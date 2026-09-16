"""GRU + fast-weight memory head. The GRU wins bytes; give it the exact-match memory."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from torch.nn import functional as F

from flybrain.model.sparsela import Block, KeyCode, SparseKeyLM, sparse_memory_read


class MinGRULayer(nn.Module):
    """Selective diagonal recurrence (minGRU / Mamba-class): h_t = (1-z_t) h_{t-1} + z_t (W x_t)."""

    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.z = nn.Linear(d, d)
        self.h = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    L = 32

    def forward(self, x: Tensor) -> Tensor:
        u = self.ln1(x)
        z = torch.sigmoid(self.z(u)); cand = self.h(u)
        a = (1 - z).clamp_min(1e-4); b = z * cand
        # chunked closed form, all chunks at once: within a chunk h_t = A_t (h_0 + sum_{s<=t} b_s / A_s), A_t = prod a;
        # the carry h_0 of each chunk is a short sequential loop over [B, d] tensors only (no per-chunk slicing of big tensors).
        B, T, d = u.shape; L = self.L
        pad = (-T) % L
        if pad:
            a = F.pad(a, (0, 0, 0, pad), value=1.0); b = F.pad(b, (0, 0, 0, pad))
        nC = (T + pad) // L
        a = a.reshape(B, nC, L, d).double(); b = b.reshape(B, nC, L, d).double()
        A = torch.exp(torch.cumsum(torch.log(a), dim=2))          # [B, nC, L, d]
        local = A * torch.cumsum(b / A, dim=2)                     # chunk output with h_0 = 0
        A_end, local_end = A[:, :, -1], local[:, :, -1]            # [B, nC, d]
        carries = [torch.zeros(B, d, dtype=torch.float64, device=u.device)]
        for i in range(nC - 1):
            carries.append(A_end[:, i] * carries[-1] + local_end[:, i])
        h0 = torch.stack(carries, dim=1)                           # [B, nC, d]
        h = (A * h0[:, :, None, :] + local).reshape(B, nC * L, d)[:, :T]
        x = x + h.to(x.dtype)
        return x + self.ff(self.ln2(x))


class MinGRU(nn.Module):
    def __init__(self, d: int, n_layers: int = 4, d_ff: int = 256, vocab_size: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([MinGRULayer(d, d_ff) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)

    def forward(self, e: Tensor) -> Tensor:
        for l in self.layers:
            e = l(e)
        return self.ln(e)


class GLABackbone(nn.Module):
    """Gated linear attention blocks: learned key/query, data-dependent decay, FFN."""

    def __init__(self, d: int, n_layers: int = 4, d_ff: int = 256, key_dim: int = 64, key_conv: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, KeyCode("gla", d, key_dim, 4, 4, 1.0, i), 1.0, d_ff, key_conv=key_conv, normalize=True) for i in range(n_layers)])
        self.ln = nn.LayerNorm(d)

    def forward(self, e: Tensor) -> Tensor:
        for b in self.blocks:
            e = b(e)
        return self.ln(e)


def rope(x: Tensor, base: float = 10000.0) -> Tensor:
    """Rotary position embedding on the last dim of [B, H, T, Dh]."""
    B, H, T, Dh = x.shape
    half = Dh // 2
    freqs = base ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * freqs[None, :]  # [T, half]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def rope_at(x: Tensor, pos: Tensor, base: float = 10000.0) -> Tensor:
    """Rotary embedding of [B, H, T, Dh] at explicit integer positions pos [T]."""
    Dh = x.size(-1); half = Dh // 2
    freqs = base ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    ang = pos.to(torch.float32)[:, None] * freqs[None, :]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class LocalAttention(nn.Module):
    """Sliding-window causal attention (each position sees the previous `window` positions incl. itself), computed
    block-wise so cost is O(T * window) and it runs unchanged at any length. Rotary positions."""

    def __init__(self, d: int, n_heads: int, window: int):
        super().__init__()
        self.h, self.dh, self.w = n_heads, d // n_heads, int(window)
        self.ln = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def _qkv(self, x: Tensor):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln(x)).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        pos = torch.arange(T, device=x.device)
        return rope_at(q, pos), rope_at(k, pos), v

    def forward(self, x: Tensor) -> Tensor:
        B, T, d = x.shape; w = self.w; H = self.h
        q, k, v = self._qkv(x)                                            # [B, H, T, dh]
        pad = (-T) % w; nB = (T + pad) // w
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        q = q.view(B, H, nB, w, self.dh)
        k = k.view(B, H, nB, w, self.dh); v = v.view(B, H, nB, w, self.dh)
        kp = torch.cat([torch.zeros_like(k[:, :, :1]), k[:, :, :-1]], dim=2)   # previous block's keys
        vp = torch.cat([torch.zeros_like(v[:, :, :1]), v[:, :, :-1]], dim=2)
        k2 = torch.cat([kp, k], dim=3); v2 = torch.cat([vp, v], dim=3)         # [B, H, nB, 2w, dh]
        i = torch.arange(w, device=x.device)[:, None]; j = torch.arange(2 * w, device=x.device)[None, :]
        mask = (j >= i + 1) & (j <= i + w)                                       # t - s in [0, w-1]
        mask = mask[None].repeat(nB, 1, 1); mask[0, :, :w] = False              # block 0 has no previous block
        mask = mask[None, None].expand(B, H, nB, w, 2 * w).reshape(B * H * nB, 1, w, 2 * w)
        qq = q.permute(0, 1, 2, 3, 4).reshape(B * H * nB, 1, w, self.dh)
        kk = k2.reshape(B * H * nB, 1, 2 * w, self.dh); vv = v2.reshape(B * H * nB, 1, 2 * w, self.dh)
        y = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask)          # [B*H*nB, 1, w, dh]
        y = y.view(B, H, nB * w, self.dh)[:, :, :T].transpose(1, 2).reshape(B, T, d)
        return self.out(y)

    def reference(self, x: Tensor) -> Tensor:
        """Banded full attention, for tests."""
        B, T, d = x.shape
        q, k, v = self._qkv(x)
        t = torch.arange(T, device=x.device)
        mask = ((t[:, None] - t[None, :]) >= 0) & ((t[:, None] - t[None, :]) < self.w)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(B, T, d))


class RopeAttention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.rope_base = 10000.0  # eval-time NTK-aware length extension changes this (set_rope_scale)

    def forward(self, u: Tensor) -> Tensor:
        B, T, d = u.shape
        q, k, v = self.qkv(u).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)  # 3 x [B, H, T, Dh]
        q, k = rope(q, self.rope_base), rope(k, self.rope_base)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, d))


class TransformerBackbone(nn.Module):
    """Pre-LN causal transformer: the negative control. Full attention already retrieves exactly.
    rope=True swaps the learned position table for rotary positions (the modern recipe)."""

    def __init__(self, d: int, n_layers: int = 4, d_ff: int = 256, n_heads: int | None = None, max_len: int = 1024, rope: bool = False):
        super().__init__()
        n_heads = n_heads or max(1, d // 32)
        self.rope = bool(rope)
        self.pos = None if rope else nn.Embedding(max_len, d)
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.attn = nn.ModuleList([RopeAttention(d, n_heads) if rope else nn.MultiheadAttention(d, n_heads, batch_first=True) for _ in range(n_layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d)) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)

    @property
    def layers(self):
        return self.attn

    def set_rope_scale(self, s: float, base: float = 10000.0) -> None:
        """NTK-aware RoPE extension for evaluating at s x the training length: base' = base * s^(Dh/(Dh-2))."""
        if not self.rope:
            raise ValueError("set_rope_scale needs rope=True")
        for a in self.attn:
            a.rope_base = float(base * s ** (a.dh / (a.dh - 2)))

    def forward(self, e: Tensor) -> Tensor:
        T = e.size(1)
        if self.pos is not None:
            e = e + self.pos(torch.arange(T, device=e.device))[None]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=e.device), diagonal=1)
        for ln1, ln2, attn, ff in zip(self.ln1, self.ln2, self.attn, self.ff):
            u = ln1(e)
            e = e + (attn(u) if self.rope else attn(u, u, u, attn_mask=mask, need_weights=False)[0])
            e = e + ff(ln2(e))
        return self.ln(e)


class CountHead(nn.Module):
    """Memory that writes the *next byte* (one-hot) under the context key and reads back an empirical next-byte histogram.
    Same frozen sparse address and exact index form as the value heads; the read is mixed into the output distribution
    with a gate driven by how much matched (pointer-cache with sparse addressing; KC -> MBON drives the output directly)."""

    def __init__(self, d: int, n_key: int, claws: int, top_k: int, theta: float, seed: int, key_conv: int, chunk: int, vocab: int = 256):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.key_conv = int(key_conv)
        if self.key_conv > 1:
            self.conv = nn.Conv1d(d, d, self.key_conv, groups=d)
        self.code = KeyCode("sparse", d, n_key, claws, top_k, theta, seed)
        self.chunk = int(chunk); self.vocab = int(vocab)
        self.gate = nn.Linear(2, 1); nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x: Tensor, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """x: [B, T, d] stream the key is taken from; tokens: [B, T] input bytes. Returns (log p_mem [B, T, vocab], gate g [B, T, 1])."""
        xk = self.ln(x)
        if self.key_conv > 1:
            xk = self.conv(F.pad(xk.transpose(1, 2), (self.key_conv - 1, 0))).transpose(1, 2)
        v = F.one_hot(tokens, self.vocab).to(xk.dtype)
        v = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)            # extra channel carries the match mass
        out = sparse_memory_read(self.code, xk, v, self.chunk, 1.0, True, None)   # normalised: [B, T, vocab + 1]
        r = out[..., -1:].clamp_min(0.0)                                  # mass / (mass + 1e-3) in [0, 1)
        mass = 1e-3 * r / (1 - r).clamp_min(1e-6)
        eps = 1e-4
        p_mem = (out[..., :-1].clamp_min(0.0) + eps) / (r + self.vocab * eps)
        g = torch.sigmoid(self.gate(torch.cat([r, torch.log1p(mass)], dim=-1)))
        return p_mem.log(), g


WHERE = ("pre", "post", "none")
BACKBONES = ("gru", "mingru", "gla", "transformer")


class GRUHybrid(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        hidden: int = 330,
        key: str = "sparse",
        where: str = "post",
        n_key: int = 1024,
        top_k: int = 4,
        claws: int = 4,
        theta: float = 1.0,
        learned_key: int = 64,
        normalize: bool = True,
        key_conv: int = 2,
        decay: float = 1.0,
        vocab_size: int = 256,
        seed: int = 0,
        shuffle_keys: bool = False,
        backbone: str = "gru",
        n_layers: int = 4,
        d_ff: int = 256,
        gate: bool = False,
        max_len: int = 1024,
        rope: bool = False,
        chunk: int = 0,
        mem_every: int = 0,
        key_convs: tuple | None = None,
        count_head: bool = False,
        learn_decay: bool = False,
        local_attn: int = 0,
    ) -> None:
        super().__init__()
        if where not in WHERE or backbone not in BACKBONES:
            raise ValueError("bad where/backbone")
        if mem_every and (backbone not in ("mingru", "gla") or where != "post"):
            raise ValueError("mem_every needs a layered recurrent backbone and where='post'")
        self.mem_every = int(mem_every)
        self.backbone = backbone
        self.d_model, self.hidden, self.where, self.key = int(d_model), int(hidden), where, key
        self.vocab_size = self.n_out = int(vocab_size)
        self.shuffle_keys = bool(shuffle_keys)  # leakage control: keys/queries from another sequence
        self.embed = nn.Embedding(vocab_size, d_model)
        if backbone == "gru":
            self.gru = nn.GRU(d_model, hidden, batch_first=True)
            out_dim = hidden
        elif backbone == "mingru":
            self.core = MinGRU(d_model, n_layers, d_ff, vocab_size); out_dim = d_model
        elif backbone == "gla":
            self.core = GLABackbone(d_model, n_layers, d_ff, learned_key, key_conv); out_dim = d_model
        else:
            self.core = TransformerBackbone(d_model, n_layers, d_ff, max_len=max_len, rope=rope); out_dim = d_model
        self.d_ff = int(d_ff)
        self.hidden = out_dim
        self.decoder = nn.Linear(out_dim, vocab_size)
        dim = d_model if where == "pre" else out_dim
        nk = learned_key if key in ("learned", "softmax") else n_key
        kcs = tuple(int(k) for k in key_convs) if key_convs else (int(key_conv),)  # per-head key width, cycled over heads
        if where != "none":
            self.mem = Block(dim, KeyCode(key, dim, nk, claws, top_k, theta, seed), decay, d_ff=1, key_conv=kcs[0], normalize=normalize, gate=gate, learn_decay=learn_decay)
            self.mem.chunk = int(chunk)
        # memory that grows with depth: one extra head after every mem_every-th backbone layer (the post head covers the last),
        # each with its own random address (seed offset) so heads at different depths do not share slots.
        self.mems = nn.ModuleList()
        if self.mem_every:
            n_inner = (n_layers - 1) // self.mem_every
            for i in range(n_inner):
                blk = Block(dim, KeyCode(key, dim, nk, claws, top_k, theta, seed + 1000 * (i + 1)), decay, d_ff=1, key_conv=kcs[(i + 1) % len(kcs)], normalize=normalize, gate=gate, learn_decay=learn_decay)
                blk.chunk = int(chunk)
                self.mems.append(blk)
        self.count = CountHead(out_dim, nk, claws, top_k, theta, seed + 777, kcs[0], chunk) if count_head else None
        # one sliding-window attention layer in the middle of the recurrent stack (local content-weighted selection)
        self.local_attn = int(local_attn)
        if self.local_attn:
            if backbone not in ("mingru", "gla"):
                raise ValueError("local_attn needs a layered recurrent backbone")
            self.local = LocalAttention(d_model, max(1, d_model // 64), self.local_attn)
            self.local_after = max(1, n_layers // 2)  # applied after this many layers

    def expand_slots(self, n_key: int) -> None:
        """Eval-time slot expansion on every head (see KeyCode.expand_slots); each head gets its own fresh rows."""
        for i, h in enumerate(self.all_heads()):
            h.code.expand_slots(n_key, seed=10_000 + 7 * i)

    def all_heads(self) -> list:
        return ([self.mem] if self.where != "none" else []) + list(self.mems)

    @property
    def n_heads(self) -> int:
        return len(self.all_heads())

    def _head(self, blk, x: Tensor) -> Tensor:
        z = blk.ln1(x)
        return x + blk.memory(z, z.roll(1, dims=0) if self.shuffle_keys else None)

    def _run_core(self, e: Tensor) -> Tensor:
        if not self.mem_every and not self.local_attn:
            return self.core(e)
        layers = self.core.layers if hasattr(self.core, "layers") else self.core.blocks
        k = 0
        for i, l in enumerate(layers):
            e = l(e)
            if self.local_attn and i + 1 == self.local_after:
                e = e + self.local(e)
            if self.mem_every and (i + 1) % self.mem_every == 0 and i + 1 < len(layers):
                e = self._head(self.mems[k], e); k += 1
        assert k == len(self.mems)
        return self.core.ln(e)

    def persistent_state_scalars(self, batch: int = 1) -> int:
        return int(batch * self.hidden)

    def flops_per_step(self, batch: int = 1) -> int:
        e, h, v = self.d_model, self.hidden, self.vocab_size
        if self.backbone == "gru":
            base = 2 * 3 * (e * h + h * h) + 2 * h * v
        else:  # per-token recurrent cost; for the transformer this omits the O(T) attention term on purpose
            n_l = len(self.core.layers if hasattr(self.core, "layers") else self.core.blocks)
            base = 2 * e * v + n_l * (2 * 4 * e * e + 2 * 2 * e * self.d_ff)
        if self.where == "none":
            return int(batch * base)
        c = self.mem.code; dim = e if self.where == "pre" else h; k = c.active_units()
        key_cost = 2 * c.claws * c.n_key if c.kind == "sparse" else 2 * dim * c.n_key * (2 if c.kind in ("learned", "softmax") else 1)
        head = key_cost + 2 * dim * dim * 2 + 4 * k * dim + 2 * dim * self.mem.key_conv
        return int(batch * (base + self.n_heads * head))

    def forward(self, tokens: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        e = self.embed(tokens)
        if self.where == "pre":
            z = self.mem.ln1(e)
            e = e + self.mem.memory(z, z.roll(1, dims=0) if self.shuffle_keys else None)
        if self.backbone == "gru":
            out, hstate = self.gru(e, state)
        else:
            out = self._run_core(e); hstate = out[:, -1]
        if self.where == "post":
            z = self.mem.ln1(out)
            out = out + self.mem.memory(z, z.roll(1, dims=0) if self.shuffle_keys else None)
        logits = self.decoder(out)
        if self.count is not None:  # mixture of the backbone's distribution and the memory's next-byte histogram
            log_pmem, g = self.count(out, tokens)
            logits = torch.logaddexp(torch.log1p(-g) + F.log_softmax(logits, dim=-1), g.log() + log_pmem)
        return logits, hstate
