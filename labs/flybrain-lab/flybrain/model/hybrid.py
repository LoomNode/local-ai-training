"""GRU + fast-weight memory head. The GRU wins bytes; give it the exact-match memory."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from torch.nn import functional as F

from flybrain.model.sparsela import Block, KeyCode, SparseKeyLM


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
        if where != "none":
            self.mem = Block(dim, KeyCode(key, dim, nk, claws, top_k, theta, seed), decay, d_ff=1, key_conv=key_conv, normalize=normalize, gate=gate)
            self.mem.chunk = int(chunk)
        # memory that grows with depth: one extra head after every mem_every-th backbone layer (the post head covers the last),
        # each with its own random address (seed offset) so heads at different depths do not share slots.
        self.mems = nn.ModuleList()
        if self.mem_every:
            n_inner = (n_layers - 1) // self.mem_every
            for i in range(n_inner):
                blk = Block(dim, KeyCode(key, dim, nk, claws, top_k, theta, seed + 1000 * (i + 1)), decay, d_ff=1, key_conv=key_conv, normalize=normalize, gate=gate)
                blk.chunk = int(chunk)
                self.mems.append(blk)

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
        if not self.mem_every:
            return self.core(e)
        layers = self.core.layers if hasattr(self.core, "layers") else self.core.blocks
        k = 0
        for i, l in enumerate(layers):
            e = l(e)
            if (i + 1) % self.mem_every == 0 and i + 1 < len(layers):
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
        return self.decoder(out), hstate
