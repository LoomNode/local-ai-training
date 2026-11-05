"""GRU + fast-weight memory head. The GRU wins bytes; give it the exact-match memory."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from torch.nn import functional as F

from flybrain.model.sparsela import Block, KeyCode, SparseKeyLM, ngram_hash, sparse_memory_read
from flybrain.model.state import AgentState, AttnCache, HeadState


class MinGRULayer(nn.Module):
    """Selective diagonal recurrence (minGRU / Mamba-class): h_t = (1-z_t) h_{t-1} + z_t (W x_t)."""

    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.z = nn.Linear(d, d)
        self.h = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    L = 32

    def scan(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """The layer with an explicit carry. h0 [B, d] is the recurrent state before x[:, 0] (None = zeros).
        Returns (layer output, recurrent state after the last position, float64)."""
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
        carries = [torch.zeros(B, d, dtype=torch.float64, device=u.device) if h0 is None else h0.to(torch.float64)]
        for i in range(nC - 1):
            carries.append(A_end[:, i] * carries[-1] + local_end[:, i])
        h0s = torch.stack(carries, dim=1)                          # [B, nC, d]
        h = (A * h0s[:, :, None, :] + local).reshape(B, nC * L, d)[:, :T]
        x = x + h.to(x.dtype)
        # the carry is a copy: a view of h would keep the whole [B, nC * L, d] float64 scan buffer alive for as long as the state is held
        return x + self.ff(self.ln2(x)), (h[:, -1].clone() if T else h0s[:, 0])   # an empty segment leaves the carry as it was

    def forward(self, x: Tensor) -> Tensor:
        return self.scan(x)[0]


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

    rebase = 65536  # rotary positions restart from the cached block every `rebase` bytes, so float32 angles stay small on long streams (bounded error; relative offsets inside the window are unchanged)

    def _stream_block(self, x: Tensor, cache: AttnCache) -> tuple[Tensor, AttnCache]:
        """At most `window` new positions against the cached window - 1."""
        B, T, d = x.shape; w = self.w
        q, k, v = self.qkv(self.ln(x)).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)   # unrotated, [B, H, T, dh]
        n = 0 if cache.k is None else cache.k.size(2)
        k_all = k if n == 0 else torch.cat([cache.k.to(k.dtype), k], dim=2)
        v_all = v if n == 0 else torch.cat([cache.v.to(v.dtype), v], dim=2)
        pos = (cache.offset - n) % self.rebase + torch.arange(n + T, device=x.device)
        i = torch.arange(n, n + T, device=x.device)[:, None]; j = torch.arange(n + T, device=x.device)[None, :]
        mask = (j <= i) & (j > i - w)                                                            # t - s in [0, w - 1]
        y = F.scaled_dot_product_attention(rope_at(q, pos[n:]), rope_at(k_all, pos), v_all, attn_mask=mask)
        keep = min(w - 1, n + T)
        new = AttnCache(k=k_all[:, :, n + T - keep :], v=v_all[:, :, n + T - keep :], offset=cache.offset + T)
        return self.out(y.transpose(1, 2).reshape(B, T, d)), new

    def stream(self, x: Tensor, cache: AttnCache) -> tuple[Tensor, AttnCache]:
        """forward() continued across calls: the same numbers as one pass over the whole stream."""
        if x.size(1) == 0: return x.new_zeros(x.shape), AttnCache(k=cache.k, v=cache.v, offset=cache.offset)   # an empty segment: no output, the cache as it was
        outs = []
        for c0 in range(0, x.size(1), self.w):
            y, cache = self._stream_block(x[:, c0 : c0 + self.w], cache)
            outs.append(y)
        return torch.cat(outs, dim=1), cache

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


class NgramEmbed(nn.Module):
    """Qwen-style n-gram embedding (Qwen3.8-Flash-Next, arm B of the one-table design): one hashed table per n-gram order,
    looked up from the input bytes, summed, and added to the stream through a zero-initialised projection, so it starts silent."""

    def __init__(self, d: int, rows: int, orders: tuple = (2, 3), seed: int = 0):
        super().__init__()
        self.rows, self.orders, self.seed = int(rows), tuple(int(o) for o in orders), int(seed)
        self.tables = nn.ModuleList(nn.Embedding(self.rows, d) for _ in self.orders)
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, tokens: Tensor) -> Tensor:
        e = 0
        for o, t in zip(self.orders, self.tables):
            e = e + t(ngram_hash(tokens, o, self.rows, salt=self.seed + 101 * o))
        return self.proj(e)


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
        decays: tuple | None = None,
        local_at: tuple | None = None,
        mem_prior: bool = False,
        ngram_addr: tuple | None = None,
        ngram_embed: int = 0,
        ngram_at: int = 2,
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
        dcs = tuple(float(x) for x in decays) if decays else (float(decay),)      # per-head decay, cycled; 1.0 = archive, <1 = recent
        ld = lambda dv: bool(learn_decay) and dv < 1.0                              # only decaying heads learn their rate
        # one table (design 2026-11-05): a learned prior under every head; n-gram orders cycled over heads (all_heads order)
        self.mem_prior = bool(mem_prior)
        self.ngram_addr = tuple(int(o) for o in ngram_addr) if ngram_addr else ()
        table = lambda i: dict(prior=self.mem_prior, ngram_order=self.ngram_addr[i % len(self.ngram_addr)] if self.ngram_addr else 0,
                               ngram_salt=seed + 31 * i)
        if where != "none":
            self.mem = Block(dim, KeyCode(key, dim, nk, claws, top_k, theta, seed), dcs[0], d_ff=1, key_conv=kcs[0], normalize=normalize, gate=gate, learn_decay=ld(dcs[0]), **table(0))
            self.mem.chunk = int(chunk)
        # memory that grows with depth: one extra head after every mem_every-th backbone layer (the post head covers the last),
        # each with its own random address (seed offset) so heads at different depths do not share slots.
        self.mems = nn.ModuleList()
        if self.mem_every:
            n_inner = (n_layers - 1) // self.mem_every
            for i in range(n_inner):
                dv = dcs[(i + 1) % len(dcs)]
                blk = Block(dim, KeyCode(key, dim, nk, claws, top_k, theta, seed + 1000 * (i + 1)), dv, d_ff=1, key_conv=kcs[(i + 1) % len(kcs)], normalize=normalize, gate=gate, learn_decay=ld(dv), **table(i + 1))
                blk.chunk = int(chunk)
                self.mems.append(blk)
        self.count = CountHead(out_dim, nk, claws, top_k, theta, seed + 777, kcs[0], chunk) if count_head else None
        # one sliding-window attention layer in the middle of the recurrent stack (local content-weighted selection)
        self.local_attn = int(local_attn)
        if self.local_attn:
            if backbone not in ("mingru", "gla"):
                raise ValueError("local_attn needs a layered recurrent backbone")
            # applied after these many layers (1-based); default one layer mid-stack. The first keeps the name `local`
            # so single-layer checkpoints load unchanged; further layers live in `local_more`.
            self.local_at = [int(i) for i in (local_at or (max(1, n_layers // 2),))]
            if any(i < 1 or i > n_layers for i in self.local_at) or len(set(self.local_at)) != len(self.local_at):
                raise ValueError(f"local_at must be distinct layer counts in 1..{n_layers}, got {self.local_at}")
            nh = max(1, d_model // 64)
            self.local = LocalAttention(d_model, nh, self.local_attn)
            self.local_more = nn.ModuleList(LocalAttention(d_model, nh, self.local_attn) for _ in self.local_at[1:])
            self.local_after = self.local_at[0]

        # arm B: a separate n-gram embedding added after layer `ngram_at` (Qwen adds its at layer 2)
        self.ngram = None
        if ngram_embed:
            if backbone not in ("mingru", "gla"):
                raise ValueError("ngram_embed needs a layered recurrent backbone")
            self.ngram = NgramEmbed(d_model, int(ngram_embed), (2, 3), seed)
            self.ngram_after = int(ngram_at)

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
        return x + blk.memory(z, z.roll(1, dims=0) if self.shuffle_keys else None, tokens=self._tokens)

    def _run_core(self, e: Tensor) -> Tensor:
        if not self.mem_every and not self.local_attn and getattr(self, "ngram", None) is None:
            return self.core(e)
        layers = self.core.layers if hasattr(self.core, "layers") else self.core.blocks
        k = 0
        for i, l in enumerate(layers):
            e = l(e)
            if getattr(self, "ngram", None) is not None and i + 1 == self.ngram_after:
                e = e + self.ngram(self._tokens)
            if self.local_attn and i + 1 in self.local_at:
                j = self.local_at.index(i + 1)
                e = e + (self.local if j == 0 else self.local_more[j - 1])(e)
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
        self._tokens = tokens       # the heads' n-gram address and the n-gram embedding read the input bytes
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
            out = out + self.mem.memory(z, z.roll(1, dims=0) if self.shuffle_keys else None, tokens=tokens)
        logits = self.decoder(out)
        if self.count is not None:  # mixture of the backbone's distribution and the memory's next-byte histogram
            log_pmem, g = self.count(out, tokens)
            logits = torch.logaddexp(torch.log1p(-g) + F.log_softmax(logits, dim=-1), g.log() + log_pmem)
        return logits, hstate

    def _check_streamable(self) -> None:
        if self.backbone != "mingru":
            raise ValueError("streaming is implemented for the mingru backbone only")
        if self.where == "pre" or self.count is not None or self.shuffle_keys:
            raise ValueError("streaming does not support where='pre', the count head or shuffled keys")
        if getattr(self, "mem_prior", False) or getattr(self, "ngram_addr", ()) or getattr(self, "ngram", None) is not None:
            raise ValueError("streaming does not support a table prior, an n-gram address or the n-gram embedding yet")
        if any(h.chunk <= 0 for h in self.all_heads()):
            raise ValueError("streaming needs the chunked head (chunk > 0)")

    def init_state(self, batch: int, device=None) -> AgentState:
        """A fresh agent: empty tables, no carry, empty attention cache."""
        self._check_streamable()
        device = self.embed.weight.device if device is None else device
        heads = []
        for blk in self.all_heads():
            d = blk.val.in_features
            tail = torch.zeros(batch, blk.key_conv - 1, d, device=device) if blk.key_conv > 1 else None
            heads.append(HeadState(S=torch.zeros(batch, blk.code.n_key, d, device=device), Ksum=torch.zeros(batch, blk.code.n_key, device=device), conv_tail=tail))
        n_attn = len(self.local_at) if self.local_attn else 0
        return AgentState(h=[None] * len(self.core.layers), heads=heads, attn=[AttnCache() for _ in range(n_attn)])

    def stream(self, tokens: Tensor, state: AgentState) -> tuple[Tensor, AgentState]:
        """forward() continued from `state`: the same logits as one pass over everything the agent has read.
        `state` is not modified; the new state is returned. A state built for another model (another number of layers, heads or
        attention layers) is refused. Quantised heads (`state_bits > 0`) give the same logits when every segment is a whole number of
        head chunks (the last may be shorter), because a one-pass read quantises the tables at the same chunk ends; any other
        segmentation also quantises at call boundaries, and the logits then match only approximately."""
        self._check_streamable()
        want = (len(self.core.layers), len(self.all_heads()), len(self.local_at) if self.local_attn else 0)
        for name, have, n in zip(("carries", "heads", "attention caches"), (len(state.h), len(state.heads), len(state.attn)), want):
            if have != n:
                raise ValueError(f"state has {have} {name}, this model has {n}: it was built for another model")
        if state.batch is not None and state.batch != tokens.size(0):
            raise ValueError(f"state is for {state.batch} agents, tokens are for {tokens.size(0)}")
        h, heads, attn = list(state.h), list(state.heads), list(state.attn)
        e = self.embed(tokens)
        layers = self.core.layers; k = 0
        for i, l in enumerate(layers):
            e, carry = l.scan(e, h[i])
            if tokens.size(1) or h[i] is not None:     # an empty segment leaves a missing carry missing: zeros would bind a head-less state to this batch size
                h[i] = carry
            if self.local_attn and i + 1 in self.local_at:
                j = self.local_at.index(i + 1)
                y, attn[j] = (self.local if j == 0 else self.local_more[j - 1]).stream(e, attn[j])
                e = e + y
            if self.mem_every and (i + 1) % self.mem_every == 0 and i + 1 < len(layers):
                y, heads[1 + k] = self.mems[k].memory(self.mems[k].ln1(e), state=heads[1 + k])
                e = e + y; k += 1
        out = self.core.ln(e)
        if self.where == "post":
            y, heads[0] = self.mem.memory(self.mem.ln1(out), state=heads[0])
            out = out + y
        return self.decoder(out), AgentState(h=h, heads=heads, attn=attn)
