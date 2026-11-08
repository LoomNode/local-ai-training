"""Parallel-form linear attention with pluggable keys (GPU, long sequences).

Equivalent to the FastWeightLM recurrence with a constant gate, written as a
causal T×T product so it trains in parallel:

    out_t = Σ_{s<t} decay^{t-1-s} · (q_t · key_{s-1}) · v_s

Key kinds:
    sparse    frozen ±1 claws → ReLU(z−θ) → top-k        (the fly hash)
    gausstopk frozen Gaussian → ReLU(z−θ) → top-k         (top-k without claws)
    gauss     frozen Gaussian, all active                  (orthogonal dense)
    learned   learned key + query (gated-linear-attention style)
    softmax   ordinary causal softmax attention (ceiling reference)

Each layer: x + W_o(out) then x + FFN(x). No recurrent hidden state.
"""

from __future__ import annotations

import math
import os

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from flybrain.model.seqmixer import signed_expand_matrix
from flybrain.model.select_kernel import fused_select, select_available
from flybrain.model.state import HeadState

KEY_KINDS = ("sparse", "gausstopk", "gauss", "learned", "softmax", "tied", "mixed", "gla")


# Speed switches (2026-11-05): on a card the head's one-pass read is torch.compiled (fuses each chunk's compare, multiply,
# sum and scale into single kernels; same values to fp32 rounding) and slot selection is one Triton kernel.
# FLYBRAIN_COMPILE=0 turns both off (eager, for debugging or a box without a working compiler).
COMPILE_ON = os.environ.get("FLYBRAIN_COMPILE", "1") != "0"
# FLYBRAIN_FLEX=1: the sliding-window attention through flex_attention (4 ms a step at 20M). Off by default: under bf16
# autocast it reorders the arithmetic, and a 4000-step proof run moved by 0.003 bpb (the seed spread), so it is not a
# drop-in against older runs; turn it on for every arm of an experiment, or for none.
FLEX_ON = os.environ.get("FLYBRAIN_FLEX", "0") == "1"
# FLYBRAIN_TABLE16=1: the memory tables (S, Ksum) held in float16 between chunks (half the head's table traffic, -4 ms a
# step at 20M). REJECTED 2026-11-05: the 4000-step proof run read 0.949 bpb against 0.925, the loss in the copyable bytes
# (0.240 vs 0.210), so 11 bits is not enough for the tables (Run 47: 8-bit state hurt, int16 with a scale was free).
# Kept only as the record of that test; do not use for results.
TABLE16 = os.environ.get("FLYBRAIN_TABLE16", "0") == "1"
_COMPILED_READ = None


def compiled_read():
    global _COMPILED_READ
    if _COMPILED_READ is None:
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)   # one entry per (shape, flags): eval sweeps many lengths
        _COMPILED_READ = torch.compile(sparse_memory_read, dynamic=False)
    return _COMPILED_READ


class KeyCode(nn.Module):
    def __init__(self, kind: str, d: int, n_key: int, claws: int, top_k: int, theta: float, seed: int, signed: bool = False):
        super().__init__()
        self.signed = bool(signed)
        if kind not in KEY_KINDS:
            raise ValueError(f"kind must be one of {KEY_KINDS}")
        self.kind, self.n_key, self.top_k, self.claws = kind, int(n_key), int(top_k), int(claws)
        if kind == "sparse":
            self.register_buffer("proj", signed_expand_matrix(n_key, d, claws, seed))
            self.theta = nn.Parameter(torch.full((n_key,), float(theta)))
            self._build_index_form()
            self.register_load_state_dict_post_hook(lambda module, incompatible: module._build_index_form())
        elif kind in ("gausstopk", "gauss"):
            g = torch.Generator().manual_seed(seed)
            self.register_buffer("proj", torch.randn(n_key, d, generator=g) / d**0.5)
            if kind == "gausstopk":
                self.theta = nn.Parameter(torch.full((n_key,), float(theta)))
        elif kind == "tied":  # trainable, q = k, Gaussian init (warm start of the frozen 'gauss' key)
            g = torch.Generator().manual_seed(seed)
            self.proj = nn.Parameter(torch.randn(n_key, d, generator=g) / d**0.5)
        else:  # learned / softmax / gla
            self.key = nn.Linear(d, n_key, bias=False)
            self.query = nn.Linear(d, n_key, bias=False)
            if kind == "gla":  # data-dependent scalar decay per position, init near 1
                self.gate = nn.Linear(d, 1)
                nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, 4.0)

    def expand_slots(self, n_key: int, seed: int) -> None:
        """Eval-time capacity: append fresh random address rows (and mean-theta thresholds) so a checkpoint trained with fewer
        slots can be scored at a longer window under the same writes-per-slot load. Trained rows are untouched."""
        if self.kind != "sparse":
            raise ValueError("expand_slots is for the sparse key")
        if n_key < self.n_key:
            raise ValueError("cannot shrink the address table")
        if n_key == self.n_key:
            return
        extra = signed_expand_matrix(n_key - self.n_key, self.proj.shape[1], self.claws, seed).to(self.proj)
        self.proj = torch.cat([self.proj, extra], dim=0)
        theta = torch.cat([self.theta.data, self.theta.data.mean().expand(n_key - self.n_key)])
        self.theta = nn.Parameter(theta, requires_grad=self.theta.requires_grad)
        self.n_key = int(n_key)
        self._build_index_form()

    def active_units(self) -> int:
        return min(self.top_k, self.n_key) if self.kind in ("sparse", "gausstopk") else self.n_key

    def log_decay(self, x: Tensor) -> Tensor | None:
        if self.kind != "gla":
            return None
        return F.logsigmoid(self.gate(x)).squeeze(-1)  # [B, T], log alpha_t

    code_slice = 512  # positions per slice when selecting the top-k (peak memory = B x slice x n_key, never B x T x n_key)

    def sparse_code(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Index form of the sparse key: (idx [B, T, k] long, val [B, T, k] >= 0). Same numbers as forward() restricted to its nonzeros.
        Selection runs without grad in slices of positions; the k selected scores are then recomputed sparsely so memory does not
        scale with n_key (the dense [B, T, n_key] projection is never kept)."""
        if self.kind not in ("sparse", "gausstopk") or self.signed:
            raise ValueError("index form needs nonnegative sparse keys")
        kk = min(self.top_k, self.n_key)
        with torch.no_grad():
            if self.kind == "sparse" and select_available(x) and COMPILE_ON and not (self.n_key & (self.n_key - 1)):
                # one kernel: 4 signed picks per slot, threshold, top-k (select_kernel.py)
                cols, signs = self._index_form()
                idx = fused_select(x.reshape(-1, x.size(-1)).detach(), cols, signs, self.theta.detach(), kk).view(x.size(0), x.size(1), kk)
            else:
                parts = []
                for c0 in range(0, x.size(1), self.code_slice):
                    z = x[:, c0 : c0 + self.code_slice].matmul(self.proj.t())
                    parts.append(F.relu(z - self.theta).topk(kk, dim=-1).indices)
                idx = torch.cat(parts, dim=1)                              # [B, T, k]
        if self.kind == "sparse":
            # each frozen row has exactly `claws` nonzeros (+-1), so its dot product is `claws` picks from x: the same sum
            # as gathering the dense rows ([B, T, k, d]) at 1/d of the memory traffic (2026-11-05)
            cols, signs = self._index_form()
            ci, si = cols[idx], signs[idx]                                  # [B, T, k, claws]
            B, T, kk = idx.shape
            xg = x.gather(-1, ci.reshape(B, T, kk * ci.size(-1))).reshape(B, T, kk, -1)
            z_sel = (xg * si).sum(-1)
        else:
            rows = self.proj[idx]                                          # [B, T, k, d] (frozen)
            z_sel = (x.unsqueeze(2) * rows).sum(-1)                        # [B, T, k]
        val = F.relu(z_sel - self.theta[idx])
        return idx, val

    def _index_form(self) -> tuple[Tensor, Tensor]:
        """The sparse key matrix as (columns [n_key, claws] int32, signs [n_key, claws]): non-persistent buffers built
        at construction, after expand_slots and after a checkpoint load, so nothing data-dependent runs per call."""
        return self.index_cols, self.index_signs

    def _build_index_form(self) -> None:
        nz = (self.proj != 0).nonzero()
        cols = nz[:, 1].reshape(self.proj.size(0), -1)
        if cols.size(1) != self.claws:
            raise ValueError("every sparse key row must have exactly `claws` nonzeros")
        self.register_buffer("index_cols", cols.to(torch.int32).contiguous(), persistent=False)
        self.register_buffer("index_signs", self.proj.gather(1, cols).contiguous(), persistent=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (query, key), each [B, T, n_key]."""
        if self.kind in ("learned", "softmax", "gla"):
            return self.query(x), self.key(x)
        z = x.matmul(self.proj.t())
        if self.kind in ("gauss", "tied"):
            k = z
        elif self.signed:
            kk = min(self.top_k, self.n_key)
            idx = z.abs().topk(kk, dim=-1).indices
            k = z * torch.zeros_like(z).scatter_(-1, idx, 1.0)
        else:
            c = F.relu(z - self.theta)
            kk = min(self.top_k, self.n_key)
            if 0 < kk < self.n_key:
                idx = c.topk(kk, dim=-1).indices
                c = c * torch.zeros_like(c).scatter_(-1, idx, 1.0)
            k = c
        return k, k


def ngram_hash(tokens: Tensor, order: int, mod: int, salt: int = 0) -> Tensor:
    """Hash of the last `order` input bytes up to and including each position, in [0, mod). Causal: position t sees bytes
    t-order+1..t; positions before the start read as byte 256. Different salts give independent-looking hashes."""
    g = torch.zeros_like(tokens, dtype=torch.long)
    for i in range(order):
        prev = torch.full_like(tokens, 256) if i >= tokens.size(1) else torch.cat([torch.full_like(tokens[:, :i], 256), tokens[:, : tokens.size(1) - i]], dim=1)
        g = g + prev.long() * (257 ** i)
    a = 2 * ((salt * 2654435761 + 97) % (1 << 20)) + 1          # odd multiplier from the salt
    b = (salt * 40503 + 12345) % 2147483647
    return ((g * a + b) % 2147483647) % mod


def fake_quant(S: Tensor, bits: int) -> Tensor:
    """Symmetric per-row fake quantisation to `bits` (0 = identity): what an integer memory state would hold.

    Straight-through: the forward value is the quantised one, the backward pass sees the identity. Plain `.round()`
    has zero gradient, which silently cuts the memory's write path at every chunk boundary during training
    (eval is unaffected; that cost 0.02 bpb on retrieval in Run 47 before this was fixed)."""
    if not bits:
        return S
    qmax = 2 ** (bits - 1) - 1
    scale = S.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    q = (S / scale).round().clamp(-qmax, qmax) * scale
    return S + (q - S).detach()


def sparse_memory_read(code: KeyCode, xk: Tensor, v: Tensor, chunk: int, decay: float, normalize: bool, gate_tau: Tensor | None, state_bits: int = 0, state: HeadState | None = None, prior: tuple[Tensor, Tensor] | None = None, idx_offset: Tensor | None = None):
    """Linear-cost index form of the parallel path below, exact for scalar decay in (0, 1] + nonnegative sparse keys.
    out_t = g(m_t) * q_t . S_t / (m_t + 1e-3),  S_t = sum_{1<=s<t} k_{s-1} v_s^T,  m_t = q_t . sum_{1<=s<t} k_{s-1}.
    The key has only top_k nonzeros, so reads are gathers and writes are index-adds; the T x T term is chunk-local.
    With `state` the tables and the previous position's key continue from an earlier call, and the return value is
    (out, new HeadState); the state passed in is not modified. S is carried aged to the first position of the next call.
    `prior` = (S0 [n_key, d], K0 [n_key]): the tables start from these (shared by the batch) instead of zeros, so the read
    is a weighted mean of a learned value and this sequence's writes (design 2026-11-05). `idx_offset` [B, T] shifts every
    slot index at that position (mod n_key), for reads and writes alike: an n-gram term in the address."""
    B, T, _ = xk.shape; L = chunk; d = v.size(-1); n_key = code.n_key
    if state is not None and T == 0:  # an empty segment: no output, and the state as it was (copied, so the new state shares nothing with the old)
        cp = lambda t: None if t is None else t.clone()
        return v.new_zeros(B, 0, d), HeadState(S=cp(state.S), Ksum=cp(state.Ksum), last_idx=cp(state.last_idx), last_val=cp(state.last_val), conv_tail=cp(state.conv_tail))
    lam = decay if torch.is_tensor(decay) else torch.tensor(float(decay), device=xk.device, dtype=xk.dtype)
    archive = (not torch.is_tensor(decay)) and float(decay) == 1.0     # no ageing anywhere: skip every multiply by 1 (2026-11-05)
    idx, val = code.sparse_code(xk)                                   # [B, T, k]
    if idx_offset is not None:
        idx = (idx + idx_offset[..., None]) % n_key
    val = val / (val.norm(dim=-1, keepdim=True) + 1e-6)              # q = k = unit-norm sparse code
    k = idx.size(-1)
    # key_{s-1} at slot s (slot 0 reads nothing, unless an earlier call left its last key in `state`)
    if state is not None and state.last_idx is not None:
        idx_s = torch.cat([state.last_idx[:, None], idx[:, :-1]], dim=1)
        val_s = torch.cat([state.last_val[:, None].to(val.dtype), val[:, :-1]], dim=1)
    else:
        idx_s = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        val_s = torch.cat([torch.zeros_like(val[:, :1]), val[:, :-1]], dim=1)
    tdt = torch.float16 if (TABLE16 and xk.is_cuda and not state_bits) else xk.dtype   # table storage dtype
    if state is None and prior is not None:
        if state_bits:
            raise ValueError("a prior with a quantised state is not supported")
        S = prior[0].to(tdt).repeat(B, 1); Ksum = prior[1].to(tdt).repeat(B)
    elif state is None:
        S = xk.new_zeros(B * n_key, d, dtype=tdt); Ksum = xk.new_zeros(B * n_key, dtype=tdt)
    else:
        if prior is not None:
            raise ValueError("a streamed state already holds its prior")
        # copies: the tables are updated in place below, and the state passed in must come back unchanged
        S = state.S.to(tdt).reshape(B * n_key, d).clone(); Ksum = state.Ksum.to(tdt).reshape(B * n_key).clone()
    boff = (torch.arange(B, device=xk.device) * n_key)[:, None, None]  # [B, 1, 1]
    outs = []
    for c in range(0, T, L):
        iq, vq = idx[:, c : c + L], val[:, c : c + L]                  # queries in chunk
        ik, vk, vc = idx_s[:, c : c + L], val_s[:, c : c + L], v[:, c : c + L]
        Lc = iq.size(1)
        # intra-chunk scores by index equality: sc[t, s] = sum_{i,j} vq[t, i] vk[s, j] [iq[t, i] == ik[s, j]]  (no n_key-sized tensor)
        eq = (iq[:, :, None, :, None] == ik[:, None, :, None, :])       # [B, t, s, i, j]
        sc = (eq * (vq[:, :, None, :, None] * vk[:, None, :, None, :])).sum((-1, -2))  # [B, t, s]
        ar = torch.arange(Lc, device=xk.device)
        lag = (ar[:, None] - 1 - ar[None, :]).clamp(min=0).to(xk.dtype)          # t-1-s within the chunk
        tril = torch.ones(Lc, Lc, device=xk.device, dtype=torch.bool).tril(-1)
        sc = sc * tril if archive else sc * tril * lam ** lag
        age_in = 1.0 if archive else (lam ** ar.to(xk.dtype))[None, :, None]          # reads: state aged from chunk start to t
        age_out = 1.0 if archive else (lam ** (Lc - 1 - ar).to(xk.dtype))[None, :, None]   # writes: aged to the next chunk start
        # inter-chunk reads from the running state
        flat = (iq + boff).reshape(B, Lc * k)                          # [B, Lc*k] rows of S
        Srows = S.index_select(0, flat.reshape(-1)).view(B, Lc, k, d).to(xk.dtype)
        Krows = Ksum.index_select(0, flat.reshape(-1)).view(B, Lc, k).to(xk.dtype)
        num = sc.matmul(vc) + age_in * (Srows * vq[..., None]).sum(2)  # [B, Lc, d]
        mass = sc.sum(-1, keepdim=True) + age_in * (Krows * vq).sum(-1, keepdim=True)
        out = num
        if normalize:
            out = out / (mass + 1e-3)
        if gate_tau is not None:
            out = out * (mass / (mass + F.softplus(gate_tau)))
        outs.append(out)
        # write this chunk's keys into the state
        wflat = (ik + boff).reshape(-1)
        vk_aged = vk if archive else vk * age_out
        # one pass: the out-of-place index_add writes into a copy of the (aged) table, instead of ageing, zero-filling,
        # scattering and adding in four passes over B x n_key x d
        # in place: nothing saved for backward holds the old table values (index_select keeps only its indices), so
        # the write needs no fresh 67 MB table per chunk
        S = (S if archive else S * (lam ** Lc).to(tdt)).index_add_(0, wflat, (vk_aged[..., None] * vc[:, :, None, :]).reshape(-1, d).to(tdt))
        Ksum = (Ksum if archive else Ksum * (lam ** Lc).to(tdt)).index_add_(0, wflat, vk_aged.reshape(-1).to(tdt))
        if state_bits:  # integer memory state: quantise what is carried between chunks
            S = fake_quant(S, state_bits); Ksum = fake_quant(Ksum[None], state_bits)[0]
    out = torch.cat(outs, dim=1)
    if state is None:
        return out
    # the carried keys are copies: a view would keep the whole per-call idx / val tensor alive for as long as the state is held
    return out, HeadState(S=S.reshape(B, n_key, d).to(xk.dtype), Ksum=Ksum.reshape(B, n_key).to(xk.dtype), last_idx=idx[:, -1].clone(), last_val=val[:, -1].clone(), conv_tail=state.conv_tail)


class Block(nn.Module):
    def __init__(self, d: int, code, decay: float, d_ff: int, key_conv: int = 1, freeze_conv: bool = False, normalize: bool = False, gate: bool = False, learn_decay: bool = False,
                 prior: bool = False, ngram_order: int = 0, ngram_salt: int = 0):
        super().__init__()
        self.gate = bool(gate)
        self.chunk = 0  # >0: linear-cost chunked form (exact for decay 1 + nonnegative sparse keys)
        if gate:  # match-strength gate: g = mass / (mass + softplus(tau)); zero output when nothing in memory matched
            self.gate_tau = nn.Parameter(torch.zeros(()))
        self.codes = nn.ModuleList(code if isinstance(code, (list, tuple)) else [code])
        self.code = self.codes[0]
        self.decay, self.normalize = float(decay), bool(normalize)
        self.learn_decay = bool(learn_decay)
        self.state_bits = 0  # eval-time: fake-quantise the memory state carried between chunks (0 = fp32)
        if learn_decay:  # recency: decay = sigmoid(logit), one scalar per head, initialised at the given decay
            if not (0.0 < decay < 1.0):
                raise ValueError("learn_decay needs an initial decay in (0, 1)")
            self.decay_logit = nn.Parameter(torch.tensor(math.log(decay / (1.0 - decay))))
        self.outs = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in self.codes])
        self.key_conv = int(key_conv)
        if key_conv > 1:  # causal depthwise conv so the key can see the last `key_conv` tokens
            self.conv = nn.Conv1d(d, d, key_conv, groups=d)
            if freeze_conv:
                for prm in self.conv.parameters():
                    prm.requires_grad_(False)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.val = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        # one table (design 2026-11-05): the head's table starts from a learned prior, each slot a value V0 counted as
        # m0 = softplus(r) earlier writes; and/or the slot index is shifted by a hash of the last `ngram_order` bytes
        self.prior = bool(prior)
        self.ngram_order, self.ngram_salt = int(ngram_order), int(ngram_salt)
        if self.prior or self.ngram_order:
            if len(self.codes) != 1 or self.code.kind != "sparse":
                raise ValueError("a prior or an n-gram address needs one sparse key code")
        if self.prior:
            self.prior_V = nn.Parameter(torch.zeros(self.code.n_key, d))
            self.prior_r = nn.Parameter(torch.zeros(self.code.n_key))

    def prior_tables(self) -> tuple[Tensor, Tensor] | None:
        """(S0, K0) the head's tables start from, or None without a prior."""
        if not self.prior:
            return None
        m0 = F.softplus(self.prior_r.float())
        return m0[:, None] * self.prior_V.float(), m0

    def causal_conv(self, x: Tensor) -> Tensor:
        """The depthwise causal conv of width key_conv on [B, T, d], as the weighted sum of shifted copies it is
        (out_t = sum_j w_j x_{t-(K-1-j)} + bias, zeros before the start): the same numbers as Conv1d on the transposed,
        padded input, without the two transposes and the padded copy (2026-11-05: ~1.6 ms a step at 20M)."""
        K = self.key_conv; w = self.conv.weight[:, 0, :]                      # [d, K]
        out = x * w[:, K - 1]
        for j in range(K - 1):
            shift = K - 1 - j
            out = out + F.pad(x[:, :-shift], (0, 0, shift, 0)) * w[:, j] if x.size(1) > shift else out
        return out + self.conv.bias

    def decay_value(self):
        return torch.sigmoid(self.decay_logit) if self.learn_decay else self.decay

    def memory(self, x: Tensor, key_src: Tensor | None = None, state: HeadState | None = None, tokens: Tensor | None = None):
        """The head's read. With `state` the head continues an earlier call and returns (output, new HeadState).
        `tokens` [B, T] are the input bytes, needed when the head has an n-gram address."""
        if (self.prior or self.ngram_order) and (state is not None or self.chunk <= 0):
            raise ValueError("a prior or an n-gram address needs the chunked head and no streamed state")
        if self.ngram_order and tokens is None:
            raise ValueError("an n-gram address needs the input bytes")
        self._offset = ngram_hash(tokens, self.ngram_order, self.code.n_key, self.ngram_salt) if self.ngram_order else None
        xk = x if key_src is None else key_src
        tail = None
        if self.key_conv > 1:
            if state is None:
                xk = self.causal_conv(xk)
            else:  # the conv's left context is the previous call's last key_conv - 1 inputs instead of zeros
                src = torch.cat([state.conv_tail.to(xk.dtype), xk], dim=1)
                tail = src[:, -(self.key_conv - 1) :].clone()   # a copy: a view would keep the whole key source of the segment alive
                xk = self.causal_conv(src)[:, self.key_conv - 1 :] if xk.size(1) else xk   # an empty segment has no positions to convolve
        v = self.val(x)
        if state is None:
            total = 0
            for code, out in zip(self.codes, self.outs):
                total = total + out(self._one_head(code, xk, v))
            return total
        if len(self.codes) != 1:
            raise ValueError("a streamed head has exactly one key code")
        y, new = self._one_head(self.codes[0], xk, v, state)
        new.conv_tail = tail
        return self.outs[0](y), new

    def _one_head_chunked(self, code: KeyCode, xk: Tensor, v: Tensor, state: HeadState | None = None):
        # under a whole-model compile the plain function is inlined into that graph; the separately compiled read is
        # for eager use (calling it from inside a traced region would break the graph)
        read = compiled_read() if (state is None and xk.is_cuda and COMPILE_ON and not torch.compiler.is_compiling()) else sparse_memory_read
        return read(code, xk, v, self.chunk, self.decay_value(), self.normalize, self.gate_tau if self.gate else None, self.state_bits, state,
                    prior=self.prior_tables() if getattr(self, "prior", False) else None, idx_offset=getattr(self, "_offset", None))

    def _one_head(self, code: KeyCode, xk: Tensor, v: Tensor, state: HeadState | None = None):
        # the fast-weight sums accumulate in fp32 whatever autocast does to the rest of the model;
        # the state's precision is controlled explicitly by state_bits (fake-quantised between chunks)
        with torch.autocast(device_type=xk.device.type, enabled=False):
            return self._one_head_fp32(code, xk.float(), v.float(), state)

    def _one_head_fp32(self, code: KeyCode, xk: Tensor, v: Tensor, state: HeadState | None = None):
        if self.chunk > 0:
            return self._one_head_chunked(code, xk, v, state)
        if state is not None:
            raise ValueError("a streamed head needs the chunked form (chunk > 0)")
        B, T, _ = xk.shape
        q, k = code(xk)
        if code.kind == "softmax":
            k = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)  # same key_{s-1} shift as the other arms
            scores = q.matmul(k.transpose(1, 2)) / math.sqrt(q.size(-1))
            mask = torch.ones(T, T, device=xk.device, dtype=torch.bool).tril(-1)  # strictly past
            scores = scores.masked_fill(~mask, float("-inf"))
            att = torch.softmax(scores, dim=-1).nan_to_num(0.0)
            return att.matmul(v)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        k_shift = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)  # key_{s-1} at slot s
        scores = q.matmul(k_shift.transpose(1, 2))  # [B, T, T] = q_t · key_{s-1}
        t = torch.arange(T, device=xk.device)
        lag = (t[:, None] - 1 - t[None, :]).float()  # (t-1) - s
        la = code.log_decay(xk)
        if la is None:
            weight = torch.where(lag >= 0, self.decay_value() ** lag.clamp(min=0), torch.zeros_like(lag))
        else:  # data-dependent: prod_{u=s+1}^{t-1} alpha_u = exp(L[t-1] - L[s])
            L = torch.cumsum(la, dim=1)  # [B, T]
            Lt = torch.cat([torch.zeros_like(L[:, :1]), L[:, :-1]], dim=1)  # L[t-1]
            weight = torch.exp((Lt[:, :, None] - L[:, None, :]).clamp(max=0.0)) * (lag >= 0).float()
        weight[..., 0] = 0.0  # slot 0 has no previous key
        w = scores * weight
        mass = w.abs().sum(-1, keepdim=True)
        if self.normalize:  # linear-attention normalizer: average over matches instead of summing
            w = w / (mass + 1e-3)
        out = w.matmul(v)
        if self.gate:
            out = out * (mass / (mass + F.softplus(self.gate_tau)))
        return out

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.memory(self.ln1(x))
        return x + self.ff(self.ln2(x))


class SparseKeyLM(nn.Module):
    def __init__(
        self,
        key: str = "sparse",
        d_model: int = 64,
        n_key: int = 1024,
        claws: int = 4,
        top_k: int = 16,
        theta: float = 1.0,
        n_layers: int = 2,
        d_ff: int = 128,
        decay: float = 0.98,
        vocab_size: int = 256,
        seed: int = 0,
        key_conv: int = 1,
        freeze_conv: bool = False,
        normalize: bool = False,
        signed: bool = False,
        learned_key: int = 64,
    ) -> None:
        super().__init__()
        self.key, self.d_model, self.n_key, self.vocab_size = key, int(d_model), int(n_key), int(vocab_size)
        self.n_out = int(vocab_size)
        self.embed = nn.Embedding(vocab_size, d_model)
        def make(i: int):
            if key == "mixed":
                return [KeyCode("learned", d_model, learned_key, claws, top_k, theta, seed + i),
                        KeyCode("sparse", d_model, n_key, claws, top_k, theta, seed + i, signed)]
            nk = learned_key if key in ("learned", "softmax", "gla") else n_key
            return KeyCode(key, d_model, nk, claws, top_k, theta, seed + i, signed)

        self.blocks = nn.ModuleList(
            [Block(d_model, make(i), decay, d_ff, key_conv, freeze_conv, normalize) for i in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def flops_per_token_recurrent(self) -> int:
        """Per-token cost in the recurrent (inference) form, excluding the softmax arm (O(T))."""
        d, V = self.d_model, self.vocab_size
        total = 2 * d * V
        for b in self.blocks:
            total += 2 * d * d + 4 * d * b.ff[0].out_features + 2 * d * b.key_conv  # value, FFN, conv
            for c in b.codes:
                k = c.active_units()
                key_cost = 2 * c.claws * c.n_key if c.kind == "sparse" else 2 * d * c.n_key
                if c.kind in ("learned", "softmax", "gla"):
                    key_cost *= 2  # key and query
                total += key_cost + 2 * k * d + 2 * k * d + 2 * d * d  # key, read, write, out
        return int(total)

    def forward(self, tokens: Tensor, state=None) -> tuple[Tensor, Tensor]:
        x = self.embed(tokens)
        for b in self.blocks:
            x = b(x)
        return self.decoder(self.ln(x)), tokens.new_zeros(tokens.size(0), 1)
