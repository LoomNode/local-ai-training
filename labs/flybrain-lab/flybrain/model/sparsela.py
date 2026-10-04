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

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from flybrain.model.seqmixer import signed_expand_matrix
from flybrain.model.state import HeadState

KEY_KINDS = ("sparse", "gausstopk", "gauss", "learned", "softmax", "tied", "mixed", "gla")


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
        idx = self.select(x)                                               # [B, T, k]
        return idx, self.recompute(x, idx)

    def select(self, x: Tensor) -> Tensor:
        """Which k units fire at every position: idx [B, T, k] long. A dense matmul against the frozen projection, in slices of
        positions, without grad (the selection is discrete). Under autocast the matmul follows it unless the caller disabled it."""
        kk = min(self.top_k, self.n_key)
        with torch.no_grad():
            parts = []
            for c0 in range(0, x.size(1), self.code_slice):
                z = x[:, c0 : c0 + self.code_slice].matmul(self.proj.t())
                parts.append(F.relu(z - self.theta).topk(kk, dim=-1).indices)
            idx = torch.cat(parts, dim=1)                                  # [B, T, k]
        return idx

    def recompute(self, x: Tensor, idx: Tensor) -> Tensor:
        """The activations of the selected units, with gradient to x and theta: val [B, T, k] >= 0."""
        rows = self.proj[idx]                                              # [B, T, k, d] (frozen)
        z_sel = (x.unsqueeze(2) * rows).sum(-1)                            # [B, T, k]
        val = F.relu(z_sel - self.theta[idx])
        return val

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


def _normalize_gate(num: Tensor, mass: Tensor, normalize: bool, gate_tau: Tensor | None) -> Tensor:
    """The read's normaliser (average over matches instead of summing) and match-strength gate, for one chunk."""
    out = num
    if normalize:
        out = out / (mass + 1e-3)
    if gate_tau is not None:
        out = out * (mass / (mass + F.softplus(gate_tau)))
    return out


def sparse_memory_read(code: KeyCode, xk: Tensor, v: Tensor, chunk: int, decay: float, normalize: bool, gate_tau: Tensor | None, state_bits: int = 0, state: HeadState | None = None, detach_state: bool = False):
    """Linear-cost index form of the parallel path below, exact for scalar decay in (0, 1] + nonnegative sparse keys.
    out_t = g(m_t) * q_t . S_t / (m_t + 1e-3),  S_t = sum_{1<=s<t} k_{s-1} v_s^T,  m_t = q_t . sum_{1<=s<t} k_{s-1}.
    The key has only top_k nonzeros, so reads are gathers and writes are index-adds; the T x T term is chunk-local.
    With `state` the tables and the previous position's key continue from an earlier call, and the return value is
    (out, new HeadState); the state passed in is not modified. S is carried aged to the first position of the next call.
    `detach_state` (--detach-state): the tables carried from earlier chunks (and the incoming `state`) are constants for the gradient;
    the forward is unchanged. Cut: the gradient from a read in chunk c into the writes (values and key magnitudes) of chunks before c.
    Kept: everything inside a chunk (its intra-chunk scores and its own read of the carried tables, which still reaches the query)."""
    B, T, _ = xk.shape; L = chunk; d = v.size(-1); n_key = code.n_key
    if state is not None and T == 0:  # an empty segment: no output, and the state as it was (copied, so the new state shares nothing with the old)
        cp = lambda t: None if t is None else t.clone()
        return v.new_zeros(B, 0, d), HeadState(S=cp(state.S), Ksum=cp(state.Ksum), last_idx=cp(state.last_idx), last_val=cp(state.last_val), conv_tail=cp(state.conv_tail))
    cdt = torch.float32 if xk.dtype in (torch.bfloat16, torch.float16) else xk.dtype   # the tables and the decay arithmetic are never below fp32 (only matters under --head-amp)
    lam = decay if torch.is_tensor(decay) else torch.tensor(float(decay), device=xk.device, dtype=cdt)
    idx, val = code.sparse_code(xk)                                   # [B, T, k]
    val = val / (val.norm(dim=-1, keepdim=True) + 1e-6)              # q = k = unit-norm sparse code
    k = idx.size(-1)
    # key_{s-1} at slot s (slot 0 reads nothing, unless an earlier call left its last key in `state`)
    if state is not None and state.last_idx is not None:
        idx_s = torch.cat([state.last_idx[:, None], idx[:, :-1]], dim=1)
        val_s = torch.cat([state.last_val[:, None].to(val.dtype), val[:, :-1]], dim=1)
    else:
        idx_s = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        val_s = torch.cat([torch.zeros_like(val[:, :1]), val[:, :-1]], dim=1)
    if state is None:
        S = xk.new_zeros(B * n_key, d, dtype=cdt); Ksum = xk.new_zeros(B * n_key, dtype=cdt)
    else:
        S = state.S.to(cdt).reshape(B * n_key, d); Ksum = state.Ksum.to(cdt).reshape(B * n_key)
        if detach_state:
            S = S.detach(); Ksum = Ksum.detach()
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
        lag = (ar[:, None] - 1 - ar[None, :]).clamp(min=0).to(cdt)               # t-1-s within the chunk
        sc = sc * torch.ones(Lc, Lc, device=xk.device, dtype=torch.bool).tril(-1) * lam ** lag
        age_in = (lam ** ar.to(cdt))[None, :, None]                                 # reads: state aged from chunk start to t
        age_out = (lam ** (Lc - 1 - ar).to(cdt))[None, :, None]                     # writes: aged to the next chunk start
        # inter-chunk reads from the running state
        flat = (iq + boff).reshape(B, Lc * k)                          # [B, Lc*k] rows of S
        Srows = S.index_select(0, flat.reshape(-1)).view(B, Lc, k, d)
        Krows = Ksum.index_select(0, flat.reshape(-1)).view(B, Lc, k)
        num = sc.matmul(vc) + age_in * (Srows * vq[..., None]).sum(2)  # [B, Lc, d]
        mass = sc.sum(-1, keepdim=True) + age_in * (Krows * vq).sum(-1, keepdim=True)
        outs.append(_normalize_gate(num, mass, normalize, gate_tau))
        # write this chunk's keys into the state
        wflat = (ik + boff).reshape(-1)
        vk_aged = vk * age_out
        S = S * lam ** Lc + torch.zeros_like(S).index_add(0, wflat, (vk_aged[..., None] * vc[:, :, None, :]).reshape(-1, d))
        Ksum = Ksum * lam ** Lc + torch.zeros_like(Ksum).index_add(0, wflat, vk_aged.reshape(-1))
        if state_bits:  # integer memory state: quantise what is carried between chunks
            S = fake_quant(S, state_bits); Ksum = fake_quant(Ksum[None], state_bits)[0]
        if detach_state:  # the next chunk reads these tables as constants: no gradient flows back into this chunk's writes through them
            S = S.detach(); Ksum = Ksum.detach()
    out = torch.cat(outs, dim=1)
    if state is None:
        return out
    # the carried keys are copies: a view would keep the whole per-call idx / val tensor alive for as long as the state is held
    return out, HeadState(S=S.reshape(B, n_key, d), Ksum=Ksum.reshape(B, n_key), last_idx=idx[:, -1].clone(), last_val=val[:, -1].clone(), conv_tail=state.conv_tail)


class Block(nn.Module):
    def __init__(self, d: int, code, decay: float, d_ff: int, key_conv: int = 1, freeze_conv: bool = False, normalize: bool = False, gate: bool = False, learn_decay: bool = False):
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
        self.head_amp = False      # --head-amp: no fp32 pin, the head follows the surrounding autocast (see _one_head)
        self.detach_state = False  # --detach-state: the tables carried between chunks are constants for the gradient (see sparse_memory_read)
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

    def decay_value(self):
        return torch.sigmoid(self.decay_logit) if self.learn_decay else self.decay

    def memory(self, x: Tensor, key_src: Tensor | None = None, state: HeadState | None = None):
        """The head's read. With `state` the head continues an earlier call and returns (output, new HeadState)."""
        xk = x if key_src is None else key_src
        tail = None
        if self.key_conv > 1:
            if state is None:
                xk = self.conv(F.pad(xk.transpose(1, 2), (self.key_conv - 1, 0))).transpose(1, 2)
            else:  # the conv's left context is the previous call's last key_conv - 1 inputs instead of zeros
                src = torch.cat([state.conv_tail.to(xk.dtype), xk], dim=1)
                tail = src[:, -(self.key_conv - 1) :].clone()   # a copy: a view would keep the whole key source of the segment alive
                xk = self.conv(src.transpose(1, 2)).transpose(1, 2) if xk.size(1) else xk   # an empty segment has no positions to convolve
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
        return sparse_memory_read(code, xk, v, self.chunk, self.decay_value(), self.normalize, self.gate_tau if self.gate else None, self.state_bits, state, self.detach_state)

    def _one_head(self, code: KeyCode, xk: Tensor, v: Tensor, state: HeadState | None = None):
        if self.head_amp:
            # --head-amp: no fp32 pin. The head's matmuls (the top-k selection matmul included) follow the surrounding autocast; the
            # tables S and Ksum and the decay arithmetic stay at least fp32 (sparse_memory_read). Without autocast this is the same as below.
            return self._one_head_fp32(code, xk, v, state)
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
