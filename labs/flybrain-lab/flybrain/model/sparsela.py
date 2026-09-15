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
        kk = min(self.top_k, self.n_key)
        with torch.no_grad():
            parts = []
            for c0 in range(0, x.size(1), self.code_slice):
                z = x[:, c0 : c0 + self.code_slice].matmul(self.proj.t())
                parts.append(F.relu(z - self.theta).topk(kk, dim=-1).indices)
            idx = torch.cat(parts, dim=1)                                  # [B, T, k]
        rows = self.proj[idx]                                              # [B, T, k, d] (frozen)
        z_sel = (x.unsqueeze(2) * rows).sum(-1)                            # [B, T, k]
        val = F.relu(z_sel - self.theta[idx])
        return idx, val

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


class Block(nn.Module):
    def __init__(self, d: int, code, decay: float, d_ff: int, key_conv: int = 1, freeze_conv: bool = False, normalize: bool = False, gate: bool = False):
        super().__init__()
        self.gate = bool(gate)
        self.chunk = 0  # >0: linear-cost chunked form (exact for decay 1 + nonnegative sparse keys)
        if gate:  # match-strength gate: g = mass / (mass + softplus(tau)); zero output when nothing in memory matched
            self.gate_tau = nn.Parameter(torch.zeros(()))
        self.codes = nn.ModuleList(code if isinstance(code, (list, tuple)) else [code])
        self.code = self.codes[0]
        self.decay, self.normalize = float(decay), bool(normalize)
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

    def memory(self, x: Tensor, key_src: Tensor | None = None) -> Tensor:
        xk = x if key_src is None else key_src
        if self.key_conv > 1:
            xk = self.conv(F.pad(xk.transpose(1, 2), (self.key_conv - 1, 0))).transpose(1, 2)
        v = self.val(x)
        total = 0
        for code, out in zip(self.codes, self.outs):
            total = total + out(self._one_head(code, xk, v))
        return total

    def _one_head_chunked(self, code: KeyCode, xk: Tensor, v: Tensor) -> Tensor:
        """Linear-cost index form of the parallel path below, exact for decay 1 + nonnegative sparse keys.
        out_t = g(m_t) * q_t . S_t / (m_t + 1e-3),  S_t = sum_{1<=s<t} k_{s-1} v_s^T,  m_t = q_t . sum_{1<=s<t} k_{s-1}.
        The key has only top_k nonzeros, so reads are gathers and writes are index-adds; the T x T term is chunk-local."""
        if self.decay != 1.0:
            raise ValueError("chunked memory needs decay 1.0")
        B, T, _ = xk.shape; L = self.chunk; d = v.size(-1); n_key = code.n_key
        idx, val = code.sparse_code(xk)                                   # [B, T, k]
        val = val / (val.norm(dim=-1, keepdim=True) + 1e-6)              # q = k = unit-norm sparse code
        k = idx.size(-1)
        # key_{s-1} at slot s (slot 0 reads nothing)
        idx_s = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        val_s = torch.cat([torch.zeros_like(val[:, :1]), val[:, :-1]], dim=1)
        S = xk.new_zeros(B * n_key, d); Ksum = xk.new_zeros(B * n_key)
        boff = (torch.arange(B, device=xk.device) * n_key)[:, None, None]  # [B, 1, 1]
        outs = []
        for c in range(0, T, L):
            iq, vq = idx[:, c : c + L], val[:, c : c + L]                  # queries in chunk
            ik, vk, vc = idx_s[:, c : c + L], val_s[:, c : c + L], v[:, c : c + L]
            Lc = iq.size(1)
            # intra-chunk scores by index equality: sc[t, s] = sum_{i,j} vq[t, i] vk[s, j] [iq[t, i] == ik[s, j]]  (no n_key-sized tensor)
            eq = (iq[:, :, None, :, None] == ik[:, None, :, None, :])       # [B, t, s, i, j]
            sc = (eq * (vq[:, :, None, :, None] * vk[:, None, :, None, :])).sum((-1, -2))  # [B, t, s]
            sc = sc * torch.ones(Lc, Lc, device=xk.device, dtype=torch.bool).tril(-1)
            # inter-chunk reads from the running state
            flat = (iq + boff).reshape(B, Lc * k)                          # [B, Lc*k] rows of S
            Srows = S.index_select(0, flat.reshape(-1)).view(B, Lc, k, d)
            Krows = Ksum.index_select(0, flat.reshape(-1)).view(B, Lc, k)
            num = sc.matmul(vc) + (Srows * vq[..., None]).sum(2)           # [B, Lc, d]
            mass = sc.sum(-1, keepdim=True) + (Krows * vq).sum(-1, keepdim=True)
            out = num
            if self.normalize:
                out = out / (mass + 1e-3)
            if self.gate:
                out = out * (mass / (mass + F.softplus(self.gate_tau)))
            outs.append(out)
            # write this chunk's keys into the state
            wflat = (ik + boff).reshape(-1)
            S = S.index_add(0, wflat, (vk[..., None] * vc[:, :, None, :]).reshape(-1, d))
            Ksum = Ksum.index_add(0, wflat, vk.reshape(-1))
        return torch.cat(outs, dim=1)

    def _one_head(self, code: KeyCode, xk: Tensor, v: Tensor) -> Tensor:
        if self.chunk > 0:
            return self._one_head_chunked(code, xk, v)
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
            weight = torch.where(lag >= 0, self.decay ** lag.clamp(min=0), torch.zeros_like(lag))
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
