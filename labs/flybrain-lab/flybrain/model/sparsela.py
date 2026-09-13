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

KEY_KINDS = ("sparse", "gausstopk", "gauss", "learned", "softmax")


class KeyCode(nn.Module):
    def __init__(self, kind: str, d: int, n_key: int, claws: int, top_k: int, theta: float, seed: int):
        super().__init__()
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
        else:  # learned / softmax
            self.key = nn.Linear(d, n_key, bias=False)
            self.query = nn.Linear(d, n_key, bias=False)

    def active_units(self) -> int:
        return min(self.top_k, self.n_key) if self.kind in ("sparse", "gausstopk") else self.n_key

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (query, key), each [B, T, n_key]."""
        if self.kind in ("learned", "softmax"):
            return self.query(x), self.key(x)
        z = x.matmul(self.proj.t())
        if self.kind == "gauss":
            k = z
        else:
            c = F.relu(z - self.theta)
            kk = min(self.top_k, self.n_key)
            if 0 < kk < self.n_key:
                idx = c.topk(kk, dim=-1).indices
                c = c * torch.zeros_like(c).scatter_(-1, idx, 1.0)
            k = c
        return k, k


class Block(nn.Module):
    def __init__(self, d: int, code: KeyCode, decay: float, d_ff: int, key_conv: int = 1):
        super().__init__()
        self.code, self.decay = code, float(decay)
        self.key_conv = int(key_conv)
        if key_conv > 1:  # causal depthwise conv so the key can see the last `key_conv` tokens
            self.conv = nn.Conv1d(d, d, key_conv, groups=d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.val = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def memory(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        xk = x
        if self.key_conv > 1:
            xk = self.conv(F.pad(x.transpose(1, 2), (self.key_conv - 1, 0))).transpose(1, 2)
        q, k = self.code(xk)
        v = self.val(x)
        if self.code.kind == "softmax":
            k = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)  # same key_{s-1} shift as the other arms
            scores = q.matmul(k.transpose(1, 2)) / math.sqrt(q.size(-1))
            mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril(-1)  # strictly past
            scores = scores.masked_fill(~mask, float("-inf"))
            att = torch.softmax(scores, dim=-1).nan_to_num(0.0)
            return att.matmul(v)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        k_shift = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)  # key_{s-1} at slot s
        scores = q.matmul(k_shift.transpose(1, 2))  # [B, T, T] = q_t · key_{s-1}
        t = torch.arange(T, device=x.device)
        lag = (t[:, None] - 1 - t[None, :]).float()  # (t-1) - s
        weight = torch.where(lag >= 0, self.decay ** lag.clamp(min=0), torch.zeros_like(lag))
        weight[:, 0] = 0.0  # slot 0 has no previous key
        return (scores * weight).matmul(v)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.out(self.memory(self.ln1(x)))
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
    ) -> None:
        super().__init__()
        self.key, self.d_model, self.n_key, self.vocab_size = key, int(d_model), int(n_key), int(vocab_size)
        self.n_out = int(vocab_size)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, KeyCode(key, d_model, n_key, claws, top_k, theta, seed + i), decay, d_ff, key_conv) for i in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def flops_per_token_recurrent(self) -> int:
        """Per-token cost in the recurrent (inference) form, excluding the softmax arm (O(T))."""
        d, V = self.d_model, self.vocab_size
        total = 2 * d * V
        for b in self.blocks:
            c = b.code
            k = c.active_units()
            key_cost = 2 * c.claws * c.n_key if c.kind == "sparse" else 2 * d * c.n_key
            if c.kind in ("learned", "softmax"):
                key_cost *= 2  # key and query
            total += key_cost + 2 * d * d + 2 * k * d + 2 * k * d + 2 * d * d + 4 * d * b.ff[0].out_features + 2 * d * b.key_conv
        return int(total)

    def forward(self, tokens: Tensor, state=None) -> tuple[Tensor, Tensor]:
        x = self.embed(tokens)
        for b in self.blocks:
            x = b(x)
        return self.decoder(self.ln(x)), tokens.new_zeros(tokens.size(0), 1)
