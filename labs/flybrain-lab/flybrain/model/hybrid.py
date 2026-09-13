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

    def forward(self, x: Tensor) -> Tensor:
        u = self.ln1(x)
        z = torch.sigmoid(self.z(u)); cand = self.h(u)
        a = (1 - z).clamp_min(1e-4); b = z * cand
        # chunked closed form: h_t = A_t (h_0 + sum_{s<=t} b_s / A_s), A_t = prod_{u<=t} a_u, in float64 per chunk
        B, T, d = u.shape; L = 32
        outs = []; h0 = torch.zeros(B, d, dtype=torch.float64, device=u.device)
        for c in range(0, T, L):
            ac = a[:, c : c + L].double(); bc = b[:, c : c + L].double()
            A = torch.exp(torch.cumsum(torch.log(ac), dim=1))
            hc = A * (h0[:, None, :] + torch.cumsum(bc / A, dim=1))
            outs.append(hc); h0 = hc[:, -1]
        x = x + torch.cat(outs, dim=1).to(x.dtype)
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


WHERE = ("pre", "post", "none")
BACKBONES = ("gru", "mingru", "gla")


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
    ) -> None:
        super().__init__()
        if where not in WHERE or backbone not in BACKBONES:
            raise ValueError("bad where/backbone")
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
        else:
            self.core = GLABackbone(d_model, n_layers, d_ff, learned_key, key_conv); out_dim = d_model
        self.hidden = out_dim
        self.decoder = nn.Linear(out_dim, vocab_size)
        dim = d_model if where == "pre" else out_dim
        nk = learned_key if key in ("learned", "softmax") else n_key
        if where != "none":
            self.mem = Block(dim, KeyCode(key, dim, nk, claws, top_k, theta, seed), decay, d_ff=1, key_conv=key_conv, normalize=normalize)

    def persistent_state_scalars(self, batch: int = 1) -> int:
        return int(batch * self.hidden)

    def flops_per_step(self, batch: int = 1) -> int:
        e, h, v = self.d_model, self.hidden, self.vocab_size
        base = 2 * 3 * (e * h + h * h) + 2 * h * v if self.backbone == "gru" else 2 * e * v + 4 * len(self.core.layers if hasattr(self.core, "layers") else self.core.blocks) * (e * e + 2 * e * 256)
        if self.where == "none":
            return int(batch * base)
        c = self.mem.code; dim = e if self.where == "pre" else h; k = c.active_units()
        key_cost = 2 * c.claws * c.n_key if c.kind == "sparse" else 2 * dim * c.n_key * (2 if c.kind in ("learned", "softmax") else 1)
        return int(batch * (base + key_cost + 2 * dim * dim * 2 + 4 * k * dim + 2 * dim * self.mem.key_conv))

    def forward(self, tokens: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        e = self.embed(tokens)
        if self.where == "pre":
            z = self.mem.ln1(e)
            e = e + self.mem.memory(z, z.roll(1, dims=0) if self.shuffle_keys else None)
        if self.backbone == "gru":
            out, hstate = self.gru(e, state)
        else:
            out = self.core(e); hstate = out[:, -1]
        if self.where == "post":
            z = self.mem.ln1(out)
            out = out + self.mem.memory(z, z.roll(1, dims=0) if self.shuffle_keys else None)
        return self.decoder(out), hstate
