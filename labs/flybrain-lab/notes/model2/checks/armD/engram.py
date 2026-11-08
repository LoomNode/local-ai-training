"""Arm D (2026-11-06): an Engram-style hashed n-gram memory bolted onto a finished Qwen3.5 student. The preceding 2- and 3-grams are
hashed by --heads multiplicative hashes per order into prime-sized embedding tables; the concatenated rows go through a zero-initialised
up-projection, a short depthwise causal convolution, and a sigmoid gate conditioned on the hidden state, and are added to the residual
stream after decoder layer --after (DeepSeek Engram: layer 2 was the best single insertion point). Zero-initialised, so the student starts
unchanged. attach() installs it with forward hooks: a pre-hook on the text model captures the input ids, a hook on the layer adds the
memory. Numbers only."""
import torch, torch.nn as nn, torch.nn.functional as F

PRIMES = [1000000007, 998244353, 1000000009, 999999937, 1000000021, 999999929, 1000000033, 999999893]

class Engram(nn.Module):
    def __init__(self, d_model: int, rows: int = 500009, dim: int = 64, heads: int = 4, orders=(2, 3), kernel: int = 4):
        super().__init__(); self.rows, self.dim, self.heads, self.orders, self.kernel = rows, dim, heads, tuple(orders), kernel
        self.tables = nn.ModuleList([nn.Embedding(rows, dim) for _ in range(heads * len(orders))])
        for t in self.tables: nn.init.normal_(t.weight, std=0.02)
        self.up = nn.Linear(dim * heads * len(orders), d_model, bias=False); nn.init.zeros_(self.up.weight)
        self.conv = nn.Conv1d(d_model, d_model, kernel, groups=d_model, bias=False); nn.init.zeros_(self.conv.weight); self.conv.weight.data[:, 0, -1] = 1.0
        self.gate = nn.Linear(d_model, d_model); nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.ids = None
    def hashes(self, ids: torch.Tensor):
        """[B, L] token ids -> list over (order, head) of [B, L] row indices; the n-gram ending at each position"""
        out = []; B, L = ids.shape
        for n in self.orders:
            for k in range(self.heads):
                m = PRIMES[(k + n) % len(PRIMES)]; h = torch.zeros_like(ids)
                for j in range(n - 1, -1, -1):                            # oldest token first
                    t = F.pad(ids, (j, 0), value=0)[:, :L] if j else ids
                    h = (h * m + t + 1) % self.rows
                out.append(h)
        return out
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        ids = self.ids; assert ids is not None and ids.shape[:2] == h.shape[:2]
        e = torch.cat([t(x) for t, x in zip(self.tables, self.hashes(ids))], -1)   # [B, L, heads*orders*dim]
        v = self.up(e.to(self.up.weight.dtype))                                    # [B, L, d]
        v = self.conv(F.pad(v.transpose(1, 2), (self.kernel - 1, 0))).transpose(1, 2)   # causal depthwise conv
        g = torch.sigmoid(self.gate(h.to(self.gate.weight.dtype)))
        return h + (g * v).to(h.dtype)
    def config(self): return {"rows": self.rows, "dim": self.dim, "heads": self.heads, "orders": list(self.orders), "kernel": self.kernel}

def attach(text_model: nn.Module, eng: Engram, after: int = 2):
    """install on a Qwen3_5TextModel: ids captured on entry, the memory added to the output of decoder layer `after` (1-based)"""
    def pre(mod, args, kwargs):
        eng.ids = kwargs.get("input_ids", args[0] if args else None)
    def post(mod, args, out):
        if isinstance(out, tuple): return (eng(out[0]),) + tuple(out[1:])
        return eng(out)
    text_model.register_forward_pre_hook(pre, with_kwargs=True); text_model.layers[after - 1].register_forward_hook(post)

def load(path: str, d_model: int, device="cuda") -> Engram:
    st = torch.load(path, map_location=device); eng = Engram(d_model, **st["config"]).to(device); eng.load_state_dict(st["state"]); return eng
