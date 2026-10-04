"""Extra input channels for GRUHybrid, all off by default and all zero-initialised, so a model built with one starts as the plain model.

FeatureChannel: for each of K "orders" a predicted byte and a class of how much to trust it (class 0 = no prediction), entering the
network twice. Input side: a learned embedding of the byte and one of the class are summed over orders and added to the byte
embedding (a miss adds only its class-0 embedding). Output side: for each order with a prediction, a learned bonus is added to the
logit of the predicted byte, a linear function of the final hidden state with one output per class; the class selects the output.
The corpus table of scripts/build_corpus_memory.py (class from its confidence counter, `conf_class`) is one source of such
predictions.

The embeddings and the bonus are all additive, so zero-initialising both is not a dead start: the gradient of the loss with respect
to a zero embedding row is the gradient with respect to the input it is added to, which is not zero.

NgramEmbedding: for each order n an embedding row chosen by a fixed hash of the n bytes ending at each window index (computed on the
device from the byte window; an index with fewer than n bytes of history inside the window uses a reserved row), the rows summed over
orders, projected to the model width and added to the byte embedding. The fixed address with a learned table of the hash layers and
hashed n-gram embeddings of the literature, trained by gradient with the rest of the network. The projection starts at zero (the model
starts as the plain model) and the rows at small random values, so that the projection gets a gradient on the first step and the rows
from the second."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

CONF_EDGES = (2, 4, 16)                  # the table's counter: 1 | 2-3 | 4-15 | 16+ (the buckets of scripts/probe_memory_mix.py)
N_CONF_CLASSES = len(CONF_EDGES) + 2     # miss, 1, 2-3, 4-15, 16+


def conf_class(conf: Tensor) -> Tensor:
    """The table's confidence counter (uint8, 0 = a miss) as a class in 0..4: miss, 1, 2-3, 4-15, 16+. Long."""
    c = conf.long()
    return (c > 0).long() + sum((c >= e).long() for e in CONF_EDGES)


def parse_orders(text: str, flag: str) -> tuple[int, ...]:
    """'8,16,32' -> (8, 16, 32): positive integers, no repeats, in the order given (it is the order of the parameters)."""
    try:
        orders = tuple(int(s) for s in str(text).split(","))
    except ValueError:
        raise ValueError(f"{flag} wants a comma list of integers, got {text!r}") from None
    if not orders or min(orders) < 1:
        raise ValueError(f"{flag} orders must be positive integers, got {text!r}")
    if len(set(orders)) != len(orders):
        raise ValueError(f"{flag} orders must be distinct, got {text!r}")
    return orders


class FeatureChannel(nn.Module):
    """One predicted byte and one class per order: input embeddings and an output logit bonus (see the module docstring).
    n_classes counts the miss class: the bonus has n_classes - 1 outputs, for the classes 1 .. n_classes - 1."""

    def __init__(self, d_model: int, d_out: int, n_orders: int, n_classes: int, vocab: int = 256):
        super().__init__()
        self.n_orders, self.n_classes = int(n_orders), int(n_classes)
        self.byte_emb = nn.ModuleList(nn.Embedding(vocab, d_model) for _ in range(n_orders))
        self.class_emb = nn.ModuleList(nn.Embedding(n_classes, d_model) for _ in range(n_orders))
        self.bonus = nn.ModuleList(nn.Linear(d_out, n_classes - 1) for _ in range(n_orders))
        for p in self.parameters():
            nn.init.zeros_(p)

    def embed(self, pred: Tensor, cls: Tensor) -> Tensor:
        """[B, T, d_model] to add to the byte embedding. pred, cls: long [B, T, n_orders]."""
        e = 0
        for o in range(self.n_orders):
            c = cls[..., o]
            e = e + self.byte_emb[o](pred[..., o]) * (c > 0).unsqueeze(-1) + self.class_emb[o](c)
        return e

    def add_bonus(self, logits: Tensor, h: Tensor, pred: Tensor, cls: Tensor) -> Tensor:
        """logits [B, T, V] (float32 out, whatever came in) plus, on the logit of each order's predicted byte, the bonus of that
        order's class: (W h + b)[class - 1] where the order has a prediction, nothing where it does not. Two orders that predict the same
        byte add their bonuses. h: the final hidden state [B, T, d_out]."""
        bonus = []
        for o in range(self.n_orders):
            c = cls[..., o]
            out = self.bonus[o](h).float()                                          # [B, T, n_classes - 1]
            bonus.append(out.gather(-1, (c - 1).clamp_min(0)[..., None])[..., 0] * (c > 0))
        return logits.float().scatter_add(-1, pred, torch.stack(bonus, dim=-1))


# ---- hashed n-gram embeddings --------------------------------------------------------------------------------------------------

NGRAM_ROWS, NGRAM_DIM = 262_144, 64      # defaults of --ngram-rows and --ngram-dim
NGRAM_P = (1 << 31) - 1                  # the hash works modulo this prime, so every product of two residues stays below 2 ** 62 in int64
NGRAM_MULT, NGRAM_SALT = 1_540_483_477, 2_654_435_761
NGRAM_A1, NGRAM_B1, NGRAM_A2 = 1_597_334_677, 1_013_904_223, 1_103_515_245


def ngram_sizes(orders, rows: int) -> tuple[int, ...]:
    """Rows of each order's block, not counting its reserved row: `rows`, except an order whose every n-gram fits (256 ** n <= rows:
    orders 1 and 2 at the default 262,144) is indexed directly by the bytes themselves, with no collisions."""
    return tuple(256 ** n if 256 ** n <= rows else int(rows) for n in orders)


def ngram_index(tokens: Tensor, orders, rows: int) -> Tensor:
    """[B, T, len(orders)] long: for each window index t and order n, the row of the n bytes ending at t inside that order's block of
    ngram_sizes(orders, rows)[k] rows: row size_k, the reserved one, where t < n - 1 (fewer than n bytes in the window). tokens [B, T] long.
    Hashed orders: v = byte + 1, acc = v_t, then acc = (acc * MULT + v_(t-k+1)) mod P for k = 2..n, h = (acc + n * SALT) mod P,
    h = (h * A1 + B1) mod P, h ^= h >> 15, h = (h * A2) mod P, row = h mod size. Only int64 arithmetic below 2 ** 62, so CPU and GPU agree
    exactly; no randomness, no gradient. Only bytes at or before t enter the row at t."""
    B, T = tokens.shape
    orders = tuple(int(n) for n in orders)
    size_of = dict(zip(orders, ngram_sizes(orders, rows)))
    tok = tokens.long()
    v = tok + 1
    pos = torch.arange(T, device=tokens.device)
    acc, direct, found = v, tok, {}
    for n in range(1, max(orders) + 1):
        if n > 1:
            acc = (acc * NGRAM_MULT + F.pad(v, (n - 1, 0))[:, :T]) % NGRAM_P      # + v_(t-n+1), zero where that is before the window
            direct = direct + F.pad(tok, (n - 1, 0))[:, :T] * 256 ** (n - 1)
        if n in size_of:
            if 256 ** n <= rows:
                row = direct
            else:
                h = (acc + n * NGRAM_SALT) % NGRAM_P
                h = (h * NGRAM_A1 + NGRAM_B1) % NGRAM_P
                h = h ^ (h >> 15)
                row = ((h * NGRAM_A2) % NGRAM_P) % size_of[n]
            found[n] = torch.where(pos[None, :] < n - 1, torch.full_like(row, size_of[n]), row)
    return torch.stack([found[n] for n in orders], dim=-1)


class NgramEmbedding(nn.Module):
    """Hashed n-gram embeddings for the orders given: one block of (size + 1) rows of `dim` numbers per order in one table, the rows of
    all orders summed, then a bias-free linear projection to d_model. `sparse`: the table's gradient is a sparse tensor (give it to
    torch.optim.SparseAdam). The table is built from its own generator: the global one is not touched."""

    def __init__(self, orders, rows: int, dim: int, d_model: int, sparse: bool = False, seed: int = 0):
        super().__init__()
        self.orders, self.rows, self.dim, self.sparse = tuple(int(n) for n in orders), int(rows), int(dim), bool(sparse)
        sizes = ngram_sizes(self.orders, self.rows)
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size + 1)                   # a block is its hashed rows then its reserved row
        g = torch.Generator().manual_seed(int(seed) + 4242)
        self.table = nn.Embedding(offsets[-1], self.dim, sparse=self.sparse, _weight=torch.randn(offsets[-1], self.dim, generator=g) * self.dim ** -0.5)
        with torch.random.fork_rng(devices=[]):                      # nn.Linear draws a weight that is zeroed at once: leave the global generator as it was
            self.proj = nn.Linear(self.dim, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.register_buffer("offsets", torch.tensor(offsets[:-1], dtype=torch.long), persistent=False)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens: Tensor) -> Tensor:
        """[B, T, d_model] to add to the byte embedding."""
        B, T = tokens.shape
        idx = ngram_index(tokens, self.orders, self.rows) + self.offsets
        summed = F.embedding_bag(idx.reshape(B * T, -1), self.table.weight, mode="sum", sparse=self.sparse)
        return self.proj(summed.view(B, T, self.dim))


def channel_kwargs(a) -> dict:
    """GRUHybrid keyword arguments for the extra input channels, from a run's recorded arguments (a dict: vars(args) or a checkpoint's
    "args"; keys that are absent or off give nothing, so a legacy checkpoint rebuilds as it always did)."""
    kw: dict = {}
    if a.get("mem_feat"):
        kw["mem_orders"] = parse_orders(a.get("mem_orders") or "8,16,32", "--mem-orders")
    if a.get("ngram_emb"):
        kw.update(ngram_orders=parse_orders(a["ngram_emb"], "--ngram-emb"), ngram_rows=int(a.get("ngram_rows", NGRAM_ROWS)),
                  ngram_dim=int(a.get("ngram_dim", NGRAM_DIM)), ngram_sparse=bool(a.get("ngram_sparse", False)))
    return kw
