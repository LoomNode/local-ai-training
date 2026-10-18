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
from the second.

match_features: the in-window match model. For each order n and window index t >= n - 1, the most recent earlier index s < t whose n
bytes ending at s equal the n bytes ending at t gives the byte at s + 1 (already known at t) and a class from how many earlier indices
have that context (1, 2-3, 4+); no such index is a miss. Computed on the device from the byte window with no gradient and no table
anywhere: hash every position's n bytes, sort (key, index) per row, read the previous occurrence from the sorted order. It enters the
network through a FeatureChannel exactly as the corpus table's prediction does (separate parameters). Two word orders (ids 900 and 902)
key on the identifier runs of the window instead of the last n bytes: the word prefix so far, and the last two words plus the prefix."""

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


class FusedFeatureChannel(nn.Module):
    """FeatureChannel's computation without its loop over orders: the same parameters in one byte table of n_orders * vocab rows, one
    class table of n_orders * n_classes rows and one bonus Linear with n_orders * (n_classes - 1) outputs, read with one embedding_bag
    each (the sum over orders is the bag) and one gather. The same numbers up to float summation order; built the same way (zeros).
    With 18 orders the loop was 11 % of a training step. `from_channel` copies a FeatureChannel's weights into this layout."""

    def __init__(self, d_model: int, d_out: int, n_orders: int, n_classes: int, vocab: int = 256):
        super().__init__()
        self.n_orders, self.n_classes, self.vocab = int(n_orders), int(n_classes), int(vocab)
        self.byte_emb = nn.Embedding(self.n_orders * self.vocab, d_model)
        self.class_emb = nn.Embedding(self.n_orders * self.n_classes, d_model)
        self.bonus = nn.Linear(d_out, self.n_orders * (self.n_classes - 1))
        for p in self.parameters():
            nn.init.zeros_(p)
        o = torch.arange(self.n_orders)
        self.register_buffer("byte_off", o * self.vocab, persistent=False)             # row offsets of order o in each table
        self.register_buffer("class_off", o * self.n_classes, persistent=False)
        self.register_buffer("bonus_off", o * (self.n_classes - 1), persistent=False)

    @classmethod
    def from_channel(cls, ch: FeatureChannel, d_out: int) -> "FusedFeatureChannel":
        """A fused copy of `ch` (its parameters rearranged, nothing else)."""
        d_model = ch.byte_emb[0].weight.size(1)
        vocab = ch.byte_emb[0].weight.size(0)
        m = cls(d_model, d_out, ch.n_orders, ch.n_classes, vocab)
        with torch.no_grad():
            m.byte_emb.weight.copy_(torch.cat([e.weight for e in ch.byte_emb]))
            m.class_emb.weight.copy_(torch.cat([e.weight for e in ch.class_emb]))
            m.bonus.weight.copy_(torch.cat([b.weight for b in ch.bonus]))
            m.bonus.bias.copy_(torch.cat([b.bias for b in ch.bonus]))
        return m

    def embed(self, pred: Tensor, cls: Tensor) -> Tensor:
        """[B, T, d_model] to add to the byte embedding. pred, cls: long [B, T, n_orders]."""
        B, T, K = pred.shape
        w = self.byte_emb.weight
        e = F.embedding_bag((pred + self.byte_off).view(B * T, K), w, mode="sum", per_sample_weights=(cls > 0).to(w.dtype).view(B * T, K))
        e = e + F.embedding_bag((cls + self.class_off).view(B * T, K), self.class_emb.weight, mode="sum")
        return e.view(B, T, -1)

    def add_bonus(self, logits: Tensor, h: Tensor, pred: Tensor, cls: Tensor) -> Tensor:
        """As FeatureChannel.add_bonus: (W h + b)[class - 1] of each order on the logit of its predicted byte, where it has one."""
        out = self.bonus(h).float()                                                     # [B, T, n_orders * (n_classes - 1)]
        bonus = out.gather(-1, (cls - 1).clamp_min(0) + self.bonus_off) * (cls > 0)      # [B, T, n_orders]
        return logits.float().scatter_add(-1, pred, bonus)


class SharedFeatureChannel(nn.Module):
    """The table channel for a large vocabulary (model #2 on tokens): the predicted token of each order is embedded with the model's own
    token embedding (shared; a reference, not a submodule, so the parameter stays the model's), scaled by a learned per-order gate (zero
    at the start), plus a per-order class embedding (zeros); no per-order vocabulary table (18 orders x 32k x 512 would outweigh the
    transformer). The output bonus is FusedFeatureChannel's: per order and class, on the logit of the predicted token. Zero-initialised
    gate, class table and bonus: the model starts as the plain model; the gate's gradient is the inner product of the input gradient
    with the token embedding, which is not zero."""

    def __init__(self, d_model: int, d_out: int, n_orders: int, n_classes: int, embed: nn.Embedding):
        super().__init__()
        self.n_orders, self.n_classes = int(n_orders), int(n_classes)
        self._embed = [embed]
        self.gate = nn.Parameter(torch.zeros(self.n_orders))
        self.class_emb = nn.Embedding(self.n_orders * self.n_classes, d_model)
        self.bonus = nn.Linear(d_out, self.n_orders * (self.n_classes - 1))
        nn.init.zeros_(self.class_emb.weight); nn.init.zeros_(self.bonus.weight); nn.init.zeros_(self.bonus.bias)
        o = torch.arange(self.n_orders)
        self.register_buffer("class_off", o * self.n_classes, persistent=False)
        self.register_buffer("bonus_off", o * (self.n_classes - 1), persistent=False)

    @property
    def vocab(self) -> int:
        return int(self._embed[0].weight.size(0))

    def embed(self, pred: Tensor, cls: Tensor) -> Tensor:
        """[B, T, d_model] to add to the token embedding. pred, cls: long [B, T, n_orders]."""
        B, T, K = pred.shape
        w = self._embed[0].weight
        weights = (self.gate.to(w.dtype)[None, None, :] * (cls > 0).to(w.dtype)).reshape(B * T, K)
        e = F.embedding_bag(pred.reshape(B * T, K), w, mode="sum", per_sample_weights=weights)
        e = e + F.embedding_bag((cls + self.class_off).reshape(B * T, K), self.class_emb.weight, mode="sum")
        return e.view(B, T, -1)

    def add_bonus(self, logits: Tensor, h: Tensor, pred: Tensor, cls: Tensor) -> Tensor:
        out = self.bonus(h).float()
        bonus = out.gather(-1, (cls - 1).clamp_min(0) + self.bonus_off) * (cls > 0)
        return logits.float().scatter_add(-1, pred, bonus)


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
            if 256 ** n <= rows:                                                  # the direct row is only read while every n-gram fits the rows;
                direct = direct + F.pad(tok, (n - 1, 0))[:, :T] * 256 ** (n - 1)  # past that 256 ** (n - 1) would leave int64 at order 9
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


# ---- in-window match features ---------------------------------------------------------------------------------------------------

MATCH_CLASSES = 4                       # miss, 1, 2-3, 4+ earlier occurrences of the context in the window
MATCH_P = (1 << 31) - 1
MATCH_M1, MATCH_M2 = 1_540_483_477, 1_947_135_679
MATCH_KEY_BITS = 62                     # the key of an n-gram is two independent 31-bit polynomial hashes side by side: 62 bits
MATCH_WORD_PREFIX, MATCH_WORD_PAIR = 900, 902            # the two word orders (reserved ids: parse_orders and --match-feat take them as they are)
MATCH_WORD_ORDERS = (MATCH_WORD_PREFIX, MATCH_WORD_PAIR)
MATCH_MAX_BYTE_ORDER = 64
MATCH_WORD_MIX = ((1_366_822_397, 1_755_094_927, 1_590_157_987),   # per 31-bit half: odd multipliers of (the word before the last, the last word,
                  (1_707_041_821, 1_228_776_599, 1_685_489_703))   # the prefix) in the key of order 902; no relation with coefficients in +-300 mod P
MATCH_ID_RANGES = ((48, 57), (65, 90), (97, 122), (95, 95))        # identifier bytes [0-9A-Za-z_]: the words of orders 900 and 902


def _match_previous(key: Tensor, tokens: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
    """(pred, cls) [B, T] for one order from its keys [B, T] (a miss carries a key of its own, -(t + 1), below every real key): a stable
    sort of (key, index) per row; in a run of equal keys the indices ascend, so the element before an index in the run is its latest
    earlier occurrence and its position in the run is the number of earlier ones. pred is the token after that occurrence, cls the class
    of the count (1, 2-3, 4+), both 0 at a miss."""
    B, T = key.shape
    sk, si = torch.sort(key, dim=1, stable=True)
    same = torch.cat([torch.zeros_like(sk[:, :1], dtype=torch.bool), sk[:, 1:] == sk[:, :-1]], dim=1)   # same key as the sorted element before it
    before = torch.cat([torch.zeros_like(si[:, :1]), si[:, :-1]], dim=1)      # the index of that element
    run_start = torch.cummax(torch.where(same, 0, pos.expand(B, T)), dim=1).values
    prev = torch.empty_like(si).scatter_(1, si, torch.where(same, before, -1))                        # previous occurrence of each index
    count = torch.empty_like(si).scatter_(1, si, pos.expand(B, T) - run_start)                        # earlier occurrences of each index
    hit = prev >= 0
    pred = torch.where(hit, tokens.long().gather(1, (prev + 1).clamp_min(0)), 0)
    cls = torch.where(hit, 1 + (count >= 2).long() + (count >= 4).long(), 0)
    return pred, cls


def _match_powers(bases: tuple[int, ...], pos: Tensor) -> Tensor:
    """[len(bases), T]: base ** t mod MATCH_P at every window index t, from the bits of t (log2 T steps of tensor work, no loop over t)."""
    p = pos.reshape(-1)
    T = p.numel()
    out = torch.ones(len(bases), T, dtype=torch.long, device=pos.device)
    i = 0
    while (1 << i) < T:
        sq = torch.tensor([pow(b, 1 << i, MATCH_P) for b in bases], dtype=torch.long, device=pos.device)[:, None]
        out = torch.where(((p >> i) & 1).bool(), out * sq % MATCH_P, out)
        i += 1
    return out


def _identifier_runs(tokens: Tensor, pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The identifier runs of a window, for the word orders. isid [B, T] bool: the token at t is an identifier byte (MATCH_ID_RANGES);
    isid_prev [B, T]: the same for t - 1 (False at t = 0); H [2, B, T]: the two polynomial hashes (multipliers MATCH_M1, MATCH_M2, mod
    MATCH_P) of the run so far, the maximal run of identifier bytes ending at t and including t, with the latest byte at power 0, so equal
    runs give equal hashes wherever they stand. Along the whole row W_t = sum_{j <= t} v_j M^-j (M^-1 by Fermat, P prime; a cumulative sum of
    T terms below 2^31, exact in int64 for T < 2^32), and the run from its start r to t is (W_t - W_{r-1}) M^t = sum_{j=r..t} v_j M^(t-j).
    The run start is a cummax over the indices where a run begins. H is meaningless (and never read) at a non-identifier byte."""
    B, T = tokens.shape
    x = tokens.long()
    isid = (x >= MATCH_ID_RANGES[0][0]) & (x <= MATCH_ID_RANGES[0][1])
    for lo, hi in MATCH_ID_RANGES[1:]:
        isid = isid | ((x >= lo) & (x <= hi))
    isid_prev = torch.cat([torch.zeros_like(isid[:, :1]), isid[:, :-1]], dim=1)
    start = torch.cummax(torch.where(isid & isid_prev, 0, pos.expand(B, T)), dim=1).values          # the first index of the run containing t
    bases = (MATCH_M1, MATCH_M2)
    powers = _match_powers(bases, pos)[:, None, :]                                                    # M^t
    ipowers = _match_powers(tuple(pow(m, MATCH_P - 2, MATCH_P) for m in bases), pos)[:, None, :]     # M^-t
    W = torch.cumsum((x[None] + 1) * ipowers % MATCH_P, dim=2) % MATCH_P
    W_before = F.pad(W, (1, 0)).gather(2, start[None].expand(2, B, T))                                # W_{r-1}, with W_{-1} = 0
    H = (((W - W_before) % MATCH_P) * powers) % MATCH_P
    return isid, isid_prev, H


def match_features(tokens: Tensor, orders) -> tuple[Tensor, Tensor]:
    """(pred, cls), long [B, T, len(orders)]: for each window index t and order n, with t >= n - 1, the byte at s + 1 for the most recent
    earlier index s < t whose n bytes ending at s are the n bytes ending at t (s + 1 <= t, so it is already known), and a class by the
    number of earlier indices with that context: 1 for one, 2 for two or three, 3 for four or more. A miss (0, 0) where there is none,
    and where t < n - 1. This is the feature for predicting the byte at t + 1. Keys: two polynomial hashes mod 2 ** 31 - 1 with different
    multipliers, 31 bits each, side by side (62 bits), so two different n-grams of one window of 2,048 bytes share a key with probability
    about 2,048 ** 2 / 2 / 2 ** 62 = 2 ** -41, about 5 in 10 ** 13 windows. The previous occurrence comes from a stable sort of (key, index) per row:
    in a run of equal keys the indices ascend, so the element before an index in the run is its latest earlier occurrence and its
    position in the run is the number of earlier ones. Integer operations only: no gradient, no randomness, no value-dependent branch.

    Byte orders are 1 .. MATCH_MAX_BYTE_ORDER. Two word orders, on the identifier runs of the window (maximal runs of [0-9A-Za-z_]):
    900 (MATCH_WORD_PREFIX), the dictionary in the window: the context at t is the run so far, the identifier run ending at t and
      including t (a miss where t is not an identifier byte); the prediction is what followed this prefix the last time it was typed
      (s may lie inside a longer word, and inside the current one: "pyt" of "python python").
    902 (MATCH_WORD_PAIR), the last two words plus the prefix: the context at t is (the completed run before the last, the last completed
      run, the run so far), where a completed run is one a non-identifier byte has closed at an index <= t and the run so far is empty at
      a non-identifier byte (not a miss: there the order predicts the first byte of the next word from the last two words). Other
      non-identifier bytes are only separators ("foo bar" and "foo, bar" give the same words). A miss where fewer than two completed
      runs precede t. The key of each half mixes the three run hashes with the multipliers of MATCH_WORD_MIX mod P.
    Both carry the same classes and the same miss convention as the byte orders."""
    B, T = tokens.shape
    orders = tuple(int(n) for n in orders)
    bad = [n for n in orders if not (1 <= n <= MATCH_MAX_BYTE_ORDER or n in MATCH_WORD_ORDERS)]
    if bad:
        raise ValueError(f"match orders are 1..{MATCH_MAX_BYTE_ORDER} (bytes) or {MATCH_WORD_ORDERS} (words), got {bad}")
    v = tokens.long() + 1
    pos = torch.arange(T, device=tokens.device)[None, :]
    found = {}
    byte_orders = [n for n in orders if n not in MATCH_WORD_ORDERS]
    if byte_orders:
        mult = torch.tensor([MATCH_M1, MATCH_M2], dtype=torch.long, device=tokens.device).view(2, 1, 1)
        acc = v[None].expand(2, B, T)
        for n in range(1, max(byte_orders) + 1):
            if n > 1:
                acc = (acc * mult + F.pad(v, (n - 1, 0))[:, :T][None]) % MATCH_P          # + v_(t-n+1): the key of the n bytes ending at t
            if n in byte_orders:
                key = torch.where(pos >= n - 1, acc[0] * (1 << 31) + acc[1], -(pos + 1))  # no full context: a key of its own, below every real key
                found[n] = _match_previous(key, tokens, pos)
    if any(n in MATCH_WORD_ORDERS for n in orders):
        isid, isid_prev, H = _identifier_runs(tokens, pos)
        if MATCH_WORD_PREFIX in orders:
            key = torch.where(isid, H[0] * (1 << 31) + H[1], -(pos + 1))
            found[MATCH_WORD_PREFIX] = _match_previous(key, tokens, pos)
        if MATCH_WORD_PAIR in orders:
            ends = torch.cummax(torch.where(isid_prev & ~isid, pos.expand(B, T), -1), dim=1).values   # the separator that closed the latest completed run at or before t (-1: none)
            e2 = ends.gather(1, (ends - 1).clamp_min(0))                                             # the one before it (ends[0] is -1: nothing closes at t = 0)
            w1 = H.gather(2, (ends - 1).clamp_min(0)[None].expand(2, B, T))                          # the last completed run: the run so far at its last byte
            w2 = H.gather(2, (e2 - 1).clamp_min(0)[None].expand(2, B, T))
            pre = torch.where(isid[None], H, 0)                                                      # the run so far, 0 = empty
            mix = torch.tensor(MATCH_WORD_MIX, dtype=torch.long, device=tokens.device)[:, :, None, None]
            kh = (w2 * mix[:, 0] % MATCH_P + w1 * mix[:, 1] % MATCH_P + pre * mix[:, 2] % MATCH_P) % MATCH_P
            key = torch.where(e2 >= 0, kh[0] * (1 << 31) + kh[1], -(pos + 1))
            found[MATCH_WORD_PAIR] = _match_previous(key, tokens, pos)
    return torch.stack([found[n][0] for n in orders], dim=-1), torch.stack([found[n][1] for n in orders], dim=-1)


def channel_kwargs(a) -> dict:
    """GRUHybrid keyword arguments for the extra input channels, from a run's recorded arguments (a dict: vars(args) or a checkpoint's
    "args"; keys that are absent or off give nothing, so a legacy checkpoint rebuilds as it always did)."""
    kw: dict = {}
    if a.get("mem_feat"):
        kw["mem_orders"] = parse_orders(a.get("mem_orders") or "8,16,32", "--mem-orders")
    if a.get("ngram_emb"):
        kw.update(ngram_orders=parse_orders(a["ngram_emb"], "--ngram-emb"), ngram_rows=int(a.get("ngram_rows", NGRAM_ROWS)),
                  ngram_dim=int(a.get("ngram_dim", NGRAM_DIM)), ngram_sparse=bool(a.get("ngram_sparse", False)))
    if a.get("match_feat"):
        kw["match_orders"] = parse_orders(a["match_feat"], "--match-feat")
    if a.get("fused_channels"):
        kw["fused_channels"] = True
    # model #2 on tokens: the vocabulary, the value orders (counters served as values, not predictions) and the shared-embedding channel;
    # absent or at their defaults (bytes) they give nothing, so a byte checkpoint rebuilds as it always did
    if int(a.get("vocab", 256) or 256) != 256:
        kw["vocab_size"] = int(a["vocab"])
    if a.get("mem_value_orders"):
        kw["mem_value_orders"] = parse_orders(a["mem_value_orders"], "--mem-value-orders")
    if a.get("shared_table"):
        kw["shared_table"] = True
    return kw
