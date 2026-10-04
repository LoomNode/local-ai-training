"""Extra input channels for GRUHybrid, all off by default and all zero-initialised, so a model built with one starts as the plain model.

FeatureChannel: for each of K "orders" a predicted byte and a class of how much to trust it (class 0 = no prediction), entering the
network twice. Input side: a learned embedding of the byte and one of the class are summed over orders and added to the byte
embedding (a miss adds only its class-0 embedding). Output side: for each order with a prediction, a learned bonus is added to the
logit of the predicted byte, a linear function of the final hidden state with one output per class; the class selects the output.
The corpus table of scripts/build_corpus_memory.py (class from its confidence counter, `conf_class`) is one source of such
predictions.

The embeddings and the bonus are all additive, so zero-initialising both is not a dead start: the gradient of the loss with respect
to a zero embedding row is the gradient with respect to the input it is added to, which is not zero."""

from __future__ import annotations

import torch
from torch import Tensor, nn

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


def channel_kwargs(a) -> dict:
    """GRUHybrid keyword arguments for the extra input channels, from a run's recorded arguments (a dict: vars(args) or a checkpoint's
    "args"; keys that are absent or off give nothing, so a legacy checkpoint rebuilds as it always did)."""
    kw: dict = {}
    if a.get("mem_feat"):
        kw["mem_orders"] = parse_orders(a.get("mem_orders") or "8,16,32", "--mem-orders")
    return kw
