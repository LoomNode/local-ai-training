"""Agent state: everything a GRUHybrid needs to continue a byte stream where it stopped."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass
class HeadState:
    """One memory head. S and Ksum are what the head has learned; the rest is what the next byte needs."""

    S: Tensor                         # [B, n_key, d] fast-weight table
    Ksum: Tensor                      # [B, n_key] key mass per slot
    last_idx: Tensor | None = None    # [B, k] address of the previous position (None before the first byte of a document)
    last_val: Tensor | None = None    # [B, k] its unit-normalised values
    conv_tail: Tensor | None = None   # [B, key_conv - 1, d] inputs the causal key conv still needs (None when key_conv == 1)


@dataclass
class AttnCache:
    """One local-attention layer: unrotated keys and values of the last n <= window - 1 positions."""

    k: Tensor | None = None           # [B, H, n, dh]
    v: Tensor | None = None           # [B, H, n, dh]
    offset: int = 0                   # absolute position of the next byte


@dataclass
class AgentState:
    h: list                           # per minGRU layer: [B, d] float64 carry, or None before the first byte
    heads: list                       # HeadState per head, in GRUHybrid.all_heads() order (post head first, then inner heads bottom-up)
    attn: list                        # AttnCache per local-attention layer, in GRUHybrid.local_at order

    @property
    def batch(self) -> int | None:
        """Number of agents this state belongs to (None for a fresh state of a model without memory heads)."""
        for head in self.heads:
            return int(head.S.size(0))
        for carry in self.h:
            if carry is not None:
                return int(carry.size(0))
        return None

    def new_document(self) -> "AgentState":
        """Start a new document: keep what was learned (S, Ksum), drop everything tied to the previous bytes.
        S and Ksum are the same tensors as this state's, not copies; states are never written in place."""
        heads = [replace(h, last_idx=None, last_val=None, conv_tail=None if h.conv_tail is None else torch.zeros_like(h.conv_tail))
                 for h in self.heads]
        return AgentState(h=[None] * len(self.h), heads=heads, attn=[AttnCache() for _ in self.attn])
