"""A group of agents with one brain: read different streams, pool what was written, hand it to an agent that did not read it.

Agents with the same weights compute the same sparse addresses, so one agent's tables are readable by another.
That is the only reason any of this works, and why a memory file is bound to the weights that wrote it."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from flybrain.model.state import AgentState


@dataclass
class Commons:
    """Per-head sum of a group's tables."""

    S: list       # per head [n_key, d]
    Ksum: list    # per head [n_key]


def weights_hash(model) -> str:
    """sha256 over every tensor of the state dict, by name and bytes, in key order. It identifies the weights, not the configuration:
    settings that live outside the state dict, such as `top_k`, are not covered."""
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def read(model, streams: Tensor, state: AgentState | None = None, seg: int = 2048) -> tuple[Tensor, AgentState]:
    """Each row of `streams` [B, T] is one agent reading its own bytes, `seg` at a time.
    Returns (ce [B, T - 1] in nats: the loss of byte t + 1 given bytes up to t, the agents' state after all T bytes).
    When a read continues from `state`, the loss of its first byte is not returned. The state passed in is not modified.
    A stream of fewer than two bytes has no loss to return: ce is [B, 0]."""
    if seg < 1:
        raise ValueError(f"seg must be at least 1, got {seg}")
    B, T = streams.shape
    state = model.init_state(B, streams.device) if state is None else state
    ces = []
    for c0 in range(0, T, seg):
        logits, state = model.stream(streams[:, c0 : c0 + seg], state)
        y = streams[:, c0 + 1 : c0 + seg + 1]
        if y.size(1):
            ces.append(F.cross_entropy(logits[:, : y.size(1)].float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(B, -1))
    ce = torch.cat(ces, dim=1) if ces else torch.zeros(B, 0, device=streams.device)
    return ce, state


def merge(states: AgentState | Sequence[AgentState]) -> Commons:
    """Per-head sum of S and Ksum over every agent of every state."""
    states = [states] if isinstance(states, AgentState) else list(states)
    if not states:
        raise ValueError("merge needs at least one agent state")
    n = len(states[0].heads)
    if any(len(st.heads) != n for st in states):
        raise ValueError(f"merge needs states with the same number of heads, got {sorted({len(st.heads) for st in states})}")
    return Commons(S=[sum(st.heads[i].S.sum(0) for st in states) for i in range(n)],
                   Ksum=[sum(st.heads[i].Ksum.sum(0) for st in states) for i in range(n)])


def preload(state: AgentState, commons: Commons, alpha: float = 1.0) -> AgentState:
    """A new state whose tables also hold alpha x commons; every agent gets the same commons, and `state` is not modified.
    It is not a deep copy: only the tables are new tensors, the other fields are shared with `state`, which is safe because states are never written in place."""
    if not len(commons.S) == len(commons.Ksum) == len(state.heads):
        raise ValueError(f"commons holds {len(commons.S)} S and {len(commons.Ksum)} Ksum tables, the state has {len(state.heads)} heads")
    heads = [replace(h, S=h.S + alpha * S.to(h.S)[None], Ksum=h.Ksum + alpha * K.to(h.Ksum)[None])
             for h, S, K in zip(state.heads, commons.S, commons.Ksum)]
    return AgentState(h=list(state.h), heads=heads, attn=list(state.attn))


def save_commons(path, commons: Commons, model) -> None:
    torch.save({"weights": weights_hash(model), "S": [s.cpu() for s in commons.S], "Ksum": [k.cpu() for k in commons.Ksum]}, path)


def load_commons(path, model) -> Commons:
    """Refuses a memory written by different weights: its addresses would mean nothing to this brain."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob["weights"] != weights_hash(model):
        raise ValueError("this memory was written by different weights")
    return Commons(S=blob["S"], Ksum=blob["Ksum"])
