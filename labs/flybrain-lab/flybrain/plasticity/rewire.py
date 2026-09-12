"""Prune weak synapses and grow new sparse candidates. Topology may change; signs stay locked."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flybrain.model.brain import FlyLanguageModel


def _edge_magnitude(model: FlyLanguageModel) -> Tensor:
    w = model.synapse_weights().detach()
    if w.ndim == 2:
        return w.abs().mean(dim=-1)
    return w.abs()


def _install_edges(
    model: FlyLanguageModel,
    src: Tensor,
    dst: Tensor,
    sign: Tensor,
    synapse_param: Tensor,
) -> None:
    if src.numel() == 0:
        raise ValueError("graph would have zero edges")
    model.register_buffer("src", src.long().contiguous())
    model.register_buffer("dst", dst.long().contiguous())
    model.register_buffer("sign", sign.float().contiguous())
    model.synapse_param = nn.Parameter(synapse_param.detach().clone())
    model.connectome.src = model.src
    model.connectome.dst = model.dst
    model.connectome.sign = model.sign


def prune_weakest(model: FlyLanguageModel, drop: int) -> int:
    """Remove the `drop` lowest-|w| edges. Returns how many were removed."""
    e = model.n_synapses
    if drop <= 0:
        return 0
    keep_n = e - drop
    if keep_n < 1:
        raise ValueError("prune would remove every synapse")
    mag = _edge_magnitude(model)
    keep = torch.topk(mag, k=keep_n, largest=True).indices.sort().values
    p = model.synapse_param.detach()
    _install_edges(
        model,
        model.src[keep],
        model.dst[keep],
        model.sign[keep],
        p[keep],
    )
    return int(drop)


def grow_random(
    model: FlyLanguageModel,
    n_new: int,
    seed: int,
    max_edges: int | None = None,
) -> int:
    """Add unused src→dst pairs under a sparse edge budget. Returns how many were added."""
    if n_new <= 0:
        return 0
    n = int(getattr(model, "capacity", model.connectome.n))
    live = getattr(model, "live_mask", None)
    if live is None:
        live_idx = torch.arange(n)
    else:
        live_idx = live.nonzero(as_tuple=False).view(-1)
        if live_idx.numel() == 0:
            return 0
    existing = set(zip(model.src.tolist(), model.dst.tolist()))
    budget = max_edges if max_edges is not None else n * n
    room = max(0, budget - model.n_synapses)
    want = min(n_new, room)
    if want == 0:
        return 0
    rng = torch.Generator().manual_seed(seed)
    src_new: list[int] = []
    dst_new: list[int] = []
    sign_new: list[float] = []
    attempts = 0
    live_list = live_idx.tolist()
    n_live = len(live_list)
    while len(src_new) < want and attempts < want * 50 + 32:
        attempts += 1
        i = int(live_list[int(torch.randint(0, n_live, (1,), generator=rng))])
        j = int(live_list[int(torch.randint(0, n_live, (1,), generator=rng))])
        if i == j or (i, j) in existing:
            continue
        existing.add((i, j))
        src_new.append(i)
        dst_new.append(j)
        sign_new.append(-1.0 if float(torch.rand(1, generator=rng)) < 0.2 else 1.0)
    added = len(src_new)
    if added == 0:
        return 0
    p = model.synapse_param.detach()
    # small init so new edges start weak; match projected vs scalar rank
    if p.ndim == 2:
        extra_p = torch.full((added, p.shape[1]), -2.0, dtype=p.dtype)
    else:
        extra_p = torch.full((added,), -2.0, dtype=p.dtype)
    _install_edges(
        model,
        torch.cat([model.src, torch.tensor(src_new, dtype=torch.long)]),
        torch.cat([model.dst, torch.tensor(dst_new, dtype=torch.long)]),
        torch.cat([model.sign, torch.tensor(sign_new, dtype=model.sign.dtype)]),
        torch.cat([p, extra_p], dim=0),
    )
    return added
