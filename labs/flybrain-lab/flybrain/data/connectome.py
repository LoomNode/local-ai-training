"""Sparse connectome: synthetic fly-like motifs now, FlyWire NPZ later."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

CELL_KENYON = 0
CELL_DOPAMINE = 1
CELL_PROJECTION = 2
CELL_GENERIC = 3
N_CELL_TYPES = 4

REGION_SENSORY = 0
REGION_KENYON = 1
REGION_CORE = 2
REGION_DOPAMINE = 3
REGION_READOUT = 4


@dataclass
class Connectome:
    n: int
    src: Tensor
    dst: Tensor
    contact_count: Tensor
    sign: Tensor
    cell_type: Tensor
    region: Tensor
    sensory_idx: Tensor
    readout_idx: Tensor

    @property
    def n_edges(self) -> int:
        return int(self.src.numel())


def _unique_claws(
    src_pool: Tensor,
    dst_ids: Tensor,
    claws: int,
    rng: torch.Generator,
) -> tuple[list[int], list[int]]:
    """Each destination draws `claws` distinct sources (Kenyon-style PN claws)."""
    src: list[int] = []
    dst: list[int] = []
    if src_pool.numel() == 0 or dst_ids.numel() == 0 or claws <= 0:
        return src, dst
    claws = min(claws, int(src_pool.numel()))
    for d in dst_ids.tolist():
        perm = src_pool[torch.randperm(src_pool.numel(), generator=rng)]
        taken = 0
        used: set[int] = set()
        for s in perm.tolist():
            if s == d or s in used:
                continue
            src.append(int(s))
            dst.append(int(d))
            used.add(int(s))
            taken += 1
            if taken >= claws:
                break
    return src, dst


def _edges_from(
    src_ids: Tensor,
    dst_pool: Tensor,
    k: int,
    rng: torch.Generator,
) -> tuple[list[int], list[int]]:
    src: list[int] = []
    dst: list[int] = []
    if src_ids.numel() == 0 or dst_pool.numel() == 0 or k <= 0:
        return src, dst
    k = min(k, int(dst_pool.numel()))
    for s in src_ids.tolist():
        perm = dst_pool[torch.randperm(dst_pool.numel(), generator=rng)]
        taken = 0
        for d in perm.tolist():
            if d == s:
                continue
            src.append(int(s))
            dst.append(int(d))
            taken += 1
            if taken >= k:
                break
    return src, dst


def synthetic_flylike(
    n: int,
    seed: int,
    n_sensory: int | None = None,
    n_readout: int | None = None,
    n_kenyon: int | None = None,
    claws: int | None = None,
    class_claws: Tensor | None = None,
) -> Connectome:
    """Motif graph: sensory → expansion → recurrent core → readout, plus DA.

    `claws` is Kenyon-centric: each Kenyon cell draws that many distinct
    sensory sources. Omit it to keep the original sensory-fanout construction.
    """
    if n < 16:
        raise ValueError("synthetic_flylike needs n >= 16")
    rng = torch.Generator().manual_seed(seed)
    n_sensory = max(4, n // 8) if n_sensory is None else n_sensory
    n_readout = max(4, n // 8) if n_readout is None else n_readout
    n_da = max(2, n // 20)
    reserved = n_sensory + n_readout + n_da
    if reserved + 6 > n:
        raise ValueError(f"n={n} too small for sensory/readout/DA split")
    if n_kenyon is None:
        n_kenyon = min(max(8, n // 3), n - reserved - 2)
    n_core = n - reserved - n_kenyon
    if n_kenyon < 2 or n_core < 0:
        raise ValueError(f"n={n} cannot fit n_kenyon={n_kenyon}")

    sensory_idx = torch.arange(0, n_sensory)
    kenyon_idx = torch.arange(n_sensory, n_sensory + n_kenyon)
    da_idx = torch.arange(n_sensory + n_kenyon, n_sensory + n_kenyon + n_da)
    core_idx = torch.arange(
        n_sensory + n_kenyon + n_da,
        n_sensory + n_kenyon + n_da + n_core,
    )
    readout_idx = torch.arange(n - n_readout, n)

    cell_type = torch.full((n,), CELL_GENERIC, dtype=torch.long)
    cell_type[kenyon_idx] = CELL_KENYON
    cell_type[da_idx] = CELL_DOPAMINE
    cell_type[core_idx] = CELL_PROJECTION
    cell_type[readout_idx] = CELL_PROJECTION
    if core_idx.numel() >= 2:
        cell_type[core_idx[::2]] = CELL_GENERIC

    region = torch.full((n,), REGION_CORE, dtype=torch.long)
    region[sensory_idx] = REGION_SENSORY
    region[kenyon_idx] = REGION_KENYON
    region[da_idx] = REGION_DOPAMINE
    region[readout_idx] = REGION_READOUT

    src: list[int] = []
    dst: list[int] = []

    def add(a: list[int], b: list[int]) -> None:
        src.extend(a)
        dst.extend(b)

    if claws is None and class_claws is None:
        s, d = _edges_from(sensory_idx, kenyon_idx, k=min(3, n_kenyon), rng=rng)
        add(s, d)
    else:
        claw_k = 3 if claws is None else int(claws)
        if class_claws is not None:
            n_tuned = min(int(class_claws.shape[0]), int(kenyon_idx.numel()))
            for i in range(n_tuned):
                dest = int(kenyon_idx[i])
                for feat in class_claws[i].tolist():
                    src.append(int(sensory_idx[int(feat)]))
                    dst.append(dest)
            rest = kenyon_idx[n_tuned:]
        else:
            rest = kenyon_idx
        if rest.numel() > 0:
            s, d = _unique_claws(sensory_idx, rest, claws=claw_k, rng=rng)
            add(s, d)
    s, d = _edges_from(kenyon_idx, core_idx if n_core else readout_idx, k=2, rng=rng)
    add(s, d)
    s, d = _edges_from(kenyon_idx, readout_idx, k=1, rng=rng)
    add(s, d)
    if n_core:
        s, d = _edges_from(core_idx, core_idx, k=min(3, n_core - 1), rng=rng)
        add(s, d)
        s, d = _edges_from(core_idx, readout_idx, k=min(2, n_readout), rng=rng)
        add(s, d)
    da_targets = torch.cat([kenyon_idx, core_idx, readout_idx]) if n_core else torch.cat(
        [kenyon_idx, readout_idx]
    )
    s, d = _edges_from(da_idx, da_targets, k=min(8, int(da_targets.numel())), rng=rng)
    add(s, d)
    s, d = _edges_from(readout_idx, core_idx if n_core else kenyon_idx, k=1, rng=rng)
    add(s, d)

    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)
    e = src_t.numel()
    # Lognormal contact counts; ~20% inhibitory, biased toward core/DA sources.
    contact = torch.exp(torch.randn(e, generator=rng) * 0.6 + 1.0).clamp(min=1.0)
    sign = torch.ones(e)
    inhib = torch.rand(e, generator=rng) < 0.2
    sign[inhib] = -1.0

    return Connectome(
        n=n,
        src=src_t,
        dst=dst_t,
        contact_count=contact,
        sign=sign,
        cell_type=cell_type,
        region=region,
        sensory_idx=sensory_idx.long(),
        readout_idx=readout_idx.long(),
    )


def in_out_degree(graph: Connectome) -> tuple[Tensor, Tensor]:
    din = torch.zeros(graph.n, dtype=torch.long)
    dout = torch.zeros(graph.n, dtype=torch.long)
    if graph.n_edges == 0:
        return din, dout
    dout.index_add_(0, graph.src, torch.ones_like(graph.src))
    din.index_add_(0, graph.dst, torch.ones_like(graph.dst))
    return din, dout


def _clone_graph(graph: Connectome, src: Tensor, dst: Tensor) -> Connectome:
    return Connectome(
        n=graph.n,
        src=src,
        dst=dst,
        contact_count=graph.contact_count.clone(),
        sign=graph.sign.clone(),
        cell_type=graph.cell_type.clone(),
        region=graph.region.clone(),
        sensory_idx=graph.sensory_idx.clone(),
        readout_idx=graph.readout_idx.clone(),
    )


def degree_matched_shuffle(graph: Connectome, seed: int) -> Connectome:
    """Permute destinations; keep degrees, signs, counts, cell types, and I/O sets."""
    if graph.n_edges < 2:
        return _clone_graph(graph, graph.src.clone(), graph.dst.clone())
    rng = torch.Generator().manual_seed(seed)
    src = graph.src.clone()
    new_dst = graph.dst[torch.randperm(graph.n_edges, generator=rng)]
    for _ in range(16):
        loops = (src == new_dst).nonzero(as_tuple=False).view(-1)
        if loops.numel() == 0:
            break
        for i in loops.tolist():
            j = int(torch.randint(0, new_dst.numel(), (1,), generator=rng))
            new_dst[i], new_dst[j] = new_dst[j].clone(), new_dst[i].clone()
    return _clone_graph(graph, src, new_dst)


def region_preserving_shuffle(graph: Connectome, seed: int) -> Connectome:
    """Permute destinations inside each (src-region, dst-region) class.

    Keeps motif adjacency types (no sensory→readout shortcuts the fly graph
    lacks) while still rewiring which neuron in the allowed region is targeted.
    """
    if graph.n_edges < 2:
        return _clone_graph(graph, graph.src.clone(), graph.dst.clone())
    rng = torch.Generator().manual_seed(seed)
    src = graph.src.clone()
    dst = graph.dst.clone()
    pair = graph.region[src] * 8 + graph.region[dst]
    for p in pair.unique().tolist():
        idx = (pair == p).nonzero(as_tuple=False).view(-1)
        if idx.numel() < 2:
            continue
        perm = torch.randperm(idx.numel(), generator=rng)
        dst[idx] = dst[idx][perm]
        for _ in range(16):
            loops = idx[src[idx] == dst[idx]]
            if loops.numel() == 0:
                break
            for i in loops.tolist():
                j = int(idx[int(torch.randint(0, idx.numel(), (1,), generator=rng))])
                dst[i], dst[j] = dst[j].clone(), dst[i].clone()
    return _clone_graph(graph, src, dst)


def collapse_parallel_edges(graph: Connectome) -> Connectome:
    """Merge duplicate (src, dst) pairs. Shuffle can create them; fly claws do not."""
    seen: dict[tuple[int, int], int] = {}
    keep: list[int] = []
    contact = graph.contact_count.clone()
    for i, (s, d) in enumerate(zip(graph.src.tolist(), graph.dst.tolist())):
        key = (int(s), int(d))
        if key in seen:
            j = seen[key]
            contact[j] = contact[j] + contact[i]
            continue
        seen[key] = len(keep)
        keep.append(i)
    idx = torch.tensor(keep, dtype=torch.long)
    return Connectome(
        n=graph.n,
        src=graph.src[idx],
        dst=graph.dst[idx],
        contact_count=contact[idx],
        sign=graph.sign[idx],
        cell_type=graph.cell_type.clone(),
        region=graph.region.clone(),
        sensory_idx=graph.sensory_idx.clone(),
        readout_idx=graph.readout_idx.clone(),
    )


def load_npz(path: str | Path) -> Connectome:
    blob = np.load(path)
    cell_type = torch.as_tensor(blob["cell_type"], dtype=torch.long)
    return Connectome(
        n=int(cell_type.numel()),
        src=torch.as_tensor(blob["src"], dtype=torch.long),
        dst=torch.as_tensor(blob["dst"], dtype=torch.long),
        contact_count=torch.as_tensor(blob["contact_count"], dtype=torch.float32),
        sign=torch.as_tensor(blob["sign"], dtype=torch.float32),
        cell_type=cell_type,
        region=torch.as_tensor(blob["region"], dtype=torch.long),
        sensory_idx=torch.as_tensor(blob["sensory_idx"], dtype=torch.long),
        readout_idx=torch.as_tensor(blob["readout_idx"], dtype=torch.long),
    )
