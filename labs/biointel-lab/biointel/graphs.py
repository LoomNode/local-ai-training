"""Sparse directed graphs and a degree-matched shuffle control."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SparseGraph:
    n: int
    edge_index: torch.Tensor  # [2, E] src -> dst
    cell_type: torch.Tensor  # [N]

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


def erdos_renyi(n: int, p: float, seed: int) -> SparseGraph:
    rng = torch.Generator().manual_seed(seed)
    src: list[int] = []
    dst: list[int] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if float(torch.rand(1, generator=rng)) < p:
                src.append(i)
                dst.append(j)
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    return SparseGraph(n=n, edge_index=edge_index, cell_type=torch.zeros(n, dtype=torch.long))


def in_out_degree(graph: SparseGraph) -> tuple[torch.Tensor, torch.Tensor]:
    din = torch.zeros(graph.n, dtype=torch.long)
    dout = torch.zeros(graph.n, dtype=torch.long)
    if graph.n_edges == 0:
        return din, dout
    src, dst = graph.edge_index
    dout.index_add_(0, src, torch.ones_like(src))
    din.index_add_(0, dst, torch.ones_like(dst))
    return din, dout


def degree_matched_shuffle(graph: SparseGraph, seed: int) -> SparseGraph:
    """Permute destinations so each node's in- and out-degree is unchanged."""
    if graph.n_edges < 2:
        return SparseGraph(
            n=graph.n,
            edge_index=graph.edge_index.clone(),
            cell_type=graph.cell_type.clone(),
        )
    rng = torch.Generator().manual_seed(seed)
    src = graph.edge_index[0].clone()
    dst = graph.edge_index[1].clone()
    new_dst = dst[torch.randperm(dst.numel(), generator=rng)]
    for _ in range(16):
        loops = (src == new_dst).nonzero(as_tuple=False).view(-1)
        if loops.numel() == 0:
            break
        for i in loops.tolist():
            j = int(torch.randint(0, new_dst.numel(), (1,), generator=rng))
            new_dst[i], new_dst[j] = new_dst[j].clone(), new_dst[i].clone()
    edge_index = torch.stack([src, new_dst], dim=0)
    return SparseGraph(n=graph.n, edge_index=edge_index, cell_type=graph.cell_type.clone())
