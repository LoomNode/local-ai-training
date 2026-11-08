"""Shared neuron/synapse programs over a sparse graph, plus a unique-weight GRU baseline."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from biointel.graphs import SparseGraph

NeuronKind = Literal["point", "recurrent", "dendritic"]
SynapseKind = Literal["shared", "unique", "stateful"]


class UniqueGRU(nn.Module):
    """Conventional unique-weight recurrent baseline (not a graph)."""

    def __init__(self, input_dim: int, hidden: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, output_dim)

    def forward_sequence(self, xs: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(xs)
        return self.readout(out[:, -1])

    def persistent_state_scalars(self, batch: int = 1) -> int:
        return int(batch * self.hidden)

    def flops_per_step(self, batch: int = 1) -> int:
        # 3 GRU gates: each is a matmul on [input+hidden -> hidden]
        h, d = self.hidden, self.input_dim
        return int(batch * 3 * 2 * ((d + h) * h))


def unique_gru(input_dim: int, hidden: int, output_dim: int) -> UniqueGRU:
    return UniqueGRU(input_dim=input_dim, hidden=hidden, output_dim=output_dim)


class SharedRuleNet(nn.Module):
    """One learned F_theta applied at every neuron; optional per-edge weights."""

    def __init__(
        self,
        graph: SparseGraph,
        neuron: NeuronKind = "recurrent",
        synapse: SynapseKind = "shared",
        hidden: int = 8,
        input_dim: int = 4,
        output_dim: int = 4,
        n_input: int = 4,
        n_output: int = 4,
    ) -> None:
        super().__init__()
        if neuron not in ("point", "recurrent", "dendritic"):
            raise ValueError(f"unknown neuron {neuron}")
        if synapse not in ("shared", "unique", "stateful"):
            raise ValueError(f"unknown synapse {synapse}")
        self.graph = graph
        self.neuron = neuron
        self.synapse = synapse
        self.hidden = hidden
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_input = n_input
        self.n_output = n_output
        self.n = graph.n
        self.n_compartments = 2 if neuron == "dendritic" else 1
        self.msg = nn.Linear(hidden, hidden, bias=False)
        if synapse == "unique":
            self.edge_weight = nn.Parameter(torch.ones(graph.n_edges))
        else:
            self.register_parameter("edge_weight", None)
        if synapse == "stateful":
            self.synapse_state = nn.Parameter(torch.ones(graph.n_edges), requires_grad=False)
            self.readout_adapter = nn.Parameter(torch.zeros(output_dim, hidden), requires_grad=False)
        else:
            self.register_parameter("synapse_state", None)
            self.register_parameter("readout_adapter", None)
        cell_in = (2 * hidden if neuron == "dendritic" else hidden) + input_dim
        if neuron == "point":
            self.cell = nn.Linear(cell_in, hidden)
        else:
            self.cell = nn.GRUCell(cell_in, hidden)
        self.readout = nn.Linear(hidden, output_dim)
        self.register_buffer("edge_index", graph.edge_index.clone())
        if graph.n_edges:
            comp = torch.arange(graph.n_edges) % max(self.n_compartments, 1)
        else:
            comp = torch.zeros(0, dtype=torch.long)
        self.register_buffer("comp", comp)
        self.state: torch.Tensor | None = None

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def persistent_state_scalars(self, batch: int = 1) -> int:
        n_state = batch * self.n * self.hidden
        if self.synapse_state is not None:
            n_state += int(self.synapse_state.numel())
        if self.readout_adapter is not None:
            n_state += int(self.readout_adapter.numel())
        return int(n_state)

    def flops_per_step(self, batch: int = 1) -> int:
        h, n, e, d = self.hidden, self.n, self.n_edges, self.input_dim
        flops = 2 * batch * e * h * h
        if self.neuron == "dendritic":
            # same projection, two scatter-adds; count the extra add as E*H
            flops += batch * e * h
        cell_in = (2 * h if self.neuron == "dendritic" else h) + d
        if self.neuron == "point":
            flops += 2 * batch * n * cell_in * h
        else:
            flops += batch * n * 3 * 2 * (cell_in * h + h * h)
        flops += 2 * batch * min(self.n_input, n) * d * h
        return int(flops)

    def reset_state(self, batch: int, device: torch.device | None = None) -> None:
        device = device or self.msg.weight.device
        self.state = torch.zeros(batch, self.n, self.hidden, device=device)

    def _inject(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        loc = x.new_zeros(batch, self.n, self.input_dim)
        n_in = min(self.n_input, self.n)
        loc[:, :n_in, :] = x.unsqueeze(1).expand(-1, n_in, -1)
        return loc

    def _messages(self, h: torch.Tensor) -> torch.Tensor:
        src = self.edge_index[0]
        if src.numel() == 0:
            return h.new_zeros(h.size(0), 0, self.hidden)
        m = self.msg(h[:, src, :])
        if self.edge_weight is not None:
            m = m * self.edge_weight.view(1, -1, 1)
        if self.synapse_state is not None:
            m = m * self.synapse_state.view(1, -1, 1)
        return m

    def _aggregate(self, h: torch.Tensor) -> torch.Tensor:
        dst = self.edge_index[1]
        m = self._messages(h)
        agg = h.new_zeros(h.size(0), self.n, self.hidden)
        if m.size(1) == 0:
            return agg
        idx = dst.view(1, -1, 1).expand(h.size(0), -1, self.hidden)
        agg.scatter_add_(1, idx, m)
        return agg

    def _aggregate_compartments(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dst = self.edge_index[1]
        m = self._messages(h)
        agg1 = h.new_zeros(h.size(0), self.n, self.hidden)
        agg2 = h.new_zeros(h.size(0), self.n, self.hidden)
        if m.size(1) == 0:
            return agg1, agg2
        idx = dst.view(1, -1, 1).expand(h.size(0), -1, self.hidden)
        c = self.comp.to(device=m.device)
        w1 = (c == 0).to(dtype=m.dtype).view(1, -1, 1)
        w2 = (c == 1).to(dtype=m.dtype).view(1, -1, 1)
        agg1.scatter_add_(1, idx, m * w1)
        agg2.scatter_add_(1, idx, m * w2)
        return agg1, agg2

    def step(self, x: torch.Tensor) -> torch.Tensor:
        if self.state is None or self.state.size(0) != x.size(0):
            self.reset_state(x.size(0), device=x.device)
        assert self.state is not None
        h = self.state
        loc = self._inject(x)
        if self.neuron == "dendritic":
            a1, a2 = self._aggregate_compartments(h)
            inp = torch.cat([a1, a2, loc], dim=-1)
        else:
            agg = self._aggregate(h)
            inp = torch.cat([agg, loc], dim=-1)
        flat = inp.reshape(-1, inp.size(-1))
        if self.neuron == "point":
            h_new = torch.tanh(self.cell(flat)).view_as(h)
        else:
            h_new = self.cell(flat, h.reshape(-1, self.hidden)).view_as(h)
        self.state = h_new
        return h_new

    def forward_sequence(self, xs: torch.Tensor) -> torch.Tensor:
        self.reset_state(xs.size(0), device=xs.device)
        for t in range(xs.size(1)):
            self.step(xs[:, t])
        assert self.state is not None
        pooled = self.state.mean(dim=1)
        weight = self.readout.weight
        if self.readout_adapter is not None:
            weight = weight + self.readout_adapter
        return F.linear(pooled, weight, self.readout.bias)
