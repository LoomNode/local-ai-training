"""Sparse graphs: generators plus a degree-matched shuffle that is a real control."""

from __future__ import annotations

import torch

from biointel.graphs import degree_matched_shuffle, erdos_renyi, in_out_degree


def test_erdos_renyi_is_deterministic_and_simple():
    g1 = erdos_renyi(n=24, p=0.2, seed=7)
    g2 = erdos_renyi(n=24, p=0.2, seed=7)
    g3 = erdos_renyi(n=24, p=0.2, seed=8)
    assert g1.n == 24
    assert g1.edge_index.shape[0] == 2
    assert torch.equal(g1.edge_index, g2.edge_index)
    assert not torch.equal(g1.edge_index, g3.edge_index)
    assert (g1.edge_index[0] != g1.edge_index[1]).all()
    assert int(g1.edge_index.max()) < 24


def test_degree_matched_shuffle_preserves_in_and_out_degree():
    g = erdos_renyi(n=40, p=0.15, seed=3)
    shuffled = degree_matched_shuffle(g, seed=11)
    assert shuffled.n == g.n
    assert shuffled.edge_index.shape[1] == g.edge_index.shape[1]
    din_a, dout_a = in_out_degree(g)
    din_b, dout_b = in_out_degree(shuffled)
    assert torch.equal(din_a, din_b)
    assert torch.equal(dout_a, dout_b)
    # Control is useless if wiring is identical.
    assert not torch.equal(g.edge_index, shuffled.edge_index)
