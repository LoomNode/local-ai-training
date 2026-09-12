"""Honest scalar counts: an (8, 32) tensor is 256 scalars; state is not theta."""

from __future__ import annotations

from flybrain.accounting import count_resources, param_breakdown
from flybrain.baselines.gru import matched_gru
from flybrain.data.connectome import synthetic_flylike
from flybrain.model.brain import FlyLanguageModel


def test_count_resources_uses_numel_not_tensor_count():
    g = synthetic_flylike(n=32, seed=0)
    m = FlyLanguageModel(g, state_dim=4)
    report = count_resources(
        m,
        persistent_state_scalars=m.persistent_state_scalars(batch=2),
        neuron_count=g.n,
        synapse_count=g.n_edges,
        bits_per_scalar=32,
    )
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert report.trainable_scalars == trainable
    assert report.trainable_bits == trainable * 32
    assert report.neuron_count == 32
    assert report.synapse_count == g.n_edges
    assert report.persistent_state_scalars == 2 * 32 * 4
    assert report.persistent_state_scalars != report.trainable_scalars


def test_matched_gru_is_within_budget_band():
    g = synthetic_flylike(n=40, seed=1)
    fly = FlyLanguageModel(g, state_dim=4)
    gru = matched_gru(fly)
    n_fly = sum(p.numel() for p in fly.parameters() if p.requires_grad)
    n_gru = sum(p.numel() for p in gru.parameters() if p.requires_grad)
    # Same order of trainable scalars, not a straw-man GRU.
    assert n_gru > 0
    assert abs(n_gru - n_fly) / n_fly < 0.35


def test_param_breakdown_sums_and_flatten_head_dominates():
    g = synthetic_flylike(n=32, seed=0)
    flat = FlyLanguageModel(g, state_dim=4, readout="flatten")
    pooled = FlyLanguageModel(g, state_dim=4, readout="mean")
    b_flat = param_breakdown(flat)
    b_pool = param_breakdown(pooled)
    assert b_flat["total"] == sum(p.numel() for p in flat.parameters() if p.requires_grad)
    assert b_flat["decoder"] > b_flat["synapse"] + b_flat["program"]
    assert b_pool["decoder"] < b_flat["decoder"]
    assert b_pool["synapse"] == b_flat["synapse"]
