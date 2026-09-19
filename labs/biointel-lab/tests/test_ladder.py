"""Ablation ladder: every arm reports the same resource keys; wiring is swappable."""

from __future__ import annotations

from biointel.ladder import LADDER_ARMS, run_ladder


def test_ladder_tiny_reports_all_arms_with_honest_counts():
    rows = run_ladder(tiny=True, seed=0)
    names = [r["arm"] for r in rows]
    assert names == list(LADDER_ARMS)
    required = {
        "arm",
        "neuron",
        "synapse",
        "graph",
        "trainable_scalars",
        "trainable_bits",
        "persistent_state_scalars",
        "neuron_count",
        "synapse_count",
        "flops_per_step",
        "train_acc",
        "eval_acc",
        "loss",
    }
    for row in rows:
        assert required.issubset(row)
        assert row["trainable_scalars"] > 0
        assert row["trainable_bits"] == row["trainable_scalars"] * 32
        if row["arm"] != "unique_gru":
            assert row["neuron_count"] > 0
            assert row["synapse_count"] > 0

    shared = next(r for r in rows if r["arm"] == "shared_recurrent")
    shuffled = next(r for r in rows if r["arm"] == "shared_recurrent_shuffled")
    assert shared["trainable_scalars"] == shuffled["trainable_scalars"]
    assert shared["graph"] == "erdos_renyi"
    assert shuffled["graph"] == "degree_matched_shuffle"

    point = next(r for r in rows if r["arm"] == "shared_point")
    recurrent = next(r for r in rows if r["arm"] == "shared_recurrent")
    assert recurrent["persistent_state_scalars"] >= point["persistent_state_scalars"]
