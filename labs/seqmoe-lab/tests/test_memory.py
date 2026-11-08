"""CPU memory-plan dict and sequential-vs-joint optimizer comparison."""

from __future__ import annotations

from seqmoe.config import tiny_config
from seqmoe.memory import PLAN_KEYS, compare_sequential_vs_joint, plan
from seqmoe.model import SeqMoE


def test_memory_plan_cpu():
    cfg = tiny_config("cpu")
    m = SeqMoE(cfg.model)
    d = plan(m, "pretrain", "cpu", cfg.train)
    for key in PLAN_KEYS:
        assert key in d, f"missing plan key {key}"
    assert d["param_bytes"] > 0
    assert d["optim_bytes"] == 2 * d["grad_bytes"]
    assert d["swap_counted"] is False
    assert d["cuda_margin"] == 512 * 1024 * 1024
    m.set_mode("expert", 0)
    d2 = plan(m, "expert", "cpu", cfg.train)
    assert d2["resident_expert_count"] == 1
    assert d2["optimizer_numel"] > 0
    assert d2["optimizer_numel"] < d["optimizer_numel"] or d["grad_bytes"] >= d2["grad_bytes"]


def test_compare_sequential_vs_joint_cpu():
    cfg = tiny_config("cpu")
    cfg.train.batch_size = 2
    cfg.train.seq_len = 16
    cfg.model.block_size = 16
    out = compare_sequential_vs_joint(cfg, steps=2)
    n_seq = out["sequential"]["optimizer_numel"]
    n_joint = out["joint"]["optimizer_numel"]
    assert n_joint == 2 * n_seq
    assert out["sequential"]["resident_expert_count"] == 1
    assert out["joint"]["resident_expert_count"] == 2
    assert out["sequential"]["peak_vram_mb"] == 0.0
    assert out["joint"]["peak_vram_mb"] == 0.0
