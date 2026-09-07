"""4-arm ablation: budget match + consolidate gating (shipped train path)."""

from __future__ import annotations

from pathlib import Path

import torch

from scrt.budget import build_matched_models, consolidates, scrt_budget
from scrt.config import ModelConfig, TrainConfig, ablation_config
from scrt.consolidate import consolidate
from scrt.model import SCRTGPT, build_model, count_params, model_forward
from scrt.train import (
    ABLATION_ARMS,
    should_run_consolidate,
    train_arm,
)
from scrt.config import ExperimentConfig


def test_consolidates_policy_by_arm():
    assert consolidates("scrt") is True
    assert consolidates("scrt_static") is False
    assert consolidates("looped") is False
    assert consolidates("baseline") is False


def test_should_run_consolidate_gates_static_and_frac():
    # scrt_static never
    assert not should_run_consolidate(
        "scrt_static",
        100,
        consolidate_every=50,
        steps=1000,
        consolidate_until_frac=1.0,
    )
    # scrt on schedule within frac
    assert should_run_consolidate(
        "scrt",
        100,
        consolidate_every=50,
        steps=1000,
        consolidate_until_frac=0.5,
    )
    # scrt past half-run with frac=0.5
    assert not should_run_consolidate(
        "scrt",
        600,
        consolidate_every=50,
        steps=1000,
        consolidate_until_frac=0.5,
    )
    # boundary inclusive
    assert should_run_consolidate(
        "scrt",
        500,
        consolidate_every=50,
        steps=1000,
        consolidate_until_frac=0.5,
    )
    # off-schedule
    assert not should_run_consolidate(
        "scrt",
        51,
        consolidate_every=50,
        steps=1000,
        consolidate_until_frac=1.0,
    )


def test_four_arms_matched_budget():
    cfg = ModelConfig(
        vocab_size=48,
        n_embd=64,
        n_head=4,
        block_size=32,
        n_layer=4,
        n_loops=4,
        n_slots=8,
        lora_rank=4,
        slots_per_loop=1,
    )
    matched = build_matched_models(cfg, exact=True)
    target = scrt_budget(cfg)
    for arm in ABLATION_ARMS:
        assert arm in matched
        m, _, n = matched[arm]
        assert n == target
        assert count_params(m) == target
        # forward works
        x = torch.randint(0, 48, (2, 16))
        y = torch.randint(0, 48, (2, 16))
        fr = model_forward(m, x, y)
        assert fr.loss is not None


def test_scrt_and_scrt_static_same_architecture():
    cfg = ModelConfig(
        vocab_size=32,
        n_embd=32,
        n_head=4,
        block_size=16,
        n_loops=3,
        n_slots=4,
        lora_rank=2,
        slots_per_loop=1,
    )
    a = build_model("scrt", cfg)
    b = build_model("scrt_static", cfg)
    assert type(a) is type(b) is SCRTGPT
    assert count_params(a) == count_params(b)


def test_consolidate_preserves_numel():
    cfg = ModelConfig(
        vocab_size=32,
        n_embd=32,
        n_head=4,
        block_size=16,
        n_loops=3,
        n_slots=4,
        lora_rank=2,
        slots_per_loop=1,
    )
    m = SCRTGPT(cfg)
    n0 = count_params(m)
    tcfg = TrainConfig(
        merge_sim_threshold=0.5,
        residual_free_threshold=0.05,
        max_merges_per_cycle=2,
        probe_batches=1,
        device="cpu",
        arm="scrt",
    )
    probes = [(torch.randint(0, 32, (2, 16)), torch.randint(0, 32, (2, 16)))]
    stats = consolidate(m, probes, tcfg, step=1)
    assert count_params(m) == n0 == stats.n_params


def test_train_arm_scrt_static_zero_consolidate(tmp_path: Path):
    """Real train_arm path: scrt_static must log zero consolidate events."""
    cfg = ExperimentConfig(
        model=ModelConfig(
            vocab_size=48,
            n_embd=32,
            n_head=4,
            block_size=32,
            n_layer=2,
            n_loops=2,
            n_slots=4,
            lora_rank=2,
            slots_per_loop=1,
        ),
        train=TrainConfig(
            seed=0,
            batch_size=8,
            seq_len=32,
            steps=12,
            lr=3e-3,
            eval_every=6,
            consolidate_every=3,
            consolidate_until_frac=1.0,
            log_every=6,
            probe_batches=1,
            device="cpu",
            arm="scrt_static",
            corpus="structured",
            corpus_length=4000,
            run_name="ablation_gate",
            results_dir=str(tmp_path),
            max_merges_per_cycle=2,
        ),
    )
    r = train_arm(cfg, arm="scrt_static", match_params=True, write_results=True)
    assert r.n_params > 0
    assert len(r.consolidate_log) == 0
    assert (tmp_path / "ablation_gate_scrt_static.json").exists()


def test_train_arm_scrt_respects_until_frac(tmp_path: Path):
    """scrt consolidates only in first half when until_frac=0.5."""
    steps = 20
    every = 5
    # Would hit steps 5,10,15,20 if frac=1; with 0.5 only 5,10.
    cfg = ExperimentConfig(
        model=ModelConfig(
            vocab_size=48,
            n_embd=32,
            n_head=4,
            block_size=32,
            n_layer=2,
            n_loops=2,
            n_slots=4,
            lora_rank=2,
            slots_per_loop=1,
        ),
        train=TrainConfig(
            seed=1,
            batch_size=8,
            seq_len=32,
            steps=steps,
            lr=3e-3,
            eval_every=10,
            consolidate_every=every,
            consolidate_until_frac=0.5,
            log_every=10,
            probe_batches=1,
            device="cpu",
            arm="scrt",
            corpus="structured",
            corpus_length=4000,
            run_name="ablation_frac",
            results_dir=str(tmp_path),
            max_merges_per_cycle=1,
        ),
    )
    r = train_arm(cfg, arm="scrt", match_params=True, write_results=True)
    assert len(r.consolidate_log) >= 1
    for entry in r.consolidate_log:
        assert entry["step"] <= steps * 0.5
        assert entry["n_params"] == r.n_params


def test_ablation_config_defaults():
    cfg = ablation_config(device="cpu", steps=100)
    assert cfg.train.run_name == "ablation"
    assert cfg.train.consolidate_until_frac == 0.5
    assert cfg.train.corpus == "structured"
