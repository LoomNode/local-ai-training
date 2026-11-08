"""Saturation metrics + late-window consolidate gating."""

from __future__ import annotations

from pathlib import Path

from scrt.config import ExperimentConfig, ModelConfig, TrainConfig, saturation_config
from scrt.train import (
    detect_plateau,
    last_quarter_improvement,
    post_plateau_improvement,
    should_run_consolidate,
    train_arm,
)


def test_should_run_consolidate_start_frac():
    # Before start window
    assert not should_run_consolidate(
        "scrt",
        500,
        consolidate_every=100,
        steps=1000,
        consolidate_until_frac=1.0,
        consolidate_start_frac=0.5,
    )
    # Inside window
    assert should_run_consolidate(
        "scrt",
        600,
        consolidate_every=100,
        steps=1000,
        consolidate_until_frac=1.0,
        consolidate_start_frac=0.5,
    )
    # Boundary: step == steps * start is excluded
    assert not should_run_consolidate(
        "scrt",
        500,
        consolidate_every=100,
        steps=1000,
        consolidate_until_frac=1.0,
        consolidate_start_frac=0.5,
    )


def test_detect_plateau_and_post_improvement():
    # Improving then flat then slight improve
    history = []
    loss = 3.0
    for step in range(100, 1001, 100):
        if step <= 400:
            loss -= 0.1
        elif step <= 800:
            loss = loss  # flat
        else:
            loss -= 0.05
        history.append({"step": step, "eval_loss": loss})
    plat_step, plat_loss = detect_plateau(
        history, steps=1000, patience_evals=3, epsilon=0.02, min_frac=0.2
    )
    assert plat_step is not None
    assert plat_step >= 400
    post = post_plateau_improvement(history, plat_step, plat_loss)
    # after long flat, late drop should register if after plateau
    assert isinstance(post, float)


def test_last_quarter_improvement_sign():
    history = []
    for step in range(100, 1001, 100):
        # worse mid, better late
        if step <= 750:
            history.append({"step": step, "eval_loss": 2.5})
        else:
            history.append({"step": step, "eval_loss": 2.0})
    assert last_quarter_improvement(history, 1000) > 0


def test_saturation_config_windows():
    cfg = saturation_config(device="cpu", steps=20000)
    assert cfg.train.consolidate_start_frac == 0.5
    assert cfg.train.consolidate_until_frac == 1.0
    assert cfg.train.consolidate_start_frac < cfg.train.consolidate_until_frac


def test_merge_only_config_disables_free_regrow():
    from scrt.config import saturation_merge_only_config

    cfg = saturation_merge_only_config(device="cpu", steps=1000)
    assert cfg.train.consolidate_enable_free is False
    assert cfg.train.consolidate_enable_regrow is False
    assert cfg.model.dropout > 0


def test_train_scrt_consolidates_only_second_half(tmp_path: Path):
    steps = 40
    every = 5
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
            seed=2,
            batch_size=8,
            seq_len=32,
            steps=steps,
            lr=3e-3,
            eval_every=10,
            consolidate_every=every,
            consolidate_start_frac=0.5,
            consolidate_until_frac=1.0,
            log_every=10,
            probe_batches=1,
            device="cpu",
            arm="scrt",
            corpus="structured",
            corpus_length=4000,
            run_name="sat_gate",
            results_dir=str(tmp_path),
            max_merges_per_cycle=1,
            plateau_patience_evals=2,
            plateau_epsilon=0.01,
            plateau_min_frac=0.2,
        ),
    )
    r = train_arm(cfg, arm="scrt", match_params=True, write_results=True)
    assert len(r.consolidate_log) >= 1
    for entry in r.consolidate_log:
        assert entry["step"] > steps * 0.5
        assert entry["step"] <= steps
    # plateau fields present (may be None if never flat)
    assert hasattr(r, "post_plateau_improvement")
    assert hasattr(r, "last_quarter_improvement")
