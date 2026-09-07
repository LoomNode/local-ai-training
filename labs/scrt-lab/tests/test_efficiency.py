"""Efficiency metrics + native-size builders (no forced param match)."""

from __future__ import annotations

from dataclasses import replace

import torch

from scrt.budget import (
    BORN_ARMS,
    build_born_matched_models,
    build_native_models,
    native_param_report,
)
from scrt.config import ModelConfig, efficiency_config
from scrt.efficiency import (
    arm_efficiency_row,
    best_eval_per_mparam,
    dominates,
    efficiency_report,
    format_efficiency_table,
    pareto_frontier,
    rows_from_run_results,
    tokens_per_sec,
)
from scrt.model import count_params, model_forward
from scrt.train import train_arm


def _cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=48,
        n_embd=64,
        n_head=4,
        block_size=32,
        n_layer=4,
        n_loops=4,
        n_slots=8,
        lora_rank=4,
        slots_per_loop=1,
        tiny_lora_rank=2,
    )


def test_tokens_per_sec_and_eval_per_mparam():
    assert tokens_per_sec(1000, 2.0) == 500.0
    assert tokens_per_sec(1000, 0.0) == 0.0
    # 2.0 loss * 1.5M params → 3.0
    assert abs(best_eval_per_mparam(2.0, 1_500_000) - 3.0) < 1e-9


def test_dominates_and_pareto():
    # A: smaller and better loss → dominates B
    assert dominates(100, 1.0, 200, 1.5)
    # equal params, better loss
    assert dominates(100, 1.0, 100, 1.2)
    # worse on one axis → not dominate
    assert not dominates(200, 1.0, 100, 1.5)
    assert not dominates(100, 1.5, 100, 1.0)

    points = [
        {"arm": "big_good", "n_params": 3_000_000, "best_eval_loss": 1.0},
        {"arm": "small_ok", "n_params": 1_000_000, "best_eval_loss": 1.2},
        {"arm": "small_bad", "n_params": 1_100_000, "best_eval_loss": 1.5},
        {"arm": "tiny_worse", "n_params": 500_000, "best_eval_loss": 2.0},
    ]
    front = pareto_frontier(points)
    # big_good (best loss), small_ok (mid), tiny_worse (smallest) should survive;
    # small_bad is dominated by small_ok (fewer params and better loss? wait small_ok
    # has fewer params AND better loss than small_bad).
    assert "small_bad" not in front
    assert "big_good" in front
    assert "tiny_worse" in front
    assert "small_ok" in front


def test_efficiency_report_winners():
    rows = [
        arm_efficiency_row(
            "baseline",
            n_params=3_000_000,
            best_eval_loss=1.0,
            final_eval_loss=1.05,
            wall_s=100.0,
            tokens_seen=1_000_000,
        ),
        arm_efficiency_row(
            "looped",
            n_params=1_000_000,
            best_eval_loss=1.1,
            final_eval_loss=1.15,
            wall_s=300.0,
            tokens_seen=1_000_000,
        ),
    ]
    rep = efficiency_report(rows)
    assert rep["params_equal"] is False
    assert rep["winners"]["best_eval"]["arm"] == "baseline"
    assert rep["winners"]["smallest"]["arm"] == "looped"
    # looped: 1.1 * 1.0 = 1.1; baseline: 1.0 * 3.0 = 3.0 → looped wins eval/M
    assert rep["winners"]["best_eval_per_mparam"]["arm"] == "looped"
    assert "baseline" in rep["pareto_frontier"]
    assert "looped" in rep["pareto_frontier"]
    table = format_efficiency_table(rep)
    assert "looped" in table
    assert "baseline" in table
    assert "pareto frontier" in table.lower()


def test_native_models_unequal_params():
    cfg = _cfg()
    native = build_native_models(cfg, arms=BORN_ARMS)
    counts = {arm: n for arm, (_, _, n) in native.items()}
    assert set(counts) == set(BORN_ARMS)
    # Baseline unique stack should be larger than shared-core arms.
    assert counts["baseline"] > counts["looped"]
    assert counts["baseline"] > counts["shared_residual"]
    # No pad on any arm.
    for arm, (m, mcfg, n) in native.items():
        assert count_params(m) == n
        assert mcfg.n_layer == cfg.n_layer
        assert not hasattr(m, "param_pad") or m.param_pad.pad is None

    rep = native_param_report(cfg, arms=BORN_ARMS)
    assert rep["params_equal"] is False
    assert rep["max_params"] == counts["baseline"]
    assert rep["min_params"] == min(counts.values())


def test_native_vs_matched_baseline_layers():
    """Matched protocol rematches baseline; native keeps cfg.n_layer."""
    cfg = _cfg()
    native_m, native_cfg, native_n = build_native_models(cfg)["baseline"]
    matched_m, matched_cfg, matched_n = build_born_matched_models(cfg, exact=True)[
        "baseline"
    ]
    assert native_cfg.n_layer == cfg.n_layer
    assert count_params(native_m) == native_n
    # All matched arms share one budget; native baseline is the full unique stack.
    target = max(
        count_params(build_native_models(cfg)[a][0])
        for a in ("looped", "shared_residual", "modulated", "shared_plus_tiny")
    )
    assert matched_n == target
    # Full unique n_layer stack is larger than the matched-down baseline.
    assert native_n >= matched_n
    assert native_cfg.n_layer >= matched_cfg.n_layer


def test_efficiency_config():
    cfg = efficiency_config(device="cpu", steps=50)
    assert cfg.train.run_name == "efficiency"
    assert cfg.train.consolidate_every == 0
    assert cfg.train.steps == 50
    # Default must be capacity-bound, not structured entropy floor.
    assert cfg.train.corpus == "wikitext2"
    assert cfg.train.max_chars >= 10_000_000


def test_train_arm_native_records_efficiency_fields(tmp_path):
    from scrt.config import ExperimentConfig, TrainConfig

    mcfg = _cfg()
    tcfg = TrainConfig(
        seed=0,
        batch_size=8,
        seq_len=16,
        steps=4,
        lr=1e-3,
        eval_every=2,
        log_every=2,
        consolidate_every=0,
        device="cpu",
        results_dir=str(tmp_path),
        run_name="eff_smoke",
        corpus="synonym",
        corpus_length=5_000,
    )
    cfg = ExperimentConfig(model=mcfg, train=tcfg)
    r = train_arm(cfg, arm="looped", match_params=False, write_results=True)
    assert r.match_params is False
    assert r.pad_params == 0
    assert r.n_params == count_params(
        __import__("scrt.model", fromlist=["build_model"]).build_model("looped", mcfg)
    )
    assert r.tokens_per_sec > 0
    payload = (tmp_path / "eff_smoke_looped.json").read_text()
    assert "n_unique_params" in payload
    assert "tokens_per_sec" in payload


def test_rows_from_run_results_dict():
    results = {
        "a": {
            "n_params": 100,
            "best_eval_loss": 2.0,
            "final_eval_loss": 2.1,
            "wall_s": 10.0,
            "tokens_seen": 1000,
            "best_eval_step": 5,
            "model_config": {"n_layer": 4, "n_loops": 4},
        }
    }
    rows = rows_from_run_results(results, match_params=False)
    assert rows[0]["arm"] == "a"
    assert rows[0]["tokens_per_sec"] == 100.0
