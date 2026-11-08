"""End-to-end smoke: a few training steps on each arm (CPU)."""

from __future__ import annotations

from scrt.config import ExperimentConfig, ModelConfig, TrainConfig
from scrt.train import train_arm


def _tiny_cfg(arm: str) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=32,
            n_embd=32,
            n_head=4,
            block_size=16,
            n_layer=2,
            n_loops=2,
            n_slots=4,
            lora_rank=2,
            slots_per_loop=1,
        ),
        train=TrainConfig(
            seed=0,
            batch_size=8,
            seq_len=16,
            steps=6,
            lr=3e-3,
            eval_every=3,
            consolidate_every=3,
            log_every=3,
            probe_batches=1,
            device="cpu",
            arm=arm,
            run_name="pytest_smoke",
            results_dir="results",
        ),
    )


def test_train_baseline_smoke():
    r = train_arm(_tiny_cfg("baseline"), write_results=False)
    assert r.n_params > 0
    assert r.final_eval_loss > 0


def test_train_looped_smoke():
    r = train_arm(_tiny_cfg("looped"), write_results=False)
    assert r.final_eval_loss > 0


def test_train_scrt_smoke_runs_consolidate():
    r = train_arm(_tiny_cfg("scrt"), write_results=False)
    assert r.final_eval_loss > 0
    # steps=6, consolidate_every=3 → steps 3 and 6
    assert len(r.consolidate_log) >= 1
    for entry in r.consolidate_log:
        assert entry["n_params"] == r.n_params
