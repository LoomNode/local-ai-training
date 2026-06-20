import csv
import json
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import load_file

from local_ai_training.checkpoint import load_checkpoint, save_checkpoint
from local_ai_training.config import ExperimentConfig
from local_ai_training.data import build_char_corpus
from local_ai_training.metrics import collect_ratchet_metrics
from local_ai_training.model import ModelConfig, build_seeded_model
from local_ai_training.train import train_run


def small_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        block_size=8,
        batch_size=4,
        n_layer=1,
        n_head=1,
        n_embd=8,
        steps=24,
        eval_interval=8,
        eval_batches=2,
        support_learning_rate=0.02,
        pressure_threshold=2,
        seeds=(7,),
        device="cpu",
    )


def test_toml_config_loads_and_validates_known_sections(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(
        """
[model]
block_size = 16
n_layer = 2
n_head = 2
n_embd = 32
dropout = 0.0

[ratchet]
pressure_threshold = 8
bucket_low = 0.5
bucket_high = 1.5

[training]
batch_size = 4
steps = 10
eval_interval = 5
eval_batches = 2
support_learning_rate = 0.001
seeds = [1, 2]
device = "cpu"
""".strip()
    )

    config = ExperimentConfig.from_toml(path)

    assert config.n_embd == 32
    assert config.seeds == (1, 2)
    assert config.model_config(vocab_size=65).vocab_size == 65


def test_metrics_report_code_pressure_and_memory() -> None:
    model = build_seeded_model(
        ModelConfig(vocab_size=5, block_size=4, n_layer=1, n_head=1, n_embd=8),
        max_code=2,
        seed=3,
    )

    metrics = collect_ratchet_metrics(model)

    assert sum(json.loads(metrics["code_histogram"]).values()) == metrics["ratchet_weights"]
    assert sum(json.loads(metrics["pressure_histogram"]).values()) == metrics["ratchet_weights"]
    assert 0 <= metrics["zero_percent"] <= 100
    assert metrics["ratchet_state_bytes"] > 0
    assert metrics["support_parameter_bytes"] > 0


def test_safetensors_checkpoint_round_trip_includes_optimizer_state(tmp_path: Path) -> None:
    config = ModelConfig(vocab_size=5, block_size=4, n_layer=1, n_head=1, n_embd=8)
    model = build_seeded_model(config, max_code=2, seed=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    tokens = torch.randint(0, 5, (2, 4))
    _, loss = model(tokens, tokens)
    assert loss is not None
    loss.backward()
    model.ratchet_update()
    optimizer.step()

    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        step=1,
        max_code=2,
        vocabulary=tuple("abcde"),
        experiment_config={"name": "test"},
    )

    restored = build_seeded_model(config, max_code=2, seed=99)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    metadata = load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        expected_max_code=2,
        expected_vocabulary=tuple("abcde"),
    )

    assert metadata["step"] == 1
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor.cpu(), restored.state_dict()[name].cpu())
    assert len(restored_optimizer.state) == len(optimizer.state)


def test_short_repetitive_corpus_run_moves_codes_and_reduces_validation_loss(
    tmp_path: Path,
) -> None:
    corpus = build_char_corpus("abcd" * 400)

    result = train_run(
        corpus=corpus,
        config=small_experiment_config(),
        max_code=2,
        seed=7,
        run_dir=tmp_path / "run",
    )

    assert result.total_code_moves > 0
    assert result.final_validation_loss < result.initial_validation_loss
    assert result.metrics_csv.is_file()
    assert result.checkpoint.with_suffix(".safetensors").is_file()


def test_resumed_run_matches_uninterrupted_run(tmp_path: Path) -> None:
    corpus = build_char_corpus("abcd" * 400)
    full_config = replace(small_experiment_config(), steps=12, eval_interval=6)
    partial_config = replace(full_config, steps=6)

    uninterrupted = train_run(
        corpus=corpus,
        config=full_config,
        max_code=3,
        seed=7,
        run_dir=tmp_path / "uninterrupted",
    )
    partial = train_run(
        corpus=corpus,
        config=partial_config,
        max_code=3,
        seed=7,
        run_dir=tmp_path / "resumed",
    )
    resumed = train_run(
        corpus=corpus,
        config=full_config,
        max_code=3,
        seed=7,
        run_dir=tmp_path / "resumed",
        resume_from=partial.checkpoint,
    )

    full_tensors = load_file(uninterrupted.checkpoint.with_suffix(".safetensors"))
    resumed_tensors = load_file(resumed.checkpoint.with_suffix(".safetensors"))
    for name in full_tensors:
        if not name.startswith("rng::"):
            assert torch.equal(full_tensors[name], resumed_tensors[name]), name


def test_metrics_csv_records_cumulative_code_moves(tmp_path: Path) -> None:
    corpus = build_char_corpus("abcd" * 400)

    result = train_run(
        corpus=corpus,
        config=small_experiment_config(),
        max_code=2,
        seed=7,
        run_dir=tmp_path / "run",
    )

    rows = list(csv.DictReader(result.metrics_csv.open()))
    cumulative = [int(row["cumulative_code_moves"]) for row in rows]
    # Seed row logs zero cumulative moves and the series never decreases.
    assert cumulative[0] == 0
    assert all(b >= a for a, b in zip(cumulative, cumulative[1:], strict=False))
    # The persisted total matches the in-memory running total.
    assert cumulative[-1] == result.total_code_moves > 0


def test_resumed_run_continues_cumulative_code_moves(tmp_path: Path) -> None:
    corpus = build_char_corpus("abcd" * 400)
    full_config = replace(small_experiment_config(), steps=12, eval_interval=6)
    partial_config = replace(full_config, steps=6)

    uninterrupted = train_run(
        corpus=corpus,
        config=full_config,
        max_code=3,
        seed=7,
        run_dir=tmp_path / "uninterrupted",
    )
    partial = train_run(
        corpus=corpus,
        config=partial_config,
        max_code=3,
        seed=7,
        run_dir=tmp_path / "resumed",
    )
    resumed = train_run(
        corpus=corpus,
        config=full_config,
        max_code=3,
        seed=7,
        run_dir=tmp_path / "resumed",
        resume_from=partial.checkpoint,
    )

    # Resume seeds the counter from the CSV rather than restarting at zero.
    assert resumed.total_code_moves == uninterrupted.total_code_moves
    resumed_rows = list(csv.DictReader(resumed.metrics_csv.open()))
    assert int(resumed_rows[-1]["cumulative_code_moves"]) == uninterrupted.total_code_moves


def test_frozen_control_never_moves_codes(tmp_path: Path) -> None:
    corpus = build_char_corpus("abcd" * 400)
    result = train_run(
        corpus=corpus,
        config=replace(small_experiment_config(), steps=4, eval_interval=2),
        max_code=2,
        seed=7,
        run_dir=tmp_path / "frozen",
        weight_mode="frozen",
    )

    assert result.total_code_moves == 0


def test_fp32_control_runs_without_ratchet_state(tmp_path: Path) -> None:
    corpus = build_char_corpus("abcd" * 400)
    result = train_run(
        corpus=corpus,
        config=replace(small_experiment_config(), steps=4, eval_interval=2),
        max_code=None,
        seed=7,
        run_dir=tmp_path / "fp32",
        weight_mode="fp32",
    )

    rows = list(csv.DictReader(result.metrics_csv.open()))
    assert rows[-1]["ratchet_weights"] == "0"
