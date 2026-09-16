import csv
from pathlib import Path

import pytest
import torch
from torch import nn

from local_ai_training.checkpoint import save_checkpoint
from local_ai_training.cli import build_parser
from local_ai_training.config import ExperimentConfig
from local_ai_training.data import build_char_corpus
from local_ai_training.generate import load_for_generation
from local_ai_training.int8_master import Int8MasterLinear
from local_ai_training.metrics import collect_ratchet_metrics
from local_ai_training.model import ModelConfig, build_seeded_model
from local_ai_training.ratchet import DiscreteRatchetLinear, audit_no_master_weights
from local_ai_training.train import train_run


def test_init_matches_seeded_kaiming_uniform_reference_and_scale() -> None:
    torch.manual_seed(0)
    reference = torch.empty(12, 20, dtype=torch.float32)
    nn.init.kaiming_uniform_(reference, a=5**0.5)

    layer = Int8MasterLinear.from_reference(reference)

    row_max = reference.abs().amax(dim=1)
    expected_scale = (row_max / 127).clamp_min(torch.finfo(torch.float32).eps)
    expected_code = torch.round(reference / expected_scale[:, None]).clamp(-127, 127)

    assert layer.weight_int8.dtype == torch.int8
    assert torch.equal(layer.weight_int8.to(torch.float32), expected_code)
    assert torch.allclose(layer.scale, expected_scale)


def test_default_init_matches_nn_linear_kaiming_uniform_same_seed() -> None:
    # Same seed/RNG draw as nn.Linear -> shared logical FP init across arms.
    torch.manual_seed(7)
    layer = Int8MasterLinear(20, 12)
    torch.manual_seed(7)
    reference_linear = nn.Linear(20, 12, bias=False)

    effective = layer.effective_weight()
    assert effective.shape == reference_linear.weight.shape
    half_grid = (layer.scale / 2)[:, None].expand_as(effective)
    assert torch.all((effective - reference_linear.weight).abs() <= half_grid + 1e-6)


def test_scale_is_a_fixed_buffer_not_a_parameter() -> None:
    layer = Int8MasterLinear(4, 3)
    assert dict(layer.named_parameters()) == {}
    assert "_scale" in dict(layer.named_buffers())
    assert "weight_int8" in dict(layer.named_buffers())


def test_forward_keeps_temporary_weight_until_update_then_releases_it() -> None:
    layer = Int8MasterLinear.from_reference(torch.eye(2))
    layer.train()
    inputs = torch.ones(1, 2, requires_grad=True)

    layer(inputs).sum().backward()

    assert layer.has_pending_gradient
    stats = layer.int8_update(0.1)
    assert stats.total_weights == 4
    assert not layer.has_pending_gradient
    assert layer._effective_weight is None


def test_update_with_no_pending_gradient_raises() -> None:
    layer = Int8MasterLinear(4, 3)
    with pytest.raises(RuntimeError, match="no pending"):
        layer.int8_update(0.1)


def test_lr_one_moves_every_weight_by_exactly_one_against_gradient_sign() -> None:
    torch.manual_seed(0)
    layer = Int8MasterLinear(64, 64)
    layer.weight_int8.fill_(0)
    before = layer.weight_int8.clone().to(torch.float32)
    layer.train()
    inputs = torch.ones(1, 64, requires_grad=True)
    output = layer(inputs)
    # Force every output-row gradient positive so sign(grad) == +1 everywhere,
    # i.e. delta = -lr == -1 (every weight should move by exactly -1).
    output.sum().backward()

    stats = layer.int8_update(1.0)

    after = layer.weight_int8.to(torch.float32)
    assert torch.equal(after, before - 1.0)
    assert stats.code_moves == before.numel()
    assert stats.negative_moves == before.numel()
    assert stats.positive_moves == 0


def test_lr_quarter_moves_by_one_with_empirical_probability_near_quarter() -> None:
    torch.manual_seed(1234)
    n = 10_000
    layer = Int8MasterLinear(n, 1)
    layer.weight_int8.fill_(0)
    before = layer.weight_int8.clone().to(torch.float32)
    layer.train()
    inputs = torch.eye(n)
    output = layer(inputs)
    output.sum().backward()

    stats = layer.int8_update(0.25)

    after = layer.weight_int8.to(torch.float32)
    delta = after - before
    assert torch.all((delta == 0) | (delta == -1))  # never moves by more than 1
    moved_fraction = (delta == -1).float().mean().item()
    assert abs(moved_fraction - 0.25) < 0.05
    assert stats.negative_moves == int((delta == -1).sum().item())


def test_clamping_at_boundary_records_blocked_moves() -> None:
    layer = Int8MasterLinear(3, 1)
    layer.weight_int8.fill_(127)
    layer.train()
    inputs = torch.ones(1, 3, requires_grad=True)
    output = layer(inputs)
    # Negative gradient -> sign(grad) < 0 -> delta = -lr*sign(grad) > 0 -> pushes
    # further past +127, which must be blocked (and clamped back to 127).
    (-output).sum().backward()

    stats = layer.int8_update(1.0)

    assert torch.equal(layer.weight_int8.to(torch.float32), torch.full((1, 3), 127.0))
    assert stats.blocked_positive_moves == 3
    assert stats.positive_moves == 0
    assert stats.code_moves == 0


def test_audit_reports_zero_violations_and_correct_persistent_bytes() -> None:
    model = nn.Sequential(
        Int8MasterLinear(16, 8),
        Int8MasterLinear(8, 4),
    )
    report = audit_no_master_weights(model, raise_on_violation=True)

    assert report.violations == ()
    assert report.ratchet_layers == 2
    assert report.ratchet_weights == 16 * 8 + 8 * 4
    expected_bytes = (16 * 8 + 8 * 4) * 1 + (8 + 4) * 4
    assert report.ratchet_state_bytes == expected_bytes

    for layer in model:
        assert layer.persistent_state_bytes == layer.weight_int8.numel() + 4 * layer.out_features


def test_state_histogram_counts_sum_to_total_and_flags_saturation() -> None:
    layer = Int8MasterLinear(8, 4)
    layer.weight_int8.copy_(
        torch.tensor(
            [[0, 127, -127, 5, 0, -5, 10, -10]] * 4, dtype=torch.int8
        )
    )
    stats = layer.state_histogram()
    assert stats["total"] == 32
    assert stats["zero"] == 8
    assert stats["saturated"] == 8
    assert sum(stats["histogram"].values()) == 32


def _small_model_config(**overrides) -> ModelConfig:
    base = dict(vocab_size=11, block_size=8, n_layer=1, n_head=1, n_embd=8)
    base.update(overrides)
    return ModelConfig(**base)


def test_linear_dispatch_produces_int8_master_layers() -> None:
    model = build_seeded_model(_small_model_config(int8_master=True), max_code=2, seed=3)
    linears = [m for m in model.modules() if isinstance(m, Int8MasterLinear)]
    assert linears  # qkv, projection, expand, contract, lm_head
    assert not any(isinstance(m, DiscreteRatchetLinear) for m in model.modules())


def test_int8_master_and_qat_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _small_model_config(qat=True, int8_master=True)


def test_int8_lr_must_be_positive() -> None:
    with pytest.raises(ValueError, match="int8_lr must be positive"):
        _small_model_config(int8_lr=0.0)


def test_ratchet_gpt_int8_master_update_aggregates_stats() -> None:
    model = build_seeded_model(
        _small_model_config(int8_master=True, int8_lr=0.5), max_code=2, seed=1
    )
    model.train()
    tokens = torch.randint(0, 11, (2, 8))
    _, loss = model(tokens, tokens)
    loss.backward()

    stats = model.int8_master_update()

    assert stats.total_weights > 0
    layers = [m for m in model.modules() if isinstance(m, Int8MasterLinear)]
    assert all(not layer.has_pending_gradient for layer in layers)


def test_toml_int8master_section_parses_lr_into_int8_lr(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text('[int8master]\nlr = 0.2\n')

    config = ExperimentConfig.from_toml(path)

    assert config.int8_lr == 0.2
    assert config.model_config(vocab_size=11).int8_lr == 0.2


def test_cli_weight_mode_int8master_and_int8_lr_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--weight-mode", "int8master", "--int8-lr", "0.3"])
    assert args.weight_mode == "int8master"
    assert args.int8_lr == 0.3
    assert parser.parse_args(["train"]).int8_lr is None


def _small_experiment_config(**overrides) -> ExperimentConfig:
    base = dict(
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
    base.update(overrides)
    return ExperimentConfig(**base)


def test_train_run_int8master_smoke_reduces_loss_and_moves_codes(tmp_path: Path) -> None:
    corpus = build_char_corpus("abcd" * 400)
    result = train_run(
        corpus=corpus,
        config=_small_experiment_config(int8_lr=0.5),
        max_code=2,
        seed=7,
        run_dir=tmp_path / "int8master",
        weight_mode="int8master",
    )
    rows = list(csv.DictReader(result.metrics_csv.open()))
    assert result.final_validation_loss < float(rows[0]["validation_loss"])
    assert result.total_code_moves > 0
    assert int(rows[-1]["code_moves"]) >= 0
    assert any(int(row["code_moves"]) > 0 for row in rows)


def test_train_run_int8master_ratchet_path_never_touched(tmp_path: Path) -> None:
    # weight_mode="int8master" must produce zero DiscreteRatchetLinear layers.
    corpus = build_char_corpus("abcd" * 400)
    result = train_run(
        corpus=corpus,
        config=_small_experiment_config(steps=2, eval_interval=2),
        max_code=2,
        seed=7,
        run_dir=tmp_path / "int8master-audit",
        weight_mode="int8master",
    )
    rows = list(csv.DictReader(result.metrics_csv.open()))
    assert int(rows[-1]["ratchet_layers"]) > 0  # int8master layers counted here


def test_checkpoint_round_trip_identical_logits_via_load_for_generation(tmp_path: Path) -> None:
    model_config = _small_model_config(int8_master=True, int8_lr=0.2)
    model = build_seeded_model(model_config, max_code=2, seed=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    vocabulary = tuple(chr(ord("a") + i) for i in range(11))
    tokens = torch.randint(0, 11, (2, 8))

    # Mutate state so a bit-identical round trip is a real check, not a no-op.
    model.train()
    _, loss = model(tokens, tokens)
    loss.backward()
    model.int8_master_update()
    optimizer.step()

    base = save_checkpoint(
        tmp_path / "ckpt",
        model=model,
        optimizer=optimizer,
        step=1,
        max_code=2,
        vocabulary=vocabulary,
        experiment_config={
            "block_size": model_config.block_size,
            "n_layer": model_config.n_layer,
            "n_head": model_config.n_head,
            "n_embd": model_config.n_embd,
            "matmul_mode": "fp32",
            "weight_mode": "int8master",
            "int8_lr": 0.2,
        },
    )

    model.eval()
    with torch.no_grad():
        expected_logits, _ = model(tokens)

    loaded_model, loaded_vocab = load_for_generation(base, device="cpu")
    assert loaded_vocab == vocabulary
    linears = [m for m in loaded_model.modules() if isinstance(m, Int8MasterLinear)]
    assert linears
    with torch.no_grad():
        actual_logits, _ = loaded_model(tokens)
    assert torch.equal(expected_logits, actual_logits)


def test_collect_ratchet_metrics_counts_int8_master_layers() -> None:
    model = build_seeded_model(_small_model_config(int8_master=True), max_code=2, seed=3)
    metrics = collect_ratchet_metrics(model)
    assert metrics["ratchet_layers"] > 0
    assert metrics["ratchet_weights"] > 0
    assert metrics["ratchet_state_bytes"] > 0
    assert 0 <= metrics["zero_percent"] <= 100
    assert 0 <= metrics["saturated_percent"] <= 100
