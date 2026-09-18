"""Tests for the three opt-in Int8MasterLinear levers: lr decay, live row scale,
and configurable bit width.

See docs/superpowers/specs/2026-09-16-int8-levers-design.md. Each knob is threaded
exactly like ``int8_lr``: ModelConfig, ExperimentConfig ([int8master] section), CLI,
checkpoint resume defaults, and the generation loader. Defaults
(int8_lr_final=0.0, int8_live_scale=False, int8_bits=8) must reproduce the pre-lever
behaviour bit-for-bit.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from torch import nn

from local_ai_training.checkpoint import _RESUME_CONFIG_DEFAULTS, save_checkpoint
from local_ai_training.cli import build_parser
from local_ai_training.config import ExperimentConfig
from local_ai_training.data import build_char_corpus
from local_ai_training.generate import load_for_generation
from local_ai_training.int8_master import Int8MasterLinear, scheduled_lr
from local_ai_training.metrics import collect_ratchet_metrics
from local_ai_training.model import ModelConfig, build_seeded_model
from local_ai_training.ratchet import audit_no_master_weights
from local_ai_training.train import train_run

# ---------------------------------------------------------------------------
# Hard rule: defaults are bit-identical to the pre-lever stateless sign step.
# ---------------------------------------------------------------------------


def test_default_update_matches_original_stateless_sign_step_formula() -> None:
    torch.manual_seed(3)
    layer = Int8MasterLinear(6, 5)
    layer.train()
    inputs = torch.randn(2, 6, requires_grad=True)
    output = layer(inputs)
    output.sum().backward()
    gradient = layer._effective_weight.grad.clone()
    old_code = layer.weight_int8.to(torch.float32).clone()
    max_value = 127

    # Independently reproduce the exact pre-lever formula: -lr * sign(grad), no
    # init_scale/scale term, clamped at +-127.
    torch.manual_seed(99)
    expected_u = torch.rand(old_code.shape, dtype=torch.float32)
    expected_delta = -0.3 * torch.sign(gradient).to(torch.float32)
    expected = torch.floor(old_code + expected_delta + expected_u).clamp(-max_value, max_value)

    torch.manual_seed(99)  # reset the default generator so int8_update draws the same u
    layer.int8_update(0.3)

    assert torch.equal(layer.weight_int8.to(torch.float32), expected)


def test_default_train_run_is_bit_identical_before_and_after_lever_defaults(
    tmp_path: Path,
) -> None:
    """Same seed, defaults everywhere -> identical final weights across two runs."""
    corpus = build_char_corpus("abcd" * 400)

    def run(run_dir: Path) -> torch.Tensor:
        result = train_run(
            corpus=corpus,
            config=ExperimentConfig(
                block_size=8,
                batch_size=4,
                n_layer=1,
                n_head=1,
                n_embd=8,
                steps=3,
                eval_interval=3,
                eval_batches=2,
                support_learning_rate=0.02,
                pressure_threshold=2,
                seeds=(7,),
                device="cpu",
                int8_lr=0.5,
            ),
            max_code=2,
            seed=7,
            run_dir=run_dir,
            weight_mode="int8master",
        )
        loaded, _ = load_for_generation(result.checkpoint, device="cpu")
        layer = next(m for m in loaded.modules() if isinstance(m, Int8MasterLinear))
        return layer.weight_int8.clone()

    first = run(tmp_path / "a")
    second = run(tmp_path / "b")
    assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# Lever 1: decay (int8_lr_final)
# ---------------------------------------------------------------------------


def test_scheduled_lr_at_step_zero_half_and_total() -> None:
    assert scheduled_lr(1.0, 0.5, 0, 10) == 1.0
    assert scheduled_lr(1.0, 0.5, 5, 10) == pytest.approx(0.75)
    assert scheduled_lr(1.0, 0.5, 10, 10) == pytest.approx(0.5)


def test_scheduled_lr_clamps_fraction_to_zero_one() -> None:
    # step beyond total_steps must not overshoot past lr_final.
    assert scheduled_lr(1.0, 0.5, 20, 10) == pytest.approx(0.5)


def test_scheduled_lr_lr_final_zero_returns_lr_exactly() -> None:
    assert scheduled_lr(0.37, 0.0, 5, 10) == 0.37
    assert scheduled_lr(0.37, 0.0, 0, 10) == 0.37


def test_model_config_int8_lr_final_defaults_zero_and_validates() -> None:
    config = ModelConfig(vocab_size=11, block_size=8, n_layer=1, n_head=1, n_embd=8)
    assert config.int8_lr_final == 0.0

    with pytest.raises(ValueError, match="int8_lr_final"):
        ModelConfig(vocab_size=11, block_size=8, n_layer=1, n_head=1, n_embd=8, int8_lr_final=-0.1)

    with pytest.raises(ValueError, match="int8_lr_final"):
        ModelConfig(
            vocab_size=11,
            block_size=8,
            n_layer=1,
            n_head=1,
            n_embd=8,
            int8_lr=0.1,
            int8_lr_final=0.2,
        )


def test_ratchet_gpt_int8_master_update_lr_none_uses_config_int8_lr() -> None:
    torch.manual_seed(5)
    model_a = build_seeded_model(
        ModelConfig(vocab_size=11, block_size=8, n_layer=1, n_head=1, n_embd=8, int8_lr=0.4),
        max_code=2,
        seed=1,
    )
    model_b = build_seeded_model(
        ModelConfig(vocab_size=11, block_size=8, n_layer=1, n_head=1, n_embd=8, int8_lr=0.4),
        max_code=2,
        seed=1,
    )
    tokens = torch.randint(0, 11, (2, 8))
    for model in (model_a, model_b):
        model.train()
        _, loss = model(tokens, tokens)
        loss.backward()

    torch.manual_seed(123)
    model_a.int8_master_update()
    torch.manual_seed(123)
    model_b.int8_master_update(lr=0.4)

    layers_a = [m for m in model_a.modules() if isinstance(m, Int8MasterLinear)]
    layers_b = [m for m in model_b.modules() if isinstance(m, Int8MasterLinear)]
    for layer_a, layer_b in zip(layers_a, layers_b, strict=True):
        assert torch.equal(layer_a.weight_int8, layer_b.weight_int8)


def test_toml_int8master_section_parses_lr_final(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text("[int8master]\nlr = 0.2\nlr_final = 0.05\n")

    config = ExperimentConfig.from_toml(path)

    assert config.int8_lr_final == 0.05
    assert config.model_config(vocab_size=11).int8_lr_final == 0.05


def test_cli_int8_lr_final_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--int8-lr-final", "0.02"])
    assert args.int8_lr_final == 0.02
    assert parser.parse_args(["train"]).int8_lr_final is None


def test_resume_config_defaults_include_int8_lr_final() -> None:
    assert _RESUME_CONFIG_DEFAULTS["int8_lr_final"] == 0.0


def test_load_for_generation_threads_int8_lr_final(tmp_path: Path) -> None:
    corpus = build_char_corpus("hello world " * 20)
    model_config = ModelConfig(
        vocab_size=len(corpus.vocabulary),
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=8,
        int8_master=True,
        int8_lr=0.3,
        int8_lr_final=0.05,
    )
    model = build_seeded_model(model_config, max_code=2, seed=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    base = save_checkpoint(
        tmp_path / "ckpt_lr_final",
        model=model,
        optimizer=optimizer,
        step=0,
        max_code=2,
        vocabulary=corpus.vocabulary,
        experiment_config={
            "block_size": 16,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "matmul_mode": "fp32",
            "weight_mode": "int8master",
            "int8_lr": 0.3,
            "int8_lr_final": 0.05,
        },
    )

    loaded, vocab = load_for_generation(base, device="cpu")
    assert vocab == corpus.vocabulary
    assert loaded.config.int8_lr_final == 0.05


def test_train_run_with_lr_final_moves_layer_less_than_constant_lr(tmp_path: Path) -> None:
    """A decaying schedule should end at a strictly smaller lr than the constant-lr
    arm, so by the final step the sign-step magnitude (in expectation) is smaller."""
    corpus = build_char_corpus("abcd" * 400)

    config_kwargs = dict(
        block_size=8,
        batch_size=4,
        n_layer=1,
        n_head=1,
        n_embd=8,
        steps=3,
        eval_interval=3,
        eval_batches=2,
        support_learning_rate=0.02,
        pressure_threshold=2,
        seeds=(7,),
        device="cpu",
        int8_lr=1.0,
    )

    constant_result = train_run(
        corpus=corpus,
        config=ExperimentConfig(**config_kwargs),
        max_code=2,
        seed=7,
        run_dir=tmp_path / "constant",
        weight_mode="int8master",
    )
    decayed_result = train_run(
        corpus=corpus,
        config=ExperimentConfig(**config_kwargs, int8_lr_final=0.0001),
        max_code=2,
        seed=7,
        run_dir=tmp_path / "decayed",
        weight_mode="int8master",
    )

    constant_rows = list(csv.DictReader(constant_result.metrics_csv.open()))
    decayed_rows = list(csv.DictReader(decayed_result.metrics_csv.open()))
    # Both runs share the same seed/batches through the final step; a decayed lr must
    # not move more codes than the constant lr=1.0 arm by the last logged step.
    assert int(decayed_rows[-1]["cumulative_code_moves"]) <= int(
        constant_rows[-1]["cumulative_code_moves"]
    )


# ---------------------------------------------------------------------------
# Lever 2: live row scale (block exponent)
# ---------------------------------------------------------------------------


def test_live_scale_default_off_registers_no_init_scale_buffer() -> None:
    layer = Int8MasterLinear(4, 3)
    assert layer.live_scale is False
    assert "_init_scale" not in dict(layer.named_buffers())


def test_live_scale_on_registers_init_scale_buffer_cloned_from_init_scale() -> None:
    torch.manual_seed(2)
    layer = Int8MasterLinear(4, 3, live_scale=True)
    assert layer.live_scale is True
    buffers = dict(layer.named_buffers())
    assert "_init_scale" in buffers
    assert torch.equal(buffers["_init_scale"], layer._scale)
    # A clone, not an alias -- mutating _scale later must not move _init_scale.
    layer._scale.mul_(2.0)
    assert not torch.equal(buffers["_init_scale"], layer._scale)


def test_live_scale_persistent_state_bytes_includes_init_scale() -> None:
    plain = Int8MasterLinear(4, 3)
    live = Int8MasterLinear(4, 3, live_scale=True)
    assert live.persistent_state_bytes == plain.persistent_state_bytes + live.out_features * 4


def test_live_scale_grow_doubles_scale_and_preserves_effective_weight_within_one_grid_unit() -> (
    None
):
    torch.manual_seed(11)
    layer = Int8MasterLinear(8, 1, live_scale=True)
    layer._scale.fill_(1.0)
    layer._init_scale.fill_(1.0)
    layer.weight_int8.copy_(torch.tensor([[127, 127, 0, 0, 0, 0, 0, 0]], dtype=torch.int8))
    before_effective = layer.effective_weight().clone()

    layer._rescale_rows()

    assert torch.equal(layer._scale, torch.tensor([2.0]))
    after_effective = layer.effective_weight()
    grid_unit = layer._scale[:, None]
    assert torch.all((after_effective - before_effective).abs() <= grid_unit + 1e-6)
    assert layer.rows_grown == 1
    assert layer.rows_shrunk == 0


def test_live_scale_shrink_is_exact_and_preserves_effective_weight() -> None:
    layer = Int8MasterLinear(8, 1, live_scale=True)
    layer._scale.fill_(2.0)
    layer._init_scale.fill_(2.0)
    layer.weight_int8.copy_(torch.tensor([[10, -5, 3, 0, 1, -1, 2, -2]], dtype=torch.int8))
    before_effective = layer.effective_weight().clone()

    layer._rescale_rows()

    assert torch.equal(layer._scale, torch.tensor([1.0]))
    after_effective = layer.effective_weight()
    assert torch.equal(after_effective, before_effective)
    assert layer.rows_shrunk == 1
    assert layer.rows_grown == 0


def test_live_scale_no_rescale_between_both_thresholds() -> None:
    layer = Int8MasterLinear(8, 1, live_scale=True)
    layer._scale.fill_(1.0)
    layer._init_scale.fill_(1.0)
    # sat_frac == 0 (nothing at +-127) and row max (40) > max_value // 4 (31): neither
    # grow nor shrink should fire.
    layer.weight_int8.copy_(torch.tensor([[40, -30, 3, 0, 1, -1, 2, -2]], dtype=torch.int8))
    before = layer.weight_int8.clone()

    layer._rescale_rows()

    assert torch.equal(layer._scale, torch.tensor([1.0]))
    assert torch.equal(layer.weight_int8, before)
    assert layer.rows_grown == 0
    assert layer.rows_shrunk == 0


def test_live_scale_holds_effective_step_constant_across_grown_and_ungrown_rows() -> None:
    torch.manual_seed(21)
    n = 4000
    layer = Int8MasterLinear(n, 2, live_scale=True)
    layer.weight_int8.fill_(50)
    layer._scale.copy_(torch.tensor([1.0, 2.0]))
    layer._init_scale.copy_(torch.tensor([1.0, 1.0]))  # row 1 already grown once (ratio 0.5)
    before = layer.weight_int8.clone().to(torch.float32)
    layer.train()
    inputs = torch.ones(1, n, requires_grad=True)
    output = layer(inputs)
    output.sum().backward()

    layer.int8_update(1.0)

    after = layer.weight_int8.to(torch.float32)
    grid_delta = after - before
    # Neither row should have crossed a rescale threshold (row max stays ~49-50).
    assert layer.rows_grown == 0
    assert layer.rows_shrunk == 0
    effective_delta_row0 = (grid_delta[0] * layer._scale[0]).mean().item()
    effective_delta_row1 = (grid_delta[1] * layer._scale[1]).mean().item()
    assert abs(effective_delta_row0 - effective_delta_row1) < 0.1


def test_live_scale_delta_formula_reduces_to_plain_sign_step_when_ratio_is_one() -> None:
    """``-lr * (init_scale / scale) * sign(grad)`` must equal the plain
    ``-lr * sign(grad)`` path whenever a row's scale still equals its init_scale
    (ratio == 1), independent of quantization or any prior rescale history."""
    layer = Int8MasterLinear(8, 3, live_scale=True)
    layer._scale.copy_(torch.tensor([0.7, 1.0, 3.25]))
    layer._init_scale.copy_(layer._scale.clone())
    layer.weight_int8.fill_(50)
    before = layer.weight_int8.clone().to(torch.float32)
    layer.train()
    inputs = torch.randn(1, 8, requires_grad=True)
    layer(inputs).sum().backward()
    gradient = layer._effective_weight.grad.clone()

    torch.manual_seed(55)
    layer.int8_update(0.2)
    actual = layer.weight_int8.to(torch.float32)

    torch.manual_seed(55)
    u = torch.rand(before.shape, dtype=torch.float32)
    plain_delta = -0.2 * torch.sign(gradient).to(torch.float32)
    expected = torch.floor(before + plain_delta + u).clamp(-127, 127)
    # No row should have crossed a rescale threshold from this single small step.
    assert layer.rows_grown == 0
    assert layer.rows_shrunk == 0
    assert torch.equal(actual, expected)


def test_audit_clean_with_live_scale_enabled() -> None:
    model = nn.Sequential(
        Int8MasterLinear(16, 8, live_scale=True),
        Int8MasterLinear(8, 4, live_scale=True),
    )
    report = audit_no_master_weights(model, raise_on_violation=True)
    assert report.violations == ()


def test_model_config_int8_live_scale_defaults_false_and_threads_to_layers() -> None:
    config = ModelConfig(
        vocab_size=11, block_size=8, n_layer=1, n_head=1, n_embd=8, int8_master=True
    )
    assert config.int8_live_scale is False
    model = build_seeded_model(config, max_code=2, seed=3)
    layers = [m for m in model.modules() if isinstance(m, Int8MasterLinear)]
    assert layers and all(not layer.live_scale for layer in layers)

    live_config = ModelConfig(
        vocab_size=11,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=8,
        int8_master=True,
        int8_live_scale=True,
    )
    live_model = build_seeded_model(live_config, max_code=2, seed=3)
    live_layers = [m for m in live_model.modules() if isinstance(m, Int8MasterLinear)]
    assert live_layers and all(layer.live_scale for layer in live_layers)


def test_toml_int8master_section_parses_live_scale(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text("[int8master]\nlive_scale = true\n")

    config = ExperimentConfig.from_toml(path)

    assert config.int8_live_scale is True
    assert config.model_config(vocab_size=11).int8_live_scale is True


def test_cli_int8_live_scale_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--int8-live-scale"])
    assert args.int8_live_scale is True
    assert parser.parse_args(["train"]).int8_live_scale is False


def test_resume_config_defaults_include_int8_live_scale() -> None:
    assert _RESUME_CONFIG_DEFAULTS["int8_live_scale"] is False


def test_checkpoint_round_trip_preserves_init_scale_buffer(tmp_path: Path) -> None:
    torch.manual_seed(6)
    model_config = ModelConfig(
        vocab_size=11,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=8,
        int8_master=True,
        int8_live_scale=True,
    )
    model = build_seeded_model(model_config, max_code=2, seed=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    vocabulary = tuple(chr(ord("a") + i) for i in range(11))
    tokens = torch.randint(0, 11, (2, 8))

    model.train()
    _, loss = model(tokens, tokens)
    loss.backward()
    model.int8_master_update()
    optimizer.step()

    base = save_checkpoint(
        tmp_path / "ckpt_live_scale",
        model=model,
        optimizer=optimizer,
        step=1,
        max_code=2,
        vocabulary=vocabulary,
        experiment_config={
            "block_size": 8,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "matmul_mode": "fp32",
            "weight_mode": "int8master",
            "int8_lr": 0.1,
            "int8_live_scale": True,
        },
    )

    model.eval()
    with torch.no_grad():
        expected_logits, _ = model(tokens)

    loaded_model, loaded_vocab = load_for_generation(base, device="cpu")
    assert loaded_vocab == vocabulary
    layers = [m for m in loaded_model.modules() if isinstance(m, Int8MasterLinear)]
    assert layers and all(layer.live_scale for layer in layers)
    assert all("_init_scale" in dict(layer.named_buffers()) for layer in layers)
    with torch.no_grad():
        actual_logits, _ = loaded_model(tokens)
    assert torch.equal(expected_logits, actual_logits)


# ---------------------------------------------------------------------------
# Lever 3: bits (configurable grid width)
# ---------------------------------------------------------------------------


def test_bits_default_is_eight_with_max_value_127() -> None:
    layer = Int8MasterLinear(4, 3)
    assert layer.bits == 8
    assert layer.max_value == 127


@pytest.mark.parametrize("bits,expected_max", [(4, 7), (5, 15), (6, 31), (7, 63), (8, 127)])
def test_max_value_per_bits(bits: int, expected_max: int) -> None:
    layer = Int8MasterLinear(4, 3, bits=bits)
    assert layer.max_value == expected_max


@pytest.mark.parametrize("bits", [1, 2, 3, 9, 0, -1])
def test_invalid_bits_rejected_by_layer(bits: int) -> None:
    with pytest.raises(ValueError, match="bits"):
        Int8MasterLinear(4, 3, bits=bits)


def test_init_scale_and_clamp_use_max_value_for_given_bits() -> None:
    torch.manual_seed(9)
    reference = torch.empty(5, 6, dtype=torch.float32)
    nn.init.kaiming_uniform_(reference, a=5**0.5)

    layer = Int8MasterLinear.from_reference(reference, bits=4)

    row_max = reference.abs().amax(dim=1)
    expected_scale = (row_max / 7).clamp_min(torch.finfo(torch.float32).eps)
    expected_code = torch.round(reference / expected_scale[:, None]).clamp(-7, 7)
    assert torch.equal(layer.weight_int8.to(torch.float32), expected_code)
    assert torch.allclose(layer.scale, expected_scale)
    assert layer.weight_int8.abs().max().item() <= 7


def test_clamping_at_bits_boundary_records_blocked_moves() -> None:
    layer = Int8MasterLinear(3, 1, bits=4)
    layer.weight_int8.fill_(7)
    layer.train()
    inputs = torch.ones(1, 3, requires_grad=True)
    output = layer(inputs)
    (-output).sum().backward()  # pushes further past +7

    stats = layer.int8_update(1.0)

    assert torch.equal(layer.weight_int8.to(torch.float32), torch.full((1, 3), 7.0))
    assert stats.blocked_positive_moves == 3
    assert stats.positive_moves == 0


def test_persistent_state_bytes_logical_for_four_bits() -> None:
    layer = Int8MasterLinear(16, 8, bits=4)
    expected = layer.weight_int8.numel() * 4 // 8 + layer.out_features * 4
    assert layer.persistent_state_bytes == expected
    assert layer.persistent_state_bytes == layer.weight_int8.numel() // 2 + layer.out_features * 4


def test_persistent_state_bytes_default_bits_matches_prior_one_byte_per_weight() -> None:
    layer = Int8MasterLinear(16, 8)
    assert layer.persistent_state_bytes == layer.weight_int8.numel() + layer.out_features * 4


def test_extra_repr_reports_states_for_bits() -> None:
    layer = Int8MasterLinear(4, 3, bits=4)
    assert "states=15" in layer.extra_repr()  # 2*7+1
    assert Int8MasterLinear(4, 3).extra_repr().__contains__("states=255")


def test_state_histogram_bin_width_scales_with_bits() -> None:
    layer = Int8MasterLinear(8, 4, bits=4)
    layer.weight_int8.copy_(
        torch.tensor([[0, 7, -7, 3, 0, -3, 5, -5]] * 4, dtype=torch.int8)
    )
    stats = layer.state_histogram()
    assert stats["total"] == 32
    assert stats["zero"] == 8
    assert stats["saturated"] == 8  # abs == max_value (7), not the literal 127
    assert sum(stats["histogram"].values()) == 32


def test_metrics_saturated_percent_correct_for_four_bit_layer() -> None:
    model = nn.Sequential(Int8MasterLinear(4, 2, bits=4))
    layer = model[0]
    # 4 of 8 weights saturated at +-7 (the 4-bit max), not the 8-bit 127.
    layer.weight_int8.copy_(torch.tensor([[7, -7, 0, 1], [7, -7, 2, -1]], dtype=torch.int8))

    metrics = collect_ratchet_metrics(model)

    assert metrics["saturated_percent"] == pytest.approx(100.0 * 4 / 8)


def test_model_config_int8_bits_defaults_eight_and_validates() -> None:
    config = ModelConfig(vocab_size=11, block_size=8, n_layer=1, n_head=1, n_embd=8)
    assert config.int8_bits == 8

    for bits in (4, 5, 6, 7, 8):
        ModelConfig(
            vocab_size=11, block_size=8, n_layer=1, n_head=1, n_embd=8, int8_bits=bits
        )

    for bad_bits in (3, 9, 0):
        with pytest.raises(ValueError, match="int8_bits"):
            ModelConfig(
                vocab_size=11, block_size=8, n_layer=1, n_head=1, n_embd=8, int8_bits=bad_bits
            )


def test_model_config_int8_bits_threads_to_layers() -> None:
    config = ModelConfig(
        vocab_size=11, block_size=8, n_layer=1, n_head=1, n_embd=8, int8_master=True, int8_bits=4
    )
    model = build_seeded_model(config, max_code=2, seed=3)
    layers = [m for m in model.modules() if isinstance(m, Int8MasterLinear)]
    assert layers and all(layer.bits == 4 and layer.max_value == 7 for layer in layers)


def test_toml_int8master_section_parses_bits(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text("[int8master]\nbits = 6\n")

    config = ExperimentConfig.from_toml(path)

    assert config.int8_bits == 6
    assert config.model_config(vocab_size=11).int8_bits == 6


def test_experiment_config_int8_bits_validates() -> None:
    with pytest.raises(ValueError, match="int8_bits"):
        ExperimentConfig(int8_bits=9)


def test_cli_int8_bits_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--int8-bits", "4"])
    assert args.int8_bits == 4
    assert parser.parse_args(["train"]).int8_bits is None


def test_resume_config_defaults_include_int8_bits() -> None:
    assert _RESUME_CONFIG_DEFAULTS["int8_bits"] == 8


def test_load_for_generation_threads_int8_bits(tmp_path: Path) -> None:
    corpus = build_char_corpus("hello world " * 20)
    model_config = ModelConfig(
        vocab_size=len(corpus.vocabulary),
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=8,
        int8_master=True,
        int8_bits=4,
    )
    model = build_seeded_model(model_config, max_code=2, seed=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    base = save_checkpoint(
        tmp_path / "ckpt_bits",
        model=model,
        optimizer=optimizer,
        step=0,
        max_code=2,
        vocabulary=corpus.vocabulary,
        experiment_config={
            "block_size": 16,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "matmul_mode": "fp32",
            "weight_mode": "int8master",
            "int8_lr": 0.1,
            "int8_bits": 4,
        },
    )

    loaded, vocab = load_for_generation(base, device="cpu")
    assert vocab == corpus.vocabulary
    layers = [m for m in loaded.modules() if isinstance(m, Int8MasterLinear)]
    assert layers and all(layer.bits == 4 and layer.max_value == 7 for layer in layers)


# ---------------------------------------------------------------------------
# Hard rule: audit_no_master_weights stays clean for every knob combination.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("live_scale", [False, True])
@pytest.mark.parametrize("bits", [4, 6, 8])
@pytest.mark.parametrize("lr_final", [0.0, 0.05])
def test_audit_clean_for_every_lever_combination(
    live_scale: bool, bits: int, lr_final: float
) -> None:
    config = ModelConfig(
        vocab_size=11,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=8,
        int8_master=True,
        int8_lr=0.5 if lr_final == 0.0 else max(lr_final, 0.5),
        int8_lr_final=lr_final,
        int8_live_scale=live_scale,
        int8_bits=bits,
    )
    model = build_seeded_model(config, max_code=2, seed=0)
    report = audit_no_master_weights(model, raise_on_violation=True)
    assert report.violations == ()


def test_histogram_bin_width_is_sixteen_at_eight_bits() -> None:
    from local_ai_training.int8_master import histogram_bin_width

    # Must match the width the 2026-09-15 int8 runs reported so histograms compare.
    assert histogram_bin_width(Int8MasterLinear(4, 3).max_value) == 16
    assert histogram_bin_width(Int8MasterLinear(4, 3, bits=6).max_value) == 4
    assert histogram_bin_width(Int8MasterLinear(4, 3, bits=4).max_value) == 1


def test_schedule_shapes_and_back_compat() -> None:
    from local_ai_training.int8_master import resolve_schedule, scheduled_lr

    # Back-compat: the 2026-09-17 runs set lr_final alone and got a linear anneal.
    assert resolve_schedule("constant", 0.25) == "linear"
    assert resolve_schedule("constant", 0.0) == "constant"
    assert scheduled_lr(1.0, 0.25, 15_000, 30_000) == pytest.approx(0.625)
    # Constant with no target is the untouched default path.
    assert scheduled_lr(1.0, 0.0, 15_000, 30_000) == pytest.approx(1.0)
    # Explicit schedules may anneal all the way to zero, which "constant" cannot express.
    assert scheduled_lr(1.0, 0.0, 15_000, 30_000, "linear") == pytest.approx(0.5)
    assert scheduled_lr(1.0, 0.0, 30_000, 30_000, "linear") == pytest.approx(0.0)
    # Cosine: flat at both ends, half way down at the midpoint.
    assert scheduled_lr(1.0, 0.0, 0, 30_000, "cosine") == pytest.approx(1.0)
    assert scheduled_lr(1.0, 0.0, 15_000, 30_000, "cosine") == pytest.approx(0.5)
    assert scheduled_lr(1.0, 0.0, 30_000, 30_000, "cosine") == pytest.approx(0.0)
    assert scheduled_lr(1.0, 0.2, 15_000, 30_000, "cosine") == pytest.approx(0.6)
    # Clamped past the budget.
    assert scheduled_lr(1.0, 0.25, 40_000, 30_000, "linear") == pytest.approx(0.25)


def test_schedule_knob_threads_through_config_cli_and_checkpoint(tmp_path) -> None:
    from local_ai_training.cli import build_parser
    from local_ai_training.config import ExperimentConfig
    from local_ai_training.model import ModelConfig

    base = dict(vocab_size=8, block_size=4, n_layer=1, n_head=1, n_embd=8)
    assert ModelConfig(**base).int8_lr_schedule == "constant"
    with pytest.raises(ValueError, match="int8_lr_schedule"):
        ModelConfig(**base, int8_lr_schedule="quadratic")
    with pytest.raises(ValueError, match="int8_lr_schedule"):
        ExperimentConfig(int8_lr_schedule="quadratic")

    toml = tmp_path / "c.toml"
    toml.write_text('[int8master]\nlr = 1.0\nlr_final = 0.0\nlr_schedule = "cosine"\n')
    assert ExperimentConfig.from_toml(toml).int8_lr_schedule == "cosine"

    args = build_parser().parse_args(
        ["train", "--weight-mode", "int8master",
         "--int8-lr-schedule", "cosine", "--int8-lr-final", "0.0"]
    )
    assert args.int8_lr_schedule == "cosine"
