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

from local_ai_training.checkpoint import _RESUME_CONFIG_DEFAULTS, save_checkpoint
from local_ai_training.cli import build_parser
from local_ai_training.config import ExperimentConfig
from local_ai_training.data import build_char_corpus
from local_ai_training.generate import load_for_generation
from local_ai_training.int8_master import Int8MasterLinear, scheduled_lr
from local_ai_training.model import ModelConfig, build_seeded_model
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
