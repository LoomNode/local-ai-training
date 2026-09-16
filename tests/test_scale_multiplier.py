"""Tests for the opt-in ``scale_multiplier`` init-scale knob.

See docs/superpowers/specs/2026-09-15-init-scale-screen-design.md. The knob is threaded
exactly like ``stochastic_bucket``: DiscreteRatchetLinear/RatchetEmbedding, ModelConfig,
ExperimentConfig ([ratchet] scale_multiplier), CLI --scale-multiplier, checkpoint resume
defaults, and the generation loader.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from local_ai_training.checkpoint import _RESUME_CONFIG_DEFAULTS, save_checkpoint
from local_ai_training.config import ExperimentConfig
from local_ai_training.data import build_char_corpus
from local_ai_training.generate import load_for_generation
from local_ai_training.model import ModelConfig, build_seeded_model
from local_ai_training.ratchet import DiscreteRatchetLinear, audit_no_master_weights


def _reference() -> torch.Tensor:
    torch.manual_seed(1234)
    return torch.randn(6, 4) * 3.0


def test_scale_multiplier_default_is_bit_identical_to_unset() -> None:
    reference = _reference()

    default_layer = DiscreteRatchetLinear(4, 6, max_code=3, initial_weight=reference)
    explicit_layer = DiscreteRatchetLinear(
        4, 6, max_code=3, initial_weight=reference, scale_multiplier=1.0
    )

    assert torch.equal(default_layer.packed, explicit_layer.packed)
    assert torch.equal(default_layer._scale, explicit_layer._scale)


def test_scale_multiplier_four_scales_rows_and_shrinks_code_magnitudes() -> None:
    reference = _reference()
    max_code = 3

    default_layer = DiscreteRatchetLinear(4, 6, max_code=max_code, initial_weight=reference)
    scaled_layer = DiscreteRatchetLinear(
        4, 6, max_code=max_code, initial_weight=reference, scale_multiplier=4.0
    )

    assert torch.allclose(scaled_layer._scale, default_layer._scale * 4.0)
    assert torch.all(scaled_layer.code.abs() <= default_layer.code.abs())

    expected = torch.round(reference / scaled_layer.scale[:, None]).clamp(-max_code, max_code)
    expected = expected * scaled_layer.scale[:, None]
    assert torch.equal(scaled_layer.effective_weight(), expected)


def test_scale_multiplier_stored_on_layer() -> None:
    layer = DiscreteRatchetLinear(4, 6, max_code=3, scale_multiplier=2.5)
    assert layer.scale_multiplier == 2.5


def test_train_scale_multiplier_flag_defaults_one_and_parses() -> None:
    from local_ai_training.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["train"]).scale_multiplier == 1.0
    assert parser.parse_args(["train", "--scale-multiplier", "4.0"]).scale_multiplier == 4.0


@pytest.mark.parametrize("bad_value", [0.0, -1.0, -0.5])
def test_scale_multiplier_must_be_positive(bad_value: float) -> None:
    with pytest.raises(ValueError, match="scale_multiplier"):
        DiscreteRatchetLinear(4, 6, max_code=3, scale_multiplier=bad_value)


def test_toml_config_parses_scale_multiplier(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(
        """
[ratchet]
pressure_threshold = 8
scale_multiplier = 2.0
""".strip()
    )

    config = ExperimentConfig.from_toml(path)

    assert config.scale_multiplier == 2.0
    model_config = config.model_config(vocab_size=65)
    assert model_config.scale_multiplier == 2.0


def test_audit_clean_on_tiny_model_with_scale_multiplier_four() -> None:
    config = ModelConfig(
        vocab_size=16, block_size=8, n_layer=1, n_head=1, n_embd=8, scale_multiplier=4.0
    )

    model = build_seeded_model(config, max_code=2, seed=0)

    report = audit_no_master_weights(model, raise_on_violation=True)
    assert report.violations == ()


def test_resume_config_defaults_include_scale_multiplier() -> None:
    assert _RESUME_CONFIG_DEFAULTS["scale_multiplier"] == 1.0


def test_load_for_generation_rebuilds_scale_multiplier_checkpoint(tmp_path: Path) -> None:
    corpus = build_char_corpus("hello world " * 20)
    model_config = ModelConfig(
        vocab_size=len(corpus.vocabulary),
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=8,
        scale_multiplier=2.0,
    )
    model = build_seeded_model(model_config, max_code=3, seed=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    base = save_checkpoint(
        tmp_path / "ckpt_scale_multiplier",
        model=model,
        optimizer=optimizer,
        step=0,
        max_code=3,
        vocabulary=corpus.vocabulary,
        experiment_config={
            "block_size": 16,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "matmul_mode": "fp32",
            "weight_mode": "ratchet",
            "scale_multiplier": 2.0,
        },
    )

    loaded, vocab = load_for_generation(base, device="cpu")

    assert vocab == corpus.vocabulary
    tokens = corpus.validation_ids[:16][None]
    with torch.no_grad():
        expected, _ = model.eval()(tokens)
        actual, _ = loaded(tokens)
    assert torch.equal(actual, expected)
