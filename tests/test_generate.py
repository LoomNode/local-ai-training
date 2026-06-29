"""Autoregressive sampling from a saved ratchet checkpoint."""

from pathlib import Path

import pytest
import torch

from local_ai_training.checkpoint import save_checkpoint
from local_ai_training.data import build_char_corpus
from local_ai_training.generate import (
    _generation_window,
    generate,
    load_for_generation,
    warm_up_generation,
)
from local_ai_training.model import ModelConfig, build_seeded_model
from local_ai_training.ratchet import DiscreteRatchetLinear
from local_ai_training.tokenizer import BpeTokenizer


def _save_tiny_checkpoint(tmp_path: Path):
    corpus = build_char_corpus("hello world this is a tiny corpus for testing generation " * 8)
    model_config = ModelConfig(
        vocab_size=len(corpus.vocabulary), block_size=16, n_layer=1, n_head=1, n_embd=8
    )
    model = build_seeded_model(model_config, max_code=2, seed=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    base = save_checkpoint(
        tmp_path / "ckpt",
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
        },
    )
    return base, corpus.vocabulary


def test_load_rebuilds_model_and_vocab(tmp_path: Path) -> None:
    base, vocabulary = _save_tiny_checkpoint(tmp_path)
    model, loaded_vocab = load_for_generation(base, device="cpu")
    assert loaded_vocab == vocabulary
    assert model.config.vocab_size == len(vocabulary)
    assert model.config.block_size == 16


def test_greedy_generation_is_deterministic_and_in_vocab(tmp_path: Path) -> None:
    base, vocabulary = _save_tiny_checkpoint(tmp_path)
    model, vocab = load_for_generation(base, device="cpu")

    out1 = generate(model, vocab, "hello", max_new_tokens=20, temperature=0.0)
    out2 = generate(model, vocab, "hello", max_new_tokens=20, temperature=0.0)

    assert out1 == out2  # greedy is deterministic
    assert len(out1) == 20  # returns exactly the new characters
    assert all(character in vocab for character in out1)


def test_seeded_sampling_is_reproducible(tmp_path: Path) -> None:
    base, vocabulary = _save_tiny_checkpoint(tmp_path)
    model, vocab = load_for_generation(base, device="cpu")

    sample1 = generate(model, vocab, "hello", max_new_tokens=20, temperature=1.0, top_k=5, seed=42)
    sample2 = generate(model, vocab, "hello", max_new_tokens=20, temperature=1.0, top_k=5, seed=42)

    assert sample1 == sample2
    assert all(character in vocab for character in sample1)


def test_generation_past_block_size_keeps_running(tmp_path: Path) -> None:
    base, vocabulary = _save_tiny_checkpoint(tmp_path)
    model, vocab = load_for_generation(base, device="cpu")
    # block_size is 16; generating more than that must not raise (context is cropped).
    out = generate(model, vocab, "hello", max_new_tokens=40, temperature=0.0)
    assert len(out) == 40


def test_unknown_prompt_character_is_rejected(tmp_path: Path) -> None:
    base, vocabulary = _save_tiny_checkpoint(tmp_path)
    model, vocab = load_for_generation(base, device="cpu")
    with pytest.raises(ValueError, match="not in the model vocabulary"):
        generate(model, vocab, "HELLO!", max_new_tokens=5, temperature=0.0)


# ---------------------------------------------------------------------------
# Subword (BPE) tests
# ---------------------------------------------------------------------------

_SUBWORD_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump! "
) * 40


def _save_subword_checkpoint(tmp_path: Path):
    tok = BpeTokenizer.train(_SUBWORD_TEXT, vocab_size=300)
    model_config = ModelConfig(
        vocab_size=tok.vocab_size,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=8,
        ratchet_embedding=True,
    )
    model = build_seeded_model(model_config, max_code=2, seed=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    base = save_checkpoint(
        tmp_path / "ckpt_subword",
        model=model,
        optimizer=optimizer,
        step=0,
        max_code=2,
        vocabulary=(),
        experiment_config={
            "block_size": 16,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "matmul_mode": "fp32",
            "ratchet_embedding": True,
        },
        tokenizer_kind="subword",
        tokenizer_json=tok.to_json(),
    )
    return base, tok


def test_subword_load_returns_bpe_tokenizer(tmp_path: Path) -> None:
    base, tok = _save_subword_checkpoint(tmp_path)
    model, decoder = load_for_generation(base, device="cpu")
    assert isinstance(decoder, BpeTokenizer)
    assert model.config.vocab_size == tok.vocab_size


def test_subword_generate_returns_nonempty_str(tmp_path: Path) -> None:
    base, _ = _save_subword_checkpoint(tmp_path)
    model, decoder = load_for_generation(base, device="cpu")
    out = generate(model, decoder, "the", max_new_tokens=5, temperature=0.0)
    assert isinstance(out, str)
    assert len(out) > 0


def test_load_for_generation_with_rms_ema_beta(tmp_path: Path) -> None:
    # Regression: a model trained with rms_ema_beta > 0 registers the conditional
    # rms_ema buffers; load_for_generation must thread that knob from the saved config
    # so the rebuilt model has matching buffers (otherwise state_dict load fails with
    # "Unexpected key(s) ... rms_ema").
    tok = BpeTokenizer.train(_SUBWORD_TEXT, vocab_size=300)
    model_config = ModelConfig(
        vocab_size=tok.vocab_size,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=8,
        ratchet_embedding=True,
        rms_ema_beta=0.9,
    )
    model = build_seeded_model(model_config, max_code=2, seed=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    base = save_checkpoint(
        tmp_path / "ckpt_ema",
        model=model,
        optimizer=optimizer,
        step=0,
        max_code=2,
        vocabulary=(),
        experiment_config={
            "block_size": 16,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "matmul_mode": "fp32",
            "ratchet_embedding": True,
            "rms_ema_beta": 0.9,
        },
        tokenizer_kind="subword",
        tokenizer_json=tok.to_json(),
    )
    # Must not raise on state_dict load.
    model, decoder = load_for_generation(base, device="cpu")
    out = generate(model, decoder, "the", max_new_tokens=3, temperature=0.0)
    assert isinstance(out, str) and len(out) > 0


def test_load_for_generation_can_override_int8_matmul_for_inference(tmp_path: Path) -> None:
    tok = BpeTokenizer.train(_SUBWORD_TEXT, vocab_size=300)
    model_config = ModelConfig(
        vocab_size=tok.vocab_size,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=8,
        matmul_mode="int8",
    )
    model = build_seeded_model(model_config, max_code=2, seed=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    base = save_checkpoint(
        tmp_path / "ckpt_int8",
        model=model,
        optimizer=optimizer,
        step=0,
        max_code=2,
        vocabulary=(),
        experiment_config={
            "block_size": 16,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "matmul_mode": "int8",
        },
        tokenizer_kind="subword",
        tokenizer_json=tok.to_json(),
    )

    loaded, _ = load_for_generation(base, device="cpu", inference_matmul_mode="fp32")

    ratchet_layers = [
        module for module in loaded.modules() if isinstance(module, DiscreteRatchetLinear)
    ]
    assert ratchet_layers
    assert {module.matmul_mode for module in ratchet_layers} == {"fp32"}


def test_generation_window_right_pads_short_context_to_fixed_shape() -> None:
    ids = torch.tensor([[4, 5, 6]], dtype=torch.long)

    window, next_index = _generation_window(ids, block_size=8, fixed_shape=True)

    assert window.tolist() == [[4, 5, 6, 0, 0, 0, 0, 0]]
    assert next_index == 2


def test_generation_window_crops_long_context_to_fixed_shape() -> None:
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    window, next_index = _generation_window(ids, block_size=3, fixed_shape=True)

    assert window.tolist() == [[3, 4, 5]]
    assert next_index == 2


def test_generation_window_keeps_variable_shape_for_non_int8_path() -> None:
    ids = torch.tensor([[4, 5, 6]], dtype=torch.long)

    window, next_index = _generation_window(ids, block_size=8, fixed_shape=False)

    assert window.tolist() == [[4, 5, 6]]
    assert next_index == 2


def test_int8_generation_uses_fixed_block_size_inputs(tmp_path: Path) -> None:
    base, _ = _save_subword_checkpoint(tmp_path)
    model, decoder = load_for_generation(base, device="cpu")
    for module in model.modules():
        if isinstance(module, DiscreteRatchetLinear):
            module.matmul_mode = "int8"
    seen_lengths: list[int] = []

    def recording_forward(tokens, targets=None):
        seen_lengths.append(tokens.shape[1])
        logits = torch.zeros(
            tokens.shape[0],
            tokens.shape[1],
            model.config.vocab_size,
            dtype=torch.float32,
        )
        return logits, None

    model.forward = recording_forward

    generate(model, decoder, "the", max_new_tokens=5, temperature=0.0, device="cpu")

    assert seen_lengths == [model.config.block_size] * 5


def test_warm_up_generation_runs_one_fixed_shape_forward_for_int8(tmp_path: Path) -> None:
    base, _ = _save_subword_checkpoint(tmp_path)
    model, _ = load_for_generation(base, device="cpu")
    for module in model.modules():
        if isinstance(module, DiscreteRatchetLinear):
            module.matmul_mode = "int8"
    seen_lengths: list[int] = []

    def recording_forward(tokens, targets=None):
        seen_lengths.append(tokens.shape[1])
        logits = torch.zeros(
            tokens.shape[0],
            tokens.shape[1],
            model.config.vocab_size,
            dtype=torch.float32,
        )
        return logits, None

    model.forward = recording_forward

    assert warm_up_generation(model, device="cpu") is True
    assert seen_lengths == [model.config.block_size]


def test_warm_up_generation_skips_non_int8_models(tmp_path: Path) -> None:
    base, _ = _save_subword_checkpoint(tmp_path)
    model, _ = load_for_generation(base, device="cpu")
    calls = 0

    def recording_forward(tokens, targets=None):
        nonlocal calls
        calls += 1
        raise AssertionError("warmup should not run for non-int8 generation")

    model.forward = recording_forward

    assert warm_up_generation(model, device="cpu") is False
    assert calls == 0
