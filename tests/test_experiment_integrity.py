import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from local_ai_training.checkpoint import load_checkpoint, save_checkpoint
from local_ai_training.config import ExperimentConfig
from local_ai_training.data import build_char_corpus, build_subword_corpus
from local_ai_training.generate import generate, load_for_generation
from local_ai_training.model import ModelConfig, build_seeded_model
from local_ai_training.ratchet import DiscreteRatchetLinear, RatchetEmbedding
from local_ai_training.tokenizer import BpeTokenizer
from local_ai_training.train import train_run


def _config(**changes) -> ExperimentConfig:
    base = ExperimentConfig(
        block_size=4,
        batch_size=2,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
        steps=3,
        eval_interval=3,
        eval_batches=1,
        support_learning_rate=0.02,
        pressure_threshold=2,
        seeds=(17,),
        device="cpu",
    )
    return replace(base, **changes)


def _model_and_optimizer(*, seed: int = 1):
    config = ModelConfig(vocab_size=5, block_size=4, n_layer=1, n_head=1, n_embd=8)
    model = build_seeded_model(config, max_code=2, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    return model, optimizer


def _rewrite_metadata(base: Path, update) -> None:
    path = base.with_suffix(".json")
    metadata = json.loads(path.read_text())
    update(metadata)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_seeded_cpu_model_construction_preserves_cuda_rng() -> None:
    torch.cuda.manual_seed(9182)
    before = torch.cuda.get_rng_state()

    build_seeded_model(
        ModelConfig(vocab_size=5, block_size=4, n_layer=1, n_head=1, n_embd=8),
        max_code=2,
        seed=17,
    )

    assert torch.equal(torch.cuda.get_rng_state(), before)


def test_frozen_run_preserves_every_ratchet_state_but_learns_support_parameters(
    tmp_path: Path,
) -> None:
    corpus = build_char_corpus("abcde" * 200)
    config = _config(
        trainable_scale=True,
        rms_ema_beta=0.9,
        pressure_leak_period=1,
        ratchet_embedding=True,
    )
    initial = build_seeded_model(config.model_config(vocab_size=5), max_code=2, seed=17)
    initial_state = {name: value.detach().clone() for name, value in initial.state_dict().items()}
    ratchet_prefixes = tuple(
        f"{name}." for name, module in initial.named_modules()
        if isinstance(module, DiscreteRatchetLinear)
    )

    result = train_run(
        corpus=corpus,
        config=config,
        max_code=2,
        seed=17,
        run_dir=tmp_path / "frozen",
        weight_mode="frozen",
    )

    tensors = load_file(result.checkpoint.with_suffix(".safetensors"))
    for name, expected in initial_state.items():
        if name.startswith(ratchet_prefixes):
            assert torch.equal(tensors[f"model::{name}"], expected), name
    assert not torch.equal(
        tensors["model::blocks.0.attention_norm.weight"],
        initial_state["blocks.0.attention_norm.weight"],
    )
    metadata = json.loads(result.checkpoint.with_suffix(".json").read_text())
    assert set(metadata["ratchet_update_counts"].values()) == {0}


def test_frozen_run_with_fp_embedding_checkpointing_and_compiled_update_learns_embedding(
    tmp_path: Path,
) -> None:
    corpus = build_char_corpus("abcde" * 200)
    config = _config(
        compile_update=True,
        gradient_checkpointing=True,
        trainable_scale=True,
        rms_ema_beta=0.9,
        pressure_leak_period=1,
    )
    initial = build_seeded_model(config.model_config(vocab_size=5), max_code=2, seed=17)
    initial_state = {name: value.detach().clone() for name, value in initial.state_dict().items()}
    ratchet_prefixes = tuple(
        f"{name}." for name, module in initial.named_modules()
        if isinstance(module, DiscreteRatchetLinear)
    )

    result = train_run(
        corpus=corpus,
        config=config,
        max_code=2,
        seed=17,
        run_dir=tmp_path / "frozen-fp-embedding",
        weight_mode="frozen",
    )

    tensors = load_file(result.checkpoint.with_suffix(".safetensors"))
    for name, expected in initial_state.items():
        if name.startswith(ratchet_prefixes):
            assert torch.equal(tensors[f"model::{name}"], expected), name
    assert not torch.equal(
        tensors["model::token_embedding.weight"], initial_state["token_embedding.weight"]
    )


@pytest.mark.parametrize("layer_kind", ["linear", "embedding"])
def test_disabled_ratchet_update_guard_preserves_state_and_upstream_gradients(
    layer_kind: str,
) -> None:
    if layer_kind == "linear":
        layer = DiscreteRatchetLinear(
            4,
            3,
            max_code=2,
            trainable_scale=True,
            rms_ema_beta=0.9,
            pressure_leak_period=1,
            fuse_backward_update=True,
        )
        inputs = torch.randn(2, 4, requires_grad=True)
        forward = partial(layer, inputs)
    else:
        layer = RatchetEmbedding(
            5,
            4,
            max_code=2,
            trainable_scale=True,
            rms_ema_beta=0.9,
            pressure_leak_period=1,
        )
        inputs = torch.tensor([[0, 1, 2], [2, 3, 4]])
        forward = partial(layer, inputs)
    support = torch.nn.Parameter(torch.ones(4 if layer_kind == "embedding" else 3))
    layer.log_scale.grad = torch.ones_like(layer.log_scale)
    layer.set_update_enabled(False)
    before = {name: value.detach().clone() for name, value in layer.state_dict().items()}

    (forward() * support).sum().backward()
    stats = layer.ratchet_update()

    assert support.grad is not None and torch.count_nonzero(support.grad)
    if layer_kind == "linear":
        assert inputs.grad is not None and torch.count_nonzero(inputs.grad)
    assert all(torch.equal(value, before[name]) for name, value in layer.state_dict().items())
    assert layer._update_count == 0
    assert not layer.has_pending_gradient
    assert stats.code_moves == 0
    assert layer.log_scale.grad is None


@pytest.mark.parametrize("matmul_mode", ["bf16", "int8"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_frozen_cuda_modes_preserve_ratchet_state(
    tmp_path: Path, matmul_mode: str
) -> None:
    corpus = build_char_corpus("abcde" * 200)
    config = _config(
        steps=1,
        eval_interval=1,
        device="cuda",
        matmul_mode=matmul_mode,
        int8_backward=matmul_mode == "int8",
        trainable_scale=True,
        pressure_leak_period=1,
    )
    initial = build_seeded_model(config.model_config(vocab_size=5), max_code=2, seed=17)
    prefixes = tuple(
        f"{name}." for name, module in initial.named_modules()
        if isinstance(module, DiscreteRatchetLinear)
    )
    expected = {
        name: value.clone() for name, value in initial.state_dict().items()
        if name.startswith(prefixes)
    }

    result = train_run(
        corpus=corpus,
        config=config,
        max_code=2,
        seed=17,
        run_dir=tmp_path / matmul_mode,
        weight_mode="frozen",
    )

    tensors = load_file(result.checkpoint.with_suffix(".safetensors"))
    assert all(torch.equal(tensors[f"model::{name}"], value) for name, value in expected.items())


def test_subword_resume_rejects_different_equal_size_tokenizer_before_mutation(
    tmp_path: Path,
) -> None:
    saved_tokenizer = BpeTokenizer.train("abab abab abab", vocab_size=258)
    other_tokenizer = BpeTokenizer.train("cdcd cdcd cdcd", vocab_size=258)
    assert saved_tokenizer.vocab_size == other_tokenizer.vocab_size
    model, optimizer = _model_and_optimizer(seed=1)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        step=1,
        max_code=2,
        vocabulary=(),
        experiment_config={"matmul_mode": "fp32"},
        tokenizer_kind="subword",
        tokenizer_json=saved_tokenizer.to_json(),
        run_seed=17,
    )
    restored, restored_optimizer = _model_and_optimizer(seed=99)
    before_model = {name: value.clone() for name, value in restored.state_dict().items()}
    before_rng = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="checkpoint tokenizer does not match dataset"):
        load_checkpoint(
            checkpoint,
            model=restored,
            optimizer=restored_optimizer,
            expected_max_code=2,
            expected_vocabulary=(),
            expected_tokenizer_kind="subword",
            expected_tokenizer_json=other_tokenizer.to_json(),
            expected_run_seed=17,
        )

    assert all(
        torch.equal(value, before_model[name]) for name, value in restored.state_dict().items()
    )
    assert restored_optimizer.state == {}
    assert torch.equal(torch.get_rng_state(), before_rng)


def test_subword_resume_accepts_equivalent_json_serialization(tmp_path: Path) -> None:
    tokenizer = BpeTokenizer.train("abab abab abab", vocab_size=258)
    equivalent_json = json.dumps(json.loads(tokenizer.to_json()), indent=2, sort_keys=True)
    model, optimizer = _model_and_optimizer()
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        step=1,
        max_code=2,
        vocabulary=(),
        experiment_config={"matmul_mode": "fp32"},
        tokenizer_kind="subword",
        tokenizer_json=tokenizer.to_json(),
        run_seed=17,
    )

    metadata = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected_max_code=2,
        expected_vocabulary=(),
        expected_tokenizer_kind="subword",
        expected_tokenizer_json=equivalent_json,
        expected_run_seed=17,
    )

    assert json.loads(metadata["tokenizer_json"]) == json.loads(equivalent_json)


def test_checkpoint_v2_saves_seed_cpu_rng_and_leak_counters(tmp_path: Path) -> None:
    model, optimizer = _model_and_optimizer()
    layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, DiscreteRatchetLinear)
    }
    for index, module in enumerate(layers.values(), start=1):
        module._update_count = index
    expected_rng = torch.get_rng_state().clone()

    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        step=1,
        max_code=2,
        vocabulary=tuple("abcde"),
        experiment_config={"matmul_mode": "fp32", "pressure_leak_period": 3},
        run_seed=17,
    )

    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    tensors = load_file(checkpoint.with_suffix(".safetensors"))
    assert metadata["format_version"] == 2
    assert metadata["run_seed"] == 17
    assert metadata["ratchet_update_counts"] == {
        name: index for index, name in enumerate(layers, start=1)
    }
    assert torch.equal(tensors["rng::cpu"], expected_rng)


@pytest.mark.parametrize("malformation", ["missing", "negative"])
def test_invalid_leak_counter_metadata_rejects_before_model_mutation(
    tmp_path: Path, malformation: str
) -> None:
    model, optimizer = _model_and_optimizer(seed=1)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        step=1,
        max_code=2,
        vocabulary=tuple("abcde"),
        experiment_config={"matmul_mode": "fp32", "pressure_leak_period": 3},
        run_seed=17,
    )

    def corrupt(metadata) -> None:
        counters = metadata["ratchet_update_counts"]
        name = next(iter(counters))
        if malformation == "missing":
            counters.pop(name)
        else:
            counters[name] = -1

    _rewrite_metadata(checkpoint, corrupt)
    restored, restored_optimizer = _model_and_optimizer(seed=99)
    before = {name: value.clone() for name, value in restored.state_dict().items()}

    with pytest.raises(ValueError, match="ratchet leak counters"):
        load_checkpoint(
            checkpoint,
            model=restored,
            optimizer=restored_optimizer,
            expected_max_code=2,
            expected_vocabulary=tuple("abcde"),
            expected_experiment_config={"pressure_leak_period": 3},
            expected_run_seed=17,
        )

    assert all(torch.equal(value, before[name]) for name, value in restored.state_dict().items())
    assert restored_optimizer.state == {}


@pytest.mark.parametrize(
    ("config_update", "message"),
    [
        ({"pressure_leak_period": 3}, "legacy checkpoint is missing ratchet leak counters"),
        ({"dropout": 0.25}, "legacy CUDA checkpoint is missing CUDA RNG state"),
    ],
)
def test_legacy_training_resume_rejects_only_relevant_missing_state(
    tmp_path: Path, config_update: dict[str, object], message: str
) -> None:
    model, optimizer = _model_and_optimizer()
    config = {"matmul_mode": "fp32", **config_update}
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        step=1,
        max_code=2,
        vocabulary=tuple("abcde"),
        experiment_config=config,
        run_seed=17,
    )
    _rewrite_metadata(
        checkpoint,
        lambda metadata: (
            metadata.update(format_version=1),
            metadata.pop("ratchet_update_counts", None),
        ),
    )

    with pytest.raises(ValueError, match=message):
        load_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_max_code=2,
            expected_vocabulary=tuple("abcde"),
            expected_experiment_config=config,
            expected_run_seed=17,
            training_device=("cuda" if "dropout" in config_update else "cpu"),
        )


def test_legacy_frozen_checkpoint_does_not_require_unused_leak_counters(
    tmp_path: Path,
) -> None:
    model, optimizer = _model_and_optimizer()
    model.set_ratchet_updates_enabled(False)
    config = {
        "matmul_mode": "fp32",
        "pressure_leak_period": 3,
        "weight_mode": "frozen",
    }
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        step=1,
        max_code=2,
        vocabulary=tuple("abcde"),
        experiment_config=config,
        run_seed=17,
    )
    _rewrite_metadata(
        checkpoint,
        lambda metadata: (
            metadata.update(format_version=1),
            metadata.pop("ratchet_update_counts", None),
        ),
    )

    metadata = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected_max_code=2,
        expected_vocabulary=tuple("abcde"),
        expected_experiment_config=config,
        expected_run_seed=17,
    )

    assert metadata["format_version"] == 1


def test_legacy_deterministic_checkpoint_can_resume_and_generate(tmp_path: Path) -> None:
    model, optimizer = _model_and_optimizer()
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        step=1,
        max_code=2,
        vocabulary=tuple("abcde"),
        experiment_config={
            "block_size": 4,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "matmul_mode": "fp32",
            "dropout": 0.0,
            "pressure_leak_period": 0,
            "int8_backward": True,
        },
        run_seed=17,
    )
    _rewrite_metadata(
        checkpoint,
        lambda metadata: (
            metadata.update(format_version=1),
            metadata.pop("ratchet_update_counts", None),
        ),
    )

    metadata = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected_max_code=2,
        expected_vocabulary=tuple("abcde"),
        expected_experiment_config={
            "dropout": 0.0,
            "pressure_leak_period": 0,
            "int8_backward": True,
        },
        expected_run_seed=17,
        training_device="cuda",
    )
    generated_model, vocabulary = load_for_generation(checkpoint, device="cpu")

    assert metadata["format_version"] == 1
    assert len(generate(generated_model, vocabulary, "a", max_new_tokens=2, temperature=0.0)) == 2


def test_legacy_subword_checkpoint_remains_usable_for_generation(tmp_path: Path) -> None:
    tokenizer = BpeTokenizer.train("abab abab abab", vocab_size=258)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=4,
        n_layer=1,
        n_head=1,
        n_embd=8,
    )
    model = build_seeded_model(config, max_code=2, seed=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        step=1,
        max_code=2,
        vocabulary=(),
        experiment_config={
            "block_size": 4,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "matmul_mode": "fp32",
        },
        tokenizer_kind="subword",
        tokenizer_json=tokenizer.to_json(),
        run_seed=17,
    )
    _rewrite_metadata(
        checkpoint,
        lambda metadata: (
            metadata.update(format_version=1),
            metadata.pop("run_seed", None),
        ),
    )

    with pytest.raises(ValueError, match="legacy checkpoint is missing run seed"):
        load_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_max_code=2,
            expected_vocabulary=(),
            expected_tokenizer_kind="subword",
            expected_tokenizer_json=tokenizer.to_json(),
            expected_run_seed=17,
        )

    generated_model, decoder = load_for_generation(checkpoint, device="cpu")

    assert isinstance(decoder, BpeTokenizer)
    assert generate(generated_model, decoder, "ab", max_new_tokens=1, temperature=0.0)


def test_legacy_wrong_run_seed_rejects_before_mutation(tmp_path: Path) -> None:
    model, optimizer = _model_and_optimizer(seed=1)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        step=1,
        max_code=2,
        vocabulary=tuple("abcde"),
        experiment_config={"matmul_mode": "fp32"},
        run_seed=17,
    )
    _rewrite_metadata(
        checkpoint,
        lambda metadata: metadata.update(format_version=1, run_seed=99),
    )
    restored, restored_optimizer = _model_and_optimizer(seed=99)
    before = {name: value.clone() for name, value in restored.state_dict().items()}
    before_rng = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="checkpoint run seed does not match requested run"):
        load_checkpoint(
            checkpoint,
            model=restored,
            optimizer=restored_optimizer,
            expected_max_code=2,
            expected_vocabulary=tuple("abcde"),
            expected_run_seed=17,
        )

    assert all(torch.equal(value, before[name]) for name, value in restored.state_dict().items())
    assert restored_optimizer.state == {}
    assert torch.equal(torch.get_rng_state(), before_rng)


def test_resume_config_mismatch_rejects_before_mutating_model_optimizer_or_rng(
    tmp_path: Path,
) -> None:
    model, optimizer = _model_and_optimizer(seed=1)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        step=1,
        max_code=2,
        vocabulary=tuple("abcde"),
        experiment_config={"matmul_mode": "fp32", "dropout": 0.25},
        run_seed=17,
    )
    restored, restored_optimizer = _model_and_optimizer(seed=99)
    before_model = {name: value.clone() for name, value in restored.state_dict().items()}
    before_rng = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="checkpoint dropout does not match requested run"):
        load_checkpoint(
            checkpoint,
            model=restored,
            optimizer=restored_optimizer,
            expected_max_code=2,
            expected_vocabulary=tuple("abcde"),
            expected_experiment_config={"dropout": 0.5},
            expected_run_seed=17,
        )

    assert all(
        torch.equal(value, before_model[name]) for name, value in restored.state_dict().items()
    )
    assert restored_optimizer.state == {}
    assert torch.equal(torch.get_rng_state(), before_rng)


def test_dropout_and_off_boundary_leak_resume_matches_uninterrupted_exactly(
    tmp_path: Path,
) -> None:
    corpus = build_char_corpus("abcde" * 200)
    full_config = _config(
        steps=7,
        eval_interval=7,
        dropout=0.2,
        pressure_leak_period=3,
        rms_ema_beta=0.9,
    )
    partial_config = replace(full_config, steps=4)
    uninterrupted = train_run(
        corpus=corpus,
        config=full_config,
        max_code=2,
        seed=17,
        run_dir=tmp_path / "full",
    )
    partial = train_run(
        corpus=corpus,
        config=partial_config,
        max_code=2,
        seed=17,
        run_dir=tmp_path / "split",
    )
    resumed = train_run(
        corpus=corpus,
        config=full_config,
        max_code=2,
        seed=17,
        run_dir=tmp_path / "split",
        resume_from=partial.checkpoint,
    )

    full_tensors = load_file(uninterrupted.checkpoint.with_suffix(".safetensors"))
    resumed_tensors = load_file(resumed.checkpoint.with_suffix(".safetensors"))
    assert full_tensors.keys() == resumed_tensors.keys()
    for name in full_tensors:
        assert torch.equal(full_tensors[name], resumed_tensors[name]), name
    full_metadata = json.loads(uninterrupted.checkpoint.with_suffix(".json").read_text())
    resumed_metadata = json.loads(resumed.checkpoint.with_suffix(".json").read_text())
    assert full_metadata["ratchet_update_counts"] == resumed_metadata["ratchet_update_counts"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_dropout_int8_backward_resume_matches_uninterrupted_exactly(
    tmp_path: Path,
) -> None:
    corpus = build_char_corpus("abcde" * 200)
    full_config = _config(
        steps=4,
        eval_interval=4,
        dropout=0.2,
        pressure_leak_period=3,
        matmul_mode="int8",
        int8_backward=True,
        device="cuda",
    )
    partial_config = replace(full_config, steps=2)
    uninterrupted = train_run(
        corpus=corpus, config=full_config, max_code=2, seed=17, run_dir=tmp_path / "full"
    )
    partial = train_run(
        corpus=corpus, config=partial_config, max_code=2, seed=17, run_dir=tmp_path / "split"
    )
    resumed = train_run(
        corpus=corpus,
        config=full_config,
        max_code=2,
        seed=17,
        run_dir=tmp_path / "split",
        resume_from=partial.checkpoint,
    )

    full_tensors = load_file(uninterrupted.checkpoint.with_suffix(".safetensors"))
    resumed_tensors = load_file(resumed.checkpoint.with_suffix(".safetensors"))
    assert full_tensors.keys() == resumed_tensors.keys()
    for name in full_tensors:
        assert torch.equal(full_tensors[name], resumed_tensors[name]), name


def test_train_run_rejects_equal_size_subword_tokenizer_change(tmp_path: Path) -> None:
    saved_tokenizer = BpeTokenizer.train("abab abab abab " * 100, vocab_size=258)
    other_tokenizer = BpeTokenizer.train("cdcd cdcd cdcd " * 100, vocab_size=258)
    saved_corpus = build_subword_corpus("abab abab abab " * 100, saved_tokenizer)
    other_corpus = build_subword_corpus("cdcd cdcd cdcd " * 100, other_tokenizer)
    partial = train_run(
        corpus=saved_corpus,
        config=_config(steps=1, eval_interval=1),
        max_code=2,
        seed=17,
        run_dir=tmp_path / "run",
    )
    metrics_before = partial.metrics_csv.read_bytes()
    tensors_before = partial.checkpoint.with_suffix(".safetensors").read_bytes()
    metadata_before = partial.checkpoint.with_suffix(".json").read_bytes()
    cuda_rng_before = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    with pytest.raises(ValueError, match="checkpoint tokenizer does not match dataset"):
        train_run(
            corpus=other_corpus,
            config=_config(steps=2, eval_interval=2),
            max_code=2,
            seed=17,
            run_dir=tmp_path / "run",
            resume_from=partial.checkpoint,
        )
    assert partial.metrics_csv.read_bytes() == metrics_before
    assert partial.checkpoint.with_suffix(".safetensors").read_bytes() == tensors_before
    assert partial.checkpoint.with_suffix(".json").read_bytes() == metadata_before
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng_before)
