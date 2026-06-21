import json
from dataclasses import replace

import pytest
import torch

from local_ai_training.config import ExperimentConfig
from local_ai_training.int8_convergence_preflight import (
    PreflightMismatch,
    assert_configs_matched,
    assert_model_states_matched,
    assert_rng_unchanged,
    assert_schedules_matched,
    cuda_uuid_text,
)
from local_ai_training.model import ModelConfig, build_seeded_model


def tiny_model_config(mode: str) -> ModelConfig:
    return ModelConfig(
        vocab_size=11,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        matmul_mode=mode,
    )


def test_configs_may_differ_only_at_matmul_mode() -> None:
    bf16 = ExperimentConfig(matmul_mode="bf16", seeds=(1337,))
    int8 = replace(bf16, matmul_mode="int8")

    assert assert_configs_matched(bf16, int8) == "matmul_mode"


def test_config_comparison_rejects_an_additional_mismatch() -> None:
    bf16 = ExperimentConfig(matmul_mode="bf16", seeds=(1337,))
    int8 = replace(bf16, matmul_mode="int8", steps=bf16.steps + 1)

    with pytest.raises(PreflightMismatch, match="steps"):
        assert_configs_matched(bf16, int8)


def test_seeded_models_match_all_persistent_state() -> None:
    bf16 = build_seeded_model(tiny_model_config("bf16"), max_code=4, seed=1337)
    int8 = build_seeded_model(tiny_model_config("int8"), max_code=4, seed=1337)

    summary = assert_model_states_matched(bf16, int8)

    assert summary["tensor_count"] == len(bf16.state_dict())
    assert summary["persistent_bytes"] > 0


@pytest.mark.parametrize(
    ("state_name", "replacement"),
    [
        ("blocks.0.attention.qkv.packed", lambda value: value ^ 1),
        ("blocks.0.attention.qkv._scale", lambda value: value + 1),
        ("token_embedding.weight", lambda value: value + 1),
        ("position_encoding", lambda value: value + 1),
    ],
)
def test_model_comparison_rejects_each_state_category(state_name, replacement) -> None:
    bf16 = build_seeded_model(tiny_model_config("bf16"), max_code=4, seed=1337)
    int8 = build_seeded_model(tiny_model_config("int8"), max_code=4, seed=1337)
    target = dict(int8.named_parameters()) | dict(int8.named_buffers())
    with torch.no_grad():
        target[state_name].copy_(replacement(target[state_name]))

    with pytest.raises(PreflightMismatch, match=state_name):
        assert_model_states_matched(bf16, int8)


def test_complete_train_and_evaluation_schedules_match() -> None:
    train = torch.arange(24).reshape(6, 4)
    evaluation = torch.arange(12).reshape(3, 4)

    summary = assert_schedules_matched(train, train.clone(), evaluation, evaluation.clone())

    assert summary["training"]["shape"] == [6, 4]
    assert summary["evaluation"]["shape"] == [3, 4]
    assert summary["training"]["sha256"] != summary["evaluation"]["sha256"]


@pytest.mark.parametrize("schedule", ["training", "evaluation"])
def test_schedule_comparison_rejects_mismatch(schedule: str) -> None:
    train_bf16 = torch.arange(24).reshape(6, 4)
    train_int8 = train_bf16.clone()
    eval_bf16 = torch.arange(12).reshape(3, 4)
    eval_int8 = eval_bf16.clone()
    if schedule == "training":
        train_int8[-1, -1] += 1
    else:
        eval_int8[-1, -1] += 1

    with pytest.raises(PreflightMismatch, match=schedule):
        assert_schedules_matched(train_bf16, train_int8, eval_bf16, eval_int8)


def test_rng_check_accepts_an_rng_neutral_operation() -> None:
    summary = assert_rng_unchanged("neutral", lambda: torch.ones(2) + 1)

    assert summary == {"name": "neutral", "cpu": "unchanged", "cuda": "unchanged"}


def test_rng_check_rejects_cpu_rng_consumption() -> None:
    with pytest.raises(PreflightMismatch, match="CPU RNG"):
        assert_rng_unchanged("consumer", lambda: torch.rand(1))


def test_cuda_uuid_provenance_is_json_serializable() -> None:
    class PrivateCudaUuid:
        def __str__(self) -> str:
            return "GPU-test-uuid"

    assert json.dumps({"uuid": cuda_uuid_text(PrivateCudaUuid())}) == (
        '{"uuid": "GPU-test-uuid"}'
    )
