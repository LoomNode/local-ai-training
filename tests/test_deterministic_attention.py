"""Opt-in deterministic attention backend.

The flash SDPA backward accumulates the query gradient with atomic adds. At head size 128
(the 1B recipe: n_embd 3072 / 24 heads) that makes every backward pass bit-different, so a
resumed run can never match an uninterrupted one. Head size 64 (the 25M recipes) happens to
split keys into two blocks, and two-term float addition is commutative, which is why the
existing exact-resume tests never saw it. `deterministic_attention` pins the efficient/math
backends, whose backward is repeatable.
"""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from local_ai_training.checkpoint import load_checkpoint, save_checkpoint
from local_ai_training.cli import build_parser
from local_ai_training.config import ExperimentConfig
from local_ai_training.model import ModelConfig, build_seeded_model


def test_deterministic_attention_defaults_off_and_round_trips_through_toml(tmp_path: Path):
    assert ExperimentConfig().deterministic_attention is False
    assert ModelConfig(vocab_size=5).deterministic_attention is False
    path = tmp_path / "config.toml"
    path.write_text("[training]\ndeterministic_attention = true\n")
    config = ExperimentConfig.from_toml(path)
    assert config.deterministic_attention is True
    assert config.model_config(vocab_size=5).deterministic_attention is True
    assert config.to_dict()["deterministic_attention"] is True


def test_train_deterministic_attention_flag_defaults_off_and_can_enable():
    parser = build_parser()
    assert parser.parse_args(["train"]).deterministic_attention is False
    assert parser.parse_args(["train", "--deterministic-attention"]).deterministic_attention


def test_resume_rejects_deterministic_attention_mismatch(tmp_path: Path):
    config = ModelConfig(vocab_size=5, block_size=4, n_layer=1, n_head=1, n_embd=8)
    model = build_seeded_model(config, max_code=2, seed=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
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
    with pytest.raises(ValueError, match="deterministic_attention does not match"):
        load_checkpoint(
            checkpoint,
            model=build_seeded_model(config, max_code=2, seed=1),
            optimizer=optimizer,
            expected_max_code=2,
            expected_vocabulary=tuple("abcde"),
            expected_experiment_config={"deterministic_attention": True},
            expected_run_seed=17,
        )


def _attention_grads(model_config: ModelConfig, *, passes: int) -> list[torch.Tensor]:
    torch.manual_seed(0)
    model = build_seeded_model(model_config, max_code=None, seed=3).cuda()
    tokens = torch.randint(0, model_config.vocab_size, (8, model_config.block_size), device="cuda")
    grads = []
    for _ in range(passes):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(tokens)
        logits.float().sum().backward()
        grads.append(model.blocks[0].attention.qkv.weight.grad.detach().clone())
    return grads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA attention backends")
def test_deterministic_attention_makes_head_size_128_backward_repeatable():
    base = ModelConfig(vocab_size=32, block_size=256, n_layer=1, n_head=4, n_embd=512)
    assert base.n_embd // base.n_head == 128
    grads = _attention_grads(replace(base, deterministic_attention=True), passes=4)
    assert all(torch.equal(grads[0], other) for other in grads[1:])
