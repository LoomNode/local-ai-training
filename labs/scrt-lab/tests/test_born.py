"""Born-consolidated arms: shared residual by construction."""

from __future__ import annotations

import torch

from scrt.budget import BORN_ARMS, born_budget, build_born_matched_models
from scrt.config import ModelConfig, born_config
from scrt.model import (
    ModulatedResidualGPT,
    SharedPlusTinyGPT,
    SharedResidualGPT,
    build_model,
    count_params,
    model_forward,
)


def _cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=48,
        n_embd=64,
        n_head=4,
        block_size=32,
        n_layer=4,
        n_loops=4,
        n_slots=8,
        lora_rank=4,
        slots_per_loop=1,
    )


def test_build_born_models_forward():
    cfg = _cfg()
    for arm in (
        "shared_residual",
        "modulated",
        "shared_plus_tiny",
        "looped",
        "baseline",
    ):
        m = build_model(arm, cfg)
        x = torch.randint(0, 48, (2, 16))
        y = torch.randint(0, 48, (2, 16))
        fr = model_forward(m, x, y)
        assert fr.logits.shape == (2, 16, 48)
        assert fr.loss is not None


def test_shared_plus_tiny_is_shared_plus_difference():
    cfg = _cfg()
    m = SharedPlusTinyGPT(cfg)
    assert count_params(m) > count_params(SharedResidualGPT(cfg))
    assert len(m.tiny) == cfg.n_loops
    # shared rank full; tiny rank smaller
    assert m.shared_lora.down.out_features == cfg.lora_rank
    assert m.tiny[0].down.out_features == cfg.tiny_lora_rank


def test_shared_residual_fewer_params_than_scrt_bank():
    from scrt.model import SCRTGPT

    cfg = _cfg()
    assert count_params(SharedResidualGPT(cfg)) < count_params(SCRTGPT(cfg))
    assert count_params(SharedResidualGPT(cfg)) > count_params(
        build_model("looped", cfg)
    )


def test_born_matched_equal_params():
    cfg = _cfg()
    matched = build_born_matched_models(cfg, exact=True)
    target = born_budget(cfg)
    assert set(BORN_ARMS) == set(matched.keys())
    for arm in BORN_ARMS:
        assert count_params(matched[arm][0]) == target


def test_modulated_gate_shapes():
    cfg = _cfg()
    m = ModulatedResidualGPT(cfg)
    x = torch.randint(0, 48, (3, 16))
    logits, loss = m(x, x)
    assert logits.shape[0] == 3
    assert loss is None or loss.ndim == 0


def test_born_config():
    cfg = born_config(device="cpu", steps=100)
    assert cfg.train.consolidate_every == 0
    assert cfg.train.run_name == "born"
