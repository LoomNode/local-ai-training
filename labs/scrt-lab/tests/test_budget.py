"""Matched parameter budget tests."""

from __future__ import annotations

from scrt.budget import build_matched_models, budget_report, scrt_budget
from scrt.config import ModelConfig, stepped_config
from scrt.model import count_params, model_forward
import torch


def test_matched_models_equal_params():
    cfg = ModelConfig(
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
    matched = build_matched_models(cfg, exact=True)
    target = scrt_budget(cfg)
    for arm, (m, _, n) in matched.items():
        assert n == target, (arm, n, target)
        assert count_params(m) == target


def test_matched_forward_all_arms():
    cfg = ModelConfig(
        vocab_size=32,
        n_embd=32,
        n_head=4,
        block_size=16,
        n_loops=3,
        n_slots=4,
        lora_rank=2,
        slots_per_loop=1,
    )
    matched = build_matched_models(cfg, exact=True)
    x = torch.randint(0, 32, (2, 16))
    y = torch.randint(0, 32, (2, 16))
    for arm, (m, _, _) in matched.items():
        fr = model_forward(m, x, y)
        assert fr.logits.shape[-1] == 32
        assert fr.loss is not None
        fr.loss.backward()


def test_stepped_config_budget_report():
    cfg = stepped_config(device="cpu", steps=10).model
    rep = budget_report(cfg)
    assert rep["target"] > 100_000
    assert "baseline" in rep and "scrt" in rep
