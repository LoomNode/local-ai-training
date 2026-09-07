"""Model shape and param-budget smoke tests."""

from __future__ import annotations

import torch

from scrt.config import ModelConfig
from scrt.model import (
    BaselineGPT,
    LoopedGPT,
    SCRTGPT,
    build_model,
    count_params,
    model_forward,
)


def _cfg(**kwargs) -> ModelConfig:
    base = dict(
        vocab_size=32,
        n_embd=32,
        n_head=4,
        block_size=16,
        n_layer=2,
        n_loops=3,
        n_slots=4,
        lora_rank=2,
        slots_per_loop=1,
    )
    base.update(kwargs)
    return ModelConfig(**base)


def test_baseline_forward_shapes():
    cfg = _cfg()
    m = BaselineGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 16))
    y = torch.randint(0, cfg.vocab_size, (4, 16))
    logits, loss = m(x, y)
    assert logits.shape == (4, 16, cfg.vocab_size)
    assert loss is not None and loss.ndim == 0


def test_looped_and_scrt_forward():
    cfg = _cfg()
    for arm in ("looped", "scrt"):
        m = build_model(arm, cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))
        fr = model_forward(m, x, y)
        assert fr.logits.shape == (2, 16, cfg.vocab_size)
        assert fr.loss is not None


def test_scrt_param_count_stable_after_forward_backward():
    cfg = _cfg()
    m = SCRTGPT(cfg)
    n0 = count_params(m)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss, _ = m(x, y)
    loss.backward()
    assert count_params(m) == n0


def test_looped_has_fewer_params_than_deep_baseline():
    """Shared core should beat a same-depth stack on param count."""
    cfg = _cfg(n_layer=6, n_loops=6)
    base = BaselineGPT(ModelConfig(**{**cfg.__dict__, "n_layer": 6}))
    looped = LoopedGPT(cfg)
    # prelude + core + coda = 3 blocks vs 6 independent
    assert count_params(looped) < count_params(base)


def test_slot_bank_active_init():
    cfg = _cfg(n_slots=6, n_loops=3, slots_per_loop=1)
    m = SCRTGPT(cfg)
    assert m.bank.n_active() == 3
    assert m.bank.n_free() == 3
