"""Consolidation: residual merge / free / regrow under hard param budget."""

from __future__ import annotations

import torch

from scrt.config import ModelConfig, TrainConfig
from scrt.consolidate import (
    consolidate,
    free_weak_slots,
    functional_signatures,
    merge_slot_pair,
    regrow_free_slots,
)
from scrt.model import SCRTGPT, count_params


def _model() -> SCRTGPT:
    cfg = ModelConfig(
        vocab_size=32,
        n_embd=32,
        n_head=4,
        block_size=16,
        n_loops=3,
        n_slots=6,
        lora_rank=2,
        slots_per_loop=1,
    )
    return SCRTGPT(cfg)


def test_merge_keeps_residual_slot_and_param_count():
    m = _model()
    n0 = count_params(m)
    bank = m.bank
    bank.occupied[:] = False
    bank.assignment[:] = -1
    bank.occupied[0] = True
    bank.occupied[1] = True
    bank.assignment[0] = 0
    bank.assignment[1] = 0
    with torch.no_grad():
        bank.down[1].copy_(bank.down[0])
        bank.up[1].copy_(bank.up[0])
        bank.up[0].add_(0.2)
        bank.up[1].add_(0.1)
    bank.importance[0] = 1.0
    bank.importance[1] = 0.5
    energy = merge_slot_pair(bank, 0, 1)
    # v2: both stay occupied (shared + residual)
    assert bank.n_active() == 2
    assert energy >= 0.0
    assert count_params(m) == n0


def test_regrow_respects_keep_free():
    m = _model()
    bank = m.bank
    # free several
    for s in range(bank.n_slots):
        bank.free_slot(s)
    n = regrow_free_slots(bank, max_regrow=2, keep_free=2)
    assert n == 2
    assert bank.n_free() == bank.n_slots - 2


def test_functional_signatures_shape():
    m = _model()
    x = torch.randint(0, 32, (2, 16))
    y = torch.randint(0, 32, (2, 16))
    sigs = functional_signatures(m, m.bank, [(x, y)])
    assert sigs.shape == (m.bank.n_slots, m.cfg.n_embd)


def test_consolidate_hard_budget_invariant():
    m = _model()
    n0 = count_params(m)
    with torch.no_grad():
        m.bank.down[1].copy_(m.bank.down[0])
        m.bank.up[1].copy_(m.bank.up[0])
        m.bank.up[0].add_(0.1)
        m.bank.up[1].add_(0.1)
        m.bank.importance[0] = 1.0
        m.bank.importance[1] = 1.0

    tcfg = TrainConfig(
        merge_sim_threshold=0.5,
        residual_free_threshold=0.05,
        max_merges_per_cycle=2,
        max_regrow_per_cycle=2,
        keep_free_slots=1,
        probe_batches=1,
        device="cpu",
        arm="scrt",
    )
    probes = [
        (
            torch.randint(0, 32, (2, 16)),
            torch.randint(0, 32, (2, 16)),
        )
    ]
    stats = consolidate(m, probes, tcfg, step=1)
    assert count_params(m) == n0 == stats.n_params
    assert stats.n_active_after <= m.bank.n_slots


def test_free_weak_skips_zero_init_bank():
    m = _model()
    before = m.bank.n_active()
    freed = free_weak_slots(m.bank, threshold=0.05)
    assert freed == 0
    assert m.bank.n_active() == before


def test_merge_only_consolidate_never_frees():
    m = _model()
    n0 = count_params(m)
    active0 = m.bank.n_active()
    with torch.no_grad():
        m.bank.down[1].copy_(m.bank.down[0])
        m.bank.up[1].copy_(m.bank.up[0])
        m.bank.up[0].add_(0.2)
        m.bank.importance[:] = 1.0
    tcfg = TrainConfig(
        merge_sim_threshold=0.5,
        residual_free_threshold=0.99,  # would free aggressively if enabled
        max_merges_per_cycle=2,
        consolidate_enable_free=False,
        consolidate_enable_regrow=False,
        probe_batches=1,
        device="cpu",
        arm="scrt",
    )
    probes = [(torch.randint(0, 32, (2, 16)), torch.randint(0, 32, (2, 16)))]
    stats = consolidate(m, probes, tcfg, step=1)
    assert stats.n_freed == 0
    assert stats.n_regrown == 0
    assert count_params(m) == n0
    # occupancy only shrinks via free; merge-only keeps slots
    assert m.bank.n_active() >= active0 - stats.n_merged  # residual path keeps both
