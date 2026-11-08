"""Free/regrow thrash: a regrown slot must get time to bootstrap before it can die.

The capacity run (4 seeds, results/capacity_agg.json) produced 50 merges against
1351 frees and 1335 regrows over 800 cycles, with active slots collapsing 14 -> 4
of 16 by step 8000 in every seed. Cause: ``allocate_slot`` zeroes ``up`` and resets
``importance``, and ``free_weak_slots`` frees on low energy AND low importance — so
a freshly regrown slot satisfies the free condition by construction, and frees were
the only unthrottled operation in the cycle.
"""

from __future__ import annotations

import torch

from scrt.config import ModelConfig, TrainConfig
from scrt.consolidate import consolidate, free_weak_slots
from scrt.model import SCRTGPT, count_params


def _model(n_slots: int = 8, n_loops: int = 4) -> SCRTGPT:
    cfg = ModelConfig(
        vocab_size=32,
        n_embd=32,
        n_head=4,
        block_size=16,
        n_loops=n_loops,
        n_slots=n_slots,
        lora_rank=2,
        slots_per_loop=1,
    )
    return SCRTGPT(cfg)


def _probes(m: SCRTGPT, n: int = 2) -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(0)
    return [
        (
            torch.randint(0, 32, (2, 16), generator=g),
            torch.randint(0, 32, (2, 16), generator=g),
        )
        for _ in range(n)
    ]


def test_new_slots_start_at_age_zero_and_tick():
    m = _model()
    bank = m.bank
    assert bank.age.shape == (bank.n_slots,)
    assert int(bank.age.max().item()) == 0
    bank.tick_age()
    # Only occupied slots accumulate age.
    for s in range(bank.n_slots):
        expected = 1 if bool(bank.occupied[s]) else 0
        assert int(bank.age[s].item()) == expected


def test_allocate_resets_age():
    m = _model()
    bank = m.bank
    for _ in range(5):
        bank.tick_age()
    free_i = int((~bank.occupied).nonzero().view(-1)[0].item())
    bank.allocate_slot(free_i, 0)
    assert int(bank.age[free_i].item()) == 0


def test_young_slot_is_not_freed():
    """A slot regrown this cycle must survive to the next one."""
    m = _model()
    bank = m.bank
    with torch.no_grad():
        # One strong established slot so the weak ones look weak by comparison.
        bank.up[0].add_(1.0)
        bank.importance[0] = 1.0
    for _ in range(10):
        bank.tick_age()

    free_i = int((~bank.occupied).nonzero().view(-1)[0].item())
    bank.allocate_slot(free_i, 0)  # age 0, up==0, importance 0 -> maximally weak

    before = {s for s in range(bank.n_slots) if bool(bank.occupied[s])}
    free_weak_slots(bank, threshold=0.12, min_age=3)
    after = {s for s in range(bank.n_slots) if bool(bank.occupied[s])}

    assert bool(bank.occupied[free_i]), "young slot was freed before it could learn"
    # The old weak slots are fair game; only the young one is protected.
    assert free_i not in (before - after)


def test_slot_becomes_eligible_once_old_enough():
    m = _model()
    bank = m.bank
    with torch.no_grad():
        bank.up[0].add_(1.0)
        bank.importance[0] = 1.0
    free_i = int((~bank.occupied).nonzero().view(-1)[0].item())
    bank.allocate_slot(free_i, 0)
    for _ in range(5):
        bank.tick_age()
    freed = free_weak_slots(bank, threshold=0.12, min_age=3)
    assert freed >= 1
    assert not bool(bank.occupied[free_i])


def test_frees_are_capped_per_cycle():
    m = _model(n_slots=8)
    bank = m.bank
    with torch.no_grad():
        bank.occupied[:] = True
        bank.assignment[:] = 0
        bank.up.zero_()
        bank.down.zero_()
        bank.up[0].add_(1.0)  # single strong slot; all others are weak
        bank.importance.zero_()
        bank.importance[0] = 1.0
    for _ in range(10):
        bank.tick_age()
    freed = free_weak_slots(bank, threshold=0.12, max_free=2, min_age=0)
    assert freed == 2


def test_consolidate_does_not_starve_the_bank():
    """Repeated cycles must not drive occupancy toward zero.

    This is the regression the capacity run exposed: 200 cycles took the bank from
    8 active slots to 4 while doing almost no merging.
    """
    torch.manual_seed(0)
    m = _model(n_slots=16, n_loops=8)
    cfg = TrainConfig(
        arm="scrt",
        merge_sim_threshold=0.90,
        residual_free_threshold=0.12,
        max_merges_per_cycle=2,
        max_regrow_per_cycle=2,
        max_frees_per_cycle=2,
        min_slot_age_cycles=5,
        keep_free_slots=2,
    )
    probes = _probes(m)
    n0 = count_params(m)
    start_active = m.bank.n_active()
    for step in range(1, 41):
        stats = consolidate(m, probes, cfg, step * 100)
        assert stats.n_params == n0

    assert m.bank.n_active() >= start_active, (
        f"bank starved: {start_active} -> {m.bank.n_active()} active slots"
    )


def test_init_occupied_sets_exact_count_without_changing_params():
    """Occupancy is a buffer, so varying it must not move the parameter count."""
    base = ModelConfig(
        vocab_size=32, n_embd=32, n_head=4, block_size=16,
        n_loops=8, n_slots=16, lora_rank=2, slots_per_loop=1,
    )
    counts = set()
    for n in (4, 8, 16):
        import dataclasses

        m = SCRTGPT(dataclasses.replace(base, init_occupied=n))
        assert m.bank.n_active() == n
        # every occupied slot has a real loop assignment
        for s in range(m.bank.n_slots):
            if bool(m.bank.occupied[s]):
                assert 0 <= int(m.bank.assignment[s].item()) < base.n_loops
        counts.add(count_params(m))
    assert len(counts) == 1, f"param count varied with occupancy: {counts}"
