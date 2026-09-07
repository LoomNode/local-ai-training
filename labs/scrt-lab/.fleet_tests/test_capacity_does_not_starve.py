"""Capacity-config consolidate must not starve the slot bank (the open lab issue)."""
import torch
from scrt.config import capacity_config
from scrt.consolidate import consolidate
from scrt.model import SCRTGPT, count_params


def test_capacity_config_has_grace_and_free_cap():
    cfg = capacity_config(device="cpu", steps=100)
    assert cfg.train.min_slot_age_cycles >= 5
    assert cfg.train.max_frees_per_cycle <= 2
    assert cfg.train.max_regrow_per_cycle >= 1


def test_capacity_knobs_do_not_starve_over_80_cycles():
    torch.manual_seed(0)
    from scrt.config import ModelConfig

    m = SCRTGPT(
        ModelConfig(
            vocab_size=32,
            n_embd=32,
            n_head=4,
            block_size=16,
            n_loops=8,
            n_slots=16,
            lora_rank=2,
            slots_per_loop=1,
        )
    )
    cap = capacity_config(device="cpu", steps=100)
    tcfg = cap.train
    g = torch.Generator().manual_seed(0)
    probes = [
        (
            torch.randint(0, 32, (2, 16), generator=g),
            torch.randint(0, 32, (2, 16), generator=g),
        )
        for _ in range(2)
    ]
    n0 = count_params(m)
    start = m.bank.n_active()
    for step in range(1, 81):
        stats = consolidate(m, probes, tcfg, step * 100)
        assert stats.n_params == n0
    assert m.bank.n_active() >= start, f"starved {start} -> {m.bank.n_active()}"
