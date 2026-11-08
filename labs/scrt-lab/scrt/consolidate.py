"""Functional consolidation under a hard residual-slot budget.

v2 changes vs thrashy v1:
  - merge keeps **shared** on survivor and **residual difference** on the other
    slot (only free when residual energy is tiny)
  - cap merges per cycle
  - optional free-capacity floor (do not regrow every free slot)
  - higher default similarity bar (set in TrainConfig)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from scrt.config import TrainConfig
from scrt.model import ResidualSlotBank, SCRTGPT


@dataclass
class ConsolidateStats:
    step: int
    n_active_before: int
    n_active_after: int
    n_merged: int
    n_freed: int
    n_regrown: int
    n_params: int
    pair_sims: list[float]
    residual_energies: list[float]

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "n_active_before": self.n_active_before,
            "n_active_after": self.n_active_after,
            "n_merged": self.n_merged,
            "n_freed": self.n_freed,
            "n_regrown": self.n_regrown,
            "n_params": self.n_params,
            "max_pair_sim": max(self.pair_sims) if self.pair_sims else None,
            "mean_pair_sim": (
                sum(self.pair_sims) / len(self.pair_sims) if self.pair_sims else None
            ),
            "mean_residual_energy": (
                sum(self.residual_energies) / len(self.residual_energies)
                if self.residual_energies
                else None
            ),
        }


def _flatten_signature(delta: torch.Tensor) -> torch.Tensor:
    v = delta.detach().float().mean(dim=(0, 1)).reshape(-1)
    n = v.norm().clamp_min(1e-8)
    return v / n


@torch.no_grad()
def functional_signatures(
    model: SCRTGPT,
    bank: ResidualSlotBank,
    probe_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """(n_slots, D) unit signatures from residual deltas on probe inputs."""
    device = bank.down.device
    D = bank.n_embd
    sigs = torch.zeros(bank.n_slots, D, device=device)
    counts = torch.zeros(bank.n_slots, device=device)

    model.eval()
    for idx, _targets in probe_batches:
        idx = idx.to(device)
        x = model.embed(idx)
        x = model.prelude(x)
        for loop_i in range(model.n_loops):
            x = model.core(x)
            for s in range(bank.n_slots):
                if not bool(bank.occupied[s]):
                    continue
                if int(bank.assignment[s].item()) != loop_i:
                    continue
                delta = bank.slot_delta(x, s)
                sigs[s] += _flatten_signature(delta)
                counts[s] += 1.0
            x = x + bank(x, loop_i)

    for s in range(bank.n_slots):
        if counts[s] > 0:
            v = sigs[s] / counts[s]
            sigs[s] = v / v.norm().clamp_min(1e-8)
    model.train()
    return sigs


@torch.no_grad()
def merge_slot_pair(bank: ResidualSlotBank, i: int, j: int) -> float:
    """Merge into shared + residual (not average-and-drop).

    Survivor (higher importance) gets the shared component.
    Other slot keeps the residual difference on the same loop assignment.
    Returns residual energy of the residual slot (for free decisions).
    """
    assert bool(bank.occupied[i] and bank.occupied[j])
    keep, drop = (i, j) if bank.importance[i] >= bank.importance[j] else (j, i)

    d_k, d_d = bank.down[keep].clone(), bank.down[drop].clone()
    u_k, u_d = bank.up[keep].clone(), bank.up[drop].clone()

    shared_down = 0.5 * (d_k + d_d)
    shared_up = 0.5 * (u_k + u_d)

    # Residual relative to shared (what drop uniquely contributed).
    res_down = d_d - shared_down
    res_up = u_d - shared_up

    bank.down[keep].copy_(shared_down)
    bank.up[keep].copy_(shared_up)
    bank.down[drop].copy_(res_down)
    bank.up[drop].copy_(res_up)

    # Both stay occupied; residual stays on its loop (or keep's loop if drop was -1).
    if int(bank.assignment[drop].item()) < 0:
        bank.assignment[drop] = bank.assignment[keep]
    bank.occupied[drop] = True

    # Importance: shared keeps most, residual gets remainder fraction.
    total_imp = bank.importance[keep] + bank.importance[drop]
    bank.importance[keep] = total_imp * 0.75
    bank.importance[drop] = total_imp * 0.25

    energy = float(res_up.pow(2).mean().item() + res_down.pow(2).mean().item())
    return energy


@torch.no_grad()
def free_weak_slots(
    bank: ResidualSlotBank,
    threshold: float,
    max_free: Optional[int] = None,
    min_age: int = 0,
) -> int:
    """Free occupied slots whose residual energy is tiny vs peers.

    Slots younger than ``min_age`` consolidation cycles are exempt: a regrown slot
    has ``up == 0`` and zero importance, so it looks maximally weak precisely when
    it has had no opportunity to learn. Without the exemption, regrow and free form
    a closed loop that reinitialises the same slots forever.

    ``max_free`` caps frees per cycle so the bank cannot lose many slots at once;
    weakest slots go first.
    """
    freed = 0
    energies = bank.up.detach().float().pow(2).mean(dim=(1, 2))
    energies = energies + bank.down.detach().float().pow(2).mean(dim=(1, 2))
    active = bank.occupied.nonzero(as_tuple=False).view(-1)
    if active.numel() <= 1:
        return 0
    max_e = float(energies[active].max().item())
    if max_e < 1e-10:
        return 0
    max_imp = float(bank.importance[active].max().item())

    eligible: list[tuple[float, int]] = []
    for s in active.tolist():
        if int(bank.age[s].item()) < min_age:
            continue
        rel = float(energies[s] / max_e)
        imp_rel = float(bank.importance[s] / max_imp) if max_imp > 1e-12 else 0.0
        if rel < threshold and imp_rel < threshold:
            eligible.append((rel, s))

    eligible.sort()  # weakest first
    for _rel, s in eligible:
        if max_free is not None and freed >= max_free:
            break
        if bank.n_active() <= 1:
            break
        bank.free_slot(s)
        freed += 1
    return freed


@torch.no_grad()
def regrow_free_slots(
    bank: ResidualSlotBank,
    max_regrow: Optional[int] = None,
    keep_free: int = 0,
) -> int:
    """Allocate free slots to high-pressure loops; optionally leave some free."""
    free = (~bank.occupied).nonzero(as_tuple=False).view(-1).tolist()
    if keep_free > 0:
        free = free[:-keep_free] if len(free) > keep_free else []
    if not free:
        return 0
    if max_regrow is not None:
        free = free[:max_regrow]
    regrown = 0
    pressure = bank.loop_pressure.detach().clone()
    if float(pressure.sum()) <= 0:
        pressure = torch.ones_like(pressure)
    for slot_i in free:
        loop_i = int(torch.argmax(pressure).item())
        bank.allocate_slot(slot_i, loop_i)
        pressure[loop_i] *= 0.5
        regrown += 1
    return regrown


def consolidate(
    model: SCRTGPT,
    probe_batches: list[tuple[torch.Tensor, torch.Tensor]],
    cfg: TrainConfig,
    step: int,
) -> ConsolidateStats:
    """One consolidation cycle. Param count must be invariant."""
    bank = model.bank
    n_before = bank.n_active()
    n_params_before = sum(p.numel() for p in model.parameters())

    bank.tick_age()
    bank.update_importance_from_grads()
    sigs = functional_signatures(model, bank, probe_batches)

    occupied = bank.occupied.nonzero(as_tuple=False).view(-1).tolist()
    pair_sims: list[float] = []
    residual_energies: list[float] = []
    merged = 0
    used: set[int] = set()
    candidates: list[tuple[float, int, int]] = []
    for a_i, a in enumerate(occupied):
        for b in occupied[a_i + 1 :]:
            sim = float(F.cosine_similarity(sigs[a], sigs[b], dim=0, eps=1e-8).item())
            pair_sims.append(sim)
            if sim >= cfg.merge_sim_threshold:
                candidates.append((sim, a, b))
    candidates.sort(reverse=True)

    max_merges = getattr(cfg, "max_merges_per_cycle", 2)
    with torch.no_grad():
        for sim, a, b in candidates:
            if merged >= max_merges:
                break
            if a in used or b in used:
                continue
            if not (bool(bank.occupied[a]) and bool(bank.occupied[b])):
                continue
            # Prefer merging slots on the same loop (more meaningful redundancy).
            if int(bank.assignment[a]) != int(bank.assignment[b]):
                # allow cross-loop only if very high sim
                if sim < min(0.97, cfg.merge_sim_threshold + 0.05):
                    continue
            energy = merge_slot_pair(bank, a, b)
            residual_energies.append(energy)
            used.add(a)
            used.add(b)
            merged += 1

    freed = 0
    if getattr(cfg, "consolidate_enable_free", True):
        freed = free_weak_slots(
            bank,
            cfg.residual_free_threshold,
            max_free=getattr(cfg, "max_frees_per_cycle", None),
            min_age=getattr(cfg, "min_slot_age_cycles", 0),
        )

    regrown = 0
    if getattr(cfg, "consolidate_enable_regrow", True):
        keep_free = getattr(cfg, "keep_free_slots", 2)
        max_regrow = getattr(cfg, "max_regrow_per_cycle", 2)
        regrown = regrow_free_slots(bank, max_regrow=max_regrow, keep_free=keep_free)

    n_params_after = sum(p.numel() for p in model.parameters())
    if n_params_after != n_params_before:
        raise RuntimeError(
            f"HARD BUDGET VIOLATED: params {n_params_before} → {n_params_after}"
        )

    return ConsolidateStats(
        step=step,
        n_active_before=n_before,
        n_active_after=bank.n_active(),
        n_merged=merged,
        n_freed=freed,
        n_regrown=regrown,
        n_params=n_params_after,
        pair_sims=pair_sims,
        residual_energies=residual_energies,
    )
