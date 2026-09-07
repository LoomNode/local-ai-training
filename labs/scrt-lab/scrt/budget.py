"""Parameter-budget matching across arms.

Research question needs a fair comparison:
  same optimizer, data, steps, and approximately the same physical parameter count.

SCRT is the reference architecture. Baseline / looped are adjusted (layer count
and/or a tiny learned pad) so ``count_params`` lands within a tight tolerance
of the SCRT budget — never by growing SCRT after the fact.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import torch
import torch.nn as nn

from scrt.config import ModelConfig
from scrt.model import (
    BaselineGPT,
    LoopedGPT,
    ModulatedResidualGPT,
    SCRTGPT,
    SharedPlusTinyGPT,
    SharedResidualGPT,
    build_model,
    count_params,
)


class ParamPad(nn.Module):
    """Learned pad parameters that do not change the forward map (×0).

    Used only to equalize optimizer/param budgets when discrete arch knobs
    cannot hit the target exactly. The pad is optimized but cannot help
    prediction except via optimizer noise — so it is slightly *pessimistic*
    for the padded arm, which is the conservative direction for claiming SCRT wins.
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        if n < 0:
            raise ValueError("pad size must be >= 0")
        self.pad = nn.Parameter(torch.zeros(n)) if n > 0 else None

    def forward_bias(self) -> torch.Tensor:
        if self.pad is None:
            return torch.tensor(0.0)
        return self.pad.sum() * 0.0


class PaddedModel(nn.Module):
    """Wrap any arm model with an optional param pad for exact budget match."""

    def __init__(self, core: nn.Module, pad_n: int = 0) -> None:
        super().__init__()
        self.core = core
        self.param_pad = ParamPad(pad_n)

    def forward(self, idx, targets=None, **kwargs):
        out = self.core(idx, targets, **kwargs) if kwargs else self.core(idx, targets)
        # Touch pad so it participates in the graph / .grad when needed.
        bias = self.param_pad.forward_bias()
        if isinstance(out, tuple):
            # logits, loss, ...
            logits = out[0]
            if bias.device != logits.device:
                bias = bias.to(logits.device)
            logits = logits + bias
            return (logits,) + out[1:]
        return out

    def residual_bank(self):
        if hasattr(self.core, "residual_bank"):
            return self.core.residual_bank()
        return None

    @property
    def bank(self):
        return getattr(self.core, "bank", None)

    @property
    def cfg(self):
        return self.core.cfg


def is_scrt(model: nn.Module) -> bool:
    """True if model has an SCRT residual bank (scrt or scrt_static)."""
    core = model.core if isinstance(model, PaddedModel) else model
    return isinstance(core, SCRTGPT)


def uses_residual_bank(arm: str) -> bool:
    return arm in ("scrt", "scrt_static")


def consolidates(arm: str) -> bool:
    """Only full scrt runs the consolidate cycle; scrt_static is bank-only."""
    return arm == "scrt"


def unwrap(model: nn.Module) -> nn.Module:
    return model.core if isinstance(model, PaddedModel) else model


def scrt_budget(cfg: ModelConfig) -> int:
    return count_params(SCRTGPT(cfg))


def closest_baseline_layers(cfg: ModelConfig, target: int, max_layers: int = 24) -> int:
    """Pick n_layer minimizing |params - target| (prefer not exceeding by much)."""
    best_L, best_err = 1, float("inf")
    for L in range(1, max_layers + 1):
        mcfg = replace(cfg, n_layer=L)
        n = count_params(BaselineGPT(mcfg))
        err = abs(n - target)
        # Prefer under/equal target when errors tie.
        if err < best_err or (err == best_err and n <= target):
            best_err, best_L = err, L
    return best_L


def pad_needed(model: nn.Module, target: int) -> int:
    n = count_params(model)
    if n > target:
        # Cannot pad negative; caller should choose smaller arch.
        return 0
    return target - n


def build_matched_models(
    cfg: ModelConfig,
    *,
    exact: bool = True,
) -> dict[str, tuple[nn.Module, ModelConfig, int]]:
    """Build baseline / looped / scrt near a shared budget.

    Returns arm -> (model, model_config_used, n_params).
    """
    target = scrt_budget(cfg)
    out: dict[str, tuple[nn.Module, ModelConfig, int]] = {}

    # SCRT reference (no pad). scrt_static shares the same architecture.
    scrt = SCRTGPT(cfg)
    assert count_params(scrt) == target
    out["scrt"] = (scrt, cfg, target)
    out["scrt_static"] = (SCRTGPT(cfg), cfg, target)

    # Looped: same cfg without bank — fewer params → pad up to target.
    looped = LoopedGPT(cfg)
    lp = pad_needed(looped, target) if exact else 0
    if exact and count_params(looped) > target:
        raise RuntimeError(
            f"looped params {count_params(looped)} exceed SCRT target {target}"
        )
    looped_m = PaddedModel(looped, lp) if lp else looped
    out["looped"] = (looped_m, cfg, count_params(looped_m))

    # Baseline: choose n_layer closest at or under target, then pad.
    L = closest_baseline_layers(cfg, target)
    bcfg = replace(cfg, n_layer=L)
    base = BaselineGPT(bcfg)
    # If still over target, step down.
    while count_params(base) > target and L > 1:
        L -= 1
        bcfg = replace(cfg, n_layer=L)
        base = BaselineGPT(bcfg)
    bp = pad_needed(base, target) if exact else 0
    base_m = PaddedModel(base, bp) if bp else base
    out["baseline"] = (base_m, bcfg, count_params(base_m))

    if exact:
        for arm, (m, _, n) in out.items():
            if n != target:
                raise RuntimeError(f"{arm} params {n} != target {target}")

    return out


BORN_ARMS = (
    "baseline",
    "looped",
    "shared_residual",
    "modulated",
    "shared_plus_tiny",
)


def born_budget(cfg: ModelConfig) -> int:
    """Param target for born-consolidated suite: richest of the sharing arms."""
    return max(
        count_params(LoopedGPT(cfg)),
        count_params(SharedResidualGPT(cfg)),
        count_params(ModulatedResidualGPT(cfg)),
        count_params(SharedPlusTinyGPT(cfg)),
    )


def build_born_matched_models(
    cfg: ModelConfig,
    *,
    exact: bool = True,
) -> dict[str, tuple[nn.Module, ModelConfig, int]]:
    """Match born-consolidated arms to one budget.

    Target = max over sharing arms; pad smaller ones; baseline uses layer search + pad.
    """
    target = born_budget(cfg)
    out: dict[str, tuple[nn.Module, ModelConfig, int]] = {}

    for arm, ctor in (
        ("looped", LoopedGPT),
        ("shared_residual", SharedResidualGPT),
        ("modulated", ModulatedResidualGPT),
        ("shared_plus_tiny", SharedPlusTinyGPT),
    ):
        m = ctor(cfg)
        n = count_params(m)
        if exact and n > target:
            raise RuntimeError(f"{arm} params {n} exceed born target {target}")
        pad = pad_needed(m, target) if exact else 0
        wrapped = PaddedModel(m, pad) if pad else m
        out[arm] = (wrapped, cfg, count_params(wrapped))

    L = closest_baseline_layers(cfg, target)
    bcfg = replace(cfg, n_layer=L)
    base = BaselineGPT(bcfg)
    while count_params(base) > target and L > 1:
        L -= 1
        bcfg = replace(cfg, n_layer=L)
        base = BaselineGPT(bcfg)
    bp = pad_needed(base, target) if exact else 0
    base_m = PaddedModel(base, bp) if bp else base
    out["baseline"] = (base_m, bcfg, count_params(base_m))

    if exact:
        for arm, (m, mcfg, _) in list(out.items()):
            n = count_params(m)
            out[arm] = (m, mcfg, n)
            if n != target:
                raise RuntimeError(f"{arm} params {n} != born target {target}")

    return out


def budget_report(cfg: ModelConfig) -> dict:
    """Human-readable budget breakdown (no pad)."""
    rows = {}
    for arm in ("baseline", "looped", "scrt"):
        # For baseline use matched layer count preview.
        if arm == "baseline":
            target = scrt_budget(cfg)
            L = closest_baseline_layers(cfg, target)
            mcfg = replace(cfg, n_layer=L)
            while count_params(BaselineGPT(mcfg)) > target and L > 1:
                L -= 1
                mcfg = replace(cfg, n_layer=L)
            m = BaselineGPT(mcfg)
            rows[arm] = {
                "n_layer": L,
                "raw_params": count_params(m),
                "pad_to_scrt": max(0, target - count_params(m)),
            }
        else:
            m = build_model(arm, cfg)
            rows[arm] = {
                "raw_params": count_params(m),
                "pad_to_scrt": max(0, scrt_budget(cfg) - count_params(m))
                if arm != "scrt"
                else 0,
            }
    rows["target"] = scrt_budget(cfg)
    return rows


def build_native_models(
    cfg: ModelConfig,
    arms: Optional[tuple[str, ...]] = None,
) -> dict[str, tuple[nn.Module, ModelConfig, int]]:
    """Build arms at native size — no ParamPad, no layer rematch.

    Baseline keeps ``cfg.n_layer`` unique blocks. Sharing arms keep their
    natural unique weights. Param count becomes an *outcome* for efficiency
    comparisons (Pareto over loss vs size).
    """
    arms = arms or BORN_ARMS
    out: dict[str, tuple[nn.Module, ModelConfig, int]] = {}
    for arm in arms:
        m = build_model(arm, cfg)
        out[arm] = (m, cfg, count_params(m))
    return out


def native_param_report(
    cfg: ModelConfig,
    arms: Optional[tuple[str, ...]] = None,
) -> dict[str, Any]:
    """Per-arm native param counts and size ratios (no pad)."""
    arms = arms or BORN_ARMS
    counts: dict[str, int] = {}
    for arm in arms:
        counts[arm] = count_params(build_model(arm, cfg))
    max_p = max(counts.values()) if counts else 0
    min_p = min(counts.values()) if counts else 0
    rows = {
        arm: {
            "n_params": n,
            "frac_of_largest": (n / max_p) if max_p else 1.0,
            "saving_vs_largest": ((max_p - n) / max_p) if max_p else 0.0,
        }
        for arm, n in counts.items()
    }
    return {
        "arms": rows,
        "min_params": min_p,
        "max_params": max_p,
        "params_equal": min_p == max_p,
        "n_layer": cfg.n_layer,
        "n_loops": cfg.n_loops,
    }
