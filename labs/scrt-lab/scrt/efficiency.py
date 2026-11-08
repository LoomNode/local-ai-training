"""Efficiency metrics: quality vs cost, no forced parameter equality.

Matched-param races answer "same optimizer budget, different topology."
This module answers a different question:

  given each arm at its *native* size, where does it sit on the
  (eval loss, unique params, wall time) tradeoff?

Primary objects:
  - quality-per-size scores (lower loss at smaller n is better)
  - Pareto frontier over (n_params, best_eval) — minimize both
  - optional wall-time axis for compute-ish comparison
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence


def tokens_per_sec(tokens_seen: int, wall_s: float) -> float:
    if wall_s <= 0:
        return 0.0
    return float(tokens_seen) / float(wall_s)


def best_eval_per_mparam(best_eval_loss: float, n_params: int) -> float:
    """Eval loss scaled by millions of unique params.

    Lower is better on both axes separately; the product is a crude single
    number that rewards small models that still get low loss. Not a substitute
    for the Pareto frontier when arms cross.
    """
    if n_params <= 0:
        return float("inf")
    return float(best_eval_loss) * (float(n_params) / 1_000_000.0)


def loss_params_product(best_eval_loss: float, n_params: int) -> float:
    """Same idea without the 1e6 scale — useful for ranking tiny models."""
    return float(best_eval_loss) * float(n_params)


def dominates(
    a_params: int,
    a_loss: float,
    b_params: int,
    b_loss: float,
    *,
    param_tol: int = 0,
    loss_tol: float = 0.0,
) -> bool:
    """True if A is no worse on both axes and strictly better on at least one.

    Minimization on both n_params and best_eval_loss.
    """
    a_le_params = a_params <= b_params + param_tol
    a_le_loss = a_loss <= b_loss + loss_tol
    a_lt_params = a_params < b_params - param_tol
    a_lt_loss = a_loss < b_loss - loss_tol
    return a_le_params and a_le_loss and (a_lt_params or a_lt_loss)


def pareto_frontier(
    points: Sequence[Mapping[str, Any]],
    *,
    param_key: str = "n_params",
    loss_key: str = "best_eval_loss",
    param_tol: int = 0,
    loss_tol: float = 0.0,
) -> list[str]:
    """Return arm names on the non-dominated frontier (stable input order)."""
    items = [
        (str(p["arm"]), int(p[param_key]), float(p[loss_key])) for p in points
    ]
    frontier: list[str] = []
    for i, (arm_i, pi, li) in enumerate(items):
        dominated = False
        for j, (_, pj, lj) in enumerate(items):
            if i == j:
                continue
            if dominates(pj, lj, pi, li, param_tol=param_tol, loss_tol=loss_tol):
                dominated = True
                break
        if not dominated:
            frontier.append(arm_i)
    return frontier


def arm_efficiency_row(
    arm: str,
    *,
    n_params: int,
    best_eval_loss: float,
    final_eval_loss: float,
    wall_s: float,
    tokens_seen: int,
    best_eval_step: int = 0,
    pad_params: int = 0,
    match_params: bool = False,
    n_loops: Optional[int] = None,
    n_layer: Optional[int] = None,
) -> dict[str, Any]:
    """One arm's efficiency record for tables / JSON."""
    tps = tokens_per_sec(tokens_seen, wall_s)
    return {
        "arm": arm,
        "n_params": int(n_params),
        "n_unique_params": int(n_params) - int(pad_params),
        "pad_params": int(pad_params),
        "match_params": bool(match_params),
        "best_eval_loss": float(best_eval_loss),
        "final_eval_loss": float(final_eval_loss),
        "best_eval_step": int(best_eval_step),
        "wall_s": float(wall_s),
        "tokens_seen": int(tokens_seen),
        "tokens_per_sec": tps,
        "best_eval_per_mparam": best_eval_per_mparam(best_eval_loss, n_params),
        "loss_params_product": loss_params_product(best_eval_loss, n_params),
        # Crude compute proxy: lower loss * less wall is better. Not a true
        # equal-wall iso-curve, but surfaces the loop FLOP tax when loss is close.
        "loss_wall_product": float(best_eval_loss) * float(wall_s),
        "n_loops": n_loops,
        "n_layer": n_layer,
    }


def efficiency_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    loss_tol: float = 0.0,
) -> dict[str, Any]:
    """Aggregate efficiency table + Pareto + simple winners.

    Expects rows shaped like :func:`arm_efficiency_row` (at least arm,
    n_params, best_eval_loss, wall_s, best_eval_per_mparam).
    """
    rows = list(rows)
    if not rows:
        return {
            "arms": {},
            "pareto_frontier": [],
            "params_equal": True,
            "_note": "empty",
        }

    by_arm = {str(r["arm"]): dict(r) for r in rows}
    frontier = pareto_frontier(rows, loss_tol=loss_tol)
    params = {a: int(r["n_params"]) for a, r in by_arm.items()}
    params_equal = len(set(params.values())) == 1

    best_eval_arm = min(rows, key=lambda r: float(r["best_eval_loss"]))
    smallest = min(rows, key=lambda r: int(r["n_params"]))
    best_per_m = min(rows, key=lambda r: float(r["best_eval_per_mparam"]))
    best_wall = min(rows, key=lambda r: float(r["wall_s"]))

    # Relative to largest model: how much smaller is each arm?
    max_p = max(params.values())
    min_p = min(params.values())
    for arm, r in by_arm.items():
        p = int(r["n_params"])
        r["param_frac_of_largest"] = p / max_p if max_p else 1.0
        r["param_saving_vs_largest"] = (max_p - p) / max_p if max_p else 0.0
        r["on_pareto_frontier"] = arm in frontier

    return {
        "arms": by_arm,
        "pareto_frontier": frontier,
        "params_equal": params_equal,
        "n_params": params,
        "param_range": {"min": min_p, "max": max_p},
        "winners": {
            "best_eval": {
                "arm": best_eval_arm["arm"],
                "best_eval_loss": float(best_eval_arm["best_eval_loss"]),
                "n_params": int(best_eval_arm["n_params"]),
            },
            "smallest": {
                "arm": smallest["arm"],
                "n_params": int(smallest["n_params"]),
                "best_eval_loss": float(smallest["best_eval_loss"]),
            },
            "best_eval_per_mparam": {
                "arm": best_per_m["arm"],
                "best_eval_per_mparam": float(best_per_m["best_eval_per_mparam"]),
                "best_eval_loss": float(best_per_m["best_eval_loss"]),
                "n_params": int(best_per_m["n_params"]),
            },
            "fastest_wall": {
                "arm": best_wall["arm"],
                "wall_s": float(best_wall["wall_s"]),
                "best_eval_loss": float(best_wall["best_eval_loss"]),
            },
        },
        "_protocol": (
            "Native parameter counts (no ParamPad). Primary ranking is the "
            "Pareto frontier over (n_params, best_eval_loss); single-number "
            "best_eval_per_mparam is secondary."
        ),
    }


def format_efficiency_table(report: Mapping[str, Any]) -> str:
    """Human-readable multi-metric table."""
    arms = report.get("arms") or {}
    if not arms:
        return "(no arms)"
    lines = [
        f"{'arm':18} {'params':>10} {'best':>8} {'final':>8} "
        f"{'eval/M':>10} {'wall':>8} {'tok/s':>8} {'pareto':>6}",
    ]
    frontier = set(report.get("pareto_frontier") or [])
    # Sort by params ascending for a size-ordered view.
    ordered = sorted(arms.items(), key=lambda kv: int(kv[1]["n_params"]))
    for arm, r in ordered:
        mark = "yes" if arm in frontier else ""
        lines.append(
            f"{arm:18} {int(r['n_params']):10d} "
            f"{float(r['best_eval_loss']):8.4f} "
            f"{float(r.get('final_eval_loss', float('nan'))):8.4f} "
            f"{float(r['best_eval_per_mparam']):10.4f} "
            f"{float(r['wall_s']):8.1f} "
            f"{float(r.get('tokens_per_sec', 0.0)):8.0f} "
            f"{mark:>6}"
        )
    lines.append(f"pareto frontier: {', '.join(report.get('pareto_frontier') or [])}")
    w = report.get("winners") or {}
    if w:
        lines.append(
            f"winners: best_eval={w.get('best_eval', {}).get('arm')}  "
            f"eval/Mparam={w.get('best_eval_per_mparam', {}).get('arm')}  "
            f"smallest={w.get('smallest', {}).get('arm')}  "
            f"fastest={w.get('fastest_wall', {}).get('arm')}"
        )
    return "\n".join(lines)


def rows_from_run_results(
    results: Mapping[str, Any],
    *,
    match_params: bool = False,
    pad_by_arm: Optional[Mapping[str, int]] = None,
) -> list[dict[str, Any]]:
    """Build efficiency rows from train.RunResult-like objects or dicts."""
    pad_by_arm = pad_by_arm or {}
    rows: list[dict[str, Any]] = []
    for arm, r in results.items():
        if hasattr(r, "n_params"):
            n_params = int(r.n_params)
            best = float(r.best_eval_loss)
            final = float(r.final_eval_loss)
            wall = float(r.wall_s)
            tokens = int(r.tokens_seen)
            best_step = int(r.best_eval_step)
            mcfg = getattr(r, "model_config", None) or {}
        else:
            n_params = int(r["n_params"])
            best = float(r["best_eval_loss"])
            final = float(r["final_eval_loss"])
            wall = float(r["wall_s"])
            tokens = int(r["tokens_seen"])
            best_step = int(r.get("best_eval_step", 0))
            mcfg = r.get("model_config") or {}
        rows.append(
            arm_efficiency_row(
                arm,
                n_params=n_params,
                best_eval_loss=best,
                final_eval_loss=final,
                wall_s=wall,
                tokens_seen=tokens,
                best_eval_step=best_step,
                pad_params=int(pad_by_arm.get(arm, 0)),
                match_params=match_params,
                n_loops=mcfg.get("n_loops") if isinstance(mcfg, dict) else None,
                n_layer=mcfg.get("n_layer") if isinstance(mcfg, dict) else None,
            )
        )
    return rows
