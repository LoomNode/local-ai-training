"""Training loop for baseline / looped / SCRT arms."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

try:
    import fleetjob
except ImportError:            # not running under the fleet
    class fleetjob:            # noqa: N801
        steps = staticmethod(lambda n: n)
        metrics = staticmethod(lambda *a, **k: None)
        stop_requested = staticmethod(lambda *a, **k: False)

from scrt.budget import (
    BORN_ARMS,
    born_budget,
    build_born_matched_models,
    build_matched_models,
    build_native_models,
    budget_report,
    consolidates,
    native_param_report,
    unwrap,
)
from scrt.config import (
    ExperimentConfig,
    ablation_config,
    born_config,
    capacity_config,
    efficiency_config,
    resolve_device,
    saturation_config,
    saturation_merge_only_config,
    stepped_config,
    tiny_mvp_config,
    v2_config,
)
from scrt.efficiency import (
    efficiency_report,
    format_efficiency_table,
    rows_from_run_results,
    tokens_per_sec,
)
from scrt.consolidate import consolidate
from scrt.data import TokenBatcher, build_corpus
from scrt.model import SCRTGPT, build_model, count_params, model_forward

# Default arm order for the bank-vs-consolidate-vs-loop ablation.
ABLATION_ARMS = ("baseline", "looped", "scrt_static", "scrt")


def should_run_consolidate(
    arm: str,
    step: int,
    *,
    consolidate_every: int,
    steps: int,
    consolidate_until_frac: float,
    consolidate_start_frac: float = 0.0,
) -> bool:
    """Policy gate used by the training loop (and tests).

    - ``scrt_static`` never consolidates (bank only).
    - ``scrt`` consolidates on schedule while
      ``steps * start_frac < step <= steps * until_frac``
      (start bound is exclusive so step==0 never fires; step at exact
      ``floor(steps*start)`` is excluded — first eligible is the next on-schedule step).
    """
    if not consolidates(arm):
        return False
    if consolidate_every <= 0:
        return False
    if step % consolidate_every != 0:
        return False
    if step <= steps * consolidate_start_frac:
        return False
    # Inclusive upper bound so step == floor(steps * frac) still runs when on schedule.
    if step > steps * consolidate_until_frac:
        return False
    return True


@dataclass
class RunResult:
    arm: str
    n_params: int
    final_train_loss: float
    final_eval_loss: float
    best_eval_loss: float
    best_eval_step: int
    history: list[dict[str, Any]] = field(default_factory=list)
    consolidate_log: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0
    tokens_seen: int = 0
    path: Optional[str] = None
    model_config: Optional[dict[str, Any]] = None
    corpus_name: str = ""
    late_improvement: float = 0.0  # best_early - best_late (positive = still improving)
    plateau_step: Optional[int] = None
    loss_at_plateau: Optional[float] = None
    # loss_at_plateau - min(eval after plateau); >0 means improved after plateau.
    post_plateau_improvement: float = 0.0
    # min(eval in third quarter) - min(eval in last quarter); >0 means last quarter better.
    last_quarter_improvement: float = 0.0
    # Efficiency fields (meaningful when match_params=False).
    pad_params: int = 0
    match_params: bool = False
    tokens_per_sec: float = 0.0


@torch.no_grad()
def eval_loss(
    model: torch.nn.Module, batcher: TokenBatcher, batches: int = 10
) -> float:
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = batcher.sample()
        fr = model_forward(model, x, y)
        assert fr.loss is not None
        total += float(fr.loss.item())
    model.train()
    return total / batches


def _late_improvement(history: list[dict[str, Any]], steps: int) -> float:
    """How much best eval improved in the second half vs first half."""
    evals = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    if len(evals) < 4:
        return 0.0
    mid = steps // 2
    first = [e for s, e in evals if s <= mid]
    second = [e for s, e in evals if s > mid]
    if not first or not second:
        return 0.0
    return float(min(first) - min(second))  # >0 means second half found better loss


def detect_plateau(
    history: list[dict[str, Any]],
    *,
    steps: int,
    patience_evals: int = 5,
    epsilon: float = 0.002,
    min_frac: float = 0.25,
) -> tuple[Optional[int], Optional[float]]:
    """First eval step where loss has not improved by >epsilon for ``patience`` evals.

    Returns ``(plateau_step, loss_at_that_eval)`` or ``(None, None)`` if never flat.
    Only considers steps after ``steps * min_frac``.
    """
    evals = [(h["step"], float(h["eval_loss"])) for h in history if "eval_loss" in h]
    if len(evals) < patience_evals + 1:
        return None, None
    min_step = steps * min_frac
    best = float("inf")
    stagnant = 0
    for step, loss in evals:
        if step < min_step:
            if loss < best - epsilon:
                best = loss
            continue
        if loss < best - epsilon:
            best = loss
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience_evals:
                return step, loss
    return None, None


def post_plateau_improvement(
    history: list[dict[str, Any]],
    plateau_step: Optional[int],
    loss_at_plateau: Optional[float],
) -> float:
    """Positive ⇒ best eval after plateau is lower than loss at plateau onset."""
    if plateau_step is None or loss_at_plateau is None:
        return 0.0
    after = [
        float(h["eval_loss"])
        for h in history
        if "eval_loss" in h and h["step"] > plateau_step
    ]
    if not after:
        return 0.0
    return float(loss_at_plateau - min(after))


def last_quarter_improvement(history: list[dict[str, Any]], steps: int) -> float:
    """Positive ⇒ best loss in last 25% of steps beats best in third quarter."""
    evals = [(h["step"], float(h["eval_loss"])) for h in history if "eval_loss" in h]
    if len(evals) < 4:
        return 0.0
    q3 = steps * 0.5
    q4 = steps * 0.75
    third = [e for s, e in evals if q3 < s <= q4]
    last = [e for s, e in evals if s > q4]
    if not third or not last:
        return 0.0
    return float(min(third) - min(last))


def train_arm(
    cfg: ExperimentConfig,
    arm: Optional[str] = None,
    *,
    write_results: bool = True,
    match_params: bool = False,
    match_family: str = "scrt",
    corpus_length: Optional[int] = None,
    eval_batches: int = 10,
    final_eval_batches: int = 20,
    model: Optional[torch.nn.Module] = None,
) -> RunResult:
    tcfg = cfg.train
    mcfg = cfg.model
    arm = arm or tcfg.arm
    device = resolve_device(tcfg.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    torch.manual_seed(tcfg.seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(tcfg.seed)

    length = corpus_length if corpus_length is not None else tcfg.corpus_length
    corpus = build_corpus(
        kind=tcfg.corpus,
        vocab_size=mcfg.vocab_size,
        length=length,
        seed=tcfg.seed,
        max_chars=tcfg.max_chars,
        vocab_cap=tcfg.char_vocab_cap,
    )
    # Char corpora define their own vocab — update model config.
    if corpus.vocab_size != mcfg.vocab_size:
        from dataclasses import replace

        mcfg = replace(mcfg, vocab_size=corpus.vocab_size)

    split = int(corpus.data.numel() * 0.9)
    train_data = corpus.data[:split]
    eval_data = corpus.data[split:]

    train_batcher = TokenBatcher(
        train_data, tcfg.seq_len, tcfg.batch_size, device=device, seed=tcfg.seed
    )
    eval_batcher = TokenBatcher(
        eval_data,
        tcfg.seq_len,
        tcfg.batch_size,
        device=device,
        seed=tcfg.seed + 1,
    )
    probe_batcher = TokenBatcher(
        train_data,
        tcfg.seq_len,
        min(16, tcfg.batch_size),
        device=device,
        seed=tcfg.seed + 2,
    )

    used_mcfg = mcfg
    pad_params = 0
    if model is None:
        if match_params:
            if match_family == "born":
                matched = build_born_matched_models(mcfg, exact=True)
            else:
                matched = build_matched_models(mcfg, exact=True)
            model, used_mcfg, n_params = matched[arm]
            model = model.to(device)
            core_n = count_params(unwrap(model))
            pad_params = max(0, n_params - core_n)
        else:
            model = build_model(arm, mcfg).to(device)
            n_params = count_params(model)
            pad_params = 0
    else:
        model = model.to(device)
        n_params = count_params(model)
        core_n = count_params(unwrap(model))
        pad_params = max(0, n_params - core_n)

    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(tcfg.steps, 1))

    history: list[dict[str, Any]] = []
    consolidate_log: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    last_loss = float("nan")
    best_eval = float("inf")
    best_step = 0
    tokens_per_step = tcfg.batch_size * tcfg.seq_len

    model.train()
    total = fleetjob.steps(tcfg.steps)
    series = f"{arm}_s{tcfg.seed}"
    evaluated = False
    for step in range(1, total + 1):
        x, y = train_batcher.sample()
        fr = model_forward(model, x, y)
        assert fr.loss is not None
        opt.zero_grad(set_to_none=True)
        fr.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        last_loss = float(fr.loss.item())

        core = unwrap(model)
        if isinstance(core, SCRTGPT):
            core.bank.update_importance_from_grads()

        if step % tcfg.log_every == 0 or step == 1:
            history.append(
                {
                    "step": step,
                    "train_loss": last_loss,
                    "lr": float(sched.get_last_lr()[0]),
                    "tokens": step * tokens_per_step,
                }
            )

        if step % tcfg.eval_every == 0 or step == total:
            evaluated = True
            el = eval_loss(model, eval_batcher, batches=eval_batches)
            fleetjob.metrics(step, series=series, eval_loss=el, train_loss=last_loss)
            if el < best_eval:
                best_eval = el
                best_step = step
            history.append(
                {
                    "step": step,
                    "eval_loss": el,
                    "best_eval_loss": best_eval,
                    "tokens": step * tokens_per_step,
                }
            )
            extra = ""
            if isinstance(core, SCRTGPT):
                extra = f"  slots={core.bank.n_active()}/{core.bank.n_slots}"
            print(
                f"[{arm}] step {step}/{tcfg.steps}  "
                f"train={last_loss:.4f}  eval={el:.4f}  best={best_eval:.4f}  "
                f"params={n_params}{extra}"
            )

        if should_run_consolidate(
            arm,
            step,
            consolidate_every=tcfg.consolidate_every,
            steps=tcfg.steps,
            consolidate_until_frac=tcfg.consolidate_until_frac,
            consolidate_start_frac=tcfg.consolidate_start_frac,
        ):
            core = unwrap(model)
            assert isinstance(core, SCRTGPT)
            probes = [probe_batcher.sample() for _ in range(tcfg.probe_batches)]
            stats = consolidate(core, probes, tcfg, step)
            assert count_params(model) == n_params
            consolidate_log.append(stats.as_dict())
            print(
                f"  [consolidate] merged={stats.n_merged} freed={stats.n_freed} "
                f"regrown={stats.n_regrown} active={stats.n_active_after}/{core.bank.n_slots} "
                f"params={stats.n_params}"
            )

        if evaluated and fleetjob.stop_requested(series):
            break

    wall = time.perf_counter() - t0
    final_eval = eval_loss(model, eval_batcher, batches=final_eval_batches)
    if final_eval < best_eval:
        best_eval = final_eval
        best_step = tcfg.steps

    late_imp = _late_improvement(history, tcfg.steps)
    plat_step, plat_loss = detect_plateau(
        history,
        steps=tcfg.steps,
        patience_evals=tcfg.plateau_patience_evals,
        epsilon=tcfg.plateau_epsilon,
        min_frac=tcfg.plateau_min_frac,
    )
    post_plat = post_plateau_improvement(history, plat_step, plat_loss)
    last_q = last_quarter_improvement(history, tcfg.steps)
    tokens_seen = tcfg.steps * tokens_per_step
    tps = tokens_per_sec(tokens_seen, wall)
    result = RunResult(
        arm=arm,
        n_params=n_params,
        final_train_loss=last_loss,
        final_eval_loss=final_eval,
        best_eval_loss=best_eval,
        best_eval_step=best_step,
        history=history,
        consolidate_log=consolidate_log,
        wall_s=wall,
        tokens_seen=tokens_seen,
        model_config=asdict(used_mcfg),
        corpus_name=corpus.name,
        late_improvement=late_imp,
        plateau_step=plat_step,
        loss_at_plateau=plat_loss,
        post_plateau_improvement=post_plat,
        last_quarter_improvement=last_q,
        pad_params=pad_params,
        match_params=match_params,
        tokens_per_sec=tps,
    )

    if write_results:
        out_dir = Path(tcfg.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{tcfg.run_name}_{arm}.json"
        payload = {
            "config": cfg.to_dict(),
            "arm": arm,
            "n_params": n_params,
            "n_unique_params": n_params - pad_params,
            "pad_params": pad_params,
            "match_params": match_params,
            "model_config": result.model_config,
            "corpus": corpus.name,
            "final_train_loss": last_loss,
            "final_eval_loss": final_eval,
            "best_eval_loss": best_eval,
            "best_eval_step": best_step,
            "late_improvement": late_imp,
            "plateau_step": plat_step,
            "loss_at_plateau": plat_loss,
            "post_plateau_improvement": post_plat,
            "last_quarter_improvement": last_q,
            "wall_s": wall,
            "tokens_seen": result.tokens_seen,
            "tokens_per_sec": tps,
            "history": history,
            "consolidate_log": consolidate_log,
        }
        path.write_text(json.dumps(payload, indent=2))
        result.path = str(path)
        print(f"[{arm}] wrote {path}")

    return result


def run_mvp_comparison(
    steps: int = 200,
    device: str = "cuda:1",
    run_name: str = "mvp",
) -> dict[str, RunResult]:
    cfg = tiny_mvp_config(device=resolve_device(device), steps=steps)
    cfg.train.run_name = run_name
    results: dict[str, RunResult] = {}
    for arm in ("baseline", "looped", "scrt"):
        print(f"\n=== ARM: {arm} ===")
        results[arm] = train_arm(cfg, arm=arm, write_results=True, match_params=False)
    _print_summary("MVP comparison", results, cfg.train.results_dir, run_name)
    return results


def run_stepped_comparison(
    steps: int = 3000,
    device: str = "cuda:1",
    run_name: str = "stepped",
    *,
    match_params: bool = True,
    corpus_length: int = 500_000,
) -> dict[str, RunResult]:
    cfg = stepped_config(device=resolve_device(device), steps=steps)
    cfg.train.run_name = run_name
    cfg.train.corpus_length = corpus_length
    return _run_matched(cfg, match_params=match_params, title="Stepped comparison")


def run_v2_comparison(
    steps: int = 8000,
    device: str = "cuda:1",
    run_name: str = "v2",
    *,
    match_params: bool = True,
    corpus: Optional[str] = None,
) -> dict[str, RunResult]:
    """Harder data + gentler residual consolidation (the next real test)."""
    cfg = v2_config(device=resolve_device(device), steps=steps)
    cfg.train.run_name = run_name
    if corpus is not None:
        cfg.train.corpus = corpus
        if corpus in ("shakespeare", "wikitext2"):
            cfg.train.run_name = run_name if run_name != "v2" else f"v2_{corpus}"
    return _run_matched(
        cfg,
        match_params=match_params,
        title=f"v2 comparison (corpus={cfg.train.corpus})",
        eval_batches=20,
        final_eval_batches=50,
        arms=("baseline", "looped", "scrt"),
    )


def run_ablation_comparison(
    steps: int = 8000,
    device: str = "cuda:1",
    run_name: str = "ablation",
    *,
    match_params: bool = True,
    corpus: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> dict[str, RunResult]:
    """4-arm ablation: baseline / looped / scrt_static / scrt.

    Isolates residual bank (scrt_static) from consolidate (scrt). Full scrt uses
    ``consolidate_until_frac`` from ``ablation_config`` (default 0.5).
    """
    cfg = ablation_config(device=resolve_device(device), steps=steps)
    cfg.train.run_name = run_name
    if results_dir is not None:
        cfg.train.results_dir = results_dir
    if corpus is not None:
        cfg.train.corpus = corpus
    return _run_matched(
        cfg,
        match_params=match_params,
        title=(
            f"Ablation (corpus={cfg.train.corpus}, "
            f"until_frac={cfg.train.consolidate_until_frac})"
        ),
        eval_batches=20,
        final_eval_batches=50,
        arms=ABLATION_ARMS,
    )


def run_saturation_comparison(
    steps: int = 20000,
    device: str = "cuda:1",
    run_name: str = "saturation",
    *,
    match_params: bool = True,
    corpus: Optional[str] = None,
    results_dir: Optional[str] = None,
    merge_only: bool = False,
) -> dict[str, RunResult]:
    """Saturation-focused 4-arm run: consolidate in the second half only.

    Primary research metric: ``post_plateau_improvement`` / ``last_quarter_improvement``
    — does SCRT keep improving after losses flatten?

    If ``merge_only``, use free/regrow-disabled consolidate + stronger regularisation.
    """
    if merge_only:
        cfg = saturation_merge_only_config(device=resolve_device(device), steps=steps)
    else:
        cfg = saturation_config(device=resolve_device(device), steps=steps)
    cfg.train.run_name = run_name
    if results_dir is not None:
        cfg.train.results_dir = results_dir
    if corpus is not None:
        cfg.train.corpus = corpus
    mode = "merge-only" if merge_only else "merge+free+regrow"
    return _run_matched(
        cfg,
        match_params=match_params,
        title=(
            f"Saturation/{mode} (corpus={cfg.train.corpus}, "
            f"consolidate=({cfg.train.consolidate_start_frac},"
            f"{cfg.train.consolidate_until_frac}])"
        ),
        eval_batches=20,
        final_eval_batches=50,
        arms=ABLATION_ARMS,
    )


def run_born_comparison(
    steps: int = 20000,
    device: str = "cuda:1",
    run_name: str = "born",
    *,
    match_params: bool = True,
    corpus: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> dict[str, RunResult]:
    """Born-consolidated arms: share by construction (no merge/free/regrow)."""
    cfg = born_config(device=resolve_device(device), steps=steps)
    cfg.train.run_name = run_name
    if results_dir is not None:
        cfg.train.results_dir = results_dir
    if corpus is not None:
        cfg.train.corpus = corpus
    return _run_matched(
        cfg,
        match_params=match_params,
        match_family="born",
        title=f"Born-consolidated (corpus={cfg.train.corpus})",
        eval_batches=20,
        final_eval_batches=50,
        arms=BORN_ARMS,
    )


def run_efficiency_comparison(
    steps: int = 20000,
    device: str = "cuda:1",
    run_name: str = "efficiency",
    *,
    seed: int = 0,
    corpus: Optional[str] = None,
    results_dir: Optional[str] = None,
    arms: tuple[str, ...] = BORN_ARMS,
    match_params: bool = False,
) -> dict[str, RunResult]:
    """One seed of the native-size efficiency suite (no ParamPad by default).

    Writes ``{run_name}_s{seed}_{arm}.json``. Prefer **wikitext2** (default):
    structured synthetic pins near its entropy floor so loss cannot separate
    arms and a native-size Pareto just ranks by parameter count.

    Run several seeds then :func:`aggregate_seeds` (and the efficiency JSON
    under each seed summary). ``match_params=True`` falls back to the old
    forced-budget protocol for A/B of protocols.
    """
    cfg = efficiency_config(device=resolve_device(device), steps=steps)
    cfg.train.seed = seed
    cfg.train.run_name = f"{run_name}_s{seed}"
    if results_dir is not None:
        cfg.train.results_dir = results_dir
    if corpus is not None:
        cfg.train.corpus = corpus
        if corpus == "wikitext2":
            cfg.train.max_chars = 11_000_000
            cfg.train.char_vocab_cap = 256
            cfg.model.vocab_size = 256
        elif corpus in ("structured", "synonym"):
            # Explicit synthetic override for smoke only.
            if corpus == "structured":
                cfg.train.corpus_length = 1_500_000
    title = (
        f"Efficiency native-size (corpus={cfg.train.corpus}, "
        f"match_params={match_params}, seed={seed})"
    )
    return _run_matched(
        cfg,
        match_params=match_params,
        match_family="born",
        title=title,
        eval_batches=20,
        final_eval_batches=50,
        arms=arms,
        efficiency_mode=not match_params,
    )


def run_capacity_comparison(
    steps: int = 20000,
    device: str = "cuda:1",
    run_name: str = "capacity",
    *,
    seed: int = 0,
    match_params: bool = True,
    results_dir: Optional[str] = None,
    arms: tuple[str, ...] = ABLATION_ARMS,
) -> dict[str, RunResult]:
    """One seed of the capacity-bound 4-arm run on full char-level WikiText-2.

    Writes ``{run_name}_s{seed}_{arm}.json``. Run several seeds (in parallel across
    GPUs if you have them), then aggregate with :func:`aggregate_seeds` — a single
    seed cannot separate arms that finish within a few thousandths of a nat.
    """
    cfg = capacity_config(device=resolve_device(device), steps=steps)
    cfg.train.seed = seed
    cfg.train.run_name = f"{run_name}_s{seed}"
    if results_dir is not None:
        cfg.train.results_dir = results_dir
    return _run_matched(
        cfg,
        match_params=match_params,
        title=f"Capacity (corpus={cfg.train.corpus}, seed={seed})",
        eval_batches=20,
        final_eval_batches=50,
        arms=arms,
    )


_AGG_METRICS = (
    "best_eval_loss",
    "final_eval_loss",
    "last_quarter_improvement",
    "post_plateau_improvement",
    "late_improvement",
    "wall_s",
    "tokens_per_sec",
    "n_params",
)


def aggregate_seeds(
    run_name: str,
    seeds: Sequence[int],
    *,
    arms: Sequence[str] = ABLATION_ARMS,
    results_dir: str = "results",
) -> dict[str, Any]:
    """Mean +/- std across seeds for each arm; writes ``{run_name}_agg.json``.

    Missing per-seed files are reported rather than skipped silently — a partial
    aggregate that looks complete is how a crashed arm turns into a conclusion.
    """
    root = Path(results_dir)
    agg: dict[str, Any] = {"run_name": run_name, "seeds": list(seeds), "arms": {}}
    missing: list[str] = []
    for arm in arms:
        per_seed: dict[str, list[float]] = {m: [] for m in _AGG_METRICS}
        n_params: set[int] = set()
        merged: list[int] = []
        for s in seeds:
            path = root / f"{run_name}_s{s}_{arm}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            d = json.loads(path.read_text())
            for m in _AGG_METRICS:
                if m not in d:
                    continue
                per_seed[m].append(float(d[m]))
            n_params.add(int(d["n_params"]))
            # Derived efficiency score if not stored.
            if "best_eval_per_mparam" not in d and d.get("n_params"):
                d_bem = float(d["best_eval_loss"]) * (
                    float(d["n_params"]) / 1_000_000.0
                )
            else:
                d_bem = float(d.get("best_eval_per_mparam", float("nan")))
            if "best_eval_per_mparam" not in per_seed:
                per_seed["best_eval_per_mparam"] = []
            if d.get("n_params"):
                per_seed["best_eval_per_mparam"].append(
                    float(d["best_eval_loss"]) * (float(d["n_params"]) / 1_000_000.0)
                    if "best_eval_per_mparam" not in d
                    else d_bem
                )
            # loss × wall: crude equal-wall proxy (lower better if loss similar).
            if "loss_wall_product" not in per_seed:
                per_seed["loss_wall_product"] = []
            per_seed["loss_wall_product"].append(
                float(d["best_eval_loss"]) * float(d["wall_s"])
            )
            merged.append(sum(c.get("n_merged", 0) for c in d.get("consolidate_log", [])))
        entry: dict[str, Any] = {
            "n_seeds": len(per_seed["best_eval_loss"]),
            "n_params": sorted(n_params),
            "total_merged_per_seed": merged,
        }
        for m, vals in per_seed.items():
            if not vals:
                continue
            mean = statistics.fmean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            entry[m] = {"mean": mean, "std": sd, "values": vals}
        agg["arms"][arm] = entry
    agg["missing"] = missing

    ranked = [
        (a, e["best_eval_loss"]["mean"])
        for a, e in agg["arms"].items()
        if "best_eval_loss" in e
    ]
    ranked.sort(key=lambda kv: kv[1])
    agg["_ranking_best_eval"] = ranked
    if len(ranked) > 1:
        # Is the gap between the top two arms bigger than their seed noise?
        top, second = ranked[0][0], ranked[1][0]
        pooled = max(
            agg["arms"][top]["best_eval_loss"]["std"],
            agg["arms"][second]["best_eval_loss"]["std"],
        )
        gap = ranked[1][1] - ranked[0][1]
        agg["_separation"] = {
            "top": top,
            "second": second,
            "gap": gap,
            "max_std": pooled,
            "gap_exceeds_noise": bool(gap > pooled),
        }

    # Mean-point Pareto over (n_params, best_eval) across seeds.
    pareto_points = []
    for a, e in agg["arms"].items():
        if "best_eval_loss" not in e or not e.get("n_params"):
            continue
        pareto_points.append(
            {
                "arm": a,
                "n_params": int(e["n_params"][0]),
                "best_eval_loss": float(e["best_eval_loss"]["mean"]),
            }
        )
    if pareto_points:
        from scrt.efficiency import efficiency_report

        # Reconstruct wall means for report.
        rows = []
        for p in pareto_points:
            e = agg["arms"][p["arm"]]
            rows.append(
                {
                    "arm": p["arm"],
                    "n_params": p["n_params"],
                    "best_eval_loss": p["best_eval_loss"],
                    "final_eval_loss": float(
                        e.get("final_eval_loss", {}).get("mean", p["best_eval_loss"])
                    ),
                    "wall_s": float(e.get("wall_s", {}).get("mean", 0.0)),
                    "tokens_seen": 0,
                    "tokens_per_sec": float(
                        e.get("tokens_per_sec", {}).get("mean", 0.0)
                    ),
                    "best_eval_per_mparam": float(
                        e.get("best_eval_per_mparam", {}).get(
                            "mean",
                            p["best_eval_loss"] * p["n_params"] / 1_000_000.0,
                        )
                    ),
                }
            )
        agg["_efficiency"] = efficiency_report(rows)

    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{run_name}_agg.json"
    out.write_text(json.dumps(agg, indent=2))

    print(f"\n=== {run_name}: {len(seeds)} seeds ===")
    print(
        f"{'arm':16} {'best_eval (mean+/-std)':>26} {'params':>10} "
        f"{'wall_s':>9} {'eval/M':>10}"
    )
    for arm, entry in agg["arms"].items():
        if "best_eval_loss" not in entry:
            print(f"{arm:16} {'NO DATA':>26}")
            continue
        b = entry["best_eval_loss"]
        params = entry["n_params"][0] if entry.get("n_params") else 0
        wall = entry.get("wall_s", {}).get("mean", 0.0)
        bem = entry.get("best_eval_per_mparam", {}).get("mean", float("nan"))
        print(
            f"{arm:16} {b['mean']:>14.4f} +/- {b['std']:<7.4f} "
            f"{params:10d} {wall:9.1f} {bem:10.4f}"
        )
    if missing:
        print(f"\nMISSING {len(missing)} per-seed files:")
        for m in missing:
            print(f"  {m}")
    sep = agg.get("_separation")
    if sep:
        verdict = "SEPARATED" if sep["gap_exceeds_noise"] else "WITHIN NOISE"
        print(
            f"\n{sep['top']} beats {sep['second']} by {sep['gap']:.4f} nats; "
            f"max seed std {sep['max_std']:.4f} -> {verdict}"
        )
    print(f"wrote {out}")
    return agg


def _run_matched(
    cfg: ExperimentConfig,
    *,
    match_params: bool,
    title: str,
    eval_batches: int = 15,
    final_eval_batches: int = 40,
    arms: tuple[str, ...] = ("baseline", "looped", "scrt"),
    match_family: str = "scrt",
    efficiency_mode: bool = False,
) -> dict[str, RunResult]:
    print(f"=== Budget plan (pre-pad) corpus={cfg.train.corpus} family={match_family} ===")
    # For char corpora vocab is unknown until load; build once for scrt budget print.
    if cfg.train.corpus in ("shakespeare", "wikitext2"):
        probe = build_corpus(
            kind=cfg.train.corpus,
            vocab_size=cfg.model.vocab_size,
            length=1000,
            seed=cfg.train.seed,
            max_chars=min(cfg.train.max_chars, 50_000),
            vocab_cap=cfg.train.char_vocab_cap,
        )
        from dataclasses import replace

        cfg.model = replace(cfg.model, vocab_size=probe.vocab_size)
        print(f"char vocab_size={probe.vocab_size} corpus={probe.name}")

    if match_family == "born":
        print(json.dumps({"born_target": born_budget(cfg.model)}, indent=2))
        matcher = build_born_matched_models
    else:
        print(json.dumps(budget_report(cfg.model), indent=2))
        matcher = build_matched_models

    if match_params:
        matched = matcher(cfg.model, exact=True)
        print("=== Matched budgets ===")
        for arm in arms:
            m, mcfg, n = matched[arm]
            pad = 0
            if hasattr(m, "param_pad") and m.param_pad.pad is not None:
                pad = m.param_pad.pad.numel()
            print(
                f"  {arm:16} params={n}  pad={pad}  "
                f"n_layer={getattr(mcfg, 'n_layer', None)}"
            )
    else:
        # Native sizes — param count is an outcome.
        native = native_param_report(cfg.model, arms=arms)
        print("=== Native param budgets (no pad) ===")
        print(json.dumps(native, indent=2))
        # Touch build_native_models so the path stays exercised / import-live.
        _ = build_native_models(cfg.model, arms=arms)

    results: dict[str, RunResult] = {}
    for arm in arms:
        print(f"\n=== ARM: {arm} ===")
        results[arm] = train_arm(
            cfg,
            arm=arm,
            write_results=True,
            match_params=match_params,
            match_family=match_family,
            eval_batches=eval_batches,
            final_eval_batches=final_eval_batches,
        )
    _print_summary(
        title,
        results,
        cfg.train.results_dir,
        cfg.train.run_name,
        efficiency_mode=efficiency_mode or not match_params,
        match_params=match_params,
    )
    return results


def _print_summary(
    title: str,
    results: dict[str, RunResult],
    results_dir: str,
    run_name: str,
    *,
    efficiency_mode: bool = False,
    match_params: bool = True,
) -> None:
    print(f"\n=== {title} ===")
    print(
        f"{'arm':12} {'params':>10} {'best':>10} {'best@':>7} "
        f"{'plat@':>7} {'postΔ':>8} {'lastQΔ':>8} {'lateΔ':>8} {'wall':>7}"
    )
    for arm, r in results.items():
        plat = r.plateau_step if r.plateau_step is not None else -1
        print(
            f"{arm:12} {r.n_params:10d} {r.best_eval_loss:10.4f} "
            f"{r.best_eval_step:7d} {plat:7d} "
            f"{r.post_plateau_improvement:8.4f} {r.last_quarter_improvement:8.4f} "
            f"{r.late_improvement:8.4f} {r.wall_s:7.1f}"
        )

    summary: dict[str, Any] = {
        arm: {
            "n_params": r.n_params,
            "n_unique_params": r.n_params - r.pad_params,
            "pad_params": r.pad_params,
            "match_params": r.match_params,
            "final_eval_loss": r.final_eval_loss,
            "best_eval_loss": r.best_eval_loss,
            "best_eval_step": r.best_eval_step,
            "late_improvement": r.late_improvement,
            "plateau_step": r.plateau_step,
            "loss_at_plateau": r.loss_at_plateau,
            "post_plateau_improvement": r.post_plateau_improvement,
            "last_quarter_improvement": r.last_quarter_improvement,
            "final_train_loss": r.final_train_loss,
            "wall_s": r.wall_s,
            "tokens_seen": r.tokens_seen,
            "tokens_per_sec": r.tokens_per_sec,
            "corpus": r.corpus_name,
            "n_consolidate": len(r.consolidate_log),
            "total_merged": sum(c.get("n_merged", 0) for c in r.consolidate_log),
            "total_freed": sum(c.get("n_freed", 0) for c in r.consolidate_log),
            "first_consolidate_step": (
                r.consolidate_log[0]["step"] if r.consolidate_log else None
            ),
            "last_consolidate_step": (
                r.consolidate_log[-1]["step"] if r.consolidate_log else None
            ),
        }
        for arm, r in results.items()
    }
    winner = min(results.items(), key=lambda kv: kv[1].best_eval_loss)
    summary["_winner_best_eval"] = {
        "arm": winner[0],
        "best_eval_loss": winner[1].best_eval_loss,
        "best_eval_step": winner[1].best_eval_step,
        "n_params": winner[1].n_params,
    }
    late_winner = max(results.items(), key=lambda kv: kv[1].late_improvement)
    summary["_winner_late_improvement"] = {
        "arm": late_winner[0],
        "late_improvement": late_winner[1].late_improvement,
    }
    post_winner = max(results.items(), key=lambda kv: kv[1].post_plateau_improvement)
    summary["_winner_post_plateau"] = {
        "arm": post_winner[0],
        "post_plateau_improvement": post_winner[1].post_plateau_improvement,
        "plateau_step": post_winner[1].plateau_step,
    }
    lq_winner = max(results.items(), key=lambda kv: kv[1].last_quarter_improvement)
    summary["_winner_last_quarter"] = {
        "arm": lq_winner[0],
        "last_quarter_improvement": lq_winner[1].last_quarter_improvement,
    }
    params = {arm: r.n_params for arm, r in results.items()}
    summary["_params_equal"] = len(set(params.values())) == 1
    summary["_params"] = params
    summary["_match_params"] = match_params
    summary["_efficiency_mode"] = efficiency_mode

    # Always attach efficiency view when sizes differ or efficiency_mode is on.
    pad_by_arm = {arm: r.pad_params for arm, r in results.items()}
    eff_rows = rows_from_run_results(
        results, match_params=match_params, pad_by_arm=pad_by_arm
    )
    eff = efficiency_report(eff_rows)
    summary["_efficiency"] = eff

    summary_path = Path(results_dir) / f"{run_name}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary → {summary_path}")

    if efficiency_mode or not summary["_params_equal"]:
        print(f"\n=== Efficiency view (native size / multi-metric) ===")
        print(format_efficiency_table(eff))
        eff_path = Path(results_dir) / f"{run_name}_efficiency.json"
        eff_path.write_text(json.dumps(eff, indent=2))
        print(f"efficiency → {eff_path}")

    print(
        f"winner (best eval): {winner[0]}  "
        f"loss={winner[1].best_eval_loss:.4f} @ step {winner[1].best_eval_step}  "
        f"params={winner[1].n_params}"
    )
    print(
        f"winner (late improvement): {late_winner[0]}  "
        f"Δ={late_winner[1].late_improvement:.4f}"
    )
    print(
        f"winner (post-plateau): {post_winner[0]}  "
        f"Δ={post_winner[1].post_plateau_improvement:.4f}  "
        f"plat@={post_winner[1].plateau_step}"
    )
    print(
        f"winner (last quarter): {lq_winner[0]}  "
        f"Δ={lq_winner[1].last_quarter_improvement:.4f}"
    )
    if efficiency_mode or not summary["_params_equal"]:
        w = eff.get("winners") or {}
        print(
            f"winner (eval/Mparam): {w.get('best_eval_per_mparam', {}).get('arm')}  "
            f"score={w.get('best_eval_per_mparam', {}).get('best_eval_per_mparam')}"
        )
        print(f"pareto frontier: {eff.get('pareto_frontier')}")


if __name__ == "__main__":
    run_saturation_comparison()
