"""Eval arms → JSON. Same holdouts for every arm."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from seqmoe.assembly import assemble, load_router, load_trunk
from seqmoe.config import ExperimentConfig
from seqmoe.data import PocData, TokenBatcher
from seqmoe.model import SeqMoE

LN2 = math.log(2.0)


def nll_to_bpb(nll: float) -> float:
    return nll / LN2


@torch.no_grad()
def eval_nll(
    model: SeqMoE,
    data: torch.Tensor,
    cfg: ExperimentConfig,
    *,
    batches: int | None = None,
    **forward_kwargs: Any,
) -> dict[str, float]:
    tcfg = cfg.train
    device = str(next(model.parameters()).device)
    seq = tcfg.seq_len
    if data.numel() <= seq + 2:
        # Score the whole short stream as one sequence (cross templates).
        seq = min(cfg.model.block_size, int(data.numel()) - 1)
        if seq < 2:
            return {"nll": float("nan"), "bpb": float("nan")}
        x = data[:seq].unsqueeze(0).to(device)
        y = data[1 : seq + 1].unsqueeze(0).to(device)
        model.eval()
        fr = model(x, y, **forward_kwargs)
        assert fr.loss is not None
        nll = float(fr.loss.item())
        return {"nll": nll, "bpb": nll_to_bpb(nll)}
    batcher = TokenBatcher(
        data, seq, tcfg.batch_size, device=device, seed=tcfg.seed + 123
    )
    n_batches = batches if batches is not None else tcfg.eval_batches
    model.eval()
    total = 0.0
    route_acc: list[torch.Tensor] = []
    agree_acc: list[float] = []
    for _ in range(n_batches):
        x, y = batcher.sample()
        fr = model(x, y, **forward_kwargs)
        assert fr.loss is not None
        total += float(fr.loss.item())
        if fr.route_frac is not None:
            route_acc.append(fr.route_frac.detach().cpu())
        if fr.gold_agree is not None:
            agree_acc.append(float(fr.gold_agree.item()))
    nll = total / n_batches
    out: dict[str, float] = {"nll": nll, "bpb": nll_to_bpb(nll)}
    if route_acc:
        mean_frac = torch.stack(route_acc).mean(dim=0)
        p = mean_frac.clamp_min(1e-12)
        p = p / p.sum()
        entropy = float((-(p * p.log()).sum()).item())
        out["entropy"] = entropy
        for i, v in enumerate(mean_frac.tolist()):
            out[f"route_frac_{i}"] = float(v)
    if agree_acc:
        out["gold_agree"] = sum(agree_acc) / len(agree_acc)
    return out


def _metrics(model: SeqMoE, data: PocData, cfg: ExperimentConfig, **kw: Any) -> dict[str, Any]:
    out = {}
    for name, tensor in data.splits.items():
        out[name] = eval_nll(model, tensor, cfg, **kw)
    return out


def evaluate_poc(
    cfg: ExperimentConfig,
    data: PocData,
    *,
    trunk_path: Path | str,
    expert_paths: list[Path | str],
    router_path: Path | str | None,
    dense_ft_paths: dict[str, Path | str] | None = None,
) -> dict[str, Any]:
    """Run comparison arms. Tiny PoC requires trunk, force-expert, random, learned Top-K."""
    tcfg = cfg.train
    device = tcfg.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    arms: dict[str, Any] = {}

    trunk = load_trunk(trunk_path, map_location="cpu").to(device)
    trunk.eval()
    arms["trunk"] = _metrics(trunk, data, cfg)
    del trunk

    if dense_ft_paths:
        for name, path in dense_ft_paths.items():
            ft = load_trunk(path, map_location="cpu").to(device)
            ft.eval()
            arms[name] = _metrics(ft, data, cfg)
            del ft

    assembled = assemble(trunk_path, expert_paths).to(device)
    assembled.eval()
    if router_path is not None and Path(router_path).is_file():
        load_router(assembled, router_path)

    arms["expert_only_code"] = _metrics(assembled, data, cfg, force_expert=0)
    arms["expert_only_math"] = _metrics(assembled, data, cfg, force_expert=1)
    arms["random_route"] = _metrics(assembled, data, cfg, route="random", top_k=1)

    # hard_route only on code/math (gold domain).
    hard: dict[str, Any] = {}
    if "code" in data.splits:
        hard["code"] = eval_nll(assembled, data.splits["code"], cfg, route="hard", hard_id=0)
        # gold_agree is 1 by construction for hard route
    if "math" in data.splits:
        hard["math"] = eval_nll(assembled, data.splits["math"], cfg, route="hard", hard_id=1)
    arms["hard_route"] = hard

    assembled.set_mode("eval")
    gold_code = torch.tensor(0, dtype=torch.long, device=device)
    gold_math = torch.tensor(1, dtype=torch.long, device=device)
    top1 = _metrics(assembled, data, cfg, route="learned", top_k=1)
    top2 = _metrics(assembled, data, cfg, route="learned", top_k=2)
    # Gold agree on domain holdouts (sequences that carry a domain id).
    if "code" in data.splits:
        extra = eval_nll(
            assembled,
            data.splits["code"],
            cfg,
            route="learned",
            top_k=1,
            gold_id=gold_code,
        )
        top1.setdefault("code", {}).update(
            {k: extra[k] for k in extra if k in ("gold_agree", "entropy") or k.startswith("route_frac")}
        )
    if "math" in data.splits:
        extra = eval_nll(
            assembled,
            data.splits["math"],
            cfg,
            route="learned",
            top_k=1,
            gold_id=gold_math,
        )
        top1.setdefault("math", {}).update(
            {k: extra[k] for k in extra if k in ("gold_agree", "entropy") or k.startswith("route_frac")}
        )
    arms["seqmoe_top1"] = top1
    arms["seqmoe_top2"] = top2

    payload = {
        "run_name": tcfg.run_name,
        "scientific_grade": data.scientific_grade,
        "code_source": data.code_source,
        "math_source": data.math_source,
        "wiki_source": data.wiki_source,
        "device": device,
        "model": {
            "n_embd": cfg.model.n_embd,
            "n_layer": cfg.model.n_layer,
            "n_experts": cfg.model.n_experts,
            "top_k": cfg.model.top_k,
        },
        "steps": {
            "pretrain": tcfg.pretrain_steps,
            "expert": tcfg.expert_steps,
            "router": tcfg.router_steps,
            "seqmoe_wall_steps": tcfg.pretrain_steps
            + 2 * tcfg.expert_steps
            + tcfg.router_steps,
            "dense_ft_mix_wall_steps": tcfg.pretrain_steps + tcfg.expert_steps,
        },
        "arms": arms,
        "splits": sorted(data.splits),
    }
    return payload


def write_results(
    payload: dict[str, Any],
    results_dir: Path | str,
    run_name: str,
) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison = results_dir / f"{run_name}_comparison.json"
    comparison.write_text(json.dumps(payload, indent=2))
    for arm, metrics in payload.get("arms", {}).items():
        arm_path = results_dir / f"{run_name}_{arm}.json"
        arm_path.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "arm": arm,
                    "scientific_grade": payload.get("scientific_grade"),
                    "metrics": metrics,
                    "device": payload.get("device"),
                },
                indent=2,
            )
        )
    return comparison


def _finite_nll_bpb(cell: dict[str, Any] | None) -> bool:
    if not isinstance(cell, dict):
        return False
    nll, bpb = cell.get("nll"), cell.get("bpb")
    if not isinstance(nll, (int, float)) or not isinstance(bpb, (int, float)):
        return False
    return nll == nll and bpb == bpb


def has_required_tiny_arms(payload: dict[str, Any]) -> bool:
    arms = payload.get("arms") or {}
    needed = (
        "trunk",
        "expert_only_code",
        "expert_only_math",
        "random_route",
    )
    if any(k not in arms for k in needed):
        return False
    if "seqmoe_top1" not in arms and "seqmoe_top2" not in arms:
        return False
    splits = ("code", "math", "mixed")
    learned = arms.get("seqmoe_top2") or arms.get("seqmoe_top1") or {}
    for arm_name in ("trunk", "expert_only_code", "random_route"):
        block = arms[arm_name]
        for sp in splits:
            cell = block.get(sp) or {}
            if not isinstance(cell.get("nll"), (int, float)) or not isinstance(
                cell.get("bpb"), (int, float)
            ):
                return False
            if cell["nll"] != cell["nll"]:  # NaN
                return False
    for sp in splits:
        cell = learned.get(sp) or {}
        if not isinstance(cell.get("nll"), (int, float)):
            return False
    return True


def has_required_eval_arms(payload: dict[str, Any]) -> bool:
    if not has_required_tiny_arms(payload):
        return False
    arms = payload.get("arms") or {}
    splits = ("code", "math", "mixed")
    for name in ("dense_ft_mix", "dense_ft_code", "dense_ft_math"):
        if name not in arms:
            return False
        for sp in splits:
            if not _finite_nll_bpb(arms[name].get(sp)):
                return False
    return True
