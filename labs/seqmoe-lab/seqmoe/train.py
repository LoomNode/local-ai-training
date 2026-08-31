"""Pretrain, sequential expert train, router-only train, eval_loss."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from seqmoe.assembly import assemble, load_trunk, save_expert, save_router, save_trunk
from seqmoe.config import DOMAIN_TO_ID, ExperimentConfig, resolve_device
from seqmoe.data import MixedDomainBatcher, PocData, TokenBatcher
from seqmoe.memory import assert_expert_stage, optimizer_numel, peak_vram_mb, plan, reset_peak
from seqmoe.model import SeqMoE, count_params


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device: str) -> str:
    device = resolve_device(device)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


@torch.no_grad()
def eval_loss(
    model: SeqMoE,
    batcher: TokenBatcher,
    batches: int = 8,
    **forward_kwargs: Any,
) -> float:
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = batcher.sample()
        fr = model(x, y, **forward_kwargs)
        assert fr.loss is not None
        total += float(fr.loss.item())
    model.train()
    return total / batches


def _adamw(model: SeqMoE, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _step_loss(
    model: SeqMoE,
    opt: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    balance_coef: float = 0.0,
    route_coef: float = 0.0,
    gold_id: torch.Tensor | None = None,
    **forward_kwargs: Any,
) -> float:
    fr = model(x, y, gold_id=gold_id, **forward_kwargs)
    assert fr.loss is not None
    loss = fr.loss
    if fr.aux_loss is not None and balance_coef:
        loss = loss + balance_coef * fr.aux_loss
    if fr.route_ce is not None and route_coef:
        loss = loss + route_coef * fr.route_ce
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    return float(fr.loss.item())


@dataclass
class StageStats:
    stage: str
    steps: int
    wall_s: float
    tokens_seen: int
    optimizer_numel: int
    peak_vram_mb: float
    resident_expert_count: int
    final_train_loss: float
    final_eval_loss: float | None = None
    plan: dict[str, Any] = field(default_factory=dict)


def pretrain_trunk(
    model: SeqMoE,
    data: torch.Tensor,
    cfg: ExperimentConfig,
    *,
    eval_data: torch.Tensor | None = None,
) -> StageStats:
    tcfg = cfg.train
    device = str(next(model.parameters()).device)
    model.set_mode("pretrain")
    opt = _adamw(model, tcfg.lr, tcfg.weight_decay)
    mem = plan(model, "pretrain", device, tcfg, opt=opt)
    batcher = TokenBatcher(data, tcfg.seq_len, tcfg.batch_size, device=device, seed=tcfg.seed)
    eval_batcher = (
        TokenBatcher(
            eval_data, tcfg.seq_len, tcfg.batch_size, device=device, seed=tcfg.seed + 1
        )
        if eval_data is not None and eval_data.numel() > tcfg.seq_len + 2
        else None
    )
    model.train()
    reset_peak(device)
    t0 = time.perf_counter()
    last = float("nan")
    tokens = tcfg.batch_size * tcfg.seq_len
    for step in range(1, tcfg.pretrain_steps + 1):
        x, y = batcher.sample()
        last = _step_loss(model, opt, x, y)
        if step % tcfg.log_every == 0 or step == 1:
            print(f"[pretrain] step {step}/{tcfg.pretrain_steps} loss={last:.4f}")
    wall = time.perf_counter() - t0
    el = eval_loss(model, eval_batcher, tcfg.eval_batches) if eval_batcher else None
    return StageStats(
        stage="pretrain",
        steps=tcfg.pretrain_steps,
        wall_s=wall,
        tokens_seen=tcfg.pretrain_steps * tokens,
        optimizer_numel=optimizer_numel(opt),
        peak_vram_mb=peak_vram_mb(device),
        resident_expert_count=0,
        final_train_loss=last,
        final_eval_loss=el,
        plan=mem,
    )


def apply_dense_ft_grad_policy(model: SeqMoE) -> None:
    """Freeze trunk; train only the dense MLP (pretrain residency, no expert bank)."""
    model.set_mode("pretrain")
    for name, p in model.named_parameters():
        p.requires_grad = ".mlp." in name


def train_dense_ft(
    model: SeqMoE,
    data: torch.Tensor,
    cfg: ExperimentConfig,
    *,
    tag: str,
    eval_data: torch.Tensor | None = None,
    steps: int | None = None,
) -> StageStats:
    """Fine-tune trunk MLPs only for ``expert_steps``. Does not install a bank."""
    tcfg = cfg.train
    device = str(next(model.parameters()).device)
    apply_dense_ft_grad_policy(model)
    opt = _adamw(model, tcfg.lr, tcfg.weight_decay)
    mem = plan(model, tag, device, tcfg, opt=opt)
    batcher = TokenBatcher(
        data, tcfg.seq_len, tcfg.batch_size, device=device, seed=tcfg.seed + 50 + hash(tag) % 17
    )
    eval_batcher = (
        TokenBatcher(
            eval_data,
            tcfg.seq_len,
            tcfg.batch_size,
            device=device,
            seed=tcfg.seed + 70,
        )
        if eval_data is not None and eval_data.numel() > tcfg.seq_len + 2
        else None
    )
    model.train()
    reset_peak(device)
    t0 = time.perf_counter()
    last = float("nan")
    tokens = tcfg.batch_size * tcfg.seq_len
    n_steps = tcfg.expert_steps if steps is None else int(steps)
    for step in range(1, n_steps + 1):
        x, y = batcher.sample()
        last = _step_loss(model, opt, x, y)
        if step % tcfg.log_every == 0 or step == 1:
            print(
                f"[{tag}] step {step}/{n_steps} "
                f"loss={last:.4f} opt_numel={optimizer_numel(opt)}"
            )
    wall = time.perf_counter() - t0
    el = eval_loss(model, eval_batcher, tcfg.eval_batches) if eval_batcher else None
    return StageStats(
        stage=tag,
        steps=n_steps,
        wall_s=wall,
        tokens_seen=n_steps * tokens,
        optimizer_numel=optimizer_numel(opt),
        peak_vram_mb=peak_vram_mb(device),
        resident_expert_count=0,
        final_train_loss=last,
        final_eval_loss=el,
        plan=mem,
    )


def train_expert(
    model: SeqMoE,
    data: torch.Tensor,
    cfg: ExperimentConfig,
    logical_id: int,
    *,
    eval_data: torch.Tensor | None = None,
    domain: str = "",
) -> StageStats:
    tcfg = cfg.train
    device = str(next(model.parameters()).device)
    torch.manual_seed(tcfg.seed + 1000 * int(logical_id))
    model.set_mode("expert", logical_id, expert_init=tcfg.expert_init)
    opt = _adamw(model, tcfg.lr, tcfg.weight_decay)
    assert_expert_stage(model, opt, logical_id)
    mem = plan(model, f"expert_{logical_id}", device, tcfg, opt=opt)
    batcher = TokenBatcher(
        data, tcfg.seq_len, tcfg.batch_size, device=device, seed=tcfg.seed + 10 + logical_id
    )
    eval_batcher = (
        TokenBatcher(
            eval_data,
            tcfg.seq_len,
            tcfg.batch_size,
            device=device,
            seed=tcfg.seed + 30 + logical_id,
        )
        if eval_data is not None and eval_data.numel() > tcfg.seq_len + 2
        else None
    )
    model.train()
    reset_peak(device)
    t0 = time.perf_counter()
    last = float("nan")
    tokens = tcfg.batch_size * tcfg.seq_len
    for step in range(1, tcfg.expert_steps + 1):
        x, y = batcher.sample()
        last = _step_loss(model, opt, x, y)
        if step % tcfg.log_every == 0 or step == 1:
            print(
                f"[expert {logical_id} {domain}] step {step}/{tcfg.expert_steps} "
                f"loss={last:.4f} opt_numel={optimizer_numel(opt)}"
            )
    wall = time.perf_counter() - t0
    el = eval_loss(model, eval_batcher, tcfg.eval_batches) if eval_batcher else None
    return StageStats(
        stage=f"expert_{logical_id}",
        steps=tcfg.expert_steps,
        wall_s=wall,
        tokens_seen=tcfg.expert_steps * tokens,
        optimizer_numel=optimizer_numel(opt),
        peak_vram_mb=peak_vram_mb(device),
        resident_expert_count=model.resident_expert_count(),
        final_train_loss=last,
        final_eval_loss=el,
        plan=mem,
    )


def train_router(
    model: SeqMoE,
    domain_data: dict[int, torch.Tensor],
    cfg: ExperimentConfig,
    *,
    probs: dict[int, float] | None = None,
) -> StageStats:
    tcfg = cfg.train
    device = str(next(model.parameters()).device)
    model.set_mode("router")
    opt = _adamw(model, tcfg.lr, tcfg.weight_decay)
    mem = plan(model, "router", device, tcfg, opt=opt)
    if probs is None:
        if 2 in domain_data or any(k not in (0, 1) for k in domain_data):
            # scientific-style: code/math dominate; leftover (wiki) gets the rest
            probs = {i: (0.4 if i in (0, 1) else 0.2) for i in domain_data}
        else:
            n = max(len(domain_data), 1)
            probs = {i: 1.0 / n for i in domain_data}
    batcher = MixedDomainBatcher(
        domain_data,
        probs,
        tcfg.seq_len,
        tcfg.batch_size,
        device=device,
        seed=tcfg.seed + 99,
    )
    model.train()
    reset_peak(device)
    t0 = time.perf_counter()
    last = float("nan")
    tokens = tcfg.batch_size * tcfg.seq_len
    for step in range(1, tcfg.router_steps + 1):
        x, y, gold = batcher.sample()
        last = _step_loss(
            model,
            opt,
            x,
            y,
            balance_coef=tcfg.balance_coef,
            route_coef=tcfg.route_coef,
            gold_id=gold if tcfg.route_coef else None,
        )
        if step % tcfg.log_every == 0 or step == 1:
            print(f"[router] step {step}/{tcfg.router_steps} loss={last:.4f}")
    wall = time.perf_counter() - t0
    return StageStats(
        stage="router",
        steps=tcfg.router_steps,
        wall_s=wall,
        tokens_seen=tcfg.router_steps * tokens,
        optimizer_numel=optimizer_numel(opt),
        peak_vram_mb=peak_vram_mb(device),
        resident_expert_count=model.resident_expert_count(),
        final_train_loss=last,
        plan=mem,
    )


def _release(*objs: Any) -> None:
    for _ in objs:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sequential_poc(
    cfg: ExperimentConfig,
    data: PocData,
    *,
    work_dir: Path | str,
) -> dict[str, Any]:
    """Stage A (pretrain) → B (sequential experts) → C (router). Writes checkpoints."""
    tcfg = cfg.train
    device = pick_device(tcfg.device)
    tcfg.device = device
    work = Path(work_dir)
    ckpt_dir = work / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed_all(tcfg.seed)

    trunk_path = ckpt_dir / "trunk.pt"
    expert_paths: dict[int, Path] = {}
    stats: dict[str, Any] = {}

    print(f"== stage A pretrain device={device} ==")
    model = SeqMoE(cfg.model).to(device)
    mixed_hold = data.splits.get("mixed")
    stats["pretrain"] = pretrain_trunk(
        model, data.mixed_train, cfg, eval_data=mixed_hold
    ).__dict__
    digest = save_trunk(
        model,
        trunk_path,
        pretrain={
            "steps": tcfg.pretrain_steps,
            "corpus": "mixed",
            "final_eval_nll": stats["pretrain"]["final_eval_loss"],
            "scientific_grade": data.scientific_grade,
        },
    )
    stats["trunk_sha256"] = digest
    print(f"saved {trunk_path} sha256={digest[:12]}…")
    del model
    _release()

    print("== dense FT (MLP-only, frozen trunk copy) ==")
    mix_cm = torch.cat(
        [data.domains["code"].train, data.domains["math"].train]
    )
    dft_streams = {
        "dense_ft_mix": mix_cm,
        "dense_ft_code": data.domains["code"].train,
        "dense_ft_math": data.domains["math"].train,
    }
    dense_ft_paths: dict[str, Path] = {}
    for tag, stream in dft_streams.items():
        print(f"-- {tag} --")
        model = load_trunk(trunk_path, map_location="cpu").to(device)
        st = train_dense_ft(model, stream, cfg, tag=tag)
        path = ckpt_dir / f"{tag}.pt"
        save_trunk(
            model,
            path,
            pretrain={
                "steps": tcfg.expert_steps,
                "corpus": tag,
                "device": device,
                "peak_vram_mb": st.peak_vram_mb,
                "optimizer_numel": st.optimizer_numel,
                "dense_ft": True,
            },
        )
        dense_ft_paths[tag] = path
        stats[tag] = st.__dict__
        del model
        _release()

    if tcfg.expert_corpus == "mix":
        matched_steps = tcfg.expert_steps * cfg.model.n_experts
        print(f"-- dense_ft_mix_matched steps={matched_steps} --")
        model = load_trunk(trunk_path, map_location="cpu").to(device)
        st = train_dense_ft(
            model, mix_cm, cfg, tag="dense_ft_mix_matched", steps=matched_steps
        )
        path = ckpt_dir / "dense_ft_mix_matched.pt"
        save_trunk(
            model,
            path,
            pretrain={
                "steps": matched_steps,
                "corpus": "dense_ft_mix_matched",
                "device": device,
                "peak_vram_mb": st.peak_vram_mb,
                "optimizer_numel": st.optimizer_numel,
                "dense_ft": True,
            },
        )
        dense_ft_paths["dense_ft_mix_matched"] = path
        stats["dense_ft_mix_matched"] = st.__dict__
        del model
        _release()

    print("== stage B sequential experts ==")
    if tcfg.expert_corpus == "mix":
        jobs = [
            (i, f"seed{i}", data.mixed_train, data.splits.get("mixed"))
            for i in range(cfg.model.n_experts)
        ]
    else:
        jobs = [
            (0, "code", data.domains["code"].train, data.domains["code"].holdout),
            (1, "math", data.domains["math"].train, data.domains["math"].holdout),
        ]
        if cfg.model.n_experts != 2:
            raise ValueError("domain corpus currently supports n_experts=2")
    for logical_id, domain, train_stream, hold in jobs:
        print(f"-- expert {logical_id} {domain} init={tcfg.expert_init} --")
        model = SeqMoE.from_trunk(trunk_path, map_location="cpu").to(device)
        st = train_expert(
            model,
            train_stream,
            cfg,
            logical_id,
            eval_data=hold,
            domain=domain,
        )
        path = ckpt_dir / f"expert_{logical_id}_{domain}.pt"
        save_expert(
            model,
            path,
            logical_id=logical_id,
            domain=domain,
            train={
                "steps": tcfg.expert_steps,
                "corpus": domain,
                "expert_init": tcfg.expert_init,
                "device": device,
                "peak_vram_mb": st.peak_vram_mb,
                "optimizer_numel": st.optimizer_numel,
            },
        )
        expert_paths[logical_id] = path
        stats[f"expert_{logical_id}"] = st.__dict__
        for blk in model.blocks:
            if blk.bank is not None:
                blk.bank.uninstall(logical_id)
        del model
        _release()

    print("== stage C router ==")
    ordered_experts = [expert_paths[i] for i in range(cfg.model.n_experts)]
    model = assemble(trunk_path, ordered_experts).to(device)
    domain_data = {
        DOMAIN_TO_ID["code"]: data.domains["code"].train,
        DOMAIN_TO_ID["math"]: data.domains["math"].train,
    }
    if "wiki" in data.domains:
        # wiki has no gold expert; MixedDomainBatcher still needs int keys.
        # Use a dummy id only if we want wiki in the mix — spec uses wiki=0.2
        # for scientific. Tokens from wiki should not use gold route CE
        # (route_coef defaults to 0).
        domain_data[2] = data.domains["wiki"].train
    elif "fill" in data.domains:
        domain_data[2] = data.domains["fill"].train
    stats["router"] = train_router(model, domain_data, cfg).__dict__
    router_path = ckpt_dir / "router.pt"
    save_router(model, router_path)
    stats["paths"] = {
        "trunk": str(trunk_path),
        "router": str(router_path),
        **{f"expert_{i}": str(expert_paths[i]) for i in range(cfg.model.n_experts)},
        **{k: str(v) for k, v in dense_ft_paths.items()},
    }
    stats["n_params_pretrain"] = count_params(
        load_trunk(trunk_path, map_location="cpu"), trainable_only=False
    )
    del model
    _release()
    return stats


def run_and_eval(
    cfg: ExperimentConfig,
    data: PocData,
    *,
    work_dir: Path | str,
    results_dir: Path | str | None = None,
) -> dict[str, Any]:
    from seqmoe.eval import evaluate_poc, has_required_eval_arms, write_results

    tcfg = cfg.train
    work = Path(work_dir)
    stats = run_sequential_poc(cfg, data, work_dir=work)
    paths = stats["paths"]
    n_exp = cfg.model.n_experts
    payload = evaluate_poc(
        cfg,
        data,
        trunk_path=paths["trunk"],
        expert_paths=[paths[f"expert_{i}"] for i in range(n_exp)],
        router_path=paths["router"],
        dense_ft_paths={k: v for k, v in paths.items() if k.startswith("dense_ft_")},
    )
    payload["resources"] = {
        k: v for k, v in stats.items() if k not in ("paths",)
    }
    payload["paths"] = paths
    out_dir = Path(results_dir) if results_dir is not None else Path(tcfg.results_dir)
    comparison = write_results(payload, out_dir, tcfg.run_name)
    payload["comparison_path"] = str(comparison)
    if not has_required_eval_arms(payload):
        raise RuntimeError(f"comparison JSON missing required arms/NLL: {comparison}")
    print(f"wrote {comparison}")
    return payload
