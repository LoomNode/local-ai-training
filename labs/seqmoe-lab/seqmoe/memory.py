"""Fail-closed VRAM envelope. Swap is not capacity."""

from __future__ import annotations

import gc
from typing import Any

import torch

from seqmoe.config import ExperimentConfig, TrainConfig, tiny_config
from seqmoe.errors import MemoryPlanError
from seqmoe.model import SeqMoE, count_params, expert_mlp_numel

CUDA_MARGIN = 512 * 1024 * 1024
PLAN_KEYS = (
    "param_bytes",
    "grad_bytes",
    "optim_bytes",
    "activation_est",
    "cuda_margin",
    "total",
    "resident_expert_count",
    "optimizer_numel",
    "stage",
    "device",
)


def optimizer_numel(opt: torch.optim.Optimizer) -> int:
    return sum(p.numel() for g in opt.param_groups for p in g["params"])


def peak_vram_mb(device: str) -> float:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return 0.0
    idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
    return torch.cuda.max_memory_allocated(idx) / (1024 * 1024)


def reset_peak(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
        torch.cuda.reset_peak_memory_stats(idx)


def plan(
    model: SeqMoE,
    stage: str,
    device: str,
    train: TrainConfig,
    *,
    opt: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Estimate bytes, then abort rather than allocate past the envelope."""
    cfg = model.cfg
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )
    optim_bytes = 2 * grad_bytes
    activation_est = (
        train.batch_size * train.seq_len * cfg.n_embd * cfg.n_layer * 4 * 12
    )
    total = param_bytes + grad_bytes + optim_bytes + activation_est + CUDA_MARGIN
    opt_n = optimizer_numel(opt) if opt is not None else count_params(model, True)
    out: dict[str, Any] = {
        "param_bytes": int(param_bytes),
        "grad_bytes": int(grad_bytes),
        "optim_bytes": int(optim_bytes),
        "activation_est": int(activation_est),
        "cuda_margin": int(CUDA_MARGIN),
        "total": int(total),
        "resident_expert_count": int(model.resident_expert_count()),
        "optimizer_numel": int(opt_n),
        "stage": stage,
        "device": device,
        "swap_counted": False,
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
        free, tot = torch.cuda.mem_get_info(idx)
        already_used = tot - free
        cap = min(train.max_vram_mb * 1024 * 1024, tot - already_used)
        out["device_total_bytes"] = int(tot)
        out["already_used_bytes"] = int(already_used)
        out["cap_bytes"] = int(cap)
        if total > cap:
            raise MemoryPlanError(
                "memory plan exceeds envelope: "
                + ", ".join(f"{k}={out[k]}" for k in PLAN_KEYS)
                + f", cap_bytes={int(cap)}"
            )
    return out


def assert_expert_stage(
    model: SeqMoE, opt: torch.optim.Optimizer, logical_id: int
) -> None:
    if model.resident_expert_count() != 1:
        raise RuntimeError(
            f"expert stage resident_expert_count={model.resident_expert_count()} != 1"
        )
    want = expert_mlp_numel(model.cfg)
    got = optimizer_numel(opt)
    if got != want:
        raise RuntimeError(f"optimizer_numel={got} != expert_mlp_numel={want}")
    needle = f"bank.experts.{int(logical_id)}."
    for name, _ in model.named_parameters():
        if "bank.experts." in name and needle not in name:
            raise RuntimeError(f"non-resident expert param in tree: {name}")
    # No parked logical id in the state dict.
    for name in model.state_dict():
        if "bank.experts." in name and needle not in name:
            raise RuntimeError(f"non-resident expert key in state_dict: {name}")


def prepare_joint_trainable(model: SeqMoE, *, init: str = "clone") -> None:
    """Both expert FFNs resident and trainable. Memory counterfactual, not a v1 stage."""
    for eid in range(model.cfg.n_experts):
        model.clone_and_install(eid, init=init)
    for blk in model.blocks:
        blk.router = None
    model.mode = "eval"
    model._expert_id = None
    for n, p in model.named_parameters():
        p.requires_grad = ".bank.experts." in n


def _few_steps(
    model: SeqMoE,
    opt: torch.optim.Optimizer,
    device: str,
    seq_len: int,
    batch: int,
    steps: int,
    vocab: int,
    **forward_kwargs,
) -> float:
    model.train()
    reset_peak(device)
    for _ in range(steps):
        x = torch.randint(0, vocab, (batch, seq_len), device=device)
        y = torch.randint(0, vocab, (batch, seq_len), device=device)
        fr = model(x, y, **forward_kwargs)
        assert fr.loss is not None
        opt.zero_grad(set_to_none=True)
        fr.loss.backward()
        opt.step()
    return peak_vram_mb(device)


def compare_sequential_vs_joint(
    cfg: ExperimentConfig | None = None,
    *,
    steps: int = 4,
) -> dict[str, Any]:
    """Tiny sequential vs joint (both experts in one optimizer). CPU peak is 0."""
    cfg = cfg or tiny_config("cpu")
    device = cfg.train.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    seq_len = min(cfg.train.seq_len, cfg.model.block_size)
    batch = cfg.train.batch_size

    seq_model = SeqMoE(cfg.model).to(device)
    seq_model.set_mode("expert", 0, expert_init=cfg.train.expert_init)
    seq_params = [p for p in seq_model.parameters() if p.requires_grad]
    seq_opt = torch.optim.AdamW(seq_params, lr=cfg.train.lr)
    seq_plan = plan(seq_model, "expert", device, cfg.train, opt=seq_opt)
    assert_expert_stage(seq_model, seq_opt, 0)
    seq_peak = _few_steps(
        seq_model,
        seq_opt,
        device,
        seq_len,
        batch,
        steps,
        cfg.model.vocab_size,
        force_expert=0,
    )
    seq_numel = optimizer_numel(seq_opt)
    seq_resident = seq_model.resident_expert_count()
    del seq_model, seq_opt
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    joint = SeqMoE(cfg.model).to(device)
    prepare_joint_trainable(joint, init=cfg.train.expert_init)
    joint_params = [p for p in joint.parameters() if p.requires_grad]
    joint_opt = torch.optim.AdamW(joint_params, lr=cfg.train.lr)
    joint_plan = plan(joint, "joint", device, cfg.train, opt=joint_opt)
    joint_peak = _few_steps(
        joint,
        joint_opt,
        device,
        seq_len,
        batch,
        steps,
        cfg.model.vocab_size,
        route="random",
        top_k=cfg.model.n_experts,
    )
    joint_numel = optimizer_numel(joint_opt)
    joint_resident = joint.resident_expert_count()

    return {
        "sequential": {
            "optimizer_numel": int(seq_numel),
            "peak_vram_mb": float(seq_peak),
            "resident_expert_count": int(seq_resident),
            "plan": seq_plan,
        },
        "joint": {
            "optimizer_numel": int(joint_numel),
            "peak_vram_mb": float(joint_peak),
            "resident_expert_count": int(joint_resident),
            "plan": joint_plan,
        },
        "ratio": (float(joint_numel) / float(seq_numel)) if seq_numel else None,
        "device": device,
    }
