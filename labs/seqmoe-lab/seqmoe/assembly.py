"""Trunk hash, canonical expert save/load, fail-closed assemble."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from seqmoe.config import ModelConfig
from seqmoe.errors import (
    DuplicateExpertError,
    ExpertConfigMismatch,
    MissingExpertError,
    TrunkHashMismatch,
)
from seqmoe.model import MLP, SeqMoE

TRUNK_FORMAT = "seqmoe-trunk-v1"
EXPERT_FORMAT = "seqmoe-expert-v1"
ROUTER_FORMAT = "seqmoe-router-v1"

_TRUNK_BLOCK = re.compile(r"^blocks\.(\d+)\.(ln1|attn|ln2)\.")


def is_trunk_key(key: str) -> bool:
    if key.startswith("tok_emb.") or key.startswith("pos_emb."):
        return True
    if key.startswith("ln_f.") or key.startswith("lm_head."):
        return True
    return bool(_TRUNK_BLOCK.match(key))


def trunk_sha256_from_state(state_dict: dict[str, torch.Tensor]) -> str:
    """SHA-256 over shared tensors only, sorted key order, CPU bytes."""
    h = hashlib.sha256()
    for key in sorted(k for k in state_dict if is_trunk_key(k)):
        value = state_dict[key].detach().cpu().contiguous().numpy().tobytes()
        h.update(value)
    return h.hexdigest()


def trunk_sha256(model: SeqMoE) -> str:
    return trunk_sha256_from_state(model.state_dict())


def _torch_load(path: Path | str, map_location: str = "cpu") -> Any:
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_trunk(
    model: SeqMoE,
    path: Path | str,
    *,
    pretrain: dict[str, Any] | None = None,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = trunk_sha256(model)
    payload = {
        "format": TRUNK_FORMAT,
        "model_config": asdict(model.cfg),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "trunk_sha256": digest,
        "tokenizer": "bytes256",
        "pretrain": pretrain or {},
    }
    torch.save(payload, path)
    model._stored_trunk_sha256 = digest
    return digest


def load_trunk(path: Path | str, map_location: str = "cpu") -> SeqMoE:
    ckpt = _torch_load(path, map_location=map_location)
    if ckpt.get("format") != TRUNK_FORMAT:
        raise ValueError(f"not a trunk checkpoint: {ckpt.get('format')}")
    cfg = ModelConfig(**ckpt["model_config"])
    model = SeqMoE(cfg)
    model.load_state_dict(ckpt["state_dict"])
    recomputed = trunk_sha256_from_state(ckpt["state_dict"])
    stored = ckpt.get("trunk_sha256")
    if stored is not None and stored != recomputed:
        raise TrunkHashMismatch(
            f"trunk.pt hash field {stored} != recomputed {recomputed}"
        )
    model._stored_trunk_sha256 = recomputed
    return model


# Canonical on-disk MLP keys, independent of bank residency.
def canonical_mlp_key(layer: int, which: str) -> str:
    if which not in ("fc", "proj"):
        raise ValueError(which)
    return f"blocks.{layer}.mlp.{which}.weight"


def save_expert(
    model: SeqMoE,
    path: Path | str,
    *,
    logical_id: int,
    domain: str,
    train: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logical_id = int(logical_id)
    state: dict[str, torch.Tensor] = {}
    for i, blk in enumerate(model.blocks):
        if blk.bank is None or logical_id not in blk.bank.resident_ids():
            raise ValueError(
                f"expert {logical_id} is not resident at layer {i}; cannot save"
            )
        sd = blk.bank.get(logical_id).state_dict()
        state[canonical_mlp_key(i, "fc")] = sd["fc.weight"].detach().cpu()
        state[canonical_mlp_key(i, "proj")] = sd["proj.weight"].detach().cpu()
    digest = model._stored_trunk_sha256 or trunk_sha256(model)
    payload = {
        "format": EXPERT_FORMAT,
        "logical_id": logical_id,
        "domain": domain,
        "trunk_sha256": digest,
        "model_config": asdict(model.cfg),
        "state_dict": state,
        "train": train or {},
    }
    torch.save(payload, path)


def _check_expert_config(trunk_cfg: ModelConfig, expert_cfg: dict[str, Any]) -> None:
    for key in ("n_layer", "n_embd", "mlp_mult"):
        got = expert_cfg.get(key)
        want = getattr(trunk_cfg, key)
        if got != want:
            raise ExpertConfigMismatch(f"{key}: expert={got} trunk={want}")


def load_expert_checkpoint(path: Path | str, map_location: str = "cpu") -> dict[str, Any]:
    ckpt = _torch_load(path, map_location=map_location)
    if ckpt.get("format") != EXPERT_FORMAT:
        raise ValueError(f"not an expert checkpoint: {ckpt.get('format')}")
    return ckpt


def install_expert(model: SeqMoE, ckpt: dict[str, Any]) -> int:
    """Install one expert from a checkpoint into the live bank. Validates hash/config."""
    if model.mode == "pretrain":
        model.set_mode("eval")
    _check_expert_config(model.cfg, ckpt["model_config"])
    current = model._stored_trunk_sha256 or trunk_sha256(model)
    embedded = ckpt["trunk_sha256"]
    if embedded != current:
        raise TrunkHashMismatch(
            f"expert logical_id={ckpt.get('logical_id')} hash {embedded} != trunk {current}"
        )
    logical_id = int(ckpt["logical_id"])
    state = ckpt["state_dict"]
    for i, blk in enumerate(model.blocks):
        assert blk.bank is not None
        mlp = MLP(model.cfg.n_embd, model.cfg.mlp_mult, model.cfg.dropout)
        mlp.load_state_dict(
            {
                "fc.weight": state[canonical_mlp_key(i, "fc")],
                "proj.weight": state[canonical_mlp_key(i, "proj")],
            }
        )
        blk.bank.install(logical_id, model._placed(mlp))
    return logical_id


def load_expert(model: SeqMoE, path: Path | str) -> int:
    return install_expert(model, load_expert_checkpoint(path))


def assemble(trunk_path: Path | str, expert_paths: list[Path | str]) -> SeqMoE:
    """Load trunk, drop dense MLPs, install every expert, random routers.

    Raises on hash/config mismatch, duplicates, or missing logical ids.
    No partial assemble.
    """
    model = load_trunk(trunk_path)
    # Drop dense MLPs, empty banks, Kaiming routers.
    model.set_mode("router")
    ckpts = [load_expert_checkpoint(p) for p in expert_paths]
    ids = [int(c["logical_id"]) for c in ckpts]
    if len(ids) != len(set(ids)):
        raise DuplicateExpertError(f"duplicate logical ids: {ids}")
    # Validate all before installing any (fail closed, no partial).
    current = model._stored_trunk_sha256 or trunk_sha256(model)
    for ckpt in ckpts:
        _check_expert_config(model.cfg, ckpt["model_config"])
        if ckpt["trunk_sha256"] != current:
            raise TrunkHashMismatch(
                f"expert logical_id={ckpt.get('logical_id')} hash "
                f"{ckpt['trunk_sha256']} != trunk {current}"
            )
    want = set(range(model.cfg.n_experts))
    got = set(ids)
    if got != want:
        raise MissingExpertError(f"logical ids {sorted(got)} != {sorted(want)}")
    for ckpt in ckpts:
        install_expert(model, ckpt)
    model.set_mode("router")  # routers-only grad policy after installs
    return model


def save_router(model: SeqMoE, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if ".router." in k
    }
    torch.save(
        {
            "format": ROUTER_FORMAT,
            "trunk_sha256": model._stored_trunk_sha256 or trunk_sha256(model),
            "model_config": asdict(model.cfg),
            "state_dict": state,
        },
        path,
    )


def load_router(model: SeqMoE, path: Path | str) -> None:
    ckpt = _torch_load(path)
    if ckpt.get("format") != ROUTER_FORMAT:
        raise ValueError(f"not a router checkpoint: {ckpt.get('format')}")
    current = model._stored_trunk_sha256 or trunk_sha256(model)
    if ckpt["trunk_sha256"] != current:
        raise TrunkHashMismatch("router trunk hash mismatch")
    model.load_state_dict(ckpt["state_dict"], strict=False)
