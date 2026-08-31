"""Hash, canonical expert I/O, fail-closed assemble."""

from __future__ import annotations

from pathlib import Path

import torch

from seqmoe.assembly import (
    assemble,
    load_expert,
    load_expert_checkpoint,
    save_expert,
    save_trunk,
    trunk_sha256,
)
from seqmoe.config import ModelConfig
from seqmoe.errors import ExpertConfigMismatch, TrunkHashMismatch
from seqmoe.model import SeqMoE


def _cfg(**kwargs) -> ModelConfig:
    base = dict(
        vocab_size=256,
        n_embd=32,
        n_head=4,
        n_layer=2,
        block_size=16,
        n_experts=2,
        top_k=2,
        mlp_mult=4,
    )
    base.update(kwargs)
    return ModelConfig(**base)


def _pair(tmp: Path) -> tuple[Path, Path, Path]:
    trunk = tmp / "trunk.pt"
    e0 = tmp / "expert_0_code.pt"
    e1 = tmp / "expert_1_math.pt"
    return trunk, e0, e1


def test_save_load_expert_roundtrip(tmp_path: Path):
    cfg = _cfg()
    m = SeqMoE(cfg)
    trunk, e0, _ = _pair(tmp_path)
    save_trunk(m, trunk)
    m.set_mode("expert", 0)
    with torch.no_grad():
        m.blocks[0].bank.get(0).fc.weight.add_(0.25)
    want = m.blocks[0].bank.get(0).fc.weight.detach().clone()
    save_expert(m, e0, logical_id=0, domain="code")
    m2 = SeqMoE.from_trunk(trunk)
    m2.set_mode("expert", 0)
    logical_id = load_expert(m2, e0)
    assert logical_id == 0
    got = m2.blocks[0].bank.get(0).fc.weight.detach()
    assert torch.equal(got.cpu(), want.cpu())
    ckpt = load_expert_checkpoint(e0)
    assert ckpt["logical_id"] == 0
    assert ckpt["domain"] == "code"
    assert "blocks.0.mlp.fc.weight" in ckpt["state_dict"]
    assert not any("bank.experts" in k for k in ckpt["state_dict"])


def test_assemble_rejects_hash_mismatch(tmp_path: Path):
    cfg = _cfg()
    m = SeqMoE(cfg)
    trunk, e0, e1 = _pair(tmp_path)
    save_trunk(m, trunk)
    m.set_mode("expert", 0)
    save_expert(m, e0, logical_id=0, domain="code")
    m = SeqMoE.from_trunk(trunk)
    m.set_mode("expert", 1)
    save_expert(m, e1, logical_id=1, domain="math")

    bad = SeqMoE.from_trunk(trunk)
    with torch.no_grad():
        bad.blocks[0].attn.qkv.weight.add_(1.0)
    bad_path = tmp_path / "trunk_bad.pt"
    save_trunk(bad, bad_path)
    try:
        assemble(bad_path, [e0, e1])
    except TrunkHashMismatch:
        return
    raise AssertionError("assemble must refuse a flipped trunk")


def test_assemble_rejects_wrong_n_embd(tmp_path: Path):
    cfg = _cfg()
    m = SeqMoE(cfg)
    trunk, e0, e1 = _pair(tmp_path)
    save_trunk(m, trunk)
    m.set_mode("expert", 0)
    save_expert(m, e0, logical_id=0, domain="code")
    m = SeqMoE.from_trunk(trunk)
    m.set_mode("expert", 1)
    save_expert(m, e1, logical_id=1, domain="math")
    ckpt = load_expert_checkpoint(e0)
    ckpt["model_config"]["n_embd"] = 64
    torch.save(ckpt, e0)
    try:
        assemble(trunk, [e0, e1])
    except ExpertConfigMismatch:
        return
    raise AssertionError("assemble must refuse n_embd mismatch")


def test_assemble_success_same_trunk_hash(tmp_path: Path):
    m = SeqMoE(_cfg())
    trunk, e0, e1 = _pair(tmp_path)
    digest = save_trunk(m, trunk)
    m.set_mode("expert", 0)
    save_expert(m, e0, logical_id=0, domain="code")
    m = SeqMoE.from_trunk(trunk)
    m.set_mode("expert", 1)
    save_expert(m, e1, logical_id=1, domain="math")
    moe = assemble(trunk, [e0, e1])
    assert moe.resident_expert_ids() == (0, 1)
    assert trunk_sha256(moe) == digest
    x = torch.randint(0, 256, (2, 16))
    fr = moe(x, route="learned", top_k=2)
    assert fr.logits.shape[-1] == 256
