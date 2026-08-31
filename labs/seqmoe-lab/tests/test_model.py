"""Drive the shipped SeqMoE: dense forward, clone, residency, Top-K sparsity."""

from __future__ import annotations

import torch

from seqmoe.config import ModelConfig, resolve_device
from seqmoe.model import SeqMoE


def _cfg(**kwargs) -> ModelConfig:
    base = dict(
        vocab_size=256,
        n_embd=32,
        n_head=4,
        n_layer=2,
        block_size=16,
        dropout=0.0,
        n_experts=2,
        top_k=2,
        mlp_mult=4,
    )
    base.update(kwargs)
    return ModelConfig(**base)


def test_resolve_device_maps_bare_cuda():
    assert resolve_device("cuda") == "cuda:1"
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:0") == "cuda:0"


def test_forward_loss_shapes():
    cfg = _cfg()
    m = SeqMoE(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 16))
    y = torch.randint(0, cfg.vocab_size, (4, 16))
    fr = m(x, y)
    assert fr.logits.shape == (4, 16, cfg.vocab_size)
    assert fr.loss is not None and fr.loss.ndim == 0
    assert torch.isfinite(fr.loss)


def test_tied_lm_head():
    m = SeqMoE(_cfg())
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()


def test_clone_equals_trunk_mlp():
    cfg = _cfg()
    m = SeqMoE(cfg)
    fc = [blk.mlp.fc.weight.detach().clone() for blk in m.blocks]
    proj = [blk.mlp.proj.weight.detach().clone() for blk in m.blocks]
    m.set_mode("expert", 0)
    for i, blk in enumerate(m.blocks):
        mlp = blk.bank.get(0)
        assert (mlp.fc.weight.detach().cpu() - fc[i].cpu()).abs().max().item() == 0
        assert (mlp.proj.weight.detach().cpu() - proj[i].cpu()).abs().max().item() == 0


def test_expert_mode_trainable_only_active_mlp():
    m = SeqMoE(_cfg())
    m.set_mode("expert", 1)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable, "expert mode must train the resident MLP"
    assert all("bank.experts.1." in n for n in trainable)
    frozen = [n for n, p in m.named_parameters() if not p.requires_grad]
    assert any("attn" in n for n in frozen)
    assert any("tok_emb" in n for n in frozen)
    assert not any("bank.experts.0." in n for n, _ in m.named_parameters())


def test_expert1_does_not_allocate_expert0():
    m = SeqMoE(_cfg())
    m.set_mode("expert", 1)
    for blk in m.blocks:
        assert blk.bank is not None
        assert "0" not in blk.bank.experts
        assert 0 not in blk.bank.resident_ids()
        assert blk.mlp is None
    names = [n for n, _ in m.named_modules()]
    assert not any(n.endswith("experts.0") or ".experts.0." in n for n in names)


def test_router_topk_sparsity():
    cfg = _cfg(n_layer=1, top_k=2, block_size=8)
    m = SeqMoE(cfg)
    m.clone_and_install(0)
    m.clone_and_install(1)
    m.set_mode("router")
    x = torch.randint(0, cfg.vocab_size, (3, 8))
    y = torch.randint(0, cfg.vocab_size, (3, 8))
    fr1 = m(x, y, route="learned", top_k=1)
    w1 = fr1.token_route_weights
    assert w1 is not None and w1.shape == (3, 8, 2)
    nz1 = (w1 > 1e-6).sum(dim=-1)
    assert torch.all(nz1 == 1), "Top-K=1 must be one expert per token"
    fr2 = m(x, y, route="learned", top_k=2)
    w2 = fr2.token_route_weights
    assert w2 is not None
    nz2 = (w2 > 1e-6).sum(dim=-1)
    assert torch.all(nz2 <= 2)
    assert torch.all(nz2 >= 1)


def test_set_mode_expert_requires_id():
    m = SeqMoE(_cfg())
    try:
        m.set_mode("expert")
    except ValueError as e:
        assert "expert_id" in str(e)
    else:
        raise AssertionError("expected ValueError")
