"""Learned Top-2 vs random on real dumps, tiny sequential train on cuda:0."""
from pathlib import Path

import torch

from seqmoe.assembly import assemble, save_expert, save_router, save_trunk
from seqmoe.config import tiny_config
from seqmoe.data import build_scientific_data
from seqmoe.eval import evaluate_poc
from seqmoe.model import SeqMoE
from seqmoe.train import pretrain_trunk, train_expert, train_router


def test_scientific_dumps_are_full():
    data = build_scientific_data(seed=0)
    assert data.scientific_grade == "full", data.scientific_grade
    assert data.code_source == "dump"
    assert data.math_source == "dump"


def test_top2_mixed_nll_beats_random_after_short_train(tmp_path: Path):
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
    data = build_scientific_data(seed=1)
    assert data.scientific_grade == "full"
    cfg = tiny_config("cuda:0")
    cfg.train.device = "cuda:0"
    cfg.train.pretrain_steps = 8
    cfg.train.expert_steps = 8
    cfg.train.router_steps = 12
    cfg.train.eval_batches = 2
    cfg.train.batch_size = 4
    cfg.train.seq_len = 64
    cfg.train.results_dir = str(tmp_path)

    m = SeqMoE(cfg.model).to("cuda:0")
    pretrain_trunk(m, data.mixed_train, cfg, eval_data=data.splits["mixed"])
    trunk = tmp_path / "trunk.pt"
    save_trunk(m, trunk)
    del m

    e0 = tmp_path / "e0.pt"
    m0 = SeqMoE.from_trunk(trunk).to("cuda:0")
    train_expert(m0, data.domains["code"].train, cfg, 0, domain="code")
    save_expert(m0, e0, logical_id=0, domain="code")
    del m0

    e1 = tmp_path / "e1.pt"
    m1 = SeqMoE.from_trunk(trunk).to("cuda:0")
    train_expert(m1, data.domains["math"].train, cfg, 1, domain="math")
    save_expert(m1, e1, logical_id=1, domain="math")
    del m1

    rpath = tmp_path / "router.pt"
    assembled = assemble(trunk, [e0, e1]).to("cuda:0")
    train_router(
        assembled,
        {
            0: data.domains["code"].train,
            1: data.domains["math"].train,
            2: data.domains["wiki"].train,
        },
        cfg,
    )
    save_router(assembled, rpath)

    payload = evaluate_poc(
        cfg, data, trunk_path=trunk, expert_paths=[e0, e1], router_path=rpath
    )
    top2 = payload["arms"]["seqmoe_top2"]["mixed"]["nll"]
    rnd = payload["arms"]["random_route"]["mixed"]["nll"]
    assert top2 == top2 and rnd == rnd
    assert top2 <= rnd, f"top2 {top2} vs random {rnd}"
