"""Parametrized sweep: which config makes seqmoe_top2 robustly beat random_route?

Run: ROUTE_COEF=0.1 EXPERT_INIT=clone python3 scratch_sweep.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seqmoe.assembly import assemble, save_expert, save_router, save_trunk
from seqmoe.config import tiny_config
from seqmoe.data import build_scientific_data
from seqmoe.eval import evaluate_poc
from seqmoe.model import SeqMoE
from seqmoe.train import pretrain_trunk, train_expert, train_router


def main() -> None:
    data = build_scientific_data(seed=1)
    assert data.scientific_grade == "full"
    route_coef = float(os.environ.get("ROUTE_COEF", "0.0"))
    expert_init = os.environ.get("EXPERT_INIT", "clone")
    balance_coef = float(os.environ.get("BALANCE_COEF", "0.01"))

    cfg = tiny_config("cuda:0")
    cfg.train.device = "cuda:0"
    cfg.train.pretrain_steps = 8
    cfg.train.expert_steps = 8
    cfg.train.router_steps = 12
    cfg.train.eval_batches = 2
    cfg.train.batch_size = 4
    cfg.train.seq_len = 64
    cfg.train.route_coef = route_coef
    cfg.train.balance_coef = balance_coef
    cfg.train.expert_init = expert_init
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
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
            {0: data.domains["code"].train, 1: data.domains["math"].train, 2: data.domains["wiki"].train},
            cfg,
        )
        save_router(assembled, rpath)

        payload = evaluate_poc(cfg, data, trunk_path=trunk, expert_paths=[e0, e1], router_path=rpath)
        arms = payload["arms"]
        top2 = arms["seqmoe_top2"]["mixed"]["nll"]
        top1 = arms["seqmoe_top1"]["mixed"]["nll"]
        rnd = arms["random_route"]["mixed"]["nll"]
        fr = arms["seqmoe_top2"]["mixed"]
        print(f"route_coef={route_coef} init={expert_init} bal={balance_coef} "
              f"top2={top2:.4f} top1={top1:.4f} random={rnd:.4f} "
              f"gap(top2-rand)={top2 - rnd:+.4f} PASS={top2 <= rnd} "
              f"frac0={fr.get('route_frac_0', 0):.3f} frac1={fr.get('route_frac_1', 0):.3f}")


if __name__ == "__main__":
    main()
