"""Scratch: mirror of .fleet_tests/test_top2_beats_random_dumps.py with verbose prints.

Run: python3 scratch_dump_check.py
"""

from __future__ import annotations

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
    print("grade:", data.scientific_grade, data.code_source, data.math_source)
    cfg = tiny_config("cuda:0")
    cfg.train.device = "cuda:0"
    cfg.train.pretrain_steps = 8
    cfg.train.expert_steps = 8
    cfg.train.router_steps = 12
    cfg.train.eval_batches = 2
    cfg.train.batch_size = 4
    cfg.train.seq_len = 64
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
        arms = payload["arms"]
        for name in ("trunk", "random_route", "seqmoe_top1", "seqmoe_top2"):
            blk = arms[name]
            print(name, {k: round(v["nll"], 4) for k, v in blk.items() if isinstance(v, dict) and "nll" in v})
        print("hard_route", {k: round(v["nll"], 4) for k, v in arms["hard_route"].items() if isinstance(v, dict)})
        if "entropy" in arms["seqmoe_top2"].get("mixed", {}):
            print("top2 mixed entropy/route_frac:",
                  arms["seqmoe_top2"]["mixed"].get("entropy"),
                  {k: round(v, 3) for k, v in arms["seqmoe_top2"]["mixed"].items() if k.startswith("route_frac")})
        top2 = arms["seqmoe_top2"]["mixed"]["nll"]
        rnd = arms["random_route"]["mixed"]["nll"]
        print(f"RESULT top2={top2:.4f} random={rnd:.4f} diff={top2 - rnd:+.4f} pass={top2 <= rnd}")


if __name__ == "__main__":
    main()
