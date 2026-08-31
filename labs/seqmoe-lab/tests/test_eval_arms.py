"""Eval arms write numeric NLL/bpb from the shipped evaluate path."""

from __future__ import annotations

import math
from pathlib import Path

from seqmoe.assembly import save_expert, save_trunk
from seqmoe.config import ExperimentConfig, ModelConfig, TrainConfig
from seqmoe.data import build_synth_data
from seqmoe.eval import evaluate_poc, has_required_tiny_arms, write_results
from seqmoe.model import SeqMoE


def test_eval_arms_numeric_json(tmp_path: Path):
    cfg = ExperimentConfig(
        model=ModelConfig(
            vocab_size=256,
            n_embd=32,
            n_head=4,
            n_layer=2,
            block_size=32,
            n_experts=2,
            top_k=2,
        ),
        train=TrainConfig(
            seed=0,
            batch_size=4,
            seq_len=32,
            device="cpu",
            eval_batches=2,
            results_dir=str(tmp_path),
            run_name="eval_smoke",
        ),
    )
    data = build_synth_data(seed=0)
    trunk = tmp_path / "trunk.pt"
    e0 = tmp_path / "expert_0_code.pt"
    e1 = tmp_path / "expert_1_math.pt"
    m = SeqMoE(cfg.model)
    save_trunk(m, trunk)
    m.set_mode("expert", 0)
    save_expert(m, e0, logical_id=0, domain="code")
    m = SeqMoE.from_trunk(trunk)
    m.set_mode("expert", 1)
    save_expert(m, e1, logical_id=1, domain="math")

    payload = evaluate_poc(
        cfg, data, trunk_path=trunk, expert_paths=[e0, e1], router_path=None
    )
    assert has_required_tiny_arms(payload)
    path = write_results(payload, tmp_path, cfg.train.run_name)
    assert path.is_file()
    for arm in ("trunk", "expert_only_code", "expert_only_math", "random_route"):
        for split in ("code", "math", "mixed"):
            cell = payload["arms"][arm][split]
            assert isinstance(cell["nll"], float)
            assert isinstance(cell["bpb"], float)
            assert cell["nll"] == cell["nll"]
            assert cell["bpb"] > 0
    learned = payload["arms"]["seqmoe_top2"]
    for split in ("code", "math", "mixed"):
        assert isinstance(learned[split]["nll"], float)
        assert abs(learned[split]["bpb"] - learned[split]["nll"] / math.log(2.0)) < 1e-9
