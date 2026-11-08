"""Dense MLP fine-tune arms: MLP-only grads and eval JSON on the shipped path."""

from __future__ import annotations

from pathlib import Path

import torch

from seqmoe.assembly import save_expert, save_trunk
from seqmoe.config import ExperimentConfig, ModelConfig, TrainConfig
from seqmoe.data import build_synth_data
from seqmoe.eval import evaluate_poc, has_required_eval_arms
from seqmoe.model import SeqMoE
from seqmoe.train import train_dense_ft


def _cfg(tmp: Path, expert_steps: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
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
            lr=3e-3,
            expert_steps=expert_steps,
            device="cpu",
            eval_batches=2,
            results_dir=str(tmp),
            run_name="dft_smoke",
            log_every=1,
        ),
    )


def test_dense_ft_trainable_only_mlp(tmp_path: Path):
    cfg = _cfg(tmp_path)
    data = build_synth_data(seed=0)
    m = SeqMoE(cfg.model)
    save_trunk(m, tmp_path / "trunk.pt")
    train_dense_ft(m, data.domains["code"].train, cfg, tag="dense_ft_code")
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable, "dense_ft must train the MLP"
    assert all(".mlp." in n for n in trainable)
    frozen = [n for n, p in m.named_parameters() if not p.requires_grad]
    assert any("attn" in n for n in frozen)
    assert any("tok_emb" in n for n in frozen)
    assert any("ln_f" in n or n.startswith("ln_f") for n in frozen)
    assert m.mode == "pretrain"
    assert m.blocks[0].mlp is not None
    assert m.blocks[0].bank is None


def test_dense_ft_arms_numeric_json(tmp_path: Path):
    cfg = _cfg(tmp_path, expert_steps=2)
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

    dft_paths: dict[str, Path] = {}
    mix = torch.cat([data.domains["code"].train, data.domains["math"].train])
    streams = {
        "dense_ft_mix": mix,
        "dense_ft_code": data.domains["code"].train,
        "dense_ft_math": data.domains["math"].train,
    }
    for tag, stream in streams.items():
        model = SeqMoE.from_trunk(trunk)
        train_dense_ft(model, stream, cfg, tag=tag)
        path = tmp_path / f"{tag}.pt"
        save_trunk(model, path)
        dft_paths[tag] = path

    payload = evaluate_poc(
        cfg,
        data,
        trunk_path=trunk,
        expert_paths=[e0, e1],
        router_path=None,
        dense_ft_paths=dft_paths,
    )
    assert has_required_eval_arms(payload)
    for arm in ("dense_ft_mix", "dense_ft_code", "dense_ft_math"):
        for split in ("code", "math", "mixed"):
            cell = payload["arms"][arm][split]
            assert isinstance(cell["nll"], float)
            assert isinstance(cell["bpb"], float)
            assert cell["nll"] == cell["nll"]
            assert cell["bpb"] > 0
    steps = payload["steps"]
    assert steps["dense_ft_mix_wall_steps"] == steps["pretrain"] + steps["expert"]
    assert steps["seqmoe_wall_steps"] == steps["pretrain"] + 2 * steps["expert"] + steps["router"]
