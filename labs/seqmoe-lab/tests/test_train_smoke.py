"""Sequential train invariants: independence, specialization smoke, optimizer numel."""

from __future__ import annotations

from pathlib import Path

import torch

from seqmoe.assembly import save_expert, save_trunk, trunk_sha256
from seqmoe.config import ExperimentConfig, ModelConfig, TrainConfig
from seqmoe.data import TokenBatcher, build_synth_data
from seqmoe.memory import optimizer_numel, prepare_joint_trainable
from seqmoe.model import SeqMoE, expert_mlp_numel
from seqmoe.train import eval_loss, train_expert


def _exp(tmp: Path, expert_steps: int = 4) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=256,
            n_embd=32,
            n_head=4,
            n_layer=2,
            block_size=32,
            n_experts=2,
            top_k=2,
            mlp_mult=4,
        ),
        train=TrainConfig(
            seed=0,
            batch_size=4,
            seq_len=32,
            lr=3e-3,
            pretrain_steps=2,
            expert_steps=expert_steps,
            router_steps=2,
            eval_every=2,
            log_every=2,
            device="cpu",
            results_dir=str(tmp),
            run_name="pytest",
            eval_batches=2,
        ),
    )


def test_sequential_independence(tmp_path: Path):
    cfg = _exp(tmp_path, expert_steps=3)
    data = build_synth_data(seed=0)
    trunk_path = tmp_path / "trunk.pt"
    e0_path = tmp_path / "expert_0_code.pt"

    trunk = SeqMoE(cfg.model)
    digest0 = save_trunk(trunk, trunk_path)

    m0 = SeqMoE.from_trunk(trunk_path)
    train_expert(m0, data.domains["code"].train, cfg, 0, domain="code")
    save_expert(m0, e0_path, logical_id=0, domain="code")
    bytes0 = e0_path.read_bytes()
    hash_after_e0 = trunk_sha256(m0)

    m1 = SeqMoE.from_trunk(trunk_path)
    train_expert(m1, data.domains["math"].train, cfg, 1, domain="math")
    assert e0_path.read_bytes() == bytes0
    assert trunk_sha256(m1) == digest0
    assert hash_after_e0 == digest0
    assert m1.resident_expert_ids() == (1,)
    assert 0 not in m1.resident_expert_ids()


def test_synth_specialization_smoke():
    cfg = _exp(Path("."), expert_steps=20)
    data = build_synth_data(seed=1)
    m = SeqMoE(cfg.model)
    m.set_mode("expert", 0)
    code_b = TokenBatcher(data.splits["code"], cfg.train.seq_len, 4, device="cpu", seed=1)
    math_b = TokenBatcher(data.splits["math"], cfg.train.seq_len, 4, device="cpu", seed=2)
    nll_code_0 = eval_loss(m, code_b, batches=4)
    nll_math_0 = eval_loss(m, math_b, batches=4)
    train_expert(m, data.domains["code"].train, cfg, 0, domain="code")
    nll_code_1 = eval_loss(m, code_b, batches=4)
    nll_math_1 = eval_loss(m, math_b, batches=4)
    improved_code = nll_code_1 < nll_code_0
    code_better_than_math = nll_code_1 < nll_math_1
    code_rose_and_math_dropped = (nll_code_1 > nll_code_0) and (nll_math_1 < nll_math_0)
    assert not code_rose_and_math_dropped, (
        f"code expert got worse on code ({nll_code_0:.4f}->{nll_code_1:.4f}) "
        f"while improving on math ({nll_math_0:.4f}->{nll_math_1:.4f})"
    )
    assert improved_code or code_better_than_math, (
        f"no specialization signal: code {nll_code_0:.4f}->{nll_code_1:.4f} "
        f"math {nll_math_0:.4f}->{nll_math_1:.4f}"
    )


def test_optimizer_numel_sequential_vs_joint():
    cfg = ModelConfig(
        vocab_size=256, n_embd=32, n_head=4, n_layer=2, block_size=16, n_experts=2
    )
    seq = SeqMoE(cfg)
    seq.set_mode("expert", 0)
    opt_seq = torch.optim.AdamW(
        [p for p in seq.parameters() if p.requires_grad], lr=1e-3
    )
    n_seq = optimizer_numel(opt_seq)
    assert n_seq == expert_mlp_numel(cfg)
    assert n_seq == sum(p.numel() for p in seq.parameters() if p.requires_grad)

    joint = SeqMoE(cfg)
    prepare_joint_trainable(joint)
    opt_joint = torch.optim.AdamW(
        [p for p in joint.parameters() if p.requires_grad], lr=1e-3
    )
    n_joint = optimizer_numel(opt_joint)
    assert n_joint == 2 * n_seq
    assert joint.resident_expert_count() == 2
    assert seq.resident_expert_count() == 1
