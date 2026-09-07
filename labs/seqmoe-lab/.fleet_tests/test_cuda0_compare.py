"""Sequential vs joint on GPU0. Sequential optimizer is half; peak VRAM not higher than joint."""
import torch
from seqmoe.config import tiny_config
from seqmoe.memory import compare_sequential_vs_joint


def test_compare_sequential_vs_joint_on_cuda0():
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
    cfg = tiny_config("cuda:0")
    cfg.train.device = "cuda:0"
    cfg.train.batch_size = 2
    cfg.train.seq_len = 16
    cfg.model.block_size = 16
    out = compare_sequential_vs_joint(cfg, steps=2)
    seq, joint = out["sequential"], out["joint"]
    assert joint["optimizer_numel"] == 2 * seq["optimizer_numel"]
    assert seq["resident_expert_count"] == 1
    assert joint["resident_expert_count"] == 2
    assert seq["peak_vram_mb"] > 0.0
    assert seq["peak_vram_mb"] <= joint["peak_vram_mb"] + 1.0  # tiny noise
