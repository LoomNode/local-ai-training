"""GPU0 (gaming card) envelope: plan must fail closed without allocating."""
import pytest
import torch
from seqmoe.config import tiny_config
from seqmoe.errors import MemoryPlanError
from seqmoe.memory import plan
from seqmoe.model import SeqMoE


def test_plan_raises_on_cuda0_when_envelope_is_tiny():
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
    cfg = tiny_config("cuda:0")
    cfg.train.device = "cuda:0"
    cfg.train.max_vram_mb = 1
    m = SeqMoE(cfg.model)
    with pytest.raises(MemoryPlanError):
        plan(m, "pretrain", "cuda:0", cfg.train)
