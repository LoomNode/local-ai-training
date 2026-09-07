"""Fail-closed memory plan on CPU (no CUDA required)."""
import pytest
from seqmoe.config import tiny_config
from seqmoe.errors import MemoryPlanError
from seqmoe.memory import plan
from seqmoe.model import SeqMoE


def test_plan_raises_on_cpu_when_envelope_is_tiny():
    cfg = tiny_config("cpu")
    cfg.train.max_vram_mb = 1  # 1 MiB — smaller than any real plan
    m = SeqMoE(cfg.model)
    with pytest.raises(MemoryPlanError):
        plan(m, "pretrain", "cpu", cfg.train)
