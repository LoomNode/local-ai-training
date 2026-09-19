"""
Runtime device / mode config for envelope-lab.

GPU policy on this box:
  GPU 1 = dedicated AI card (always part of the lab)
  GPU 0 = second card (display-friendly; used when turbo is on)

solo  = DEFAULT. single-card: ONLY GPU 1 (AI_GPU). GPU 0 left alone.
dual  = turbo TOGGLE. USE BOTH — train on TRAIN_GPU (default 0) while AI/KAT
        stays on AI_GPU (default 1).

Env vars (optional overrides):
  ENVELOPE_MODE=solo|dual   # default solo
  ENVELOPE_AI_GPU=1
  ENVELOPE_TRAIN_GPU=0      # second card when dual; ignored in solo
  ENVELOPE_TIME_BUDGET=300
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LabConfig:
    mode: str  # solo | dual
    ai_gpu: int
    train_gpu: int
    time_budget_s: float

    @property
    def experiment_gpu(self) -> int:
        """GPU used by train.py for a trial."""
        if self.mode == "dual":
            # turbo: train on the second card; AI card stays available for KAT
            return self.train_gpu
        # solo: dedicated AI card only
        return self.ai_gpu

    @property
    def uses_both_gpus(self) -> bool:
        return self.mode == "dual"


def load_lab_config() -> LabConfig:
    mode = os.environ.get("ENVELOPE_MODE", "solo").strip().lower()
    if mode not in ("solo", "dual"):
        raise ValueError(f"ENVELOPE_MODE must be solo|dual, got {mode!r}")
    ai_gpu = int(os.environ.get("ENVELOPE_AI_GPU", "1"))
    train_gpu = int(os.environ.get("ENVELOPE_TRAIN_GPU", "0"))
    time_budget = float(os.environ.get("ENVELOPE_TIME_BUDGET", "300"))
    return LabConfig(
        mode=mode,
        ai_gpu=ai_gpu,
        train_gpu=train_gpu,
        time_budget_s=time_budget,
    )


def apply_cuda_device(gpu_index: int) -> None:
    """Pin this process to one GPU via CUDA_VISIBLE_DEVICES if not already set."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
