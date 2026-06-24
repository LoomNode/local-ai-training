"""Validated experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import tomllib

from .model import ModelConfig


@dataclass(frozen=True)
class ExperimentConfig:
    block_size: int = 128
    batch_size: int = 32
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    steps: int = 2_000
    eval_interval: int = 100
    eval_batches: int = 20
    support_learning_rate: float = 3e-4
    pressure_threshold: int = 8
    bucket_low: float = 0.5
    bucket_high: float = 1.5
    trainable_scale: bool = False
    rms_ema_beta: float = 0.0
    compile_update: bool = False
    matmul_mode: Literal["fp32", "bf16", "int8"] = "fp32"
    seeds: tuple[int, ...] = (1337, 1338, 1339)
    device: str = "auto"
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            self.block_size,
            self.batch_size,
            self.n_layer,
            self.n_head,
            self.n_embd,
            self.steps,
            self.eval_interval,
            self.eval_batches,
            self.pressure_threshold,
        )
        if min(integer_fields) <= 0:
            raise ValueError("all experiment dimensions and intervals must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.support_learning_rate <= 0:
            raise ValueError("support_learning_rate must be positive")
        if not 0 <= self.bucket_low < self.bucket_high:
            raise ValueError("bucket thresholds must satisfy 0 <= low < high")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.matmul_mode not in {"fp32", "bf16", "int8"}:
            raise ValueError("matmul_mode must be fp32, bf16, or int8")

    @classmethod
    def from_toml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("rb") as handle:
            document = tomllib.load(handle)
        allowed = {
            "model": {"block_size", "n_layer", "n_head", "n_embd", "dropout"},
            "ratchet": {
                "pressure_threshold",
                "bucket_low",
                "bucket_high",
                "trainable_scale",
                "rms_ema_beta",
                "compile_update",
            },
            "training": {
                "batch_size",
                "steps",
                "eval_interval",
                "eval_batches",
                "support_learning_rate",
                "seeds",
                "device",
                "matmul_mode",
                "gradient_checkpointing",
            },
        }
        unknown_sections = set(document) - set(allowed)
        if unknown_sections:
            raise ValueError(f"unknown config sections: {sorted(unknown_sections)}")
        values: dict[str, Any] = {}
        for section, section_values in document.items():
            unknown_keys = set(section_values) - allowed[section]
            if unknown_keys:
                raise ValueError(f"unknown keys in [{section}]: {sorted(unknown_keys)}")
            values.update(section_values)
        if "seeds" in values:
            values["seeds"] = tuple(values["seeds"])
        return cls(**values)

    def model_config(self, *, vocab_size: int) -> ModelConfig:
        return ModelConfig(
            vocab_size=vocab_size,
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
            pressure_threshold=self.pressure_threshold,
            bucket_low=self.bucket_low,
            bucket_high=self.bucket_high,
            trainable_scale=self.trainable_scale,
            rms_ema_beta=self.rms_ema_beta,
            compile_update=self.compile_update,
            matmul_mode=self.matmul_mode,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        return result
