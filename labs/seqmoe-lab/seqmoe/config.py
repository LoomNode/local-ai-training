"""Model, train, and experiment configuration for sequential MoE."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


DOMAIN_TO_ID = {"code": 0, "math": 1, "fill": 2, "wiki": 2}
ID_TO_DOMAIN = {0: "code", 1: "math", 2: "other"}
SKILL_ORDER = ("code", "math", "fill", "wiki")

MAX_VRAM_MB = 22_000


@dataclass
class ModelConfig:
    vocab_size: int = 256
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 64
    dropout: float = 0.0
    n_experts: int = 2
    top_k: int = 2
    mlp_mult: int = 4

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_experts < 2:
            raise ValueError("n_experts must be >= 2")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError("top_k must be in [1, n_experts]")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if self.mlp_mult < 1:
            raise ValueError("mlp_mult must be >= 1")


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    seq_len: int = 64
    lr: float = 3e-3
    weight_decay: float = 0.01
    pretrain_steps: int = 200
    expert_steps: int = 200
    router_steps: int = 200
    eval_every: int = 50
    log_every: int = 25
    device: str = "cuda:1"
    expert_init: str = "clone"  # clone | random
    expert_corpus: str = "domain"  # domain (code/math) | mix (same stream, seed ensemble)
    balance_coef: float = 0.01
    route_coef: float = 0.0
    max_vram_mb: int = MAX_VRAM_MB
    results_dir: str = "results"
    run_name: str = "poc"
    scientific: bool = False
    eval_batches: int = 8

    def __post_init__(self) -> None:
        if self.expert_init not in ("clone", "random"):
            raise ValueError("expert_init must be 'clone' or 'random'")
        if self.expert_corpus not in ("domain", "mix"):
            raise ValueError("expert_corpus must be 'domain' or 'mix'")
        if self.batch_size < 1 or self.seq_len < 1:
            raise ValueError("batch_size and seq_len must be >= 1")
        if self.max_vram_mb < 1:
            raise ValueError("max_vram_mb must be >= 1")


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return {"model": asdict(self.model), "train": asdict(self.train)}


def resolve_device(device: str) -> str:
    """Map bare ``cuda`` → ``cuda:1`` (dedicated research GPU on this box)."""
    if device == "cuda":
        return "cuda:1"
    return device


def tiny_config(device: str = "cpu") -> ExperimentConfig:
    """CPU/pytest and GPU plumbing. Full sequential smoke is 20/20/20 steps."""
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=256,
            n_embd=32,
            n_head=4,
            n_layer=2,
            block_size=64,
            dropout=0.0,
            n_experts=2,
            top_k=2,
            mlp_mult=4,
        ),
        train=TrainConfig(
            seed=42,
            batch_size=8,
            seq_len=64,
            lr=3e-3,
            weight_decay=0.01,
            pretrain_steps=20,
            expert_steps=20,
            router_steps=20,
            eval_every=10,
            log_every=10,
            device=resolve_device(device),
            expert_init="clone",
            balance_coef=0.01,
            route_coef=0.0,
            max_vram_mb=MAX_VRAM_MB,
            results_dir="results",
            run_name="poc_tiny",
            scientific=False,
            eval_batches=4,
        ),
    )


def tiny_seed_config(device: str = "cpu") -> ExperimentConfig:
    """Same-mix sequential seeds: random FFN inits, one expert at a time. CPU only."""
    del device  # GPU is reserved; this recipe stays on CPU.
    cfg = tiny_config(device="cpu")
    cfg.model.n_experts = 3
    cfg.model.top_k = 2
    cfg.train.expert_init = "random"
    cfg.train.expert_corpus = "mix"
    cfg.train.run_name = "poc_tiny_seeds"
    cfg.train.device = "cpu"
    return cfg


def tiny_skill_config(device: str = "cpu") -> ExperimentConfig:
    """One cloned expert per synth skill (code/math/fill). CPU only."""
    del device
    cfg = tiny_config(device="cpu")
    cfg.model.n_experts = 3
    cfg.model.top_k = 2
    cfg.train.expert_init = "clone"
    cfg.train.expert_corpus = "domain"
    cfg.train.run_name = "poc_tiny_skills"
    cfg.train.device = "cpu"
    return cfg


def scientific_config(device: str = "cuda:1") -> ExperimentConfig:
    """GPU1 scientific run (wiki + code + math)."""
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=256,
            n_embd=256,
            n_head=8,
            n_layer=8,
            block_size=256,
            dropout=0.0,
            n_experts=2,
            top_k=2,
            mlp_mult=4,
        ),
        train=TrainConfig(
            seed=42,
            batch_size=16,
            seq_len=256,
            lr=8e-4,
            weight_decay=0.01,
            pretrain_steps=5000,
            expert_steps=3000,
            router_steps=2000,
            eval_every=200,
            log_every=100,
            device=resolve_device(device),
            expert_init="clone",
            balance_coef=0.01,
            route_coef=0.0,
            max_vram_mb=MAX_VRAM_MB,
            results_dir="results",
            run_name="poc_scientific",
            scientific=True,
            eval_batches=20,
        ),
    )
