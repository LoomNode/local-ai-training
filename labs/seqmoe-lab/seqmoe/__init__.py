"""seqmoe-lab: sequential Mixture-of-Experts against a frozen shared trunk."""

from seqmoe.config import (
    DOMAIN_TO_ID,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    resolve_device,
    scientific_config,
    tiny_config,
)
from seqmoe.errors import (
    DuplicateExpertError,
    ExpertConfigMismatch,
    MemoryPlanError,
    MissingExpertError,
    TrunkHashMismatch,
)
from seqmoe.model import ExpertBank, ForwardResult, MLP, Router, SeqMoE

__all__ = [
    "DOMAIN_TO_ID",
    "DuplicateExpertError",
    "ExperimentConfig",
    "ExpertBank",
    "ExpertConfigMismatch",
    "ForwardResult",
    "MLP",
    "MemoryPlanError",
    "MissingExpertError",
    "ModelConfig",
    "Router",
    "SeqMoE",
    "TrainConfig",
    "TrunkHashMismatch",
    "resolve_device",
    "scientific_config",
    "tiny_config",
]
