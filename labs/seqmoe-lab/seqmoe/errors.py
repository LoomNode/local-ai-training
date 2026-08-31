"""Fail-closed errors for sequential MoE assembly and memory planning."""

from __future__ import annotations


class SeqMoEError(Exception):
    """Base error for seqmoe-lab."""


class TrunkHashMismatch(SeqMoEError):
    """Expert was trained against a different trunk than the one being assembled."""


class ExpertConfigMismatch(SeqMoEError):
    """Expert checkpoint architecture does not match the trunk."""


class MissingExpertError(SeqMoEError):
    """Assemble did not receive every logical id in ``range(n_experts)``."""


class DuplicateExpertError(SeqMoEError):
    """Two expert files claimed the same logical id."""


class MemoryPlanError(SeqMoEError):
    """Planned VRAM exceeds the envelope. Do not allocate."""
