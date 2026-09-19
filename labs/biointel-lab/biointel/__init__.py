"""Shared-rule neuron/synapse lab: capability per trainable bit."""

from biointel.accounting import ResourceReport, count_resources
from biointel.graphs import SparseGraph, degree_matched_shuffle, erdos_renyi, in_out_degree
from biointel.ladder import LADDER_ARMS, run_ladder
from biointel.network import SharedRuleNet, unique_gru
from biointel.plasticity import (
    derangement,
    evaluate_accuracy_perm,
    inner_adapt,
    local_adapt,
    program_tensors,
    reset_adapters,
)
from biointel.task import delayed_identity_batch, evaluate_accuracy, last_cue_batch, train_delayed_identity, train_task

__all__ = [
    "LADDER_ARMS",
    "ResourceReport",
    "SharedRuleNet",
    "SparseGraph",
    "count_resources",
    "degree_matched_shuffle",
    "delayed_identity_batch",
    "derangement",
    "erdos_renyi",
    "evaluate_accuracy",
    "evaluate_accuracy_perm",
    "in_out_degree",
    "inner_adapt",
    "last_cue_batch",
    "local_adapt",
    "program_tensors",
    "reset_adapters",
    "run_ladder",
    "train_delayed_identity",
    "train_task",
    "unique_gru",
]
