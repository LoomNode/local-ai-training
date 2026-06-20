# Ratchet Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible PyTorch feasibility experiment comparing master-weight-free
quinary and septenary ratchet matrices on Tiny Shakespeare.

**Architecture:** A generic eager `DiscreteRatchetLinear` owns integer code and pressure
buffers plus fixed row scales. A tiny GPT uses those layers throughout its dense path, and
an explicit trainer performs backward, ratchet updates, support-parameter optimization,
logging, safe checkpointing, comparison, and plotting.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Datasets, safetensors, Matplotlib, pytest.

---

## Tasks

- [x] Implement and test code initialization, effective-weight lifecycle, pressure bucketing,
  code clicks, saturation handling, statistics, and persistent-state audits.
- [x] Implement and test deterministic character data, paired batch schedules, fixed positions,
  RMSNorm GPT blocks, and all-ratchet dense matrices.
- [x] Implement and test configuration loading, training/evaluation, CSV metrics, safe
  checkpoints, resume equivalence, and failure validation.
- [x] Implement and test dataset, train, compare, plot, and audit CLI commands.
- [ ] Run CPU tests and lint, a synthetic overfit check, a Tiny Shakespeare smoke run, audits,
  plots, documentation checks, and final Git verification.
