"""Coincidence mixer: sparse expand, set inject, thin head. No connectome."""

from __future__ import annotations

import torch

from flybrain.baselines.gru import matched_gru
from flybrain.baselines.mlp import matched_mlp
from flybrain.data.task import make_sparse_patterns, sparse_binding_batch
from flybrain.model.mixer import CoincidenceMixer, sparse_expand_matrix
from flybrain.train import train_last_step


def test_expand_rows_have_unique_claws():
    w = sparse_expand_matrix(80, 12, 4, seed=0)
    assert w.shape == (80, 12)
    assert torch.all(w.sum(dim=1) == 4)
    assert set(w.unique().tolist()) <= {0.0, 1.0}


def test_mixer_head_is_thin_not_flatten64():
    m = CoincidenceMixer(n_features=12, n_classes=16, n_expand=80, n_read=16)
    assert m.decoder.in_features == 16
    assert m.read.in_features == 80
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == 80 * 16 + 16 + 16 * 16 + 16


def test_bag_duplicates_do_not_inflate_counts():
    m = CoincidenceMixer(n_features=12, n_classes=16, theta=0.0, top_k=80, seed=0)
    a = torch.tensor([[0, 1, 2]])
    b = torch.tensor([[0, 0, 0, 1, 1, 2]])
    with torch.no_grad():
        za = m.encode(a)
        zb = m.encode(b)
    assert torch.allclose(za, zb)


def test_expand_is_not_the_class_pattern_matrix():
    patterns = make_sparse_patterns(16, 12, 3, seed=0)
    m = CoincidenceMixer(n_features=12, n_classes=16, n_expand=80, claws=4, seed=0)
    rows = [tuple(sorted(int(x) for x in row.tolist())) for row in patterns]
    claws = []
    for i in range(m.n_expand):
        idx = (m.expand[i] > 0).nonzero(as_tuple=False).view(-1).tolist()
        claws.append(tuple(sorted(int(j) for j in idx)))
    aligned = sum(1 for i, row in enumerate(rows) if i < len(claws) and claws[i] == row)
    assert aligned == 0


def test_mixer_trains_above_chance_on_tiny_binding():
    patterns = make_sparse_patterns(8, 12, 3, seed=1)
    m = CoincidenceMixer(n_features=12, n_classes=8, n_expand=40, claws=3, top_k=8, n_read=8, theta=1.5, seed=1)

    def batch_fn(batch: int, seed: int):
        return sparse_binding_batch(batch, patterns, seed, n_features=12, n_distractors=2)

    stats = train_last_step(m, batch_fn, steps=40, batch=16, lr=0.05, seed=0, eval_batches=4)
    assert stats["val_acc"] > 1.0 / 8
    mlp = matched_mlp(m)
    n_m = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_p = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    assert abs(n_p - n_m) / n_m < 0.35
    gru = matched_gru(m, vocab_size=8)
    n_g = sum(p.numel() for p in gru.parameters() if p.requires_grad)
    assert abs(n_g - n_m) / n_m < 0.35
