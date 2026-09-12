"""Delayed identity: cue then silence, predict the cue."""

from __future__ import annotations

import torch

from flybrain.baselines.gru import matched_gru
from flybrain.data.connectome import synthetic_flylike
from flybrain.data.task import delayed_identity_batch, make_sparse_patterns, sparse_binding_batch
from flybrain.model.brain import FlyLanguageModel
from flybrain.train import eval_delay, train_delay, train_last_step


def test_delayed_identity_puts_cue_first_and_silence_after():
    x, y = delayed_identity_batch(8, n_classes=16, delay=3, seed=0, silence=255)
    assert x.shape == (8, 4)
    assert y.shape == (8,)
    assert torch.equal(x[:, 0], y)
    assert torch.all(x[:, 1:] == 255)
    assert int(y.min()) >= 0 and int(y.max()) < 16


def test_delay_zero_is_immediate_copy():
    x, y = delayed_identity_batch(4, n_classes=8, delay=0, seed=1)
    assert x.shape == (4, 1)
    assert torch.equal(x[:, 0], y)


def test_silence_must_not_collide_with_classes():
    try:
        delayed_identity_batch(2, n_classes=16, delay=1, seed=0, silence=3)
    except ValueError as exc:
        assert "silence" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_delay_loss_is_last_step_only():
    g = synthetic_flylike(n=32, seed=0)
    m = FlyLanguageModel(g, state_dim=4, readout="mean", update="ordered")
    x, y = delayed_identity_batch(4, n_classes=16, delay=2, seed=0)
    logits, _ = m(x)
    assert logits.shape == (4, 3, 256)
    last = logits[:, -1]
    loss = torch.nn.functional.cross_entropy(last, y)
    assert torch.isfinite(loss)


def test_flatten_ordered_decoder_reads_all_readout_states():
    g = synthetic_flylike(n=32, seed=0)
    m = FlyLanguageModel(g, state_dim=4, readout="flatten", update="ordered")
    assert m.decoder.in_features == int(g.readout_idx.numel()) * 4
    x, y = delayed_identity_batch(3, n_classes=16, delay=0, seed=0)
    logits, state = m(x)
    assert logits.shape == (3, 1, 256)
    assert state.shape == (3, 32, 4)


def test_train_delay_reports_accuracy_and_gru_delay0_beats_chance():
    g = synthetic_flylike(n=32, seed=0)
    fly = FlyLanguageModel(g, state_dim=4, readout="mean", update="ordered")
    gru = matched_gru(fly)
    stats = train_delay(
        gru, n_classes=16, delay=0, steps=40, batch=16, lr=0.05, seed=0, eval_batches=4
    )
    assert 0.0 <= stats["val_acc"] <= 1.0
    assert stats["val_acc"] > 1.0 / 16
    held = eval_delay(fly, n_classes=16, delay=0, batch=8, n_batches=2, seed=1)
    assert 0.0 <= held["val_acc"] <= 1.0


def test_sparse_patterns_are_unique_khot_and_overlap():
    patterns = make_sparse_patterns(n_classes=16, n_features=16, k=3, seed=0)
    assert patterns.shape == (16, 3)
    rows = [tuple(sorted(int(x) for x in row.tolist())) for row in patterns]
    assert len(set(rows)) == 16
    counts = torch.bincount(patterns.reshape(-1), minlength=16)
    assert int((counts > 1).sum()) >= 1
    assert patterns.min() >= 0 and patterns.max() < 16


def test_binding_batch_is_a_permutation_of_the_class_pattern():
    patterns = make_sparse_patterns(8, 12, 3, seed=1)
    x, y = sparse_binding_batch(10, patterns, seed=2)
    assert x.shape == (10, 3)
    for i in range(10):
        got = tuple(sorted(int(t) for t in x[i].tolist()))
        want = tuple(sorted(int(t) for t in patterns[int(y[i])].tolist()))
        assert got == want


def test_binding_batch_with_distractors_keeps_true_features():
    patterns = make_sparse_patterns(8, 12, 3, seed=1)
    x, y = sparse_binding_batch(10, patterns, seed=3, n_features=12, n_distractors=2)
    assert x.shape == (10, 5)
    for i in range(10):
        bag = set(int(t) for t in x[i].tolist())
        true = set(int(t) for t in patterns[int(y[i])].tolist())
        assert true.issubset(bag)
        assert len(bag) == 5


def test_train_binding_gru_beats_chance():
    g = synthetic_flylike(n=32, seed=0)
    fly = FlyLanguageModel(g, state_dim=4, readout="flatten", update="ordered")
    gru = matched_gru(fly)
    patterns = make_sparse_patterns(8, 12, 3, seed=0)

    def batch_fn(batch: int, seed: int):
        return sparse_binding_batch(batch, patterns, seed)

    stats = train_last_step(
        gru, batch_fn, steps=50, batch=16, lr=0.05, seed=0, eval_batches=4
    )
    assert stats["val_acc"] > 1.0 / 8
