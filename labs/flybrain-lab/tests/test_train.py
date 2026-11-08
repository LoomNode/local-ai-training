"""Plumbing: next-byte training runs on all three arms and loss is finite."""

from __future__ import annotations

import torch

from flybrain.baselines.gru import matched_gru
from flybrain.data.connectome import degree_matched_shuffle, synthetic_flylike
from flybrain.data.text import bytes_tensor
from flybrain.model.brain import FlyLanguageModel
from flybrain.train import bits_per_byte, eval_bpb, train_steps


def test_bits_per_byte_is_ce_over_ln2():
    ce = torch.tensor(0.693147)  # ln 2
    assert abs(float(bits_per_byte(ce)) - 1.0) < 1e-4


def test_three_arms_train_and_fly_loss_drops():
    g = synthetic_flylike(n=32, seed=0)
    sh = degree_matched_shuffle(g, seed=1)
    fly = FlyLanguageModel(g, state_dim=4)
    rnd = FlyLanguageModel(sh, state_dim=4)
    gru = matched_gru(fly)
    data = bytes_tensor("abcabcabcabcabcabcabcabc")
    fly_stats = train_steps(fly, data, steps=25, seq_len=12, batch=4, lr=0.02, seed=0)
    rnd_stats = train_steps(rnd, data, steps=8, seq_len=12, batch=4, lr=0.02, seed=0)
    gru_stats = train_steps(gru, data, steps=8, seq_len=12, batch=4, lr=0.02, seed=0)
    assert fly_stats["final_loss"] < fly_stats["initial_loss"]
    assert fly_stats["final_bpb"] < fly_stats["initial_bpb"]
    assert rnd_stats["final_loss"] > 0
    assert gru_stats["final_loss"] > 0
    assert fly_stats["trainable_scalars"] == rnd_stats["trainable_scalars"]


def test_universal_cell_train_steps_drop_loss():
    g = synthetic_flylike(n=32, seed=0)
    m = FlyLanguageModel(g, state_dim=4, readout="mean", cell="universal")
    data = bytes_tensor("abcabcabcabcabcabcabcabc")
    stats = train_steps(m, data, steps=25, seq_len=12, batch=4, lr=0.02, seed=0)
    assert stats["final_loss"] < stats["initial_loss"]


def test_ordered_update_train_steps_drop_loss():
    g = synthetic_flylike(n=32, seed=0)
    m = FlyLanguageModel(g, state_dim=4, readout="mean", update="ordered")
    data = bytes_tensor("abcabcabcabcabcabcabcabc")
    stats = train_steps(m, data, steps=25, seq_len=12, batch=4, lr=0.02, seed=0)
    assert stats["final_loss"] < stats["initial_loss"]


def test_sensory_readout_skip_train_steps_drop_loss():
    g = synthetic_flylike(n=32, seed=0)
    m = FlyLanguageModel(g, state_dim=4, readout="mean", skip="sensory_readout")
    data = bytes_tensor("abcabcabcabcabcabcabcabc")
    stats = train_steps(m, data, steps=25, seq_len=12, batch=4, lr=0.02, seed=0)
    assert stats["final_loss"] < stats["initial_loss"]


def test_sensory_decode_train_steps_drop_loss():
    g = synthetic_flylike(n=32, seed=0)
    m = FlyLanguageModel(g, state_dim=4, readout="mean", decode_pop="sensory")
    data = bytes_tensor("abcabcabcabcabcabcabcabc")
    stats = train_steps(m, data, steps=25, seq_len=12, batch=4, lr=0.02, seed=0)
    assert stats["final_loss"] < stats["initial_loss"]


def test_unique_cell_train_steps_drop_loss():
    g = synthetic_flylike(n=32, seed=0)
    m = FlyLanguageModel(g, state_dim=4, readout="mean", cell="unique")
    data = bytes_tensor("abcabcabcabcabcabcabcabc")
    stats = train_steps(m, data, steps=25, seq_len=12, batch=4, lr=0.02, seed=0)
    assert stats["final_loss"] < stats["initial_loss"]


def test_vector_message_fly_loss_drops():
    g = synthetic_flylike(n=32, seed=0)
    m = FlyLanguageModel(g, state_dim=4, readout="mean", message="vector")
    data = bytes_tensor("abcabcabcabcabcabcabcabc")
    stats = train_steps(m, data, steps=25, seq_len=12, batch=4, lr=0.02, seed=0)
    assert stats["final_loss"] < stats["initial_loss"]


def test_vector2_and_projected_train_steps_drop_loss():
    g = synthetic_flylike(n=32, seed=0)
    data = bytes_tensor("abcabcabcabcabcabcabcabc")
    for kind in ("vector2", "projected"):
        m = FlyLanguageModel(g, state_dim=4, readout="mean", message=kind)
        stats = train_steps(m, data, steps=25, seq_len=12, batch=4, lr=0.02, seed=0)
        assert stats["final_loss"] < stats["initial_loss"], kind


def test_eval_bpb_does_not_update_theta():
    g = synthetic_flylike(n=24, seed=2)
    m = FlyLanguageModel(g, state_dim=4)
    data = bytes_tensor("hello world hello world hello")
    before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    bpb = eval_bpb(m, data, seq_len=8, batch=2, n_batches=3, seed=0)
    assert bpb > 0
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k])


def test_train_steps_logs_heldout_curve():
    g = synthetic_flylike(n=24, seed=3)
    m = FlyLanguageModel(g, state_dim=4)
    train = bytes_tensor("abcabcabcabcabcabcabcabc")
    val = bytes_tensor("xyzxyzxyzxyzxyzxyzxyzxyz")
    stats = train_steps(
        m, train, steps=6, seq_len=8, batch=2, lr=0.02, seed=0, val_data=val, log_every=3
    )
    assert "val_bpb" in stats
    assert stats["val_bpb"] > 0
    assert len(stats["curve"]) >= 2
    assert set(stats["curve"][0]) >= {"step", "train_bpb", "val_bpb"}
