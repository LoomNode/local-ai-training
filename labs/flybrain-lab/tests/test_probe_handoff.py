"""scripts/probe_commons.py --mode handoff. Needs the agent-state plan: flybrain.model.state, GRUHybrid.init_state / stream, flybrain.group."""

from __future__ import annotations

import numpy as np
import torch

HEAD = b"\n// ==== k/x.c ====\n"
TEXT = b"static int probe_device(struct device *dev) { return -ENODEV; }\n" * 4


def test_transfer_ratio_needs_a_context_gain_and_best_alpha_breaks_ties_low():
    import pytest

    from scripts.probe_commons import MIN_GAIN, best_alpha, transfer_ratio
    assert MIN_GAIN == 0.01
    assert transfer_ratio(1.0, 0.8, 0.6) == pytest.approx(0.5)
    assert transfer_ratio(1.0, 1.05, 0.6) == pytest.approx(-0.125)           # a handoff that hurts is reported, not clipped
    assert transfer_ratio(1.0, 0.9, 0.995) is None                           # context barely helps: the ratio would be noise
    assert best_alpha({0.03: 0.9, 0.1: 0.8, 0.3: 0.8, 1.0: 0.95}) == 0.1


def test_handoff_with_alpha_zero_equals_alone_exactly_and_a_real_alpha_changes_the_scores(tiny_hybrid):
    from scripts.probe_commons import handoff_target
    m = tiny_hybrid()
    stream = HEAD + TEXT
    out = handoff_target(m, "cpu", stream, len(HEAD), 120, [stream], (0.0, 1.0), seg=64)   # the reader reads the very bytes that are scored
    assert len(out["alone"]) == 120 and np.isfinite(out["alone"]).all()
    assert np.array_equal(out["handoff"][0.0], out["alone"])                               # alpha = 0 adds nothing: identical, not close
    assert not np.allclose(out["handoff"][1.0], out["alone"], atol=1e-4)
    assert not np.allclose(out["sequential"], out["alone"], atol=1e-4)


def test_add_commons_matches_merging_all_reader_states_at_once(tiny_hybrid):
    from flybrain.group import merge, read
    from scripts.probe_commons import add_commons, to_tokens
    m = tiny_hybrid()
    with torch.no_grad():
        states = [read(m, to_tokens(b, "cpu"))[1] for b in (b"alpha beta gamma delta " * 3, b"gamma delta epsilon " * 4, b"zeta eta theta " * 5)]
        whole, acc = merge(states), None
        for s in states:
            c = merge(s)
            acc = c if acc is None else add_commons(acc, c)
    assert len(whole.S) == len(acc.S) == 2
    for x, y in zip(whole.S + whole.Ksum, acc.S + acc.Ksum):
        assert torch.allclose(x, y, atol=1e-6)


def test_handoff_effects_are_gains_and_ratios_and_their_intervals_follow_the_targets():
    from scripts.probe_commons import bootstrap_handoff, handoff_effects
    same = {"alone": np.array([[20.0, 10]] * 3), "context": np.array([[15.0, 10]] * 3), "sequential": np.array([[18.0, 10]] * 3), "handoff:0.5": np.array([[17.0, 10]] * 3)}
    e = handoff_effects(same, (0.5,))
    assert abs(e["gain_context"] - 0.5) < 1e-12 and abs(e["gain_handoff:0.5"] - 0.3) < 1e-12 and abs(e["gain_sequential"] - 0.2) < 1e-12
    assert abs(e["transfer:0.5"] - 0.6) < 1e-12 and abs(e["transfer_sequential"] - 0.4) < 1e-12
    ci = bootstrap_handoff(same, (0.5,), 50, 1)
    assert all(abs(ci[k][0] - e[k]) < 1e-9 and abs(ci[k][1] - e[k]) < 1e-9 for k in e)       # identical targets: the interval is the point
    mixed = {"alone": np.array([[20.0, 10], [10.0, 10], [30.0, 10]]), "context": np.array([[15.0, 10], [9.0, 10], [20.0, 10]]),
             "sequential": np.array([[18.0, 10], [9.5, 10], [28.0, 10]]), "handoff:0.5": np.array([[17.0, 10], [9.5, 10], [24.0, 10]])}
    a = bootstrap_handoff(mixed, (0.5,), 100, 1)
    assert a == bootstrap_handoff(mixed, (0.5,), 100, 1) and a["gain_context"][0] < a["gain_context"][1]
    flat = {k: v.copy() for k, v in same.items()}; flat["context"] = flat["alone"].copy()
    assert handoff_effects(flat, (0.5,))["transfer:0.5"] is None and bootstrap_handoff(flat, (0.5,), 20, 1)["transfer:0.5"] is None   # no context gain: no ratio


READER_A = b"\n// ==== k/y.c ====\n" + b"void zeta_flush(struct zeta *z) { z->pending = 0; wmb(); }\n" * 3
READER_B = b"\n// ==== k/z.c ====\n" + b"int omega_open(struct inode *i, struct file *f) { return 0; }\n" * 3


def test_handoff_target_is_deterministic_and_leaves_the_model_unchanged(tiny_hybrid):
    from scripts.probe_commons import handoff_target, to_tokens
    m = tiny_hybrid()
    stream, alphas = HEAD + TEXT, (0.3, 1.0)
    x = to_tokens(stream, "cpu")
    weights = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        plain = m(x)[0]
    one, two = (handoff_target(m, "cpu", stream, len(HEAD), 120, [READER_A, READER_B], alphas, seg=64) for _ in range(2))
    assert np.array_equal(one["alone"], two["alone"]) and np.array_equal(one["sequential"], two["sequential"])      # bit-equal, not close
    assert all(np.array_equal(one["handoff"][a], two["handoff"][a]) for a in alphas) and one["head_diff"] == two["head_diff"]
    now = m.state_dict()
    assert not m.training and weights.keys() == now.keys() and all(torch.equal(w, now[k]) for k, w in weights.items())   # no weight or buffer moved
    with torch.no_grad():
        assert torch.equal(m(x)[0], plain)                                                                          # the plain forward scores as before


def test_alone_row_is_the_plain_forward_score_of_the_same_bytes(tiny_hybrid):
    from scripts.probe_commons import handoff_target, score_with_prefix
    m = tiny_hybrid()
    stream = HEAD + TEXT
    for n_scored in (120, len(TEXT)):                                         # a window inside the body, and the whole body up to the last byte
        out = handoff_target(m, "cpu", stream, len(HEAD), n_scored, [READER_A], (1.0,), seg=64)
        plain = score_with_prefix(m, b"", stream, len(HEAD), n_scored, "cpu")
        assert len(out["alone"]) == len(plain) == n_scored
        assert np.abs(out["alone"] - plain).max() < 1e-4                      # the stream API and the plain forward score the same bytes at the same positions


def test_handoff_does_not_depend_on_the_order_of_the_readers(tiny_hybrid):
    from scripts.probe_commons import handoff_target
    m = tiny_hybrid()
    stream, alphas = HEAD + TEXT, (0.3, 1.0)
    ab = handoff_target(m, "cpu", stream, len(HEAD), 120, [READER_A, READER_B], alphas, seg=64)
    ba = handoff_target(m, "cpu", stream, len(HEAD), 120, [READER_B, READER_A], alphas, seg=64)
    for a in alphas:
        assert np.abs(ab["handoff"][a] - ba["handoff"][a]).max() < 1e-5       # a sum does not depend on order; "sequential" may differ, and is not compared
    assert not np.allclose(ab["handoff"][1.0], ab["alone"], atol=1e-4)        # the tables of the readers do change the scores


def test_handoff_without_readers_is_a_clear_error(tiny_hybrid):
    import pytest

    from scripts.probe_commons import handoff_target
    with pytest.raises(ValueError, match="handoff needs at least one reader"):
        handoff_target(tiny_hybrid(), "cpu", HEAD + TEXT, len(HEAD), 120, [], (1.0,), seg=64)
