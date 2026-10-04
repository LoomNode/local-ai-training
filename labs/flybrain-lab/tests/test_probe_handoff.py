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


def _handoff(world, tiny_hybrid, alphas=(0.0, 0.5), limit=0, budget=600, per_file=300):
    from scripts.probe_commons import probe_handoff
    return probe_handoff(tiny_hybrid(), "cpu", world.path, "dev", alphas, 0, 300, budget, per_file, 64, world.train_end, world.tail_start, world.dev_files, limit)


def test_probe_handoff_alone_alpha_zero_and_sequential_head_difference(world, tiny_hybrid):
    res = _handoff(world, tiny_hybrid)
    b = res["bpb"]
    assert b["handoff"]["0.0"] == b["alone"]                                  # alpha = 0 gives exactly the alone numbers
    assert abs(b["alone"] - b["alone_forward"]) < 1e-3                         # the stream API agrees with the plain forward
    assert b["handoff"]["0.5"] != b["alone"] and b["context"] != b["alone"]
    assert res["n_targets"] == 3 and res["n_scored_bytes"] == 690 and res["readers"]["mean_files"] == 2.0
    lowest = res["head_diff"][1]                                               # all_heads(): post head first, then the inner head
    assert lowest["S_rel"] < 1e-4 and lowest["Ksum_rel"] < 1e-4                # the lowest head's keys do not depend on earlier reads
    assert res["head_diff"][0]["S_rel"] > 1e-4                                 # the post head's do, so its tables differ from the sum
    assert res["best_alpha"] in (0.0, 0.5) and res["transfer_best"] == res["transfer"][str(res["best_alpha"])]
    ci = res["ci95"]
    assert set(ci) == {"gain_context", "gain_sequential", "gain_handoff", "transfer", "transfer_best", "transfer_sequential"}
    assert len(ci["gain_context"]) == 2 and set(ci["gain_handoff"]) == {"0.0", "0.5"}
    assert ci["gain_handoff"]["0.0"] == [0.0, 0.0] and res["gain_handoff"]["0.0"] == 0.0     # alpha = 0 changes nothing, in every resample
    assert abs(res["gain_context"] - (b["alone"] - b["context"])) < 1e-12 and abs(res["gain_sequential"] - (b["alone"] - b["sequential"])) < 1e-12


def test_readers_are_cut_by_file_and_by_budget(world, tiny_hybrid):
    res = _handoff(world, tiny_hybrid, budget=1000, per_file=300)
    assert [t["readers"] for t in res["targets"]] == [4, 4, 4] and [t["reader_bytes"] for t in res["targets"]] == [1000, 1000, 1000]
    one = _handoff(world, tiny_hybrid, budget=250, per_file=300)
    assert [t["readers"] for t in one["targets"]] == [1, 1, 1] and [t["reader_bytes"] for t in one["targets"]] == [250, 250, 250]


def test_handoff_probe_is_deterministic_and_json_serialisable(world, tiny_hybrid):
    import json
    a, b = _handoff(world, tiny_hybrid), _handoff(world, tiny_hybrid)
    assert json.dumps(a) == json.dumps(b)


def test_main_handoff_scores_the_given_alpha_or_all_of_them(world, tiny_checkpoint, tmp_path, monkeypatch):
    import json

    import pytest
    import scripts.probe_commons as pc
    monkeypatch.setattr(pc, "RESULTS", tmp_path)
    common = ["--mode", "handoff", "--ckpt", str(tiny_checkpoint()), "--split", "dev", "--corpus", str(world.path), "--device", "cpu", "--n-scored", "300",
              "--budget", "600", "--per-file", "300", "--train-end", str(world.train_end), "--tail-start", str(world.tail_start), "--dev-files", "3"]
    pc.main(["--run-name", "ho_all", *common])
    res = json.loads((tmp_path / "ho_all.json").read_text())
    assert {"run_name", "mode", "ckpt", "arm", "chunk", "args", "split", "seed", "alphas", "budget", "per_file", "n_targets", "n_scored_bytes", "bpb",
            "best_alpha", "gain_context", "transfer", "transfer_best", "transfer_sequential", "head_diff", "readers", "targets"} <= set(res)
    assert res["alphas"] == [0.03, 0.1, 0.3, 1.0] and set(res["bpb"]["handoff"]) == {"0.03", "0.1", "0.3", "1.0"} and res["chunk"] == 0
    pc.main(["--run-name", "ho_one", "--alpha", str(res["best_alpha"]), *common])
    one = json.loads((tmp_path / "ho_one.json").read_text())
    assert one["alphas"] == [res["best_alpha"]] and one["best_alpha"] == res["best_alpha"]
    assert one["bpb"]["handoff"][str(res["best_alpha"])] == res["bpb"]["handoff"][str(res["best_alpha"])]
    with pytest.raises(SystemExit):
        pc.main(["--run-name", "ho_one", *common])


def test_render_handoff_prints_every_condition_and_the_head_table(world, tiny_hybrid):
    from scripts.probe_commons import render_handoff
    text = render_handoff({"run_name": "tiny", **_handoff(world, tiny_hybrid)})
    assert "| condition | bits/byte | gain vs alone (95 % interval) | transfer ratio (95 % interval) |" in text and "| sequential reader |" in text and "| handoff alpha 0.5 |" in text
    assert "| head (all_heads() order) | S relative difference | Ksum relative difference |" in text


def _spy_head_chunks(pc, monkeypatch):
    """load_checkpoint wrapped so a test sees, per call, the chunk main asked for and the chunk every head ended up with."""
    loaded, real = [], pc.load_checkpoint

    def spy(path, device, chunk):
        model, ck = real(path, device, chunk)
        loaded.append((chunk, [h.chunk for h in model.all_heads()]))
        return model, ck
    monkeypatch.setattr(pc, "load_checkpoint", spy)
    return loaded


def test_main_handoff_without_chunk_is_chunk_zero_and_keeps_the_checkpoints_own(world, tiny_checkpoint, tmp_path, monkeypatch):
    import json

    import scripts.probe_commons as pc
    monkeypatch.setattr(pc, "RESULTS", tmp_path)
    loaded = _spy_head_chunks(pc, monkeypatch)
    common = ["--mode", "handoff", "--ckpt", str(tiny_checkpoint()), "--split", "dev", "--corpus", str(world.path), "--device", "cpu", "--n-scored", "300",
              "--budget", "600", "--per-file", "300", "--train-end", str(world.train_end), "--tail-start", str(world.tail_start), "--dev-files", "3"]
    pc.main(["--run-name", "ho_chunk0", "--chunk", "0", *common])
    pc.main(["--run-name", "ho_nochunk", *common])
    zero, none = (json.loads((tmp_path / f"{n}.json").read_text()) for n in ("ho_chunk0", "ho_nochunk"))
    assert loaded == [(0, [8, 8]), (0, [8, 8])]                                 # the tiny checkpoint was trained with chunk 8: no head got the context default
    assert zero["chunk"] == none["chunk"] == 0 and zero["args"]["chunk"] == 0 and none["args"]["chunk"] is None     # the chunk in force, and the flag as given
    same = lambda r: {k: v for k, v in r.items() if k not in ("run_name", "args")}
    assert same(zero) == same(none)                                             # the same numbers


def test_main_context_without_chunk_still_runs_and_records_4096(world, tiny_checkpoint, tmp_path, monkeypatch):
    import json

    import scripts.probe_commons as pc
    monkeypatch.setattr(pc, "RESULTS", tmp_path)
    loaded = _spy_head_chunks(pc, monkeypatch)
    pc.main(["--mode", "context", "--ckpt", str(tiny_checkpoint()), "--run-name", "ctx_nochunk", "--split", "dev", "--corpus", str(world.path), "--device", "cpu",
             "--ps", "128,256", "--n-scored", "300", "--train-end", str(world.train_end), "--tail-start", str(world.tail_start), "--dev-files", "3"])
    res = json.loads((tmp_path / "ctx_nochunk.json").read_text())
    assert res["chunk"] == 4096 and res["mode"] == "context"
    assert loaded == [(4096, [4096, 4096])]                                     # the heads really ran at the context default, not only the record


def test_handoff_limit_scores_exactly_the_first_target_of_the_full_run(world, tiny_hybrid):
    full, one = _handoff(world, tiny_hybrid), _handoff(world, tiny_hybrid, limit=1)
    assert full["n_targets"] == 3 and one["n_targets"] == 1 and len(one["targets"]) == 1
    assert one["targets"][0]["index"] == full["targets"][0]["index"]
    assert one["targets"][0]["bpb"] == full["targets"][0]["bpb"]               # the same floats, not close ones


def test_render_handoff_prints_each_number_from_the_result_it_was_given(world, tiny_hybrid):
    import copy

    from scripts.probe_commons import render_handoff
    res = {"run_name": "tiny", **_handoff(world, tiny_hybrid)}
    base, new = render_handoff(res), copy.deepcopy(res)
    new["bpb"]["sequential"] = 7.7777
    new["head_diff"][0].update(S_rel=0.0123, Ksum_rel=0.00456)                 # two different values, so a swapped column would show
    text = render_handoff(new)
    assert "| sequential reader | 7.7777 |" in text and "| sequential reader | 7.7777 |" not in base
    assert "| 0 | 1.23e-02 | 4.56e-03 |" in text and "| 0 | 1.23e-02 | 4.56e-03 |" not in base
    other = res["head_diff"][1]
    assert f"| 1 | {other['S_rel']:.2e} | {other['Ksum_rel']:.2e} |" in text      # the other head's row still comes from its own entry
    old = base.splitlines()
    assert len([line for line in text.splitlines() if line not in old]) == 2   # only the two edited rows changed
