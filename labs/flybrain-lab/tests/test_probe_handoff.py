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


def test_pooling_is_a_ratio_of_sums_over_targets_not_a_mean_of_ratios():
    from scripts.probe_commons import handoff_effects
    arr = {"alone": np.array([[20.0, 10], [100.0, 100]]), "context": np.array([[10.0, 10], [80.0, 100]]),
           "sequential": np.array([[14.0, 10], [96.0, 100]]), "handoff:0.5": np.array([[15.0, 10], [98.0, 100]])}
    e = handoff_effects(arr, (0.5,))                                         # targets of 10 and 100 scored bytes: pooled bits/byte 120/110, 90/110, 110/110, 113/110
    assert abs(e["gain_context"] - 30 / 110) < 1e-12 and abs(e["gain_sequential"] - 10 / 110) < 1e-12 and abs(e["gain_handoff:0.5"] - 7 / 110) < 1e-12
    assert abs(e["transfer:0.5"] - 7 / 30) < 1e-12 and abs(e["transfer_sequential"] - 10 / 30) < 1e-12   # not the mean of the targets' own ratios, 0.3 and 0.4


def test_a_row_equal_to_alone_has_a_zero_gain_in_every_resample():
    from scripts.probe_commons import bootstrap_handoff
    alone = np.array([[20.0, 10], [10.0, 10], [30.0, 10], [7.0, 5]])
    arr = {"alone": alone, "context": alone * [0.5, 1], "sequential": alone * [0.9, 1], "handoff:0.5": alone.copy()}
    ci = bootstrap_handoff(arr, (0.5,), 60, 3)
    assert ci["gain_handoff:0.5"] == [0.0, 0.0]                              # the same targets are drawn for every row, so alone - alone is exactly 0 in each resample
    assert ci["gain_sequential"][0] < ci["gain_sequential"][1]               # while a row that differs from alone by another amount per target does vary


def test_best_alpha_breaks_ties_to_the_smaller_alpha_whatever_the_order_of_the_dict():
    from scripts.probe_commons import best_alpha
    assert best_alpha({1.0: 0.95, 0.3: 0.8, 0.1: 0.8, 0.03: 0.9}) == 0.1     # the larger of the tied alphas comes first here: breaking ties by position would pick 0.3
    assert best_alpha({0.3: 0.5, 0.1: 0.5, 0.03: 0.5}) == 0.03               # every alpha tied, in descending order


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


def test_head_diff_adds_up_only_for_the_lowest_head_and_one_reader_makes_sequential_the_alpha_one_handoff(tiny_hybrid):
    from scripts.probe_commons import handoff_target
    m = tiny_hybrid()
    two = handoff_target(m, "cpu", HEAD + TEXT, len(HEAD), 120, [READER_A, READER_B], (1.0,), seg=64)
    (post_s, post_k), (low_s, low_k) = two["head_diff"]       # all_heads() order: post head first, then inner heads bottom-up
    assert max(low_s, low_k) < 1e-4                           # the lowest head's keys do not depend on earlier reads: its tables just add up
    assert min(post_s, post_k) > 1e-3                         # the post head's keys do
    one = handoff_target(m, "cpu", HEAD + TEXT, len(HEAD), 120, [READER_A], (1.0,), seg=64)
    assert np.array_equal(one["sequential"], one["handoff"][1.0]) and one["head_diff"] == [(0.0, 0.0)] * 2


def test_only_the_pooled_tables_are_alive_while_the_target_is_scored(tiny_hybrid):
    import gc

    from flybrain.group import Commons
    from scripts.probe_commons import handoff_target

    def n_commons():
        gc.collect()
        return sum(type(o) is Commons for o in gc.get_objects())
    m, stream = tiny_hybrid(), HEAD + TEXT
    base, alive, real = n_commons(), [], m.stream
    def spy(tokens, state):
        if tokens.size(1) == len(stream):                                   # a scoring call: the readers are read 64 bytes at a time
            alive.append(n_commons() - base)
        return real(tokens, state)
    m.stream = spy
    handoff_target(m, "cpu", stream, len(HEAD), 120, [READER_A, READER_B], (0.3, 1.0), seg=64)
    assert alive == [1] * 4                                                 # alone, two alphas, sequential: the pooled tables, not also the last reader's


def test_a_readers_own_tables_are_freed_once_they_are_in_the_running_sum(tiny_hybrid, monkeypatch):
    import gc

    import flybrain.group as group
    from scripts.probe_commons import handoff_target

    def n_commons():
        gc.collect()
        return sum(type(o) is group.Commons for o in gc.get_objects())
    base, alive, real = n_commons(), [], group.read

    def spy(model, streams, state=None, seg=2048):                          # every read: a reader by a fresh agent, then the same reader by the sequential agent
        alive.append(n_commons() - base)
        return real(model, streams, state, seg)
    monkeypatch.setattr(group, "read", spy)
    handoff_target(tiny_hybrid(), "cpu", HEAD + TEXT, len(HEAD), 120, [READER_A, READER_B, READER_A, READER_B], (1.0,), seg=64)
    assert alive == [0] + [1] * 7                                           # no merged tables before the first read, then the running sum alone: not also the last reader's own


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


def _by_hand(world, i, budget, per_file, n_scored=300):
    """What the driver must give handoff_target for target i, rebuilt from the corpus bytes without the driver's helpers: the readers (same-directory files that end inside
    the training range and differ in content from the target, in path order, the first per_file bytes of each, budget bytes in all) and the target's stream."""
    data, e = world.path.read_bytes(), world.table[i]
    related = sorted((j for j, f in enumerate(world.table) if j != i and f.dirname == e.dirname and f.end <= world.train_end and world.hashes[j] != world.hashes[i]),
                     key=lambda j: world.table[j].path)
    readers, have = [], 0
    for j in related:
        f = world.table[j]
        n = min(budget - have, per_file, f.end - f.start)
        if n <= 0:
            break
        readers.append(data[f.start : f.start + n])
        have += n
    scored = min(n_scored, e.end - e.body)
    return readers, data[e.start : e.body + scored], e.body - e.start, scored


def _driver_equals_its_building_blocks_on_hand_built_readers(world, tiny_hybrid, monkeypatch, budget, per_file):
    import pytest
    import scripts.probe_commons as pc
    monkeypatch.setattr(pc, "MIN_GAIN", -100.0)                                 # a ratio at every row, so a wrong alpha or a wrong row shows
    alphas = (0.5, 0.0, 1.0)                                                    # the best alpha is neither the first nor the last one
    res = pc.probe_handoff(tiny_hybrid(), "cpu", world.path, "dev", alphas, 7, 300, budget, per_file, 64, world.train_end, world.tail_start, 6, 0)   # six dev targets, in two directories
    model = tiny_hybrid()                                                       # (with three, every seed gives the same bootstrap interval)
    names = ["alone", "context", "alone_forward", "sequential", *(f"handoff:{a}" for a in alphas)]
    bits, heads, counts, sizes = {k: [] for k in names}, [], [], []
    for rec in res["targets"]:
        readers, stream, body_start, scored = _by_hand(world, rec["index"], budget, per_file)
        out = pc.handoff_target(model, "cpu", stream, body_start, scored, readers, alphas, 64)
        row = {"alone": out["alone"], "sequential": out["sequential"], **{f"handoff:{a}": out["handoff"][a] for a in alphas},
               "context": pc.score_with_prefix(model, b"".join(readers), stream, body_start, scored, "cpu"),         # a plain forward over the reader bytes, then the target
               "alone_forward": pc.score_with_prefix(model, b"", stream, body_start, scored, "cpu")}
        for k in names:
            bits[k].append(row[k])
        for k in ("alone", "context", "alone_forward", "sequential"):
            assert rec["bpb"][k] == float(row[k].mean()), (rec["path"], k)                                           # equal, not close: the same computation on the same bytes
        assert all(rec["bpb"]["handoff"][str(a)] == float(row[f"handoff:{a}"].mean()) for a in alphas), rec["path"]
        assert (rec["readers"], rec["reader_bytes"], rec["n_scored"]) == (len(readers), sum(map(len, readers)), scored), rec["path"]
        heads.append(out["head_diff"])
        counts.append(len(readers))
        sizes.append(sum(map(len, readers)))
    assert res["n_targets"] == 6 and res["readers"] == {"mean_files": pytest.approx(np.mean(counts)), "mean_bytes": pytest.approx(np.mean(sizes))}     # the mean, not the median
    arr = {k: np.array([(float(b.sum()), len(b)) for b in v]) for k, v in bits.items()}      # [target, (sum of bits, count)]
    pooled = {k: a[:, 0].sum() / a[:, 1].sum() for k, a in arr.items()}
    for k in ("alone", "context", "alone_forward", "sequential"):
        assert res["bpb"][k] == pytest.approx(pooled[k], rel=1e-12), k
    for a in alphas:
        assert res["bpb"]["handoff"][str(a)] == pytest.approx(pooled[f"handoff:{a}"], rel=1e-12), a
        assert res["gain_handoff"][str(a)] == pytest.approx(pooled["alone"] - pooled[f"handoff:{a}"], abs=1e-12), a
        assert res["transfer"][str(a)] == pytest.approx((pooled["alone"] - pooled[f"handoff:{a}"]) / (pooled["alone"] - pooled["context"]), rel=1e-9), a
    best = min(alphas, key=lambda a: (pooled[f"handoff:{a}"], a))
    assert res["best_alpha"] == best and res["transfer_best"] == res["transfer"][str(best)]
    assert res["gain_context"] == pytest.approx(pooled["alone"] - pooled["context"], abs=1e-12) and res["gain_sequential"] == pytest.approx(pooled["alone"] - pooled["sequential"], abs=1e-12)
    assert res["transfer_sequential"] == pytest.approx((pooled["alone"] - pooled["sequential"]) / (pooled["alone"] - pooled["context"]), rel=1e-9)
    ci = pc.bootstrap_handoff(arr, alphas, pc.N_BOOT, 7)                         # the intervals come from the same resamples, slot by slot
    assert res["ci95"]["gain_context"] == ci["gain_context"] and res["ci95"]["gain_sequential"] == ci["gain_sequential"] and res["ci95"]["transfer_sequential"] == ci["transfer_sequential"]
    assert res["ci95"]["transfer_best"] == ci[f"transfer:{best}"]
    assert all(res["ci95"]["gain_handoff"][str(a)] == ci[f"gain_handoff:{a}"] and res["ci95"]["transfer"][str(a)] == ci[f"transfer:{a}"] for a in alphas)
    d = np.array(heads)                                                          # [target, head, (S, Ksum)]
    assert [h["head"] for h in res["head_diff"]] == [0, 1]
    for h in (0, 1):
        assert res["head_diff"][h]["S_rel"] == pytest.approx(float(d[:, h, 0].mean()), rel=1e-12) and res["head_diff"][h]["Ksum_rel"] == pytest.approx(float(d[:, h, 1].mean()), rel=1e-12)
    assert res["n_scored_bytes"] == sum(r["n_scored"] for r in res["targets"])


def test_probe_handoff_equals_its_building_blocks_on_two_hand_built_readers_per_target(world, tiny_hybrid, monkeypatch):
    _driver_equals_its_building_blocks_on_hand_built_readers(world, tiny_hybrid, monkeypatch, 600, 300)


def test_probe_handoff_equals_its_building_blocks_on_every_related_file_cut_at_300_bytes(world, tiny_hybrid, monkeypatch):
    _driver_equals_its_building_blocks_on_hand_built_readers(world, tiny_hybrid, monkeypatch, 5000, 300)


def test_without_a_context_gain_the_driver_reports_no_transfer_ratio_anywhere(world, tiny_hybrid):
    from scripts.probe_commons import MIN_GAIN
    res = _handoff(world, tiny_hybrid)                                           # on the tiny world a plain prefix does not help, so the ratios of the tests above need MIN_GAIN lowered
    assert res["gain_context"] < MIN_GAIN
    assert all(v is None for v in res["transfer"].values()) and res["transfer_best"] is None and res["transfer_sequential"] is None
    ci = res["ci95"]
    assert all(v is None for v in ci["transfer"].values()) and ci["transfer_best"] is None and ci["transfer_sequential"] is None


def test_the_stream_rows_come_from_the_stream_api_and_the_plain_rows_from_the_plain_forward(world, tiny_hybrid, monkeypatch):
    from flybrain.model.hybrid import GRUHybrid
    base = _handoff(world, tiny_hybrid)["bpb"]
    real = GRUHybrid.stream

    def skewed(self, tokens, state):
        logits, state = real(self, tokens, state)
        return logits * 1.5, state                                               # on the tiny model both APIs agree to the last bit: now they disagree
    monkeypatch.setattr(GRUHybrid, "stream", skewed)
    moved = _handoff(world, tiny_hybrid)["bpb"]
    assert moved["alone_forward"] == base["alone_forward"] and moved["context"] == base["context"]        # the plain forwards do not use the stream API
    assert moved["alone"] != base["alone"] and moved["sequential"] != base["sequential"] and moved["handoff"]["0.5"] != base["handoff"]["0.5"]
    assert abs(moved["alone"] - moved["alone_forward"]) > 1e-2                                            # so the two alone rows are two measurements


def test_a_target_with_an_empty_body_is_skipped_not_scored_as_nan(world, tiny_hybrid, tmp_path):
    import json

    from scripts.probe_commons import probe_handoff
    path = tmp_path / "with_empty.bytes"
    path.write_bytes(world.path.read_bytes() + b"\n// ==== k/t6.c ====\n")        # the world plus a seventh dev file that ends with its header
    res = probe_handoff(tiny_hybrid(), "cpu", path, "dev", (0.5,), 0, 300, 600, 300, 64, world.train_end, world.tail_start, 7, 0)
    assert res["n_targets"] == 6 and "k/t6.c" not in [t["path"] for t in res["targets"]]
    assert np.isfinite(res["bpb"]["alone"]) and not any(np.isnan(t["bpb"]["alone"]) for t in res["targets"])
    json.dumps(res, allow_nan=False)                                             # no NaN anywhere in the result


def test_main_hands_every_handoff_flag_to_probe_handoff_on_both_splits(world, tiny_checkpoint, tmp_path, monkeypatch):
    """The command line is probe_handoff with the flags in the right places: the same numbers as a direct call, on readers rebuilt from the corpus bytes."""
    import json

    import scripts.probe_commons as pc
    monkeypatch.setattr(pc, "RESULTS", tmp_path)
    ckpt = tiny_checkpoint()
    model = pc.load_checkpoint(str(ckpt), "cpu", 0)[0]
    flags = ["--mode", "handoff", "--ckpt", str(ckpt), "--corpus", str(world.path), "--device", "cpu", "--n-scored", "250", "--budget", "2500", "--per-file", "250", "--seed", "5",
             "--train-end", str(world.train_end), "--tail-start", str(world.tail_start), "--dev-files", "3"]
    for split, extra, alphas, limit in (("dev", ["--limit", "2"], pc.ALPHAS, 2), ("test", ["--alpha", "0.1"], (0.1,), 0)):
        pc.main(["--run-name", f"flags_{split}", "--split", split, *extra, *flags])
        got = json.loads((tmp_path / f"flags_{split}.json").read_text())
        want = pc.probe_handoff(model, "cpu", world.path, split, alphas, 5, 250, 2500, 250, 2048, world.train_end, world.tail_start, 3, limit)
        assert {k: got[k] for k in want} == json.loads(json.dumps(want)), split
        assert (got["mode"], got["split"], got["ckpt"], got["arm"], got["chunk"], got["run_name"]) == ("handoff", split, ckpt.name, "mingru_sparse_post", 0, f"flags_{split}")
        assert got["n_targets"] == len(got["targets"]) == (2 if split == "dev" else 3)
        for t in got["targets"]:                                                 # the readers are the ones rebuilt from the corpus bytes: all of them inside the training range
            readers, _, _, scored = _by_hand(world, t["index"], 2500, 250, 250)
            assert (t["readers"], t["reader_bytes"], t["n_scored"]) == (len(readers), sum(map(len, readers)), scored), t["path"]


def test_main_handoff_on_the_test_split_needs_the_alpha_chosen_on_dev(world, tiny_checkpoint, tmp_path, monkeypatch):
    import json

    import pytest
    import scripts.probe_commons as pc
    monkeypatch.setattr(pc, "RESULTS", tmp_path)
    loaded = _spy_head_chunks(pc, monkeypatch)
    common = ["--mode", "handoff", "--ckpt", str(tiny_checkpoint()), "--corpus", str(world.path), "--device", "cpu", "--n-scored", "300", "--budget", "600", "--per-file", "300",
              "--train-end", str(world.train_end), "--tail-start", str(world.tail_start), "--dev-files", "3"]
    with pytest.raises(SystemExit) as refused:                                  # without --alpha the test split would choose its own alpha: it is scored once, at the dev alpha
        pc.main(["--run-name", "ho_test_all", "--split", "test", *common])
    assert refused.value.code == "handoff on the test split needs --alpha (the alpha chosen on dev)"
    assert loaded == [] and not (tmp_path / "ho_test_all.json").exists()         # refused before the checkpoint was loaded, and nothing was written
    pc.main(["--run-name", "ho_test_one", "--split", "test", "--alpha", "0.1", *common])   # with the alpha the test split runs
    one = json.loads((tmp_path / "ho_test_one.json").read_text())
    assert one["split"] == "test" and one["alphas"] == [0.1] and one["n_targets"] == 3 and len(loaded) == 1
    pc.main(["--run-name", "ho_test_zero", "--split", "test", "--alpha", "0.0", *common])   # alpha 0.0 is a value, not a missing flag
    assert json.loads((tmp_path / "ho_test_zero.json").read_text())["alphas"] == [0.0] and len(loaded) == 2
    pc.main(["--run-name", "ho_dev_all", "--split", "dev", *common])             # the dev split chooses the alpha: it scores every one, no flag needed
    assert json.loads((tmp_path / "ho_dev_all.json").read_text())["alphas"] == list(pc.ALPHAS) and len(loaded) == 3


def _render_cells(text):
    """Every table row of a rendered text as {first cell: the other cells}: the condition table by condition, the head table by head number."""
    rows = {}
    for line in text.splitlines():
        if line.startswith("|") and not line.startswith("|---"):
            first, *rest = (cell.strip() for cell in line.strip("|").split("|"))
            rows[first] = rest
    return rows


def _check_render(res):
    """Every number render_handoff prints against the result it was given: the title line, each cell of the condition table and of the head table, the plain-forward line,
    and the n/a legend, which is printed exactly when a cell reads n/a, in a paragraph of its own between the two tables."""
    import scripts.probe_commons as pc
    b, ci = res["bpb"], res["ci95"]
    gain = lambda v, c: f"{v:+.3f} ({c[0]:+.3f}, {c[1]:+.3f})"
    ratio = lambda v, c: "n/a" if v is None else f"{v:.2f}" if c is None else f"{v:.2f} ({c[0]:.2f}, {c[1]:.2f})"
    want = {"condition": ["bits/byte", "gain vs alone (95 % interval)", "transfer ratio (95 % interval)"], "alone": [f"{b['alone']:.4f}", "", ""],
            "context (same bytes as a plain prefix)": [f"{b['context']:.4f}", gain(res["gain_context"], ci["gain_context"]), "1.00" if res["gain_context"] >= pc.MIN_GAIN else "n/a"],
            "sequential reader": [f"{b['sequential']:.4f}", gain(res["gain_sequential"], ci["gain_sequential"]), ratio(res["transfer_sequential"], ci["transfer_sequential"])],
            "head (all_heads() order)": ["S relative difference", "Ksum relative difference"]}
    want.update({f"handoff alpha {a}": [f"{v:.4f}", gain(res["gain_handoff"][a], ci["gain_handoff"][a]), ratio(res["transfer"][a], ci["transfer"][a])] for a, v in b["handoff"].items()})
    want.update({str(h["head"]): [f"{h['S_rel']:.2e}", f"{h['Ksum_rel']:.2e}"] for h in res["head_diff"]})
    lines = pc.render_handoff(res).splitlines()
    assert _render_cells("\n".join(lines)) == want
    assert lines[0] == (f"{res['run_name']} {res['split']}: {res['n_targets']} targets, {res['n_scored_bytes']} scored bytes, readers {res['readers']['mean_files']:.1f} files / "
                        f"{res['readers']['mean_bytes']:.0f} bytes on average, best alpha {res['best_alpha']}")
    assert f"alone through the plain forward: {b['alone_forward']:.4f}" in lines
    legend = f"n/a: the context gain is below {pc.MIN_GAIN} bits/byte, so no transfer ratio is reported"
    some_na = any(c == "n/a" for cells in want.values() for c in cells)
    assert [line for line in lines if line.startswith("n/a:")] == ([legend] if some_na else [])       # printed exactly when a cell reads n/a
    if some_na:
        seq, head = (next(i for i, line in enumerate(lines) if line.startswith(start)) for start in ("| sequential reader |", "| head (all_heads() order)"))
        assert seq < lines.index(legend) - 1 < lines.index(legend) < head and lines[lines.index(legend) - 1] == ""


def _hand_result(gain_context, defined):
    """A result for render_handoff with a different number in every field, so a cell read from the wrong field shows. Without ratios (defined=False) they are all None, as the driver gives them."""
    ratio = (lambda v: v) if defined else (lambda v: None)
    return {"run_name": "hand", "split": "dev", "n_targets": 7, "n_scored_bytes": 12345, "readers": {"mean_files": 6.5, "mean_bytes": 32768.0}, "best_alpha": 0.3,
            "bpb": {"alone": 1.5, "alone_forward": 1.5001, "context": 1.5 - gain_context, "sequential": 1.43, "handoff": {"0.1": 1.47, "0.3": 1.40, "1.0": 1.44}},
            "gain_context": gain_context, "gain_sequential": 0.07, "gain_handoff": {"0.1": 0.03, "0.3": 0.10, "1.0": 0.06},
            "transfer": {"0.1": ratio(0.15), "0.3": ratio(0.5), "1.0": ratio(0.3)}, "transfer_best": ratio(0.5), "transfer_sequential": ratio(0.35),
            "ci95": {"gain_context": [gain_context - 0.03, gain_context + 0.02], "gain_sequential": [0.04, 0.10],
                     "gain_handoff": {"0.1": [0.01, 0.05], "0.3": [0.07, 0.13], "1.0": [0.02, 0.09]},
                     "transfer": {"0.1": ratio([0.05, 0.25]), "0.3": ratio([0.35, 0.65]), "1.0": None}, "transfer_best": ratio([0.35, 0.65]), "transfer_sequential": ratio([0.2, 0.5])},
            "head_diff": [{"head": 0, "S_rel": 0.123456, "Ksum_rel": 0.0456}, {"head": 1, "S_rel": 1.5e-07, "Ksum_rel": 2.5e-07}, {"head": 2, "S_rel": 0.2, "Ksum_rel": 0.07}]}


def test_render_handoff_prints_every_cell_of_a_real_result_with_and_without_ratios(world, tiny_hybrid, monkeypatch):
    import scripts.probe_commons as pc
    alphas = (0.5, 0.0, 1.0)
    real = {"run_name": "tiny", **_handoff(world, tiny_hybrid, alphas)}
    assert real["gain_context"] < pc.MIN_GAIN and real["transfer_best"] is None      # the tiny world: no context gain, so no ratio is defined
    _check_render(real)
    monkeypatch.setattr(pc, "MIN_GAIN", -100.0)                                      # a ratio at every row
    defined = {"run_name": "tiny", **_handoff(world, tiny_hybrid, alphas)}
    assert defined["transfer_best"] is not None and defined["ci95"]["transfer_best"] is not None
    _check_render(defined)


def test_render_handoff_prints_every_cell_of_a_hand_built_result_and_n_a_where_no_ratio_is_defined():
    from scripts.probe_commons import render_handoff
    for gain_context in (0.2, 0.01, 0.0099, 0.0, -0.065):
        _check_render(_hand_result(gain_context, gain_context >= 0.01))
    context_ratio = lambda g: _render_cells(render_handoff(_hand_result(g, g >= 0.01)))["context (same bytes as a plain prefix)"][2]
    assert [context_ratio(g) for g in (0.2, 0.01, 0.0099, -0.065)] == ["1.00", "1.00", "n/a", "n/a"]    # 1.00 where the other ratios are defined (at MIN_GAIN too), n/a below it
    assert "n/a: the context gain is below 0.01 bits/byte, so no transfer ratio is reported" in render_handoff(_hand_result(-0.065, False)).splitlines()
