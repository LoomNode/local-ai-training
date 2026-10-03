"""scripts/census.py on a tiny corpus in the real header format."""

from __future__ import annotations

import json

import pytest


def _run(world, split="dev", seed=0, never=True, ps=(128, 256, 100_000), ns=(4, 8), limit=0):
    from scripts.census import run_census
    return run_census(world.path, split, ns, ps, seed, never, 300, world.train_end, world.tail_start, world.dev_files, limit)


def test_counts_partition_the_scored_bytes_and_related_files_beat_random_ones(world):
    res = _run(world)
    assert [r["path"] for r in res["targets"]] == ["k/t0.c", "k/t1.c", "k/t2.c"]
    assert [r["n_scored"] for r in res["targets"]] == [300, 300, 90]           # the short body is scored whole, nothing padded
    assert res["n_targets"] == 3 and res["n_scored_bytes"] == 690
    for n, row in res["trained"]["cells"].items():
        for p, c in row.items():
            assert c["total"] == 690
            assert c["A"] + c["B"] + c["C"] == c["total"] == c["A"] + c["Bp"] + c["Cp"], (n, p)
            assert abs(sum(c["share"][k] for k in ("A", "B", "C")) - 1.0) < 1e-12
    t0 = res["targets"][0]["cells"]["8"]["100000"]                              # k/t0.c: its copy in y/ is not drawn, so the random pool has no k phrase
    assert t0["B"] > t0["Bp"] > 0                                              # same-directory files share the phrase, random ones only the includes
    small, large = res["trained"]["cells"]["8"]["128"], res["trained"]["cells"]["8"]["100000"]
    assert large["B"] >= small["B"] and large["Bp"] >= small["Bp"]            # more prefix, no fewer matches


def test_never_trained_pass_uses_only_the_targets_own_split(world):
    dev = _run(world, "dev")
    nt = dev["never_trained"]
    assert nt["n_targets"] == 3 and [r["never_files"] for r in dev["targets"]] == [2, 2, 2]
    assert nt["n_scored_bytes"] == 690 and set(nt["trained_cells"]) == {"4", "8"}
    test = _run(world, "test")
    assert [r["path"] for r in test["targets"]] == ["k/t3.c", "z/t4.c", "z/t5.c"]
    assert test["never_trained"]["n_targets"] == 0                           # no target has two other files in the test split
    assert all(c["share"]["A"] is None for row in test["never_trained"]["cells"].values() for c in row.values())
    assert _run(world, "dev", never=False)["never_trained"] is None


def test_census_is_deterministic_for_a_fixed_seed_and_the_seed_reaches_only_the_random_prefix(world, monkeypatch):
    import scripts.census as census
    a, b = _run(world, seed=0), _run(world, seed=0)
    assert json.dumps(a) == json.dumps(b)
    seen, real = [], census.prefix_pairs
    monkeypatch.setattr(census, "prefix_pairs", lambda *args, **kw: (seen.append(args[5]), real(*args, **kw))[1])   # args[5] is the seed
    c = _run(world, seed=7)
    assert set(seen) == {7} and len(seen) == 6                                  # one trained and one never-trained draw per target
    for n, row in a["trained"]["cells"].items():
        for p, cell in row.items():
            other = c["trained"]["cells"][n][p]
            assert (cell["A"], cell["B"], cell["C"]) == (other["A"], other["B"], other["C"])   # related prefixes and self-matches do not depend on the seed


def test_limit_keeps_the_first_targets_only(world):
    assert [r["path"] for r in _run(world, limit=2)["targets"]] == ["k/t0.c", "k/t1.c"]


def test_render_prints_one_table_per_n(world):
    from scripts.census import render
    text = render({"run_name": "census_tiny", **_run(world)})
    assert text.count("| P | A | B | B' | C | C' |") == 2 * 3                  # n = 4, 8 for the trained, matched-trained and never-trained blocks
    assert "n = 8" in text and "| 100000 |" in text and "never trained" in text


def test_main_writes_the_json_and_refuses_an_existing_run_name(world, tmp_path, monkeypatch, capsys):
    import scripts.census as census
    monkeypatch.setattr(census, "RESULTS", tmp_path)
    argv = ["--split", "dev", "--run-name", "census_tiny", "--corpus", str(world.path), "--ns", "4,8", "--ps", "128,256", "--n-scored", "300",
            "--never-trained", "--train-end", str(world.train_end), "--tail-start", str(world.tail_start), "--dev-files", "3"]
    census.main(argv)
    res = json.loads((tmp_path / "census_tiny.json").read_text())
    assert {"run_name", "args", "split", "ns", "ps", "n_targets", "n_scored_bytes", "trained", "never_trained", "targets"} <= set(res)
    assert res["ns"] == [4, 8] and res["ps"] == [128, 256] and "census_tiny" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        census.main(argv)


def test_a_target_with_an_empty_body_is_skipped(corpus):
    from flybrain.data.files import file_table
    from scripts.census import run_census
    files = [(f"d/a{j}.c", (b"int a%d;\n" % j) * 30) for j in range(4)] + [("d/ok.c", b"int ok;\n" * 30), ("d/empty.c", b"")]
    path = corpus(files)
    cut = file_table(path)[4].start                    # training ends where d/ok.c starts; the tail is ok.c and a last, empty header
    res = run_census(path, "dev", (4,), (64,), 0, False, 100, cut, cut, 2)
    assert [r["path"] for r in res["targets"]] == ["d/ok.c"] and res["n_scored_bytes"] == 100
