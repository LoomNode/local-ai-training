"""--wsd-cooldown and --resume (scripts/run_sparsela.py): the schedule's shape, and a run resumed from one of its --ckpt-every
checkpoints ends where the unbroken run ends, with the record (curves, clock, flags) continued."""
import json
import sys

import pytest
import torch

SMALL = ["--task", "code", "--train-bytes", "3000", "--val-bytes", "4000", "--d-model", "16", "--n-layers", "2", "--d-ff", "32",
         "--n-key", "64", "--top-k", "4", "--key-conv", "2", "--normalize", "--chunk", "8", "--batch", "2", "--seq-len", "32",
         "--lr", "0.003", "--save", "--seed", "0", "--device", "cpu", "--run-name", "tiny", "--arms", "transformer", "--rope"]


def run_main(monkeypatch, tmp_path, tag, corpus, *flags):
    """The runner's main() in-process, results into tmp_path/tag: (the run's JSON, the results dir)."""
    import scripts.run_sparsela as sra
    res = tmp_path / tag
    res.mkdir()
    monkeypatch.setattr(sra, "RESULTS", res)
    monkeypatch.setattr(sra, "_CMASK", {})
    monkeypatch.setattr(sys, "argv", ["run_sparsela.py", "--corpus-file", str(corpus), *SMALL, *flags])
    sra.main()
    return json.loads((res / "tiny.json").read_text()), res


def test_wsd_schedule_shape():
    from scripts.run_sparsela import lr_lambda
    f = lr_lambda(1000, 100, False, cooldown=200)
    assert abs(f(0) - 0.01) < 1e-9 and f(100) == 1.0 and f(799) == 1.0
    assert abs(f(800) - (1 - 0.9 / 200)) < 1e-9 and abs(f(999) - 0.1) < 1e-9
    g = lr_lambda(2000, 100, False, cooldown=200)   # the stable phase does not depend on steps: a run extended by --steps continues at full lr
    assert g(799) == 1.0 and g(1500) == 1.0 and abs(g(1999) - 0.1) < 1e-9
    assert lr_lambda(1000, 100, True)(500) < 1.0   # the cosine is untouched


def test_the_two_schedules_cannot_both_be_named(tmp_path):
    from scripts.run_sparsela import parse_args
    c = tmp_path / "c.bytes"; c.write_bytes(b"x" * 100)
    with pytest.raises(SystemExit):
        parse_args([*SMALL, "--corpus-file", str(c), "--steps", "10", "--cosine", "--wsd-cooldown", "2"])
    with pytest.raises(SystemExit):
        parse_args([*SMALL, "--corpus-file", str(c), "--steps", "10", "--wsd-cooldown", "10"])
    with pytest.raises(SystemExit):
        parse_args([*SMALL, "--corpus-file", str(c), "--steps", "10", "--resume", str(tmp_path / "missing.pt")])


def test_resume_ends_where_the_unbroken_run_ends(monkeypatch, tmp_path):
    corpus = tmp_path / "c.bytes"
    corpus.write_bytes(b"\n// ==== a.c ====\n" + b"int main(void) { return 0; }\nstatic int n = 0;\n" * 2600)   # ~120 kB: the legacy validation bytes start at --train-bytes + 100,000
    flags = ["--steps", "12", "--warmup", "2", "--wsd-cooldown", "4", "--ckpt-every", "6", "--val-fixed", "2", "--val-offset", "110000", "--eval-every", "3"]
    a, ra = run_main(monkeypatch, tmp_path, "a", corpus, *flags)                       # unbroken, with a checkpoint at step 6
    ck6 = ra / "ckpt" / "tiny_transformer.step6.pt"
    assert ck6.is_file()
    b, rb = run_main(monkeypatch, tmp_path, "b", corpus, *flags, "--resume", str(ck6))   # steps 7-12 only
    fa = torch.load(ra / "ckpt" / "tiny_transformer.pt", map_location="cpu", weights_only=False)["state"]
    fb = torch.load(rb / "ckpt" / "tiny_transformer.pt", map_location="cpu", weights_only=False)["state"]
    assert fa.keys() == fb.keys() and all(torch.allclose(fa[k], fb[k], atol=1e-6, rtol=0) for k in fa), "the resumed weights differ"
    ra_, rb_ = a["arms"][0], b["arms"][0]
    assert rb_["resumed_from"] == {"path": str(ck6), "step": 6} and "resumed_from" not in ra_
    ca, cb = ra_["curve_fixed"], rb_["curve_fixed"]
    assert [r["step"] for r in ca] == [r["step"] for r in cb] == [3, 6, 9, 12]           # the rows before the checkpoint come from it
    assert all(abs(x["bpb"] - y["bpb"]) < 1e-6 for x, y in zip(ca, cb))
    assert cb[0]["train_seconds"] == ca[0]["train_seconds"] and cb[-1]["train_seconds"] > cb[1]["train_seconds"]   # the clock continued
    assert rb_["bytes_seen"] == ra_["bytes_seen"] == 12 * 2 * 32
    assert "resume" not in a and "wsd_cooldown" not in json.loads((ra / "tiny.json").read_text()) or a["wsd_cooldown"] == 4
    assert b["resume"] == str(ck6) and b["wsd_cooldown"] == 4


def test_a_checkpoint_of_another_arm_or_without_an_optimizer_is_refused(monkeypatch, tmp_path):
    corpus = tmp_path / "c.bytes"
    corpus.write_bytes(b"\n// ==== a.c ====\n" + b"int main(void) { return 0; }\n" * 4000)   # past the legacy validation bytes at 103,000
    _, ra = run_main(monkeypatch, tmp_path, "a", corpus, "--steps", "4")   # --save writes the model-only checkpoint
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, "b", corpus, "--steps", "4", "--resume", str(ra / "ckpt" / "tiny_transformer.pt"))
