"""The yardstick's scorer inside the runner: fixed windows, cold and warm bits/byte. CPU only, no corpus."""

from __future__ import annotations

import argparse
import math

import pytest
import torch
from torch.nn import functional as F

from flybrain.data.text import fixed_windows
from flybrain.model.hybrid import GRUHybrid


def _tiny():
    torch.manual_seed(0)
    return GRUHybrid(d_model=16, backbone="mingru", where="none", n_layers=1, d_ff=32)


def test_eval_fixed_is_the_per_position_mean_cold_and_its_second_half_warm():
    from scripts.run_sparsela import eval_fixed
    m = _tiny()
    xf, yf = fixed_windows(torch.randint(0, 256, (7 * 16 + 1,)), 16, 7)   # 7 windows in batches of 3: the last batch is short
    out = eval_fixed(m, xf, yf, argparse.Namespace(batch=3, amp=False), torch.device("cpu"))
    assert m.training                                                    # left in train mode, like the runner's other evals
    m.eval()
    with torch.no_grad():                                                 # reference: one window at a time
        ce = torch.stack([F.cross_entropy(m(xf[i:i + 1])[0][0], yf[i], reduction="none") for i in range(7)])
    assert out["bpb"] == pytest.approx(float(ce.mean()) / math.log(2), abs=1e-5)
    assert out["bpb_warm"] == pytest.approx(float(ce[:, 8:].mean()) / math.log(2), abs=1e-5)
    assert out["bpb_warm"] != pytest.approx(out["bpb"], abs=1e-6)         # the two numbers really are different cuts
    again = eval_fixed(m, xf, yf, argparse.Namespace(batch=7, amp=False), torch.device("cpu"))
    assert again["bpb"] == pytest.approx(out["bpb"], abs=1e-6)            # the batch size does not change the score


def test_eval_fixed_uses_no_global_rng():
    from scripts.run_sparsela import eval_fixed
    m = _tiny()
    xf, yf = fixed_windows(torch.randint(0, 256, (4 * 16 + 1,)), 16, 4)
    state = torch.get_rng_state()
    eval_fixed(m, xf, yf, argparse.Namespace(batch=2, amp=False), torch.device("cpu"))
    assert torch.equal(state, torch.get_rng_state())


# ---- the runner's yardstick flags: tiny CPU runs through main(), no corpus beyond a file written here ----

GOLDEN_ARM_KEYS = ["val_bpb", "val_bpb_first_q", "val_bpb_last_q", "val_bpb_copyable", "copyable_frac", "val_bpb_noncopyable",
                   "arm", "seconds", "trainable_scalars", "curve", "flops_per_token_recurrent"]  # results/m20_linux_depth4local_s0.json
NEW_ARM_KEYS = {"train_seconds", "eval_seconds", "bytes_seen", "peak_vram_bytes", "curve_fixed", "val_fixed", "joules", "mean_power_w",
                "power_limit_w", "max_temp_c", "contended", "samples", "telemetry_lost", "gpu_uuid"}
SMALL = ["--task", "code", "--train-bytes", "3000", "--val-bytes", "4000", "--d-model", "16", "--n-layers", "2", "--d-ff", "32",
         "--n-key", "64", "--top-k", "4", "--key-conv", "2", "--normalize", "--chunk", "8", "--batch", "2", "--seq-len", "32",
         "--steps", "6", "--lr", "0.003", "--arms", "mingru_sparse_post", "--save", "--seed", "0", "--device", "cpu", "--run-name", "tiny"]
FIXED = ["--val-fixed", "4", "--val-offset", "110000"]  # 4 windows of 32 bytes, past the 3000 train and the val bytes at 103000


def _corpus(tmp_path):
    path = tmp_path / "tiny.bytes"
    path.write_bytes((b"int main(void) { return 0; }\n" * 5000)[:130_000])
    return path


def _run(monkeypatch, tmp_path, tag, *flags):
    """main() in-process, results into tmp_path/tag; returns (the run's JSON, the results dir)."""
    import json
    import sys

    import scripts.run_sparsela as sra
    res = tmp_path / tag
    res.mkdir()
    monkeypatch.setattr(sra, "RESULTS", res)
    monkeypatch.setattr(sra, "_CMASK", {})
    monkeypatch.setattr(sys, "argv", ["run_sparsela.py", "--corpus-file", str(_corpus(tmp_path)), *SMALL, *flags])
    sra.main()
    return json.loads((res / "tiny.json").read_text()), res


def _ckpt(res, suffix=""):
    return torch.load(res / "ckpt" / f"tiny_mingru_sparse_post{suffix}.pt", map_location="cpu", weights_only=False)


def test_yardstick_flags_leave_training_untouched(monkeypatch, tmp_path):
    """The required runner test: the same final weights and the same legacy JSON with and without the new flags."""
    legacy, ra = _run(monkeypatch, tmp_path, "a")
    rng_a = torch.get_rng_state()
    flagged, rb = _run(monkeypatch, tmp_path, "b", *FIXED, "--eval-every", "2", "--ckpt-every", "2")
    assert torch.equal(rng_a, torch.get_rng_state())                                        # the global generator ends where it ended without the flags
    sa, sb = _ckpt(ra)["state"], _ckpt(rb)["state"]
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)            # identical state_dict, bit for bit
    for k, v in legacy.items():
        if k != "arms":
            assert flagged[k] == v, k                                                       # every legacy top-level field
    assert len(legacy["arms"]) == len(flagged["arms"]) == 1
    for k, v in legacy["arms"][0].items():
        if k != "seconds":
            assert flagged["arms"][0][k] == v, k                                            # every legacy per-arm field but the clock


def test_with_every_flag_absent_the_json_and_checkpoint_are_the_legacy_ones(monkeypatch, tmp_path):
    from scripts.run_sparsela import YARDSTICK_FLAGS
    legacy, res = _run(monkeypatch, tmp_path, "a")
    assert not set(YARDSTICK_FLAGS) & set(legacy)                                           # no new key among the recorded args
    assert list(legacy["arms"][0]) == GOLDEN_ARM_KEYS                                       # same keys, same order, nothing added
    assert not NEW_ARM_KEYS & set(legacy["arms"][0])
    assert not set(YARDSTICK_FLAGS) & set(_ckpt(res)["args"])
    assert sorted(p.name for p in (res / "ckpt").iterdir()) == ["tiny_mingru_sparse_post.pt"]


def test_flagged_run_writes_the_new_fields_and_checkpoints(monkeypatch, tmp_path):
    out, res = _run(monkeypatch, tmp_path, "b", *FIXED, "--eval-every", "2", "--ckpt-every", "2")
    arm = out["arms"][0]
    assert out["val_fixed"] == 4 and out["val_offset"] == 110000 and out["eval_every"] == 2 and out["ckpt_every"] == 2
    assert [r["step"] for r in arm["curve_fixed"]] == [2, 4, 6]
    assert all(set(r) == {"step", "train_seconds", "bpb", "bpb_warm"} for r in arm["curve_fixed"])
    ts = [r["train_seconds"] for r in arm["curve_fixed"]]
    assert ts == sorted(ts) and 0 < ts[-1] == arm["train_seconds"] <= arm["seconds"]        # the last row closes the training clock
    assert arm["val_fixed"] == {"offset": 110000, "n_windows": 4, "bpb": arm["curve_fixed"][-1]["bpb"], "bpb_warm": arm["curve_fixed"][-1]["bpb_warm"]}
    assert arm["eval_seconds"] > 0 and arm["train_seconds"] + arm["eval_seconds"] <= arm["seconds"] + 1e-3
    assert arm["bytes_seen"] == 6 * 2 * 32 and arm["peak_vram_bytes"] == 0                   # CPU
    assert not {"joules", "gpu_uuid"} & set(arm)                                             # no meter was asked for
    assert sorted(p.name for p in (res / "ckpt").iterdir()) == [
        "tiny_mingru_sparse_post.pt", "tiny_mingru_sparse_post.step2.pt", "tiny_mingru_sparse_post.step4.pt", "tiny_mingru_sparse_post.step6.pt"]
    ck = _ckpt(res, ".step4")
    assert set(ck) == {"arm", "args", "step", "state", "opt", "sched"} and ck["step"] == 4 and ck["arm"] == "mingru_sparse_post"
    assert ck["sched"]["last_epoch"] == 4 and ck["args"]["ckpt_every"] == 2


@pytest.mark.parametrize("steps, every, expected", [
    (6, 2, [2, 4, 6]),                  # the last periodic eval is the final one: scored once, not twice
    (4, 100, [4]),                      # --eval-every larger than --steps: only the final eval
    (4, 4, [4]),
    (12, 0, [2, 4, 6, 8, 10, 12]),      # 0 keeps steps // 6
    (1, 0, [1]),
])
def test_fixed_eval_cadence(monkeypatch, tmp_path, steps, every, expected):
    flags = [*FIXED, "--steps", str(steps)] + (["--eval-every", str(every)] if every else [])
    out, _ = _run(monkeypatch, tmp_path, "c", *flags)
    arm = out["arms"][0]
    assert [r["step"] for r in arm["curve_fixed"]] == expected
    assert arm["val_fixed"]["bpb"] == arm["curve_fixed"][-1]["bpb"]
    assert [r["step"] for r in arm["curve"]] == [s for s in range(max(1, steps // 6), steps + 1, max(1, steps // 6))]  # legacy cadence kept


def test_a_fixed_region_the_corpus_cannot_supply_fails_before_any_training(monkeypatch, tmp_path):
    import scripts.run_sparsela as sra
    monkeypatch.setattr(sra, "build", lambda *a, **k: pytest.fail("a model was built"))
    near_the_end = str(_corpus(tmp_path).stat().st_size - 100)
    with pytest.raises(ValueError, match="need 129 bytes for 4 windows of 32"):
        _run(monkeypatch, tmp_path, "d", "--val-fixed", "4", "--val-offset", near_the_end)
    assert list((tmp_path / "d").iterdir()) == []                                            # nothing written


def test_evals_and_saves_never_count_as_training_time(monkeypatch, tmp_path):
    """Fake clock: a training step costs 1 s, a legacy eval 10 s, a fixed eval 100 s, a checkpoint save 1000 s."""
    import scripts.run_sparsela as sra
    from flybrain.meter import SplitClock
    t = [0.0]

    def costing(fn, dt):
        def wrapped(*a, **k):
            t[0] += dt
            return fn(*a, **k)
        return wrapped

    monkeypatch.setattr(sra, "SplitClock", lambda sync: SplitClock(sync=sync, now=lambda: t[0]))
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", costing(torch.nn.utils.clip_grad_norm_, 1.0))  # once per training step
    monkeypatch.setattr(sra, "eval_wiki", costing(sra.eval_wiki, 10.0))     # 6 periodic + 1 final
    monkeypatch.setattr(sra, "eval_fixed", costing(sra.eval_fixed, 100.0))  # steps 2, 4 and the final 6
    monkeypatch.setattr(sra, "save_ckpt", costing(sra.save_ckpt, 1000.0))   # steps 2, 4, 6
    arm = _run(monkeypatch, tmp_path, "t", *FIXED, "--eval-every", "2", "--ckpt-every", "2")[0]["arms"][0]
    assert arm["train_seconds"] == 6.0
    assert arm["eval_seconds"] == 7 * 10.0 + 3 * 100.0 + 3 * 1000.0
    assert [r["train_seconds"] for r in arm["curve_fixed"]] == [2.0, 4.0, 6.0]


def test_cpu_meter_is_a_no_op_that_still_gets_the_timing_fields(monkeypatch, tmp_path, capsys):
    out, _ = _run(monkeypatch, tmp_path, "e", "--meter")
    arm = out["arms"][0]
    assert {"train_seconds", "eval_seconds", "bytes_seen", "peak_vram_bytes"} <= set(arm)
    assert not {"joules", "mean_power_w", "contended", "telemetry_lost", "gpu_uuid", "curve_fixed", "val_fixed"} & set(arm)
    assert "meter: skipped, device is cpu" in capsys.readouterr().out


def test_meter_summary_reaches_the_json_and_the_thread_is_stopped(monkeypatch, tmp_path):
    import os

    import scripts.run_sparsela as sra
    from flybrain.meter import GpuMeter, Sample
    ok, made = Sample(100.0, 150.0, 60.0, 1000.0, ()), []

    def fake(args, device):
        m = GpuMeter("GPU-test", os.getpid(), interval=0.01, sampler=lambda: ok)
        m.ingest(ok, 0.0)                    # one sample is certain, whatever the thread has done by then
        made.append(m)
        return m

    monkeypatch.setattr(sra, "make_meter", fake)
    arm = _run(monkeypatch, tmp_path, "f", "--meter")[0]["arms"][0]
    assert arm["gpu_uuid"] == "GPU-test" and arm["contended"] is False and arm["telemetry_lost"] is False
    assert arm["power_limit_w"] == 150.0 and arm["max_temp_c"] == 60.0 and arm["samples"] >= 1
    assert made[0]._thread is not None and not made[0]._thread.is_alive()


def test_an_overheated_card_aborts_the_run_and_stops_the_thread(monkeypatch, tmp_path):
    import os

    import scripts.run_sparsela as sra
    from flybrain.meter import GpuMeter, Sample
    hot, made = Sample(100.0, 150.0, 90.0, 1000.0, ()), []

    def fake(args, device):
        m = GpuMeter("GPU-test", os.getpid(), interval=0.01, sampler=lambda: hot)
        m.ingest(hot, 0.0)
        made.append(m)
        return m

    monkeypatch.setattr(sra, "make_meter", fake)
    monkeypatch.setattr(sra, "CHECK_EVERY", 2)
    with pytest.raises(RuntimeError, match="reached 90 C"):
        _run(monkeypatch, tmp_path, "g", "--meter")
    assert not (tmp_path / "g" / "tiny.json").exists()                                       # an aborted run leaves no result to skip
    assert made[0]._thread is not None and not made[0]._thread.is_alive()


@pytest.mark.parametrize("argv", [["--val-offset", "5"], ["--eval-every", "5"], ["--val-fixed", "4", "--task", "copy"],
                                  ["--val-fixed", "-1"], ["--ckpt-every", "-2"]])
def test_flags_that_cannot_work_are_refused_at_parse_time(argv):
    from scripts.run_sparsela import parse_args
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_a_fixed_region_inside_the_training_bytes_is_refused_at_parse_time(capsys):
    """Windows scored from inside [0, --train-bytes) would record a training-set score as the held-out fixed-window score."""
    from scripts.run_sparsela import parse_args
    for argv in (["--val-fixed", "4"],                                                     # --val-offset left at 0: the first bytes of the corpus
                 ["--val-fixed", "4", "--val-offset", "1999999"],                          # the last byte of the default 2,000,000 training bytes
                 ["--val-fixed", "4", "--val-offset", "2999", "--train-bytes", "3000"]):   # the same, at another --train-bytes
        with pytest.raises(SystemExit) as refused:
            parse_args(["--task", "code", *argv])
        assert refused.value.code == 2 and "--val-offset must lie past the training bytes" in capsys.readouterr().err, argv
    for offset, train_bytes in ((2_000_000, "2000000"), (3000, "3000"), (110_000, "3000")):  # the first byte past the training bytes is allowed, and any later one
        args = parse_args(["--task", "code", "--val-fixed", "4", "--val-offset", str(offset), "--train-bytes", train_bytes])
        assert (args.val_fixed, args.val_offset, args.train_bytes) == (4, offset, int(train_bytes))


def test_the_parser_defaults_are_the_legacy_behaviour():
    from scripts.run_sparsela import parse_args, yardstick_on
    args = parse_args([])
    assert (args.val_fixed, args.val_offset, args.eval_every, args.meter, args.ckpt_every) == (0, 0, 0, False, 0)
    assert yardstick_on(args) is False and yardstick_on(parse_args(["--meter"])) is True
