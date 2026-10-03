"""scripts/run_spine.sh: the three spine commands, parsed by the runner's own argparse, so a typo or a drifted value fails here
and not hours into a run. No GPU, no corpus (an empty stand-in file), no uv (a stub on PATH)."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_spine.sh"

COMMON = dict(task="code", train_bytes=200_000_000, val_bytes=2_000_000, normalize=True, key_conv=2, top_k=4, n_key=4096, lr=0.0006,
              decay=1.0, d_model=512, d_ff=1024, n_layers=8, batch=8, seq_len=2048, steps=16000, save=True, seed=0, device="cuda:1",
              warmup=1000, cosine=True, wd=0.1, chunk=256, amp=True, meter=True, val_fixed=128, val_offset=202_117_357,
              eval_every=1000, ckpt_every=0)
ARMS = {"hybrid": dict(arms="mingru_sparse_post", gate=True, mem_every=2, local_attn=512, key_convs="2,4", rope=False),
        "tf": dict(arms="transformer", gate=False, mem_every=0, local_attn=0, key_convs="", rope=True),
        "zeroheads": dict(arms="mingru", gate=False, mem_every=0, local_attn=512, key_convs="", rope=False)}


def _dry(tmp_path, *argv):
    corpora = tmp_path / "corpora"
    corpora.mkdir(exist_ok=True)
    (corpora / "linux_full.bytes").write_bytes(b"x")
    return subprocess.run(["bash", str(SCRIPT), *argv], capture_output=True, text=True,
                          env={**os.environ, "DRY_RUN": "1", "CORPORA": str(corpora)})


def _parsed(tmp_path, *argv):
    from scripts.run_sparsela import parse_args
    r = _dry(tmp_path, *argv)
    assert r.returncode == 0, r.stderr
    name, flags = r.stdout.strip().split(" ", 1)
    return name, parse_args(shlex.split(flags))


def test_the_script_is_executable_and_valid_bash():
    assert os.access(SCRIPT, os.X_OK)
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


@pytest.mark.parametrize("arm", ["hybrid", "tf", "zeroheads"])
def test_each_arm_parses_to_the_decided_flags(tmp_path, arm):
    name, args = _parsed(tmp_path, "cuda:1", arm, "0")
    assert name == f"spine16_{arm}_s0"
    for k, v in {**COMMON, **ARMS[arm]}.items():
        assert getattr(args, k) == v, (arm, k)
    assert args.corpus_file == str(tmp_path / "corpora" / "linux_full.bytes")


def test_steps_seed_device_and_tag_reach_the_command(tmp_path):
    name, args = _parsed(tmp_path, "cuda:0", "tf", "1", "18286", "eqsec")
    assert name == "spine16_tf_eqsec_s1" and (args.device, args.seed, args.steps, args.val_fixed, args.eval_every) == ("cuda:0", 1, 18286, 128, 1000)


@pytest.mark.parametrize("arm, baseline", [("hybrid", "m20_linux_depth4local_s0.json"), ("tf", "m20_linux_s1.json")])
def test_each_arm_is_the_recorded_16m_recipe_plus_the_yardstick_flags(tmp_path, arm, baseline):
    """Every training flag the committed 16M run recorded is the value the spine uses (seed, device, corpus path aside)."""
    args = vars(_parsed(tmp_path, "cuda:1", arm, "0")[1])
    recorded = json.loads((ROOT / "results" / baseline).read_text())
    skip = {"run_name", "device", "save", "arms", "seed", "corpus_file", "unigram_bpb", "bigram_bpb"}
    checked = [k for k in recorded if k not in skip and k in args]
    assert len(checked) >= 24
    for k in checked:
        assert args[k] == recorded[k], (arm, k)


def test_an_unknown_arm_and_a_missing_corpus_are_refused(tmp_path):
    assert _dry(tmp_path, "cuda:1", "nope", "0").returncode == 2
    r = subprocess.run(["bash", str(SCRIPT), "cuda:1", "tf", "0"], capture_output=True, text=True,
                       env={**os.environ, "CORPORA": str(tmp_path / "empty")})
    assert r.returncode == 2 and "corpus not found" in r.stdout


def _sandbox(tmp_path, uv_body: str) -> tuple[Path, dict]:
    """A copy of the script in a scratch repo root, a stand-in corpus, and a stub uv on PATH."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "run_spine.sh")
    (tmp_path / "results").mkdir()
    (tmp_path / "corpora").mkdir()
    (tmp_path / "corpora" / "linux_full.bytes").write_bytes(b"x")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "uv").write_text("#!/usr/bin/env bash\n" + uv_body)
    (tmp_path / "bin" / "uv").chmod(0o755)
    return tmp_path / "scripts" / "run_spine.sh", {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "CORPORA": str(tmp_path / "corpora")}


def test_a_finished_run_name_is_skipped_without_starting_anything(tmp_path):
    script, env = _sandbox(tmp_path, "echo STARTED; exit 0\n")
    (tmp_path / "results" / "spine16_hybrid_s0.json").write_text("{}")
    r = subprocess.run(["bash", str(script), "cuda:1", "hybrid", "0"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "skip spine16_hybrid_s0" in r.stdout and "STARTED" not in r.stdout


def test_the_runner_is_started_with_the_run_name_and_its_output_is_filtered_and_unbuffered(tmp_path):
    script, env = _sandbox(tmp_path, 'echo "uv $*"; echo "  curve line"; echo "kept line"; exit 0\n')
    r = subprocess.run(["bash", str(script), "cuda:1", "zeroheads", "1"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "=== spine16_zeroheads_s1" in r.stdout
    assert "uv run python scripts/run_sparsela.py --run-name spine16_zeroheads_s1" in r.stdout
    assert "kept line" in r.stdout and "curve line" not in r.stdout           # two-space lines are dropped, like run_50m.sh
    assert "--line-buffered" in SCRIPT.read_text()                            # the log is read while the run is going


def test_a_failing_runner_fails_the_script_so_a_chain_of_arms_stops(tmp_path):
    script, env = _sandbox(tmp_path, 'echo "Traceback: RuntimeError: GPU reached 90 C"; exit 3\n')
    r = subprocess.run(["bash", str(script), "cuda:1", "tf", "0"], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "Traceback" in r.stdout
