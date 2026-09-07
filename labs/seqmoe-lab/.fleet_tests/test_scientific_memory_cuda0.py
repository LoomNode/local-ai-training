"""Sequential vs joint at scientific model size on GPU0 (full card OK)."""
import json
import subprocess
import sys
from pathlib import Path

import torch


def test_run_memory_scientific_cuda0(tmp_path):
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
    root = Path(__file__).resolve().parents[1]
    # Hidden tests are copied to worktree/.fleet_tests; repo root is cwd for the grader.
    script = Path("scripts/run_memory.py")
    assert script.is_file(), "run from the seqmoe-lab worktree"
    out = tmp_path / "mem.json"
    py = sys.executable
    p = subprocess.run(
        [py, str(script), "--scientific", "--device", "cuda:0", "--steps", "2", "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert p.returncode == 0, p.stdout + "\n" + p.stderr
    data = json.loads(out.read_text())
    seq, joint = data["sequential"], data["joint"]
    assert joint["optimizer_numel"] == 2 * seq["optimizer_numel"]
    assert seq["resident_expert_count"] == 1
    assert joint["resident_expert_count"] == 2
    assert seq["peak_vram_mb"] > 0.0
    assert joint["peak_vram_mb"] >= seq["peak_vram_mb"] - 1.0
    assert seq["peak_vram_mb"] < 24_000
