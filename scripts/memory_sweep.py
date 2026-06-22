"""Training-peak memory scaling sweep: fp32 (nn.Linear+Adam) vs bf16-ratchet vs int8-ratchet.

Sweeps model width/depth with a tiny batch/block so WEIGHTS dominate memory (the regime where
the ratchet's 1-byte state matters), and records the peak CUDA memory of a *completed* training
step under each mode, plus the audited persistent footprint. Pushes each mode until OOM.

Two comparisons matter and the writeup must keep them distinct:
  * fp32 (nn.Linear + full AdamW) vs ratchet  -> storage + optimizer-state win (1 byte vs ~12).
  * bf16-ratchet vs int8-ratchet              -> both 1-byte persistent; isolates the matmul
                                                 materialization win this branch built.

Hardening vs the first attempt: run >=3 steps and only trust a peak from a run that REACHED the
final step (the old int8@2048 number was a step-0 row from an interrupted run and was invalid,
since max_memory_allocated is a cumulative peak that only reflects the backward once a full step
has completed).
"""

import csv
import json
import subprocess
from pathlib import Path

SWEEP_DIR = Path("runs/sweep")
STEPS = 3  # need >=2 completed steps so the cumulative peak reflects a full backward
VOCAB = 65
CODES = 5  # max_code 2 (quinary)

SIZES = [
    (512, 8, 8),
    (1024, 12, 16),
    (2048, 16, 16),
    (3072, 24, 24),
    (4096, 32, 32),
    (5120, 40, 40),
    (6144, 48, 48),
]
MODES = ["fp32", "bf16", "int8"]


def write_config(n_embd, n_layer, n_head, matmul_mode) -> str:
    cfg = f"""
[model]
block_size = 32
n_layer = {n_layer}
n_head = {n_head}
n_embd = {n_embd}
dropout = 0.0

[ratchet]
pressure_threshold = 8
bucket_low = 0.5
bucket_high = 1.5

[training]
batch_size = 2
steps = {STEPS}
eval_interval = 1
eval_batches = 1
support_learning_rate = 0.0003
seeds = [1337]
device = "cuda"
matmul_mode = "{matmul_mode}"
gradient_checkpointing = true
"""
    path = SWEEP_DIR / "temp_sweep.toml"
    path.write_text(cfg)
    return str(path)


def audit_footprint(cfg_path):
    out = subprocess.check_output(
        ["uv", "run", "lat", "audit", "--model", cfg_path, "--codes", str(CODES),
         "--vocab-size", str(VOCAB)],
        text=True,
    )
    d = json.loads(out)
    pf = d["persistent_footprint"]
    return {
        "ratchet_state_MB": d["ratchet_state_bytes"] / 1024**2,
        "fp32_matrix_MB": pf["fp32_matrix_bytes"] / 1024**2,
        "fp32_master_plus_opt_MB": (pf["fp32_master_bytes"] + pf["fp32_optimizer_bytes"]) / 1024**2,
        "reduction_ratio": pf["reduction_ratio"],
    }


def peak_of_completed_run(run_dir) -> float | str:
    """Return peak CUDA MB only if the run reached the final step; else 'INCOMPLETE'."""
    metrics = Path(run_dir) / "metrics.csv"
    if not metrics.exists():
        return "OOM"
    rows = list(csv.DictReader(metrics.open()))
    if not rows or int(rows[-1]["step"]) < STEPS - 1:
        return "INCOMPLETE"  # crashed/killed before a full step's backward was captured
    # max_memory_allocated is cumulative; the last row holds the true peak
    return max(float(r["cuda_memory_bytes"]) for r in rows) / 1024**2


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    stop_modes = set()  # once a mode OOMs, larger sizes will too -> skip
    for embd, layer, head in SIZES:
        row = {"n_embd": embd, "n_layer": layer, "n_head": head}
        try:
            row.update(audit_footprint(write_config(embd, layer, head, "fp32")))
        except Exception as e:  # noqa: BLE001
            print(f"audit failed @ {embd}: {e}")
        for mode in MODES:
            if mode in stop_modes:
                row[f"peak_MB_{mode}"] = "skip(OOM)"
                continue
            cfg = write_config(embd, layer, head, mode)
            run_dir = SWEEP_DIR / f"mode_{mode}_embd_{embd}"
            cmd = ["uv", "run", "lat", "train", "--config", cfg, "--codes", str(CODES),
                   "--output", str(run_dir)]
            if mode == "fp32":
                cmd += ["--weight-mode", "fp32"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            peak = peak_of_completed_run(run_dir)
            if proc.returncode != 0 and peak in ("OOM", "INCOMPLETE"):
                oom = "out of memory" in proc.stdout.lower()
                peak = "OOM" if oom else "ERROR"
                if oom:
                    stop_modes.add(mode)
                print(f"{mode}@{embd}: {peak}\n{proc.stdout[-400:]}")
            row[f"peak_MB_{mode}"] = peak
        results.append(row)
        print(row)
        (SWEEP_DIR / "sweep_results.json").write_text(json.dumps(results, indent=2))
    print("\nDone ->", SWEEP_DIR / "sweep_results.json")


if __name__ == "__main__":
    main()
