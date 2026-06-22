"""Settle whether the int8 OOM at 3072-width is a training limit or an observability artifact.

The corrected sweep (docs/results) shows int8 OOMs at 3072 while bf16 completes, reproducing the
historical claim that justified Direction 2. But bf16's *training* peak at 3072 is only ~3 GiB while
its *observability* peak (collect_ratchet_metrics) is ~23 GiB. This probe re-runs the real train_run
path with collect_ratchet_metrics neutralised, so the only thing that can OOM is the training step
itself. If int8 now completes, the historical OOM was the observability spike, not training.

Run one mode per fresh child process (CUDA peak isolation):

    CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache \
      uv run python scripts/int8_3072_observability_probe.py --modes bf16 int8 --embd 3072
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MIB = 1024**2
OUT = Path("runs/int8-3072-probe")


def run_child(mode: str, embd: int, layer: int, head: int) -> dict:
    import torch

    import local_ai_training.train as train_module
    from local_ai_training.config import ExperimentConfig
    from local_ai_training.data import build_char_corpus

    # Neutralise the observability spike: collect_ratchet_metrics is the ~7x high-water mark.
    train_module.collect_ratchet_metrics = lambda model: {}

    corpus = build_char_corpus("abcde" * 4000)
    config = ExperimentConfig(
        block_size=32,
        batch_size=2,
        n_layer=layer,
        n_head=head,
        n_embd=embd,
        steps=3,
        eval_interval=1,
        eval_batches=1,
        support_learning_rate=3e-4,
        pressure_threshold=8,
        seeds=(1337,),
        device="cuda",
        matmul_mode=mode,
        gradient_checkpointing=True,
    )
    run_dir = OUT / f"mode_{mode}_embd_{embd}"
    try:
        result = train_module.train_run(
            corpus=corpus, config=config, max_code=2, seed=1337, run_dir=run_dir
        )
        import csv

        rows = list(csv.DictReader(result.metrics_csv.open()))
        train_peak = max(float(r["cuda_train_peak_bytes"]) for r in rows) / MIB
        obs_peak = max(float(r["cuda_observability_peak_bytes"]) for r in rows) / MIB
        return {"mode": mode, "status": "completed", "train_peak_MB": train_peak,
                "obs_peak_MB": obs_peak}
    except torch.cuda.OutOfMemoryError as exc:  # noqa: PERF203
        return {"mode": mode, "status": "OOM", "detail": str(exc)[:200]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["bf16", "int8"])
    parser.add_argument("--embd", type=int, default=3072)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--head", type=int, default=24)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.child:
        print(json.dumps(run_child(args.mode, args.embd, args.layer, args.head)))
        return

    results = []
    for mode in args.modes:
        proc = subprocess.run(
            [sys.executable, __file__, "--child", "--mode", mode, "--embd", str(args.embd),
             "--layer", str(args.layer), "--head", str(args.head)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = {"mode": mode, "status": "ERROR", "detail": proc.stdout[-300:]}
        results.append(row)
        print(row)
    (OUT / "probe_results.json").write_text(json.dumps(results, indent=2))
    print("\nDone ->", OUT / "probe_results.json")


if __name__ == "__main__":
    main()
