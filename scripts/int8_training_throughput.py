"""End-to-end training throughput: does the tuned int8 GEMM make a real training step faster?

The bare-GEMM bench showed the custom Triton int8 kernel hits ~2x bf16. The eager-throughput note
measured fp32 vs the *fp32-effective-weight* ratchet (which materializes the FP weight and so cannot
beat fp32). What was never measured is the **integrated int8 path** (matmul_mode="int8", tuned
kernel in the forward and the two backward GEMMs) running a full training step. This benchmark
times that.

The 2x lives in the compute-bound regime, so it uses the matched 25M config (n_embd 512, 8 layers,
block 256, batch 64 -> M = batch*block = 16384). Random tokens (throughput is data-independent).
Each mode runs in a fresh process for clean Triton autotuning; warmup excludes the autotune/compile
step, then steady-state per-step time is measured with explicit CUDA syncs.

    CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache \
      uv run python scripts/int8_training_throughput.py --modes fp32 bf16 int8
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("runs/int8-throughput")
VOCAB = 65
WARMUP = 8
TIMED = 30


def run_child(mode: str, embd: int, layer: int, head: int, block: int, batch: int,
              checkpointing: bool) -> dict:
    import torch

    from local_ai_training.model import ModelConfig, build_seeded_model
    from local_ai_training.ratchet import RatchetUpdateStats

    device = torch.device("cuda")
    max_code = None if mode == "fp32" else 2
    matmul_mode = "fp32" if mode == "fp32" else mode
    config = ModelConfig(
        vocab_size=VOCAB, block_size=block, n_layer=layer, n_head=head, n_embd=embd,
        dropout=0.0, matmul_mode=matmul_mode, gradient_checkpointing=checkpointing,
    )
    model = build_seeded_model(config, max_code=max_code, seed=1337).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)

    gen = torch.Generator(device="cpu").manual_seed(0)
    tokens = torch.randint(0, VOCAB, (batch, block + 1), generator=gen).to(device)
    inputs, targets = tokens[:, :block], tokens[:, 1:]
    is_ratchet = max_code is not None

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        loss.backward()
        if is_ratchet:
            model.ratchet_update()
        else:
            _ = RatchetUpdateStats(0, 0, 0, 0, 0, 0.0)
        optimizer.step()

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()

    per_step_ms = []
    for _ in range(TIMED):
        torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        per_step_ms.append((time.perf_counter() - start) * 1e3)

    median_ms = statistics.median(per_step_ms)
    tokens_per_step = batch * block
    return {
        "mode": mode,
        "median_ms": median_ms,
        "p10_ms": statistics.quantiles(per_step_ms, n=10)[0],
        "tokens_per_second": tokens_per_step / (median_ms / 1e3),
        "peak_MB": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["fp32", "bf16", "int8"])
    parser.add_argument("--embd", type=int, default=512)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.child:
        print(json.dumps(run_child(args.mode, args.embd, args.layer, args.head, args.block,
                                    args.batch, args.checkpointing)))
        return

    results = []
    for mode in args.modes:
        cmd = [sys.executable, __file__, "--child", "--mode", mode, "--embd", str(args.embd),
               "--layer", str(args.layer), "--head", str(args.head), "--block", str(args.block),
               "--batch", str(args.batch)]
        if args.checkpointing:
            cmd.append("--checkpointing")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = {"mode": mode, "status": "ERROR", "detail": proc.stdout[-400:]}
        results.append(row)
        print(row)

    base = next((r["tokens_per_second"] for r in results
                 if r.get("mode") == "fp32" and "tokens_per_second" in r), None)
    if base:
        for r in results:
            if "tokens_per_second" in r:
                r["speedup_vs_fp32"] = r["tokens_per_second"] / base
    (OUT / "throughput.json").write_text(json.dumps(results, indent=2))
    print("\nDone ->", OUT / "throughput.json")
    for r in results:
        if "tokens_per_second" in r:
            speedup = r.get("speedup_vs_fp32", float("nan"))
            print(f"{r['mode']:>5}: {r['tokens_per_second']:>10,.0f} tok/s  "
                  f"{r['median_ms']:.2f} ms/step  x{speedup:.3f} vs fp32")


if __name__ == "__main__":
    main()
