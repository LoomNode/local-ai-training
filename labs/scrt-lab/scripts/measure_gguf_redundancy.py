#!/usr/bin/env python3
"""Weight-space FFN redundancy: 1.45M lab model vs GLM-5.2's dense FFN blocks.

Our 1.45M model showed no redundancy on either axis — zero neuron pairs above 0.8,
and rank-truncating the MLP to a 10% param saving cost +0.084 nats (14x seed noise).
The open question is whether that is a property of the architecture or of the scale.

GLM-5.2's first three FFN blocks are dense (the rest are MoE) and, despite the
"UD-IQ1_S" label, are quantised at Q6_K (ffn_down, ~6.6 bit) and Q5_K (gate/up,
~5.5 bit) — the 1-bit tensors are the expert stacks. 5-6 bits is plenty for a
similarity statistic. Each block is 12,288 neurons at model dim 6144: 226M params,
156x our whole model.

Metric (weight-space, no forward pass, so it applies identically to both models):

    neuron i's input direction   w_in_i  = up[i, :]      (what it reads)
    neuron i's output direction  w_out_i = down[:, i]    (what it writes)
    sim(i,j) = cos(w_in_i, w_in_j) * cos(w_out_i, w_out_j)

Two neurons scoring high both read the same thing and write the same thing — the
merge target. This is the weight-space analogue of the activation-based measure used
on the lab model, and the lab model is re-measured here the same way so the two
numbers are comparable.

Controls, both essential:
  * **shuffled** — permute entries within each tensor. Preserves the quantised value
    distribution exactly while destroying any functional pairing, so it absorbs the
    "few distinct levels inflate cosine" artifact that low-bit quantisation creates.
  * **spectrum** — singular values of ffn_down give effective rank, the basis-style
    compressibility that pairwise similarity cannot see.

    python scripts/measure_gguf_redundancy.py --blocks 0 1 2
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

GLM_DIR = Path("/games/ailab/models/GLM-5.2-UD-IQ1_S")


def load_tensor(name: str, files: list[str]) -> np.ndarray:
    from gguf import GGUFReader
    from gguf.quants import dequantize

    for f in files:
        r = GGUFReader(f)
        for t in r.tensors:
            if t.name == name:
                arr = dequantize(t.data, t.tensor_type)
                # gguf reports ne reversed vs row-major; reshape to (rows, cols)
                shape = tuple(int(x) for x in reversed(t.shape))
                return np.asarray(arr, dtype=np.float32).reshape(shape)
        del r
    raise KeyError(name)


def pair_stats(w_in: torch.Tensor, w_out: torch.Tensor, tag: str) -> dict:
    """w_in: (N, C) rows = input dirs. w_out: (N, C) rows = output dirs."""
    A = F.normalize(w_in.float(), dim=1)
    B = F.normalize(w_out.float(), dim=1)
    N = A.shape[0]
    S = (A @ A.T) * (B @ B.T)
    S.fill_diagonal_(-2.0)
    best = S.max(dim=1).values
    absS = S.abs()
    absS.fill_diagonal_(0.0)

    g = torch.Generator(device=S.device).manual_seed(0)
    idx = torch.randint(0, N * N, (2_000_000,), generator=g, device=S.device)
    samp = absS.view(-1)[idx]

    out = {
        "tag": tag,
        "n_neurons": int(N),
        "pairs_over": {
            str(t): int((absS >= t).sum().item() // 2) for t in (0.5, 0.7, 0.8, 0.9)
        },
        "neurons_with_twin_over": {
            str(t): int((best >= t).sum().item()) for t in (0.5, 0.7, 0.8, 0.9)
        },
        "best_partner_median": float(best.median().item()),
        "best_partner_max": float(best.max().item()),
        "abs_p99": float(torch.quantile(samp, 0.99).item()),
        "abs_p999": float(torch.quantile(samp, 0.999).item()),
    }
    del S, absS, A, B
    torch.cuda.empty_cache()
    return out


def spectrum(W: torch.Tensor, tag: str) -> dict:
    s = torch.linalg.svdvals(W.float())
    e = s.pow(2)
    tot = e.sum()
    cum = torch.cumsum(e, 0) / tot
    pr = float((tot.pow(2) / e.pow(2).sum()).item())  # participation ratio
    full = int(min(W.shape))
    return {
        "tag": tag,
        "full_rank": full,
        "participation_ratio": pr,
        "pr_frac": pr / full,
        "rank_50pct_energy": int((cum < 0.50).sum().item()) + 1,
        "rank_90pct_energy": int((cum < 0.90).sum().item()) + 1,
        "rank_99pct_energy": int((cum < 0.99).sum().item()) + 1,
        "frac_rank_for_90pct": (int((cum < 0.90).sum().item()) + 1) / full,
    }


def shuffled(t: torch.Tensor) -> torch.Tensor:
    flat = t.reshape(-1)
    perm = torch.randperm(flat.numel(), device=t.device)
    return flat[perm].reshape(t.shape)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--lab-checkpoint", default="results_redund/baseline_11000k.pt")
    p.add_argument("--out", default="results_redund/gguf_redundancy.json")
    args = p.parse_args()

    dev = args.device
    torch.manual_seed(0)
    results = {"glm": [], "lab": []}

    files = sorted(glob.glob(str(GLM_DIR / "*.gguf")))
    for b in args.blocks:
        print(f"\n=== GLM-5.2 dense block {b} ===", flush=True)
        up = torch.from_numpy(load_tensor(f"blk.{b}.ffn_up.weight", files)).to(dev)
        down = torch.from_numpy(load_tensor(f"blk.{b}.ffn_down.weight", files)).to(dev)
        # up: (N, C) rows are neurons. down: (C, N) columns are neurons.
        if up.shape[0] < up.shape[1]:
            up = up.T.contiguous()
        w_in = up
        w_out = down.T.contiguous() if down.shape[0] < down.shape[1] else down
        print(f"  w_in {tuple(w_in.shape)}  w_out {tuple(w_out.shape)}", flush=True)

        real = pair_stats(w_in, w_out, f"glm_blk{b}")
        ctrl = pair_stats(shuffled(w_in), shuffled(w_out), f"glm_blk{b}_shuffled")
        spec = spectrum(down, f"glm_blk{b}_down")
        spec_c = spectrum(shuffled(down), f"glm_blk{b}_down_shuffled")
        results["glm"].append(
            {"block": b, "real": real, "shuffled": ctrl, "spectrum": spec,
             "spectrum_shuffled": spec_c}
        )
        for r in (real, ctrl):
            print(f"  {r['tag']:26} twins>=0.5:{r['neurons_with_twin_over']['0.5']:>6} "
                  f">=0.7:{r['neurons_with_twin_over']['0.7']:>5} "
                  f">=0.8:{r['neurons_with_twin_over']['0.8']:>5} "
                  f"| med_best={r['best_partner_median']:.3f} "
                  f"max={r['best_partner_max']:.3f} p99.9={r['abs_p999']:.3f}")
        for s in (spec, spec_c):
            print(f"  {s['tag']:26} full={s['full_rank']} "
                  f"rank@90%={s['rank_90pct_energy']} ({s['frac_rank_for_90pct']:.2f}) "
                  f"PR={s['participation_ratio']:.0f} ({s['pr_frac']:.2f})")
        del up, down, w_in, w_out
        torch.cuda.empty_cache()

    # Same metric on the lab model, so the numbers are comparable.
    ck = Path(args.lab_checkpoint)
    if ck.exists():
        from dataclasses import replace
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scrt.budget import build_matched_models, unwrap
        from scrt.config import capacity_config

        cfg = capacity_config(device="cpu")
        holder, _mc, _ = build_matched_models(
            replace(cfg.model, vocab_size=256), exact=True
        )["baseline"]
        holder.load_state_dict(
            torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
        )
        m = unwrap(holder)
        print(f"\n=== lab baseline (1.45M) — same weight-space metric ===")
        for i, blk in enumerate(m.blocks):
            w_in = blk.mlp.fc.weight.data.to(dev)          # (4C, C)
            w_out = blk.mlp.proj.weight.data.T.to(dev)     # (4C, C)
            real = pair_stats(w_in, w_out, f"lab_blk{i}")
            ctrl = pair_stats(shuffled(w_in), shuffled(w_out), f"lab_blk{i}_shuffled")
            spec = spectrum(blk.mlp.proj.weight.data.to(dev), f"lab_blk{i}_down")
            results["lab"].append(
                {"block": i, "real": real, "shuffled": ctrl, "spectrum": spec}
            )
            for r in (real, ctrl):
                print(f"  {r['tag']:26} twins>=0.5:{r['neurons_with_twin_over']['0.5']:>6} "
                      f">=0.7:{r['neurons_with_twin_over']['0.7']:>5} "
                      f">=0.8:{r['neurons_with_twin_over']['0.8']:>5} "
                      f"| med_best={r['best_partner_median']:.3f} "
                      f"max={r['best_partner_max']:.3f} p99.9={r['abs_p999']:.3f}")
            print(f"  {spec['tag']:26} full={spec['full_rank']} "
                  f"rank@90%={spec['rank_90pct_energy']} "
                  f"({spec['frac_rank_for_90pct']:.2f}) "
                  f"PR={spec['participation_ratio']:.0f} ({spec['pr_frac']:.2f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
