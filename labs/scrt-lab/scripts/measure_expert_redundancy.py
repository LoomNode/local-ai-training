#!/usr/bin/env python3
"""Cross-expert redundancy in GLM-5.2's MoE layers.

Intra-block FFN redundancy came back empty at every scale tested: zero neuron pairs
above 0.7 in our 1.45M model, and zero in GLM's 12,288-neuron dense blocks (75M pairs
each). But the origin thread's sharpest example was never intra-block — it was
cross-*expert*:

    Expert A: programming / Expert B: mathematics / Expert C: engineering
    ...all independently learn logic, arithmetic, units, basic syntax.

That is the one place duplication is architecturally *likely*: experts are trained on
disjoint routed token streams and have no mechanism forcing them to share anything.

GLM-5.2 layer 8: 256 experts, 2048 neurons each, model dim 6144.
  ffn_down_exps  IQ4_XS  (4.25 bit) -> output directions, the clean tensor
  ffn_up_exps    IQ2_XXS (2.06 bit) -> input directions, noisy; reported separately

Three measures, all with shuffled controls:

1. **cross-expert neuron twins** — for neuron i in expert a, its best match among
   expert b's 2048 neurons. Same metric as the dense study so numbers are comparable.
2. **output-direction-only twins** — rests solely on IQ4_XS, immune to the 2-bit
   noise in ffn_up.
3. **subspace overlap** — mean cos^2 of principal angles between the column spaces of
   two experts' down matrices. Two random 2048-dim subspaces of R^6144 overlap at
   2048/6144 = 0.333 by chance; meaningfully above that is a shared core, which is
   exactly "store the common part once, spend capacity on residuals".

    python scripts/measure_expert_redundancy.py --layer 8 --experts 16
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

GLM_DIR = Path("/games/ailab/models/GLM-5.2-UD-IQ1_S")


def expert_slices(name: str, n_take: int):
    """Yield (idx, np.ndarray) per expert, dequantising only that expert's bytes."""
    from gguf import GGUFReader
    from gguf.constants import GGML_QUANT_SIZES
    from gguf.quants import dequantize

    for f in sorted(glob.glob(str(GLM_DIR / "*.gguf"))):
        r = GGUFReader(f)
        hits = [t for t in r.tensors if t.name == name]
        if not hits:
            del r
            continue
        t = hits[0]
        ne = [int(x) for x in t.shape]
        bs, ts = GGML_QUANT_SIZES[t.tensor_type]
        n_exp, rows, cols = ne[2], ne[1], ne[0]
        per = rows * cols // bs * ts
        raw = t.data.reshape(-1)
        for e in range(min(n_take, n_exp)):
            arr = dequantize(raw[e * per : (e + 1) * per], t.tensor_type)
            yield e, np.asarray(arr, dtype=np.float32).reshape(rows, cols), t.tensor_type.name
        del r
        return
    raise KeyError(name)


def shuffled(t: torch.Tensor) -> torch.Tensor:
    flat = t.reshape(-1)
    return flat[torch.randperm(flat.numel(), device=t.device)].reshape(t.shape)


def twin_stats(Sa_in, Sa_out, Sb_in, Sb_out, same: bool) -> dict:
    """Best cross-set match per neuron, for combined and output-only metrics."""
    comb = (Sa_in @ Sb_in.T) * (Sa_out @ Sb_out.T)
    outo = (Sa_out @ Sb_out.T).abs()
    if same:
        comb.fill_diagonal_(-2.0)
        outo.fill_diagonal_(0.0)
    best_c = comb.max(dim=1).values
    best_o = outo.max(dim=1).values
    ac = comb.abs()
    return {
        "combined": {
            "pairs_over": {str(t): int((ac >= t).sum().item()) for t in (0.5, 0.7, 0.8)},
            "neurons_with_twin_over": {
                str(t): int((best_c >= t).sum().item()) for t in (0.5, 0.7, 0.8)
            },
            "best_median": float(best_c.median().item()),
            "best_max": float(best_c.max().item()),
        },
        "out_only": {
            "neurons_with_twin_over": {
                str(t): int((best_o >= t).sum().item()) for t in (0.5, 0.7, 0.8, 0.9)
            },
            "best_median": float(best_o.median().item()),
            "best_max": float(best_o.max().item()),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", default="results_redund/expert_redundancy.json")
    args = p.parse_args()

    dev = args.device
    torch.manual_seed(0)
    L, E = args.layer, args.experts

    print(f"loading {E} experts from layer {L} ...", flush=True)
    downs, ups = {}, {}
    qt_down = qt_up = "?"
    for e, arr, qt in expert_slices(f"blk.{L}.ffn_down_exps.weight", E):
        downs[e] = torch.from_numpy(arr).to(dev)  # (6144, 2048)
        qt_down = qt
    for e, arr, qt in expert_slices(f"blk.{L}.ffn_up_exps.weight", E):
        ups[e] = torch.from_numpy(arr).to(dev)  # (2048, 6144)
        qt_up = qt
    print(f"  down {tuple(downs[0].shape)} [{qt_down}]  up {tuple(ups[0].shape)} [{qt_up}]")

    # Per-expert normalised neuron directions.
    n_in, n_out, Q = {}, {}, {}
    for e in downs:
        n_in[e] = F.normalize(ups[e].float(), dim=1)              # (2048, 6144)
        n_out[e] = F.normalize(downs[e].float().T, dim=1)         # (2048, 6144)
        Q[e] = torch.linalg.qr(downs[e].float())[0]               # (6144, 2048)

    res = {"layer": L, "n_experts": E, "quant": {"down": qt_down, "up": qt_up}}

    within = twin_stats(n_in[0], n_out[0], n_in[0], n_out[0], same=True)
    res["within_expert_0"] = within
    print(f"\nwithin expert 0 (control): "
          f"comb twins>=0.5:{within['combined']['neurons_with_twin_over']['0.5']} "
          f"med={within['combined']['best_median']:.3f} "
          f"| out-only >=0.5:{within['out_only']['neurons_with_twin_over']['0.5']} "
          f"med={within['out_only']['best_median']:.3f}")

    pairs = list(itertools.combinations(sorted(downs), 2))
    agg = {"comb_twin05": [], "comb_med": [], "out_twin05": [], "out_twin07": [],
           "out_med": [], "overlap": []}
    for a, b in pairs:
        st = twin_stats(n_in[a], n_out[a], n_in[b], n_out[b], same=False)
        ov = float((Q[a].T @ Q[b]).pow(2).sum().item() / Q[a].shape[1])
        agg["comb_twin05"].append(st["combined"]["neurons_with_twin_over"]["0.5"])
        agg["comb_med"].append(st["combined"]["best_median"])
        agg["out_twin05"].append(st["out_only"]["neurons_with_twin_over"]["0.5"])
        agg["out_twin07"].append(st["out_only"]["neurons_with_twin_over"]["0.7"])
        agg["out_med"].append(st["out_only"]["best_median"])
        agg["overlap"].append(ov)

    # Shuffled control: destroys pairing, keeps the quantised value distribution.
    sa_in, sa_out = F.normalize(shuffled(ups[0]).float(), dim=1), F.normalize(shuffled(downs[0]).float().T, dim=1)
    sb_in, sb_out = F.normalize(shuffled(ups[1]).float(), dim=1), F.normalize(shuffled(downs[1]).float().T, dim=1)
    sh = twin_stats(sa_in, sa_out, sb_in, sb_out, same=False)
    Qs_a = torch.linalg.qr(shuffled(downs[0]).float())[0]
    Qs_b = torch.linalg.qr(shuffled(downs[1]).float())[0]
    sh_ov = float((Qs_a.T @ Qs_b).pow(2).sum().item() / Qs_a.shape[1])
    res["shuffled_control"] = {"twins": sh, "overlap": sh_ov}

    import statistics as st_

    res["cross_expert"] = {
        "n_pairs": len(pairs),
        "comb_twins_over_0.5_mean": st_.fmean(agg["comb_twin05"]),
        "comb_best_median_mean": st_.fmean(agg["comb_med"]),
        "out_twins_over_0.5_mean": st_.fmean(agg["out_twin05"]),
        "out_twins_over_0.7_mean": st_.fmean(agg["out_twin07"]),
        "out_best_median_mean": st_.fmean(agg["out_med"]),
        "overlap_mean": st_.fmean(agg["overlap"]),
        "overlap_min": min(agg["overlap"]),
        "overlap_max": max(agg["overlap"]),
    }
    chance = Q[0].shape[1] / Q[0].shape[0]
    res["overlap_chance"] = chance

    c = res["cross_expert"]
    print(f"\ncross-expert over {len(pairs)} pairs (2048 neurons each):")
    print(f"  combined  twins>=0.5 (mean/pair): {c['comb_twins_over_0.5_mean']:.1f}"
          f"   best-match median {c['comb_best_median_mean']:.3f}")
    print(f"  out-only  twins>=0.5: {c['out_twins_over_0.5_mean']:.1f}"
          f"  >=0.7: {c['out_twins_over_0.7_mean']:.1f}"
          f"   best-match median {c['out_best_median_mean']:.3f}")
    print(f"  shuffled  twins>=0.5: {sh['out_only']['neurons_with_twin_over']['0.5']}"
          f"   best-match median {sh['out_only']['best_median']:.3f}")
    print(f"\n  subspace overlap mean={c['overlap_mean']:.4f} "
          f"(min {c['overlap_min']:.4f} max {c['overlap_max']:.4f})")
    print(f"  chance (2048/6144)   ={chance:.4f}    shuffled={sh_ov:.4f}")
    ratio = c["overlap_mean"] / chance
    print(f"  -> {ratio:.2f}x chance "
          f"{'(SHARED STRUCTURE)' if ratio > 1.15 else '(no shared core)'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
