#!/usr/bin/env python
"""Arm C (2026-11-06): the Memory Decoder interpolated with a frozen model at the output, p = lam * p_dec + (1 - lam) * p_lm, on the
held-out windows. Both models' tokens.jsonl (same windows, same target tokens) give log p(true) per token; the decoder's record also
gives its entropy, which buckets lam (a confident decoder gets more weight). lam per bucket is fitted on the other set, five-fold within
the set, and in-sample (the ceiling); a single scalar lam too. --since keeps the time split's windows. Numbers only.
usage: mix_decoder.py --dec tokens.jsonl --lm tokens.jsonl --out DIR [--since 2026-09-01] [--tag name]"""
import argparse, json, math, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dec", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True); ap.add_argument("--since", default="")
ap.add_argument("--tag", default="mix"); ap.add_argument("--T", type=int, default=2048); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.umask(0o077); os.makedirs(a.out, exist_ok=True); LN2 = math.log(2); HALF = a.T // 2
def load(p): return {(r["set"], r["w"]): r for r in map(json.loads, open(p)) if not a.since or (r["date"] or "") >= a.since}
A = load(a.dec); Bm = load(a.lm); keys = sorted(k for k in A if k in Bm)
LA, LB, EN, SET, WIN = [], [], [], [], []
for i, k in enumerate(keys):
    ra, rb = A[k], Bm[k]; assert ra["tgt"] == rb["tgt"], f"target tokens differ at {k}"
    LA.append(ra["lp"]); LB.append(rb["lp"]); EN.append(ra["ent"]); SET.append(np.full(len(ra["lp"]), 0 if k[0] == "kyle" else 1, np.int8)); WIN.append(np.full(len(ra["lp"]), i, np.int32))
LA, LB, EN, SET, WIN = (np.concatenate(x) for x in (LA, LB, EN, SET, WIN)); pa, pb = np.exp(LA), np.exp(LB)
EB = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 99]; bucket = np.searchsorted(np.asarray(EB), EN, "right") - 1; NBK = len(EB) - 1
GRID = np.concatenate([[0.0], np.linspace(0.01, 0.99, 99)])
def fit(mask):
    lam = np.zeros(NBK)
    for b in range(NBK):
        m = mask & (bucket == b)
        if not m.any(): continue
        bits = [-np.log(g * pa[m] + (1 - g) * pb[m]).sum() for g in GRID]; lam[b] = GRID[int(np.argmin(bits))]
    return lam
def bits(mask, lam): g = lam[bucket[mask]]; return float(-np.log(g * pa[mask] + (1 - g) * pb[mask]).sum() / LN2)
def bpb(mask, lam): return bits(mask, lam) / (len(np.unique(WIN[mask])) * HALF)
nwin = {s: int(len(np.unique(WIN[SET == i]))) for i, s in enumerate(("kyle", "dlg"))}
out = {"dec": a.dec, "lm": a.lm, "since": a.since, "entropy_edges": EB, "sets": {}}
K, D = SET == 0, SET == 1
for name, m, other in (("kyle", K, D), ("dlg", D, K)):
    r = {"windows": nwin[name], "lm": bpb(m, np.zeros(NBK)), "dec": bpb(m, np.ones(NBK))}
    lam_o = fit(other); r["mixed_fit_on_other"] = bpb(m, lam_o); r["lam_fit_on_other"] = [round(float(x), 2) for x in lam_o]
    r["mixed_in_sample_ceiling"] = bpb(m, fit(m))
    best = min(GRID, key=lambda g: bits(other, np.full(NBK, g))); r["scalar_lam_fit_on_other"] = float(best); r["mixed_scalar"] = bpb(m, np.full(NBK, best))
    rng = np.random.default_rng(a.seed); wins_ = np.unique(WIN[m]); perm = rng.permutation(wins_); fold_of = {int(w): i % a.folds for i, w in enumerate(perm)}
    fw = np.asarray([fold_of.get(int(w), -1) for w in WIN]); cv = 0.0
    for f in range(a.folds):
        test = m & (fw == f); train = m & (fw != f) & (fw >= 0); cv += bits(test, fit(train))
    r["mixed_cv_within"] = cv / (len(wins_) * HALF)
    r["by_bucket"] = [{"entropy": f"{EB[b]}-{EB[b + 1]}", "tokens": int((m & (bucket == b)).sum()), "lam": float(lam_o[b]),
                       "bits_change": round(float((-np.log(lam_o[b] * pa + (1 - lam_o[b]) * pb)[m & (bucket == b)] - (-LB[m & (bucket == b)])).sum() / LN2), 1)} for b in range(NBK) if (m & (bucket == b)).any()]
    out["sets"][name] = r
    print(f"{name} ({nwin[name]} windows): LM {r['lm']:.4f}, decoder alone {r['dec']:.4f}; mixed (lam by entropy, fitted on the other set) {r['mixed_fit_on_other']:.4f} ({r['mixed_fit_on_other'] - r['lm']:+.4f}); "
          f"scalar lam {best:.2f}: {r['mixed_scalar']:.4f}; five-fold within {r['mixed_cv_within']:.4f} ({r['mixed_cv_within'] - r['lm']:+.4f}); in-sample {r['mixed_in_sample_ceiling']:.4f}", flush=True)
json.dump(out, open(os.path.join(a.out, f"{a.tag}.json"), "w"), indent=1)
