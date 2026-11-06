#!/usr/bin/env python
"""Arm A (2026-11-05): the gated n-gram table mixed into Qwen at every scored token. For each token the table's probability of the
true token is count(context suffix + token) / count(context suffix) at the longest suffix order the as-of world has seen at least
--cmin times (n_eff; 0 = unseen). The mixture p = lam * p_table + (1 - lam) * p_qwen, with lam a function of the bucket
(n_eff, count of the suffix), fitted by maximum likelihood on one set and scored on the other (and five-fold within a set), never
fitted and scored on the same windows except the labelled in-sample ceiling. Bits per byte = summed bits over the scored windows'
second halves (1,024 bytes each), the same bar as qwen_memory.py. Prints numbers only, never text.
usage: mix.py --tokens tokens.jsonl --index index.npz --out DIR [--cmin 1] [--features features.npz (cached)]"""
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ngram_index import Index, day

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", required=True); ap.add_argument("--index", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cmin", type=int, default=1, help="a suffix order counts as seen at this many occurrences")
ap.add_argument("--features", default=None, help="cache of the table counts per token (built if missing)"); ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.umask(0o077); os.makedirs(a.out, exist_ok=True)
LN2 = math.log(2); HALF = a.T // 2

# ---- the per-token table: counts at every order ------------------------------------------------------------------------------------------
recs = [json.loads(l) for l in open(a.tokens)]
feat_path = a.features or os.path.join(a.out, "features.npz")
if os.path.exists(feat_path):
    z = np.load(feat_path); CTX, TRU, LP, SET, WIN, N = z["ctx"], z["tru"], z["lp"], z["set"], z["win"], int(z["N"])
else:
    ix = Index(a.index); N = ix.N; ctxs = []; trus = []; lps = []; sets = []; wins = []; t0 = time.time()
    for i, r in enumerate(recs):
        c, t = ix.features(np.asarray(r["ctx"]), np.asarray(r["tgt"]), day(r["date"]))
        ctxs.append(c); trus.append(t); lps.append(np.asarray(r["lp"], dtype=np.float64)); sets.append(np.full(len(r["tgt"]), 0 if r["set"] == "kyle" else 1, np.int8)); wins.append(np.full(len(r["tgt"]), i, np.int32))
        if (i + 1) % 100 == 0: print(f"  features: {i + 1} windows ({time.time() - t0:.0f} s)", flush=True)
    CTX = np.concatenate(ctxs, axis=1); TRU = np.concatenate(trus, axis=1); LP = np.concatenate(lps); SET = np.concatenate(sets); WIN = np.concatenate(wins)
    np.savez(feat_path, ctx=CTX, tru=TRU, lp=LP, set=SET, win=WIN, N=N)
nwin = {s: sum(1 for r in recs if r["set"] == s) for s in ("kyle", "dlg")}
print(f"{len(recs)} windows ({nwin}), {len(LP):,} tokens, orders 1..{N}", flush=True)

# ---- n_eff, the table's probability of the true token, the bucket -----------------------------------------------------------------------
seen = CTX >= a.cmin                                                   # [N, k]
n_eff = np.where(seen.any(axis=0), seen.shape[0] - np.argmax(seen[::-1], axis=0), 0)   # the longest seen order (1..N), 0 if none
j = np.arange(len(LP)); cnt = np.where(n_eff > 0, CTX[np.maximum(n_eff - 1, 0), j], 0); tru = np.where(n_eff > 0, TRU[np.maximum(n_eff - 1, 0), j], 0)
p_tab = np.where(cnt > 0, tru / np.maximum(cnt, 1), 0.0); p_lm = np.exp(LP)
NB = [0, 1, 2, 3, 4, 5, 6, 8, 10, 99]; CB = [1, 2, 4, 16, 64, 10**9]              # bucket edges: n_eff bins x count bins
nb = np.searchsorted(np.asarray(NB), n_eff, "right") - 1; cb = np.searchsorted(np.asarray(CB), np.maximum(cnt, 1), "right") - 1   # bin i holds [edge i, edge i+1)
bucket = np.where(n_eff > 0, 1 + nb * len(CB) + cb, 0); NBK = 1 + len(NB) * len(CB)
GRID = np.concatenate([[0.0], np.linspace(0.01, 0.99, 99)])

def fit(mask: np.ndarray) -> np.ndarray:
    """lam per bucket maximising the log-likelihood of the tokens in `mask`."""
    lam = np.zeros(NBK)
    for b in range(1, NBK):
        m = mask & (bucket == b)
        if not m.any(): continue
        ll = [np.log(g * p_tab[m] + (1 - g) * p_lm[m]).sum() for g in GRID]; lam[b] = GRID[int(np.argmax(ll))]
    return lam
def bits(mask: np.ndarray, lam: np.ndarray) -> float:
    g = lam[bucket[mask]]; return float(-np.log(g * p_tab[mask] + (1 - g) * p_lm[mask]).sum() / LN2)
def bpb(mask: np.ndarray, lam) -> float:
    w = np.unique(WIN[mask]); return bits(mask, lam) / (len(w) * HALF)
def base(mask: np.ndarray) -> float:
    w = np.unique(WIN[mask]); return float(-LP[mask].sum() / LN2) / (len(w) * HALF)

out = {"cmin": a.cmin, "orders": N, "buckets": {"n_eff_edges": NB, "count_edges": CB}, "sets": {}}
K, D = SET == 0, SET == 1
for name, m, other in (("kyle", K, D), ("dlg", D, K)):
    r = {"qwen": base(m)}
    lam_o = fit(other); r["mixed_fit_on_other"] = bpb(m, lam_o)
    lam_i = fit(m); r["mixed_in_sample_ceiling"] = bpb(m, lam_i)
    rng = np.random.default_rng(a.seed); wins_ = np.unique(WIN[m]); perm = rng.permutation(wins_); fold_of = {int(w): i % a.folds for i, w in enumerate(perm)}
    fw = np.asarray([fold_of.get(int(w), -1) for w in WIN]); cv_bits = 0.0
    for f in range(a.folds):
        test = m & (fw == f); train = m & (fw != f) & (fw >= 0); cv_bits += bits(test, fit(train))
    r["mixed_cv_within"] = cv_bits / (len(wins_) * HALF)
    for g in (0.05, 0.1, 0.2): r[f"fixed_lam_{g}"] = bpb(m, np.full(NBK, g) * (np.arange(NBK) > 0))
    r["tokens"] = int(m.sum()); r["seen_frac"] = float((n_eff[m] > 0).mean()); r["n_eff_mean_seen"] = float(n_eff[m][n_eff[m] > 0].mean()) if (n_eff[m] > 0).any() else 0
    r["p_tab_true_mean_seen"] = float(p_tab[m][n_eff[m] > 0].mean()) if (n_eff[m] > 0).any() else 0
    r["lam_fit_on_other"] = [round(float(x), 2) for x in lam_o]
    out["sets"][name] = r
    print(f"{name}: Qwen {r['qwen']:.4f}; mixed (gate fitted on the other set) {r['mixed_fit_on_other']:.4f} ({r['mixed_fit_on_other'] - r['qwen']:+.4f}); "
          f"five-fold within {r['mixed_cv_within']:.4f} ({r['mixed_cv_within'] - r['qwen']:+.4f}); in-sample ceiling {r['mixed_in_sample_ceiling']:.4f}; "
          f"fixed lam .05/.1/.2 {r['fixed_lam_0.05']:.4f}/{r['fixed_lam_0.1']:.4f}/{r['fixed_lam_0.2']:.4f}; table saw {r['seen_frac']:.0%} of tokens, "
          f"mean order {r['n_eff_mean_seen']:.1f}, p_table(true) {r['p_tab_true_mean_seen']:.3f}", flush=True)
# where the gain lives: by n_eff bin, on each set with the other set's gate
for name, m, other in (("kyle", K, D), ("dlg", D, K)):
    lam_o = fit(other); rows = []
    for i in range(len(NB) - 1):
        mm = m & (nb == i)
        if not mm.any(): continue
        g = lam_o[bucket[mm]]; d = float((-np.log(g * p_tab[mm] + (1 - g) * p_lm[mm]) + LP[mm]).sum() / LN2)
        rows.append({"n_eff_bin": f"{NB[i]}-{NB[i + 1] - 1}" if i else "unseen", "tokens": int(mm.sum()), "bits_change": round(d, 1), "p_tab_true": round(float(p_tab[mm].mean()), 3), "lam_mean": round(float(g.mean()), 3)})
    out["sets"][name]["by_order"] = rows
    print(f"  {name} by order (other set's gate): " + "; ".join(f"n{r['n_eff_bin']}: {r['tokens']} tok, {r['bits_change']:+.0f} bits, p {r['p_tab_true']:.2f}, lam {r['lam_mean']:.2f}" for r in rows), flush=True)
json.dump(out, open(os.path.join(a.out, f"mix_cmin{a.cmin}.json"), "w"), indent=1)
