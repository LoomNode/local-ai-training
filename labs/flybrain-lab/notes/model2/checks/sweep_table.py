#!/usr/bin/env python
"""The sweep's table (2026-10-20): every arm's memory condition paired against the reference run's warm and memory per window. For each set:
the arm's gain (mean, 10 %-trimmed mean, median), its difference from the reference chooser (mean with a bootstrap 95 % interval, windows
better and worse by more than 0.02) and, at the end, the oracle over all arms (the best memory per window: what a perfect reranker over
these candidates would reach). usage: sweep_table.py --ref results/tables/qwen/qwen_memory.json "label=path/qwen_memory.json" ... [--out table.md]"""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--ref", required=True); ap.add_argument("--out", default=None); ap.add_argument("--boot", type=int, default=4000)
ap.add_argument("runs", nargs="+"); a = ap.parse_args()
ref = json.load(open(a.ref)); rng = np.random.default_rng(0)
def tm(x, f=0.1):
    k = int(len(x) * f); y = np.sort(x); return y[k:len(y) - k].mean()
def ci(d):
    m = rng.choice(d, size=(a.boot, len(d)), replace=True).mean(axis=1); return np.percentile(m, 2.5), np.percentile(m, 97.5)
rows = []; best = {}
for spec in a.runs:
    label, path = spec.split("=", 1)
    if not os.path.exists(path): rows.append((label, None)); continue
    R = json.load(open(path)); cells = {}
    for s_ in ("kyle", "dlg"):
        if "memory" not in R[s_]: continue
        new = np.asarray(R[s_]["memory"]["per_window"]); n_ = len(new)
        w = np.asarray(ref[s_]["warm"]["per_window"][:n_]); old = np.asarray(ref[s_]["memory"]["per_window"][:n_])
        if len(w) != n_: continue
        g = new - w; d = new - old; lo, hi = ci(d)
        cells[s_] = (n_, new.mean(), g.mean(), tm(g), np.median(g), d.mean(), lo, hi, int((d < -0.02).sum()), int((d > 0.02).sum()))
        best[s_] = np.minimum(best[s_], new) if s_ in best else new.copy()
    rows.append((label, cells))
wk = np.asarray(ref["kyle"]["warm"]["per_window"]); wd = np.asarray(ref["dlg"]["warm"]["per_window"])
lines = ["| arm | his turns: bits (gain) | trim10 / median | vs the spans (95 % CI) | better / worse | the dialogue: bits (gain) | trim10 / median | vs the spans (95 % CI) | better / worse |",
         "|---|---|---|---|---|---|---|---|---|"]
def fmt(c):
    if c is None: return "not run | | | "
    n_, m, g, t, md, d, lo, hi, b, w = c
    return f"{m:.4f} ({g:+.4f}) | {t:+.4f} / {md:+.4f} | {d:+.4f} [{lo:+.3f}, {hi:+.3f}] | {b} / {w}"
order = sorted(range(len(rows)), key=lambda i: rows[i][1]["kyle"][2] if rows[i][1] and "kyle" in rows[i][1] else 9)
for i in order:
    label, cells = rows[i]
    if cells is None: lines.append(f"| {label} | not run | | | | | | | |"); continue
    lines.append(f"| {label} | " + fmt(cells.get("kyle")) + " | " + fmt(cells.get("dlg")) + " |")
if best:
    lines.append(f"| the oracle over these arms (the best memory per window) | {best['kyle'].mean():.4f} ({best['kyle'].mean() - wk.mean():+.4f}) | | | | {best['dlg'].mean():.4f} ({best['dlg'].mean() - wd.mean():+.4f}) | | | |")
    lines.append(f"| the oracle with warm (skip where every memory hurts) | {np.minimum(best['kyle'], wk).mean():.4f} ({np.minimum(best['kyle'], wk).mean() - wk.mean():+.4f}) | | | | {np.minimum(best['dlg'], wd).mean():.4f} ({np.minimum(best['dlg'], wd).mean() - wd.mean():+.4f}) | | | |")
out = "\n".join(lines); print(out)
if a.out: open(a.out, "w").write(out + "\n")
