#!/usr/bin/env python
"""Per-window score files (score_tables.py's k_*.json: per_window_cold / per_window_warm in window order) re-read on the clean windows
of a leak mask (leak_scan.py's json: sets[<set>]["clean"], window indices). For each file: all windows, the clean windows, the touched
windows, cold and warm, and the paired difference against a reference file on the same set when given. Counts and bits only.
usage: leak_table.py --mask results/tables/leak/leak_windows_talkB.json --set kyle FILE[=REF] ...   (the set name: kyle | dialogue)"""
import argparse, json, os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mask", required=True); ap.add_argument("--set", required=True); ap.add_argument("--min-covered", type=int, default=1, help="a window counts as touched from this many covered bytes")
ap.add_argument("files", nargs="+", help="k_*.json, or k_*.json=k_ref.json for a paired difference")
a = ap.parse_args()
m = json.load(open(a.mask))["sets"][a.set]
cov = np.zeros(int(m["windows"])); 
for w, c in m["covered_bytes"].items(): cov[int(w)] = c
touched = cov >= a.min_covered; clean = ~touched
print(f"{a.set}: {int(m['windows'])} windows, {int(touched.sum())} touched (covered bytes >= {a.min_covered}), {int(clean.sum())} clean")


def load(p):
    d = json.load(open(p)); return np.asarray(d["per_window_cold"]), np.asarray(d["per_window_warm"]), d


for spec in a.files:
    f, _, ref = spec.partition("=")
    c, w, d = load(f); n = len(c)
    assert n == len(cov), f"{f}: {n} windows, the mask has {len(cov)}"
    line = f"{os.path.basename(f):28s} all cold {c.mean():.4f} warm {w.mean():.4f} | clean cold {c[clean].mean():.4f} warm {w[clean].mean():.4f} | touched cold {c[touched].mean() if touched.any() else float('nan'):.4f} warm {w[touched].mean() if touched.any() else float('nan'):.4f}"
    if ref:
        rc, rw, _ = load(ref); dc, dw = c - rc, w - rw
        line += f" | vs {os.path.basename(ref)}: all {dc.mean():+.4f}/{dw.mean():+.4f} clean {dc[clean].mean():+.4f}/{dw[clean].mean():+.4f} touched {dc[touched].mean() if touched.any() else float('nan'):+.4f}/{dw[touched].mean() if touched.any() else float('nan'):+.4f}"
    print(line)
