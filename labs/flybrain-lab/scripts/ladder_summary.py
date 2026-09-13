#!/usr/bin/env python3
"""Width ladder: does the sparse-head gain shrink as the backbone gets wider?"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAT = re.compile(r"ladder_(?P<bb>[a-z]+)_(?P<corpus>[a-z]+)_d(?P<d>\d+)_s(?P<s>\d+)\.json$")


def ladder_table(results_dir: Path, backbone: str, corpus: str, head: str = "sparse_post") -> list[dict]:
    by_d: dict[int, list[dict]] = defaultdict(list)
    for f in sorted(Path(results_dir).glob("ladder_*.json")):
        m = PAT.search(f.name)
        if not m or m["bb"] != backbone or m["corpus"] != corpus:
            continue
        by_d[int(m["d"])].append(json.loads(f.read_text()))
    rows = []
    for d in sorted(by_d):
        plain, headed, cp_plain, cp_head, n_plain, n_head = [], [], [], [], 0, 0
        for rep in by_d[d]:
            arms = {a["arm"]: a for a in rep["arms"]}
            if backbone in arms:
                plain.append(arms[backbone]["val_bpb"]); cp_plain.append(arms[backbone]["val_bpb_copyable"]); n_plain = arms[backbone]["trainable_scalars"]
            if f"{backbone}_{head}" in arms:
                a = arms[f"{backbone}_{head}"]
                headed.append(a["val_bpb"]); cp_head.append(a["val_bpb_copyable"]); n_head = a["trainable_scalars"]
        if not plain or not headed:
            continue
        mean = lambda xs: sum(xs) / len(xs)
        rows.append({"d_model": d, "n_seeds": min(len(plain), len(headed)), "plain": mean(plain), "head": mean(headed),
                     "gap": mean(plain) - mean(headed), "copyable_plain": mean(cp_plain), "copyable_head": mean(cp_head),
                     "gap_copyable": mean(cp_plain) - mean(cp_head), "scalars_plain": n_plain, "scalars_head": n_head,
                     "plain_seeds": plain, "head_seeds": headed})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--backbones", default="mingru,gla")
    p.add_argument("--corpora", default="code,cpp,wiki")
    p.add_argument("--head", default="sparse_post")
    args = p.parse_args()
    for bb in args.backbones.split(","):
        for corpus in args.corpora.split(","):
            rows = ladder_table(Path(args.results), bb, corpus, args.head)
            if not rows:
                continue
            print(f"\n{bb} / {corpus} / +{args.head}")
            print(f"{'d':>5} {'params':>9} {'seeds':>5} {'plain':>7} {'+head':>7} {'gap':>7} {'cp_plain':>8} {'cp_head':>8} {'cp_gap':>7}")
            for r in rows:
                print(f"{r['d_model']:>5} {r['scalars_plain']:>9} {r['n_seeds']:>5} {r['plain']:>7.3f} {r['head']:>7.3f} {r['gap']:>7.3f} "
                      f"{r['copyable_plain']:>8.3f} {r['copyable_head']:>8.3f} {r['gap_copyable']:>7.3f}")


if __name__ == "__main__":
    main()
