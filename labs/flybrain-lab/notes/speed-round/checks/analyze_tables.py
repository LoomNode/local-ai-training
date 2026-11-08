"""Read the two probes' per-window scores and print the comparisons (paired over windows, 95 % interval by the normal approximation)."""
import json
import math
from pathlib import Path

D = Path("/games/ailab/lab/flybrain-lab/.worktrees/speed/results/tables")
WIN = json.load(open("/games/ailab/lab/flybrain-lab/.worktrees/speed/results/mem/feat_ext/windows.json"))["windows"]


def load(name):
    p = D / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def paired(a, b, idx=None):
    """mean(b - a) over windows idx, with a 95 % interval."""
    idx = range(len(a)) if idx is None else idx
    d = [b[i] - a[i] for i in idx]
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1))
    return m, 1.96 * sd / math.sqrt(len(d)), len(d)


print("=== probe 1: the kernel table extended after training with 11.6 MB of unseen kernel files (dev windows, cold bits/byte)")
print(f"{'model':8s} {'base':>7s} {'evict':>8s} {'change':>18s} {'noevict':>8s} {'change':>18s}   change on the 406 windows with same-directory material / the other 618")
related = [r["window"] for r in WIN if r["same_dir_files"] > 0]
other = [r["window"] for r in WIN if r["same_dir_files"] == 0]
for model in ("tf_tab", "zh_tab", "tf_mt", "zh_mt"):
    base = load(f"p1_{model}_feat")
    if base is None:
        continue
    line = f"{model:8s} {base['bpb']:7.4f}"
    for cond in ("feat_ext", "feat_ext_noevict"):
        r = load(f"p1_{model}_{cond}")
        if r is None:
            line += " " * 27; continue
        m, ci, n = paired(base["per_window_cold"], r["per_window_cold"])
        mr, cir, _ = paired(base["per_window_cold"], r["per_window_cold"], related)
        mo, cio, _ = paired(base["per_window_cold"], r["per_window_cold"], other)
        line += f" {r['bpb']:8.4f} {m:+.4f} ({m - ci:+.4f}, {m + ci:+.4f})"
        line += f"   [{cond[5:] or 'evict'}: related {mr:+.4f} ± {cir:.4f} / other {mo:+.4f} ± {cio:.4f}]"
    print(line)
for j, n in enumerate((8, 16, 32)):
    tot = len(WIN) * 2048
    print(f"  order {n}: hits base {sum(r['base_hits'][j] for r in WIN) / tot:.3f}, extended (evict) {sum(r['ext_hits'][j] for r in WIN) / tot:.3f}; "
          f"new {sum(r['new_hits'][j] for r in WIN) / tot:.4f}, lost {sum(r['lost_hits'][j] for r in WIN) / tot:.4f}")

print("\n=== probe 2: CPython windows (cold / warm bits/byte); models trained on the kernel with its table, read with four tables")
print(f"{'model':8s} " + " ".join(f"{c:>15s}" for c in ("none", "kernel", "own", "merged")) + "   (own - none, 95 % interval)")
for model in ("tf_tab", "zh_tab", "tf_mt", "zh_mt"):
    cells = []
    rows = {}
    for cond in ("none", "kernel", "own", "merged"):
        r = load(f"p2_{model}_{cond}")
        rows[cond] = r
        cells.append(f"{r['bpb']:7.3f}/{r['bpb_warm']:.3f}" if r else " " * 15)
    extra = ""
    if rows.get("none") and rows.get("own"):
        m, ci, _ = paired(rows["none"]["per_window_cold"], rows["own"]["per_window_cold"])
        extra = f"   {m:+.4f} ({m - ci:+.4f}, {m + ci:+.4f})"
    print(f"{model:8s} " + " ".join(cells) + extra)
print("plain models on the same windows:")
for model in ("tf_r", "tf_c", "zh_c", "hyb_c"):
    r = load(f"p2_{model}")
    if r:
        print(f"  {model:8s} {r['bpb']:.3f} / {r['bpb_warm']:.3f}")
