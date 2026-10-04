#!/usr/bin/env python3
"""How many bits/byte does a table written in one pass over the training bytes, with no gradient, add to a trained model? CPU only.

Stage `score`: the per-byte loss (nats, fp32, no autocast) of each checkpoint on the yardstick windows, exactly the number
cross_eval.evaluate_fixed averages, kept per byte under results/mem/loss/ so the mixing can be redone without the model. Windows come
in blocks; several worker processes may run at once, each claiming blocks (an exclusive lock file per block), and every block is
saved as soon as it is done, so a stopped run keeps what it scored. Blocks go round-robin over (split, checkpoint) so that every run
is complete up to the same number of windows at any moment.

Stage `analyse`: the memory's lookups (scripts/build_corpus_memory.py) at the same positions, mixed with the model by a rule fitted on
dev, scored on test. The rule is in the comment above in_window() and in the docstrings of mix_bits() and analyse_checkpoint()."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.corpus_memory import ORDERS
from flybrain.data.files import TRAIN_END, read_region

RESULTS = ROOT / "results"
CORPUS = ROOT / "corpora" / "linux_full.bytes"
SPLITS = {"dev": 202_117_357, "test": 211_576_468}      # the first N_WINDOWS non-overlapping windows from each offset
CKPTS = {"hybrid": "spine16_hybrid_s0_mingru_sparse_post", "transformer": "spine16_tf_s0_transformer", "zeroheads": "spine16_zeroheads_s0_mingru"}
T = 2048
N_WINDOWS = 1024
BLOCK = 64                                              # windows per scoring block (a multiple of the batch)
BATCH = 8192 // T                                       # windows per forward, as cross_eval scores them
LN2 = math.log(2.0)


# ---- stage score ----------------------------------------------------------------------------------------------------------

def load_windows(corpus, offset: int, n: int, seq_len: int = T):
    """(x, y): the n fixed non-overlapping windows from byte `offset`, as flybrain.data.text.fixed_windows cuts them
    (x[i] = bytes [offset + i * T, offset + (i + 1) * T), y[i] one byte ahead); torch long tensors [n, T]."""
    import torch
    from flybrain.data.text import fixed_windows
    raw = read_region(corpus, offset, offset + n * seq_len + 1)
    return fixed_windows(torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).astype(np.int64)), seq_len, n)


def load_model(ckpt_dir: Path, name: str):
    import torch
    from scripts.cross_eval import rebuild
    ck = torch.load(Path(ckpt_dir) / f"{name}.pt", map_location="cpu", weights_only=False)
    return rebuild(ck).eval()


def score_per_byte(model, x, y, batch: int = BATCH) -> np.ndarray:
    """Per-byte cross entropy in nats, float32 [n, T]: the call cross_eval.evaluate_fixed makes (fp32 logits, no autocast), kept per byte."""
    import torch
    from torch.nn import functional as F
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            logits, _ = model(x[i:i + batch])
            rows.append(F.cross_entropy(logits.reshape(-1, 256), y[i:i + batch].reshape(-1), reduction="none").view(-1, x.size(1)).float().numpy())
    return np.concatenate(rows)


def block_path(loss_dir: Path, name: str, split: str, b: int) -> Path:
    return Path(loss_dir) / f"{name}_{split}_b{b:03d}.npy"


def load_loss(loss_dir: Path, name: str, split: str, n_windows: int, block: int | None = None) -> np.ndarray:
    """Per-byte nats [n_windows, T] of the first n_windows windows, from their blocks (every one must exist)."""
    block = block or BLOCK
    if n_windows % block:
        raise ValueError(f"n_windows {n_windows} is not a multiple of the block size {block}")
    missing = [b for b in range(n_windows // block) if not block_path(loss_dir, name, split, b).exists()]
    if missing:
        raise FileNotFoundError(f"{name}/{split}: blocks {missing} are not scored yet")
    return np.concatenate([np.load(block_path(loss_dir, name, split, b)) for b in range(n_windows // block)])


def complete_windows(loss_dir: Path, names, splits, n_blocks: int) -> int:
    """How many windows (a multiple of BLOCK) every (checkpoint, split) has scored from window 0."""
    done = n_blocks
    for name in names:
        for split in splits:
            k = 0
            while k < n_blocks and block_path(loss_dir, name, split, k).exists():
                k += 1
            done = min(done, k)
    return done * BLOCK


def run_worker(args) -> None:
    import torch
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:                                         # already fixed in this process (a second call in one interpreter)
        pass
    loss_dir = Path(args.loss_dir)
    (loss_dir / "claims").mkdir(parents=True, exist_ok=True)
    names, splits = args.ckpts.split(","), args.splits.split(",")
    n_windows = args.windows or N_WINDOWS
    n_blocks = n_windows // BLOCK
    windows = {s: load_windows(args.corpus, SPLITS[s], n_windows, T) for s in splits}
    models: dict = {}
    for b in range(n_blocks):
        for s in splits:
            for name in names:
                out = block_path(loss_dir, name, s, b)
                claim = loss_dir / "claims" / (out.stem + ".lock")
                if out.exists():
                    continue
                try:
                    os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                except FileExistsError:
                    continue                                             # another worker has it
                if name not in models:
                    models[name] = load_model(args.ckpt_dir, CKPTS[name])
                x, y = windows[s]
                t0 = time.perf_counter()
                ce = score_per_byte(models[name], x[b * BLOCK:(b + 1) * BLOCK], y[b * BLOCK:(b + 1) * BLOCK])
                secs = time.perf_counter() - t0
                tmp = out.with_name(out.stem + ".tmp.npy")
                np.save(tmp, ce)
                os.replace(tmp, out)
                out.with_suffix(".json").write_text(json.dumps({"seconds": secs, "threads": args.threads, "loadavg": os.getloadavg()[0], "windows": BLOCK}) + "\n")
                print(f"{name:11s} {s:4s} block {b:2d} ({(b + 1) * BLOCK:4d} windows)  {secs:6.1f} s  load {os.getloadavg()[0]:5.1f}  bpb {float(ce.mean()) / LN2:.4f}", flush=True)


# ---- stage analyse: features, buckets, the mixture, its fit, the intervals --------------------------------------------------

# The rule. A scored byte (window position t, true byte y) has, at each order n, the memory's lookup (byte m_n, confidence c_n; 0 = miss). A
# lookup counts only if its whole n-byte context lies inside the model's window (t + 1 >= n; see in_window). The longest order with a hit
# gives the byte m and the confidence c, and the bucket b = (that order, class of c: 1 | 2-3 | 4-15 | 16+); no hit is its own bucket.
#   p_mix(y) = (1 - lambda_b) p_model(y) + lambda_b [m == y],  lambda = 0 without a hit.
# lambda_b maximises the mean log2 p_mix over the dev bytes of bucket b (a concave one-dimensional problem, fit_lambda); nothing else is
# fitted, and test only ever sees the dev weights.
CONF_EDGES = (2, 4, 16)                                  # confidence classes: 1 | 2-3 | 4-15 | 16+
CONF_LABELS = ("1", "2-3", "4-15", "16+")
LAM_MAX = 0.999                                          # a weight of exactly 1 would put zero probability on every miss of a bucket that never missed on dev
N_BOOT = 1000
CLUSTER = 8                                              # windows per cluster of the extra bootstrap (8 x 2,048 bytes ~ one file)


def bucket_labels(orders=ORDERS) -> list[str]:
    return [f"n={n} conf {c}" for n in orders for c in CONF_LABELS] + ["no hit"]


def in_window(conf: np.ndarray, orders=ORDERS) -> np.ndarray:
    """conf [n_orders, W, T] of the raw lookups at the scored bytes -> a copy with the hits whose context starts before the model's window
    set to misses. Scored byte t of a window is predicted from x[0..t], at absolute position p = (window start) + t + 1; its order-n
    context starts at p - n, which is not before the window's first byte iff t + 1 >= n."""
    out = conf.copy()
    t1 = np.arange(1, conf.shape[2] + 1)
    for k, n in enumerate(orders):
        out[k][:, t1 < n] = 0
    return out


def assign_buckets(pred: np.ndarray, conf: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pred, conf [n_orders, W, T] (orders ascending, after in_window), y [W, T] the true bytes -> (bucket, right), both [W, T].
    bucket = 4 * (index of the longest order with a hit) + confidence class of that hit, or 4 * n_orders for no hit;
    right = the longest hit's byte is the true byte (False without a hit)."""
    n_orders = conf.shape[0]
    hit = conf > 0
    any_hit = hit.any(0)
    longest = n_orders - 1 - np.argmax(hit[::-1], axis=0)
    c = np.take_along_axis(conf, longest[None], 0)[0]
    pr = np.take_along_axis(pred, longest[None], 0)[0]
    cls = np.searchsorted(np.array(CONF_EDGES), c, side="right")
    return np.where(any_hit, 4 * longest + cls, 4 * n_orders).astype(np.int8), any_hit & (pr == y)


def fit_lambda(a: np.ndarray, right: np.ndarray, lam_max: float = LAM_MAX) -> float:
    """The weight in [0, lam_max] that maximises the mean of log((1 - lam) a + lam right), a = the model's probability of the true byte and
    right in {0, 1} (the memory's byte is the true byte). The objective is a sum of logs of affine functions of lam, so it is concave:
    its derivative falls monotonically and bisection finds the root. 0 when the memory never helps (or there is nothing to fit)."""
    a = np.maximum(np.asarray(a, dtype=np.float64), 1e-300)
    r = np.asarray(right, dtype=bool)
    if a.size == 0:
        return 0.0
    n_miss, a_hit = int((~r).sum()), a[r]
    d_hit = 1.0 - a_hit

    def slope(lam: float) -> float:                       # d/dlam of the summed log-likelihood
        return float((d_hit / (a_hit + lam * d_hit)).sum()) - n_miss / (1.0 - lam)

    if slope(0.0) <= 0.0:
        return 0.0
    if slope(lam_max) >= 0.0:
        return lam_max
    lo, hi = 0.0, lam_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if slope(mid) > 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def fit_lambdas(ce: np.ndarray, right: np.ndarray, bucket: np.ndarray, n_buckets: int, mask: np.ndarray | None = None) -> np.ndarray:
    """One weight per bucket from per-byte nats `ce`; the last bucket (no hit) stays 0. `mask` picks the bytes the fit may see."""
    keep = np.ones(ce.shape, dtype=bool) if mask is None else mask
    a = np.exp(-ce.astype(np.float64))
    lam = np.zeros(n_buckets)
    for b in range(n_buckets - 1):
        sel = keep & (bucket == b)
        lam[b] = fit_lambda(a[sel], right[sel])
    return lam


def mix_bits(ce: np.ndarray, right: np.ndarray, bucket: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Bits of the true byte under p_mix = (1 - lam_b) p_model + lam_b [the memory's byte is the true byte], from per-byte nats `ce`.
    Where lam_b = 0 the result is the model's own bits, bit for bit."""
    ce = ce.astype(np.float64)
    lam_b = lam[bucket]
    p = (1.0 - lam_b) * np.exp(-ce) + lam_b * right
    return np.where(lam_b > 0.0, -np.log2(np.where(lam_b > 0.0, p, 1.0)), ce / LN2)


def bootstrap_gain(gain_by_window: np.ndarray, bytes_per_window: int, n_boot: int = N_BOOT, seed: int = 0, block: int = 1) -> list[float]:
    """95 % interval of the mean gain per byte, resampling windows with replacement (a paired bootstrap: each window's gain is the
    difference of its own sums, alone minus mixed). block > 1 resamples runs of `block` consecutive windows instead (neighbouring windows share
    files, so they are not independent: a cluster bootstrap is the more honest interval); the windows must divide into whole blocks."""
    w = len(gain_by_window)
    if w % block:
        raise ValueError(f"{w} windows do not divide into blocks of {block}")
    units = gain_by_window.reshape(w // block, block).sum(1)
    idx = np.random.default_rng(seed).integers(0, len(units), size=(n_boot, len(units)))
    g = units[idx].sum(1) / (w * bytes_per_window)
    return [float(v) for v in np.percentile(g, [2.5, 97.5])]


REGIMES = {"cold": lambda t: slice(0, t), "warm": lambda t: slice(t // 2, t)}   # cold: every byte of the window; warm: its second half


def evaluate(ce: np.ndarray, right: np.ndarray, bucket: np.ndarray, lam: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Alone and mixed bits/byte, cold and warm, with the paired-bootstrap interval of the gain, over the windows given (rows)."""
    w, t = ce.shape
    alone, mixed = ce.astype(np.float64) / LN2, mix_bits(ce, right, bucket, lam)
    out: dict = {"n_windows": int(w)}
    for name, sl in ((k, f(t)) for k, f in REGIMES.items()):
        n = sl.stop - sl.start
        a_w, m_w = alone[:, sl].sum(1), mixed[:, sl].sum(1)
        out[name] = {"alone": float(a_w.sum() / (w * n)), "mixed": float(m_w.sum() / (w * n)), "gain": float((a_w - m_w).sum() / (w * n)),
                     "gain_ci95": bootstrap_gain(a_w - m_w, n, n_boot, seed)}
        if w % CLUSTER == 0 and w // CLUSTER >= 2:                      # an extra, not asked for: the same interval over runs of CLUSTER neighbouring windows
            out[name]["gain_ci95_clusters"] = bootstrap_gain(a_w - m_w, n, n_boot, seed, block=CLUSTER)
    return out


def bucket_table(ce: np.ndarray, right: np.ndarray, bucket: np.ndarray, lam: np.ndarray, regime: str, orders=ORDERS) -> list[dict]:
    """Per bucket, over the bytes of a regime: share of the bytes, how often the memory's byte is right, lambda, the model's and the mixture's
    bits/byte there, and the bucket's contribution to the gain (bits per byte of the whole regime: the contributions add up to the gain)."""
    sl = REGIMES[regime](ce.shape[1])
    ce, right, bucket = ce[:, sl], right[:, sl], bucket[:, sl]
    alone, mixed = ce.astype(np.float64) / LN2, mix_bits(ce, right, bucket, lam)
    n_all, rows = bucket.size, []
    for b, label in enumerate(bucket_labels(orders)):
        m = bucket == b
        n = int(m.sum())
        rows.append({"bucket": label, "bytes": n, "share": n / n_all, "memory_right": float(right[m].mean()) if n else None, "lambda": float(lam[b]),
                     "alone_bits": float(alone[m].mean()) if n else None, "mixed_bits": float(mixed[m].mean()) if n else None,
                     "gain_contribution": float((alone[m].sum() - mixed[m].sum()) / n_all)})
    return rows


class SplitFeatures:
    """The memory's view of one split's windows: the window rule applied, every scored byte bucketed, the memory's byte checked against the truth."""

    def __init__(self, name: str, y: np.ndarray, pred: np.ndarray, conf: np.ndarray, orders=ORDERS):
        self.name, self.y, self.orders = name, y, tuple(orders)
        self.raw_conf = conf
        self.conf = in_window(conf, self.orders)
        self.bucket, self.right = assign_buckets(pred, self.conf, y)


def memory_features(mem_dir: Path, offset: int, n_windows: int, orders=ORDERS, seq_len: int = T) -> tuple[np.ndarray, np.ndarray]:
    """(pred, conf) uint8 [n_orders, n_windows, T]: the stored lookups at the windows' scored positions. Byte t of window i is predicted at
    absolute position offset + i * T + t + 1, and the memory's files are indexed by absolute position."""
    info = json.loads((Path(mem_dir) / "build.json").read_text())
    lo, hi = offset + 1, offset + 1 + n_windows * seq_len
    if lo < info["valid_from"]:
        raise ValueError(f"positions from {lo} are training bytes: the memory only holds lookups from byte {info['valid_from']}")
    pred = np.stack([np.load(Path(mem_dir) / f"pred_n{n}.npy", mmap_mode="r")[lo:hi].reshape(n_windows, seq_len) for n in orders])
    conf = np.stack([np.load(Path(mem_dir) / f"conf_n{n}.npy", mmap_mode="r")[lo:hi].reshape(n_windows, seq_len) for n in orders])
    return pred, conf


def corpus_files(corpus) -> tuple[list, list[str]]:
    """The corpus's file table and the content hash of every file body (flybrain.data.files)."""
    from flybrain.data.files import file_table, hash_table
    table = file_table(corpus)
    return table, hash_table(corpus, table)


def duplicate_windows(table: list, hashes: list[str], offset: int, n_windows: int, seq_len: int = T, train_end: int = TRAIN_END) -> dict:
    """Which windows overlap a held-out file whose content hash also occurs among the training files. A training file lies entirely inside
    [0, train_end); every other file is held out (the one straddling train_end included). A window covers bytes [start, start + T + 1):
    the T it reads and the last byte it predicts. A file's extent is its header line and body."""
    train_hashes = {h for e, h in zip(table, hashes) if e.end <= train_end}
    held = [(e, h) for e, h in zip(table, hashes) if e.end > train_end]
    twins = [e for e, h in held if h in train_hashes]
    starts = offset + seq_len * np.arange(n_windows)
    ends = starts + seq_len + 1
    hit = np.zeros(n_windows, dtype=bool)
    touching = 0
    for e in twins:
        o = (starts < e.end) & (ends > e.start)
        hit |= o
        touching += int(o.any())
    return {"overlaps": hit, "training_files": len(table) - len(held), "heldout_files": len(held), "heldout_files_with_a_training_twin": len(twins),
            "twins_touching_these_windows": touching}


def committed_check(out_dir: Path, split: str, ckpt_file: str, ce: np.ndarray) -> dict | None:
    """Our cold and warm bits/byte against the committed fp32 scores: the full 1,024 windows against results/spine16_score_<split>_s0.json, or,
    for dev, the first 128 windows against spine16_score_dev128_s0.json. None when no committed number covers the windows scored."""
    n = ce.shape[0]
    if n == N_WINDOWS:
        path = f"spine16_score_{split}_s0.json"
    elif split == "dev" and n >= 128:
        path, ce = "spine16_score_dev128_s0.json", ce[:128]
    else:
        return None
    if not (Path(out_dir) / path).exists():
        return None
    row = next(r for r in json.loads((Path(out_dir) / path).read_text())["rows"] if r["ckpt"] == ckpt_file)
    mine = {"cold": float(ce.astype(np.float64).mean() / LN2), "warm": float(ce[:, ce.shape[1] // 2:].astype(np.float64).mean() / LN2)}
    theirs = {"cold": float(row["bpb"]), "warm": float(row["bpb_warm"])}
    return {"file": path, "n_windows": int(row["n_windows"]), "committed": theirs, "ours": mine, "abs_diff": {k: abs(mine[k] - theirs[k]) for k in mine},
            "equal_to_3_decimals": all(round(mine[k], 3) == round(theirs[k], 3) for k in mine)}


def strata(ce: np.ndarray, f: SplitFeatures, lam: np.ndarray, n_strata: int = 4) -> list[dict]:
    """Where the gain sits: the windows sorted by the share of their bytes with a hit at the longest order (the memory's own sign that the text
    was seen before: a stratifier of the inputs, not of the loss), cut into n_strata equal groups, lowest share first. Cold bits/byte alone and
    mixed, the gain, and the group's share of the total gain; the weights are the dev ones."""
    share = (f.conf[-1] > 0).mean(1)
    alone, mixed = ce.astype(np.float64) / LN2, mix_bits(ce, f.right, f.bucket, lam)
    by_window = (alone - mixed).sum(1)
    rows = []
    for g in np.array_split(np.argsort(share, kind="stable"), n_strata):
        rows.append({"windows": int(len(g)), "longest_order_hit_share": float(share[g].mean()), "alone_cold": float(alone[g].mean()),
                     "mixed_cold": float(mixed[g].mean()), "gain_cold": float(by_window[g].sum() / (len(g) * ce.shape[1])),
                     "gain_warm": float((alone[g][:, ce.shape[1] // 2:] - mixed[g][:, ce.shape[1] // 2:]).mean()),
                     "share_of_total_gain": float(by_window[g].sum() / by_window.sum()) if by_window.sum() else None})
    return rows


def analyse_checkpoint(ce: dict[str, np.ndarray], feats: dict[str, SplitFeatures], keep: dict[str, np.ndarray], orders=ORDERS) -> dict:
    """What one checkpoint contributes. The weights lambda_b are fitted on dev, on every scored byte of every dev window (the cold bytes), and
    nothing else is fitted. For each split: the mixture over all windows, over the windows that overlap no held-out file with a training
    twin (weights as fitted, and refitted on the dev windows that remain), and, as a sensitivity row, weights fitted on dev's warm bytes only.
    ce[split] [W, T] nats; keep[split] [W] bool, False for a window that overlaps a file with a training twin."""
    n_b = 4 * len(orders) + 1
    dev, t = feats["dev"], ce["dev"].shape[1]
    warm = np.zeros(ce["dev"].shape, dtype=bool)
    warm[:, t // 2:] = True
    kd = keep["dev"]
    lam = fit_lambdas(ce["dev"], dev.right, dev.bucket, n_b)
    lam_warm = fit_lambdas(ce["dev"], dev.right, dev.bucket, n_b, mask=warm)
    lam_clean = fit_lambdas(ce["dev"][kd], dev.right[kd], dev.bucket[kd], n_b) if kd.any() else np.zeros(n_b)   # nothing left to fit: no mixing
    out: dict = {"lambda": [float(v) for v in lam], "bucket_labels": bucket_labels(orders), "splits": {}}
    for s, f in feats.items():
        k = keep[s]
        if k.any():
            excl = {"n_windows": int(k.sum()),
                    "fixed_lambda": evaluate(ce[s][k], f.right[k], f.bucket[k], lam),
                    "refit_lambda": {"lambda": [float(v) for v in lam_clean], **evaluate(ce[s][k], f.right[k], f.bucket[k], lam_clean)},
                    "buckets_cold_refit": bucket_table(ce[s][k], f.right[k], f.bucket[k], lam_clean, "cold", orders)}
        else:                                              # every window of the split overlaps a file with a training twin
            excl = {"n_windows": 0, "fixed_lambda": None, "refit_lambda": None, "buckets_cold_refit": None}
        out["splits"][s] = {
            "all": evaluate(ce[s], f.right, f.bucket, lam),
            "buckets_cold": bucket_table(ce[s], f.right, f.bucket, lam, "cold", orders),
            "buckets_warm": bucket_table(ce[s], f.right, f.bucket, lam, "warm", orders),
            "strata_by_longest_order_hit_share": strata(ce[s], f, lam),
            "warm_fit": {"lambda": [float(v) for v in lam_warm], **evaluate(ce[s], f.right, f.bucket, lam_warm)},
            "excl_twins": excl}
    return out


def md_table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---:" if i else ":---" for i in range(len(header))) + "|", *("| " + " | ".join(r) + " |" for r in rows)])


def fmt_gain(r: dict) -> str:
    lo, hi = r["gain_ci95"]
    return f"{r['gain']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def results_markdown(res: dict, n_windows: int, dup: dict, build: dict) -> str:
    """The tables printed at the end of a run: bits/byte alone and mixed, the gain with its interval, the duplicate-excluded rows, the
    sensitivity row and the per-bucket table of every checkpoint on test."""
    head = ["checkpoint", "alone cold", "mixed cold", "gain cold [95 % CI]", "alone warm", "mixed warm", "gain warm [95 % CI]"]

    def row(name: str, r: dict) -> list[str]:
        return [name, f"{r['cold']['alone']:.4f}", f"{r['cold']['mixed']:.4f}", fmt_gain(r["cold"]), f"{r['warm']['alone']:.4f}", f"{r['warm']['mixed']:.4f}", fmt_gain(r["warm"])]

    out = [f"Windows scored per split: {n_windows} of {N_WINDOWS}; memory built in {build['build_seconds']:.1f} s wall ({build['build_cpu_seconds']:.1f} s CPU), "
           f"occupancy " + ", ".join(f"n={n}: {o:.3f}" for n, o in build["occupancy"].items()) + "\n"]
    for s, label in (("test", "test (weights fitted on dev)"), ("dev", "dev (in sample: weights fitted on these bytes)")):
        out += [f"### {label}", md_table(head, [row(n, r["splits"][s]["all"]) for n, r in res.items()]), ""]
    for s in ("test", "dev"):
        k = dup[s]
        out += [f"### {s}, without the windows that overlap a held-out file with a training twin "
                f"({n_windows - int(k['overlaps'].sum())} of {n_windows} windows left, {int(k['overlaps'].sum())} removed)"]
        rows = ([row(f"{n}, weights from all dev", r["splits"][s]["excl_twins"]["fixed_lambda"]) for n, r in res.items() if r["splits"][s]["excl_twins"]["fixed_lambda"]]
                + [row(f"{n}, weights refit on clean dev", r["splits"][s]["excl_twins"]["refit_lambda"]) for n, r in res.items() if r["splits"][s]["excl_twins"]["refit_lambda"]])
        out += [md_table(head, rows) if rows else "(no window left)", ""]
    out += ["### test, weights fitted on dev's warm bytes only (sensitivity)", md_table(head, [row(n, r["splits"]["test"]["warm_fit"]) for n, r in res.items()]), ""]
    bh = ["bucket", "share of bytes", "memory right", "lambda", "model bits", "mixed bits", "gain contribution"]
    for n, r in res.items():
        rows = [[b["bucket"], f"{b['share']:.4f}", "-" if b["memory_right"] is None else f"{b['memory_right']:.3f}", f"{b['lambda']:.3f}",
                 "-" if b["alone_bits"] is None else f"{b['alone_bits']:.3f}", "-" if b["mixed_bits"] is None else f"{b['mixed_bits']:.3f}",
                 f"{b['gain_contribution']:+.5f}"] for b in r["splits"]["test"]["buckets_cold"]]
        out += [f"### {n}: test, cold, by bucket", md_table(bh, rows), ""]
    sh = ["checkpoint", "windows", "longest-order hit share", "alone cold", "mixed cold", "gain cold", "gain warm", "share of the total gain"]
    rows = [[n, str(g["windows"]), f"{g['longest_order_hit_share']:.3f}", f"{g['alone_cold']:.4f}", f"{g['mixed_cold']:.4f}", f"{g['gain_cold']:+.4f}",
             f"{g['gain_warm']:+.4f}", "-" if g["share_of_total_gain"] is None else f"{g['share_of_total_gain']:.3f}"]
            for n, r in res.items() for g in r["splits"]["test"]["strata_by_longest_order_hit_share"]]
    out += ["### test: windows in four groups by the share of their bytes with a hit at the longest order (lowest first)", md_table(sh, rows), ""]
    return "\n".join(out)


def run_analysis(args) -> dict:
    mem_dir, loss_dir, out_dir, corpus = Path(args.mem_dir), Path(args.loss_dir), Path(args.out_dir), Path(args.corpus)
    names = args.ckpts.split(",")
    build = json.loads((mem_dir / "build.json").read_text())
    orders = tuple(build["orders"])
    n = args.windows or complete_windows(loss_dir, names, tuple(SPLITS), N_WINDOWS // BLOCK)
    if n < BLOCK:
        raise SystemExit("no complete block of windows is scored yet")
    table, hashes = corpus_files(corpus)
    feats, dup, ce = {}, {}, {name: {} for name in names}
    for s, offset in SPLITS.items():
        _, y = load_windows(corpus, offset, n, T)
        y = y.numpy().astype(np.uint8)
        assert np.array_equal(y.reshape(-1), np.frombuffer(read_region(corpus, offset + 1, offset + 1 + n * T), dtype=np.uint8))   # window i byte t = absolute offset + i * T + t + 1
        pred, conf = memory_features(mem_dir, offset, n, orders, T)
        feats[s] = SplitFeatures(s, y, pred, conf, orders)
        dup[s] = duplicate_windows(table, hashes, offset, n, T, build["train_end"])
        for name in names:
            ce[name][s] = load_loss(loss_dir, name, s, n)
    res = {name: analyse_checkpoint(ce[name], feats, {s: ~dup[s]["overlaps"] for s in SPLITS}, orders) for name in names}
    why = args.why or ("all windows scored" if n == N_WINDOWS else "first windows only")
    for s, offset in SPLITS.items():
        checks = {name: committed_check(out_dir, s, CKPTS[name] + ".pt", ce[name][s]) for name in names}
        rec = {
            "split": s, "what": "next-byte memory written in one pass over the training bytes, mixed with a trained model by per-bucket weights fitted on dev",
            "yardstick": {"corpus": str(corpus.resolve()), "offset": offset, "window": T, "windows_in_the_yardstick": N_WINDOWS, "windows_scored": n,
                          "full_yardstick": n == N_WINDOWS, "why_this_many_windows": why, "scored_bytes_cold": n * T, "scored_bytes_warm": n * T // 2,
                          "scoring": "fp32, no autocast, CPU, batch 4 windows; cold = every byte of the window, warm = its second half"},
            "memory": {k: build[k] for k in ("orders", "log2_slots", "slots_per_order", "table_bytes_per_order", "chunk", "train_end", "build_seconds",
                                             "build_cpu_seconds", "read_seconds", "lookup_seconds", "occupancy", "heldout_hit_rate", "update_counts", "table_digest")},
            "mixture": {"rule": "p_mix(true byte) = (1 - lambda_b) p_model(true byte) + lambda_b [memory's byte == true byte]; lambda = 0 without a hit",
                        "bucket": "(longest order with a hit whose whole context lies inside the model's window, confidence class 1 | 2-3 | 4-15 | 16+), or no hit",
                        "lambda_fit": f"dev, all {n * T} scored bytes, maximising the mean log2 p_mix per bucket, lambda in [0, {LAM_MAX}]",
                        "bootstrap": {"resamples": N_BOOT, "seed": 0, "unit": "windows", "interval": "95 %, percentile; the weights are held fixed",
                                      "clusters": f"gain_ci95_clusters (an extra): the same interval resampling runs of {CLUSTER} consecutive windows, since neighbouring windows share files"},
                        "bucket_labels": bucket_labels(orders)},
            "duplicates": {"windows_removed": int(dup[s]["overlaps"].sum()), "windows_kept": int((~dup[s]["overlaps"]).sum()),
                           **{k: v for k, v in dup[s].items() if k != "overlaps"},
                           "rule": "a window is removed when it overlaps (header line and body) a held-out file whose body hash also occurs among the training files (files entirely inside [0, train_end))"},
            "checkpoints": {name: {"ckpt": CKPTS[name] + ".pt", "score_check": checks[name], "lambda": res[name]["lambda"], **res[name]["splits"][s]} for name in names},
        }
        (out_dir / f"mem_mix_{s}.json").write_text(json.dumps(rec, indent=2) + "\n")
    md = results_markdown(res, n, dup, build)
    print(md)
    for name in names:
        for s in SPLITS:
            c = committed_check(out_dir, s, CKPTS[name] + ".pt", ce[name][s])
            print(f"score check {name:11s} {s:4s}: " + ("no committed number covers these windows" if c is None else
                  f"cold {c['ours']['cold']:.4f} vs {c['committed']['cold']:.4f}, warm {c['ours']['warm']:.4f} vs {c['committed']['warm']:.4f} "
                  f"({c['file']}, {c['n_windows']} windows): equal to 3 decimals: {c['equal_to_3_decimals']}"))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage", choices=("score", "analyse"), default="analyse")
    p.add_argument("--corpus", default=str(CORPUS))
    p.add_argument("--ckpt-dir", default=str(RESULTS / "ckpt"))
    p.add_argument("--mem-dir", default=str(RESULTS / "mem"))
    p.add_argument("--loss-dir", default=str(RESULTS / "mem" / "loss"))
    p.add_argument("--out-dir", default=str(RESULTS))
    p.add_argument("--ckpts", default=",".join(CKPTS))
    p.add_argument("--splits", default=",".join(SPLITS))
    p.add_argument("--windows", type=int, default=None, help="windows per split, a multiple of 64 (score: default 1,024; analyse: default every window all runs have scored)")
    p.add_argument("--threads", type=int, default=2, help="torch threads of this worker (score stage)")
    p.add_argument("--why", default="", help="analyse stage: why fewer than 1,024 windows were scored (written into the JSON)")
    args = p.parse_args()
    if args.stage == "score":
        run_worker(args)
    else:
        run_analysis(args)


if __name__ == "__main__":
    main()
