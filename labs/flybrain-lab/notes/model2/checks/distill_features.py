#!/usr/bin/env python
"""The slice's table inputs (T2 of the distillation plan, 2026-10-20): the reader takes 18 inputs a position from the saved tables
(use_features.text_features: the byte orders 4..64 and the keys looked up in the frozen tables, the seven counters from the text).
Writes a FeatureStore directory for the slice (pred_n<order>.npy, conf_n<order>.npy, uint8 [S]; features.json) so run_sparsela.py
--mem-feat reads it like a corpus's own features. --check DIR: before that, the exactness check of this path against a corpus's stored
features at its train_end (the frozen table's lookups): the inputs of --check-n windows computed from the text alone, from 64 KB
before, must equal the stored ones (0 differences), which verifies both the rebuilt tables and the alignment.
usage: distill_features.py --tables /vault/.../talkB/t28 --slice distill.bytes --out distill-tables/feat_all28 [--check talkB-tables/feat_all28 --check-n 64]"""
import argparse, hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); CK = os.path.abspath(os.path.join(HERE, "..", "..", "speed-round", "checks"))
sys.path.insert(0, CK)
from use_features import Tables, text_features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tables", required=True); ap.add_argument("--slice", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--orders", default="4,8,12,16,32,64,116,132,202,203,301,1001,1002,1003,1004,1005,1006,1007")
ap.add_argument("--check", default=None); ap.add_argument("--check-n", type=int, default=64); ap.add_argument("--T", type=int, default=2048); ap.add_argument("--back", type=int, default=65536)
a = ap.parse_args(); os.umask(0o077)
orders = [int(x) for x in a.orders.split(",")]
t0 = time.time(); tables = Tables(a.tables)
print(f"tables: {a.tables} ({', '.join(str(n) for n in tables.meta['orders'])}; 2^{tables.log2} slots) loaded in {time.time() - t0:.0f} s", flush=True)

if a.check:
    info = json.load(open(f"{a.check}/features.json")); corpus = np.memmap(info["corpus"], dtype=np.uint8, mode="r")
    A = int(info["train_end"]); B = A + a.check_n * a.T; H = max(0, A - a.back)
    t1 = time.time(); pred, conf = text_features(tables, np.asarray(corpus[H:B]), orders)
    bad = 0; stateful = 0
    for j, o in enumerate(orders):
        fp = np.load(f"{a.check}/pred_n{o}.npy", mmap_mode="r")[A:B]; fc = np.load(f"{a.check}/conf_n{o}.npy", mmap_mode="r")[A:B]
        d = int((pred[A - H:B - H, j] != fp).sum() + (conf[A - H:B - H, j] != fc).sum())
        if o in (1004, 1005, 1006, 1007): stateful += d  # brace and paren depth, in-string, in-comment carry state from the file's start: a check from 64 KB back cannot match them
        else: bad += d
        if d: print(f"  order {o}: {d} differences" + (" (file-state counter, not exact from mid-file)" if o in (1004, 1005, 1006, 1007) else ""), flush=True)
    print(f"check: {B - A:,} positions from {A:,} ({a.check_n} windows): {bad} differences in the table orders and the position counters, {stateful} in the file-state counters ({time.time() - t1:.0f} s)", flush=True)
    if bad: raise SystemExit("the text features do not reproduce the stored ones: stop")

data = np.memmap(a.slice, dtype=np.uint8, mode="r"); S = len(data)
t1 = time.time(); pred, conf = text_features(tables, np.asarray(data), orders)
os.makedirs(a.out, exist_ok=True); os.chmod(a.out, 0o700)
for j, o in enumerate(orders):
    for kind, arr in (("pred", pred), ("conf", conf)):
        f = os.path.join(a.out, f"{kind}_n{o}.npy"); np.save(f, np.ascontiguousarray(arr[:S, j])); os.chmod(f, 0o600)
hit = {str(o): float((conf[:S, j] > 0).mean()) for j, o in enumerate(orders) if o < 1000}
meta = {"kind": "features", "corpus": os.path.abspath(a.slice), "corpus_bytes": S, "train_end": S, "heldout_positions": 0, "chunk": int(tables.meta.get("chunk", 1 << 20)),
        "files": {"pred": "pred_n<order>.npy", "conf": "conf_n<order>.npy"},
        "alignment": "feature[p] is the frozen table's prediction for the byte at absolute position p from the n bytes before it (conf 0 = miss); the slice is read through the saved tables of another corpus (use_features.text_features), never a leak-free pass of its own",
        "orders": orders, "tables": os.path.abspath(a.tables), "table_digest": hashlib.sha256(open(f"{a.tables}/tables.json", "rb").read()).hexdigest()[:16],
        "slots_per_order": 1 << tables.log2, "hit_rate": hit, "built": time.strftime("%F %T"),
        "note": f"distill_features.py: the slice's inputs from the saved tables {a.tables}"}
json.dump(meta, open(os.path.join(a.out, "features.json"), "w"), indent=1); os.chmod(os.path.join(a.out, "features.json"), 0o600)
print(f"wrote {a.out}: {S:,} positions, {len(orders)} orders ({time.time() - t1:.0f} s); hit rates " + ", ".join(f"{k}:{v:.2f}" for k, v in hit.items()))
