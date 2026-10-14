"""Index a project into a copy of the saved tables (the use layer, second pass, 2026-10-14): "learn a codebase in minutes", with no
training. The project's files are laid out as the corpus was (the header "\\n// ==== <name>/<path> ====\\n" before each file's bytes,
the files in a seeded shuffled order) into <out>/index.bytes, a share of them held out into <out>/heldout.bytes (the viability test
scores those: the model never sees them, the table never holds them), and the index stream is written into a copy of the tables with
the corpus build's own rule (assoc_all.py's write(), copied below unchanged: two-way slots, confirm or decay a matching tag, else fill an
empty slot, else replace the weaker unless both are confirmed), chunk by chunk as the build did, the byte orders' hashes from
context_hashes and the key orders' from compute_keys (as use_features.text_features reads them). The result is a tables directory any
reader uses as it uses the corpus tables (sample_tables.py --tables <out>): the project's contexts answer where the corpus's did not.

usage:
  index_project.py --tables DIR --out DIR [--name NAME] [--hold-out 0.2] [--seed 0] [--ext gd,py,...] [--chunk 1048576] PROJECT_DIR
  index_project.py --check STREAM OUT       the exactness check: the stream indexed into empty 2^20 tables here, and by assoc_all.py
                                            TABLES_ONLY=1 (its keys from compute_keys saved as key_n<id>.npy); the tables must be bit-identical
The copy of the tables is a plain file copy (2^28: 12 GB, about a minute); the writes map it read-write and touch the slots the
project's contexts hash to. Memory: the index stream and its keys (a few bytes per byte of project); the tables stay on disk."""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_common import SPEED  # noqa: E402

sys.path.insert(0, SPEED)
from flybrain.corpus_memory import CONF_MAX, SLOT, context_hashes, slot_and_tag  # noqa: E402
from key_tables import KEY_IDS, compute_keys  # noqa: E402
from use_features import BYTE_ORDERS, HEADER  # noqa: E402

CODE_EXT = ("gd", "gdshader", "tscn", "tres", "cfg", "py", "c", "h", "cc", "cpp", "hpp", "cxx", "hh", "js", "ts", "mjs", "rs", "go", "java",
            "kt", "cs", "sh", "bash", "lua", "rb", "php", "pl", "m", "swift", "json", "toml", "yaml", "yml", "ini", "cmake", "mk", "md",
            "txt", "sql", "html", "css", "xml", "proto", "vert", "frag", "glsl", "wgsl")
SKIP_DIRS = (".git", ".godot", ".import", "node_modules", "__pycache__", ".venv", "venv", "build", "dist", ".worktrees", ".superpowers")
MAX_FILE = 1 << 20          # the corpus kept files of a few KB to a few hundred KB; a generated megabyte is not code to learn


def write(table, slot, tag, nxt):
    """assoc_all.py's write(), unchanged: one chunk's (slot, tag, next byte) triples into a 2-way table. Duplicate contexts inside a
    chunk resolve by numpy's last write."""
    slot = slot.astype(np.int64)
    a, b = slot & ~1, slot | 1
    ra, rb = table[a], table[b]
    hit_a = (ra["conf"] > 0) & (ra["tag"] == tag)
    hit_b = ~hit_a & (rb["conf"] > 0) & (rb["tag"] == tag)
    for hit, idx, rec in ((hit_a, a, ra), (hit_b, b, rb)):          # confirm or decay the matching buddy
        if hit.any():
            r = rec[hit].copy(); same = r["byte"] == nxt[hit]
            c = r["conf"].astype(np.int32)
            new_c = np.where(same, np.minimum(c + 1, CONF_MAX), c - 1)
            r["byte"] = np.where(new_c <= 0, nxt[hit], r["byte"]); r["conf"] = np.where(new_c <= 0, 1, new_c).astype(np.uint8)
            table[idx[hit]] = r
    miss = ~hit_a & ~hit_b
    if not miss.any():
        return
    a, b, tag, nxt, ra, rb = a[miss], b[miss], tag[miss], nxt[miss], ra[miss], rb[miss]
    empty_a, empty_b = ra["conf"] == 0, rb["conf"] == 0
    target = np.where(empty_a, a, np.where(empty_b, b, np.where(ra["conf"] <= rb["conf"], a, b)))
    ok = ~((ra["conf"] >= 2) & (rb["conf"] >= 2))
    rec = np.empty(int(ok.sum()), dtype=SLOT); rec["tag"] = tag[ok]; rec["byte"] = nxt[ok]; rec["conf"] = 1
    table[target[ok]] = rec


def index_stream(tables: dict, log2: int, data: np.ndarray, orders, chunk: int, verbose=True) -> None:
    """Write `data` (a corpus-format stream) into the tables, as assoc_all.py TABLES_ONLY=1 does: chunks of `chunk`, each chunk's byte
    orders' hashes from context_hashes and key orders' from compute_keys (on the whole stream, as the build's key files were)."""
    N = len(data)
    key_ids = [n for n in orders if n in KEY_IDS]
    keys = compute_keys(data) if key_ids else {}
    t0 = time.time()
    bounds = sorted(set(range(0, N, chunk)) | {N})
    byte_orders = [n for n in orders if n in BYTE_ORDERS]
    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        hashes = context_hashes(data, lo, hi, byte_orders) if byte_orders else {}
        for kid in key_ids:
            hashes[kid] = np.asarray(keys[kid][lo:hi])
        nxt = data[lo:hi]
        for n in orders:
            first = min(max(lo, n) - lo, hi - lo)
            slot, tag = slot_and_tag(hashes[n][first:], log2)
            write(tables[n], slot, tag, nxt[first:])
        if verbose and i % 20 == 0:
            print(f"  {hi:,} / {N:,} bytes  {time.time() - t0:.0f} s", flush=True)


def collect(root: str, exts, skip=SKIP_DIRS) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip and not d.startswith("."))
        for f in sorted(filenames):
            if f.rsplit(".", 1)[-1].lower() in exts and 0 < os.path.getsize(os.path.join(dirpath, f)) <= MAX_FILE:
                out.append(os.path.relpath(os.path.join(dirpath, f), root))
    return out


def stream_of(root: str, name: str, files) -> bytes:
    parts = []
    for rel in files:
        parts.append(HEADER + f"{name}/{rel}".encode() + b" ====\n")
        parts.append(open(os.path.join(root, rel), "rb").read())
    return b"".join(parts)


def index_project(tables_dir: str, out: str, root: str, name: str, hold_out: float, seed: int, exts, chunk: int) -> dict:
    meta = json.load(open(f"{tables_dir}/tables.json"))
    log2, orders = int(meta["log2_slots"]), [int(n) for n in meta["orders"]]
    files = collect(root, exts)
    if not files:
        raise SystemExit(f"no files with extensions {sorted(exts)} under {root}")
    rng = random.Random(seed); rng.shuffle(files)
    n_held = int(round(len(files) * hold_out))
    held, kept = sorted(files[:n_held]), files[n_held:]          # the index keeps the shuffled order, as the corpus did
    os.makedirs(out, exist_ok=True)
    idx = stream_of(root, name, kept); hold = stream_of(root, name, held)
    open(f"{out}/index.bytes", "wb").write(idx); open(f"{out}/heldout.bytes", "wb").write(hold)
    print(f"{name}: {len(files)} files; indexing {len(kept)} ({len(idx):,} bytes), holding out {len(held)} ({len(hold):,} bytes)", flush=True)
    t0 = time.time()
    for n in orders:
        shutil.copyfile(f"{tables_dir}/table_n{n}.npy", f"{out}/table_n{n}.npy")
    print(f"tables copied from {tables_dir} in {time.time() - t0:.0f} s", flush=True)
    tabs = {n: np.load(f"{out}/table_n{n}.npy", mmap_mode="r+") for n in orders}
    data = np.frombuffer(idx, dtype=np.uint8)
    t0 = time.time()
    index_stream(tabs, log2, data, orders, chunk)
    for t in tabs.values():
        t.flush()
    secs = time.time() - t0
    info = {**meta, "base": tables_dir, "indexed": {"project": root, "name": name, "files": len(kept), "bytes": len(idx), "seed": seed,
                                                     "held_out_files": len(held), "held_out_bytes": len(hold), "hold_out": hold_out,
                                                     "exts": sorted(exts), "index_seconds": round(secs, 1),
                                                     "index_sha256": hashlib.sha256(idx).hexdigest()}}
    json.dump(info, open(f"{out}/tables.json", "w"), indent=2)
    json.dump({"indexed": kept, "held_out": held}, open(f"{out}/split.json", "w"), indent=1)
    print(f"indexed in {secs:.0f} s: {out}", flush=True)
    return info


def check(stream_path: str, out: str, log2: int = 20) -> int:
    """index_stream from empty tables against assoc_all.py TABLES_ONLY=1 on the same bytes (its key files from compute_keys)."""
    data = np.fromfile(stream_path, dtype=np.uint8)
    orders = [*BYTE_ORDERS, *KEY_IDS]
    os.makedirs(f"{out}/keys", exist_ok=True); os.makedirs(f"{out}/mine", exist_ok=True)
    keys = compute_keys(data)
    for kid in KEY_IDS:
        np.save(f"{out}/keys/key_n{kid}.npy", np.asarray(keys[kid]))
    env = {**os.environ, "CORPUS": stream_path, "TRAIN_END": str(len(data)), "KEYS": f"{out}/keys", "TABLES_ONLY": "1", "MEM": out}
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, f"{here}/assoc_all.py", str(log2), f"{out}/ref"], env=env, check=True, stdout=subprocess.DEVNULL)
    tabs = {n: np.zeros(1 << log2, dtype=SLOT) for n in orders}
    index_stream(tabs, log2, data, orders, 1 << 20, verbose=False)
    bad = 0
    for n in orders:
        ref = np.load(f"{out}/ref/table_n{n}.npy")
        d = int((ref != tabs[n]).sum())
        print(f"order {n}: {int((tabs[n]['conf'] > 0).sum()):,} slots written, {d} differences from assoc_all.py")
        bad += d
    print("index_project.py writes the tables assoc_all.py writes" if bad == 0 else f"MISMATCH: {bad} differences")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("project", nargs="?")
    ap.add_argument("--tables"); ap.add_argument("--out"); ap.add_argument("--name", default=None)
    ap.add_argument("--hold-out", type=float, default=0.2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ext", default=",".join(CODE_EXT)); ap.add_argument("--chunk", type=int, default=1 << 20)
    ap.add_argument("--check", nargs=2, metavar=("STREAM", "OUT"), default=None)
    a = ap.parse_args()
    if a.check:
        sys.exit(1 if check(*a.check) else 0)
    if not (a.project and a.tables and a.out):
        ap.error("--tables, --out and the project directory are required")
    root = os.path.abspath(a.project)
    index_project(a.tables, a.out, root, a.name or os.path.basename(root).lower(), a.hold_out, a.seed, set(a.ext.lower().split(",")), a.chunk)


if __name__ == "__main__":
    main()
