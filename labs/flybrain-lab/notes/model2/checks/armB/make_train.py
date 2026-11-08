#!/usr/bin/env python
"""Arm B (2026-11-06): the student's training sequences. His material (every world file, and with --sessions every held-out session
document, both dated; --before keeps only those created before a cutoff) is cut into byte segments; each segment gets the span
retriever's memory in front (two chunks with file headers, chosen as of the document's date, never from the document's own file nor a
chunk holding the segment's text), as the deployed prompt will have it, and the loss covers the segment only. Replay (general prose
and Python source, no memory) is added until it is --replay-frac of the loss tokens. Writes train.npz (tokens int32, seq_start, loss_start,
kind 0 files / 1 sessions / 2 replay, day) and train.json (counts). Numbers and ids only, never text.
usage: make_train.py --out DIR [--before 2026-09-01] [--sessions] [--replay-frac 0.3] [--seg 6000]"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import ASOF, Retriever, Tokenizer, day, WORLD

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--before", default=""); ap.add_argument("--sessions", action="store_true")
ap.add_argument("--replay-frac", type=float, default=0.3); ap.add_argument("--seg", type=int, default=6000); ap.add_argument("--query", type=int, default=1024)
ap.add_argument("--replay", default="/vault/ailab/flybrain-mem/armB/replay/fineweb_edu.bytes:2,/games/ailab/lab/flybrain-lab/corpora/cpython.bytes:1")
ap.add_argument("--session-dates", default="/games/ailab/corpora/kyle/session_dates.json"); ap.add_argument("--max-tokens", type=int, default=4000, help="drop a sequence longer than this")
a = ap.parse_args(); os.umask(0o077); os.makedirs(a.out, exist_ok=True); t0 = time.time()
tok = Tokenizer(); BOS = tok.bos; r = Retriever(); cut = day(a.before) if a.before else 0
seqs = []   # (tokens list, loss_start, kind, day)
stats = {"docs": 0, "segments": 0, "with_memory": 0, "dropped_long": 0, "loss_tokens": 0, "memory_tokens": 0}
def add_his(doc: bytes, d: int, kind: int, own_file: int):
    stats["docs"] += 1
    for s in range(0, len(doc), a.seg):
        seg = doc[s:s + a.seg]
        if len(seg) < 200: continue
        before = min(d, cut) if (d and cut) else (d or cut)
        ids = r.retrieve(seg[:a.query], exclude_bytes=seg, before_day=before, exclude_file=own_file)
        mem = r.memory(ids, headers=True); mt = tok(mem) if mem else []; st = tok(seg)
        if 1 + len(mt) + len(st) > a.max_tokens: stats["dropped_long"] += 1; continue
        seqs.append(([BOS] + mt + st, 1 + len(mt), kind, d)); stats["segments"] += 1; stats["with_memory"] += bool(ids); stats["loss_tokens"] += len(st); stats["memory_tokens"] += len(mt)
for fi, rel in enumerate(r.files):
    d = r.file_day[rel]
    if cut and d and d >= cut: continue
    add_his(open(os.path.join(WORLD, rel), "rb").read(), d, 0, fi)
files_loss = stats["loss_tokens"]; files_docs = stats["docs"]
if a.sessions:
    sd = json.load(open(a.session_dates))
    for name, fn in (("kyle", "kyle.bytes"), ("dialogue", "dialogue.bytes")):
        b = open(os.path.join(os.path.dirname(a.session_dates), fn), "rb").read()
        for s, e, dt in sd[name]:
            if not dt: continue
            d = day(dt)
            if cut and d >= cut: continue
            add_his(b[s:e], d, 1, -1)
his_loss = stats["loss_tokens"]
print(f"his material: {files_docs} files + {stats['docs'] - files_docs} sessions, {stats['segments']} segments ({stats['with_memory']} with memory, {stats['dropped_long']} dropped as too long), "
      f"{his_loss:,} loss tokens, {stats['memory_tokens']:,} memory tokens ({time.time() - t0:.0f} s)", flush=True)
want = int(his_loss * a.replay_frac / (1 - a.replay_frac)); got = 0; reps = {}
parts = [(p.split(":")[0], float(p.split(":")[1])) for p in a.replay.split(",")]; wsum = sum(w for _, w in parts)
for path, w in parts:
    b = open(path, "rb").read(); target = int(want * w / wsum); n = 0
    for s in range(0, len(b), a.seg):
        if n >= target: break
        t = tok(b[s:s + a.seg])
        if len(t) < 50: continue
        seqs.append(([BOS] + t, 1, 2, 0)); n += len(t)
    reps[os.path.basename(path)] = n; got += n
print(f"replay: {got:,} tokens {reps} = {got / (got + his_loss):.1%} of the loss tokens", flush=True)
L = [len(s[0]) for s in seqs]; tokens = np.concatenate([np.asarray(s[0], dtype=np.int32) for s in seqs]); seq_start = np.concatenate([[0], np.cumsum(L)]).astype(np.int64)
np.savez(os.path.join(a.out, "train.npz"), tokens=tokens, seq_start=seq_start, loss_start=np.asarray([s[1] for s in seqs], dtype=np.int64),
         kind=np.asarray([s[2] for s in seqs], dtype=np.int8), day=np.asarray([s[3] for s in seqs], dtype=np.int32))
meta = {"before": a.before, "sessions": a.sessions, "replay_frac": a.replay_frac, "seg": a.seg, "sequences": len(seqs), "tokens": int(len(tokens)), "loss_tokens": his_loss + got,
        "his_loss_tokens": his_loss, "files_loss_tokens": files_loss, "session_loss_tokens": his_loss - files_loss, "replay_tokens": got, "replay": reps, "stats": stats,
        "max_len": int(max(L)), "mean_len": float(np.mean(L)), "seconds": round(time.time() - t0)}
json.dump(meta, open(os.path.join(a.out, "train.json"), "w"), indent=1); print(json.dumps({k: v for k, v in meta.items() if k != "stats"}))
