#!/usr/bin/env python
"""The reader as the judge of what the big model reads (the memory-head test proper, 2026-10-20; the side agent's point: the retrieval
tests only ever compared matchers, never the small model's judgement). No new Qwen time: for every held-out window the runs so far have
scored Qwen on several different memories (the spans', the tables', the breadth vote's, the dedup chooser's, the sweep's arms), so a
chooser that picks among those candidates can be scored by looking up Qwen's bits under its pick. The in-house reader (a 4 KB
table-fed reader through its saved tables) reads each candidate memory followed by the window's FIRST half and is charged its bits on
the first half: the use-time signal (the second half is never read for the choice). Its pick is the candidate with the fewest bits.
Reported per set: Qwen's bits under the reader's pick (among memories; and with "no memory" as a candidate, the gate), the span
chooser's, the oracle over the same candidates, the mean rank correlation between the reader's judgement and Qwen's bits across a
window's candidates, and how often the pick is the oracle's. --second also scores the reader on the second half given each memory
(its own gain, a diagnostic, with the memory cut to fit the window). Prints numbers only, never text. CPU.
usage: reader_judge.py --ckpt tok/results/ckpt/th_w4k_r_transformer.pt --tables /games/ailab/corpora/talk-tables/t28 --out DIR
       --ref results/tables/qwen/qwen_memory.json --spans-choices results/tables/qwen-dedup/choices.json
       --runs "label=qwen_memory.json:choices.json[:choices|span_choices]" ... [--sets kyle,dlg] [--limit N] [--second] [--threads 6]"""
import argparse, glob, hashlib, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); CK = os.path.abspath(os.path.join(HERE, "..", "..", "speed-round", "checks"))
TOK = "/games/ailab/lab/flybrain-lab/.worktrees/tok"
# one flybrain package for everything, the tok worktree's (the readers were trained there and the speed worktree's model code is older);
# use_features would put the speed worktree first, so flybrain is imported before it and stays cached
sys.path.insert(0, TOK)
import flybrain, flybrain.corpus_memory  # noqa: E402,F401
from scripts.cross_eval import rebuild  # noqa: E402
from flybrain.model.inputs import channel_kwargs  # noqa: E402
sys.path.insert(1, CK)
from use_features import Tables, text_features  # noqa: E402
assert flybrain.__file__.startswith(TOK), flybrain.__file__

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--tables", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ref", required=True, help="the span run: warm per window"); ap.add_argument("--spans-choices", required=True, help="a choices.json whose span_choices are the span run's picks")
ap.add_argument("--runs", nargs="*", default=[], help="label=qwen_memory.json:choices.json[:key] candidate sources (standard prompt format, the 1,500-byte pool)")
ap.add_argument("--world", default="/games/ailab/corpora/kyle/world/files"); ap.add_argument("--chunk", type=int, default=1500); ap.add_argument("--sep", default="\n\n")
ap.add_argument("--heldout", default="/games/ailab/corpora/kyle/heldout.bytes"); ap.add_argument("--json", default="/games/ailab/corpora/kyle/heldout.json")
ap.add_argument("--T", type=int, default=2048); ap.add_argument("--kyle-every", type=int, default=1); ap.add_argument("--dlg-every", type=int, default=4)
ap.add_argument("--sets", default="kyle,dlg"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--second", action="store_true"); ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args(); os.umask(0o077); os.makedirs(a.out, exist_ok=True); torch.set_num_threads(a.threads)

# ---- the pool (qwen_memory.py's, byte for byte) and the candidates --------------------------------------------------------------------------
t0 = time.time(); chunks = []
for p in sorted(glob.glob(os.path.join(a.world, "**", "*"), recursive=True)):
    if not os.path.isfile(p): continue
    b = open(p, "rb").read()
    for s in range(0, max(1, len(b) - a.chunk // 2), a.chunk // 2 if len(b) > a.chunk else a.chunk):
        c = b[s:s + a.chunk]
        if len(c) < 64: continue
        chunks.append(c)
digest = hashlib.sha256(b"".join(hashlib.sha256(c).digest() for c in chunks)).hexdigest()
ref = json.load(open(a.ref)); SETS = [s for s in a.sets.split(",")]
sources = [("spans", a.ref, a.spans_choices, "span_choices")]
for r in a.runs:
    label, rest = r.split("=", 1); parts = rest.split(":"); sources.append((label, parts[0], parts[1], parts[2] if len(parts) > 2 else "choices"))
cands = {s: {} for s in SETS}                                                  # set -> window -> {ids tuple: (qwen bits, labels)}
for label, mj, cj, key in sources:
    if not (os.path.exists(mj) and os.path.exists(cj)): print(f"  {label}: missing, skipped"); continue
    M = json.load(open(mj)); C = json.load(open(cj))
    if C["pool"]["digest"] != digest: print(f"  {label}: another pool, skipped"); continue
    rt = M.get("retriever", {})
    if rt.get("order", "top-first") != "top-first" or rt.get("sep", "\\n\\n") != "\\n\\n" or rt.get("headers"): print(f"  {label}: another prompt format, skipped"); continue
    for s in SETS:
        pw = M[s]["memory"]["per_window"]; ch = C[key][s]
        for n in range(min(len(pw), len(ch))):
            d = cands[s].setdefault(n, {}); t = tuple(int(i) for i in ch[n])
            if t in d and abs(d[t][0] - pw[n]) > 0.001: print(f"  {label}: window {s}/{n} scores {pw[n]} vs {d[t][0]} for the same memory?")
            d.setdefault(t, (pw[n], []))[1].append(label)
for s in SETS:
    for n, d in cands[s].items(): d.setdefault((), (ref[s]["warm"]["per_window"][n], ["none"]))
print(f"pool {len(chunks):,} chunks; candidates: " + ", ".join(f"{s}: {len(cands[s])} windows, {np.mean([len(d) for d in cands[s].values()]):.1f} a window" for s in SETS) + f" ({time.time() - t0:.0f} s)", flush=True)

# ---- the reader ---------------------------------------------------------------------------------------------------------------------------
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); orders = list(channel_kwargs(ck["args"])["mem_orders"]); L = int(ck["args"]["seq_len"])
m = rebuild(ck).eval(); tables = Tables(a.tables); LN2 = math.log(2)
# the features through this import path must reproduce a corpus's stored ones (the table orders; the file-state counters cannot from mid-file)
feat_dir = ck["args"]["mem_feat"]
if os.path.isdir(feat_dir) and os.path.exists(f"{feat_dir}/features.json"):
    info = json.load(open(f"{feat_dir}/features.json")); corpus = np.memmap(info["corpus"], dtype=np.uint8, mode="r"); A = int(info["train_end"]); B = A + 8 * T if False else A + 8 * 2048; H = A - 65536
    pr, cf = text_features(tables, np.asarray(corpus[H:B]), orders); bad = 0
    for j, o in enumerate(orders):
        if o in (1004, 1005, 1006, 1007): continue
        bad += int((pr[A - H:B - H, j] != np.load(f"{feat_dir}/pred_n{o}.npy", mmap_mode="r")[A:B]).sum() + (cf[A - H:B - H, j] != np.load(f"{feat_dir}/conf_n{o}.npy", mmap_mode="r")[A:B]).sum())
    if bad: raise SystemExit(f"the features through this path differ from {feat_dir} at {bad} positions: stop")
    print(f"features check against {feat_dir}: 0 differences on 8 windows", flush=True)
print(f"reader {os.path.basename(a.ckpt)}: {ck['arm']}, window {L}, {len(orders)} table inputs from {a.tables}", flush=True)
truncs = 0


@torch.no_grad()
def bits(prefix: bytes, target: bytes) -> float:
    """The reader's bits per byte of `target` after `prefix`; a prefix longer than the window allows is cut from its front."""
    global truncs
    room = L - len(target)
    if len(prefix) > room: prefix = prefix[-room:]; truncs += 1
    text = prefix + target; N = len(text)
    pred, conf = text_features(tables, text, orders)
    x = torch.from_numpy(np.frombuffer(text[:-1], np.uint8).astype(np.int64))[None]
    y = torch.from_numpy(np.frombuffer(text[1:], np.uint8).astype(np.int64))[None]
    kw = {"mem_pred": torch.from_numpy(pred[1:N].copy())[None], "mem_conf": torch.from_numpy(conf[1:N].copy())[None]}
    logits, _ = m(x, **kw)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 256).float(), y.reshape(-1), reduction="none")
    return float(ce[-len(target):].sum() / len(target) / LN2)


held = open(a.heldout, "rb").read(); meta = json.load(open(a.json)); T = a.T
offs = {"kyle": (meta["kyle"]["offset"], meta["kyle"]["windows"], a.kyle_every), "dlg": (meta["dialogue"]["offset"], meta["dialogue"]["windows"], a.dlg_every)}
out = {"ckpt": a.ckpt, "tables": a.tables, "sets": {}}; lines = []
for s in SETS:
    off, nwin, every = offs[s]; rows = []; t1 = time.time(); n = 0
    for w in range(0, nwin, every):
        if n not in cands[s]: n += 1; continue
        if a.limit and len(rows) >= a.limit: break
        W = held[off + w * T: off + (w + 1) * T]; first, second = W[:T // 2], W[T // 2:]
        row = {"n": n, "cands": []}
        for ids, (qb, labels) in cands[s][n].items():
            mem = b"".join(chunks[i] + a.sep.encode() for i in ids)
            j1 = bits(mem, first)
            j2 = bits(mem + first, second) if a.second else None
            row["cands"].append({"ids": list(ids), "qwen": qb, "j1": j1, "j2": j2, "labels": labels})
        rows.append(row); n += 1
        if len(rows) % 20 == 0: print(f"  {s}: {len(rows)} windows ({time.time() - t1:.0f} s, {truncs} prefixes cut)", flush=True)
    # ---- the readings
    def pick(row, with_none):
        cs = [c for c in row["cands"] if with_none or c["ids"]]; return min(cs, key=lambda c: c["j1"])
    warm = np.array([next(c["qwen"] for c in r["cands"] if not c["ids"]) for r in rows])
    spans = np.array([next((c["qwen"] for c in r["cands"] if "spans" in c["labels"]), np.nan) for r in rows])
    picked = np.array([pick(r, False)["qwen"] for r in rows]); gated = np.array([pick(r, True)["qwen"] for r in rows])
    oracle = np.array([min(c["qwen"] for c in r["cands"] if c["ids"]) for r in rows]); oracle_g = np.array([min(c["qwen"] for c in r["cands"]) for r in rows])
    hit = np.mean([pick(r, False)["ids"] == min((c for c in r["cands"] if c["ids"]), key=lambda c: c["qwen"])["ids"] for r in rows])
    rc = []
    for r in rows:
        cs = [c for c in r["cands"] if c["ids"]]
        if len(cs) >= 3:
            x_ = np.array([c["j1"] for c in cs]); y_ = np.array([c["qwen"] for c in cs])
            if x_.std() > 0 and y_.std() > 0: rc.append(np.corrcoef(np.argsort(np.argsort(x_)), np.argsort(np.argsort(y_)))[0, 1])
    ncand = np.mean([len([c for c in r["cands"] if c["ids"]]) for r in rows])
    nm = "Kyle's turns" if s == "kyle" else "the dialogue"; ok = ~np.isnan(spans)
    line = (f"judge ({os.path.basename(a.ckpt)}) on {nm} ({len(rows)} windows, {ncand:.1f} memories a window): warm {warm.mean():.4f}; the spans' memory {spans[ok].mean():.4f} ({(spans[ok] - warm[ok]).mean():+.4f}); "
            f"the reader's pick {picked.mean():.4f} ({(picked - warm).mean():+.4f}; vs the spans {(picked[ok] - spans[ok]).mean():+.4f}, better >0.02 on {int(((picked - spans)[ok] < -0.02).sum())}, worse on {int(((picked - spans)[ok] > 0.02).sum())}); "
            f"the reader's pick with 'no memory' allowed {gated.mean():.4f} ({(gated - warm).mean():+.4f}); the oracle over the same memories {oracle.mean():.4f} ({(oracle - warm).mean():+.4f}; with none {(oracle_g - warm).mean():+.4f}); "
            f"the pick is the oracle's on {hit:.0%}; rank corr of the reader's first-half bits with Qwen's bits across a window's memories {np.mean(rc):+.3f} (over {len(rc)} windows)")
    if a.second:
        rc2 = []
        for r in rows:
            cs = [c for c in r["cands"] if c["ids"]]
            if len(cs) >= 3:
                x_ = np.array([c["j2"] for c in cs]); y_ = np.array([c["qwen"] for c in cs])
                if x_.std() > 0 and y_.std() > 0: rc2.append(np.corrcoef(np.argsort(np.argsort(x_)), np.argsort(np.argsort(y_)))[0, 1])
        line += f"; the reader's own second-half bits vs Qwen's: rank corr {np.mean(rc2):+.3f}"
    print(line, flush=True); lines.append(line); out["sets"][s] = rows
json.dump(out, open(os.path.join(a.out, "judge.json"), "w")); open(os.path.join(a.out, "judge.txt"), "a").write("\n".join(lines) + "\n")
print(f"wrote {a.out} ({truncs} prefixes cut to fit the window)")
