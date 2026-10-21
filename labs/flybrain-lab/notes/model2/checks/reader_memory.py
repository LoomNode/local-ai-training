#!/usr/bin/env python
"""Does the reader read memory in its context? (2026-10-21, Kyle: "do 1 now".) For each checkpoint, the reader's bits per byte on each
held-out window's second half given (a) the first half alone, (b) the first half behind the plain chunks (qwen_memory.py's standard
format, chunk + '\\n\\n'), (c) the first half behind the labelled memory (the sweep's winner: a '## <path>' line, the chunk, a rule
separator; the format world-rag.bytes trains on). The chunks are a choices.json's (table_retrieve.py at --chunk); a prefix longer than
the window allows is cut from its front. Per set and checkpoint: the means, the gains against (a), better/worse counts; the per-window
bits are saved (no text). The features through this path are checked against the checkpoint's stored ones at startup.
usage: reader_memory.py --tables DIR --choices choices.json --out DIR --ckpts "label=path,..." [--chunk 900 --sets kyle,dlg --limit N --threads 6]"""
import argparse, glob, hashlib, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); CK = os.path.abspath(os.path.join(HERE, "..", "..", "speed-round", "checks"))
TOK = "/games/ailab/lab/flybrain-lab/.worktrees/tok"
sys.path.insert(0, TOK)
import flybrain, flybrain.corpus_memory  # noqa: E402,F401
from scripts.cross_eval import rebuild  # noqa: E402
from flybrain.model.inputs import channel_kwargs  # noqa: E402
sys.path.insert(1, CK)
from use_features import Tables, text_features  # noqa: E402
assert flybrain.__file__.startswith(TOK), flybrain.__file__

ap = argparse.ArgumentParser()
ap.add_argument("--tables", required=True); ap.add_argument("--choices", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ckpts", required=True)
ap.add_argument("--world", default="/games/ailab/corpora/kyle/world/files"); ap.add_argument("--chunk", type=int, default=900)
ap.add_argument("--sep", default="\n\n---\n\n"); ap.add_argument("--plain-sep", default="\n\n")
ap.add_argument("--heldout", default="/games/ailab/corpora/kyle/heldout.bytes"); ap.add_argument("--json", default="/games/ailab/corpora/kyle/heldout.json")
ap.add_argument("--T", type=int, default=2048); ap.add_argument("--kyle-every", type=int, default=1); ap.add_argument("--dlg-every", type=int, default=4)
ap.add_argument("--sets", default="kyle,dlg"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args(); os.umask(0o077); os.makedirs(a.out, exist_ok=True); torch.set_num_threads(a.threads)

t0 = time.time(); chunks = []; paths = []
for p in sorted(glob.glob(os.path.join(a.world, "**", "*"), recursive=True)):
    if not os.path.isfile(p): continue
    b = open(p, "rb").read(); rel = os.path.relpath(p, a.world)
    for s in range(0, max(1, len(b) - a.chunk // 2), a.chunk // 2 if len(b) > a.chunk else a.chunk):
        c = b[s:s + a.chunk]
        if len(c) < 64: continue
        chunks.append(c); paths.append(rel)
digest = hashlib.sha256(b"".join(hashlib.sha256(c).digest() for c in chunks)).hexdigest()
C = json.load(open(a.choices))
if C["pool"]["digest"] != digest: raise SystemExit(f"the choices' pool ({C['pool']['digest'][:16]}, chunk {C['pool']['chunk']}) is not this one ({digest[:16]}, chunk {a.chunk})")
held = open(a.heldout, "rb").read(); meta = json.load(open(a.json)); T = a.T; SETS = a.sets.split(",")
offs = {"kyle": (meta["kyle"]["offset"], meta["kyle"]["windows"], a.kyle_every), "dlg": (meta["dialogue"]["offset"], meta["dialogue"]["windows"], a.dlg_every)}
print(f"pool {len(chunks):,} chunks of <= {a.chunk} bytes; choices: " + ", ".join(f"{s}: {len(C['choices'][s])} windows" for s in SETS) + f" ({time.time() - t0:.0f} s)", flush=True)
tables = Tables(a.tables); LN2 = math.log(2)
out = {"tables": a.tables, "choices": a.choices, "chunk": a.chunk, "ckpts": {}}; lines = []
for spec in a.ckpts.split(","):
    label, ck_path = spec.split("=", 1)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False); orders = list(channel_kwargs(ck["args"])["mem_orders"]); L = int(ck["args"]["seq_len"])
    m = rebuild(ck).eval(); feat_dir = ck["args"]["mem_feat"]
    if os.path.isdir(feat_dir) and os.path.exists(f"{feat_dir}/features.json"):
        info = json.load(open(f"{feat_dir}/features.json")); corpus = np.memmap(info["corpus"], dtype=np.uint8, mode="r")
        A = min(int(info["train_end"]), len(corpus) - 8 * 2048 - 1); B = A + 8 * 2048; H = max(0, A - 65536)
        pr, cf = text_features(tables, np.asarray(corpus[H:B]), orders); bad = 0
        for j, o in enumerate(orders):
            if o in (1004, 1005, 1006, 1007): continue
            bad += int((pr[A - H:B - H, j] != np.load(f"{feat_dir}/pred_n{o}.npy", mmap_mode="r")[A:B]).sum() + (cf[A - H:B - H, j] != np.load(f"{feat_dir}/conf_n{o}.npy", mmap_mode="r")[A:B]).sum())
        if bad: raise SystemExit(f"{label}: the features through this path differ from {feat_dir} at {bad} positions: stop")
        print(f"{label}: features check against {feat_dir}: 0 differences on 8 windows at {A:,}", flush=True)
    truncs = [0]

    @torch.no_grad()
    def bits(prefix: bytes, target: bytes) -> float:
        room = L - len(target)
        if len(prefix) > room: prefix = prefix[-room:]; truncs[0] += 1
        text = prefix + target; N = len(text)
        pred, conf = text_features(tables, text, orders)
        x = torch.from_numpy(np.frombuffer(text[:-1], np.uint8).astype(np.int64))[None]; y = torch.from_numpy(np.frombuffer(text[1:], np.uint8).astype(np.int64))[None]
        logits, _ = m(x, mem_pred=torch.from_numpy(pred[1:N].copy())[None], mem_conf=torch.from_numpy(conf[1:N].copy())[None])
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 256).float(), y.reshape(-1), reduction="none")
        return float(ce[-len(target):].sum() / len(target) / LN2)

    out["ckpts"][label] = {"ckpt": ck_path, "window": L, "sets": {}}
    for s in SETS:
        off, nwin, every = offs[s]; ch = C["choices"][s]; none = []; plain = []; lab = []; has = []; t1 = time.time()
        for n, w in enumerate(range(0, nwin, every)):
            if n >= len(ch) or (a.limit and n >= a.limit): break
            W = held[off + w * T: off + (w + 1) * T]; first, second = W[:T // 2], W[T // 2:]; ids = ch[n]
            b0 = bits(first, second)
            if ids:
                mp = b"".join(chunks[i] + a.plain_sep.encode() for i in ids); ml = b"".join(b"## " + paths[i].encode("utf-8", "replace") + b"\n" + chunks[i] + a.sep.encode() for i in ids)
                b1, b2 = bits(mp + first, second), bits(ml + first, second)
            else: b1 = b2 = b0
            none.append(b0); plain.append(b1); lab.append(b2); has.append(bool(ids))
            if (n + 1) % 50 == 0: print(f"  {label} {s}: {n + 1} windows ({time.time() - t1:.0f} s, {truncs[0]} prefixes cut)", flush=True)
        none, plain, lab, has = map(np.asarray, (none, plain, lab, has))
        out["ckpts"][label]["sets"][s] = {"none": none.tolist(), "plain": plain.tolist(), "labelled": lab.tolist(), "has_memory": has.tolist()}
        gp, gl = plain - none, lab - none
        line = (f"{label} on {'his turns' if s == 'kyle' else 'the dialogue'} ({len(none)} windows, {int(has.sum())} with a memory of <= {a.chunk}x{C['chooser']['top']}): "
                f"the second half after the first half alone {none.mean():.4f}; behind the plain chunks {plain.mean():.4f} ({gp.mean():+.4f}; better >0.02 on {int((gp < -0.02).sum())}, worse on {int((gp > 0.02).sum())}); "
                f"behind the labelled memory {lab.mean():.4f} ({gl.mean():+.4f}; better >0.02 on {int((gl < -0.02).sum())}, worse on {int((gl > 0.02).sum())}; median {np.median(gl):+.4f}); "
                f"{truncs[0]} prefixes cut to fit the window")
        print(line, flush=True); lines.append(line)
json.dump(out, open(os.path.join(a.out, "memory.json"), "w")); open(os.path.join(a.out, "memory.txt"), "a").write("\n".join(lines) + "\n")
print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
