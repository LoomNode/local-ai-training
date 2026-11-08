#!/usr/bin/env python
"""Arm B (2026-11-06): the 27B's sampled targets over the training sequences (Sparse Logit Sampling: K ids drawn from the teacher's
distribution at every loss position, an unbiased estimate of the full-vocabulary cross-entropy, instead of a biased top-k list). The GGUF
teacher runs through llama-cpp-python with every position's logits kept; torch samples on the GPU. Writes memmaps under --out: ids.npy
[P, K] int32, lp.npy [P, K] float16 (the teacher's log-probability of each sample), true_lp.npy [P] float32 (of the true token), P = the
loss positions of train.npz in order; progress.json for resume. Numbers and ids only.
usage: teacher_pass.py --data DIR --out DIR [--model GGUF] [--K 16] [--n-ctx 6144] [--limit N]"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import TEACHER_GGUF

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--model", default=TEACHER_GGUF); ap.add_argument("--K", type=int, default=16)
ap.add_argument("--n-ctx", type=int, default=6144); ap.add_argument("--n-batch", type=int, default=1024); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.umask(0o077); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed)
z = np.load(os.path.join(a.data, "train.npz")); tokens, seq_start, loss_start, kind = z["tokens"], z["seq_start"], z["loss_start"], z["kind"]
S = len(loss_start) if not a.limit else min(a.limit, len(loss_start)); L = seq_start[1:] - seq_start[:-1]; nloss = L - loss_start; pos_start = np.concatenate([[0], np.cumsum(nloss)])
P = int(pos_start[S]); assert L[:S].max() <= a.n_ctx, "a sequence is longer than n_ctx"
prog = os.path.join(a.out, "progress.json"); done = json.load(open(prog))["done"] if os.path.exists(prog) else 0
mode = "r+" if done else "w+"
ids = np.lib.format.open_memmap(os.path.join(a.out, "ids.npy"), mode=mode, dtype=np.int32, shape=(P, a.K))
lps = np.lib.format.open_memmap(os.path.join(a.out, "lp.npy"), mode=mode, dtype=np.float16, shape=(P, a.K))
tlp = np.lib.format.open_memmap(os.path.join(a.out, "true_lp.npy"), mode=mode, dtype=np.float32, shape=(P,))
from llama_cpp import Llama
llm = Llama(model_path=a.model, n_gpu_layers=-1, n_ctx=a.n_ctx, n_batch=a.n_batch, logits_all=True, verbose=False)
print(f"teacher pass: {S:,} sequences, {P:,} loss positions, K={a.K}, resuming at {done}" if done else f"teacher pass: {S:,} sequences, {P:,} loss positions, K={a.K}", flush=True)
t0 = time.time(); nats = {0: [0.0, 0], 1: [0.0, 0], 2: [0.0, 0]}; ntok = 0
for i in range(done, S):
    toks = tokens[seq_start[i]:seq_start[i + 1]].tolist(); ls = int(loss_start[i]); n = len(toks)
    llm.reset(); llm.eval(toks)
    pred = np.asarray(llm.scores[ls - 1:n - 1], dtype=np.float32); targ = torch.tensor(toks[ls:], device="cuda")
    p0 = int(pos_start[i])
    for c in range(0, len(pred), 512):
        x = torch.from_numpy(pred[c:c + 512]).cuda(); lsm = torch.log_softmax(x, -1); t = targ[c:c + 512]
        s = torch.multinomial(lsm.exp(), a.K, replacement=True)
        ids[p0 + c:p0 + c + len(x)] = s.cpu().numpy(); lps[p0 + c:p0 + c + len(x)] = lsm.gather(1, s).cpu().numpy().astype(np.float16)
        tl = lsm.gather(1, t[:, None])[:, 0]; tlp[p0 + c:p0 + c + len(x)] = tl.cpu().numpy(); nats[int(kind[i])][0] += float(-tl.sum()); nats[int(kind[i])][1] += len(x)
    ntok += n
    if (i + 1) % 50 == 0 or i + 1 == S:
        ids.flush(); lps.flush(); tlp.flush(); json.dump({"done": i + 1, "of": S, "positions": P, "K": a.K, "seconds": round(time.time() - t0)}, open(prog + ".tmp", "w")); os.replace(prog + ".tmp", prog)
        print(f"  {i + 1}/{S} sequences, {ntok:,} tokens, {ntok / (time.time() - t0):.0f} tok/s; teacher nats/token so far: files {nats[0][0] / max(1, nats[0][1]):.3f}, sessions {nats[1][0] / max(1, nats[1][1]):.3f}, replay {nats[2][0] / max(1, nats[2][1]):.3f}", flush=True)
json.dump({"done": S, "of": S, "positions": P, "K": a.K, "seconds": round(time.time() - t0), "teacher_nats_per_token": {k: v[0] / max(1, v[1]) for k, v in nats.items()}, "model": a.model}, open(os.path.join(a.out, "teacher.json"), "w"), indent=1)
open(os.path.join(a.out, "pass.done"), "w").close(); print("done", flush=True)
