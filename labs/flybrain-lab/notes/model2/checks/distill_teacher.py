#!/usr/bin/env python
"""The teacher pass (T4 of the distillation plan, 2026-10-20; the spec: docs/superpowers/specs/2026-10-20-distillation-design.md): the big
model reads the slice in windows and, at every byte position, its next-token distribution is turned into a next-byte distribution
(distill_bytes.py: the top K tokens by probability, exact at a token boundary, prefix-conditioned inside a token). Stored per position:
the top 8 bytes and their probabilities (teacher_ids.npy [S, 8] uint8, teacher_p.npy [S, 8] float16), plus teacher.json (the teacher,
K, the dropped mass, the cover, the fallbacks, KB/s) and progress.json (windows done: a stopped pass resumes). Windows of --window
bytes overlap by --overlap: the first --overlap bytes of a window are context only. Prints counts and numbers only, never text.
usage: distill_teacher.py --slice distill.bytes --out DIR --model GGUF [--window 5120 --overlap 1024 --n-ctx 6144 --top-k 512 --top 8]
       [--bench MB (measure KB/s on the first MB, write nothing)] [--limit N windows] [--cpu]"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_bytes import ByteVocab, next_bytes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slice", required=True); ap.add_argument("--out", default=None); ap.add_argument("--model", required=True)
ap.add_argument("--window", type=int, default=5120); ap.add_argument("--overlap", type=int, default=1024); ap.add_argument("--n-ctx", type=int, default=6144)
ap.add_argument("--n-batch", type=int, default=2048); ap.add_argument("--top-k", type=int, default=512); ap.add_argument("--top", type=int, default=8)
ap.add_argument("--bench", type=float, default=0.0, help="MB: measure on the first MB, write nothing"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--cpu", action="store_true"); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args(); os.umask(0o077); torch.set_num_threads(a.threads)
if not a.bench and not a.out: raise SystemExit("--out DIR (or --bench MB)")
from llama_cpp import Llama  # noqa: E402

data = np.memmap(a.slice, dtype=np.uint8, mode="r"); S = len(data)
t0 = time.time()
llm = Llama(model_path=a.model, n_gpu_layers=(0 if a.cpu else -1), n_ctx=a.n_ctx, n_batch=a.n_batch, logits_all=True, verbose=False)
vb = ByteVocab.from_llama(llm); BOS = llm.token_bos(); V = llm.n_vocab()
print(f"teacher {os.path.basename(a.model)}: {V:,} tokens; slice {S:,} bytes; window {a.window} overlap {a.overlap} K {a.top_k} top {a.top} ({time.time() - t0:.0f} s to load)", flush=True)

step = a.window - a.overlap
starts = list(range(0, S, step))
if a.bench: starts = starts[:max(1, math.ceil(a.bench * 1e6 / step))]
if a.limit: starts = starts[:a.limit]
ids_out = p_out = None; done = 0; stats = {"positions": 0, "fallbacks": 0, "dropped_sum": 0.0, "cover_sum": 0.0, "tokens": 0, "skipped_windows": 0, "gpu_s": 0.0, "cpu_s": 0.0}
if not a.bench:
    os.makedirs(a.out, exist_ok=True); os.chmod(a.out, 0o700)
    fi, fp, pr = os.path.join(a.out, "teacher_ids.npy"), os.path.join(a.out, "teacher_p.npy"), os.path.join(a.out, "progress.json")
    if os.path.exists(pr) and os.path.exists(fi) and os.path.exists(fp):
        prog = json.load(open(pr)); done = int(prog["done"]); stats = prog["stats"]
        ids_out = np.load(fi, mmap_mode="r+"); p_out = np.load(fp, mmap_mode="r+")
        if ids_out.shape != (S, a.top): raise SystemExit(f"{fi} is {ids_out.shape}, the slice needs {(S, a.top)}")
        print(f"resuming after {done} of {len(starts)} windows", flush=True)
    else:
        ids_out = np.lib.format.open_memmap(fi, mode="w+", dtype=np.uint8, shape=(S, a.top)); p_out = np.lib.format.open_memmap(fp, mode="w+", dtype=np.float16, shape=(S, a.top))
        for f in (fi, fp): os.chmod(f, 0o600)


def kept_rows(scores: np.ndarray):
    """The top K tokens of every row of logits, with their probabilities under the full softmax, and the dropped mass per row."""
    ids = np.empty((len(scores), a.top_k), np.int64); probs = np.empty((len(scores), a.top_k), np.float32); dropped = np.empty(len(scores), np.float32)
    for lo in range(0, len(scores), 128):
        t = torch.from_numpy(np.ascontiguousarray(scores[lo:lo + 128]))
        lse = torch.logsumexp(t, dim=1); v, i = torch.topk(t, a.top_k, dim=1); pk = torch.exp(v - lse[:, None])
        ids[lo:lo + 128] = i.numpy(); probs[lo:lo + 128] = pk.numpy(); dropped[lo:lo + 128] = (1.0 - pk.sum(1)).numpy()
    return ids, probs, dropped


t_run = time.time(); emitted_bytes = 0
for w in range(done, len(starts)):
    s = starts[w]; e = min(S, s + a.window); emit_from = s + (a.overlap if s > 0 else 0)
    if e <= emit_from: continue
    wb = bytes(data[s:e])
    toks = llm.tokenize(wb, add_bos=False, special=False)
    if len(toks) + 1 > a.n_ctx:                                   # a window of byte-fallback tokens: halve it
        toks = llm.tokenize(wb[: (e - s) // 2], add_bos=False, special=False); e = s + (e - s) // 2; wb = wb[:e - s]
    tb = [vb.tok[t, :vb.len[t]].tobytes() for t in toks]; offs = np.cumsum([0] + [len(b) for b in tb])
    if offs[-1] != len(wb): stats["skipped_windows"] += 1; print(f"  window {w}: tokens cover {offs[-1]} of {len(wb)} bytes, skipped", flush=True); continue
    g0 = time.time(); llm.reset(); llm.eval([BOS] + toks); n = len(toks)
    scores = np.asarray(llm.scores[:n], dtype=np.float32)        # row i predicts token i
    stats["gpu_s"] += time.time() - g0; c0 = time.time()
    ids, probs, dropped = kept_rows(scores)
    first_tok = int(np.searchsorted(offs, emit_from - s, side="right") - 1)   # the token holding the first emitted byte
    for i in range(max(0, first_tok), n):
        tok_k, len_k = vb.kept(ids[i]); p_k = probs[i]; t_bytes = tb[i]
        for j in range(len(t_bytes)):
            pos = s + int(offs[i]) + j
            if pos < emit_from: continue
            bi, bp, mass, cover = next_bytes(tok_k, len_k, p_k, t_bytes[:j], a.top)
            if mass == 0:
                bi = np.zeros(a.top, np.uint8); bp = np.zeros(a.top, np.float32); bi[0] = t_bytes[j]; bp[0] = 1.0; stats["fallbacks"] += 1
            else:
                stats["cover_sum"] += cover
            if ids_out is not None: ids_out[pos] = bi; p_out[pos] = bp.astype(np.float16)
            stats["positions"] += 1
        stats["dropped_sum"] += float(dropped[i]); stats["tokens"] += 1
    stats["cpu_s"] += time.time() - c0; emitted_bytes += e - emit_from; done = w + 1
    if ids_out is not None and (done % 10 == 0 or done == len(starts)):   # progress written atomically: a kill mid-write (the GPU 0 guard) must not leave a torn file
        ids_out.flush(); p_out.flush(); json.dump({"done": done, "windows": len(starts), "stats": stats}, open(pr + ".tmp", "w")); os.replace(pr + ".tmp", pr)
    if done % 50 == 0 or done == len(starts) or a.bench:
        el = time.time() - t_run; print(f"  {done}/{len(starts)} windows, {emitted_bytes / 1e3:.0f} KB at {emitted_bytes / 1e3 / el:.2f} KB/s (gpu {stats['gpu_s']:.0f} s, cpu {stats['cpu_s']:.0f} s); "
                                        f"dropped mass {stats['dropped_sum'] / max(1, stats['tokens']):.4f}, cover {stats['cover_sum'] / max(1, stats['positions'] - stats['fallbacks']):.4f}, fallbacks {stats['fallbacks']} of {stats['positions']}", flush=True)
el = time.time() - t_run
meta = {"teacher": a.model, "slice": a.slice, "slice_bytes": S, "window": a.window, "overlap": a.overlap, "n_ctx": a.n_ctx, "top_k": a.top_k, "top": a.top,
        "windows": len(starts), "done": done, "kb_per_s": emitted_bytes / 1e3 / max(el, 1e-9), "stats": stats,
        "dropped_mass_mean": stats["dropped_sum"] / max(1, stats["tokens"]), "cover_mean": stats["cover_sum"] / max(1, stats["positions"] - stats["fallbacks"]),
        "fallback_rate": stats["fallbacks"] / max(1, stats["positions"]), "built": time.strftime("%F %T")}
if a.bench: print("bench:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in meta.items() if k not in ("stats",)}))
else:
    ids_out.flush(); p_out.flush(); json.dump(meta, open(os.path.join(a.out, "teacher.json"), "w"), indent=1)
    print(f"wrote {a.out}: {stats['positions']:,} positions in {done} windows, {meta['kb_per_s']:.2f} KB/s")
