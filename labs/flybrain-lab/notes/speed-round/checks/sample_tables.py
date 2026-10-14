"""Write with a table-fed reader (the use layer, 2026-10-11). The model reads a window of the text and, for every position in it, the 18
table inputs computed from the text and the saved tables (use_features.py); the next byte is drawn from its output, appended, the new
position's inputs computed, and so on. Model #1 is a checkpoint, its saved tables and this.

usage:
  sample_tables.py --ckpt CKPT --tables DIR [--file PATH] [--prompt TEXT | --prompt-file F] [--n 400] [--temperature 0.8] [--top-p 0.95]
                   [--window 2048] [--seed 0] [--device cuda:0]
      --file PATH starts the text with the corpus's file header (\\n// ==== PATH ====\\n): the path tells the model the project and the
      language, as it did in training. The written bytes stream to stdout; a summary line goes to stderr.
  sample_tables.py --ckpt CKPT --tables DIR --score-windows 0,1,2 [--offset O] [--T 2048]
      The end-to-end check: fixed windows of the corpus (from O, default the tables' train_end: the package dev windows) scored through
      this path, the inputs computed from the text from a file header 64 KB before the window; equals score_tables.py's per-window number
      (which reads the stored features), up to the card's float noise.
The model code is the run's (SCORE_CODE, default .worktrees/big1); run it with CUDA_VISIBLE_DEVICES set to one card, or --device cpu."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

CODE = os.environ.get("SCORE_CODE", "/games/ailab/lab/flybrain-lab/.worktrees/big1")
sys.path.insert(0, CODE)
from flybrain.model.inputs import channel_kwargs  # noqa: E402
from scripts.cross_eval import rebuild  # noqa: E402

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from use_features import HEADER, Tables, text_features  # noqa: E402


def load(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    orders = [int(n) for n in channel_kwargs(ck["args"]).get("mem_orders", ())]
    return rebuild(ck).to(device).eval(), orders, ck


@torch.no_grad()
def window_logits(model, data: np.ndarray, pred, conf, lo: int, hi: int, device) -> torch.Tensor:
    """Logits [hi - lo, 256] for the inputs data[lo:hi]: index t predicts byte lo + t + 1 and takes the inputs of that position
    (rows lo + 1 .. hi of pred / conf, which text_features aligns as row p = the inputs for the byte at p)."""
    x = torch.from_numpy(data[lo:hi].astype(np.int64))[None].to(device)
    kw = {}
    if pred is not None:
        kw = {"mem_pred": torch.from_numpy(np.ascontiguousarray(pred[lo + 1:hi + 1]))[None].to(device),
              "mem_conf": torch.from_numpy(np.ascontiguousarray(conf[lo + 1:hi + 1]))[None].to(device)}
    logits, _ = model(x, **kw)
    return logits[0].float()


def draw(logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = F.softmax(logits / temperature, dim=-1).cpu()
    if top_p < 1.0:
        p, idx = probs.sort(descending=True)
        keep = p.cumsum(0) - p < top_p                              # the smallest set whose mass reaches top_p
        probs = torch.zeros_like(probs).scatter(0, idx[keep], p[keep])
    return int(torch.multinomial(probs / probs.sum(), 1, generator=gen))


def write(model, orders, tables, prompt: bytes, n: int, window: int, temperature: float, top_p: float, seed: int, device, out=None):
    text = bytearray(prompt)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(n):
        data = np.frombuffer(bytes(text), dtype=np.uint8)
        pred, conf = text_features(tables, data, orders) if orders else (None, None)
        N = len(data)
        lo = max(0, N - window)
        b = draw(window_logits(model, data, pred, conf, lo, N, device)[-1], temperature, top_p, gen)
        text.append(b)
        if out is not None:
            out.write(bytes([b])); out.flush()
    return bytes(text[len(prompt):]), time.time() - t0


def score_windows(model, orders, tables, corpus, offset: int, T: int, which, device, back=1 << 16):
    """Bits/byte of fixed windows (n non-overlapping windows of T from offset) through this path, the inputs from the text alone
    (computed from the last file header at least `back` bytes before the window, or the nearest one when none is that far back)."""
    res = {}
    for i in which:
        A = offset + i * T
        lo = max(0, A - back - (1 << 17))
        H = bytes(corpus[lo:max(lo, A - back)]).rfind(HEADER)
        if H < 0:                                                      # a window at or near a stream's start, or deep in a long file
            lo = 0; H = bytes(corpus[:A + len(HEADER)]).rfind(HEADER)
        if H < 0:
            raise SystemExit(f"no file header before window {i}")
        H += lo
        data = np.asarray(corpus[H:A + T + 1])
        pred, conf = text_features(tables, data, orders) if orders else (None, None)
        logits = window_logits(model, data, pred, conf, A - H, A - H + T, device)
        y = torch.from_numpy(data[A - H + 1:A - H + T + 1].astype(np.int64)).to(logits.device)
        res[i] = float(F.cross_entropy(logits, y) / math.log(2))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tables", required=True)
    ap.add_argument("--prompt", default=None); ap.add_argument("--prompt-file", default=None); ap.add_argument("--file", default=None)
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95); ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score-windows", default=None); ap.add_argument("--offset", type=int, default=None); ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--score-file", default=None, help="a corpus-format stream (index_project.py's heldout.bytes): every full window of T from its start")
    ap.add_argument("--out", default=None, help="with --score-file: write the per-window bits and the mean to this JSON")
    a = ap.parse_args()
    device = torch.device(a.device)
    tables = Tables(a.tables)
    model, orders, ck = load(a.ckpt, device)
    if a.score_file is not None:
        stream = np.fromfile(a.score_file, dtype=np.uint8)
        n = (len(stream) - 1) // a.T
        res = score_windows(model, orders, tables, stream, 0, a.T, range(n), device)
        per = [res[i] for i in range(n)]
        mean = float(np.mean(per))
        print(f"{a.score_file}: {n} windows of {a.T}, {mean:.4f} bits/byte (ckpt {os.path.basename(a.ckpt)}, tables {a.tables})")
        if a.out:
            json.dump({"bpb": mean, "n": n, "T": a.T, "ckpt": a.ckpt, "tables": a.tables, "stream": a.score_file, "per_window_cold": per},
                      open(a.out, "w"))
        return
    if a.score_windows is not None:
        corpus = np.memmap(tables.meta["corpus"], dtype=np.uint8, mode="r")
        off = tables.meta["train_end"] if a.offset is None else a.offset
        for i, v in score_windows(model, orders, tables, corpus, off, a.T, [int(x) for x in a.score_windows.split(",")], device).items():
            print(f"window {i}: {v:.6f} bits/byte")
        return
    prompt = (HEADER + a.file.encode() + b" ====\n") if a.file else b""
    if a.prompt_file:
        prompt += open(a.prompt_file, "rb").read()
    elif a.prompt:
        prompt += a.prompt.encode().decode("unicode_escape").encode("latin-1")
    out, secs = write(model, orders, tables, prompt, a.n, a.window, a.temperature, a.top_p, a.seed, device, out=sys.stdout.buffer)
    print(f"\n[{len(out)} bytes in {secs:.1f} s, {len(out) / max(secs, 1e-9):.0f} bytes/s; {os.path.basename(a.ckpt)}, tables {a.tables}, "
          f"{len(orders)} inputs, T={a.temperature}, top-p {a.top_p}]", file=sys.stderr)


if __name__ == "__main__":
    main()
