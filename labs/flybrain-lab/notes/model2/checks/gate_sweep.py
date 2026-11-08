#!/usr/bin/env python
"""The easy-window gate (follow-up 3 of the tables' arm, 2026-10-20, Kyle: "Queue up those 3"): a memory of any kind hurts the windows
Qwen already reads easily (the 13 turns under 0.5 warm: +0.27 under the spans' chunks, +1.14 under the tables') and helps the hard ones
most. Offline, from per-window numbers already scored: for each memory run, the gated result reads the memory's bits where the gate is
open and warm's where it is shut. Three signals known before the second half is read: Qwen's bits on the first half alone (--first:
qwen_memory.py --conditions first), the plain span chooser's hit count for its top chunk (--choices: table_retrieve.py's span_scores),
and the first half's self-repetition (the share of its 12-byte spans seen earlier in it; text only). Each signal is swept over a grid of
its own quantiles; the honest number is the threshold chosen on the other set (the dialogue's applied to his turns and the reverse);
the ceiling is the oracle gate (open exactly where the memory helps). Prints numbers only, never text.
usage: gate_sweep.py --ref qwen/qwen_memory.json [--first qwen-gate/qwen_memory.json] [--choices DIR/choices.json] [--runs "label=json" ...] --out gate.txt"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="the span run: warm and the spans' memory per window"); ap.add_argument("--first", default=None)
ap.add_argument("--choices", default=None); ap.add_argument("--runs", nargs="*", default=[], help="label=qwen_memory.json of other memory runs to gate")
ap.add_argument("--heldout", default="/games/ailab/corpora/kyle/heldout.bytes"); ap.add_argument("--json", default="/games/ailab/corpora/kyle/heldout.json")
ap.add_argument("--T", type=int, default=2048); ap.add_argument("--kyle-every", type=int, default=1); ap.add_argument("--dlg-every", type=int, default=4)
ap.add_argument("--span", type=int, default=12); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.umask(0o077)
MUL = np.uint64(0x9E3779B97F4A7C15)
def span_hashes(b: bytes, S: int) -> np.ndarray:
    d = np.frombuffer(b, dtype=np.uint8).astype(np.uint64); h = np.zeros(len(d) - S + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(S): h = h * MUL + d[j:len(d) - S + 1 + j] + np.uint64(1)
    return h
def repetition(b: bytes) -> float:
    """The share of the text's spans that occurred earlier in the same text."""
    h = span_hashes(b, a.span); _, first = np.unique(h, return_index=True); return 1.0 - len(first) / len(h)

ref = json.load(open(a.ref)); held = open(a.heldout, "rb").read(); meta = json.load(open(a.json)); T = a.T
sets = {"kyle": ("his turns", meta["kyle"]["offset"], meta["kyle"]["windows"], a.kyle_every), "dlg": ("the dialogue", meta["dialogue"]["offset"], meta["dialogue"]["windows"], a.dlg_every)}
first = json.load(open(a.first)) if a.first and os.path.exists(a.first) else None
choices = json.load(open(a.choices)) if a.choices and os.path.exists(a.choices) else None
runs = [("the spans", ref)] + [(r.split("=", 1)[0], json.load(open(r.split("=", 1)[1]))) for r in a.runs if os.path.exists(r.split("=", 1)[1])]

signals = {}                                                          # name -> (direction, {set: values}); open where value >= t (+1) or <= t (-1)
for s_, (nm, off, nwin, every) in sets.items():
    n = len(ref[s_]["warm"]["per_window"])
    rep = np.array([repetition(held[off + w * T: off + w * T + T // 2]) for w in list(range(0, nwin, every))[:n]])
    signals.setdefault("repetition", (-1, {}))[1][s_] = rep
    if first is not None and "first" in first.get(s_, {}) and len(first[s_]["first"]["per_window"]) >= n:
        signals.setdefault("first-half bits", (+1, {}))[1][s_] = np.array(first[s_]["first"]["per_window"][:n])
    if choices is not None and len(choices["span_scores"][s_]) >= n:
        signals.setdefault("span hits", (+1, {}))[1][s_] = np.array([(x[0] if x else 0) for x in choices["span_scores"][s_][:n]], dtype=float)

def gated(g: np.ndarray, open_: np.ndarray) -> float: return float(np.where(open_, g, 0.0).mean())
def grid(v: np.ndarray) -> np.ndarray: return np.unique(np.quantile(v, np.linspace(0, 0.95, 20)))
def best_threshold(g, v, direction):
    """The threshold on the grid of v with the lowest gated gain (most negative), and that gain."""
    best = (0.0, None)
    for t in grid(v):
        gg = gated(g, v >= t if direction > 0 else v <= t)
        if gg < best[0]: best = (gg, float(t))
    return best
lines = []
for label, run in runs:
    for s_, (nm, *_rest) in sets.items():
        if "memory" not in run.get(s_, {}): continue
        warm = np.array(ref[s_]["warm"]["per_window"]); mem = np.array(run[s_]["memory"]["per_window"][:len(warm)])
        if len(mem) != len(warm): continue
        g = mem - warm; n = len(g); other = "dlg" if s_ == "kyle" else "kyle"
        parts = [f"gate: {label} on {nm} ({n}): warm {warm.mean():.4f}, memory {g.mean():+.4f}, oracle gate {np.minimum(g, 0).mean():+.4f} (open {int((g < 0).sum())})"]
        for sig, (direction, vals) in signals.items():
            if s_ not in vals or other not in vals: continue
            v = vals[s_]; rank_corr = float(np.corrcoef(np.argsort(np.argsort(v)), np.argsort(np.argsort(g)))[0, 1])
            gb, tb = best_threshold(g, v, direction)
            # the honest number: the other set's best threshold, applied here
            go, to = best_threshold(run_other := (np.array(run[other]["memory"]["per_window"]) - np.array(ref[other]["warm"]["per_window"])), vals[other], direction)
            if to is None: cross = g.mean(); frac = 1.0
            else: op = v >= to if direction > 0 else v <= to; cross = gated(g, op); frac = float(op.mean())
            own = f"in-set best {gb:+.4f} at {tb:.3g} (open {float((v >= tb).mean() if direction > 0 else (v <= tb).mean()):.0%})" if tb is not None else "in-set best: no gate helps"
            parts.append(f"{sig}: rank corr with the gain {rank_corr:+.2f}; {own}; the other set's threshold {('%.3g' % to) if to is not None else 'none'} gives {cross:+.4f} (open {frac:.0%})")
        lines.append("; ".join(parts))
for l_ in lines: print(l_)
open(a.out, "w").write("\n".join(lines) + "\n"); print("wrote", a.out)
