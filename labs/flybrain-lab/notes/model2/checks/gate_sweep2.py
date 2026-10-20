#!/usr/bin/env python
"""The gate sweep, wider (Kyle, 2026-10-20: "exhaustively test all 3 separately"; gate_sweep.py was three signals, one threshold each).
Offline over per-window numbers already scored: for a memory run, the gated result reads the memory's bits where the gate is open and
warm's where it is shut. Every signal is known at use time, before the second half is read:
  the window's own: the first half's self-repetition; Qwen's bits on the first half alone (--first); the plain span chooser's top and
  second hit counts, their margin and the top's share of the first half's distinct spans (--window-choices: span_scores, query_spans);
  the chooser's own (each run's choices.json): its best vote; whether its top chunk is a plain-text file; whether its chunks are one
  file; the memory's bytes; and, when a first_mem run is given, Qwen's bits on the first half GIVEN the memory and the memory's
  effect on the first half (that minus the first half alone).
Four methods, each with the honest number (chosen on the other set, applied here) beside the in-set number: one signal with one
threshold (both directions tried); the AND of two such gates; a linear probe over every signal (least squares on the gain, fitted on
the other set, its threshold chosen there too); and, when a top-1 run is given, the soft gate choosing none / one chunk / two per window
by two probes. The ceiling is the oracle gate (open exactly where the memory helps). Prints numbers only, never text.
usage: gate_sweep2.py --ref qwen/qwen_memory.json --first qwen-gate/qwen_memory.json --window-choices qwen-dedup/choices.json
       --runs "label=memory.json[:choices.json[:first_mem.json[:top1.json]]]" ... --out gate2.txt"""
import argparse, itertools, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True); ap.add_argument("--first", default=None); ap.add_argument("--window-choices", default=None)
ap.add_argument("--runs", nargs="*", default=[]); ap.add_argument("--out", required=True)
ap.add_argument("--heldout", default="/games/ailab/corpora/kyle/heldout.bytes"); ap.add_argument("--json", default="/games/ailab/corpora/kyle/heldout.json")
ap.add_argument("--T", type=int, default=2048); ap.add_argument("--kyle-every", type=int, default=1); ap.add_argument("--dlg-every", type=int, default=4)
ap.add_argument("--span", type=int, default=12); ap.add_argument("--grid", type=int, default=20); ap.add_argument("--pair-grid", type=int, default=10)
a = ap.parse_args(); os.umask(0o077)
MUL = np.uint64(0x9E3779B97F4A7C15)
def span_hashes(b: bytes, S: int) -> np.ndarray:
    d = np.frombuffer(b, dtype=np.uint8).astype(np.uint64); h = np.zeros(len(d) - S + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(S): h = h * MUL + d[j:len(d) - S + 1 + j] + np.uint64(1)
    return h
def repetition(b: bytes) -> float:
    h = span_hashes(b, a.span); _, first = np.unique(h, return_index=True); return 1.0 - len(first) / len(h)
def load(p): return json.load(open(p)) if p and os.path.exists(p) else None

ref = json.load(open(a.ref)); held = open(a.heldout, "rb").read(); meta = json.load(open(a.json)); T = a.T
SETS = {"kyle": ("his turns", meta["kyle"]["offset"], meta["kyle"]["windows"], a.kyle_every), "dlg": ("the dialogue", meta["dialogue"]["offset"], meta["dialogue"]["windows"], a.dlg_every)}
N = {s_: len(ref[s_]["warm"]["per_window"]) for s_ in SETS}
first = load(a.first); wch = load(a.window_choices)

# ---- the window's own signals ----------------------------------------------------------------------------------------------------------------
W_SIG = {}                                                                   # name -> {set: values}
for s_, (nm, off, nwin, every) in SETS.items():
    n = N[s_]
    W_SIG.setdefault("repetition", {})[s_] = np.array([repetition(held[off + w * T: off + w * T + T // 2]) for w in list(range(0, nwin, every))[:n]])
    if first is not None and len(first[s_].get("first", {}).get("per_window", [])) >= n:
        W_SIG.setdefault("first-half bits", {})[s_] = np.array(first[s_]["first"]["per_window"][:n])
    if wch is not None and len(wch["span_scores"][s_]) >= n:
        top = np.array([(x[0] if x else 0) for x in wch["span_scores"][s_][:n]], float); sec = np.array([(x[1] if len(x) > 1 else 0) for x in wch["span_scores"][s_][:n]], float)
        W_SIG.setdefault("top hits", {})[s_] = top; W_SIG.setdefault("second hits", {})[s_] = sec; W_SIG.setdefault("hit margin", {})[s_] = top - sec
        if "query_spans" in wch: W_SIG.setdefault("matched share", {})[s_] = top / np.maximum(1, np.array(wch["query_spans"][s_][:n], float))

def run_signals(ch, fm, s_):
    """The chooser's own signals for one set, from its choices.json and its first_mem run."""
    out = {}; n = N[s_]
    if ch is not None and len(ch["choices"][s_]) >= n:
        C = ch["choices"][s_][:n]
        if "best_vote" in ch: out["best vote"] = np.array(ch["best_vote"][s_][:n], float)
        if "ext" in ch["pool"]:
            ext = ch["pool"]["ext"]; fil = ch["pool"]["file"]
            out["top is .txt"] = np.array([1.0 if c and ext[fil[c[0]]] == ".txt" else 0.0 for c in C])
            out["one file"] = np.array([1.0 if len(c) > 1 and len({fil[i] for i in c}) == 1 else 0.0 for c in C])
        out["memory bytes"] = np.array([sum(min(ch["pool"]["chunk"], 1 << 30) for _ in c) for c in C], float)   # chunks are <= chunk bytes; the count is the signal
    if fm is not None and len(fm[s_].get("first_mem", {}).get("per_window", [])) >= n:
        v = np.array(fm[s_]["first_mem"]["per_window"][:n]); out["first-half bits with the memory"] = v
        if "first-half bits" in W_SIG and s_ in W_SIG["first-half bits"]: out["the memory's effect on the first half"] = v - W_SIG["first-half bits"][s_]
    return out

# ---- the methods ------------------------------------------------------------------------------------------------------------------------------
def gated(g, open_): return float(np.where(open_, g, 0.0).mean())
def grid(v, k): return np.unique(np.quantile(v, np.linspace(0, 1, k + 1)[:-1]))
def single_best(g, v, k):
    """(gain, direction, threshold) of the best one-threshold gate on v; direction +1 opens where v >= t, -1 where v <= t."""
    best = (gated(g, np.ones_like(g, bool)), 0, None)
    for t in grid(v, k):
        for d in (+1, -1):
            gg = gated(g, v >= t if d > 0 else v <= t)
            if gg < best[0] - 1e-12: best = (gg, d, float(t))
    return best
def apply1(v, d, t): return np.ones_like(v, bool) if t is None else (v >= t if d > 0 else v <= t)
def pair_best(g, v1, v2, k):
    best = (gated(g, np.ones_like(g, bool)), None)
    for t1 in grid(v1, k):
        for d1 in (+1, -1):
            o1 = apply1(v1, d1, t1)
            for t2 in grid(v2, k):
                for d2 in (+1, -1):
                    gg = gated(g, o1 & apply1(v2, d2, t2))
                    if gg < best[0] - 1e-12: best = (gg, (d1, float(t1), d2, float(t2)))
    return best
def probe_fit(X, g):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta = np.linalg.lstsq(Z, g, rcond=None)[0]; return mu, sd, beta
def probe_pred(X, fit):
    mu, sd, beta = fit; return np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ beta

lines = []
runs = [("the spans", a.ref, None, None, None)]
for r in a.runs:
    label, rest = r.split("=", 1); parts = (rest.split(":") + [None] * 4)[:4]; runs.append((label, *parts))
for label, mp, chp, fmp, t1p in runs:
    run = load(mp); ch = load(chp); fm = load(fmp); t1 = load(t1p)
    if run is None: lines.append(f"gate2: {label}: no run at {mp}"); continue
    G = {}; S = {}; G1 = {}
    for s_ in SETS:
        if "memory" not in run.get(s_, {}): continue
        warm = np.array(ref[s_]["warm"]["per_window"]); mem = np.array(run[s_]["memory"]["per_window"][:len(warm)])
        if len(mem) != len(warm): continue
        G[s_] = mem - warm; S[s_] = {**{k: v[s_] for k, v in W_SIG.items() if s_ in v}, **run_signals(ch, fm, s_)}
        if t1 is not None and len(t1[s_].get("memory", {}).get("per_window", [])) >= len(warm): G1[s_] = np.array(t1[s_]["memory"]["per_window"][:len(warm)]) - warm
    for s_ in G:
        o = "dlg" if s_ == "kyle" else "kyle"
        if o not in G: continue
        g, go = G[s_], G[o]; nm = SETS[s_][0]; names = [k for k in S[s_] if k in S[o] and np.std(S[s_][k]) > 0 and np.std(S[o][k]) > 0]
        lines.append(f"gate2: {label} on {nm} ({len(g)}): memory {g.mean():+.4f}, oracle gate {np.minimum(g, 0).mean():+.4f} (open {int((g < 0).sum())}); signals: {len(names)}")
        # one signal
        rows = []
        for k in names:
            v, vo = S[s_][k], S[o][k]; rc = float(np.corrcoef(np.argsort(np.argsort(v)), np.argsort(np.argsort(g)))[0, 1])
            gb, db, tb = single_best(g, v, a.grid); _, do, to = single_best(go, vo, a.grid)
            op = apply1(v, do, to); rows.append((gated(g, op), k, rc, gb, float(op.mean()), do, to))
        rows.sort()
        lines.append("  one signal (honest first): " + "; ".join(f"{k} {h:+.4f} (open {fr:.0%}, {'v>=' if d > 0 else 'v<='}{t:.3g}; in-set {gb:+.4f}; rank corr {rc:+.2f})" if t is not None else f"{k} {h:+.4f} (no gate; in-set {gb:+.4f}; rank corr {rc:+.2f})" for h, k, rc, gb, fr, d, t in rows))
        # two signals
        best_pair = None
        for k1, k2 in itertools.combinations(names, 2):
            gb, pb = pair_best(g, S[s_][k1], S[s_][k2], a.pair_grid); _, po = pair_best(go, S[o][k1], S[o][k2], a.pair_grid)
            if po is None: h = g.mean(); fr = 1.0
            else:
                d1, t1_, d2, t2 = po; op = apply1(S[s_][k1], d1, t1_) & apply1(S[s_][k2], d2, t2); h = gated(g, op); fr = float(op.mean())
            if best_pair is None or h < best_pair[0]: best_pair = (h, k1, k2, fr, gb)
        if best_pair: lines.append(f"  two signals, best honest: {best_pair[1]} & {best_pair[2]} {best_pair[0]:+.4f} (open {best_pair[3]:.0%}; its in-set {best_pair[4]:+.4f})")
        # the probe
        if len(names) >= 2:
            X, Xo = np.column_stack([S[s_][k] for k in names]), np.column_stack([S[o][k] for k in names])
            fo = probe_fit(Xo, go); po_ = probe_pred(Xo, fo); _, _, to = single_best(go, po_, a.grid)            # threshold chosen on the other set's own predictions
            p = probe_pred(X, fo); op = apply1(p, -1, to) if to is not None else p < 0
            fi = probe_fit(X, g); pi = probe_pred(X, fi); gi, _, _ = single_best(g, pi, a.grid)
            rc = float(np.corrcoef(np.argsort(np.argsort(p)), np.argsort(np.argsort(g)))[0, 1])
            lines.append(f"  the probe (fitted on the other set): {gated(g, op):+.4f} (open {float(op.mean()):.0%}; rank corr of its prediction with the gain {rc:+.2f}; fitted and thresholded in-set {gi:+.4f})")
        # the soft gate
        if s_ in G1 and o in G1:
            g1, g1o = G1[s_], G1[o]
            lines.append(f"  one chunk only {g1.mean():+.4f}; oracle over none / one / two {np.minimum(np.minimum(g, g1), 0).mean():+.4f}")
            if len(names) >= 2:
                f2 = probe_fit(Xo, go); f1 = probe_fit(Xo, g1o); p2, p1 = probe_pred(X, f2), probe_pred(X, f1)
                choice = np.argmin(np.column_stack([np.zeros(len(g)), p1, p2]), axis=1); soft = np.where(choice == 0, 0.0, np.where(choice == 1, g1, g))
                lines.append(f"  the soft gate (two probes from the other set): {soft.mean():+.4f} (none {int((choice == 0).sum())}, one {int((choice == 1).sum())}, two {int((choice == 2).sum())})")
for l_ in lines: print(l_)
open(a.out, "w").write("\n".join(lines) + "\n"); print("wrote", a.out)
