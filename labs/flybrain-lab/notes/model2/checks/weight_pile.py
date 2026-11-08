"""The weights' pile: where a reader spends its bits once its memories are credited (Kyle, 2026-10-18: "check that pile then and we can
brainstorm the jobs"). Reads dump_window.py's JSONs (per-byte cost with everything on and with each memory withdrawn, the top-3 guesses).
A byte is credited to the memory whose withdrawal costs it the most, if that is at least 0.25 bit; otherwise it is the weights'. The
weights' bytes are then sorted by kind (a letter starting a word, a letter inside one, a digit, a space, a newline, punctuation) with
their share of all the bits, their mean cost, how often the truth was in the top three (a near miss: the job half done) and how many
were cheap (under 1 bit) or expensive (over 4 bits). Prints counts only, never text (Kyle's held-out windows are in the dumps).
usage: weight_pile.py <label>=<glob> [<label>=<glob> ...]   e.g. kyle='/games/ailab/corpora/kyle/dumps/thb_kyle_w*.json'"""
import glob
import json
import sys
from collections import defaultdict

THRESH = 0.25
KINDS = ("word-start", "word-inside", "digit", "space", "newline", "punct", "other")


def kind(b: int, prev: int) -> str:
    c = chr(b)
    if c.isalpha():
        return "word-inside" if chr(prev).isalpha() else "word-start"
    if c.isdigit(): return "digit"
    if c == " ": return "space"
    if c == "\n": return "newline"
    if 33 <= b < 127: return "punct"
    return "other"


def analyse(label: str, paths: list[str]) -> None:
    src_bytes, src_bits = defaultdict(int), defaultdict(float)
    kind_all_bytes, kind_all_bits = defaultdict(int), defaultdict(float)
    w_bytes, w_bits, w_top3, w_cheap, w_dear = (defaultdict(int), defaultdict(float), defaultdict(int), defaultdict(int), defaultdict(int))
    total_bits = 0.0; total = 0
    for p in paths:
        d = json.load(open(p)); text = d["text"].encode("latin-1"); full = d["cost"]["full"]; top3 = d.get("top3")
        offs = {c[:-4]: v for c, v in d["cost"].items() if c.endswith("_off") and c != "both_off"}
        for i, c in enumerate(full):
            b = text[i]; prev = text[i - 1] if i else 32; k = kind(b, prev)
            deltas = {m: v[i] - c for m, v in offs.items()}
            src = max(deltas, key=deltas.get) if deltas and max(deltas.values()) >= THRESH else "weights"
            total += 1; total_bits += c
            src_bytes[src] += 1; src_bits[src] += c; kind_all_bytes[k] += 1; kind_all_bits[k] += c
            if src == "weights":
                w_bytes[k] += 1; w_bits[k] += c
                if top3 and b in (top3[i][0], top3[i][2], top3[i][4]): w_top3[k] += 1
                if c < 1: w_cheap[k] += 1
                if c > 4: w_dear[k] += 1
    print(f"\n== {label}: {len(paths)} windows, {total:,} bytes, {total_bits / total:.4f} bits/byte")
    print("credited to      bytes   bits   mean")
    for s in sorted(src_bytes, key=lambda s: -src_bits[s]):
        print(f"  {s:12s} {100 * src_bytes[s] / total:5.1f}%  {100 * src_bits[s] / total_bits:5.1f}%  {src_bits[s] / src_bytes[s]:.2f}")
    wb = sum(w_bytes.values()); wbits = sum(w_bits.values())
    print(f"the weights' pile by kind ({wb:,} bytes, {100 * wbits / total_bits:.1f}% of the bits)")
    print("  kind         bytes  of all bits  mean  top3-hit  <1 bit  >4 bits   | all bytes of this kind: share, mean")
    for k in KINDS:
        if not w_bytes[k]: continue
        n = w_bytes[k]
        print(f"  {k:11s} {100 * n / total:5.1f}%   {100 * w_bits[k] / total_bits:5.1f}%     {w_bits[k] / n:.2f}   {100 * w_top3[k] / n:4.0f}%   {100 * w_cheap[k] / n:4.0f}%   {100 * w_dear[k] / n:4.0f}%     | "
              f"{100 * kind_all_bytes[k] / total:5.1f}%  {kind_all_bits[k] / kind_all_bytes[k]:.2f}")


for arg in sys.argv[1:]:
    label, pat = arg.split("=", 1)
    paths = sorted(glob.glob(pat))
    if not paths: print(f"{label}: no files match {pat}"); continue
    analyse(label, paths)
