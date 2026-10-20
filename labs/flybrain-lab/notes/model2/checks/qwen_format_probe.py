#!/usr/bin/env python
"""The probe behind the morning of 2026-10-20 (counts only, GPU 1, .venv-qwen): on the three smoke windows of Kyle's turns the tables' chunks cost
+0.09 / +0.35 / +0.58 against warm, each chunk alone helped or cost little, and the same two chunks in the other order swung window 0 from
+0.09 to -0.01: the second-half cost depends on which chunk sits next to the window. This scores the six chunk pairs under four memory
formats (raw "\\n\\n" joins as the runs use; a "---" rule; corpus-style headers; whole lines + the rule). Result: the formats shuffle the
outliers (lines fixes window 2 and breaks window 1, headers the reverse), the scorer is deterministic at n_batch 512/1024/2048, no code
fence or chat marker is involved, and the spans' own run has the same tails (17 windows worse by >0.3, 58 better by >0.3, extremes
+-2.6 on his turns): the per-window cost of a raw memory is heavy-tailed by content, so the paired medians are the honest reading."""
from llama_cpp import Llama
WORLD = "/games/ailab/corpora/kyle/world/files"; CH = 1500; chunks = []
for p in sorted(glob.glob(os.path.join(WORLD, "**", "*"), recursive=True)):
    if not os.path.isfile(p): continue
    b = open(p, "rb").read()
    for s in range(0, max(1, len(b) - CH // 2), CH // 2 if len(b) > CH else CH):
        c = b[s:s + CH]
        if len(c) >= 64: chunks.append(c)
held = open("/games/ailab/corpora/kyle/heldout.bytes", "rb").read(); meta = json.load(open("/games/ailab/corpora/kyle/heldout.json")); T = 2048; off = meta["kyle"]["offset"]
llm = Llama(model_path="/vault/ailab/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf", n_gpu_layers=-1, n_ctx=6144, n_batch=1024, logits_all=True, verbose=False); BOS = llm.token_bos()
def score(prefix, target):
    tp = llm.tokenize(prefix, add_bos=True, special=False) if prefix else [BOS]; tt = llm.tokenize(target, add_bos=False, special=False)
    toks = (tp + tt)[-6144:]; k = len(tt); llm.reset(); llm.eval(toks)
    logits = np.asarray(llm.scores[:len(toks)], dtype=np.float32); pred = logits[len(toks) - k - 1:len(toks) - 1]; targ = np.asarray(toks[len(toks) - k:])
    m = pred.max(axis=1, keepdims=True); lse = m[:, 0] + np.log(np.exp(pred - m).sum(axis=1)); lp = pred[np.arange(k), targ] - lse
    return float(-lp.sum() / math.log(2)) / len(target)
def build(ids, first, fmt):
    cs = [chunks[i] for i in ids]
    if fmt == "raw": return b"".join(c + b"\n\n" for c in cs) + first
    if fmt == "rule": return b"".join(c + b"\n\n---\n\n" for c in cs) + first
    if fmt == "header": return b"".join(b"// ==== notes ====\n" + c + b"\n\n" for c in cs) + b"// ==== chat ====\n" + first
    if fmt == "lines":   # whole lines only: drop the cut first and last lines, then the rule
        out = b""
        for c in cs:
            a = c.find(b"\n") + 1; z = c.rfind(b"\n"); out += (c[a:z] if 0 < a < z else c) + b"\n\n---\n\n"
        return out + first
cases = {"w2 tables": (2, [23583, 18952]), "w2 swapped": (2, [18952, 23583]), "w2 spans": (2, [24483, 24482]), "w0 tables": (0, [18340, 18339]), "w0 swapped": (0, [18339, 18340]),
         "w1 tables": (1, [18890, 18932]), "w1 spans": (1, [18890, 18521]), "w0 spans": (0, [28506, 28524])}
for w in range(3):
    W = held[off + w * T: off + (w + 1) * T]; print(f"w{w} warm {score(W[:1024], W[1024:]):.4f}; with a header line only {score(b'// ==== chat ====' + chr(10).encode() + W[:1024], W[1024:]):.4f}", flush=True)
for fmt in ("raw", "rule", "header", "lines"):
    row = []
    for name, (w, ids) in cases.items():
        W = held[off + w * T: off + (w + 1) * T]; row.append(f"{name} {score(build(ids, W[:1024], fmt), W[1024:]):.4f}")
    print(f"{fmt:7s}: " + "; ".join(row), flush=True)
