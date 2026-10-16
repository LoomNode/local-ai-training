# bytes/token of each trained BPE on the three samples, and the per-byte compute of a 12x512 reader at each vocabulary.
import sys, glob, os
from tokenizers import Tokenizer
d = sys.argv[1]
samples = {s: open(f"{d}/data/{s}.bytes", "rb").read() for s in ("dev_clean", "vox", "lab")}
d_model, n_layers, d_ff, T = 512, 12, 1024, 2048
def macs_per_token(V):  # per-token multiply-adds: attention projections + ffn per layer, attention scores over the window, output softmax
    layer = 4 * d_model * d_model + 2 * d_model * d_ff + 2 * d_model * T  # qkvo + ffn + (scores + weighted sum over T positions)
    return n_layers * layer, d_model * V
print("tokenizer   | bytes/token dev vox lab | transformer MACs/token | output MACs/token | total MACs/byte (dev)")
for V in (8192, 16384, 32768, 65536):
    tok = Tokenizer.from_file(f"{d}/bpe_{V}.json")
    bpt = {}
    for s, b in samples.items():
        n = len(tok.encode(b.decode("utf-8", errors="replace")).ids)
        bpt[s] = len(b) / n
    tr, out = macs_per_token(V)
    print(f"bpe {V:6d}  | {bpt['dev_clean']:.3f} {bpt['vox']:.3f} {bpt['lab']:.3f} | {tr/1e6:.1f} M | {out/1e6:.1f} M | {(tr + out) / bpt['dev_clean'] / 1e6:.1f} M")
for name, V, bpt in (("qwen 151k", 151936, 3.317), ("bytes (#1)", 256, 1.0)):
    tr, out = macs_per_token(V)
    print(f"{name:11s}| {bpt:.3f}             | {tr/1e6:.1f} M | {out/1e6:.1f} M | {(tr + out) / bpt / 1e6:.1f} M")
