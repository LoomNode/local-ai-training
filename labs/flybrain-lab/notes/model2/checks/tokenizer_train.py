#!/usr/bin/env python
"""Model #2's tokenizer: a byte-level BPE (HF tokenizers; GPT-2 byte alphabet and regex pre-tokenizer, no special tokens) trained on the
package corpus's train split (layout.json's train_end), vocabulary 32,768 (Kyle, 2026-10-15: "ok 32k it is"; the probe in
results/tables/tok/: 32k reads 3.18 bytes/token on the clean dev windows, Qwen's 151k 3.64, at a third of the output layer's cost).
Reads the split line by line as UTF-8 with invalid bytes dropped (the trainer wants text; the byte-level model then sees every byte).
usage: tokenizer_train.py [--corpus /games/ailab/corpora/pkgs.bytes] [--layout /games/ailab/corpora/pkgs/layout.json] [--vocab 32768]
                          [--out /games/ailab/corpora/pkgs-tok/bpe32k.json] [--max-bytes N]   (N: train on the split's first N bytes only)
"""
import argparse, json, time
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="/games/ailab/corpora/pkgs.bytes")
ap.add_argument("--layout", default="/games/ailab/corpora/pkgs/layout.json")
ap.add_argument("--vocab", type=int, default=32768)
ap.add_argument("--out", default="/games/ailab/corpora/pkgs-tok/bpe32k.json")
ap.add_argument("--max-bytes", type=int, default=0)
a = ap.parse_args()
end = int(json.load(open(a.layout))["train_end"])
if a.max_bytes: end = min(end, a.max_bytes)

def lines():
    done = 0
    with open(a.corpus, "rb") as f:
        while done < end:
            chunk = f.read(min(64 << 20, end - done))
            if not chunk: break
            done += len(chunk)
            for ln in chunk.splitlines(keepends=True):
                yield ln.decode("utf-8", errors="ignore")

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
tok.decoder = decoders.ByteLevel()
tr = trainers.BpeTrainer(vocab_size=a.vocab, min_frequency=2, show_progress=False, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
t0 = time.time()
tok.train_from_iterator(lines(), tr)
tok.save(a.out)
print(f"trained V={tok.get_vocab_size()} on {end:,} bytes of {a.corpus} in {time.time() - t0:.0f} s -> {a.out}", flush=True)
