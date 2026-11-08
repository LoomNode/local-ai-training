#!/usr/bin/env bash
# The yardstick for model #1's viability test (Kyle, 2026-10-13: "we could use qwen 3.8 27b right?"): Qwen3.8-27B (the 4-bit build on
# /vault) reading the same held-out stream our reader scores (index_project.py's heldout.bytes: the project's files the index never
# saw, with the corpus's file headers), in bits per byte. llama-perplexity (the lab's CUDA build in lab/llama.cpp-qwen4exp, its CUDA 12
# runtime in .deps) in its strided mode (it prints no 'Final estimate' line: the last chunk's running estimate is the PPL): windows of C tokens moved by S, the last S tokens of each scored with C - S tokens of context
# (-c 2048 --ppl-stride 1024: 1,024 to 2,048 tokens, 4 to 8 KB, of context, more than our reader's 2,048 bytes, and across the stream's
# file boundaries; the first C - S tokens of the stream are never scored). It reports PPL = exp(nll / count) over the scored tokens;
# bits/byte = count * ln(PPL) / ln 2 / (the bytes of the scored tokens), the bytes taken as the stream's bytes times count / tokens.
# Runs on one card (16 GB of VRAM); about ten minutes for a megabyte on a 3090. Not the model server: a one-off scoring, no port.
# usage: CUDA_VISIBLE_DEVICES=<uuid> qwen_baseline.sh <heldout.bytes> <out.json> [C=2048] [S=1024]
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
IN=${1:?usage: qwen_baseline.sh <heldout.bytes> <out.json> [C] [S]}; OUT=${2:?}; C=${3:-2048}; S=${4:-1024}
MODEL=${MODEL:-/vault/ailab/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf}
LL=/games/ailab/lab/llama.cpp-qwen4exp
LOG=${OUT%.json}.log
export LD_LIBRARY_PATH=$LL/.deps/cuda-12.8.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
complete() {  # 1 if the log already holds every chunk of a strided run (then the run is not repeated: its numbers are in the log)
  python3 - "$1" "$S" <<'PPL'
import re, sys
text = open(sys.argv[1]).read(); S = int(sys.argv[2])
toks = re.search(r"have (\d+) tokens\. Calculation chunk = (\d+)", text); last = re.findall(r"\[(\d+)\][0-9.]+,", text)
print(1 if toks and last and int(last[-1]) == (int(toks.group(1)) - int(toks.group(2)) + S - 1) // S else 0)
PPL
}
if [ -f "$LOG" ] && [ "$(complete "$LOG")" = 1 ]; then
  echo "qwen baseline $(date '+%F %T'): $LOG holds a complete run; parsing it" >> "$LOG"
else
  echo "qwen baseline $(date '+%F %T'): $MODEL on $IN, C=$C S=$S" > "$LOG"
  "$LL/build-cuda/bin/llama-perplexity" -m "$MODEL" -f "$IN" -c "$C" -b "$C" --ppl-stride "$S" -ngl 99 -t 8 --no-mmap 2>&1 | tee -a "$LOG" | grep --line-buffered "Final estimate\|calculating perplexity\|error" || true
fi
python3 - "$LOG" "$IN" "$OUT" "$MODEL" "$C" "$S" <<'PY'
import json, math, os, re, sys
log, inp, out, model, C, S = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
text = open(log).read()
toks = re.search(r"have (\d+) tokens\. Calculation chunk = (\d+)", text)      # the strided mode's own count (n_ctx grows by S/2 there)
if not toks:
    sys.exit("no token count in " + log)
n_bytes = os.path.getsize(inp)
n_tokens, calc_chunk = int(toks.group(1)), int(toks.group(2))
n_chunks = (n_tokens - calc_chunk + S - 1) // S                        # every chunk scores its last S tokens
m = re.search(r"Final estimate: PPL = ([0-9.]+)", text)
chunks = re.findall(r"\[(\d+)\]([0-9.]+),", text)                     # the strided mode prints no final line; its running estimate
if m:                                                                   # exp(nll / count) after the last chunk is the final PPL
    ppl = float(m.group(1))
elif chunks and int(chunks[-1][0]) == n_chunks:
    ppl = float(chunks[-1][1])
else:
    sys.exit(f"no final estimate in {log}: {chunks[-1][0] if chunks else 0} of {n_chunks} chunks")
count = n_chunks * S
bytes_scored = n_bytes * count / n_tokens
bpb = count * math.log(ppl) / math.log(2) / bytes_scored
res = {"model": os.path.basename(model), "stream": inp, "bytes": n_bytes, "C": C, "S": S, "calc_chunk": calc_chunk, "chunks": n_chunks,
       "scored_tokens": count, "tokens": n_tokens, "ppl": ppl, "bits_per_token": math.log(ppl) / math.log(2), "bpb": bpb, "log": log}
json.dump(res, open(out, "w"), indent=1)
print(f"{os.path.basename(model)} on {os.path.basename(inp)}: PPL {ppl:.4f} over {count} tokens = {res['bits_per_token']:.4f} bits/token, "
      f"about {bpb:.4f} bits/byte ({n_tokens:,} tokens for {n_bytes:,} bytes)")
PY
