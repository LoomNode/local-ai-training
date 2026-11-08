#!/usr/bin/env bash
# The hour arms' corpus pair for model #2 (plan docs/superpowers/plans/2026-10-16-model2-tokens.md, Task 3): pkgs1g.bytes = the package
# corpus's first 1 GB of training text + its dev region (82 MB), with a layout (train_end = dev_offset = 1e9), then on it:
#   1. the token stream  /games/ailab/corpora/pkgs1g-tok/            (tokenize_corpus.py with the 32k tokenizer)
#   2. the byte features /games/ailab/corpora/pkgs1g-tables/feat_all28 (key_tables.py --keys-only, counter_tables.py, assoc_all.py 28;
#      the keys on /vault (43 GB, deleted after the table pass), the rest on /games)
#   3. the token features /games/ailab/corpora/pkgs1g-tok/feat28       (tok_tables.py at 2^28)
# A step whose output exists is skipped. CPU only, niced and pinned off the owner's cores. log: results/tables/tok/pkgs1g_build.log
# usage: systemd-run --user --unit pkgs1g-build -p MemoryMax=30G --setenv CUDA_VISIBLE_DEVICES= /bin/bash hour_corpus.sh
set -euo pipefail
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed
C2=$SP/notes/model2/checks; CK=$SP/notes/speed-round/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
SRC=/games/ailab/corpora/pkgs.bytes; SRCLAY=/games/ailab/corpora/pkgs/layout.json
CORPUS=/games/ailab/corpora/pkgs1g.bytes; LAYDIR=/games/ailab/corpora/pkgs1g; LAYOUT=$LAYDIR/layout.json
TOKD=/games/ailab/corpora/pkgs1g-tok; TABD=/games/ailab/corpora/pkgs1g-tables; KEYD=/vault/ailab/flybrain-mem/pkgs1g/feat_keys
TOKENIZER=/games/ailab/corpora/pkgs-tok/bpe32k.json
LOG=$SP/results/tables/tok/pkgs1g_build.log
RUN=(nice -n 19 taskset -c 0-5,8-13)
export CUDA_VISIBLE_DEVICES="" RAYON_NUM_THREADS=12
mkdir -p "$(dirname "$LOG")" "$LAYDIR" "$TOKD" "$TABD" "$KEYD"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
step() { local name=$1; shift; log "$name: start"; local t0=$SECONDS; if ! "${RUN[@]}" "$@" >> "$LOG" 2>&1; then log "$name: FAILED (see above)"; exit 1; fi; log "$name: done in $(( SECONDS - t0 )) s"; }
cd "$SP"
if [ -f "$LAYOUT" ] && [ -f "$CORPUS" ]; then log "corpus: exists, skipped"; else
  step "corpus" "$PY" - "$SRC" "$SRCLAY" "$CORPUS" "$LAYOUT" <<'PY'
import json, sys, numpy as np
src, srclay, dst, lay = sys.argv[1:5]
L = json.load(open(srclay)); TRAIN = 1_000_000_000
m = np.memmap(src, dtype=np.uint8, mode="r")
# cut the train part at the last file header before 1 GB so the train split ends on a file boundary
hdr = b"\n\n// ==== "
tail = m[TRAIN - (1 << 22):TRAIN].tobytes(); k = tail.rfind(hdr)
assert k >= 0, "no file header in the last 4 MB before 1 GB"
train_end = TRAIN - (1 << 22) + k + 2            # the header's "/" (table_common's cut point)
dev_off, dev_n = int(L["dev_offset"]), int(L["dev_bytes"])
with open(dst, "wb") as f:
    f.write(m[:train_end].tobytes()); f.write(m[dev_off:dev_off + dev_n].tobytes())
out = {"corpus": dst, "built_from": src, "built": "hour_corpus.sh", "corpus_bytes": train_end + dev_n, "train_end": train_end, "dev_offset": train_end,
       "dev_bytes": dev_n, "test_offset": train_end + dev_n, "test_bytes": 0, "note": "the package corpus's first 1 GB (cut at a file header) + its dev region; the dev windows are the same bytes as pkgs.bytes's at dev_offset"}
json.dump(out, open(lay, "w"), indent=1); print(out)
PY
fi
TRAIN_END=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["train_end"]))' "$LAYOUT")
export TRAIN_END CORPUS
if [ -f "$TOKD/layout.json" ]; then log "stream: exists, skipped"; else
  step "stream" "$PY" "$C2/tokenize_corpus.py" --corpus "$CORPUS" --layout "$LAYOUT" --tokenizer "$TOKENIZER" --out "$TOKD"
fi
export MEM=$TABD JOBS=4
unset OUT ORDERS CHUNK KEYS COUNTERS
keys_done=1; for kid in 116 132 202 203 301; do [ -f "$KEYD/key_n$kid.npy" ] || keys_done=0; done
if [ -f "$TABD/feat_all28/features.json" ]; then log "byte features: exist, skipped"; else
  if [ "$keys_done" = 1 ]; then log "keys: exist, skipped"; else step "keys" env OUT="$KEYD" "$PY" "$CK/key_tables.py" --keys-only; fi
  if [ -f "$TABD/feat_counters/features.json" ]; then log "counters: exist, skipped"; else step "counters" env OUT="$TABD/feat_counters" "$PY" "$CK/counter_tables.py"; fi
  step "assoc 28" env KEYS="$KEYD" COUNTERS="$TABD/feat_counters" "$PY" "$CK/assoc_all.py" 28 "$TABD/feat_all28"
  rm -f "$KEYD"/key_n*.npy; log "keys deleted"
fi
if [ -f "$TOKD/feat28/features.json" ]; then log "token features: exist, skipped"; else
  step "token tables 28" "$PY" "$C2/tok_tables.py" --stream "$TOKD" --out "$TOKD/feat28" --log2 28 --jobs 6
fi
log "pkgs1g build: all done"
