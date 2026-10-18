#!/usr/bin/env bash
# The prose corpus proper (R124, 2026-10-17: mixD, the union of the mixes that paid on Kyle's held-out sets) and everything model #2's
# pair at the hour needs on it, hour_corpus.sh's pipeline on build_corpus_talk.py's output:
#   1. the corpus       /games/ailab/corpora/talk.bytes + talk/layout.json, manifest.json: mixD's recipe (stackexchange 1 GB, wikipedia
#                       250 MB, the package corpus's first GB 500 MB, oasst2 and docs whole), 5 % of each source's documents held out for
#                       dev and as many for test (cut on document headers; exact duplicates dropped), then Kyle's held-out sets as the
#                       trailing region (tail_offset: his turns, then the dialogue; heldout.json has their sizes). Mode 0600: his text is in it.
#   2. the tokenizer    /games/ailab/corpora/talk-tok/bpe32k.json, a 32k byte-level BPE on the train split (tokenizer_train.py)
#   3. the token stream /games/ailab/corpora/talk-tok/ (tokenize_corpus.py; the directory lives on /vault, linked from /games)
#   4. byte features    /games/ailab/corpora/talk-tables/feat_all28 (keys on /vault, deleted after; ~65 GB)
#   5. token features   /games/ailab/corpora/talk-tok/feat28 (tok_tables.py at 2^28; ~36 GB)
# A step whose output exists is skipped. CPU only, niced and pinned off the owner's cores. Done file talk/build.done.
# NAME=talkB RECIPE="..." builds another mix beside it (talkB.bytes, talkB/, talkB-tok, talkB-tables; log buildB.log, done file talkB/build.done):
#   2026-10-18, Kyle ("mixB is fine for now", R125): NAME=talkB with mixB's recipe (stackexchange 1 GB, wikipedia 250 MB, oasst2 and docs; no code).
# usage: systemd-run --user --unit talk-build -p MemoryMax=30G --setenv CUDA_VISIBLE_DEVICES= [--setenv NAME=talkB --setenv RECIPE="..."] /bin/bash talk_corpus.sh   (log results/tables/talk/build.log)
set -euo pipefail
umask 077
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed
C2=$SP/notes/model2/checks; CK=$SP/notes/speed-round/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
SRCD=/vault/ailab/corpora-src/talk
NAME=${NAME:-talk}; SUF=${NAME#talk}
CORPUS=/games/ailab/corpora/$NAME.bytes; LAYDIR=/games/ailab/corpora/$NAME; LAYOUT=$LAYDIR/layout.json
TOKD=/games/ailab/corpora/$NAME-tok; TOKD_REAL=/vault/ailab/flybrain-mem/$NAME-tok
TABD=/games/ailab/corpora/$NAME-tables; KEYD=/vault/ailab/flybrain-mem/$NAME/final/feat_keys
HELDOUT=/games/ailab/corpora/kyle/heldout.bytes
RECIPE=${RECIPE:-"stackexchange=1000000000 oasst2=0 docs=0 wikipedia=250000000 pkgs1g=500000000"}
LOG=$SP/results/tables/talk/build$SUF.log
RUN=(nice -n 19 taskset -c 0-5,8-13)
export CUDA_VISIBLE_DEVICES="" RAYON_NUM_THREADS=12
mkdir -p "$(dirname "$LOG")" "$LAYDIR" "$TOKD_REAL" "$TABD" "$KEYD"
[ -e "$TOKD" ] || ln -s "$TOKD_REAL" "$TOKD"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
step() { local name=$1; shift; log "$name: start"; local t0=$SECONDS; if ! "${RUN[@]}" "$@" >> "$LOG" 2>&1; then log "$name: FAILED (see above)"; exit 1; fi; log "$name: done in $(( SECONDS - t0 )) s"; }
cd "$SP"
log "$NAME corpus build start: $RECIPE"
if [ -f "$LAYOUT" ] && [ -f "$CORPUS" ]; then log "corpus: exists, skipped"; else
  step "corpus" "$PY" "$C2/talk/build_corpus_talk.py" --src-dir "$SRCD" --out "$CORPUS" --layout-dir "$LAYDIR" --seed 0 --holdout 0.05 --dedup --tail-file "$HELDOUT" $RECIPE
  chmod 600 "$CORPUS"
fi
TRAIN_END=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["train_end"]))' "$LAYOUT")
export TRAIN_END CORPUS
if [ -f "$TOKD/bpe32k.json" ]; then log "tokenizer: exists, skipped"; else
  step "tokenizer" "$PY" "$C2/tokenizer_train.py" --corpus "$CORPUS" --layout "$LAYOUT" --vocab 32768 --out "$TOKD/bpe32k.json"
fi
if [ -f "$TOKD/layout.json" ]; then log "stream: exists, skipped"; else
  step "stream" "$PY" "$C2/tokenize_corpus.py" --corpus "$CORPUS" --layout "$LAYOUT" --tokenizer "$TOKD/bpe32k.json" --out "$TOKD"
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
touch "$LAYDIR/build.done"
log "$NAME corpus build: all done"
