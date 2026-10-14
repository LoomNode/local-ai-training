#!/usr/bin/env bash
# The table features of the terminal test's corpus pair (checks/term_corpus.py: corpora/term/term.bytes and plain.bytes, block-aligned),
# each through the package corpus's pipeline (pkgs_tables.sh): keys, counters, the 2^28 two-way tables and features, into
# /vault/ailab/flybrain-mem/term/<term|plain>. CPU only, niced and pinned off the owner's cores; about 0.3 s/MB for the keys, 0.03 for
# the counters, 0.8 for the table pass: about ten minutes a stream at 460 MB. A step whose output exists is skipped. 76 bytes of
# features per corpus byte while building (the key files, 40 of them, can go once the tables are built: feat_keys/).
# usage: term_tables.sh [term|plain ...]   (default: both)       log: results/tables/term_tables.log
set -euo pipefail
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed
CHECKS=$WT/notes/speed-round/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
DIR=/games/ailab/corpora/term
MEMROOT=/vault/ailab/flybrain-mem/term
LOG=$WT/results/tables/term_tables.log
RUN=(nice -n 19 taskset -c 0-5,8-13)
export CUDA_VISIBLE_DEVICES=""
export JOBS=${JOBS:-4}
unset OUT ORDERS CHUNK KEYS COUNTERS
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
step() { local name=$1; shift; log "$name: start"; local t0=$SECONDS; if ! "${RUN[@]}" "$@" >> "$LOG" 2>&1; then log "$name: FAILED (see above)"; exit 1; fi; log "$name: done in $(( SECONDS - t0 )) s"; }
TRAIN_END=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["train_end"]))' "$DIR/layout.json")
export TRAIN_END
[ $# -gt 0 ] || set -- term plain
for which in "$@"; do
  export CORPUS=$DIR/$which.bytes MEM=$MEMROOT/$which
  N=$(stat -c %s "$CORPUS")
  mkdir -p "$MEM"
  log "$which: corpus $CORPUS ($N bytes), train_end $TRAIN_END, mem $MEM"
  keys_done=1; for kid in 116 132 202 203 301; do [ -f "$MEM/feat_keys/key_n$kid.npy" ] || keys_done=0; done
  if [ "$keys_done" = 1 ]; then log "$which keys: exist, skipped"; else step "$which keys" env OUT="$MEM/feat_keys" "$PY" "$CHECKS/key_tables.py" --keys-only; fi
  if [ -f "$MEM/feat_counters/features.json" ]; then log "$which counters: exist, skipped"; else step "$which counters" env OUT="$MEM/feat_counters" "$PY" "$CHECKS/counter_tables.py"; fi
  if [ -f "$MEM/feat_all28/features.json" ]; then log "$which assoc 28: exists, skipped"; else step "$which assoc 28" "$PY" "$CHECKS/assoc_all.py" 28 "$MEM/feat_all28"; fi
done
log "term tables: all done"
