#!/usr/bin/env bash
# The four tests' table side (Kyle, 2026-10-18: "test all 4"): on the prose corpus talk.bytes,
#   1. the key pass again (its keys were deleted after the build), now nine families: the five of ALL18 plus the dictionary (200: the
#      current partial identifier alone; 201: with the last token) and the template at prose scale (104, 108: the last 4 / 8 symbols);
#   2. feat_all28x: feat_all28's eleven lookup orders (hard links) plus the four new orders built by assoc_all.py (ORDERS=200,201,104,108),
#      the counters linked as always: one feature directory for the th_dict_r and th_skel_r arms;
#   3. t28: the saved tables (assoc_all.py TABLES_ONLY=1) of all fifteen lookup orders, for index_project.py (the right book, test 1).
# CPU only, niced and pinned off the owner's cores. usage: systemd-run --user --unit talk-keys2 -p MemoryMax=30G --setenv CUDA_VISIBLE_DEVICES= /bin/bash talk_keys2.sh
set -euo pipefail
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; CK=$SP/notes/speed-round/checks; PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
export CORPUS=/games/ailab/corpora/talk.bytes TRAIN_END=1656781807 MEM=/games/ailab/corpora/talk-tables JOBS=4
TABD=/games/ailab/corpora/talk-tables; KEYD=/vault/ailab/flybrain-mem/talk/final/feat_keys
LOG=$SP/results/tables/talk/keys2.log; RUN=(nice -n 19 taskset -c 0-5,8-13); export CUDA_VISIBLE_DEVICES=""
mkdir -p "$KEYD" "$TABD/feat_all28x" "$TABD/t28"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
step() { local name=$1; shift; log "$name: start"; local t0=$SECONDS; if ! "${RUN[@]}" "$@" >> "$LOG" 2>&1; then log "$name: FAILED (see above)"; exit 1; fi; log "$name: done in $(( SECONDS - t0 )) s"; }
cd "$SP"
keys_done=1; for kid in 116 132 202 203 301 200 201 104 108; do [ -f "$KEYD/key_n$kid.npy" ] || keys_done=0; done
if [ "$keys_done" = 1 ]; then log "keys: exist, skipped"; else step "keys (9 families)" env OUT="$KEYD" "$PY" "$CK/key_tables.py" --keys-only; fi
if [ -f "$TABD/feat_all28x/pred_n200.npy" ] && grep -q '"200"' "$TABD/feat_all28x/features.json" 2>/dev/null; then log "feat_all28x: exists, skipped"; else
  for n in 4 8 12 16 32 64 116 132 202 203 301; do for k in pred conf; do [ -e "$TABD/feat_all28x/${k}_n$n.npy" ] || ln "$TABD/feat_all28/${k}_n$n.npy" "$TABD/feat_all28x/${k}_n$n.npy"; done; done
  [ -f "$TABD/feat_all28x/features.json" ] || cp "$TABD/feat_all28/features.json" "$TABD/feat_all28x/features.json"
  step "assoc 28 (200,201,104,108)" env ORDERS=200,201,104,108 KEYS="$KEYD" COUNTERS="$TABD/feat_counters" "$PY" "$CK/assoc_all.py" 28 "$TABD/feat_all28x"
fi
if [ -f "$TABD/t28/tables.json" ] && [ -f "$TABD/t28/table_n200.npy" ]; then log "t28: exists, skipped"; else
  step "tables only 28 (15 orders)" env TABLES_ONLY=1 KEYS="$KEYD" COUNTERS="$TABD/feat_counters" "$PY" "$CK/assoc_all.py" 28 "$TABD/t28"
fi
touch "$SP/results/tables/talk/keys2.done"; log "keys2: all done"
