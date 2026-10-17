#!/usr/bin/env bash
# The prose sources' scoring corpora and tables (R119): for each source, the budgeted selection of its documents as train + Kyle's
# held-out sets (his turns, then the two-sided dialogue) as the dev region (build_corpus_talk.py --dev-file), then its 2^28 byte tables (key_tables --keys-only, counter_tables,
# assoc_all 28: hour_corpus.sh's pipeline) over that file, so the held-out windows have features under this source's tables.
# Waits for each source's sidecar (/vault/ailab/corpora-src/talk/<src>.json, written last by its extractor); processes sources as they
# land; a source with score/<src>/tables.done is skipped. CPU only, niced and pinned off the owner's cores.
# usage: systemd-run --user --unit talk-tables -p MemoryMax=30G --setenv CUDA_VISIBLE_DEVICES= /bin/bash talk_tables.sh <src> [<src>...]
set -u -o pipefail
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; C2=$SP/notes/model2/checks; CK=$SP/notes/speed-round/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
SRCD=/vault/ailab/corpora-src/talk; SCD=$SRCD/score; KEYROOT=/vault/ailab/flybrain-mem/talk
HELDOUT=/games/ailab/corpora/kyle/heldout.bytes   # Kyle's turns + the two-sided dialogue (kyle_heldout.py)
LOG=$SP/results/tables/talk/tables.log
RUN=(nice -n 19 taskset -c 0-5,8-13)
export CUDA_VISIBLE_DEVICES="" RAYON_NUM_THREADS=12
declare -A BUDGET=([stackexchange]=1000000000 [ubuntu-irc]=300000000 [oasst2]=0 [wikipedia]=500000000 [wikipedia-simple]=0 [docs]=200000000)
mkdir -p "$(dirname "$LOG")" "$SCD" "$KEYROOT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
step() { local name=$1; shift; log "$name: start"; local t0=$SECONDS; if ! "${RUN[@]}" "$@" >> "$LOG" 2>&1; then log "$name: FAILED (see above)"; return 1; fi; log "$name: done in $(( SECONDS - t0 )) s"; }
build() {  # build <src>
  local s=$1 lay="$SCD/$s/layout.json" corpus="$SCD/$s.bytes" mem="$SCD/$s-tables" keyd="$KEYROOT/$s/feat_keys"
  [ -n "${BUDGET[$s]+x}" ] || { log "$s: no budget known"; return 1; }
  mkdir -p "$SCD/$s" "$mem" "$keyd"
  if [ -f "$lay" ] && [ -f "$corpus" ]; then log "$s corpus: exists, skipped"; else
    step "$s corpus" "$PY" "$C2/talk/build_corpus_talk.py" --src-dir "$SRCD" --out "$corpus" --layout-dir "$SCD/$s" --dev-file "$HELDOUT" "$s=${BUDGET[$s]}" || return 1
  fi
  local train_end; train_end=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["train_end"]))' "$lay")
  export CORPUS=$corpus TRAIN_END=$train_end MEM=$mem JOBS=4
  unset OUT ORDERS CHUNK KEYS COUNTERS
  if [ -f "$mem/feat_all28/features.json" ]; then log "$s features: exist, skipped"; else
    local keys_done=1 kid; for kid in 116 132 202 203 301; do [ -f "$keyd/key_n$kid.npy" ] || keys_done=0; done
    if [ "$keys_done" = 1 ]; then log "$s keys: exist, skipped"; else step "$s keys" env OUT="$keyd" "$PY" "$CK/key_tables.py" --keys-only || return 1; fi
    if [ -f "$mem/feat_counters/features.json" ]; then log "$s counters: exist, skipped"; else step "$s counters" env OUT="$mem/feat_counters" "$PY" "$CK/counter_tables.py" || return 1; fi
    step "$s assoc 28" env KEYS="$keyd" COUNTERS="$mem/feat_counters" "$PY" "$CK/assoc_all.py" 28 "$mem/feat_all28" || return 1
    rm -f "$keyd"/key_n*.npy; log "$s keys deleted"
  fi
  touch "$SCD/$s/tables.done"; log "$s: tables done (train $train_end bytes)"
}
cd "$SP"
log "talk tables start: $*"
pending=("$@"); failed=()
while [ ${#pending[@]} -gt 0 ]; do
  progressed=0; rest=()
  for s in "${pending[@]}"; do
    if [ -f "$SCD/$s/tables.done" ]; then log "$s: done already"; progressed=1; continue; fi
    if [ ! -f "$SRCD/$s.json" ]; then rest+=("$s"); continue; fi
    if build "$s"; then progressed=1; else failed+=("$s"); log "$s: FAILED"; fi
  done
  pending=("${rest[@]}")
  if [ ${#pending[@]} -gt 0 ] && [ "$progressed" = 0 ]; then sleep 120; fi
done
log "talk tables: all done (failed: ${failed[*]:-none})"
