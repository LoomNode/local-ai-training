#!/usr/bin/env bash
# mixB's saved tables (T1 of the distillation plan, 2026-10-20): the slice's table inputs come from the frozen tables through
# use_features.text_features, and the mixB build kept only its features. Rebuild the key hashes (transient, ~54 GB on /vault), run the
# table pass with TABLES_ONLY=1 into /vault/ailab/flybrain-mem/talkB/t28 (16 GB; the same recipe as the features: 2^28, two-way, the
# byte orders 4..64 and the keys), delete the keys. CPU only, ~30 min. Log results/tables/distill/t28.log.
set -u -o pipefail
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; CK=$SP/notes/speed-round/checks; PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
export CORPUS=/games/ailab/corpora/talkB.bytes TRAIN_END=1207223865 MEM=/games/ailab/corpora/talkB-tables JOBS=4 CUDA_VISIBLE_DEVICES="" RAYON_NUM_THREADS=12
KEYD=/vault/ailab/flybrain-mem/talkB/final/feat_keys; T28=/vault/ailab/flybrain-mem/talkB/t28; LOG=$SP/results/tables/distill/t28.log
RUN=(nice -n 19 taskset -c 0-5,8-13)
mkdir -p "$KEYD" "$(dirname "$T28")"; cd "$SP"; exec >> "$LOG" 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
free_gb() { df --output=avail -BG "$1" | tail -1 | tr -dc '0-9'; }
log "talkB t28: start; /vault free $(free_gb /vault) GB"
[ "$(free_gb /vault)" -ge 90 ] || { log "under 90 GB free on /vault: stop"; exit 1; }
keys_done=1; for kid in 116 132 202 203 301; do [ -f "$KEYD/key_n$kid.npy" ] || keys_done=0; done
if [ "$keys_done" = 1 ]; then log "keys: exist"; else
  t0=$SECONDS; "${RUN[@]}" env OUT="$KEYD" "$PY" "$CK/key_tables.py" --keys-only || { log "keys FAILED"; exit 1; }; log "keys: done in $((SECONDS - t0)) s; /vault free $(free_gb /vault) GB"
fi
if [ -f "$T28/tables.json" ] && [ -f "$T28/table_n301.npy" ]; then log "t28: exists"; else
  t0=$SECONDS; "${RUN[@]}" env TABLES_ONLY=1 KEYS="$KEYD" COUNTERS="$MEM/feat_counters" "$PY" "$CK/assoc_all.py" 28 "$T28" || { log "t28 FAILED"; exit 1; }; log "t28: done in $((SECONDS - t0)) s"
fi
rm -f "$KEYD"/key_n*.npy; log "keys deleted; t28 $(du -sh "$T28" | cut -f1); /vault free $(free_gb /vault) GB"
[ -e /games/ailab/corpora/talkB-tables/t28 ] || ln -s "$T28" /games/ailab/corpora/talkB-tables/t28
touch "$T28/t28.done"; log "talkB t28: all done"
