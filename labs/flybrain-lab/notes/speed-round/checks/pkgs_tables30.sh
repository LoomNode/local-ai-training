#!/usr/bin/env bash
# The package corpus's 2^30 tables saved for use (the use layer; TODO: "the 2^30 tables saved for use, 44 GB on /games; the key hashes
# first"): the same build as the training features' (assoc_all.py, 2^30, the key orders from key_tables.py's hashes) with TABLES_ONLY=1
# (no features written, each order's final table saved) in four passes of orders so the tables in RAM stay under 16 GB a pass, into
# /games/ailab/corpora/pkgs-tables/t30; then check_tables.py against the stored 2^30 held-out features (feat_all30: a lookup in the saved
# tables must reproduce them exactly, as t28 did), and the 220 GB of key hashes on /vault go once the check passes. A step whose output
# exists is skipped. CPU only, niced and pinned off the owner's cores; log results/tables/pkgs_tables30.log.
# usage: systemd-run --user --unit pkgs-t30 /bin/bash pkgs_tables30.sh
set -euo pipefail
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed
CHECKS=$WT/notes/speed-round/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
LAYOUT=/games/ailab/corpora/pkgs/layout.json
LOG=$WT/results/tables/pkgs_tables30.log
T30=/games/ailab/corpora/pkgs-tables/t30
FEAT30=/vault/ailab/flybrain-mem/pkgs/feat_all30
RUN=(nice -n 19 taskset -c 0-5,8-13)
export CUDA_VISIBLE_DEVICES=""
export CORPUS=/games/ailab/corpora/pkgs.bytes
export MEM=/vault/ailab/flybrain-mem/pkgs
export JOBS=${JOBS:-4}
unset OUT ORDERS CHUNK KEYS COUNTERS
mkdir -p "$(dirname "$LOG")" "$T30"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
step() { local name=$1; shift; log "$name: start"; local t0=$SECONDS; if ! "${RUN[@]}" "$@" >> "$LOG" 2>&1; then log "$name: FAILED (see above)"; exit 1; fi; log "$name: done in $(( SECONDS - t0 )) s"; }
TRAIN_END=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["train_end"]))' "$LAYOUT")
export TRAIN_END
N=$(stat -c %s "$CORPUS")
log "pkgs 2^30 use tables: corpus $CORPUS ($N bytes), train_end $TRAIN_END, keys under $MEM, tables to $T30"
keys_done=1; for kid in 116 132 202 203 301; do [ -f "$MEM/feat_keys/key_n$kid.npy" ] || keys_done=0; done
if [ "$keys_done" = 1 ]; then log "keys: exist, skipped"; else
  avail=$(df --output=avail -B1 "$MEM" | tail -1); [ "$avail" -gt $(( N * 40 )) ] || { log "keys: need $(( N * 40 )) bytes on $MEM, $avail free"; exit 1; }
  step keys env OUT="$MEM/feat_keys" "$PY" "$CHECKS/key_tables.py" --keys-only
fi
avail=$(df --output=avail -B1 "$T30" | tail -1); [ "$avail" -gt $(( 48 << 30 )) ] || { log "tables: need 48 GB on $T30, $avail free"; exit 1; }
for pass in "4,8,12" "16,32,64" "116,132" "202,203,301"; do
  done_=1; for o in ${pass//,/ }; do [ -f "$T30/table_n$o.npy" ] || done_=0; done
  if [ "$done_" = 1 ]; then log "pass $pass: tables exist, skipped"; else step "pass $pass" env TABLES_ONLY=1 ORDERS="$pass" "$PY" "$CHECKS/assoc_all.py" 30 "$T30"; fi
done
step "check against feat_all30" "$PY" "$CHECKS/check_tables.py" "$T30" "$FEAT30"
if tail -n 20 "$LOG" | grep -q "tables reproduce the features"; then
  log "check passed: removing the key hashes ($MEM/feat_keys)"; rm -rf "$MEM/feat_keys"
else
  log "check output unclear: the key hashes stay (see above)"
fi
log "pkgs 2^30 use tables: all done ($(du -sh "$T30" | cut -f1))"
