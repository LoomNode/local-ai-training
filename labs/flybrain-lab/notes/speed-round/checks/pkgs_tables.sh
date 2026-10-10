#!/usr/bin/env bash
# The table features of the packages corpus (/games/ailab/corpora/pkgs.bytes, ~5.46 GB), in order:
#   1. key_tables.py --keys-only  -> $MEM/feat_keys/key_n{116,132,202,203,301}.npy   (uint64: 40 bytes per corpus byte, ~220 GB)
#   2. counter_tables.py          -> $MEM/feat_counters (pred/conf of 1001-1007, 14 bytes per corpus byte, ~76 GB)
#   3. assoc_all.py 28            -> $MEM/feat_all28 (two-way 2^28 tables for 11 orders, 22 bytes per corpus byte, ~120 GB; tables 11.8 GB RAM)
# A step is skipped when its output is already complete (the five key files; features.json for the other two), so the script can be
# rerun after a failure. TRAIN_END comes from the corpus's layout.json. CPU only, niced and pinned off the owner's cores; log in
# results/tables/pkgs_tables.log. Takes hours: about 0.3 s/MB for the keys and 0.03 s/MB for the counters with 4 workers, 0.8 s/MB for
# the one-CPU table pass (rates of a 70 MB slice of the Linux corpus).
set -euo pipefail

WT=/games/ailab/lab/flybrain-lab/.worktrees/speed
CHECKS=$WT/notes/speed-round/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
LAYOUT=/games/ailab/corpora/pkgs/layout.json
LOG=$WT/results/tables/pkgs_tables.log
RUN=(nice -n 19 taskset -c 0-5,8-13)

export CUDA_VISIBLE_DEVICES=""
export CORPUS=/games/ailab/corpora/pkgs.bytes
export MEM=/vault/ailab/flybrain-mem/pkgs
export JOBS=${JOBS:-4}
unset OUT ORDERS CHUNK KEYS COUNTERS

mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ -f "$CORPUS" ] || { log "no corpus at $CORPUS"; exit 1; }
[ -f "$LAYOUT" ] || { log "no layout at $LAYOUT"; exit 1; }
TRAIN_END=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["train_end"]))' "$LAYOUT")
export TRAIN_END
N=$(stat -c %s "$CORPUS")
[ "$TRAIN_END" -gt 0 ] && [ "$TRAIN_END" -lt "$N" ] || { log "train_end $TRAIN_END is not inside the corpus ($N bytes)"; exit 1; }
mkdir -p "$MEM"
log "pkgs tables: corpus $CORPUS ($N bytes), train_end $TRAIN_END, mem $MEM, $JOBS workers"

need_space() {   # need_space <bytes per corpus byte> <step>: refuse to start a step the disk cannot hold
    local avail need
    avail=$(df --output=avail -B1 "$MEM" | tail -1)
    need=$(( N * $1 ))
    if [ "$avail" -lt "$need" ]; then log "$2: needs $need bytes, $avail free on $MEM"; exit 1; fi
}

step() {         # step <name> <command...>: run under nice/taskset, output appended to the log
    local name=$1; shift
    log "$name: start"
    local t0=$SECONDS
    if ! "${RUN[@]}" "$@" >> "$LOG" 2>&1; then log "$name: FAILED (see above)"; exit 1; fi
    log "$name: done in $(( SECONDS - t0 )) s"
}

keys_done=1
for kid in 116 132 202 203 301; do [ -f "$MEM/feat_keys/key_n$kid.npy" ] || keys_done=0; done
if [ "$keys_done" = 1 ]; then
    log "keys: all five key files exist, skipped"
else
    need_space 40 keys
    step keys env OUT="$MEM/feat_keys" "$PY" "$CHECKS/key_tables.py" --keys-only
fi

if [ -f "$MEM/feat_counters/features.json" ]; then
    log "counters: features.json exists, skipped"
else
    need_space 14 counters
    step counters env OUT="$MEM/feat_counters" "$PY" "$CHECKS/counter_tables.py"
fi

if [ -f "$MEM/feat_all28/features.json" ]; then
    log "assoc 28: features.json exists, skipped"
else
    need_space 22 "assoc 28"
    step "assoc 28" "$PY" "$CHECKS/assoc_all.py" 28 "$MEM/feat_all28"
fi
log "pkgs tables: all done"
