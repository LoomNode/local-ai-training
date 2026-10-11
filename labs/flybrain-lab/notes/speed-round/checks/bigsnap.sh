#!/usr/bin/env bash
# Cooled-down snapshots of a big run at doubling milestones (a side agent's point, 2026-10-11, checked on this session's restart test:
# the constant-rate run read 0.034 above the cosine run at step 20,000 and its cooldown then dropped it 0.05; the stable phase sits above
# the model's level, so the stop rule and model #1 come from snapshots, not from the raw curve). At each milestone m, once the run has
# saved its checkpoint there, a cooldown of K = FRAC x m steps resumes from it on this card beside the run (bigtrain.sh with
# NAME=<run>_snap<m>, STEPS=m+K, COOLDOWN=K: the rate falls linearly to a tenth over K steps; CKPT_EVERY=K), then the cooled model
# (results/ckpt/<run>_snap<m>_transformer.pt in .worktrees/big1) is scored on the clean windows: dev through 2^28 (the bar), the kernel
# windows, and both through 2^30. The scores land in results/tables/big/k_<run>snap_s<m>_<tag>.json (summary.json lists them as
# "<run>snap"). The stop rule: a doubling of the steps buys less than STOP_DELTA (0.005) on the clean dev between two snapshots; this
# script reports it in its log ("STOP RULE MET") and goes on; the stop is Kyle's call.
# usage: RUN=big12 LAYERS=12 CUDA_VISIBLE_DEVICES=<GPU 1 uuid> UNIT=big12-snap.service bigsnap.sh   (env: MILESTONES FRAC STOP_DELTA)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"; : "${RUN:?}"; : "${LAYERS:?}"; : "${UNIT:?}"
MILESTONES=${MILESTONES:-"50000 100000 200000 400000"}; FRAC=${FRAC:-0.1}; STOP_DELTA=${STOP_DELTA:-0.005}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
WT1=/games/ailab/lab/flybrain-lab/.worktrees/big1
BIG=$OUT/big
C2=/games/ailab/corpora/pkgs.bytes; DEV=5283527978; OLDK=5460445051
PK28=/vault/ailab/flybrain-mem/pkgs/feat_all28; PK30=/vault/ailab/flybrain-mem/pkgs/feat_all30
mkdir -p "$BIG"
exec >> "$BIG/${RUN}_snap.log" 2>&1
echo "snapshots of $RUN start $(date '+%F %T'): milestones $MILESTONES, cooldown $FRAC of the steps"
score() {  # score <step> <ckpt> <tag> <offset> <featdir> <windows file>
  local out="$BIG/k_${RUN}snap_s$1_$3.json"
  [ -f "$out" ] && return 0
  SCORE_CODE=$WT1 "$PY" "$S/score_tables.py" --corpus "$C2" --offset "$4" --n 1024 --T 2048 --windows "$6" --device cuda:0 --out-dir "$BIG" "k_${RUN}snap_s$1_$3=$2@$5" >> "$BIG/score_runs.log" 2>&1 || echo "$(date +%T) scoring snapshot $1 $3 failed"
}
prev=""
for m in $MILESTONES; do
  K=$(python3 -c "print(int(round($m * $FRAC)))")
  name=${RUN}_snap$m
  final=$WT1/results/ckpt/${name}_transformer.pt
  if [ ! -f "$final" ]; then
    until grep -q "saved ${RUN}_transformer.step$m.pt" "$BIG/$RUN.log" 2>/dev/null; do sleep 120; done
    ck=$WT1/results/ckpt/${RUN}_transformer.step$m.pt
    [ -f "$ck" ] || { echo "$(date +%T) the checkpoint at $m is gone (pruned?): skipping this milestone"; continue; }
    cp "$ck" "$WT1/results/ckpt/${name}_from.pt"                       # the pruner keeps per-10,000 checkpoints, but a copy costs nothing
    echo; echo "=== $name $(date +%T): cooldown of $K steps from step $m"
    NAME=$name LAYERS=$LAYERS UNIT=$UNIT STEPS=$((m + K)) COOLDOWN=$K CKPT_EVERY=$K RESUME_FROM=$WT1/results/ckpt/${name}_from.pt \
      bash "$S/bigtrain.sh"
    [ -f "$final" ] || { echo "$(date +%T) $name did not save its model: see $BIG/$name.log"; continue; }
    rm -f "$WT1/results/ckpt/${name}_from.pt"
  fi
  score "$m" "$final" dev "$DEV" "$PK28" "$BIG/clean_dev.txt"
  score "$m" "$final" oldk "$OLDK" "$PK28" "$BIG/clean_kernel.txt"
  score "$m" "$final" dev30 "$DEV" "$PK30" "$BIG/clean_dev.txt"
  score "$m" "$final" oldk30 "$OLDK" "$PK30" "$BIG/clean_kernel.txt"
  line=$("$PY" - "$BIG" "$RUN" "$m" "$prev" "$STOP_DELTA" <<'PY'
import json, sys
big, run, m, prev, delta = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], float(sys.argv[5])
r = lambda s, t: json.load(open(f"{big}/k_{run}snap_s{s}_{t}.json"))["bpb"]
v = {t: r(m, t) for t in ("dev", "oldk", "dev30", "oldk30")}
msg = f"snapshot {run} at {m:,}: clean dev {v['dev']:.4f} (2^30 {v['dev30']:.4f}), clean kernel {v['oldk']:.4f} (2^30 {v['oldk30']:.4f})"
if prev:
    gain = r(int(prev), "dev") - v["dev"]
    msg += f"; the doubling from {int(prev):,} bought {gain:+.4f} on the clean dev" + (" -- STOP RULE MET" if gain < delta else "")
print(msg)
PY
)
  echo "$(date +%T) $line"
  prev=$m
done
echo "snapshots of $RUN done $(date '+%F %T')"
