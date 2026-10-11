#!/usr/bin/env bash
# Scores the big trains' checkpoints as they appear (written every 2,000 steps; scored every SCORE_EVERY = 4,000, the extras every
# EXTRA_EVERY = 20,000: at every 2,000 the scorer cost the 12-layer run 11 to 18 % of its pace, measured 2026-10-11), on GPU 1 beside the 12-layer run (no guard), on the CLEAN
# windows only (score_tables.py --windows): every SCORE_EVERY-th checkpoint on the clean package dev windows through the training table
# (2^28: the bar, 613 windows, the file-level mask); every EXTRA_EVERY-th also on the clean old kernel windows (890, the window-level
# mask: new files of an indexed project) and both through the 2^30 table (the use-time number). Then prunes a run's checkpoints to the
# last three plus one per KEEP_EVERY (10,000), and writes results/tables/big/summary.json (per run, per step, per tag: cold and warm).
# Masks: results/tables/leak_windows2_2026-10-11.json -> big/clean_dev.txt, big/clean_kernel.txt. Stop: touch results/tables/big/STOP_score
set -u
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the UUID of GPU 1}"   # no apostrophe inside ${...:?...}: bash pairs it
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
WT1=/games/ailab/lab/flybrain-lab/.worktrees/big1
BIG=$OUT/big
C2=/games/ailab/corpora/pkgs.bytes; DEV=5283527978; OLDK=5460445051
PK28=/vault/ailab/flybrain-mem/pkgs/feat_all28; PK30=/vault/ailab/flybrain-mem/pkgs/feat_all30
SCORE_EVERY=${SCORE_EVERY:-4000}; EXTRA_EVERY=${EXTRA_EVERY:-20000}; KEEP_EVERY=${KEEP_EVERY:-10000}
mkdir -p "$BIG"
exec >> "$BIG/score.log" 2>&1
echo "scorer start $(date '+%F %T')"
"$PY" - "$OUT/leak_windows2_2026-10-11.json" "$BIG" <<'PY' || exit 1
import json, sys
m = json.load(open(sys.argv[1]))
open(f"{sys.argv[2]}/clean_dev.txt", "w").write("\n".join(map(str, m["dev_clean_windows"])) + "\n")
flag = set(m["kernel_flagged"])
open(f"{sys.argv[2]}/clean_kernel.txt", "w").write("\n".join(str(i) for i in range(1024) if i not in flag) + "\n")
print(f"clean windows: {len(m['dev_clean_windows'])} dev, {1024 - len(flag)} kernel")
PY
score() {  # score <name> <step> <ckpt> <tag> <offset> <featdir> <windows file>
  local out="$BIG/k_$1_s$2_$4.json"
  [ -f "$out" ] && return 0
  SCORE_CODE=$WT1 "$PY" "$S/score_tables.py" --corpus "$C2" --offset "$5" --n 1024 --T 2048 --windows "$7" --device cuda:0 --out-dir "$BIG" "k_$1_s$2_$4=$3@$6" >> "$BIG/score_runs.log" 2>&1 || echo "$(date +%T) scoring $1 step $2 $4 failed"
}
while [ ! -f "$BIG/STOP_score" ]; do
  for ck in $(ls -v "$WT1"/results/ckpt/big*_transformer.step*.pt 2>/dev/null); do
    base=$(basename "$ck" .pt); name=${base%%_transformer.step*}; step=${base##*.step}
    [ $((step % SCORE_EVERY)) = 0 ] || continue
    new=0; [ -f "$BIG/k_${name}_s${step}_dev.json" ] || new=1
    score "$name" "$step" "$ck" dev "$DEV" "$PK28" "$BIG/clean_dev.txt"
    if [ $((step % EXTRA_EVERY)) = 0 ]; then
      score "$name" "$step" "$ck" oldk "$OLDK" "$PK28" "$BIG/clean_kernel.txt"
      score "$name" "$step" "$ck" dev30 "$DEV" "$PK30" "$BIG/clean_dev.txt"
      score "$name" "$step" "$ck" oldk30 "$OLDK" "$PK30" "$BIG/clean_kernel.txt"
    fi
    [ "$new" = 1 ] && echo "$(date +%T) scored $name step $step"
  done
  # prune: per run, keep the last three checkpoints and every KEEP_EVERY-th; one that is due a score goes only once its dev score exists
  for name in $(ls "$WT1"/results/ckpt/big*_transformer.step*.pt 2>/dev/null | sed 's#.*/##; s#_transformer.step.*##' | sort -u); do
    files=$(ls -v "$WT1"/results/ckpt/${name}_transformer.step*.pt); n=$(echo "$files" | wc -l)
    echo "$files" | head -n $((n > 3 ? n - 3 : 0)) | while read -r f; do
      s=${f##*.step}; s=${s%.pt}
      if [ $((s % KEEP_EVERY)) != 0 ] && { [ $((s % SCORE_EVERY)) != 0 ] || [ -f "$BIG/k_${name}_s${s}_dev.json" ]; }; then rm -f "$f"; fi
    done
  done
  "$PY" - "$BIG" <<'PY' 2>>"$BIG/score.log"
import json, sys, glob, re
big = sys.argv[1]
rows = {}
for f in glob.glob(f"{big}/k_big*_s*_*.json"):
    mm = re.match(r".*/k_(big\w+?)_s(\d+)_(\w+)\.json$", f)
    if not mm:
        continue
    d = json.load(open(f))
    rows.setdefault(mm.group(1), {}).setdefault(int(mm.group(2)), {})[mm.group(3)] = {"cold": round(d["bpb"], 4), "warm": round(d["bpb_warm"], 4), "n": len(d["per_window_cold"])}
json.dump({name: [{"step": s, **r[s]} for s in sorted(r)] for name, r in rows.items()}, open(f"{big}/summary.json", "w"), indent=1)
PY
  sleep 60
done
echo "scorer stopped $(date '+%F %T')"
