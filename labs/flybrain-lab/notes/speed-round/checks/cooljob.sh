#!/usr/bin/env bash
# One cooling job of the GPU 0 queue (coolq_gpu0.sh). Cools a copy of <RUN>'s checkpoint at step M for K steps (bigtrain.sh with
# STEPS=M+K COOLDOWN=K: the rate falls linearly to a tenth over the K steps) with --seed SEED (a cooldown's seed is its batch order, so
# the same checkpoint cooled at two seeds shows the noise), checkpointing every 2,000 steps so the game guard costs at most ten minutes
# (the queue resumes the job from its newest checkpoint: RESUME_FROM=auto, with the milestone checkpoint copied in as the job's step-M
# checkpoint to start), then scores the cooled model on the clean windows (dev through 2^28, the bar; the kernel windows; both through
# 2^30) and drops the job's step checkpoints. Names: cool_<run>_<M>_k<K>_s<seed> (not big*, so bigscore.sh neither scores nor prunes
# them); scores results/tables/big/k_<name>_<tag>.json; the log of the cooldown big/<name>.log; the summary big/cool_summary.json.
# TAG=snap marks a doubling milestone of the stop rule: the model and scores are also linked under bigsnap.sh's names
# (<run>_snap<M>_transformer.pt, k_<run>snap_s<M>_<tag>.json: the summary's "<run>snap" series the board shows) and the stop-rule line
# goes to big/<run>_snap.log.
# usage: RUN=big12 LAYERS=12 M=200000 K=40000 SEED=0 TAG= UNIT=cool-gpu0.service CUDA_VISIBLE_DEVICES=<GPU 0 uuid> cooljob.sh
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the UUID of GPU 0}"; : "${RUN:?}"; : "${LAYERS:?}"; : "${M:?}"; : "${K:?}"; : "${SEED:?}"; : "${UNIT:?}"
TAG=${TAG:-}; STOP_DELTA=${STOP_DELTA:-0.005}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
WT1=/games/ailab/lab/flybrain-lab/.worktrees/big1
BIG=$OUT/big; CK=$WT1/results/ckpt
C2=/games/ailab/corpora/pkgs.bytes; DEV=5283527978; OLDK=5460445051
PK28=/vault/ailab/flybrain-mem/pkgs/feat_all28; PK30=/vault/ailab/flybrain-mem/pkgs/feat_all30
name=cool_${RUN}_${M}_k${K}_s${SEED}
final=$CK/${name}_transformer.pt
exec >> "$BIG/coolq.log" 2>&1
if [ ! -f "$final" ]; then
  if ! ls "$CK/${name}"_transformer.step*.pt >/dev/null 2>&1; then
    src=$CK/${RUN}_transformer.step$M.pt
    [ -f "$src" ] || { echo "$(date +%T) $name: no checkpoint of $RUN at step $M ($src)"; exit 1; }
    cp "$src" "$CK/${name}_transformer.step$M.pt"     # the start: RESUME_FROM=auto picks the newest checkpoint of the name
  fi
  echo "=== $name $(date '+%F %T'): cooldown of $K steps from step $M of $RUN, seed $SEED (its log: $BIG/$name.log)"
  NAME=$name LAYERS=$LAYERS UNIT=$UNIT STEPS=$((M + K)) COOLDOWN=$K CKPT_EVERY=2000 SEED=$SEED RESUME_FROM=auto bash "$S/bigtrain.sh"
  [ -f "$final" ] || { echo "$(date +%T) $name stopped before its model was saved (the queue resumes it)"; exit 0; }
fi
score() {  # score <tag> <offset> <featdir> <windows file>
  local out="$BIG/k_${name}_$1.json"
  [ -f "$out" ] && return 0
  SCORE_CODE=$WT1 "$PY" "$S/score_tables.py" --corpus "$C2" --offset "$2" --n 1024 --T 2048 --windows "$4" --device cuda:0 --out-dir "$BIG" "k_${name}_$1=$final@$3" >> "$BIG/score_runs.log" 2>&1 || echo "$(date +%T) scoring $name $1 failed"
}
score dev "$DEV" "$PK28" "$BIG/clean_dev.txt"
score oldk "$OLDK" "$PK28" "$BIG/clean_kernel.txt"
score dev30 "$DEV" "$PK30" "$BIG/clean_dev.txt"
score oldk30 "$OLDK" "$PK30" "$BIG/clean_kernel.txt"
for t in dev oldk dev30 oldk30; do [ -f "$BIG/k_${name}_$t.json" ] || exit 1; done
rm -f "${CK:?}/${name:?}"_transformer.step*.pt      # the job's step checkpoints (340 MB each); the cooled model stays
line=$("$PY" - "$BIG" "$name" <<'PY'
import json, sys
big, name = sys.argv[1:]
v = {t: json.load(open(f"{big}/k_{name}_{t}.json"))["bpb"] for t in ("dev", "oldk", "dev30", "oldk30")}
print(f"clean dev {v['dev']:.4f} (2^30 {v['dev30']:.4f}), clean kernel {v['oldk']:.4f} (2^30 {v['oldk30']:.4f})")
PY
)
echo "$(date '+%F %T') $name done: $line"
if [ "$TAG" = snap ]; then   # a doubling milestone: also under bigsnap.sh's names, and the stop rule against the previous milestone
  ln -f "$final" "$CK/${RUN}_snap${M}_transformer.pt"
  for t in dev oldk dev30 oldk30; do ln -f "$BIG/k_${name}_$t.json" "$BIG/k_${RUN}snap_s${M}_$t.json"; done
  msg=$("$PY" - "$BIG" "$RUN" "$M" "$STOP_DELTA" <<'PY'
import json, sys, glob, re
big, run, m, delta = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
r = lambda s, t: json.load(open(f"{big}/k_{run}snap_s{s}_{t}.json"))["bpb"]
prev = max([int(x) for f in glob.glob(f"{big}/k_{run}snap_s*_dev.json") for x in re.findall(r"_s(\d+)_dev", f) if int(x) < m], default=None)
msg = f"snapshot {run} at {m:,}: clean dev {r(m, 'dev'):.4f} (2^30 {r(m, 'dev30'):.4f}), clean kernel {r(m, 'oldk'):.4f} (2^30 {r(m, 'oldk30'):.4f})"
if prev:
    gain = r(prev, "dev") - r(m, "dev")
    msg += f"; the doubling from {prev:,} bought {gain:+.4f} on the clean dev" + (" -- STOP RULE MET" if gain < delta else "")
print(msg)
PY
)
  echo "$(date +%T) $msg" | tee -a "$BIG/${RUN}_snap.log"
fi
"$PY" - "$BIG" <<'PY'   # the sweep's summary: every finished job, by run, milestone, cooldown and seed
import json, sys, glob, re
big = sys.argv[1]; rows = {}
for f in glob.glob(f"{big}/k_cool_*_*.json"):
    mm = re.match(r".*/k_(cool_(\w+?)_(\d+)_k(\d+)_s(\d+))_(dev|oldk|dev30|oldk30)\.json$", f)
    if not mm:
        continue
    d = json.load(open(f))
    key = (mm.group(2), int(mm.group(3)), int(mm.group(4)), int(mm.group(5)))
    rows.setdefault(key, {"name": mm.group(1), "run": key[0], "milestone": key[1], "cooldown": key[2], "frac": round(key[2] / key[1], 3), "seed": key[3]})[mm.group(6)] = {"cold": round(d["bpb"], 4), "warm": round(d["bpb_warm"], 4), "n": len(d["per_window_cold"])}
json.dump([rows[k] for k in sorted(rows)], open(f"{big}/cool_summary.json", "w"), indent=1)
PY
