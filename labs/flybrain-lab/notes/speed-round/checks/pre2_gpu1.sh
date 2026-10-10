#!/usr/bin/env bash
# GPU 1's follow-up to pre_gpu1.sh (which cannot be edited while it runs). After it ends: one width arm of item 3 (the other is first
# on GPU 0's list, pre_gpu0.sh, so the two cards take different arms; the guard yielded GPU 0 to a game at 10:39 seventeen minutes
# into s12x4, so this card may end up with both), then item 1, the restart test, then the other width arm.
# Item 1: the hour arm (l4b: 4 layers, 18 inputs, 2^28 tables, 25,562 steps) under warmup-stable-decay (--wsd-cooldown 5112, the last
# 20 %) with a checkpoint every 5,000 steps, from the run worktree at the commit that holds --wsd-cooldown and --resume (speed-run5):
#   wsd_r   unbroken
#   wsdk_r  killed by a 30-minute timeout (about step 12,000, so the last checkpoint is 10,000), then resumed from that checkpoint
# Passes if the cool-down lands at l4b's 0.652 or better (the schedule) and the two agree within the per-window error (the restart;
# the batches are a function of (seed, step), TF32 and the compiler leave about 1e-3 on the card).
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-pre2-gpu1.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
WT5=/games/ailab/lab/flybrain-lab/.worktrees/speed-run5
cd "$WT"
exec > >(tee -a "$OUT/pre2_gpu1.log") 2>&1
echo "pre-train tests 2 gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD); the restart test from $WT5 at $(git -C "$WT5" rev-parse --short HEAD)"
while systemctl --user is-active --quiet tables-pre-gpu1.service; do sleep 60; done
sleep 10
FLAGS[s12x4_tf_mt_r]="$TFI --d-model 384 --n-layers 12 --d-ff 768"
FLAGS[s6x4_tf_mt_r]="$TFI --d-model 768 --n-layers 6 --d-ff 1536"
STEPS[s12x4_tf_mt_r]=52588; STEPS[s6x4_tf_mt_r]=38264
declare -A FEAT=([s12x4_tf_mt_r]=feat_all28 [s6x4_tf_mt_r]=feat_all28 [wsd_r]=feat_all28 [wsdk_r]=feat_all28)
score2() {  # score2 <name>: as scale_common's score, with the arm's own feature directory, from the current directory's results/ckpt
  name=$1; short=${name%_r}
  [ -f "$OUT/k_$short.json" ] && return 0
  "$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "k_$short=results/ckpt/${name}_transformer.pt@$MEM/${FEAT[$name]}" || echo "scoring $name failed"
}
preflight
cat "$MEM"/feat_all28/pred_n8.npy > /dev/null
sweep s6x4_tf_mt_r
echo; echo "=== scoring s6x4_tf_mt_r $(date +%T)"; score2 s6x4_tf_mt_r

echo; echo "=== item 1: the restart test $(date +%T), from $WT5"
cd "$WT5"
COMMON_WSD=${COMMON//--cosine/}   # the two schedules exclude each other
L4B="$TFI --n-layers 4 --wsd-cooldown 5112 --ckpt-every 5000"
run5() {  # run5 <name> [more flags]: the hour arm at 25,562 steps from speed-run5; the claim goes where the board reads them
  name=$1; shift
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo "$UNIT" > "$WT/results/claim_$name"
  echo; echo "=== $name $(date +%T): 25562 steps $*"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON_WSD $L4B --steps 25562 --save --val-fixed 128 --eval-every 2000 "$@" 2>&1 | grep --line-buffered -v "^  "
}
run5 wsd_r
if [ ! -f results/wsdk_r.json ]; then
  echo "$UNIT" > "$WT/results/claim_wsdk_r"
  echo; echo "=== wsdk_r $(date +%T): 25562 steps, to be killed after 1,800 s"
  timeout 1800 "$PY" scripts/run_sparsela.py --run-name wsdk_r $COMMON_WSD $L4B --steps 25562 --save --val-fixed 128 --eval-every 2000 2>&1 | grep --line-buffered -v "^  "
  echo "wsdk_r stopped at $(date +%T) (exit ${PIPESTATUS[0]}; 124 is the timeout)"
  last=$(ls -v results/ckpt/wsdk_r_transformer.step*.pt 2>/dev/null | tail -n 1)
  if [ -n "$last" ]; then run5 wsdk_r --resume "$last"; else echo "no checkpoint to resume wsdk_r from"; fi
fi
for n in wsd_r wsdk_r; do cp -n "results/$n.json" "$SP/results/" 2>/dev/null; done
echo; echo "=== scoring the restart test $(date +%T)"
score2 wsd_r; score2 wsdk_r
cd "$WT"
sweep s12x4_tf_mt_r
echo; echo "=== scoring s12x4_tf_mt_r $(date +%T)"; score2 s12x4_tf_mt_r
echo "pre-train tests 2 gpu1 done $(date '+%F %T')"
