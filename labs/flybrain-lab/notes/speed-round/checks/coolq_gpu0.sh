#!/usr/bin/env bash
# The GPU 0 cooling queue (Kyle, 2026-10-13: "let's sweep the cooling period. tbh i feel like it could vary seed to seed"; the 8-layer
# question is answered, so the card cools the big run instead: the doubling milestones of the stop rule, a sweep of the cooldown length
# and seed from one checkpoint, and the 8-layer run as the filler when nothing is runnable). Replaces bigsuper_gpu0.sh.
# The queue is results/tables/big/coolq.txt, one job per line, re-read every round (edit it any time; # comments and blank lines skipped):
#   <run> <milestone step M> <cooldown steps K> <seed> [snap]
# A job (cooljob.sh, unit cool-gpu0 under the game guard probe_guard.sh) cools a copy of <run>'s checkpoint at M for K steps, scores it,
# and records it; 'snap' marks a milestone of the stop rule (it also lands under bigsnap.sh's names, so the board and the stop line see
# it). The queue is a priority list: a job is runnable once its milestone checkpoint is saved and until it is done, and each round the
# first runnable job runs; a job that becomes runnable ahead of the running one (the milestone landing during a sweep job) preempts it:
# its unit is stopped (at most 2,000 steps, ten minutes, lost) and it resumes later. Removing the running job's line stops it too. With
# no runnable job the filler runs (FILLER=<name>:<layers>, default big8:8, as unit <name>-gpu0 with its own guard; FILLER=none or
# touch big/STOP_<name> keeps it off), preempted by any job. The card must be idle for three minutes before anything starts (Kyle's games).
# Log: big/coolq.log (the jobs' own lines too). Stop the queue: touch results/tables/big/STOP_coolq (the running unit is stopped).
# usage: CUDA_VISIBLE_DEVICES=<GPU 0 uuid> coolq_gpu0.sh
set -u
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the UUID of GPU 0}"   # no apostrophe inside ${...:?...}: bash pairs it
FILLER=${FILLER:-big8:8}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
WT1=/games/ailab/lab/flybrain-lab/.worktrees/big1
BIG=$OUT/big; CK=$WT1/results/ckpt; Q=$BIG/coolq.txt; STOP=$BIG/STOP_coolq
mkdir -p "$BIG"
exec >> "$BIG/coolq.log" 2>&1
echo "cooling queue start $(date '+%F %T') (queue $Q, filler $FILLER)"
JOBS=()
read_queue() {
  JOBS=()
  local run m k seed tag
  while read -r run m k seed tag; do
    case "$run" in ''|'#'*) continue ;; esac
    [ -n "$seed" ] || { echo "$(date +%T) bad queue line: '$run $m $k $seed'"; continue; }
    JOBS+=("$run $m $k $seed ${tag:-}")
  done < "$Q"
}
job_name() { set -- $1; echo "cool_$1_$2_k$3_s$4"; }
job_runnable() {  # job_runnable "<line>": done jobs and jobs whose milestone is not saved yet are not
  local run m k seed tag name; read -r run m k seed tag <<< "$1"; name=$(job_name "$1")
  [ -f "$BIG/k_${name}_oldk30.json" ] && [ -f "$BIG/k_${name}_dev.json" ] && [ -f "$BIG/k_${name}_oldk.json" ] && [ -f "$BIG/k_${name}_dev30.json" ] && return 1
  [ -f "$CK/${name}_transformer.pt" ] && return 0
  ls "$CK/${name}"_transformer.step*.pt >/dev/null 2>&1 && return 0
  grep -q "saved ${run}_transformer.step$m.pt" "$BIG/$run.log" 2>/dev/null && [ -f "$CK/${run}_transformer.step$m.pt" ]
}
first_runnable() { local i; for i in "${!JOBS[@]}"; do job_runnable "${JOBS[$i]}" && { echo "$i"; return; }; done; }
index_of() { local i; for i in "${!JOBS[@]}"; do [ "${JOBS[$i]}" = "$1" ] && { echo "$i"; return; }; done; echo 999; }
prune() { ls -v "${CK:?}/${1:?}"_transformer.step*.pt 2>/dev/null | head -n -2 | xargs -r rm -f; }   # a running job keeps its newest two
while [ ! -f "$STOP" ]; do
  idle=0
  while [ "$idle" -lt 6 ] && [ ! -f "$STOP" ]; do
    sleep 30
    n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
    used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "$n" = 0 ] && [ "$used" -lt 2500 ]; then idle=$((idle + 1)); else idle=0; fi
  done
  [ -f "$STOP" ] && break
  read_queue; i=$(first_runnable)
  if [ -n "$i" ]; then
    cur=${JOBS[$i]}; read -r run m k seed tag <<< "$cur"; name=$(job_name "$cur"); unit=cool-gpu0.service
    layers=12; [ "$run" = big8 ] && layers=8
    echo "$(date '+%F %T') job $i ($cur): starting $unit"
    systemctl --user reset-failed "$unit" 2>/dev/null
    systemd-run --user --unit "${unit%.service}" --setenv CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" --setenv RUN="$run" --setenv LAYERS="$layers" \
      --setenv M="$m" --setenv K="$k" --setenv SEED="$seed" --setenv TAG="$tag" --setenv UNIT="$unit" /bin/bash "$S/cooljob.sh" 2>&1 | tail -n 1
  elif [ "$FILLER" != none ] && [ ! -f "$BIG/STOP_${FILLER%%:*}" ]; then
    cur=filler; name=${FILLER%%:*}; unit=$name-gpu0.service
    echo "$(date '+%F %T') nothing runnable: starting the filler $unit"
    systemctl --user reset-failed "$unit" 2>/dev/null
    systemd-run --user --unit "${unit%.service}" --setenv CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" --setenv NAME="$name" --setenv LAYERS="${FILLER##*:}" --setenv UNIT="$unit" /bin/bash "$S/bigtrain.sh" 2>&1 | tail -n 1
  else
    sleep 300; continue
  fi
  sleep 5
  systemd-run --user --unit "${unit%.service}-guard" /bin/bash "$S/probe_guard.sh" "$unit" "$OUT/guard.log" 2>&1 | tail -n 1
  while systemctl --user is-active --quiet "$unit"; do
    sleep 60
    [ "$cur" = filler ] || prune "$name"
    read_queue; j=$(first_runnable)
    if [ "$cur" = filler ]; then pos=999; else pos=$(index_of "$cur"); fi
    why=""
    if [ -f "$STOP" ]; then why="stop file"
    elif [ "$cur" != filler ] && [ "$pos" = 999 ]; then why="its line left the queue"
    elif [ -n "$j" ] && [ "$j" -lt "$pos" ]; then why="job $j (${JOBS[$j]}) is runnable ahead of it"
    fi
    if [ -n "$why" ]; then echo "$(date '+%F %T') stopping $unit: $why"; systemctl --user stop "$unit"; break; fi
  done
  echo "$(date '+%F %T') $unit ended ($(tail -n 1 "$OUT/guard.log"))"
  sleep 60
done
echo "cooling queue stopped $(date '+%F %T') (stop file)"
