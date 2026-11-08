#!/usr/bin/env bash
# Starts gpu0_queue.sh on GPU 0 only inside 22:00-04:00 and only when the card lists no compute process and holds under
# 1,500 MiB; starts the probes worktree's gpu0_guard.sh for it (yields to a foreign compute process, stops at 05:00); and itself
# yields when anything that is not a compute process of ours uses over 2,500 MiB (a game is a graphics client).
set -u
GPU0=GPU-cdb4714c-560c-9a81-6420-f07eb59b5a5d
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-heads
LOG=$WT/results/speed_gpu0_guard.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
h=$(date +%H%M)
if [ "$h" -ge 0400 ] && [ "$h" -lt 2200 ]; then say "outside the start window: nothing started"; exit 0; fi
apps=$(nvidia-smi -i "$GPU0" --query-compute-apps=pid --format=csv,noheader | grep -c .)
mem=$(nvidia-smi -i "$GPU0" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$apps" != 0 ] || [ "$mem" -ge 1500 ]; then say "GPU 0 in use ($apps compute processes, $mem MiB): nothing started"; exit 0; fi
say "starting the two-hour runs on GPU 0"
systemd-run --user --unit=speed-gpu0 --setenv=CUDA_VISIBLE_DEVICES="$GPU0" /bin/bash "$WT/gpu0_queue.sh"
sleep 5
systemd-run --user --unit=speed-gpu0-guard /bin/bash /games/ailab/lab/flybrain-lab/.worktrees/probes/gpu0_guard.sh speed-gpu0.service "$LOG" "$WT"
while systemctl --user is-active --quiet speed-gpu0.service; do
  total=$(nvidia-smi -i "$GPU0" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  ours=$(nvidia-smi -i "$GPU0" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{s += $1} END {print s + 0}')
  if [ $((total - ours)) -gt 2500 ]; then
    say "yielding GPU 0: $((total - ours)) MiB used by something else"
    systemctl --user stop speed-gpu0.service; exit 0
  fi
  sleep 60
done
say "launcher done"
