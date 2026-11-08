#!/usr/bin/env bash
# Stops the tables-probe unit on GPU 0 when a compute process that is not ours appears, or when something that is not a compute
# process (a game is a graphics client) holds more than 2,500 MiB of the card. No time window: Kyle said "use gpu when available".
# usage: probe_guard.sh <unit> <log>
set -u
UNIT=$1; LOG=$2
GPU0=GPU-cdb4714c-560c-9a81-6420-f07eb59b5a5d
pids_of() { cg=$(systemctl --user show -p ControlGroup --value "$1" 2>/dev/null); [ -n "$cg" ] && cat "/sys/fs/cgroup$cg/cgroup.procs" 2>/dev/null; }
while systemctl --user is-active --quiet "$UNIT"; do
  ours=" $(pids_of "$UNIT" | tr '\n' ' ') "
  for pid in $(nvidia-smi -i "$GPU0" --query-compute-apps=pid --format=csv,noheader); do
    case "$ours" in *" $pid "*) continue ;; esac
    state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null)
    if [ -z "$state" ] || [ "$state" = Z ]; then continue; fi    # gone or a zombie: a scorer of ours that just exited and nvidia-smi still lists (the cgroup no longer does); it yielded to three of those (06:13, 08:05, 23:38 on 2026-10-10)
    echo "$(date +%T) yielding GPU 0 to pid $pid ($(ps -o comm= -p "$pid"))" >> "$LOG"; systemctl --user stop "$UNIT"; exit 0
  done
  total=$(nvidia-smi -i "$GPU0" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  mine=$(nvidia-smi -i "$GPU0" --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{s += $1} END {print s + 0}')
  if [ $((total - mine)) -gt 2500 ]; then echo "$(date +%T) yielding GPU 0: $((total - mine)) MiB used by something else" >> "$LOG"; systemctl --user stop "$UNIT"; exit 0; fi
  sleep 20
done
echo "$(date +%T) probe queue ended on its own" >> "$LOG"
