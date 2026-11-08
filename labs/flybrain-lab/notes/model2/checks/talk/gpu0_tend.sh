#!/usr/bin/env bash
# Tends a resumable job on GPU 0, Kyle's card (2026-10-20: "gpu 0 is idle most of the time ... with the guard i can just interrupt if i want
# to game"): stops the job's unit the moment a compute process that is not the job's appears on GPU 0 (a game), and relaunches the unit
# once the card has been free of foreign processes for two minutes, until the job's done file exists. No time window. Runs as its own
# unit. usage: gpu0_tend.sh <job unit> <job script> <done file> <log>
set -u
UNIT=$1; SCRIPT=$2; DONE=$3; LOG=$4
GPU0=GPU-cdb4714c-560c-9a81-6420-f07eb59b5a5d
pids_of() { local cg; cg=$(systemctl --user show -p ControlGroup --value "$1" 2>/dev/null); [ -n "$cg" ] && cat "/sys/fs/cgroup$cg/cgroup.procs" 2>/dev/null; }
foreign() {  # prints the first compute pid on GPU 0 that is not the job's, if any
  local ours pid; ours=" $(pids_of "$UNIT" | tr '\n' ' ') "
  for pid in $(nvidia-smi -i "$GPU0" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do case "$ours" in *" $pid "*) ;; *) echo "$pid"; return 0 ;; esac; done
  return 1
}
free_for=0
echo "$(date '+%F %T') tending $UNIT on GPU 0 until $DONE" >> "$LOG"
while [ ! -f "$DONE" ]; do
  if systemctl --user is-active --quiet "$UNIT"; then
    if pid=$(foreign); then echo "$(date '+%F %T') yielding GPU 0 to pid $pid ($(ps -o comm= -p "$pid" 2>/dev/null))" >> "$LOG"; systemctl --user stop "$UNIT"; free_for=0; fi
  else
    if foreign >/dev/null; then free_for=0; else free_for=$((free_for + 20)); fi
    if [ "$free_for" -ge 120 ]; then echo "$(date '+%F %T') GPU 0 free: starting $UNIT" >> "$LOG"; systemctl --user reset-failed "$UNIT" 2>/dev/null; systemd-run --user --unit "$UNIT" --setenv CUDA_VISIBLE_DEVICES=$GPU0 "$SCRIPT" >/dev/null 2>&1 || echo "$(date '+%F %T') could not start $UNIT" >> "$LOG"; free_for=0; sleep 30; fi
  fi
  sleep 20
done
echo "$(date '+%F %T') $DONE exists: tending over" >> "$LOG"
