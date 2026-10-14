#!/usr/bin/env bash
# Runs the Qwen side of the viability test (qwen_runs.sh) on GPU 0 as soon as the cooling sweep is done, without touching the queue's
# rules: waits for the sweep's last job to be scored, parks the queue's filler (touch big/STOP_big8, then stops big8-gpu0 right after
# its next checkpoint save, so nothing is lost), waits for the card to be idle for three minutes (Kyle's games), runs qwen_runs.sh as
# the unit qwen-eval under the game guard (probe_guard.sh stops it for a game; it resumes with the jobs left), and when all four
# results exist releases the filler (rm STOP_big8: the queue restarts big8 within minutes). No GPU of its own.
# usage: systemd-run --user --unit qwen-orch /bin/bash qwen_orchestrate.sh        log: results/tables/use/qwen_orch.log
set -u
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; BIG=$SP/results/tables/big; U=$SP/results/tables/use
G0=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)
LAST=${LAST:-k_cool_big12_200000_k80000_s0_oldk30.json}      # the sweep's last job's last score
mkdir -p "$U"
exec >> "$U/qwen_orch.log" 2>&1
echo "orchestrator start $(date '+%F %T'): waiting for $BIG/$LAST"
until [ -f "$BIG/$LAST" ]; do sleep 120; done
echo "$(date '+%F %T') the sweep is scored: parking the filler"
touch "$BIG/STOP_big8"
if systemctl --user is-active --quiet big8-gpu0.service; then
  n=$(grep -c "^saved big8_transformer" "$BIG/big8.log")
  until [ "$(grep -c "^saved big8_transformer" "$BIG/big8.log")" -gt "$n" ] || ! systemctl --user is-active --quiet big8-gpu0.service; do sleep 10; done
  sleep 3; systemctl --user stop big8-gpu0.service; echo "$(date '+%F %T') big8-gpu0 stopped after $(grep '^saved big8_transformer' "$BIG/big8.log" | tail -n 1)"
fi
all_done() { for f in qwen_vox_heldout qwen_lab_heldout qwen_vox_complete_gd qwen_lab_complete_py; do [ -f "$U/$f.json" ] || return 1; done; }
until all_done; do
  idle=0
  while [ "$idle" -lt 6 ]; do
    sleep 30
    n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$G0")
    used=$(nvidia-smi -i "$G0" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "$n" = 0 ] && [ "$used" -lt 2500 ]; then idle=$((idle + 1)); else idle=0; fi
  done
  echo "$(date '+%F %T') card idle: starting qwen-eval"
  systemctl --user reset-failed qwen-eval.service 2>/dev/null
  systemd-run --user --unit qwen-eval --setenv CUDA_VISIBLE_DEVICES="$G0" /bin/bash "$S/qwen_runs.sh" 2>&1 | tail -n 1
  sleep 5
  systemd-run --user --unit qwen-eval-guard /bin/bash "$S/probe_guard.sh" qwen-eval.service "$SP/results/tables/guard.log" 2>&1 | tail -n 1
  while systemctl --user is-active --quiet qwen-eval.service; do sleep 60; done
  echo "$(date '+%F %T') qwen-eval ended ($(tail -n 1 "$SP/results/tables/guard.log"))"
  sleep 60
done
rm -f "$BIG/STOP_big8"
echo "$(date '+%F %T') all four Qwen results in; the filler is released"
tail -n 5 "$U/qwen_runs.log"
