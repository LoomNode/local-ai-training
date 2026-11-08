#!/usr/bin/env bash
# The big train's end (Kyle, 2026-10-15: "Stop #1 at 800k. we can evaluate it and get #2 locked in"). Unit big-finish, no GPU of its own:
# 1. parks the GPU 0 filler for good (STOP_big8; big8-gpu0 stopped after its next checkpoint save: its question was answered on 10-14);
# 2. waits for the 12-layer run's checkpoint at step 800,000 (big12.log "saved big12_transformer.step800000.pt", the file settled),
#    then stops big12-gpu1: the constant-rate phase ends there; the steps after the save are discarded;
# 3. starts the final cooldown on GPU 1 as unit cool-final-gpu1 (cooljob.sh: cool_big12_800000_k160000_s0, 160,000 steps = 20 % of the
#    run, the sweep's finding that each doubling of the cooldown buys 0.008; about 19 hours; it resumes from its newest checkpoint after
#    a stop, bigtrain_restart.sh restarts it). The queue's own 800k job (10 %, GPU 0, the stop rule's snapshot) runs beside it, so the
#    doublings compare under the same cooldown fraction;
# 4. stops the scorer once it has scored step 800,000 with the extras (STOP_score), and the queue once its 800k snapshot is scored
#    (STOP_coolq: the queue stops whatever it runs and exits);
# 5. when the final cooldown's model is scored, links it as model #1 (results/ckpt/big12_final_transformer.pt in .worktrees/big1; the
#    scores as big/k_big12final_<tag>.json) and runs final_eval.sh on it (the held-out streams and completions through the indexes, t28, t30).
# Every step is skipped when its result exists, so the unit can be restarted. Log: results/tables/big/finish.log
# usage: systemd-run --user --unit big-finish /bin/bash big_finish.sh
set -u
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; BIG=$SP/results/tables/big; WT1=/games/ailab/lab/flybrain-lab/.worktrees/big1; CK=$WT1/results/ckpt
G1=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 1)
M=800000; KF=160000; FINAL=cool_big12_${M}_k${KF}_s0; SNAP=cool_big12_${M}_k80000_s0
exec >> "$BIG/finish.log" 2>&1
log() { echo "$(date '+%F %T') $*"; }
log "big-finish start: stop big12 at $M, final cooldown $KF steps on GPU 1, the queue's 10 % snapshot on GPU 0"
park_filler() {
  touch "$BIG/STOP_big8"
  if systemctl --user is-active --quiet big8-gpu0.service; then
    local n; n=$(grep -c "^saved big8_transformer" "$BIG/big8.log")
    until [ "$(grep -c "^saved big8_transformer" "$BIG/big8.log")" -gt "$n" ] || ! systemctl --user is-active --quiet big8-gpu0.service; do sleep 10; done
    sleep 3; systemctl --user stop big8-gpu0.service; log "big8-gpu0 (the filler) stopped after $(grep '^saved big8_transformer' "$BIG/big8.log" | tail -n 1)"
  fi
}
park_filler
# 2. the 800k checkpoint, then the run stops
if systemctl --user is-active --quiet big12-gpu1.service; then
  until grep -q "saved big12_transformer.step$M.pt" "$BIG/big12.log" && [ -f "$CK/big12_transformer.step$M.pt" ]; do sleep 60; park_filler; done
  until [ $(( $(date +%s) - $(stat -c %Y "$CK/big12_transformer.step$M.pt") )) -ge 30 ]; do sleep 5; done
  systemctl --user stop big12-gpu1.service
  log "big12-gpu1 stopped after its checkpoint at step $M ($(grep -c '^saved big12_transformer' "$BIG/big12.log") saves in its log)"
fi
# 3. the final cooldown on GPU 1
if [ ! -f "$CK/${FINAL}_transformer.pt" ] && ! systemctl --user is-active --quiet cool-final-gpu1.service; then
  sleep 30
  systemctl --user reset-failed cool-final-gpu1.service 2>/dev/null
  systemd-run --user --unit cool-final-gpu1 --setenv CUDA_VISIBLE_DEVICES="$G1" --setenv RUN=big12 --setenv LAYERS=12 --setenv M="$M" --setenv K="$KF" \
    --setenv SEED=0 --setenv TAG= --setenv UNIT=cool-final-gpu1.service /bin/bash "$S/cooljob.sh" 2>&1 | tail -n 1
  log "cool-final-gpu1 started: $FINAL (about 19 hours)"
fi
# 4. the scorer and the queue wind down; 5. the final model
scorer_done=0; queue_done=0
until [ -f "$BIG/k_${FINAL}_oldk30.json" ]; do
  sleep 60
  [ "$queue_done" = 1 ] || park_filler
  if [ "$scorer_done" = 0 ] && [ -f "$BIG/k_big12_s${M}_oldk30.json" ]; then touch "$BIG/STOP_score"; scorer_done=1; log "step $M scored by the scorer: STOP_score"; fi
  if [ "$queue_done" = 0 ] && [ -f "$BIG/k_${SNAP}_oldk30.json" ]; then
    touch "$BIG/STOP_coolq"; queue_done=1; log "the queue's 800k snapshot is scored: STOP_coolq ($(tail -n 1 "$BIG/big12_snap.log"))"
  fi
  if ! systemctl --user is-active --quiet cool-final-gpu1.service && [ ! -f "$CK/${FINAL}_transformer.pt" ]; then
    log "cool-final-gpu1 is not running and the final model is not saved: restarting it"
    systemctl --user reset-failed cool-final-gpu1.service 2>/dev/null
    systemd-run --user --unit cool-final-gpu1 --setenv CUDA_VISIBLE_DEVICES="$G1" --setenv RUN=big12 --setenv LAYERS=12 --setenv M="$M" --setenv K="$KF" \
      --setenv SEED=0 --setenv TAG= --setenv UNIT=cool-final-gpu1.service /bin/bash "$S/cooljob.sh" 2>&1 | tail -n 1
    sleep 600
  fi
done
ln -f "$CK/${FINAL}_transformer.pt" "$CK/big12_final_transformer.pt"
for t in dev oldk dev30 oldk30; do ln -f "$BIG/k_${FINAL}_$t.json" "$BIG/k_big12final_$t.json"; done
log "model #1 is $CK/big12_final_transformer.pt: $(grep "$FINAL done" "$BIG/coolq.log" | tail -n 1)"
CUDA_VISIBLE_DEVICES=$G1 bash "$S/final_eval.sh" "$CK/big12_final_transformer.pt" final
tail -n 2 "$SP/results/tables/use/final_eval.log"
log "big-finish done"
