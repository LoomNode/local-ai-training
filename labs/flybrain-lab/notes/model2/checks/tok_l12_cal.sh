#!/usr/bin/env bash
# Model #2, the second scale point (Kyle, 2026-10-16 evening: "let's keep testing both as we scale for now. i'm just thinking our setup
# may change things"): the calibration for tok_l12_gpu.sh. Times the 12-layer byte arm and the 12-layer token arm for 600 steps each on
# one quiet card (the calibrated-steps rule: the step-time ratio from one session, steps as the budget), the pace from the curve's clock
# between steps 200 and 600 (compile and warmup excluded), then a 40-step token smoke with a checkpoint at 20 and a resume from it (the
# GPU 0 arms resume after a game), and at the very end writes results/tables/tok/l12g_steps.env (STEPS_BYTES and STEPS_TOK = BUDGET_S /
# pace, BUDGET_S = 4 x the hour pair's byte clock of 3,992 s: four hours of card for each arm at its own pace) and l12g_cal.done.
# usage: CUDA_VISIBLE_DEVICES=<uuid> UNIT=<unit>.service tok_l12_cal.sh     (GPU 1, idle)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the card UUID}"; : "${UNIT:?}"
BUDGET_S=${BUDGET_S:-15968}
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
OUT=$SP/results/tables/tok
CORPUS=/games/ailab/corpora/pkgs1g.bytes; LAYOUT=/games/ailab/corpora/pkgs1g/layout.json
TOKD=/games/ailab/corpora/pkgs1g-tok; TABD=/games/ailab/corpora/pkgs1g-tables
TRAIN_END=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["train_end"]))' "$LAYOUT")
TRAIN_END_TOK=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["train_end_tok"]))' "$TOKD/layout.json")
LOOK="4,8,12,16,32,64,116,132,202,203,301"; ALL18="$LOOK,1001,1002,1003,1004,1005,1006,1007"
TOK20="1,2,3,4,6,8,12,16,116,132,202,203,301,1001,1002,1003,1004,1005,1006,1007"
COMMON="--task code --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 12 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-fixed 128 --eval-every 200"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --fused-channels"
BYTEFLAGS=(--corpus-file "$CORPUS" --train-bytes "$TRAIN_END" --val-bytes 2000000 --val-offset "$TRAIN_END" --match-feat 8,16,32 --mem-feat "$TABD/feat_all28" --mem-orders "$ALL18")
TOKFLAGS=(--tokens --vocab 32768 --corpus-file "$TOKD/tokens.u16" --train-bytes "$TRAIN_END_TOK" --val-bytes 700000 --val-offset "$TRAIN_END_TOK"
          --tok-len "$TOKD/tok_len.u8" --match-feat 2,4,8 --mem-feat "$TOKD/feat28" --mem-orders "$TOK20" --shared-table)
mkdir -p "$OUT" "$WT/results"
cd "$WT"
exec >> "$OUT/l12g_cal.log" 2>&1
echo "12-layer calibration, start $(date '+%F %T'): 600 steps an arm, budget $BUDGET_S s an arm"
for f in "$TABD/feat_all28/features.json" "$TOKD/feat28/features.json" "$TOKD/tokens.u16"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$apps" = 0 ] && [ "$used" -lt 2500 ] || { echo "pre-flight failed: $apps compute processes, $used MiB in use"; exit 1; }
run() {  # run <name> <flags...>: 600 steps, no checkpoint
  local name=$1; shift
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo; echo "=== $name $(date +%T): 600 steps (12 layers)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON $TF --steps 600 "$@" 2>&1 | tee "results/$name.out" | grep --line-buffered -v "^  "
  [ -f "results/$name.json" ] || { echo "$name did not finish; its last lines:"; tail -n 20 "results/$name.out"; }
}
run cal12_bytes "${BYTEFLAGS[@]}"
run cal12_tok "${TOKFLAGS[@]}"
pace() {  # pace <name>: seconds per step between the curve points at 200 and 600
  "$PY" - "results/$1.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["arms"][0]["curve_fixed"]
t = {p["step"]: p["train_seconds"] for p in c}
print((t[600] - t[200]) / 400)
PY
}
PB=$(pace cal12_bytes) && PT=$(pace cal12_tok) || { echo "calibration failed: no paces"; exit 1; }
SB=$("$PY" -c "print(round($BUDGET_S / $PB))"); ST=$("$PY" -c "print(round($BUDGET_S / $PT))")
echo "paces: bytes $PB s/step, tokens $PT s/step (ratio $("$PY" -c "print(round($PT / $PB, 3))")); steps at $BUDGET_S s: bytes $SB, tokens $ST"
echo; echo "=== smoke12_tok $(date +%T): 40 steps, checkpoint at 20, then a resume from it"
rm -f results/smoke12_tok.json results/ckpt/smoke12_tok_transformer.step*.pt
"$PY" scripts/run_sparsela.py --run-name smoke12_tok $COMMON $TF --steps 40 --ckpt-every 20 "${TOKFLAGS[@]}" 2>&1 | grep -v "^  " | grep "saved\|yardstick\|rror" | tail -n 4
rm -f results/smoke12_tok.json
"$PY" scripts/run_sparsela.py --run-name smoke12_tok $COMMON $TF --steps 40 --resume results/ckpt/smoke12_tok_transformer.step20.pt "${TOKFLAGS[@]}" 2>&1 | grep -v "^  " | grep "resumed\|yardstick\|rror" | tail -n 4
if [ -f results/smoke12_tok.json ]; then echo "resume smoke ok"; else echo "RESUME SMOKE FAILED"; fi
rm -f results/ckpt/smoke12_tok_transformer.step*.pt
printf 'PB=%s\nPT=%s\nBUDGET_S=%s\nSTEPS_BYTES=%s\nSTEPS_TOK=%s\n' "$PB" "$PT" "$BUDGET_S" "$SB" "$ST" > "$OUT/l12g_steps.env"
touch "$OUT/l12g_cal.done"
echo "calibration done $(date '+%F %T')"
