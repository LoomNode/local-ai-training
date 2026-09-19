#!/usr/bin/env bash
# Run one envelope-lab trial; log to /vault runs dir.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Default solo = GPU1. dual is an explicit turbo toggle (both cards).
export ENVELOPE_MODE="${ENVELOPE_MODE:-solo}"
export ENVELOPE_AI_GPU="${ENVELOPE_AI_GPU:-1}"
export ENVELOPE_TRAIN_GPU="${ENVELOPE_TRAIN_GPU:-0}"
export ENVELOPE_CACHE="${ENVELOPE_CACHE:-/vault/ailab/envelope-lab/cache}"
export ENVELOPE_RUNS="${ENVELOPE_RUNS:-/vault/ailab/envelope-lab/runs}"
export ENVELOPE_TIME_BUDGET="${ENVELOPE_TIME_BUDGET:-300}"

mkdir -p "$ENVELOPE_CACHE" "$ENVELOPE_RUNS"

if [[ "$ENVELOPE_MODE" == "dual" ]]; then
  TRIAL_GPU="$ENVELOPE_TRAIN_GPU"
else
  TRIAL_GPU="$ENVELOPE_AI_GPU"
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$ENVELOPE_RUNS/run-${STAMP}.log"
ln -sfn "$LOG" "$ENVELOPE_RUNS/run.log"

echo "mode=$ENVELOPE_MODE trial_gpu=$TRIAL_GPU ai_gpu=$ENVELOPE_AI_GPU second_gpu=$ENVELOPE_TRAIN_GPU time_budget=${ENVELOPE_TIME_BUDGET}s"
echo "log=$LOG"
if [[ "$ENVELOPE_MODE" == "solo" ]]; then
  echo "note: solo (default) = GPU1 only — unload KAT if VRAM tight; toggle: ENVELOPE_MODE=dual"
else
  echo "note: dual/turbo = both GPUs (train on second_gpu, AI/KAT on ai_gpu)"
fi

# Unbuffered so tail -f shows steps live (not one dump at the end)
export PYTHONUNBUFFERED=1

# Prefer uv if available
if command -v uv >/dev/null 2>&1; then
  uv run python -u train.py >"$LOG" 2>&1
else
  python3 -u train.py >"$LOG" 2>&1
fi

echo "---- metrics ----"
grep -E '^(val_bpb|peak_vram_mb|decode_tps|envelope|num_params_M|training_seconds|num_steps):' "$LOG" || {
  echo "WARN: no metrics found; see $LOG" >&2
  tail -n 40 "$LOG" >&2 || true
  exit 1
}
