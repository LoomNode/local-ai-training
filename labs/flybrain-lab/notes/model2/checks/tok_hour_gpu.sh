#!/usr/bin/env bash
# Model #2's first number (plan docs/superpowers/plans/2026-10-16-model2-tokens.md, Task 6): tokens against bytes at the hour, on the
# package corpus's first GB (hour_corpus.sh: pkgs1g.bytes and its token stream, each with its own 2^28 features). Three arms of the
# 4-layer reader from the tok worktree (spike/tokens; the byte path is the big train's code with --vocab 256):
#   h1g_bytes_r   bytes, the 18 table inputs and the match 8/16/32, 25,562 steps (the speed round's 4-layer hour)
#   h1g_tok_r     32k tokens, the 20 token inputs (8 n-gram orders, 5 keys, 7 counters as values), the match 2/4/8, the shared-embedding
#                 channel, the same 25,562 steps (equal steps: the token arm sees ~3x the text and does ~1.3x the work per step)
#   h1g_tok_eq_r  the token arm at the steps that cost what the byte arm cost: 25,562 x (byte s/step / token s/step) from the two arms'
#                 logged clocks (the calibrated-steps rule: the ratio from the same session)
# Scored on the 613 clean dev windows (results/tables/big/clean_dev.txt; the same bytes as pkgs.bytes's dev region): the byte arm by
# score_tables.py, the token arms by score_tokens.py (bits over the windows' bytes). Verdict line at the end into tok/hour.log.
# usage: CUDA_VISIBLE_DEVICES=<uuid> UNIT=<unit>.service tok_hour_gpu.sh     (through gpu0_slot.sh on GPU 0, or directly on GPU 1)
#        an arm whose results/<name>.json exists is skipped; a guard stop reruns an arm from scratch (the claim files follow scale_common.sh)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the card's UUID}"; : "${UNIT:?}"
STEPS_HOUR=${STEPS_HOUR:-25562}
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
C2=$SP/notes/model2/checks; CK=$SP/notes/speed-round/checks
OUT=$SP/results/tables/tok
CORPUS=/games/ailab/corpora/pkgs1g.bytes; LAYOUT=/games/ailab/corpora/pkgs1g/layout.json
TOKD=/games/ailab/corpora/pkgs1g-tok; TABD=/games/ailab/corpora/pkgs1g-tables
WINDOWS=$SP/results/tables/big/clean_dev.txt
TRAIN_END=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["train_end"]))' "$LAYOUT")
TRAIN_END_TOK=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["train_end_tok"]))' "$TOKD/layout.json")
LOOK="4,8,12,16,32,64,116,132,202,203,301"; ALL18="$LOOK,1001,1002,1003,1004,1005,1006,1007"
TOK20="1,2,3,4,6,8,12,16,116,132,202,203,301,1001,1002,1003,1004,1005,1006,1007"
COMMON="--task code --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 4 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-fixed 128 --eval-every 2000"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --fused-channels"
mkdir -p "$OUT" "$WT/results"
cd "$WT"
exec >> "$OUT/hour.log" 2>&1
echo "tokens vs bytes at the hour, start $(date '+%F %T'): $STEPS_HOUR steps an arm; train_end $TRAIN_END bytes / $TRAIN_END_TOK tokens"
for f in "$TABD/feat_all28/features.json" "$TOKD/feat28/features.json" "$TOKD/tokens.u16"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$apps" = 0 ] && [ "$used" -lt 2500 ] || { echo "pre-flight failed: $apps compute processes, $used MiB in use"; exit 1; }
run() {  # run <name> <steps> <flags...>
  local name=$1 steps=$2; shift 2
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo "$UNIT" > "results/claim_$name"
  echo; echo "=== $name $(date +%T): $steps steps (4 layers)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON $TF --steps "$steps" --save "$@" 2>&1 | grep --line-buffered -v "^  "
  rm -f "results/claim_$name"
}
run h1g_bytes_r "$STEPS_HOUR" --corpus-file "$CORPUS" --train-bytes "$TRAIN_END" --val-bytes 2000000 --val-offset "$TRAIN_END" \
  --match-feat 8,16,32 --mem-feat "$TABD/feat_all28" --mem-orders "$ALL18"
TOKFLAGS=(--tokens --vocab 32768 --corpus-file "$TOKD/tokens.u16" --train-bytes "$TRAIN_END_TOK" --val-bytes 700000 --val-offset "$TRAIN_END_TOK"
          --tok-len "$TOKD/tok_len.u8" --match-feat 2,4,8 --mem-feat "$TOKD/feat28" --mem-orders "$TOK20" --shared-table)
run h1g_tok_r "$STEPS_HOUR" "${TOKFLAGS[@]}"
pace() { "$PY" -c 'import json, sys; a = json.load(open(sys.argv[1]))["arms"][0]; print(a["train_seconds"] / int(sys.argv[2]))' "results/$1.json" "$2" 2>/dev/null; }
if [ -f results/h1g_bytes_r.json ] && [ -f results/h1g_tok_r.json ]; then
  PB=$(pace h1g_bytes_r "$STEPS_HOUR"); PT=$(pace h1g_tok_r "$STEPS_HOUR")
  STEPS_EQ=$("$PY" -c "print(max(1000, round($STEPS_HOUR * $PB / $PT)))")
  echo "paces: bytes $PB s/step, tokens $PT s/step (ratio $("$PY" -c "print(round($PT / $PB, 3))")); equal-compute token arm: $STEPS_EQ steps"
  run h1g_tok_eq_r "$STEPS_EQ" "${TOKFLAGS[@]}"
fi
score_bytes() {  # the byte arm on the clean dev windows of pkgs1g.bytes
  [ -f "$OUT/k_h1g_bytes.json" ] && return 0
  [ -f results/ckpt/h1g_bytes_r_transformer.pt ] || { echo "no checkpoint for h1g_bytes_r"; return 0; }
  SCORE_CODE=$WT "$PY" "$CK/score_tables.py" --corpus "$CORPUS" --offset "$TRAIN_END" --n 1024 --T 2048 --windows "$WINDOWS" --device cuda:0 \
    --out-dir "$OUT" "k_h1g_bytes=results/ckpt/h1g_bytes_r_transformer.pt@$TABD/feat_all28" || echo "scoring h1g_bytes_r failed"
}
score_tok() {  # score_tok <name> <short>: a token arm on the same byte windows
  [ -f "$OUT/k_$2.json" ] && return 0
  [ -f "results/ckpt/$1_transformer.pt" ] || { echo "no checkpoint for $1"; return 0; }
  SCORE_CODE=$WT "$PY" "$C2/score_tokens.py" --stream "$TOKD" --offset "$TRAIN_END" --n 1024 --T 2048 --windows "$WINDOWS" --warm-bytes 2048 --device cuda:0 \
    --out-dir "$OUT" "k_$2=results/ckpt/$1_transformer.pt@$TOKD/feat28" || echo "scoring $1 failed"
}
score_bytes; score_tok h1g_tok_r h1g_tok; score_tok h1g_tok_eq_r h1g_tok_eq
"$PY" - "$OUT" "$WT/results" "$STEPS_HOUR" <<'PY'
import json, sys
d, r, S = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    b, t = (json.load(open(f"{d}/k_{w}.json")) for w in ("h1g_bytes", "h1g_tok"))
    eq = json.load(open(f"{d}/k_h1g_tok_eq.json")) if __import__("os").path.exists(f"{d}/k_h1g_tok_eq.json") else None
    secs = {w: json.load(open(f"{r}/{w}_r.json"))["arms"][0]["train_seconds"] for w in ("h1g_bytes", "h1g_tok")}
    eqs = json.load(open(f"{r}/h1g_tok_eq_r.json"))["arms"][0] if eq else None
    print(f"tokens vs bytes, clean dev (613 windows): bytes {b['bpb']:.4f} (warm {b['bpb_warm']:.4f}), tokens {t['bpb']:.4f} (warm {t['bpb_warm']:.4f}, "
          f"{t['bpt']:.3f} bits/token) at equal steps ({S}; clocks {secs['h1g_bytes']:.0f} s vs {secs['h1g_tok']:.0f} s, ratio {secs['h1g_tok'] / secs['h1g_bytes']:.3f})"
          + (f"; tokens {eq['bpb']:.4f} (warm {eq['bpb_warm']:.4f}) at equal compute ({json.load(open(f'{r}/h1g_tok_eq_r.json'))['steps']} steps, {eqs['train_seconds']:.0f} s)" if eq else ""))
except Exception as e:
    print(f"no verdict yet: {e}")
PY
touch "$OUT/hour.done"
echo "tokens vs bytes done $(date '+%F %T')"
