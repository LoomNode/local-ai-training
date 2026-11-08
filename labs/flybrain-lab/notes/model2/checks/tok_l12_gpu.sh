#!/usr/bin/env bash
# Model #2, the second scale point (Kyle, 2026-10-16: "let's keep testing both as we scale for now. i'm just thinking our setup may change
# things"): tokens against bytes at the big train's depth, 12 layers, at four times the hour's budget on the first GB, each arm at the
# steps that cost four hours on a quiet card (tok_l12_cal.sh -> results/tables/tok/l12g_steps.env). Arms (12 layers, the hour pair's inputs):
#   l12g_tok_r       32k tokens, the 20 token inputs, the match 2/4/8, the shared channel, STEPS_TOK steps        (GPU 1)
#   l12g_bytes_r     bytes, the 18 inputs, the match 8/16/32, STEPS_BYTES steps                                   (GPU 0, through the slot)
#   l12g_bytes_sh_r  the byte arm with the shared channel (the channel's cost at this depth), STEPS_BYTES steps   (GPU 0, after)
#   l12g_tok_eq_r    the token arm at the byte arm's steps (equal steps, 1.77x the cost: quality only)             (GPU 1, after the verdict)
# Checkpoints every 2,000 steps and an automatic resume (bigtrain.sh's pattern), so a game on GPU 0 costs at most 2,000 steps; the step
# checkpoints are pruned once the arm's record exists. Each arm is scored on the 613 clean dev windows when it finishes; the verdict line
# (tokens vs bytes at 4x, 12 layers) goes into tok/l12g_verdict.txt and this card's log when both main scores exist.
# usage: CUDA_VISIBLE_DEVICES=<uuid> UNIT=<unit>.service tok_l12_gpu.sh <arm> [<arm>...]
#   waits for l12g_steps.env; log results/tables/tok/l12g_<unit>.log; done file results/tables/tok/<unit>.done once every listed arm is scored
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the card UUID}"; : "${UNIT:?}"
[ $# -ge 1 ] || { echo "usage: tok_l12_gpu.sh <arm> [<arm>...]"; exit 1; }
U=${UNIT%.service}
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
COMMON="--task code --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 12 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-fixed 128 --eval-every 2000"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --fused-channels"
BYTEFLAGS=(--corpus-file "$CORPUS" --train-bytes "$TRAIN_END" --val-bytes 2000000 --val-offset "$TRAIN_END" --match-feat 8,16,32 --mem-feat "$TABD/feat_all28" --mem-orders "$ALL18")
TOKFLAGS=(--tokens --vocab 32768 --corpus-file "$TOKD/tokens.u16" --train-bytes "$TRAIN_END_TOK" --val-bytes 700000 --val-offset "$TRAIN_END_TOK"
          --tok-len "$TOKD/tok_len.u8" --match-feat 2,4,8 --mem-feat "$TOKD/feat28" --mem-orders "$TOK20" --shared-table)
mkdir -p "$OUT" "$WT/results"
cd "$WT"
exec >> "$OUT/l12g_$U.log" 2>&1
echo "12 layers at 4x, $U start $(date '+%F %T'): arms $*"
until [ -f "$OUT/l12g_steps.env" ]; do sleep 60; done
. "$OUT/l12g_steps.env"
echo "steps: bytes $STEPS_BYTES, tokens $STEPS_TOK (paces $PB / $PT s/step, budget $BUDGET_S s)"
for f in "$TABD/feat_all28/features.json" "$TOKD/feat28/features.json" "$TOKD/tokens.u16"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
for i in $(seq 1 40); do  # up to 20 minutes for the calibration or a scorer to leave the card
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "$apps" = 0 ] && [ "$used" -lt 2500 ] && break
  sleep 30
done
[ "$apps" = 0 ] && [ "$used" -lt 2500 ] || { echo "pre-flight failed: $apps compute processes, $used MiB in use"; exit 1; }
run() {  # run <name> <steps> <flags...>: resumes from the arm's newest checkpoint that loads; prunes the step checkpoints when done
  local name=$1 steps=$2; shift 2
  [ -f "results/$name.json" ] && { echo "skip $name (done)"; return 0; }
  local resume="" c
  for c in $(ls -v results/ckpt/${name}_transformer.step*.pt 2>/dev/null | tac); do
    if "$PY" -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); sys.exit(0 if {'opt', 'state', 'step'} <= set(ck) else 1)" "$c" 2>/dev/null; then
      resume="--resume $c"; break
    fi
    echo "$(date +%T) checkpoint $c does not load (cut off mid-save?): set aside as $c.bad"; mv "$c" "$c.bad"
  done
  echo "$UNIT" > "results/claim_$name"
  echo; echo "=== $name $(date +%T): $steps steps (12 layers, ${resume:-fresh})"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON $TF --steps "$steps" --ckpt-every 2000 --save $resume "$@" 2>&1 | tee "results/$name.out" | grep --line-buffered -v "^  "
  rm -f "results/claim_$name"
  if [ -f "results/$name.json" ] && [ -f "results/ckpt/${name}_transformer.pt" ]; then
    rm -f results/ckpt/${name}_transformer.step*.pt results/ckpt/${name}_transformer.step*.pt.bad
  else
    echo "$name did not finish; its last lines:"; tail -n 20 "results/$name.out"
  fi
}
score() {  # score <name> <short> bytes|tok: the arm on the 613 clean dev windows
  [ -f "$OUT/k_$2.json" ] && return 0
  [ -f "results/ckpt/$1_transformer.pt" ] || { echo "no checkpoint for $1"; return 1; }
  if [ "$3" = tok ]; then
    SCORE_CODE=$WT "$PY" "$C2/score_tokens.py" --stream "$TOKD" --offset "$TRAIN_END" --n 1024 --T 2048 --windows "$WINDOWS" --warm-bytes 2048 --device cuda:0 \
      --out-dir "$OUT" "k_$2=results/ckpt/$1_transformer.pt@$TOKD/feat28" || echo "scoring $1 failed"
  else
    SCORE_CODE=$WT "$PY" "$CK/score_tables.py" --corpus "$CORPUS" --offset "$TRAIN_END" --n 1024 --T 2048 --windows "$WINDOWS" --device cuda:0 \
      --out-dir "$OUT" "k_$2=results/ckpt/$1_transformer.pt@$TABD/feat_all28" || echo "scoring $1 failed"
  fi
}
verdict() {
  "$PY" - "$OUT" "$WT/results" <<'PY'
import json, os, sys
d, r = sys.argv[1], sys.argv[2]
def load(p): return json.load(open(p)) if os.path.exists(p) else None
b, t, s, e = (load(f"{d}/k_l12g_{w}.json") for w in ("bytes", "tok", "bytes_sh", "tok_eq"))
rb, rt, re_ = (load(f"{r}/l12g_{w}_r.json") for w in ("bytes", "tok", "tok_eq"))
v = f"{d}/l12g_verdict.txt"; have = open(v).read() if os.path.exists(v) else ""
if b and t and rb and rt and "tokens vs bytes" not in have:
    line = (f"tokens vs bytes at 4x, 12 layers, clean dev (613 windows): bytes {b['bpb']:.4f} (warm {b['bpb_warm']:.4f}) at {rb['steps']} steps, "
            f"tokens {t['bpb']:.4f} (warm {t['bpb_warm']:.4f}, {t['bpt']:.3f} bits/token) at {rt['steps']} steps; "
            f"clocks {rb['arms'][0]['train_seconds']:.0f} s vs {rt['arms'][0]['train_seconds']:.0f} s")
    print(line); open(v, "a").write(line + "\n")
if b and s and "shared-channel control" not in have:
    line = (f"shared-channel control at 12 layers, clean dev: bytes {b['bpb']:.4f} (warm {b['bpb_warm']:.4f}), bytes+shared {s['bpb']:.4f} "
            f"(warm {s['bpb_warm']:.4f}), difference {s['bpb'] - b['bpb']:+.4f} cold / {s['bpb_warm'] - b['bpb_warm']:+.4f} warm")
    print(line); open(v, "a").write(line + "\n")
if b and e and rb and re_ and "tokens at equal steps" not in have:
    line = (f"tokens at equal steps, 12 layers, clean dev: bytes {b['bpb']:.4f} (warm {b['bpb_warm']:.4f}) at {rb['steps']} steps, "
            f"tokens {e['bpb']:.4f} (warm {e['bpb_warm']:.4f}, {e['bpt']:.3f} bits/token) at {re_['steps']} steps; "
            f"difference {e['bpb'] - b['bpb']:+.4f} cold / {e['bpb_warm'] - b['bpb_warm']:+.4f} warm; clocks {rb['arms'][0]['train_seconds']:.0f} s vs {re_['arms'][0]['train_seconds']:.0f} s")
    print(line); open(v, "a").write(line + "\n")
PY
}
for arm in "$@"; do
  case $arm in
    l12g_bytes_r)    run "$arm" "$STEPS_BYTES" "${BYTEFLAGS[@]}"; score "$arm" l12g_bytes bytes ;;
    l12g_bytes_sh_r) run "$arm" "$STEPS_BYTES" "${BYTEFLAGS[@]}" --shared-table; score "$arm" l12g_bytes_sh bytes ;;
    l12g_tok_r)      run "$arm" "$STEPS_TOK" "${TOKFLAGS[@]}"; score "$arm" l12g_tok tok ;;
    l12g_tok_eq_r)   run "$arm" "$STEPS_BYTES" "${TOKFLAGS[@]}"; score "$arm" l12g_tok_eq tok ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  verdict
done
all=1
for arm in "$@"; do short=${arm#l12g_}; short=${short%_r}; [ -f "$OUT/k_l12g_$short.json" ] || all=0; done
[ "$all" = 1 ] && touch "$OUT/$U.done"
echo "$U done $(date '+%F %T') (all scored: $all)"
