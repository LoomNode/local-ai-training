#!/usr/bin/env bash
# The control for the hour verdict (spec risk note: "if the token arm loses, test a byte arm with the shared channel before blaming
# tokens"): h1g_bytes_sh_r = tok_hour_gpu.sh's byte arm with --shared-table (the table's and the match's predicted byte embedded with
# the model's own byte embedding through a learned per-order gate, no per-order tables), the same 25,562 steps, scored on the 613 clean
# dev windows -> results/tables/tok/k_h1g_bytes_sh.json. If it lands near h1g_bytes_r (0.8100), the channel is not the handicap.
# usage: CUDA_VISIBLE_DEVICES=<uuid> UNIT=<unit>.service tok_hour_ctrl.sh     (through gpu0_slot.sh on GPU 0, or directly on GPU 1)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the card UUID}"; : "${UNIT:?}"
STEPS_HOUR=${STEPS_HOUR:-25562}
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
CK=$SP/notes/speed-round/checks; OUT=$SP/results/tables/tok
CORPUS=/games/ailab/corpora/pkgs1g.bytes; LAYOUT=/games/ailab/corpora/pkgs1g/layout.json; TABD=/games/ailab/corpora/pkgs1g-tables
WINDOWS=$SP/results/tables/big/clean_dev.txt
TRAIN_END=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["train_end"]))' "$LAYOUT")
LOOK="4,8,12,16,32,64,116,132,202,203,301"; ALL18="$LOOK,1001,1002,1003,1004,1005,1006,1007"
COMMON="--task code --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 4 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-fixed 128 --eval-every 2000"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --fused-channels"
cd "$WT"
exec >> "$OUT/hour_ctrl.log" 2>&1
echo "shared-channel byte control, start $(date '+%F %T'): $STEPS_HOUR steps"
[ -f "$TABD/feat_all28/features.json" ] || { echo "missing $TABD/feat_all28/features.json"; exit 1; }
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$apps" = 0 ] && [ "$used" -lt 2500 ] || { echo "pre-flight failed: $apps compute processes, $used MiB in use"; exit 1; }
name=h1g_bytes_sh_r
if [ ! -f "results/$name.json" ]; then
  echo "$UNIT" > "results/claim_$name"
  echo; echo "=== $name $(date +%T): $STEPS_HOUR steps (4 layers, shared channel)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON $TF --steps "$STEPS_HOUR" --save --corpus-file "$CORPUS" --train-bytes "$TRAIN_END" \
    --val-bytes 2000000 --val-offset "$TRAIN_END" --match-feat 8,16,32 --mem-feat "$TABD/feat_all28" --mem-orders "$ALL18" --shared-table 2>&1 | grep --line-buffered -v "^  "
  rm -f "results/claim_$name"
fi
if [ ! -f "$OUT/k_h1g_bytes_sh.json" ] && [ -f "results/ckpt/${name}_transformer.pt" ]; then
  SCORE_CODE=$WT "$PY" "$CK/score_tables.py" --corpus "$CORPUS" --offset "$TRAIN_END" --n 1024 --T 2048 --windows "$WINDOWS" --device cuda:0 \
    --out-dir "$OUT" "k_h1g_bytes_sh=results/ckpt/${name}_transformer.pt@$TABD/feat_all28" || echo "scoring $name failed"
fi
"$PY" - "$OUT" <<'PY'
import json, sys
d = sys.argv[1]
try:
    b, s = (json.load(open(f"{d}/k_{w}.json")) for w in ("h1g_bytes", "h1g_bytes_sh"))
    print(f"shared-channel control, clean dev (613 windows): bytes {b['bpb']:.4f} (warm {b['bpb_warm']:.4f}), bytes+shared {s['bpb']:.4f} (warm {s['bpb_warm']:.4f}), "
          f"difference {s['bpb'] - b['bpb']:+.4f} cold / {s['bpb_warm'] - b['bpb_warm']:+.4f} warm")
except Exception as e:
    print(f"no verdict yet: {e}")
PY
touch "$OUT/hour_ctrl.done"
echo "shared-channel control done $(date '+%F %T')"
