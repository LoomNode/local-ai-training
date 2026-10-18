#!/usr/bin/env bash
# The pace and the memory of a bigger token reader on this card (Kyle, 2026-10-18: "i still want to train a local ai model ... a larger model";
# time is not the constraint): three shapes, 80 steps each after the compile, on the prose corpus's token stream and its 2^28 tables.
# The pace is the clock between the eval points at steps 40, 60 and 80 (the compile and the first evals excluded); the memory is the
# yardstick's peak_vram_gib. Not a result: a probe for sizing the always-on train.
# usage: CUDA_VISIBLE_DEVICES=<uuid> big2_cal.sh   (log results/tables/big2/cal.log; summary big2/cal.txt; done file big2/cal.done)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?}"
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
OUT=$SP/results/tables/big2; TOKD=/games/ailab/corpora/talk-tok
TRAIN_END_TOK=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["train_end_tok"]))' "$TOKD/layout.json")
TOK20="1,2,3,4,6,8,12,16,116,132,202,203,301,1001,1002,1003,1004,1005,1006,1007"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --fused-channels"
TOKFLAGS=(--tokens --vocab 32768 --corpus-file "$TOKD/tokens.u16" --train-bytes "$TRAIN_END_TOK" --val-bytes 700000 --val-offset "$TRAIN_END_TOK"
          --tok-len "$TOKD/tok_len.u8" --match-feat 2,4,8 --mem-feat "$TOKD/feat28" --mem-orders "$TOK20" --shared-table)
mkdir -p "$OUT"; cd "$WT"
exec >> "$OUT/cal.log" 2>&1
echo "big2 calibration start $(date '+%F %T') on $CUDA_VISIBLE_DEVICES"
cal() {  # cal <name> <layers> <d_model> <d_ff> <batch>
  local name=$1 L=$2 D=$3 F=$4 B=$5
  echo; echo "=== $name $(date +%T): $L x $D (ff $F), batch $B x 2048 tokens"
  "$PY" scripts/run_sparsela.py --run-name "$name" --task code --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0003 --decay 1.0 \
    --d-model "$D" --d-ff "$F" --n-layers "$L" --batch "$B" --seq-len 2048 --seed 0 --device cuda:0 --warmup 20 --cosine --wd 0.1 --chunk 256 --amp \
    --meter --val-fixed 32 --eval-every 20 --steps 80 $TF "${TOKFLAGS[@]}" 2>&1 | tee "results/$name.out" | grep --line-buffered -E "^fixed|yardstick|Error|error|OOM|out of memory" | cut -c1-220
  "$PY" - "$name" "$L" "$D" "$F" "$B" "results/$name.out" <<'PY' | tee -a "$OUT/cal.txt"
import ast, re, sys
name, L, D, F, B, path = sys.argv[1:7]
pts = [ast.literal_eval(m.group(1)) for m in re.finditer(r"^fixed transformer (\{.*\})", open(path).read(), re.M)]
vram = re.search(r"peak_vram_gib=([\d.]+)", open(path).read())
if len(pts) >= 3:
    a, b = pts[-3], pts[-1]
    pace = (b["train_seconds"] - a["train_seconds"]) / (b["step"] - a["step"])
    print(f"{name}: {L} x {D} (ff {F}), batch {B}: {pace:.3f} s/step = {int(B) * 2048 / pace:,.0f} tokens/s, peak VRAM {vram.group(1) if vram else '?'} GiB")
else:
    print(f"{name}: {L} x {D} (ff {F}), batch {B}: did not reach step 80 ({len(pts)} eval points; OOM or error, see cal.log)")
PY
}
cal b2cal_24x1024_b8 24 1024 4096 8
cal b2cal_24x1024_b4 24 1024 4096 4
cal b2cal_16x1024_b8 16 1024 4096 8
cal b2cal_12x768_b8 12 768 3072 8
touch "$OUT/cal.done"; echo "big2 calibration done $(date '+%F %T')"
