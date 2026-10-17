#!/usr/bin/env bash
# The prose sources measured on Kyle's held-out set (R119): for each source, once talk_tables.sh has its scoring corpus and tables, the
# hour reader (4 layers, the hour pair's byte arm: 25,562 steps, the 18 inputs, match 8/16/32) trained on that source alone with its own
# 2^28 table, then scored on the held-out windows (score_tables.py over the corpus's dev region = kyle.bytes). First, the baseline: model #1
# with the code corpus's t28 tables on the same text (sample_tables.py --score-file). Lines land in results/tables/talk/scores.txt.
# usage: CUDA_VISIBLE_DEVICES=<uuid> UNIT=<unit>.service talk_readers.sh <src> [<src>...]   (log results/tables/talk/readers_<unit>.log)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the card UUID}"; : "${UNIT:?}"
U=${UNIT%.service}
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok; WT1=/games/ailab/lab/flybrain-lab/.worktrees/big1
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python; CK=$SP/notes/speed-round/checks
SRCD=/vault/ailab/corpora-src/talk; SCD=$SRCD/score; OUT=$SP/results/tables/talk
KD=/games/ailab/corpora/kyle; KYLE=$KD/kyle.bytes; DLG=$KD/dialogue.bytes; KWIN=$KD/windows.txt
# the dev region of every scoring corpus is heldout.bytes = kyle.bytes (his turns) then dialogue.bytes (both sides); heldout.json has the regions
NWIN=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["kyle"]["windows"])')
DOFF=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["dialogue"]["offset"])')
DWIN=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["dialogue"]["windows"])')
DEVB=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["bytes"])')
STEPS=25562
LOOK="4,8,12,16,32,64,116,132,202,203,301"; ALL18="$LOOK,1001,1002,1003,1004,1005,1006,1007"
COMMON="--task code --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 4 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-fixed 128 --eval-every 2000"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --fused-channels"
mkdir -p "$OUT" "$WT/results"
exec >> "$OUT/readers_$U.log" 2>&1
echo "talk readers, $U start $(date '+%F %T'): $* (Kyle's turns: $NWIN windows; the dialogue: $DWIN windows at +$DOFF; dev $DEVB bytes)"
idle() {  # wait for this card to be free (the equal-steps arm or a scorer may hold it)
  while :; do
    local apps used
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
    used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "$apps" = 0 ] && [ "$used" -lt 2500 ] && return 0
    sleep 60
  done
}
cd "$WT"
for set in kyle dialogue; do  # the baseline: model #1 with the code corpus's t28 tables on both sets
  f=$KD/$set.bytes; o=$OUT/k_${set}_model1_t28.json
  if [ -f "$o" ]; then echo "baseline $set: exists"; continue; fi
  idle; echo "=== baseline $(date +%T): model #1 + t28 on $set"
  SCORE_CODE=$WT1 nice -n 19 "$PY" "$CK/sample_tables.py" --ckpt "$WT1/results/ckpt/big12_final_transformer.pt" --tables /games/ailab/corpora/pkgs-tables/t28 \
    --score-file "$f" --device cuda:0 --out "$o" || echo "baseline $set failed"
  [ -f "$o" ] && "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"model #1 + t28 on {sys.argv[2]} ({d['n']} windows): {d['bpb']:.4f} bits/byte (cold)\")" "$o" "$set" | tee -a "$OUT/scores.txt"
done
for s in "$@"; do
  name="talk_${s//-/_}_r"; corpus="$SCD/$s.bytes"; mem="$SCD/$s-tables"; lay="$SCD/$s/layout.json"
  until [ -f "$SCD/$s/tables.done" ]; do sleep 120; done
  train_end=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["train_end"]))' "$lay")
  if [ -f "results/$name.json" ] && [ -f "results/ckpt/${name}_transformer.pt" ]; then echo "skip $name (done)"; else
    idle; echo "$UNIT" > "results/claim_$name"
    echo; echo "=== $name $(date +%T): $STEPS steps on $s ($train_end train bytes)"
    "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON $TF --steps "$STEPS" --save --corpus-file "$corpus" --train-bytes "$train_end" \
      --val-bytes "$DEVB" --val-offset "$train_end" --match-feat 8,16,32 --mem-feat "$mem/feat_all28" --mem-orders "$ALL18" 2>&1 | tee "results/$name.out" | grep --line-buffered -v "^  "
    rm -f "results/claim_$name"
    [ -f "results/$name.json" ] || { echo "$name did not finish; its last lines:"; tail -n 20 "results/$name.out"; continue; }
  fi
  if [ -f "$OUT/k_talk_$s.json" ]; then echo "score $s (Kyle's turns): exists"; else
    idle
    SCORE_CODE=$WT "$PY" "$CK/score_tables.py" --corpus "$corpus" --offset "$train_end" --n "$NWIN" --T 2048 --windows "$KWIN" --device cuda:0 \
      --out-dir "$OUT" "k_talk_$s=results/ckpt/${name}_transformer.pt@$mem/feat_all28" || { echo "scoring $s failed"; continue; }
  fi
  if [ -f "$OUT/k_talk_${s}_dlg.json" ]; then echo "score $s (dialogue): exists"; else
    idle
    SCORE_CODE=$WT "$PY" "$CK/score_tables.py" --corpus "$corpus" --offset $(( train_end + DOFF )) --n "$DWIN" --T 2048 --device cuda:0 \
      --out-dir "$OUT" "k_talk_${s}_dlg=results/ckpt/${name}_transformer.pt@$mem/feat_all28" || { echo "scoring $s (dialogue) failed"; continue; }
  fi
  "$PY" - "$s" "$OUT/k_talk_$s.json" "$OUT/k_talk_${s}_dlg.json" "results/$name.json" "$train_end" <<'PY' | tee -a "$OUT/scores.txt"
import json, sys
s, kp, dp, rp, train_end = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
k = json.load(open(kp)); g = json.load(open(dp)); r = json.load(open(rp)); a = r["arms"][0]; last = a["curve_fixed"][-1]
print(f"{s}: Kyle's turns ({k['n']} windows) cold {k['bpb']:.4f} warm {k['bpb_warm']:.4f}; the dialogue ({g['n']} windows) cold {g['bpb']:.4f} "
      f"warm {g['bpb_warm']:.4f}; the hour reader on {train_end:,} train bytes, {r['steps']} steps (in-run val {last['bpb']:.4f}, clock {a['train_seconds']:.0f} s)")
PY
done
touch "$OUT/$U.done"; echo "$U done $(date '+%F %T')"
