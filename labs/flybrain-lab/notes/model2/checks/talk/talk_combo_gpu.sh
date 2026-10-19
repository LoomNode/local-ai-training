#!/usr/bin/env bash
# The three use-time levers combined (Kyle, 2026-10-19: "Go for it. Sounds promising"): the 4 KB-window reader th_w4k_r (context, −0.109 on
# his turns with 3 KB in view), fine-tuned with replay on his world (weights, −0.037) and read through the corpus tables with his files
# indexed in (memory, −0.049). One arm, th_combo_r = th_w4k_r + 3,000 steps on world-mix at lr 1e-4; then every model on the same bytes:
# the last 1,024 bytes of every 4,096-byte window of Kyle's turns (170) and of every eighth of the dialogue's (128), in heldout.bytes,
# whose inputs are computed through the saved tables alone (heldout-feat-plain) and through the book (heldout-feat-book):
#   bytes2k  th_bytes_r on the second 2 KB of each window (1 KB of context: the hour reader as it is)
#   w4k      th_w4k_r, plain tables           w4k+book    th_w4k_r through the book
#   combo    th_combo_r, plain tables         combo+book  th_combo_r through the book (all three levers)
# usage: CUDA_VISIBLE_DEVICES=<uuid> UNIT=<unit>.service talk_combo_gpu.sh   (log results/tables/talk/combo.log; lines combo.txt + leads.txt; done file combo.done)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?}"; : "${UNIT:?}"
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python; C2=$SP/notes/model2/checks; CK=$SP/notes/speed-round/checks
OUT=$SP/results/tables/talk; KD=/games/ailab/corpora/kyle; HELD=$KD/heldout.bytes; KDUMP=$KD/dumps
CORPUS=/games/ailab/corpora/talk.bytes; LAYOUT=/games/ailab/corpora/talk/layout.json; TABD=/games/ailab/corpora/talk-tables
LOOK="4,8,12,16,32,64,116,132,202,203,301"; ALL18="$LOOK,1001,1002,1003,1004,1005,1006,1007"
COMMON="--task code --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0001 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 4 --batch 4 --seq-len 4096 --seed 0 --device cuda:0 --warmup 100 --cosine --wd 0.1 --chunk 256 --amp --meter --val-fixed 64 --eval-every 1000"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --fused-channels"
mkdir -p "$OUT" "$WT/results"; ( umask 077; mkdir -p "$KDUMP" )
cd "$WT"; exec >> "$OUT/combo.log" 2>&1
echo "the three levers combined: $UNIT start $(date '+%F %T')"
lay() { "$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))[sys.argv[2]]))' "$1" "$2"; }
TRAIN_END=$(lay "$LAYOUT" train_end); DEV_OFF=$(lay "$LAYOUT" dev_offset); DEVWIN=$(lay "$LAYOUT" dev_windows); TAIL=$(lay "$LAYOUT" tail_offset)
NWIN=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["kyle"]["windows"])')
DOFF=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["dialogue"]["offset"])')
DWIN=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["dialogue"]["windows"])')
idle() { while :; do local apps used; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$CUDA_VISIBLE_DEVICES"); used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '); [ "$apps" = 0 ] && [ -n "$used" ] && [ "$used" -lt 2500 ] && return 0; sleep 60; done; }
# 1. the held-out file's inputs through the tables alone and through the book (CPU, a minute each)
for pair in "plain:$TABD/t28" "book:$KD/index-files"; do
  lab=${pair%%:*}; tab=${pair#*:}; F=$KD/heldout-feat-$lab
  [ -f "$F/features.json" ] && continue
  echo "== inputs of heldout.bytes through $tab -> $F $(date +%T)"
  ( umask 077; CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 0-5,8-13 "$PY" "$C2/world_features.py" --corpus "$HELD" --tables "$tab" --out "$F" | tail -n 1 )
done
# 2. the combined arm: th_w4k_r fine-tuned with replay on the world (4 KB window, batch 4)
MIX=$KD/world-mix.bytes; MN=$(stat -c %s "$MIX")
if ! { [ -f results/th_combo_r.json ] && [ -f results/ckpt/th_combo_r_transformer.pt ]; }; then
  idle; echo; echo "=== th_combo_r $(date +%T): 3000 steps (th_w4k_r + the world with replay, 4 KB window)"
  "$PY" scripts/run_sparsela.py --run-name th_combo_r $COMMON $TF --steps 3000 --ckpt-every 1000 --save \
    --corpus-file "$MIX" --train-bytes $(( MN - 600000 )) --val-bytes 500000 --val-offset $(( MN - 600000 )) --match-feat 8,16,32 --mem-feat "$KD/world-mix" --mem-orders "$ALL18" \
    --init-from results/ckpt/th_w4k_r_transformer.pt 2>&1 | tee -a results/th_combo_r.out | grep --line-buffered -v "^  "
  rm -f results/ckpt/th_combo_r_transformer.step*.pt
  [ -f results/th_combo_r.json ] || { echo "th_combo_r did not finish"; tail -n 20 results/th_combo_r.out; exit 1; }
fi
# 3. the standard scores for the record (2 KB windows of talk.bytes, the plain features)
score() { local name=$1 short=$2 set=$3 off=$4 n=$5 win=${6:-}; local o="k_${short}_$set"; [ -f "$OUT/$o.json" ] && return 0; local wf=(); [ -n "$win" ] && wf=(--windows "$win"); idle
  SCORE_CODE=$WT "$PY" "$CK/score_tables.py" --corpus "$CORPUS" --offset "$off" --n "$n" --T 2048 "${wf[@]}" --device cuda:0 --out-dir "$OUT" "$o=results/ckpt/${name}_transformer.pt@$TABD/feat_all28" || echo "scoring $name on $set failed"; }
score th_combo_r th_combo dev "$DEV_OFF" "$DEVWIN"; score th_combo_r th_combo kyle "$TAIL" "$NWIN" "$KD/windows.txt"; score th_combo_r th_combo dlg $(( TAIL + DOFF )) "$DWIN"
# 4. every model on the same bytes of heldout.bytes: 4 KB windows (the last 1,024 bytes compared), the 2 KB reader on their second halves
K4=$(( NWIN / 2 )); KW4=$(seq -s, 0 $((K4 - 1))); KW2=$(seq -s, 0 2 $((2 * K4 - 1)))          # his turns: 170 windows of 4 KB; the 2 KB reader's windows 0,2,4,.. from offset 2048
D4=$(( DWIN / 2 )); DW4=$(seq -s, 0 8 $((D4 - 1))); DW2=$(seq -s, 0 16 $((2 * D4 - 1)))      # the dialogue: every eighth of its 1,023 windows of 4 KB
dump() {  # dump <ckpt name> <feat dir> <tag> <offset> <n> <T> <windows>
  local name=$1 feat=$2 tag=$3 off=$4 n=$5 T=$6 wins=$7
  [ -f "$KDUMP/${tag}_w0.json" ] && return 0
  idle; ( umask 077; cd "$SP"; SCORE_CODE=$WT "$PY" "$CK/dump_window.py" --corpus "$HELD" --offset "$off" --n "$n" --T "$T" --windows "$wins" --feat "$feat" --ckpt "$WT/results/ckpt/${name}_transformer.pt" --name "$tag" --out-dir "$KDUMP" --device cuda:0 | grep -c window )
}
for set in kyle dlg; do
  if [ $set = kyle ]; then off=0; n4=$K4; w4=$KW4; w2=$KW2; else off=$DOFF; n4=$D4; w4=$DW4; w2=$DW2; fi
  dump th_bytes_r "$KD/heldout-feat-plain" "cb_bytes2k_$set" $(( off + 2048 )) $(( 2 * n4 - 1 )) 2048 "$w2"
  dump th_w4k_r   "$KD/heldout-feat-plain" "cb_w4k_$set"     "$off" "$n4" 4096 "$w4"
  dump th_w4k_r   "$KD/heldout-feat-book"  "cb_w4kbook_$set" "$off" "$n4" 4096 "$w4"
  dump th_combo_r "$KD/heldout-feat-plain" "cb_combo_$set"   "$off" "$n4" 4096 "$w4"
  dump th_combo_r "$KD/heldout-feat-book"  "cb_combobook_$set" "$off" "$n4" 4096 "$w4"
done
"$PY" - "$KDUMP" <<'PY' | tee "$OUT/combo.txt" | tee -a "$OUT/leads.txt"
import json, glob, sys
d = sys.argv[1]
def last(tag, tail):  # the mean cost of the last `tail` bytes of each dumped window, in window order
    ps = sorted(glob.glob(f"{d}/{tag}_w*.json"), key=lambda p: int(p.split("_w")[-1][:-5]))
    vals = [sum(json.load(open(p))["cost"]["full"][-tail:]) / tail for p in ps]
    return sum(vals) / len(vals), len(vals)
for s, name in (("kyle", "Kyle's turns"), ("dlg", "the dialogue")):
    b, nb = last(f"cb_bytes2k_{s}", 1024); rows = [("bytes2k (the hour reader, 1 KB of context)", b)]
    for tag, what in (("w4k", "w4k (3 KB of context)"), ("w4kbook", "w4k + the book"), ("combo", "w4k + the world fine-tune"), ("combobook", "w4k + the fine-tune + the book")):
        v, n = last(f"cb_{tag}_{s}", 1024); rows.append((what, v))
    print(f"the levers combined on the last 1,024 bytes of {nb} 4 KB windows of {name}: " + "; ".join(f"{w} {v:.4f} ({v - b:+.4f})" for w, v in rows))
PY
touch "$OUT/combo.done"; echo "$UNIT done $(date '+%F %T')"
