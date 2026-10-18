#!/usr/bin/env bash
# Model #2's pair at the hour on the prose corpus (R118: both ways as we scale; R124: the corpus = mixD): tok_hour_gpu.sh's three arms on
# talk.bytes and its token stream (talk_corpus.sh), each with the corpus's own 2^28 features, scored on three sets: the mix's dev windows
# (the first 1,024 windows of the held-out documents at dev_offset), Kyle's turns (340 windows at tail_offset, the held-out set's
# windows.txt) and the dialogue (2,047 windows after them). Arms of the 4-layer reader from the tok worktree:
#   th_bytes_r   bytes, the 18 table inputs, the match 8/16/32, 25,562 steps (the hour)
#   th_tok_r     32k tokens (the corpus's own BPE), the 20 token inputs, the match 2/4/8, the shared channel, 25,562 steps (equal steps)
#   th_tok_eq_r  the token arm at the steps that cost the byte arm's clock (the ratio of the two arms' logged clocks, this session)
# Checkpoints every 2,000 steps with a resume (R123). Waits for the corpus build (talk/build.done) and for the card to be free.
# usage: CUDA_VISIBLE_DEVICES=<uuid> UNIT=<unit>.service talk_hour_gpu.sh   (log results/tables/talk/hour.log; verdicts talk/hour.txt; done file talk/hour.done)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the card UUID}"; : "${UNIT:?}"
STEPS_HOUR=${STEPS_HOUR:-25562}
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
C2=$SP/notes/model2/checks; CK=$SP/notes/speed-round/checks
OUT=$SP/results/tables/talk
CORPUS=/games/ailab/corpora/talk.bytes; LAYDIR=/games/ailab/corpora/talk; LAYOUT=$LAYDIR/layout.json
TOKD=/games/ailab/corpora/talk-tok; TABD=/games/ailab/corpora/talk-tables
KD=/games/ailab/corpora/kyle; KWIN=$KD/windows.txt
LOOK="4,8,12,16,32,64,116,132,202,203,301"; ALL18="$LOOK,1001,1002,1003,1004,1005,1006,1007"
TOK20="1,2,3,4,6,8,12,16,116,132,202,203,301,1001,1002,1003,1004,1005,1006,1007"
COMMON="--task code --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 4 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-fixed 128 --eval-every 2000"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --fused-channels"
mkdir -p "$OUT" "$WT/results"
cd "$WT"
exec >> "$OUT/hour.log" 2>&1
echo "the pair at the hour on the prose corpus, $UNIT start $(date '+%F %T'): $STEPS_HOUR steps an arm"
until [ -f "$LAYDIR/build.done" ]; do sleep 300; done
lay() { "$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))[sys.argv[2]]))' "$1" "$2"; }
TRAIN_END=$(lay "$LAYOUT" train_end); DEV_OFF=$(lay "$LAYOUT" dev_offset); DEVWIN=$(lay "$LAYOUT" dev_windows); TAIL=$(lay "$LAYOUT" tail_offset)
TRAIN_END_TOK=$(lay "$TOKD/layout.json" train_end_tok)
NWIN=$(lay "$KD/heldout.json" kyle.windows 2>/dev/null || "$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["kyle"]["windows"])')
DOFF=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["dialogue"]["offset"])')
DWIN=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["dialogue"]["windows"])')
echo "corpus: train_end $TRAIN_END bytes / $TRAIN_END_TOK tokens; dev at $DEV_OFF ($DEVWIN windows); Kyle's turns at $TAIL ($NWIN windows), the dialogue at +$DOFF ($DWIN windows)"
for f in "$TABD/feat_all28/features.json" "$TOKD/feat28/features.json" "$TOKD/tokens.u16"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
idle() {  # wait for this card to be free
  while :; do
    local apps used
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
    used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "$apps" = 0 ] && [ "$used" -lt 2500 ] && return 0
    sleep 60
  done
}
run() {  # run <name> <steps> <flags...>: resumes from the arm's newest checkpoint that loads; prunes the step checkpoints when done
  local name=$1 steps=$2; shift 2
  [ -f "results/$name.json" ] && [ -f "results/ckpt/${name}_transformer.pt" ] && { echo "skip $name (done)"; return 0; }
  local resume="" c
  for c in $(ls -v results/ckpt/${name}_transformer.step*.pt 2>/dev/null | tac); do
    if "$PY" -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); sys.exit(0 if {'opt', 'state', 'step'} <= set(ck) else 1)" "$c" 2>/dev/null; then
      resume="--resume $c"; break
    fi
    echo "$(date +%T) checkpoint $c does not load (cut off mid-save?): set aside as $c.bad"; mv "$c" "$c.bad"
  done
  idle; echo "$UNIT" > "results/claim_$name"
  echo; echo "=== $name $(date +%T): $steps steps (4 layers, ${resume:-fresh})"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON $TF --steps "$steps" --ckpt-every 2000 --save $resume "$@" 2>&1 | tee -a "results/$name.out" | grep --line-buffered -v "^  "
  rm -f "results/claim_$name"
  if [ -f "results/$name.json" ] && [ -f "results/ckpt/${name}_transformer.pt" ]; then
    rm -f results/ckpt/${name}_transformer.step*.pt results/ckpt/${name}_transformer.step*.pt.bad
  else
    echo "$name did not finish; its last lines:"; tail -n 20 "results/$name.out"
  fi
}
BYTEFLAGS=(--corpus-file "$CORPUS" --train-bytes "$TRAIN_END" --val-bytes 2000000 --val-offset "$DEV_OFF" --match-feat 8,16,32 --mem-feat "$TABD/feat_all28" --mem-orders "$ALL18")
TOKFLAGS=(--tokens --vocab 32768 --corpus-file "$TOKD/tokens.u16" --train-bytes "$TRAIN_END_TOK" --val-bytes 700000 --val-offset "$TRAIN_END_TOK"
          --tok-len "$TOKD/tok_len.u8" --match-feat 2,4,8 --mem-feat "$TOKD/feat28" --mem-orders "$TOK20" --shared-table)
run th_bytes_r "$STEPS_HOUR" "${BYTEFLAGS[@]}"
run th_tok_r "$STEPS_HOUR" "${TOKFLAGS[@]}"
pace() { "$PY" -c 'import json, sys; a = json.load(open(sys.argv[1]))["arms"][0]; print(a["train_seconds"] / int(sys.argv[2]))' "results/$1.json" "$2" 2>/dev/null; }
if [ -f results/th_bytes_r.json ] && [ -f results/th_tok_r.json ]; then
  PB=$(pace th_bytes_r "$STEPS_HOUR"); PT=$(pace th_tok_r "$STEPS_HOUR")
  STEPS_EQ=$("$PY" -c "print(max(1000, round($STEPS_HOUR * $PB / $PT)))")
  echo "paces: bytes $PB s/step, tokens $PT s/step (ratio $("$PY" -c "print(round($PT / $PB, 3))")); equal-compute token arm: $STEPS_EQ steps"
  run th_tok_eq_r "$STEPS_EQ" "${TOKFLAGS[@]}"
fi
score() {  # score <arm> <short> bytes|tok <set> <offset> <n> [<windows file>]: one arm on one window set
  local name=$1 short=$2 kind=$3 set=$4 off=$5 n=$6 win=${7:-}; local o="k_${short}_$set"
  [ -f "$OUT/$o.json" ] && return 0
  [ -f "results/ckpt/${name}_transformer.pt" ] || { echo "no checkpoint for $name"; return 0; }
  local wf=(); [ -n "$win" ] && wf=(--windows "$win")
  idle
  if [ "$kind" = tok ]; then
    SCORE_CODE=$WT "$PY" "$C2/score_tokens.py" --stream "$TOKD" --offset "$off" --n "$n" --T 2048 "${wf[@]}" --warm-bytes 2048 --device cuda:0 \
      --out-dir "$OUT" "$o=results/ckpt/${name}_transformer.pt@$TOKD/feat28" || echo "scoring $name on $set failed"
  else
    SCORE_CODE=$WT "$PY" "$CK/score_tables.py" --corpus "$CORPUS" --offset "$off" --n "$n" --T 2048 "${wf[@]}" --device cuda:0 \
      --out-dir "$OUT" "$o=results/ckpt/${name}_transformer.pt@$TABD/feat_all28" || echo "scoring $name on $set failed"
  fi
}
for arm in "th_bytes_r th_bytes bytes" "th_tok_r th_tok tok" "th_tok_eq_r th_tok_eq tok"; do
  set -- $arm
  score "$1" "$2" "$3" dev "$DEV_OFF" "$DEVWIN"
  score "$1" "$2" "$3" kyle "$TAIL" "$NWIN" "$KWIN"
  score "$1" "$2" "$3" dlg $(( TAIL + DOFF )) "$DWIN"
done
"$PY" - "$OUT" "$WT/results" "$STEPS_HOUR" <<'PY' | tee -a "$OUT/hour.txt"
import json, os, sys
d, r, S = sys.argv[1], sys.argv[2], int(sys.argv[3])
def load(p): return json.load(open(p)) if os.path.exists(p) else None
for name, short, what in (("th_bytes_r", "th_bytes", "bytes"), ("th_tok_r", "th_tok", "tokens, equal steps"), ("th_tok_eq_r", "th_tok_eq", "tokens, equal compute")):
    run = load(f"{r}/{name}.json"); sets = {s: load(f"{d}/k_{short}_{s}.json") for s in ("dev", "kyle", "dlg")}
    if not run or not all(sets.values()): print(f"{short}: not complete"); continue
    a = run["arms"][0]; bpt = f", {sets['dev']['bpt']:.3f} bits/token on dev" if "bpt" in sets["dev"] else ""
    print(f"{short} ({what}, {run['steps']} steps, clock {a['train_seconds']:.0f} s): the mix's dev ({sets['dev']['n']} windows) cold {sets['dev']['bpb']:.4f} warm {sets['dev']['bpb_warm']:.4f}; "
          f"Kyle's turns ({sets['kyle']['n']}) cold {sets['kyle']['bpb']:.4f} warm {sets['kyle']['bpb_warm']:.4f}; the dialogue ({sets['dlg']['n']}) cold {sets['dlg']['bpb']:.4f} warm {sets['dlg']['bpb_warm']:.4f}{bpt}")
PY
touch "$OUT/hour.done"
echo "the pair at the hour on the prose corpus done $(date '+%F %T')"
