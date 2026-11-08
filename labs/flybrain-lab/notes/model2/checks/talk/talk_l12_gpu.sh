#!/usr/bin/env bash
# Model #2's second scale point on the prose corpus (R118: both ways as we scale): talk_hour_gpu.sh's three arms at the big train's depth,
# 12 layers, at four times the hour's budget, the step counts of the code point (tok/l12g_steps.env, the quiet calibration of
# tok_l12_cal.sh: the per-step work is the same model with the same inputs and vocabulary, so the ratio carries over; clocks logged).
#   tl12_bytes_r    bytes, the 18 table inputs, the match 8/16/32, STEPS_BYTES steps
#   tl12_tok_eqc_r  32k tokens, the 20 token inputs, the match 2/4/8, the shared channel, STEPS_TOK steps (equal compute)
#   tl12_tok_eqs_r  the token arm at STEPS_BYTES steps (equal steps, 1.77x the compute: quality only)
# Each arm scored on the mix's dev windows, Kyle's turns and the dialogue (talk_hour_gpu.sh's three sets); checkpoints every 2,000 steps
# with a resume (R123). Waits for the card to be free before each arm and each scoring.
# usage: CUDA_VISIBLE_DEVICES=<uuid> UNIT=<unit>.service talk_l12_gpu.sh   (log results/tables/talk/l12.log; verdicts talk/l12.txt; done file talk/l12.done)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the card UUID}"; : "${UNIT:?}"
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
C2=$SP/notes/model2/checks; CK=$SP/notes/speed-round/checks
OUT=$SP/results/tables/talk
CORPUS=/games/ailab/corpora/talk.bytes; LAYDIR=/games/ailab/corpora/talk; LAYOUT=$LAYDIR/layout.json
TOKD=/games/ailab/corpora/talk-tok; TABD=/games/ailab/corpora/talk-tables
KD=/games/ailab/corpora/kyle; KWIN=$KD/windows.txt
LOOK="4,8,12,16,32,64,116,132,202,203,301"; ALL18="$LOOK,1001,1002,1003,1004,1005,1006,1007"
TOK20="1,2,3,4,6,8,12,16,116,132,202,203,301,1001,1002,1003,1004,1005,1006,1007"
COMMON="--task code --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 12 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-fixed 128 --eval-every 2000"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --fused-channels"
mkdir -p "$OUT" "$WT/results"
cd "$WT"
exec >> "$OUT/l12.log" 2>&1
echo "12 layers at 4x on the prose corpus, $UNIT start $(date '+%F %T')"
[ -f "$LAYDIR/build.done" ] || { echo "no build.done"; exit 1; }
[ -f "$SP/results/tables/tok/l12g_steps.env" ] || { echo "no l12g_steps.env"; exit 1; }
. "$SP/results/tables/tok/l12g_steps.env"
echo "steps: bytes $STEPS_BYTES, tokens at equal compute $STEPS_TOK (the code point's quiet paces $PB / $PT s/step, budget $BUDGET_S s)"
lay() { "$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))[sys.argv[2]]))' "$1" "$2"; }
TRAIN_END=$(lay "$LAYOUT" train_end); DEV_OFF=$(lay "$LAYOUT" dev_offset); DEVWIN=$(lay "$LAYOUT" dev_windows); TAIL=$(lay "$LAYOUT" tail_offset)
TRAIN_END_TOK=$(lay "$TOKD/layout.json" train_end_tok)
NWIN=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["kyle"]["windows"])')
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
  echo; echo "=== $name $(date +%T): $steps steps (12 layers, ${resume:-fresh})"
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
score3() {  # score3 <arm> <short> bytes|tok: the three sets
  score "$1" "$2" "$3" dev "$DEV_OFF" "$DEVWIN"
  score "$1" "$2" "$3" kyle "$TAIL" "$NWIN" "$KWIN"
  score "$1" "$2" "$3" dlg $(( TAIL + DOFF )) "$DWIN"
}
verdict() {
  "$PY" - "$OUT" "$WT/results" <<'PY'
import json, os, sys
d, r = sys.argv[1], sys.argv[2]
def load(p): return json.load(open(p)) if os.path.exists(p) else None
v = f"{d}/l12.txt"; have = open(v).read() if os.path.exists(v) else ""
for name, short, what in (("tl12_bytes_r", "tl12_bytes", "bytes"), ("tl12_tok_eqc_r", "tl12_tok_eqc", "tokens, equal compute"), ("tl12_tok_eqs_r", "tl12_tok_eqs", "tokens, equal steps")):
    if f"{short} (" in have: continue
    run = load(f"{r}/{name}.json"); sets = {s: load(f"{d}/k_{short}_{s}.json") for s in ("dev", "kyle", "dlg")}
    if not run or not all(sets.values()): continue
    a = run["arms"][0]; bpt = f", {sets['dev']['bpt']:.3f} bits/token on dev" if "bpt" in sets["dev"] else ""
    line = (f"{short} ({what}, 12 layers, {run['steps']} steps, clock {a['train_seconds']:.0f} s): the mix's dev ({sets['dev']['n']} windows) cold {sets['dev']['bpb']:.4f} warm {sets['dev']['bpb_warm']:.4f}; "
            f"Kyle's turns ({sets['kyle']['n']}) cold {sets['kyle']['bpb']:.4f} warm {sets['kyle']['bpb_warm']:.4f}; the dialogue ({sets['dlg']['n']}) cold {sets['dlg']['bpb']:.4f} warm {sets['dlg']['bpb_warm']:.4f}{bpt}")
    print(line); open(v, "a").write(line + "\n")
PY
}
run tl12_bytes_r "$STEPS_BYTES" "${BYTEFLAGS[@]}";   score3 tl12_bytes_r tl12_bytes bytes; verdict
run tl12_tok_eqc_r "$STEPS_TOK" "${TOKFLAGS[@]}";    score3 tl12_tok_eqc_r tl12_tok_eqc tok; verdict
run tl12_tok_eqs_r "$STEPS_BYTES" "${TOKFLAGS[@]}";  score3 tl12_tok_eqs_r tl12_tok_eqs tok; verdict
all=1; for s in tl12_bytes tl12_tok_eqc tl12_tok_eqs; do for w in dev kyle dlg; do [ -f "$OUT/k_${s}_$w.json" ] || all=0; done; done
[ "$all" = 1 ] && touch "$OUT/l12.done"
echo "12 layers at 4x on the prose corpus done $(date '+%F %T') (all scored: $all)"
