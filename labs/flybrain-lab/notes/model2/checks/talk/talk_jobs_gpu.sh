#!/usr/bin/env bash
# The weights' pile, the trained tests (Kyle, 2026-10-18: "test all 4"): the hour reader on the prose corpus with one abstraction added
# to its inputs, against th_bytes_r (ALL18, the match 8/16/32), each scored on the mix's dev, Kyle's turns and the dialogue:
#   th_dict_r   + the dictionary keys 200, 201 (feat_all28x)            spelling from a context-free lexicon
#   th_skel_r   + the short template keys 104, 108 (feat_all28x)        the skeleton at prose scale
#   th_vocab_r  + the in-window word matches 900, 902 (--match-feat)    the live vocabulary (waits for talk/match_words.ok)
# Test 1 (the right book) trains nothing: index_project.py + sample_tables.py --score-windows, in ~/world.sh and talk_book_score.sh.
# Checkpoints every 2,000 steps with a resume (R123); waits for the card and for talk/keys2.done.
# usage: CUDA_VISIBLE_DEVICES=<uuid> UNIT=<unit>.service talk_jobs_gpu.sh [arm ...]   (log results/tables/talk/jobs.log; lines talk/jobs.txt; done file talk/jobs.done)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the card UUID}"; : "${UNIT:?}"
STEPS_HOUR=${STEPS_HOUR:-25562}
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
C2=$SP/notes/model2/checks; CK=$SP/notes/speed-round/checks
OUT=$SP/results/tables/talk
CORPUS=/games/ailab/corpora/talk.bytes; LAYDIR=/games/ailab/corpora/talk; LAYOUT=$LAYDIR/layout.json
TABD=/games/ailab/corpora/talk-tables
KD=/games/ailab/corpora/kyle; KWIN=$KD/windows.txt
LOOK="4,8,12,16,32,64,116,132,202,203,301"; ALL18="$LOOK,1001,1002,1003,1004,1005,1006,1007"
COMMON="--task code --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 4 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-fixed 128 --eval-every 2000"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --fused-channels"
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(th_dict_r th_skel_r th_vocab_r)
mkdir -p "$OUT" "$WT/results"
cd "$WT"
exec >> "$OUT/jobs.log" 2>&1
echo "the weights' pile, trained tests: $UNIT start $(date '+%F %T'): ${ARMS[*]}, $STEPS_HOUR steps an arm"
until [ -f "$OUT/keys2.done" ]; do sleep 120; done
lay() { "$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))[sys.argv[2]]))' "$1" "$2"; }
TRAIN_END=$(lay "$LAYOUT" train_end); DEV_OFF=$(lay "$LAYOUT" dev_offset); DEVWIN=$(lay "$LAYOUT" dev_windows); TAIL=$(lay "$LAYOUT" tail_offset)
NWIN=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["kyle"]["windows"])')
DOFF=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["dialogue"]["offset"])')
DWIN=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["dialogue"]["windows"])')
for f in "$TABD/feat_all28/features.json" "$TABD/feat_all28x/features.json"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
idle() {
  while :; do
    local apps used
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$CUDA_VISIBLE_DEVICES")
    used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ "$apps" = 0 ] && [ -n "$used" ] && [ "$used" -lt 2500 ] && return 0
    sleep 60
  done
}
run() {
  local name=$1 steps=$2; shift 2
  [ -f "results/$name.json" ] && [ -f "results/ckpt/${name}_transformer.pt" ] && { echo "skip $name (done)"; return 0; }
  local resume="" c
  for c in $(ls -v results/ckpt/${name}_transformer.step*.pt 2>/dev/null | tac); do
    if "$PY" -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); sys.exit(0 if {'opt', 'state', 'step'} <= set(ck) else 1)" "$c" 2>/dev/null; then
      resume="--resume $c"; break
    fi
    echo "$(date +%T) checkpoint $c does not load (cut off mid-save?): set aside as $c.bad"; mv "$c" "$c.bad"
  done
  while [ -f "$OUT/jobs.hold" ]; do sleep 60; done   # a hold file (the driver window, 2026-10-18) delays the next arm
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
score() {  # score <arm> <short> <featdir> <set> <offset> <n> [<windows file>]
  local name=$1 short=$2 feat=$3 set=$4 off=$5 n=$6 win=${7:-}; local o="k_${short}_$set"
  [ -f "$OUT/$o.json" ] && return 0
  [ -f "results/ckpt/${name}_transformer.pt" ] || { echo "no checkpoint for $name"; return 0; }
  local wf=(); [ -n "$win" ] && wf=(--windows "$win")
  idle
  SCORE_CODE=$WT "$PY" "$CK/score_tables.py" --corpus "$CORPUS" --offset "$off" --n "$n" --T 2048 "${wf[@]}" --device cuda:0 \
    --out-dir "$OUT" "$o=results/ckpt/${name}_transformer.pt@$feat" || echo "scoring $name on $set failed"
}
score3() { score "$1" "$2" "$3" dev "$DEV_OFF" "$DEVWIN"; score "$1" "$2" "$3" kyle "$TAIL" "$NWIN" "$KWIN"; score "$1" "$2" "$3" dlg $(( TAIL + DOFF )) "$DWIN"; }
line() {  # line <short> <what>: the arm's three scores against th_bytes, appended once to jobs.txt
  "$PY" - "$OUT" "$WT/results" "$1" "$2" <<'PY'
import json, os, sys
d, r, short, what = sys.argv[1:5]
def load(p): return json.load(open(p)) if os.path.exists(p) else None
v = f"{d}/jobs.txt"; have = open(v).read() if os.path.exists(v) else ""
if f"{short} (" in have: sys.exit(0)
run = load(f"{r}/{short}_r.json"); sets = {s: load(f"{d}/k_{short}_{s}.json") for s in ("dev", "kyle", "dlg")}; base = {s: load(f"{d}/k_th_bytes_{s}.json") for s in ("dev", "kyle", "dlg")}
if not run or not all(sets.values()) or not all(base.values()): print(f"{short}: not complete"); sys.exit(0)
a = run["arms"][0]
parts = [f"{n} cold {sets[s]['bpb']:.4f} ({sets[s]['bpb'] - base[s]['bpb']:+.4f}) warm {sets[s]['bpb_warm']:.4f} ({sets[s]['bpb_warm'] - base[s]['bpb_warm']:+.4f})"
         for s, n in (("dev", "the mix's dev"), ("kyle", "Kyle's turns"), ("dlg", "the dialogue"))]
line = f"{short} ({what}, {run['steps']} steps, clock {a['train_seconds']:.0f} s) vs th_bytes: " + "; ".join(parts)
print(line); open(v, "a").write(line + "\n")
PY
}
BYTEFLAGS=(--corpus-file "$CORPUS" --train-bytes "$TRAIN_END" --val-bytes 2000000 --val-offset "$DEV_OFF")
for arm in "${ARMS[@]}"; do
  case $arm in
    th_dict_r)  run th_dict_r "$STEPS_HOUR" "${BYTEFLAGS[@]}" --match-feat 8,16,32 --mem-feat "$TABD/feat_all28x" --mem-orders "$ALL18,200,201"
                score3 th_dict_r th_dict "$TABD/feat_all28x"; line th_dict "the dictionary keys 200, 201" ;;
    th_skel_r)  run th_skel_r "$STEPS_HOUR" "${BYTEFLAGS[@]}" --match-feat 8,16,32 --mem-feat "$TABD/feat_all28x" --mem-orders "$ALL18,104,108"
                score3 th_skel_r th_skel "$TABD/feat_all28x"; line th_skel "the short template keys 104, 108" ;;
    th_vocab_r) until [ -f "$OUT/match_words.ok" ]; do sleep 120; done
                run th_vocab_r "$STEPS_HOUR" "${BYTEFLAGS[@]}" --match-feat 8,16,32,900,902 --mem-feat "$TABD/feat_all28" --mem-orders "$ALL18"
                score3 th_vocab_r th_vocab "$TABD/feat_all28"; line th_vocab "the in-window word matches 900, 902" ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
done
all=1; for a in "${ARMS[@]}"; do s=${a%_r}; for w in dev kyle dlg; do [ -f "$OUT/k_${s}_$w.json" ] || all=0; done; done
[ "$all" = 1 ] && touch "$OUT/jobs.done"
echo "$UNIT done $(date '+%F %T') (all scored: $all)"
