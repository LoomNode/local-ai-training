#!/usr/bin/env bash
# The distillation pilot (T6 of docs/superpowers/plans/2026-10-20-distillation.md; the spec docs/superpowers/specs/2026-10-20-distillation-design.md).
# Waits for the sweep (results/tables/qwen-sweep/sweep.done) and the slice's features. Then on GPU 1:
#   P0  bench: each teacher candidate on the first MB of the slice (KB/s, dropped mass)                -> results/tables/distill/bench.txt
#   P1  the pass: the chosen teacher over the 32 MB slice (/vault/ailab/flybrain-mem/distill/teacher)   -> teacher.json (resumable)
#   P3  seven hour arms at 25,562 steps (the hour recipe, 4 layers, one session): a_text (mixB), b_slice (the slice, text), c_a02 c_a05
#       c_a08 c_a10 (the slice, the teacher at alpha), d_mix (mixB + the slice half and half, the best alpha of c by the in-run dev eval);
#       each scored cold and warm on the mix's dev (1,024 windows), Kyle's turns (340) and the dialogue (2,047) with score_tables.py;
#       the runs from the tok worktree (the current runner, where the hour arms ran), the results in results/tables/distill/ here
#   then the table (every arm against a_text and b_slice, paired per window)                            -> results/tables/distill/distill.txt
# Disk is checked before every big step (Kyle: watch disc space). A failed step is logged and the chain goes on where it can.
# usage: CUDA_VISIBLE_DEVICES=<uuid> talk_distill_gpu.sh   (log results/tables/distill/distill.log; done file results/tables/distill/pilot.done)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?}"
STEPS=${STEPS:-25562}
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok; C2=$SP/notes/model2/checks; CK=$SP/notes/speed-round/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python; PYQ=/games/ailab/lab/flybrain-lab/.venv-qwen/bin/python
NV=/games/ailab/lab/flybrain-lab/.venv-qwen/lib/python3.12/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cuda_runtime/lib:$NV/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
C=/games/ailab/corpora; VM=/vault/ailab/flybrain-mem/distill; OUT=$SP/results/tables/distill; KD=$C/kyle
M27=/vault/ailab/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf; MOE=/vault/ailab/models/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf
LOOK="4,8,12,16,32,64,116,132,202,203,301"; ALL18="$LOOK,1001,1002,1003,1004,1005,1006,1007"
COMMON="--task code --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 4 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-fixed 128 --eval-every 2000"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --fused-channels"
mkdir -p "$OUT" "$VM" "$WT/results/ckpt"; cd "$WT"; exec >> "$OUT/distill.log" 2>&1     # the runs from the tok worktree: the current runner (the hour arms ran there)
log() { echo "[$(date '+%F %T')] $*"; }
free_gb() { df --output=avail -BG "$1" | tail -1 | tr -dc '0-9'; }
need() { [ "$(free_gb "$1")" -ge "$2" ] || { log "under $2 GB free on $1: stop"; exit 1; }; }
lay() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
idle() { while :; do local apps used; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$CUDA_VISIBLE_DEVICES"); used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ "${apps:-0}" -eq 0 ] && [ "${used:-99999}" -lt 2500 ] && return 0; sleep 60; done; }
log "the distillation pilot: start on $CUDA_VISIBLE_DEVICES, $STEPS steps an arm"
until [ -f "$SP/results/tables/qwen-sweep/sweep.done" ]; do sleep 300; done; log "the sweep is done"
until [ -f "$C/distill-tables/feat_all28/features.done" ]; do sleep 60; done; log "the slice's features are there"
S=$(lay "$C/distill/stacks.json" slice_bytes); TE=$(lay "$C/distill/stacks.json" train_end); D=$(lay "$C/distill/stacks.json" dev_tail)
STACK_SLICE=$(lay "$C/distill/stacks.json" stack_slice); STACK_MIX=$(lay "$C/distill/stacks.json" stack_mix); RANGES=$(lay "$C/distill/stacks.json" ranges_mix)
DEV_OFF=$(lay "$C/talkB/layout.json" dev_offset); DEVWIN=$(lay "$C/talkB/layout.json" dev_windows); TAIL=$(lay "$C/talkB/layout.json" tail_offset)
NWIN=$(lay "$KD/heldout.json" kyle | "$PY" -c 'import ast, sys; print(ast.literal_eval(sys.stdin.read())["windows"])'); DOFF=$(lay "$KD/heldout.json" dialogue | "$PY" -c 'import ast, sys; print(ast.literal_eval(sys.stdin.read())["offset"])'); DWIN=$(lay "$KD/heldout.json" dialogue | "$PY" -c 'import ast, sys; print(ast.literal_eval(sys.stdin.read())["windows"])')
log "slice $S bytes; mixB train_end $TE, dev at $DEV_OFF ($DEVWIN windows); Kyle's turns at $TAIL ($NWIN), the dialogue at +$DOFF ($DWIN)"
# ---- P0 the bench --------------------------------------------------------------------------------------------------------------------------
if [ ! -f "$OUT/bench.txt" ]; then
  for M in "$M27" "$MOE"; do
    idle; log "bench $(basename "$M")"
    ( umask 077; "$PYQ" "$C2/distill_teacher.py" --slice "$C/distill.bytes" --model "$M" --bench 1 ) 2>&1 | grep "^bench:\|windows," | sed "s|^|$(basename "$M"): |" | tee -a "$OUT/bench.txt" || log "bench $(basename "$M") failed"
  done
fi
# ---- P1 the pass ---------------------------------------------------------------------------------------------------------------------------
TEACHER=$M27
KBS=$(grep "^$(basename "$M27"): bench:" "$OUT/bench.txt" | tail -1 | sed 's/.*"kb_per_s": \([0-9.]*\).*/\1/')
if [ -n "${KBS:-}" ] && "$PY" -c "import sys; sys.exit(0 if float('$KBS') < 0.64 else 1)"; then TEACHER=$MOE; log "the 27B reads $KBS KB/s: 32 MB past 14 h, the MoE teaches"; else log "the 27B reads ${KBS:-?} KB/s: it teaches"; fi
if [ ! -f "$VM/teacher/teacher.json" ]; then
  need /vault 20; idle; log "the pass: $(basename "$TEACHER")"
  ( umask 077; "$PYQ" "$C2/distill_teacher.py" --slice "$C/distill.bytes" --out "$VM/teacher" --model "$TEACHER" ) || { log "the pass failed"; exit 1; }
fi
[ -e "$C/distill-teacher" ] || ln -s "$VM/teacher" "$C/distill-teacher"
log "teacher: $(grep -o '"kb_per_s": [0-9.]*\|"dropped_mass_mean": [0-9.]*\|"cover_mean": [0-9.]*\|"fallback_rate": [0-9.]*' "$VM/teacher/teacher.json" | tr '\n' ' ')"
# ---- P3 the arms ---------------------------------------------------------------------------------------------------------------------------
run() {  # run <name> <flags...>: resumes from the arm's newest checkpoint that loads; prunes the step checkpoints when done
  local name=$1; shift
  [ -f "results/$name.json" ] && [ -f "results/ckpt/${name}_transformer.pt" ] && { log "skip $name (done)"; return 0; }
  need /games 15
  local resume="" c
  for c in $(ls -v results/ckpt/${name}_transformer.step*.pt 2>/dev/null | tac); do
    if "$PY" -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); sys.exit(0 if {'opt', 'state', 'step'} <= set(ck) else 1)" "$c" 2>/dev/null; then resume="--resume $c"; break; fi
    mv "$c" "$c.bad"
  done
  idle; log "=== $name: $STEPS steps (${resume:-fresh})"
  "$PY" "$WT/scripts/run_sparsela.py" --run-name "$name" $COMMON $TF --steps "$STEPS" --ckpt-every 2000 --save $resume "$@" 2>&1 | tee -a "results/$name.out" | grep --line-buffered -v "^  "
  if [ -f "results/$name.json" ] && [ -f "results/ckpt/${name}_transformer.pt" ]; then rm -f results/ckpt/${name}_transformer.step*.pt results/ckpt/${name}_transformer.step*.pt.bad
  else log "$name did not finish; its last lines:"; tail -n 12 "results/$name.out"; fi
}
score() {  # score <arm> <set> <offset> <n> [<windows file>]
  local name=$1 set=$2 off=$3 n=$4 win=${5:-}; local o="k_${name}_$set"
  [ -f "$OUT/$o.json" ] && return 0; [ -f "results/ckpt/${name}_transformer.pt" ] || { log "no checkpoint for $name"; return 0; }
  local wf=(); [ -n "$win" ] && wf=(--windows "$win")
  idle; SCORE_CODE=$WT "$PY" "$CK/score_tables.py" --corpus "$C/talkB.bytes" --offset "$off" --n "$n" --T 2048 "${wf[@]}" --device cuda:0 --out-dir "$OUT" "$o=results/ckpt/${name}_transformer.pt@$C/talkB-tables/feat_all28" || log "scoring $name on $set failed"
}
SLICE=(--corpus-file "$C/distillD.bytes" --train-bytes "$S" --val-bytes 2000000 --val-offset "$S" --match-feat 8,16,32 --mem-feat-stack "$STACK_SLICE" --mem-orders "$ALL18")
run a_text --corpus-file "$C/talkB.bytes" --train-bytes "$TE" --val-bytes 2000000 --val-offset "$DEV_OFF" --match-feat 8,16,32 --mem-feat "$C/talkB-tables/feat_all28" --mem-orders "$ALL18"
run b_slice "${SLICE[@]}"
for A in 02 05 08 10; do run "c_a$A" "${SLICE[@]}" --teacher-store "$C/distill-teacher:0" --teacher-alpha "0.$A"; done
BEST=$("$PY" - <<'PY'
import json, os
best = None
for A in ("02", "05", "08", "10"):
    p = f"results/c_a{A}.json"
    if not os.path.exists(p): continue
    v = json.load(open(p))["arms"][0].get("val_fixed", {}).get("bpb_warm")
    if v is not None and (best is None or v < best[0]): best = (v, A)
print(best[1] if best else "05")
PY
)
log "the best alpha of c by the in-run dev eval: 0.$BEST"
run d_mix --corpus-file "$C/talkD.bytes" --train-bytes $((TE + S)) --val-bytes 2000000 --val-offset $((TE + S)) --match-feat 8,16,32 --mem-feat-stack "$STACK_MIX" --mem-orders "$ALL18" --sample-ranges "$RANGES" --teacher-store "$C/distill-teacher:$TE" --teacher-alpha "0.$BEST"
for name in a_text b_slice c_a02 c_a05 c_a08 c_a10 d_mix; do
  score "$name" dev "$DEV_OFF" "$DEVWIN"; score "$name" kyle "$TAIL" "$NWIN" "$KD/windows.txt"; score "$name" dlg $((TAIL + DOFF)) "$DWIN"
done
# ---- the table ---------------------------------------------------------------------------------------------------------------------------------
"$PY" - "$OUT" <<'PY' | tee -a "$OUT/distill.txt"
import json, os, sys
import numpy as np
d = sys.argv[1]
def load(n, s): p = f"{d}/k_{n}_{s}.json"; return json.load(open(p)) if os.path.exists(p) else None
arms = ["a_text", "b_slice", "c_a02", "c_a05", "c_a08", "c_a10", "d_mix"]; sets = [("dev", "the mix's dev"), ("kyle", "Kyle's turns"), ("dlg", "the dialogue")]
R = {(n, s): load(n, s) for n in arms for s, _ in sets}
def paired(r, ref, k):
    a, b = np.asarray(r[f"per_window_{k}"]), np.asarray(ref[f"per_window_{k}"]); n = min(len(a), len(b)); dd = a[:n] - b[:n]
    return f"{dd.mean():+.4f} (median {np.median(dd):+.4f}, better >0.02 on {int((dd < -0.02).sum())}, worse on {int((dd > 0.02).sum())})"
print("the distillation pilot (the hour recipe, 25,562 steps an arm, one session): cold / warm bits per byte; differences paired per window against a_text (mixB, text) and, for the c arms, b_slice (the slice, text):")
for n in arms:
    for s, nm in sets:
        r = R[(n, s)]
        if r is None: print(f"  {n} on {nm}: not scored"); continue
        line = f"  {n} on {nm}: cold {r['bpb']:.4f}, warm {r['bpb_warm']:.4f}"
        if n != "a_text" and R[("a_text", s)]: line += f"; vs a cold {paired(r, R[('a_text', s)], 'cold')}, warm {paired(r, R[('a_text', s)], 'warm')}"
        if n.startswith("c_") and R[("b_slice", s)]: line += f"; vs b cold {paired(r, R[('b_slice', s)], 'cold')}, warm {paired(r, R[('b_slice', s)], 'warm')}"
        print(line)
PY
touch "$OUT/pilot.done"; log "pilot done"
