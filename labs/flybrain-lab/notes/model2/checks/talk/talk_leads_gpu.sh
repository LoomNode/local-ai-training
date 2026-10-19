#!/usr/bin/env bash
# The leads after the four tests (Kyle, 2026-10-18 evening: "Queue it all up then. Let's get it testing.", R127), on the prose corpus,
# each scored on the mix's dev, Kyle's turns and the dialogue against th_bytes_r:
#   th_l0_r     0 layers: the channels and the output head alone (what the network adds at word starts vs inside words: the probe for
#   th_l1_r     1 layer     the autocomplete split, a decider per word + a shallow speller); both dumped by kind (weight_pile.py)
#   th_w4k_r    a 4,096-byte window, batch 4 (the same bytes a step): the context lead; also dumped on 4,096-byte windows of his turns
#   th_world_r  th_bytes_r's weights fine-tuned 2,000 steps on Kyle's world (world.bytes, inputs through the corpus tables alone,
#               world_features.py), lr 1e-4: the register in the weights
#   th_world2_r the same with replay: the world followed by a 32 MB slice of the corpus's train region (world-mix), 3,000 steps
# Checkpoints every 2,000 steps with a resume (R123); waits for the card.
# usage: CUDA_VISIBLE_DEVICES=<uuid> UNIT=<unit>.service talk_leads_gpu.sh [arm ...]   (log results/tables/talk/leads.log; lines talk/leads.txt; done file talk/leads.done)
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
WORLD=$KD/world.bytes; WFEAT=$KD/world-feat
LOOK="4,8,12,16,32,64,116,132,202,203,301"; ALL18="$LOOK,1001,1002,1003,1004,1005,1006,1007"
COMMON="--task code --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-fixed 128 --eval-every 2000"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --fused-channels"
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(th_l0_r th_l1_r th_w4k_r th_world_r)
mkdir -p "$OUT" "$WT/results"
cd "$WT"
exec >> "$OUT/leads.log" 2>&1
echo "the leads: $UNIT start $(date '+%F %T'): ${ARMS[*]}"
lay() { "$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))[sys.argv[2]]))' "$1" "$2"; }
TRAIN_END=$(lay "$LAYOUT" train_end); DEV_OFF=$(lay "$LAYOUT" dev_offset); DEVWIN=$(lay "$LAYOUT" dev_windows); TAIL=$(lay "$LAYOUT" tail_offset)
NWIN=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["kyle"]["windows"])')
DOFF=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["dialogue"]["offset"])')
DWIN=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["dialogue"]["windows"])')
idle() {
  while :; do
    local apps used
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$CUDA_VISIBLE_DEVICES")
    used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ "$apps" = 0 ] && [ -n "$used" ] && [ "$used" -lt 2500 ] && return 0
    sleep 60
  done
}
run() {  # run <name> <steps> <layers> <flags...>
  local name=$1 steps=$2 layers=$3; shift 3
  [ -f "results/$name.json" ] && [ -f "results/ckpt/${name}_transformer.pt" ] && { echo "skip $name (done)"; return 0; }
  local resume="" c
  for c in $(ls -v results/ckpt/${name}_transformer.step*.pt 2>/dev/null | tac); do
    if "$PY" -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); sys.exit(0 if {'opt', 'state', 'step'} <= set(ck) else 1)" "$c" 2>/dev/null; then
      resume="--resume $c"; break
    fi
    echo "$(date +%T) checkpoint $c does not load (cut off mid-save?): set aside as $c.bad"; mv "$c" "$c.bad"
  done
  while [ -f "$OUT/leads.hold" ]; do sleep 60; done
  idle; echo "$UNIT" > "results/claim_$name"
  echo; echo "=== $name $(date +%T): $steps steps ($layers layers, ${resume:-fresh})"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON --n-layers "$layers" $TF --steps "$steps" --ckpt-every 2000 --save $resume "$@" 2>&1 | tee -a "results/$name.out" | grep --line-buffered -v "^  "
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
line() {  # line <short> <what>
  "$PY" - "$OUT" "$WT/results" "$1" "$2" <<'PY'
import json, os, sys
d, r, short, what = sys.argv[1:5]
def load(p): return json.load(open(p)) if os.path.exists(p) else None
v = f"{d}/leads.txt"; have = open(v).read() if os.path.exists(v) else ""
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
dump() {  # dump <arm> <short> <featdir> [<T>]: the per-byte costs on the pile's windows (his turns, the dialogue, the mix's dev)
  local name=$1 short=$2 feat=$3 T=${4:-2048}; local KDUMP=$KD/dumps
  [ -f "$KDUMP/${short}_kyle_w0.json" ] && return 0
  [ -f "results/ckpt/${name}_transformer.pt" ] || return 0
  idle; ( umask 077; mkdir -p "$KDUMP"; cd "$SP"
  if [ "$T" = 2048 ]; then
    SCORE_CODE=$WT "$PY" "$CK/dump_window.py" --corpus "$CORPUS" --offset "$TAIL" --n "$NWIN" --windows 0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300 --feat "$feat" --ckpt "$WT/results/ckpt/${name}_transformer.pt" --name "${short}_kyle" --out-dir "$KDUMP" --device cuda:0 | grep -c window
    SCORE_CODE=$WT "$PY" "$CK/dump_window.py" --corpus "$CORPUS" --offset $(( TAIL + DOFF )) --n "$DWIN" --windows 0,128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920 --feat "$feat" --ckpt "$WT/results/ckpt/${name}_transformer.pt" --name "${short}_dlg" --out-dir "$KDUMP" --device cuda:0 | grep -c window
    umask 022; SCORE_CODE=$WT "$PY" "$CK/dump_window.py" --corpus "$CORPUS" --offset "$DEV_OFF" --n "$DEVWIN" --windows 0,128,256,384,512,640,768,896 --feat "$feat" --ckpt "$WT/results/ckpt/${name}_transformer.pt" --name "${short}_dev" --out-dir "$SP/tools/board/data" --device cuda:0 | grep -c window
    "$PY" "$C2/weight_pile.py" "his turns, $short=$KDUMP/${short}_kyle_w*.json" "the dialogue, $short=$KDUMP/${short}_dlg_w*.json" "the mix's dev, $short=$SP/tools/board/data/${short}_dev_w*.json" > "$OUT/weight_pile_$short.txt" 2>&1
  else  # the long window: T-byte windows of his turns, 16 of them, and the 2 KB reader on their last 2 KB; the last 1,024 bytes compared
    local nl=$(( NWIN * 2048 / T )) ratio=$(( T / 2048 )) step=$(( NWIN * 2048 / T / 16 )); local wl="" wb=""
    for i in $(seq 0 15); do wl="$wl,$(( i * step ))"; wb="$wb,$(( i * step * ratio ))"; done; wl=${wl#,}; wb=${wb#,}
    SCORE_CODE=$WT "$PY" "$CK/dump_window.py" --corpus "$CORPUS" --offset "$TAIL" --n "$nl" --T "$T" --windows "$wl" --feat "$feat" --ckpt "$WT/results/ckpt/${name}_transformer.pt" --name "${short}_kyle_long" --out-dir "$KDUMP" --device cuda:0 | grep -c window
    SCORE_CODE=$WT "$PY" "$CK/dump_window.py" --corpus "$CORPUS" --offset $(( TAIL + T - 2048 )) --n $(( nl * ratio - ratio + 1 )) --T 2048 --windows "$wb" --feat "$TABD/feat_all28" --ckpt "$WT/results/ckpt/th_bytes_r_transformer.pt" --name "${short}_kyle_base" --out-dir "$KDUMP" --device cuda:0 | grep -c window
    "$PY" - "$KDUMP" "$short" "$T" <<'PY' | tee -a "$OUT/leads.txt"
import json, sys, glob
d, short, T = sys.argv[1], sys.argv[2], int(sys.argv[3])
key = lambda p: int(p.split("_w")[-1][:-5])
a = [json.load(open(p)) for p in sorted(glob.glob(f"{d}/{short}_kyle_long_w*.json"), key=key)]; b = [json.load(open(p)) for p in sorted(glob.glob(f"{d}/{short}_kyle_base_w*.json"), key=key)]
la = [sum(x["cost"]["full"][-1024:]) / 1024 for x in a]; lb = [sum(y["cost"]["full"][-1024:]) / 1024 for y in b]
print(f"{short} vs th_bytes on the last 1,024 bytes of {len(a)} {T:,}-byte windows of Kyle's turns ({T - 1024:,} vs 1,024 bytes of context): {sum(la) / len(la):.4f} vs {sum(lb) / len(lb):.4f} ({sum(la) / len(la) - sum(lb) / len(lb):+.4f})")
PY
  fi )
}
BYTEFLAGS=(--corpus-file "$CORPUS" --train-bytes "$TRAIN_END" --val-bytes 2000000 --val-offset "$DEV_OFF" --match-feat 8,16,32 --mem-feat "$TABD/feat_all28" --mem-orders "$ALL18")
for arm in "${ARMS[@]}"; do
  case $arm in
    th_l0_r)  run th_l0_r "$STEPS_HOUR" 0 "${BYTEFLAGS[@]}"; score3 th_l0_r th_l0 "$TABD/feat_all28"; line th_l0 "0 layers: the channels and the head alone"; dump th_l0_r thl0 "$TABD/feat_all28" ;;
    th_l1_r)  run th_l1_r "$STEPS_HOUR" 1 "${BYTEFLAGS[@]}"; score3 th_l1_r th_l1 "$TABD/feat_all28"; line th_l1 "1 layer"; dump th_l1_r thl1 "$TABD/feat_all28" ;;
    th_w4k_r) run th_w4k_r "$STEPS_HOUR" 4 "${BYTEFLAGS[@]}" --seq-len 4096 --batch 4; score3 th_w4k_r th_w4k "$TABD/feat_all28"; line th_w4k "a 4,096-byte window, batch 4, scored on 2,048 windows"; dump th_w4k_r thw4k "$TABD/feat_all28" 4096 ;;
    th_world_r) WN=$(stat -c %s "$WORLD"); [ -f "$WFEAT/features.json" ] || { echo "no world features"; continue; }
              run th_world_r 2000 4 --corpus-file "$WORLD" --train-bytes $(( WN - 600000 )) --val-bytes 500000 --val-offset $(( WN - 600000 )) --match-feat 8,16,32 --mem-feat "$WFEAT" --mem-orders "$ALL18" \
                  --init-from results/ckpt/th_bytes_r_transformer.pt --lr 0.0001 --warmup 100
              score3 th_world_r th_world "$TABD/feat_all28"; line th_world "th_bytes_r fine-tuned 2,000 steps on Kyle's world, lr 1e-4"; dump th_world_r thwo "$TABD/feat_all28" ;;
    th_world2_r) MIX=$KD/world-mix.bytes; MN=$(stat -c %s "$MIX"); [ -f "$KD/world-mix/features.json" ] || { echo "no world-mix features"; continue; }
              run th_world2_r 3000 4 --corpus-file "$MIX" --train-bytes $(( MN - 600000 )) --val-bytes 500000 --val-offset $(( MN - 600000 )) --match-feat 8,16,32 --mem-feat "$KD/world-mix" --mem-orders "$ALL18" \
                  --init-from results/ckpt/th_bytes_r_transformer.pt --lr 0.0001 --warmup 100
              score3 th_world2_r th_world2 "$TABD/feat_all28"; line th_world2 "th_bytes_r fine-tuned 3,000 steps on Kyle's world + a 32 MB corpus slice (replay), lr 1e-4" ;;
    th_world3_r) MIX=$KD/world-mix.bytes; MN=$(stat -c %s "$MIX")
              run th_world3_r 6000 4 --corpus-file "$MIX" --train-bytes $(( MN - 600000 )) --val-bytes 500000 --val-offset $(( MN - 600000 )) --match-feat 8,16,32 --mem-feat "$KD/world-mix" --mem-orders "$ALL18" --init-from results/ckpt/th_bytes_r_transformer.pt --lr 0.0001 --warmup 100
              score3 th_world3_r th_world3 "$TABD/feat_all28"; line th_world3 "the replay fine-tune at 6,000 steps, lr 1e-4" ;;
    th_world4_r) MIX=$KD/world-mix.bytes; MN=$(stat -c %s "$MIX")
              run th_world4_r 3000 4 --corpus-file "$MIX" --train-bytes $(( MN - 600000 )) --val-bytes 500000 --val-offset $(( MN - 600000 )) --match-feat 8,16,32 --mem-feat "$KD/world-mix" --mem-orders "$ALL18" --init-from results/ckpt/th_bytes_r_transformer.pt --lr 0.0003 --warmup 100
              score3 th_world4_r th_world4 "$TABD/feat_all28"; line th_world4 "the replay fine-tune at 3,000 steps, lr 3e-4" ;;
    th_w8k_r) run th_w8k_r "$STEPS_HOUR" 4 "${BYTEFLAGS[@]}" --seq-len 8192 --batch 2; score3 th_w8k_r th_w8k "$TABD/feat_all28"; line th_w8k "an 8,192-byte window, batch 2, scored on 2,048 windows"; dump th_w8k_r thw8k "$TABD/feat_all28" 8192 ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
done
touch "$OUT/leads.done"; echo "$UNIT done $(date '+%F %T')"
