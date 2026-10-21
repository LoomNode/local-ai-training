#!/usr/bin/env bash
# Teach the reader to read memory (Kyle, 2026-10-21: "do 1 now"): th_rag_r = th_w4k_r fine-tuned 3,000 steps (the combo recipe: lr 1e-4,
# batch 4, 4 KB window, replay) on world-rag.bytes (rag_corpus.py: his world in pieces, each behind the labelled memory the span chooser
# retrieves for its first half from OTHER files, in the format the sweep found best for Qwen), inputs through the corpus tables alone
# (world_features.py). Then reader_memory.py on the CPU: the second half behind the first half alone / the plain chunks / the labelled
# memory, his turns and the dialogue (the baselines th_w4k_r and th_combo_r in results/tables/rag/base/). Resumable (the tender may stop
# it): a done arm is skipped, a stopped one resumes from its newest step checkpoint. The run from the tok worktree (the current runner).
# usage: CUDA_VISIBLE_DEVICES=<uuid> talk_rag_gpu.sh   (log results/tables/rag/rag.log; done file results/tables/rag/rag.done)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?}"
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok; C2=$SP/notes/model2/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python; KD=/games/ailab/corpora/kyle; OUT=$SP/results/tables/rag; TAB=/games/ailab/corpora/talk-tables/t28
LOOK="4,8,12,16,32,64,116,132,202,203,301"; ALL18="$LOOK,1001,1002,1003,1004,1005,1006,1007"
COMMON="--task code --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0001 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 4 --batch 4 --seq-len 4096 --seed 0 --device cuda:0 --warmup 100 --cosine --wd 0.1 --chunk 256 --amp --meter --val-fixed 64 --eval-every 1000"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --fused-channels"
mkdir -p "$OUT"; cd "$WT"; exec >> "$OUT/rag.log" 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
idle() { while :; do local apps used; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$CUDA_VISIBLE_DEVICES"); used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ "${apps:-0}" -eq 0 ] && [ "${used:-99999}" -lt 2500 ] && return 0; sleep 60; done; }
log "the reader reads memory: start on $CUDA_VISIBLE_DEVICES"
until [ -f "$KD/world-rag/features.json" ]; do sleep 60; done
MIX=$KD/world-rag.bytes; MN=$(stat -c %s "$MIX")
run() {  # run <name> <flags...>: resumes from the arm's newest checkpoint that loads
  local name=$1; shift
  [ -f "results/$name.json" ] && [ -f "results/ckpt/${name}_transformer.pt" ] && { log "skip $name (done)"; return 0; }
  local resume="" c
  for c in $(ls -v results/ckpt/${name}_transformer.step*.pt 2>/dev/null | tac); do
    if "$PY" -c "import sys, torch; ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); sys.exit(0 if {'opt', 'state', 'step'} <= set(ck) else 1)" "$c" 2>/dev/null; then resume="--resume $c"; break; fi
    mv "$c" "$c.bad"
  done
  idle; log "=== $name: 3000 steps (${resume:-fresh}) on $MIX ($MN bytes)"
  "$PY" "$WT/scripts/run_sparsela.py" --run-name "$name" $COMMON $TF --steps 3000 --ckpt-every 1000 --save $resume "$@" 2>&1 | tee -a "results/$name.out" | grep --line-buffered -v "^  "
  if [ -f "results/$name.json" ] && [ -f "results/ckpt/${name}_transformer.pt" ]; then rm -f results/ckpt/${name}_transformer.step*.pt results/ckpt/${name}_transformer.step*.pt.bad
  else log "$name did not finish; its last lines:"; tail -n 12 "results/$name.out"; fi
}
run th_rag_r --corpus-file "$MIX" --train-bytes $(( MN - 600000 )) --val-bytes 500000 --val-offset $(( MN - 600000 )) --match-feat 8,16,32 --mem-feat "$KD/world-rag" --mem-orders "$ALL18" \
    --init-from results/ckpt/th_w4k_r_transformer.pt
if [ -f "results/ckpt/th_rag_r_transformer.pt" ] && [ ! -f "$OUT/rag/memory.json" ]; then
  log "the test (CPU): th_rag_r"
  ( umask 077; CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 0-5,8-13 "$PY" "$C2/reader_memory.py" --tables "$TAB" --choices "$OUT/choices/choices.json" --out "$OUT/rag" \
      --ckpts "th_rag_r=$WT/results/ckpt/th_rag_r_transformer.pt" 2>&1 | grep -v "^  " )
fi
[ -f "$OUT/rag/memory.json" ] && { cat "$OUT/base/memory.txt" "$OUT/rag/memory.txt" 2>/dev/null > "$OUT/rag.txt"; touch "$OUT/rag.done"; log "rag done"; } || log "no result"
