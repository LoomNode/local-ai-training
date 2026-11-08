#!/usr/bin/env bash
# The reader as the judge (reader_judge.py), on the CPU while both cards are busy (2026-10-20 evening): first the plain 4 KB reader
# (th_w4k_r, trained on mixD, its saved tables talk-tables/t28), then the reader fine-tuned on Kyle's world (th_combo_r, its features
# from the book index corpora/kyle/index). Candidates: every standard-format run on the 1,500-byte pool so far (the spans, the tables, the
# breadth vote, the dedup chooser, the sweep's floor6/floor10/top1/dedup100/dedup500/dedupfrac/top4 and the breadth arms if scored).
# usage: judge_cpu.sh   (log results/tables/judge/judge.log; results results/tables/judge/<reader>/judge.{json,txt})
set -u -o pipefail
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; C2=$SP/notes/model2/checks; PY=/games/ailab/lab/flybrain-lab/.venv/bin/python; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok
R=$SP/results/tables; SW=$R/qwen-sweep; OUT=$R/judge; export CUDA_VISIBLE_DEVICES=""
mkdir -p "$OUT"; cd "$SP"; exec >> "$OUT/judge.log" 2>&1; umask 077
RUNS=("tables=$R/qwen-tables/qwen_memory.json:$R/qwen-tables/choices.json" "breadth=$R/qwen-breadth/qwen_memory.json:$R/qwen-breadth/choices.json" "dedup=$R/qwen-dedup/qwen_memory.json:$R/qwen-dedup/choices.json")
for arm in floor6 floor10 top1 dedup100 dedup500 dedupfrac top4 b12 bbytes bkeys bdedup border brarity bcap0 bcap4 bcap64 b16 b8 b321612 b12words; do
  [ -f "$SW/$arm/qwen_memory.json" ] && RUNS+=("$arm=$SW/$arm/qwen_memory.json:$SW/$arm/choices.json")
done
echo "[$(date '+%F %T')] the judge: ${#RUNS[@]} candidate sources beside the spans"
for spec in "w4k=$WT/results/ckpt/th_w4k_r_transformer.pt=/games/ailab/corpora/talk-tables/t28" "combo=$WT/results/ckpt/th_combo_r_transformer.pt=/games/ailab/corpora/kyle/index"; do
  IFS='=' read -r name ckpt tables <<< "$spec"
  [ -f "$OUT/$name/judge.json" ] && { echo "skip $name"; continue; }
  echo "[$(date '+%F %T')] == $name"
  nice -n 19 taskset -c 0-5,8-13 "$PY" "$C2/reader_judge.py" --ckpt "$ckpt" --tables "$tables" --out "$OUT/$name" --ref "$R/qwen/qwen_memory.json" --spans-choices "$R/qwen-dedup/choices.json" --runs "${RUNS[@]}" --second --threads 6 || echo "$name failed"
  grep "^judge" "$OUT/$name/judge.txt" 2>/dev/null | tail -2
done
touch "$OUT/judge.done"; echo "[$(date '+%F %T')] judge done"
