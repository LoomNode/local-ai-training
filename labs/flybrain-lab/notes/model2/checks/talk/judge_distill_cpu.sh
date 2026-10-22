#!/usr/bin/env bash
# The distilled readers as judges, free (2026-10-21 evening; MODEL2.md "The reader as the judge": the plain and the world-fine-tuned
# readers were noise on Kyle's turns; the question whether a reader trained against Qwen's distribution judges Qwen's memories better).
# Waits for the pilot (results/tables/distill/pilot.done), then (1) reads the pilot's table on the leak mask's clean windows
# (leak_table.py: every arm's level and the pairs c_a*=b_slice, d_mix=a_text, b_slice=a_text) -> results/tables/distill/clean.txt;
# (2) runs reader_judge.py over the seven pilot readers with the baseline's nine candidate sources (the same set the w4k and combo
# judges saw, so the numbers compare) -> results/tables/judge/<arm>/judge.{json,txt}; marker results/tables/judge/distill.done.
# CPU only, nice 19 on cores 0-5,8-13 (~47 min a reader). usage: judge_distill_cpu.sh   (log results/tables/judge/judge_distill.log)
set -u -o pipefail
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; C2=$SP/notes/model2/checks; PY=/games/ailab/lab/flybrain-lab/.venv/bin/python; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok
R=$SP/results/tables; SW=$R/qwen-sweep; D=$R/distill; OUT=$R/judge; TABLES=/games/ailab/corpora/talkB-tables/t28; export CUDA_VISIBLE_DEVICES=""
mkdir -p "$OUT"; cd "$SP"; exec >> "$OUT/judge_distill.log" 2>&1; umask 077
log() { echo "[$(date '+%F %T')] $*"; }
NICE=(nice -n 19 taskset -c 0-5,8-13)
until [ -f "$D/pilot.done" ]; do sleep 120; done
log "pilot.done seen"
[ -d "$TABLES" ] || { log "no tables at $TABLES"; exit 1; }
# (1) the pilot's table on the clean windows
if [ ! -f "$D/clean.txt" ]; then
  { echo "the pilot on the leak mask's clean windows ($(date '+%F %T'))"
    for set in kyle dlg; do
      name=$set; [ "$set" = dlg ] && name=dialogue
      files=(); pairs=()
      for arm in a_text b_slice c_a02 c_a05 c_a08 c_a10 d_mix; do [ -f "$D/k_${arm}_$set.json" ] && files+=("$D/k_${arm}_$set.json"); done
      for arm in c_a02 c_a05 c_a08 c_a10; do [ -f "$D/k_${arm}_$set.json" ] && [ -f "$D/k_b_slice_$set.json" ] && pairs+=("$D/k_${arm}_$set.json=$D/k_b_slice_$set.json"); done
      [ -f "$D/k_d_mix_$set.json" ] && [ -f "$D/k_a_text_$set.json" ] && pairs+=("$D/k_d_mix_$set.json=$D/k_a_text_$set.json")
      [ -f "$D/k_b_slice_$set.json" ] && [ -f "$D/k_a_text_$set.json" ] && pairs+=("$D/k_b_slice_$set.json=$D/k_a_text_$set.json")
      echo "== $name: levels"; [ ${#files[@]} -gt 0 ] && "${NICE[@]}" "$PY" "$C2/leak_table.py" --mask "$R/leak/leak_windows_talkB.json" --set "$name" "${files[@]}"
      echo "== $name: pairs"; [ ${#pairs[@]} -gt 0 ] && "${NICE[@]}" "$PY" "$C2/leak_table.py" --mask "$R/leak/leak_windows_talkB.json" --set "$name" "${pairs[@]}"
    done; } > "$D/clean.txt" 2>&1
  log "clean.txt written"
fi
# (2) the judges: the baseline's nine candidate sources, pinned
RUNS=("tables=$R/qwen-tables/qwen_memory.json:$R/qwen-tables/choices.json" "breadth=$R/qwen-breadth/qwen_memory.json:$R/qwen-breadth/choices.json" "dedup=$R/qwen-dedup/qwen_memory.json:$R/qwen-dedup/choices.json")
for arm in floor6 floor10 top1 dedup100 dedup500 dedupfrac; do RUNS+=("$arm=$SW/$arm/qwen_memory.json:$SW/$arm/choices.json"); done
log "the distilled readers as judges: ${#RUNS[@]} candidate sources beside the spans"
for name in b_slice c_a05 c_a02 c_a08 c_a10 a_text d_mix; do
  ckpt=$WT/results/ckpt/${name}_transformer.pt
  [ -f "$ckpt" ] || { log "no checkpoint for $name"; continue; }
  [ -f "$OUT/$name/judge.json" ] && { log "skip $name"; continue; }
  log "== $name"
  "${NICE[@]}" "$PY" "$C2/reader_judge.py" --ckpt "$ckpt" --tables "$TABLES" --out "$OUT/$name" --ref "$R/qwen/qwen_memory.json" --spans-choices "$R/qwen-dedup/choices.json" --runs "${RUNS[@]}" --second --threads 6 || log "$name failed"
  grep "^judge" "$OUT/$name/judge.txt" 2>/dev/null | tail -2 | cut -c1-200
done
touch "$OUT/distill.done"; log "the distilled judges done"
