#!/usr/bin/env bash
# The vocabulary arms' streams for model #2's sweep (plan Task 6 / spec "The sweep"): a 16k and a 64k BPE on the full train split, the
# pkgs1g corpus tokenized with each, and their 2^28 token features, so the vocabulary arms can run the moment tokens win at the hour.
# Waits for hour_corpus.sh's build to finish (its "all done" line) so the two never share the RAM. CPU only. log: results/tables/tok/sweep_corpus.log
set -euo pipefail
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; C2=$SP/notes/model2/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
LOG=$SP/results/tables/tok/sweep_corpus.log; BUILD=$SP/results/tables/tok/pkgs1g_build.log
RUN=(nice -n 19 taskset -c 0-5,8-13)
export CUDA_VISIBLE_DEVICES="" RAYON_NUM_THREADS=12
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
step() { local name=$1; shift; log "$name: start"; local t0=$SECONDS; if ! "${RUN[@]}" "$@" >> "$LOG" 2>&1; then log "$name: FAILED (see above)"; exit 1; fi; log "$name: done in $(( SECONDS - t0 )) s"; }
until grep -q "pkgs1g build: all done" "$BUILD" 2>/dev/null; do sleep 120; done
for V in 16384 65536; do
  tag=$((V / 1024))k; TOK=/games/ailab/corpora/pkgs-tok/bpe$tag.json; D=/games/ailab/corpora/pkgs1g-tok$tag
  [ -f "$TOK" ] && log "tokenizer $tag: exists" || step "tokenizer $tag" "$PY" "$C2/tokenizer_train.py" --vocab "$V" --out "$TOK"
  [ -f "$D/layout.json" ] && log "stream $tag: exists" || step "stream $tag" "$PY" "$C2/tokenize_corpus.py" --corpus /games/ailab/corpora/pkgs1g.bytes --layout /games/ailab/corpora/pkgs1g/layout.json --tokenizer "$TOK" --out "$D"
  [ -f "$D/feat28/features.json" ] && log "features $tag: exist" || step "features $tag" "$PY" "$C2/tok_tables.py" --stream "$D" --out "$D/feat28" --log2 28 --jobs 6
done
log "sweep corpus: all done"
