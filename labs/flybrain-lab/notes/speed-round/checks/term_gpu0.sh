#!/usr/bin/env bash
# The terminal test, step 1, on GPU 0 under the game guard (Kyle, 2026-10-14: "Both 1 and 2 sound worth testing"): two hour arms of the
# 4-layer reader, identical in everything but the corpus (checks/term_corpus.py's pair: the same files, blocks and content bytes; the
# preamble area holds a terminal's output in `term` and the file's own previous bytes in `plain`), block-aligned windows
# (--aligned-windows, code .worktrees/term at b6a72d4 on spike/term), each fed its own stream's 2^28 features (term_tables.sh), the
# hour arm's calibrated steps (25,562: the 4-layer arm's hour, pre2_gpu1.sh), then scored on the 1,411 dev blocks (clean held-out
# files) with score_tables.py --score-from 768: the content bytes only, the preamble read but not scored. The question: is a terminal's
# output worth more to the reader than the same room filled with the file's own earlier text?
# usage: CUDA_VISIBLE_DEVICES=<GPU 0 uuid> UNIT=term-gpu0.service term_gpu0.sh      (arms: term plain; a guard stop resumes nothing:
#        an hour arm is rerun from scratch, as the queue's arms were; the claim files follow scale_common.sh's rule)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the UUID of GPU 0}"; : "${UNIT:?}"
STEPS_HOUR=${STEPS_HOUR:-25562}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
WTT=/games/ailab/lab/flybrain-lab/.worktrees/term
DIR=/games/ailab/corpora/term; MEMROOT=/vault/ailab/flybrain-mem/term
LAY=$DIR/layout.json
TRAIN_END=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["train_end"]))' "$LAY")
NDEV=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["dev_windows"]))' "$LAY")
PRE=$("$PY" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["pre"]))' "$LAY")
mkdir -p "$OUT/term" "$WTT/results"
cd "$WTT"
exec >> "$OUT/term/term_gpu0.log" 2>&1
echo "terminal test start $(date '+%F %T'): $STEPS_HOUR steps an arm, train_end $TRAIN_END, $NDEV dev windows, pre $PRE"
preflight
termarm() {  # termarm <term|plain>
  local which=$1 name=term_$1_r
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo "$UNIT" > "results/claim_$name"
  echo; echo "=== $name $(date +%T): $STEPS_HOUR steps (4 layers, aligned windows, corpus $DIR/$which.bytes)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON $TFI --mem-feat "$MEMROOT/$which/feat_all28" --mem-orders "$ALL18" --n-layers 4 \
    --corpus-file "$DIR/$which.bytes" --train-bytes "$TRAIN_END" --val-offset "$TRAIN_END" --val-bytes $((NDEV * 2048 + 1)) --aligned-windows \
    --steps "$STEPS_HOUR" --save --val-fixed 128 --eval-every 2000 2>&1 | grep --line-buffered -v "^  "
  rm -f "results/claim_$name"
}
termscore() {  # termscore <term|plain>: the dev blocks through the stream's own features, scored on the content
  local which=$1 name=term_$1_r
  [ -f "$OUT/term/k_term_$which.json" ] && return 0
  [ -f "results/ckpt/${name}_transformer.pt" ] || { echo "no checkpoint for $name"; return 0; }
  SCORE_CODE=$WTT "$PY" "$S/score_tables.py" --corpus "$DIR/$which.bytes" --offset "$TRAIN_END" --n "$NDEV" --T 2048 --score-from "$PRE" --device cuda:0 \
    --out-dir "$OUT/term" "k_term_$which=results/ckpt/${name}_transformer.pt@$MEMROOT/$which/feat_all28" || echo "scoring $name failed"
}
for w in term plain; do termarm "$w"; termscore "$w"; done
"$PY" - "$OUT/term" <<'PY'
import json, sys
d = sys.argv[1]
try:
    t, p = (json.load(open(f"{d}/k_term_{w}.json")) for w in ("term", "plain"))
    print(f"content bits/byte: term {t['bpb_from']:.4f}, plain {p['bpb_from']:.4f}, difference {t['bpb_from'] - p['bpb_from']:+.4f} "
          f"(whole window: {t['bpb']:.4f} / {p['bpb']:.4f}; {t['n']} dev blocks)")
except Exception as e:
    print(f"no verdict yet: {e}")
PY
echo "terminal test done $(date '+%F %T')"
