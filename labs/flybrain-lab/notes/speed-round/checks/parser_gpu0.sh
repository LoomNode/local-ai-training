#!/usr/bin/env bash
# The parser test on GPU 0 under the game guard (Kyle, 2026-10-14: "test a real parser too"): two hour arms of the 4-layer reader on
# the kernel corpus at the hour's calibrated steps (25,562), identical but for the inputs: `pbase` reads the 18 table inputs
# (feat_all28), `pparse` those plus the four parser pseudo-orders (checks/parser_features.py: node kind, node depth, defined-name
# completion, used-name completion; feat_parser links the 18 and adds its four). Both from .worktrees/term (b6a72d4 on spike/term).
# Scored on the 890 clean old-kernel dev windows (results/tables/big/clean_kernel.txt) through each arm's own features. The per-step work
# differs by four inputs through the fused channel (one embedding bag): the paces are logged and the ratio reported with the verdict.
# usage: CUDA_VISIBLE_DEVICES=<GPU 0 uuid> UNIT=parser-gpu0.service parser_gpu0.sh     (a guard stop reruns an arm from scratch)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the UUID of GPU 0}"; : "${UNIT:?}"
STEPS_HOUR=${STEPS_HOUR:-25562}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
WTT=/games/ailab/lab/flybrain-lab/.worktrees/term
FEAT_P=$SP/results/mem/feat_parser
PARSER_ORDERS=2001,2002,2003,2004
mkdir -p "$OUT/parser" "$WTT/results"
cd "$WTT"
exec >> "$OUT/parser/parser_gpu0.log" 2>&1
echo "parser test start $(date '+%F %T'): $STEPS_HOUR steps an arm"
[ -f "$FEAT_P/features.json" ] || { echo "no parser features at $FEAT_P"; exit 1; }
preflight
parm() {  # parm <pbase|pparse>
  local name=${1}_r feat orders
  if [ "$1" = pparse ]; then feat=$FEAT_P; orders=$ALL18,$PARSER_ORDERS; else feat=$MEM/feat_all28; orders=$ALL18; fi
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo "$UNIT" > "results/claim_$name"
  echo; echo "=== $name $(date +%T): $STEPS_HOUR steps (4 layers, inputs $orders)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON $TFI --mem-feat "$feat" --mem-orders "$orders" --n-layers 4 \
    --steps "$STEPS_HOUR" --save --val-fixed 128 --eval-every 2000 2>&1 | grep --line-buffered -v "^  "
  rm -f "results/claim_$name"
}
pscore() {  # pscore <pbase|pparse>: the clean kernel dev windows through the arm's own features
  local name=${1}_r feat
  if [ "$1" = pparse ]; then feat=$FEAT_P; else feat=$MEM/feat_all28; fi
  [ -f "$OUT/parser/k_$1.json" ] && return 0
  [ -f "results/ckpt/${name}_transformer.pt" ] || { echo "no checkpoint for $name"; return 0; }
  SCORE_CODE=$WTT "$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --windows "$OUT/big/clean_kernel.txt" --device cuda:0 \
    --out-dir "$OUT/parser" "k_$1=results/ckpt/${name}_transformer.pt@$feat" || echo "scoring $name failed"
}
for a in pbase pparse; do parm "$a"; pscore "$a"; done
"$PY" - "$OUT/parser" "$WTT/results" <<'PY'
import json, sys
d, r = sys.argv[1], sys.argv[2]
try:
    b, p = (json.load(open(f"{d}/k_{w}.json")) for w in ("pbase", "pparse"))
    secs = {w: json.load(open(f"{r}/{w}_r.json"))["arms"][0].get("train_seconds") for w in ("pbase", "pparse")}
    print(f"clean kernel dev: base {b['bpb']:.4f}, parser {p['bpb']:.4f}, difference {p['bpb'] - b['bpb']:+.4f} ({b['n']} windows); "
          f"train seconds base {secs['pbase']:.0f}, parser {secs['pparse']:.0f} (ratio {secs['pparse'] / secs['pbase']:.3f})")
except Exception as e:
    print(f"no verdict yet: {e}")
PY
echo "parser test done $(date '+%F %T')"
