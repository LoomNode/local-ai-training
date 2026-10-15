#!/usr/bin/env bash
# The parser test's honest arm (after the leak check, parser_prefix.sh: the full-parse node kind and depth at byte p-1 encode the bytes
# after p; on prefix-only inputs the pparse arm scores worse than the base): the same hour arm with the two prefix-only parser inputs
# alone, 2003 (defined-name completion) and 2004 (used-name completion), which are functions of the earlier bytes by construction.
# Against k_pbase on the 890 clean kernel windows. usage: as parser_gpu0.sh (CUDA_VISIBLE_DEVICES=<GPU 0 uuid> UNIT=pnames-gpu0.service)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the UUID of GPU 0}"; : "${UNIT:?}"
STEPS_HOUR=${STEPS_HOUR:-25562}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
WTT=/games/ailab/lab/flybrain-lab/.worktrees/term
FEAT_P=$SP/results/mem/feat_parser
mkdir -p "$OUT/parser" "$WTT/results"
cd "$WTT"
exec >> "$OUT/parser/parser_gpu0.log" 2>&1
echo "parser names arm start $(date '+%F %T'): $STEPS_HOUR steps"
preflight
name=pnames_r
if [ -f "results/$name.json" ]; then echo "skip $name"; else
  echo "$UNIT" > "results/claim_$name"
  echo; echo "=== $name $(date +%T): $STEPS_HOUR steps (4 layers, inputs $ALL18,2003,2004)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON $TFI --mem-feat "$FEAT_P" --mem-orders "$ALL18,2003,2004" --n-layers 4 \
    --steps "$STEPS_HOUR" --save --val-fixed 128 --eval-every 2000 2>&1 | grep --line-buffered -v "^  "
  rm -f "results/claim_$name"
fi
if [ ! -f "$OUT/parser/k_pnames.json" ] && [ -f "results/ckpt/${name}_transformer.pt" ]; then
  SCORE_CODE=$WTT "$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --windows "$OUT/big/clean_kernel.txt" --device cuda:0 \
    --out-dir "$OUT/parser" "k_pnames=results/ckpt/${name}_transformer.pt@$FEAT_P" || echo "scoring $name failed"
fi
"$PY" - "$OUT/parser" "$WTT/results" <<'PY'
import json, sys
d, r = sys.argv[1], sys.argv[2]
try:
    b, p = (json.load(open(f"{d}/k_{w}.json")) for w in ("pbase", "pnames"))
    secs = {w: json.load(open(f"{r}/{w}_r.json"))["arms"][0].get("train_seconds") for w in ("pbase", "pnames")}
    print(f"clean kernel dev: base {b['bpb']:.4f}, names {p['bpb']:.4f}, difference {p['bpb'] - b['bpb']:+.4f} ({b['n']} windows); "
          f"train seconds base {secs['pbase']:.0f}, names {secs['pnames']:.0f} (ratio {secs['pnames'] / secs['pbase']:.3f})")
except Exception as e:
    print(f"no verdict yet: {e}")
PY
echo "parser names arm done $(date '+%F %T')"
