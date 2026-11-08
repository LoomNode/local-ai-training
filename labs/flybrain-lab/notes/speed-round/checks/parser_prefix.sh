#!/usr/bin/env bash
# The parser test's leak check (parser_prefix.py): the pparse arm rescored on the 890 clean kernel windows with its node kind and
# depth (2001/2002) recomputed from prefix-only parses, and once more with 2001/2002 withheld (conf 0: the name completions alone).
# Against k_pparse (the full-parse features, what it trained on) and k_pbase. CPU for the features (niced, off the owner's cores), GPU 1
# for the two scorings (half a minute each, beside the big run). usage: parser_prefix.sh      log: results/tables/parser/parser_prefix.log
set -u -o pipefail
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; S=$SP/notes/speed-round/checks; WTT=/games/ailab/lab/flybrain-lab/.worktrees/term
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python; OUT=$SP/results/tables/parser; MEM=$SP/results/mem
C=/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes; WIN=$SP/results/tables/big/clean_kernel.txt
G1=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 1)
exec >> "$OUT/parser_prefix.log" 2>&1
echo "parser leak check start $(date '+%F %T')"
if [ ! -f "$MEM/feat_parser_prefix/features.json" ]; then
  CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 0-5,8-13 "$PY" "$S/parser_prefix.py" --feat "$MEM/feat_parser" --out "$MEM/feat_parser_prefix" \
    --corpus "$C" --offset 202117357 --T 2048 --n 1024 --windows "$WIN" --jobs 6 --check 64 --check-windows 8 || { echo "feature build failed"; exit 1; }
fi
if [ ! -f "$MEM/feat_parser_nokd/features.json" ]; then   # 2001/2002 withheld: conf 0 everywhere, pred 0; the rest linked
  mkdir -p "$MEM/feat_parser_nokd"
  "$PY" - "$MEM/feat_parser" "$MEM/feat_parser_nokd" <<'PY'
import json, os, sys
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
N = os.path.getsize(f"{src}/pred_n2001.npy") - 128
for kid in (2001, 2002):
    for kind in ("pred", "conf"):
        a = np.lib.format.open_memmap(f"{dst}/{kind}_n{kid}.npy", mode="w+", dtype=np.uint8, shape=(N,)); a.flush(); del a
for kid in (4, 8, 12, 16, 32, 64, 116, 132, 202, 203, 301, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 2003, 2004):
    for kind in ("pred", "conf"):
        d = f"{dst}/{kind}_n{kid}.npy"
        if not os.path.lexists(d):
            os.symlink(os.path.realpath(f"{src}/{kind}_n{kid}.npy"), d)
info = json.load(open(f"{src}/features.json")); info["note"] = info.get("note", "") + " | 2001/2002 withheld (conf 0)"
json.dump(info, open(f"{dst}/features.json", "w"), indent=2)
print("withheld dir written")
PY
fi
cd "$WTT"
for v in prefix nokd; do
  [ -f "$OUT/k_pparse_$v.json" ] && continue
  SCORE_CODE=$WTT CUDA_VISIBLE_DEVICES=$G1 "$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --windows "$WIN" --device cuda:0 \
    --out-dir "$OUT" "k_pparse_$v=results/ckpt/pparse_r_transformer.pt@$MEM/feat_parser_$v" || echo "scoring $v failed"
done
"$PY" - "$OUT" <<'PY'
import json, sys
d = sys.argv[1]
try:
    r = {w: json.load(open(f"{d}/k_{w}.json"))["bpb"] for w in ("pbase", "pparse", "pparse_prefix", "pparse_nokd")}
    print(f"clean kernel dev: base {r['pbase']:.4f}; parser arm on full-parse inputs {r['pparse']:.4f}, on prefix-only inputs {r['pparse_prefix']:.4f} "
          f"({r['pparse_prefix'] - r['pbase']:+.4f} vs base), with kind/depth withheld {r['pparse_nokd']:.4f} ({r['pparse_nokd'] - r['pbase']:+.4f})")
except Exception as e:
    print(f"no verdict yet: {e}")
PY
echo "parser leak check done $(date '+%F %T')"
