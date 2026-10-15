#!/usr/bin/env bash
# The use-layer evaluation of a model #1 checkpoint (the viability test's own side, SPEED.md "The use layer, second pass", repeated for
# the final model): bits/byte on the two held-out streams (Voxelheim's GDScript, the lab's Python) through the project's index (vox28,
# lab28), the corpus's 2^28 tables (t28) and its 2^30 tables (t30, memory-mapped: slow and RAM-hungry the first time), and the line
# completions on the same 60 GDScript and 42 Python spots (seed 0, greedy). GPU 1 (CUDA_VISIBLE_DEVICES), about 20 minutes.
# usage: CUDA_VISIBLE_DEVICES=<GPU 1 uuid> final_eval.sh <checkpoint> <tag>     -> results/tables/use/{vox,lab}_heldout_<tag>_<tables>.json,
#        {vox,lab}_complete_{gd,py}_<tag>_<index>.json; log results/tables/use/final_eval.log
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the UUID of GPU 1}"
CK=${1:?usage: final_eval.sh <checkpoint> <tag>}; TAG=${2:?}
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; WT1=/games/ailab/lab/flybrain-lab/.worktrees/big1
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python; T=/games/ailab/corpora/pkgs-tables; U=$SP/results/tables/use
mkdir -p "$U"
exec >> "$U/final_eval.log" 2>&1
echo "final eval of $CK as $TAG, start $(date '+%F %T')"
cd "$SP"
for s in vox lab; do
  for t in ${s}28 t28 t30; do
    out=$U/${s}_heldout_${TAG}_$t.json
    [ -f "$out" ] && continue
    SCORE_CODE=$WT1 nice -n 19 "$PY" "$S/sample_tables.py" --ckpt "$CK" --tables "$T/$t" --score-file "$T/${s}28/heldout.bytes" --device cuda:0 --out "$out" || echo "scoring $s through $t failed"
  done
done
[ -f "$U/vox_complete_gd_${TAG}_vox28.json" ] || SCORE_CODE=$WT1 nice -n 19 "$PY" "$S/complete_test.py" --ckpt "$CK" --tables "$T/vox28" --stream "$T/vox28/heldout.bytes" --spots 60 --ext gd --seed 0 --device cuda:0 --show 4 --out "$U/vox_complete_gd_${TAG}_vox28.json" || echo "vox completion failed"
[ -f "$U/lab_complete_py_${TAG}_lab28.json" ] || SCORE_CODE=$WT1 nice -n 19 "$PY" "$S/complete_test.py" --ckpt "$CK" --tables "$T/lab28" --stream "$T/lab28/heldout.bytes" --spots 60 --ext py --seed 0 --device cuda:0 --show 4 --out "$U/lab_complete_py_${TAG}_lab28.json" || echo "lab completion failed"
"$PY" - "$U" "$TAG" <<'PY'
import json, os, sys
u, tag = sys.argv[1], sys.argv[2]
def bpb(f):
    p = f"{u}/{f}"; return f"{json.load(open(p))['bpb']:.4f}" if os.path.exists(p) else "-"
def comp(f):
    p = f"{u}/{f}"
    if not os.path.exists(p): return "-"
    d = json.load(open(p)); return f"{d['exact']:.0%} / {d['first_token']:.0%} / {d['half_or_more']:.0%}"
print(f"{tag}: Voxelheim held-out bits/byte index {bpb(f'vox_heldout_{tag}_vox28.json')}, t28 {bpb(f'vox_heldout_{tag}_t28.json')}, t30 {bpb(f'vox_heldout_{tag}_t30.json')}; "
      f"lab Python index {bpb(f'lab_heldout_{tag}_lab28.json')}, t28 {bpb(f'lab_heldout_{tag}_t28.json')}, t30 {bpb(f'lab_heldout_{tag}_t30.json')}; "
      f"completion gd {comp(f'vox_complete_gd_{tag}_vox28.json')}, py {comp(f'lab_complete_py_{tag}_lab28.json')} (exact / first token / half)")
PY
echo "final eval of $TAG done $(date '+%F %T')"
