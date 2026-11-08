#!/usr/bin/env bash
# Qwen as the yardstick, pointed at the memory (Kyle, 2026-10-19: "go ahead and run the qwen tests"): qwen_memory.py scores the second
# half of every 2 KB held-out window given its first half, alone and with Kyle's files retrieved by the first half (Qwen3.8-27B Q4_K_XL
# on GPU 1 through llama-cpp-python, .venv-qwen); then the lab's readers on the same bytes: th_bytes_r (the hour reader) and th_combo_r
# (4 KB, fine-tuned on the world) dumped on the same 2 KB windows of heldout.bytes through the tables alone and through the files-only
# book (heldout-feat-plain / -book, from talk_combo_gpu.sh), their second-half costs beside Qwen's.
# usage: CUDA_VISIBLE_DEVICES=<uuid> talk_qwen_gpu.sh   (log results/tables/qwen/qwen.log; lines qwen/qwen_memory.txt + talk/leads.txt; done file qwen/qwen.done)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?}"
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok; C2=$SP/notes/model2/checks; CK=$SP/notes/speed-round/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python; PYQ=/games/ailab/lab/flybrain-lab/.venv-qwen/bin/python
NV=/games/ailab/lab/flybrain-lab/.venv-qwen/lib/python3.12/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cuda_runtime/lib:$NV/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}   # the cu12 wheel's runtime (no toolkit on the box)
OUT=$SP/results/tables/qwen; TOUT=$SP/results/tables/talk; KD=/games/ailab/corpora/kyle; KDUMP=$KD/dumps; HELD=$KD/heldout.bytes
MODEL=/vault/ailab/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf
mkdir -p "$OUT"; cd "$SP"; exec >> "$OUT/qwen.log" 2>&1
echo "Qwen + the memory: start $(date '+%F %T') on $CUDA_VISIBLE_DEVICES"
until [ -f "$OUT/venv.done" ]; do sleep 30; done
idle() { while :; do local apps used; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$CUDA_VISIBLE_DEVICES"); used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '); [ "$apps" = 0 ] && [ -n "$used" ] && [ "$used" -lt 2500 ] && return 0; sleep 60; done; }
NWIN=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["kyle"]["windows"])')
DOFF=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["dialogue"]["offset"])')
DWIN=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["dialogue"]["windows"])')
# 1. Qwen: a smoke of 3 windows a set, then the full run
if [ ! -f "$OUT/qwen_memory.json" ]; then
  idle; echo "== smoke $(date +%T)"; ( umask 077; "$PYQ" "$C2/qwen_memory.py" --model "$MODEL" --out "$OUT/smoke" --limit 3 ) || { echo "smoke failed"; exit 1; }
  idle; echo "== full run $(date +%T)"; ( umask 077; "$PYQ" "$C2/qwen_memory.py" --model "$MODEL" --out "$OUT" ) || { echo "full run failed"; exit 1; }
fi
# 2. the readers on the same windows (the per-byte dumps: the second half's mean = warm on these bytes)
dump() { local name=$1 feat=$2 tag=$3 off=$4 n=$5 wins=$6; [ -f "$KDUMP/${tag}_w0.json" ] && return 0; idle
  ( umask 077; SCORE_CODE=$WT "$PY" "$CK/dump_window.py" --corpus "$HELD" --offset "$off" --n "$n" --T 2048 --windows "$wins" --feat "$feat" --ckpt "$WT/results/ckpt/${name}_transformer.pt" --name "$tag" --out-dir "$KDUMP" --device cuda:0 | grep -c window ); }
KW=$(seq -s, 0 $((NWIN - 1))); DW=$(seq -s, 0 4 $((DWIN - 1)))
for m in th_bytes_r:qb th_combo_r:qc; do name=${m%%:*}; tag=${m#*:}
  dump $name "$KD/heldout-feat-plain" "${tag}_plain_kyle" 0 "$NWIN" "$KW";  dump $name "$KD/heldout-feat-book" "${tag}_book_kyle" 0 "$NWIN" "$KW"
  dump $name "$KD/heldout-feat-plain" "${tag}_plain_dlg" "$DOFF" "$DWIN" "$DW"; dump $name "$KD/heldout-feat-book" "${tag}_book_dlg" "$DOFF" "$DWIN" "$DW"
done
"$PY" - "$KDUMP" "$OUT" <<'PY' | tee -a "$OUT/qwen_memory.txt" | tee -a "$TOUT/leads.txt"
import json, glob, sys
d, out = sys.argv[1:3]; q = json.load(open(f"{out}/qwen_memory.json"))
def half(tag):
    ps = glob.glob(f"{d}/{tag}_w*.json"); v = [sum(json.load(open(p))["cost"]["full"][1024:]) / 1024 for p in ps]; return sum(v) / len(v), len(v)
for s, name in (("kyle", "Kyle's turns"), ("dlg", "the dialogue")):
    rows = [(f"the hour reader, tables alone", half(f"qb_plain_{s}")[0]), ("the hour reader through the book", half(f"qb_book_{s}")[0]),
            ("the 4 KB reader fine-tuned on the world, tables alone", half(f"qc_plain_{s}")[0]), ("the same through the book", half(f"qc_book_{s}")[0]),
            ("Qwen alone", q[s]["warm"]["bpb"]), ("Qwen + the files retrieved", q[s]["memory"]["bpb"])]
    print(f"the second half of 2 KB windows of {name} given the first ({q[s]['warm']['n']} windows): " + "; ".join(f"{w} {v:.4f}" for w, v in rows))
PY
touch "$OUT/qwen.done"; echo "done $(date '+%F %T')"
