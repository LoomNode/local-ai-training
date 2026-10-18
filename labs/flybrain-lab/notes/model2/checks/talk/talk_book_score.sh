#!/usr/bin/env bash
# The right book (test 1 of the weights' pile, 2026-10-18), no training: the hour reader th_bytes_r reads Kyle's turns and the dialogue
# through the prose corpus's saved tables alone (t28: the baseline, which must reproduce k_th_bytes_kyle's cold number: the exactness
# check) and through the copy of those tables with Kyle's world indexed in (kyle/index, from ~/world.sh). sample_tables.py
# --score-windows computes every window's inputs from the text and the tables (the use layer's path), so the book's contexts answer
# where the corpus's did not. Kyle's turns: all 340 windows; the dialogue: every fourth of its 2,047 (512).
# usage: CUDA_VISIBLE_DEVICES=<uuid> talk_book_score.sh   (waits for kyle/index/index.done; log results/tables/talk/book.log; results book_*.json; line in jobs.txt)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?}"
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; WT=/games/ailab/lab/flybrain-lab/.worktrees/tok; CK=$SP/notes/speed-round/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python; OUT=$SP/results/tables/talk
CKPT=$WT/results/ckpt/th_bytes_r_transformer.pt; KD=/games/ailab/corpora/kyle
TAIL=$("$PY" -c 'import json; print(int(json.load(open("/games/ailab/corpora/talk/layout.json"))["tail_offset"]))')
DOFF=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["dialogue"]["offset"])')
NWIN=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["kyle"]["windows"])')
DWIN=$("$PY" -c 'import json; print(json.load(open("/games/ailab/corpora/kyle/heldout.json"))["dialogue"]["windows"])')
KW=$(seq -s, 0 $((NWIN - 1))); DW=$(seq -s, 0 4 $((DWIN - 1)))
export SCORE_CODE=$WT
cd "$SP"; exec >> "$OUT/book.log" 2>&1
echo "the right book: start $(date '+%F %T') on $CUDA_VISIBLE_DEVICES"
until [ -f "$KD/index/index.done" ]; do sleep 120; done
score() {  # score <tables> <label> <set> <offset> <windows csv>
  local tables=$1 label=$2 set=$3 off=$4 wins=$5; local o="$OUT/book_${label}_$set.json"
  [ -f "$o" ] && return 0
  echo "== $label on $set $(date +%T)"
  "$PY" "$CK/sample_tables.py" --ckpt "$CKPT" --tables "$tables" --score-windows "$wins" --offset "$off" --T 2048 --device cuda:0 2>&1 | grep "^window " > "$o.txt" || { echo "scoring failed ($label, $set)"; return 1; }
  "$PY" - "$o.txt" "$o" "$label" "$set" "$tables" <<'PY'
import json, sys, statistics
txt, out, label, s, tables = sys.argv[1:6]
per = {int(l.split()[1][:-1]): float(l.split()[2]) for l in open(txt) if l.startswith("window ")}
json.dump({"label": label, "set": s, "tables": tables, "n": len(per), "bpb": statistics.mean(per.values()), "per_window_cold": per}, open(out, "w"))
print(f"{label} on {s}: {len(per)} windows, {statistics.mean(per.values()):.4f} bits/byte (cold)")
PY
}
score /games/ailab/corpora/talk-tables/t28 base kyle "$TAIL" "$KW"
score "$KD/index" book kyle "$TAIL" "$KW"
score /games/ailab/corpora/talk-tables/t28 base dlg $((TAIL + DOFF)) "$DW"
score "$KD/index" book dlg $((TAIL + DOFF)) "$DW"
"$PY" - "$OUT" <<'PY' | tee -a "$OUT/jobs.txt"
import json, os, sys, statistics
d = sys.argv[1]
def load(p): return json.load(open(p)) if os.path.exists(p) else None
bk, bb, dk, db = (load(f"{d}/book_{l}_{s}.json") for l, s in (("base", "kyle"), ("book", "kyle"), ("base", "dlg"), ("book", "dlg")))
ref = load(f"{d}/k_th_bytes_kyle.json"); refd = load(f"{d}/k_th_bytes_dlg.json")
if bk and ref:
    pw = ref.get("per_window_cold"); diff = max(abs(bk["per_window_cold"][i] - pw[i]) for i in range(len(pw))) if pw and len(pw) == bk["n"] else None
    print(f"exactness: th_bytes_r on Kyle's turns through t28 {bk['bpb']:.4f} vs the stored features' {ref['bpb']:.4f}" + (f" (max per-window difference {diff:.4f})" if diff is not None else ""))
if bk and bb:
    print(f"the book (Kyle's world indexed into t28, no training) vs the corpus tables alone: Kyle's turns ({bk['n']} windows) cold {bb['bpb']:.4f} vs {bk['bpb']:.4f} ({bb['bpb'] - bk['bpb']:+.4f})"
          + (f"; the dialogue ({dk['n']} windows, every fourth) cold {db['bpb']:.4f} vs {dk['bpb']:.4f} ({db['bpb'] - dk['bpb']:+.4f})" if dk and db else ""))
PY
touch "$OUT/book.done"; echo "the right book: done $(date '+%F %T')"
