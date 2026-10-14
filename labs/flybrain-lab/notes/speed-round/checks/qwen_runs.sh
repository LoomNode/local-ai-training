#!/usr/bin/env bash
# The Qwen side of model #1's viability test, all four numbers on one card (about half an hour on a 3090): Qwen3.8-27B's bits/byte on
# the two held-out streams (qwen_baseline.sh: Voxelheim's GDScript, the lab's Python) and its line completions on the same spots our
# reader was scored on (qwen_complete.sh: 60 GDScript spots, 42 Python spots, seed 0). Outputs in results/tables/use/qwen_*.json; a
# job whose JSON exists is skipped, so a run cut short (the game guard stops this unit for a game) resumes with what is left.
# usage: CUDA_VISIBLE_DEVICES=<GPU 0 uuid> qwen_runs.sh        (run as a unit under probe_guard.sh: see qwen_orchestrate in the session)
set -u
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; T=/games/ailab/corpora/pkgs-tables; U=$SP/results/tables/use
mkdir -p "$U"
exec >> "$U/qwen_runs.log" 2>&1
echo "qwen runs start $(date '+%F %T') on $CUDA_VISIBLE_DEVICES"
[ -f "$U/qwen_vox_heldout.json" ] || bash "$S/qwen_baseline.sh" "$T/vox28/heldout.bytes" "$U/qwen_vox_heldout.json"
[ -f "$U/qwen_lab_heldout.json" ] || bash "$S/qwen_baseline.sh" "$T/lab28/heldout.bytes" "$U/qwen_lab_heldout.json"
[ -f "$U/qwen_vox_complete_gd.json" ] || bash "$S/qwen_complete.sh" "$T/vox28/heldout.bytes" "$U/qwen_vox_complete_gd.json" --spots 60 --ext gd --seed 0 --show 4
[ -f "$U/qwen_lab_complete_py.json" ] || bash "$S/qwen_complete.sh" "$T/lab28/heldout.bytes" "$U/qwen_lab_complete_py.json" --spots 60 --ext py --seed 0 --show 4
echo "qwen runs done $(date '+%F %T')"
for f in qwen_vox_heldout qwen_lab_heldout qwen_vox_complete_gd qwen_lab_complete_py; do
  [ -f "$U/$f.json" ] && python3 -c "
import json, sys; d = json.load(open(sys.argv[1]))
print(sys.argv[2], f\"bpb {d['bpb']:.4f}\" if 'bpb' in d else f\"exact {d['exact']:.0%}, first token {d['first_token']:.0%}, half or more {d['half_or_more']:.0%} ({d['n']} spots)\")" "$U/$f.json" "$f"
done
