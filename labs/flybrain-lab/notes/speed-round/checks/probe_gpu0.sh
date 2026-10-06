#!/usr/bin/env bash
# Tables follow-up: the two probes' scoring jobs on GPU 0 (Kyle's card; he said "use gpu when available"). A short queue: about 32
# scorings of 1,024 windows, one minute each. Pre-flight refuses a card with another compute process. probe_guard.sh stops this
# unit if a foreign compute process or a large graphics client (a game) appears. Resumable: a job whose JSON exists is skipped.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
S=/tmp/claude-1000/-games-ailab/f5e9946e-97cc-48fe-9eb3-0b019140c726/scratchpad/tables
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed
CK=/games/ailab/lab/flybrain-lab/results/ckpt
OUT=$SP/results/tables
MEM=$SP/results/mem
CPY=$SP/results/mem_cpython
cd /games/ailab/lab/flybrain-lab
exec > >(tee -a "$OUT/gpu0.log") 2>&1
echo "tables probes start $(date '+%F %T')"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
[ "$apps" = 0 ] || { echo "pre-flight failed: $apps compute processes on the card"; exit 1; }
TF_TAB=$CK/speed_t2_tf_tab_c_3600_s0_transformer.pt
ZH_TAB=$CK/speed_t2_zh_tab_c_3600_s0_mingru.pt
TF_MT=$CK/speed_t2_tf_match_tab_c_3600_s0_transformer.pt
ZH_MT=$CK/speed_t2_zh_match_tab_c_3600_s0_mingru.pt
jobs1=()
for cond in feat feat_ext feat_ext_noevict; do
  jobs1+=("p1_tf_tab_$cond=$TF_TAB@$MEM/$cond" "p1_zh_tab_$cond=$ZH_TAB@$MEM/$cond" "p1_tf_mt_$cond=$TF_MT@$MEM/$cond" "p1_zh_mt_$cond=$ZH_MT@$MEM/$cond")
done
echo "=== probe 1: kernel dev windows, the table extended after training $(date +%T)"
"$PY" "$S/score_tables.py" --corpus corpora/linux_full.bytes --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "${jobs1[@]}"
jobs2=("p2_tf_r=$CK/speed_tf_r_3600_s0_transformer.pt" "p2_tf_c=$CK/speed_tf_c_3600_s0_transformer.pt" "p2_zh_c=$CK/speed_zh_c_3600_s0_mingru.pt" "p2_hyb_c=$CK/speed_hyb_c_3600_s0_mingru_sparse_post.pt")
for cond in none kernel own merged; do
  jobs2+=("p2_tf_tab_$cond=$TF_TAB@$CPY/$cond" "p2_zh_tab_$cond=$ZH_TAB@$CPY/$cond" "p2_tf_mt_$cond=$TF_MT@$CPY/$cond" "p2_zh_mt_$cond=$ZH_MT@$CPY/$cond")
done
echo "=== probe 2: CPython windows, four tables $(date +%T)"
"$PY" "$S/score_tables.py" --corpus corpora/cpython.bytes --offset 40000000 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "${jobs2[@]}"
echo "tables probes done $(date '+%F %T')"
