#!/usr/bin/env bash
# Speed round, pit stop 1 on GPU 1 (between the handoff test and the timed reruns): where does a training step go, and what
# does each switch of spike 1 buy? Short runs of scripts/profile_step.py from this worktree; nothing here is a timed spine run.
# A script file, not an inline command under systemd-run: systemd expands ${var} in a unit's command line itself.
# usage: pitstop1.sh [unit to wait for]
# The caller picks the card by UUID: CUDA_VISIBLE_DEVICES hides the other card, so the visible one is cuda:0.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-heads
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
cd "$WT"
if [ -n "${1:-}" ]; then
  while systemctl --user is-active --quiet "$1"; do sleep 10; done
  sleep 15
fi
mkdir -p results/profile
exec > >(tee results/profile/pitstop1.log) 2>&1
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
echo "pit stop 1 start $(date '+%F %T'): $apps compute processes on the card; load $(cut -d' ' -f1-3 /proc/loadavg)"
[ "$apps" = 0 ] || { echo "the card is not free: stopping"; exit 1; }
COMMON="--task code --corpus-file /games/ailab/lab/flybrain-lab/corpora/linux_full.bytes --train-bytes 20000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --steps 16000 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter"
HYB="--arms mingru_sparse_post --gate --mem-every 2 --local-attn 512 --key-convs 2,4"
ZH="--arms mingru --local-attn 512"
TF="--arms transformer --rope"
ALL="--tf32 --head-amp --sparse-claws --scan doubling32 --fused"
run() {  # run <name> <seconds limit> <measure steps> <flags...>
  name=$1; lim=$2; m=$3; shift 3
  echo; echo "=== $name $(date +%T)"
  timeout "$lim" "$PY" scripts/profile_step.py $COMMON "$@" --warmup-steps 5 --measure-steps "$m" --json-out "results/profile/$name.json" > "results/profile/$name.md" 2>&1
  rc=$?
  echo "rc=$rc"; grep -m1 'plain step' "results/profile/$name.md" || tail -n 3 "results/profile/$name.md"
}
# 1. the three arms as they ran in the spine runs
run hyb_base 300 20 $HYB
run tf_base 200 20 $TF
run zh_base 200 20 $ZH
# 2. one switch at a time on the hybrid
run hyb_tf32 240 15 $HYB --tf32
run hyb_headamp 240 15 $HYB --head-amp
run hyb_claws 240 15 $HYB --sparse-claws
run hyb_scan32 240 15 $HYB --scan doubling32
run hyb_scanamp 240 15 $HYB --scan doubling
run hyb_detach 240 15 $HYB --detach-state
run hyb_fused 240 15 $HYB --fused
# 3. combinations
run hyb_all 240 15 $HYB $ALL
run hyb_all_detach 240 15 $HYB $ALL --detach-state
run hyb_all_2heads 240 15 --arms mingru_sparse_post --gate --mem-every 4 --local-attn 512 --key-convs 2,4 $ALL
# 4. the same generic switches on the other two arms, so the bar moves with them
run tf_tf32_fused 200 15 $TF --tf32 --fused
run zh_scan32_fused 200 15 $ZH --scan doubling32 --tf32 --fused
# 5. torch.compile last: a long first step, and it may fail
run hyb_all_compile 600 15 $HYB $ALL --compile
run tf_compile 400 15 $TF --tf32 --fused --compile
echo; echo "pit stop 1 done $(date '+%F %T'); load $(cut -d' ' -f1-3 /proc/loadavg)"
