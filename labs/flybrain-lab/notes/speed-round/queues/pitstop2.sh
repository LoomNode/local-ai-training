#!/usr/bin/env bash
# Speed round, pit stop 2 on GPU 1: pit stop 1 found torch.compile takes the hybrid from 1.27 to 0.50 s/step with every other
# switch on (and the transformer from 0.36 to 0.28). Which switches matter under compile, and what do the other arms get?
# Short runs of scripts/profile_step.py from this worktree; nothing here is a timed spine run.
# The caller picks the card by UUID: CUDA_VISIBLE_DEVICES hides the other card, so the visible one is cuda:0.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-heads
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
cd "$WT"
mkdir -p results/profile
exec > >(tee results/profile/pitstop2.log) 2>&1
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
echo "pit stop 2 start $(date '+%F %T'): $apps compute processes on the card; load $(cut -d' ' -f1-3 /proc/loadavg)"
[ "$apps" = 0 ] || { echo "the card is not free: stopping"; exit 1; }
COMMON="--task code --corpus-file /games/ailab/lab/flybrain-lab/corpora/linux_full.bytes --train-bytes 20000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --steps 16000 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter"
HYB="--arms mingru_sparse_post --gate --mem-every 2 --local-attn 512 --key-convs 2,4"
HYB2="--arms mingru_sparse_post --gate --mem-every 4 --local-attn 512 --key-convs 2,4"
ZH="--arms mingru --local-attn 512"
TF="--arms transformer --rope"
ALL="--tf32 --head-amp --sparse-claws --scan doubling32 --fused"   # pit stop 1's combination: 1.268 eager, 0.500 compiled
run() {  # run <name> <seconds limit> <measure steps> <flags...>
  name=$1; lim=$2; m=$3; shift 3
  echo; echo "=== $name $(date +%T)"
  timeout "$lim" "$PY" scripts/profile_step.py $COMMON "$@" --warmup-steps 5 --measure-steps "$m" --json-out "results/profile/$name.json" > "results/profile/$name.md" 2>&1
  rc=$?
  echo "rc=$rc"; grep -m1 'plain step' "results/profile/$name.md" || tail -n 3 "results/profile/$name.md"
  grep -m1 'Warmup steps' "results/profile/$name.md"
}
# 1. compile alone, then one switch at a time on top of it (the hybrid)
run hyb_c 400 15 $HYB --compile
run hyb_c_d32 400 15 $HYB --compile --scan doubling32
run hyb_c_d 400 15 $HYB --compile --scan doubling
run hyb_c_headamp 400 15 $HYB --compile --head-amp
run hyb_c_claws 400 15 $HYB --compile --sparse-claws
run hyb_c_detach 400 15 $HYB --compile --detach-state
# 2. combinations under compile
run hyb_c_all_d 400 15 $HYB --tf32 --head-amp --sparse-claws --scan doubling --fused --compile
run hyb_c_all_detach 400 15 $HYB $ALL --compile --detach-state
run hyb_c_all_2heads 400 15 $HYB2 $ALL --compile
run hyb_c_all_chunk512 400 15 $HYB $ALL --compile --chunk 512
# 3. the other two arms under compile, so the bar moves with them
run zh_c 300 15 $ZH --compile
run zh_c_d32 300 15 $ZH --compile --scan doubling32 --tf32 --fused
run zh_c_d 300 15 $ZH --compile --scan doubling --tf32 --fused
run tf_c 300 15 $TF --compile
echo; echo "pit stop 2 done $(date '+%F %T'); load $(cut -d' ' -f1-3 /proc/loadavg)"
