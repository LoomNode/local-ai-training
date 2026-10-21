#!/usr/bin/env bash
# The product end to end (Kyle, 2026-10-21: "we can queue 3 up"): qwen_reply.py on GPU 1 after the sweep's combinations (waits for
# qwen-sweep/combo.done): 24 turns of the held-out dialogue, the local Qwen replying with and without the labelled memory of his files,
# written for Kyle to read (results/tables/product/replies.md, 0600, never committed). usage: CUDA_VISIBLE_DEVICES=<uuid> talk_product_gpu.sh
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?}"
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; C2=$SP/notes/model2/checks; PYQ=/games/ailab/lab/flybrain-lab/.venv-qwen/bin/python
NV=/games/ailab/lab/flybrain-lab/.venv-qwen/lib/python3.12/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cuda_runtime/lib:$NV/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
OUT=$SP/results/tables/product; mkdir -p "$OUT"; chmod 700 "$OUT"; cd "$SP"; exec >> "$OUT/product.log" 2>&1
echo "[$(date '+%F %T')] the product test: waiting for the combinations (qwen-sweep/combo.done) on $CUDA_VISIBLE_DEVICES"
until [ -f "$SP/results/tables/qwen-sweep/combo.done" ]; do sleep 300; done
idle() { while :; do local apps used; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$CUDA_VISIBLE_DEVICES"); used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ "${apps:-0}" -eq 0 ] && [ "${used:-99999}" -lt 2500 ] && return 0; sleep 60; done; }
idle; echo "[$(date '+%F %T')] generating"
( umask 077; "$PYQ" "$C2/qwen_reply.py" --out "$OUT" ) && touch "$OUT/product.done"
echo "[$(date '+%F %T')] product done"
