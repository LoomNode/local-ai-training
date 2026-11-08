#!/usr/bin/env bash
# The as-of arms on GPU 0 under the tender, after the memory-reading reader (rag.done). usage (by the tender): talk_qwen_asof_gpu0.sh
set -u; export WAIT_FOR=/games/ailab/lab/flybrain-lab/.worktrees/speed/results/tables/rag/rag.done
exec /games/ailab/lab/flybrain-lab/.worktrees/speed/notes/model2/checks/talk/talk_qwen_asof_gpu.sh
