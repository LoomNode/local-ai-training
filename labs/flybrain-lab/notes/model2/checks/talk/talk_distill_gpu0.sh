#!/usr/bin/env bash
# The distillation pilot's chain on GPU 0 under the tender (2026-10-21; Kyle 2026-10-20: "it just saves time to use both cards"): the same
# chain as talk_distill_gpu.sh without the wait for the sweep (which holds GPU 1 until the evening). The tender starts it once the teacher
# pass's unit is gone: distill-tend2-gpu0 waits for pass.done, then gpu0_tend.sh tends unit talk-distill-gpu0 until pilot.done.
# usage (by the tender, CUDA_VISIBLE_DEVICES set): talk_distill_gpu0.sh
set -u
export WAIT_SWEEP=0
exec /games/ailab/lab/flybrain-lab/.worktrees/speed/notes/model2/checks/talk/talk_distill_gpu.sh
