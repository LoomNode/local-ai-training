# Handoff: assistant-scale 1B feasibility screen

## Goal

Test whether the current master-weight-free ratchet stack can train a low-billion-parameter
subword model toward a local-assistant-scale base model. This is a 5k feasibility gate, not an
assistant-quality claim.

## Current config

Use `configs/assistant_scale_1b_5k.toml`.

- Shape: 8 layers, width 3072, 24 heads, block size 256.
- Vocab: 8K BPE subword tokenizer on enwik8.
- Approximate weights: 0.96B total, ~0.91B matrix weights.
- Ratchet settings: 15 states, `rms_ema_beta=0.9`, ratcheted token embedding, compiled update.
- Training path: `matmul_mode = "int8"`, gradient checkpointing enabled.
- Batch size: 16 for the first screen. Reduce only if the run OOMs before step 1.

## Run command

GPU 0 is available for this screen unless the user says otherwise.

```bash
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/games/ailab/.uv-cache uv run lat train \
  --codes 15 --tokenizer subword --vocab-size 8000 --ratchet-embedding \
  --rms-ema-beta 0.9 --config configs/assistant_scale_1b_5k.toml --seed 1337 \
  --dataset-path /games/ailab/local-ai-training/data/enwik8/enwik8 \
  --output runs/assistant_scale_1b_5k
```

Generate from the latest checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/games/ailab/.uv-cache uv run lat generate \
  --checkpoint runs/assistant_scale_1b_5k/checkpoint \
  --prompt "[[History of " --max-new-tokens 300 --temperature 0.8
```

## Gate criteria

Promote to a longer run only if:

- the model fits and reaches at least step 200 without OOM;
- validation loss descends cleanly through 5k;
- checkpoint save and resume work at this scale;
- generation is not obviously broken;
- metrics report peak CUDA train bytes, tokens/sec, persistent bytes, and code movement;
- `lat audit` remains clean for the configured shape.

If it fails, preserve `runs/assistant_scale_1b_5k/` and record whether the blocker was memory,
throughput, non-finite loss, checkpointing, or generation.

## Boundary

Do not jump to 10B-20B training from here. The next step after a clean 5k is a 30k+ continuation
of the same recipe or a slightly larger 1-2B sibling screen.
