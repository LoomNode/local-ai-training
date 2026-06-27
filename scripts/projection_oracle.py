"""Projection-oracle diagnostic: separate the representation floor from the optimization gap.

Given a converged FP32 checkpoint, measure three validation losses at one config/seed:
  - FP32                : the ceiling (best-case continuous weights)
  - Projection (PTQ)    : FP32 weights projected to the best per-row quinary codes, embeddings
                          and RMSNorm kept FP -> the representation floor / PTQ baseline
  - (ratchet-trained)   : supplied separately from a `lat train` ratchet run's metrics

Two gaps fall out: FP32->Projection = representation cost of quantizing; Projection->ratchet
= the optimization gap (negative => master-weight-free training BEATS PTQ at equal bits).

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/projection_oracle.py \
    --fp32-checkpoint runs/po-fp32/checkpoint --dataset-path data/text8/text8 \
    --config configs/scaleup_text8_25m_5k.toml --max-code 2 [--ratchet-val 1.31]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from local_ai_training.checkpoint import load_checkpoint
from local_ai_training.config import ExperimentConfig
from local_ai_training.data import build_char_corpus, make_batch_schedule
from local_ai_training.model import build_seeded_model
from local_ai_training.projection import project_to_codes
from local_ai_training.ratchet import DiscreteRatchetLinear, pack_code_pressure
from local_ai_training.train import evaluate


def _corpus(dataset_path: Path):
    return build_char_corpus(Path(dataset_path).read_text(encoding="utf-8"))


def _eval(model, corpus, config, device, seed):
    schedule = make_batch_schedule(
        data_length=corpus.validation_ids.numel(), steps=config.eval_batches,
        batch_size=config.batch_size, block_size=config.block_size, seed=seed + 20_000,
    )
    return evaluate(model, corpus.validation_ids, schedule,
                    block_size=config.block_size, device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32-checkpoint", required=True, type=Path)
    ap.add_argument("--dataset-path", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--max-code", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ratchet-val", type=float, default=None,
                    help="ratchet-trained val loss (from a lat train run) for the gap report")
    ap.add_argument("--output", type=Path, default=Path("runs/projection-oracle/result.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ExperimentConfig.from_toml(args.config)
    corpus = _corpus(args.dataset_path)
    mc = config.model_config(vocab_size=len(corpus.vocabulary))

    # FP32 model: load the checkpoint, evaluate the ceiling.
    fp32 = build_seeded_model(mc, max_code=None, seed=args.seed).to(device)
    fp32_opt = torch.optim.AdamW(fp32.parameters(), lr=0.0)
    # The FP32 control serializes max_code as 0 (the no-codes sentinel), not None.
    load_checkpoint(args.fp32_checkpoint, model=fp32, optimizer=fp32_opt,
                    expected_max_code=0, expected_vocabulary=corpus.vocabulary,
                    expected_matmul_mode=config.matmul_mode)
    fp32_val = _eval(fp32, corpus, config, device, args.seed)

    # Projected model: same architecture as a ratchet, but each matrix is the best PTQ of the
    # FP32 weight (codes frozen) and embeddings/RMSNorm are copied from FP32 (PTQ keeps them FP).
    ratchet = build_seeded_model(mc, max_code=args.max_code, seed=args.seed).to(device)
    fp32_mods = dict(fp32.named_modules())
    projected = 0
    for name, module in ratchet.named_modules():
        if isinstance(module, DiscreteRatchetLinear):
            src = fp32_mods[name]  # the same-named nn.Linear in the FP32 model
            code, scale = project_to_codes(src.weight.to(device), args.max_code)
            zeros = torch.zeros_like(code)
            module.packed.copy_(pack_code_pressure(code, zeros, args.max_code))
            module._scale.copy_(scale.to(device))
            projected += 1
    # Copy the trained FP support params (token/pos embeddings, RMSNorm, lm_head) from FP32.
    ratchet_state = ratchet.state_dict()
    for key, tensor in fp32.state_dict().items():
        if key in ratchet_state and ratchet_state[key].shape == tensor.shape \
                and "packed" not in key and "_scale" not in key:
            ratchet_state[key].copy_(tensor.to(device))
    ratchet.load_state_dict(ratchet_state, strict=True)
    projection_val = _eval(ratchet, corpus, config, device, args.seed)

    out = {
        "config": str(args.config), "max_code": args.max_code, "seed": args.seed,
        "matrices_projected": projected,
        "fp32_val": fp32_val,
        "projection_ptq_val": projection_val,
        "ratchet_trained_val": args.ratchet_val,
        "representation_cost_fp32_to_ptq": projection_val - fp32_val,
        "optimization_gap_ptq_to_ratchet": (
            None if args.ratchet_val is None else args.ratchet_val - projection_val
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
