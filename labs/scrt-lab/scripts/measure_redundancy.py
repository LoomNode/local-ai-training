#!/usr/bin/env python3
"""Does FFN-neuron redundancy exist at this scale? Measure before building.

The consolidation machinery only ever operated on the residual bank (3.4% of the
model) and found nothing to merge: 192/200 cycles had no slot pair above sim 0.90.
FFN weights are 60.9% of the model — 18x the bank — and are what the origin thread
actually named as the target. Before porting merge/free/regrow onto MLP neurons, ask
whether the redundancy the whole thesis depends on is there at all.

Functional similarity, exactly:

    neuron i's contribution to the block output is  a_i(t) (x) w_i
        a_i in R^T  = gelu(fc(x))[:, i] over probe tokens
        w_i in R^C  = proj[:, i]

    <a_i (x) w_i, a_j (x) w_j>_F = <a_i, a_j> <w_i, w_j>

so cos of the contributions factorises as cos(a_i,a_j) * cos(w_i,w_j) — "responds to
the same inputs" times "writes the same direction". Activations are centred first,
because post-gelu values are largely non-negative and raw cosine between non-negative
vectors is inflated toward 1 for every pair.

An untrained model of the same shape is measured as a control: with 768 neurons and
finite probe tokens, chance-level max-similarity is not 0, and a "twin" count means
nothing without it.

    python scripts/measure_redundancy.py --steps 20000 --device cuda:1
    python scripts/measure_redundancy.py --checkpoint results_redund/baseline.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scrt.config import capacity_config, resolve_device
from scrt.data import TokenBatcher, build_corpus
from scrt.budget import build_matched_models, unwrap
from scrt.model import BaselineGPT
from scrt.train import train_arm


@torch.no_grad()
def collect_activations(
    model: BaselineGPT, batcher: TokenBatcher, n_batches: int
) -> list[torch.Tensor]:
    """Per block: (n_hidden, T) post-gelu activations over probe tokens."""
    acts: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(model.blocks))}
    handles = []

    def mk_hook(i: int):
        def hook(_mod, _inp, out):
            # out: (B, T, 4C) pre-gelu
            a = F.gelu(out).detach().reshape(-1, out.shape[-1])  # (B*T, 4C)
            acts[i].append(a.float().cpu())

        return hook

    for i, blk in enumerate(model.blocks):
        handles.append(blk.mlp.fc.register_forward_hook(mk_hook(i)))

    model.eval()
    for _ in range(n_batches):
        x, _y = batcher.sample()
        model(x)
    for h in handles:
        h.remove()

    return [torch.cat(acts[i], dim=0).T.contiguous() for i in sorted(acts)]


def pairwise_stats(A: torch.Tensor, W: torch.Tensor) -> dict:
    """A: (N, T) activations. W: (C, N) output directions (proj weight)."""
    Ac = A - A.mean(dim=1, keepdim=True)
    An = F.normalize(Ac, dim=1)
    Wn = F.normalize(W.T, dim=1)  # (N, C)

    act_sim = An @ An.T
    w_sim = Wn @ Wn.T
    func = act_sim * w_sim

    N = func.shape[0]
    eye = torch.eye(N, dtype=torch.bool)
    off = ~eye
    vals = func[off]
    absvals = vals.abs()

    # Per-neuron best partner — "does this neuron have a twin?"
    f = func.clone()
    f[eye] = -2.0
    best = f.max(dim=1).values

    q = [50, 90, 99, 99.9, 100]
    return {
        "n_neurons": N,
        "pair_pct": {
            str(x): float(torch.quantile(vals.float(), x / 100).item()) for x in q
        },
        "abs_pair_pct": {
            str(x): float(torch.quantile(absvals.float(), x / 100).item()) for x in q
        },
        "pairs_over": {
            str(t): int((absvals >= t).sum().item()) for t in (0.7, 0.8, 0.9, 0.95)
        },
        "n_pairs": int(absvals.numel() // 2),
        "neurons_with_twin_over": {
            str(t): int((best >= t).sum().item()) for t in (0.7, 0.8, 0.9, 0.95)
        },
        "best_partner_pct": {
            str(x): float(torch.quantile(best.float(), x / 100).item())
            for x in [50, 90, 100]
        },
        "mean_act_sim_abs": float(act_sim[off].abs().mean().item()),
        "mean_w_sim_abs": float(w_sim[off].abs().mean().item()),
    }


def report(tag: str, blocks: list[dict]) -> None:
    print(f"\n===== {tag}")
    for i, b in enumerate(blocks):
        print(
            f" block {i}: N={b['n_neurons']}  "
            f"|func| p50={b['abs_pair_pct']['50']:.3f} "
            f"p99={b['abs_pair_pct']['99']:.3f} "
            f"p99.9={b['abs_pair_pct']['99.9']:.3f} "
            f"max={b['abs_pair_pct']['100']:.3f}"
        )
        print(
            f"          pairs |func|>=0.7:{b['pairs_over']['0.7']:>7} "
            f">=0.8:{b['pairs_over']['0.8']:>6} "
            f">=0.9:{b['pairs_over']['0.9']:>5} "
            f">=0.95:{b['pairs_over']['0.95']:>4}   (of {b['n_pairs']} pairs)"
        )
        print(
            f"          neurons with a twin >=0.7:{b['neurons_with_twin_over']['0.7']:>4} "
            f">=0.8:{b['neurons_with_twin_over']['0.8']:>4} "
            f">=0.9:{b['neurons_with_twin_over']['0.9']:>4}  "
            f"| best-partner median={b['best_partner_pct']['50']:.3f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="FFN neuron redundancy measurement")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-batches", type=int, default=16)
    p.add_argument("--results-dir", default="results_redund")
    p.add_argument("--checkpoint", default=None, help="Skip training; load this .pt")
    p.add_argument(
        "--max-chars",
        type=int,
        default=11_000_000,
        help="Corpus size. Shrinking this at fixed model size raises params-per-token, "
             "i.e. drives the model from capacity-starved toward over-parameterised.",
    )
    args = p.parse_args()

    device = resolve_device(args.device)
    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.max_chars // 1000}k"
    ckpt_path = (
        Path(args.checkpoint) if args.checkpoint else outdir / f"baseline_{tag}.pt"
    )

    cfg = capacity_config(device=device, steps=args.steps)
    cfg.train.seed = args.seed
    cfg.train.results_dir = args.results_dir
    cfg.train.run_name = f"redund_{tag}"
    cfg.train.max_chars = args.max_chars

    # Build the matched baseline ourselves and hand it to train_arm, which trains
    # the object in place — that is how we keep a reference to the trained weights.
    torch.manual_seed(cfg.train.seed)
    matched = build_matched_models(replace(cfg.model, vocab_size=256), exact=True)
    holder, used_mcfg, _n = matched["baseline"]

    if ckpt_path.exists() and args.checkpoint:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        holder = holder.to(device)
        holder.load_state_dict(blob["state_dict"])
        print(f"loaded {ckpt_path} (best_eval={blob.get('best_eval')})")
    else:
        print(f"training baseline: {args.steps} steps on {device}")
        r = train_arm(
            cfg,
            arm="baseline",
            model=holder,
            write_results=True,
            eval_batches=20,
            final_eval_batches=50,
        )
        torch.save(
            {"state_dict": holder.state_dict(), "best_eval": r.best_eval_loss},
            ckpt_path,
        )
        print(f"saved {ckpt_path}  best_eval={r.best_eval_loss:.4f}")

    model = unwrap(holder)  # strip PaddedModel wrapper if present
    assert isinstance(model, BaselineGPT), type(model)

    corpus = build_corpus(
        kind=cfg.train.corpus,
        vocab_size=cfg.model.vocab_size,
        length=cfg.train.corpus_length,
        seed=cfg.train.seed,
        max_chars=cfg.train.max_chars,
        vocab_cap=cfg.train.char_vocab_cap,
    )
    split = int(corpus.data.numel() * 0.9)
    probe = TokenBatcher(
        corpus.data[split:],
        cfg.train.seq_len,
        cfg.train.batch_size,
        device=device,
        seed=cfg.train.seed + 7,
    )

    trained_acts = collect_activations(model, probe, args.probe_batches)
    trained = [
        pairwise_stats(A, model.blocks[i].mlp.proj.weight.detach().float().cpu())
        for i, A in enumerate(trained_acts)
    ]

    torch.manual_seed(args.seed + 999)
    ctrl_model = BaselineGPT(used_mcfg).to(device)
    ctrl_acts = collect_activations(ctrl_model, probe, args.probe_batches)
    ctrl = [
        pairwise_stats(A, ctrl_model.blocks[i].mlp.proj.weight.detach().float().cpu())
        for i, A in enumerate(ctrl_acts)
    ]

    report(f"TRAINED (max_chars={args.max_chars:,}, "
           f"{1452672/args.max_chars:.2f} params/char)", trained)
    report("UNTRAINED CONTROL (chance level)", ctrl)

    out = outdir / f"redundancy_{tag}.json"
    out.write_text(
        json.dumps(
            {
                "max_chars": args.max_chars,
                "params_per_char": 1452672 / args.max_chars,
                "trained": trained,
                "control": ctrl,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
