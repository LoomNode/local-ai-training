# seqmoe-lab

Sequential Mixture-of-Experts training under limited VRAM.

**Hypothesis:** two expert FFNs, each trained alone against the same frozen trunk, can be assembled afterward and routed — so total specialist capacity is a storage problem, while training memory tracks one expert.

This lab copies small pieces from `scrt-lab` (GPT block, AdamW loop, JSON results, GPU 1), `envelope-lab` (fail-closed 22 GB VRAM envelope), and Colibri (logical expert ids). It does **not** import those trees, and it does **not** implement NVMe paging.

- Spec: [`docs/superpowers/specs/2026-08-30-sequential-moe-poc-design.md`](docs/superpowers/specs/2026-08-30-sequential-moe-poc-design.md)
- Default research device: `cuda:1` (bare `--device cuda` maps there)
- Swap is not capacity. A memory plan that would exceed the envelope raises; CUDA OOM is not retried.

## Setup

```bash
cd /games/ailab/lab/seqmoe-lab
uv sync --extra dev
```

## How to run

```bash
# unit tests (CPU, no network)
pytest -q

# tiny PoC: synth code/math/fill, 20/20/20 steps, CPU (or --device cuda:1)
python scripts/run_poc.py --tiny
python scripts/run_poc.py --tiny --device cuda:1 --run-name poc_tiny_cuda
python scripts/run_poc.py --tiny-seeds    # CPU: same mix, 3 random FFN inits
python scripts/run_poc.py --tiny-skills   # CPU: cloned experts for code/math/fill
python scripts/run_poc.py --dump-skills --device cuda:1   # 3 cloned skills, scientific steps
python scripts/run_poc.py --dump-seeds --device cuda:1    # 3 random seeds, same dumps/budget

# sequential vs joint optimizer/VRAM comparison
python scripts/run_memory.py --device cpu
python scripts/run_memory.py --device cuda:1

# sequential vs joint at scientific model size (few steps)
python scripts/run_memory.py --scientific --device cuda:1

# scientific corpora (fails closed if wiki is missing — no synth fallback)
python scripts/prepare.py --fetch    # GSM8K + CPython dump; sidecar dump only if obtained
python scripts/prepare.py
python scripts/run_poc.py --scientific --device cuda:1
```

Wiki path: `$SEQMOE_WIKI` or `/games/ailab/lab/scrt-lab/data/wiki.train.raw`.

## Reading results

JSON lands in `results/{run_name}_comparison.json` plus per-arm files. Every arm reports `nll` (nats/token) and `bpb` (bits/byte) on the same holdouts.

| Arm | What it is |
|-----|------------|
| `trunk` | Frozen dense pretrain |
| `dense_ft_mix` | Trunk MLP fine-tuned `expert_steps` on mixed code+math |
| `dense_ft_code` / `dense_ft_math` | Trunk MLP fine-tuned on one domain |
| `expert_only_code` / `expert_only_math` | Assembled model, `force_expert` |
| `random_route` | Uniform one expert per token |
| `hard_route` | Gold domain id; **code/math only** |
| `seqmoe_top1` / `seqmoe_top2` | Learned token-level Top-K |

`resources` records `optimizer_numel`, `peak_vram_mb` (0 on CPU), `resident_expert_count`, wall time, and steps per stage.

### `scientific_grade`

| Value | Meaning |
|-------|---------|
| `synth` | In-process Python-ish / arithmetic / filler. **Plumbing smoke only** — not a quality claim. |
| `weak` | Wiki is real, but code was harvested from lab `*.py` (`code_source=lab_python`) and/or math is the generated fallback (`math_source=fallback`). |
| `full` | Wiki + dumped `data/cache/code.txt` + dumped `data/cache/math.txt` (not harvest/fallback). |

Do not publish a `synth` or `weak` Top-K win as the spec's scientific success bars. Those bars (learned Top-2 vs random on real code/math, Top-2 ≤ Top-1 on cross-domain templates) are measured when a scientific run happens; they are not required to close the plumbing PoC.

Fairness: sequential MoE wall is `pretrain + N*expert + router` steps (`N = n_experts`). `dense_ft_mix` is `pretrain + expert`. `dense_ft_mix_matched` is `pretrain + N*expert`. The comparison JSON records these.

## Design in one page

1. Pretrain a dense GPT on mixed data. Freeze embeddings, attention, norms, `lm_head`.
2. For each domain (code=0, math=1, wiki/fill=2 when `n_experts=3`): clone that layer's trunk MLP, train **only** the resident expert, save canonical `blocks.{i}.mlp.{fc,proj}.weight`, unload.
3. Assemble by logical id. Hash or shape mismatch is refused (no partial assemble).
4. Train the per-layer token router with trunk and experts frozen. Top-K gather/scatter runs only the selected expert FFNs.
