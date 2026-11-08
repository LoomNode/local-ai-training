# Sequential MoE findings: weak harvest vs full dumps

**Run:** `feat/seqmoe-poc`, `python scripts/run_poc.py --scientific --device cuda:1`  
**Grade:** `scientific_grade=full` (`code_source=dump` CPython 3.11 Lib, `math_source=dump` GSM8K train.jsonl, wiki WikiText-2). Streams capped at 2e6 bytes.  
**JSON:** `{SCRATCH}/poc-scientific-comparison.json`  
**Wall steps:** seqmoe `pretrain+2*expert+router` = **13000**; `dense_ft_mix` = **8000**.

## Locked weak-run table (do not treat as full)

Harvested lab Python + generated arithmetic, `cuda:1`, same `scientific_config`:

| Arm | code | math | mixed | cross | wiki |
|-----|------|------|-------|-------|------|
| trunk | 1.815 | 0.860 | 1.345 | 2.691 | 1.664 |
| expert_only_code | **1.916** | 1.955 | 3.005 | 4.871 | 3.962 |
| expert_only_math | 4.739 | 0.800 | 2.531 | 5.713 | 3.710 |

Code expert **lost** to trunk on code (1.916 vs 1.815).

## Full-grade NLL (nats/token)

| Arm | code | math | mixed | wiki | cross |
|-----|------|------|-------|------|-------|
| trunk | 1.412 | 1.149 | **1.473** | **1.640** | 1.996 |
| dense_ft_mix | 1.338 | 1.031 | 1.514 | 1.899 | 1.885 |
| dense_ft_code | 1.343 | 1.728 | 1.853 | 2.227 | 1.963 |
| dense_ft_math | 1.970 | 0.988 | 1.751 | 2.025 | 2.462 |
| expert_only_code | **1.334** | 1.768 | 1.885 | 2.326 | 2.060 |
| expert_only_math | 1.995 | **0.983** | 1.757 | 2.015 | 2.527 |
| random_route | 1.622 | 1.297 | 1.754 | 2.090 | 2.181 |
| hard_route | 1.334 | 0.983 | — | — | — |
| seqmoe_top1 | 1.413 | 1.059 | 1.596 | 2.003 | 2.055 |
| seqmoe_top2 | 1.347 | 1.010 | 1.509 | 1.866 | **1.915** |

Dense-FT and sequential experts used the same active-param count (`optimizer_numel=4194304`). Router `optimizer_numel=4096`. Pretrain peak VRAM 923 MB; expert-stage ~794 MB.

## Diagnosis: why the weak-run code expert lost to trunk

Compare **trunk vs `dense_ft_code` vs `expert_only_code`** on this full run:

| | code | math | wiki |
|--|------|------|------|
| trunk | 1.412 | 1.149 | 1.640 |
| dense_ft_code | 1.343 | 1.728 | 2.227 |
| expert_only_code | 1.334 | 1.768 | 2.326 |

On dumped CPython, **both** extra-MLP arms beat the trunk on code and get worse on math/wiki. Sequential expert ≈ dense_ft_code (1.334 vs 1.343). Train loss for expert_0 ended at 0.69 with holdout eval 1.35 — not a collapse.

The plan’s three buckets:

- **(a) extra MLP steps overfit this code source** — **false on dumps** (`dense_ft_code` better than trunk on code). Plausible **on the harvest**: lab `*.py` sit next to wiki in the mixed pretrain, so more MLP steps on the same-ish bytes do not buy a distinct code holdout.
- **(b) sequential clone+freeze is the problem** (`dense_ft_code` beats trunk, expert does not) — **false**. Expert matches dense_ft.
- **(c) holdout overlap with pretrain mix** — **the weak-run explanation.** Harvested scrt/envelope/ailab-cli Python is the same house as the lab, mixed into pretrain with wiki; the “code” holdout was not a new domain. GSM8K + CPython Lib are.

**Call:** weak-run code-expert regression was **(c)**, with harvest behaving like **(a)**. It was not sequential clone/freeze.

## Spec success 3–5 (measured on full JSON)

**3. Specialization — PASS**

- Expert 0 code 1.334 < trunk code 1.412 and < expert 0 math 1.768
- Expert 1 math 0.983 < trunk math 1.149 and < expert 1 code 1.995

**4. Routing vs random on mixed — PASS vs random, not vs trunk**

- seqmoe_top2 mixed 1.509 < random 1.754
- seqmoe_top2 mixed 1.509 **>** trunk mixed 1.473 (committee does not beat frozen pretrain on mix)
- seqmoe_top2 mixed 1.509 ≈ dense_ft_mix 1.514, but seqmoe used **13000** wall steps vs dense_ft_mix **8000**. Do not call this a sequential-MoE win on extra steps.

Router gold_agree (top-1): code 0.847, math 0.853.

**5. Cross-domain Top-2 ≤ Top-1 — PASS**

- seqmoe_top2 cross 1.915 ≤ seqmoe_top1 cross 2.055
- Top-2 also ≤ trunk cross 1.996

## Scientific-size memory (not quality)

`scripts/run_memory.py --scientific --device cuda:1 --steps 4`

| | optimizer_numel | peak_vram_mb | resident |
|--|-----------------|--------------|----------|
| sequential | 4194304 | 793.72 | 1 |
| joint | 8388608 | 1156.37 | 2 |

Ratio exactly 2.0. Swap not counted.

## What this does not show

- A mixed-holdout win for the committee over the trunk.
- That 27B / paging / LoRA experts would behave the same.
- Quality of harvested lab Python as a “code” domain (it is not `full`).
