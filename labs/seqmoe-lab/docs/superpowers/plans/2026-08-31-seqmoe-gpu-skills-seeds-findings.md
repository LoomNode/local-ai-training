# GPU scientific-horizon 3-expert skills vs seeds (full dumps)

**Runs:** `feat/seqmoe-poc`, parallel  
`python scripts/run_poc.py --dump-skills --device cuda:1 --run-name poc_gpu_dump_skills`  
`python scripts/run_poc.py --dump-seeds --device cuda:0 --run-name poc_gpu_dump_seeds`  
**Grade:** `scientific_grade=full` (`code_source=dump` CPython 3.11 Lib, `math_source=dump` GSM8K, wiki WikiText-2).  
**JSON:** `results/poc_gpu_dump_{skills,seeds}_comparison.json`  
**Width:** `n_embd=256` `n_layer=8` `n_experts=3` `top_k=2` `seq_len=256`.  
**Budget:** `pretrain=5000` `expert=3000` `router=2000` (seqmoe wall **16000**; `dense_ft_mix_matched` wall **14000**). Not the CPU dump 120/120/120 horizon.  
**VRAM:** pretrain peak 923 MB; expert-stage ~794 MB (1 resident, `optimizer_numel=4194304`); router ~832 MB (3 resident, router `optimizer_numel=6144`). Envelope 22000 MB. Swap not counted.

Locked comparison tables (do not overwrite):

- CPU 120-step 3-expert dumps: `results/poc_cpu_dump_{skills,seeds}_comparison.json`
- 2-expert GPU `--scientific`: `docs/superpowers/plans/2026-08-30-seqmoe-weak-to-full-findings.md`

Trunk NLL on this GPU pair matches that 2-expert scientific trunk to three decimals (same seed, same pretrain): code 1.412, math 1.149, wiki 1.640, mixed **1.473**, cross 1.996.

## GPU skills NLL (clone + domain)

| Arm | code | math | wiki | mixed | cross |
|-----|------|------|------|-------|-------|
| trunk | 1.412 | 1.149 | 1.640 | 1.473 | 1.996 |
| dense_ft_mix | 1.344 | 1.033 | 1.892 | 1.521 | 1.887 |
| dense_ft_code | 1.325 | 1.736 | 2.228 | 1.833 | 2.093 |
| dense_ft_math | 2.034 | 0.990 | 2.058 | 1.786 | 2.555 |
| dense_ft_wiki | 2.013 | 1.630 | **1.488** | 1.820 | 3.100 |
| dense_ft_mix_matched | 1.343 | 0.997 | 2.051 | 1.560 | 1.921 |
| expert_only_0 (code) | **1.334** | 1.768 | 2.326 | 1.885 | 2.060 |
| expert_only_1 (math) | 1.995 | **0.983** | 2.015 | 1.757 | 2.527 |
| expert_only_2 (wiki) | 2.036 | 1.585 | **1.484** | 1.808 | 3.112 |
| random_route | 1.724 | 1.359 | 1.867 | 1.737 | 2.385 |
| hard_route | 1.334 | 0.983 | 1.484 | — | — |
| seqmoe_top1 | 1.428 | 1.105 | 1.582 | 1.499 | 2.077 |
| seqmoe_top2 | 1.364 | 1.033 | 1.522 | **1.426** | **1.953** |

Pairwise expert cosine 0.775 / 0.769 / 0.796 (moved off the clone; CPU 120-step was 0.985).

## GPU seeds NLL (random + mix)

| Arm | code | math | wiki | mixed | cross |
|-----|------|------|------|-------|-------|
| trunk | 1.412 | 1.149 | 1.640 | 1.473 | 1.996 |
| dense_ft_mix_matched | 1.331 | 0.998 | 2.039 | 1.550 | 1.921 |
| expert_only_0 | 1.387 | 1.116 | 1.601 | 1.445 | 1.910 |
| expert_only_1 | 1.385 | 1.106 | 1.611 | 1.442 | 1.919 |
| expert_only_2 | 1.397 | 1.111 | 1.603 | 1.443 | 1.932 |
| random_route | 1.442 | 1.168 | 1.674 | 1.504 | 1.964 |
| seqmoe_top1 | 1.406 | 1.128 | 1.627 | 1.462 | 1.940 |
| seqmoe_top2 | 1.348 | 1.080 | 1.573 | 1.411 | 1.846 |

Pairwise expert cosine **0.021** (CPU 120-step was 0.002).

## (a) Skills specialists vs trunk and vs matching `dense_ft_*`

On-domain sequential expert ≈ matching dense-FT MLP, both beat the trunk:

| Domain | expert_only | trunk | matching dense_ft |
|--------|-------------|-------|-------------------|
| code | 1.334 | 1.412 | dense_ft_code 1.325 |
| math | 0.983 | 1.149 | dense_ft_math 0.990 |
| wiki | 1.484 | 1.640 | dense_ft_wiki 1.488 |

Cross-domain NLL is worse (code expert on math 1.768, math expert on code 1.995). Clone+freeze is not the problem; sequential specialist ≈ dense-FT specialist. Same pattern as CPU 120-step (experts 2.795/2.564/2.645 vs trunk 3.068/2.973/2.950 vs dense_ft 2.794/2.570/2.607) and as the 2-expert GPU scientific table (code 1.334 vs trunk 1.412 vs dense_ft_code 1.343).

**Spec 3 — PASS** (now with a wiki owner too).

## (b) Skills `gold_agree` — no longer collapsed onto expert 0

| Split | GPU 3-expert 5000/3000/2000 | CPU 3-expert 120-step |
|-------|-----------------------------|------------------------|
| code | **0.845** | 0.977 |
| math | **0.760** | 0.006 |
| wiki | **0.826** | 0.035 |

CPU 120-step top-1 sat on expert 0. At scientific horizon the router actually tracks gold ids. Top-1 route mass on-domain: code `r0=0.74`, math `r1=0.62`, wiki `r2=0.68`. Mixed top-1 is balanced (`r0/r1/r2 ≈ 0.36/0.31/0.33`). The 2-expert GPU scientific run had gold_agree code 0.847 / math 0.853 — this 3-expert run is in that band, not the CPU collapse.

## (c) Mix `seqmoe_top2` vs trunk vs `dense_ft_mix_matched`

| Run | top2 mix | trunk mix | matched mix | random mix |
|-----|----------|-----------|-------------|------------|
| GPU 3-expert skills | **1.426** | 1.473 | 1.560 | 1.737 |
| GPU 3-expert seeds | **1.411** | 1.473 | 1.550 | 1.504 |
| CPU 120-step skills | 2.702 | 2.902 | 2.700 | 2.916 |
| CPU 120-step seeds | 2.726 | 2.902 | 2.718 | 2.878 |
| GPU 2-expert scientific | 1.509 | **1.473** | (no matched arm in that table; dense_ft_mix 1.514) | 1.754 |

**Sign vs CPU 120-step:** both horizons, skills Top-2 beats trunk on mix. CPU 120-step Top-2 **tied** matched (2.702 vs 2.700). GPU 3-expert Top-2 **beats** matched (1.426 vs 1.560).

**Sign vs 2-expert GPU scientific:** flipped. That run’s Top-2 mix **lost** to trunk (1.509 vs 1.473). This 3-expert run **beats** trunk (1.426 vs 1.473). The 2-expert committee had no wiki owner on a mix that includes wiki; expert 2 is that owner here (`hard_route` wiki 1.484 vs trunk 1.640).

Fairness: seqmoe wall 16000 vs matched 14000 is **+2000 router steps**, not extra MLP steps. Matched already used `N * expert_steps` on one dense MLP. Do not call this “MoE won because it trained longer on the FFN.” It won because three specialists + Top-2 beat one dense MLP trained for the same expert-step budget, and beat the frozen trunk. `dense_ft_mix` (8000 wall) mixed 1.521 also loses to Top-2.

**Spec 4 — PASS** vs random (1.426 < 1.737) **and** vs trunk (unlike the 2-expert table).  
**Spec 5 — PASS:** Top-2 cross 1.953 ≤ Top-1 cross 2.077 (also ≤ trunk cross 1.996).

## (d) Seeds remain non-specialists

Force-one NLL is interchangeable across experts (code 1.387 / 1.385 / 1.397; math 1.116 / 1.106 / 1.111; wiki 1.601 / 1.611 / 1.603). No domain split. Pairwise cosine ~0.021. Each seed is a slightly-better-than-trunk generalist on mix (~1.44 vs 1.473). Top-2 averaging them (1.411) looks as good as the skills committee on mix, but that is **not** specialization — it is three diverse generalists. Naming a skill (clone + domain corpus) is what produces the on-domain win (math expert 0.983 vs any seed ~1.11).

## What this does not show

- 27B / paging / LoRA experts.
- That seeds become specialists if you train them longer on mix (3000 steps did not).
- A mix win from extra MLP wall vs matched dense-FT (the extra wall is the router).
- That harvested lab Python is a `full` code domain (these runs are dumps).
