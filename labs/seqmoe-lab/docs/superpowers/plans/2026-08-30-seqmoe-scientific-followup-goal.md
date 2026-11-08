# Plan: Sequential MoE scientific follow-up (dense-FT arms, full-grade dumps, diagnosis, VRAM at scale)

## Goal kind
code-change

## Locked from the previous goal (do not redo)

Plumbing PoC is on `/games/ailab/lab/seqmoe-lab` branch `feat/seqmoe-poc`.

Commits: `1d6e968` spec → `540b78b` PoC → `3203948` `.to(cuda)` then `set_mode` device placement → `07052ae` harvest not relabeled `dump`.

Already gated: CPU `pytest -q` (22), tiny PoC twice on CPU + once on `cuda:1`, sequential vs joint `optimizer_numel` 16384 vs 32768 (tiny), `peak_vram_mb` tiny CUDA 20.57 vs 22.02. New modules after `.to(device)` must stay on `tok_emb.weight.device` (`SeqMoE._placed`).

Weak scientific run (`cuda:1`, `scientific_config`, ~6 min, `scientific_grade=weak`, `code_source=lab_python`, `math_source=fallback`) already measured these holdout NLLs (nats/token). Treat as the baseline table; do not claim they are `full`:

| Arm | code | math | mixed | cross | wiki |
|-----|------|------|-------|-------|------|
| trunk | 1.815 | 0.860 | **1.345** | **2.691** | 1.664 |
| expert_only_code | 1.916 | 1.955 | 3.005 | 4.871 | 3.962 |
| expert_only_math | 4.739 | **0.800** | 2.531 | 5.713 | 3.710 |
| random_route | 3.097 | 1.228 | 2.221 | 4.259 | 2.955 |
| seqmoe_top1 | 1.871 | 0.911 | 1.845 | 4.460 | 2.584 |
| seqmoe_top2 | 1.782 | 0.820 | 1.600 | 4.220 | 2.153 |

Spec success vs that table: (3) code specialist does **not** beat trunk on code; math specialist does (0.800 vs 0.860). (4) Top-2 mixed 1.600 beats random 2.221, loses to trunk 1.345. (5) Top-2 ≤ Top-1 on cross (4.220 ≤ 4.460); both lose to trunk. Expert-stage `optimizer_numel=4194304`; pretrain `peak_vram_mb≈923`. Comparison JSON: keep a copy under `{SCRATCH}/poc-scientific-comparison.json` if still present, else re-run weak only as a last resort.

## Acceptance criteria
1. **Dense fine-tune arms exist on the shipped path.** `evaluate_poc` / `run_and_eval` write numeric `nll`/`bpb` for `dense_ft_mix`, `dense_ft_code`, and `dense_ft_math` on the same holdouts as the other arms. Each arm starts from the frozen trunk, stays in `pretrain` (dense MLP, no bank), freezes embeddings/attention/norms/`lm_head`, trains **only** `blocks.*.mlp.*` for `TrainConfig.expert_steps` (same active-param count as one sequential expert). `dense_ft_mix` trains on mixed code+math; the domain arms train on that domain only. Wall-step fairness fields stay in the JSON: seqmoe `pretrain+2*expert+router`, dense_ft_mix `pretrain+expert`. Tiny pytest drives the real `train_dense_ft` + `evaluate_poc` (no mocks, no hardcoded NLL) and asserts the three arms are present with finite `nll`/`bpb` on code/math/mixed.
2. **`scientific_grade=full` is attempted honestly.** `data/cache/sources.json` remains the origin of truth (`07052ae`): harvested lab Python stays `lab_python`, generated arithmetic stays `fallback`; existing files without a sidecar may be user dumps. Obtain real dumps into `data/cache/code.txt` and `data/cache/math.txt` (math = GSM8K questions+answers or equivalent dumped text; code = a dumped Python corpus that is **not** `harvest_lab_python()`). Sidecar `code_source=dump`, `math_source=dump`. Wiki remains `SEQMOE_WIKI` or `/games/ailab/lab/scrt-lab/data/wiki.train.raw`. If a dump cannot be obtained (network down, missing file), capture the exact missing path / HTTP error under `{SCRATCH}/dump-status.txt`, leave sidecar as harvest/fallback, and **do not** label the run `full`. Do not silently promote harvest to dump.
3. **One `--scientific` run on `cuda:1`** (CPU only if both 3090s are full — then capture `nvidia-smi`) includes every spec eval arm: trunk, the three dense_ft arms, expert_only_code/math, hard_route (code/math only), random_route, seqmoe_top1, seqmoe_top2. Same holdouts. JSON has numeric `nll`/`bpb` on code, math, mixed, wiki, cross. Copy `{run}_comparison.json` to `{SCRATCH}/`. Record `scientific_grade`, `code_source`, `math_source`.
4. **Diagnosis of the code-expert regression is written from measurements, not vibes.** The note must compare trunk vs `dense_ft_code` vs `expert_only_code` on code/math/wiki holdouts (and train loss vs holdout if logged). Possible outcomes the note must distinguish: (a) any extra MLP steps on this code source overfit (dense_ft_code also worse than trunk), (b) sequential clone+freeze is the problem (dense_ft_code beats trunk, expert_only_code does not), (c) holdout is too close to wiki/pretrain mix. File: `lab/seqmoe-lab/docs/superpowers/plans/2026-08-30-seqmoe-weak-to-full-findings.md` plus `{SCRATCH}/findings.md`. Do not claim a scientific win if trunk still beats the committee on mixed.
5. **Joint vs sequential VRAM at scientific size.** `scripts/run_memory.py` (or a `--scientific` flag using `scientific_config` model dims, few steps) writes `optimizer_numel` and `peak_vram_mb` for sequential (resident 1) vs joint (both expert FFNs in one optimizer) on `cuda:1` if free else CPU (`peak_vram_mb=0`). Sequential expert-stage `optimizer_numel` must be ≈ half of joint for `n_experts=2`. Capture `{SCRATCH}/memory-compare-scientific.log` and JSON. Swap is not capacity; `MemoryPlanError` if the envelope would be exceeded — do not catch CUDA OOM and retry via swap.

## Verification plan
1. gating: CPU `pytest -q` from `lab/seqmoe-lab` still passes the existing suite **and** new tests that: (i) `train_dense_ft` on a tiny trunk leaves only `blocks.*.mlp.*` with `requires_grad`; (ii) after three short dense_ft trains, `evaluate_poc` JSON contains `dense_ft_mix` / `dense_ft_code` / `dense_ft_math` with finite `nll`/`bpb` on code/math/mixed; (iii) `prepare_scientific` still does not relabel harvest/fallback as dump (`tests/test_prepare_grade.py`). Capture `{SCRATCH}/pytest.log`. Do not mock `SeqMoE`; do not hardcode expected NLL.
2. gating: `python scripts/run_poc.py --tiny --device cpu` (or `cuda:1` if free) writes comparison JSON that includes the three dense_ft arms plus the previous tiny arms. Copy to `{SCRATCH}/poc-tiny-dft-comparison.json`. Exit 0.
3. gating: Dump attempt logged in `{SCRATCH}/dump-status.txt` (paths, byte sizes, sidecar contents, `scientific_grade`). If grade is `full`, `sources.json` has `code_source=dump` and `math_source=dump` and wiki exists. If grade is `weak`, the file states **why** (exact missing URL or path) and the sidecar still says harvest/fallback.
4. gating: `python scripts/run_poc.py --scientific --device cuda:1` exit 0 (or CPU if both GPUs full — capture `nvidia-smi`). Comparison JSON in `{SCRATCH}/` has numeric NLL/bpb for **all** arms in criterion 3 on code/math/mixed (wiki/cross too if those splits exist). `scientific_grade` matches the sidecar.
5. gating: Scientific-size memory compare as in criterion 5. Sequential `optimizer_numel * 2 == joint` (exact for two identical FFN banks). `{SCRATCH}/memory-compare-scientific.log`.
6. evidence: Findings note (criterion 4) quotes the new JSON numbers, the locked weak-run table, and a pass/fail line for spec success 3, 4, 5. A fail on 3–5 does **not** fail this goal if the numbers were measured and the diagnosis distinguishes (a)/(b)/(c). A win that is only extra steps vs `dense_ft_mix` must be called out using the wall-step fields.
7. evidence: `git -C /games/ailab/lab/seqmoe-lab status` and `diff --stat main...HEAD` in `{SCRATCH}/git-status.txt`. Touch only `seqmoe-lab` (not `scrt-lab` / `envelope-lab` / Colibri). Prefer continuing `feat/seqmoe-poc`.

## Non-goals
- NVMe/RAM/VRAM expert paging, leases, prefetch, Colibri kernels, dual-GPU layer ownership.
- Hardware-aware router terms (switch / cache-miss / memory-pressure).
- LoRA/adapter experts as the scientific path; HuggingFace / Qwen / BPE trunks; 27B scale.
- Distillation into the trunk; joint unfreeze of trunk+experts as a quality stage; training independent full models with no shared trunk.
- Autoresearch `ailab loop` campaigns; importing `scrt`, envelope-lab, or Colibri as Python dependencies.
- Re-implementing the frozen-trunk sequential PoC, Top-K gather/scatter, hash/assemble, or tiny synth plumbing (already shipped).
- Falsifying `scientific_grade=full` from harvested lab Python or `synth_math_scientific` fallback.
- Merging `feat/seqmoe-poc` to `main` unless the user asks in that session.
- Closing spec success 3–5 as a required pass — measure and diagnose; the hypothesis may be false at this scale/data.

## Assumed scope
`/games/ailab/lab/seqmoe-lab/` on `feat/seqmoe-poc`. Spec: `docs/superpowers/specs/2026-08-30-sequential-moe-poc-design.md` (§7.2 dumps, §9 memory, §10 eval arms including `dense_ft_*`, success items 3–5 in §1). Wiki: `SEQMOE_WIKI` or `/games/ailab/lab/scrt-lab/data/wiki.train.raw`. Cache: `data/cache/{code,math}.txt` + `sources.json` (gitignored). Default research GPU `cuda:1`; `resolve_device("cuda")→cuda:1`; envelope `max_vram_mb=22000`; swap is not capacity. Torch + numpy + pytest; no HuggingFace. `scientific_config`: `n_embd=256`, `n_head=8`, `n_layer=8`, `block_size=256`, `batch_size=16`, `seq_len=256`, `lr=8e-4`, `pretrain_steps=5000`, `expert_steps=3000`, `router_steps=2000`. Tiny remains 20/20/20 at `n_embd=32`, `n_layer=2`.

## Implementation approach
Add `train_dense_ft(model, data, cfg, *, tag: str) -> StageStats` in `seqmoe/train.py`: `SeqMoE.from_trunk` / `load_trunk` (still `pretrain` with dense MLPs), freeze all params except names containing `.mlp.`, AdamW on those only, `expert_steps` loops via existing `_step_loss` / `TokenBatcher`. Save three checkpoints under the run `ckpts/` (`dense_ft_mix.pt` etc. as full dense state_dicts, format `seqmoe-trunk-v1` or a dedicated `seqmoe-denseft-v1` that `load_trunk` can still eval). Call this from `run_sequential_poc` after stage A using copies of the trunk file (do not mutate the frozen trunk used for experts). Extend `evaluate_poc` to eval those three paths like `trunk`. Extend `has_required_tiny_arms` (or a sibling `has_required_eval_arms`) to require dense_ft keys. For dumps: a small `scripts/fetch_dumps.py` or `prepare.py --fetch` that writes math from GSM8K (network) and code from a dumped corpus; always write `sources.json`. Memory: `compare_sequential_vs_joint(scientific_config(device), steps=4)` — if activation plan exceeds envelope, raise `MemoryPlanError` and record it (still a valid measurement). Findings markdown is the quality deliverable. Keep gather/scatter Top-K; do not add a dense “run every expert and mask” path.

## Task checklist
- [ ] `train_dense_ft` + pytest: only MLP trainable; tiny evaluate JSON has `dense_ft_*` finite NLL/bpb
- [ ] Wire dense_ft into `run_sequential_poc` / `run_and_eval` / README arm table; tiny PoC writes all arms
- [ ] Fetch or refuse dumps; sidecar + `{SCRATCH}/dump-status.txt`; never promote harvest to `full`
- [ ] `--scientific` on `cuda:1` (or captured GPU-full) with every spec arm; copy comparison JSON
- [ ] Scientific-size sequential vs joint memory compare; log under `{SCRATCH}`
- [ ] Findings note: trunk vs dense_ft_code vs expert_only_code; spec 3/4/5 pass/fail quoted from JSON
- [ ] `pytest -q` green; `{SCRATCH}/git-status.txt` is seqmoe-lab only

## Risks / Contradictions
- OBJECTIVE for a new session may again be the word “R&D”. This plan is **the leftover scientific follow-up**, not a second plumbing rewrite and not paging/Colibri.
- Weak-run code expert already lost to trunk (1.916 vs 1.815). Criterion 4 must explain that with dense_ft; do not ignore it and run another synth PoC.
- GSM8K / a real code dump may be unavailable offline. Then grade stays `weak` and criterion 2’s fail-closed dump log is the pass condition, not a fake `full`.
- Local serving often fills both 3090s. Scientific gating may use CPU only if `nvidia-smi` shows both cards without ≥2 GB free; say so. Tiny dense_ft tests stay CPU.
- `build_scientific_data` currently caps streams at 2e6 bytes — keep or document if dumps are larger; do not silently train on a different holdout than eval.
- Joint scientific-size step may trip the 22 GB envelope via `activation_est`; that is a measurement, not a license to swap.

## How to launch
Paste this file as the goal plan (or point a new Grok goal at `lab/seqmoe-lab/docs/superpowers/plans/2026-08-30-seqmoe-scientific-followup-goal.md`). Worktree: `/games/ailab/lab/seqmoe-lab` on `feat/seqmoe-poc`. Scratch: the goal harness `{SCRATCH}` dir (never shared `/tmp/...`).
