# Sequential MoE PoC — Design

**Date:** 2026-08-30
**Repo:** `/games/ailab/lab/seqmoe-lab/`
**Status:** approved in conversation; awaiting implementation plan

This spec is **sub-project 1 only**: train two expert FFNs independently against a frozen shared trunk, assemble them, train a router, and measure whether that committee works — and whether sequential training actually saves optimizer/VRAM versus joint training.

It does not specify NVMe paging, dual-GPU layer splits, cache-aware routing, distillation, or 27B-scale experts.

---

## 1. Hypothesis

Sparse model training does not require every future expert to occupy accelerator memory simultaneously if those experts share a stable pretrained representation and their specialization can be learned independently.

If true, **total specialist capacity is a storage problem**, while **training memory and inference compute depend on the active subset**.

The PoC tests that claim at lab scale, not at 200B.

### Success (all required)

1. **Independence.** Training Expert 1 does not mutate Expert 0’s file or any shared trunk tensor. Trunk SHA-256 is identical in every expert checkpoint.
2. **Memory.** In `expert` mode, optimizer state exists only for the resident expert MLP bank. Peak allocated bytes track `trunk + 1 expert + AdamW(1 expert)`, not `N` experts. Measured on the tiny model with a joint-training arm as the counterfactual.
3. **Specialization.** After sequential training, Expert 0’s holdout NLL on code is lower than the frozen trunk’s, and lower than Expert 0’s NLL on math. Symmetric for Expert 1 on math.
4. **Routing.** On mixed holdout, learned Top-2 beats random routing. Gold-domain hard routing is the ceiling; the learned router should approach it, not collapse to one expert.
5. **Cross-domain.** On mixed code+math templates, Top-2 NLL is ≤ Top-1 NLL (measured, not assumed).

If (3) or (4) fail, stop. Do not build a pager around a committee that does not work.

---

## 2. Non-goals (explicit)

Out of this repo and this spec:

- NVMe / RAM / VRAM expert paging, leases, prefetch, Colibri kernels
- Dual-GPU contiguous layer ownership
- Router switch-cost, cache-miss, or memory-pressure penalties
- Distilling specialists back into the trunk
- LoRA / adapter experts as the scientific path
- Independent full models trained per domain with no shared trunk
- Unfreezing attention/embeddings while a specialist is being created
- Qwen 27B, HuggingFace pretrained trunks, BPE tokenizers
- Autoresearch campaign loops (`ailab loop`)

A later sub-project may copy Colibri’s logical-id → physical-slot mapping. This spec only reuses the **logical expert id** idea.

---

## 3. What we reuse from other labs

`seqmoe-lab` is a new package. It does **not** import `scrt`, `envelope-lab`, or Colibri as dependencies. It **copies and adapts** small, proven pieces.

| Source | Reuse | Do not reuse |
|--------|--------|----------------|
| `scrt-lab` | Decoder block: pre-norm, bias-free `CausalSelfAttention`, GELU MLP `fc`/`proj` with `mlp_mult=4`. `ModelConfig` / `TrainConfig` dataclasses. `resolve_device("cuda") → "cuda:1"`. `TokenBatcher` + AdamW step loop + `eval_loss`. JSON `RunResult` files. WikiText-2 at `scrt-lab/data/wiki.train.raw`. Reporting **loss vs steps and wall time**, not just params. | Residual slot bank, consolidate/merge/free/regrow, looped shared-core, synonym language as a scientific corpus. |
| `envelope-lab` | Fail-closed VRAM envelope (`MAX_VRAM_MB = 22000` on a 3090). Research default GPU 1. Startup must not “allocate until Linux swaps.” | Frozen `prepare.py` vs agent-editable `train.py` campaign loop. Shakespeare-only eval. |
| Colibri / llama.cpp | **Logical expert id** is the semantic identity. Cache location is not. Conceptual `ExpertRecord { logical_id, layer, tensor names }`. | File-backed GGUF pages, `ExpertLease`, CUDA events, `n-cpu-moe`, dual-GPU tensor-split. |
| Lab trees | Optional **code-corpus fallback**: concatenate `*.py` under `/games/ailab/lab/{scrt-lab,envelope-lab,ailab-cli}` when no dedicated Python dump is cached. Record `code_source` in results. | Using those trees as the only scientific evidence without flagging `scientific_grade=weak`. |

If scrt-lab code and this lab drift, this lab wins. Copy, do not submodule.

---

## 4. Architecture

One `SeqMoE` module, four residency **modes**. Whether an expert is in VRAM is an implementation detail of the mode. Callers pass **logical ids**.

```text
idx → tok_emb + pos_emb
    → N × Block:
         x = x + attn(ln1(x))          # shared
         x = x + expert_ffn(ln2(x))    # mode-dependent
    → ln_f → lm_head                   # shared, tied to tok_emb
```

### 4.1 Shared trunk

After pretrain, frozen forever for specialist and router stages:

- `tok_emb`, `pos_emb`
- every block’s `ln1`, `attn`, `ln2`
- `ln_f`, `lm_head` (`lm_head.weight` is the same storage as `tok_emb.weight`)

The pretrained dense MLP in each block is the **clone template**, then discarded from the live MoE. The dense checkpoint stays on disk as the `trunk` eval arm.

`Block` fields by mode:

- `pretrain`: `mlp: MLP`, `bank is None`, `router is None`
- `expert` / `router` / `eval`: `mlp is None`, `bank: ExpertBank`, `router` present only in `router`/`eval`

### 4.2 Expert FFN

Per layer, per logical id: a full `MLP` with the same shapes as the trunk MLP.

```python
class MLP(nn.Module):
    def __init__(self, n_embd: int, mlp_mult: int = 4, dropout: float = 0.0) -> None:
        ...
        self.fc = nn.Linear(n_embd, mlp_mult * n_embd, bias=False)
        self.proj = nn.Linear(mlp_mult * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))
```

**Init:** `load_state_dict` from that layer’s trunk MLP. Never clone expert *i−1*. Never default to random. `TrainConfig.expert_init = "clone" | "random"` exists only as an ablation; default `"clone"`.

**Bank:** only **resident** experts are `nn.Module` children. Parked experts are not constructed.

```python
class ExpertBank(nn.Module):
    """Per-layer map logical_id → MLP. Absent id ⇒ not in the module tree."""

    experts: nn.ModuleDict  # keys are str(logical_id)

    def install(self, logical_id: int, mlp: MLP) -> None: ...
    def uninstall(self, logical_id: int) -> None: ...
    def resident_ids(self) -> tuple[int, ...]: ...
```

### 4.3 Router

One per layer, created only in `router` and `eval` modes:

```python
class Router(nn.Module):
    def __init__(self, n_embd: int, n_experts: int) -> None:
        self.gate = nn.Linear(n_embd, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → logits (B, T, E)
        return self.gate(x)
```

Routing is **token-level, per layer**. Top-K (default K=2):

1. `logits = router(x)`
2. `val, idx = logits.topk(k, dim=-1)`
3. `w = softmax(val)`
4. Gather tokens per selected expert, run **only** those MLPs, scatter back (no dense “run every resident expert and mask” path — even the tiny model uses gather/scatter so tests cover the real kernel)
5. Weighted sum into the residual stream

v1 does **not** include switch-cost or cache-miss terms.

Load-balance auxiliary loss (Switch Transformer, default `balance_coef=0.01`):

```text
f_i = fraction of tokens whose top-1 is expert i
P_i = mean softmax probability on expert i
L_balance = n_experts * sum_i f_i * P_i
L = L_ce + balance_coef * L_balance
```

Optional `route_coef * L_route` (CE of router vs gold domain id, broadcast to all tokens in a sequence) defaults to **0**. Labels must not be the only thing that makes routing work.

### 4.4 Modes

| Mode | Resident | `requires_grad` | FFN path |
|------|----------|-----------------|----------|
| `pretrain` | dense trunk MLP per layer | all | dense MLP |
| `expert` | trunk (frozen) + expert `i` MLPs | expert `i` only | that MLP, no router |
| `router` | trunk + all experts + routers | routers only | Top-K MoE |
| `eval` | same as `router`, or force-one-expert / random / hard-route | none | selected |

`SeqMoE.set_mode(mode, expert_id=None)` is the only legal way to change residency. It must `uninstall` experts that are not needed so they drop out of `parameters()` and CUDA storage.

### 4.5 Forward contract

```python
class SeqMoE(nn.Module):
    def set_mode(self, mode: str, expert_id: int | None = None) -> None: ...
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        force_expert: int | None = None,
        route: str = "learned",  # learned | random | hard
        hard_id: int | None = None,
        top_k: int | None = None,
    ) -> ForwardResult: ...

@dataclass
class ForwardResult:
    logits: torch.Tensor
    loss: torch.Tensor | None
    aux_loss: torch.Tensor | None
    route_frac: torch.Tensor | None   # (E,) token fractions, eval/router only
```

`hard` routing uses `hard_id` for every token and layer. `random` samples uniformly per token. Gold domain for a sequence is an eval-time argument, not a model weight.

---

## 5. Assembly and on-disk format

### 5.1 Trunk hash

SHA-256 over shared tensors only, in sorted state-dict key order, each value `cpu().contiguous().numpy().tobytes()`:

- `tok_emb.weight`, `pos_emb.weight`
- `blocks.{i}.ln1.*`, `blocks.{i}.attn.*`, `blocks.{i}.ln2.*`
- `ln_f.*`, `lm_head.weight`

Expert MLP keys and router keys are excluded. Two experts trained from the same trunk file must embed this same hex digest.

### 5.2 Trunk checkpoint

`trunk.pt` is a `dict`:

```python
{
    "format": "seqmoe-trunk-v1",
    "model_config": dict,          # asdict(ModelConfig)
    "state_dict": dict,            # dense GPT (includes MLPs used as clone source)
    "trunk_sha256": str,
    "tokenizer": "bytes256",
    "pretrain": dict,              # steps, corpus names, final eval NLL
}
```

### 5.3 Expert checkpoint

`expert_{logical_id}_{domain}.pt`:

```python
{
    "format": "seqmoe-expert-v1",
    "logical_id": int,             # 0 or 1 in v1; assigned in config, not train order
    "domain": str,                 # "code" | "math"
    "trunk_sha256": str,
    "model_config": dict,          # must match trunk
    "state_dict": dict,            # only blocks.{i}.mlp.{fc,proj}.weight
    "train": dict,                 # steps, corpus, device, peak_vram_mb, optimizer_numel
}
```

On-disk MLP keys are **canonical dense names**, independent of bank residency:

```text
blocks.{layer}.mlp.fc.weight
blocks.{layer}.mlp.proj.weight
```

Live module names under a bank (`blocks.{layer}.bank.experts.{id}.fc.weight`) are never written. `save_expert` copies each layer’s resident `MLP.state_dict()` into those canonical keys. `load_expert` maps them back into `bank.install`.

`load_expert` refuses to install if `trunk_sha256`, `n_layer`, `n_embd`, or `mlp_mult` differ.

Logical ids are **stable**: config maps `code → 0`, `math → 1`. Re-running Expert 0 overwrites `expert_0_code.pt`. Training order cannot remap ids.

### 5.4 Assemble

```python
def assemble(trunk_path: Path, expert_paths: list[Path]) -> SeqMoE:
    ...
```

- Load trunk, drop dense MLPs, `set_mode("router")` constructing empty banks
- Install each expert after hash/config checks
- Require `set(logical_ids) == {0, ..., n_experts-1}` with no duplicates
- Routers start random (Kaiming), not copied from anywhere

Errors (do not catch-and-continue):

- `TrunkHashMismatch`
- `ExpertConfigMismatch`
- `MissingExpertError`
- `DuplicateExpertError`

---

## 6. Training pipeline

### 6.1 Stage A — trunk pretrain

- Mode `pretrain`
- Corpus: mixed (CI: synth_code ∪ synth_math ∪ filler; scientific: wiki + code + math)
- AdamW on all params
- Save `trunk.pt`

### 6.2 Stage B — sequential specialists

For each `ExpertSpec(logical_id, domain, corpus)`:

1. `SeqMoE.from_trunk(trunk.pt)`
2. For each layer, `MLP` ← clone trunk MLP; `install(logical_id, mlp)`
3. `set_mode("expert", expert_id=logical_id)`
4. Freeze everything except resident MLPs
5. New AdamW on `filter(lambda p: p.requires_grad, model.parameters())` only
6. Train on that domain corpus
7. `save_expert(...)`
8. `uninstall` + delete optimizer; `gc.collect()`; `torch.cuda.empty_cache()` if CUDA

**Invariant checked every expert run:** `sum(p.numel() for p in model.parameters() if p.requires_grad)` equals the MLP bank size of **one** expert, not N.

### 6.3 Stage C — router

- `assemble(...)`
- Freeze trunk and all experts
- AdamW on router gates only
- Mixed batches (see Data). Default `route_coef=0`, `balance_coef=0.01`
- Save `router.pt` (router state_dict + same `trunk_sha256`)

### 6.4 No joint unfreeze in v1

There is no stage D that trains trunk LoRA or experts together. If success criteria 3–4 fail, a follow-up spec may add a tiny trunk-only alignment pass. It is not an undocumented flag in v1.

---

## 7. Data

Byte-level tokenizer: UTF-8 bytes, `vocab_size=256`. No BOS/EOS required; language-model windows are contiguous streams. `stoi`/`itos` are identity maps `0..255`.

### 7.1 CI corpora (no network)

Generated in-process with a seed:

- `synth_code` — tokens drawn from a Python-ish alphabet: `def`, `return`, `():`, indent spaces, identifiers `[a-z_]`, brackets. Length 50_000 bytes.
- `synth_math` — digits, `+-*/=()`, short equations `12+7=19` style. Length 50_000 bytes.
- `synth_fill` — lowercase words and spaces (filler English-ish). Length 20_000 bytes.
- Mixed pretrain = concatenation of all three, shuffled by **chunk** (256-byte chunks) so the trunk sees both domains.
- Holdouts: last 10% of each domain stream, never used in training.

Specialization on CI is a **plumbing smoke**, not a scientific claim.

### 7.2 Scientific corpora

`scripts/prepare.py` writes `data/cache/` (gitignored). It never runs during pytest.

| Split | Source order |
|-------|----------------|
| wiki | `SEQMOE_WIKI` or `/games/ailab/lab/scrt-lab/data/wiki.train.raw` (required). Fail if missing. |
| code | `data/cache/code.txt` if present; else harvest `*.py` from `/games/ailab/lab/scrt-lab`, `envelope-lab`, `ailab-cli` (skip `.venv`, `__pycache__`). Set `code_source=lab_python` and `scientific_grade=weak` when harvested. |
| math | `data/cache/math.txt` if present (GSM8K questions+answers and/or other math text, pre-dumped). Else generate a larger arithmetic+word-problem stream (100k examples of multi-digit ops and “mean/sum of” templates) and set `math_source=fallback`, `scientific_grade=weak`. |

Router mix: batches sampled with probabilities `code=0.4, math=0.4, wiki=0.2`. Domain id is stored on the batch for hard-route eval and optional `L_route`, not fed as an extra token.

Cross-domain templates (eval only), 512 sequences, e.g.:

- `def mean(xs):\n    return sum(xs)/len(xs)`
- `# compute 17*23 in python\nprint(17*23)`

These are scored as NLL, not execution.

If wiki is missing, `run_poc.py --scientific` exits with a message to copy or point `SEQMOE_WIKI`. It does not silently fall back to synth.

### 7.3 Batching

Copy scrt-lab `TokenBatcher`: random slices of length `seq_len` from a 1D `LongTensor` stream. `x = buf[i:i+T]`, `y = buf[i+1:i+T+1]`.

---

## 8. Config

```python
@dataclass
class ModelConfig:
    vocab_size: int = 256
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 64
    dropout: float = 0.0
    n_experts: int = 2
    top_k: int = 2
    mlp_mult: int = 4

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_experts < 2:
            raise ValueError("n_experts must be >= 2")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError("top_k must be in [1, n_experts]")

@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    seq_len: int = 64
    lr: float = 3e-3
    weight_decay: float = 0.01
    pretrain_steps: int = 200
    expert_steps: int = 200
    router_steps: int = 200
    eval_every: int = 50
    log_every: int = 25
    device: str = "cuda:1"
    expert_init: str = "clone"          # clone | random
    balance_coef: float = 0.01
    route_coef: float = 0.0
    max_vram_mb: int = 22_000
    results_dir: str = "results"
    run_name: str = "poc"
    scientific: bool = False

def resolve_device(device: str) -> str:
    if device == "cuda":
        return "cuda:1"
    return device

def tiny_config() -> ExperimentConfig:
    """CPU/pytest and GPU plumbing."""
    ...

def scientific_config() -> ExperimentConfig:
    """GPU1 scientific run."""
    # n_embd=256, n_head=8, n_layer=8, block_size=256,
    # batch_size=16, seq_len=256, lr=8e-4,
    # pretrain_steps=5000, expert_steps=3000, router_steps=2000
```

`tiny_config` must run on CPU in pytest in under ~30s for the full sequential smoke (short steps: 20/20/20).

---

## 9. Memory plan

Copied in spirit from envelope-lab: **plan, then allocate; abort rather than swap.**

At the start of each stage, `memory.plan(model, stage, device)` computes:

```text
param_bytes     = sum(p.numel()*p.element_size() for p in model.parameters())
grad_bytes      = same, for requires_grad params
optim_bytes     = 2 * grad_bytes          # AdamW m,v  (fp32)
activation_est  = batch * seq * n_embd * n_layer * 4 * 12   # rough residual+attn
cuda_margin     = 512 * 1024 * 1024
total           = param + grad + optim + activation_est + cuda_margin
```

On CUDA, if `total > min(max_vram_mb*1024^2, torch.cuda.mem_get_info()[1] - already_used)` → raise `MemoryPlanError` with the breakdown.

Swap (`/proc/meminfo SwapTotal`) is not added to capacity.

During `expert` stage, assert:

- `resident_expert_count == 1`
- `optimizer_numel == expert_mlp_numel`
- no parameter named for a non-resident logical id exists in `state_dict()`

`scripts/run_memory.py` trains tiny sequential vs tiny **joint** (both experts trainable, one optimizer) for a few steps and writes `peak_vram_mb` / `optimizer_numel` for both. This is the VRAM-savings evidence. Joint training is **not** a quality arm of the scientific run; it exists to measure memory.

---

## 10. Evaluation

Same holdouts for every arm. JSON under `results/{run_name}_{arm}.json`.

| Arm | Definition |
|-----|------------|
| `trunk` | Frozen dense pretrain checkpoint |
| `dense_ft_mix` | Trunk MLP fine-tuned `expert_steps` on mixed code+math (active-param match) |
| `dense_ft_code` / `dense_ft_math` | MLP fine-tuned on one domain |
| `expert_only_code` / `expert_only_math` | Assembled model, `force_expert` |
| `hard_route` | Gold domain id for every token. **Only reported on `code` and `math` splits.** `wiki` has no gold expert. `mixed` / `cross` report `seqmoe_top1`/`top2` only; `gold_agree` uses sequences that still carry a domain id and ignores wiki-origin sequences |
| `random_route` | Uniform per token |
| `seqmoe_top1` / `seqmoe_top2` | Learned router |

Metrics per eval split (`code`, `math`, `wiki`, `mixed`, `cross`):

- `nll` (nats/token, mean over tokens)
- `bpb` = `nll / ln(2)` (byte-level, so bits/byte)

Router metrics on domain holdouts:

- `route_frac` per expert
- `entropy` of mean routing distribution
- `gold_agree` = fraction of tokens whose top-1 matches gold domain

Resource metrics:

- `peak_vram_mb`, `optimizer_numel`, `resident_expert_count`, `wall_s`, `steps`, `tokens_seen`

**Fairness (from scrt-lab):** sequential MoE uses `pretrain_steps + 2*expert_steps + router_steps` wall. `dense_ft_mix` uses `pretrain_steps + expert_steps`. Report both. Do not claim a win that is only extra steps.

A result is **scientific_grade=full** only when wiki, code, and math sources are real dumps (`code_source` and `math_source` not fallback/harvest). Otherwise `weak` and README must say so.

---

## 11. Tests (CPU, pytest)

| Test | Assert |
|------|--------|
| `test_clone_equals_trunk_mlp` | After clone, max abs diff vs trunk MLP is 0 |
| `test_expert_mode_trainable_only_active_mlp` | `requires_grad` names are exactly `blocks.*.mlp.*` of one id |
| `test_expert1_does_not_allocate_expert0` | After `set_mode("expert", 1)`, `"0"` not in any `ExpertBank.experts` |
| `test_save_load_expert_roundtrip` | Reloaded MLP weights equal; metadata `logical_id` preserved |
| `test_assemble_rejects_hash_mismatch` | Flip one trunk attn weight, save expert, assemble raises `TrunkHashMismatch` |
| `test_assemble_rejects_wrong_n_embd` | Config mismatch raises `ExpertConfigMismatch` |
| `test_router_topk_sparsity` | With k=1, `route_frac` is one-hot per token; k=2 at most two nonzero experts per token |
| `test_sequential_independence` | Train expert 0 a few steps, save; reload trunk, train expert 1; expert 0 file bytes unchanged; trunk hash unchanged |
| `test_synth_specialization_smoke` | 20-step expert trains: expert 0 code NLL < expert 0 math NLL **or** code NLL dropped vs init (allow weak signal; fail only if code NLL rose and math dropped) |
| `test_optimizer_numel_sequential_vs_joint` | Joint AdamW expert-param count is 2× sequential for `n_experts=2` |
| `test_memory_plan_cpu` | Plan dict has the keys above; does not require CUDA |
| `test_tied_lm_head` | `lm_head.weight.data_ptr() == tok_emb.weight.data_ptr()` |

No GPU required for pytest. `run_poc.py --tiny` is the GPU optional smoke.

---

## 12. Repository layout

```text
/games/ailab/lab/seqmoe-lab/
  README.md
  pyproject.toml
  seqmoe/
    __init__.py
    config.py          # ModelConfig, TrainConfig, resolve_device, tiny/scientific
    model.py           # MLP, CausalSelfAttention, Block, ExpertBank, Router, SeqMoE
    data.py            # bytes tokenizer, synth corpora, TokenBatcher, loaders
    assembly.py        # trunk hash, save/load trunk+expert, assemble
    train.py           # pretrain, train_expert, train_router, eval_loss
    memory.py          # plan, peak tracker, residency asserts
    eval.py            # arms, metrics, write JSON
    errors.py          # TrunkHashMismatch, ...
  tests/
    test_model.py
    test_assembly.py
    test_train_smoke.py
    test_memory.py
    test_eval_arms.py
  scripts/
    prepare.py
    run_poc.py
    run_memory.py
  data/                # gitignored cache; synth is generated
  results/             # gitignored json
  docs/superpowers/specs/2026-08-30-sequential-moe-poc-design.md
```

`pyproject.toml`: `torch`, `numpy`, `pytest` (dev). No HuggingFace dependency. Python ≥ 3.10. `uv` optional, same as scrt-lab.

Default device `cuda:1`. `CUDA_VISIBLE_DEVICES` still respected if the caller sets `device=cuda`.

---

## 13. Scripts

### `scripts/run_poc.py`

```text
python scripts/run_poc.py --tiny                  # synth, tiny_config, CPU or --device
python scripts/run_poc.py --scientific --device cuda:1
```

Runs A → B → C → eval arms. Writes `results/`. Exits nonzero if assembly raises or memory plan fails.

### `scripts/run_memory.py`

Tiny sequential vs joint, prints a two-row table: `optimizer_numel`, `peak_vram_mb` (0 on CPU), `resident_expert_count`.

### `scripts/prepare.py`

Only needed for `--scientific`. Copies/symlinks wiki; builds code/math cache as in §7.2; prints `code_source`, `math_source`, `scientific_grade`.

---

## 14. Error handling

| Condition | Behavior |
|-----------|----------|
| Wiki missing on `--scientific` | Exit 2, print path help |
| Expert file hash mismatch | Raise `TrunkHashMismatch`, no partial assemble |
| Memory plan exceeds envelope | Raise `MemoryPlanError` with byte breakdown |
| `set_mode("expert")` without `expert_id` | `ValueError` |
| `top_k > resident experts` | `ValueError` |
| CUDA OOM after a passing plan | Let it crash; do not catch and retry with swap |

---

## 15. Key decisions

1. **New lab, not scrt-lab.** scrt-lab’s archive is a negative result about merge/free. Sequential MoE is a different claim. Copy block/train/JSON patterns; do not extend `ResidualSlotBank`.
2. **Lab-pretrained trunk, not HuggingFace.** From-scratch at lab scale. Shared representation comes from mixed pretrain, not a downloaded 0.5B.
3. **Cloned full FFNs, not LoRA.** Sequential VRAM savings must show up in optimizer states of real MLPs. LoRA would make the memory win too cheap to be interesting.
4. **Clone trunk FFN, not random, not previous expert.** Random-init is an ablation flag only.
5. **Fully frozen trunk during specialist and router training.** Avoids Expert B seeing a different backbone than Expert A.
6. **Router last, experts frozen.** No joint unfreeze in v1.
7. **Token-level Top-2 per layer.** Cross-domain handled by Top-K, not sequential expert passes.
8. **Byte tokenizer, vocab 256.** Matches scrt-lab’s char-cap lesson without a BPE dependency.
9. **Logical ids in config.** File names and `logical_id` fields agree (`0=code`, `1=math`).
10. **Reuse other repos as patterns and optional corpora, not as libraries.**
11. **Two data grades.** Synth CI vs scientific. Weak scientific (harvested lab Python, fallback math) is labeled, not silently published as evidence.
12. **GPU 1 default, 22 GB envelope.** Same box law as envelope-lab / scrt-lab.

---

## 16. Later sub-projects (not this spec)

2. Hardware-aware router terms (switch / cache-miss), simulable without a pager
3. Colibri/llama.cpp paging: `acquire_expert(layer, logical_id)` leases, NVMe→RAM→VRAM
4. Scale-up: larger FFNs, more experts, optional QLoRA on Qwen3.8-27B

---

## 17. Experimental unknowns (results, not implementation forks)

These stay questions for the PoC to **measure**. v1 already picked a side so the code is determinate:

| Unknown | v1 choice | What we measure |
|---------|-----------|-----------------|
| How independent can expert training be? | Fully independent, frozen trunk | Criteria 3–4 |
| Clone vs random FFN? | Clone | Optional `--expert-init random` ablation |
| Must the backbone stay frozen? | Yes | If quality fails, a follow-up may unfreeze later |
| Is router-only enough? | Router-only | If not, spec a joint pass later |
| Token vs sequence routing? | Token | Sequence hard-route is an eval arm |
| Cache-aware routing? | Off | Sub-project 2 |
| Cross-domain multi-expert? | Top-2 | Top-1 vs Top-2 on `cross` split |
| How much overlap? | Uncontrolled | `gold_agree` and `route_frac` |

---

## 18. Implementation order (for the plan, not extra scope)

1. Package skeleton + `ModelConfig` + dense GPT + pytest
2. `ExpertBank` residency + clone + mode switching
3. Hash + save/load + assemble rejections
4. Synth data + sequential train loop + independence test
5. Router Top-K + load-balance + router train
6. Eval arms + JSON
7. Memory plan + sequential vs joint script
8. `prepare.py` + `--scientific` path
9. README with how to run and how to read `scientific_grade`
