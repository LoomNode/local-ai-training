# STE-QAT De-confounding Control Arm — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an STE-QAT control arm (FP master weight + Adam, ratchet's exact quantizer) so the ratchet's quality gap can be split into "few states" vs "master-weight-free", then run it at 25M/text8 and report the 2×2 decomposition.

**Architecture:** A new `QATLinear` (FP32 master `weight`, forward quantized with the ratchet's per-row `row_max_abs/max_code` quantizer via a straight-through estimator) plugs into the existing `_linear` factory behind a `ModelConfig.qat` flag and a new `weight_mode="qat"`. QAT is a control with master weights — deliberately not a `DiscreteRatchetLinear`, so it sits outside `audit_no_master_weights`' scope. The experiment reuses the existing FP32+ratchet `text8-25m` arms (gated by a reproduce-check) and runs only QAT{5,7,9}.

**Tech Stack:** Python, PyTorch, the `local_ai_training` package, `uv`, `lat` CLI.

## Global Constraints

- Environment prefix for all commands: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache`. Pick an idle GPU with `nvidia-smi` and set `CUDA_VISIBLE_DEVICES`.
- The QAT quantizer must match the ratchet exactly: per-row `scale = (weight.abs().amax(dim=1)/max_code).clamp_min(torch.finfo(torch.float32).eps)` (NOTE: `finfo.eps`, not `tiny`), `code = round(weight/scale).clamp(-max_code, max_code)`, `effective = code*scale`. Init is `nn.init.kaiming_uniform_(weight, a=5**0.5)` — identical to `nn.Linear` and to the ratchet's logical reference (ratchet.py:307-308).
- Gradient is pure straight-through (round passes identity; no gradient clamp on saturated entries), matching BitNet b1.58.
- `audit_no_master_weights` only inspects `DiscreteRatchetLinear`; ratchet arms must still audit clean. QAT models legitimately have master weights and are outside that scope.
- Preserve existing runs: write QAT runs to `runs/text8-25m-qat/`, never overwrite `runs/text8-25m/`.
- text8 is pinned by sha256 (`data.py`), already compliant.
- `lat` entry point: `uv run lat ...`. Single-seed (1337) trainability/attribution test, not a converged-scale claim — note it in the writeup.

---

### Task 1: `QATLinear` quantization-aware control linear

**Files:**
- Create: `src/local_ai_training/qat.py`
- Test: `tests/test_qat.py`

**Interfaces:**
- Produces: `QATLinear(in_features: int, out_features: int, *, max_code: int, initial_weight: Tensor | None = None)`; attribute `weight: nn.Parameter` (fp32, shape `(out_features, in_features)`); method `quantized_weight() -> Tensor` (the STE effective weight, forward value `code*scale`); `forward(inputs) -> Tensor`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_qat.py
import torch
from torch import nn

from local_ai_training.qat import QATLinear
from local_ai_training.ratchet import DiscreteRatchetLinear


def _ratchet_effective(layer: DiscreteRatchetLinear) -> torch.Tensor:
    return layer.code.to(torch.float32) * layer.scale.to(torch.float32)[:, None]


def test_qat_quantized_weight_matches_ratchet_quantizer_bit_exact() -> None:
    torch.manual_seed(0)
    reference = torch.empty(12, 20, dtype=torch.float32)
    nn.init.kaiming_uniform_(reference, a=5**0.5)
    for max_code in (2, 3, 4):
        qat = QATLinear(20, 12, max_code=max_code, initial_weight=reference)
        ratchet = DiscreteRatchetLinear.from_reference(reference, max_code=max_code)
        assert torch.equal(qat.quantized_weight().detach(), _ratchet_effective(ratchet))


def test_qat_init_matches_kaiming_uniform_like_nn_linear() -> None:
    # Same seed/RNG draw as nn.Linear -> shared logical FP init across arms.
    torch.manual_seed(7)
    qat = QATLinear(20, 12, max_code=2)
    torch.manual_seed(7)
    ref = nn.Linear(20, 12, bias=False)
    assert torch.equal(qat.weight, ref.weight)


def test_qat_ste_gradient_reaches_master_including_saturated() -> None:
    torch.manual_seed(1)
    qat = QATLinear(8, 4, max_code=2)
    # Force saturation: large weights so |weight/scale| > max_code for many entries.
    with torch.no_grad():
        qat.weight.mul_(50.0)
    x = torch.randn(6, 8)
    qat(x).pow(2).sum().backward()
    assert qat.weight.grad is not None
    assert torch.count_nonzero(qat.weight.grad) > 0


def test_qat_forward_equals_linear_with_quantized_weight() -> None:
    torch.manual_seed(2)
    qat = QATLinear(10, 5, max_code=3)
    x = torch.randn(7, 10)
    expected = torch.nn.functional.linear(x, qat.quantized_weight())
    assert torch.equal(qat(x), expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_qat.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'local_ai_training.qat'`.

- [ ] **Step 3: Implement `QATLinear`**

```python
# src/local_ai_training/qat.py
"""STE-QAT control linear.

A quantization-aware-training control: keeps a full-precision master weight (trained by
Adam) and quantizes it in the forward pass with the ratchet's exact per-row quantizer,
using a straight-through estimator. This is deliberately NOT a DiscreteRatchetLinear — it
HAS master weights and exists only as the control that isolates the cost of dropping them.
It is therefore outside audit_no_master_weights' scope.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class QATLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        max_code: int,
        initial_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        if max_code not in (2, 3, 4):
            raise ValueError("max_code must be 2 (quinary), 3 (septenary), or 4 (nonary)")
        self.in_features = in_features
        self.out_features = out_features
        self.max_code = max_code
        if initial_weight is None:
            weight = torch.empty(out_features, in_features, dtype=torch.float32)
            nn.init.kaiming_uniform_(weight, a=5**0.5)
        else:
            if initial_weight.shape != (out_features, in_features):
                raise ValueError(
                    f"initial_weight must have shape {(out_features, in_features)}, "
                    f"got {tuple(initial_weight.shape)}"
                )
            weight = initial_weight.detach().to(dtype=torch.float32).clone()
        self.weight = nn.Parameter(weight)

    def quantized_weight(self) -> Tensor:
        # Matches DiscreteRatchetLinear's quantizer (ratchet.py:317-319): per-row
        # row_max_abs/max_code scale (clamped to finfo.eps), round-to-code, dequantize.
        scale = (
            self.weight.detach().abs().amax(dim=1, keepdim=True) / self.max_code
        ).clamp_min(torch.finfo(torch.float32).eps)
        code = torch.round(self.weight / scale).clamp(-self.max_code, self.max_code)
        # Straight-through: forward value is code*scale; gradient is identity to weight,
        # including saturated entries (the whole quantized term is detached).
        return self.weight + (code * scale - self.weight).detach()

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.quantized_weight())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_qat.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint and commit**

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run ruff check src/local_ai_training/qat.py tests/test_qat.py
git add src/local_ai_training/qat.py tests/test_qat.py
git commit -m "feat: QATLinear STE control (ratchet's exact quantizer, FP master + STE)"
```

---

### Task 2: Wire `weight_mode="qat"` through config, model, train, CLI

**Files:**
- Modify: `src/local_ai_training/model.py` (add `ModelConfig.qat` field; `_linear` branch; import `QATLinear`)
- Modify: `src/local_ai_training/train.py` (accept `qat` in validation; set `qat=True` on model config when `weight_mode=="qat"`)
- Modify: `src/local_ai_training/cli.py` (add `"qat"` to `--weight-mode` choices)
- Test: `tests/test_experiment.py` (qat trains; audit scope)

**Interfaces:**
- Consumes: `QATLinear` from Task 1; existing `train_run(..., weight_mode=...)`, `build_seeded_model`, `audit_no_master_weights`.
- Produces: `weight_mode="qat"` end-to-end; `ModelConfig.qat: bool` (default `False`).

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_experiment.py
def test_qat_arm_trains_and_reduces_loss(tmp_path: Path) -> None:
    corpus = build_char_corpus("abcd" * 400)
    result = train_run(
        corpus=corpus,
        config=small_experiment_config(),
        max_code=2,
        seed=7,
        run_dir=tmp_path / "qat",
        weight_mode="qat",
    )
    rows = list(csv.DictReader(result.metrics_csv.open()))
    assert result.final_validation_loss < float(rows[0]["validation_loss"])


def test_qat_model_has_master_weights_outside_ratchet_audit(tmp_path: Path) -> None:
    from dataclasses import replace

    from local_ai_training.model import build_seeded_model
    from local_ai_training.ratchet import audit_no_master_weights

    mc = small_experiment_config().model_config(vocab_size=16)
    qat_model = build_seeded_model(replace(mc, qat=True), max_code=2, seed=7)
    ratchet_model = build_seeded_model(mc, max_code=2, seed=7)

    qat_report = audit_no_master_weights(qat_model)
    ratchet_report = audit_no_master_weights(ratchet_model)
    # QAT has no DiscreteRatchetLinear layers -> outside audit scope, no violations.
    assert qat_report.ratchet_layers == 0
    assert qat_report.violations == ()
    # Ratchet arm still clean, and is actually seen by the audit.
    assert ratchet_report.ratchet_layers > 0
    assert ratchet_report.violations == ()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_experiment.py -k qat -q`
Expected: FAIL — `train_run` rejects `weight_mode="qat"` (ValueError) and `ModelConfig` has no `qat` field.

- [ ] **Step 3: Add `ModelConfig.qat` and the `_linear` branch**

In `src/local_ai_training/model.py`, add the import near the existing ratchet import:

```python
from .qat import QATLinear
```

Add the field to `ModelConfig` (immediately after the `matmul_mode` field):

```python
    qat: bool = False
```

Replace `_linear` (currently lines 54-68) with:

```python
def _linear(config: ModelConfig, in_features: int, out_features: int, max_code: int | None):
    if max_code is None:
        return nn.Linear(in_features, out_features, bias=False)
    if config.qat:
        return QATLinear(in_features, out_features, max_code=max_code)
    return DiscreteRatchetLinear(
        in_features,
        out_features,
        max_code=max_code,
        pressure_threshold=config.pressure_threshold,
        bucket_low=config.bucket_low,
        bucket_high=config.bucket_high,
        trainable_scale=config.trainable_scale,
        compile_update=config.compile_update,
        matmul_mode=config.matmul_mode,
        fuse_backward_update=True,
    )
```

- [ ] **Step 4: Thread `weight_mode="qat"` through `train_run`**

In `src/local_ai_training/train.py`, add to the imports at top:

```python
from dataclasses import replace
```

Replace the validation block (currently lines 120-125) with:

```python
    if weight_mode not in {"ratchet", "frozen", "fp32", "qat"}:
        raise ValueError("weight_mode must be ratchet, frozen, fp32, or qat")
    if weight_mode == "fp32" and max_code is not None:
        raise ValueError("fp32 mode requires max_code=None")
    if weight_mode != "fp32" and max_code not in (2, 3, 4):
        raise ValueError("ratchet, frozen, and qat modes require max_code 2, 3, or 4")
```

Replace the model-build line (currently around line 134-136) with:

```python
    model_config = config.model_config(vocab_size=len(corpus.vocabulary))
    if weight_mode == "qat":
        model_config = replace(model_config, qat=True)
    model = build_seeded_model(model_config, max_code=max_code, seed=seed).to(device)
```

(The training loop needs no change: qat falls into the existing non-ratchet branch, where `discard_pending_gradients()` is a no-op because the model has no `DiscreteRatchetLinear` layers, and `optimizer.step()` updates the `QATLinear` masters — they are ordinary `nn.Parameter`s in `model.parameters()`.)

- [ ] **Step 5: Add the CLI choice**

In `src/local_ai_training/cli.py`, change the `--weight-mode` choices (line 36-37) from `choices=("ratchet", "frozen", "fp32")` to:

```python
        choices=("ratchet", "frozen", "fp32", "qat"),
```

- [ ] **Step 6: Run the qat tests, then the full suite**

Run: `CUDA_VISIBLE_DEVICES=<idle> UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_experiment.py -k qat -q`
Expected: PASS (2 tests).
Run: `CUDA_VISIBLE_DEVICES=<idle> UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest -q`
Expected: PASS (full suite).

- [ ] **Step 7: Audit + lint + commit**

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run lat audit --model configs/ratchet_tiny.toml
UV_CACHE_DIR=/games/ailab/.uv-cache uv run ruff check src/local_ai_training/model.py src/local_ai_training/train.py src/local_ai_training/cli.py tests/test_experiment.py
git add src/local_ai_training/model.py src/local_ai_training/train.py src/local_ai_training/cli.py tests/test_experiment.py
git commit -m "feat: weight_mode=qat control arm (QATLinear behind ModelConfig.qat)"
```

---

### Task 3: Reproduce-check — validate reuse of the existing arms

**Files:**
- Create: `runs/repro-check/` (git-ignored run output; not committed)

**Interfaces:**
- Consumes: `lat train` with `weight_mode=ratchet`; the stored `runs/text8-25m/quinary/seed-1337/metrics.csv`.

- [ ] **Step 1: Download text8 (idempotent)**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run lat dataset --which text8`
Expected: prints the dataset path `data/text8/text8`.

- [ ] **Step 2: Make a short-budget copy of the 30k config**

Create `configs/repro_check_quinary.toml` as a copy of `configs/scaleup_text8_25m_30k.toml` with `steps = 400` (leave everything else identical: block_size 256, n_layer 8, n_head 8, n_embd 512, batch_size 64, eval_interval 200, eval_batches 40, support_learning_rate 0.0003, seeds [1337], pressure_threshold 8).

- [ ] **Step 3: Run ratchet-quinary fresh for 400 steps**

Run:
```bash
CUDA_VISIBLE_DEVICES=<idle> MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache \
  uv run lat train --config configs/repro_check_quinary.toml --dataset-path data/text8/text8 \
  --codes 5 --weight-mode ratchet --seed 1337 --output runs/repro-check/quinary
```

- [ ] **Step 4: Compare against the stored trajectory (decision gate)**

Run:
```bash
python3 - <<'PY'
import csv
new = list(csv.DictReader(open("runs/repro-check/quinary/metrics.csv")))
old = list(csv.DictReader(open("runs/text8-25m/quinary/seed-1337/metrics.csv")))
oldmap = {int(r["step"]): float(r["validation_loss"]) for r in old}
for r in new:
    s = int(r["step"])
    if s in oldmap:
        nv = float(r["validation_loss"])
        print(f"step {s}: new={nv:.6f} old={oldmap[s]:.6f} dabs={abs(nv-oldmap[s]):.2e}")
PY
```
Expected: at steps 0, 200, 400 the new and old validation losses match to < 1e-3 (the eager fp32+ratchet path is deterministic). **If they match → proceed to Task 4 reusing the existing FP32+ratchet arms. If they diverge materially (> 1e-2) → the eager path drifted; re-run all seven arms (FP32 + ratchet{5,7,9}) fresh into `runs/text8-25m-qat/` alongside the QAT arms before analysis, and note the drift in the results doc.**

- [ ] **Step 5: Commit the repro config**

```bash
git add configs/repro_check_quinary.toml
git commit -m "test: short-budget config for the QAT reuse reproduce-check"
```

---

### Task 4: Run QAT{5,7,9} and write the 2×2 decomposition

**Files:**
- Create: `runs/text8-25m-qat/qat{5,7,9}/seed-1337/` (git-ignored)
- Create: `docs/results/2026-06-22-qat-deconfounding.md`

**Interfaces:**
- Consumes: `lat train --weight-mode qat`; existing arm metrics in `runs/text8-25m/`.

- [ ] **Step 1: Run the three QAT arms (≈30 min each on an idle 3090)**

```bash
for codes in 5 7 9; do
  CUDA_VISIBLE_DEVICES=<idle> MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache \
    uv run lat train --config configs/scaleup_text8_25m_30k.toml --dataset-path data/text8/text8 \
    --codes $codes --weight-mode qat --seed 1337 \
    --output runs/text8-25m-qat/qat${codes}/seed-1337
done
```
Expected: each writes `metrics.csv` reaching step 30000 (or near it) with a descending validation loss.

- [ ] **Step 2: Collect best validation loss per arm**

```bash
python3 - <<'PY'
import csv, pathlib
def best(p):
    rows = list(csv.DictReader(open(p)))
    return min(float(r["validation_loss"]) for r in rows)
arms = {
  "FP32": "runs/text8-25m/fp32/seed-1337/metrics.csv",
  "ratchet5": "runs/text8-25m/quinary/seed-1337/metrics.csv",
  "ratchet7": "runs/text8-25m/septenary/seed-1337/metrics.csv",
  "ratchet9": "runs/text8-25m/nonary/seed-1337/metrics.csv",
  "qat5": "runs/text8-25m-qat/qat5/seed-1337/metrics.csv",
  "qat7": "runs/text8-25m-qat/qat7/seed-1337/metrics.csv",
  "qat9": "runs/text8-25m-qat/qat9/seed-1337/metrics.csv",
}
b = {k: best(v) for k, v in arms.items() if pathlib.Path(v).exists()}
for k, v in b.items():
    print(f"{k:10s} {v:.4f}")
for s, lvl in ((5,"5"),(7,"7"),(9,"9")):
    fp, q, r = b["FP32"], b[f"qat{s}"], b[f"ratchet{s}"]
    print(f"states={lvl}: FP32->QAT (few-states cost) = {q-fp:+.4f} | "
          f"QAT->ratchet (master-weight-free cost) = {r-q:+.4f}")
PY
```

- [ ] **Step 3: Write the results note**

Create `docs/results/2026-06-22-qat-deconfounding.md` with: method (the 2×2, matched quantizer, shared init, iso-everything, reuse + reproduce-check outcome), a table of all seven best val losses, and the per-state decomposition (FP32→QAT = cost of few states; QAT→ratchet = cost of master-weight-free). State the headline attribution: which lever (few states vs master-weight-free) owns the gap, and whether it varies across 5/7/9. Note the limitations: single seed (1337), 25M/30k — a trainability/attribution test, not a converged-scale claim; if the QAT→ratchet gap is small, a multi-seed repeat is the follow-up.

- [ ] **Step 4: Commit the results**

```bash
git add docs/results/2026-06-22-qat-deconfounding.md
git commit -m "docs: QAT de-confounding result — attribute the ratchet gap (few states vs master-weight-free)"
```

---

## Self-Review

**Spec coverage:** QAT quantizer matched to ratchet (Task 1) ✓; `weight_mode="qat"` wiring incl. audit scope (Task 2) ✓; shared init via kaiming_uniform (Task 1 test) ✓; reuse + reproduce-check gate (Task 3) ✓; QAT{5,7,9} runs + 2×2 results (Task 4) ✓; CLI exposure (Task 2 Step 5) ✓; preserve runs / separate output dir (Task 4 paths) ✓; pure straight-through incl. saturated (Task 1 test) ✓.

**Placeholder scan:** `<idle>` is the GPU index the executor selects from `nvidia-smi` (intentional runtime value, not a code placeholder). All code blocks are complete.

**Type consistency:** `QATLinear(in, out, *, max_code, initial_weight=None)`, `.weight`, `.quantized_weight()`, `ModelConfig.qat`, `weight_mode="qat"` used consistently across tasks.
