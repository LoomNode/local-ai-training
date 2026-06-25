# Int8 Per-Token Speed Phase — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove (and, where a cheap bottleneck blocks it, unblock) that a master-weight-free int8-ratchet training step is wall-clock cheaper than a plain dense bf16 step at iso-fit width, reusing the width-512 convergence parity for quality.

**Architecture:** Four phases. (1) Relabel the existing dense mode in the throughput harness as the honest baseline and record the current CPU-bound state. (2) Defer the ratchet's per-step `.item()` metric syncs so the GPU stops being launch-serialized — TDD equivalence-gated — and verify the int8 GEMM. (3) Run the corrected sustained sweep 512→4096 and report the per-token crossover + the bf16-dense OOM frontier. (4) Record results and correct the stale docs.

**Tech Stack:** Python, PyTorch (eager + autocast), Triton (autotuned int8 GEMM already in-package), `uv`, pytest, single RTX 3090(s).

**Spec:** `docs/superpowers/specs/2026-06-24-int8-per-token-speed-design.md`

## Global Constraints

- `audit_no_master_weights` stays clean — no FP/BF16 Parameter mirroring a code matrix; the sync change touches only metric accounting, not stored state. Verify with `uv run lat audit --model configs/ratchet_tiny.toml`.
- Never claim packed sub-byte storage; the int8 path stores int8 codes, not 2.32/2.81-bit.
- Do not overwrite existing `runs/` (convergence runs, the six tiny-shakespeare arms). Benchmark artifacts go under git-ignored `runs/int8-throughput`.
- Every reported number names its **baseline kind** (dense vs ratchet) and **method** (sustained vs per-step-synced). Per-token claims and memory claims are stated separately.
- 3090-specific; record GPU model/UUID (`nvidia-smi -L`).
- Lint/format gate on changed files: `uv run ruff check <files>` clean, `git diff --check` clean (line-length 100).
- Work on a branch in `.worktrees/` (e.g. `feat/int8-per-token-speed`), not on `main` directly.
- Run all GPU commands with `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache` and pin the GPU via `CUDA_VISIBLE_DEVICES`. Each measured point must run on an **uncontended** GPU (`nvidia-smi` to pick an idle one).

---

## File Structure

- `scripts/int8_training_throughput.py` — Modify: relabel dense mode `fp32`→`dense_bf16`; honest report wording. The per-width throughput microbench (one width, modes as subprocesses).
- `src/local_ai_training/ratchet.py` — Modify: `RatchetUpdateStats` gains a `materialize()`; `apply_normalized_gradient` stops calling `.item()` (keeps GPU tensors); `gradient_rms_mean` kept as a tensor until materialized.
- `src/local_ai_training/model.py` — Modify: `ratchet_update()` aggregation works on tensors-or-ints (unchanged `sum(...)` semantics) and returns an unmaterialized stats object.
- `src/local_ai_training/train.py` — Modify: keep the cumulative move counter on-GPU; `.item()` / `materialize()` only on the eval cadence.
- `tests/test_ratchet.py` — Modify: add the deferred-sync equivalence test.
- `docs/results/2026-06-24-int8-per-token-speed.md` — Create: the result note.
- `docs/results/2026-06-24-int8-training-throughput-final.md` — Modify: mark superseded.
- `docs/ROADMAP.md`, `docs/README.md` — Modify: correct the "speedup only ≥2048 / bf16 owns 512" claim.

---

### Task 1: Relabel the dense baseline + record the CPU-bound starting point

**Files:**
- Modify: `scripts/int8_training_throughput.py` (the `run_child` mode dispatch and `main` defaults/printing)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a `dense_bf16` mode name used by Tasks 3–4; `runs/int8-throughput/throughput.json` rows keyed by that mode.

**Context:** `step()` (line ~60) already wraps every mode in `torch.autocast(bfloat16)`, so the current `fp32` mode (`max_code = None`) is a dense model with bf16 matmuls + fp32 master + fp32 Adam — i.e. standard mixed-precision bf16 training. We rename it to stop implying pure fp32. The dispatch today is:
```python
    max_code = None if mode == "fp32" else 2
    matmul_mode = "fp32" if mode == "fp32" else mode
```

- [ ] **Step 1: Relabel the mode in `run_child`**

In `scripts/int8_training_throughput.py`, change the dispatch so `dense_bf16` is the dense mode:
```python
    is_dense = mode == "dense_bf16"
    max_code = None if is_dense else 2
    # Dense baseline runs nn.Linear under the step()'s bf16 autocast (mixed precision).
    # The ratchet arms use their matmul_mode (bf16 ratchet / int8 ratchet).
    matmul_mode = "bf16" if is_dense else mode
```
(`matmul_mode` is ignored when `max_code=None` — the dense `nn.Linear` path does not consult it — but set it to `bf16` so nothing downstream reads `"fp32"`.)

- [ ] **Step 2: Update defaults and the fp32-relative reporting in `main`**

Change the default modes and the speedup baseline (the report divides by the dense baseline, not a "fp32" key):
```python
    parser.add_argument("--modes", nargs="+", default=["dense_bf16", "bf16", "int8"])
```
and replace the `base = next(... r.get("mode") == "fp32" ...)` lookup and the `speedup_vs_fp32` key with `dense_bf16`:
```python
    base = next((r["tokens_per_second"] for r in results
                 if r.get("mode") == "dense_bf16" and "tokens_per_second" in r), None)
    if base:
        for r in results:
            if "tokens_per_second" in r:
                r["speedup_vs_dense"] = r["tokens_per_second"] / base
```
and in the print loop use `r.get("speedup_vs_dense", float("nan"))` with label `x{speedup:.3f} vs dense_bf16`.

- [ ] **Step 3: Add a one-line honest banner to the printed summary**

Before the per-row print loop in `main`, add:
```python
    print("baseline dense_bf16 = nn.Linear, fp32 master + fp32 Adam + bf16 autocast "
          "(standard mixed-precision bf16 training); arms are master-weight-free ratchets.")
```

- [ ] **Step 4: Lint the changed file**

Run: `uv run ruff check scripts/int8_training_throughput.py && git diff --check`
Expected: no errors.

- [ ] **Step 5: Record the CPU-bound starting point at two widths**

Pick an idle GPU with `nvidia-smi`. Run (substitute the idle index for `0`):
```bash
cd /games/ailab/local-ai-training
for W in 512 2048; do
  B=64; [ "$W" = 2048 ] && B=16
  CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache \
    uv run python scripts/int8_training_throughput.py --modes dense_bf16 bf16 int8 --embd $W --batch $B
done
```
Expected: each run prints three rows with `tok/s`, `ms/step`, `x… vs dense_bf16`, and the new banner. Save the two printed blocks into the task report verbatim — these are the **before** numbers for Task 3 and the CPU-bound evidence (int8 `ms/step` will be inflated by per-step `.item()` syncs; note the int8 `isolated-latency` vs `sustained` gap).

- [ ] **Step 6: Commit**

```bash
git add scripts/int8_training_throughput.py
git commit -m "bench: relabel dense baseline (dense_bf16) and report speedup vs dense"
```

---

### Task 2: Defer the ratchet's per-step `.item()` metric syncs

**Files:**
- Modify: `src/local_ai_training/ratchet.py` (`RatchetUpdateStats`, `apply_normalized_gradient`, and the `RatchetUpdateStats` construction at lines ~520 and ~544)
- Modify: `src/local_ai_training/model.py` (`ratchet_update`, lines ~235–250)
- Modify: `src/local_ai_training/train.py` (the per-step move counter + eval-cadence materialization)
- Test: `tests/test_ratchet.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `RatchetUpdateStats.materialize() -> RatchetUpdateStats` (all fields python scalars); `ratchet_update()` returns an *unmaterialized* `RatchetUpdateStats` whose count fields may be 0-d `torch.Tensor`; `code_moves` / `blocked_moves` properties return the same type as their summands.

**Context:** Today `apply_normalized_gradient` (ratchet.py ~544) builds `RatchetUpdateStats` via `int(positive.item())` ×4, and `apply_weight_gradient` (~518) sets `gradient_rms_mean = float(...item())`. Each `.item()` forces a GPU→CPU sync **every step**, serializing the pipeline. The fix keeps these as 0-d GPU tensors and only materializes (syncs) when a metrics row is written.

- [ ] **Step 1: Write the failing equivalence test**

Add to `tests/test_ratchet.py`:
```python
def test_deferred_stats_materialize_equals_eager_item():
    """Deferring the .item() sync must yield identical metric values."""
    import torch
    from local_ai_training.ratchet import DiscreteRatchetLinear

    torch.manual_seed(0)
    layer = DiscreteRatchetLinear(16, 8, max_code=2)
    grad = torch.randn(8, 16)
    normalized = grad / (grad.square().mean(dim=1, keepdim=True).sqrt() + 1e-8)

    stats = layer.apply_normalized_gradient(normalized.clone())
    # Unmaterialized: count fields are tensors (no host sync happened yet).
    assert isinstance(stats.positive_moves, torch.Tensor)
    m = stats.materialize()
    # Materialized: plain python ints/floats, equal to the eager .item() values.
    assert isinstance(m.positive_moves, int)
    assert m.positive_moves == int(stats.positive_moves.item())
    assert m.negative_moves == int(stats.negative_moves.item())
    assert m.blocked_positive_moves == int(stats.blocked_positive_moves.item())
    assert m.blocked_negative_moves == int(stats.blocked_negative_moves.item())
    assert m.code_moves == m.positive_moves + m.negative_moves
    assert m.total_weights == 8 * 16
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ratchet.py::test_deferred_stats_materialize_equals_eager_item -v`
Expected: FAIL — `apply_normalized_gradient` currently returns python ints, so `isinstance(stats.positive_moves, torch.Tensor)` fails, and `materialize` does not exist.

- [ ] **Step 3: Add `materialize()` to `RatchetUpdateStats`**

In `ratchet.py`, in the `RatchetUpdateStats` dataclass (after the `blocked_moves` property), add:
```python
    def materialize(self) -> "RatchetUpdateStats":
        """Return a copy with all fields as python scalars (forces the GPU->CPU sync).

        Count/rms fields may be 0-d tensors when produced by the no-sync update path;
        callers invoke this only when a metric value is actually needed (eval cadence).
        """
        def scalar(v):
            return v.item() if isinstance(v, torch.Tensor) else v
        return RatchetUpdateStats(
            total_weights=int(scalar(self.total_weights)),
            positive_moves=int(scalar(self.positive_moves)),
            negative_moves=int(scalar(self.negative_moves)),
            blocked_positive_moves=int(scalar(self.blocked_positive_moves)),
            blocked_negative_moves=int(scalar(self.blocked_negative_moves)),
            gradient_rms_mean=float(scalar(self.gradient_rms_mean)),
        )
```
(`torch` is already imported in `ratchet.py`.)

- [ ] **Step 4: Stop syncing in `apply_normalized_gradient`**

Replace the `int(positive.item())` ×4 construction (ratchet.py ~544) with the raw tensors:
```python
        return RatchetUpdateStats(
            total_weights=self.code.numel(),
            positive_moves=positive,
            negative_moves=negative,
            blocked_positive_moves=blocked_positive,
            blocked_negative_moves=blocked_negative,
            gradient_rms_mean=0.0,
        )
```
If `self._validate_state()` (called just above) performs any `.item()`/`bool()` host sync, gate it behind a cheap flag so the no-sync path stays sync-free; if it is pure tensor ops, leave it.

- [ ] **Step 5: Keep `gradient_rms_mean` a tensor in `apply_weight_gradient`**

At ratchet.py ~518, change:
```python
        rms_mean = float(gradient.float().square().mean(dim=1).sqrt().mean().item())
```
to keep it as a 0-d tensor:
```python
        rms_mean = gradient.float().square().mean(dim=1).sqrt().mean()
```
and pass `rms_mean` (the tensor) into the returned `RatchetUpdateStats(... gradient_rms_mean=rms_mean)`.

- [ ] **Step 6: Make `model.ratchet_update` tensor-safe**

In `model.py` (~241), `sum(update.positive_moves for update in updates)` already works for tensors (0-d tensors add), but `total_weights` must stay an int and the rms average must not divide a python int by tensor count incorrectly. Confirm the construction is:
```python
        total_weights = sum(update.total_weights for update in updates)  # ints -> int
        return RatchetUpdateStats(
            total_weights=total_weights,
            positive_moves=sum(update.positive_moves for update in updates),
            negative_moves=sum(update.negative_moves for update in updates),
            blocked_positive_moves=sum(update.blocked_positive_moves for update in updates),
            blocked_negative_moves=sum(update.blocked_negative_moves for update in updates),
            gradient_rms_mean=sum(update.gradient_rms_mean for update in updates) / len(updates),
        )
```
No code change is expected here if it already reads this way; the point is the returned stats are now *unmaterialized* (tensor-valued) and must not be `.item()`'d inside this method.

- [ ] **Step 7: Run the equivalence test**

Run: `uv run pytest tests/test_ratchet.py::test_deferred_stats_materialize_equals_eager_item -v`
Expected: PASS.

- [ ] **Step 8: Thread eval-cadence materialization through `train.py`**

In `train.py` the loop does `total_moves += update.code_moves` every step and builds metric rows only at the eval cadence. Keep the cumulative count on-GPU and materialize only when a row is written. Change the per-step accumulation to a 0-d tensor and materialize at the cadence:
```python
    total_moves_t = torch.zeros((), device=device, dtype=torch.long)
    ...
        update = model.ratchet_update()            # unmaterialized (tensor-valued)
        ...
        total_moves_t = total_moves_t + update.code_moves   # stays on GPU, no sync
        ...
        if step_index % config.eval_interval == 0 or step_index == config.steps:
            update_m = update.materialize()         # the only per-eval sync
            total_moves = int(total_moves_t.item())
            row = _metric_row(model, step=step_index, ..., update=update_m,
                              cumulative_code_moves=total_moves, ...)
```
For non-ratchet modes (`weight_mode != "ratchet"`) keep the existing `RatchetUpdateStats(0,0,0,0,0,0.0)` (already materialized ints) — `code_moves` is `0`, and `total_moves_t + 0` is fine. The `TrainResult.total_code_moves` returned at the end must `int(total_moves_t.item())` once.

- [ ] **Step 9: Run the train-loop and full ratchet tests**

Run: `uv run pytest tests/test_ratchet.py tests/test_experiment.py -q`
Expected: PASS (existing metric/cumulative-move assertions still hold — materialized values are identical).

- [ ] **Step 10: Audit + lint**

Run: `uv run lat audit --model configs/ratchet_tiny.toml` (expect `"violations": []`) and `uv run ruff check src/local_ai_training/ratchet.py src/local_ai_training/model.py src/local_ai_training/train.py tests/test_ratchet.py && git diff --check`.

- [ ] **Step 11: Re-measure the two widths (the "after")**

Run the same two-width command from Task 1 Step 5. Save the printed blocks. Compute and record `int8 ms/step` before (Task 1) vs after — the per-step sync removal should lower int8 `ms/step` and shrink the `isolated-latency` vs `sustained` gap.

- [ ] **Step 12: Commit**

```bash
git add src/local_ai_training/ratchet.py src/local_ai_training/model.py src/local_ai_training/train.py tests/test_ratchet.py
git commit -m "perf: defer ratchet metric .item() syncs to eval cadence (unblock the launch-bound step)"
```

---

### Task 3: Verify the int8 GEMM hits peak at frontier

**Files:**
- (Read-only verification; no source change expected) `src/local_ai_training/int8_matmul.py`, `scripts/int8_spike/width_sweep_bench.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a recorded "% of int8 peak" figure at width 4096 for the result note; a go/no-go on whether the GEMM needs work.

**Context:** `scaled_int8_mm` is already a hand-written `@triton.autotune` `tl.dot` int8 kernel. The tuned-kernel reversal (`docs/results/2026-06-21-int8-tuned-kernel-reversal.md`) showed the hand-written kernel reaches ~100% of int8 peak while vendor kernels stall at ~35%. This task confirms the *in-package* kernel is the fast one at the widths we benchmark, so the sweep's int8 numbers are not bottlenecked by a slow GEMM.

- [ ] **Step 1: Measure the bare GEMM at frontier and compare to peak**

Run on an idle GPU:
```bash
cd /games/ailab/local-ai-training
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache \
  uv run python scripts/int8_spike/width_sweep_bench.py
```
Expected: a table of `width | bf16 ms %pk | int8 ms %pk | spd`. Record the int8 `%pk` at width 4096 and 2048.

- [ ] **Step 2: Decide go/no-go**

If int8 `%pk` ≥ ~70% at frontier widths, the GEMM is healthy — record the numbers and proceed (no code change). If it is stalling (<~50%), STOP and escalate: a kernel fix is a separate spec, not part of this plan. Write the decision and the numbers into the task report.

- [ ] **Step 3: Commit (report only — no code)**

No source changed; nothing to commit. Record numbers in the task report for Task 5.

---

### Task 4: Full sustained sweep + crossover and OOM frontier

**Files:**
- (Uses `scripts/int8_training_throughput.py` from Task 1; no new source unless a thin sweep wrapper is desired — not required)

**Interfaces:**
- Consumes: the `dense_bf16` mode (Task 1), the sync-free ratchet (Task 2).
- Produces: the per-(width × mode) table, the crossover width, and the bf16-dense OOM width — consumed by Task 5.

- [ ] **Step 1: Run the sweep across all widths**

On an idle GPU, run each width with a batch that fits; record OOM where it occurs (the script prints `status: ERROR` rows on OOM rather than crashing the sweep):
```bash
cd /games/ailab/local-ai-training
for WB in "512 64" "1024 32" "2048 16" "4096 8"; do
  set -- $WB
  echo "### width $1 batch $2 ###"
  CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache \
    uv run python scripts/int8_training_throughput.py --modes dense_bf16 bf16 int8 --embd $1 --batch $2
done
```
Expected: four blocks. `dense_bf16` is expected to OOM at width 4096 (fp32 Adam state); record that as the memory-frontier datum. Save every block verbatim.

- [ ] **Step 2: Extract the crossover and OOM frontier**

From the saved blocks, build the table (width × {dense_bf16, int8} → ms/step, tok/s, peak_MB) and identify: (a) the smallest width where `int8` ms/step ≤ `dense_bf16` ms/step (the per-token crossover), and (b) the smallest width where `dense_bf16` OOMs while `int8` runs (the memory frontier). Write both into the task report.

- [ ] **Step 3: Commit (artifacts only)**

`runs/int8-throughput/` is git-ignored; nothing to commit. The numbers live in the task report for Task 5.

---

### Task 5: Result note + doc corrections

**Files:**
- Create: `docs/results/2026-06-24-int8-per-token-speed.md`
- Modify: `docs/results/2026-06-24-int8-training-throughput-final.md`
- Modify: `docs/ROADMAP.md` (line ~33), `docs/README.md` (line ~43)

**Interfaces:**
- Consumes: the numbers recorded in Tasks 1–4.

- [ ] **Step 1: Write the result note**

Create `docs/results/2026-06-24-int8-per-token-speed.md` leading with current numbers. Include: the GPU model/UUID; the method banner (sustained, dense_bf16 baseline = mixed-precision dense); the before/after int8 ms/step from Task 2 (the sync-removal win); the full width × mode table from Task 4; the **per-token crossover width** and the **bf16-dense OOM frontier**; the int8 GEMM `%pk` from Task 3; and the reused width-512 convergence parity (+0.0077 nats, cite `docs/results/2026-06-21-int8-convergence-25m.md`). End with the success-criterion one-liner instantiated with the measured X (crossover) and Y (OOM) widths. State explicitly that a plain *dense* fp32/bf16 model is faster at small width and that the win is a frontier-width + memory claim, not a small-width one.

- [ ] **Step 2: Supersede the stale table**

At the top of `docs/results/2026-06-24-int8-training-throughput-final.md`, add a banner:
```markdown
> **SUPERSEDED (2026-06-24).** The width-512 int8 number below (42,674 tok/s, "bf16 2.6x faster")
> was measured with the per-step-synced method, which over-penalizes kernel-heavy int8. Under the
> corrected sustained method the int8 ratchet is ~110k tok/s at width 512, and the `bf16` column
> here is the bf16 *ratchet*, not a dense baseline. See `2026-06-24-int8-per-token-speed.md` for
> the corrected sustained sweep against a dense baseline.
```
Do not delete the old numbers.

- [ ] **Step 3: Correct ROADMAP and README**

In `docs/ROADMAP.md:33`, replace the "at smaller widths … makes int8 slower … beats bf16 at width 4096" sentence with the measured result: int8 ratchet beats the bf16 *ratchet* at all measured widths under the sustained method, the per-token crossover vs the *dense* baseline is at width X, and dense bf16 OOMs at width Y while the ratchet trains on. In `docs/README.md:43`, replace "width-gated (crossover ~K=4096) … switches on at frontier scale" with the same corrected crossover figure and a pointer to the new note.

- [ ] **Step 4: Lint docs whitespace**

Run: `git diff --check`
Expected: no trailing-whitespace errors.

- [ ] **Step 5: Commit**

```bash
git add docs/results/2026-06-24-int8-per-token-speed.md docs/results/2026-06-24-int8-training-throughput-final.md docs/ROADMAP.md docs/README.md
git commit -m "docs: corrected per-token speed sweep (dense baseline, sustained); supersede stale table"
```

---

## Notes for the executor

- Phases 1, 3, 4 are measurement/doc tasks (run-and-record), not red-green-refactor; only Task 2 is TDD code.
- If Task 2's no-sync path still shows no `ms/step` improvement, the binding per-step sync may be elsewhere (e.g. `_validate_state`); profile with `scripts/int8_step_profile.py` before assuming the change failed.
- The convergence half is reused, not re-run — do not start a frontier-width training run.
