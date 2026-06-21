# 4-bit Packed Storage and Update Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pack the ratchet `code` + `pressure` into one `uint8` byte (halving persistent state, ~6x -> ~12x params/GB) and fuse the elementwise ratchet update via `torch.compile`, with provably identical behavior.

**Architecture:** `DiscreteRatchetLinear` stores a single `uint8` `packed` buffer (code in the low nibble, pressure in the high nibble, both offset-encoded). `code`/`pressure` become unpack-on-read properties; all writes go through one pack. The update math is unchanged but restructured to unpack once / pack once so `torch.compile` fuses it. The FP effective-weight matmul path is untouched.

**Tech Stack:** Python 3.10+, PyTorch >=2.5, pytest, ruff. Run everything with `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run ...`.

## Global Constraints

- Master-weight-free invariant: no FP32/BF16 matrix `Parameter` may be added; `audit_no_master_weights` must report zero violations. (verbatim from AGENTS.md)
- Behavior must be bit-identical to the current update: same `code`, `pressure`, and `RatchetUpdateStats`. The only allowed external change is the on-disk checkpoint format.
- Supported tiers only: `max_code in (2, 3, 4)`, `pressure_threshold <= 8`. Packing is lossless exactly within these bounds.
- ruff must pass (`uv run ruff check .`), line-length 100. All existing tests stay green (currently 36).
- Commit messages end with the repo's Co-Authored-By / Claude-Session trailers.

---

### Task 1: Nibble pack/unpack helpers

**Files:**
- Modify: `src/local_ai_training/ratchet.py` (add two module-level functions near the top, after imports)
- Test: `tests/test_ratchet.py`

**Interfaces:**
- Produces: `pack_code_pressure(code: Tensor, pressure: Tensor, max_code: int) -> Tensor` (returns `uint8`); `unpack_code_pressure(packed: Tensor, max_code: int) -> tuple[Tensor, Tensor]` (returns two `int8` tensors).

- [ ] **Step 1: Write the failing round-trip test**

```python
# tests/test_ratchet.py
from local_ai_training.ratchet import pack_code_pressure, unpack_code_pressure

def test_nibble_pack_unpack_round_trip_is_lossless() -> None:
    for max_code in (2, 3, 4):
        codes = torch.arange(-max_code, max_code + 1, dtype=torch.int8)
        pressures = torch.arange(-7, 8, dtype=torch.int8)
        code_grid, pressure_grid = torch.meshgrid(codes, pressures, indexing="ij")
        packed = pack_code_pressure(code_grid, pressure_grid, max_code)
        assert packed.dtype == torch.uint8
        out_code, out_pressure = unpack_code_pressure(packed, max_code)
        assert torch.equal(out_code, code_grid)
        assert torch.equal(out_pressure, pressure_grid)
        assert out_code.dtype == torch.int8 and out_pressure.dtype == torch.int8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py::test_nibble_pack_unpack_round_trip_is_lossless -v`
Expected: FAIL with ImportError (functions not defined).

- [ ] **Step 3: Implement the helpers**

```python
# src/local_ai_training/ratchet.py, after the imports
def pack_code_pressure(code: Tensor, pressure: Tensor, max_code: int) -> Tensor:
    """Pack signed code (low nibble) and pressure (high nibble) into one uint8.

    Lossless for code in [-max_code, max_code] (max_code <= 4) and pressure in [-7, 7].
    """
    low = (code.to(torch.int16) + max_code) & 0x0F
    high = (pressure.to(torch.int16) + 7) & 0x0F
    return (low | (high << 4)).to(torch.uint8)


def unpack_code_pressure(packed: Tensor, max_code: int) -> tuple[Tensor, Tensor]:
    value = packed.to(torch.int16)
    code = ((value & 0x0F) - max_code).to(torch.int8)
    pressure = ((value >> 4) - 7).to(torch.int8)
    return code, pressure
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py::test_nibble_pack_unpack_round_trip_is_lossless -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/local_ai_training/ratchet.py tests/test_ratchet.py
git commit -m "feat: add lossless nibble pack/unpack for ratchet code+pressure"
```

---

### Task 2: Golden-reference behavior test (pins current update before refactor)

**Files:**
- Test: `tests/test_ratchet.py`

**Interfaces:**
- Consumes: existing `DiscreteRatchetLinear.apply_normalized_gradient`.

- [ ] **Step 1: Write the golden test (passes on CURRENT code)**

These values were captured from the current implementation; they must remain unchanged through the storage and fusion refactors.

```python
# tests/test_ratchet.py
import pytest

@pytest.mark.parametrize(
    "max_code, code_sum, pressure_sum, total_moves",
    [(2, 15, -32, 59), (3, 14, -32, 62), (4, 9, -32, 65)],
)
def test_update_matches_golden_reference(max_code, code_sum, pressure_sum, total_moves) -> None:
    torch.manual_seed(1234)
    layer = DiscreteRatchetLinear(8, 6, max_code=max_code, pressure_threshold=8)
    total = 0
    for _ in range(60):
        total += layer.apply_normalized_gradient(torch.randn(6, 8) * 2.0).code_moves
    assert int(layer.code.sum()) == code_sum
    assert int(layer.pressure.sum()) == pressure_sum
    assert total == total_moves
```

- [ ] **Step 2: Run to verify it PASSES on current code**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py::test_update_matches_golden_reference -v`
Expected: PASS (3 parametrizations). This locks current behavior; it must keep passing after Tasks 3 and 6.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ratchet.py
git commit -m "test: pin ratchet update behavior with golden reference"
```

---

### Task 3: Switch DiscreteRatchetLinear to packed uint8 storage

**Files:**
- Modify: `src/local_ai_training/ratchet.py` (`__init__`, add `code`/`pressure` properties, `apply_normalized_gradient`, `_validate_state`, `persistent_state_bytes`)
- Test: `tests/test_ratchet.py` (existing tests + golden + round-trip cover this)

**Interfaces:**
- Consumes: `pack_code_pressure`, `unpack_code_pressure` (Task 1).
- Produces: `self.packed` (`uint8` buffer); `code`/`pressure` read-only properties returning `int8`.

- [ ] **Step 1: Replace the buffers in `__init__`**

Find the block that registers `code` and `pressure` buffers and replace with a single packed buffer (leave the `scale`/`log_scale` block unchanged):

```python
        # was: register_buffer("code", ...) and register_buffer("pressure", ...)
        zero_pressure = torch.zeros_like(code, dtype=torch.int8)
        self.register_buffer("packed", pack_code_pressure(code.to(torch.int8), zero_pressure, max_code))
```

- [ ] **Step 2: Add `code`/`pressure` properties**

Add near the existing `scale` property:

```python
    @property
    def code(self) -> Tensor:
        return unpack_code_pressure(self.packed, self.max_code)[0]

    @property
    def pressure(self) -> Tensor:
        return unpack_code_pressure(self.packed, self.max_code)[1]
```

- [ ] **Step 3: Rewrite the tail of `apply_normalized_gradient` to unpack once / pack once**

Read code+pressure from `packed` at the start of the compute, and write back with a single pack. Replace the two `copy_` calls:

```python
        # was: self.code.copy_(...); self.pressure.copy_(...)
        self.packed.copy_(
            pack_code_pressure(code.to(torch.int8), pressure.to(torch.int8), self.max_code)
        )
```

Also change the two reads at the top of the method from `self.pressure`/`self.code` (which now each unpack separately) to one unpack:

```python
        current_code, current_pressure = unpack_code_pressure(self.packed, self.max_code)
        pressure = current_pressure.to(torch.int16) + increments
        code = current_code.to(torch.int16)
```

- [ ] **Step 4: Update `_validate_state` and `persistent_state_bytes`**

```python
    def _validate_state(self) -> None:
        code, pressure = unpack_code_pressure(self.packed, self.max_code)
        if code.min().item() < -self.max_code or code.max().item() > self.max_code:
            raise RuntimeError("ratchet code escaped its allowed range")
        if pressure.abs().max().item() > 7:
            raise RuntimeError("ratchet pressure escaped the packed nibble range")
        if not torch.isfinite(self.scale).all() or torch.any(self.scale <= 0):
            raise RuntimeError("ratchet row scales must remain positive and finite")

    @property
    def persistent_state_bytes(self) -> int:
        return self.packed.numel() * self.packed.element_size() + self.scale.numel() * self.scale.element_size()
```

- [ ] **Step 5: Run the golden + round-trip + full suite**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py tests/test_experiment.py tests/test_model.py -v`
Expected: PASS — golden values unchanged, existing forward/resume/controls tests still green. The persistent-footprint test will now report half the bytes; if it asserts the old number, update it in Task 4.

- [ ] **Step 6: Commit**

```bash
git add src/local_ai_training/ratchet.py
git commit -m "feat: store ratchet code+pressure packed in one uint8 byte"
```

---

### Task 4: Update audit and footprint accounting for packed format

**Files:**
- Modify: `src/local_ai_training/ratchet.py` (`audit_no_master_weights`, `compare_persistent_footprint`)
- Test: `tests/test_ratchet.py`

**Interfaces:**
- Consumes: `self.packed`, `persistent_state_bytes` (Task 3).

- [ ] **Step 1: Write/extend the footprint test for the halved bytes**

```python
# tests/test_ratchet.py
def test_packed_footprint_is_one_byte_per_weight() -> None:
    model = nn.Sequential(DiscreteRatchetLinear(16, 8, max_code=2),
                          DiscreteRatchetLinear(8, 4, max_code=3))
    audit = audit_no_master_weights(model)
    weights = audit.ratchet_weights
    # one uint8 byte per weight, plus per-row fp32 scale
    scale_bytes = (8 + 4) * 4
    assert audit.ratchet_state_bytes == weights * 1 + scale_bytes
    fp = compare_persistent_footprint(model)
    assert fp.reduction_ratio > 10  # ~12x now (was ~6x at int8)
```

- [ ] **Step 2: Run to verify it fails**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py::test_packed_footprint_is_one_byte_per_weight -v`
Expected: FAIL on the dtype/uint8 validation or the byte count if audit still expects int8 buffers.

- [ ] **Step 3: Update the audit to validate the packed buffer**

In `audit_no_master_weights`, replace the `code`/`pressure` dtype checks with:

```python
        if module.packed.dtype != torch.uint8:
            violations.append(f"{prefix}.packed: expected uint8, got {module.packed.dtype}")
        if module.scale.ndim != 1 or module.scale.shape[0] != module.out_features:
            violations.append(f"{prefix}.scale: expected one scale per output row")
```

`ratchet_weights += module.code.numel()` and `ratchet_state_bytes += module.persistent_state_bytes` still work (property + halved bytes). `compare_persistent_footprint` needs no change — it already sums `persistent_state_bytes`, which is now halved, so `reduction_ratio` rises automatically.

- [ ] **Step 4: Run to verify it passes + full suite**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest -q`
Expected: PASS (all). Fix any footprint test that hard-coded the old int8 byte count.

- [ ] **Step 5: Commit**

```bash
git add src/local_ai_training/ratchet.py tests/test_ratchet.py
git commit -m "feat: account for packed 1-byte ratchet state in audit and footprint"
```

---

### Task 5: Commit the pressure-nibble invariant test

**Files:**
- Test: `tests/test_ratchet.py`

- [ ] **Step 1: Add the adversarial invariant test**

```python
# tests/test_ratchet.py
def test_stored_pressure_stays_within_nibble_under_adversarial_updates() -> None:
    torch.manual_seed(0)
    for max_code in (2, 3, 4):
        for threshold in (1, 2, 4, 8):
            layer = DiscreteRatchetLinear(8, 6, max_code=max_code, pressure_threshold=threshold)
            for step in range(200):
                sign = 1.0 if step % 7 < 5 else -1.0   # sustained push -> saturate codes
                layer.apply_normalized_gradient(sign * (3.0 + torch.rand(6, 8)))
                assert int(layer.pressure.abs().max()) <= 7
                assert int(layer.code.abs().max()) <= max_code
```

- [ ] **Step 2: Run to verify it passes**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py::test_stored_pressure_stays_within_nibble_under_adversarial_updates -v`
Expected: PASS (already confirmed empirically: worst |pressure| observed is 6).

- [ ] **Step 3: Commit (end of the storage stage)**

```bash
git add tests/test_ratchet.py
git commit -m "test: enforce pressure stays within packed nibble range"
```

---

### Task 6: Fuse the update with torch.compile (the speed stage)

**Files:**
- Modify: `src/local_ai_training/ratchet.py` (extract the update core, add opt-in compile)
- Test: `tests/test_ratchet.py`

**Interfaces:**
- Produces: `DiscreteRatchetLinear(..., compile_update: bool = False)`; when true, the update runs through a `torch.compile`-fused function producing identical results.

- [ ] **Step 1: Profile the current update (gate)**

Run this and record where time goes; confirms the update (not FP-weight materialization) is the target before optimizing:

```bash
MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run python -c "
import torch, time
from local_ai_training.ratchet import DiscreteRatchetLinear
layer = DiscreteRatchetLinear(2048, 2048, max_code=4).cuda()
g = torch.randn(2048, 2048, device='cuda')
for _ in range(5): layer.apply_normalized_gradient(g)   # warmup
torch.cuda.synchronize(); t=time.time()
for _ in range(50): layer.apply_normalized_gradient(g)
torch.cuda.synchronize(); print(f'update: {(time.time()-t)/50*1000:.2f} ms/call')
"
```
Expected: a per-call cost that, multiplied across all layers and steps, accounts for the ~2.3x slowdown. If it is negligible, stop and reassess (the slowdown would then be FP-weight materialization — out of scope here).

- [ ] **Step 2: Extract the update core into a standalone function**

Refactor the elementwise body of `apply_normalized_gradient` (from `increments` through the final pack) into a module-level pure function `_ratchet_update_core(packed, increments, max_code, threshold) -> tuple[Tensor, Tensor, ...]` returning the new `packed` and the move-count tensors. `apply_normalized_gradient` calls it. Verify the golden test still passes (pure refactor):

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py::test_update_matches_golden_reference -v`
Expected: PASS.

- [ ] **Step 3: Add opt-in compiled path + equality test**

```python
# tests/test_ratchet.py
def test_compiled_update_matches_eager() -> None:
    torch.manual_seed(1234)
    eager = DiscreteRatchetLinear(8, 6, max_code=4, pressure_threshold=8)
    comp = DiscreteRatchetLinear(8, 6, max_code=4, pressure_threshold=8, compile_update=True)
    for _ in range(20):
        g = torch.randn(6, 8) * 2.0
        eager.apply_normalized_gradient(g.clone())
        comp.apply_normalized_gradient(g.clone())
    assert torch.equal(eager.packed, comp.packed)
```

In `__init__`, store `self._update_fn = torch.compile(_ratchet_update_core) if compile_update else _ratchet_update_core` and call `self._update_fn(...)` in `apply_normalized_gradient`. Default stays eager so the suite is fast and deterministic.

- [ ] **Step 4: Run the equality test + full suite**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest -q`
Expected: PASS (37+). The compiled path is bit-identical to eager.

- [ ] **Step 5: Benchmark the fused speedup**

```bash
MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run python -c "
import torch, time
from local_ai_training.ratchet import DiscreteRatchetLinear
for flag in (False, True):
    l = DiscreteRatchetLinear(2048, 2048, max_code=4, compile_update=flag).cuda()
    g = torch.randn(2048, 2048, device='cuda')
    for _ in range(10): l.apply_normalized_gradient(g)
    torch.cuda.synchronize(); t=time.time()
    for _ in range(50): l.apply_normalized_gradient(g)
    torch.cuda.synchronize(); print(f'compile={flag}: {(time.time()-t)/50*1000:.2f} ms/call')
"
```
Record the speedup in the commit message. If compiled is not faster, note it and keep the eager default (Triton fallback would be the next step, but only if measurement justifies it).

- [ ] **Step 6: Commit**

```bash
git add src/local_ai_training/ratchet.py tests/test_ratchet.py
git commit -m "feat: optional torch.compile-fused ratchet update"
```

---

## Self-Review

**Spec coverage:** Section 1 (packed storage) -> Tasks 1, 3, 4. Section 2 (update fusion) -> Task 6. Section 3 (testing: golden equality, round-trip, footprint, invariant, regression) -> Tasks 1, 2, 4, 5, 6. Section 4 (sequencing: staged commits storage-then-fusion, checkpoint break) -> task ordering (Tasks 1-5 storage, Task 6 fusion); checkpoint format changes implicitly via the `packed` buffer replacing `code`/`pressure` in `state_dict` (no shim, per spec). All covered.

**Placeholder scan:** none — every code step has complete code; golden values are concrete (15/-32/59, 14/-32/62, 9/-32/65).

**Type consistency:** `pack_code_pressure`/`unpack_code_pressure` signatures match between Task 1 (definition) and Task 3 (use); `packed` is `uint8` throughout; `code`/`pressure` properties return `int8` matching prior buffer dtype; `compile_update` flag introduced in Task 6 only.

## Checkpoint / live-experiment note

The in-flight 60k convergence runs hold their code in memory and are unaffected; finish and analyze them on the old format before merging this branch. Pre-existing `runs/` checkpoints will not resume into the packed format (terminal experiments). This branch is `feat/packed-storage-update-fusion`.
