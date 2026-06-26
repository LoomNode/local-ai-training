# Ratchet the Token Embedding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the input `token_embedding` master-weight-free (integer codes + per-row scale, no FP `Parameter`) and run a matched A/B vs the FP-embedding control to measure any quality cost, at byte-level vocab.

**Architecture:** `RatchetEmbedding` subclasses `DiscreteRatchetLinear` (out_features = vocab, in_features = n_embd) and overrides only `forward` to do an `F.embedding` lookup on a transient effective weight; it inherits the entire update/normalize/audit machinery. Because every existing sweep keys on `isinstance(module, DiscreteRatchetLinear)`, the embedding auto-joins them. A `ratchet_embedding` config flag (default **False**, opt-in) selects it, so existing baselines are unchanged and the A/B's only variable is embedding precision.

**Tech Stack:** PyTorch, `uv`, pytest. Package `src/local_ai_training/`.

## Global Constraints

- Ratchet matrices persist **only** int8 code, int8 pressure (nibble-packed in one `uint8` `packed` buffer), and one FP32 scale per row. **Never** add an FP32/BF16 `Parameter` mirroring a code matrix. `audit_no_master_weights` (via `lat audit`) must stay violation-free.
- Temporary FP effective weights/gradients are fine but must be released after each ratchet update (`self._effective_weight = None`).
- No per-step host syncs over large tensors (`.item()`, `.all()`) on the hot path — gate them behind `validate` (the width-4096 "hang" lesson).
- Default behavior must not change: `ratchet_embedding` defaults to **False** (FP `nn.Embedding`), so existing configs/checkpoints/baselines are untouched. The treatment opts in.
- Screening budget: 5k steps for iteration; 30k only for converged confirmation.
- Byte-level vocab only (~205). Large-vocab sparse updates, fused-inline backward, trainable embedding scale, and weight tying are **out of scope**.
- Commit messages end with the two trailers used across this repo (`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and the `Claude-Session:` line).
- Run commands with `UV_CACHE_DIR=/games/ailab/.uv-cache`; training on `CUDA_VISIBLE_DEVICES=1`.

---

### Task 1: `RatchetEmbedding` module

**Files:**
- Modify: `src/local_ai_training/ratchet.py` (add class after `DiscreteRatchetLinear`, ends line 681)
- Test: `tests/test_ratchet_embedding.py` (create)

**Interfaces:**
- Consumes: `DiscreteRatchetLinear` (its `__init__`, `effective_weight()`, `apply_weight_gradient`, `ratchet_update`, `code`/`pressure`/`scale` properties, `_effective_weight` slot, `has_pending_gradient`), `unpack_code_pressure`, `torch.nn.functional as F` (already imported in ratchet.py).
- Produces: `RatchetEmbedding(num_embeddings: int, embedding_dim: int, *, max_code: int, pressure_threshold: int = 8, bucket_low: float = 0.5, bucket_high: float = 1.5, eps: float = 1e-8, rms_ema_beta: float = 0.0, pressure_leak_period: int = 0, trainable_scale: bool = False, compile_update: bool = False, initial_weight: Tensor | None = None)`. `forward(token_ids: Tensor) -> Tensor`. Attributes `num_embeddings`, `embedding_dim`. Inherits `code` (shape `(num_embeddings, embedding_dim)`), `effective_weight()`, `ratchet_update(validate=...)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ratchet_embedding.py`:

```python
import torch
import torch.nn.functional as F

from local_ai_training.ratchet import (
    DiscreteRatchetLinear,
    RatchetEmbedding,
    audit_no_master_weights,
)


def _embedding(seed: int = 0, *, max_code: int = 7) -> RatchetEmbedding:
    torch.manual_seed(seed)
    return RatchetEmbedding(num_embeddings=12, embedding_dim=8, max_code=max_code)


def test_is_a_discrete_ratchet_linear_with_token_rows() -> None:
    embedding = _embedding()
    assert isinstance(embedding, DiscreteRatchetLinear)  # auto-joins existing sweeps
    assert embedding.code.shape == (12, 8)  # rows are tokens, cols are embedding dim
    assert embedding.scale.shape == (12,)


def test_forward_matches_f_embedding_on_effective_weight() -> None:
    embedding = _embedding().eval()
    token_ids = torch.tensor([[0, 3, 11], [5, 5, 1]])
    expected = F.embedding(token_ids, embedding.effective_weight())
    assert torch.equal(embedding(token_ids), expected)


def test_has_no_master_weight_parameter() -> None:
    embedding = _embedding()
    assert list(embedding.parameters()) == []  # buffers only; nothing AdamW would train


def test_forward_captures_and_releases_effective_weight_gradient() -> None:
    embedding = _embedding().train()
    token_ids = torch.tensor([[0, 1, 2, 3]])
    out = embedding(token_ids)
    out.square().sum().backward()
    assert embedding.has_pending_gradient
    assert embedding._effective_weight.grad is not None
    embedding.ratchet_update(validate=True)
    assert embedding._effective_weight is None  # released after update


def test_codes_move_under_a_persistent_gradient() -> None:
    embedding = _embedding(max_code=7)
    before = embedding.code.clone()
    grad = torch.ones_like(embedding.code, dtype=torch.float32)
    # pressure_threshold default 8; bucket gives +2/step for |z|>=high, so a handful of
    # identical applications must move at least one code.
    for _ in range(8):
        embedding.apply_weight_gradient(grad, validate=False)
    assert not torch.equal(embedding.code, before)


def test_audit_reports_no_violation_and_counts_embedding_state() -> None:
    embedding = _embedding()
    report = audit_no_master_weights(embedding, raise_on_violation=True)
    assert report.ratchet_layers == 1
    assert report.ratchet_state_bytes > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet_embedding.py -q`
Expected: FAIL at import — `ImportError: cannot import name 'RatchetEmbedding'`.

- [ ] **Step 3: Implement `RatchetEmbedding`**

In `src/local_ai_training/ratchet.py`, immediately after `DiscreteRatchetLinear` (after its `extra_repr`, line 681), add:

```python
class RatchetEmbedding(DiscreteRatchetLinear):
    """Master-weight-free token embedding.

    A DiscreteRatchetLinear whose forward is an embedding lookup instead of a matmul:
    rows are tokens (out_features == num_embeddings), columns the embedding dim
    (in_features == embedding_dim). It inherits the entire integer-code update path
    (per-row scale, pressure accumulation, bucket_pressure, ratchet_update) and, because
    every sweep keys on isinstance(DiscreteRatchetLinear), auto-joins the update / discard /
    metrics / audit passes. Only the forward differs.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        max_code: int,
        pressure_threshold: int = 8,
        bucket_low: float = 0.5,
        bucket_high: float = 1.5,
        eps: float = 1e-8,
        rms_ema_beta: float = 0.0,
        pressure_leak_period: int = 0,
        trainable_scale: bool = False,
        compile_update: bool = False,
        initial_weight: Tensor | None = None,
    ) -> None:
        if initial_weight is None:
            # N(0, 1), matching nn.Embedding's default reset_parameters distribution.
            initial_weight = torch.randn(num_embeddings, embedding_dim)
        super().__init__(
            embedding_dim,
            num_embeddings,
            max_code=max_code,
            pressure_threshold=pressure_threshold,
            bucket_low=bucket_low,
            bucket_high=bucket_high,
            eps=eps,
            rms_ema_beta=rms_ema_beta,
            pressure_leak_period=pressure_leak_period,
            trainable_scale=trainable_scale,
            compile_update=compile_update,
            matmul_mode="fp32",
            initial_weight=initial_weight,
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, token_ids: Tensor) -> Tensor:  # type: ignore[override]
        effective = self.effective_weight()
        if self.training and torch.is_grad_enabled():
            # Transient leaf so autograd fills effective.grad (the weight gradient the
            # ratchet consumes). Released in ratchet_update() — no persistent FP weight.
            effective = effective.detach().requires_grad_(True)
            self._effective_weight = effective
        return F.embedding(token_ids, effective)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"states={2 * self.max_code + 1}, threshold={self.pressure_threshold}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet_embedding.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Lint + commit**

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run ruff check src/local_ai_training/ratchet.py tests/test_ratchet_embedding.py
git add src/local_ai_training/ratchet.py tests/test_ratchet_embedding.py
git commit -m "feat: RatchetEmbedding — master-free token embedding

$(printf 'Subclass of DiscreteRatchetLinear (rows=tokens) overriding only forward to do\nan F.embedding lookup on a transient effective weight; inherits the integer-code\nupdate path and auto-joins the isinstance-keyed sweeps. No nn.Parameter.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011gqtgUHpfZDiNfQ36VTbBB')"
```

---

### Task 2: Wire into the model behind a `ratchet_embedding` flag

**Files:**
- Modify: `src/local_ai_training/model.py` (import line 20; `ModelConfig` fields ~line 39-42; `RatchetGPT.__init__` line 204)
- Modify: `src/local_ai_training/config.py` (`ExperimentConfig` field, `to_dict`, `model_config`)
- Modify: `src/local_ai_training/cli.py` (`train` subparser ~line 33-47; `train_run` call ~line 163)
- Modify: `src/local_ai_training/generate.py` (`load_for_generation`, read the flag from metadata)
- Test: `tests/test_experiment.py` (add integration tests)

**Interfaces:**
- Consumes: `RatchetEmbedding` (Task 1); `ModelConfig` fields `vocab_size`, `n_embd`, `pressure_threshold`, `bucket_low`, `bucket_high`, `trainable_scale`, `compile_update`.
- Produces: `ModelConfig.ratchet_embedding: bool = False`; `ExperimentConfig.ratchet_embedding: bool = False` surfaced in `to_dict()` and `model_config()`; CLI flag `--ratchet-embedding` (store_true) threaded into `train_run`. When `max_code is not None and config.ratchet_embedding`, `RatchetGPT.token_embedding` is a `RatchetEmbedding`, else `nn.Embedding`.

- [ ] **Step 1: Write the failing integration tests**

Add to `tests/test_experiment.py`:

```python
def test_ratchet_embedding_flag_swaps_module_and_trains(tmp_path) -> None:
    from local_ai_training.ratchet import RatchetEmbedding, audit_no_master_weights

    corpus = build_char_corpus("abcdefgh " * 200)
    config = replace(small_experiment_config(), steps=2, eval_interval=2, ratchet_embedding=True)
    result = train_run(
        corpus=corpus, config=config, max_code=7, seed=7, run_dir=tmp_path / "re"
    )
    assert result.metrics_csv.is_file()


def test_ratchet_embedding_model_is_audit_clean_with_embedding_as_state() -> None:
    config = replace(small_experiment_config(), ratchet_embedding=True)
    model = build_seeded_model(
        config.model_config(vocab_size=11), max_code=7, seed=3
    )
    from local_ai_training.model import RatchetGPT  # noqa: F401
    from local_ai_training.ratchet import RatchetEmbedding, audit_no_master_weights

    assert isinstance(model.token_embedding, RatchetEmbedding)
    report = audit_no_master_weights(model, raise_on_violation=True)
    assert report.ratchet_state_bytes > 0  # no violation raised; embedding counted


def test_default_keeps_fp_embedding() -> None:
    import torch.nn as nn

    config = small_experiment_config()  # ratchet_embedding defaults False
    model = build_seeded_model(config.model_config(vocab_size=11), max_code=7, seed=3)
    assert isinstance(model.token_embedding, nn.Embedding)
```

(If `small_experiment_config()` is not already a helper in this file, use the existing config factory the other tests use — check the top of `tests/test_experiment.py` and reuse whatever builds the smoke `ExperimentConfig`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_experiment.py -k ratchet_embedding -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'ratchet_embedding'` (the field does not exist yet).

- [ ] **Step 3: Add the `ModelConfig` field + model wiring**

In `src/local_ai_training/model.py`, line 20, extend the import:

```python
from .ratchet import DiscreteRatchetLinear, RatchetEmbedding, RatchetUpdateStats
```

In `ModelConfig` (the dataclass with `matmul_mode` etc., around line 39-42), add a field next to `qat`:

```python
    ratchet_embedding: bool = False
```

In `RatchetGPT.__init__`, replace line 204:

```python
        if max_code is not None and config.ratchet_embedding:
            self.token_embedding = RatchetEmbedding(
                config.vocab_size,
                config.n_embd,
                max_code=max_code,
                pressure_threshold=config.pressure_threshold,
                bucket_low=config.bucket_low,
                bucket_high=config.bucket_high,
                trainable_scale=config.trainable_scale,
                compile_update=config.compile_update,
            )
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
```

- [ ] **Step 4: Thread the flag through `config.py`**

In `src/local_ai_training/config.py`: add `ratchet_embedding: bool = False` to `ExperimentConfig`; include `"ratchet_embedding"` in the `to_dict()` field list (next to `matmul_mode` — see `config.py:91`); and pass `ratchet_embedding=self.ratchet_embedding` into the `ModelConfig(...)` built by `.model_config()` (next to `matmul_mode=self.matmul_mode`, `config.py:124`).

- [ ] **Step 5: Run the integration tests to verify they pass**

Run: `UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_experiment.py -k ratchet_embedding -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Add the CLI flag + generate thread-through**

In `src/local_ai_training/cli.py`, in the `train` subparser block (around line 33-47), add:

```python
    train.add_argument("--ratchet-embedding", action="store_true")
```

and where `train_run`/config is assembled for the `train` command (the config is built from TOML, then overridden by flags like `--codes`/`--trainable-scale`), apply the override after loading the config, mirroring the existing `--trainable-scale` handling:

```python
        if args.ratchet_embedding:
            config = replace(config, ratchet_embedding=True)
```

In `src/local_ai_training/generate.py` `load_for_generation`, read the flag from the checkpoint metadata so a ratchet-embedding checkpoint rebuilds correctly:

```python
        ratchet_embedding=bool(config.get("ratchet_embedding", False)),
```

added to the `ModelConfig(...)` constructed there.

- [ ] **Step 7: Run the full suite + audit + lint**

Run:
```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest -q
UV_CACHE_DIR=/games/ailab/.uv-cache uv run lat audit --model configs/ratchet_tiny.toml
UV_CACHE_DIR=/games/ailab/.uv-cache uv run ruff check src/local_ai_training tests
git diff --check
```
Expected: all pass; audit reports no violations; ruff clean; no whitespace errors.

- [ ] **Step 8: Commit**

```bash
git add src/local_ai_training/model.py src/local_ai_training/config.py src/local_ai_training/cli.py src/local_ai_training/generate.py tests/test_experiment.py
git commit -m "feat: ratchet_embedding flag wires RatchetEmbedding into the model

$(printf 'Opt-in (default False, keeping FP nn.Embedding) so existing baselines are\nunchanged. When set with a ratchet max_code, token_embedding becomes a\nRatchetEmbedding and auto-joins the update/discard/metrics/audit sweeps. CLI\n--ratchet-embedding and generate metadata thread-through included.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011gqtgUHpfZDiNfQ36VTbBB')"
```

---

### Task 3: Run the A/B and write the results note

**Files:**
- Create: `docs/results/2026-06-26-ratchet-embedding-ab.md`
- Artifacts (git-ignored): `runs/embed_ab_control`, `runs/embed_ab_ratchet`

**Interfaces:**
- Consumes: the `--ratchet-embedding` flag (Task 2), `configs/scaleup_text8_25m_5k.toml`, the extracted corpus `data/enwik8/enwik8`, `lat audit`, `lat generate`.
- Produces: a results note with the val-loss gap and the embedding's saturation/histogram.

- [ ] **Step 1: Launch the control arm (FP embedding)**

```bash
cd /games/ailab/local-ai-training/.worktrees/projection-oracle
export UV_CACHE_DIR=/games/ailab/.uv-cache MPLCONFIGDIR=/tmp/mpl
CUDA_VISIBLE_DEVICES=1 uv run lat train --codes 15 \
  --config configs/scaleup_text8_25m_5k.toml --seed 1337 \
  --dataset-path /games/ailab/local-ai-training/data/enwik8/enwik8 \
  --output runs/embed_ab_control
```
Expected: completes 5000 steps, prints `final_validation_loss`. (`token_embedding` is `nn.Embedding`.)

- [ ] **Step 2: Launch the treatment arm (ratchet embedding)**

```bash
CUDA_VISIBLE_DEVICES=1 uv run lat train --codes 15 --ratchet-embedding \
  --config configs/scaleup_text8_25m_5k.toml --seed 1337 \
  --dataset-path /games/ailab/local-ai-training/data/enwik8/enwik8 \
  --output runs/embed_ab_ratchet
```
Expected: completes 5000 steps; `final_validation_loss` recorded.

- [ ] **Step 3: Verify the treatment is master-free and inspect the embedding state**

Read the last row of `runs/embed_ab_ratchet/metrics.csv` — its `code_histogram` / `saturated_percent` now include the embedding's rows. Confirm the treatment checkpoint has no FP embedding parameter by loading metadata and confirming `ratchet_embedding: true` in `experiment_config`, and that a generation still works:

```bash
CUDA_VISIBLE_DEVICES=1 uv run lat generate --checkpoint runs/embed_ab_ratchet/checkpoint \
  --prompt "[[History of " --max-new-tokens 160 --temperature 0.8 --seed 1
```
Expected: legible enwik8-style text (proves the ratcheted embedding learned a usable representation).

- [ ] **Step 4: Write the results note**

Create `docs/results/2026-06-26-ratchet-embedding-ab.md` with: a TL;DR (does ratcheting the embedding cost quality at byte-level — yes/no + the nat gap); the two final val losses; the embedding saturation %; a sample from the treatment; and the honest framing (this closes the last FP carve-out at byte-level and validates the per-row-scale + dense-update mechanism; large-vocab sparse updates remain future work). **Preserve the result whichever way it falls** — if there is a gap, report it, do not tune it away. Note whether the result justifies flipping `ratchet_embedding` to default-True in a follow-up (and update `docs/ROADMAP.md` item #5 status accordingly).

- [ ] **Step 5: Commit the results note + roadmap update**

```bash
git add docs/results/2026-06-26-ratchet-embedding-ab.md docs/ROADMAP.md
git commit -m "docs: A/B results — ratcheting the token embedding at byte-level

$(printf 'Matched FP-embedding control vs ratchet-embedding treatment (enwik8 25M\ncodes-15, 5k, seed 1337). Reports the val-loss gap + embedding saturation and\nupdates ROADMAP #5.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011gqtgUHpfZDiNfQ36VTbBB')"
```

---

## Self-Review

**Spec coverage:** RatchetEmbedding module (Task 1) ✓; shared update internals via subclass (Task 1) ✓; transient-leaf gradient capture + release (Task 1, Step 3 + test) ✓; audit-clean, embedding counted as state (Task 1 test, Task 2 test) ✓; `nn.Embedding`-vs-ratchet wiring mirroring `_linear` (Task 2) ✓; frozen-control discard (inherited; covered by existing `discard_pending_gradients` keying on `isinstance` — no new code, noted) ✓; A/B FP vs ratchet, enwik8 25M codes-15, 5k→30k, preserve result (Task 3) ✓; secondary observable = embedding saturation/histogram (Task 3, Step 3) — available because metrics keys on `isinstance` ✓; gated non-finite guard (inherited `apply_weight_gradient(validate=...)`) ✓; YAGNI boundaries stated in Global Constraints ✓.

**Gap found + resolved:** the spec's "frozen control discards embedding grads" needs no new code — `RatchetGPT.discard_pending_gradients` already iterates `isinstance(DiscreteRatchetLinear)`, so a `RatchetEmbedding` is covered automatically. No task needed; noted here.

**30k promotion:** intentionally not a separate task — Task 3 screens at 5k; if the gap is interesting, re-run Steps 1-2 with `configs/scaleup_text8_25m_30k.toml` before writing the note. Folded into Task 3's judgment rather than a standalone task.

**Type consistency:** `RatchetEmbedding(num_embeddings, embedding_dim, *, max_code, ...)` and `forward(token_ids)` are used identically in Tasks 1-2; `ratchet_embedding` flag name is identical across `ModelConfig`, `ExperimentConfig`, CLI, and `generate`. `effective_weight()` is a method call (matches `DiscreteRatchetLinear`).
