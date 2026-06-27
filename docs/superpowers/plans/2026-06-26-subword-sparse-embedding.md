# Subword + Sparse-Update Ratchet Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a polished, interactible, fully master-weight-free 25M model on enwik8 using an 8K pure-Python BPE subword vocab with a ratcheted input embedding driven by a sparse-aware per-row RMS-EMA update.

**Architecture:** Add a self-contained word-frequency BPE tokenizer (no new dependency); a `SubwordCorpus` parallel to the existing `CharCorpus`; fix the per-row RMS-EMA update so it only updates firing rows (the sparse-correctness core); thread the sparse knobs into `RatchetEmbedding`; embed the tokenizer in checkpoints so `generate` works subword. Then tune, run the 30k polished model, and a matched FP-vs-ratchet-embedding A/B.

**Tech Stack:** Python 3.10, PyTorch, pytest, ruff. No new dependencies (pure-Python BPE).

## Global Constraints

- Ratchet matrices persist ONLY int8 code, int8 pressure (nibble-packed in `packed`), and one FP32 `_scale` per output row. Never add an FP/BF16 `Parameter` mirroring a code matrix. `uv run lat audit` must stay violation-free.
- The subword ratchet-embedding model must have ZERO FP master weights: checkpoint has `model::token_embedding.packed` + `model::token_embedding._scale`, and NO `optimizer::token_embedding`.
- No new top-level dependency. Tokenizer is pure Python (stdlib `re` only).
- The existing char path (`CharCorpus`, char checkpoints, byte-level baselines/tests) MUST remain untouched and backward-compatible. New config field `tokenizer` defaults to `"char"`.
- Corpus/artifact discipline: tokenizer trained on the TRAIN split only (never the validation tail). The trained tokenizer JSON is a pinned, cached artifact — not retrained per run.
- Commands prefix: `CUDA_VISIBLE_DEVICES=1 UV_CACHE_DIR=/games/ailab/.uv-cache MPLCONFIGDIR=/tmp/mpl uv run ...`. Default to GPU 1.
- Never claim training acceleration or packed sub-byte storage. Preserve failed runs/configs and measured A/B magnitudes — do not tune them away.
- Latin-1 is the canonical byte<->str mapping used everywhere (enwik8 is read latin-1).
- Commit message footer (every commit):
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_011gqtgUHpfZDiNfQ36VTbBB
  ```

## File Structure

- `src/local_ai_training/tokenizer.py` (CREATE) — pure-Python word-frequency BPE: `BpeTokenizer` with `train`, `encode`, `decode`, `to_json`, `from_json`.
- `src/local_ai_training/ratchet.py` (MODIFY) — `_normalize`: mask the RMS-EMA update to firing rows only (sparse core).
- `src/local_ai_training/model.py` (MODIFY) — pass `rms_ema_beta` / `pressure_leak_period` into `RatchetEmbedding` (currently omitted at ~line 206).
- `src/local_ai_training/data.py` (MODIFY) — `SubwordCorpus` dataclass + `build_subword_corpus`.
- `src/local_ai_training/config.py` (MODIFY) — add `tokenizer: Literal["char","subword"] = "char"` field, parse + thread it.
- `src/local_ai_training/cli.py` (MODIFY) — `--tokenizer`/`--vocab-size` flags; `_corpus` builds SubwordCorpus (training/caching the tokenizer artifact).
- `src/local_ai_training/checkpoint.py` (MODIFY) — metadata `tokenizer_kind` + embedded tokenizer JSON; save/load.
- `src/local_ai_training/generate.py` (MODIFY) — subword encode-in/decode-out path.
- `configs/enwik8_subword_25m_5k.toml`, `configs/enwik8_subword_25m_30k.toml` (CREATE) — tuning + polished configs.
- Tests: `tests/test_tokenizer.py`, `tests/test_sparse_embedding.py`, `tests/test_subword_corpus.py`, `tests/test_generate.py` (extend), `tests/test_experiment.py` (extend).
- `docs/results/2026-06-26-subword-sparse-embedding-ab.md` (CREATE) — A/B + final write-up.

---

### Task 1: Sparse-aware RMS-EMA update (the research core)

**Files:**
- Modify: `src/local_ai_training/ratchet.py` (`_normalize`, ~lines 540-552)
- Test: `tests/test_sparse_embedding.py` (CREATE)

**Interfaces:**
- Consumes: `DiscreteRatchetLinear._normalize(self, gradient: Tensor, row_start: int, row_end: int) -> Tensor`, buffer `self.rms_ema` (shape `[out_features]`), float `self.rms_ema_beta`.
- Produces: a `_normalize` whose `rms_ema` entries change ONLY for rows with nonzero mean-square gradient this call; zero-gradient rows keep their prior EMA and yield zero normalized output.

**Background:** Current code updates the EMA for every row including zero-gradient rows, decaying rare rows' EMA toward zero so their next firing explodes. Fix: update EMA only where `ms > 0`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sparse_embedding.py
import torch
from local_ai_training.ratchet import DiscreteRatchetLinear


def _layer(rows=4, cols=8, beta=0.9):
    torch.manual_seed(0)
    return DiscreteRatchetLinear(
        cols, rows, max_code=7, pressure_threshold=8, rms_ema_beta=beta
    )


def test_nonfiring_row_keeps_ema_and_yields_zero():
    layer = _layer()
    # Seed EMA on all rows with one full-gradient step.
    g0 = torch.ones(4, 8)
    layer._normalize(g0, 0, 4)
    before = layer.rms_ema.clone()

    # Now only rows 0 and 2 fire; rows 1 and 3 have exactly zero gradient.
    g1 = torch.zeros(4, 8)
    g1[0] = 2.0
    g1[2] = 2.0
    out = layer._normalize(g1, 0, 4)

    # Non-firing rows keep their EMA unchanged.
    assert torch.allclose(layer.rms_ema[1], before[1])
    assert torch.allclose(layer.rms_ema[3], before[3])
    # Firing rows' EMA moved.
    assert not torch.allclose(layer.rms_ema[0], before[0])
    # Non-firing rows produce zero normalized output (no spurious pressure).
    assert torch.allclose(out[1], torch.zeros(8))
    assert torch.allclose(out[3], torch.zeros(8))


def test_rare_row_normalization_is_stable_across_sparse_firings():
    layer = _layer()
    # A row that fires once with magnitude m, after a long gap, must NOT explode:
    # its normalized magnitude should be ~ the same as a row firing every step.
    dense = _layer()
    g = torch.zeros(4, 8)
    g[0] = 3.0
    for _ in range(5):  # dense row fires every step
        dense_out = dense._normalize(g, 0, 4)

    sparse = _layer()
    sparse._normalize(g, 0, 4)  # one firing
    for _ in range(50):  # 50 idle steps where row 0 does NOT fire
        sparse._normalize(torch.zeros(4, 8), 0, 4)
    sparse_out = sparse._normalize(g, 0, 4)  # fires again

    # Sparse firing normalizes to the same scale as dense (within tolerance),
    # because the idle steps did not decay row 0's EMA.
    assert torch.allclose(sparse_out[0], dense_out[0], atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sparse_embedding.py -v`
Expected: both FAIL (current code decays EMA on idle rows → `test_rare_row...` mismatch; `test_nonfiring...` EMA[1]/[3] changed).

- [ ] **Step 3: Implement the masked EMA update**

Replace the `if self.rms_ema_beta > 0.0:` block in `_normalize` with:

```python
        if self.rms_ema_beta > 0.0:
            ema = self.rms_ema[row_start:row_end].unsqueeze(1)
            # Only rows that fired this step (ms > 0) update their EMA; idle rows keep
            # their prior EMA so a rarely-firing row retains a stable normalization
            # constant instead of decaying toward zero (which would explode on its
            # next firing). Uninitialized rows (ema == 0) seed from their first ms.
            fired = ms > 0
            seeded = self.rms_ema_beta * ema + (1.0 - self.rms_ema_beta) * ms
            updated = torch.where(ema == 0, ms, seeded)
            ema = torch.where(fired, updated, ema)
            self.rms_ema[row_start:row_end] = ema.squeeze(1)
            rms = ema.sqrt()
        else:
            rms = ms.sqrt()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sparse_embedding.py -v`
Expected: PASS.

- [ ] **Step 5: Run the existing ratchet suite for no regression**

Run: `uv run pytest tests/test_ratchet.py -q`
Expected: PASS (dense behavior unchanged: when every row fires, `fired` is all-True so behavior matches the prior formula).

- [ ] **Step 6: Commit**

```bash
git add src/local_ai_training/ratchet.py tests/test_sparse_embedding.py
git commit -m "fix: per-row RMS-EMA updates only firing rows (sparse embedding core)

$(printf 'Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011gqtgUHpfZDiNfQ36VTbBB')"
```

---

### Task 2: Thread sparse knobs into RatchetEmbedding

**Files:**
- Modify: `src/local_ai_training/model.py` (~lines 205-216, `RatchetEmbedding(...)` construction)
- Test: `tests/test_sparse_embedding.py` (extend)

**Interfaces:**
- Consumes: `ModelConfig.rms_ema_beta: float`, `ModelConfig.pressure_leak_period: int` (already exist), `RatchetEmbedding(..., rms_ema_beta=..., pressure_leak_period=...)` (already accepted by `__init__`).
- Produces: a `RatchetGPT` whose `token_embedding` (when ratcheted) has `rms_ema_beta`/`pressure_leak_period` equal to the config values.

**Background:** `model.py` constructs `RatchetEmbedding` without passing the two sparse knobs, so they silently default to 0 even when config sets them.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_sparse_embedding.py
from local_ai_training.model import ModelConfig, RatchetGPT
from local_ai_training.ratchet import RatchetEmbedding


def test_ratchet_embedding_receives_sparse_knobs():
    cfg = ModelConfig(
        vocab_size=32, block_size=16, n_layer=1, n_head=2, n_embd=8,
        ratchet_embedding=True, rms_ema_beta=0.9, pressure_leak_period=5,
    )
    model = RatchetGPT(cfg, max_code=7)
    emb = model.token_embedding
    assert isinstance(emb, RatchetEmbedding)
    assert emb.rms_ema_beta == 0.9
    assert emb.pressure_leak_period == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sparse_embedding.py::test_ratchet_embedding_receives_sparse_knobs -v`
Expected: FAIL (`emb.rms_ema_beta == 0.0`).

- [ ] **Step 3: Pass the knobs in `model.py`**

In the `RatchetEmbedding(...)` constructor call, add the two arguments:

```python
            self.token_embedding = RatchetEmbedding(
                config.vocab_size,
                config.n_embd,
                max_code=max_code,
                pressure_threshold=config.pressure_threshold,
                bucket_low=config.bucket_low,
                bucket_high=config.bucket_high,
                rms_ema_beta=config.rms_ema_beta,
                pressure_leak_period=config.pressure_leak_period,
                trainable_scale=config.trainable_scale,
                compile_update=config.compile_update,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sparse_embedding.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/local_ai_training/model.py tests/test_sparse_embedding.py
git commit -m "fix: thread rms_ema_beta/pressure_leak_period into RatchetEmbedding

$(printf 'Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011gqtgUHpfZDiNfQ36VTbBB')"
```

---

### Task 3: Pure-Python word-frequency BPE tokenizer

**Files:**
- Create: `src/local_ai_training/tokenizer.py`
- Test: `tests/test_tokenizer.py`

**Interfaces:**
- Produces:
  - `class BpeTokenizer` with attributes `merges: dict[tuple[int, int], int]` (ordered by insertion = rank), `vocab: dict[int, bytes]`.
  - `@classmethod train(cls, text: str, vocab_size: int) -> BpeTokenizer`
  - `encode(self, text: str) -> list[int]`
  - `decode(self, ids: list[int]) -> str`
  - `to_json(self) -> str` / `@classmethod from_json(cls, data: str) -> BpeTokenizer`
  - property `vocab_size -> int` (== `len(self.vocab)`)
- Conventions: byte-level base (256 tokens, ids 0..255 → `bytes([i])`); latin-1 for str<->bytes; merges never cross pretoken boundaries; pretokenizer regex `re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tokenizer.py
from local_ai_training.tokenizer import BpeTokenizer

SAMPLE = "the cat sat on the mat. the cat ran. " * 50 + "[[Wikipedia]] 1999 caps & PUNCT!"


def test_roundtrip_exact():
    tok = BpeTokenizer.train(SAMPLE, vocab_size=400)
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


def test_roundtrip_unseen_text():
    tok = BpeTokenizer.train(SAMPLE, vocab_size=400)
    other = "the dog sat. 2024 [[Test]]!"
    assert tok.decode(tok.encode(other)) == other


def test_training_deterministic():
    a = BpeTokenizer.train(SAMPLE, vocab_size=400)
    b = BpeTokenizer.train(SAMPLE, vocab_size=400)
    assert a.merges == b.merges
    assert a.vocab == b.vocab


def test_vocab_size_reached():
    tok = BpeTokenizer.train(SAMPLE, vocab_size=400)
    assert tok.vocab_size == 400


def test_json_roundtrip_identical():
    tok = BpeTokenizer.train(SAMPLE, vocab_size=400)
    clone = BpeTokenizer.from_json(tok.to_json())
    assert clone.merges == tok.merges
    assert clone.vocab == tok.vocab
    assert clone.encode(SAMPLE) == tok.encode(SAMPLE)


def test_merges_reduce_token_count():
    tok = BpeTokenizer.train(SAMPLE, vocab_size=400)
    raw = len(SAMPLE.encode("latin-1"))
    assert len(tok.encode(SAMPLE)) < raw
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'local_ai_training.tokenizer'`.

- [ ] **Step 3: Implement the tokenizer**

```python
# src/local_ai_training/tokenizer.py
"""Self-contained word-frequency byte-level BPE (no third-party dependency).

Pretokens are merged independently (merges never cross pretoken boundaries), and
training operates on the de-duplicated pretoken multiset for tractability over large
corpora. Bytes<->str use latin-1 so any byte sequence round-trips exactly.
"""

from __future__ import annotations

import json
import re
from collections import Counter

_PRETOKEN_RE = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+")


def _pretokens(text: str) -> list[str]:
    return _PRETOKEN_RE.findall(text)


def _count_pairs(word_freqs: dict[tuple[int, ...], int]) -> Counter:
    pairs: Counter = Counter()
    for symbols, freq in word_freqs.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs


def _merge_word(symbols: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class BpeTokenizer:
    def __init__(
        self, merges: dict[tuple[int, int], int], vocab: dict[int, bytes]
    ) -> None:
        self.merges = merges
        self.vocab = vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BpeTokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (byte base)")
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # De-duplicate pretokens into a frequency map of byte-id tuples.
        word_freqs: dict[tuple[int, ...], int] = {}
        for pretoken, freq in Counter(_pretokens(text)).items():
            symbols = tuple(pretoken.encode("latin-1"))
            if symbols:
                word_freqs[symbols] = word_freqs.get(symbols, 0) + freq
        merges: dict[tuple[int, int], int] = {}
        next_id = 256
        while next_id < vocab_size:
            pairs = _count_pairs(word_freqs)
            if not pairs:
                break
            # Most frequent pair; ties broken by smallest pair for determinism.
            best = max(pairs.items(), key=lambda kv: (kv[1], (-kv[0][0], -kv[0][1])))[0]
            merges[best] = next_id
            vocab[next_id] = vocab[best[0]] + vocab[best[1]]
            word_freqs = {
                _merge_word(sym, best, next_id): freq for sym, freq in word_freqs.items()
            }
            next_id += 1
        return cls(merges, vocab)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        symbols = list(pretoken.encode("latin-1"))
        while len(symbols) >= 2:
            # Find the pair present with the lowest merge rank (earliest learned).
            best_rank = None
            best_pos = None
            for pos, pair in enumerate(zip(symbols, symbols[1:])):
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pos = pos
            if best_pos is None:
                break
            symbols[best_pos : best_pos + 2] = [self.merges[
                (symbols[best_pos], symbols[best_pos + 1])
            ]]
        return symbols

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for pretoken in _pretokens(text):
            ids.extend(self._encode_pretoken(pretoken))
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("latin-1")

    def to_json(self) -> str:
        return json.dumps(
            {
                "merges": [[a, b, nid] for (a, b), nid in self.merges.items()],
                "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, data: str) -> "BpeTokenizer":
        obj = json.loads(data)
        merges = {(a, b): nid for a, b, nid in obj["merges"]}
        vocab = {int(k): bytes(v) for k, v in obj["vocab"].items()}
        return cls(merges, vocab)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tokenizer.py -v`
Expected: PASS (all 6).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/local_ai_training/tokenizer.py tests/test_tokenizer.py
git add src/local_ai_training/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: pure-Python word-frequency BPE tokenizer

$(printf 'Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011gqtgUHpfZDiNfQ36VTbBB')"
```

---

### Task 4: SubwordCorpus + tokenizer artifact build

**Files:**
- Modify: `src/local_ai_training/data.py`
- Test: `tests/test_subword_corpus.py`

**Interfaces:**
- Consumes: `BpeTokenizer` (Task 3); existing `validation_fraction=0.1` split convention.
- Produces:
  - `@dataclass(frozen=True) class SubwordCorpus` with fields `train_text: str`, `validation_text: str`, `tokenizer: BpeTokenizer`, `train_ids: Tensor`, `validation_ids: Tensor`, and `vocab_size: int`; method `decode(self, token_ids: Tensor) -> str`.
  - `build_subword_corpus(text: str, tokenizer: BpeTokenizer, *, validation_fraction: float = 0.1) -> SubwordCorpus`
  - `train_subword_tokenizer(text: str, *, vocab_size: int, train_chars: int = 10_000_000, validation_fraction: float = 0.1) -> BpeTokenizer` — trains on the first `train_chars` of the TRAIN split only.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_subword_corpus.py
import torch
from local_ai_training.data import build_subword_corpus, train_subword_tokenizer

TEXT = ("the cat sat on the mat. " * 2000) + ("[[Page]] 1999 PUNCT! " * 500)


def test_tokenizer_trains_on_train_split_only():
    # Put a unique marker ONLY in the validation tail; it must not become a merge.
    body = "the cat sat. " * 5000
    tail = "ZZZQQQ " * 200
    text = body + tail
    tok = train_subword_tokenizer(text, vocab_size=300, train_chars=10_000_000)
    # "ZZZQQQ" lives only in the validation 10% tail, so its bytes never merge into one token.
    assert len(tok.encode("ZZZQQQ")) > 1


def test_corpus_split_and_decode_roundtrip():
    tok = train_subword_tokenizer(TEXT, vocab_size=400)
    corpus = build_subword_corpus(TEXT, tok)
    assert corpus.vocab_size == 400
    # ids reconstruct their partitions.
    assert corpus.decode(corpus.train_ids) == corpus.train_text
    assert corpus.decode(corpus.validation_ids) == corpus.validation_text
    # 90/10 split on characters.
    assert len(corpus.validation_text) == int(len(TEXT) * 0.1)


def test_build_is_deterministic():
    tok = train_subword_tokenizer(TEXT, vocab_size=400)
    a = build_subword_corpus(TEXT, tok)
    b = build_subword_corpus(TEXT, tok)
    assert torch.equal(a.train_ids, b.train_ids)
    assert torch.equal(a.validation_ids, b.validation_ids)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_subword_corpus.py -v`
Expected: FAIL with ImportError (`build_subword_corpus`/`train_subword_tokenizer` not defined).

- [ ] **Step 3: Implement in `data.py`**

Add near `CharCorpus` / `build_char_corpus`:

```python
from .tokenizer import BpeTokenizer


@dataclass(frozen=True)
class SubwordCorpus:
    train_text: str
    validation_text: str
    tokenizer: BpeTokenizer
    train_ids: Tensor
    validation_ids: Tensor
    vocab_size: int

    def decode(self, token_ids: Tensor) -> str:
        return self.tokenizer.decode([int(i) for i in token_ids.flatten().tolist()])


def _split_text(text: str, validation_fraction: float) -> tuple[str, str]:
    if not text:
        raise ValueError("corpus text must not be empty")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    validation_length = int(len(text) * validation_fraction)
    if validation_length < 2 or len(text) - validation_length < 2:
        raise ValueError("corpus is too short for train and validation next-token splits")
    return text[:-validation_length], text[-validation_length:]


def train_subword_tokenizer(
    text: str,
    *,
    vocab_size: int,
    train_chars: int = 10_000_000,
    validation_fraction: float = 0.1,
) -> BpeTokenizer:
    train_text, _ = _split_text(text, validation_fraction)
    return BpeTokenizer.train(train_text[:train_chars], vocab_size=vocab_size)


def build_subword_corpus(
    text: str, tokenizer: BpeTokenizer, *, validation_fraction: float = 0.1
) -> SubwordCorpus:
    train_text, validation_text = _split_text(text, validation_fraction)
    return SubwordCorpus(
        train_text=train_text,
        validation_text=validation_text,
        tokenizer=tokenizer,
        train_ids=torch.tensor(tokenizer.encode(train_text), dtype=torch.long),
        validation_ids=torch.tensor(tokenizer.encode(validation_text), dtype=torch.long),
        vocab_size=tokenizer.vocab_size,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_subword_corpus.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/local_ai_training/data.py tests/test_subword_corpus.py
git commit -m "feat: SubwordCorpus + train-split-only tokenizer training

$(printf 'Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011gqtgUHpfZDiNfQ36VTbBB')"
```

---

### Task 5: Config + CLI wiring (`--tokenizer`, artifact caching)

**Files:**
- Modify: `src/local_ai_training/config.py` (add `tokenizer` field + parse + `to_dict`)
- Modify: `src/local_ai_training/cli.py` (`_corpus`, `train`/`generate` argparse)
- Test: `tests/test_experiment.py` (extend with a config-parse test)

**Interfaces:**
- Consumes: `ExperimentConfig` (dataclass with `from_toml`, `to_dict`, `model_config`); `build_subword_corpus`, `train_subword_tokenizer` (Task 4); existing `_corpus(dataset_path, cache_dir)`.
- Produces:
  - `ExperimentConfig.tokenizer: Literal["char", "subword"] = "char"` and `ExperimentConfig.vocab_size: int = 8000` (used only when subword).
  - `_corpus(dataset_path, cache_dir, *, tokenizer: str = "char", vocab_size: int = 8000)` returning `CharCorpus | SubwordCorpus`; for subword it loads `data/<stem>.bpe<vocab_size>.json` if present, else trains + writes it.
  - CLI flags `--tokenizer {char,subword}` (default char) and `--vocab-size INT` (default 8000) on the `train` and `generate` subparsers.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_experiment.py
from dataclasses import replace
from local_ai_training.config import ExperimentConfig


def test_tokenizer_field_defaults_and_parses(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text(
        "[model]\nblock_size=16\nn_layer=1\nn_head=2\nn_embd=8\n"
        "[training]\nbatch_size=2\nsteps=1\nseeds=[1]\n"
        "tokenizer='subword'\nvocab_size=512\n"
    )
    cfg = ExperimentConfig.from_toml(toml)
    assert cfg.tokenizer == "subword"
    assert cfg.vocab_size == 512
    # default stays char
    assert replace(cfg, tokenizer="char").tokenizer == "char"
```

(Confirm the `[training]` keys match the existing `from_toml` section mapping; if `tokenizer`/`vocab_size` belong under a different section in this repo's parser, place them where `from_toml` reads scalars. Check `config.py:from_toml` allowed-keys list and add `"tokenizer"`, `"vocab_size"`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_experiment.py::test_tokenizer_field_defaults_and_parses -v`
Expected: FAIL (unknown key or missing attribute).

- [ ] **Step 3: Add the fields in `config.py`**

- Add to the dataclass (near `ratchet_embedding`):
  ```python
  tokenizer: Literal["char", "subword"] = "char"
  vocab_size: int = 8000
  ```
- Add `"tokenizer"` and `"vocab_size"` to the key list parsed in `from_toml` (same list that currently includes `"rms_ema_beta"`, `"pressure_leak_period"`, `"matmul_mode"`).
- Add a validation line in `__post_init__`: `if self.tokenizer not in {"char", "subword"}: raise ValueError("tokenizer must be char or subword")`.
- Ensure `to_dict` includes them (if `to_dict` uses `asdict`, no change needed; otherwise add the keys).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_experiment.py::test_tokenizer_field_defaults_and_parses -v`
Expected: PASS.

- [ ] **Step 5: Wire `_corpus` + CLI in `cli.py`**

Replace `_corpus` with a tokenizer-aware version and thread the choice. The subword branch caches the artifact next to the corpus file:

```python
def _corpus(dataset_path, cache_dir, *, tokenizer="char", vocab_size=8000):
    path = _dataset_file(dataset_path, cache_dir)  # existing resolution logic
    text = path.read_text(encoding="latin-1")
    if tokenizer == "char":
        return build_char_corpus(text)
    artifact = path.with_suffix(path.suffix + f".bpe{vocab_size}.json")
    if artifact.is_file():
        tok = BpeTokenizer.from_json(artifact.read_text())
    else:
        tok = train_subword_tokenizer(text, vocab_size=vocab_size)
        artifact.write_text(tok.to_json())
    return build_subword_corpus(text, tok)
```

- Import `BpeTokenizer`, `build_subword_corpus`, `train_subword_tokenizer`.
- Add `--tokenizer` (`choices=["char","subword"]`, default `"char"`) and `--vocab-size` (default `8000`) to the `train` and `generate` subparsers.
- In the train handler: pass `tokenizer=args.tokenizer, vocab_size=args.vocab_size` to `_corpus`; if `args.tokenizer == "subword"`, `config = replace(config, tokenizer="subword", vocab_size=args.vocab_size)`. Build the model with `vocab_size = corpus.vocab_size` (already wired via `corpus`); the existing code that reads `len(corpus.vocabulary)` must instead use a corpus-agnostic size — use `getattr(corpus, "vocab_size", None) or len(corpus.vocabulary)`.

- [ ] **Step 6: Run the suite**

Run: `uv run pytest tests/test_experiment.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/local_ai_training/config.py src/local_ai_training/cli.py tests/test_experiment.py
git commit -m "feat: --tokenizer subword wiring with cached BPE artifact

$(printf 'Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011gqtgUHpfZDiNfQ36VTbBB')"
```

---

### Task 6: Subword checkpoint + generate

**Files:**
- Modify: `src/local_ai_training/checkpoint.py` (metadata write/validate)
- Modify: `src/local_ai_training/generate.py` (`load_for_generation`, `generate`)
- Modify: `src/local_ai_training/train.py` (pass tokenizer JSON when saving, if checkpoint save happens there)
- Test: `tests/test_generate.py` (extend)

**Interfaces:**
- Consumes: `BpeTokenizer.to_json/from_json`; existing `save_checkpoint(..., vocabulary=...)` and `load_for_generation(base_path, device=...)`.
- Produces:
  - Checkpoint metadata gains `"tokenizer_kind": "char" | "subword"` and, when subword, `"tokenizer_json": <str>`. Char checkpoints unchanged (`tokenizer_kind` defaults to `"char"` when absent → backward compatible).
  - `load_for_generation` returns `(model, decoder)` where `decoder` is the char tuple (char) or a `BpeTokenizer` (subword); `generate(...)` accepts either and encodes/decodes accordingly.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_generate.py
from local_ai_training.tokenizer import BpeTokenizer
from local_ai_training.generate import generate
# plus the repo's existing helpers to build+save a tiny model checkpoint

def test_subword_generate_roundtrips(tmp_path):
    tok = BpeTokenizer.train("the cat sat on the mat. " * 200, vocab_size=300)
    # Build a tiny ratchet model with vocab_size == tok.vocab_size, save a checkpoint
    # with tokenizer_kind="subword" and tokenizer_json=tok.to_json() (use the repo's
    # save_checkpoint plus the new metadata fields), then:
    model, decoder = load_for_generation(ckpt_base, device="cpu")
    assert isinstance(decoder, BpeTokenizer)
    out = generate(model, decoder, "the cat", max_new_tokens=5, temperature=0.0)
    assert isinstance(out, str) and len(out) > 0
```

(Use the existing checkpoint-save test helper in `tests/` as the template for constructing `ckpt_base`; mirror how `test_generate.py` already builds a char checkpoint, switching the metadata to subword.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate.py::test_subword_generate_roundtrips -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

- `checkpoint.py`: add optional params `tokenizer_kind: str = "char"`, `tokenizer_json: str | None = None` to the save function; write them into `metadata`. On load/validate, read `metadata.get("tokenizer_kind", "char")`.
- `generate.py` `load_for_generation`: after building the model, branch:
  ```python
  kind = metadata.get("tokenizer_kind", "char")
  if kind == "subword":
      decoder = BpeTokenizer.from_json(metadata["tokenizer_json"])
  else:
      decoder = tuple(metadata["vocabulary"])
  return model.to(device).eval(), decoder
  ```
  Build the model with `vocab_size` = `len(vocabulary)` for char, or `decoder.vocab_size` for subword (read from metadata `max_code`/config as today; vocab size for subword = `BpeTokenizer.from_json(...).vocab_size`).
- `generate.py` `generate`: accept `decoder: tuple[str, ...] | BpeTokenizer`. Encode prompt: char → existing `char_to_id`; subword → `decoder.encode(prompt)`. Decode output: char → `"".join(...)`; subword → `decoder.decode(produced)`.
- `train.py`: when saving the checkpoint for a subword run, pass `tokenizer_kind="subword", tokenizer_json=corpus.tokenizer.to_json()`. (Locate the existing `save_checkpoint` call; add the two kwargs guarded by whether the corpus is a `SubwordCorpus`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate.py -v`
Expected: PASS (including the existing char generate tests — no regression).

- [ ] **Step 5: Commit**

```bash
git add src/local_ai_training/checkpoint.py src/local_ai_training/generate.py src/local_ai_training/train.py tests/test_generate.py
git commit -m "feat: subword checkpoints carry embedded tokenizer; generate decodes subword

$(printf 'Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011gqtgUHpfZDiNfQ36VTbBB')"
```

---

### Task 7: End-to-end integration test + audit (subword ratchet embedding)

**Files:**
- Test: `tests/test_experiment.py` (extend)

**Interfaces:**
- Consumes: the full stack (Tasks 1-6); existing tiny-config training helper `small_experiment_config()` (tests/test_experiment.py:19), `audit_no_master_weights`.
- Produces: a fast CPU integration test proving a subword + ratchet-embedding run trains a step, saves, generates, and audits clean.

- [ ] **Step 1: Write the failing/again-green integration test**

```python
# add to tests/test_experiment.py
from local_ai_training.ratchet import audit_no_master_weights

def test_subword_ratchet_embedding_trains_and_audits(tmp_path):
    text = ("the cat sat on the mat. [[Page]] 1999 caps! " * 400)
    corpus_file = tmp_path / "mini"
    corpus_file.write_text(text, encoding="latin-1")
    # Run one short training step via the train entry with:
    #   tokenizer=subword, vocab_size=300, ratchet_embedding=True,
    #   rms_ema_beta=0.9, max_code (codes) set, steps=2, on CPU.
    # Then load the checkpoint and assert audit is clean.
    report = audit_no_master_weights(model)
    assert report.violations == []  # zero FP master weights, incl. token_embedding
    # token_embedding persists only packed + _scale, no optimizer entry.
```

(Model-build + short-run wiring mirrors the existing integration tests in this file; set `device="cpu"`, `steps=2`, `eval_interval=1`. Use `codes` consistent with the repo's CLI→max_code mapping, e.g. `--codes 15` → max_code 7.)

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_experiment.py::test_subword_ratchet_embedding_trains_and_audits -v`
Expected: PASS (all mechanism already implemented; this is the integration gate).

- [ ] **Step 3: Full suite + audit + lint**

```bash
uv run pytest -q
uv run lat audit --model configs/ratchet_tiny.toml
uv run ruff check src/local_ai_training tests
git diff --check
```
Expected: suite green; audit reports no violations; ruff clean on changed files; no whitespace errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_experiment.py
git commit -m "test: end-to-end subword ratchet-embedding train+audit integration

$(printf 'Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011gqtgUHpfZDiNfQ36VTbBB')"
```

---

### Task 8: Configs, tuning sweep, polished run, A/B, write-up

**Files:**
- Create: `configs/enwik8_subword_25m_5k.toml`, `configs/enwik8_subword_25m_30k.toml`
- Create: `docs/results/2026-06-26-subword-sparse-embedding-ab.md`

**Interfaces:**
- Consumes: the full CLI (`lat train`, `lat generate`); enwik8 dataset at `data/enwik8/enwik8`.
- Produces: the two configs, screening + A/B + 30k run artifacts under git-ignored `runs/`, and the results doc.

**Note:** This task runs experiments (GPU, hours). It has no unit test; its "test" is the produced metrics + audit. Keep all runs under `runs/` (git-ignored); never overwrite existing runs. Default GPU 1.

- [ ] **Step 1: Write the 5k tuning config**

`configs/enwik8_subword_25m_5k.toml` — copy `configs/enwik8_25m.toml`'s `[model]`/`[ratchet]` blocks; `[training]` with `steps = 5000`, `seeds = [1337]`, add `tokenizer = "subword"`, `vocab_size = 8000`. Header comment documents it as the subword screening config.

- [ ] **Step 2: Build the tokenizer artifact + sanity-check it**

```bash
CUDA_VISIBLE_DEVICES=1 UV_CACHE_DIR=/games/ailab/.uv-cache uv run lat train \
  --codes 15 --tokenizer subword --vocab-size 8000 \
  --config configs/enwik8_subword_25m_5k.toml --seed 1337 \
  --dataset-path data/enwik8/enwik8 --output runs/subword_smoke --steps 1
```
Expected: writes `data/enwik8/enwik8.bpe8000.json`; one step runs without error. (If `--steps` override is unavailable, use a 1-step config.) Confirm artifact exists and `vocab_size` ≈ 8000.

- [ ] **Step 3: Tune `rms_ema_beta` / `pressure_leak_period` (5k screens)**

Run 2-3 short screens varying `rms_ema_beta` (e.g. 0.9, 0.99) and `pressure_leak_period` (e.g. 0, 200) with `--ratchet-embedding`, into separate `runs/subword_tune_*` outputs. Pick the lowest val loss. Record the chosen values and the screen numbers in the results doc (preserve all screens).

- [ ] **Step 4: Write the 30k config with settled knobs**

`configs/enwik8_subword_25m_30k.toml` — like the 5k config but `steps = 30000`, `eval_interval = 200`, and the tuned `rms_ema_beta`/`pressure_leak_period` baked in. Header documents the recipe + generate command.

- [ ] **Step 5: Run the matched A/B (control = FP embedding, treatment = ratchet embedding)**

```bash
# Control (FP embedding)
CUDA_VISIBLE_DEVICES=1 ... uv run lat train --codes 15 --tokenizer subword --vocab-size 8000 \
  --config configs/enwik8_subword_25m_30k.toml --seed 1337 \
  --dataset-path data/enwik8/enwik8 --output runs/subword_ab_control
# Treatment (ratchet embedding)
CUDA_VISIBLE_DEVICES=1 ... uv run lat train --codes 15 --tokenizer subword --vocab-size 8000 \
  --ratchet-embedding --config configs/enwik8_subword_25m_30k.toml --seed 1337 \
  --dataset-path data/enwik8/enwik8 --output runs/subword_ab_ratchet
```
Expected: both reach step 30000; treatment audits clean (zero FP master weights).

- [ ] **Step 6: Generate samples from the polished (treatment) model**

```bash
uv run lat generate --checkpoint runs/subword_ab_ratchet/checkpoint \
  --prompt "[[History of " --max-new-tokens 200 --temperature 0.8
```
Expected: legible Wikipedia-style subword text.

- [ ] **Step 7: Write the results doc + commit**

`docs/results/2026-06-26-subword-sparse-embedding-ab.md`: tuning table, the converged control-vs-treatment val gap (the sparse-update cost at 8K subword), an audit confirmation, and a sample. State honestly whether sparse ratchet embedding is competitive. Then:

```bash
git add configs/enwik8_subword_25m_5k.toml configs/enwik8_subword_25m_30k.toml docs/results/2026-06-26-subword-sparse-embedding-ab.md
git commit -m "docs: subword sparse-embedding configs + A/B results

$(printf 'Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011gqtgUHpfZDiNfQ36VTbBB')"
```

---

## Self-Review Notes

- **Spec coverage:** Section 1 sparse fix → Task 1; sparse knobs wiring (found bug) → Task 2; Section 2 BPE → Task 3; SubwordCorpus → Task 4; config/CLI → Task 5; Section 3 checkpoint/generate → Task 6; Section 5 audit/integration → Task 7; Section 4 tuning/run/A/B → Task 8. All spec sections mapped.
- **Type consistency:** `BpeTokenizer` (merges/vocab/encode/decode/to_json/from_json/vocab_size) used identically across Tasks 3-8; `SubwordCorpus.vocab_size`/`.tokenizer` consumed in Tasks 5-6; `tokenizer_kind`/`tokenizer_json` metadata keys consistent in Task 6.
- **Known repo-specific verifications for implementers:** exact `from_toml` key-list location and section placement (Task 5 Step 3); exact `save_checkpoint` signature + call site in `train.py` (Task 6); the `len(corpus.vocabulary)` call sites that must become corpus-agnostic (Task 5 Step 5); the CLI→max_code mapping for `--codes 15` (Tasks 7-8). Each is called out inline.
