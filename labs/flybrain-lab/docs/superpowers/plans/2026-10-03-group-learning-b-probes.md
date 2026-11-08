# Group Learning B: Probes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, on the 353 dev and 339 test held-out kernel files that have at least four same-directory training files, how much exact-match material those files hold (census), how much a trained model gains from reading them with no new model code (related-reading probe) and how much of that gain survives handing over only the pooled memory tables (handoff probe), and write the three results up as notebook Runs 50-52.

**Architecture:** `flybrain/data/files.py` and `flybrain/data/prefixes.py` turn the header-delimited corpus into a file table, content hashes, dev/test splits, targets and leakage-safe context material; `flybrain/census.py` bins scored bytes by exact n-gram match in pure Python. `scripts/census.py` and `scripts/probe_commons.py` (`--mode context | handoff`) are thin command lines over importable functions; the probe pools a per-target `[scope, P, partition, condition, code, (count, bits)]` cube into every table, the headroom numbers and a bootstrap interval for the gate number. The handoff mode consumes the stream API of the agent-state plan and is the only part that cannot run before that plan is merged.

**Tech Stack:** Python 3.11, PyTorch (CPU for every test, CUDA for the probe runs), NumPy, pytest, `uv`; no new dependencies. The corpus format is the `\n// ==== <path> ====\n` header of `scripts/build_corpus.py`.

**Spec:** docs/superpowers/specs/2026-10-03-group-learning-design.md (sections 1.1-1.4)

## Global Constraints

- Repo `/games/ailab/lab/flybrain-lab`, branch `feat/group-learning`, Python 3.11 under `uv`; tests run with `uv run pytest -q` (`pyproject.toml` sets `pythonpath = ["."]`); every test is CPU-only and uses a tiny random `GRUHybrid`, never a trained checkpoint.
- Scope: spec sections 1.1-1.4 and the write-ups of their results. Sections 0.x and 1.5-1.6 belong to the other two plans: do not plan or write them, do not define `fixed_windows` (the yardstick plan adds it to `flybrain/data/text.py`), do not touch `scripts/run_sparsela.py`, `scripts/cross_eval.py` or `flybrain/model/*`.
- Corpus `corpora/linux_full.bytes`: 220,000,000 bytes, 12,497 files, each preceded by the header `\n// ==== <relative path> ====\n` (regex `rb"\n// ==== ([^\n]+?) ====\n"`; the first header starts at byte 0; the last file, `fs/bcachefs/xattr.c`, is cut mid-line at byte 220,000,000).
- Training bytes are `[0, 200,000,000)` (`TRAIN_END`). A file is a training file only if it lies entirely inside, so `drivers/net/ethernet/cavium/liquidio/lio_vf_rep.c`, which crosses byte 200,000,000, is not one. The legacy validation window `[200,100,000, 202,100,000)` is used by nothing here.
- Held-out tail = files whose header starts at or after byte 202,100,000 (`TAIL_START`): 999 files. dev = the first 500 (first header at byte 202,117,357), test = the other 499 (first header at byte 211,576,468) (`DEV_FILES = 500`). Anything tuned is tuned on dev; test is scored once per reported number.
- Targets = held-out files with at least 4 same-directory training files, content duplicates excluded: 353 on dev, 339 on test. Of these, 101 (dev) and 83 (test) have at least 2 other same-directory files in their own split (`MIN_NEVER = 2`) and take part in the never-trained pass.
- A stream is the target's header followed by its body. The scored bytes are the first 4,096 body bytes (the whole body if shorter); 2,048 for the transformer arm. Every model also reports its first 2,048 scored bytes. "Same directory" means identical `posixpath.dirname` of the header path.
- Related prefix = the target's same-directory training files, header kept, in path order, cut at P bytes. Random prefix = a seeded draw from training files outside the target's directory whose content differs from the target's (`random.Random(seed * 1_000_003 + target index)`), cut at the related prefix's length. Never-trained prefix = the same-directory files of the target's own split, cut at P bytes. The target and any file with its content hash never appear in a prefix or a reader set.
- Census: n in {8, 12, 16}, P in {8192, 16384, 32768}, seed 0. The probe bins at n = 8 (`BIN_N`). Gate number = H(B) - H(B') at n = 8 and P = 32768 (`GATE_P`) from the alone-condition loss, H(X) = share(X) x (loss(X) - loss(A)); Gate 1 threshold 0.02 bits/byte (Kyle decides).
- Context probe: `m20_linux_depth4x64k_s0` at P in {8192, 16384, 32768} with head chunk 4096; `m20_linux_depth4local_s0` and `m50_linux_hybrid_s0` at P in {2048, 4096}; the transformer `m20_linux_s0_transformer.pt` (17.09M parameters, the "16M" reference) with a 2,048-byte prefix and its first 2,048 target bytes, rotary base NTK-scaled by total length / 2048 for every sequence.
- Handoff probe: `m20_linux_depth4x64k_s0` (no local attention); readers = the first 8,192 bytes (header included) of each related file in path order, at most 32,768 reader bytes per target; alpha in {0.03, 0.1, 0.3, 1.0} chosen on dev by lowest handoff bits/byte, test scored once at that alpha; transfer ratio `(alone - handoff) / (alone - context)` reported only when `alone - context >= 0.01`.
- Comparisons are paired (every condition scores the same target bytes) and the gate number and every reported gain carry a 95 % interval from resampling targets (files) with replacement: `N_BOOT = 1000` resamples, seeded; if the gate interval straddles 0.02 the write-up says so (spec, "Comparisons are paired" and Gate 1).
- Interfaces of the agent-state plan are used exactly as specified and never redefined: `flybrain.model.state.{HeadState, AgentState}`, `GRUHybrid.init_state(batch, device)`, `GRUHybrid.stream(tokens, state)`, `flybrain.group.{Commons, read, merge, preload}`.
- Script style: argparse, `ROOT = Path(__file__).resolve().parents[1]` with the `sys.path` insert of `scripts/cross_eval.py`, JSON to `ROOT / "results" / f"{run_name}.json"`, `main()` under `if __name__ == "__main__"`, refuse to overwrite an existing JSON. Core logic in importable functions.
- Run names: `census_linux_{dev,test}`, `commons_ctx_<model>_{dev,test}` with `<model>` in `x64k`, `local`, `m50`, `tf`, and `commons_handoff_x64k_{dev,test}`.
- GPU 1 only for these probes during the day: run `systemctl --user stop qwen-server.service` first, and `systemctl --user start qwen-server.service` when the queue is empty. GPU 0 may be used only between 22:00 and 05:00.
- Before any GPU command run `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and do not start if another compute process is on the card.
- The power cap stays at 150 W. Kill jobs by explicit PID only (never `pkill -f`).
- New run names only; never overwrite an existing results JSON.
- Do not edit README.md or notes/RESULTS.md.
- Never push. Commit with the repo's configured identity; end each commit message with the Co-Authored-By trailer your session specifies (the commands in this plan use `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`).

## Review Focus

- **Scored bytes misaligned by a prefix.** The loss of byte t is read from the logits at t - 1 and shifts by `len(prefix)`; with an empty prefix a wrong shift is invisible, so it would pass every alone-only check. Pinned by `test_scored_bits_are_the_loss_of_the_right_bytes_under_any_prefix` (Task 5: a model that sees one byte of context must score identical bits under any prefix).
- **Short and empty targets.** 71 of 353 dev and 70 of 339 test targets have bodies under 4,096 bytes, 4 test bodies are under 2,048, and a corpus can end on a header with an empty body (`build_corpus.py` cuts at `max_bytes`; the real last file is cut mid-line): nothing is padded, means use the true count, an empty body is skipped. Pinned by `test_header_at_end_of_file_and_header_cut_by_eof` and `test_read_stream_is_the_header_plus_at_most_n_scored_body_bytes` (Task 1), `test_a_target_with_an_empty_body_is_skipped` (Task 4), `test_an_empty_prefix_scores_exactly_alone_and_short_bodies_are_scored_whole` and `test_a_long_target_is_cut_at_the_cap_and_a_header_at_the_end_of_the_file_is_skipped` (Task 7).
- **Content duplicates and range edges.** The corpus has three exact-duplicate pairs (all across directories, none touching a held-out file today), so only a synthetic test would notice a leak: a directory whose only related file is a copy of the target must not make it a target, a copy in another directory must not enter the random prefix, a file crossing byte 200,000,000 is not a training file, and a never-trained prefix must stay inside the target's own split. Pinned by `test_targets_count_related_files_after_duplicates_are_removed` (Task 1), `test_prefixes_never_contain_the_target_or_a_duplicate_of_it` and `test_never_trained_files_come_from_the_targets_own_split_only` (Task 2).
- **A prefix cut at exactly P bytes.** The cut lands mid-file or inside a header; 18 of 353 dev targets (and about half of the never-trained ones) have fewer than 32,768 related bytes, and the random control must then shrink to the same size or B' is inflated. Pinned by `test_chunks_cut_at_the_limit_inside_a_file_and_inside_a_header` and `test_prefix_pairs_are_nested_and_the_random_prefix_has_the_related_length` (Task 2).
- **Scored bytes with fewer than n bytes of context.** A negative slice start wraps around in Python, so a stream or header shorter than n would silently match the wrong bytes. Pinned by `test_a_scored_byte_with_fewer_than_n_bytes_before_it_is_c_and_never_wraps_around` (Task 3); `test_a_bin_is_the_labs_copyable_mask` (Task 3) ties bin A to `copyable_mask`.

---

### Task 1: File table, content hashes, related files and splits

Every later task starts from the file table of `corpora/linux_full.bytes`. This task adds the header parser, content hashes, the same-directory lookup with its range and duplicate rules, the dev/test split of the held-out tail, the target filter and the stream reader, plus the `corpus` fixture that writes tiny corpora in the real header format.

**Files:**
- Create: `flybrain/data/files.py`
- Create: `tests/conftest.py`
- Test: `tests/test_files.py`

**Interfaces:**
- Consumes: `scripts.build_corpus.build(root, exts, min_size, max_size, max_bytes, seed, out) -> (n_bytes, n_files)` (tests only, to pin the format).
- Produces: constants `TRAIN_END = 200_000_000`, `TAIL_START = 202_100_000`, `DEV_FILES = 500`, `MIN_NEVER = 2`; `FileEntry(path: str, start: int, body: int, end: int)` (frozen, with property `dirname`); `file_table(path) -> list[FileEntry]`; `read_region(path, start, end) -> bytes`; `content_hash(path, entry) -> str`; `hash_table(path, table) -> list[str]`; `related(i, table, lo, hi, hashes) -> list[int]`; `splits(table, tail_start=TAIL_START, dev_files=DEV_FILES) -> dict[str, list[int]]` (keys `dev`, `test`); `split_range(table, split_indices) -> tuple[int, int]`; `targets(table, split_indices, hashes, min_related=4, train_end=TRAIN_END) -> list[int]`; `read_stream(path, entry, n_scored) -> (stream: bytes, body_start: int, scored: int)`; pytest fixture `corpus(files, name='c.bytes') -> Path`.

- [ ] **Step 1: Write the failing test**

`random` and `dataclass` are imported in `tests/conftest.py` now for the fixtures Task 2 appends. If another plan has already created `tests/conftest.py`, append the `write_corpus` function and the `corpus` fixture to it instead of creating the file.

Create `tests/conftest.py`:

```python
"""Shared fixtures: tiny corpora written in the real header format (scripts/build_corpus.py)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pytest


def write_corpus(path: Path, files: list[tuple[str, bytes]]) -> Path:
    """Write files in the header format of scripts/build_corpus.py, in the given order."""
    path.write_bytes(b"".join(b"\n// ==== " + p.encode() + b" ====\n" + body for p, body in files))
    return path


@pytest.fixture
def corpus(tmp_path):
    """corpus(files, name) -> Path of a corpus holding `files` (path, body) in order."""
    return lambda files, name="c.bytes": write_corpus(tmp_path / name, files)
```

Create `tests/test_files.py`:

```python
"""File table, content hashes, same-directory lookups and splits of a header-delimited corpus."""

from __future__ import annotations


def _entries(spec):
    """FileEntry list from (path, start, end); bodies start 20 bytes after the header start."""
    from flybrain.data.files import FileEntry
    return [FileEntry(p, s, s + 20, e) for p, s, e in spec]


def test_file_table_round_trips_a_built_corpus(tmp_path):
    from flybrain.data.files import file_table, read_region
    from scripts.build_corpus import build
    src = tmp_path / "src"; (src / "d1").mkdir(parents=True); (src / "d2").mkdir()
    bodies = {"d1/a.c": b"int a(void){return 1;}\n" * 5, "d1/b.c": b"int b(void){return 2;}\n" * 7, "d2/c.py": b"def c():\n    return 3\n" * 4}
    for rel, b in bodies.items():
        (src / rel).write_bytes(b)
    out = tmp_path / "corpus.bytes"
    n_bytes, n_files = build(src, (".c", ".py"), 1, 100_000, 10_000_000, 0, out)
    table = file_table(out)
    assert n_files == 3 and len(table) == 3 and n_bytes == out.stat().st_size
    assert {e.path for e in table} == set(bodies)
    assert table[0].start == 0 and table[-1].end == n_bytes
    assert all(a.end == b.start for a, b in zip(table, table[1:]))
    for e in table:
        assert read_region(out, e.body, e.end) == bodies[e.path]
        assert read_region(out, e.start, e.body) == b"\n// ==== " + e.path.encode() + b" ====\n"


def test_test_corpus_writer_matches_the_real_corpus_format(tmp_path, corpus):
    from flybrain.data.files import file_table, read_region
    from scripts.build_corpus import build
    src = tmp_path / "src"; (src / "d").mkdir(parents=True)
    (src / "d" / "a.c").write_bytes(b"int a;\n" * 9); (src / "d" / "b.c").write_bytes(b"int b;\n" * 9)
    built = tmp_path / "built.bytes"
    build(src, (".c",), 1, 1000, 10**6, 0, built)
    order = [(e.path, read_region(built, e.body, e.end)) for e in file_table(built)]
    assert corpus(order, "mine.bytes").read_bytes() == built.read_bytes()


def test_header_at_end_of_file_and_header_cut_by_eof(tmp_path):
    from flybrain.data.files import file_table, read_stream
    h = lambda p: b"\n// ==== " + p + b" ====\n"
    whole = tmp_path / "whole.bytes"; whole.write_bytes(h(b"a/x.c") + b"AAA\n" + h(b"b/y.c"))   # last header, empty body
    t = file_table(whole)
    assert [e.path for e in t] == ["a/x.c", "b/y.c"] and t[0].end == t[1].start
    assert t[1].body == t[1].end == whole.stat().st_size
    stream, body_start, scored = read_stream(whole, t[1], 4096)
    assert scored == 0 and stream == h(b"b/y.c") and body_start == len(stream)
    cut = tmp_path / "cut.bytes"; cut.write_bytes(h(b"a/x.c") + b"AAA\n" + h(b"b/y.c")[:-5])      # header cut by EOF: not a file
    t2 = file_table(cut)
    assert [e.path for e in t2] == ["a/x.c"] and t2[0].end == cut.stat().st_size
    empty = tmp_path / "empty.bytes"; empty.write_bytes(b"")
    assert file_table(empty) == []


def test_read_region_reads_only_the_requested_bytes(tmp_path):
    import tracemalloc
    from flybrain.data.files import read_region
    big = tmp_path / "big.bytes"
    with open(big, "wb") as f:
        f.truncate(300_000_000)                      # sparse: no disk, no RAM
        f.seek(150_000_000); f.write(b"MARKER")
    tracemalloc.start()
    got = read_region(big, 150_000_000, 150_000_006)
    peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    assert got == b"MARKER" and peak < 1_000_000


def test_content_hash_depends_on_the_body_only(corpus):
    from flybrain.data.files import content_hash, file_table, hash_table
    p = corpus([("a/x.c", b"same body\n"), ("b/y.c", b"same body\n"), ("b/z.c", b"other body\n")])
    t = file_table(p); h = hash_table(p, t)
    assert h[0] == h[1] != h[2] and h[0] == content_hash(p, t[0])


def test_related_is_same_directory_inside_the_range_without_duplicates_sorted_by_path():
    from flybrain.data.files import related
    table = _entries([("d/c.c", 0, 100), ("d/a.c", 100, 200), ("d/dup.c", 200, 300), ("e/z.c", 300, 400),
                      ("d/b.c", 400, 500), ("d/late.c", 500, 700), ("d/self.c", 700, 800)])
    hashes = ["hc", "ha", "hself", "hz", "hb", "hl", "hself"]    # d/dup.c has the same content as the target d/self.c
    i = 6
    assert related(i, table, 0, 10_000, hashes) == [1, 4, 0, 5]  # a, b, c, late: not self, not the duplicate, not e/z.c
    assert related(i, table, 0, 600, hashes) == [1, 4, 0]        # d/late.c ends past hi: it straddles the range
    assert related(i, table, 100, 600, hashes) == [1, 4]         # d/c.c starts before lo


def test_targets_count_related_files_after_duplicates_are_removed():
    from flybrain.data.files import targets
    only_dup = _entries([("d/dup.c", 0, 100), ("d/t.c", 1000, 1100)])
    assert targets(only_dup, [1], ["ht", "ht"], min_related=1, train_end=500) == []      # the only same-directory file is a copy of the target
    three = _entries([("d/a.c", 0, 100), ("d/b.c", 100, 200), ("d/c.c", 200, 300), ("d/dup.c", 300, 400), ("d/t.c", 1000, 1100)])
    assert targets(three, [4], ["ha", "hb", "hc", "ht", "ht"], min_related=4, train_end=500) == []   # 3 distinct related files
    four = _entries([("d/a.c", 0, 100), ("d/b.c", 100, 200), ("d/c.c", 200, 300), ("d/e.c", 300, 400), ("d/dup.c", 400, 500), ("d/t.c", 1000, 1100)])
    assert targets(four, [5], ["ha", "hb", "hc", "he", "ht", "ht"], min_related=4, train_end=500) == [5]
    assert targets(four, [5], ["ha", "hb", "hc", "he", "ht", "ht"], min_related=4, train_end=350) == []  # d/e.c ends past the training range


def test_splits_take_the_first_dev_files_of_the_tail_in_file_order():
    from flybrain.data.files import DEV_FILES, MIN_NEVER, TAIL_START, TRAIN_END, split_range, splits
    table = _entries([("a", 0, 100), ("b", 100, 200), ("c", 200, 300), ("d", 300, 400), ("e", 400, 500), ("f", 500, 600)])
    sp = splits(table, tail_start=200, dev_files=2)
    assert sp == {"dev": [2, 3], "test": [4, 5]}
    assert split_range(table, sp["dev"]) == (200, 400) and split_range(table, sp["test"]) == (400, 600)
    assert (TRAIN_END, TAIL_START, DEV_FILES, MIN_NEVER) == (200_000_000, 202_100_000, 500, 2)


def test_read_stream_is_the_header_plus_at_most_n_scored_body_bytes(corpus):
    from flybrain.data.files import file_table, read_stream
    p = corpus([("a/x.c", b"0123456789" * 10), ("a/y.c", b"short")])
    t = file_table(p)
    head = b"\n// ==== a/x.c ====\n"
    s, bs, n = read_stream(p, t[0], 25)
    assert n == 25 and bs == len(head) and s == head + (b"0123456789" * 3)[:25]
    s2, bs2, n2 = read_stream(p, t[1], 25)                     # body shorter than the cap: the whole body, nothing padded
    assert n2 == 5 and s2.endswith(b"short") and len(s2) == bs2 + 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_files.py -q`

Expected: FAIL with "ModuleNotFoundError: No module named 'flybrain.data.files'" (9 failed).

- [ ] **Step 3: Write minimal implementation**

Create `flybrain/data/files.py`:

```python
"""File table of a header-delimited byte corpus (the format of scripts/build_corpus.py): offsets, content hashes,
same-directory lookups and the dev/test split of the held-out tail."""

from __future__ import annotations

import hashlib
import mmap
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

HEADER = re.compile(rb"\n// ==== ([^\n]+?) ====\n")
TRAIN_END = 200_000_000   # training reads bytes [0, TRAIN_END)
TAIL_START = 202_100_000  # held-out tail: files whose header starts here or later (past the legacy validation window)
DEV_FILES = 500           # the first DEV_FILES tail files are dev, the rest are test
MIN_NEVER = 2             # never-trained pass: a target's own split must hold this many other same-directory files


@dataclass(frozen=True)
class FileEntry:
    path: str
    start: int  # offset of the header's leading newline
    body: int   # first content byte
    end: int    # start of the next header, or EOF

    @property
    def dirname(self) -> str:
        return posixpath.dirname(self.path)


def file_table(path: str | Path) -> list[FileEntry]:
    """One entry per header, in byte order. A header cut by EOF is not a file (its bytes stay in the previous body)."""
    with open(path, "rb") as f:
        if f.seek(0, 2) == 0:
            return []
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            hits = [(m.group(1).decode("utf-8"), m.start(), m.end()) for m in HEADER.finditer(mm)]
            size = len(mm)
    return [FileEntry(p, s, b, hits[k + 1][1] if k + 1 < len(hits) else size) for k, (p, s, b) in enumerate(hits)]


def read_region(path: str | Path, start: int, end: int) -> bytes:
    """Bytes [start, end) by seek and read; never the whole file."""
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(max(0, end - start))


def content_hash(path: str | Path, entry: FileEntry) -> str:
    """Hash of the body only, so the same content under two paths hashes alike."""
    return hashlib.blake2b(read_region(path, entry.body, entry.end), digest_size=16).hexdigest()


def hash_table(path: str | Path, table: list[FileEntry]) -> list[str]:
    return [content_hash(path, e) for e in table]


def related(i: int, table: list[FileEntry], lo: int, hi: int, hashes: list[str]) -> list[int]:
    """Files in table[i]'s directory that lie entirely inside bytes [lo, hi), minus i and any file with i's content, sorted by path."""
    d = table[i].dirname
    out = [j for j, e in enumerate(table) if j != i and e.dirname == d and e.start >= lo and e.end <= hi and hashes[j] != hashes[i]]
    return sorted(out, key=lambda j: (table[j].path, table[j].start))


def splits(table: list[FileEntry], tail_start: int = TAIL_START, dev_files: int = DEV_FILES) -> dict[str, list[int]]:
    """dev = the first dev_files held-out tail files (header at or after tail_start), test = the rest, both in file order."""
    tail = [i for i, e in enumerate(table) if e.start >= tail_start]
    return {"dev": tail[:dev_files], "test": tail[dev_files:]}


def split_range(table: list[FileEntry], split_indices: list[int]) -> tuple[int, int]:
    """Byte range [lo, hi) covered by a split's files (the tail files are contiguous)."""
    return table[split_indices[0]].start, table[split_indices[-1]].end


def targets(table: list[FileEntry], split_indices: list[int], hashes: list[str], min_related: int = 4, train_end: int = TRAIN_END) -> list[int]:
    """Held-out files with at least min_related same-directory files entirely inside the training range."""
    return [i for i in split_indices if len(related(i, table, 0, train_end, hashes)) >= min_related]


def read_stream(path: str | Path, entry: FileEntry, n_scored: int) -> tuple[bytes, int, int]:
    """(header + the first n_scored body bytes, body_start = header length, bytes actually scored: fewer when the body is shorter)."""
    scored = min(n_scored, entry.end - entry.body)
    return read_region(path, entry.start, entry.body + scored), entry.body - entry.start, scored
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_files.py -q`

Expected: PASS (9 passed).

- [ ] **Step 5: Check the table against the real corpus (read-only, about 5 s)**

Run:

```bash
uv run python - <<'EOF'
from flybrain.data.files import MIN_NEVER, file_table, hash_table, related, split_range, splits, targets
path = "corpora/linux_full.bytes"
t = file_table(path); h = hash_table(path, t); sp = splits(t)
print(len(t), len(sp["dev"]) + len(sp["test"]), len(sp["dev"]), len(sp["test"]), t[sp["dev"][0]].start, t[sp["test"][0]].start)
tg = {s: targets(t, sp[s], h) for s in sp}
print(len(tg["dev"]), len(tg["test"]))
for s in sp:
    lo, hi = split_range(t, sp[s])
    print(s, "never-trained targets", sum(1 for i in tg[s] if len(related(i, t, lo, hi, h)) >= MIN_NEVER))
EOF
```

Expected output:

```text
12497 999 500 499 202117357 211576468
353 339
dev never-trained targets 101
test never-trained targets 83
```

Any other number means the header regex, the range rule or the duplicate rule differs from the spec's counts (12,497 files, 999 in the tail, 500 / 499, first headers at 202,117,357 / 211,576,468, 353 / 339 targets, about 100 / 83 never-trained targets): stop and find out why.

- [ ] **Step 6: Commit**

```bash
git add flybrain/data/files.py tests/conftest.py tests/test_files.py
git commit -m "File table, content hashes, related files and dev/test splits for the header-delimited corpus (Stage 1.1)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Leakage-safe context material

Both scripts need the same prefixes, so the leakage rules live in one module: the related prefix, the equal-size seeded random prefix, the never-trained prefix (a `[lo, hi)` byte range) and the per-file reader chunks of the handoff probe. The `world` fixture is a 22-file corpus with the awkward cases in it: a content twin of a held-out file in another directory and one in the same directory, a file straddling the training end, a 90-byte target and four directories.

**Files:**
- Create: `flybrain/data/prefixes.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_prefixes.py`

**Interfaces:**
- Consumes: Task 1: `FileEntry`, `related`, `read_region`, `TRAIN_END`; fixture `corpus`'s `write_corpus`.
- Produces: `chunks(path, table, order, limit, per_file=None) -> list[tuple[int, bytes]]` (file index, bytes; header + body, each cut at `per_file`, all cut at `limit`); `joined(cs) -> bytes`; `random_order(table, i, hashes, seed, train_end=TRAIN_END) -> list[int]`; `prefix_pairs(path, table, i, hashes, ps, seed, lo=0, hi=TRAIN_END, train_end=TRAIN_END) -> dict[int, tuple[bytes, bytes]]` (P -> (related prefix, random prefix of the same length)); `reader_chunks(path, table, i, hashes, budget=32_768, per_file=8192, lo=0, hi=TRAIN_END) -> list[tuple[int, bytes]]`; fixtures `world` (`World(path, table, hashes, train_end, tail_start, dev_files, idx)`) and constants/`tiny_world_files()` in `tests/conftest.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/conftest.py`:

```python
PHRASE_K = b"static int probe_device(struct device *dev) { return -ENODEV; }\n"
PHRASE_Z = b"void zeta_flush(struct zeta *z) { z->pending = 0; wmb(); }\n"
PHRASE_W = b"int omega_open(struct inode *i, struct file *f) { return 0; }\n"
COMMON = b"#include <linux/kernel.h>\n#include <linux/module.h>\n"


def tiny_world_files() -> list[tuple[str, bytes]]:
    """Training files (dirs k/, z/ and w/; a copy of tail target k/t0.c under y/ and a copy of k/t1.c in k/; a k/ file straddling the
    training end), then the held-out tail: k/t0 k/t1 k/t2 (dev) and k/t3 z/t4 z/t5 (test). Files of one directory share a phrase, which opens every body."""
    rng = random.Random(0)
    alphabet = b"abcdefghijklmnopqrstuvwxyz ;(){}\n"

    def body(phrase: bytes, n: int) -> bytes:
        noise = lambda m: bytes(rng.choice(alphabet) for _ in range(m))
        return COMMON + phrase + noise(n) + phrase

    t0, t1 = body(PHRASE_K, 600), body(PHRASE_K, 700)
    files = [(f"k/a{j}.c", body(PHRASE_K, 500)) for j in range(5)] + [("k/a_twin.c", t1)]
    files += [(f"z/b{j}.c", body(PHRASE_Z, 500)) for j in range(4)]
    files += [(f"w/c{j}.c", body(PHRASE_W, 500)) for j in range(4)]
    files += [("y/twin.c", t0), ("k/straddle.c", body(PHRASE_K, 500))]
    files += [("k/t0.c", t0), ("k/t1.c", t1), ("k/t2.c", (COMMON + PHRASE_K)[:90])]
    files += [("k/t3.c", body(PHRASE_K, 500)), ("z/t4.c", body(PHRASE_Z, 500)), ("z/t5.c", body(PHRASE_Z, 600))]
    return files


@dataclass
class World:
    path: Path
    table: list
    hashes: list
    train_end: int
    tail_start: int
    dev_files: int
    idx: dict  # corpus path -> index in table


@pytest.fixture
def world(tmp_path) -> World:
    from flybrain.data.files import file_table, hash_table
    path = write_corpus(tmp_path / "world.bytes", tiny_world_files())
    table = file_table(path)
    idx = {e.path: i for i, e in enumerate(table)}
    return World(path, table, hash_table(path, table), table[idx["k/straddle.c"]].start + 50, table[idx["k/t0.c"]].start, 3, idx)
```

Create `tests/test_prefixes.py`:

```python
"""Context material for held-out targets: the leakage rules, the byte cut and the equal-size random control."""

from __future__ import annotations


def _body(world, name):
    from flybrain.data.files import read_region
    e = world.table[world.idx[name]]
    return read_region(world.path, e.body, e.end)


def _region(world, name):
    from flybrain.data.files import read_region
    e = world.table[world.idx[name]]
    return read_region(world.path, e.start, e.end)


def test_prefixes_never_contain_the_target_or_a_duplicate_of_it(world):
    from flybrain.data.files import related
    from flybrain.data.prefixes import prefix_pairs, random_order, reader_chunks
    i = world.idx["k/t0.c"]
    needle = _body(world, "k/t0.c")[200:260]      # inside the random noise, which no other file shares
    assert needle in _body(world, "y/twin.c")                      # the identical body sits in another directory
    for seed in range(20):
        rel, rnd = prefix_pairs(world.path, world.table, i, world.hashes, (100_000,), seed, 0, world.train_end, world.train_end)[100_000]
        assert needle not in rel and needle not in rnd and len(rel) > 0
        assert world.idx["y/twin.c"] not in random_order(world.table, i, world.hashes, seed, world.train_end)
    j, twin = world.idx["k/t1.c"], world.idx["k/a_twin.c"]                      # a same-directory training copy of k/t1.c
    needle_j = _body(world, "k/t1.c")[200:260]
    assert needle_j in _body(world, "k/a_twin.c")
    rel, rnd = prefix_pairs(world.path, world.table, j, world.hashes, (100_000,), 0, 0, world.train_end, world.train_end)[100_000]
    assert needle_j not in rel and needle_j not in rnd
    assert twin not in related(j, world.table, 0, world.train_end, world.hashes)
    assert twin not in {k for k, _ in reader_chunks(world.path, world.table, j, world.hashes, 10**9, 10**9, 0, world.train_end)}
    assert twin in related(i, world.table, 0, world.train_end, world.hashes)    # but it is ordinary material for a different target


def test_random_prefix_has_no_same_directory_file_and_no_file_outside_the_training_range(world):
    from flybrain.data.files import splits
    from flybrain.data.prefixes import random_order
    sp = splits(world.table, world.tail_start, world.dev_files)
    for i in sp["dev"] + sp["test"]:
        order = random_order(world.table, i, world.hashes, 0, world.train_end)
        assert order and i not in order
        assert all(world.table[j].dirname != world.table[i].dirname for j in order)
        assert all(world.table[j].end <= world.train_end for j in order)
        assert all(world.hashes[j] != world.hashes[i] for j in order)
        assert world.idx["k/straddle.c"] not in order and world.idx["k/t3.c"] not in order


def test_random_order_is_deterministic_for_a_seed_and_changes_with_it(world):
    from flybrain.data.prefixes import random_order
    i = world.idx["k/t1.c"]
    a = random_order(world.table, i, world.hashes, 0, world.train_end)
    assert a == random_order(world.table, i, world.hashes, 0, world.train_end)
    assert any(random_order(world.table, i, world.hashes, s, world.train_end) != a for s in range(1, 6))
    j = world.idx["k/t2.c"]                                                   # same directory, same pool, different target: its own shuffle
    assert sorted(random_order(world.table, j, world.hashes, 0, world.train_end)) == sorted(a)
    assert random_order(world.table, j, world.hashes, 0, world.train_end) != a


def test_chunks_cut_at_the_limit_inside_a_file_and_inside_a_header(world):
    from flybrain.data.prefixes import chunks, joined
    a0, a1 = world.idx["k/a0.c"], world.idx["k/a1.c"]
    full0 = _region(world, "k/a0.c")
    cut = chunks(world.path, world.table, [a0, a1], len(full0) + 10)       # 10 bytes into the next file: inside its header
    assert [j for j, _ in cut] == [a0, a1]
    assert joined(cut) == full0 + b"\n// ==== k"
    assert joined(chunks(world.path, world.table, [a0, a1], 5)) == full0[:5]
    assert chunks(world.path, world.table, [a0, a1], 0) == []
    assert len(joined(chunks(world.path, world.table, [a0, a1], 10**9))) == len(full0) + len(_region(world, "k/a1.c"))


def test_prefix_pairs_are_nested_and_the_random_prefix_has_the_related_length(world):
    from flybrain.data.prefixes import prefix_pairs
    i = world.idx["k/t0.c"]
    pairs = prefix_pairs(world.path, world.table, i, world.hashes, (0, 128, 256, 100_000), 0, 0, world.train_end, world.train_end)
    assert pairs[0] == (b"", b"")                                           # P = 0 is the empty prefix
    assert pairs[256][0].startswith(pairs[128][0]) and pairs[256][1].startswith(pairs[128][1])
    assert all(len(r) == len(q) == p for p, (r, q) in pairs.items() if p in (128, 256))
    material = sum(world.table[j].end - world.table[j].start for j in range(len(world.table)) if world.table[j].dirname == "k" and world.table[j].end <= world.train_end)
    assert len(pairs[100_000][0]) == material                               # fewer related bytes than P: the whole related set
    assert len(pairs[100_000][1]) == material                               # and a random prefix of the same size, not of size P


def test_never_trained_files_come_from_the_targets_own_split_only(world):
    from flybrain.data.files import related, splits
    from flybrain.data.prefixes import prefix_pairs
    sp = splits(world.table, world.tail_start, world.dev_files)
    ranges = {s: (world.table[sp[s][0]].start, world.table[sp[s][-1]].end) for s in sp}
    for s in sp:
        lo, hi = ranges[s]
        for i in sp[s]:
            assert set(related(i, world.table, lo, hi, world.hashes)) <= set(sp[s]) - {i}
    lo, hi = ranges["dev"]
    i = world.idx["k/t0.c"]
    assert [world.table[j].path for j in related(i, world.table, lo, hi, world.hashes)] == ["k/t1.c", "k/t2.c"]
    rel, rnd = prefix_pairs(world.path, world.table, i, world.hashes, (100_000,), 0, lo, hi, world.train_end)[100_000]
    assert rel == _region(world, "k/t1.c") + _region(world, "k/t2.c") and len(rnd) == len(rel)
    assert _body(world, "k/t3.c")[200:260] not in rel                       # a test-split file never reaches a dev prefix


def test_reader_chunks_cap_each_file_and_the_total_budget(world):
    from flybrain.data.prefixes import reader_chunks
    i = world.idx["k/t1.c"]
    cs = reader_chunks(world.path, world.table, i, world.hashes, budget=1000, per_file=300, lo=0, hi=world.train_end)
    assert [len(b) for _, b in cs] == [300, 300, 300, 100]
    assert [world.table[j].path for j, _ in cs] == ["k/a0.c", "k/a1.c", "k/a2.c", "k/a3.c"]
    assert all(b.startswith((b"\n// ==== " + world.table[j].path.encode() + b" ====\n")[:20]) for j, b in cs)
    assert all(world.idx[p] not in {j for j, _ in cs} for p in ("k/t1.c", "k/straddle.c", "y/twin.c"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prefixes.py -q`

Expected: FAIL with "ModuleNotFoundError: No module named 'flybrain.data.prefixes'" (7 failed).

- [ ] **Step 3: Write minimal implementation**

Create `flybrain/data/prefixes.py`:

```python
"""Context material for a held-out target: its same-directory files or a seeded random draw from other directories, cut at a byte budget.

Every function here applies the leakage rules: the target and any file with its content never appear, random material never comes
from the target's directory, and only files entirely inside the training range are drawn at random."""

from __future__ import annotations

import random
from pathlib import Path

from flybrain.data.files import TRAIN_END, FileEntry, read_region, related


def chunks(path: str | Path, table: list[FileEntry], order: list[int], limit: int, per_file: int | None = None) -> list[tuple[int, bytes]]:
    """(file index, bytes) for the files of `order`: header + body, each cut at per_file bytes, the whole list cut at `limit` bytes."""
    out: list[tuple[int, bytes]] = []
    have = 0
    for j in order:
        if have >= limit:
            break
        e = table[j]
        n = min(limit - have, e.end - e.start if per_file is None else min(per_file, e.end - e.start))
        if n > 0:
            out.append((j, read_region(path, e.start, e.start + n)))
            have += n
    return out


def joined(cs: list[tuple[int, bytes]]) -> bytes:
    return b"".join(b for _, b in cs)


def random_order(table: list[FileEntry], i: int, hashes: list[str], seed: int, train_end: int = TRAIN_END) -> list[int]:
    """Seeded permutation of the training-range files outside table[i]'s directory whose content differs from table[i]'s."""
    d = table[i].dirname
    pool = [j for j, e in enumerate(table) if e.end <= train_end and e.dirname != d and hashes[j] != hashes[i]]
    random.Random(seed * 1_000_003 + i).shuffle(pool)
    return pool


def prefix_pairs(path: str | Path, table: list[FileEntry], i: int, hashes: list[str], ps: list[int] | tuple[int, ...], seed: int,
                 lo: int = 0, hi: int = TRAIN_END, train_end: int = TRAIN_END) -> dict[int, tuple[bytes, bytes]]:
    """For each P: (related prefix, random prefix). Related = table[i]'s same-directory files inside [lo, hi) in path order, cut at P bytes.
    Random = the seeded draw of `random_order`, cut at the related prefix's length so both sets have equal size."""
    rel = related(i, table, lo, hi, hashes)
    order = random_order(table, i, hashes, seed, train_end)
    out = {}
    for p in ps:
        r = joined(chunks(path, table, rel, p))
        out[p] = (r, joined(chunks(path, table, order, len(r))))
    return out


def reader_chunks(path: str | Path, table: list[FileEntry], i: int, hashes: list[str], budget: int = 32_768, per_file: int = 8192,
                  lo: int = 0, hi: int = TRAIN_END) -> list[tuple[int, bytes]]:
    """Reader material: the first per_file bytes (header included) of each related file in path order, `budget` bytes in all."""
    return chunks(path, table, related(i, table, lo, hi, hashes), budget, per_file)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_files.py tests/test_prefixes.py -q`

Expected: PASS (16 passed).

- [ ] **Step 5: Commit**

```bash
git add flybrain/data/prefixes.py tests/conftest.py tests/test_prefixes.py
git commit -m "Leakage-safe related, random and never-trained prefixes and reader chunks (Stage 1.2-1.4)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Census bins

The census is pure Python on bytes, so it is tested without a model: `bins` must reproduce the lab's `copyable_mask` for A, add B from a prefix on top of it, and never wrap a negative slice.

**Files:**
- Create: `flybrain/census.py`
- Test: `tests/test_census.py`

**Interfaces:**
- Consumes: `flybrain.data.corpus.copyable_mask(x, y, n)` (tests only).
- Produces: `A, B, C = 0, 1, 2`; `ngram_next(source: bytes, n: int) -> dict[bytes, set[int]]`; `bins(stream, body_start, n_scored, prefix, n) -> list[int]` (one code per scored byte; the same call with a random prefix gives B' in place of B).

- [ ] **Step 1: Write the failing test**

Create `tests/test_census.py`:

```python
"""Census bins: self-copyable (A), prefix-only (B) and unmatched (C) bytes."""

from __future__ import annotations


def test_ngram_next_maps_each_window_to_its_continuations():
    from flybrain.census import ngram_next
    assert ngram_next(b"abab", 2) == {b"ab": {ord("a")}, b"ba": {ord("b")}}
    assert ngram_next(b"abc", 3) == {} and ngram_next(b"", 2) == {}


def test_bins_on_a_hand_built_stream():
    from flybrain.census import A, B, C, bins
    stream = b"xxabcdabcd"                      # header "xx", body "abcdabcd"; n = 3
    prefix = b"bcdaQcdababcd"                   # holds bcd->a, cda->{Q, b}, dab->a, abc->d
    #            q:   2  3  4  5  6  7  8  9      a  b  c  d  a  b  c  d
    assert bins(stream, 2, 8, prefix, 3) == [C, C, C, B, B, B, C, A]
    # q=2 has only two bytes before it; q=3 'xxa', q=4 'xab' occur nowhere; q=5 abc->d and q=6 bcd->a come from the prefix;
    # q=7 cda->b is one of cda's two continuations in the prefix; q=8 dab->c is not (dab->a only); q=9 abc->d was seen at q=5 (A beats B)
    assert bins(stream, 2, 8, b"", 3) == [C, C, C, C, C, C, C, A]            # no prefix: only self-copies
    assert bins(stream, 6, 4, prefix, 3) == [B, B, C, A]                       # a shorter scored region keeps the header and earlier body as context


def test_a_bin_is_the_labs_copyable_mask():
    import torch
    from flybrain.census import A, bins
    from flybrain.data.corpus import copyable_mask
    torch.manual_seed(0)
    s = torch.randint(0, 4, (1, 300))           # tiny alphabet: repeats everywhere
    stream = bytes(s[0].tolist())
    for n in (2, 4, 8):
        codes = bins(stream, 1, len(stream) - 1, b"", n)
        assert [c == A for c in codes] == copyable_mask(s[:, :-1], s[:, 1:], n=n)[0].tolist(), n


def test_a_scored_byte_with_fewer_than_n_bytes_before_it_is_c_and_never_wraps_around():
    from flybrain.census import A, B, C, bins
    assert bins(b"abc", 1, 2, b"abcabcabcabc", 8) == [C, C]                   # the whole stream is shorter than n
    assert bins(b"abababab", 0, 8, b"abababab", 4) == [C, C, C, C, B, B, A, A]  # q < n has no window (a negative slice would wrap)
    assert bins(b"hello", 5, 0, b"hello", 2) == []                             # nothing scored


def test_bins_reject_a_scored_region_outside_the_stream():
    import pytest
    from flybrain.census import bins
    with pytest.raises(ValueError):
        bins(b"abc", 2, 5, b"", 2)
    with pytest.raises(ValueError):
        bins(b"abc", -1, 2, b"", 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_census.py -q`

Expected: FAIL with "ModuleNotFoundError: No module named 'flybrain.census'" (5 failed).

- [ ] **Step 3: Write minimal implementation**

Create `flybrain/census.py`:

```python
"""Exact-match census of scored bytes. A byte is A if it continues an n-gram seen earlier in its own stream, B if (not A) the
n-gram and its continuation occur in a prefix, C otherwise. Pure Python: no model, no GPU."""

from __future__ import annotations

A, B, C = 0, 1, 2  # codes returned by bins()


def ngram_next(source: bytes, n: int) -> dict[bytes, set[int]]:
    """Every n-byte window of `source` -> the set of bytes that followed it somewhere in `source`."""
    out: dict[bytes, set[int]] = {}
    for q in range(n, len(source)):
        out.setdefault(source[q - n : q], set()).add(source[q])
    return out


def bins(stream: bytes, body_start: int, n_scored: int, prefix: bytes, n: int) -> list[int]:
    """One code per scored byte stream[body_start : body_start + n_scored]. The n bytes before a scored byte may reach back into
    the header; a byte with fewer than n bytes before it has no n-gram and is C. The same call with a random prefix gives B' for B."""
    end = body_start + n_scored
    if body_start < 0 or n_scored < 0 or end > len(stream):
        raise ValueError(f"scored region [{body_start}, {end}) is outside a stream of {len(stream)} bytes")
    after = ngram_next(prefix, n)
    seen: dict[bytes, set[int]] = {}
    codes = [C] * max(0, min(n, end) - body_start)
    for q in range(n, end):
        g, b = stream[q - n : q], stream[q]
        if q >= body_start:
            codes.append(A if b in seen.get(g, ()) else B if b in after.get(g, ()) else C)
        seen.setdefault(g, set()).add(b)
    return codes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_census.py -q`

Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add flybrain/census.py tests/test_census.py
git commit -m "Census bins: self-copyable, prefix-only and unmatched bytes by exact n-gram match (Stage 1.2)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Census script

`scripts/census.py` runs the census over a split and writes `results/<run-name>.json`. All logic is in `run_census`; `main` only parses flags, refuses an existing run name and prints the tables. The never-trained pass is part of the same run (`--never-trained`), so the two passes share one JSON; its random control is drawn from the training range and cut at the never-trained prefix's length.

**Files:**
- Create: `scripts/census.py`
- Test: `tests/test_census_script.py`

**Interfaces:**
- Consumes: Task 1: `file_table`, `hash_table`, `splits`, `split_range`, `targets`, `read_stream`, `related`, `MIN_NEVER`; Task 2: `prefix_pairs`; Task 3: `bins`, `A`, `B`, `C`.
- Produces: `run_census(path, split, ns=NS, ps=PS, seed=0, never_trained=False, n_scored=4096, train_end=TRAIN_END, tail_start=TAIL_START, dev_files=DEV_FILES, limit=0) -> dict`; `count_cell`, `count_target`, `pool`, `render(res) -> str`, `main(argv=None)`; constants `NS = (8, 12, 16)`, `PS = (8192, 16384, 32768)`, `KEYS`. Writes `results/<run-name>.json` with this schema (counts are bytes; `Bp` is B', `Cp` is C' = not A and not B'):

```text
{"run_name": str, "args": {...}, "split": "dev"|"test", "ns": [8, 12, 16], "ps": [8192, 16384, 32768], "n_targets": int, "n_scored_bytes": int,
 "trained": {"n_targets": int, "n_scored_bytes": int,
             "cells": {"<n>": {"<P>": {"A": int, "B": int, "Bp": int, "C": int, "Cp": int, "total": int,
                                  "share": {"A": float, "B": float, "Bp": float, "C": float, "Cp": float}}}}},
 "never_trained": null | {"n_targets": int, "n_scored_bytes": int, "cells": {same as trained.cells},
                          "trained_cells": {same, pooled over the same targets with the trained related files}},
 "targets": [{"index": int, "path": str, "n_scored": int, "related_files": int,
              "cells": {"<n>": {"<P>": {"A","B","Bp","C","Cp","total": int}}},
              "never_files": int | null, "never_cells": same as cells | null}]}
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_census_script.py`:

```python
"""scripts/census.py on a tiny corpus in the real header format."""

from __future__ import annotations

import json

import pytest


def _run(world, split="dev", seed=0, never=True, ps=(128, 256, 100_000), ns=(4, 8), limit=0):
    from scripts.census import run_census
    return run_census(world.path, split, ns, ps, seed, never, 300, world.train_end, world.tail_start, world.dev_files, limit)


def test_counts_partition_the_scored_bytes_and_related_files_beat_random_ones(world):
    res = _run(world)
    assert [r["path"] for r in res["targets"]] == ["k/t0.c", "k/t1.c", "k/t2.c"]
    assert [r["n_scored"] for r in res["targets"]] == [300, 300, 90]           # the short body is scored whole, nothing padded
    assert res["n_targets"] == 3 and res["n_scored_bytes"] == 690
    for n, row in res["trained"]["cells"].items():
        for p, c in row.items():
            assert c["total"] == 690
            assert c["A"] + c["B"] + c["C"] == c["total"] == c["A"] + c["Bp"] + c["Cp"], (n, p)
            assert abs(sum(c["share"][k] for k in ("A", "B", "C")) - 1.0) < 1e-12
    t0 = res["targets"][0]["cells"]["8"]["100000"]                              # k/t0.c: its copy in y/ is not drawn, so the random pool has no k phrase
    assert t0["B"] > t0["Bp"] > 0                                              # same-directory files share the phrase, random ones only the includes
    small, large = res["trained"]["cells"]["8"]["128"], res["trained"]["cells"]["8"]["100000"]
    assert large["B"] >= small["B"] and large["Bp"] >= small["Bp"]            # more prefix, no fewer matches


def test_never_trained_pass_uses_only_the_targets_own_split(world):
    dev = _run(world, "dev")
    nt = dev["never_trained"]
    assert nt["n_targets"] == 3 and [r["never_files"] for r in dev["targets"]] == [2, 2, 2]
    assert nt["n_scored_bytes"] == 690 and set(nt["trained_cells"]) == {"4", "8"}
    test = _run(world, "test")
    assert [r["path"] for r in test["targets"]] == ["k/t3.c", "z/t4.c", "z/t5.c"]
    assert test["never_trained"]["n_targets"] == 0                           # no target has two other files in the test split
    assert all(c["share"]["A"] is None for row in test["never_trained"]["cells"].values() for c in row.values())
    assert _run(world, "dev", never=False)["never_trained"] is None


def test_census_is_deterministic_for_a_fixed_seed_and_the_seed_reaches_only_the_random_prefix(world, monkeypatch):
    import scripts.census as census
    a, b = _run(world, seed=0), _run(world, seed=0)
    assert json.dumps(a) == json.dumps(b)
    seen, real = [], census.prefix_pairs
    monkeypatch.setattr(census, "prefix_pairs", lambda *args, **kw: (seen.append(args[5]), real(*args, **kw))[1])   # args[5] is the seed
    c = _run(world, seed=7)
    assert set(seen) == {7} and len(seen) == 6                                  # one trained and one never-trained draw per target
    for n, row in a["trained"]["cells"].items():
        for p, cell in row.items():
            other = c["trained"]["cells"][n][p]
            assert (cell["A"], cell["B"], cell["C"]) == (other["A"], other["B"], other["C"])   # related prefixes and self-matches do not depend on the seed


def test_limit_keeps_the_first_targets_only(world):
    assert [r["path"] for r in _run(world, limit=2)["targets"]] == ["k/t0.c", "k/t1.c"]


def test_render_prints_one_table_per_n(world):
    from scripts.census import render
    text = render({"run_name": "census_tiny", **_run(world)})
    assert text.count("| P | A | B | B' | C | C' |") == 2 * 3                  # n = 4, 8 for the trained, matched-trained and never-trained blocks
    assert "n = 8" in text and "| 100000 |" in text and "never trained" in text


def test_main_writes_the_json_and_refuses_an_existing_run_name(world, tmp_path, monkeypatch, capsys):
    import scripts.census as census
    monkeypatch.setattr(census, "RESULTS", tmp_path)
    argv = ["--split", "dev", "--run-name", "census_tiny", "--corpus", str(world.path), "--ns", "4,8", "--ps", "128,256", "--n-scored", "300",
            "--never-trained", "--train-end", str(world.train_end), "--tail-start", str(world.tail_start), "--dev-files", "3"]
    census.main(argv)
    res = json.loads((tmp_path / "census_tiny.json").read_text())
    assert {"run_name", "args", "split", "ns", "ps", "n_targets", "n_scored_bytes", "trained", "never_trained", "targets"} <= set(res)
    assert res["ns"] == [4, 8] and res["ps"] == [128, 256] and "census_tiny" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        census.main(argv)


def test_a_target_with_an_empty_body_is_skipped(corpus):
    from flybrain.data.files import file_table
    from scripts.census import run_census
    files = [(f"d/a{j}.c", (b"int a%d;\n" % j) * 30) for j in range(4)] + [("d/ok.c", b"int ok;\n" * 30), ("d/empty.c", b"")]
    path = corpus(files)
    cut = file_table(path)[4].start                    # training ends where d/ok.c starts; the tail is ok.c and a last, empty header
    res = run_census(path, "dev", (4,), (64,), 0, False, 100, cut, cut, 2)
    assert [r["path"] for r in res["targets"]] == ["d/ok.c"] and res["n_scored_bytes"] == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_census_script.py -q`

Expected: FAIL with "ModuleNotFoundError: No module named 'scripts.census'" (7 failed).

- [ ] **Step 3: Write minimal implementation**

Create `scripts/census.py`:

```python
#!/usr/bin/env python3
"""Exact-match census of the held-out targets. CPU only.

Per target (a held-out file with >= 4 same-directory training files), n and P: how many of its first scored body bytes are
A  (continue an n-gram seen earlier in the target itself),
B  (not A; the n-gram and its continuation occur in the related prefix: its same-directory training files, path order, cut at P bytes),
B' (not A; the same test against an equal-size random prefix from other directories),
C  (neither). --never-trained adds a second pass whose related prefix comes from same-directory files of the target's own split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.census import A, B, C, bins
from flybrain.data.files import DEV_FILES, MIN_NEVER, TAIL_START, TRAIN_END, file_table, hash_table, read_stream, related, split_range, splits, targets
from flybrain.data.prefixes import prefix_pairs

RESULTS = ROOT / "results"
NS = (8, 12, 16)
PS = (8192, 16384, 32768)
KEYS = ("A", "B", "Bp", "C", "Cp")


def count_cell(stream: bytes, body_start: int, n_scored: int, rel: bytes, rnd: bytes, n: int) -> dict[str, int]:
    """Counts of scored bytes in A, B (related prefix), Bp (random prefix), C = not A or B, Cp = not A or Bp."""
    r, q = bins(stream, body_start, n_scored, rel, n), bins(stream, body_start, n_scored, rnd, n)
    return {"A": r.count(A), "B": r.count(B), "Bp": q.count(B), "C": r.count(C), "Cp": q.count(C), "total": len(r)}


def count_target(stream: bytes, body_start: int, n_scored: int, pairs: dict, ns) -> dict[str, dict[str, dict[str, int]]]:
    return {str(n): {str(p): count_cell(stream, body_start, n_scored, rel, rnd, n) for p, (rel, rnd) in pairs.items()} for n in ns}


def pool(recs: list[dict], key: str, ns, ps) -> dict:
    """Sum the counts of every record whose `key` is not None; shares are byte-weighted (counts / total)."""
    used = [r for r in recs if r[key] is not None]
    cells = {str(n): {str(p): dict.fromkeys((*KEYS, "total"), 0) for p in ps} for n in ns}
    for r in used:
        for n, row in cells.items():
            for p, c in row.items():
                for k in c:
                    c[k] += r[key][n][p][k]
    for row in cells.values():
        for c in row.values():
            c["share"] = {k: (c[k] / c["total"] if c["total"] else None) for k in KEYS}
    return {"n_targets": len(used), "n_scored_bytes": sum(r["n_scored"] for r in used), "cells": cells}


def run_census(path, split: str, ns=NS, ps=PS, seed: int = 0, never_trained: bool = False, n_scored: int = 4096,
               train_end: int = TRAIN_END, tail_start: int = TAIL_START, dev_files: int = DEV_FILES, limit: int = 0) -> dict:
    table = file_table(path)
    hashes = hash_table(path, table)
    idxs = splits(table, tail_start, dev_files)[split]
    if not idxs:
        raise ValueError(f"split {split!r} holds no files")
    lo, hi = split_range(table, idxs)  # the split's own byte range, for the never-trained pass
    recs = []
    for i in targets(table, idxs, hashes, 4, train_end)[: limit or None]:
        stream, body_start, scored = read_stream(path, table[i], n_scored)
        if scored == 0:
            continue
        pairs = prefix_pairs(path, table, i, hashes, ps, seed, 0, train_end, train_end)
        rec = {"index": i, "path": table[i].path, "n_scored": scored, "related_files": len(related(i, table, 0, train_end, hashes)),
               "cells": count_target(stream, body_start, scored, pairs, ns), "never_files": None, "never_cells": None}
        k = len(related(i, table, lo, hi, hashes)) if never_trained else 0
        if k >= MIN_NEVER:
            npairs = prefix_pairs(path, table, i, hashes, ps, seed, lo, hi, train_end)
            rec["never_files"], rec["never_cells"] = k, count_target(stream, body_start, scored, npairs, ns)
        recs.append(rec)
    trained = pool(recs, "cells", ns, ps)
    never = None
    if never_trained:
        sub = [r for r in recs if r["never_cells"] is not None]
        never = {**pool(recs, "never_cells", ns, ps), "trained_cells": pool(sub, "cells", ns, ps)["cells"]}
    return {"split": split, "ns": list(ns), "ps": list(ps), "n_targets": trained["n_targets"], "n_scored_bytes": trained["n_scored_bytes"],
            "trained": trained, "never_trained": never, "targets": recs}


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render(res: dict) -> str:
    """Markdown tables of the pooled shares: one table per n, one row per P."""
    lines = [f"{res.get('run_name', '')} {res['split']}: {res['n_targets']} targets, {res['n_scored_bytes']} scored bytes"]
    blocks = [("related prefix = same-directory training files", res["trained"])]
    nt = res["never_trained"]
    if nt:
        blocks += [("same targets, trained related files", {**nt, "cells": nt["trained_cells"]}), ("never trained: related files from the target's own split", nt)]
    for title, blk in blocks:
        if not blk["n_targets"]:
            continue
        lines.append(f"\n{title} ({blk['n_targets']} targets, {blk['n_scored_bytes']} bytes)")
        for n, row in blk["cells"].items():
            lines += [f"\nn = {n}", "| P | A | B | B' | C | C' |", "|---:|---:|---:|---:|---:|---:|"]
            lines += [f"| {p} | " + " | ".join(_f(c["share"][k]) for k in KEYS) + " |" for p, c in row.items()]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--corpus", default=str(ROOT / "corpora" / "linux_full.bytes"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    ap.add_argument("--ps", default=",".join(map(str, PS)))
    ap.add_argument("--n-scored", type=int, default=4096)
    ap.add_argument("--never-trained", action="store_true", help="second pass: related prefix from the target's own split")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N targets only")
    ap.add_argument("--train-end", type=int, default=TRAIN_END, help="tiny-corpus tests only")
    ap.add_argument("--tail-start", type=int, default=TAIL_START, help="tiny-corpus tests only")
    ap.add_argument("--dev-files", type=int, default=DEV_FILES, help="tiny-corpus tests only")
    a = ap.parse_args(argv)
    out = RESULTS / f"{a.run_name}.json"
    if out.exists():
        sys.exit(f"{out} exists: new run names only")
    res = run_census(a.corpus, a.split, [int(x) for x in a.ns.split(",")], [int(x) for x in a.ps.split(",")], a.seed, a.never_trained,
                     a.n_scored, a.train_end, a.tail_start, a.dev_files, a.limit)
    res = {"run_name": a.run_name, "args": vars(a), **res}
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(render(res))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_census_script.py -q`

Expected: PASS (7 passed).

- [ ] **Step 5: Census the first 10 real dev targets (read-only, about 3 s)**

Run:

```bash
uv run python - <<'EOF'
from scripts.census import render, run_census
res = run_census("corpora/linux_full.bytes", "dev", limit=10)
print(render({"run_name": "smoke", **res}).split("\n\nn = 12")[0])
EOF
```

Expected output:

```text
smoke dev: 10 targets, 38243 scored bytes

related prefix = same-directory training files (10 targets, 38243 bytes)

n = 8
| P | A | B | B' | C | C' |
|---:|---:|---:|---:|---:|---:|
| 8192 | 0.375 | 0.099 | 0.024 | 0.526 | 0.601 |
| 16384 | 0.375 | 0.143 | 0.034 | 0.482 | 0.591 |
| 32768 | 0.375 | 0.177 | 0.051 | 0.448 | 0.574 |
```

The numbers are deterministic (seed 0). If A is far from 0.37 the header is not being counted as context, or the scored region is off by the header length.

- [ ] **Step 6: Commit**

```bash
git add scripts/census.py tests/test_census_script.py
git commit -m "Census script: shares of A, B, B' and C per n and P, with a never-trained pass (Stage 1.2)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Context probe: scoring primitives

`scripts/probe_commons.py` starts here: the module header (all imports and the constants of the context probe) and the functions that score one sequence: head chunk override, per-arm settings, token conversion, per-byte bits and the offset arithmetic that picks the scored bytes out of prefix + stream. The one-byte-context model in the first test is what pins that arithmetic.

**Files:**
- Create: `scripts/probe_commons.py`
- Create: `tests/test_probe_context.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_probe_context.py`

**Interfaces:**
- Consumes: `scripts.cross_eval.rebuild(ck)` (imported now, used in Task 7); `GRUHybrid.all_heads()`, `TransformerBackbone.set_rope_scale(s)` from `flybrain/model/hybrid.py`.
- Produces: `apply_chunk(model, chunk)`; `arm_settings(arm, ps, n_scored) -> (ps, n_scored)`; `to_tokens(data: bytes, device) -> LongTensor[1, T]`; `bits_of(logits, x) -> ndarray[T]` (entry 0 is NaN); `byte_bits(model, data, device, rope_train_len=0) -> ndarray[T]`; `score_with_prefix(model, prefix, stream, body_start, n_scored, device, rope_train_len=0) -> ndarray[n_scored]`; constants `BIN_N = 8`, `FIRST = 2048`, `GATE_P = 32_768`, `GATE_THRESHOLD = 0.02`, `N_BOOT = 1000`, `SCOPES`, `PARTS`, `CONDS`, `PART_IX`, `COND_IX`, `FILL`; fixture `tiny_hybrid(chunk=8)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def tiny_hybrid():
    """tiny_hybrid(chunk=8) -> a random CPU GRUHybrid with two sparse-key heads (post + one inner), no local attention."""
    def make(chunk: int = 8):
        import torch
        from flybrain.model.hybrid import GRUHybrid
        torch.manual_seed(0)
        return GRUHybrid(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4,
                         key_conv=2, normalize=True, gate=True, chunk=chunk, mem_every=2).eval()
    return make
```

Create `tests/test_probe_context.py`:

```python
"""scripts/probe_commons.py --mode context: scoring primitives, the pooled numbers and the end-to-end pipeline on a tiny random model."""

from __future__ import annotations

import math

import numpy as np
import torch


class Bigram(torch.nn.Module):
    """Logits depend only on the previous byte, so the loss of a byte is the same under any prefix."""

    def __init__(self):
        super().__init__()
        self.table = torch.randn(256, 256, generator=torch.Generator().manual_seed(0))

    def forward(self, x):
        return self.table[x], None


def test_scored_bits_are_the_loss_of_the_right_bytes_under_any_prefix():
    from scripts.probe_commons import score_with_prefix
    m = Bigram()
    head = b"\n// ==== a/x.c ====\n"
    stream = head + b"hello world\nint x;\n" * 4
    alone = score_with_prefix(m, b"", stream, len(head), 40, "cpu")
    for prefix in (b"some other material\n" * 3, b"\n// ==== b/y.c ====\nzzz", b"q"):
        assert np.array_equal(score_with_prefix(m, prefix, stream, len(head), 40, "cpu"), alone)   # a shifted window would score other bytes
    want = [float(torch.nn.functional.cross_entropy(m.table[stream[t - 1]][None], torch.tensor([stream[t]]))) / math.log(2) for t in range(len(head), len(head) + 40)]
    assert len(alone) == 40 and np.allclose(alone, want, atol=1e-6)                               # bits of byte t given byte t - 1


def test_rope_scale_follows_each_sequence_and_does_not_leak():
    from flybrain.model.hybrid import GRUHybrid
    from scripts.probe_commons import byte_bits
    torch.manual_seed(0)
    m = GRUHybrid(d_model=32, hidden=32, key="sparse", where="none", backbone="transformer", n_layers=2, d_ff=64, rope=True, max_len=64).eval()
    short, long = bytes(range(40)), bytes(range(200))
    a = byte_bits(m, short, "cpu", rope_train_len=64)                       # shorter than the training length: scale 1
    base_short = m.core.attn[0].rope_base
    b = byte_bits(m, long, "cpu", rope_train_len=64)                        # 200 / 64 x the training length
    base_long = m.core.attn[0].rope_base
    c = byte_bits(m, short, "cpu", rope_train_len=64)
    assert base_short == 10000.0 and abs(base_long - 10000.0 * (200 / 64) ** (32 / 30)) < 1e-3 and m.core.attn[0].rope_base == 10000.0
    assert np.array_equal(a[1:], c[1:]) and np.isnan(a[0]) and len(b) == 200


def test_apply_chunk_sets_every_head_and_leaves_headless_models_alone(tiny_hybrid):
    from flybrain.model.hybrid import GRUHybrid
    from scripts.probe_commons import apply_chunk, arm_settings
    m = tiny_hybrid()
    assert [h.chunk for h in m.all_heads()] == [8, 8]
    apply_chunk(m, 0); assert [h.chunk for h in m.all_heads()] == [8, 8]    # 0 keeps the checkpoint's own chunk
    apply_chunk(m, 4096); assert [h.chunk for h in m.all_heads()] == [4096, 4096]
    apply_chunk(GRUHybrid(d_model=16, hidden=16, key="sparse", where="none", backbone="transformer", n_layers=2, d_ff=32, rope=True), 4096)
    assert arm_settings("transformer", [8192, 16384], 4096) == ([2048], 2048)
    assert arm_settings("mingru_sparse_post", [8192, 16384], 4096) == ([8192, 16384], 4096)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_probe_context.py -q`

Expected: FAIL with "ModuleNotFoundError: No module named 'scripts.probe_commons'" (3 failed).

- [ ] **Step 3: Write minimal implementation**

Create `scripts/probe_commons.py`:

```python
#!/usr/bin/env python3
"""Do related files have anything to share? Scores the held-out targets of the file-ordered Linux corpus.

--mode context: plain forward, batch 1, one sequence per condition per target, scoring the target's first bytes after
  alone (header + target), related (its same-directory training files up to P bytes, then the target), random (an equal-size
  draw from other directories) and never-trained (related files from the target's own split). The alone per-byte loss is binned by
  the census (flybrain.census.bins, n = 8) to give the headroom H(X) = share(X) * (loss_alone(X) - loss_alone(A)) for X = B and B'.
--mode handoff: reader agents (fresh state, stream API) read the related files; their pooled tables are preloaded into a fresh agent that
  scores the target. Needs flybrain.group and GRUHybrid.stream (the agent-state plan); the context mode does not."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.census import bins
from flybrain.data.files import DEV_FILES, MIN_NEVER, TAIL_START, TRAIN_END, file_table, hash_table, read_stream, related, split_range, splits, targets
from flybrain.data.prefixes import prefix_pairs, reader_chunks
from scripts.cross_eval import rebuild

RESULTS = ROOT / "results"
BIN_N = 8          # the census n of the headroom numbers: what copyable_mask has always used
FIRST = 2048       # every model also reports its first FIRST scored bytes, so rows with shorter scoring compare
GATE_P = 32_768    # the gate number is read at this prefix size
GATE_THRESHOLD = 0.02  # Gate 1: H(B) - H(B') at or above this is Stage 2
N_BOOT = 1000      # resamples of the targets for every 95 % interval
SCOPES = ("full", "first2048")
PARTS = ("related", "random", "never")             # which prefix defined the census bins
CONDS = ("alone", "related", "random", "never")    # which prefix the model read
PART_IX = {p: k for k, p in enumerate(PARTS)}
COND_IX = {c: k for k, c in enumerate(CONDS)}
FILL = {"related": ("alone", "related", "random"), "random": ("alone", "related", "random"), "never": ("alone", "never")}


def apply_chunk(model, chunk: int) -> None:
    """Set the chunk length of every memory head (0 keeps the checkpoint's own): long sequences need the chunked form."""
    if chunk > 0:
        for h in getattr(model, "all_heads", lambda: [])():
            h.chunk = int(chunk)


def arm_settings(arm: str, ps, n_scored: int) -> tuple[list[int], int]:
    """The transformer reference reads 2,048 prefix bytes and scores the first 2,048 target bytes; every other arm takes the flags."""
    return ([2048], 2048) if arm == "transformer" else (list(ps), n_scored)


def to_tokens(data: bytes, device) -> Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device).long()[None]


def bits_of(logits: Tensor, x: Tensor) -> np.ndarray:
    """Bits of every byte of x[0] given the bytes before it, from logits [1, T, 256]. Entry 0 has no context (NaN)."""
    ce = F.cross_entropy(logits[0, :-1].float(), x[0, 1:], reduction="none") / math.log(2)
    return np.concatenate([[np.nan], ce.cpu().numpy()])


@torch.no_grad()
def byte_bits(model, data: bytes, device, rope_train_len: int = 0) -> np.ndarray:
    """One plain forward over `data`. rope_train_len > 0 (the rotary transformer arm) NTK-scales the base to this sequence's length
    (1.0 when it is not longer than the training length), set afresh for every sequence so nothing leaks from the previous one."""
    if rope_train_len:
        model.core.set_rope_scale(max(1.0, len(data) / rope_train_len))
    x = to_tokens(data, device)
    logits, _ = model(x)
    return bits_of(logits, x)


def score_with_prefix(model, prefix: bytes, stream: bytes, body_start: int, n_scored: int, device, rope_train_len: int = 0) -> np.ndarray:
    """Bits of the scored bytes stream[body_start : body_start + n_scored] when `stream` is read after `prefix` (b"" = alone)."""
    bits = byte_bits(model, prefix + stream, device, rope_train_len)
    a = len(prefix) + body_start
    return bits[a : a + n_scored]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_probe_context.py -q`

Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_commons.py tests/conftest.py tests/test_probe_context.py
git commit -m "Probe: score one sequence after a prefix, with per-sequence NTK scaling and the head chunk override (Stage 1.3)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Context probe: pooled cube, gains with intervals, headroom and gate number

One target adds a `[scope, P, partition, condition, code, (count, bits)]` cube and everything reported is a sum of cubes: bits per byte per condition and per census bin, the gains over alone with their split over the three bins, the headroom H(B) and H(B'), the gate number, and for each of those a 95 % interval from resampling the targets (files) with replacement, as the spec's "Comparisons are paired" section requires of the gate number and every reported gain. The tests build cubes by hand so the arithmetic is checked against numbers worked out on paper.

**Files:**
- Modify: `scripts/probe_commons.py`
- Modify: `tests/test_probe_context.py`
- Test: `tests/test_probe_context.py`

**Interfaces:**
- Consumes: Task 5: constants `SCOPES`, `PARTS`, `CONDS`, `PART_IX`, `COND_IX`, `FILL`, `FIRST`, `N_BOOT`.
- Produces: `new_cube(n_p) -> ndarray[2, n_p, 3, 4, 3, 2]`; `fill(cube, pi, part, codes, bits_by_cond)`; `headroom(cs_p) -> dict` (keys `share_B`, `loss_B`, `H_B`, `loss_A`, `share_Bp`, `loss_Bp`, `H_Bp`, `share_Bn`, `loss_Bn`, `H_Bn`, `gate`; `None` when undefined); `ci95(values) -> [lo, hi] | None`; `effects_of(cs_p) -> dict` (`gain_related`, `gain_random`, `gain_never`, `gain_related_vs_random`, `bin_<partition>_<A|B|C>`, `H_B`, `H_Bp`, `H_Bn`, `gate`); `bootstrap_effects(cubes, ps, n_boot=N_BOOT, seed=0) -> {scope: {P: {name: [lo, hi] | None}}}`; `summarize(cubes, ps, with_never, n_boot=N_BOOT, seed=0) -> dict`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_probe_context.py`:

```python
def test_the_first_2048_scope_keeps_only_the_first_scored_bytes():
    from scripts.probe_commons import COND_IX, PART_IX, fill, new_cube
    cube = new_cube(1)
    bits = {c: np.ones(3000) for c in ("alone", "related", "random")}
    fill(cube, 0, "related", np.zeros(3000, dtype=int), bits)
    a, r = COND_IX["alone"], PART_IX["related"]
    assert cube[0, 0, r, a, 0, 0] == 3000 and cube[1, 0, r, a, 0, 0] == 2048 and cube[1, 0, r, a, 0, 1] == 2048.0
    short = new_cube(1)
    fill(short, 0, "related", np.zeros(90, dtype=int), {c: np.ones(90) for c in ("alone", "related", "random")})
    assert short[0, 0, r, a, 0, 0] == short[1, 0, r, a, 0, 0] == 90            # a short body: the two scopes coincide


def test_headroom_is_share_times_the_loss_gap_and_the_gate_is_their_difference():
    from scripts.probe_commons import fill, headroom, new_cube
    alone = np.array([1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0])
    bits = {"alone": alone, "related": alone - 0.5, "random": alone}
    cube = new_cube(1)
    fill(cube, 0, "related", [0, 0, 1, 1, 1, 1, 2, 2, 2, 2], bits)           # A A B B B B C C C C
    fill(cube, 0, "random", [0, 0, 1, 2, 2, 2, 2, 2, 2, 2], bits)            # A A B' C' C' ...
    h = headroom(cube[0, 0])
    assert (h["share_B"], h["share_Bp"]) == (0.4, 0.1) and (h["loss_A"], h["loss_B"], h["loss_Bp"]) == (1.0, 5.0, 5.0)
    assert abs(h["H_B"] - 0.4 * 4.0) < 1e-12 and abs(h["H_Bp"] - 0.1 * 4.0) < 1e-12 and abs(h["gate"] - 1.2) < 1e-12
    assert h["share_Bn"] is None and h["H_Bn"] is None                       # nobody filled the never-trained partition
    no_a = new_cube(1)
    fill(no_a, 0, "related", [1, 2], {"alone": np.ones(2), "related": np.ones(2), "random": np.ones(2)})
    fill(no_a, 0, "random", [1, 2], {"alone": np.ones(2), "related": np.ones(2), "random": np.ones(2)})
    assert headroom(no_a[0, 0])["gate"] is None                              # no self-copyable byte: H is undefined, not zero
    no_b = new_cube(1)
    for part in ("related", "random"):
        fill(no_b, 0, part, [0, 2, 2], {"alone": np.array([1.0, 6.0, 6.0]), "related": np.ones(3), "random": np.ones(3)})
    assert headroom(no_b[0, 0])["H_B"] == 0.0 and headroom(no_b[0, 0])["gate"] == 0.0   # no B byte: nothing to gain, which is zero, not unknown


def test_each_gain_splits_over_the_three_bins_and_the_related_margin_is_a_difference():
    from scripts.probe_commons import effects_of, fill, new_cube
    alone = np.array([1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0])
    cube = new_cube(1)
    related = alone - np.array([0, 0, 2, 2, 2, 2, 0.5, 0.5, 0.5, 0.5])
    fill(cube, 0, "related", [0, 0, 1, 1, 1, 1, 2, 2, 2, 2], {"alone": alone, "related": related, "random": alone - 0.1})
    fill(cube, 0, "random", [0, 0, 1, 2, 2, 2, 2, 2, 2, 2], {"alone": alone, "related": alone - 0.3, "random": alone - 0.1})
    e = effects_of(cube[0, 0])
    assert abs(e["bin_related_A"]) < 1e-12 and abs(e["bin_related_B"] - 0.8) < 1e-12 and abs(e["bin_related_C"] - 0.2) < 1e-12   # 4 of 10 bytes gain 2.0, 4 of 10 gain 0.5
    assert abs(e["gain_related"] - 1.0) < 1e-12 and abs(e["gain_random"] - 0.1) < 1e-12 and abs(e["gain_related_vs_random"] - 0.9) < 1e-12
    assert abs(sum(e[f"bin_random_{c}"] for c in "ABC") - e["gain_random"]) < 1e-12
    assert e["gain_never"] is None and e["bin_never_B"] is None and e["H_Bn"] is None          # nothing was read from never-trained files


def test_bootstrap_effects_are_seeded_collapse_for_identical_targets_and_move_for_different_ones():
    from scripts.probe_commons import bootstrap_effects, ci95, effects_of, fill, new_cube
    alone = np.array([1.0, 1.0, 5.0, 5.0, 7.0, 7.0])
    one = new_cube(1)
    fill(one, 0, "related", [0, 0, 1, 1, 2, 2], {"alone": alone, "related": alone - 0.5, "random": alone})
    fill(one, 0, "random", [0, 0, 1, 2, 2, 2], {"alone": alone, "related": alone, "random": alone})
    same = np.stack([one] * 7)
    ci, want = bootstrap_effects(same, [8], 50, 3), effects_of(same.sum(0)[0, 0])
    assert abs(ci["full"]["8"]["gate"][0] - want["gate"]) < 1e-12 and abs(ci["full"]["8"]["gain_related"][1] - want["gain_related"]) < 1e-12
    assert ci["full"]["8"]["gain_never"] is None and set(ci) == {"full", "first2048"}
    other = new_cube(1)
    fill(other, 0, "related", [0, 0, 1, 1, 1, 2], {"alone": alone, "related": alone - 1.0, "random": alone})
    fill(other, 0, "random", [0, 0, 2, 2, 2, 2], {"alone": alone, "related": alone, "random": alone})
    mixed = np.stack([one, other, one, other, other])
    a = bootstrap_effects(mixed, [8], 100, 3)
    assert a == bootstrap_effects(mixed, [8], 100, 3)
    lo, hi = a["full"]["8"]["gain_related"]
    assert lo < hi
    assert ci95([1.0, 2.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]) is not None and ci95([1.0, None, None]) is None and ci95([]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_probe_context.py -q`

Expected: FAIL with "ImportError: cannot import name 'fill' from 'scripts.probe_commons'" (4 failed, 3 passed).

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/probe_commons.py`:

```python
def new_cube(n_p: int) -> np.ndarray:
    """[scope, P, partition, condition, code, (count, summed bits)]: what one target adds to every pooled number."""
    return np.zeros((len(SCOPES), n_p, len(PARTS), len(CONDS), 3, 2))


def fill(cube: np.ndarray, pi: int, part: str, codes, bits_by_cond: dict[str, np.ndarray]) -> None:
    """Add one partition's census codes and the per-byte bits of its conditions, for the full scored region and its first FIRST bytes."""
    codes = np.asarray(codes)
    for s, cap in enumerate((len(codes), min(FIRST, len(codes)))):
        for cond in FILL[part]:
            for k in range(3):
                m = codes[:cap] == k
                cube[s, pi, PART_IX[part], COND_IX[cond], k] = (m.sum(), bits_by_cond[cond][:cap][m].sum())


def _bpb(cell: np.ndarray) -> float | None:
    """Bits per byte of a [code, (count, bits)] cell pooled over its codes; None when it holds no bytes."""
    n = cell[:, 0].sum()
    return float(cell[:, 1].sum() / n) if n else None


def _by_code(cell: np.ndarray, field: int) -> dict[str, float | None]:
    """count (field 0) or bits per byte (field 1) of each code A, B, C."""
    if field == 0:
        return {c: int(cell[k, 0]) for k, c in enumerate("ABC")}
    return {c: (float(cell[k, 1] / cell[k, 0]) if cell[k, 0] else None) for k, c in enumerate("ABC")}


def headroom(cs_p: np.ndarray) -> dict:
    """Headroom numbers of one P from a pooled cube slice [partition, condition, code, 2], in bits/byte of the alone condition:
    H(X) = share(X) * (loss(X) - loss(A)), None when A or the partition holds no bytes; gate = H(B) - H(B')."""
    out = {}
    for part, tag in (("related", "B"), ("random", "Bp"), ("never", "Bn")):
        cell = cs_p[PART_IX[part], COND_IX["alone"]]
        n, s = cell[:, 0], cell[:, 1]
        total = n.sum()
        loss = [float(s[k] / n[k]) if n[k] else None for k in range(3)]
        out[f"share_{tag}"] = float(n[1] / total) if total else None
        out[f"loss_{tag}"] = loss[1]
        out[f"H_{tag}"] = None if not n[0] or not total else (float(n[1] / total * (loss[1] - loss[0])) if n[1] else 0.0)
        if part == "related":
            out["loss_A"] = loss[0]
    out["gate"] = None if out["H_B"] is None or out["H_Bp"] is None else out["H_B"] - out["H_Bp"]
    return out


def ci95(values: list) -> list[float] | None:
    """2.5 % and 97.5 % percentiles of the resampled values; None when more than a tenth of the resamples had no value."""
    got = [v for v in values if v is not None]
    return [float(np.percentile(got, 2.5)), float(np.percentile(got, 97.5))] if got and len(got) >= 0.9 * len(values) else None


def effects_of(cs_p: np.ndarray) -> dict[str, float | None]:
    """Every gain and headroom number of one (scope, P) slice [partition, condition, code, 2] pooled over targets, in bits/byte.
    A gain is alone minus the condition on the same bytes (positive = fewer bits); gain_<partition> is read in the bins of that
    partition and bin_<partition>_<code> is the share-weighted part of it that falls on the code, so the three add up to the gain."""
    h = headroom(cs_p)
    out = {k: h[k] for k in ("H_B", "H_Bp", "H_Bn", "gate")}
    for part in PARTS:
        alone, other = cs_p[PART_IX[part], COND_IX["alone"]], cs_p[PART_IX[part], COND_IX[part]]
        n = alone[:, 0].sum()
        share_gain = (alone[:, 1] - other[:, 1]) / n if n else None
        out[f"gain_{part}"] = None if share_gain is None else float(share_gain.sum())
        for k, c in enumerate("ABC"):
            out[f"bin_{part}_{c}"] = None if share_gain is None else float(share_gain[k])
    out["gain_related_vs_random"] = None if out["gain_related"] is None or out["gain_random"] is None else out["gain_related"] - out["gain_random"]
    return out


def bootstrap_effects(cubes: np.ndarray, ps, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """95 % intervals of effects_of over resamples of the targets (files) with replacement: {scope: {P: {name: [lo, hi] | None}}}."""
    rng = np.random.default_rng(seed)
    t = len(cubes)
    runs = {(s, pi): [] for s in range(len(SCOPES)) for pi in range(len(ps))}
    for _ in range(n_boot):
        pooled = cubes[rng.integers(0, t, t)].sum(0)
        for (s, pi), acc in runs.items():
            acc.append(effects_of(pooled[s, pi]))
    return {scope: {str(ps[pi]): {k: ci95([e[k] for e in acc]) for k in acc[0]} for (s2, pi), acc in runs.items() if s2 == s} for s, scope in enumerate(SCOPES)}


def summarize(cubes: np.ndarray, ps, with_never: bool, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """The nested result for stacked per-target cubes [target, scope, P, partition, condition, code, 2]: pooled numbers, the gains and
    headroom of effects_of, and their bootstrap intervals over targets."""
    pooled = cubes.sum(0)
    out = {"n_scored_bytes": {}, "bpb": {}, "bins": {}, "headroom": {}, "gains": {}, "ci95": bootstrap_effects(cubes, ps, n_boot, seed)}
    conds = ("related", "random", "never") if with_never else ("related", "random")
    for s, scope in enumerate(SCOPES):
        cs = pooled[s]
        out["n_scored_bytes"][scope] = int(cs[0, PART_IX["related"], COND_IX["alone"], :, 0].sum())
        out["bpb"][scope] = {"alone": _bpb(cs[0, PART_IX["related"], COND_IX["alone"]])}
        for cond in conds:
            out["bpb"][scope][cond] = {str(p): _bpb(cs[pi, PART_IX[cond], COND_IX[cond]]) for pi, p in enumerate(ps)}
        out["bins"][scope] = {str(p): {part: {"count": _by_code(cs[pi, PART_IX[part], COND_IX["alone"]], 0),
                                              "bpb": {c: _by_code(cs[pi, PART_IX[part], COND_IX[c]], 1) for c in FILL[part]}}
                                       for part in PARTS if part != "never" or with_never} for pi, p in enumerate(ps)}
        out["headroom"][scope] = {str(p): headroom(cs[pi]) for pi, p in enumerate(ps)}
        out["gains"][scope] = {str(p): effects_of(cs[pi]) for pi, p in enumerate(ps)}
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_probe_context.py -q`

Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_commons.py tests/test_probe_context.py
git commit -m "Probe: pooled per-target cube, per-bin bits, gains and headroom with bootstrap intervals over targets (Stage 1.2-1.3)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Context probe: driver, tables and command line

The driver scores every target under every condition, pools the cubes, renders the tables and runs from the command line. It is tested with a random CPU model, never a trained one; the last step runs one real checkpoint on CPU as a smoke test.

**Files:**
- Modify: `scripts/probe_commons.py`
- Modify: `tests/test_probe_context.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_probe_context.py`

**Interfaces:**
- Consumes: Task 1: `file_table`, `hash_table`, `splits`, `split_range`, `targets`, `read_stream`, `related`, `MIN_NEVER`; Task 2: `prefix_pairs`; Task 3: `bins`; Tasks 5-6: scoring primitives, `N_BOOT`, `new_cube`, `fill`, `summarize`; `scripts.cross_eval.rebuild`.
- Produces: `probe_target(model, device, path, table, hashes, i, ps, seed, n_scored, rope_train_len=0, never_range=None, train_end=TRAIN_END) -> (cube, record) | None`; `probe_context(model, device, path, split, ps, seed=0, n_scored=4096, rope_train_len=0, never=True, train_end=TRAIN_END, tail_start=TAIL_START, dev_files=DEV_FILES, limit=0) -> dict`; `render_context(res) -> str`; `load_checkpoint(path, device, chunk) -> (model, ck)`; `main(argv=None)` with `--mode context`; fixture `tiny_checkpoint(arm='mingru_sparse_post') -> Path`. Writes `results/<run-name>.json` with this schema (bits are bits/byte; `<scope>` is `full` or `first2048`; `<P>` and alpha-like keys are strings):

```text
{"run_name": str, "mode": "context", "ckpt": file name, "arm": str, "chunk": int, "args": {...},
 "split": str, "ps": [P, ...], "n_scored_cap": int, "rope_train_len": int, "n_targets": int, "n_never_targets": int,
 "gate_p": 32768, "gate_number": float | null, "gate_ci95": [lo, hi] | null,
 "all": BLOCK, "never_subset": BLOCK | null,
 "targets": [{"index": int, "path": str, "n_scored": int, "never": bool,
              "bpb": {"alone": float, "related": {"<P>": float}, "random": {"<P>": float}, "never": {"<P>": float} | null},
              "prefix_bytes": {"<P>": [related_len, never_len | null]}}]}

BLOCK = {"n_scored_bytes": {"full": int, "first2048": int},
         "bpb": {"<scope>": {"alone": float, "related": {"<P>": float}, "random": {"<P>": float}, ["never": {"<P>": float}]}},   # pooled over bytes; "never" only in never_subset
         "bins": {"<scope>": {"<P>": {"<partition>": {"count": {"A": int, "B": int, "C": int},
                                                      "bpb": {"<condition>": {"A": float|null, "B": float|null, "C": float|null}}}}}},
         "headroom": {"<scope>": {"<P>": {"share_B", "loss_B", "H_B", "loss_A", "share_Bp", "loss_Bp", "H_Bp", "share_Bn", "loss_Bn", "H_Bn", "gate": float | null}}},
         "gains": {"<scope>": {"<P>": {"<name>": float | null}}},       # alone minus the condition on the same bytes, positive = fewer bits
         "ci95": {"<scope>": {"<P>": {"<name>": [lo, hi] | null}}}}      # bootstrap over targets, 1,000 resamples, seeded
partition "related" and "random": conditions alone, related, random (in "random" the B code is B'); partition "never" (never_subset only): conditions alone, never.
<name> in gains and ci95: gain_related, gain_random, gain_never, gain_related_vs_random (= gain_related - gain_random), bin_related_A|B|C, bin_random_A|B|C, bin_never_A|B|C (the share-weighted part of
the gain of that partition that falls on the code; the three add up to it), H_B, H_Bp, H_Bn, gate. gate_ci95 = all.ci95.full.<gate_p>.gate.
```

- [ ] **Step 1: Write the failing test**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def tiny_checkpoint(tmp_path):
    """tiny_checkpoint(arm) -> path of a checkpoint dict (arm, args, state) that scripts.cross_eval.rebuild loads."""
    def make(arm: str = "mingru_sparse_post"):
        import torch
        from flybrain.model.hybrid import GRUHybrid
        torch.manual_seed(0)
        args = {"d_model": 16, "gru_hidden": 330, "n_key": 64, "n_layers": 4, "d_ff": 32, "top_k": 4, "claws": 4, "theta": 1.0,
                "learned_key_dim": 64, "normalize": True, "key_conv": 2, "decay": 1.0, "seed": 0, "gate": True, "seq_len": 64,
                "chunk": 8, "mem_every": 2, "rope": arm == "transformer"}
        if arm == "transformer":
            m = GRUHybrid(d_model=16, hidden=330, key="sparse", where="none", backbone="transformer", n_layers=4, d_ff=32, key_conv=2, max_len=1024, rope=True)
        else:
            m = GRUHybrid(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4,
                          key_conv=2, normalize=True, gate=True, chunk=8, mem_every=2)
        path = tmp_path / f"tiny_{arm}.pt"
        torch.save({"arm": arm, "args": args, "state": m.state_dict()}, path)
        return path
    return make
```

Append to `tests/test_probe_context.py`:

```python
def _probe(world, tiny_hybrid, ps=(0, 128), seed=0, never=True, split="dev"):
    from scripts.probe_commons import probe_context
    return probe_context(tiny_hybrid(), "cpu", world.path, split, ps, seed, 300, 0, never, world.train_end, world.tail_start, world.dev_files)


def test_an_empty_prefix_scores_exactly_alone_and_short_bodies_are_scored_whole(world, tiny_hybrid):
    res = _probe(world, tiny_hybrid)
    full = res["all"]["bpb"]["full"]
    assert full["related"]["0"] == full["alone"] == full["random"]["0"]      # P = 0 is the alone sequence: identical, not close
    assert full["related"]["128"] != full["alone"] and full["random"]["128"] != full["alone"]
    assert [r["n_scored"] for r in res["targets"]] == [300, 300, 90]
    assert res["all"]["n_scored_bytes"] == {"full": 690, "first2048": 690}
    assert res["n_targets"] == 3 and res["n_never_targets"] == 3
    assert res["never_subset"]["bpb"]["full"]["never"]["0"] == full["alone"]
    assert res["gate_number"] is None and res["gate_ci95"] is None            # 32,768 is not among this run's P
    assert res["targets"][0]["prefix_bytes"]["128"] == [128, 128]


def test_context_probe_is_deterministic_for_a_fixed_seed_and_json_serialisable(world, tiny_hybrid):
    import json
    a, b = _probe(world, tiny_hybrid, seed=0), _probe(world, tiny_hybrid, seed=0)
    assert json.dumps(a) == json.dumps(b)
    c = _probe(world, tiny_hybrid, seed=1)
    assert a["all"]["bpb"]["full"]["related"] == c["all"]["bpb"]["full"]["related"]          # the related prefix does not depend on the seed
    assert a["all"]["bpb"]["full"]["random"] != c["all"]["bpb"]["full"]["random"]            # the random prefix does


def test_the_gate_number_is_read_at_p_32768_and_never_trained_is_optional(world, tiny_hybrid):
    res = _probe(world, tiny_hybrid, ps=(128, 32_768), never=False)
    h = res["all"]["headroom"]["full"]["32768"]
    assert res["gate_number"] == h["gate"] == h["H_B"] - h["H_Bp"] == res["all"]["gains"]["full"]["32768"]["gate"]
    assert len(res["gate_ci95"]) == 2 and res["gate_ci95"] == res["all"]["ci95"]["full"]["32768"]["gate"]
    g = res["all"]["gains"]["full"]["128"]                                                      # gains carry intervals too, on the same targets
    assert abs(g["gain_related"] - (res["all"]["bpb"]["full"]["alone"] - res["all"]["bpb"]["full"]["related"]["128"])) < 1e-9
    assert abs(sum(g[f"bin_related_{c}"] for c in "ABC") - g["gain_related"]) < 1e-9 and len(res["all"]["ci95"]["full"]["128"]["gain_related"]) == 2
    assert res["never_subset"] is None and res["n_never_targets"] == 0 and res["all"]["bpb"]["full"].get("never") is None


def test_render_context_prints_the_tables_and_the_gate_number(world, tiny_hybrid):
    from scripts.probe_commons import render_context
    res = _probe(world, tiny_hybrid, ps=(128, 32_768))
    text = render_context({"run_name": "tiny", **res})
    assert "| P | related | random | never trained |" in text and "| P | scope | share B | share B' | H(B) | H(B') | H(B) - H(B') |" in text
    assert "| P | gain related | gain random | related minus random | gain never trained |" in text and "gain, share-weighted |" in text
    assert "gate number H(B) - H(B') at n = 8, P = 32768" in text and "census bins from the never prefix" in text and "95 % interval over targets" in text


def test_main_scores_a_tiny_checkpoint_and_the_transformer_arm_is_pinned_to_2048(world, tiny_checkpoint, tmp_path, monkeypatch):
    import json

    import pytest
    import scripts.probe_commons as pc
    monkeypatch.setattr(pc, "RESULTS", tmp_path)
    common = ["--mode", "context", "--split", "dev", "--corpus", str(world.path), "--device", "cpu", "--ps", "128,256", "--n-scored", "300", "--chunk", "0",
              "--train-end", str(world.train_end), "--tail-start", str(world.tail_start), "--dev-files", "3"]
    pc.main(["--ckpt", str(tiny_checkpoint()), "--run-name", "ctx_tiny", *common])
    res = json.loads((tmp_path / "ctx_tiny.json").read_text())
    assert {"run_name", "mode", "ckpt", "arm", "chunk", "args", "split", "ps", "n_scored_cap", "rope_train_len", "n_targets", "n_never_targets",
            "gate_p", "gate_number", "gate_ci95", "all", "never_subset", "targets"} <= set(res)
    assert res["ps"] == [128, 256] and res["n_scored_cap"] == 300 and res["rope_train_len"] == 0 and res["arm"] == "mingru_sparse_post"
    pc.main(["--ckpt", str(tiny_checkpoint("transformer")), "--run-name", "ctx_tiny_tf", *common])
    tf = json.loads((tmp_path / "ctx_tiny_tf.json").read_text())
    assert tf["ps"] == [2048] and tf["n_scored_cap"] == 2048 and tf["rope_train_len"] == 64 and tf["arm"] == "transformer"
    bodies = [world.table[world.idx[p]].end - world.table[world.idx[p]].body for p in ("k/t0.c", "k/t1.c", "k/t2.c")]
    assert [t["n_scored"] for t in tf["targets"]] == bodies and max(bodies) > 300      # --n-scored 300 is ignored: the reference scores up to 2,048
    with pytest.raises(SystemExit):
        pc.main(["--ckpt", str(tiny_checkpoint()), "--run-name", "ctx_tiny", *common])


def _long_and_empty_world(corpus):
    """d/a0..a3 and e/r0, e/r1 are training files; d/big.c (2,600 bytes of repeated phrases) and an empty d/empty.c close the file."""
    import random
    from flybrain.data.files import file_table, hash_table
    rng = random.Random(1)
    noise = lambda n: bytes(rng.choice(b"abcdefghijklmnopqrstuvwxyz ;(){}\n") for _ in range(n))
    phrases = [noise(60) for _ in range(8)]
    text = lambda n: b"".join(rng.choice(phrases) for _ in range(n // 60 + 1))[:n]
    files = [(f"d/a{j}.c", text(700)) for j in range(4)] + [(f"e/r{j}.c", noise(700)) for j in range(2)] + [("d/big.c", text(2600)), ("d/empty.c", b"")]
    path = corpus(files)
    table = file_table(path)
    return path, table, hash_table(path, table)


def test_a_long_target_is_cut_at_the_cap_and_a_header_at_the_end_of_the_file_is_skipped(corpus, tiny_hybrid):
    from scripts.probe_commons import probe_context
    path, table, _ = _long_and_empty_world(corpus)
    cut = table[6].start                                                       # training ends where d/big.c starts; the tail is big.c and empty.c
    res = probe_context(tiny_hybrid(), "cpu", path, "dev", (128, 32_768), 0, 4096, 0, False, cut, cut, 2)
    assert [t["path"] for t in res["targets"]] == ["d/big.c"]                  # d/empty.c is the last header of the file: nothing to score
    assert res["all"]["n_scored_bytes"] == {"full": 2600, "first2048": 2048}
    full, first = res["all"]["headroom"]["full"]["32768"], res["all"]["headroom"]["first2048"]["32768"]
    assert full["gate"] is not None and res["gate_number"] == full["gate"] != first["gate"]   # the gate number reads the full scope
    assert res["gate_ci95"] == res["all"]["ci95"]["full"]["32768"]["gate"] != res["all"]["ci95"]["first2048"]["32768"]["gate"]   # and so does its interval
    capped = probe_context(tiny_hybrid(), "cpu", path, "dev", (128,), 0, 1000, 0, False, cut, cut, 2)
    assert capped["all"]["n_scored_bytes"] == {"full": 1000, "first2048": 1000} and capped["targets"][0]["n_scored"] == 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_probe_context.py -q`

Expected: FAIL with "ImportError: cannot import name 'probe_context' from 'scripts.probe_commons'" (6 failed, 7 passed).

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/probe_commons.py`:

```python
def probe_target(model, device, path, table, hashes, i: int, ps, seed: int, n_scored: int, rope_train_len: int = 0,
                 never_range: tuple[int, int] | None = None, train_end: int = TRAIN_END):
    """(cube, record) for target i, or None when its body is empty."""
    stream, body_start, scored = read_stream(path, table[i], n_scored)
    if scored == 0:
        return None
    score = lambda prefix: score_with_prefix(model, prefix, stream, body_start, scored, device, rope_train_len)
    alone = score(b"")
    pairs = prefix_pairs(path, table, i, hashes, ps, seed, 0, train_end, train_end)
    npairs = None
    if never_range is not None and len(related(i, table, *never_range, hashes)) >= MIN_NEVER:
        npairs = prefix_pairs(path, table, i, hashes, ps, seed, *never_range, train_end)
    cube = new_cube(len(ps))
    bpb = {"alone": float(alone.mean()), "related": {}, "random": {}, "never": {} if npairs else None}
    for pi, p in enumerate(ps):
        rel, rnd = pairs[p]
        b = {"alone": alone, "related": score(rel), "random": score(rnd)}
        fill(cube, pi, "related", bins(stream, body_start, scored, rel, BIN_N), b)
        fill(cube, pi, "random", bins(stream, body_start, scored, rnd, BIN_N), b)
        bpb["related"][str(p)], bpb["random"][str(p)] = float(b["related"].mean()), float(b["random"].mean())
        if npairs:
            nt = npairs[p][0]
            b["never"] = score(nt)
            fill(cube, pi, "never", bins(stream, body_start, scored, nt, BIN_N), b)
            bpb["never"][str(p)] = float(b["never"].mean())
    rec = {"index": i, "path": table[i].path, "n_scored": scored, "never": npairs is not None, "bpb": bpb,
           "prefix_bytes": {str(p): [len(pairs[p][0]), len(npairs[p][0]) if npairs else None] for p in ps}}
    return cube, rec


def probe_context(model, device, path, split: str, ps, seed: int = 0, n_scored: int = 4096, rope_train_len: int = 0, never: bool = True,
                  train_end: int = TRAIN_END, tail_start: int = TAIL_START, dev_files: int = DEV_FILES, limit: int = 0) -> dict:
    table = file_table(path)
    hashes = hash_table(path, table)
    idxs = splits(table, tail_start, dev_files)[split]
    if not idxs:
        raise ValueError(f"split {split!r} holds no files")
    never_range = split_range(table, idxs) if never else None
    cubes, recs = [], []
    for i in targets(table, idxs, hashes, 4, train_end)[: limit or None]:
        got = probe_target(model, device, path, table, hashes, i, ps, seed, n_scored, rope_train_len, never_range, train_end)
        if got is not None:
            cubes.append(got[0]); recs.append(got[1])
            if len(recs) % 25 == 0:
                print(f"  {len(recs)} targets", flush=True)
    if not cubes:
        raise ValueError("no target to score")
    cubes = np.stack(cubes)
    sub = np.array([r["never"] for r in recs])
    res = {"split": split, "ps": list(ps), "n_scored_cap": n_scored, "rope_train_len": rope_train_len, "n_targets": len(recs),
           "n_never_targets": int(sub.sum()), "gate_p": GATE_P, "gate_number": None, "gate_ci95": None,
           "all": summarize(cubes, ps, False, N_BOOT, seed), "never_subset": summarize(cubes[sub], ps, True, N_BOOT, seed) if sub.any() else None,
           "targets": recs}
    if GATE_P in ps:
        res["gate_number"] = res["all"]["gains"]["full"][str(GATE_P)]["gate"]
        res["gate_ci95"] = res["all"]["ci95"]["full"][str(GATE_P)]["gate"]
    return res


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _ci(v, ci, fmt: str = "+.3f") -> str:
    """A value with its 95 % interval, '+0.034 (+0.020, +0.048)'; n/a without a value."""
    return "n/a" if v is None else format(v, fmt) + ("" if ci is None else f" ({format(ci[0], fmt)}, {format(ci[1], fmt)})")


def _g(block: dict, scope: str, p, key: str) -> str:
    """A gain or headroom number of a context-probe block with its interval."""
    return _ci(block["gains"][scope][str(p)][key], block["ci95"][scope][str(p)][key])


def render_context(res: dict) -> str:
    """Markdown tables: bits/byte per condition, gains over alone with 95 % intervals, bits/byte and gain per census bin (n = 8),
    the headroom numbers and the gate number."""
    ps = res["ps"]
    lines = [f"{res.get('run_name', '')} {res['split']}: {res['n_targets']} targets ({res['n_never_targets']} with never-trained files), "
             f"{res['all']['n_scored_bytes']['full']} scored bytes, P = {ps}"]
    for block, title in (("all", "all targets"), ("never_subset", "never-trained subset")):
        b = res.get(block)
        if not b:
            continue
        for scope in SCOPES:
            lines += [f"\n{title}, {scope} ({b['n_scored_bytes'][scope]} bytes): bits/byte, alone = {_f(b['bpb'][scope]['alone'])}",
                      "| P | related | random | never trained |", "|---:|---:|---:|---:|"]
            lines += [f"| {p} | {_f(b['bpb'][scope]['related'][str(p)])} | {_f(b['bpb'][scope]['random'][str(p)])} | "
                      f"{_f(b['bpb'][scope].get('never', {}).get(str(p)))} |" for p in ps]
            lines += [f"\n{title}, {scope}: gains over alone in bits/byte (positive = fewer bits), 95 % interval over targets",
                      "| P | gain related | gain random | related minus random | gain never trained |", "|---:|---:|---:|---:|---:|"]
            lines += [f"| {p} | " + " | ".join(_g(b, scope, p, k) for k in ("gain_related", "gain_random", "gain_related_vs_random", "gain_never")) + " |" for p in ps]
        for p in ps:
            for part, tag in (("related", "B"), ("random", "B'")) + ((("never", "Bn"),) if block == "never_subset" else ()):
                cell = b["bins"]["full"][str(p)][part]
                lines += [f"\n{title}, P = {p}, census bins from the {part} prefix (full scope)",
                          "| bin | share | " + " | ".join(FILL[part]) + f" | {part} gain, share-weighted |", "|---|---:|" + "---:|" * (len(FILL[part]) + 1)]
                n = sum(cell["count"].values())
                for code, name in zip("ABC", ("A", tag, "C")):
                    lines.append(f"| {name} | {_f(cell['count'][code] / n if n else None)} | " + " | ".join(_f(cell["bpb"][c][code]) for c in FILL[part])
                                 + f" | {_g(b, 'full', p, f'bin_{part}_{code}')} |")
        lines += [f"\n{title}: headroom in bits/byte (alone-condition loss), 95 % interval over targets",
                  "| P | scope | share B | share B' | H(B) | H(B') | H(B) - H(B') |", "|---:|---|---:|---:|---:|---:|---:|"]
        for scope in SCOPES:
            for p in ps:
                h = b["headroom"][scope][str(p)]
                lines.append(f"| {p} | {scope} | {_f(h['share_B'])} | {_f(h['share_Bp'])} | " + " | ".join(_g(b, scope, p, k) for k in ("H_B", "H_Bp", "gate")) + " |")
    ci = res["gate_ci95"]
    straddle = " The interval straddles the 0.02 threshold." if ci is not None and ci[0] < GATE_THRESHOLD < ci[1] else ""
    lines.append(f"\ngate number H(B) - H(B') at n = {BIN_N}, P = {res['gate_p']}: {_f(res['gate_number'])}, 95 % interval over targets "
                 f"{None if ci is None else [round(c, 4) for c in ci]}.{straddle}")
    return "\n".join(lines)


def load_checkpoint(path: str, device, chunk: int):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = rebuild(ck).to(device).eval()
    apply_chunk(model, chunk)
    return model, ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("context",))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--corpus", default=str(ROOT / "corpora" / "linux_full.bytes"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ps", default="2048,4096", help="prefix sizes in bytes (the transformer arm always uses 2048)")
    ap.add_argument("--n-scored", type=int, default=4096, help="scored bytes per target (the transformer arm always scores 2048)")
    ap.add_argument("--chunk", type=int, default=4096, help="head chunk length for long sequences; 0 keeps the checkpoint's own")
    ap.add_argument("--no-never", action="store_true", help="skip the never-trained condition")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N targets only")
    ap.add_argument("--train-end", type=int, default=TRAIN_END, help="tiny-corpus tests only")
    ap.add_argument("--tail-start", type=int, default=TAIL_START, help="tiny-corpus tests only")
    ap.add_argument("--dev-files", type=int, default=DEV_FILES, help="tiny-corpus tests only")
    a = ap.parse_args(argv)
    out = RESULTS / f"{a.run_name}.json"
    if out.exists():
        sys.exit(f"{out} exists: new run names only")
    device = torch.device(a.device)
    model, ck = load_checkpoint(a.ckpt, device, a.chunk)
    ps, n_scored = arm_settings(ck["arm"], [int(x) for x in a.ps.split(",")], a.n_scored)
    rope_train_len = int(ck["args"]["seq_len"]) if ck["arm"] == "transformer" else 0
    print(f"arm {ck['arm']}: P = {ps}, {n_scored} scored bytes per target, head chunk {a.chunk or 'as trained'}", flush=True)
    res = probe_context(model, device, a.corpus, a.split, ps, a.seed, n_scored, rope_train_len, not a.no_never, a.train_end, a.tail_start, a.dev_files, a.limit)
    res = {"run_name": a.run_name, "mode": "context", "ckpt": Path(a.ckpt).name, "arm": ck["arm"], "chunk": a.chunk, "args": vars(a), **res}
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(render_context(res))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_probe_context.py -q`

Expected: PASS (13 passed).

- [ ] **Step 5: Smoke test on a real checkpoint, CPU, two targets (read-only, about 15 s)**

Run:

```bash
uv run python - <<'EOF'
import torch
from scripts.probe_commons import load_checkpoint, probe_context
torch.set_num_threads(4)
model, ck = load_checkpoint("results/ckpt/m20_linux_depth4local_s0_mingru_sparse_post.pt", torch.device("cpu"), 0)
res = probe_context(model, torch.device("cpu"), "corpora/linux_full.bytes", "dev", (2048,), 0, 4096, 0, False, limit=2)
b = res["all"]["bpb"]["full"]
print(res["n_targets"], res["all"]["n_scored_bytes"]["full"], round(b["alone"], 3), round(b["related"]["2048"], 3), round(b["random"]["2048"], 3))
EOF
```

Expected output:

```text
2 7254 0.766 0.781 0.789
```

`alone` is the model's own bits/byte on those bytes; it must be near the checkpoint's validation number (0.76, a few thousandths either way). A value above 1 means the scored bytes are misaligned.

- [ ] **Step 6: Commit**

```bash
git add scripts/probe_commons.py tests/conftest.py tests/test_probe_context.py
git commit -m "Probe: context-mode driver, per-condition and per-bin tables, command line (Stage 1.3)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Handoff probe: reader agents, merged tables, transfer ratio

Prerequisite: the agent-state plan (`flybrain/model/state.py`, `GRUHybrid.init_state` and `stream`, `flybrain/group.py`) must be merged on this branch. The tests below import it and fail otherwise; nothing in Tasks 1-7 depends on it. This task adds the per-target handoff measurement: one fresh reader agent per file, their tables summed one at a time, preloaded at each alpha into a fresh agent, the sequential reader, and the per-head difference between the two sets of tables.

**Files:**
- Modify: `scripts/probe_commons.py`
- Test: `tests/test_probe_handoff.py`

**Interfaces:**
- Consumes: `GRUHybrid.init_state(batch, device=None) -> AgentState`; `GRUHybrid.stream(tokens: LongTensor[B, T], state) -> (logits[B, T, 256], state)` (does not mutate `state`; needs chunk > 0 on every head); `flybrain.model.state.AgentState(h, heads, attn)` with `new_document() -> AgentState` (keeps `S` and `Ksum`), `HeadState(S [B, n_key, d], Ksum [B, n_key], ...)` in `all_heads()` order (post head first, then inner heads bottom-up); `flybrain.group.Commons(S: list[[n_key, d]], Ksum: list[[n_key]])`, `read(model, streams[B, T], state=None, seg=2048) -> (ce[B, T-1], state)`, `merge(states) -> Commons`, `preload(state, commons, alpha=1.0) -> AgentState`.
- Produces: `ALPHAS = (0.03, 0.1, 0.3, 1.0)`, `MIN_GAIN = 0.01`; `add_commons(a, b) -> Commons`; `head_diff(commons, state) -> list[(S_rel, Ksum_rel)]`; `handoff_target(model, device, stream, body_start, n_scored, readers, alphas, seg=2048) -> {"alone": ndarray, "handoff": {alpha: ndarray}, "sequential": ndarray, "head_diff": [...]}`; `transfer_ratio(alone, other, context) -> float | None`; `best_alpha({alpha: bpb}) -> float`; `handoff_effects(arr, alphas, idx=None) -> dict` and `bootstrap_handoff(arr, alphas, n_boot=N_BOOT, seed=0) -> dict` (gains over alone and transfer ratios from per-target `[sum of bits, count]` arrays, and their 95 % intervals over targets).

- [ ] **Step 1: Confirm the agent-state plan is merged**

Run: `uv run python -c "from flybrain.group import Commons, merge, preload, read; from flybrain.model.state import AgentState, HeadState; from flybrain.model.hybrid import GRUHybrid; assert hasattr(GRUHybrid, 'init_state') and hasattr(GRUHybrid, 'stream'); print('agent-state API present')"`

Expected output:

```text
agent-state API present
```

If this fails, stop: merge the agent-state plan first. Do not stub the API here.

- [ ] **Step 2: Write the failing test**

Create `tests/test_probe_handoff.py`:

```python
"""scripts/probe_commons.py --mode handoff. Needs the agent-state plan: flybrain.model.state, GRUHybrid.init_state / stream, flybrain.group."""

from __future__ import annotations

import numpy as np
import torch

HEAD = b"\n// ==== k/x.c ====\n"
TEXT = b"static int probe_device(struct device *dev) { return -ENODEV; }\n" * 4


def test_transfer_ratio_needs_a_context_gain_and_best_alpha_breaks_ties_low():
    import pytest

    from scripts.probe_commons import MIN_GAIN, best_alpha, transfer_ratio
    assert MIN_GAIN == 0.01
    assert transfer_ratio(1.0, 0.8, 0.6) == pytest.approx(0.5)
    assert transfer_ratio(1.0, 1.05, 0.6) == pytest.approx(-0.125)           # a handoff that hurts is reported, not clipped
    assert transfer_ratio(1.0, 0.9, 0.995) is None                           # context barely helps: the ratio would be noise
    assert best_alpha({0.03: 0.9, 0.1: 0.8, 0.3: 0.8, 1.0: 0.95}) == 0.1


def test_handoff_with_alpha_zero_equals_alone_exactly_and_a_real_alpha_changes_the_scores(tiny_hybrid):
    from scripts.probe_commons import handoff_target
    m = tiny_hybrid()
    stream = HEAD + TEXT
    out = handoff_target(m, "cpu", stream, len(HEAD), 120, [stream], (0.0, 1.0), seg=64)   # the reader reads the very bytes that are scored
    assert len(out["alone"]) == 120 and np.isfinite(out["alone"]).all()
    assert np.array_equal(out["handoff"][0.0], out["alone"])                               # alpha = 0 adds nothing: identical, not close
    assert not np.allclose(out["handoff"][1.0], out["alone"], atol=1e-4)
    assert not np.allclose(out["sequential"], out["alone"], atol=1e-4)


def test_add_commons_matches_merging_all_reader_states_at_once(tiny_hybrid):
    from flybrain.group import merge, read
    from scripts.probe_commons import add_commons, to_tokens
    m = tiny_hybrid()
    with torch.no_grad():
        states = [read(m, to_tokens(b, "cpu"))[1] for b in (b"alpha beta gamma delta " * 3, b"gamma delta epsilon " * 4, b"zeta eta theta " * 5)]
        whole, acc = merge(states), None
        for s in states:
            c = merge(s)
            acc = c if acc is None else add_commons(acc, c)
    assert len(whole.S) == len(acc.S) == 2
    for x, y in zip(whole.S + whole.Ksum, acc.S + acc.Ksum):
        assert torch.allclose(x, y, atol=1e-6)


def test_handoff_effects_are_gains_and_ratios_and_their_intervals_follow_the_targets():
    from scripts.probe_commons import bootstrap_handoff, handoff_effects
    same = {"alone": np.array([[20.0, 10]] * 3), "context": np.array([[15.0, 10]] * 3), "sequential": np.array([[18.0, 10]] * 3), "handoff:0.5": np.array([[17.0, 10]] * 3)}
    e = handoff_effects(same, (0.5,))
    assert abs(e["gain_context"] - 0.5) < 1e-12 and abs(e["gain_handoff:0.5"] - 0.3) < 1e-12 and abs(e["gain_sequential"] - 0.2) < 1e-12
    assert abs(e["transfer:0.5"] - 0.6) < 1e-12 and abs(e["transfer_sequential"] - 0.4) < 1e-12
    ci = bootstrap_handoff(same, (0.5,), 50, 1)
    assert all(abs(ci[k][0] - e[k]) < 1e-9 and abs(ci[k][1] - e[k]) < 1e-9 for k in e)       # identical targets: the interval is the point
    mixed = {"alone": np.array([[20.0, 10], [10.0, 10], [30.0, 10]]), "context": np.array([[15.0, 10], [9.0, 10], [20.0, 10]]),
             "sequential": np.array([[18.0, 10], [9.5, 10], [28.0, 10]]), "handoff:0.5": np.array([[17.0, 10], [9.5, 10], [24.0, 10]])}
    a = bootstrap_handoff(mixed, (0.5,), 100, 1)
    assert a == bootstrap_handoff(mixed, (0.5,), 100, 1) and a["gain_context"][0] < a["gain_context"][1]
    flat = {k: v.copy() for k, v in same.items()}; flat["context"] = flat["alone"].copy()
    assert handoff_effects(flat, (0.5,))["transfer:0.5"] is None and bootstrap_handoff(flat, (0.5,), 20, 1)["transfer:0.5"] is None   # no context gain: no ratio
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_probe_handoff.py -q`

Expected: FAIL with "ImportError: cannot import name 'MIN_GAIN' from 'scripts.probe_commons'" (4 failed).

- [ ] **Step 4: Write minimal implementation**

In `scripts/probe_commons.py`, insert the following immediately above the line `def load_checkpoint(path: str, device, chunk: int):`:

```python
ALPHAS = (0.03, 0.1, 0.3, 1.0)
MIN_GAIN = 0.01  # the transfer ratio is reported when alone - context is at least this many bits/byte


def add_commons(a, b):
    """Per-head sum of two Commons, so readers can be merged one at a time and only two sets of tables are alive."""
    from flybrain.group import Commons
    return Commons([x + y for x, y in zip(a.S, b.S)], [x + y for x, y in zip(a.Ksum, b.Ksum)])


def head_diff(commons, state) -> list[tuple[float, float]]:
    """Per head (all_heads() order): relative difference of S and of Ksum between the merged reader tables and the tables of one agent
    that read the same files in sequence. Only the lowest head's keys do not depend on earlier reads, so only its difference is ~0."""
    rel = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-12))
    return [(rel(S, h.S[0]), rel(K, h.Ksum[0])) for S, K, h in zip(commons.S, commons.Ksum, state.heads)]


@torch.no_grad()
def handoff_target(model, device, stream: bytes, body_start: int, n_scored: int, readers: list[bytes], alphas, seg: int = 2048) -> dict:
    """Bits of the scored bytes (arrays of length n_scored) with no memory ('alone', through the stream API), with the merged tables of
    one fresh reader agent per entry of `readers` preloaded at each alpha, and after one agent read them in order with new_document()
    between files; plus head_diff(merged readers, sequential agent)."""
    from flybrain.group import merge, preload, read
    x = to_tokens(stream, device)
    score = lambda state: bits_of(model.stream(x, state)[0], x)[body_start : body_start + n_scored]
    commons, seq = None, model.init_state(1, device)
    for r in readers:
        t = to_tokens(r, device)
        c = merge(read(model, t, None, seg)[1])
        commons = c if commons is None else add_commons(commons, c)
        seq = read(model, t, seq, seg)[1].new_document()
    return {"alone": score(model.init_state(1, device)),
            "handoff": {a: score(preload(model.init_state(1, device), commons, a)) for a in alphas},
            "sequential": score(seq), "head_diff": head_diff(commons, seq)}


def transfer_ratio(alone: float, other: float, context: float) -> float | None:
    """(alone - other) / (alone - context), reported only when the denominator is at least MIN_GAIN."""
    return (alone - other) / (alone - context) if alone - context >= MIN_GAIN else None


def best_alpha(bpb_by_alpha: dict[float, float]) -> float:
    """The alpha with the lowest handoff bits/byte; ties go to the smaller alpha."""
    return min(bpb_by_alpha, key=lambda a: (bpb_by_alpha[a], a))


def handoff_effects(arr: dict[str, np.ndarray], alphas, idx=None) -> dict[str, float | None]:
    """Gains over alone (positive = fewer bits) and transfer ratios from per-target [sum of bits, count] arrays, one array per row
    ('alone', 'context', 'sequential', 'handoff:<alpha>'); all targets, or the rows idx of a resample."""
    pick = (lambda a: a) if idx is None else (lambda a: a[idx])
    bpb = {k: pick(a)[:, 0].sum() / pick(a)[:, 1].sum() for k, a in arr.items()}
    out = {"gain_context": bpb["alone"] - bpb["context"], "gain_sequential": bpb["alone"] - bpb["sequential"],
           "transfer_sequential": transfer_ratio(bpb["alone"], bpb["sequential"], bpb["context"])}
    for a in alphas:
        out[f"gain_handoff:{a}"] = bpb["alone"] - bpb[f"handoff:{a}"]
        out[f"transfer:{a}"] = transfer_ratio(bpb["alone"], bpb[f"handoff:{a}"], bpb["context"])
    return out


def bootstrap_handoff(arr: dict[str, np.ndarray], alphas, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, list[float] | None]:
    """95 % intervals of handoff_effects over resamples of the targets (files) with replacement; a ratio whose denominator falls
    below MIN_GAIN in more than a tenth of the resamples gets None."""
    rng = np.random.default_rng(seed)
    t = len(next(iter(arr.values())))
    runs = [handoff_effects(arr, alphas, rng.integers(0, t, t)) for _ in range(n_boot)]
    return {k: ci95([r[k] for r in runs]) for k in runs[0]}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_probe_handoff.py -q`

Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add scripts/probe_commons.py tests/test_probe_handoff.py
git commit -m "Probe: handoff primitives, reader agents, merged tables, sequential reader, transfer ratio (Stage 1.4)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Handoff probe: driver, tables and command line

The handoff driver reads each target's related files as readers, scores every memory condition, pools bits per byte, picks alpha on dev and reports the transfer ratio. The spec gives no total for the readers; 32,768 bytes is the gate's P and the capacity it quotes for a 65,536-slot table, and the context row is a plain forward over exactly the reader bytes, so the transfer ratio compares the same material. The command line gains `--mode handoff`, `--alpha`, `--budget` and `--per-file`; `--chunk` now defaults to the checkpoint's own chunk in this mode. Needs the agent-state plan like Task 8.

**Files:**
- Modify: `scripts/probe_commons.py`
- Modify: `tests/test_probe_handoff.py`
- Test: `tests/test_probe_handoff.py`

**Interfaces:**
- Consumes: Task 2: `reader_chunks`; Task 5: `score_with_prefix`; Task 8: `handoff_target`, `transfer_ratio`, `best_alpha`, `ALPHAS`; Task 7: `load_checkpoint`, `probe_context`, `render_context`.
- Produces: `probe_handoff(model, device, path, split, alphas=ALPHAS, seed=0, n_scored=4096, budget=32_768, per_file=8192, seg=2048, train_end=TRAIN_END, tail_start=TAIL_START, dev_files=DEV_FILES, limit=0) -> dict`; `render_handoff(res) -> str`; `main(argv=None)` with `--mode context|handoff`. Writes `results/<run-name>.json` with this schema (bits/byte pooled over bytes; alpha keys are strings such as `"0.03"`; `transfer*` are `null` when alone - context < 0.01):

```text
{"run_name": str, "mode": "handoff", "ckpt": file name, "arm": str, "chunk": int, "args": {...},
 "split": str, "seed": int, "alphas": [float, ...], "budget": int, "per_file": int, "n_targets": int, "n_scored_bytes": int,
 "bpb": {"alone": float, "alone_forward": float, "context": float, "sequential": float, "handoff": {"<alpha>": float}},
 "best_alpha": float, "gain_context": float, "gain_sequential": float, "gain_handoff": {"<alpha>": float},      # alone minus the row, positive = fewer bits
 "transfer": {"<alpha>": float | null}, "transfer_best": float | null, "transfer_sequential": float | null,
 "ci95": {"gain_context": [lo, hi], "gain_sequential": [lo, hi], "gain_handoff": {"<alpha>": [lo, hi]}, "transfer": {"<alpha>": [lo, hi] | null},
          "transfer_best": [lo, hi] | null, "transfer_sequential": [lo, hi] | null},      # bootstrap over targets, 1,000 resamples, seeded
 "head_diff": [{"head": int, "S_rel": float, "Ksum_rel": float}],      # all_heads() order
 "readers": {"mean_files": float, "mean_bytes": float},
 "targets": [{"index": int, "path": str, "n_scored": int, "readers": int, "reader_bytes": int,
              "bpb": {"alone": float, "context": float, "alone_forward": float, "sequential": float, "handoff": {"<alpha>": float}}}]}
```

- [ ] **Step 1: Write the failing test**

Append to `tests/test_probe_handoff.py`:

```python
def _handoff(world, tiny_hybrid, alphas=(0.0, 0.5), limit=0, budget=600, per_file=300):
    from scripts.probe_commons import probe_handoff
    return probe_handoff(tiny_hybrid(), "cpu", world.path, "dev", alphas, 0, 300, budget, per_file, 64, world.train_end, world.tail_start, world.dev_files, limit)


def test_probe_handoff_alone_alpha_zero_and_sequential_head_difference(world, tiny_hybrid):
    res = _handoff(world, tiny_hybrid)
    b = res["bpb"]
    assert b["handoff"]["0.0"] == b["alone"]                                  # alpha = 0 gives exactly the alone numbers
    assert abs(b["alone"] - b["alone_forward"]) < 1e-3                         # the stream API agrees with the plain forward
    assert b["handoff"]["0.5"] != b["alone"] and b["context"] != b["alone"]
    assert res["n_targets"] == 3 and res["n_scored_bytes"] == 690 and res["readers"]["mean_files"] == 2.0
    lowest = res["head_diff"][1]                                               # all_heads(): post head first, then the inner head
    assert lowest["S_rel"] < 1e-4 and lowest["Ksum_rel"] < 1e-4                # the lowest head's keys do not depend on earlier reads
    assert res["head_diff"][0]["S_rel"] > 1e-4                                 # the post head's do, so its tables differ from the sum
    assert res["best_alpha"] in (0.0, 0.5) and res["transfer_best"] == res["transfer"][str(res["best_alpha"])]
    ci = res["ci95"]
    assert set(ci) == {"gain_context", "gain_sequential", "gain_handoff", "transfer", "transfer_best", "transfer_sequential"}
    assert len(ci["gain_context"]) == 2 and set(ci["gain_handoff"]) == {"0.0", "0.5"}
    assert ci["gain_handoff"]["0.0"] == [0.0, 0.0] and res["gain_handoff"]["0.0"] == 0.0     # alpha = 0 changes nothing, in every resample
    assert abs(res["gain_context"] - (b["alone"] - b["context"])) < 1e-12 and abs(res["gain_sequential"] - (b["alone"] - b["sequential"])) < 1e-12


def test_readers_are_cut_by_file_and_by_budget(world, tiny_hybrid):
    res = _handoff(world, tiny_hybrid, budget=1000, per_file=300)
    assert [t["readers"] for t in res["targets"]] == [4, 4, 4] and [t["reader_bytes"] for t in res["targets"]] == [1000, 1000, 1000]
    one = _handoff(world, tiny_hybrid, budget=250, per_file=300)
    assert [t["readers"] for t in one["targets"]] == [1, 1, 1] and [t["reader_bytes"] for t in one["targets"]] == [250, 250, 250]


def test_handoff_probe_is_deterministic_and_json_serialisable(world, tiny_hybrid):
    import json
    a, b = _handoff(world, tiny_hybrid), _handoff(world, tiny_hybrid)
    assert json.dumps(a) == json.dumps(b)


def test_main_handoff_scores_the_given_alpha_or_all_of_them(world, tiny_checkpoint, tmp_path, monkeypatch):
    import json

    import pytest
    import scripts.probe_commons as pc
    monkeypatch.setattr(pc, "RESULTS", tmp_path)
    common = ["--mode", "handoff", "--ckpt", str(tiny_checkpoint()), "--split", "dev", "--corpus", str(world.path), "--device", "cpu", "--n-scored", "300",
              "--budget", "600", "--per-file", "300", "--train-end", str(world.train_end), "--tail-start", str(world.tail_start), "--dev-files", "3"]
    pc.main(["--run-name", "ho_all", *common])
    res = json.loads((tmp_path / "ho_all.json").read_text())
    assert {"run_name", "mode", "ckpt", "arm", "chunk", "args", "split", "seed", "alphas", "budget", "per_file", "n_targets", "n_scored_bytes", "bpb",
            "best_alpha", "gain_context", "transfer", "transfer_best", "transfer_sequential", "head_diff", "readers", "targets"} <= set(res)
    assert res["alphas"] == [0.03, 0.1, 0.3, 1.0] and set(res["bpb"]["handoff"]) == {"0.03", "0.1", "0.3", "1.0"} and res["chunk"] == 0
    pc.main(["--run-name", "ho_one", "--alpha", str(res["best_alpha"]), *common])
    one = json.loads((tmp_path / "ho_one.json").read_text())
    assert one["alphas"] == [res["best_alpha"]] and one["best_alpha"] == res["best_alpha"]
    assert one["bpb"]["handoff"][str(res["best_alpha"])] == res["bpb"]["handoff"][str(res["best_alpha"])]
    with pytest.raises(SystemExit):
        pc.main(["--run-name", "ho_one", *common])


def test_render_handoff_prints_every_condition_and_the_head_table(world, tiny_hybrid):
    from scripts.probe_commons import render_handoff
    text = render_handoff({"run_name": "tiny", **_handoff(world, tiny_hybrid)})
    assert "| condition | bits/byte | gain vs alone (95 % interval) | transfer ratio (95 % interval) |" in text and "| sequential reader |" in text and "| handoff alpha 0.5 |" in text
    assert "| head (all_heads() order) | S relative difference | Ksum relative difference |" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_probe_handoff.py -q`

Expected: FAIL with "ImportError: cannot import name 'probe_handoff' from 'scripts.probe_commons'" (5 failed, 4 passed).

- [ ] **Step 3: Write minimal implementation**

In `scripts/probe_commons.py`, insert the following immediately above the line `def load_checkpoint(path: str, device, chunk: int):`:

```python
def probe_handoff(model, device, path, split: str, alphas=ALPHAS, seed: int = 0, n_scored: int = 4096, budget: int = 32_768, per_file: int = 8192,
                  seg: int = 2048, train_end: int = TRAIN_END, tail_start: int = TAIL_START, dev_files: int = DEV_FILES, limit: int = 0) -> dict:
    """Readers are the target's related files in path order, the first per_file bytes of each, budget bytes in all. The context row is a
    plain forward over exactly those bytes as a prefix, so the transfer ratio compares the same material."""
    table = file_table(path)
    hashes = hash_table(path, table)
    idxs = splits(table, tail_start, dev_files)[split]
    if not idxs:
        raise ValueError(f"split {split!r} holds no files")
    per: dict[str, list[tuple[float, int]]] = {}
    recs, diffs = [], []
    for i in targets(table, idxs, hashes, 4, train_end)[: limit or None]:
        stream, body_start, scored = read_stream(path, table[i], n_scored)
        if scored == 0:
            continue
        readers = [b for _, b in reader_chunks(path, table, i, hashes, budget, per_file, 0, train_end)]
        out = handoff_target(model, device, stream, body_start, scored, readers, alphas, seg)
        rows = {"alone": out["alone"], "context": score_with_prefix(model, b"".join(readers), stream, body_start, scored, device),
                "alone_forward": score_with_prefix(model, b"", stream, body_start, scored, device), "sequential": out["sequential"]}
        for name, bits in [*rows.items(), *((f"handoff:{a}", v) for a, v in out["handoff"].items())]:
            per.setdefault(name, []).append((float(bits.sum()), len(bits)))
        diffs.append(out["head_diff"])
        recs.append({"index": i, "path": table[i].path, "n_scored": scored, "readers": len(readers), "reader_bytes": sum(map(len, readers)),
                     "bpb": {**{k: float(v.mean()) for k, v in rows.items()}, "handoff": {str(a): float(v.mean()) for a, v in out["handoff"].items()}}})
        if len(recs) % 25 == 0:
            print(f"  {len(recs)} targets", flush=True)
    if not recs:
        raise ValueError("no target to score")
    arr = {k: np.array(v) for k, v in per.items()}  # [target, (sum of bits, count)]
    pooled = {k: a[:, 0].sum() / a[:, 1].sum() for k, a in arr.items()}
    handoff = {a: pooled[f"handoff:{a}"] for a in alphas}
    best = best_alpha(handoff)
    eff, ci = handoff_effects(arr, alphas), bootstrap_handoff(arr, alphas, N_BOOT, seed)
    d = np.array(diffs)  # [target, head, (S, Ksum)]
    return {"split": split, "seed": seed, "alphas": list(alphas), "budget": budget, "per_file": per_file, "n_targets": len(recs),
            "n_scored_bytes": sum(r["n_scored"] for r in recs),
            "bpb": {"alone": pooled["alone"], "alone_forward": pooled["alone_forward"], "context": pooled["context"], "sequential": pooled["sequential"],
                    "handoff": {str(a): v for a, v in handoff.items()}},
            "best_alpha": best, "gain_context": eff["gain_context"], "gain_sequential": eff["gain_sequential"],
            "gain_handoff": {str(a): eff[f"gain_handoff:{a}"] for a in alphas},
            "transfer": {str(a): eff[f"transfer:{a}"] for a in alphas}, "transfer_best": eff[f"transfer:{best}"], "transfer_sequential": eff["transfer_sequential"],
            "ci95": {"gain_context": ci["gain_context"], "gain_sequential": ci["gain_sequential"], "gain_handoff": {str(a): ci[f"gain_handoff:{a}"] for a in alphas},
                     "transfer": {str(a): ci[f"transfer:{a}"] for a in alphas}, "transfer_best": ci[f"transfer:{best}"], "transfer_sequential": ci["transfer_sequential"]},
            "head_diff": [{"head": h, "S_rel": float(d[:, h, 0].mean()), "Ksum_rel": float(d[:, h, 1].mean())} for h in range(d.shape[1])],
            "readers": {"mean_files": float(np.mean([r["readers"] for r in recs])), "mean_bytes": float(np.mean([r["reader_bytes"] for r in recs]))},
            "targets": recs}


def render_handoff(res: dict) -> str:
    """Markdown tables: bits/byte, gain over alone and transfer ratio (95 % intervals over targets) per memory condition, and the per-head table difference."""
    b, ci = res["bpb"], res["ci95"]
    lines = [f"{res.get('run_name', '')} {res['split']}: {res['n_targets']} targets, {res['n_scored_bytes']} scored bytes, readers {res['readers']['mean_files']:.1f} files / "
             f"{res['readers']['mean_bytes']:.0f} bytes on average, best alpha {res['best_alpha']}",
             "| condition | bits/byte | gain vs alone (95 % interval) | transfer ratio (95 % interval) |", "|---|---:|---:|---:|",
             f"| alone | {b['alone']:.4f} | | |",
             f"| context (same bytes as a plain prefix) | {b['context']:.4f} | {_ci(res['gain_context'], ci['gain_context'])} | 1.00 |"]
    lines += [f"| handoff alpha {a} | {v:.4f} | {_ci(res['gain_handoff'][a], ci['gain_handoff'][a])} | {_ci(res['transfer'][a], ci['transfer'][a], '.2f')} |" for a, v in b["handoff"].items()]
    lines += [f"| sequential reader | {b['sequential']:.4f} | {_ci(res['gain_sequential'], ci['gain_sequential'])} | {_ci(res['transfer_sequential'], ci['transfer_sequential'], '.2f')} |",
              f"\nalone through the plain forward: {b['alone_forward']:.4f}", "\n| head (all_heads() order) | S relative difference | Ksum relative difference |", "|---:|---:|---:|"]
    lines += [f"| {h['head']} | {h['S_rel']:.2e} | {h['Ksum_rel']:.2e} |" for h in res["head_diff"]]
    return "\n".join(lines)
```

In `scripts/probe_commons.py`, replace this block:

```python
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("context",))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--corpus", default=str(ROOT / "corpora" / "linux_full.bytes"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ps", default="2048,4096", help="prefix sizes in bytes (the transformer arm always uses 2048)")
    ap.add_argument("--n-scored", type=int, default=4096, help="scored bytes per target (the transformer arm always scores 2048)")
    ap.add_argument("--chunk", type=int, default=4096, help="head chunk length for long sequences; 0 keeps the checkpoint's own")
    ap.add_argument("--no-never", action="store_true", help="skip the never-trained condition")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N targets only")
    ap.add_argument("--train-end", type=int, default=TRAIN_END, help="tiny-corpus tests only")
    ap.add_argument("--tail-start", type=int, default=TAIL_START, help="tiny-corpus tests only")
    ap.add_argument("--dev-files", type=int, default=DEV_FILES, help="tiny-corpus tests only")
    a = ap.parse_args(argv)
    out = RESULTS / f"{a.run_name}.json"
    if out.exists():
        sys.exit(f"{out} exists: new run names only")
    device = torch.device(a.device)
    model, ck = load_checkpoint(a.ckpt, device, a.chunk)
    ps, n_scored = arm_settings(ck["arm"], [int(x) for x in a.ps.split(",")], a.n_scored)
    rope_train_len = int(ck["args"]["seq_len"]) if ck["arm"] == "transformer" else 0
    print(f"arm {ck['arm']}: P = {ps}, {n_scored} scored bytes per target, head chunk {a.chunk or 'as trained'}", flush=True)
    res = probe_context(model, device, a.corpus, a.split, ps, a.seed, n_scored, rope_train_len, not a.no_never, a.train_end, a.tail_start, a.dev_files, a.limit)
    res = {"run_name": a.run_name, "mode": "context", "ckpt": Path(a.ckpt).name, "arm": ck["arm"], "chunk": a.chunk, "args": vars(a), **res}
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(render_context(res))
    print(f"wrote {out}")
```

with:

```python
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("context", "handoff"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--corpus", default=str(ROOT / "corpora" / "linux_full.bytes"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ps", default="2048,4096", help="context: prefix sizes in bytes (the transformer arm always uses 2048)")
    ap.add_argument("--n-scored", type=int, default=4096, help="scored bytes per target (the transformer arm always scores 2048)")
    ap.add_argument("--chunk", type=int, default=None, help="head chunk length; 0 keeps the checkpoint's own. Default 4096 in context mode, 0 in handoff mode")
    ap.add_argument("--no-never", action="store_true", help="context: skip the never-trained condition")
    ap.add_argument("--alpha", type=float, default=None, help="handoff: score only this alpha (the one chosen on dev); default scores every alpha")
    ap.add_argument("--budget", type=int, default=32_768, help="handoff: total reader bytes per target")
    ap.add_argument("--per-file", type=int, default=8192, help="handoff: bytes each reader reads")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N targets only")
    ap.add_argument("--train-end", type=int, default=TRAIN_END, help="tiny-corpus tests only")
    ap.add_argument("--tail-start", type=int, default=TAIL_START, help="tiny-corpus tests only")
    ap.add_argument("--dev-files", type=int, default=DEV_FILES, help="tiny-corpus tests only")
    a = ap.parse_args(argv)
    out = RESULTS / f"{a.run_name}.json"
    if out.exists():
        sys.exit(f"{out} exists: new run names only")
    chunk = a.chunk if a.chunk is not None else (4096 if a.mode == "context" else 0)
    device = torch.device(a.device)
    model, ck = load_checkpoint(a.ckpt, device, chunk)
    if a.mode == "context":
        ps, n_scored = arm_settings(ck["arm"], [int(x) for x in a.ps.split(",")], a.n_scored)
        rope_train_len = int(ck["args"]["seq_len"]) if ck["arm"] == "transformer" else 0
        print(f"arm {ck['arm']}: P = {ps}, {n_scored} scored bytes per target, head chunk {chunk or 'as trained'}", flush=True)
        res = probe_context(model, device, a.corpus, a.split, ps, a.seed, n_scored, rope_train_len, not a.no_never, a.train_end, a.tail_start, a.dev_files, a.limit)
        text = render_context
    else:
        alphas = ALPHAS if a.alpha is None else (a.alpha,)
        res = probe_handoff(model, device, a.corpus, a.split, alphas, a.seed, a.n_scored, a.budget, a.per_file, 2048, a.train_end, a.tail_start, a.dev_files, a.limit)
        text = render_handoff
    res = {"run_name": a.run_name, "mode": a.mode, "ckpt": Path(a.ckpt).name, "arm": ck["arm"], "chunk": chunk, "args": vars(a), **res}
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(text(res))
    print(f"wrote {out}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_probe_context.py tests/test_probe_handoff.py -q`

Expected: PASS (22 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_commons.py tests/test_probe_handoff.py
git commit -m "Probe: handoff driver with alpha chosen on dev, transfer ratio and per-head table difference; --mode handoff (Stage 1.4)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Runs and write-up (notebook Runs 50-52)

Everything here runs after Tasks 1-9 are committed. Parts A-B (census, context probes) need nothing from the other plans; part C (handoff) needs the agent-state plan merged into the commit you pin. The GPU rules in Global Constraints apply to every step: GPU 1 only, the qwen server stopped for the duration and started again when the queue is empty, no other compute process on the card, 150 W, jobs killed by explicit PID, new run names only. Expect roughly 2 hours of GPU time in all (an estimate: context probes about 1 h, handoff about 1 h). The queue scripts skip any name whose JSON exists, so a failed run is repaired in a new commit and only the missing names are rerun.

**Files:**
- Create: `results/census_linux_{dev,test}.json`
- Create: `results/commons_ctx_{x64k,local,m50,tf}_{dev,test}.json`
- Create: `results/commons_handoff_x64k_{dev,test}.json`
- Modify: `notebook.md`

**Interfaces:**
- Consumes: The command lines of `scripts/census.py` and `scripts/probe_commons.py` and the JSON schemas of Tasks 4, 7 and 9; the agent-state plan for part C.
- Produces: `results/census_linux_{dev,test}.json`, `results/commons_ctx_{x64k,local,m50,tf}_{dev,test}.json`, `results/commons_handoff_x64k_{dev,test}.json` (new files, committed), and notebook Runs 50 (census), 51 (related reading) and 52 (handoff).

- [ ] **Step 1: Pre-flight in the main checkout**

Run:

```bash
cd /games/ailab/lab/flybrain-lab
git status --short
git log --oneline -3
uv run pytest -q
```

Expected: `git status --short` prints nothing; the log shows this plan's commits on `feat/group-learning`; pytest ends with `N passed` and no failure (this plan adds 50 tests to the 143 the repo had, plus whatever the other plans added). Fix anything red before going on.

- [ ] **Step 2: Pin a commit in a worktree and link the corpus and the checkpoints**

Run:

```bash
cd /games/ailab/lab/flybrain-lab
WT=/games/ailab/lab/flybrain-lab/.worktrees/probes
grep -qxF '.worktrees/' .gitignore || { printf '.worktrees/\n' >> .gitignore; git add .gitignore; git commit -m "Ignore .worktrees/, where pinned run checkouts live"; }
git worktree add --detach "$WT" "$(git rev-parse HEAD)"
ln -s /games/ailab/lab/flybrain-lab/corpora "$WT/corpora"
ln -s /games/ailab/lab/flybrain-lab/results/ckpt "$WT/results/ckpt"
git -C "$WT" log --oneline -1
```

Expected: the pinned commit's one-line log. `corpora/` and `results/ckpt/` are git-ignored, hence the links; the runs use the main checkout's virtualenv (`$PY` below), so nothing is installed in the worktree.

- [ ] **Step 3: Confirm the four checkpoints by name and size**

Run:

```bash
cd /games/ailab/lab/flybrain-lab
ls -l results/ckpt/m20_linux_depth4x64k_s0_mingru_sparse_post.pt results/ckpt/m20_linux_depth4local_s0_mingru_sparse_post.pt results/ckpt/m50_linux_hybrid_s0_mingru_sparse_post.pt results/ckpt/m20_linux_s0_transformer.pt | awk '{print $5, $NF}'
```

Expected output:

```text
97884597 results/ckpt/m20_linux_depth4local_s0_mingru_sparse_post.pt
597966922 results/ckpt/m20_linux_depth4x64k_s0_mingru_sparse_post.pt
68380048 results/ckpt/m20_linux_s0_transformer.pt
285400350 results/ckpt/m50_linux_hybrid_s0_mingru_sparse_post.pt
```

These are the 65,536-slot hybrid, the 20M hybrid with local attention, the 50M hybrid and the 17.09M rotary transformer (seed 0; `m20_linux_s1_transformer.pt` and `m20_linux_tf8k_s0_transformer.pt` also exist and are not used). The listing is ordered by name, so it differs from the order above; four lines with these sizes mean the right files are there.

- [ ] **Step 4: Part A: census on dev and test, CPU, about 35 s each**

Run:

```bash
WT=/games/ailab/lab/flybrain-lab/.worktrees/probes
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
cd "$WT"
$PY scripts/census.py --split dev  --run-name census_linux_dev  --never-trained > results/census_linux_dev.log
$PY scripts/census.py --split test --run-name census_linux_test --never-trained > results/census_linux_test.log
$PY - <<'EOF'
import json
for s in ("dev", "test"):
    r = json.load(open(f"results/census_linux_{s}.json"))
    print(s, r["n_targets"], r["n_scored_bytes"], r["never_trained"]["n_targets"], r["never_trained"]["n_scored_bytes"], round(r["trained"]["cells"]["8"]["32768"]["share"]["B"], 3))
EOF
```

Expected output:

```text
dev 353 1364676 101 382536 0.164
test 339 1299363 83 313317 0.152
```

These numbers are deterministic (seed 0). They must equal the spec's counts: 353 / 339 targets, 101 / 83 never-trained targets. The last column is the share of bytes in bin B at n = 8, P = 32,768 (the census half of the gate).

- [ ] **Step 5: Part B: GPU pre-flight, stop the server, then check the card and the power cap**

Run:

```bash
systemctl --user stop qwen-server.service
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv
nvidia-smi -i 1 --query-gpu=power.limit --format=csv,noheader
```

Expected: the second query (card 1, the card these runs use) prints only its header line `pid, used_gpu_memory [MiB]`; the first query lists every card, so another project's process on card 0 may appear there and is none of your business; the last line is `150.00 W`. If card 1 still lists the llama server, repeat the query until it is gone. If any other process is on card 1, or the cap is not 150 W, do not start: run `systemctl --user start qwen-server.service` and ask Kyle. Never kill a process you did not start.

- [ ] **Step 6: Part B: GPU smoke test of the 65,536-slot model, 3 dev targets, about 2 min**

Run:

```bash
cd /games/ailab/lab/flybrain-lab/.worktrees/probes
export CUDA_DEVICE_ORDER=PCI_BUS_ID
/games/ailab/lab/flybrain-lab/.venv/bin/python scripts/probe_commons.py --mode context --split dev --run-name smoke_ctx_x64k_dev --device cuda:1 --limit 3 --ps 8192,16384,32768 --ckpt /games/ailab/lab/flybrain-lab/results/ckpt/m20_linux_depth4x64k_s0_mingru_sparse_post.pt > results/smoke_ctx_x64k_dev.log 2>&1
head -14 results/smoke_ctx_x64k_dev.log
rm results/smoke_ctx_x64k_dev.json results/smoke_ctx_x64k_dev.log
```

Expected: the first line reads `arm mingru_sparse_post: P = [8192, 16384, 32768], 4096 scored bytes per target, head chunk 4096`; then a `smoke_ctx_x64k_dev dev: 3 targets ...` line and the bits/byte table with `alone` near 0.8 (0.7 to 1.0; above 1.2 means misaligned scoring). The run must finish without an out-of-memory error (estimate: the 37 k-byte sequences need about 3 GB for the intra-chunk tensors at chunk 4096). If it runs out of memory, add `--chunk 2048` to every command below: the result is the same up to float rounding. `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes `cuda:1` the card `nvidia-smi -i 1` shows. The `rm` removes only the two smoke files this step created.

- [ ] **Step 7: Part B: write the context queue (4 checkpoints x dev, test)**

Create `/games/ailab/lab/flybrain-lab/.worktrees/probes/queue_ctx.sh`:

```bash
#!/usr/bin/env bash
# Context probes: four checkpoints, dev then test, GPU 1. A name whose JSON exists is skipped.
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # cuda:1 is the card nvidia-smi calls 1
WT=/games/ailab/lab/flybrain-lab/.worktrees/probes
CK=/games/ailab/lab/flybrain-lab/results/ckpt
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
cd "$WT"
run() { name=$1; shift; [ -f "results/$name.json" ] && { echo "skip $name"; return; }; echo "=== $name $(date +%T)"; "$PY" scripts/probe_commons.py --run-name "$name" --device cuda:1 "$@" > "results/$name.log" 2>&1 || echo "FAILED $name"; }
for split in dev test; do
  run commons_ctx_x64k_$split  --mode context --split $split --ckpt $CK/m20_linux_depth4x64k_s0_mingru_sparse_post.pt --ps 8192,16384,32768
  run commons_ctx_local_$split --mode context --split $split --ckpt $CK/m20_linux_depth4local_s0_mingru_sparse_post.pt --ps 2048,4096
  run commons_ctx_m50_$split   --mode context --split $split --ckpt $CK/m50_linux_hybrid_s0_mingru_sparse_post.pt --ps 2048,4096
  run commons_ctx_tf_$split    --mode context --split $split --ckpt $CK/m20_linux_s0_transformer.pt
done
echo "queue done $(date +%T)"
```

- [ ] **Step 8: Part B: launch the context queue detached**

Run: `systemd-run --user --unit=commons-ctx --working-directory=/games/ailab/lab/flybrain-lab/.worktrees/probes /bin/bash /games/ailab/lab/flybrain-lab/.worktrees/probes/queue_ctx.sh`

Expected: `Running as unit: commons-ctx.service`. Poll with `systemctl --user is-active commons-ctx` (`active` while it runs, `inactive` when finished), `tail -n 2 /games/ailab/lab/flybrain-lab/.worktrees/probes/results/commons_ctx_*.log` for progress (a `N targets` line every 25 targets) and `nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv` (one row: the queue's python). To stop early, list the unit's processes with `systemctl --user status commons-ctx` and `kill` the bash PID and then the python PID, by number.

- [ ] **Step 9: Part B: check the eight context results**

Run:

```bash
cd /games/ailab/lab/flybrain-lab/.worktrees/probes
ls results/commons_ctx_*.json | wc -l
/games/ailab/lab/flybrain-lab/.venv/bin/python - <<'EOF'
import glob, json
for f in sorted(glob.glob("results/commons_ctx_*.json")):
    r = json.load(open(f)); b = r["all"]["bpb"]["full"]
    print(f.split("/")[-1], r["n_targets"], round(b["alone"], 3), {p: round(v, 3) for p, v in b["related"].items()}, None if r["gate_number"] is None else round(r["gate_number"], 4))
EOF
```

Expected: `8`, then eight lines. Targets: 353 on dev, 339 on test. `alone` between 0.7 and 1.0 on every line (hybrids about 0.8; the transformer scores 2,048 bytes, so its number is not comparable). The gate number is filled only on the two `x64k` lines. A `FAILED <name>` line in the queue's output means that run's `.log` holds the traceback: fix the cause in a new commit, pin it, and rerun the queue; finished names are skipped.

- [ ] **Step 10: Part C: confirm the agent-state plan is in the pinned commit**

Run:

```bash
cd /games/ailab/lab/flybrain-lab/.worktrees/probes
/games/ailab/lab/flybrain-lab/.venv/bin/python -c "import flybrain.group, flybrain.model.state; print('agent-state API present')"
```

Expected: `agent-state API present`. If not, skip to the server restart below, commit the census and context results, and come back to part C once the agent-state plan is merged (pin a new commit then; the queue skips finished names).

- [ ] **Step 11: Part C: GPU smoke test of the handoff, 3 dev targets, about 3 min**

Run:

```bash
cd /games/ailab/lab/flybrain-lab/.worktrees/probes
export CUDA_DEVICE_ORDER=PCI_BUS_ID
/games/ailab/lab/flybrain-lab/.venv/bin/python scripts/probe_commons.py --mode handoff --split dev --run-name smoke_handoff_x64k_dev --device cuda:1 --limit 3 --ckpt /games/ailab/lab/flybrain-lab/results/ckpt/m20_linux_depth4x64k_s0_mingru_sparse_post.pt > results/smoke_handoff_x64k_dev.log 2>&1
head -16 results/smoke_handoff_x64k_dev.log
rm results/smoke_handoff_x64k_dev.json results/smoke_handoff_x64k_dev.log
```

Expected: the table `| condition | bits/byte | gain vs alone | transfer ratio |` with rows alone, context, four alphas and sequential; `alone through the plain forward` within 0.001 of `alone` (the stream API agrees with the plain forward); in the head table, head 1 (the lowest inner head) at about 1e-6 or below and head 0 (the post head) clearly larger. Estimated peak memory about 4 GB (reader, merged, sequential and scoring tables at 65,536 slots x 512 x 4 heads, 537 MB each).

- [ ] **Step 12: Part C: write the handoff queue (dev, then test at the alpha chosen on dev)**

Create `/games/ailab/lab/flybrain-lab/.worktrees/probes/queue_handoff.sh`:

```bash
#!/usr/bin/env bash
# Handoff probe on the 65,536-slot checkpoint: dev scores every alpha, test scores once at the alpha chosen on dev. GPU 1.
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # cuda:1 is the card nvidia-smi calls 1
WT=/games/ailab/lab/flybrain-lab/.worktrees/probes
CK=/games/ailab/lab/flybrain-lab/results/ckpt
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
cd "$WT"
run() { name=$1; shift; [ -f "results/$name.json" ] && { echo "skip $name"; return; }; echo "=== $name $(date +%T)"; "$PY" scripts/probe_commons.py --run-name "$name" --device cuda:1 "$@" > "results/$name.log" 2>&1 || echo "FAILED $name"; }
run commons_handoff_x64k_dev --mode handoff --split dev --ckpt $CK/m20_linux_depth4x64k_s0_mingru_sparse_post.pt
[ -f results/commons_handoff_x64k_dev.json ] || { echo "no dev handoff result: stopping"; exit 1; }
ALPHA=$("$PY" -c 'import json; print(json.load(open("results/commons_handoff_x64k_dev.json"))["best_alpha"])')
echo "alpha chosen on dev: $ALPHA"
run commons_handoff_x64k_test --mode handoff --split test --ckpt $CK/m20_linux_depth4x64k_s0_mingru_sparse_post.pt --alpha "$ALPHA"
echo "queue done $(date +%T)"
```

- [ ] **Step 13: Part C: launch the handoff queue detached**

Run: `systemd-run --user --unit=commons-handoff --working-directory=/games/ailab/lab/flybrain-lab/.worktrees/probes /bin/bash /games/ailab/lab/flybrain-lab/.worktrees/probes/queue_handoff.sh`

Expected: `Running as unit: commons-handoff.service`. Poll as for the context queue. The queue prints `alpha chosen on dev: <alpha>` and scores test once at it. When it is `inactive`: `ls /games/ailab/lab/flybrain-lab/.worktrees/probes/results/commons_handoff_x64k_*.json` lists two files and `grep -c . /games/ailab/lab/flybrain-lab/.worktrees/probes/results/commons_handoff_x64k_test.log` is non-zero.

- [ ] **Step 14: Start the server again (the queues are empty)**

Run:

```bash
systemctl --user start qwen-server.service
systemctl --user is-active qwen-server.service
```

Expected: `active`. Do this only when no queue is left running; if you stopped after the context queue, stop the server again before part C.

- [ ] **Step 15: Write-up: copy the results into the main checkout and commit them**

Run:

```bash
cd /games/ailab/lab/flybrain-lab
cp -n /games/ailab/lab/flybrain-lab/.worktrees/probes/results/census_linux_*.json /games/ailab/lab/flybrain-lab/.worktrees/probes/results/commons_*.json results/
git add results/census_linux_dev.json results/census_linux_test.json results/commons_ctx_x64k_dev.json results/commons_ctx_x64k_test.json results/commons_ctx_local_dev.json results/commons_ctx_local_test.json results/commons_ctx_m50_dev.json results/commons_ctx_m50_test.json results/commons_ctx_tf_dev.json results/commons_ctx_tf_test.json results/commons_handoff_x64k_dev.json results/commons_handoff_x64k_test.json
git commit -m "Results: census, related-reading and handoff probes on the dev and test splits (Stage 1.2-1.4)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

Expected: a commit of 12 JSON files (10 if part C is still to do: drop the two `commons_handoff_x64k_*` paths from `git add`). `cp -n` never overwrites a file already in `results/`; the `.log` files and smoke files stay out.

- [ ] **Step 16: Write-up: write the notebook-entry generator (stays untracked)**

It reads the result JSONs, builds the tables of Runs 50-52 and the Claim / Not-a-claim pairs from their numbers (the gate sentence picks its branch from the 0.02 threshold; the decision itself is Kyle's), and with `--insert` splices the three entries into `notebook.md` above `## Path to V3 (ladder plumbing)`. It refuses to run twice.

Create `/games/ailab/lab/flybrain-lab/.worktrees/probes/gen_entries.py`:

```python
"""Print notebook Runs 50-52 from the result JSONs; --insert splices them into notebook.md above the Path to V3 section.
usage: RUN_DATE=2026-10-04 python gen_entries.py [--insert]   (RESULTS_DIR overrides results/)"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

RES = Path(os.environ.get("RESULTS_DIR", "results"))
DATE = os.environ["RUN_DATE"]
GATE = 0.02
load = lambda name: json.loads((RES / f"{name}.json").read_text())
f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
LABEL = {"x64k": "4 heads x 65,536 slots (`m20_linux_depth4x64k_s0`)", "local": "4 heads x 4,096 + local 512 (`m20_linux_depth4local_s0`)",
         "m50": "50M hybrid (`m50_linux_hybrid_s0`)", "tf": "transformer 17.09M (`m20_linux_s0_transformer`)"}


def ci(v, c, fmt: str = "+.3f") -> str:
    """A value with its 95 % interval over targets: '+0.034 (+0.020, +0.048)'."""
    return "n/a" if v is None else format(v, fmt) + ("" if c is None else f" ({format(c[0], fmt)}, {format(c[1], fmt)})")


def gain(block: dict, scope: str, p, key: str) -> str:
    return ci(block["gains"][scope][str(p)][key], block["ci95"][scope][str(p)][key])


def margin(res: dict, n, p, a: str = "B", b: str = "Bp", n_boot: int = 1000, seed: int = 0) -> str:
    """Pooled share of bin a minus bin b at (n, P) of a census result, with a 95 % interval from resampling its targets (files)."""
    cells = [t["cells"][str(n)][str(p)] for t in res["targets"]]
    x = np.array([[c[a] - c[b], c["total"]] for c in cells], dtype=float)
    rng = np.random.default_rng(seed)
    draws = [x[i, 0].sum() / x[i, 1].sum() for i in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
    return ci(x[:, 0].sum() / x[:, 1].sum(), [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))])


def run50() -> str:
    ct, cd = load("census_linux_test"), load("census_linux_dev")
    ps, t, nt = ct["ps"], ct["trained"], ct["never_trained"]
    sh = lambda blk, n, p, k: blk["cells"][str(n)][str(p)]["share"][k]
    gp = ps[-1]
    rel_files = statistics.median(r["related_files"] for r in ct["targets"])
    rows1 = [f"| {p:,} | {f3(sh(t, 8, p, 'A'))} | {f3(sh(t, 8, p, 'B'))} | {f3(sh(t, 8, p, 'Bp'))} | {f3(sh(t, 8, p, 'C'))} | {margin(ct, 8, p)} |" for p in ps]
    rows2 = [f"| {n} | {f3(sh(t, n, gp, 'A'))} | {f3(sh(t, n, gp, 'B'))} | {f3(sh(t, n, gp, 'Bp'))} | {margin(ct, n, gp)} |" for n in ct["ns"]]
    never = ""
    if nt and nt["n_targets"]:
        rows3 = [f"| {p:,} | {f3(sh({'cells': nt['trained_cells']}, 8, p, 'B'))} | {f3(sh(nt, 8, p, 'B'))} | {f3(sh(nt, 8, p, 'Bp'))} |" for p in ps]
        never = (f"\nTargets whose directory holds at least two other files in their own split ({nt['n_targets']} on test, {nt['n_scored_bytes']:,} bytes): the same bytes "
                 f"against related files the weights trained on and against related files they never saw (n = 8):\n\n| P | B, trained files | B, never-trained files | B' (random, same size as the never-trained prefix) |\n|---:|---:|---:|---:|\n" + "\n".join(rows3) + "\n")
    claim = (f"**Claim:** on the {ct['n_targets']} test targets at n = 8, a {gp:,}-byte related prefix holds the exact n-gram and next byte for {pct(sh(t, 8, gp, 'B'))} of the scored bytes "
             f"that the target itself cannot supply, against {pct(sh(t, 8, gp, 'Bp'))} for an equal-size random prefix from other directories ({pct(sh(t, 8, gp, 'A'))} are self-copyable, "
             f"{pct(sh(t, 8, gp, 'C'))} match nowhere); B at P = {', '.join(f'{p:,}' for p in ps)} is {', '.join(pct(sh(t, 8, p, 'B')) for p in ps)}, B' {', '.join(pct(sh(t, 8, p, 'Bp')) for p in ps)}. "
             f"The related margin B - B' at P = {gp:,} is {margin(ct, 8, gp)} at n = 8, {margin(ct, 12, gp)} at n = 12 and {margin(ct, 16, gp)} at n = 16 (95 % intervals over targets; dev, n = 8: {margin(cd, 8, gp)}). "
             + (f"For never-trained related files ({nt['n_targets']} targets) B is {pct(sh(nt, 8, gp, 'B'))} against {pct(sh({'cells': nt['trained_cells']}, 8, gp, 'B'))} for trained files on the same targets. " if never else "")
             + "**Not a claim:** that a model can turn these matches into lower loss (Run 51 measures that); which related files to read (path order is arbitrary and deterministic); "
               "anything outside kernel C and header files; the dev numbers, which are shown only for consistency.")
    return (f"## Run 50 — what could related files share? Exact-match census of the held-out kernel files (`scripts/census.py`, `results/census_linux_{{dev,test}}.json`) — {DATE}, CPU\n\n"
            f"Every long-context number in Runs 40–49 crossed into unrelated files, because the corpus is a file shuffle. This run asks what *related* files could share. Targets are the held-out files "
            f"past byte 202.1M (split by file order: dev = the first 500, test = the other 499) with at least four same-directory files entirely inside the training bytes: {cd['n_targets']} on dev, "
            f"{ct['n_targets']} on test (median {rel_files:g} related files each). The scored bytes are the first 4,096 body bytes of each target (the whole body if shorter): {ct['n_scored_bytes']:,} on test. "
            f"Each scored byte is binned by exact n-gram match: **A** continues an n-gram seen earlier in the target itself (header included); **B** is not A but the n-gram and its continuation occur in the "
            f"related prefix (the target's same-directory training files, path order, headers kept, cut at P bytes); **B'** is the same test against an equal-size random prefix from other directories "
            f"(seeded; never the target's directory, never a file with the target's content); **C** is neither. Shares are byte-weighted over targets; the margin carries a 95 % interval from resampling "
            f"the targets (files). Test, n = 8:\n\n"
            f"| P | A | B | B' | C | B - B' (95 % interval) |\n|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows1) + f"\n\nThe same at P = {gp:,} for the other match lengths:\n\n"
            f"| n | A | B | B' | B - B' (95 % interval) |\n|---:|---:|---:|---:|---:|\n" + "\n".join(rows2) + "\n" + never + "\n" + claim + "\n")


def _ctx(model: str, split: str = "test") -> dict:
    return load(f"commons_ctx_{model}_{split}")


def run51() -> str:
    models = ("x64k", "local", "m50", "tf")
    c = {m: _ctx(m) for m in models}
    x, xd = c["x64k"], _ctx("x64k", "dev")
    xa = x["all"]
    gp = str(x["gate_p"])
    alone = lambda m, scope: c[m]["all"]["bpb"][scope]["alone"]
    rows1, rows2 = [], []
    for m in models:
        nts = c[m]["never_subset"]
        for p in c[m]["ps"]:
            never = "n/a" if not nts else gain(nts, "full", p, "gain_never")
            rows1.append(f"| {LABEL[m]} | {p:,} | {f3(alone(m, 'full'))} | " + " | ".join(gain(c[m]["all"], "full", p, k) for k in ("gain_related", "gain_random", "gain_related_vs_random")) + f" | {never} |")
            rows2.append(f"| {LABEL[m]} | {p:,} | {f3(alone(m, 'first2048'))} | " + " | ".join(gain(c[m]["all"], "first2048", p, k) for k in ("gain_related", "gain_random", "gain_related_vs_random")) + " |")
    bins = xa["bins"]["full"][gp]
    tot = sum(bins["related"]["count"].values())
    brow = lambda name, part, code: f"| {name} | {f3(bins[part]['count'][code] / tot)} | {f3(bins[part]['bpb']['alone'][code])} | {gain(xa, 'full', gp, f'bin_{part}_{code}')} |"
    rows3 = [brow("A", "related", "A"), brow("B", "related", "B"), brow("C", "related", "C"), brow("B'", "random", "B")]
    rows4 = [f"| {p:,} | {f3(xa['headroom']['full'][str(p)]['share_B'])} | {f3(xa['headroom']['full'][str(p)]['share_Bp'])} | " + " | ".join(gain(xa, "full", p, k) for k in ("H_B", "H_Bp", "gate")) + " |" for p in x["ps"]]
    gate, gci = x["gate_number"], x["gate_ci95"]
    straddle = gci is not None and gci[0] < GATE < gci[1]
    verdict = (f"the interval straddles the 0.02 threshold (the point value is {'above' if gate >= GATE else 'below'} it), so this run does not settle the Gate 1 table by itself" if straddle else
               "at or above the 0.02 threshold, which the Gate 1 table reads as Stage 2 (train a brain with a shared memory)" if gate >= GATE else
               "below the 0.02 threshold, which the Gate 1 table reads as exact-match sharing being dead on kernel code, with the kNN fallback probe next")
    loc, m50, tf = c["local"], c["m50"], c["tf"]
    pl, p5, p0 = max(loc["ps"]), max(m50["ps"]), x["ps"][0]
    claim = (f"**Claim:** with {LABEL['x64k']} and no new model code, reading {int(gp):,} bytes of same-directory training files moves bits/byte on the first {x['n_scored_cap']:,} target bytes from {f3(alone('x64k', 'full'))} (alone), "
             f"a gain of {gain(xa, 'full', gp, 'gain_related')} bits/byte (positive = fewer bits; 95 % interval over targets), of which {gain(xa, 'full', gp, 'bin_related_B')} falls on bin B; "
             f"the same number of random kernel bytes gives {gain(xa, 'full', gp, 'gain_random')}, so related minus random is {gain(xa, 'full', gp, 'gain_related_vs_random')}. "
             f"The gate number H(B) - H(B') at n = 8 and P = {int(gp):,} is **{ci(gate, gci)}** bits/byte on test (dev {ci(xd['gate_number'], xd['gate_ci95'])}): {verdict} — the decision is Kyle's. "
             f"Related gains at the other models' trained load: local-attention model {gain(loc['all'], 'full', pl, 'gain_related')} (P = {pl:,}), 50M hybrid {gain(m50['all'], 'full', p5, 'gain_related')} (P = {p5:,}); "
             f"on the first 2,048 bytes the transformer reference (2,048-byte prefix, NTK-scaled) {gain(tf['all'], 'first2048', 2048, 'gain_related')}, against "
             f"{gain(xa, 'first2048', p0, 'gain_related')} for the 65,536-slot model at {p0:,} bytes. **Not a claim:** that the gain comes from the memory heads rather than the attention layer "
             f"(the rows differ in both); that a brain trained to share memory would gain this much (Stage 2); anything about a different file-selection rule or other directories' size; more than one seed per checkpoint.")
    return (f"## Run 51 — does reading related files help with no new model code? (`scripts/probe_commons.py --mode context`, `results/commons_ctx_{{x64k,local,m50,tf}}_{{dev,test}}.json`) — {DATE}, GPU 1\n\n"
            f"Plain `forward`, batch 1, one sequence per condition per test target ({x['n_targets']} targets), scoring only the target's bytes: **alone** (header + target), **related** (its same-directory training files "
            f"with headers in path order, cut at P bytes, then header + target), **random** (an equal-size seeded draw from other directories) and **never trained** (related files from the target's own split; "
            f"{x['n_never_targets']} targets). Heads run at chunk 4096 for the long sequences. Every condition scores the same target bytes, so only the differences are read: gains are alone minus the condition "
            f"(positive = fewer bits/byte) with 95 % intervals from resampling the targets (files). First {x['n_scored_cap']:,} target bytes (the transformer: its first 2,048, prefix 2,048, NTK-scaled):\n\n"
            f"| model | P | alone (bits/byte) | gain related | gain random | related minus random | gain never trained |\n|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows1) +
            "\n\nThe first 2,048 scored bytes of every model, so the rows compare:\n\n| model | P | alone (bits/byte) | gain related | gain random | related minus random |\n|---|---:|---:|---:|---:|---:|\n" + "\n".join(rows2) +
            f"\n\nWhere the gain lands, {LABEL['x64k']}, P = {int(gp):,} (census bins at n = 8; the share-weighted gains of the three bins add up to the total gain; the B' row bins by the random prefix and gains from reading it):\n\n"
            f"| bin | share | alone (bits/byte) | share-weighted gain |\n|---|---:|---:|---:|\n" + "\n".join(rows3) +
            "\n\nHeadroom H(X) = share(X) x (loss(X) - loss(A)) of the alone condition:\n\n| P | share B | share B' | H(B) | H(B') | H(B) - H(B') |\n|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows4) + "\n\n" + claim + "\n")


def run52() -> str:
    d, t = load("commons_handoff_x64k_dev"), load("commons_handoff_x64k_test")
    r = lambda res, key: ci(res["transfer"][key], res["ci95"]["transfer"][key], ".2f")
    base = [f"| {s} | alone | {f3(x['bpb']['alone'])} | | |" for s, x in (("dev", d), ("test", t))]
    base += [f"| {s} | context (same bytes as a plain prefix) | {f3(x['bpb']['context'])} | {ci(x['gain_context'], x['ci95']['gain_context'])} | 1.00 |" for s, x in (("dev", d), ("test", t))]
    rows = [f"| dev | handoff, alpha {a} | {f3(v)} | {ci(d['gain_handoff'][a], d['ci95']['gain_handoff'][a])} | {r(d, a)} |" for a, v in d["bpb"]["handoff"].items()]
    rows += [f"| test | handoff, alpha {a} (chosen on dev) | {f3(v)} | {ci(t['gain_handoff'][a], t['ci95']['gain_handoff'][a])} | {r(t, a)} |" for a, v in t["bpb"]["handoff"].items()]
    seq = [f"| {s} | sequential reader | {f3(x['bpb']['sequential'])} | {ci(x['gain_sequential'], x['ci95']['gain_sequential'])} | {ci(x['transfer_sequential'], x['ci95']['transfer_sequential'], '.2f')} |" for s, x in (("dev", d), ("test", t))]
    heads = [f"| {h['head']} | {h['S_rel']:.1e} | {h['Ksum_rel']:.1e} |" for h in t["head_diff"]]
    a = str(t["best_alpha"])
    lowest = t["head_diff"][1 if len(t["head_diff"]) > 1 else 0]
    ratio = (f"The transfer ratio on test is **{r(t, a)}** (95 % interval over targets; alone {f3(t['bpb']['alone'])}, handoff {f3(t['bpb']['handoff'][a])} for a gain of {ci(t['gain_handoff'][a], t['ci95']['gain_handoff'][a])}, "
             f"context {f3(t['bpb']['context'])} for a gain of {ci(t['gain_context'], t['ci95']['gain_context'])}); the sequential reader's is {ci(t['transfer_sequential'], t['ci95']['transfer_sequential'], '.2f')}"
             if t["transfer_best"] is not None else
             f"The context gain on test is {ci(t['gain_context'], t['ci95']['gain_context'])} bits/byte, below the 0.01 floor, so no transfer ratio is reported "
             f"(handoff gain {ci(t['gain_handoff'][a], t['ci95']['gain_handoff'][a])})")
    claim = (f"**Claim:** {ratio}. Merged reader tables equal the sequential agent's for head {lowest['head']} (relative difference {lowest['S_rel']:.1e}) and differ by {max(h['S_rel'] for h in t['head_diff']):.1e} at most in S across heads, "
             f"as the keys of deeper heads depend on earlier reads. **Not a claim:** that handoff does not work. No checkpoint was trained to start from a loaded memory, and loaded matches dilute the agent's own through the "
             f"normaliser, so this probe can confirm and cannot refute; alpha was chosen on dev ({a}) and test was scored once at it.")
    return (f"## Run 52 — can a group's pooled memory be handed to an agent that did not read the files? (`scripts/probe_commons.py --mode handoff`, `results/commons_handoff_x64k_{{dev,test}}.json`) — {DATE}, GPU 1\n\n"
            f"`m20_linux_depth4x64k_s0` (no local attention) through the stream API on {d['n_targets']} dev and {t['n_targets']} test targets. One fresh reader agent per related file (its first {t['per_file']:,} bytes, header included; "
            f"{t['budget']:,} reader bytes at most per target, {t['readers']['mean_files']:.1f} files on average on test); `merge` their tables; `preload` the sum into a fresh agent at alpha; score the target's first 4,096 bytes. "
            f"A sequential reader (one agent, the same files in order, `new_document()` between files) and the plain-forward context row on exactly the same bytes complete the comparison, so every row scores the same bytes. "
            f"Gains are alone minus the row (positive = fewer bits/byte); the transfer ratio is (alone - handoff) / (alone - context); intervals are 95 % from resampling the targets (files):\n\n"
            f"| split | condition | bits/byte | gain vs alone | transfer ratio |\n|---|---|---:|---:|---:|\n" + "\n".join(base + rows + seq) +
            f"\n\nRelative difference between the merged reader tables and the sequential agent's, per head (`all_heads()` order: the post head first, then the inner heads bottom-up), test:\n\n| head | S | Ksum |\n|---:|---:|---:|\n" + "\n".join(heads) + "\n\n" + claim + "\n")


def main() -> None:
    entries = [run50(), run51(), run52()]
    text = "".join(e + "\n---\n\n" for e in entries)
    if "--insert" not in sys.argv:
        print(text)
        return
    nb = Path("notebook.md")
    s = nb.read_text()
    marker = "## Path to V3 (ladder plumbing)\n"
    assert s.count(marker) == 1 and "## Run 50 " not in s, "notebook.md already has Run 50 or lost its marker"
    nb.write_text(s.replace(marker, text + marker))
    print("inserted Runs 50-52 into notebook.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 17: Write-up: generate the entries from the result files and insert them**

Run:

```bash
cd /games/ailab/lab/flybrain-lab
RUN_DATE=$(date +%F) uv run python /games/ailab/lab/flybrain-lab/.worktrees/probes/gen_entries.py > /games/ailab/lab/flybrain-lab/.worktrees/probes/entries.md
wc -l /games/ailab/lab/flybrain-lab/.worktrees/probes/entries.md
RUN_DATE=$(date +%F) uv run python /games/ailab/lab/flybrain-lab/.worktrees/probes/gen_entries.py --insert
grep -n '^## Run 5[0-2]\|^## Path to V3' notebook.md
```

Expected: a line count of about 100; `inserted Runs 50-52 into notebook.md`; then four lines: the headings of Runs 50, 51 and 52 followed by `## Path to V3 (ladder plumbing)`. A `KeyError` or a missing file means a result JSON is absent (run the missing part first). With part C not done, run only after the handoff results exist: the generator needs all of them.

- [ ] **Step 18: Write-up: read the inserted entries and fix any sentence its own table contradicts**

Open the three entries in `notebook.md` and read them against their tables. Every number was written from a JSON, so edit prose only: a gain printed with a minus sign under the word "lowers", a `n/a` in a sentence, a "below the threshold" that a table contradicts. Do not touch a table cell, do not add a claim the tables do not show, and keep the **Not a claim** sentences. Do not edit `README.md` or `notes/RESULTS.md`: the proposed wording changes for Gate 1 belong in the report to Kyle.

- [ ] **Step 19: Write-up: commit the notebook**

Run:

```bash
cd /games/ailab/lab/flybrain-lab
git add notebook.md
git commit -m "Notebook: Runs 50-52, census, zero-shot related reading and handoff on the held-out kernel files (Stage 1.2-1.4)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git status --short
```

Expected: one commit; `git status --short` prints nothing but untracked files you created on purpose (none). Do not push.

- [ ] **Step 20: Report to Kyle and clean up**

Run:

```bash
cd /games/ailab/lab/flybrain-lab
git worktree remove --force /games/ailab/lab/flybrain-lab/.worktrees/probes
```

Before removing the worktree, send Kyle the Gate 1 numbers from Run 51 (gate number H(B) - H(B') at n = 8, P = 32,768 on test with its interval; dev for consistency), the census shares from Run 50, the transfer ratio from Run 52, and the sentence the spec's Gate 1 table attaches to the gate number (0.02 or more: Stage 2; less: run the kNN fallback probe). The run logs are `/games/ailab/lab/flybrain-lab/.worktrees/probes/results/*.log`; copy any you want to keep before the removal. The removal deletes only the throwaway worktree.

---

## Self-review

Spec requirement (sections 1.1-1.4) and the task that implements it:

- 1.1 `file_table` from the header lines (`FileEntry` also carries `body`, decision 1): Task 1. `related(i, table, lo, hi)` with the range, self and content-hash rules: Task 1 (`hashes` is passed in, built by `hash_table`). `read_region` as a seek and read: Task 1. `fixed_windows`: not in this plan (the yardstick plan adds it to `flybrain/data/text.py`).
- 1.1 dev = first 500 tail files from byte 202,117,357, test = the other 499 from 211,576,468; targets with at least four same-directory training files (353 / 339); scored bytes = the first 4,096 after the header: Task 1 (`splits`, `targets`, `read_stream`, and the real-corpus check step that prints those counts).
- 1.2 A / B / B' / C at n in {8, 12, 16}: Task 3 (`bins`), Task 4 (`run_census`). The related set cut at P in {8k, 16k, 32k} and the equal-size seeded random set: Task 2 (`prefix_pairs`), used by Tasks 4 and 7. The never-trained second pass for targets with at least two other files in their split: Task 1 (`MIN_NEVER`, `split_range`), Task 2 (`lo`, `hi`), Task 4 (`--never-trained`).
- 1.2 per-byte alone loss per bin (share and mean loss), headroom H(X), the gate number at n = 8 and P = 32k: Task 6 (`fill`, `summarize`, `headroom`, `effects_of`, `bootstrap_effects`), Task 7 (`gate_number`, `gate_ci95`, `render_context`).
- 1.3 plain forward, one sequence per condition (alone, related, random, never trained) scoring the target bytes only: Task 5 (`byte_bits`, `score_with_prefix`), Task 7 (`probe_target`). P by model, `--chunk 4096` for the 65,536-slot model, the transformer's 2,048 + 2,048 with NTK scaling: Task 5 (`apply_chunk`, `arm_settings`, per-sequence `set_rope_scale`), Task 7 (`main`), Task 10 (commands). The first-2,048 subset for every model: Task 6 (`first2048` scope). Bits/byte per condition and per census bin: Tasks 6-7.
- 1.4 one reader agent per related file, at most 8,192 bytes, fresh state, `flybrain.group.read`: Task 2 (`reader_chunks`), Task 8 (`handoff_target`). `merge`, `preload` at alpha in {0.03, 0.1, 0.3, 1.0}, alpha chosen on dev and passed as `--alpha` for test: Task 8 (`best_alpha`), Task 9 (`--alpha`), Task 10 (handoff queue). The sequential reader with `new_document()` and the per-head difference of the tables: Task 8 (`head_diff`), Task 9 (`head_diff` in the JSON). Transfer ratio when the denominator is at least 0.01: Task 8 (`transfer_ratio`), Task 9. Reported on test: Task 10.
- Revised spec, "Comparisons are paired" and Gate 1: every condition scores the same target bytes: Tasks 5, 7, 8, 9. A 95 % interval from resampling targets on the gate number and on every reported gain: Task 6 (`effects_of`, `bootstrap_effects`), Task 8 (`bootstrap_handoff`), Task 9 (`ci95` in the handoff JSON), Task 10 (census margin interval, intervals in every table). The statement when the gate interval straddles 0.02: Task 7 (`render_context`) and Task 10 (the gate sentence of Run 51).
- Leakage rules (each with a test): target and duplicates never in a prefix or reader set: Task 2; random prefixes contain no same-directory file: Task 2; an empty prefix gives the alone numbers: Task 7; handoff at alpha = 0 gives the alone numbers: Tasks 8 and 9; determinism for a fixed seed: Tasks 4, 7 and 9.
- Spec testing bullets: `file_table` round-trips a built corpus, `related` never returns the target or a duplicate: Task 1; census bins on a hand-built example: Task 3; handoff at alpha = 0 equals alone, an empty prefix equals alone: Tasks 8 and 7.
- Notebook write-ups of the census, the related-reading probe and the handoff (Runs 50, 51, 52) with Claim / Not a claim: Task 10.
