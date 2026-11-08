# Group Learning A: Agent Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing fly-brain model promptable and give it agent state: a streaming entry point that matches the full-window forward, and a group runtime that pools memory tables and hands them to another agent.

**Architecture:** `forward` and every default path stay bit-identical. Streaming is a second entry point, `GRUHybrid.stream`, that threads an explicit `AgentState` through the three stateful parts of the model: the minGRU carry, the memory-head tables, and the local-attention cache. `flybrain/group.py` builds only on `stream`. `scripts/sample.py` is the first user-facing consumer.

**Tech Stack:** Python 3.11, PyTorch, pytest, uv. All tests run on CPU.

**Spec:** `docs/superpowers/specs/2026-10-03-group-learning-design.md` (Stage 0). Plans B (probes) and C (yardstick) are separate files; B's handoff probe consumes the interfaces this plan produces.

## Global Constraints

- Branch `feat/group-learning`. Never push. Commit with the repo's configured identity; end each commit message with the Co-Authored-By trailer your session specifies.
- `forward` and every existing default code path stay bit-identical: new behaviour lives in new functions or behind a `state=None` default.
- "Exact" means within the tolerances in the tests (1e-5 for one component, 1e-4 for the full hybrid), not bitwise: fp64 scan chunks align to the segment start and fp32 sums reorder.
- Streaming supports `backbone="mingru"`, `where` in `("post", "none")`, `mem_every`, `local_attn`, and heads with `chunk > 0`. Anything else raises `ValueError`.
- `GRUHybrid.stream` and `Block.memory(state=...)` never modify the state passed in.
- With `state_bits > 0`, streaming is exact only for segments that are multiples of `chunk`.
- Tests are CPU-only and need no corpus, no checkpoint and no network. Run them with `uv run pytest -q`.
- Manual steps that load a checkpoint use `--device cpu`. This plan runs no GPU job.
- Do not edit `README.md` or `notes/RESULTS.md`.

## Review Focus

1. **One-byte segments** (generation feeds one byte per call): every equivalence test in Tasks 3–6 includes a byte-by-byte segmentation.
2. **A stream longer than the rotary rebase period**: Task 4 repeats its test with `rebase = 16`, so positions wrap mid-stream.
3. **A state built for a different number of agents than the tokens**: `stream` raises `ValueError` (Task 6, `test_stream_refuses_what_it_cannot_do`).
4. **An empty prompt, and sampled bytes that are not valid UTF-8**: an empty prompt starts from a newline; output is decoded incrementally with replacement, so a split multi-byte character prints once (Task 1).
5. **A memory file loaded under different weights**: `load_commons` raises `ValueError` (Task 8).

## File Structure

| File | Responsibility |
|---|---|
| `flybrain/model/state.py` (new) | `HeadState`, `AttnCache`, `AgentState`: the data an agent carries. No logic beyond `new_document()`. |
| `flybrain/model/hybrid.py` (modify) | `MinGRULayer.scan`, `LocalAttention.stream`, `GRUHybrid.init_state / stream` |
| `flybrain/model/sparsela.py` (modify) | `sparse_memory_read(..., state=None)`, `Block.memory(..., state=None)` |
| `flybrain/group.py` (new) | `Commons`, `weights_hash`, `read`, `merge`, `preload`, `save_commons`, `load_commons` |
| `scripts/sample.py` (new) | `sample_next`, `generate`, `streamable`, `Utf8Printer`, CLI |
| `tests/test_sample.py`, `tests/test_stream.py`, `tests/test_group.py` (new) | One test file per unit above |

Task order: 1 is independent. 2 comes before 3–5. 3 and 4 both edit `hybrid.py`, so run them one after the other. 6 needs 3, 4 and 5. 7 needs 1 and 6. 8 needs 6.

---

### Task 1: Recompute sampler

**Files:**
- Create: `scripts/sample.py`
- Test: `tests/test_sample.py`

**Interfaces:**
- Consumes: `scripts.cross_eval.rebuild(ck: dict)` (builds the model and loads its weights).
- Produces: `sample_next(logits: Tensor, temperature: float, top_k: int, gen: torch.Generator) -> int`; `generate(model, prompt: bytes, n: int, temperature: float = 0.8, top_k: int = 40, seed: int = 0, window: int = 2048, on_byte=None) -> bytes`; `load_model(path: str, device: torch.device)`; `Utf8Printer(write)` with `feed(b: int)` and `close()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sample.py`:

```python
import torch

from flybrain.model.hybrid import GRUHybrid


def _tiny():
    torch.manual_seed(0)
    return GRUHybrid(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4,
                     key_conv=2, normalize=True, gate=True, chunk=8, mem_every=2, local_attn=8, key_convs=(2, 4)).eval()


def test_sample_next_is_greedy_at_zero_temperature_and_respects_top_k():
    from scripts.sample import sample_next
    logits = torch.tensor([0.1, 2.0, 0.3, 1.9] + [-5.0] * 252)
    gen = torch.Generator().manual_seed(0)
    assert sample_next(logits, 0.0, 0, gen) == 1
    assert all(sample_next(logits, 5.0, 1, gen) == 1 for _ in range(20))       # top-1 is the argmax at any temperature
    assert {sample_next(logits, 5.0, 2, gen) for _ in range(200)} == {1, 3}     # top-2 never leaves the two best

    def draw(seed):
        g = torch.Generator().manual_seed(seed)
        return [sample_next(logits, 1.0, 0, g) for _ in range(8)]

    assert draw(7) == draw(7)


def test_generate_recompute_matches_a_manual_greedy_loop_and_is_deterministic():
    from scripts.sample import generate
    m = _tiny()
    prompt = b"int main(void) {"
    out = generate(m, prompt, 12, temperature=0.0)
    buf = list(prompt)
    for _ in range(12):
        with torch.no_grad():
            logits, _ = m(torch.tensor([buf]))
        buf.append(int(logits[0, -1].argmax()))
    assert out == bytes(buf[len(prompt):]) and len(out) == 12
    assert generate(m, prompt, 12, temperature=0.9, seed=3) == generate(m, prompt, 12, temperature=0.9, seed=3)


def test_generate_uses_only_the_last_window_bytes_and_starts_an_empty_prompt_from_a_newline():
    from scripts.sample import generate
    m = _tiny()
    prompt = b"abcdefghij"
    out = generate(m, prompt, 6, temperature=0.0, window=4)
    buf = list(prompt)
    for _ in range(6):
        with torch.no_grad():
            logits, _ = m(torch.tensor([buf[-4:]]))
        buf.append(int(logits[0, -1].argmax()))
    assert out == bytes(buf[len(prompt):])
    seen = []
    assert generate(m, b"", 5, temperature=0.0, on_byte=seen.append) == generate(m, b"\n", 5, temperature=0.0)
    assert len(seen) == 5


def test_utf8_printer_prints_split_characters_once_and_replaces_invalid_bytes():
    from scripts.sample import Utf8Printer
    got = []
    p = Utf8Printer(got.append)
    for b in "é".encode("utf-8"):      # two bytes, one character
        p.feed(b)
    p.feed(0xFF)                        # never valid in UTF-8
    p.feed(ord("x"))
    p.close()
    assert "".join(got) == "é�x"
    got.clear()
    p = Utf8Printer(got.append)
    p.feed(0xC3)                        # a character cut off at the end
    p.close()
    assert "".join(got) == "�"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sample.py -q`
Expected: 4 failures with `ModuleNotFoundError: No module named 'scripts.sample'`

- [ ] **Step 3: Write the implementation**

Create `scripts/sample.py`:

```python
#!/usr/bin/env python3
"""Prompt a saved checkpoint: bytes in, sampled bytes out.

The recompute path runs the plain forward over the last --window bytes for every new byte. It works for any arm
and needs nothing from the model beyond forward()."""

from __future__ import annotations

import argparse
import codecs
import sys
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cross_eval import rebuild


def sample_next(logits: Tensor, temperature: float, top_k: int, gen: torch.Generator) -> int:
    """One byte from next-byte logits [256]. temperature <= 0 is greedy; top_k > 0 keeps the k most likely bytes."""
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if 0 < top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen))


@torch.no_grad()
def generate(model, prompt: bytes, n: int, temperature: float = 0.8, top_k: int = 40, seed: int = 0, window: int = 2048, on_byte=None) -> bytes:
    """Sample n bytes after `prompt` (an empty prompt starts from a newline). `on_byte` is called with each byte as it is drawn."""
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    buf = list(prompt) or [10]
    out = []
    for _ in range(n):
        logits, _ = model(torch.tensor([buf[-window:]], device=device))
        b = sample_next(logits[0, -1].float().cpu(), temperature, top_k, gen)
        buf.append(b); out.append(b)
        if on_byte is not None:
            on_byte(b)
    return bytes(out)


class Utf8Printer:
    """Writes sampled bytes as text as they arrive: a multi-byte character prints once it is complete,
    and a byte that cannot be part of UTF-8 prints as U+FFFD."""

    def __init__(self, write):
        self._write = write
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, b: int) -> None:
        s = self._dec.decode(bytes([b]))
        if s:
            self._write(s)

    def close(self) -> None:
        s = self._dec.decode(b"", final=True)
        if s:
            self._write(s)


def load_model(path: str, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return rebuild(ck).to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--prompt", default="")
    src.add_argument("--prompt-file", default="")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    prompt = Path(args.prompt_file).read_bytes() if args.prompt_file else args.prompt.encode("utf-8")
    model = load_model(args.ckpt, torch.device(args.device))

    def write(s: str) -> None:
        sys.stdout.write(s); sys.stdout.flush()

    write(prompt.decode("utf-8", errors="replace"))
    printer = Utf8Printer(write)
    generate(model, prompt, args.n, args.temperature, args.top_k, args.seed, args.window, on_byte=printer.feed)
    printer.close()
    write("\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sample.py -q`
Expected: 4 passed

- [ ] **Step 5: Prompt the real 50M checkpoint on CPU**

Run: `uv run python scripts/sample.py --ckpt results/ckpt/m50_linux_hybrid_s0_mingru_sparse_post.pt --prompt 'static int ' --n 120 --window 256 --device cpu`
Expected: the prompt followed by 120 bytes of kernel-style C, appearing a byte at a time. On CPU this takes a minute or two (an estimate). If the checkpoint file is missing, stop and report it; do not train one.

- [ ] **Step 6: Commit**

```bash
git add scripts/sample.py tests/test_sample.py
git commit -m "Sampler: prompt a saved checkpoint by recomputing the last window per byte"
```

---

### Task 2: Agent state

**Files:**
- Create: `flybrain/model/state.py`
- Test: `tests/test_stream.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `HeadState(S, Ksum, last_idx=None, last_val=None, conv_tail=None)`; `AttnCache(k=None, v=None, offset=0)`; `AgentState(h, heads, attn)` with `.batch -> int | None` and `.new_document() -> AgentState`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stream.py`:

```python
import torch


def test_new_document_keeps_tables_and_drops_everything_else():
    from flybrain.model.state import AgentState, AttnCache, HeadState
    S, K = torch.randn(2, 8, 4), torch.rand(2, 8)
    st = AgentState(
        h=[torch.randn(2, 4, dtype=torch.float64), torch.randn(2, 4, dtype=torch.float64)],
        heads=[HeadState(S, K, torch.zeros(2, 3, dtype=torch.long), torch.rand(2, 3), torch.randn(2, 1, 4)), HeadState(S.clone(), K.clone())],
        attn=[AttnCache(torch.randn(2, 1, 5, 4), torch.randn(2, 1, 5, 4), 37)],
    )
    assert st.batch == 2
    nd = st.new_document()
    assert nd.h == [None, None]
    assert torch.equal(nd.heads[0].S, S) and torch.equal(nd.heads[0].Ksum, K)
    assert nd.heads[0].last_idx is None and nd.heads[0].last_val is None
    assert torch.equal(nd.heads[0].conv_tail, torch.zeros(2, 1, 4)) and nd.heads[1].conv_tail is None
    assert nd.attn[0].k is None and nd.attn[0].v is None and nd.attn[0].offset == 0
    assert st.heads[0].last_idx is not None and st.attn[0].offset == 37      # the original is untouched
    assert AgentState(h=[None], heads=[], attn=[]).batch is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_stream.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flybrain.model.state'`

- [ ] **Step 3: Write the implementation**

Create `flybrain/model/state.py`:

```python
"""Agent state: everything a GRUHybrid needs to continue a byte stream where it stopped."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass
class HeadState:
    """One memory head. S and Ksum are what the head has learned; the rest is what the next byte needs."""

    S: Tensor                         # [B, n_key, d] fast-weight table
    Ksum: Tensor                      # [B, n_key] key mass per slot
    last_idx: Tensor | None = None    # [B, k] address of the previous position (None before the first byte of a document)
    last_val: Tensor | None = None    # [B, k] its unit-normalised values
    conv_tail: Tensor | None = None   # [B, key_conv - 1, d] inputs the causal key conv still needs (None when key_conv == 1)


@dataclass
class AttnCache:
    """One local-attention layer: unrotated keys and values of the last n <= window - 1 positions."""

    k: Tensor | None = None           # [B, H, n, dh]
    v: Tensor | None = None           # [B, H, n, dh]
    offset: int = 0                   # absolute position of the next byte


@dataclass
class AgentState:
    h: list                           # per minGRU layer: [B, d] float64 carry, or None before the first byte
    heads: list                       # HeadState per head, in GRUHybrid.all_heads() order (post head first, then inner heads bottom-up)
    attn: list                        # AttnCache per local-attention layer, in GRUHybrid.local_at order

    @property
    def batch(self) -> int | None:
        """Number of agents this state belongs to (None for a fresh state of a model without memory heads)."""
        for head in self.heads:
            return int(head.S.size(0))
        for carry in self.h:
            if carry is not None:
                return int(carry.size(0))
        return None

    def new_document(self) -> "AgentState":
        """Start a new document: keep what was learned (S, Ksum), drop everything tied to the previous bytes."""
        heads = [replace(h, last_idx=None, last_val=None, conv_tail=None if h.conv_tail is None else torch.zeros_like(h.conv_tail))
                 for h in self.heads]
        return AgentState(h=[None] * len(self.h), heads=heads, attn=[AttnCache() for _ in self.attn])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_stream.py -q`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add flybrain/model/state.py tests/test_stream.py
git commit -m "Agent state: tables, carry and attention cache as explicit data"
```

---

### Task 3: minGRU scan with a carry

**Files:**
- Modify: `flybrain/model/hybrid.py` (class `MinGRULayer`: replace `forward`)
- Test: `tests/test_stream.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `MinGRULayer.scan(x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]`: the layer output `[B, T, d]` and the recurrent state after the last position, `[B, d]` float64. `MinGRULayer.forward(x)` returns `scan(x)[0]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stream.py`:

```python
def test_mingru_scan_carry_matches_one_pass():
    from flybrain.model.hybrid import MinGRULayer
    torch.manual_seed(0)
    layer = MinGRULayer(12, 24).eval()
    x = torch.randn(2, 70, 12)
    with torch.no_grad():
        ref = layer(x)
        for cut in (1, 31, 32, 33):                      # around the 32-position scan chunk
            a, h = layer.scan(x[:, :cut])
            b, _ = layer.scan(x[:, cut:], h)
            assert torch.allclose(torch.cat([a, b], 1), ref, atol=1e-5), cut
            assert h.dtype == torch.float64 and h.shape == (2, 12)
        h = None; outs = []
        for t in range(70):                              # byte by byte
            y, h = layer.scan(x[:, t : t + 1], h)
            outs.append(y)
        assert torch.allclose(torch.cat(outs, 1), ref, atol=1e-5)
        full, h_full = layer.scan(x)
        assert torch.equal(full, ref)                    # forward is scan with no carry
        assert torch.allclose(h, h_full, atol=1e-8)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_stream.py::test_mingru_scan_carry_matches_one_pass -q`
Expected: FAIL with `AttributeError: 'MinGRULayer' object has no attribute 'scan'`

- [ ] **Step 3: Write the implementation**

In `flybrain/model/hybrid.py`, replace the whole `forward` method of `MinGRULayer` (from `def forward(self, x: Tensor) -> Tensor:` through `return x + self.ff(self.ln2(x))`) with:

```python
    def scan(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """The layer with an explicit carry. h0 [B, d] is the recurrent state before x[:, 0] (None = zeros).
        Returns (layer output, recurrent state after the last position, float64)."""
        u = self.ln1(x)
        z = torch.sigmoid(self.z(u)); cand = self.h(u)
        a = (1 - z).clamp_min(1e-4); b = z * cand
        # chunked closed form, all chunks at once: within a chunk h_t = A_t (h_0 + sum_{s<=t} b_s / A_s), A_t = prod a;
        # the carry h_0 of each chunk is a short sequential loop over [B, d] tensors only (no per-chunk slicing of big tensors).
        B, T, d = u.shape; L = self.L
        pad = (-T) % L
        if pad:
            a = F.pad(a, (0, 0, 0, pad), value=1.0); b = F.pad(b, (0, 0, 0, pad))
        nC = (T + pad) // L
        a = a.reshape(B, nC, L, d).double(); b = b.reshape(B, nC, L, d).double()
        A = torch.exp(torch.cumsum(torch.log(a), dim=2))          # [B, nC, L, d]
        local = A * torch.cumsum(b / A, dim=2)                     # chunk output with h_0 = 0
        A_end, local_end = A[:, :, -1], local[:, :, -1]            # [B, nC, d]
        carries = [torch.zeros(B, d, dtype=torch.float64, device=u.device) if h0 is None else h0.to(torch.float64)]
        for i in range(nC - 1):
            carries.append(A_end[:, i] * carries[-1] + local_end[:, i])
        h0s = torch.stack(carries, dim=1)                          # [B, nC, d]
        h = (A * h0s[:, :, None, :] + local).reshape(B, nC * L, d)[:, :T]
        x = x + h.to(x.dtype)
        return x + self.ff(self.ln2(x)), h[:, -1]

    def forward(self, x: Tensor) -> Tensor:
        return self.scan(x)[0]
```

The only behavioural change is the first carry: `h0` when given, zeros otherwise. The local name `h0` of the old body is renamed `h0s` because `h0` is now the argument.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_stream.py tests/test_sparsela.py -q`
Expected: all pass, including the existing `test_mingru_chunked_scan_matches_sequential_loop`

- [ ] **Step 5: Commit**

```bash
git add flybrain/model/hybrid.py tests/test_stream.py
git commit -m "minGRU: scan with an explicit carry; forward is scan with none"
```

---

### Task 4: Local attention with a cache

**Files:**
- Modify: `flybrain/model/hybrid.py` (class `LocalAttention`: add a class attribute and two methods; imports)
- Test: `tests/test_stream.py`

**Interfaces:**
- Consumes: `AttnCache` from Task 2; the existing module-level `rope_at(x, pos)`.
- Produces: `LocalAttention.rebase: int = 65536`; `LocalAttention.stream(x: Tensor, cache: AttnCache) -> tuple[Tensor, AttnCache]`, the same numbers as `forward` / `reference` on the whole stream.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stream.py`:

```python
def test_local_attention_stream_matches_reference_across_segmentations():
    from flybrain.model.hybrid import LocalAttention
    from flybrain.model.state import AttnCache
    torch.manual_seed(0)
    la = LocalAttention(16, n_heads=2, window=8).eval()
    x = torch.randn(2, 45, 16)
    with torch.no_grad():
        ref = la.reference(x)
        for rebase in (65536, 16):                        # 16: positions wrap mid-stream; only relative offsets may matter
            la.rebase = rebase
            for seg in (1, 3, 8, 9, 45):
                cache = AttnCache(); outs = []
                for c0 in range(0, 45, seg):
                    y, cache = la.stream(x[:, c0 : c0 + seg], cache)
                    outs.append(y)
                out = torch.cat(outs, 1)
                assert torch.allclose(out, ref, atol=1e-5), (rebase, seg, float((out - ref).abs().max()))
                assert cache.offset == 45 and cache.k.size(2) == 7
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_stream.py::test_local_attention_stream_matches_reference_across_segmentations -q`
Expected: FAIL with `AttributeError: 'LocalAttention' object has no attribute 'stream'`

- [ ] **Step 3: Write the implementation**

In `flybrain/model/hybrid.py`, add this import below the existing `from flybrain.model.sparsela import ...` line:

```python
from flybrain.model.state import AttnCache
```

In class `LocalAttention`, insert the following between the end of `forward` and `def reference`:

```python
    rebase = 65536  # rotary positions restart from the cached block every `rebase` bytes, so float32 angles stay exact on long streams

    def _stream_block(self, x: Tensor, cache: AttnCache) -> tuple[Tensor, AttnCache]:
        """At most `window` new positions against the cached window - 1."""
        B, T, d = x.shape; w = self.w
        q, k, v = self.qkv(self.ln(x)).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)   # unrotated, [B, H, T, dh]
        n = 0 if cache.k is None else cache.k.size(2)
        k_all = k if n == 0 else torch.cat([cache.k.to(k.dtype), k], dim=2)
        v_all = v if n == 0 else torch.cat([cache.v.to(v.dtype), v], dim=2)
        pos = (cache.offset - n) % self.rebase + torch.arange(n + T, device=x.device)
        i = torch.arange(n, n + T, device=x.device)[:, None]; j = torch.arange(n + T, device=x.device)[None, :]
        mask = (j <= i) & (j > i - w)                                                            # t - s in [0, w - 1]
        y = F.scaled_dot_product_attention(rope_at(q, pos[n:]), rope_at(k_all, pos), v_all, attn_mask=mask)
        keep = min(w - 1, n + T)
        new = AttnCache(k=k_all[:, :, n + T - keep :], v=v_all[:, :, n + T - keep :], offset=cache.offset + T)
        return self.out(y.transpose(1, 2).reshape(B, T, d)), new

    def stream(self, x: Tensor, cache: AttnCache) -> tuple[Tensor, AttnCache]:
        """forward() continued across calls: the same numbers as one pass over the whole stream."""
        outs = []
        for c0 in range(0, x.size(1), self.w):
            y, cache = self._stream_block(x[:, c0 : c0 + self.w], cache)
            outs.append(y)
        return torch.cat(outs, dim=1), cache
```

Why it is exact: `forward` lets position t see positions s with t − s in [0, w − 1]; the mask here is the same band over the cached and the new positions. Rotary scores depend only on t − s, so starting the position count anywhere gives the same scores (the repo's `test_rope_scores_depend_only_on_relative_position` pins that).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_stream.py tests/test_sparsela.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add flybrain/model/hybrid.py tests/test_stream.py
git commit -m "Local attention: stream() with a window-1 cache and rebased rotary positions"
```

---

### Task 5: Memory head with carried tables

**Files:**
- Modify: `flybrain/model/sparsela.py` (`sparse_memory_read`; `Block.memory`, `Block._one_head_chunked`, `Block._one_head`, `Block._one_head_fp32`; imports)
- Test: `tests/test_stream.py`

**Interfaces:**
- Consumes: `HeadState` from Task 2.
- Produces: `sparse_memory_read(code, xk, v, chunk, decay, normalize, gate_tau, state_bits=0, state=None)`: returns `Tensor` when `state is None` (unchanged), else `(Tensor, HeadState)`. `Block.memory(x, key_src=None, state=None)`: returns `Tensor` when `state is None` (unchanged), else `(Tensor, HeadState)`. Neither modifies the state passed in.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stream.py`:

```python
def _fresh_head_state(b, B):
    from flybrain.model.state import HeadState
    d = b.val.in_features
    tail = torch.zeros(B, b.key_conv - 1, d) if b.key_conv > 1 else None
    return HeadState(S=torch.zeros(B, b.code.n_key, d), Ksum=torch.zeros(B, b.code.n_key), conv_tail=tail)


def test_head_with_state_matches_the_parallel_form_across_segmentations():
    from flybrain.model.sparsela import Block, KeyCode
    for decay in (1.0, 0.9):
        for gate in (False, True):
            for key_conv in (2, 4):
                torch.manual_seed(0)
                b = Block(24, KeyCode("sparse", 24, 96, 4, 4, 1.0, 0), decay, d_ff=1, key_conv=key_conv, normalize=True, gate=gate)
                if gate:
                    with torch.no_grad():
                        b.gate_tau.fill_(0.3)
                x = torch.randn(3, 37, 24)
                with torch.no_grad():
                    ref = b.memory(x)                    # chunk 0: the T x T parallel form
                    b.chunk = 8
                    for cuts in ([37], [1, 36], [8, 8, 8, 13], [5, 11, 21], [1] * 37):
                        st = _fresh_head_state(b, 3); outs = []; c0 = 0
                        for n in cuts:
                            y, st = b.memory(x[:, c0 : c0 + n], state=st)
                            outs.append(y); c0 += n
                        out = torch.cat(outs, 1)
                        assert torch.allclose(out, ref, atol=1e-5), (decay, gate, key_conv, cuts, float((out - ref).abs().max()))


def test_head_state_is_not_mutated_and_the_stateless_call_is_unchanged():
    import pytest
    from flybrain.model.sparsela import Block, KeyCode
    torch.manual_seed(0)
    b = Block(24, KeyCode("sparse", 24, 96, 4, 4, 1.0, 0), 1.0, d_ff=1, key_conv=2, normalize=True, gate=True)
    x = torch.randn(2, 20, 24)
    with torch.no_grad():
        b.chunk = 8
        st = _fresh_head_state(b, 2); S0 = st.S.clone()
        y, st2 = b.memory(x, state=st)
        assert torch.equal(st.S, S0) and st.last_idx is None           # the state passed in is untouched
        assert st2.last_idx.shape == (2, 4) and float(st2.Ksum.sum()) > 0 and st2.conv_tail.shape == (2, 1, 24)
        assert torch.allclose(y, b.memory(x), atol=1e-6)               # a fresh state is the stateless call
        assert isinstance(b.memory(x), torch.Tensor)                   # no state: same return type as before
        b.chunk = 0
        with pytest.raises(ValueError):                                # the parallel form has no tables to carry
            b.memory(x, state=_fresh_head_state(b, 2))


def test_quantised_state_survives_single_byte_segments():
    from flybrain.model.sparsela import Block, KeyCode, fake_quant
    torch.manual_seed(0)
    b = Block(24, KeyCode("sparse", 24, 96, 4, 4, 1.0, 0), 1.0, d_ff=1, key_conv=2, normalize=True, gate=True)
    b.chunk = 8; b.state_bits = 16
    x = torch.randn(1, 12, 24)
    with torch.no_grad():
        st = _fresh_head_state(b, 1)
        for t in range(12):
            y, st = b.memory(x[:, t : t + 1], state=st)
        assert torch.isfinite(y).all()
        assert torch.allclose(st.S, fake_quant(st.S.reshape(-1, 24), 16).view_as(st.S), atol=1e-6)   # what is carried is on the 16-bit grid
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_stream.py -q -k "head"`
Expected: 3 failures with `TypeError: Block.memory() got an unexpected keyword argument 'state'`

- [ ] **Step 3: Write the implementation**

In `flybrain/model/sparsela.py`:

**3a.** Add this import below `from flybrain.model.seqmixer import signed_expand_matrix`:

```python
from flybrain.model.state import HeadState
```

**3b.** In `sparse_memory_read`, replace the signature and docstring

```python
def sparse_memory_read(code: KeyCode, xk: Tensor, v: Tensor, chunk: int, decay: float, normalize: bool, gate_tau: Tensor | None, state_bits: int = 0) -> Tensor:
    """Linear-cost index form of the parallel path below, exact for scalar decay in (0, 1] + nonnegative sparse keys.
    out_t = g(m_t) * q_t . S_t / (m_t + 1e-3),  S_t = sum_{1<=s<t} k_{s-1} v_s^T,  m_t = q_t . sum_{1<=s<t} k_{s-1}.
    The key has only top_k nonzeros, so reads are gathers and writes are index-adds; the T x T term is chunk-local."""
```

with

```python
def sparse_memory_read(code: KeyCode, xk: Tensor, v: Tensor, chunk: int, decay: float, normalize: bool, gate_tau: Tensor | None, state_bits: int = 0, state: HeadState | None = None):
    """Linear-cost index form of the parallel path below, exact for scalar decay in (0, 1] + nonnegative sparse keys.
    out_t = g(m_t) * q_t . S_t / (m_t + 1e-3),  S_t = sum_{1<=s<t} k_{s-1} v_s^T,  m_t = q_t . sum_{1<=s<t} k_{s-1}.
    The key has only top_k nonzeros, so reads are gathers and writes are index-adds; the T x T term is chunk-local.
    With `state` the tables and the previous position's key continue from an earlier call, and the return value is
    (out, new HeadState); the state passed in is not modified. S is carried aged to the first position of the next call."""
```

**3c.** In the same function, replace

```python
    # key_{s-1} at slot s (slot 0 reads nothing)
    idx_s = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
    val_s = torch.cat([torch.zeros_like(val[:, :1]), val[:, :-1]], dim=1)
    S = xk.new_zeros(B * n_key, d); Ksum = xk.new_zeros(B * n_key)
```

with

```python
    # key_{s-1} at slot s (slot 0 reads nothing, unless an earlier call left its last key in `state`)
    if state is not None and state.last_idx is not None:
        idx_s = torch.cat([state.last_idx[:, None], idx[:, :-1]], dim=1)
        val_s = torch.cat([state.last_val[:, None].to(val.dtype), val[:, :-1]], dim=1)
    else:
        idx_s = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        val_s = torch.cat([torch.zeros_like(val[:, :1]), val[:, :-1]], dim=1)
    if state is None:
        S = xk.new_zeros(B * n_key, d); Ksum = xk.new_zeros(B * n_key)
    else:
        S = state.S.to(xk.dtype).reshape(B * n_key, d); Ksum = state.Ksum.to(xk.dtype).reshape(B * n_key)
```

**3d.** Replace the function's last line `    return torch.cat(outs, dim=1)` with

```python
    out = torch.cat(outs, dim=1)
    if state is None:
        return out
    return out, HeadState(S=S.reshape(B, n_key, d), Ksum=Ksum.reshape(B, n_key), last_idx=idx[:, -1], last_val=val[:, -1], conv_tail=state.conv_tail)
```

The loop between 3c and 3d is not touched. Every update of `S` and `Ksum` inside it is out of place, so the tensors in `state` are never written.

**3e.** In class `Block`, replace the methods `memory`, `_one_head_chunked` and `_one_head`, and the first two lines of `_one_head_fp32`.

Replace

```python
    def memory(self, x: Tensor, key_src: Tensor | None = None) -> Tensor:
        xk = x if key_src is None else key_src
        if self.key_conv > 1:
            xk = self.conv(F.pad(xk.transpose(1, 2), (self.key_conv - 1, 0))).transpose(1, 2)
        v = self.val(x)
        total = 0
        for code, out in zip(self.codes, self.outs):
            total = total + out(self._one_head(code, xk, v))
        return total

    def _one_head_chunked(self, code: KeyCode, xk: Tensor, v: Tensor) -> Tensor:
        return sparse_memory_read(code, xk, v, self.chunk, self.decay_value(), self.normalize, self.gate_tau if self.gate else None, self.state_bits)

    def _one_head(self, code: KeyCode, xk: Tensor, v: Tensor) -> Tensor:
        # the fast-weight sums accumulate in fp32 whatever autocast does to the rest of the model;
        # the state's precision is controlled explicitly by state_bits (fake-quantised between chunks)
        with torch.autocast(device_type=xk.device.type, enabled=False):
            return self._one_head_fp32(code, xk.float(), v.float())

    def _one_head_fp32(self, code: KeyCode, xk: Tensor, v: Tensor) -> Tensor:
        if self.chunk > 0:
            return self._one_head_chunked(code, xk, v)
```

with

```python
    def memory(self, x: Tensor, key_src: Tensor | None = None, state: HeadState | None = None):
        """The head's read. With `state` the head continues an earlier call and returns (output, new HeadState)."""
        xk = x if key_src is None else key_src
        tail = None
        if self.key_conv > 1:
            if state is None:
                xk = self.conv(F.pad(xk.transpose(1, 2), (self.key_conv - 1, 0))).transpose(1, 2)
            else:  # the conv's left context is the previous call's last key_conv - 1 inputs instead of zeros
                src = torch.cat([state.conv_tail.to(xk.dtype), xk], dim=1)
                tail = src[:, -(self.key_conv - 1) :]
                xk = self.conv(src.transpose(1, 2)).transpose(1, 2)
        v = self.val(x)
        if state is None:
            total = 0
            for code, out in zip(self.codes, self.outs):
                total = total + out(self._one_head(code, xk, v))
            return total
        if len(self.codes) != 1:
            raise ValueError("a streamed head has exactly one key code")
        y, new = self._one_head(self.codes[0], xk, v, state)
        new.conv_tail = tail
        return self.outs[0](y), new

    def _one_head_chunked(self, code: KeyCode, xk: Tensor, v: Tensor, state: HeadState | None = None):
        return sparse_memory_read(code, xk, v, self.chunk, self.decay_value(), self.normalize, self.gate_tau if self.gate else None, self.state_bits, state)

    def _one_head(self, code: KeyCode, xk: Tensor, v: Tensor, state: HeadState | None = None):
        # the fast-weight sums accumulate in fp32 whatever autocast does to the rest of the model;
        # the state's precision is controlled explicitly by state_bits (fake-quantised between chunks)
        with torch.autocast(device_type=xk.device.type, enabled=False):
            return self._one_head_fp32(code, xk.float(), v.float(), state)

    def _one_head_fp32(self, code: KeyCode, xk: Tensor, v: Tensor, state: HeadState | None = None):
        if self.chunk > 0:
            return self._one_head_chunked(code, xk, v, state)
        if state is not None:
            raise ValueError("a streamed head needs the chunked form (chunk > 0)")
```

The rest of `_one_head_fp32` (from `B, T, _ = xk.shape` on) is not touched.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_stream.py tests/test_sparsela.py -q`
Expected: all pass. `tests/test_sparsela.py` passing is the evidence that the stateless path is unchanged.

- [ ] **Step 5: Commit**

```bash
git add flybrain/model/sparsela.py tests/test_stream.py
git commit -m "Memory head: carry the tables, the last key and the conv tail across calls"
```

---

### Task 6: The hybrid as an agent

**Files:**
- Modify: `flybrain/model/hybrid.py` (class `GRUHybrid`: three new methods; imports)
- Test: `tests/test_stream.py`

**Interfaces:**
- Consumes: `MinGRULayer.scan` (Task 3), `LocalAttention.stream` (Task 4), `Block.memory(..., state=)` (Task 5), `AgentState`, `AttnCache`, `HeadState` (Task 2).
- Produces: `GRUHybrid.init_state(batch: int, device=None) -> AgentState`; `GRUHybrid.stream(tokens: Tensor, state: AgentState) -> tuple[Tensor, AgentState]` with logits `[B, T, vocab]`. `state.heads[0]` is the post head; `state.heads[1 + k]` is `self.mems[k]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stream.py`:

```python
def _hybrid(**kw):
    from flybrain.model.hybrid import GRUHybrid
    torch.manual_seed(0)
    args = dict(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4, key_conv=2,
                normalize=True, gate=True, chunk=8, mem_every=2, local_attn=8, key_convs=(2, 4))
    args.update(kw)
    return GRUHybrid(**args).eval()


def _stream_all(m, x, cuts):
    st = m.init_state(x.size(0)); outs = []; c0 = 0
    for n in cuts:
        y, st = m.stream(x[:, c0 : c0 + n], st)
        outs.append(y); c0 += n
    return torch.cat(outs, 1), st


def test_hybrid_stream_matches_forward_across_segmentations():
    m = _hybrid()
    x = torch.randint(0, 256, (2, 61))
    with torch.no_grad():
        ref, _ = m(x)
        for cuts in ([61], [1, 60], [8, 8, 45], [7, 9, 13, 32], [1] * 61):
            out, st = _stream_all(m, x, cuts)
            assert torch.allclose(out, ref, atol=1e-4), (cuts, float((out - ref).abs().max()))
    assert len(st.heads) == 2 and len(st.attn) == 1 and st.attn[0].offset == 61 and st.batch == 2


def test_stream_covers_the_zero_heads_cell_and_the_heads_only_cell():
    x = torch.randint(0, 256, (2, 40))
    for kw in (dict(where="none", mem_every=0, key_convs=None), dict(local_attn=0)):
        m = _hybrid(**kw)
        with torch.no_grad():
            ref, _ = m(x)
            out, _ = _stream_all(m, x, [3, 17, 20])
        assert torch.allclose(out, ref, atol=1e-4), kw
    assert _hybrid(where="none", mem_every=0, key_convs=None).init_state(1).heads == []


def test_stream_refuses_what_it_cannot_do():
    import pytest
    from flybrain.model.hybrid import GRUHybrid
    with pytest.raises(ValueError):
        GRUHybrid(d_model=16, backbone="transformer", where="none", n_layers=2, d_ff=32).init_state(1)
    with pytest.raises(ValueError):
        _hybrid(chunk=0).init_state(1)
    with pytest.raises(ValueError):
        _hybrid(count_head=True).init_state(1)
    m = _hybrid()
    with pytest.raises(ValueError):                        # a state for 2 agents, tokens for 3
        m.stream(torch.randint(0, 256, (3, 5)), m.init_state(2))


def test_new_document_resets_the_reader_and_keeps_the_tables():
    m = _hybrid()
    a = torch.randint(0, 256, (1, 30)); b = torch.randint(0, 256, (1, 26))
    with torch.no_grad():
        _, st_a = m.stream(a, m.init_state(1))
        # with the tables emptied as well, a new document is exactly a fresh forward
        nd = st_a.new_document()
        for head, empty in zip(nd.heads, m.init_state(1).heads):
            head.S, head.Ksum = empty.S, empty.Ksum
        out, _ = m.stream(b, nd)
        ref, _ = m(b)
        assert torch.allclose(out, ref, atol=1e-4)
        # the lowest head's keys depend on no memory read, so its table is additive across documents
        _, st_ab = m.stream(b, st_a.new_document())
        _, st_b = m.stream(b, m.init_state(1))
        assert torch.allclose(st_ab.heads[1].S, st_a.heads[1].S + st_b.heads[1].S, atol=1e-5)
        assert not torch.allclose(st_ab.heads[0].S, st_a.heads[0].S + st_b.heads[0].S, atol=1e-5)   # the top head reads the lower one


def test_stream_runs_under_bf16_autocast_close_to_fp32():
    m = _hybrid()
    x = torch.randint(0, 256, (2, 33))
    with torch.no_grad():
        ref, _ = _stream_all(m, x, [9, 24])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out, _ = _stream_all(m, x, [9, 24])
    assert (out.float() - ref).abs().max() < 0.5          # the same budget the repo's bf16 forward test uses
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_stream.py -q -k "hybrid or stream or new_document_resets"`
Expected: failures with `AttributeError: 'GRUHybrid' object has no attribute 'init_state'`

- [ ] **Step 3: Write the implementation**

In `flybrain/model/hybrid.py`, change the import added in Task 4 to:

```python
from flybrain.model.state import AgentState, AttnCache, HeadState
```

At the end of class `GRUHybrid` (after `forward`), add:

```python
    def _check_streamable(self) -> None:
        if self.backbone != "mingru":
            raise ValueError("streaming is implemented for the mingru backbone only")
        if self.where == "pre" or self.count is not None or self.shuffle_keys:
            raise ValueError("streaming does not support where='pre', the count head or shuffled keys")
        if any(h.chunk <= 0 for h in self.all_heads()):
            raise ValueError("streaming needs the chunked head (chunk > 0)")

    def init_state(self, batch: int, device=None) -> AgentState:
        """A fresh agent: empty tables, no carry, empty attention cache."""
        self._check_streamable()
        device = self.embed.weight.device if device is None else device
        heads = []
        for blk in self.all_heads():
            d = blk.val.in_features
            tail = torch.zeros(batch, blk.key_conv - 1, d, device=device) if blk.key_conv > 1 else None
            heads.append(HeadState(S=torch.zeros(batch, blk.code.n_key, d, device=device), Ksum=torch.zeros(batch, blk.code.n_key, device=device), conv_tail=tail))
        n_attn = len(self.local_at) if self.local_attn else 0
        return AgentState(h=[None] * len(self.core.layers), heads=heads, attn=[AttnCache() for _ in range(n_attn)])

    def stream(self, tokens: Tensor, state: AgentState) -> tuple[Tensor, AgentState]:
        """forward() continued from `state`: the same logits as one pass over everything the agent has read.
        `state` is not modified; the new state is returned."""
        self._check_streamable()
        if state.batch is not None and state.batch != tokens.size(0):
            raise ValueError(f"state is for {state.batch} agents, tokens are for {tokens.size(0)}")
        h, heads, attn = list(state.h), list(state.heads), list(state.attn)
        e = self.embed(tokens)
        layers = self.core.layers; k = 0
        for i, l in enumerate(layers):
            e, h[i] = l.scan(e, h[i])
            if self.local_attn and i + 1 in self.local_at:
                j = self.local_at.index(i + 1)
                y, attn[j] = (self.local if j == 0 else self.local_more[j - 1]).stream(e, attn[j])
                e = e + y
            if self.mem_every and (i + 1) % self.mem_every == 0 and i + 1 < len(layers):
                y, heads[1 + k] = self.mems[k].memory(self.mems[k].ln1(e), state=heads[1 + k])
                e = e + y; k += 1
        out = self.core.ln(e)
        if self.where == "post":
            y, heads[0] = self.mem.memory(self.mem.ln1(out), state=heads[0])
            out = out + y
        return self.decoder(out), AgentState(h=h, heads=heads, attn=attn)
```

This mirrors `_run_core` and the mingru branch of `forward` line for line: layer, then local attention if one sits after this layer, then the inner head if one sits after this layer; final `LayerNorm`; post head; decoder. `mem_every` requires `where="post"` (the constructor enforces it), so whenever inner heads exist the post head is `heads[0]` and inner head k is `heads[1 + k]`.

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass (the existing tests plus the new ones in `tests/test_sample.py` and `tests/test_stream.py`)

- [ ] **Step 5: Commit**

```bash
git add flybrain/model/hybrid.py tests/test_stream.py
git commit -m "GRUHybrid: init_state and stream, matching forward across any segmentation"
```

---

### Task 7: Streamed sampler

**Files:**
- Modify: `scripts/sample.py` (`generate`, CLI; add `streamable`)
- Test: `tests/test_sample.py`

**Interfaces:**
- Consumes: `GRUHybrid.init_state`, `GRUHybrid.stream` (Task 6); `sample_next`, `Utf8Printer` (Task 1).
- Produces: `streamable(model) -> bool`; `generate(model, prompt, n, temperature=0.8, top_k=40, seed=0, window=2048, stream=False, on_byte=None) -> bytes`. CLI gains `--no-stream`; the streamed path is the default whenever the model supports it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sample.py`:

```python
def test_streamed_and_recompute_samplers_agree_greedily():
    from scripts.sample import generate, streamable
    m = _tiny()
    prompt = b"static int probe(struct device *dev)\n{\n"
    assert streamable(m)
    assert generate(m, prompt, 40, temperature=0.0, stream=True) == generate(m, prompt, 40, temperature=0.0, stream=False)
    assert generate(m, prompt, 20, temperature=0.9, seed=5, stream=True) == generate(m, prompt, 20, temperature=0.9, seed=5, stream=True)
    seen = []
    assert len(generate(m, b"", 3, temperature=0.0, stream=True, on_byte=seen.append)) == 3 and len(seen) == 3
    assert not streamable(GRUHybrid(d_model=16, backbone="transformer", where="none", n_layers=2, d_ff=32))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_sample.py::test_streamed_and_recompute_samplers_agree_greedily -q`
Expected: FAIL with `ImportError: cannot import name 'streamable' from 'scripts.sample'`

- [ ] **Step 3: Write the implementation**

In `scripts/sample.py`, replace the module docstring with:

```python
"""Prompt a saved checkpoint: bytes in, sampled bytes out.

The streamed path carries agent state and costs one position per new byte (mingru arms). --no-stream recomputes the
last --window bytes with the plain forward for every new byte; it works for any arm and is the oracle for the streamed path."""
```

Replace the whole `generate` function with:

```python
def streamable(model) -> bool:
    """True if the model has the streamed entry point and supports it in this configuration."""
    try:
        model.init_state(1)
    except (AttributeError, ValueError):
        return False
    return True


@torch.no_grad()
def generate(model, prompt: bytes, n: int, temperature: float = 0.8, top_k: int = 40, seed: int = 0, window: int = 2048, stream: bool = False, on_byte=None) -> bytes:
    """Sample n bytes after `prompt` (an empty prompt starts from a newline). `on_byte` is called with each byte as it is drawn.
    stream=True reads the prompt once and then feeds one byte per step; it ignores `window` because the agent's state is the context."""
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    buf = list(prompt) or [10]
    out = []
    if stream:
        logits, state = model.stream(torch.tensor([buf], device=device), model.init_state(1, device))
        for i in range(n):
            b = sample_next(logits[0, -1].float().cpu(), temperature, top_k, gen)
            out.append(b)
            if on_byte is not None:
                on_byte(b)
            if i + 1 < n:
                logits, state = model.stream(torch.tensor([[b]], device=device), state)
        return bytes(out)
    for _ in range(n):
        logits, _ = model(torch.tensor([buf[-window:]], device=device))
        b = sample_next(logits[0, -1].float().cpu(), temperature, top_k, gen)
        buf.append(b); out.append(b)
        if on_byte is not None:
            on_byte(b)
    return bytes(out)
```

In `main`, add this argument after `--device`:

```python
    p.add_argument("--no-stream", action="store_true", help="recompute the last --window bytes per new byte instead of carrying state")
```

and replace the `generate(...)` call in `main` with:

```python
    use_stream = not args.no_stream and streamable(model)
    generate(model, prompt, args.n, args.temperature, args.top_k, args.seed, args.window, stream=use_stream, on_byte=printer.feed)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sample.py -q`
Expected: 5 passed

If the greedy comparison fails on a single byte while Task 6's tests pass, two logits were within float error of each other at that step: print both outputs, confirm they differ from one position on, and change the prompt in the test to another ASCII string of similar length. Do not loosen Task 6's tolerances.

- [ ] **Step 5: Prompt the real 50M checkpoint, streamed, on CPU**

Run: `uv run python scripts/sample.py --ckpt results/ckpt/m50_linux_hybrid_s0_mingru_sparse_post.pt --prompt 'static int ' --n 400 --device cpu`
Expected: 400 bytes of kernel-style C, much faster than Task 1's run (tens of bytes per second, an estimate).

Then confirm the two paths agree on the real model:

Run: `uv run python - <<'EOF'
import torch
from scripts.sample import generate, load_model
m = load_model("results/ckpt/m50_linux_hybrid_s0_mingru_sparse_post.pt", torch.device("cpu"))
p = b"static int "
a = generate(m, p, 48, temperature=0.0, stream=True)
b = generate(m, p, 48, temperature=0.0, stream=False, window=2048)
print(a == b); print(a.decode("utf-8", errors="replace"))
EOF`
Expected: `True`, then the greedy continuation. If it prints `False`, stop and report the first differing position with both logit rows; do not adjust anything.

- [ ] **Step 6: Commit**

```bash
git add scripts/sample.py tests/test_sample.py
git commit -m "Sampler: streamed generation by default, recompute path as the oracle"
```

---

### Task 8: Group runtime

**Files:**
- Create: `flybrain/group.py`
- Test: `tests/test_group.py`

**Interfaces:**
- Consumes: `GRUHybrid.init_state`, `GRUHybrid.stream` (Task 6); `AgentState` (Task 2).
- Produces:
  - `Commons(S: list[Tensor], Ksum: list[Tensor])`: per head, `[n_key, d]` and `[n_key]`.
  - `weights_hash(model) -> str`
  - `read(model, streams: Tensor, state: AgentState | None = None, seg: int = 2048) -> tuple[Tensor, AgentState]`: `streams` is `[B, T]`; the first value is cross-entropy in nats, `[B, T - 1]`, the loss of byte t + 1 given bytes up to t.
  - `merge(states: AgentState | Sequence[AgentState]) -> Commons`; raises `ValueError` on an empty sequence.
  - `preload(state: AgentState, commons: Commons, alpha: float = 1.0) -> AgentState`
  - `save_commons(path, commons: Commons, model) -> None`; `load_commons(path, model) -> Commons`, raising `ValueError` when the weights differ.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_group.py`:

```python
import pytest
import torch
from torch.nn import functional as F

from flybrain.model.hybrid import GRUHybrid


def _hybrid(**kw):
    torch.manual_seed(0)
    args = dict(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4, key_conv=2,
                normalize=True, gate=True, chunk=8, mem_every=2, local_attn=8, key_convs=(2, 4))
    args.update(kw)
    return GRUHybrid(**args).eval()


def test_read_in_segments_scores_like_forward_and_merge_sums_agents():
    from flybrain.group import merge, read
    m = _hybrid()
    x = torch.randint(0, 256, (3, 50))
    ce, st = read(m, x, seg=16)
    with torch.no_grad():
        logits, _ = m(x)
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1), reduction="none").view(3, 49)
    assert ce.shape == (3, 49) and torch.allclose(ce, ref, atol=1e-4)
    c = merge(st)
    assert len(c.S) == len(st.heads) == 2 and c.S[0].shape == (64, 16) and c.Ksum[0].shape == (64,)
    assert torch.allclose(c.S[0], st.heads[0].S.sum(0)) and torch.allclose(c.Ksum[1], st.heads[1].Ksum.sum(0))
    singles = [read(m, x[i : i + 1], seg=16)[1] for i in range(3)]   # three agents one at a time == one batch of three
    c2 = merge(singles)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(c.S, c2.S))
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(c.Ksum, c2.Ksum))
    with pytest.raises(ValueError):
        merge([])


def test_preload_zero_is_alone_and_a_handoff_gives_the_lowest_head_what_reading_would():
    from flybrain.group import merge, preload, read
    m = _hybrid()
    a = torch.randint(0, 256, (1, 30)); b = torch.randint(0, 256, (1, 26))
    _, st_a = read(m, a)
    with torch.no_grad():
        alone, _ = m.stream(b, m.init_state(1))
        zero, _ = m.stream(b, preload(m.init_state(1), merge(st_a), 0.0))
        assert torch.equal(alone, zero)
        fresh = m.init_state(1)
        handed_logits, handed = m.stream(b, preload(fresh, merge(st_a), 1.0))
        _, sequential = m.stream(b, st_a.new_document())
        assert float(fresh.heads[0].S.abs().sum()) == 0.0                 # preload returned a copy
        assert torch.allclose(handed.heads[1].S, sequential.heads[1].S, atol=1e-5)
        assert not torch.allclose(alone, handed_logits, atol=1e-4)        # the handoff changes what the agent predicts


def test_preload_gives_every_agent_the_same_commons():
    from flybrain.group import merge, preload, read
    m = _hybrid()
    _, st = read(m, torch.randint(0, 256, (2, 24)))
    c = merge(st)
    two = preload(m.init_state(2), c, 0.5)
    assert torch.allclose(two.heads[0].S[0], 0.5 * c.S[0]) and torch.equal(two.heads[0].S[0], two.heads[0].S[1])


def test_commons_file_is_bound_to_the_weights_that_wrote_it(tmp_path):
    from flybrain.group import load_commons, merge, read, save_commons, weights_hash
    m = _hybrid()
    c = merge(read(m, torch.randint(0, 256, (2, 20)))[1])
    p = tmp_path / "commons.pt"
    save_commons(p, c, m)
    back = load_commons(p, m)
    assert all(torch.equal(a, b) for a, b in zip(c.S, back.S)) and all(torch.equal(a, b) for a, b in zip(c.Ksum, back.Ksum))
    assert weights_hash(m) == weights_hash(_hybrid())                     # same seed, same weights, same hash
    other = _hybrid()
    with torch.no_grad():
        other.decoder.bias.add_(1.0)
    assert weights_hash(other) != weights_hash(m)
    with pytest.raises(ValueError):
        load_commons(p, other)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_group.py -q`
Expected: 4 failures with `ModuleNotFoundError: No module named 'flybrain.group'`

- [ ] **Step 3: Write the implementation**

Create `flybrain/group.py`:

```python
"""A group of agents with one brain: read different streams, pool what was written, hand it to an agent that did not read it.

Agents with the same weights compute the same sparse addresses, so one agent's tables are readable by another.
That is the only reason any of this works, and why a memory file is bound to the weights that wrote it."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from flybrain.model.state import AgentState


@dataclass
class Commons:
    """Per-head sum of a group's tables."""

    S: list       # per head [n_key, d]
    Ksum: list    # per head [n_key]


def weights_hash(model) -> str:
    """Identity of the brain that wrote a memory: sha256 over every tensor in the state dict, in key order."""
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def read(model, streams: Tensor, state: AgentState | None = None, seg: int = 2048) -> tuple[Tensor, AgentState]:
    """Each row of `streams` [B, T] is one agent reading its own bytes, `seg` at a time.
    Returns (ce [B, T - 1] in nats: the loss of byte t + 1 given bytes up to t, the agents' state after all T bytes)."""
    B, T = streams.shape
    state = model.init_state(B, streams.device) if state is None else state
    ces = []
    for c0 in range(0, T, seg):
        logits, state = model.stream(streams[:, c0 : c0 + seg], state)
        y = streams[:, c0 + 1 : c0 + seg + 1]
        if y.size(1):
            ces.append(F.cross_entropy(logits[:, : y.size(1)].float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(B, -1))
    ce = torch.cat(ces, dim=1) if ces else torch.zeros(B, 0, device=streams.device)
    return ce, state


def merge(states: AgentState | Sequence[AgentState]) -> Commons:
    """Per-head sum of S and Ksum over every agent of every state."""
    states = [states] if isinstance(states, AgentState) else list(states)
    if not states:
        raise ValueError("merge needs at least one agent state")
    n = len(states[0].heads)
    return Commons(S=[sum(st.heads[i].S.sum(0) for st in states) for i in range(n)],
                   Ksum=[sum(st.heads[i].Ksum.sum(0) for st in states) for i in range(n)])


def preload(state: AgentState, commons: Commons, alpha: float = 1.0) -> AgentState:
    """A copy of `state` whose tables also hold alpha x commons; every agent gets the same commons."""
    heads = [replace(h, S=h.S + alpha * S.to(h.S)[None], Ksum=h.Ksum + alpha * K.to(h.Ksum)[None])
             for h, S, K in zip(state.heads, commons.S, commons.Ksum)]
    return AgentState(h=list(state.h), heads=heads, attn=list(state.attn))


def save_commons(path, commons: Commons, model) -> None:
    torch.save({"weights": weights_hash(model), "S": [s.cpu() for s in commons.S], "Ksum": [k.cpu() for k in commons.Ksum]}, path)


def load_commons(path, model) -> Commons:
    """Refuses a memory written by different weights: its addresses would mean nothing to this brain."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob["weights"] != weights_hash(model):
        raise ValueError("this memory was written by different weights")
    return Commons(S=blob["S"], Ksum=blob["Ksum"])
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add flybrain/group.py tests/test_group.py
git commit -m "Group runtime: read, merge, preload, and memory files bound to their weights"
```

---

## Self-review

| Spec requirement (Stage 0) | Task |
|---|---|
| 0.1 sampler CLI, `--no-stream` as the oracle, UTF-8 with replacement, loads through `cross_eval.rebuild` | 1, 7 |
| `HeadState`, `AttnCache`, `AgentState.new_document()` | 2 |
| `MinGRULayer.scan(x, h0=None) -> (y, h_last)`; `forward` returns `scan(x)[0]` | 3 |
| `LocalAttention.stream(x, cache)` with rebased rotary positions | 4 |
| `sparse_memory_read(..., state=None)`; `Block.memory(x, key_src=None, state=None)` with the conv tail | 5 |
| `GRUHybrid.init_state / stream`; unsupported configurations raise | 6 |
| Exact within tolerances; `state_bits > 0` exact only for chunk multiples | 3–6 (tolerances), 5 (`test_quantised_state_survives_single_byte_segments`) |
| `read`, `merge`, `preload`, `save_commons`, `load_commons` with the weights hash | 8 |
| Spec tests 1–5 and 7 | 3, 4, 5, 6, 6, 7 |

Spec test 6 (the runner's default path) belongs to plan C.

One deliberate refinement of the spec's wording: `read` takes a `[B, T]` tensor, so agents in one call read streams of equal length. Readers of unequal length are called one at a time (`B = 1`) and combined with `merge`, which the first group test shows gives the same commons.
