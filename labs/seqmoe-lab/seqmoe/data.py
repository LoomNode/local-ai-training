"""Byte-level tokenizer, synth corpora, TokenBatcher, scientific loaders."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import torch

from seqmoe.config import DOMAIN_TO_ID

VOCAB_SIZE = 256
CHUNK = 256
DEFAULT_WIKI = Path("/games/ailab/lab/scrt-lab/data/wiki.train.raw")
LAB_PYTHON_ROOTS = (
    Path("/games/ailab/lab/scrt-lab"),
    Path("/games/ailab/lab/envelope-lab"),
    Path("/games/ailab/lab/ailab-cli"),
)
SKIP_DIR_NAMES = {".venv", "__pycache__", ".git", "node_modules", ".uv-cache", "results"}


def encode_bytes(data: bytes) -> torch.Tensor:
    return torch.tensor(list(data), dtype=torch.long)


def encode_text(text: str) -> torch.Tensor:
    return encode_bytes(text.encode("utf-8"))


def decode_bytes(tokens: torch.Tensor) -> str:
    return bytes(int(t) for t in tokens.tolist()).decode("utf-8", errors="replace")


def split_holdout(data: torch.Tensor, frac: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(data.numel() * (1.0 - frac))
    n = max(1, min(n, data.numel() - 1))
    return data[:n].contiguous(), data[n:].contiguous()


def chunk_shuffle(parts: Iterable[bytes], *, chunk: int = CHUNK, seed: int = 0) -> bytes:
    chunks: list[bytes] = []
    for part in parts:
        if not part:
            continue
        for i in range(0, len(part), chunk):
            chunks.append(part[i : i + chunk])
    rng = random.Random(seed)
    rng.shuffle(chunks)
    return b"".join(chunks)


def _fill_to(gen, length: int) -> bytes:
    buf = bytearray()
    while len(buf) < length:
        buf.extend(gen())
    return bytes(buf[:length])


def synth_code(length: int = 50_000, seed: int = 0) -> bytes:
    rng = random.Random(seed)
    names = [
        "foo",
        "bar",
        "baz",
        "xs",
        "n",
        "self",
        "data",
        "val",
        "item",
        "count",
        "total",
        "path",
        "cfg",
        "model",
        "loss",
    ]
    ops = ["+", "-", "*", "//"]

    def one() -> bytes:
        name = rng.choice(names)
        arg = rng.choice(names)
        other = rng.choice(names)
        n = rng.randint(0, 9)
        kind = rng.randint(0, 4)
        if kind == 0:
            body = (
                f"def {name}({arg}):\n"
                f"    return {arg} {rng.choice(ops)} {n}\n"
            )
        elif kind == 1:
            body = (
                f"def {name}({arg}, {other}):\n"
                f"    if {arg}:\n"
                f"        return {other}\n"
                f"    return {n}\n"
            )
        elif kind == 2:
            body = (
                f"class {name.title()}:\n"
                f"    def __init__(self, {arg}):\n"
                f"        self.{arg} = {arg}\n"
            )
        elif kind == 3:
            body = (
                f"for {arg} in {other}:\n"
                f"    {name}.append({arg})\n"
            )
        else:
            body = f"from {name} import {arg}\n{name} = {arg}({n})\n"
        return body.encode("utf-8")

    return _fill_to(one, length)


def synth_math(length: int = 50_000, seed: int = 1) -> bytes:
    rng = random.Random(seed)

    def one() -> bytes:
        a = rng.randint(0, 99)
        b = rng.randint(0, 99)
        op = rng.choice(["+", "-", "*", "/"])
        if op == "+":
            c: int | float = a + b
        elif op == "-":
            c = a - b
        elif op == "*":
            c = a * b
        else:
            b = rng.randint(1, 9)
            c = a // b
            op = "//"
        kind = rng.randint(0, 3)
        if kind == 0:
            s = f"{a}{op}{b}={c}\n"
        elif kind == 1:
            s = f"{a} + {b} = {a + b}; {a} * {b} = {a * b}\n"
        elif kind == 2:
            xs = [rng.randint(0, 20) for _ in range(4)]
            s = f"sum({xs})={sum(xs)} mean={sum(xs)/len(xs):.1f}\n"
        else:
            s = f"({a}+{b})*{rng.randint(1, 9)}={(a + b) * (1 + rng.randint(0, 8))}\n"
        return s.encode("utf-8")

    return _fill_to(one, length)


_FILL_WORDS = (
    "the a an of to and in is it on for as with that this from or by "
    "cat sat mat then went home later after before over under again "
    "people world time year day way man woman child life hand part "
    "place case week company system program number group problem"
).split()


def synth_fill(length: int = 20_000, seed: int = 2) -> bytes:
    rng = random.Random(seed)

    def one() -> bytes:
        n = rng.randint(6, 14)
        words = [rng.choice(_FILL_WORDS) for _ in range(n)]
        return (" ".join(words) + ". ").encode("utf-8")

    return _fill_to(one, length)


def build_cross_domain(n: int = 512, seed: int = 0) -> bytes:
    rng = random.Random(seed)
    parts: list[str] = []
    while len(parts) < n:
        a = rng.randint(2, 40)
        b = rng.randint(2, 40)
        parts.append(f"def mean(xs):\n    return sum(xs)/len(xs)\n")
        parts.append(f"# compute {a}*{b} in python\nprint({a}*{b})\n")
        parts.append(f"def add(a, b):\n    return a + b  # {a}+{b}={a + b}\n")
        parts.append(f"result = {a} + {b}\nprint(result)\n")
        parts.append(f"xs = [{a}, {b}, {a + b}]\nprint(sum(xs)/len(xs))\n")
    return "".join(parts[:n]).encode("utf-8")


@dataclass
class DomainSplit:
    name: str
    logical_id: Optional[int]
    train: torch.Tensor
    holdout: torch.Tensor
    source: str


@dataclass
class PocData:
    mixed_train: torch.Tensor
    splits: dict[str, torch.Tensor]
    domains: dict[str, DomainSplit]
    scientific_grade: str
    code_source: str
    math_source: str
    wiki_source: str = "none"
    meta: dict = field(default_factory=dict)


class TokenBatcher:
    def __init__(
        self,
        data: torch.Tensor,
        seq_len: int,
        batch_size: int,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.g = torch.Generator().manual_seed(seed)
        self.n = int(data.numel()) - seq_len - 1
        if self.n <= 0:
            raise ValueError("corpus too short for seq_len")

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(0, self.n, (self.batch_size,), generator=self.g)
        x = torch.stack([self.data[i : i + self.seq_len] for i in ix.tolist()])
        y = torch.stack(
            [self.data[i + 1 : i + 1 + self.seq_len] for i in ix.tolist()]
        )
        return x.to(self.device), y.to(self.device)


class MixedDomainBatcher:
    """Sample sequences from domain streams with fixed probabilities."""

    def __init__(
        self,
        domain_data: dict[int, torch.Tensor],
        probs: dict[int, float],
        seq_len: int,
        batch_size: int,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.ids = sorted(domain_data)
        total = sum(probs[i] for i in self.ids)
        self.p = torch.tensor([probs[i] / total for i in self.ids], dtype=torch.float64)
        self.batchers = {
            i: TokenBatcher(domain_data[i], seq_len, 1, device=device, seed=seed + i)
            for i in self.ids
        }
        self.batch_size = batch_size
        self.device = device
        self.g = torch.Generator().manual_seed(seed + 17)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        picks = torch.multinomial(self.p.float(), self.batch_size, replacement=True, generator=self.g)
        xs = []
        ys = []
        domains = []
        for row in picks.tolist():
            did = self.ids[int(row)]
            x, y = self.batchers[did].sample()
            xs.append(x[0])
            ys.append(y[0])
            domains.append(did)
        return (
            torch.stack(xs).to(self.device),
            torch.stack(ys).to(self.device),
            torch.tensor(domains, dtype=torch.long, device=self.device),
        )


def _domain(name: str, raw: bytes, source: str, holdout_frac: float = 0.1) -> DomainSplit:
    data = encode_bytes(raw)
    train, holdout = split_holdout(data, holdout_frac)
    return DomainSplit(
        name=name,
        logical_id=DOMAIN_TO_ID.get(name),
        train=train,
        holdout=holdout,
        source=source,
    )


def build_synth_data(seed: int = 42) -> PocData:
    code = _domain("code", synth_code(50_000, seed=seed), "synth")
    math = _domain("math", synth_math(50_000, seed=seed + 1), "synth")
    fill = _domain("fill", synth_fill(20_000, seed=seed + 2), "synth")
    mixed_bytes = chunk_shuffle(
        [
            bytes(code.train.tolist()),
            bytes(math.train.tolist()),
            bytes(fill.train.tolist()),
        ],
        seed=seed + 3,
    )
    mixed_hold = chunk_shuffle(
        [
            bytes(code.holdout.tolist()),
            bytes(math.holdout.tolist()),
            bytes(fill.holdout.tolist()),
        ],
        seed=seed + 4,
    )
    cross = encode_bytes(build_cross_domain(512, seed=seed + 5))
    splits = {
        "code": code.holdout,
        "math": math.holdout,
        "fill": fill.holdout,
        "mixed": encode_bytes(mixed_hold),
        "cross": cross,
    }
    return PocData(
        mixed_train=encode_bytes(mixed_bytes),
        splits=splits,
        domains={"code": code, "math": math, "fill": fill},
        scientific_grade="synth",
        code_source="synth",
        math_source="synth",
        wiki_source="none",
    )


def _cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "cache"


def wiki_path() -> Path:
    env = os.environ.get("SEQMOE_WIKI")
    if env:
        return Path(env)
    return DEFAULT_WIKI


def harvest_lab_python() -> bytes:
    chunks: list[bytes] = []
    for root in LAB_PYTHON_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            try:
                chunks.append(path.read_bytes() + b"\n")
            except OSError:
                continue
    return b"".join(chunks)


def synth_math_scientific(n_examples: int = 100_000, seed: int = 0) -> bytes:
    rng = random.Random(seed)
    parts: list[bytes] = []
    for _ in range(n_examples):
        a = rng.randint(10, 9999)
        b = rng.randint(10, 9999)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            c = a + b
            s = f"What is {a} plus {b}? {c}\n"
        elif op == "-":
            c = a - b
            s = f"Compute {a} minus {b}. Answer: {c}\n"
        else:
            c = a * b
            s = f"Product of {a} and {b} is {c}\n"
        if rng.random() < 0.3:
            xs = [rng.randint(1, 50) for _ in range(rng.randint(3, 8))]
            s += f"mean of {xs} is {sum(xs)/len(xs):.4f}; sum is {sum(xs)}\n"
        parts.append(s.encode("utf-8"))
    return b"".join(parts)


def prepare_scientific(cache_dir: Optional[Path] = None) -> dict:
    """Write data/cache for the scientific path. Never called from pytest."""
    cache = cache_dir or _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    wiki = wiki_path()
    wiki_ok = wiki.is_file() and wiki.stat().st_size > 10_000
    code_path = cache / "code.txt"
    math_path = cache / "math.txt"
    if code_path.is_file() and code_path.stat().st_size > 1000:
        code_source = "dump"
    else:
        harvested = harvest_lab_python()
        code_path.write_bytes(harvested)
        code_source = "lab_python"
    if math_path.is_file() and math_path.stat().st_size > 1000:
        math_source = "dump"
    else:
        math_path.write_bytes(synth_math_scientific())
        math_source = "fallback"
    grade = (
        "full"
        if wiki_ok and code_source == "dump" and math_source == "dump"
        else "weak"
    )
    meta = {
        "wiki_ok": wiki_ok,
        "wiki_path": str(wiki),
        "code_source": code_source,
        "math_source": math_source,
        "scientific_grade": grade,
        "code_path": str(code_path),
        "math_path": str(math_path),
    }
    print(
        f"prepare: wiki_ok={wiki_ok} code_source={code_source} "
        f"math_source={math_source} scientific_grade={grade}"
    )
    return meta


def build_scientific_data(seed: int = 42) -> PocData:
    meta = prepare_scientific()
    if not meta["wiki_ok"]:
        raise FileNotFoundError(
            f"wiki missing at {meta['wiki_path']}; set SEQMOE_WIKI or copy "
            f"WikiText-2 train to that path"
        )
    wiki_raw = Path(meta["wiki_path"]).read_bytes()
    code_raw = Path(meta["code_path"]).read_bytes()
    math_raw = Path(meta["math_path"]).read_bytes()
    # Cap huge harvests so tiny-adjacent scientific still fits RAM/time.
    code_raw = code_raw[: 2_000_000]
    math_raw = math_raw[: 2_000_000]
    wiki_raw = wiki_raw[: 2_000_000]
    code = _domain("code", code_raw, meta["code_source"])
    math = _domain("math", math_raw, meta["math_source"])
    wiki = _domain("wiki", wiki_raw, "wikitext2")
    mixed_bytes = chunk_shuffle(
        [
            bytes(code.train.tolist()),
            bytes(math.train.tolist()),
            bytes(wiki.train.tolist()),
        ],
        seed=seed + 3,
    )
    mixed_hold = chunk_shuffle(
        [
            bytes(code.holdout.tolist()),
            bytes(math.holdout.tolist()),
            bytes(wiki.holdout.tolist()),
        ],
        seed=seed + 4,
    )
    cross = encode_bytes(build_cross_domain(512, seed=seed + 5))
    splits = {
        "code": code.holdout,
        "math": math.holdout,
        "wiki": wiki.holdout,
        "mixed": encode_bytes(mixed_hold),
        "cross": cross,
    }
    return PocData(
        mixed_train=encode_bytes(mixed_bytes),
        splits=splits,
        domains={"code": code, "math": math, "wiki": wiki},
        scientific_grade=meta["scientific_grade"],
        code_source=meta["code_source"],
        math_source=meta["math_source"],
        wiki_source="wikitext2",
        meta=meta,
    )
