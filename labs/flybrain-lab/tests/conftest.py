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
