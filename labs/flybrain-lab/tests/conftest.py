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
