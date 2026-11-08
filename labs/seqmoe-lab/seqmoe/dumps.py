"""Fetch real scientific dumps. Never relabel harvest/fallback as dump on failure."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from seqmoe.data import _cache_dir, _write_source_sidecar, harvest_lab_python, wiki_path

GSM8K_TRAIN = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/train.jsonl"
)

# Public Python corpus that is not lab harvest. The installed CPython Lib is a
# dumped stdlib; used only if a network code dump cannot be fetched.
CPYTHON_LIB = Path(
    "~/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/lib/python3.11"
).expanduser()


def _http_get(url: str, timeout: int = 60) -> tuple[bytes | None, str | None]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "seqmoe-lab/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), None
    except Exception as exc:  # noqa: BLE001 — log exact miss
        return None, f"{type(exc).__name__}: {exc} url={url}"


def gsm8k_to_text(raw: bytes) -> bytes:
    lines_out: list[str] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        q = str(obj.get("question") or "").strip()
        a = str(obj.get("answer") or "").strip()
        if q and a:
            lines_out.append(f"Question: {q}\nAnswer: {a}\n")
    return "".join(lines_out).encode("utf-8")


def collect_cpython_lib(root: Path = CPYTHON_LIB) -> bytes:
    chunks: list[bytes] = []
    if not root.is_dir():
        return b""
    skip = {"test", "tests", "__pycache__", "site-packages", "lib-dynload", "idlelib"}
    for path in sorted(root.rglob("*.py")):
        if any(part in skip for part in path.parts):
            continue
        try:
            chunks.append(path.read_bytes() + b"\n")
        except OSError:
            continue
        if sum(len(c) for c in chunks) > 3_000_000:
            break
    return b"".join(chunks)


def fetch_dumps(cache_dir: Optional[Path] = None) -> dict[str, Any]:
    """Write dump files + sidecar. On miss, leave harvest/fallback labels alone."""
    cache = cache_dir or _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "cache": str(cache),
        "gsm8k_url": GSM8K_TRAIN,
        "code_origin": None,
        "math_origin": None,
        "errors": [],
    }
    code_path = cache / "code.txt"
    math_path = cache / "math.txt"

    math_raw, math_err = _http_get(GSM8K_TRAIN)
    math_ok = False
    if math_raw:
        text = gsm8k_to_text(math_raw)
        if len(text) > 1000:
            math_path.write_bytes(text)
            math_ok = True
            status["math_origin"] = "gsm8k_train_jsonl"
            status["math_bytes"] = len(text)
        else:
            status["errors"].append("GSM8K decoded to <1000 bytes")
    else:
        status["errors"].append(math_err or "GSM8K fetch failed")

    # Code: prefer CPython stdlib dump (not harvest_lab_python / not lab trees).
    code_blob = collect_cpython_lib()
    code_ok = False
    if len(code_blob) > 1000:
        code_path.write_bytes(code_blob)
        code_ok = True
        status["code_origin"] = f"cpython_stdlib:{CPYTHON_LIB}"
        status["code_bytes"] = len(code_blob)
    else:
        status["errors"].append(f"cpython lib missing or tiny at {CPYTHON_LIB}")

    wiki = wiki_path()
    wiki_ok = wiki.is_file() and wiki.stat().st_size > 10_000
    status["wiki_ok"] = wiki_ok
    status["wiki_path"] = str(wiki)
    if wiki_ok:
        status["wiki_bytes"] = wiki.stat().st_size

    # Sidecar: only write dump labels for sources we actually obtained.
    from seqmoe.data import _read_source_sidecar

    prev = _read_source_sidecar(cache)
    code_source = "dump" if code_ok else str(prev.get("code_source") or "lab_python")
    math_source = "dump" if math_ok else str(prev.get("math_source") or "fallback")
    if not code_ok and not code_path.is_file():
        code_path.write_bytes(harvest_lab_python())
        code_source = "lab_python"
    _write_source_sidecar(cache, code_source, math_source)
    grade = (
        "full"
        if wiki_ok and code_source == "dump" and math_source == "dump"
        else "weak"
    )
    status["code_source"] = code_source
    status["math_source"] = math_source
    status["scientific_grade"] = grade
    status["code_path"] = str(code_path)
    status["math_path"] = str(math_path)
    if code_path.is_file():
        status["code_bytes_on_disk"] = code_path.stat().st_size
    if math_path.is_file():
        status["math_bytes_on_disk"] = math_path.stat().st_size
    return status
