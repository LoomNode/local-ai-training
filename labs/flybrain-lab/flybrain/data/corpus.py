"""Repeat-rich byte corpora and a repeat-aware scoring mask."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import Tensor

SKIP_DIRS = {".venv", "node_modules", "__pycache__", "hf-cache", "ComfyUI", "open-webui", "llama.cpp-qwen4exp", ".git", "build"}


def build_code_corpus(root: str | Path, max_bytes: int, out: str | Path) -> int:
    """Concatenate .py files under root (sorted, deterministic) up to max_bytes."""
    files = []
    for p in sorted(Path(root).rglob("*.py")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        files.append(p)
    buf = bytearray()
    for p in files:
        try:
            b = p.read_bytes()
        except OSError:
            continue
        buf += b"\n# ==== " + str(p.relative_to(root)).encode() + b" ====\n" + b
        if len(buf) >= max_bytes:
            break
    Path(out).write_bytes(bytes(buf[:max_bytes]))
    return min(len(buf), max_bytes)


_WORDS = ("data", "model", "train", "eval", "config", "utils", "main", "test", "run", "build", "notes", "results", "src", "lib", "tmp", "log")
_EXT = (".py", ".txt", ".md", ".json", ".log", ".sh")


def terminal_transcript(n_bytes: int, seed: int = 0) -> bytes:
    """Synthetic shell session: a small filesystem, ls/cat/grep/echo/python with real repeated content."""
    rng = random.Random(seed)
    fs: dict[str, str] = {}
    cwd = "/home/user/" + rng.choice(_WORDS)

    def new_name() -> str:
        return rng.choice(_WORDS) + ("_" + rng.choice(_WORDS) if rng.random() < 0.4 else "") + rng.choice(_EXT)

    def new_content() -> str:
        kind = rng.random()
        if kind < 0.4:
            lines = [f"{rng.choice(_WORDS)} = {rng.randint(0, 999)}" for _ in range(rng.randint(2, 6))]
            return "\n".join(lines) + "\n"
        if kind < 0.7:
            return "{" + ", ".join(f'"{rng.choice(_WORDS)}": {rng.randint(0, 99)}' for _ in range(rng.randint(2, 5))) + "}\n"
        return "".join(f"[{rng.randint(10, 23)}:{rng.randint(10, 59)}:{rng.randint(10, 59)}] {rng.choice(('INFO', 'WARN', 'ERROR'))} {rng.choice(_WORDS)} {rng.choice(_WORDS)} {rng.randint(0, 9999)}\n" for _ in range(rng.randint(2, 5)))

    for _ in range(rng.randint(3, 6)):
        fs[new_name()] = new_content()
    out = []
    size = 0
    while size < n_bytes:
        prompt = f"user@box:{cwd}$ "
        r = rng.random()
        names = sorted(fs)
        if r < 0.22:
            cmd, resp = "ls", "  ".join(names) + "\n"
        elif r < 0.45 and names:
            n = rng.choice(names)
            cmd, resp = f"cat {n}", fs[n]
        elif r < 0.55 and names:
            n, w = rng.choice(names), rng.choice(_WORDS)
            hits = [ln for ln in fs[n].splitlines() if w in ln]
            cmd, resp = f"grep {w} {n}", ("\n".join(hits) + "\n") if hits else ""
        elif r < 0.68:
            n, c = new_name(), new_content()
            fs[n] = c
            cmd, resp = f"cat > {n} << 'EOF'\n{c}EOF", ""
        elif r < 0.78 and names:
            n = rng.choice(names)
            cmd, resp = f"python {n}", (f"Traceback (most recent call last):\n  File \"{n}\", line {rng.randint(1, 40)}, in <module>\nNameError: name '{rng.choice(_WORDS)}' is not defined\n" if rng.random() < 0.5 else f"ok {rng.randint(0, 99)}\n")
        elif r < 0.86:
            cmd, resp = "pwd", cwd + "\n"
        elif r < 0.93 and names:
            n = rng.choice(names)
            cmd, resp = f"wc -l {n}", f"{fs[n].count(chr(10))} {n}\n"
        else:
            n = rng.choice(names) if names else "x"
            cmd, resp = f"cat {n}_missing", f"cat: {n}_missing: No such file or directory\n"
        chunk = prompt + cmd + "\n" + resp
        out.append(chunk)
        size += len(chunk)
    return "".join(out).encode("utf-8")[:n_bytes]


def copyable_mask(x: Tensor, y: Tensor, n: int = 8) -> Tensor:
    """True where target y[t] continues an n-gram (x[t-n+1..t]) that occurred earlier in the window
    and was followed by the same byte there. These are the bytes an exact-match memory can predict."""
    B, T = x.shape
    mask = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        seen: dict[tuple, set] = {}
        row = x[b].tolist(); tgt = y[b].tolist()
        for t in range(T):
            if t >= n - 1:
                g = tuple(row[t - n + 1 : t + 1])
                if g in seen and tgt[t] in seen[g]:
                    mask[b, t] = True
                seen.setdefault(g, set()).add(tgt[t])
    return mask
