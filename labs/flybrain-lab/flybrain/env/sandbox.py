"""Contained relative-command sandbox. No host bash; paths must stay under root."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

ALLOWED = frozenset({"echo", "cat", "ls", "pwd"})


@dataclass
class Observation:
    text: str
    refused: bool = False
    exit_code: int = 0


class Sandbox:
    def __init__(self, root: str | Path, timeout: float = 2.0) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def reset(self) -> str:
        return f"fly@box:{self.root.name}$ \n"

    def _confine(self, spec: str) -> Path | None:
        raw = spec.strip()
        if not raw or raw.startswith("-"):
            return None
        path = Path(raw)
        if path.is_absolute():
            candidate = path
        else:
            candidate = self.root / path
        try:
            resolved = candidate.resolve()
            resolved.relative_to(self.root)
        except ValueError:
            return None
        return resolved

    def _parse(self, action: str) -> tuple[str, list[str], Path | None] | None:
        # Redirects and shells are not executed; only a single allowlisted argv.
        if any(tok in action for tok in ("<", ";", "|", "&", "`", "$(", "\n")):
            return None
        redir: Path | None = None
        body = action
        if ">" in action:
            if action.count(">") != 1:
                return None
            body, dest = action.split(">", 1)
            redir = self._confine(dest.strip())
            if redir is None:
                return None
        try:
            argv = shlex.split(body, posix=True)
        except ValueError:
            return None
        if not argv or argv[0] not in ALLOWED:
            return None
        return argv[0], argv[1:], redir

    def _run(self, argv0: str, args: list[str], redir: Path | None) -> str:
        if argv0 == "pwd":
            text = str(self.root) + "\n"
        elif argv0 == "ls":
            target = self.root
            if args:
                if len(args) != 1:
                    raise ValueError("ls")
                confined = self._confine(args[0])
                if confined is None or not confined.exists():
                    raise FileNotFoundError(args[0])
                target = confined
            if target.is_dir():
                names = sorted(p.name for p in target.iterdir())
                text = ("\n".join(names) + "\n") if names else ""
            else:
                text = target.name + "\n"
        elif argv0 == "cat":
            if len(args) != 1:
                raise ValueError("cat")
            confined = self._confine(args[0])
            if confined is None or not confined.is_file():
                raise FileNotFoundError(args[0])
            text = confined.read_text()
        elif argv0 == "echo":
            text = " ".join(args) + "\n"
        else:
            raise ValueError(argv0)
        if redir is not None:
            if not redir.is_relative_to(self.root):
                raise ValueError("redir")
            redir.parent.mkdir(parents=True, exist_ok=True)
            redir.write_text(text)
            return ""
        return text

    def step(self, action: str) -> Observation:
        action = action.strip()
        if not action:
            return Observation(text=self.reset(), refused=False)
        prompt = f"fly@box:{self.root.name}$ {action}\n"
        parsed = self._parse(action)
        if parsed is None:
            return Observation(text=prompt + "refused: escape\n", refused=True, exit_code=1)
        argv0, args, redir = parsed
        try:
            out = self._run(argv0, args, redir)
        except (OSError, ValueError):
            return Observation(text=prompt + "refused: escape\n", refused=True, exit_code=1)
        if out and not out.endswith("\n"):
            out += "\n"
        return Observation(text=prompt + out, refused=False, exit_code=0)
