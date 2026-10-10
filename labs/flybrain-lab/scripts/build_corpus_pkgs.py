#!/usr/bin/env python3
"""Build the byte corpus from a tree of unpacked source packages (one directory per package, as `apt-get source` leaves them:
<root>/<package>/<package>-<version>/...), split by whole package into train, dev and test, so a held-out number measures reading a
project the model never saw.

Per file the rules of build_corpus.py (source extensions, a per-file size window, the hex-dump filter, UTF-8 only), plus: a cap on
the bytes any one package contributes (LLVM, GCC, LibreOffice and the kernel would otherwise be most of the corpus), exact duplicates
dropped by content hash across the whole tree (vendored copies of the same file are everywhere in a distro; the first copy, in a
training package, wins, so a held-out package never repeats a training file), and an optional list of paths excluded from one
package (the files of an older corpus's held-out windows, kept clean for continuity). Files keep the header format every run reads:
`\\n// ==== <package>/<path inside the package> ====\\n`. The training files are shuffled with the seed, the held-out ones kept in
package order. A manifest beside the three byte files records every package's split, bytes and file count.

    python scripts/build_corpus_pkgs.py /games/ailab/corpora/apt-src corpora/pkgs --max-bytes 3000000000 --per-package 20000000 \\
        --holdout 0.05 --exclude linux=notes/speed-round/old_dev_test_paths.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_corpus import looks_like_hex_dump  # noqa: E402

EXTS = (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".py", ".rs", ".go", ".sh", ".bash", ".pl", ".pm", ".js", ".ts", ".java", ".rb",
        ".lua", ".cs", ".zig", ".gd", ".m", ".mm", ".S", ".s", ".mk", ".cmake", ".toml", ".vala", ".ml", ".hs", ".el", ".tcl")
NAMES = ("Makefile", "Kbuild", "Kconfig", "CMakeLists.txt", "meson.build", "configure.ac")
SKIP_DIRS = {".git", ".svn", ".hg", "debian", "node_modules", "__pycache__", "testdata", "test-data", "third_party", "thirdparty"}


def source_files(pkg_dir: Path, exts: tuple[str, ...], min_size: int, max_size: int) -> list[Path]:
    """The candidate files of one package directory, sorted: the right extension or name, inside the size window, not under a skipped
    directory (the Debian packaging, vendored trees, test data)."""
    out = []
    for p in sorted(pkg_dir.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(pkg_dir).parts[:-1]):
            continue
        if p.suffix not in exts and p.name not in NAMES:
            continue
        size = p.stat().st_size
        if min_size <= size <= max_size:
            out.append(p)
    return out


def read_source(p: Path) -> bytes | None:
    """The file's bytes if it is UTF-8 text and not a hex dump, else None."""
    b = p.read_bytes()
    if looks_like_hex_dump(b):
        return None
    try:
        b.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return b


def split_packages(packages: list[str], holdout: float, seed: int) -> dict[str, str]:
    """Package -> split. A seeded shuffle of the names; the first `holdout` of them are dev, the next `holdout` test, the rest train.
    A list too short for one held-out package of each kind gets one anyway when it has at least three packages."""
    names = sorted(packages)
    random.Random(seed).shuffle(names)
    k = max(1, int(round(len(names) * holdout))) if len(names) >= 3 else 0
    return {**{n: "dev" for n in names[:k]}, **{n: "test" for n in names[k:2 * k]}, **{n: "train" for n in names[2 * k:]}}


def build(root: Path, out: Path, exts: tuple[str, ...], min_size: int, max_size: int, per_package: int, max_bytes: int,
          holdout: float, seed: int, exclude: dict[str, set[str]]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    packages = [d.name for d in sorted(root.iterdir()) if d.is_dir() and any(c.is_dir() for c in d.iterdir())]
    split = split_packages(packages, holdout, seed)
    seen: set[str] = set()                       # content hashes, across every split: the first copy wins
    files: dict[str, list[tuple[str, bytes]]] = {"train": [], "dev": [], "test": []}
    manifest: dict[str, dict] = {}
    for s in ("train", "dev", "test"):           # train first, so a held-out package never repeats a training file
        for pkg in packages:
            if split[pkg] != s:
                continue
            pkg_dir = root / pkg
            kept = dropped_dup = dropped_bad = dropped_excl = 0
            total = 0
            pkg_excl = exclude.get(pkg, set())
            for p in source_files(pkg_dir, exts, min_size, max_size):
                if total >= per_package:
                    break
                rel = p.relative_to(pkg_dir)
                inner = Path(*rel.parts[1:]) if len(rel.parts) > 1 else rel    # the path inside <package>-<version>/
                if str(inner) in pkg_excl:
                    dropped_excl += 1
                    continue
                b = read_source(p)
                if b is None:
                    dropped_bad += 1
                    continue
                h = hashlib.sha1(b).hexdigest()
                if h in seen:
                    dropped_dup += 1
                    continue
                seen.add(h)
                files[s].append((f"{pkg}/{inner}", b))
                kept += 1
                total += len(b)
            manifest[pkg] = {"split": s, "files": kept, "bytes": total, "dropped_duplicates": dropped_dup, "dropped_binary_or_hex": dropped_bad,
                             "dropped_excluded": dropped_excl, "capped": total >= per_package}
    random.Random(seed).shuffle(files["train"])
    sizes = {}
    for s, lst in files.items():
        buf = bytearray()
        for path, b in lst:
            buf += b"\n// ==== " + path.encode() + b" ====\n" + b
            if s == "train" and len(buf) >= max_bytes:
                break
        if s == "train":
            buf = buf[:max_bytes]
        (out / f"{s}.bytes").write_bytes(bytes(buf))
        sizes[s] = {"bytes": len(buf), "files": len(lst), "packages": sum(1 for p in manifest.values() if p["split"] == s)}
    summary = {"root": str(root), "seed": seed, "holdout": holdout, "per_package": per_package, "max_bytes": max_bytes, "exts": list(exts),
               "splits": sizes, "packages": manifest}
    (out / "manifest.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def parse_exclude(specs: list[str]) -> dict[str, set[str]]:
    """--exclude PACKAGE=FILE, one path inside the package per line."""
    out: dict[str, set[str]] = {}
    for spec in specs:
        pkg, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--exclude wants PACKAGE=FILE, got {spec!r}")
        out[pkg] = {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--exts", default=",".join(EXTS))
    ap.add_argument("--min-size", type=int, default=2_000)
    ap.add_argument("--max-size", type=int, default=60_000)
    ap.add_argument("--per-package", type=int, default=20_000_000, help="bytes any one package may contribute")
    ap.add_argument("--max-bytes", type=int, default=3_000_000_000, help="the training file's cap")
    ap.add_argument("--holdout", type=float, default=0.05, help="fraction of packages held out as dev, and the same again as test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", action="append", default=[], help="PACKAGE=FILE: paths inside that package to leave out (one per line)")
    a = ap.parse_args()
    s = build(Path(a.root), Path(a.out), tuple(a.exts.split(",")), a.min_size, a.max_size, a.per_package, a.max_bytes, a.holdout, a.seed, parse_exclude(a.exclude))
    for k, v in s["splits"].items():
        print(f"{k}: {v['bytes']:,} bytes, {v['files']:,} files, {v['packages']} packages")
    print(f"manifest: {Path(a.out) / 'manifest.json'}")


if __name__ == "__main__":
    main()
