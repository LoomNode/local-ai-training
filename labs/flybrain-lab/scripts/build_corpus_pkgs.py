#!/usr/bin/env python3
"""Build the byte corpus from a tree of unpacked source packages (one directory per package, as `apt-get source` leaves them:
<root>/<package>/<package>-<version>/...), split by whole package into train, dev and test, so a held-out number measures reading a
project the model never saw.

Per file the rules of build_corpus.py (source extensions, a per-file size window, the hex-dump filter, UTF-8 only), plus: a cap on
the bytes any one package contributes (LLVM, GCC and LibreOffice would otherwise be most of the corpus), with its own cap for a package
that should stay large (the kernel: every number so far is measured on kernel files, and the old 200 MB split is its training share
to beat, not to shrink); a distro's several versions of one project (gcc-11 and gcc-12, four LLVMs, the kernel's variants) count as one
family that shares a split side, a cap and an exclusion list, so a twin never trains while its sibling is held out; exact duplicates
dropped by content hash across the whole tree (vendored copies of the same file are everywhere in a distro; the first copy, in a
training package, wins, so a held-out package never repeats a training file), and an optional list of paths excluded from one
package (the files of an older corpus's held-out windows, kept clean for continuity). Files keep the header format every run reads:
`\\n// ==== <package>/<path inside the package> ====\\n`. Every split is shuffled at the file level with the seed, so the fixed dev
windows (a contiguous 2 MB slice) span many packages; a package's own files are found by their headers. A manifest beside the three
byte files records every package's split, family, bytes and file count.

    python scripts/build_corpus_pkgs.py /games/ailab/corpora/apt-src corpora/pkgs --max-bytes 3000000000 --per-package 20000000 \\
        --cap linux=1500000000 --holdout 0.05 --exclude linux=notes/speed-round/old_dev_test_paths.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
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


VERSIONED = re.compile(r"(-[0-9][0-9.]*|[0-9]+\.[0-9]+)$")    # gcc-12, llvm-toolchain-15, openjdk-17, mysql-8.0, python3.10, boost1.74


def family(pkg: str, pkg_dir: Path | None = None) -> str:
    """The project a package is a copy of: its name without a trailing version, and `linux` for any linux-* package whose tree is a
    kernel (linux-aws, linux-hwe-6.8: near-identical copies of the kernel). A distro ships several versions of the same project side
    by side; they share a split side, a cap and an exclusion list, so a twin never trains while its sibling is held out, and a dozen
    copies cannot each take a full cap. Names whose trailing digits are part of the name (libx11, pcre2, lz4) stay as they are."""
    name = VERSIONED.sub("", pkg) or pkg
    if name.startswith("linux-") and pkg_dir is not None:
        if any(t.is_dir() and (t / "Kbuild").is_file() and (t / "MAINTAINERS").is_file() for t in pkg_dir.iterdir()):
            return "linux"
    return name


def families(packages: list[str], root: Path | None = None) -> dict[str, str]:
    """Package -> family across a whole tree: family() per name, then names that still end in a bare number join the package with
    the same stem when the tree holds more than one of them (dotnet6 and dotnet8, pcre2 and pcre3, gnupg and gnupg2); a lone
    libx11 or lz4 stays its own family."""
    fam = {p: family(p, root / p if root is not None else None) for p in packages}
    stems: dict[str, list[str]] = {}
    for p, f in fam.items():
        m = re.fullmatch(r"(.*[^0-9])[0-9]+", f)
        if m:
            stems.setdefault(m.group(1), []).append(p)
    named = set(fam.values())
    for stem, members in stems.items():
        if len(members) > 1 or stem in named:
            for p in members:
                fam[p] = stem
    return fam


def split_packages(packages: list[str], holdout: float, seed: int) -> dict[str, str]:
    """Name -> split (names are families in build(), so twins land on the same side). A seeded shuffle; the first `holdout` of them are
    dev, the next `holdout` test, the rest train. A list too short for one held-out name of each kind gets one anyway from three up."""
    names = sorted(packages)
    random.Random(seed).shuffle(names)
    k = max(1, int(round(len(names) * holdout))) if len(names) >= 3 else 0
    return {**{n: "dev" for n in names[:k]}, **{n: "test" for n in names[k:2 * k]}, **{n: "train" for n in names[2 * k:]}}


def package_sizes(root: Path, exts: tuple[str, ...], min_size: int, max_size: int) -> dict[str, int]:
    """Bytes of candidate source per package (stat only, before the cap, the dedup and the hex filter): the numbers to set caps by."""
    sizes = {}
    for d in sorted(root.iterdir()):
        if d.is_dir() and any(c.is_dir() for c in d.iterdir()):
            sizes[d.name] = sum(p.stat().st_size for p in source_files(d, exts, min_size, max_size))
    return sizes


def build(root: Path, out: Path, exts: tuple[str, ...], min_size: int, max_size: int, per_package: int, max_bytes: int,
          holdout: float, seed: int, exclude: dict[str, set[str]], caps: dict[str, int] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    packages = [d.name for d in sorted(root.iterdir()) if d.is_dir() and any(c.is_dir() for c in d.iterdir())]
    fam = families(packages, root)
    fsplit = split_packages(sorted(set(fam.values())), holdout, seed)
    split = {pkg: fsplit[fam[pkg]] for pkg in packages}
    seen: set[str] = set()                       # content hashes, across every split: the first copy wins
    files: dict[str, list[tuple[str, bytes]]] = {"train": [], "dev": [], "test": []}
    manifest: dict[str, dict] = {}
    family_total: dict[str, int] = {}            # bytes kept per family: the cap, the exclusion list and the split are the family's
    for s in ("train", "dev", "test"):           # train first, so a held-out package never repeats a training file
        for pkg in packages:                     # sorted, so the plain name (linux, gcc) comes before its twins (linux-aws, gcc-12)
            if split[pkg] != s:
                continue
            pkg_dir = root / pkg
            f = fam[pkg]
            kept = dropped_dup = dropped_bad = dropped_excl = 0
            total = 0
            pkg_excl = exclude.get(f, set())
            cap = (caps or {}).get(f, per_package)
            candidates = source_files(pkg_dir, exts, min_size, max_size)
            available = sum(p.stat().st_size for p in candidates)   # what the package holds before the cap and the filters: the number to set caps by
            for p in candidates:
                if family_total.get(f, 0) >= cap:
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
                family_total[f] = family_total.get(f, 0) + len(b)
            manifest[pkg] = {"split": s, "family": f, "files": kept, "bytes": total, "dropped_duplicates": dropped_dup,
                             "dropped_binary_or_hex": dropped_bad, "dropped_excluded": dropped_excl, "available": available, "cap": cap,
                             "capped": family_total.get(f, 0) >= cap}
    for s, lst in files.items():                 # every split shuffled at the file level: a contiguous slice of dev (the fixed windows) spans many packages
        random.Random(seed).shuffle(lst)
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
    summary = {"root": str(root), "seed": seed, "holdout": holdout, "per_package": per_package, "caps": caps or {}, "max_bytes": max_bytes, "exts": list(exts),
               "splits": sizes, "packages": manifest}
    (out / "manifest.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def parse_caps(specs: list[str]) -> dict[str, int]:
    """--cap PACKAGE=BYTES: that package's own cap instead of --per-package."""
    out: dict[str, int] = {}
    for spec in specs:
        pkg, _, n = spec.partition("=")
        if not n.isdigit():
            raise SystemExit(f"--cap wants PACKAGE=BYTES, got {spec!r}")
        out[pkg] = int(n)
    return out


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
    ap.add_argument("--cap", action="append", default=[], help="PACKAGE=BYTES: that package's own cap (the kernel keeps its size this way)")
    ap.add_argument("--max-bytes", type=int, default=3_000_000_000, help="the training file's cap")
    ap.add_argument("--holdout", type=float, default=0.05, help="fraction of packages held out as dev, and the same again as test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", action="append", default=[], help="PACKAGE=FILE: paths inside that package to leave out (one per line)")
    ap.add_argument("--sizes-only", action="store_true", help="only list what each package holds (largest first) and write sizes.json; no corpus")
    a = ap.parse_args()
    if a.sizes_only:
        sizes = package_sizes(Path(a.root), tuple(a.exts.split(",")), a.min_size, a.max_size)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "sizes.json").write_text(json.dumps(sizes, indent=1) + "\n")
        total = sum(sizes.values())
        by_family: dict[str, list[str]] = {}
        for pkg, f in families(list(sizes), Path(a.root)).items():
            by_family.setdefault(f, []).append(pkg)
        print(f"{len(sizes)} packages in {len(by_family)} families, {total / 1e9:.2f} GB of candidate source in all "
              f"(files of {a.min_size:,}-{a.max_size:,} bytes, before dedup); the largest families, which is what a cap applies to:")
        for f, members in sorted(by_family.items(), key=lambda kv: -sum(sizes[m] for m in kv[1]))[:30]:
            print(f"{sum(sizes[m] for m in members) / 1e6:10.1f} MB  {f}" + (f"  ({', '.join(members)})" if len(members) > 1 else ""))
        print(f"sizes: {Path(a.out) / 'sizes.json'}")
        return
    s = build(Path(a.root), Path(a.out), tuple(a.exts.split(",")), a.min_size, a.max_size, a.per_package, a.max_bytes, a.holdout, a.seed, parse_exclude(a.exclude), parse_caps(a.cap))
    for k, v in s["splits"].items():
        print(f"{k}: {v['bytes']:,} bytes, {v['files']:,} files, {v['packages']} packages")
    print(f"manifest: {Path(a.out) / 'manifest.json'}")


if __name__ == "__main__":
    main()
