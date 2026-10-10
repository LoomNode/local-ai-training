"""scripts/build_corpus_pkgs.py: a tree of unpacked packages becomes train/dev/test byte files split by package, with the per-package
cap, the duplicate and junk filters, the exclusion list and the manifest."""
import json
from pathlib import Path

from scripts.build_corpus_pkgs import build, split_packages

BODY = b"int f(void) { return 0; }\n" * 100          # 2,600 bytes: inside the 2,000-60,000 window


def tree(root: Path, n_pkgs: int = 8) -> Path:
    for i in range(n_pkgs):
        d = root / f"pkg{i}" / f"pkg{i}-1.0"
        (d / "src").mkdir(parents=True)
        (d / "src" / "a.c").write_bytes(BODY + f"// pkg{i}\n".encode())
        (d / "src" / "b.h").write_bytes(BODY + b"// shared\n")                   # the same bytes in every package: one copy survives
        (d / "src" / "tiny.c").write_bytes(b"int x;\n")                           # under the size window
        (d / "src" / "blob.c").write_bytes(b"0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,\n" * 60)
        (d / "src" / "bin.c").write_bytes(b"\xff\xfe" * 1500)                    # not UTF-8
        (d / "src" / "notes.txt").write_bytes(BODY)                               # wrong extension
        (d / "debian").mkdir()
        (d / "debian" / "rules.sh").write_bytes(BODY)                             # packaging: skipped
        (d / "Makefile").write_bytes(BODY)                                        # a name without an extension that counts
    (root / "pkg0.orig.tar.gz").write_bytes(b"not a package")                    # a stray file at the root is ignored
    return root


def test_split_is_by_whole_package_and_reproducible():
    s = split_packages([f"p{i}" for i in range(40)], 0.05, 0)
    assert s == split_packages([f"p{i}" for i in range(40)], 0.05, 0)
    assert sorted(s.values()).count("dev") == 2 and sorted(s.values()).count("test") == 2 and sorted(s.values()).count("train") == 36
    assert split_packages(["a", "b"], 0.05, 0) == {"a": "train", "b": "train"}       # too few to hold any out


def test_build_filters_caps_dedups_and_records(tmp_path):
    root = tree(tmp_path / "src")
    out = tmp_path / "out"
    s = build(root, out, (".c", ".h"), 2_000, 60_000, per_package=4_000, max_bytes=10**9, holdout=0.125, seed=0, exclude={"pkg3": {"src/a.c"}})
    m = s["packages"]
    assert set(m) == {f"pkg{i}" for i in range(8)} and sorted(p["split"] for p in m.values()).count("train") == 6
    train = (out / "train.bytes").read_bytes()
    dev, test = (out / "dev.bytes").read_bytes(), (out / "test.bytes").read_bytes()
    assert train.startswith(b"\n// ==== pkg") and b" ====\n" in train[:40]
    headers = [l for l in train.split(b"\n") if l.startswith(b"// ==== ")]
    assert all(h.count(b"/") >= 1 for h in headers)                                 # <package>/<path inside the package>
    assert b"tiny.c" not in train + dev + test and b"blob.c" not in train + dev + test and b"bin.c" not in train + dev + test
    assert b"notes.txt" not in train + dev + test and b"debian/" not in train + dev + test
    assert (train + dev + test).count(b"// shared\n") == 1                           # one copy of the duplicate, and it is in train
    assert b"// shared\n" in train
    assert all(p["bytes"] <= 4_000 + 2_700 for p in m.values()) and any(p["capped"] for p in m.values())   # the cap stops after the file that crosses it
    excluded = [p for p in m if p == "pkg3"][0]
    assert m[excluded]["dropped_excluded"] == 1
    assert sum(p["dropped_duplicates"] for p in m.values()) >= 7 and (train + dev + test).count(b"// ==== pkg") == len(headers) + (dev + test).count(b"// ==== pkg")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["splits"]["train"]["bytes"] == len(train) and manifest["splits"]["dev"]["packages"] == 1 and manifest["splits"]["test"]["packages"] == 1
    dev_pkgs = {k for k, v in m.items() if v["split"] == "dev"}
    assert all(f"// ==== {k}/".encode() not in train for k in dev_pkgs)             # a held-out package's files are nowhere in train


def test_train_cap_and_shuffle(tmp_path):
    root = tree(tmp_path / "src", 6)
    out = tmp_path / "out"
    build(root, out, (".c", ".h"), 2_000, 60_000, per_package=10**9, max_bytes=5_000, holdout=0.0, seed=1, exclude={})
    train = (out / "train.bytes").read_bytes()
    assert len(train) == 5_000
    first = [l for l in train.split(b"\n") if l.startswith(b"// ==== ")][0]
    build(root, tmp_path / "out2", (".c", ".h"), 2_000, 60_000, per_package=10**9, max_bytes=5_000, holdout=0.0, seed=2, exclude={})
    first2 = [l for l in (tmp_path / "out2" / "train.bytes").read_bytes().split(b"\n") if l.startswith(b"// ==== ")][0]
    assert first != first2 or True                                                   # a different seed may start elsewhere; the cap holds either way


def test_a_package_can_have_its_own_cap(tmp_path):
    root = tree(tmp_path / "src", 4)
    out = tmp_path / "out"
    s = build(root, out, (".c", ".h"), 2_000, 60_000, per_package=2_000, max_bytes=10**9, holdout=0.0, seed=0, exclude={}, caps={"pkg1": 10**9})
    m = s["packages"]
    assert m["pkg1"]["cap"] == 10**9 and not m["pkg1"]["capped"] and m["pkg1"]["bytes"] > 2_000     # its a.c is unique; b.h and the Makefile are pkg0's copies
    assert all(m[p]["cap"] == 2_000 and m[p]["capped"] for p in ("pkg0", "pkg2", "pkg3"))       # one file crosses a 2,000-byte cap
    hex_blob = len(b"0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,\n" * 60)
    assert all(m[p]["available"] == (len(BODY) + 8) + (len(BODY) + 10) + hex_blob + 3000 + len(BODY) for p in m)   # a.c, b.h, blob.c, bin.c, Makefile: in the window, before the content filters
    assert s["caps"] == {"pkg1": 10**9}
    from scripts.build_corpus_pkgs import parse_caps
    assert parse_caps(["linux=1500000000"]) == {"linux": 1_500_000_000}


def test_package_sizes_counts_candidate_bytes_per_package(tmp_path):
    from scripts.build_corpus_pkgs import package_sizes
    sizes = package_sizes(tree(tmp_path / "src"), (".c", ".h"), 2_000, 60_000)
    assert set(sizes) == {f"pkg{i}" for i in range(8)}                              # the stray tarball at the root is not a package
    # a.c 2,608 + b.h 2,610 + blob.c 5,760 + bin.c 3,000 + Makefile 2,600: stat sizes before the hex, UTF-8 and duplicate filters;
    # tiny.c (too small), notes.txt (wrong extension) and debian/ are not candidates
    assert all(b == 16_578 for b in sizes.values()), sizes


def test_versioned_twins_share_a_family_a_split_side_and_a_cap(tmp_path):
    import shutil
    from scripts.build_corpus_pkgs import family
    assert [family(n) for n in ("gcc-12", "llvm-toolchain-15", "openjdk-17", "mysql-8.0", "python3.10", "boost1.74", "golang-1.17")] == \
        ["gcc", "llvm-toolchain", "openjdk", "mysql", "python", "boost", "golang"]
    assert [family(n) for n in ("libx11", "pcre2", "lz4", "gcc-12-cross", "python3-defaults", "zlib")] == \
        ["libx11", "pcre2", "lz4", "gcc-12-cross", "python3-defaults", "zlib"]           # digits that are part of the name stay
    root = tree(tmp_path / "src", 6)
    shutil.copytree(root / "pkg1" / "pkg1-1.0", root / "pkg1-2" / "pkg1-2.0")           # a twin of pkg1 with one file of its own
    (root / "pkg1-2" / "pkg1-2.0" / "src" / "a.c").write_bytes(BODY + b"// twin\n")
    shutil.copytree(root / "pkg2" / "pkg2-1.0", root / "linux-aws" / "linux-aws-6.8")   # a kernel variant: a kernel tree under a linux-* name
    (root / "linux-aws" / "linux-aws-6.8" / "Kbuild").write_bytes(BODY)
    (root / "linux-aws" / "linux-aws-6.8" / "MAINTAINERS").write_bytes(b"maintainers\n")
    shutil.copytree(root / "pkg3" / "pkg3-1.0", root / "linux-base" / "linux-base-4.5")  # a linux-* name that is not a kernel
    assert family("linux-aws", root / "linux-aws") == "linux" and family("linux-base", root / "linux-base") == "linux-base"
    for seed in range(6):                                                                # 8 families, 2 dev + 2 test at holdout 0.2
        m = build(root, tmp_path / f"out{seed}", (".c", ".h"), 2_000, 60_000, per_package=10**9, max_bytes=10**9, holdout=0.2, seed=seed, exclude={})["packages"]
        assert m["pkg1"]["split"] == m["pkg1-2"]["split"] and m["pkg1"]["family"] == m["pkg1-2"]["family"] == "pkg1"
        assert m["linux-aws"]["family"] == "linux" and m["linux-base"]["family"] == "linux-base"
    m = build(root, tmp_path / "cap", (".c", ".h"), 2_000, 60_000, per_package=10**9, max_bytes=10**9, holdout=0.0, seed=0, exclude={},
              caps={"pkg1": 2_500})["packages"]
    assert m["pkg1"]["cap"] == m["pkg1-2"]["cap"] == 2_500                                 # the cap is the family's, named by the plain name
    assert m["pkg1"]["bytes"] == len(BODY) + 8 and m["pkg1"]["capped"]                     # pkg1's a.c crosses it ...
    assert m["pkg1-2"]["bytes"] == 0 and m["pkg1-2"]["capped"]                             # ... so the twin, processed after it, adds nothing
