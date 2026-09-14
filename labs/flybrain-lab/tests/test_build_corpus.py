from pathlib import Path


def test_build_corpus_filters_and_shuffles(tmp_path: Path):
    from scripts.build_corpus import build, looks_like_hex_dump
    src = tmp_path / "src"; src.mkdir()
    (src / "a.c").write_bytes(b"int main(){return 0;}\n" * 200)
    (src / "b.py").write_bytes(b"def f():\n    return 1\n" * 200)
    (src / "tiny.c").write_bytes(b"x")
    (src / "dump.c").write_bytes(b"0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,\n" * 200)
    (src / "skip.md").write_bytes(b"# not code\n" * 200)
    assert looks_like_hex_dump((src / "dump.c").read_bytes()) and not looks_like_hex_dump((src / "a.c").read_bytes())
    out = tmp_path / "out.bytes"
    n, files = build(src, (".c", ".py"), 100, 100_000, 10_000_000, 0, out)
    b = out.read_bytes()
    assert files == 2 and b"int main" in b and b"def f" in b and b"0xaa" not in b and b"not code" not in b and n == len(b)
    n2, _ = build(src, (".c", ".py"), 100, 100_000, 10_000_000, 1, tmp_path / "out2.bytes")
    assert n2 == n and (tmp_path / "out2.bytes").read_bytes() != b  # different seed, different file order
