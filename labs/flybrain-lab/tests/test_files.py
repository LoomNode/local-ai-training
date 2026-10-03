"""File table, content hashes, same-directory lookups and splits of a header-delimited corpus."""

from __future__ import annotations


def _entries(spec):
    """FileEntry list from (path, start, end); bodies start 20 bytes after the header start."""
    from flybrain.data.files import FileEntry
    return [FileEntry(p, s, s + 20, e) for p, s, e in spec]


def test_file_table_round_trips_a_built_corpus(tmp_path):
    from flybrain.data.files import file_table, read_region
    from scripts.build_corpus import build
    src = tmp_path / "src"; (src / "d1").mkdir(parents=True); (src / "d2").mkdir()
    bodies = {"d1/a.c": b"int a(void){return 1;}\n" * 5, "d1/b.c": b"int b(void){return 2;}\n" * 7, "d2/c.py": b"def c():\n    return 3\n" * 4}
    for rel, b in bodies.items():
        (src / rel).write_bytes(b)
    out = tmp_path / "corpus.bytes"
    n_bytes, n_files = build(src, (".c", ".py"), 1, 100_000, 10_000_000, 0, out)
    table = file_table(out)
    assert n_files == 3 and len(table) == 3 and n_bytes == out.stat().st_size
    assert {e.path for e in table} == set(bodies)
    assert table[0].start == 0 and table[-1].end == n_bytes
    assert all(a.end == b.start for a, b in zip(table, table[1:]))
    for e in table:
        assert read_region(out, e.body, e.end) == bodies[e.path]
        assert read_region(out, e.start, e.body) == b"\n// ==== " + e.path.encode() + b" ====\n"


def test_test_corpus_writer_matches_the_real_corpus_format(tmp_path, corpus):
    from flybrain.data.files import file_table, read_region
    from scripts.build_corpus import build
    src = tmp_path / "src"; (src / "d").mkdir(parents=True)
    (src / "d" / "a.c").write_bytes(b"int a;\n" * 9); (src / "d" / "b.c").write_bytes(b"int b;\n" * 9)
    built = tmp_path / "built.bytes"
    build(src, (".c",), 1, 1000, 10**6, 0, built)
    order = [(e.path, read_region(built, e.body, e.end)) for e in file_table(built)]
    assert corpus(order, "mine.bytes").read_bytes() == built.read_bytes()


def test_header_at_end_of_file_and_header_cut_by_eof(tmp_path):
    from flybrain.data.files import file_table, read_stream
    h = lambda p: b"\n// ==== " + p + b" ====\n"
    whole = tmp_path / "whole.bytes"; whole.write_bytes(h(b"a/x.c") + b"AAA\n" + h(b"b/y.c"))   # last header, empty body
    t = file_table(whole)
    assert [e.path for e in t] == ["a/x.c", "b/y.c"] and t[0].end == t[1].start
    assert t[1].body == t[1].end == whole.stat().st_size
    stream, body_start, scored = read_stream(whole, t[1], 4096)
    assert scored == 0 and stream == h(b"b/y.c") and body_start == len(stream)
    cut = tmp_path / "cut.bytes"; cut.write_bytes(h(b"a/x.c") + b"AAA\n" + h(b"b/y.c")[:-5])      # header cut by EOF: not a file
    t2 = file_table(cut)
    assert [e.path for e in t2] == ["a/x.c"] and t2[0].end == cut.stat().st_size
    empty = tmp_path / "empty.bytes"; empty.write_bytes(b"")
    assert file_table(empty) == []


def test_read_region_reads_only_the_requested_bytes(tmp_path):
    import tracemalloc
    from flybrain.data.files import read_region
    big = tmp_path / "big.bytes"
    with open(big, "wb") as f:
        f.truncate(300_000_000)                      # sparse: no disk, no RAM
        f.seek(150_000_000); f.write(b"MARKER")
    tracemalloc.start()
    got = read_region(big, 150_000_000, 150_000_006)
    peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    assert got == b"MARKER" and peak < 1_000_000


def test_content_hash_depends_on_the_body_only(corpus):
    from flybrain.data.files import content_hash, file_table, hash_table
    p = corpus([("a/x.c", b"same body\n"), ("b/y.c", b"same body\n"), ("b/z.c", b"other body\n")])
    t = file_table(p); h = hash_table(p, t)
    assert h[0] == h[1] != h[2] and h[0] == content_hash(p, t[0])


def test_related_is_same_directory_inside_the_range_without_duplicates_sorted_by_path():
    from flybrain.data.files import related
    table = _entries([("d/c.c", 0, 100), ("d/a.c", 100, 200), ("d/dup.c", 200, 300), ("e/z.c", 300, 400),
                      ("d/b.c", 400, 500), ("d/late.c", 500, 700), ("d/self.c", 700, 800)])
    hashes = ["hc", "ha", "hself", "hz", "hb", "hl", "hself"]    # d/dup.c has the same content as the target d/self.c
    i = 6
    assert related(i, table, 0, 10_000, hashes) == [1, 4, 0, 5]  # a, b, c, late: not self, not the duplicate, not e/z.c
    assert related(i, table, 0, 600, hashes) == [1, 4, 0]        # d/late.c ends past hi: it straddles the range
    assert related(i, table, 100, 600, hashes) == [1, 4]         # d/c.c starts before lo


def test_targets_count_related_files_after_duplicates_are_removed():
    from flybrain.data.files import targets
    only_dup = _entries([("d/dup.c", 0, 100), ("d/t.c", 1000, 1100)])
    assert targets(only_dup, [1], ["ht", "ht"], min_related=1, train_end=500) == []      # the only same-directory file is a copy of the target
    three = _entries([("d/a.c", 0, 100), ("d/b.c", 100, 200), ("d/c.c", 200, 300), ("d/dup.c", 300, 400), ("d/t.c", 1000, 1100)])
    assert targets(three, [4], ["ha", "hb", "hc", "ht", "ht"], min_related=4, train_end=500) == []   # 3 distinct related files
    four = _entries([("d/a.c", 0, 100), ("d/b.c", 100, 200), ("d/c.c", 200, 300), ("d/e.c", 300, 400), ("d/dup.c", 400, 500), ("d/t.c", 1000, 1100)])
    assert targets(four, [5], ["ha", "hb", "hc", "he", "ht", "ht"], min_related=4, train_end=500) == [5]
    assert targets(four, [5], ["ha", "hb", "hc", "he", "ht", "ht"], min_related=4, train_end=350) == []  # d/e.c ends past the training range


def test_splits_take_the_first_dev_files_of_the_tail_in_file_order():
    from flybrain.data.files import DEV_FILES, MIN_NEVER, TAIL_START, TRAIN_END, split_range, splits
    table = _entries([("a", 0, 100), ("b", 100, 200), ("c", 200, 300), ("d", 300, 400), ("e", 400, 500), ("f", 500, 600)])
    sp = splits(table, tail_start=200, dev_files=2)
    assert sp == {"dev": [2, 3], "test": [4, 5]}
    assert split_range(table, sp["dev"]) == (200, 400) and split_range(table, sp["test"]) == (400, 600)
    assert (TRAIN_END, TAIL_START, DEV_FILES, MIN_NEVER) == (200_000_000, 202_100_000, 500, 2)


def test_read_stream_is_the_header_plus_at_most_n_scored_body_bytes(corpus):
    from flybrain.data.files import file_table, read_stream
    p = corpus([("a/x.c", b"0123456789" * 10), ("a/y.c", b"short")])
    t = file_table(p)
    head = b"\n// ==== a/x.c ====\n"
    s, bs, n = read_stream(p, t[0], 25)
    assert n == 25 and bs == len(head) and s == head + (b"0123456789" * 3)[:25]
    s2, bs2, n2 = read_stream(p, t[1], 25)                     # body shorter than the cap: the whole body, nothing padded
    assert n2 == 5 and s2.endswith(b"short") and len(s2) == bs2 + 5
