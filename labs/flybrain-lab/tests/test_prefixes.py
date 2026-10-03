"""Context material for held-out targets: the leakage rules, the byte cut and the equal-size random control."""

from __future__ import annotations


def _body(world, name):
    from flybrain.data.files import read_region
    e = world.table[world.idx[name]]
    return read_region(world.path, e.body, e.end)


def _region(world, name):
    from flybrain.data.files import read_region
    e = world.table[world.idx[name]]
    return read_region(world.path, e.start, e.end)


def test_prefixes_never_contain_the_target_or_a_duplicate_of_it(world):
    from flybrain.data.files import related
    from flybrain.data.prefixes import prefix_pairs, random_order, reader_chunks
    i = world.idx["k/t0.c"]
    needle = _body(world, "k/t0.c")[200:260]      # inside the random noise, which no other file shares
    assert needle in _body(world, "y/twin.c")                      # the identical body sits in another directory
    for seed in range(20):
        rel, rnd = prefix_pairs(world.path, world.table, i, world.hashes, (100_000,), seed, 0, world.train_end, world.train_end)[100_000]
        assert needle not in rel and needle not in rnd and len(rel) > 0
        assert world.idx["y/twin.c"] not in random_order(world.table, i, world.hashes, seed, world.train_end)
    j, twin = world.idx["k/t1.c"], world.idx["k/a_twin.c"]                      # a same-directory training copy of k/t1.c
    needle_j = _body(world, "k/t1.c")[200:260]
    assert needle_j in _body(world, "k/a_twin.c")
    rel, rnd = prefix_pairs(world.path, world.table, j, world.hashes, (100_000,), 0, 0, world.train_end, world.train_end)[100_000]
    assert needle_j not in rel and needle_j not in rnd
    assert twin not in related(j, world.table, 0, world.train_end, world.hashes)
    assert twin not in {k for k, _ in reader_chunks(world.path, world.table, j, world.hashes, 10**9, 10**9, 0, world.train_end)}
    assert twin in related(i, world.table, 0, world.train_end, world.hashes)    # but it is ordinary material for a different target


def test_random_prefix_has_no_same_directory_file_and_no_file_outside_the_training_range(world):
    from flybrain.data.files import splits
    from flybrain.data.prefixes import random_order
    sp = splits(world.table, world.tail_start, world.dev_files)
    for i in sp["dev"] + sp["test"]:
        order = random_order(world.table, i, world.hashes, 0, world.train_end)
        assert order and i not in order
        assert all(world.table[j].dirname != world.table[i].dirname for j in order)
        assert all(world.table[j].end <= world.train_end for j in order)
        assert all(world.hashes[j] != world.hashes[i] for j in order)
        assert world.idx["k/straddle.c"] not in order and world.idx["k/t3.c"] not in order


def test_random_order_is_deterministic_for_a_seed_and_changes_with_it(world):
    from flybrain.data.prefixes import random_order
    i = world.idx["k/t1.c"]
    a = random_order(world.table, i, world.hashes, 0, world.train_end)
    assert a == random_order(world.table, i, world.hashes, 0, world.train_end)
    assert any(random_order(world.table, i, world.hashes, s, world.train_end) != a for s in range(1, 6))
    j = world.idx["k/t2.c"]                                                   # same directory, same pool, different target: its own shuffle
    assert sorted(random_order(world.table, j, world.hashes, 0, world.train_end)) == sorted(a)
    assert random_order(world.table, j, world.hashes, 0, world.train_end) != a


def test_chunks_cut_at_the_limit_inside_a_file_and_inside_a_header(world):
    from flybrain.data.prefixes import chunks, joined
    a0, a1 = world.idx["k/a0.c"], world.idx["k/a1.c"]
    full0 = _region(world, "k/a0.c")
    cut = chunks(world.path, world.table, [a0, a1], len(full0) + 10)       # 10 bytes into the next file: inside its header
    assert [j for j, _ in cut] == [a0, a1]
    assert joined(cut) == full0 + b"\n// ==== k"
    assert joined(chunks(world.path, world.table, [a0, a1], 5)) == full0[:5]
    assert chunks(world.path, world.table, [a0, a1], 0) == []
    assert len(joined(chunks(world.path, world.table, [a0, a1], 10**9))) == len(full0) + len(_region(world, "k/a1.c"))


def test_prefix_pairs_are_nested_and_the_random_prefix_has_the_related_length(world):
    from flybrain.data.prefixes import prefix_pairs
    i = world.idx["k/t0.c"]
    pairs = prefix_pairs(world.path, world.table, i, world.hashes, (0, 128, 256, 100_000), 0, 0, world.train_end, world.train_end)
    assert pairs[0] == (b"", b"")                                           # P = 0 is the empty prefix
    assert pairs[256][0].startswith(pairs[128][0]) and pairs[256][1].startswith(pairs[128][1])
    assert all(len(r) == len(q) == p for p, (r, q) in pairs.items() if p in (128, 256))
    material = sum(world.table[j].end - world.table[j].start for j in range(len(world.table)) if world.table[j].dirname == "k" and world.table[j].end <= world.train_end)
    assert len(pairs[100_000][0]) == material                               # fewer related bytes than P: the whole related set
    assert len(pairs[100_000][1]) == material                               # and a random prefix of the same size, not of size P


def test_never_trained_files_come_from_the_targets_own_split_only(world):
    from flybrain.data.files import related, splits
    from flybrain.data.prefixes import prefix_pairs
    sp = splits(world.table, world.tail_start, world.dev_files)
    ranges = {s: (world.table[sp[s][0]].start, world.table[sp[s][-1]].end) for s in sp}
    for s in sp:
        lo, hi = ranges[s]
        for i in sp[s]:
            assert set(related(i, world.table, lo, hi, world.hashes)) <= set(sp[s]) - {i}
    lo, hi = ranges["dev"]
    i = world.idx["k/t0.c"]
    assert [world.table[j].path for j in related(i, world.table, lo, hi, world.hashes)] == ["k/t1.c", "k/t2.c"]
    rel, rnd = prefix_pairs(world.path, world.table, i, world.hashes, (100_000,), 0, lo, hi, world.train_end)[100_000]
    assert rel == _region(world, "k/t1.c") + _region(world, "k/t2.c") and len(rnd) == len(rel)
    assert _body(world, "k/t3.c")[200:260] not in rel                       # a test-split file never reaches a dev prefix


def test_prefix_pairs_draws_its_random_side_from_the_training_range_for_every_seed(world):
    from flybrain.data.files import splits
    from flybrain.data.prefixes import chunks, joined, prefix_pairs, random_order
    sp = splits(world.table, world.tail_start, world.dev_files)
    dev = (world.table[sp["dev"][0]].start, world.table[sp["dev"][-1]].end)
    i = world.idx["k/t0.c"]
    for lo, hi in ((0, world.train_end), dev):                      # the related side is bounded by [lo, hi), the random side only by train_end
        for seed in range(3):
            order = random_order(world.table, i, world.hashes, seed, world.train_end)    # training-range files outside k/: never the tail's z/t4.c or z/t5.c
            for p, (rel, rnd) in prefix_pairs(world.path, world.table, i, world.hashes, (700, 2500, 100_000), seed, lo, hi, world.train_end).items():
                assert len(rel) > 0 and rnd == joined(chunks(world.path, world.table, order, len(rel))), (lo, seed, p)


def test_reader_chunks_cap_each_file_and_the_total_budget(world):
    from flybrain.data.prefixes import reader_chunks
    i = world.idx["k/t1.c"]
    cs = reader_chunks(world.path, world.table, i, world.hashes, budget=1000, per_file=300, lo=0, hi=world.train_end)
    assert [len(b) for _, b in cs] == [300, 300, 300, 100]
    assert [world.table[j].path for j, _ in cs] == ["k/a0.c", "k/a1.c", "k/a2.c", "k/a3.c"]
    assert all(b.startswith((b"\n// ==== " + world.table[j].path.encode() + b" ====\n")[:20]) for j, b in cs)
    cs = reader_chunks(world.path, world.table, i, world.hashes, 10**9, 10**9, 0, world.train_end)    # no cut at all: the whole related set
    assert [world.table[j].path for j, _ in cs] == [f"k/a{n}.c" for n in range(5)]                      # not the target, its copy k/a_twin.c, the straddler or a tail file


def test_reader_chunks_stop_at_a_files_end_and_use_the_given_range(world):
    from flybrain.data.files import splits
    from flybrain.data.prefixes import joined, reader_chunks
    sp = splits(world.table, world.tail_start, world.dev_files)
    lo, hi = world.table[sp["dev"][0]].start, world.table[sp["dev"][-1]].end
    cs = reader_chunks(world.path, world.table, world.idx["k/t0.c"], world.hashes, lo=lo, hi=hi)       # the default budget and per_file dwarf these files
    assert [j for j, _ in cs] == [world.idx["k/t1.c"], world.idx["k/t2.c"]]                             # the target's own split only
    assert joined(cs) == _region(world, "k/t1.c") + _region(world, "k/t2.c")                            # a chunk ends where its file ends, not in the next file
