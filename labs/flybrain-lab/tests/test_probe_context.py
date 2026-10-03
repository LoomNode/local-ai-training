"""scripts/probe_commons.py --mode context: scoring primitives, the pooled numbers and the end-to-end pipeline on a tiny random model."""

from __future__ import annotations

import math

import numpy as np
import torch


class Bigram(torch.nn.Module):
    """Logits depend only on the previous byte, so the loss of a byte is the same under any prefix."""

    def __init__(self):
        super().__init__()
        self.table = torch.randn(256, 256, generator=torch.Generator().manual_seed(0))

    def forward(self, x):
        return self.table[x], None


def test_scored_bits_are_the_loss_of_the_right_bytes_under_any_prefix():
    from scripts.probe_commons import score_with_prefix
    m = Bigram()
    head = b"\n// ==== a/x.c ====\n"
    stream = head + b"hello world\nint x;\n" * 4
    alone = score_with_prefix(m, b"", stream, len(head), 40, "cpu")
    for prefix in (b"some other material\n" * 3, b"\n// ==== b/y.c ====\nzzz", b"q"):
        assert np.array_equal(score_with_prefix(m, prefix, stream, len(head), 40, "cpu"), alone)   # a shifted window would score other bytes
    want = [float(torch.nn.functional.cross_entropy(m.table[stream[t - 1]][None], torch.tensor([stream[t]]))) / math.log(2) for t in range(len(head), len(head) + 40)]
    assert len(alone) == 40 and np.allclose(alone, want, atol=1e-6)                               # bits of byte t given byte t - 1


def test_rope_scale_follows_each_sequence_and_does_not_leak():
    from flybrain.model.hybrid import GRUHybrid
    from scripts.probe_commons import byte_bits
    torch.manual_seed(0)
    m = GRUHybrid(d_model=32, hidden=32, key="sparse", where="none", backbone="transformer", n_layers=2, d_ff=64, rope=True, max_len=64).eval()
    short, long = bytes(range(40)), bytes(range(200))
    a = byte_bits(m, short, "cpu", rope_train_len=64)                       # shorter than the training length: scale 1
    base_short = m.core.attn[0].rope_base
    b = byte_bits(m, long, "cpu", rope_train_len=64)                        # 200 / 64 x the training length
    base_long = m.core.attn[0].rope_base
    c = byte_bits(m, short, "cpu", rope_train_len=64)
    assert base_short == 10000.0 and abs(base_long - 10000.0 * (200 / 64) ** (32 / 30)) < 1e-3 and m.core.attn[0].rope_base == 10000.0
    assert np.array_equal(a[1:], c[1:]) and np.isnan(a[0]) and len(b) == 200


def test_apply_chunk_sets_every_head_and_leaves_headless_models_alone(tiny_hybrid):
    from flybrain.model.hybrid import GRUHybrid
    from scripts.probe_commons import apply_chunk, arm_settings
    m = tiny_hybrid()
    assert [h.chunk for h in m.all_heads()] == [8, 8]
    apply_chunk(m, 0); assert [h.chunk for h in m.all_heads()] == [8, 8]    # 0 keeps the checkpoint's own chunk
    apply_chunk(m, 4096); assert [h.chunk for h in m.all_heads()] == [4096, 4096]
    apply_chunk(GRUHybrid(d_model=16, hidden=16, key="sparse", where="none", backbone="transformer", n_layers=2, d_ff=32, rope=True), 4096)
    assert arm_settings("transformer", [8192, 16384], 4096) == ([2048], 2048)
    assert arm_settings("mingru_sparse_post", [8192, 16384], 4096) == ([8192, 16384], 4096)


def test_the_first_2048_scope_keeps_only_the_first_scored_bytes():
    from scripts.probe_commons import COND_IX, PART_IX, fill, new_cube
    a, r = COND_IX["alone"], PART_IX["related"]
    for n in (90, 2048, 2049, 3000):                                          # a short body, exactly the cap, one byte over it, a long body
        cube = new_cube(1)
        fill(cube, 0, "related", np.zeros(n, dtype=int), {c: np.arange(float(n)) for c in ("alone", "related", "random")})   # byte i costs i bits
        cut = min(n, 2048)
        assert cube[0, 0, r, a, 0, 0] == n and cube[1, 0, r, a, 0, 0] == cut                                 # the two scopes coincide up to the cap
        assert cube[0, 0, r, a, 0, 1] == sum(range(n)) and cube[1, 0, r, a, 0, 1] == sum(range(cut))         # bytes 0..cut-1, not the last cut bytes
    cube = new_cube(1)
    fill(cube, 0, "related", np.arange(3000) % 3, {c: np.arange(3000.0) for c in ("alone", "related", "random")})   # codes cycle A B C A B C ...
    for k in range(3):                                                        # the codes are cut at the same byte as the bits
        assert cube[1, 0, r, a, k, 0] == len(range(k, 2048, 3)) and cube[1, 0, r, a, k, 1] == sum(range(k, 2048, 3))
    assert [cube[1, 0, r, a, k, 0] for k in range(3)] == [683, 683, 682]      # 2048 = 3 x 682 + 2
    assert cube[1, 0, r, a, :, 1].sum() == 2_096_128                          # 0 + 1 + ... + 2047


def test_headroom_is_share_times_the_loss_gap_and_the_gate_is_their_difference():
    from scripts.probe_commons import fill, headroom, new_cube
    alone = np.array([1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0])
    bits = {"alone": alone, "related": alone - 0.5, "random": alone}
    cube = new_cube(1)
    fill(cube, 0, "related", [0, 0, 1, 1, 1, 1, 2, 2, 2, 2], bits)           # A A B B B B C C C C
    fill(cube, 0, "random", [0, 0, 2, 2, 2, 2, 1, 2, 2, 2], bits)            # A A C' C' C' C' B' C' C' C': B' is a byte that costs 7, not 5
    h = headroom(cube[0, 0])
    assert (h["share_B"], h["share_Bp"]) == (0.4, 0.1) and (h["loss_A"], h["loss_B"], h["loss_Bp"]) == (1.0, 5.0, 7.0)
    assert abs(h["H_B"] - 0.4 * 4.0) < 1e-12 and abs(h["H_Bp"] - 0.1 * 6.0) < 1e-12 and abs(h["gate"] - 1.0) < 1e-12   # 1.6 - 0.6
    assert h["share_Bn"] is None and h["H_Bn"] is None                       # nobody filled the never-trained partition
    no_a = new_cube(1)
    fill(no_a, 0, "related", [1, 2], {"alone": np.ones(2), "related": np.ones(2), "random": np.ones(2)})
    fill(no_a, 0, "random", [1, 2], {"alone": np.ones(2), "related": np.ones(2), "random": np.ones(2)})
    assert headroom(no_a[0, 0])["gate"] is None                              # no self-copyable byte: H is undefined, not zero
    no_b = new_cube(1)
    for part in ("related", "random"):
        fill(no_b, 0, part, [0, 2, 2], {"alone": np.array([1.0, 6.0, 6.0]), "related": np.ones(3), "random": np.ones(3)})
    assert headroom(no_b[0, 0])["H_B"] == 0.0 and headroom(no_b[0, 0])["gate"] == 0.0   # no B byte: nothing to gain, which is zero, not unknown


def test_each_gain_splits_over_the_three_bins_and_the_related_margin_is_a_difference():
    from scripts.probe_commons import effects_of, fill, new_cube
    alone = np.array([1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0])
    cube = new_cube(1)
    related = alone - np.array([0, 0, 2, 2, 2, 2, 0.5, 0.5, 0.5, 0.5])
    fill(cube, 0, "related", [0, 0, 1, 1, 1, 1, 2, 2, 2, 2], {"alone": alone, "related": related, "random": alone - 0.1})
    fill(cube, 0, "random", [0, 0, 1, 2, 2, 2, 2, 2, 2, 2], {"alone": alone, "related": alone - 0.3, "random": alone - 0.1})
    e = effects_of(cube[0, 0])
    assert abs(e["bin_related_A"]) < 1e-12 and abs(e["bin_related_B"] - 0.8) < 1e-12 and abs(e["bin_related_C"] - 0.2) < 1e-12   # 4 of 10 bytes gain 2.0, 4 of 10 gain 0.5
    assert abs(e["gain_related"] - 1.0) < 1e-12 and abs(e["gain_random"] - 0.1) < 1e-12 and abs(e["gain_related_vs_random"] - 0.9) < 1e-12
    assert abs(e["bin_random_A"] - 0.02) < 1e-12 and abs(e["bin_random_B"] - 0.01) < 1e-12 and abs(e["bin_random_C"] - 0.07) < 1e-12   # 0.1 saved on 2 A, 1 B', 7 C' bytes of 10
    assert abs(sum(e[f"bin_random_{c}"] for c in "ABC") - e["gain_random"]) < 1e-12
    assert e["gain_never"] is None and e["bin_never_B"] is None and e["H_Bn"] is None          # nothing was read from never-trained files


def test_the_never_trained_partition_is_read_from_its_own_cells_and_leaves_the_gate_alone():
    from scripts.probe_commons import effects_of, fill, headroom, new_cube
    alone = np.array([1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0])
    cube = new_cube(1)
    both = {"alone": alone, "related": alone - np.array([0, 0, 2, 2, 2, 2, 0.5, 0.5, 0.5, 0.5]), "random": alone - 0.1}
    fill(cube, 0, "related", [0, 0, 1, 1, 1, 1, 2, 2, 2, 2], both)           # A A B B B B C C C C
    fill(cube, 0, "random", [0, 0, 2, 2, 2, 2, 1, 2, 2, 2], both)            # A A C' C' C' C' B' C' C' C'
    saved = np.array([0.25, 0.25, 2.0, 0.25, 1.0, 0.25, 0.5, 0.25, 0.25, 0.25])
    fill(cube, 0, "never", [0, 0, 1, 2, 1, 2, 1, 2, 2, 2], {"alone": alone, "never": alone - saved})   # A A Bn C Bn C Bn C C C
    h, e = headroom(cube[0, 0]), effects_of(cube[0, 0])
    assert abs(h["share_Bn"] - 0.3) < 1e-12 and abs(h["loss_Bn"] - 17 / 3) < 1e-12 and abs(h["H_Bn"] - 1.4) < 1e-12   # 3 of 10 bytes costing 5, 5, 7: 0.3 x (17/3 - 1)
    assert abs(h["H_B"] - 1.6) < 1e-12 and abs(h["H_Bp"] - 0.6) < 1e-12 and abs(h["gate"] - 1.0) < 1e-12              # the gate is H(B) - H(B'), whatever was read from never
    assert abs(e["H_Bn"] - 1.4) < 1e-12 and abs(e["gain_never"] - 0.525) < 1e-12                                       # 0.5 + 3.5 + 1.25 bits saved over 10 bytes
    assert abs(e["bin_never_A"] - 0.05) < 1e-12 and abs(e["bin_never_B"] - 0.35) < 1e-12 and abs(e["bin_never_C"] - 0.125) < 1e-12
    assert abs(e["gain_related"] - 1.0) < 1e-12 and abs(e["gain_random"] - 0.1) < 1e-12                                # the other two partitions are not disturbed
    assert abs(e["bin_related_B"] - 0.8) < 1e-12 and abs(e["bin_random_B"] - 0.01) < 1e-12 and abs(e["bin_random_C"] - 0.07) < 1e-12


def test_bootstrap_effects_are_seeded_collapse_for_identical_targets_and_move_for_different_ones():
    from scripts.probe_commons import bootstrap_effects, ci95, effects_of, fill, new_cube
    alone = np.array([1.0, 1.0, 5.0, 5.0, 7.0, 7.0])
    one = new_cube(1)
    fill(one, 0, "related", [0, 0, 1, 1, 2, 2], {"alone": alone, "related": alone - 0.5, "random": alone})
    fill(one, 0, "random", [0, 0, 1, 2, 2, 2], {"alone": alone, "related": alone, "random": alone})
    same = np.stack([one] * 7)
    ci, want = bootstrap_effects(same, [8], 50, 3), effects_of(same.sum(0)[0, 0])
    assert abs(ci["full"]["8"]["gate"][0] - want["gate"]) < 1e-12 and abs(ci["full"]["8"]["gain_related"][1] - want["gain_related"]) < 1e-12
    assert ci["full"]["8"]["gain_never"] is None and set(ci) == {"full", "first2048"}
    other = new_cube(1)
    fill(other, 0, "related", [0, 0, 1, 1, 1, 2], {"alone": alone, "related": alone - 1.0, "random": alone})
    fill(other, 0, "random", [0, 0, 2, 2, 2, 2], {"alone": alone, "related": alone, "random": alone})
    mixed = np.stack([one, other, one, other, other])
    lo, hi = bootstrap_effects(mixed, [8], 100, 3)["full"]["8"]["gain_related"]
    assert lo < hi
    rng = np.random.default_rng(5)

    def varied():                                                            # a random target of 8 to 39 bytes, the same bits under both partitions
        n = int(rng.integers(8, 40))
        base = rng.uniform(0.5, 8.0, n)
        bits = {"alone": base, "related": base - rng.uniform(0.0, 2.0, n), "random": base - rng.uniform(0.0, 0.3, n)}
        cube = new_cube(1)
        fill(cube, 0, "related", rng.integers(0, 3, n), bits)
        fill(cube, 0, "random", rng.integers(0, 3, n), bits)
        return cube
    many = np.stack([varied() for _ in range(30)])                           # with 30 different targets only a seeded resampling repeats exactly
    seeded = bootstrap_effects(many, [8], 60, 3)
    assert seeded == bootstrap_effects(many, [8], 60, 3) and seeded != bootstrap_effects(many, [8], 60, 4)
    pair = bootstrap_effects(np.stack([one, other]), [8], 200, 3)["full"]["8"]   # two targets: a resample can be two copies of either one
    for name in ("gain_related", "gate"):
        low, high = sorted(effects_of(c[0, 0])[name] for c in (one, other))   # gain_related 0.5 and 1.0, gate 2/3 and 7/3
        assert abs(pair[name][0] - low) < 1e-12 and abs(pair[name][1] - high) < 1e-12
    assert ci95(list(range(101))) == [2.5, 97.5]                             # the 2.5 % and 97.5 % points
    assert ci95([1.0, 2.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]) is not None and ci95([1.0, None, None]) is None and ci95([]) is None
    assert ci95([1.0] * 8 + [None] * 2) is None                              # a tenth of the resamples may be undefined, not a fifth


def test_a_bootstrap_resample_has_as_many_targets_as_the_sample():
    from scripts.probe_commons import bootstrap_effects, fill, new_cube

    def target(saved):                                                       # 6 bytes; the related prefix saves `saved` bits on each of them
        alone = np.array([1.0, 1.0, 5.0, 5.0, 7.0, 7.0])
        cube = new_cube(1)
        fill(cube, 0, "related", [0, 0, 1, 1, 2, 2], {"alone": alone, "related": alone - saved, "random": alone})
        fill(cube, 0, "random", [0, 0, 1, 2, 2, 2], {"alone": alone, "related": alone, "random": alone})
        return cube
    four = np.stack([target(0.0), target(0.0), target(0.0), target(1.0)])    # related gain 0, 0, 0, 1: a resample of 4 has gain k/4, k ~ Binomial(4, 1/4)
    lo, hi = bootstrap_effects(four, [8], 1000, 0)["full"]["8"]["gain_related"]
    assert lo == 0.0 and abs(hi - 0.75) < 1e-12                              # P(k <= 2) = 243/256 = 0.949 < 0.975 <= P(k <= 3) = 255/256: with 3 or 8 draws it is 2/3 or less


def _assert_close(got, want, path="result", tol=1e-12):
    """got equals want: dicts key for key, numbers within tol, None only against None."""
    if isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), f"{path}: keys {sorted(got) if isinstance(got, dict) else got} != {sorted(want)}"
        for k in want:
            _assert_close(got[k], want[k], f"{path}[{k!r}]", tol)
    elif want is None or got is None:
        assert got is None and want is None, f"{path}: {got} != {want}"
    else:
        assert abs(got - want) < tol, f"{path}: {got} != {want}"


def _two_long_targets():
    """Two targets of 4,096 scored bytes, each 512 blocks of a 4-byte pattern (bytes 0..2047) followed by 512 blocks of another (bytes
    2048..4095): their cubes for P = 8 and 16, stacked, and the per-byte arrays the cubes were filled from. Byte 0 of every block is A in
    every partition. The tables give, per target, the alone bits of a block (first half, second half), the code of each byte of a block per
    partition and P, and the bits a condition saves (alone minus condition) on each byte of a first-half block; a second-half block saves
    one bit less, never below 0. Only t2 has a never-trained partition."""
    from scripts.probe_commons import fill, new_cube
    alone_blocks = {"t1": ([1, 7, 4, 8], [3, 10, 4, 6]), "t2": ([1, 7, 5, 10], [3, 7, 8, 4])}
    codes = {("t1", "related"): ([0, 1, 2, 2], [0, 1, 1, 2]), ("t1", "random"): ([0, 2, 1, 2], [0, 2, 1, 1]),            # (P = 8, P = 16)
             ("t2", "related"): ([0, 2, 1, 2], [0, 2, 1, 1]), ("t2", "random"): ([0, 2, 2, 2], [0, 2, 1, 2]), ("t2", "never"): ([0, 1, 2, 2], [0, 1, 1, 2])}
    saved = {("t1", "related"): ([0, 1, 1, 1], [0, 3, 2, 1]), ("t1", "random"): ([0, 0, 1, 0], [0, 0, 1, 2]),
             ("t2", "related"): ([0, 0, 3, 0], [0, 0, 1, 3]), ("t2", "random"): ([0, 1, 0, 1], [0, 0, 2, 1]), ("t2", "never"): ([0, 3, 1, 1], [0, 3, 3, 1])}
    cubes, raws = [], []
    for t, (first, second) in alone_blocks.items():
        alone = np.concatenate([np.tile(first, 512), np.tile(second, 512)]).astype(float)
        cube, raw = new_cube(2), {"codes": {}, "bits": {}}
        for pi in range(2):
            bits = {"alone": alone}
            for cond in ("related", "random", "never"):
                if (t, cond) in saved:
                    s = np.array(saved[t, cond][pi])
                    bits[cond] = alone - np.concatenate([np.tile(s, 512), np.tile(np.maximum(s - 1, 0), 512)])
            raw["bits"][pi] = bits
            for part in ("related", "random", "never"):
                if (t, part) in codes:
                    raw["codes"][pi, part] = np.tile(codes[t, part][pi], 1024)
                    fill(cube, pi, part, raw["codes"][pi, part], bits)
        cubes.append(cube)
        raws.append(raw)
    return np.stack(cubes), raws


def test_summarize_pools_the_targets_per_scope_and_per_prefix_size_in_numbers_worked_out_by_hand():
    from scripts.probe_commons import summarize
    cubes, raws = _two_long_targets()
    s = summarize(cubes, [8, 16], True, 20, 0)
    assert s["n_scored_bytes"] == {"full": 8192, "first2048": 4096}                                        # 2 x 4,096 bytes, 2 x 2,048 in the first 2,048
    # alone bits of one block: t1 20 then 23, t2 23 then 22, so 43/8 per byte over the first halves and 88/16 over all. A condition's bits per byte
    # are (alone - bits it saves) over the same bytes, e.g. related at P = 8: first halves (43 - 3 - 3)/8, all (88 - 3 - 0 - 3 - 2)/16.
    # The never numbers pool t2 alone: first half (23 - 5)/4 = 9/2, all (45 - 5 - 2)/8 = 19/4 at P = 8.
    _assert_close(s["bpb"], {
        "full": {"alone": 11 / 2, "related": {"8": 5, "16": 73 / 16}, "random": {"8": 85 / 16, "16": 5}, "never": {"8": 19 / 4, "16": 17 / 4}},
        "first2048": {"alone": 43 / 8, "related": {"8": 37 / 8, "16": 33 / 8}, "random": {"8": 5, "16": 37 / 8}, "never": {"8": 9 / 2, "16": 4}}})
    # H(X) = (sum over the bytes of bin X of alone bits - loss(A)) / all bytes, with loss(A) = 1 in the first halves and 2 over all.
    # First 2,048 at P = 8: B is t1 byte 1 (7 bits) and t2 byte 2 (5 bits): (6 + 4)/8 = 5/4; B' is t1 byte 2 (4 bits): 3/8; gate 7/8.
    heads = ("H_B", "H_Bp", "H_Bn", "gate")
    for (scope, p), want in {("full", "8"): (11 / 8, 1 / 4, 5 / 4, 9 / 8), ("full", "16"): (9 / 4, 23 / 16, 19 / 8, 13 / 16),
                             ("first2048", "8"): (5 / 4, 3 / 8, 3 / 2, 7 / 8), ("first2048", "16"): (11 / 4, 7 / 4, 5 / 2, 1)}.items():
        _assert_close({k: s["headroom"][scope][p][k] for k in heads}, dict(zip(heads, want)), f"headroom[{scope}][{p}]")
    # gain = bits saved over bytes: related at P = 8 saves 3 + 3 per block pair in the first halves (6/8 = 3/4) and 8 in all (8/16 = 1/2).
    names = ("gain_related", "gain_random", "gain_never", "gain_related_vs_random")
    for (scope, p), want in {("full", "8"): (1 / 2, 3 / 16, 7 / 8, 5 / 16), ("full", "16"): (15 / 16, 1 / 2, 11 / 8, 7 / 16),
                             ("first2048", "8"): (3 / 4, 3 / 8, 5 / 4, 3 / 8), ("first2048", "16"): (5 / 4, 3 / 4, 7 / 4, 1 / 2)}.items():
        _assert_close({k: s["gains"][scope][p][k] for k in names}, dict(zip(names, want)), f"gains[{scope}][{p}]")
    # the never numbers pool t2 only: its gain is alone minus never on t2's own bytes (45/8 - 19/4 = 7/8), not the all-target alone minus never (11/2 - 19/4 = 3/4)
    assert abs(s["gains"]["full"]["8"]["gain_never"] - (45 / 8 - s["bpb"]["full"]["never"]["8"])) < 1e-12
    assert abs((s["bpb"]["full"]["alone"] - s["bpb"]["full"]["never"]["8"]) - 3 / 4) < 1e-12
    counts = {("full", "8"): ((2048, 2048, 4096), (2048, 1024, 5120), (1024, 1024, 2048)),                     # A, B, C bytes of related, random, never
              ("full", "16"): ((2048, 4096, 2048), (2048, 3072, 3072), (1024, 2048, 1024)),
              ("first2048", "8"): ((1024, 1024, 2048), (1024, 512, 2560), (512, 512, 1024)),
              ("first2048", "16"): ((1024, 2048, 1024), (1024, 1536, 1536), (512, 1024, 512))}
    for (scope, p), by_part in counts.items():
        for part, want in zip(("related", "random", "never"), by_part):
            assert tuple(s["bins"][scope][p][part]["count"].values()) == want, (scope, p, part)
    # One cell by hand (all bytes, P = 16, related partition): its B bytes are t1 bytes 1, 2 and t2 bytes 2, 3 of both halves, 8 byte slots costing
    # 7 + 4 + 10 + 4 (t1) and 5 + 10 + 8 + 4 (t2) = 52 bits alone, 13/2 per byte; the related prefix saves 8 + 6 = 14 of them, so 38/8 = 19/4 per byte.
    _assert_close(s["bins"]["full"]["16"], {
        "related": {"count": {"A": 2048, "B": 4096, "C": 2048}, "bpb": {"alone": {"A": 2, "B": 13 / 2, "C": 7}, "related": {"A": 2, "B": 19 / 4, "C": 27 / 4},
                                                                         "random": {"A": 2, "B": 47 / 8, "C": 25 / 4}}},
        "random": {"count": {"A": 2048, "B": 3072, "C": 3072}, "bpb": {"alone": {"A": 2, "B": 35 / 6, "C": 15 / 2}, "related": {"A": 2, "B": 5, "C": 35 / 6},
                                                                        "random": {"A": 2, "B": 14 / 3, "C": 22 / 3}}},
        "never": {"count": {"A": 1024, "B": 2048, "C": 1024}, "bpb": {"alone": {"A": 2, "B": 27 / 4, "C": 7}, "never": {"A": 2, "B": 17 / 4, "C": 13 / 2}}}})
    for scope, cap in (("full", 4096), ("first2048", 2048)):                                               # the bits per byte of every bin of every cell, from the raw bytes
        for pi, p in enumerate(("8", "16")):
            for part in ("related", "random", "never"):
                have = [r for r in raws if (pi, part) in r["codes"]]
                codes = np.concatenate([r["codes"][pi, part][:cap] for r in have])
                want = {"count": {c: int((codes == k).sum()) for k, c in enumerate("ABC")}, "bpb": {}}
                for cond in (("alone", "never") if part == "never" else ("alone", "related", "random")):
                    bits = np.concatenate([r["bits"][pi][cond][:cap] for r in have])
                    want["bpb"][cond] = {c: (float(bits[codes == k].mean()) if (codes == k).any() else None) for k, c in enumerate("ABC")}
                _assert_close(s["bins"][scope][p][part], want, f"bins[{scope}][{p}][{part}]")
    every = set(s["gains"]["full"]["8"])
    assert set(s["ci95"]) == {"full", "first2048"}                                                         # an interval for every scope, P and name
    assert all(set(s["ci95"][sc]) == {"8", "16"} and all(set(s["ci95"][sc][p]) == every for p in ("8", "16")) for sc in s["ci95"])


def test_summarize_intervals_collapse_onto_the_gains_for_identical_targets_and_with_never_false_drops_the_never_entries():
    from scripts.probe_commons import summarize
    cubes, _ = _two_long_targets()
    for t in (0, 1):                                                         # t1 has no never partition, t2 has one
        s = summarize(np.stack([cubes[t]] * 4), [8, 16], True, 20, 0)
        for scope in ("full", "first2048"):
            for p in ("8", "16"):
                assert set(s["ci95"][scope][p]) == set(s["gains"][scope][p])
                for name, point in s["gains"][scope][p].items():             # identical targets: every resample pools to the same numbers
                    ci = s["ci95"][scope][p][name]
                    assert ci is None if point is None else (abs(ci[0] - point) < 1e-12 and abs(ci[1] - point) < 1e-12), (t, scope, p, name)
        assert (s["gains"]["full"]["8"]["gain_never"] is None) == (t == 0)
    nobody = summarize(np.stack([cubes[0]] * 2), [8, 16], True, 5, 0)        # with_never, but no target filled the never partition: empty, not zero
    assert nobody["bpb"]["full"]["never"] == {"8": None, "16": None}
    assert nobody["bins"]["first2048"]["16"]["never"] == {"count": {"A": 0, "B": 0, "C": 0}, "bpb": {"alone": {"A": None, "B": None, "C": None}, "never": {"A": None, "B": None, "C": None}}}
    off = summarize(np.stack([cubes[0]] * 2), [8, 16], False, 5, 0)
    assert set(off["bpb"]["full"]) == {"alone", "related", "random"} and set(off["bins"]["full"]["8"]) == {"related", "random"}


def test_the_prefix_reaches_the_model_and_the_scored_bits_are_one_forward_over_prefix_and_stream(tiny_hybrid):
    from scripts.probe_commons import byte_bits, score_with_prefix
    m = tiny_hybrid()                                                        # a model with memory: unlike Bigram it can see the prefix
    head = b"\n// ==== k/t0.c ====\n"
    prefix = b"\n// ==== k/a0.c ====\n" + b"static int probe_device(struct device *dev) { return -ENODEV; }\n" * 2
    stream = head + b"static int probe_device(struct device *dev) { return 0; }\n" * 3
    n = 100
    assert len(prefix) >= 40 and n < len(stream) - len(head)
    got = score_with_prefix(m, prefix, stream, len(head), n, "cpu")
    a = len(prefix) + len(head)
    assert len(got) == n and np.array_equal(got, byte_bits(m, prefix + stream, "cpu")[a : a + n])   # one forward over prefix and stream together
    alone = score_with_prefix(m, b"", stream, len(head), n, "cpu")
    assert np.abs(got - alone).max() > 1e-3                                  # a probe that dropped its prefix would report "related files gain nothing"


def _probe(world, tiny_hybrid, ps=(0, 128), seed=0, never=True, split="dev"):
    from scripts.probe_commons import probe_context
    return probe_context(tiny_hybrid(), "cpu", world.path, split, ps, seed, 300, 0, never, world.train_end, world.tail_start, world.dev_files)


def test_an_empty_prefix_scores_exactly_alone_and_short_bodies_are_scored_whole(world, tiny_hybrid):
    res = _probe(world, tiny_hybrid)
    full = res["all"]["bpb"]["full"]
    assert full["related"]["0"] == full["alone"] == full["random"]["0"]      # P = 0 is the alone sequence: identical, not close
    assert full["related"]["128"] != full["alone"] and full["random"]["128"] != full["alone"]
    assert [r["n_scored"] for r in res["targets"]] == [300, 300, 90]
    assert res["all"]["n_scored_bytes"] == {"full": 690, "first2048": 690}
    assert res["n_targets"] == 3 and res["n_never_targets"] == 3
    assert res["never_subset"]["bpb"]["full"]["never"]["0"] == full["alone"]
    assert res["gate_number"] is None and res["gate_ci95"] is None            # 32,768 is not among this run's P
    assert res["targets"][0]["prefix_bytes"]["128"] == [128, 128]


def test_context_probe_is_deterministic_for_a_fixed_seed_and_json_serialisable(world, tiny_hybrid):
    import json
    a, b = _probe(world, tiny_hybrid, seed=0), _probe(world, tiny_hybrid, seed=0)
    assert json.dumps(a) == json.dumps(b)
    c = _probe(world, tiny_hybrid, seed=1)
    assert a["all"]["bpb"]["full"]["related"] == c["all"]["bpb"]["full"]["related"]          # the related prefix does not depend on the seed
    assert a["all"]["bpb"]["full"]["random"] != c["all"]["bpb"]["full"]["random"]            # the random prefix does


def test_the_gate_number_is_read_at_p_32768_and_never_trained_is_optional(world, tiny_hybrid):
    res = _probe(world, tiny_hybrid, ps=(128, 32_768), never=False)
    h = res["all"]["headroom"]["full"]["32768"]
    assert res["gate_number"] == h["gate"] == h["H_B"] - h["H_Bp"] == res["all"]["gains"]["full"]["32768"]["gate"]
    assert len(res["gate_ci95"]) == 2 and res["gate_ci95"] == res["all"]["ci95"]["full"]["32768"]["gate"]
    g = res["all"]["gains"]["full"]["128"]                                                      # gains carry intervals too, on the same targets
    assert abs(g["gain_related"] - (res["all"]["bpb"]["full"]["alone"] - res["all"]["bpb"]["full"]["related"]["128"])) < 1e-9
    assert abs(sum(g[f"bin_related_{c}"] for c in "ABC") - g["gain_related"]) < 1e-9 and len(res["all"]["ci95"]["full"]["128"]["gain_related"]) == 2
    assert res["never_subset"] is None and res["n_never_targets"] == 0 and res["all"]["bpb"]["full"].get("never") is None


def test_render_context_prints_the_tables_and_the_gate_number(world, tiny_hybrid):
    from scripts.probe_commons import render_context
    res = _probe(world, tiny_hybrid, ps=(128, 32_768))
    text = render_context({"run_name": "tiny", **res})
    assert "| P | related | random | never trained |" in text and "| P | scope | share B | share B' | H(B) | H(B') | H(B) - H(B') |" in text
    assert "| P | gain related | gain random | related minus random | gain never trained |" in text and "gain, share-weighted |" in text
    assert "gate number H(B) - H(B') at n = 8, P = 32768" in text and "census bins from the never prefix" in text and "95 % interval over targets" in text


def test_main_scores_a_tiny_checkpoint_and_the_transformer_arm_is_pinned_to_2048(world, tiny_checkpoint, tmp_path, monkeypatch):
    import json

    import pytest
    import scripts.probe_commons as pc
    monkeypatch.setattr(pc, "RESULTS", tmp_path)
    common = ["--mode", "context", "--split", "dev", "--corpus", str(world.path), "--device", "cpu", "--ps", "128,256", "--n-scored", "300", "--chunk", "0",
              "--train-end", str(world.train_end), "--tail-start", str(world.tail_start), "--dev-files", "3"]
    pc.main(["--ckpt", str(tiny_checkpoint()), "--run-name", "ctx_tiny", *common])
    res = json.loads((tmp_path / "ctx_tiny.json").read_text())
    assert {"run_name", "mode", "ckpt", "arm", "chunk", "args", "split", "ps", "n_scored_cap", "rope_train_len", "n_targets", "n_never_targets",
            "gate_p", "gate_number", "gate_ci95", "all", "never_subset", "targets"} <= set(res)
    assert res["ps"] == [128, 256] and res["n_scored_cap"] == 300 and res["rope_train_len"] == 0 and res["arm"] == "mingru_sparse_post"
    pc.main(["--ckpt", str(tiny_checkpoint("transformer")), "--run-name", "ctx_tiny_tf", *common])
    tf = json.loads((tmp_path / "ctx_tiny_tf.json").read_text())
    assert tf["ps"] == [2048] and tf["n_scored_cap"] == 2048 and tf["rope_train_len"] == 64 and tf["arm"] == "transformer"
    bodies = [world.table[world.idx[p]].end - world.table[world.idx[p]].body for p in ("k/t0.c", "k/t1.c", "k/t2.c")]
    assert [t["n_scored"] for t in tf["targets"]] == bodies and max(bodies) > 300      # --n-scored 300 is ignored: the reference scores up to 2,048
    with pytest.raises(SystemExit):
        pc.main(["--ckpt", str(tiny_checkpoint()), "--run-name", "ctx_tiny", *common])


def _long_and_empty_world(corpus):
    """d/a0..a3 and e/r0, e/r1 are training files; d/big.c (2,600 bytes of repeated phrases) and an empty d/empty.c close the file."""
    import random
    from flybrain.data.files import file_table, hash_table
    rng = random.Random(1)
    noise = lambda n: bytes(rng.choice(b"abcdefghijklmnopqrstuvwxyz ;(){}\n") for _ in range(n))
    phrases = [noise(60) for _ in range(8)]
    text = lambda n: b"".join(rng.choice(phrases) for _ in range(n // 60 + 1))[:n]
    files = [(f"d/a{j}.c", text(700)) for j in range(4)] + [(f"e/r{j}.c", noise(700)) for j in range(2)] + [("d/big.c", text(2600)), ("d/empty.c", b"")]
    path = corpus(files)
    table = file_table(path)
    return path, table, hash_table(path, table)


def test_a_long_target_is_cut_at_the_cap_and_a_header_at_the_end_of_the_file_is_skipped(corpus, tiny_hybrid):
    from scripts.probe_commons import probe_context
    path, table, _ = _long_and_empty_world(corpus)
    cut = table[6].start                                                       # training ends where d/big.c starts; the tail is big.c and empty.c
    res = probe_context(tiny_hybrid(), "cpu", path, "dev", (128, 32_768), 0, 4096, 0, False, cut, cut, 2)
    assert [t["path"] for t in res["targets"]] == ["d/big.c"]                  # d/empty.c is the last header of the file: nothing to score
    assert res["all"]["n_scored_bytes"] == {"full": 2600, "first2048": 2048}
    full, first = res["all"]["headroom"]["full"]["32768"], res["all"]["headroom"]["first2048"]["32768"]
    assert full["gate"] is not None and res["gate_number"] == full["gate"] != first["gate"]   # the gate number reads the full scope
    assert res["gate_ci95"] == res["all"]["ci95"]["full"]["32768"]["gate"] != res["all"]["ci95"]["first2048"]["32768"]["gate"]   # and so does its interval
    capped = probe_context(tiny_hybrid(), "cpu", path, "dev", (128,), 0, 1000, 0, False, cut, cut, 2)
    assert capped["all"]["n_scored_bytes"] == {"full": 1000, "first2048": 1000} and capped["targets"][0]["n_scored"] == 1000


class Spy(torch.nn.Module):
    """Wraps a model and keeps every token sequence it is asked to score."""

    def __init__(self, model):
        super().__init__()
        self.model, self.seen = model, []

    def forward(self, x):
        self.seen.append(bytes(x[0].tolist()))
        return self.model(x)


def test_every_sequence_the_model_reads_ends_at_the_scored_window_and_holds_no_leaked_file(world, tiny_hybrid):
    """A spy keeps what probe_context feeds the model: for each target alone, then related, random and never-trained at each P. P = 20,000 is more
    than any prefix holds, so there each prefix is the complete set of files its rule allows. Every sequence is a prefix and then the target's
    stream, so its last byte is the last scored byte; no prefix holds the target or a copy of it, a test-split byte or (the never-trained one) a
    training byte."""
    from flybrain.data.files import HEADER
    from scripts.probe_commons import probe_context
    t, ix, raw = world.table, world.idx, world.path.read_bytes()
    train = {e.path for e in t if e.end <= world.train_end}
    k_train = ["k/a0.c", "k/a1.c", "k/a2.c", "k/a3.c", "k/a4.c", "k/a_twin.c"]          # k/a_twin.c is a copy of k/t1.c, y/twin.c a copy of k/t0.c
    allowed = {"k/t0.c": (k_train, ["k/t1.c", "k/t2.c"]), "k/t1.c": (k_train[:5], ["k/t0.c", "k/t2.c"]), "k/t2.c": (k_train, ["k/t0.c", "k/t1.c"])}
    spy = Spy(tiny_hybrid())
    res = probe_context(spy, "cpu", world.path, "dev", (128, 20_000), 0, 300, 0, True, world.train_end, world.tail_start, world.dev_files)
    assert [r["path"] for r in res["targets"]] == list(allowed) and len(spy.seen) == 3 * 7      # per target: alone, then 3 prefixes at each of 2 P
    for k, (path, (related_files, never_files)) in enumerate(allowed.items()):
        e, i = t[ix[path]], ix[path]
        stream = raw[e.start : e.body + min(300, e.end - e.body)]                      # the header and the first 300 body bytes, straight from the file
        for n, data in enumerate(spy.seen[7 * k : 7 * k + 7]):
            kind = "alone" if n == 0 else ("related", "random", "never")[(n - 1) % 3]
            assert data.endswith(stream)
            prefix = data[: len(data) - len(stream)]
            paths = [m.group(1).decode() for m in HEADER.finditer(prefix)]
            assert path not in paths and all(world.hashes[ix[q]] != world.hashes[i] for q in paths), (path, kind)
            if kind == "alone":
                assert prefix == b""
            elif n <= 3:
                assert len(prefix) == 128 and paths, (path, kind)                      # the P = 128 triple: every kind is cut at exactly P bytes
            elif kind == "related":
                assert paths == related_files, path                                     # the same-directory training files, no copy, nothing from the tail
            elif kind == "random":
                assert paths and set(paths) <= train and not any(q.startswith("k/") for q in paths), path
            else:
                assert paths == never_files, path                                       # the same-directory files of its own split and nothing else


def _three_sources_world(corpus):
    """Training files d/a0..a3 (all hold a 60-byte phrase X) and e/r0, e/r1 (a 24-byte phrase R); the target d/t0.c holds X, a phrase Y and R, and its
    dev neighbours d/u0.c and d/u1.c hold Y only. Each kind of prefix covers a different part of the target, so no two partitions bin alike."""
    import random
    from flybrain.data.files import file_table, hash_table
    rng = random.Random(2)
    noise = lambda n: bytes(rng.choice(b"abcdefghijklmnopqrstuvwxyz ;(){}\n") for _ in range(n))
    x, y, r = noise(60), noise(40), noise(24)
    files = [(f"d/a{j}.c", noise(80) + x + noise(80)) for j in range(4)] + [(f"e/r{j}.c", noise(80) + r + noise(80)) for j in range(2)]
    files += [("d/t0.c", noise(60) + x + noise(60) + y + noise(60) + r + noise(60)), ("d/u0.c", noise(80) + y + noise(80)), ("d/u1.c", noise(50) + y + noise(90))]
    path = corpus(files)
    table = file_table(path)
    return path, table, hash_table(path, table)


def test_a_targets_cube_holds_each_conditions_bits_in_the_bins_of_each_partitions_own_prefix(corpus, tiny_hybrid):
    """Cell by cell against the primitives: the related, random and never-trained partitions are binned from their own prefix (swap two of them and
    the gate number changes sign) and each condition's cells hold the bits of the sequence read after its own prefix."""
    from flybrain.census import bins
    from flybrain.data.files import read_stream, split_range
    from flybrain.data.prefixes import prefix_pairs
    from scripts.probe_commons import BIN_N, COND_IX, FILL, PART_IX, probe_target, score_with_prefix
    path, table, hashes = _three_sources_world(corpus)
    m, ps, i, cut = tiny_hybrid(), (100, 200, 20_000), 6, table[6].start                   # d/t0.c; training ends where it starts
    dev = split_range(table, [6, 7, 8])
    cube, rec = probe_target(m, "cpu", path, table, hashes, i, ps, 0, 300, 0, dev, cut)
    stream, body_start, scored = read_stream(path, table[i], 300)
    pairs = prefix_pairs(path, table, i, hashes, ps, 0, 0, cut, cut)
    never = prefix_pairs(path, table, i, hashes, ps, 0, *dev, cut)
    for pi, p in enumerate(ps):
        prefix = {"alone": b"", "related": pairs[p][0], "random": pairs[p][1], "never": never[p][0]}
        bits = {c: score_with_prefix(m, x, stream, body_start, scored, "cpu") for c, x in prefix.items()}
        codes = {part: np.array(bins(stream, body_start, scored, prefix[part], BIN_N)) for part in ("related", "random", "never")}
        assert p == 100 or len({c.tobytes() for c in codes.values()}) == 3                  # once the prefixes reach their phrases the three bin differently
        for part, c in codes.items():
            for cond in FILL[part]:
                for k in range(3):
                    count, total = cube[0, pi, PART_IX[part], COND_IX[cond], k]
                    assert count == (c == k).sum() and abs(total - bits[cond][c == k].sum()) < 1e-9, (p, part, cond, "ABC"[k])
        assert all(abs(rec["bpb"][c][str(p)] - bits[c].mean()) < 1e-12 for c in ("related", "random", "never"))
        assert rec["prefix_bytes"][str(p)] == [len(prefix["related"]), len(prefix["never"])]


def test_the_never_subset_pools_only_the_targets_that_have_never_trained_files(world, tiny_hybrid):
    """With five dev files z/t4.c is the only z/ file of its split, so it has no never-trained prefix: the never numbers pool the four k/ targets,
    the all-target numbers pool five, and the never gain is alone minus never over the same four targets' bytes in both blocks."""
    from scripts.probe_commons import probe_context
    res = probe_context(tiny_hybrid(), "cpu", world.path, "dev", (128,), 0, 300, 0, True, world.train_end, world.tail_start, 5)
    recs = res["targets"]
    sub = [r for r in recs if r["never"]]
    assert [(r["path"], r["never"]) for r in recs] == [("k/t0.c", True), ("k/t1.c", True), ("k/t2.c", True), ("k/t3.c", True), ("z/t4.c", False)]
    assert [r["index"] for r in recs] == [world.idx[p] for p in ("k/t0.c", "k/t1.c", "k/t2.c", "k/t3.c", "z/t4.c")]        # each record's index is its place in the file table
    assert res["n_targets"] == 5 and res["n_never_targets"] == 4 and recs[4]["bpb"]["never"] is None and recs[4]["prefix_bytes"]["128"][1] is None
    pooled = lambda rs, get: sum(r["n_scored"] * get(r) for r in rs) / sum(r["n_scored"] for r in rs)         # bits per byte over these targets' bytes
    alone, related, never = (lambda r: r["bpb"]["alone"]), (lambda r: r["bpb"]["related"]["128"]), (lambda r: r["bpb"]["never"]["128"])
    a, n = res["all"], res["never_subset"]
    assert a["n_scored_bytes"]["full"] == sum(r["n_scored"] for r in recs) and n["n_scored_bytes"]["full"] == sum(r["n_scored"] for r in sub)
    assert abs(a["bpb"]["full"]["alone"] - pooled(recs, alone)) < 1e-9 and abs(n["bpb"]["full"]["alone"] - pooled(sub, alone)) < 1e-9
    assert abs(a["bpb"]["full"]["related"]["128"] - pooled(recs, related)) < 1e-9 and abs(n["bpb"]["full"]["related"]["128"] - pooled(sub, related)) < 1e-9
    assert abs(n["bpb"]["full"]["never"]["128"] - pooled(sub, never)) < 1e-9
    assert "never" not in a["bpb"]["full"] and "never" not in a["bins"]["full"]["128"]                      # all targets: no never numbers beside the all-target alone
    for block in (a, n):                                                                                     # the paired number, the same four targets in both blocks
        assert abs(block["gains"]["full"]["128"]["gain_never"] - (pooled(sub, alone) - pooled(sub, never))) < 1e-9


def test_render_context_says_when_the_gate_interval_straddles_the_threshold(world, tiny_hybrid):
    from scripts.probe_commons import render_context
    res = _probe(world, tiny_hybrid, ps=(128, 32_768), never=False)
    say = "The interval straddles the 0.02 threshold."
    for ci, straddles in (([0.01, 0.03], True), ([-0.02, 0.05], True), ([0.021, 0.04], False), ([-0.03, 0.019], False), (None, False)):
        assert (say in render_context({**res, "gate_ci95": ci})) == straddles, ci
    last = render_context({**res, "gate_number": 0.0215, "gate_ci95": [0.01, 0.03]}).splitlines()[-1]
    assert last == f"gate number H(B) - H(B') at n = 8, P = 32768: 0.0215 (above the 0.02 threshold), 95 % interval over targets [0.01, 0.03]. {say}"
    assert render_context(res).splitlines()[-1].startswith(f"gate number H(B) - H(B') at n = 8, P = 32768: {res['gate_number']:.4f} (")     # the run's own gate number


def test_the_gate_line_prints_four_decimals_and_the_side_of_the_threshold_the_unrounded_number_falls_on(world, tiny_hybrid):
    from scripts.probe_commons import render_context
    res = _probe(world, tiny_hybrid, ps=(128, 32_768), never=False)
    head = "gate number H(B) - H(B') at n = 8, P = 32768: "
    cases = ((0.0215, "0.0215", "above"), (0.0185, "0.0185", "below"), (-0.003, "-0.0030", "below"),
             (0.02, "0.0200", "above"),                                              # the criterion is 0.02 or more: exactly the threshold is above it
             (0.019996, "0.0200", "below"), (0.0200004, "0.0200", "above"))          # both print as the threshold itself, one on each side of it
    for gate, shown, side in cases:
        last = render_context({**res, "gate_number": gate, "gate_ci95": [0.03, 0.04]}).splitlines()[-1]
        assert last == f"{head}{shown} ({side} the 0.02 threshold), 95 % interval over targets [0.03, 0.04].", gate
    assert render_context({**res, "gate_number": None, "gate_ci95": None}).splitlines()[-1] == f"{head}n/a, 95 % interval over targets None."     # no number, no side
    last = render_context({**res, "gate_number": 0.0215, "gate_ci95": [0.0123456, 0.0345678]}).splitlines()[-1]                                  # the interval keeps its four decimals
    assert last == f"{head}0.0215 (above the 0.02 threshold), 95 % interval over targets [0.0123, 0.0346]. The interval straddles the 0.02 threshold."


def test_the_all_targets_gain_table_prints_no_never_subset_gain_and_the_subset_table_does(world, tiny_hybrid):
    """The never gain pools only the never subset (101 of 353 dev targets) and has its own interval in each block, so it is not printed beside gains that
    pool every target: those rows read n/a. The subset block prints it next to gains on the same targets. Rendering changes nothing in the result."""
    import copy

    from scripts.probe_commons import probe_context, render_context
    res = probe_context(tiny_hybrid(), "cpu", world.path, "dev", (128, 256), 0, 300, 0, True, world.train_end, world.tail_start, 5)   # 5 targets, 4 of them never
    frozen = copy.deepcopy(res)
    text = render_context({"run_name": "tiny", **res})
    assert res == frozen and res["n_targets"] == 5 and res["n_never_targets"] == 4
    shown = lambda blk, scope, p, k: "{:+.3f} ({:+.3f}, {:+.3f})".format(blk["gains"][scope][str(p)][k], *blk["ci95"][scope][str(p)][k])
    rows = {}                                                                          # (block, scope, P) -> the four gain cells of that row
    for section in text.split("\n\n"):
        head, *table = section.splitlines()
        if ": gains over alone" in head:
            block, scope = ("all" if head.startswith("all targets") else "never_subset"), head.split(", ")[1].split(":")[0]
            for row in table[2:]:
                cells = [c.strip() for c in row.strip("|").split("|")]
                rows[block, scope, int(cells[0])] = cells[1:]
    assert sorted(rows) == [(b, s, p) for b in ("all", "never_subset") for s in ("first2048", "full") for p in (128, 256)]
    for (block, scope, p), cells in rows.items():
        names = ("gain_related", "gain_random", "gain_related_vs_random")
        assert cells[:3] == [shown(res[block], scope, p, k) for k in names], (block, scope, p)                                     # the other gains print as before
        assert cells[3] == ("n/a (never subset only)" if block == "all" else shown(res[block], scope, p, "gain_never")), (block, scope, p)
    assert text.count("n/a (never subset only)") == 4                                  # two scopes x two P in the all block, nowhere else


def test_load_checkpoint_returns_an_eval_model_with_the_requested_head_chunk(tiny_checkpoint):
    from scripts.probe_commons import load_checkpoint
    for arm in ("mingru_sparse_post", "transformer"):
        assert not load_checkpoint(str(tiny_checkpoint(arm)), "cpu", 0)[0].training          # rebuild returns a model in train mode and byte_bits never switches it
    m, ck = load_checkpoint(str(tiny_checkpoint()), "cpu", 0)
    assert ck["arm"] == "mingru_sparse_post" and [h.chunk for h in m.all_heads()] == [8, 8]  # 0 keeps the checkpoint's own chunk
    assert [h.chunk for h in load_checkpoint(str(tiny_checkpoint()), "cpu", 4096)[0].all_heads()] == [4096, 4096]


def test_main_hands_every_flag_to_probe_context_on_both_splits(world, tiny_checkpoint, tmp_path, monkeypatch):
    """The command line is probe_context with the flags in the right places: the same numbers as a direct call."""
    import json
    from pathlib import Path

    import scripts.probe_commons as pc
    monkeypatch.setattr(pc, "RESULTS", tmp_path)
    ckpt = str(tiny_checkpoint())
    flags = ["--mode", "context", "--ckpt", ckpt, "--corpus", str(world.path), "--device", "cpu", "--ps", "128,700", "--n-scored", "300", "--chunk", "16",
             "--train-end", str(world.train_end), "--tail-start", str(world.tail_start)]
    runs = (("dev", 5, 1, 0, True, ["--seed", "1"], ["k/t0.c", "k/t1.c", "k/t2.c", "k/t3.c", "z/t4.c"]),
            ("test", 3, 0, 2, False, ["--limit", "2", "--no-never"], ["k/t3.c", "z/t4.c"]))
    for split, dev_files, seed, limit, never, extra, paths in runs:
        pc.main(["--run-name", f"cli_{split}", "--split", split, "--dev-files", str(dev_files), *extra, *flags])
        got = json.loads((tmp_path / f"cli_{split}.json").read_text())
        want = pc.probe_context(pc.load_checkpoint(ckpt, "cpu", 16)[0], "cpu", world.path, split, [128, 700], seed, 300, 0, never,
                                world.train_end, world.tail_start, dev_files, limit)
        assert {k: got[k] for k in want} == json.loads(json.dumps(want)), split
        assert [r["path"] for r in got["targets"]] == paths                                         # the split, its size and the limit, from outside
        assert got["run_name"] == f"cli_{split}" and got["chunk"] == 16 and got["args"]["split"] == split
        assert got["split"] == split                                                                # the label of the run is the split it scored (a test run is not "dev")
        assert got["mode"] == "context" and got["ckpt"] == Path(ckpt).name                          # provenance: the mode and the checkpoint file the numbers came from
        assert (got["args"]["ps"], got["args"]["seed"], got["args"]["limit"], got["args"]["no_never"]) == ("128,700", seed, limit, not never)   # and the flags of the run


def test_the_rope_training_length_reaches_every_forward_of_the_transformer_arm_and_of_no_other_arm(world, tiny_checkpoint, tmp_path, monkeypatch):
    import scripts.probe_commons as pc
    monkeypatch.setattr(pc, "RESULTS", tmp_path)
    seen, real = [], pc.byte_bits
    monkeypatch.setattr(pc, "byte_bits", lambda model, data, device, rope_train_len=0: (seen.append(rope_train_len), real(model, data, device, rope_train_len))[1])
    for arm, want in (("transformer", 64), ("mingru_sparse_post", 0)):                  # the checkpoint's own seq_len for the transformer, nothing for the others
        seen.clear()
        pc.main(["--mode", "context", "--split", "dev", "--ckpt", str(tiny_checkpoint(arm)), "--run-name", f"rope_{arm}", "--corpus", str(world.path), "--device", "cpu",
                 "--ps", "128", "--n-scored", "300", "--chunk", "0", "--train-end", str(world.train_end), "--tail-start", str(world.tail_start), "--dev-files", "3", "--no-never"])
        assert seen and set(seen) == {want}, arm
