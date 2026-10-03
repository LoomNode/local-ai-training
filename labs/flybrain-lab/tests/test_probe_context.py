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
    cube = new_cube(1)
    bits = {c: np.ones(3000) for c in ("alone", "related", "random")}
    fill(cube, 0, "related", np.zeros(3000, dtype=int), bits)
    a, r = COND_IX["alone"], PART_IX["related"]
    assert cube[0, 0, r, a, 0, 0] == 3000 and cube[1, 0, r, a, 0, 0] == 2048 and cube[1, 0, r, a, 0, 1] == 2048.0
    short = new_cube(1)
    fill(short, 0, "related", np.zeros(90, dtype=int), {c: np.ones(90) for c in ("alone", "related", "random")})
    assert short[0, 0, r, a, 0, 0] == short[1, 0, r, a, 0, 0] == 90            # a short body: the two scopes coincide


def test_headroom_is_share_times_the_loss_gap_and_the_gate_is_their_difference():
    from scripts.probe_commons import fill, headroom, new_cube
    alone = np.array([1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0])
    bits = {"alone": alone, "related": alone - 0.5, "random": alone}
    cube = new_cube(1)
    fill(cube, 0, "related", [0, 0, 1, 1, 1, 1, 2, 2, 2, 2], bits)           # A A B B B B C C C C
    fill(cube, 0, "random", [0, 0, 1, 2, 2, 2, 2, 2, 2, 2], bits)            # A A B' C' C' ...
    h = headroom(cube[0, 0])
    assert (h["share_B"], h["share_Bp"]) == (0.4, 0.1) and (h["loss_A"], h["loss_B"], h["loss_Bp"]) == (1.0, 5.0, 5.0)
    assert abs(h["H_B"] - 0.4 * 4.0) < 1e-12 and abs(h["H_Bp"] - 0.1 * 4.0) < 1e-12 and abs(h["gate"] - 1.2) < 1e-12
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
    assert abs(sum(e[f"bin_random_{c}"] for c in "ABC") - e["gain_random"]) < 1e-12
    assert e["gain_never"] is None and e["bin_never_B"] is None and e["H_Bn"] is None          # nothing was read from never-trained files


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
    a = bootstrap_effects(mixed, [8], 100, 3)
    assert a == bootstrap_effects(mixed, [8], 100, 3)
    lo, hi = a["full"]["8"]["gain_related"]
    assert lo < hi
    assert ci95([1.0, 2.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]) is not None and ci95([1.0, None, None]) is None and ci95([]) is None


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
