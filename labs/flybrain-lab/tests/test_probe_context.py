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
