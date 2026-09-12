"""Synthetic fly-like graphs stay sparse and shuffle is a real topology control."""

from __future__ import annotations

import numpy as np
import torch

from flybrain.data.connectome import (
    REGION_KENYON,
    REGION_READOUT,
    REGION_SENSORY,
    Connectome,
    degree_matched_shuffle,
    in_out_degree,
    load_npz,
    region_preserving_shuffle,
    synthetic_flylike,
)


def test_synthetic_flylike_is_deterministic_and_sparse():
    a = synthetic_flylike(n=48, seed=3)
    b = synthetic_flylike(n=48, seed=3)
    c = synthetic_flylike(n=48, seed=4)
    assert a.n == 48
    assert a.src.ndim == 1 and a.dst.ndim == 1
    assert a.src.shape == a.dst.shape == a.contact_count.shape == a.sign.shape
    assert a.n_edges < a.n * a.n
    assert a.n_edges > a.n  # not a disconnected toy
    assert torch.equal(a.src, b.src) and torch.equal(a.dst, b.dst)
    assert not torch.equal(a.src, c.src) or not torch.equal(a.dst, c.dst)
    assert (a.src != a.dst).all()
    assert int(a.src.max()) < 48 and int(a.dst.max()) < 48


def test_synthetic_has_flylike_structure_not_a_blob():
    g = synthetic_flylike(n=64, seed=1)
    assert g.sensory_idx.numel() >= 4
    assert g.readout_idx.numel() >= 4
    assert g.sensory_idx.numel() < g.n
    assert g.readout_idx.numel() < g.n
    sensory = set(g.sensory_idx.tolist())
    readout = set(g.readout_idx.tolist())
    assert sensory.isdisjoint(readout)
    assert int(g.cell_type.max()) >= 2
    assert (g.sign < 0).any() and (g.sign > 0).any()
    assert float(g.contact_count.std()) > 0
    assert g.region.numel() == g.n


def _region_pairs(g: Connectome) -> list[tuple[int, int]]:
    return list(
        zip(g.region[g.src].tolist(), g.region[g.dst].tolist(), strict=True)
    )


def test_degree_matched_shuffle_preserves_degrees_signs_and_io():
    g = synthetic_flylike(n=56, seed=9)
    sh = degree_matched_shuffle(g, seed=11)
    assert sh.n == g.n
    assert sh.n_edges == g.n_edges
    din_a, dout_a = in_out_degree(g)
    din_b, dout_b = in_out_degree(sh)
    assert torch.equal(din_a, din_b)
    assert torch.equal(dout_a, dout_b)
    assert torch.equal(sh.sign, g.sign)
    assert torch.equal(sh.contact_count, g.contact_count)
    assert torch.equal(sh.cell_type, g.cell_type)
    assert torch.equal(sh.sensory_idx, g.sensory_idx)
    assert torch.equal(sh.readout_idx, g.readout_idx)
    assert not torch.equal(torch.stack([g.src, g.dst]), torch.stack([sh.src, sh.dst]))


def test_region_preserving_shuffle_keeps_region_pairs_and_io_bottleneck():
    g = synthetic_flylike(n=128, seed=0)
    sh = region_preserving_shuffle(g, seed=1)
    dest = degree_matched_shuffle(g, seed=1)
    assert sh.n_edges == g.n_edges
    din_a, dout_a = in_out_degree(g)
    din_b, dout_b = in_out_degree(sh)
    assert torch.equal(din_a, din_b)
    assert torch.equal(dout_a, dout_b)
    assert torch.equal(sh.sign, g.sign)
    assert torch.equal(sh.contact_count, g.contact_count)
    assert torch.equal(sh.sensory_idx, g.sensory_idx)
    assert torch.equal(sh.readout_idx, g.readout_idx)
    assert sorted(_region_pairs(sh)) == sorted(_region_pairs(g))
    fly_sr = sum(
        1
        for s, d in zip(g.src.tolist(), g.dst.tolist())
        if int(g.region[s]) == REGION_SENSORY and int(g.region[d]) == REGION_READOUT
    )
    sh_sr = sum(
        1
        for s, d in zip(sh.src.tolist(), sh.dst.tolist())
        if int(sh.region[s]) == REGION_SENSORY and int(sh.region[d]) == REGION_READOUT
    )
    dest_sr = sum(
        1
        for s, d in zip(dest.src.tolist(), dest.dst.tolist())
        if int(dest.region[s]) == REGION_SENSORY and int(dest.region[d]) == REGION_READOUT
    )
    assert fly_sr == 0
    assert sh_sr == 0
    assert dest_sr > 0
    assert not torch.equal(torch.stack([g.src, g.dst]), torch.stack([sh.src, sh.dst]))
    assert (sh.src == sh.dst).sum() == 0


def test_mushroom_body_expands_kenyon_with_unique_claws():
    g = synthetic_flylike(n=128, seed=0, n_sensory=16, n_readout=8, n_kenyon=96, claws=3)
    assert g.n == 128
    assert int(g.sensory_idx.numel()) == 16
    assert int(g.readout_idx.numel()) == 8
    n_k = int((g.region == REGION_KENYON).sum())
    assert n_k == 96
    assert n_k > 4 * int(g.sensory_idx.numel())
    sk = [
        (int(s), int(d))
        for s, d in zip(g.src.tolist(), g.dst.tolist())
        if int(g.region[s]) == REGION_SENSORY and int(g.region[d]) == REGION_KENYON
    ]
    assert len(sk) == 96 * 3
    assert len(set(sk)) == len(sk)
    sr = sum(
        1
        for s, d in zip(g.src.tolist(), g.dst.tolist())
        if int(g.region[s]) == REGION_SENSORY and int(g.region[d]) == REGION_READOUT
    )
    assert sr == 0
    sh = region_preserving_shuffle(g, seed=1)
    assert sorted(_region_pairs(sh)) == sorted(_region_pairs(g))
    sh_sr = sum(
        1
        for s, d in zip(sh.src.tolist(), sh.dst.tolist())
        if int(sh.region[s]) == REGION_SENSORY and int(sh.region[d]) == REGION_READOUT
    )
    assert sh_sr == 0


def test_class_claws_tune_first_kenyon_cells():
    from flybrain.data.task import make_sparse_patterns

    patterns = make_sparse_patterns(8, 16, 3, seed=2)
    g = synthetic_flylike(
        n=128, seed=0, n_sensory=16, n_readout=8, n_kenyon=96, claws=3, class_claws=patterns
    )
    k0 = int(g.sensory_idx.numel())
    for i in range(8):
        claws = sorted(
            int(s)
            for s, d in zip(g.src.tolist(), g.dst.tolist())
            if int(d) == k0 + i and int(g.region[s]) == REGION_SENSORY
        )
        want = sorted(int(x) for x in patterns[i].tolist())
        assert claws == want
    sh = region_preserving_shuffle(g, seed=3)
    broken = 0
    for i in range(8):
        claws = sorted(
            int(s)
            for s, d in zip(sh.src.tolist(), sh.dst.tolist())
            if int(d) == k0 + i and int(sh.region[s]) == REGION_SENSORY
        )
        want = sorted(int(x) for x in patterns[i].tolist())
        if claws != want:
            broken += 1
    assert broken >= 1
    for i in range(8):
        kc = k0 + i
        r = int(g.readout_idx[i])
        assert any(
            int(s) == kc and int(d) == r for s, d in zip(g.src.tolist(), g.dst.tolist())
        )


def test_n_sensory_is_honored_and_stays_disjoint_from_readout():
    g = synthetic_flylike(n=128, seed=0, n_sensory=32)
    assert g.sensory_idx.numel() == 32
    assert g.n == 128
    tiny = synthetic_flylike(n=128, seed=0, n_sensory=13)
    assert tiny.sensory_idx.numel() == 13
    assert set(tiny.sensory_idx.tolist()).isdisjoint(set(tiny.readout_idx.tolist()))
    wide = synthetic_flylike(n=128, seed=0, n_sensory=64)
    assert wide.sensory_idx.numel() == 64
    assert set(wide.sensory_idx.tolist()).isdisjoint(set(wide.readout_idx.tolist()))


def test_npz_roundtrip(tmp_path):
    g = synthetic_flylike(n=24, seed=2)
    path = tmp_path / "tiny.npz"
    np.savez(
        path,
        src=g.src.numpy(),
        dst=g.dst.numpy(),
        contact_count=g.contact_count.numpy(),
        sign=g.sign.numpy(),
        cell_type=g.cell_type.numpy(),
        region=g.region.numpy(),
        sensory_idx=g.sensory_idx.numpy(),
        readout_idx=g.readout_idx.numpy(),
    )
    loaded = load_npz(path)
    assert isinstance(loaded, Connectome)
    assert loaded.n == g.n
    assert torch.equal(loaded.src, g.src)
    assert torch.equal(loaded.sensory_idx, g.sensory_idx)
