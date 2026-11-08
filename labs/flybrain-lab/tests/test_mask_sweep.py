import torch


def test_copyable_masks_are_nested_and_bins_partition():
    from flybrain.data.corpus import copyable_mask
    from scripts.mask_sweep import longest_match_bin, NS
    torch.manual_seed(0)
    x = torch.randint(0, 4, (3, 200))  # tiny alphabet so repeats are common
    y = torch.cat([x[:, 1:], x[:, :1]], dim=1)
    masks = {n: copyable_mask(x, y, n=n) for n in NS}
    for a, b in zip(NS, NS[1:]):  # copyable at longer n implies copyable at shorter n
        assert bool((masks[b] & ~masks[a]).sum() == 0), (a, b)
    bins = longest_match_bin(x, y)
    assert bins.shape == x.shape
    assert set(bins.unique().tolist()) <= set([0] + list(NS))
    assert bool((bins == 8).sum() == masks[8].sum())
    assert bool(((bins == 0) == ~masks[NS[0]]).all())
