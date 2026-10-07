"""--fused-channels: FeatureChannel without its loop over orders. The same parameters, the same numbers (up to float summation order),
the same gradients; a checkpoint records the flag only when it is on, and rebuilds with it."""
import torch

from flybrain.model.hybrid import GRUHybrid
from flybrain.model.inputs import FeatureChannel, FusedFeatureChannel, N_CONF_CLASSES, channel_kwargs


def _pair(n_orders=5, n_classes=N_CONF_CLASSES, d_model=32, d_out=48, seed=0):
    torch.manual_seed(seed)
    ch = FeatureChannel(d_model, d_out, n_orders, n_classes)
    for p in ch.parameters():                       # zero-initialised by construction: give both something to compare
        p.data.normal_()
    return ch, FusedFeatureChannel.from_channel(ch, d_out)


def _inputs(n_orders, n_classes, B=3, T=17, seed=1):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randint(0, 256, (B, T, n_orders), generator=g)
    cls = torch.randint(0, n_classes, (B, T, n_orders), generator=g)   # class 0 = a miss, with its own pred value that must be ignored
    return pred, cls


def test_same_parameter_count_embedding_and_bonus():
    ch, fu = _pair()
    assert sum(p.numel() for p in ch.parameters()) == sum(p.numel() for p in fu.parameters())
    pred, cls = _inputs(5, N_CONF_CLASSES)
    assert torch.allclose(ch.embed(pred, cls), fu.embed(pred, cls), atol=1e-5)
    h = torch.randn(3, 17, 48); logits = torch.randn(3, 17, 256)
    assert torch.allclose(ch.add_bonus(logits, h, pred, cls), fu.add_bonus(logits, h, pred, cls), atol=1e-5)


def test_misses_add_only_their_class_embedding_and_no_bonus():
    _, fu = _pair()
    pred, cls = _inputs(5, N_CONF_CLASSES)
    cls = torch.zeros_like(cls)
    e = fu.embed(pred, cls)
    expect = fu.class_emb.weight[fu.class_off].sum(0)        # row 0 of every order's class table
    assert torch.allclose(e, expect.expand_as(e), atol=1e-5)
    logits = torch.randn(3, 17, 256)
    assert torch.equal(fu.add_bonus(logits, torch.randn(3, 17, 48), pred, cls), logits)


def test_same_gradients():
    ch, fu = _pair()
    pred, cls = _inputs(5, N_CONF_CLASSES)
    h = torch.randn(3, 17, 48); logits = torch.randn(3, 17, 256)
    for m in (ch, fu):
        (m.embed(pred, cls).pow(2).sum() + m.add_bonus(logits, h, pred, cls).pow(2).sum()).backward()
    assert torch.allclose(torch.cat([e.weight.grad for e in ch.byte_emb]), fu.byte_emb.weight.grad, atol=1e-4)
    assert torch.allclose(torch.cat([e.weight.grad for e in ch.class_emb]), fu.class_emb.weight.grad, atol=1e-4)
    assert torch.allclose(torch.cat([b.weight.grad for b in ch.bonus]), fu.bonus.weight.grad, atol=1e-4)
    assert torch.allclose(torch.cat([b.bias.grad for b in ch.bonus]), fu.bonus.bias.grad, atol=1e-4)


def _tf(fused: bool, seed: int = 0) -> GRUHybrid:
    torch.manual_seed(seed)
    return GRUHybrid(d_model=64, key="sparse", where="none", backbone="transformer", n_layers=2, d_ff=128, max_len=64, rope=True,
                     mem_orders=(8, 16, 32, 4), match_orders=(8, 16), fused_channels=fused)


def test_model_with_fused_channels_matches_the_loop():
    a, b = _tf(False), _tf(True)
    for p in [*a.table.parameters(), *a.match.parameters()]:
        p.data.normal_(std=0.1)
    b.table, b.match = FusedFeatureChannel.from_channel(a.table, 64), FusedFeatureChannel.from_channel(a.match, 64)
    x = torch.randint(0, 256, (2, 40))
    pred = torch.randint(0, 256, (2, 40, 4), dtype=torch.uint8); conf = torch.randint(0, 20, (2, 40, 4), dtype=torch.uint8)
    la, lb = a(x, mem_pred=pred, mem_conf=conf)[0], b(x, mem_pred=pred, mem_conf=conf)[0]
    assert torch.allclose(la, lb, atol=1e-4)
    assert isinstance(b.table, FusedFeatureChannel) and isinstance(a.table, FeatureChannel)


def test_runner_flag_is_recorded_only_when_on_and_rebuilds():
    from scripts.run_sparsela import make_model, parse_args, recorded_args
    base = ["--arms", "transformer", "--rope", "--d-model", "64", "--d-ff", "128", "--n-layers", "2", "--seq-len", "32", "--device", "cpu", "--match-feat", "8,16"]
    off, on = parse_args(base), parse_args([*base, "--fused-channels"])
    assert "fused_channels" not in recorded_args(off) and recorded_args(on)["fused_channels"] is True
    assert channel_kwargs(recorded_args(on)) == {"match_orders": (8, 16), "fused_channels": True}
    assert channel_kwargs(recorded_args(off)) == {"match_orders": (8, 16)}
    m = make_model("transformer", on, torch.device("cpu"))
    assert isinstance(m.match, FusedFeatureChannel) and m.table is None
    assert isinstance(make_model("transformer", off, torch.device("cpu")).match, FeatureChannel)
    from scripts.cross_eval import rebuild           # constructs from the recorded args and loads the state
    ck = {"arm": "transformer", "args": recorded_args(on), "state": m.state_dict()}
    r = rebuild(ck)
    assert isinstance(r.match, FusedFeatureChannel) and all(torch.equal(p, q) for p, q in zip(r.parameters(), m.parameters()))
