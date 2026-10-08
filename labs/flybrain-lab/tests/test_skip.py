"""--skip-frac (bet C): the stack runs only where a cheap 0-layer gate over the input is unsure. With everything kept the model is the plain
model; a skipped position carries the last kept position's output; nothing after a position changes its logits; the gate's threshold is a
running quantile that eval freezes; the runner records the flag and rebuilds with it."""
import math

import torch
import torch.nn.functional as F

from flybrain.model.hybrid import GRUHybrid
from flybrain.model.inputs import channel_kwargs


def _model(skip_frac=0.0, seed=0, **kw):
    torch.manual_seed(seed)
    m = GRUHybrid(d_model=32, hidden=32, key="sparse", where="none", backbone="transformer", n_layers=2, d_ff=64, rope=True, max_len=1024,
                  seed=seed, match_orders=(4, 8), fused_channels=True, skip_frac=skip_frac, **kw)
    for a in m.core.attn:
        a.use_rope_table(1024)
    return m


def _batch(B=3, T=48, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (B, T), generator=g)


def test_everything_kept_is_the_plain_model():
    plain, skip = _model(), _model(skip_frac=0.5)
    for p in skip.match.parameters():
        p.data.normal_(std=0.1)
    plain.match.load_state_dict(skip.match.state_dict())
    assert all(torch.equal(p, q) for p, q in zip(plain.core.parameters(), skip.core.parameters()))   # the plain parameters are drawn as before
    skip.eval(); plain.eval()
    skip.skip_tau.fill_(-1.0)                      # every entropy is above the threshold: nothing is skipped
    x = _batch()
    assert torch.allclose(plain(x)[0], skip(x)[0], atol=1e-5)
    assert skip.skip_params() == sum(p.numel() for m in (skip.skip_ln, skip.skip_head, skip.skip_pool, skip.skip_local) for p in m.parameters())


def test_skipped_positions_carry_the_last_kept_output_and_nothing_else_at_init():
    m = _model(skip_frac=0.5).eval()
    x = _batch()
    e = m.embed(x)
    mpred, mcls = __import__("flybrain.model.inputs", fromlist=["match_features"]).match_features(x, m.match_orders)
    e = e + m.match.embed(mpred, mcls)
    lc = m.skip_head(m.skip_ln(e)); lp = F.log_softmax(lc, -1); ent = -(lp.exp() * lp).sum(-1)
    m.skip_tau.fill_(float(ent.median()))
    keep = ent > m.skip_tau; keep[:, 0] = True
    out = m._skip_forward(e, None)
    # the kept rows: the plain stack on the kept subsequence at its original positions (the pooled projection is zero at init)
    for b in range(x.size(0)):
        idx = keep[b].nonzero().flatten()
        ref = m.core(e[b:b + 1, idx], idx[None])
        assert torch.allclose(out[b, idx], ref[0], atol=1e-5)
        # a skipped row carries the output of the last kept row before it (the local projection is zero at init)
        for p in (~keep[b]).nonzero().flatten().tolist():
            last = idx[idx < p].max()
            assert torch.allclose(out[b, p], out[b, last], atol=1e-6)
    assert 0.3 < (~keep).float().mean() < 0.7


def test_nothing_after_a_position_changes_its_logits():
    m = _model(skip_frac=0.4)
    m.train(); x = _batch(T=64)
    m.forward_skip(x)                              # sets the threshold from this batch
    m.eval()
    y = x.clone(); y[:, 40:] = torch.randint(0, 256, (x.size(0), 24))
    la, lb = m(x)[0], m(y)[0]
    assert torch.allclose(la[:, :40], lb[:, :40], atol=1e-5)


def test_threshold_is_a_running_quantile_frozen_at_eval():
    m = _model(skip_frac=0.3, skip_momentum=0.5)
    assert torch.isnan(m.skip_tau)
    m.train(); x = _batch(T=64)
    logits, gate = m.forward_skip(x)
    assert logits.shape == gate.shape == (3, 64, 256)
    t1 = float(m.skip_tau); assert math.isfinite(t1)
    m.forward_skip(_batch(T=64, seed=5))
    t2 = float(m.skip_tau); assert t2 != t1
    m.eval(); m(x)
    assert float(m.skip_tau) == t2
    loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1)) + F.cross_entropy(gate.reshape(-1, 256), x.reshape(-1))
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in (m.skip_head.weight, m.skip_pool.weight, m.skip_local.weight))


def test_runner_records_the_flag_and_rebuilds():
    from scripts.cross_eval import rebuild
    from scripts.run_sparsela import make_model, parse_args, recorded_args
    base = ["--arms", "transformer", "--rope", "--rope-table", "--d-model", "32", "--d-ff", "64", "--n-layers", "2", "--seq-len", "48", "--device", "cpu", "--match-feat", "4,8", "--fused-channels"]
    off, on = parse_args(base), parse_args([*base, "--skip-frac", "0.4", "--skip-momentum", "0.9"])
    assert channel_kwargs(recorded_args(off)) == {"match_orders": (4, 8), "fused_channels": True}
    assert channel_kwargs(recorded_args(on)) == {"match_orders": (4, 8), "fused_channels": True, "skip_frac": 0.4, "skip_momentum": 0.9}
    m = make_model("transformer", on, torch.device("cpu"))
    assert m.skip_frac == 0.4 and m.skip_momentum == 0.9
    m.train(); m.forward_skip(_batch())
    m.eval(); x = _batch(seed=7); la = m(x)[0]
    ck = {"arm": "transformer", "args": recorded_args(on), "state": m.state_dict()}
    r = rebuild(ck).eval()
    assert r.skip_frac == 0.4 and float(r.skip_tau) == float(m.skip_tau)
    for a in r.core.attn:
        a.use_rope_table(1024)
    assert torch.allclose(r(x)[0], la, atol=1e-5)
