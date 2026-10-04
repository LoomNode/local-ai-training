"""The local-learning switches of scripts/run_sparsela.py (speed round, row 3): --train-only, --freeze-layers, --local-blocks.
CPU only, tiny models; every checkpoint is written to a temporary directory."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from flybrain.data.text import fixed_windows, load_bytes
from flybrain.model.hybrid import GRUHybrid
from flybrain.model.sparsela import SparseKeyLM

CPU = torch.device("cpu")
ROOT = Path(__file__).resolve().parents[1]
READOUT = 16 * 2 + 16 * 256 + 256       # a final (or block) readout at d_model 16: LayerNorm + Linear(16, 256)


def tiny(n_layers: int = 4, **kw) -> GRUHybrid:
    """The spine hybrid in miniature: minGRU stack, an inner head after every 2nd layer plus the top head, one local-attention layer
    mid-stack, key widths 2 and 4. With 4 layers: one inner head after layer 1 (0-based) and local attention after layer 1. With 6: inner
    heads after layers 1 and 3, local attention after layer 2."""
    torch.manual_seed(0)
    cfg = dict(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=n_layers, d_ff=32, n_key=64, top_k=4, key_conv=2,
               normalize=True, gate=True, chunk=8, mem_every=2, local_attn=8, key_convs=(2, 4))
    cfg.update(kw)
    return GRUHybrid(**cfg)


def batch(n: int = 2, t: int = 32, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, t), generator=g), torch.randint(0, 256, (n, t), generator=g)


def ce(logits, y):
    return F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))


def ids(params) -> set:
    return {id(p) for p in params}


def numel(params) -> int:
    return sum(p.numel() for p in params)


def hybrid_args(*flags):
    """parse_args on the hybrid arm: the switches refuse the other arms at parse time."""
    from scripts.run_sparsela import parse_args
    return parse_args(["--arms", "mingru_sparse_post", *flags])


def snapshot(m) -> dict:
    return {k: v.clone() for k, v in m.state_dict().items()}   # parameters and buffers


def grads(m) -> dict:
    return {n: None if p.grad is None else p.grad.clone() for n, p in m.named_parameters()}


def one_step(m, args, x, y):
    """The runner's step: loss, backward, clip, optimizer step. Returns (optimizer, loss, block losses)."""
    from scripts.run_sparsela import make_optimizer, train_loss
    opt = make_optimizer(m, args)
    loss, parts = train_loss(m, x, y, args, CPU)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return opt, loss, parts


# ---- with the switches off nothing changes ---------------------------------------------------------------------------------------

def _old_forward(m, tokens):
    """GRUHybrid.forward and _run_core exactly as they were before the switches (commit 3cee1fe), for the mingru backbone."""
    e = m.embed(tokens)
    layers = m.core.layers
    if not m.mem_every and not m.local_attn:
        out = m.core(e)
    else:
        k = 0
        for i, l in enumerate(layers):
            e = l(e)
            if m.local_attn and i + 1 in m.local_at:
                j = m.local_at.index(i + 1)
                e = e + (m.local if j == 0 else m.local_more[j - 1])(e)
            if m.mem_every and (i + 1) % m.mem_every == 0 and i + 1 < len(layers):
                e = m._head(m.mems[k], e); k += 1
        assert k == len(m.mems)
        out = m.core.ln(e)
    if m.where == "post":
        z = m.mem.ln1(out)
        out = out + m.mem.memory(z, None)
    return m.decoder(out)


@pytest.mark.parametrize("kw", [{}, {"local_blocks": 0}, {"where": "none", "mem_every": 0}, {"where": "none", "mem_every": 0, "local_attn": 0}],
                         ids=["hybrid", "hybrid-explicit-off", "zero-heads", "plain-backbone"])
def test_without_the_switches_logits_loss_and_gradients_are_bit_identical_to_the_old_forward(kw):
    m = tiny(**kw)
    x, y = batch()
    logits = m(x)[0]
    loss = ce(logits, y); loss.backward()
    new = grads(m)
    m.zero_grad(set_to_none=True)
    ref_logits = _old_forward(m, x)
    ref_loss = ce(ref_logits, y); ref_loss.backward()
    assert torch.equal(logits, ref_logits) and torch.equal(loss, ref_loss)
    for n, p in m.named_parameters():
        assert (new[n] is None) == (p.grad is None), n
        if p.grad is not None:
            assert torch.equal(new[n], p.grad), n
    assert not any("aux" in k for k in m.state_dict()) and m.aux_scalars == 0     # an unswitched model has exactly the old state_dict keys


def test_the_switch_defaults_are_off_and_the_default_optimizer_holds_every_parameter():
    from scripts.run_sparsela import make_optimizer, parse_args
    args = parse_args(["--wd", "0.1"])
    assert (args.train_only, args.freeze_layers, args.local_blocks) == ("", "", 0)
    m = tiny()
    for a in (args, parse_args([])):                                              # AdamW with decay groups, and plain Adam
        held = [p for g in make_optimizer(m, a).param_groups for p in g["params"]]
        assert len(held) == len(list(m.parameters())) and ids(held) == ids(m.parameters())


# ---- runs through main(): tiny, in-process, results in tmp_path -----------------------------------------------------------------

SMALL = ["--task", "code", "--train-bytes", "3000", "--val-bytes", "4000", "--d-model", "16", "--n-layers", "4", "--d-ff", "32",
         "--n-key", "64", "--top-k", "4", "--key-conv", "2", "--normalize", "--chunk", "8", "--batch", "2", "--seq-len", "32",
         "--steps", "6", "--lr", "0.003", "--arms", "mingru_sparse_post", "--gate", "--mem-every", "2", "--local-attn", "8",
         "--key-convs", "2,4", "--wd", "0.1", "--save", "--seed", "0", "--device", "cpu", "--run-name", "tiny"]
FIXED = ["--val-fixed", "4", "--val-offset", "110000"]    # 4 windows of 32 bytes, past the 3000 train and the val bytes at 103000
SWITCH_KEYS = ("train_only", "freeze_layers", "local_blocks")
NEW_ARM_KEYS = {"train_only", "frozen_layers", "trainable_by_group", "total_scalars", "local_blocks", "n_blocks", "added_scalars"}


def _corpus(tmp_path):
    path = tmp_path / "tiny.bytes"
    path.write_bytes((b"int main(void) { return 0; }\n" * 5000)[:130_000])
    return path


def _argv(tmp_path, *flags):
    return ["run_sparsela.py", "--corpus-file", str(_corpus(tmp_path)), *SMALL, *flags]


def _run(monkeypatch, tmp_path, tag, *flags):
    """main() in-process with the results in tmp_path/tag; returns (the run's JSON, the results dir)."""
    import scripts.run_sparsela as sra
    res = tmp_path / tag
    res.mkdir()
    monkeypatch.setattr(sra, "RESULTS", res)
    monkeypatch.setattr(sra, "_CMASK", {})
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, *flags))
    sra.main()
    return json.loads((res / "tiny.json").read_text()), res


def _ckpt(res):
    return torch.load(res / "ckpt" / "tiny_mingru_sparse_post.pt", map_location="cpu", weights_only=False)


def test_a_run_with_the_switches_written_out_as_off_is_the_run_without_them(monkeypatch, tmp_path):
    a, ra = _run(monkeypatch, tmp_path, "a")
    b, rb = _run(monkeypatch, tmp_path, "b", "--train-only", "", "--freeze-layers", "", "--local-blocks", "0")
    sa, sb = _ckpt(ra)["state"], _ckpt(rb)["state"]
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)             # same weights, bit for bit
    for out, res in ((a, ra), (b, rb)):
        assert not set(SWITCH_KEYS) & set(out) and not set(SWITCH_KEYS) & set(_ckpt(res)["args"])   # the legacy file: no new key anywhere
        assert list(out["arms"][0]) == ["val_bpb", "val_bpb_first_q", "val_bpb_last_q", "val_bpb_copyable", "copyable_frac", "val_bpb_noncopyable",
                                        "arm", "seconds", "trainable_scalars", "curve", "flops_per_token_recurrent"]
        assert all(set(r) == {"step", "train_bpb", "val_bpb", "val_bpb_first_q", "val_bpb_last_q", "val_bpb_copyable", "copyable_frac",
                              "val_bpb_noncopyable"} for r in out["arms"][0]["curve"])
    for k, v in a["arms"][0].items():
        if k != "seconds":
            assert b["arms"][0][k] == v, k


# ---- --train-only ---------------------------------------------------------------------------------------------------------------

def test_train_only_refuses_an_unknown_group(capsys):
    from scripts.run_sparsela import apply_freezing, parse_args
    for spec in ("readout,backbone", "Readout", "readout,,embed", "readout,"):
        with pytest.raises(SystemExit):
            parse_args(["--train-only", spec])
        err = capsys.readouterr().err
        assert "--train-only" in err and "readout, embed, heads, norms" in err, spec           # says what the groups are
        m = tiny()
        with pytest.raises(ValueError, match="readout, embed, heads, norms"):
            apply_freezing(m, train_only=spec)
        assert all(p.requires_grad for p in m.parameters())                                    # a refused spec freezes nothing


def test_the_groups_map_to_the_parameters_the_spec_names():
    m = tiny(n_layers=6)
    g = m.param_groups()
    assert set(g) == {"readout", "embed", "heads", "norms"}
    assert ids(g["readout"]) == ids(m.core.ln.parameters()) | ids(m.decoder.parameters())                 # the final norm and the output projection
    assert ids(g["embed"]) == ids(m.embed.parameters())
    assert len(m.mems) == 2 and ids(g["heads"]) == ids(p for h in (m.mem, *m.mems) for p in h.parameters())   # every parameter of every head
    for h in (m.mem, *m.mems):                                                                   # the pieces the lab's record lists for one head
        assert ids([h.gate_tau, h.code.theta, h.val.weight, h.outs[0].weight, h.conv.weight, h.ln1.weight]) <= ids(g["heads"])
    assert all("proj" not in n for n, _ in m.named_parameters())                                 # the frozen address is a buffer, in no group
    norms = [p for mod in m.modules() if isinstance(mod, nn.LayerNorm) for p in mod.parameters()]
    assert ids(g["norms"]) == ids(norms) and len(norms) == 2 * (2 * 6 + 1 + 1 + 3 * 2)                  # 6 layers x 2, final, local attention, 3 heads x 2
    assert ids(g["readout"]) & ids(g["norms"]) == ids(m.core.ln.parameters())                      # the groups overlap on the final norm
    assert not ids(g["embed"]) & (ids(g["readout"]) | ids(g["heads"]) | ids(g["norms"]))
    for name, ps in g.items():
        assert len(ps) == len(ids(ps)), name                                                       # no parameter listed twice


def test_train_only_readout_trains_only_the_readout_and_prints_the_counts(capsys):
    from scripts.run_sparsela import apply_freezing, make_optimizer
    m = tiny(); init = snapshot(m)
    total = numel(m.parameters())
    rep = apply_freezing(m, train_only="readout")
    readout = ids(m.param_groups()["readout"])
    assert ids(p for p in m.parameters() if p.requires_grad) == readout
    assert rep["trainable_scalars"] == READOUT and rep["total_scalars"] == total
    assert rep["trainable_by_group"] == {"readout": READOUT, "embed": 0, "heads": 0, "norms": 32, "other": 0}
    args = hybrid_args("--wd", "0.1", "--lr", "0.01", "--train-only", "readout")
    opt = make_optimizer(m, args)
    assert ids(p for g in opt.param_groups for p in g["params"]) == readout                         # left out of the optimizer
    plain = make_optimizer(m, hybrid_args("--lr", "0.01", "--train-only", "readout"))               # the plain-Adam branch (no weight decay) too
    assert ids(p for g in plain.param_groups for p in g["params"]) == readout
    one_step(m, args, *batch())
    for k, v in m.state_dict().items():                                                          # parameters and the frozen address buffers
        if k.startswith(("core.ln.", "decoder.")):
            assert not torch.equal(v, init[k]), k
        else:
            assert torch.equal(v, init[k]), k                                                     # bit-equal to the initial value, weight decay included
    out = capsys.readouterr().out
    assert f"trainable {READOUT:,} of {total:,}" in out
    for name, n in rep["trainable_by_group"].items():
        assert re.search(rf"\b{name} {n:,}/", out), (name, out)                                    # each group: trainable/size


def test_train_only_readout_embed_and_heads_leaves_the_stack_frozen():
    from scripts.run_sparsela import apply_freezing, parse_args
    m = tiny(n_layers=6); init = snapshot(m)
    rep = apply_freezing(m, train_only="readout,embed,heads")
    g = m.param_groups()
    want = ids(g["readout"]) | ids(g["embed"]) | ids(g["heads"])
    assert ids(p for p in m.parameters() if p.requires_grad) == want
    assert not any(p.requires_grad for l in m.core.layers for p in l.parameters())                  # the stack's layers (and their norms)
    assert not any(p.requires_grad for p in m.local.parameters())
    assert rep["trainable_by_group"]["heads"] == numel(g["heads"])                                  # all of the heads, the placeholder FFNs included
    one_step(m, parse_args(["--wd", "0.1", "--lr", "0.01"]), *batch())
    for n, p in m.named_parameters():
        if id(p) not in want:
            assert torch.equal(p, init[n]), n
    assert not torch.equal(m.embed.weight, init["embed.weight"])                                    # a gradient reaches the embedding through the frozen stack
    assert not torch.equal(m.mem.val.weight, init["mem.val.weight"])


def test_a_short_readout_only_run_lowers_the_loss(monkeypatch, tmp_path):
    import scripts.run_sparsela as sra
    flags = [*FIXED, "--steps", "60", "--lr", "0.01", "--train-only", "readout"]
    out, res = _run(monkeypatch, tmp_path, "r", *flags)
    args = sra.parse_args(_argv(tmp_path, *flags)[1:])
    xf, yf = fixed_windows(load_bytes(_corpus(tmp_path), max_bytes=4 * 32 + 1, offset=110000), 32, 4)
    init = sra.build("mingru_sparse_post", args)                                                     # same seed: the model the run started from
    before = sra.eval_fixed(init, xf, yf, args, CPU)["bpb"]
    after = out["arms"][0]["val_fixed"]["bpb"]
    assert before > 7.5 and after < before - 4.0, (before, after)      # 8.0 -> 0.86 bits/byte in 60 steps
    ck = _ckpt(res)["state"]
    assert all(torch.equal(ck[k], v) for k, v in init.state_dict().items() if not k.startswith(("core.ln.", "decoder.")))   # all else is still its initial value
    assert out["arms"][0]["trainable_scalars"] == READOUT and out["arms"][0]["train_only"] == ["readout"]


# ---- --freeze-layers ------------------------------------------------------------------------------------------------------------

def test_freeze_layers_refuses_an_index_outside_the_stack(capsys):
    from scripts.run_sparsela import apply_freezing, parse_args
    for spec in ("1,4", "-1", "8", "1,x", "1,,2"):
        with pytest.raises(SystemExit):
            parse_args(["--n-layers", "4", f"--freeze-layers={spec}"])
        err = capsys.readouterr().err
        assert "--freeze-layers" in err and ("outside" in err or "integer" in err), (spec, err)
        m = tiny()                                                                                 # a stack of 4 layers
        with pytest.raises(ValueError, match="--freeze-layers"):
            apply_freezing(m, freeze_layers=spec)
        assert all(p.requires_grad for p in m.parameters())
    assert hybrid_args("--n-layers", "8", "--freeze-layers", "1,3,5,7").freeze_layers == "1,3,5,7"
    with pytest.raises(ValueError, match="layered"):                                               # a single nn.GRU has no stack to index
        apply_freezing(GRUHybrid(d_model=16, hidden=16, key="sparse", where="none", backbone="gru"), freeze_layers="0")
    for spec in ({"train_only": "readout"}, {"freeze_layers": "0"}):                               # and the switches are for the GRUHybrid arms only
        with pytest.raises(ValueError, match="GRUHybrid"):
            apply_freezing(SparseKeyLM(key="sparse", d_model=16, n_key=64, n_layers=1), **spec)


def test_layer_units_follow_the_order_the_modules_run_in():
    """The head placed after a layer is the one that runs right after it: found from the call order, not from the placement rule."""
    m = tiny(n_layers=6)
    order = []
    names = {id(l): f"L{i}" for i, l in enumerate(m.core.layers)}
    for l in m.core.layers:
        l.register_forward_hook(lambda mod, i, o: order.append(names[id(mod)]))
    for k, h in enumerate(m.mems):
        h.ln1.register_forward_hook(lambda mod, i, o, k=k: order.append(f"head{k}"))
    m.local.register_forward_hook(lambda mod, i, o: order.append("local"))
    m(batch()[0])
    assert order == ["L0", "L1", "head0", "L2", "local", "L3", "head1", "L4", "L5"]
    units = m.layer_units()
    assert [len(u) for u in units] == [1, 2, 1, 2, 1, 1]
    assert [id(u[0]) for u in units] == [id(l) for l in m.core.layers]
    assert id(units[1][1]) == id(m.mems[0]) and id(units[3][1]) == id(m.mems[1])                   # the heads that ran after layers 1 and 3
    assert not any(id(mod) in {id(m.local), id(m.mem)} for u in units for mod in u)                # the local attention and the top head are in no layer


def test_freeze_layers_freezes_the_layer_its_norms_and_its_head_and_nothing_else():
    from scripts.run_sparsela import apply_freezing, make_optimizer
    m = tiny(n_layers=6); init = snapshot(m)
    rep = apply_freezing(m, freeze_layers="1,3")
    frozen_mods = (m.core.layers[1], m.core.layers[3], m.mems[0], m.mems[1])
    frozen = ids(p for mod in frozen_mods for p in mod.parameters())
    assert frozen == ids(p for i in (1, 3) for mod in m.layer_units()[i] for p in mod.parameters())
    assert ids(p for p in m.parameters() if not p.requires_grad) == frozen
    assert ids(m.core.layers[1].ln1.parameters()) <= frozen and ids(m.core.layers[3].ln2.parameters()) <= frozen   # the layer's norms
    for mod in (m.core.layers[0], m.core.layers[2], m.core.layers[4], m.core.layers[5], m.local, m.mem, m.core.ln, m.decoder, m.embed):
        assert all(p.requires_grad for p in mod.parameters())                                       # the rest of the model still learns
    assert rep["frozen_layers"] == [1, 3] and rep["trainable_scalars"] == numel(p for p in m.parameters() if p.requires_grad)
    assert rep["total_scalars"] - rep["trainable_scalars"] == numel(p for mod in frozen_mods for p in mod.parameters())
    args = hybrid_args("--wd", "0.1", "--lr", "0.01", "--n-layers", "6", "--freeze-layers", "1,3")
    assert ids(p for g in make_optimizer(m, args).param_groups for p in g["params"]) == ids(m.parameters()) - frozen   # left out of the optimizer
    one_step(m, args, *batch())
    for k, v in m.state_dict().items():
        if k.startswith(("core.layers.1.", "core.layers.3.", "mems.")):
            assert torch.equal(v, init[k]), k                                                        # frozen: bit-equal after the step (the heads' address too)
    for k in ("core.layers.0.z.weight", "core.layers.2.z.weight", "core.layers.5.z.weight", "local.qkv.weight", "mem.val.weight"):
        assert not torch.equal(m.state_dict()[k], init[k]), k                                        # the local attention is in no layer: it keeps learning


def test_a_trainable_layer_below_a_frozen_one_still_gets_the_gradient():
    from scripts.run_sparsela import apply_freezing
    x, y = batch()
    full, frozen = tiny(), tiny()
    apply_freezing(frozen, freeze_layers="1")           # layer 1 and the head after it, with the trainable layer 0 below and layers 2, 3 above
    for m in (full, frozen):
        ce(m(x)[0], y).backward()
    g = frozen.core.layers[0].z.weight.grad
    assert g is not None and float(g.abs().sum()) > 0
    gf, gz = grads(full), grads(frozen)
    for (n, p) in frozen.named_parameters():
        if p.requires_grad:                                                                          # freezing a layer changes no gradient of the others
            assert (gz[n] is None) == (gf[n] is None), n
            if gz[n] is not None:
                assert torch.allclose(gz[n], gf[n], rtol=1e-5, atol=1e-8), n
        else:
            assert gz[n] is None, n
    assert float(frozen.embed.weight.grad.abs().sum()) > 0                                           # all the way down to the embedding


def test_freeze_layers_on_the_transformer_arm():
    from scripts.run_sparsela import apply_freezing
    torch.manual_seed(0)
    m = GRUHybrid(d_model=16, hidden=16, key="sparse", where="none", backbone="transformer", n_layers=4, d_ff=32, rope=True, max_len=64)
    apply_freezing(m, freeze_layers="0,2")
    c = m.core
    unit = lambda i: [p for mod in (c.ln1[i], c.ln2[i], c.attn[i], c.ff[i]) for p in mod.parameters()]
    assert not any(p.requires_grad for i in (0, 2) for p in unit(i))
    assert all(p.requires_grad for i in (1, 3) for p in unit(i))
    assert all(p.requires_grad for mod in (m.embed, m.decoder, c.ln) for p in mod.parameters())
    x, y = batch()
    ce(m(x)[0], y).backward()
    assert float(c.attn[1].qkv.weight.grad.abs().sum()) > 0                                          # a trainable layer between frozen ones learns


def test_train_only_and_freeze_layers_combine_as_an_intersection():
    from scripts.run_sparsela import apply_freezing
    m = tiny(n_layers=6)
    rep = apply_freezing(m, train_only="heads,norms", freeze_layers="1")
    g = m.param_groups()
    unit1 = ids(p for mod in m.layer_units()[1] for p in mod.parameters())
    assert ids(p for p in m.parameters() if p.requires_grad) == (ids(g["heads"]) | ids(g["norms"])) - unit1
    assert not any(p.requires_grad for p in m.mems[0].parameters())                                  # the head in the frozen layer is out although "heads" is on
    assert all(p.requires_grad for p in m.mems[1].parameters()) and all(p.requires_grad for p in m.mem.parameters())
    assert rep["train_only"] == ["heads", "norms"] and rep["frozen_layers"] == [1]
    assert rep["trainable_scalars"] == numel(p for p in m.parameters() if p.requires_grad)


def test_a_spec_that_leaves_nothing_to_train_is_refused_before_anything_is_frozen():
    from scripts.run_sparsela import apply_freezing
    m = tiny(where="none", mem_every=0)                                                                # the zero-heads arm: no head to train
    with pytest.raises(ValueError, match="leaves no parameter to train"):
        apply_freezing(m, train_only="heads")
    assert all(p.requires_grad for p in m.parameters())                                                # nothing was frozen


def test_train_only_norms_has_no_weight_decay_group_and_still_trains():
    from scripts.run_sparsela import apply_freezing, make_optimizer
    m = tiny(); init = snapshot(m)
    apply_freezing(m, train_only="norms")
    args = hybrid_args("--wd", "0.1", "--lr", "0.01", "--train-only", "norms")
    opt = make_optimizer(m, args)                                                                      # every norm parameter is 1-D: the decay group would be empty
    assert len(opt.param_groups) == 1 and opt.param_groups[0]["weight_decay"] == 0.0
    one_step(m, args, *batch())
    norms = ids(m.param_groups()["norms"])
    for n, p in m.named_parameters():
        if id(p) not in norms:
            assert torch.equal(p, init[n]), n
    for k in ("core.layers.0.ln1.weight", "core.layers.3.ln2.bias", "core.ln.weight", "mem.ln1.weight"):
        assert not torch.equal(m.state_dict()[k], init[k]), k


@pytest.mark.parametrize("flags, kw", [(["--local-blocks", "2"], {"local_blocks": 2}), (["--train-only", "readout"], {}), (["--freeze-layers", "1"], {})],
                         ids=["local-blocks", "train-only", "freeze-layers"])
def test_the_switches_run_under_bf16_autocast(flags, kw):
    """The GPU runs use --amp: the block readouts and the frozen stack must work under autocast too (CPU bf16 here)."""
    from scripts.run_sparsela import apply_freezing
    m = tiny(**kw)
    args = hybrid_args("--amp", "--wd", "0.1", "--lr", "0.01", *flags)
    apply_freezing(m, args.train_only, args.freeze_layers)
    x, y = batch()
    _, loss, parts = one_step(m, args, x, y)
    assert torch.isfinite(loss) and all(torch.isfinite(p) for p in parts) and len(parts) == (2 if kw else 1)
    assert all(torch.isfinite(p).all() for p in m.parameters())


# ---- --local-blocks -------------------------------------------------------------------------------------------------------------

def test_local_blocks_leave_the_initial_weights_and_the_normal_output_unchanged():
    plain, blocked = tiny(), tiny(local_blocks=2)
    sp, sb = plain.state_dict(), blocked.state_dict()
    assert set(sb) - set(sp) == {f"aux_{part}.0.{w}" for part in ("norms", "decoders") for w in ("weight", "bias")}
    assert set(sp) <= set(sb) and all(torch.equal(sp[k], sb[k]) for k in sp)                          # same seed, same initial weights (the readouts are built last)
    x, _ = batch()
    for train in (True, False):
        plain.train(train); blocked.train(train)
        with torch.set_grad_enabled(train):
            want = plain(x)[0]
            assert torch.equal(want, blocked(x)[0])                                                   # the detach changes no value
            assert torch.equal(want, blocked.forward_blocks(x)[0])
    assert blocked.n_blocks == 2 and blocked.aux_scalars == READOUT and plain.aux_scalars == 0


def test_the_first_block_gets_no_gradient_from_the_last_loss_and_some_from_its_own():
    m = tiny(local_blocks=2)                      # blocks: layers 0-1 | layers 2-3
    x, y = batch()
    logits, aux = m.forward_blocks(x)
    assert len(aux) == 1 and aux[0].shape == logits.shape
    first = {"embed": m.embed.weight, "layer0": m.core.layers[0].z.weight, "layer1": m.core.layers[1].ff[0].weight,
             "local": m.local.qkv.weight, "head": m.mems[0].val.weight, "aux_norm": m.aux_norms[0].weight, "aux_proj": m.aux_decoders[0].weight}
    from_last = torch.autograd.grad(ce(logits, y), list(first.values()), retain_graph=True, allow_unused=True)
    from_own = torch.autograd.grad(ce(aux[0], y), list(first.values()), retain_graph=True, allow_unused=True)
    for name, gl, go in zip(first, from_last, from_own):
        assert gl is None or not gl.any(), name                                                       # exactly zero from the last block's loss
        assert go is not None and float(go.abs().sum()) > 0, name                                      # and a real gradient from its own
    last = [m.core.layers[2].z.weight, m.core.layers[3].z.weight, m.core.ln.weight, m.decoder.weight, m.mem.val.weight]
    assert all(g is not None and float(g.abs().sum()) > 0 for g in torch.autograd.grad(ce(logits, y), last, retain_graph=True))
    assert all(g is None for g in torch.autograd.grad(ce(aux[0], y), last, retain_graph=True, allow_unused=True))   # the other way round: none


def _reach(m, losses):
    """For each loss: the modules (embed, core.layers.i, mems.k, local, ...) that receive a non-zero gradient from that loss alone."""
    def owner(n):
        parts = n.split(".")
        if parts[0] == "core":
            return ".".join(parts[:3]) if parts[1] == "layers" else "core.ln"
        return ".".join(parts[:2]) if parts[0] in ("mems", "aux_norms", "aux_decoders") else parts[0]
    names = [n for n, _ in m.named_parameters()]
    out = []
    for loss in losses:
        gs = torch.autograd.grad(loss, list(m.parameters()), retain_graph=True, allow_unused=True)
        out.append({owner(n) for n, g in zip(names, gs) if g is not None and g.any()})
    return out


def test_heads_and_the_local_attention_belong_to_the_block_of_their_position():
    m = tiny(n_layers=6, local_blocks=2)          # blocks: layers 0-1 | 2-3 | 4-5; heads after layers 1 and 3; local attention after layer 2; top head
    x, y = batch()
    logits, aux = m.forward_blocks(x)
    reach = _reach(m, [ce(a, y) for a in aux] + [ce(logits, y)])
    assert reach[0] == {"embed", "core.layers.0", "core.layers.1", "mems.0", "aux_norms.0", "aux_decoders.0"}
    assert reach[1] == {"core.layers.2", "core.layers.3", "local", "mems.1", "aux_norms.1", "aux_decoders.1"}
    assert reach[2] == {"core.layers.4", "core.layers.5", "core.ln", "mem", "decoder"}                  # the head after the stack: the last block


@pytest.mark.parametrize("k, ends", [(1, [0, 1, 2, 3]), (2, [1, 3]), (3, [2, 3]), (4, [3]), (9, [3])])
def test_blocks_are_k_layers_and_the_last_may_be_shorter(k, ends):
    m = tiny(local_blocks=k)
    assert m.block_ends == ends and m.n_blocks == len(ends) and len(m.aux_decoders) == len(ends) - 1
    x, _ = batch()
    logits, aux = m.forward_blocks(x)
    assert len(aux) == len(ends) - 1 and torch.equal(logits, tiny()(x)[0])


def test_the_training_loss_is_the_sum_of_the_block_losses():
    from scripts.run_sparsela import parse_args, train_loss
    x, y = batch()
    m = tiny(n_layers=6, local_blocks=2)
    loss, parts = train_loss(m, x, y, hybrid_args("--local-blocks", "2"), CPU)
    logits, aux = m.forward_blocks(x)
    ref = [ce(a, y) for a in aux] + [ce(logits, y)]
    assert len(parts) == m.n_blocks == 3
    for p, r in zip(parts, ref):
        assert p.item() == pytest.approx(r.item(), rel=1e-6)
    assert loss.item() == pytest.approx(sum(p.item() for p in parts), rel=1e-6)
    assert parts[-1].item() == pytest.approx(ce(m(x)[0], y).item(), rel=1e-6)                         # the last part is the model's own loss
    g_sum = torch.autograd.grad(loss, [m.embed.weight, m.decoder.weight], retain_graph=True)           # through the sum: each block trains on its own loss only
    assert torch.equal(g_sum[0], torch.autograd.grad(parts[0], m.embed.weight, retain_graph=True)[0])
    assert torch.equal(g_sum[1], torch.autograd.grad(parts[-1], m.decoder.weight, retain_graph=True)[0])
    m0 = tiny()                                                                                       # without the switch: the plain cross-entropy
    loss0, parts0 = train_loss(m0, x, y, parse_args([]), CPU)
    assert parts0 == [loss0] and loss0.item() == pytest.approx(ce(m0(x)[0], y).item(), rel=1e-6)


def test_evaluation_ignores_the_auxiliary_readouts():
    from scripts.run_sparsela import eval_fixed
    m = tiny(local_blocks=2)
    xf, yf = fixed_windows(torch.randint(0, 256, (4 * 32 + 1,), generator=torch.Generator().manual_seed(3)), 32, 4)
    ns = argparse.Namespace(batch=2, amp=False)
    x, _ = batch()
    base = eval_fixed(m, xf, yf, ns, CPU)
    want = m.eval()(x)[0]
    with torch.no_grad():
        for p in (*m.aux_norms.parameters(), *m.aux_decoders.parameters()):
            p.fill_(float("nan"))                                                                     # whatever a readout could hold
    assert eval_fixed(m, xf, yf, ns, CPU) == base and torch.isfinite(torch.tensor(list(base.values()))).all()
    assert torch.equal(m(x)[0], want) and torch.isfinite(want).all()


@pytest.mark.parametrize("kw, why", [(dict(backbone="transformer", where="none", rope=True, local_attn=0, mem_every=0), "layered recurrent"),
                                     (dict(backbone="gru", where="post", local_attn=0, mem_every=0), "layered recurrent"),
                                     (dict(where="pre", mem_every=0, local_attn=0), "'pre'"),
                                     (dict(local_blocks=-1), "local_blocks")])
def test_local_blocks_that_cannot_work_are_refused_with_a_message(kw, why):
    with pytest.raises(ValueError, match=re.escape(why)):
        tiny(**{"local_blocks": 2, **kw})


def test_the_non_hybrid_arms_refuse_the_switches_at_parse_time(capsys):
    from scripts.run_sparsela import parse_args
    for flag in (["--local-blocks", "2"], ["--train-only", "readout"], ["--freeze-layers", "0"]):
        for arms in ("sparse", "gru", "mingru_sparse_post,softmax"):
            with pytest.raises(SystemExit):
                parse_args(["--arms", arms, *flag])
            assert "GRUHybrid" in capsys.readouterr().err
        parse_args(["--arms", "mingru,mingru_sparse_post,transformer", *flag])                       # the hybrid arms are fine


def test_a_negative_local_blocks_is_refused_at_parse_time(capsys):
    with pytest.raises(SystemExit):
        hybrid_args("--local-blocks=-1")
    assert "--local-blocks must be >= 0" in capsys.readouterr().err


# ---- the switches together --------------------------------------------------------------------------------------------------------

def test_frozen_blocks_with_local_losses_is_a_legal_combination():
    from scripts.run_sparsela import apply_freezing
    m = tiny(local_blocks=2); init = snapshot(m)
    rep = apply_freezing(m, train_only="readout")
    readout = ids(m.param_groups()["readout"])
    assert ids(m.aux_norms.parameters()) | ids(m.aux_decoders.parameters()) <= readout               # the block readouts are readouts
    assert ids(p for p in m.parameters() if p.requires_grad) == readout and rep["trainable_scalars"] == 2 * READOUT
    args = hybrid_args("--wd", "0.1", "--lr", "0.01", "--train-only", "readout", "--local-blocks", "2")
    x, y = batch()
    _, _, parts = one_step(m, args, x, y)
    assert len(parts) == 2
    for k, v in m.state_dict().items():
        if k.startswith(("core.ln.", "decoder.", "aux_")):
            assert not torch.equal(v, init[k]), k                                                      # every readout learned
        else:
            assert torch.equal(v, init[k]), k                                                          # the frozen blocks did not
    # each readout is a probe of its block's frozen features: the last sees only the last block's loss, the first only its own
    logits, aux = m.forward_blocks(x)
    total = ce(aux[0], y) + ce(logits, y)
    for p, own in ((m.decoder.weight, logits), (m.aux_decoders[0].weight, aux[0])):
        assert torch.equal(torch.autograd.grad(total, p, retain_graph=True)[0], torch.autograd.grad(ce(own, y), p, retain_graph=True)[0])


def test_embed_and_heads_with_local_losses_train_each_block_on_its_own_loss_only():
    from scripts.run_sparsela import apply_freezing, train_loss
    m = tiny(n_layers=6, local_blocks=2)
    apply_freezing(m, train_only="readout,embed,heads")
    x, y = batch()
    loss, parts = train_loss(m, x, y, hybrid_args("--local-blocks", "2"), CPU)
    probe = [m.embed.weight, m.mems[0].val.weight, m.mems[1].val.weight, m.mem.val.weight, m.decoder.weight]
    embed_from_first = torch.autograd.grad(parts[0], m.embed.weight, retain_graph=True)[0]
    from_last = torch.autograd.grad(parts[-1], probe, retain_graph=True, allow_unused=True)
    assert [g is None for g in from_last] == [True, True, True, False, False]                          # the last loss reaches the top head and readout only
    loss.backward()
    assert torch.equal(m.embed.weight.grad, embed_from_first)                                         # the embedding hears block 0 and nothing else
    assert all(float(p.grad.abs().sum()) > 0 for p in (m.mems[0].val.weight, m.mems[1].val.weight, m.mem.val.weight, m.aux_decoders[1].weight))
    assert not any(p.requires_grad for l in m.core.layers for p in l.parameters())


def test_frozen_layers_and_local_blocks_compose():
    from scripts.run_sparsela import apply_freezing
    m = tiny(local_blocks=2)                      # blocks: layers 0-1 | 2-3
    apply_freezing(m, freeze_layers="0,2")
    assert all(p.requires_grad for p in (*m.aux_norms.parameters(), *m.aux_decoders.parameters()))   # the block readouts are in no layer
    x, y = batch()
    logits, aux = m.forward_blocks(x)
    (ce(aux[0], y) + ce(logits, y)).backward()
    assert m.core.layers[0].z.weight.grad is None and m.core.layers[2].z.weight.grad is None
    assert float(m.core.layers[1].z.weight.grad.abs().sum()) > 0 and float(m.core.layers[3].z.weight.grad.abs().sum()) > 0
    assert float(m.embed.weight.grad.abs().sum()) > 0                                                  # through frozen layer 0, from block 0's loss


# ---- results and checkpoints --------------------------------------------------------------------------------------------------------

CASES = {   # name -> (extra flags, the switches that end up recorded)
    "none": ([], set()),
    "train_only": (["--train-only", "readout"], {"train_only"}),
    "freeze": (["--freeze-layers", "1"], {"freeze_layers"}),
    "blocks": (["--local-blocks", "2"], {"local_blocks"}),
    "all": (["--train-only", "readout,embed,heads", "--freeze-layers", "1", "--local-blocks", "2"], {"train_only", "freeze_layers", "local_blocks"}),
}


@pytest.mark.parametrize("case", list(CASES))
def test_results_and_a_checkpoint_saved_with_each_switch_load_in_rebuild_and_score(monkeypatch, tmp_path, capsys, case):
    from scripts.cross_eval import evaluate_fixed, rebuild
    flags, switched = CASES[case]
    out, res = _run(monkeypatch, tmp_path, "r", *FIXED, *flags)
    arm, ck = out["arms"][0], _ckpt(res)
    assert {k for k in SWITCH_KEYS if out.get(k)} == switched == {k for k in SWITCH_KEYS if ck["args"].get(k)}   # recorded exactly when on, JSON and checkpoint
    model = rebuild(ck)                                                                               # saved with or without the switches: it loads
    live = numel(model.parameters())
    g = model.param_groups()
    frozen = set()                                                                                    # what the flags should have frozen, worked out again from the rebuilt model
    if "train_only" in switched:
        allowed = flags[flags.index("--train-only") + 1].split(",")
        frozen |= ids(model.parameters()) - ids(p for name in allowed for p in g[name])
    if "freeze_layers" in switched:
        frozen |= ids(p for mod in model.layer_units()[1] for p in mod.parameters())
    assert model.local_blocks == (2 if "local_blocks" in switched else 0)
    assert arm["trainable_scalars"] == live - numel(p for p in model.parameters() if id(p) in frozen)
    assert model.aux_scalars == (READOUT if "local_blocks" in switched else 0)
    if switched & {"train_only", "freeze_layers"}:
        in_groups = ids(p for ps in g.values() for p in ps)
        by_group = {name: numel(p for p in ps if id(p) not in frozen) for name, ps in g.items()}
        by_group["other"] = numel(p for p in model.parameters() if id(p) not in in_groups | frozen)
        assert arm["total_scalars"] == live and arm["trainable_by_group"] == by_group and arm["trainable_scalars"] < live
    if "train_only" in switched:
        assert arm["train_only"] == flags[flags.index("--train-only") + 1].split(",")
    if "freeze_layers" in switched:
        assert arm["frozen_layers"] == [1]
    if "local_blocks" in switched:
        assert (arm["local_blocks"], arm["n_blocks"], arm["added_scalars"]) == (2, 2, READOUT)
        assert live - arm["added_scalars"] == numel(tiny().parameters())                             # the model that is scored is the unswitched one
        assert all(len(r["block_bpb"]) == 2 and r["block_bpb"][-1] == r["train_bpb"] for r in arm["curve"])
    if not switched:
        assert not NEW_ARM_KEYS & set(arm) and not any("block_bpb" in r for r in arm["curve"])
    data = load_bytes(_corpus(tmp_path), max_bytes=4 * 32 + 1, offset=110000)
    row = evaluate_fixed(model, data, 32, 2, 4, CPU)                                                  # it scores, and scores what the run scored
    assert row["bpb"] == pytest.approx(arm["val_fixed"]["bpb"], abs=1e-4)
    assert row["bpb_warm"] == pytest.approx(arm["val_fixed"]["bpb_warm"], abs=1e-4)
    printed = capsys.readouterr().out
    assert (f"trainable {arm['trainable_scalars']:,} of {live:,}" in printed) == bool(switched & {"train_only", "freeze_layers"})
    assert ("local-blocks 2" in printed) == ("local_blocks" in switched)


def test_a_checkpoint_from_before_the_switches_loads_in_rebuild(tiny_checkpoint):
    from scripts.cross_eval import rebuild
    ck = torch.load(tiny_checkpoint("mingru_sparse_post"), map_location="cpu", weights_only=False)
    assert not set(SWITCH_KEYS) & set(ck["args"])
    m = rebuild(ck)
    assert m.local_blocks == 0 and m(batch()[0])[0].shape == (2, 32, 256)


# ---- the controller's GPU runs parse on top of the spine recipe -------------------------------------------------------------------

GPU_RUNS = {
    "readout only": (["--train-only", "readout"], dict(train_only="readout")),
    "readout, embed, heads": (["--train-only", "readout,embed,heads"], dict(train_only="readout,embed,heads")),
    "local blocks of 2": (["--local-blocks", "2"], dict(local_blocks=2)),
    "local blocks of 4": (["--local-blocks", "4"], dict(local_blocks=4)),
    "odd layers frozen": (["--freeze-layers", "1,3,5,7"], dict(freeze_layers="1,3,5,7")),
    "even layers frozen": (["--freeze-layers", "0,2,4,6"], dict(freeze_layers="0,2,4,6")),
}


@pytest.mark.parametrize("run", list(GPU_RUNS))
def test_the_gpu_run_flags_parse_on_top_of_the_spine_hybrid_recipe(tmp_path, run):
    from scripts.run_sparsela import parse_args
    corpora = tmp_path / "corpora"
    corpora.mkdir()
    (corpora / "linux_full.bytes").write_bytes(b"x")
    r = subprocess.run(["bash", str(ROOT / "scripts" / "run_spine.sh"), "cuda:0", "hybrid", "0"], capture_output=True, text=True,
                       env={**os.environ, "DRY_RUN": "1", "CORPORA": str(corpora)})              # DRY_RUN: prints the recipe, runs nothing
    assert r.returncode == 0, r.stderr
    flags = r.stdout.strip().split(" ", 1)[1]
    extra, want = GPU_RUNS[run]
    args = parse_args(shlex.split(flags) + extra)
    assert (args.n_layers, args.arms, args.mem_every, args.local_attn, args.amp) == (8, "mingru_sparse_post", 2, 512, True)
    for k, v in want.items():
        assert getattr(args, k) == v, k
