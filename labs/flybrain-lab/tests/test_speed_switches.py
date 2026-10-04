"""The speed round's switches (notes/SPEED.md row 1) and the step profiler. CPU only, tiny models.

With no switch the model must be the model it was: logits and every gradient bit-identical to the code before the switches existed (the
`_old_*` functions below are that code, copied unchanged). Each switch is then checked against the default path it replaces."""

from __future__ import annotations

import json
import math
import sys

import pytest
import torch
from torch.nn import functional as F

from flybrain.model import sparsela
from flybrain.model.hybrid import GRUHybrid, MinGRULayer, configure_speed
from flybrain.model.sparsela import Block, KeyCode, sparse_memory_read
from flybrain.model.state import HeadState

# ---------------------------------------------------------------------------------------------------------------------
# the code before the switches, verbatim (flybrain/model/hybrid.py MinGRULayer.scan, flybrain/model/sparsela.py at 3cee1fe)
# ---------------------------------------------------------------------------------------------------------------------


def _old_scan(self, x, h0=None):
    u = self.ln1(x)
    z = torch.sigmoid(self.z(u)); cand = self.h(u)
    a = (1 - z).clamp_min(1e-4); b = z * cand
    B, T, d = u.shape; L = self.L
    pad = (-T) % L
    if pad:
        a = F.pad(a, (0, 0, 0, pad), value=1.0); b = F.pad(b, (0, 0, 0, pad))
    nC = (T + pad) // L
    a = a.reshape(B, nC, L, d).double(); b = b.reshape(B, nC, L, d).double()
    A = torch.exp(torch.cumsum(torch.log(a), dim=2))
    local = A * torch.cumsum(b / A, dim=2)
    A_end, local_end = A[:, :, -1], local[:, :, -1]
    carries = [torch.zeros(B, d, dtype=torch.float64, device=u.device) if h0 is None else h0.to(torch.float64)]
    for i in range(nC - 1):
        carries.append(A_end[:, i] * carries[-1] + local_end[:, i])
    h0s = torch.stack(carries, dim=1)
    h = (A * h0s[:, :, None, :] + local).reshape(B, nC * L, d)[:, :T]
    x = x + h.to(x.dtype)
    return x + self.ff(self.ln2(x)), (h[:, -1].clone() if T else h0s[:, 0])


def _old_sparse_code(self, x):
    if self.kind not in ("sparse", "gausstopk") or self.signed:
        raise ValueError("index form needs nonnegative sparse keys")
    kk = min(self.top_k, self.n_key)
    with torch.no_grad():
        parts = []
        for c0 in range(0, x.size(1), self.code_slice):
            z = x[:, c0 : c0 + self.code_slice].matmul(self.proj.t())
            parts.append(F.relu(z - self.theta).topk(kk, dim=-1).indices)
        idx = torch.cat(parts, dim=1)
    rows = self.proj[idx]
    z_sel = (x.unsqueeze(2) * rows).sum(-1)
    val = F.relu(z_sel - self.theta[idx])
    return idx, val


def _old_one_head(self, code, xk, v, state=None):
    with torch.autocast(device_type=xk.device.type, enabled=False):
        return self._one_head_fp32(code, xk.float(), v.float(), state)


def _old_one_head_chunked(self, code, xk, v, state=None):
    return _old_sparse_memory_read(code, xk, v, self.chunk, self.decay_value(), self.normalize, self.gate_tau if self.gate else None, self.state_bits, state)


def _old_sparse_memory_read(code, xk, v, chunk, decay, normalize, gate_tau, state_bits=0, state=None):
    fake_quant = sparsela.fake_quant
    B, T, _ = xk.shape; L = chunk; d = v.size(-1); n_key = code.n_key
    if state is not None and T == 0:
        cp = lambda t: None if t is None else t.clone()
        return v.new_zeros(B, 0, d), HeadState(S=cp(state.S), Ksum=cp(state.Ksum), last_idx=cp(state.last_idx), last_val=cp(state.last_val), conv_tail=cp(state.conv_tail))
    lam = decay if torch.is_tensor(decay) else torch.tensor(float(decay), device=xk.device, dtype=xk.dtype)
    idx, val = code.sparse_code(xk)
    val = val / (val.norm(dim=-1, keepdim=True) + 1e-6)
    k = idx.size(-1)
    if state is not None and state.last_idx is not None:
        idx_s = torch.cat([state.last_idx[:, None], idx[:, :-1]], dim=1)
        val_s = torch.cat([state.last_val[:, None].to(val.dtype), val[:, :-1]], dim=1)
    else:
        idx_s = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        val_s = torch.cat([torch.zeros_like(val[:, :1]), val[:, :-1]], dim=1)
    if state is None:
        S = xk.new_zeros(B * n_key, d); Ksum = xk.new_zeros(B * n_key)
    else:
        S = state.S.to(xk.dtype).reshape(B * n_key, d); Ksum = state.Ksum.to(xk.dtype).reshape(B * n_key)
    boff = (torch.arange(B, device=xk.device) * n_key)[:, None, None]
    outs = []
    for c in range(0, T, L):
        iq, vq = idx[:, c : c + L], val[:, c : c + L]
        ik, vk, vc = idx_s[:, c : c + L], val_s[:, c : c + L], v[:, c : c + L]
        Lc = iq.size(1)
        eq = (iq[:, :, None, :, None] == ik[:, None, :, None, :])
        sc = (eq * (vq[:, :, None, :, None] * vk[:, None, :, None, :])).sum((-1, -2))
        ar = torch.arange(Lc, device=xk.device)
        lag = (ar[:, None] - 1 - ar[None, :]).clamp(min=0).to(xk.dtype)
        sc = sc * torch.ones(Lc, Lc, device=xk.device, dtype=torch.bool).tril(-1) * lam ** lag
        age_in = (lam ** ar.to(xk.dtype))[None, :, None]
        age_out = (lam ** (Lc - 1 - ar).to(xk.dtype))[None, :, None]
        flat = (iq + boff).reshape(B, Lc * k)
        Srows = S.index_select(0, flat.reshape(-1)).view(B, Lc, k, d)
        Krows = Ksum.index_select(0, flat.reshape(-1)).view(B, Lc, k)
        num = sc.matmul(vc) + age_in * (Srows * vq[..., None]).sum(2)
        mass = sc.sum(-1, keepdim=True) + age_in * (Krows * vq).sum(-1, keepdim=True)
        out = num
        if normalize:
            out = out / (mass + 1e-3)
        if gate_tau is not None:
            out = out * (mass / (mass + F.softplus(gate_tau)))
        outs.append(out)
        wflat = (ik + boff).reshape(-1)
        vk_aged = vk * age_out
        S = S * lam ** Lc + torch.zeros_like(S).index_add(0, wflat, (vk_aged[..., None] * vc[:, :, None, :]).reshape(-1, d))
        Ksum = Ksum * lam ** Lc + torch.zeros_like(Ksum).index_add(0, wflat, vk_aged.reshape(-1))
        if state_bits:
            S = fake_quant(S, state_bits); Ksum = fake_quant(Ksum[None], state_bits)[0]
    out = torch.cat(outs, dim=1)
    if state is None:
        return out
    return out, HeadState(S=S.reshape(B, n_key, d), Ksum=Ksum.reshape(B, n_key), last_idx=idx[:, -1].clone(), last_val=val[:, -1].clone(), conv_tail=state.conv_tail)


# ---------------------------------------------------------------------------------------------------------------------
# the three arms, tiny
# ---------------------------------------------------------------------------------------------------------------------

KW = dict(d_model=16, n_layers=4, d_ff=32, key="sparse")
ARMS = {"hybrid": dict(KW, where="post", backbone="mingru", n_key=64, top_k=4, key_conv=2, normalize=True, gate=True, chunk=8, mem_every=2,
                       local_attn=8, key_convs=(2, 4)),
        "zeroheads": dict(KW, where="none", backbone="mingru", local_attn=8),
        "tf": dict(KW, where="none", backbone="transformer", rope=True)}


def _model(arm, **extra):
    torch.manual_seed(0)
    return GRUHybrid(**{**ARMS[arm], **extra})


def _batch(T=40, B=2):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 256, (B, T), generator=g), torch.randint(0, 256, (B, T), generator=g)


def _step(m, amp=False, T=40):
    """logits, loss and every parameter's gradient of one forward/backward on the fixed batch."""
    x, y = _batch(T)
    m.zero_grad()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        logits, _ = m(x)
        loss = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))
    loss.backward()
    return logits.detach().clone(), loss.detach().clone(), {n: (None if p.grad is None else p.grad.clone()) for n, p in m.named_parameters()}


def _same(a, b):
    """Two _step results equal bit for bit."""
    return (torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2].keys() == b[2].keys()
            and all((a[2][n] is None) == (b[2][n] is None) and (a[2][n] is None or torch.equal(a[2][n], b[2][n])) for n in a[2]))


def _maxdiff(a, b):
    return float((a - b).abs().max())


# ---------------------------------------------------------------------------------------------------------------------
# no switch: the model it was
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("arm", ["hybrid", "zeroheads", "tf"])
def test_with_no_switch_logits_and_gradients_are_bit_identical_to_the_code_before(arm, amp, monkeypatch):
    now = _step(_model(arm), amp)
    with monkeypatch.context() as mp:
        mp.setattr(MinGRULayer, "scan", _old_scan)
        mp.setattr(KeyCode, "sparse_code", _old_sparse_code)
        mp.setattr(Block, "_one_head", _old_one_head)
        mp.setattr(Block, "_one_head_chunked", _old_one_head_chunked)
        mp.setattr(sparsela, "sparse_memory_read", _old_sparse_memory_read)
        before = _step(_model(arm), amp)
    assert _same(now, before)
    if arm == "hybrid":
        assert not _same(now, _step(_model("zeroheads"), amp)) and len(now[2]) > 70   # the comparison covers the heads' parameters too


def test_every_switch_is_off_by_default_and_the_constructor_arguments_keep_that():
    m = _model("hybrid")
    assert all(not b.head_amp and not b.detach_state for b in m.all_heads())
    assert configure_speed(m) == {"head_amp": 0, "detach_state": 0, "scan": 0}            # None leaves a switch alone, and touches nothing
    assert all(not b.head_amp and not b.detach_state for b in m.all_heads())
    assert all(l.scan_mode == "closed64" for l in m.core.layers)
    on = _model("hybrid", head_amp=True, detach_state=True)
    assert all(b.head_amp and b.detach_state for b in on.all_heads())
    assert configure_speed(_model("zeroheads"), head_amp=True, detach_state=True) == {"head_amp": 0, "detach_state": 0, "scan": 0}   # an arm without heads: nothing to reach
    assert configure_speed(m, head_amp=True) == {"head_amp": 2, "detach_state": 0, "scan": 0} and all(b.head_amp for b in m.all_heads())


# ---------------------------------------------------------------------------------------------------------------------
# --detach-state
# ---------------------------------------------------------------------------------------------------------------------

def _read(detach, T=24, chunk=8, state=None):
    torch.manual_seed(0)
    code = KeyCode("sparse", 16, 64, 4, 4, 0.5, 0)
    xk = torch.randn(2, T, 16, requires_grad=True); v = torch.randn(2, T, 16, requires_grad=True)
    tau = torch.zeros((), requires_grad=True)
    res = sparse_memory_read(code, xk, v, chunk, 1.0, True, tau, 0, state, detach)
    return xk, v, res


def test_detach_state_leaves_the_forward_unchanged_bit_for_bit():
    out_a = _read(False)[2]; out_b = _read(True)[2]
    assert torch.equal(out_a, out_b) and out_a.abs().sum() > 0


def test_detach_state_zeroes_the_gradient_that_crosses_a_chunk_boundary_through_the_tables_and_nothing_inside_a_chunk():
    """The loss reads one chunk's outputs. Without the switch the values written in the chunks before it get gradient through the carried
    tables; with it they get none, while the chunk's own values (intra-chunk scores) and every query-side path are untouched."""
    for c0 in (8, 16):
        grads = {}
        for detach in (False, True):
            xk, v, out = _read(detach)
            out[:, c0 : c0 + 8].pow(2).sum().backward()
            grads[detach] = (xk.grad.clone(), v.grad.clone())
        gv_plain, gv_cut = grads[False][1], grads[True][1]
        assert gv_plain[:, :c0].abs().max() > 0                                  # the gradient does cross, today
        assert gv_cut[:, :c0].abs().max() == 0                                   # and with the switch it is exactly zero
        assert gv_cut[:, c0 : c0 + 8].abs().max() > 0                            # inside the chunk it is alive ...
        assert torch.equal(gv_cut[:, c0:], gv_plain[:, c0:])                     # ... and what it was
        gx_plain, gx_cut = grads[False][0], grads[True][0]
        assert gx_plain[:, : c0 - 1].abs().max() > 0 and gx_cut[:, : c0 - 1].abs().max() == 0   # keys: the same cut (c0 - 1 feeds slot c0 as the previous key)
        assert gx_cut[:, c0 - 1].abs().max() > 0                                 # the previous-key path does not go through the tables
        assert torch.equal(gx_cut[:, c0:], gx_plain[:, c0:])


def test_detach_state_cuts_the_incoming_state_of_a_streamed_head_and_keeps_the_forward():
    torch.manual_seed(0)
    n_key, d = 64, 16
    def fresh():
        S = torch.randn(2, n_key, d, requires_grad=True); K = torch.rand(2, n_key, requires_grad=True)
        return S, K, HeadState(S=S, Ksum=K)
    S0, K0, st0 = fresh(); S1, K1, st1 = fresh()
    S1.data.copy_(S0.data); K1.data.copy_(K0.data)
    xk0, v0, (o0, new0) = _read(False, state=st0)
    xk1, v1, (o1, new1) = _read(True, state=st1)
    assert torch.equal(o0, o1)
    o0.sum().backward(); o1.sum().backward()
    assert S0.grad is not None and S0.grad.abs().max() > 0 and K0.grad.abs().max() > 0     # plain: the table passed in gets gradient
    assert S1.grad is None and K1.grad is None                                            # detached: it is a constant
    assert not new1.S.requires_grad and not new1.Ksum.requires_grad and new0.S.requires_grad


def test_detach_state_in_a_model_changes_only_the_gradients_and_only_when_there_is_more_than_one_chunk():
    plain, cut = _model("hybrid"), _model("hybrid", detach_state=True)
    a, b = _step(plain, T=40), _step(cut, T=40)                                           # 5 chunks of 8
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])                            # forward and loss: identical
    differs = [n for n in a[2] if a[2][n] is not None and not torch.equal(a[2][n], b[2][n])]
    assert any(n.startswith(("mem.", "mems.")) for n in differs)                          # the heads' parameters see another gradient
    assert all(torch.isfinite(g).all() for g in b[2].values() if g is not None)
    one_a, one_b = _step(_model("hybrid"), T=8), _step(_model("hybrid", detach_state=True), T=8)   # one chunk: nothing to cut
    assert _same(one_a, one_b)


# ---------------------------------------------------------------------------------------------------------------------
# --head-amp
# ---------------------------------------------------------------------------------------------------------------------

def test_head_amp_without_autocast_is_the_pinned_head_bit_for_bit():
    assert _same(_step(_model("hybrid"), amp=False), _step(_model("hybrid", head_amp=True), amp=False))


class _Dtypes(torch.utils._python_dispatch.TorchDispatchMode):
    """Records the operand dtypes of every aten matmul whose output has n_key columns (the top-k selection matmul)."""
    def __init__(self, n_key): super().__init__(); self.n_key, self.seen = n_key, []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func.overloadpacket in (torch.ops.aten.mm, torch.ops.aten.bmm) and isinstance(out, torch.Tensor) and out.shape[-1] == self.n_key:
            self.seen.append((out.dtype, tuple(a.dtype for a in args if isinstance(a, torch.Tensor))))
        return out


def test_head_amp_runs_the_selection_matmul_in_bf16_and_the_pinned_head_in_fp32():
    x, _ = _batch()
    for head_amp, want in ((False, torch.float32), (True, torch.bfloat16)):
        m = _model("hybrid", head_amp=head_amp)
        spy = _Dtypes(n_key=64)
        with spy, torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            m(x)
        sel = [s for s in spy.seen if s[0] in (torch.float32, torch.bfloat16)]
        assert sel and all(s[0] == want and set(s[1]) == {want} for s in sel), (head_amp, sel[:2])


def test_head_amp_under_bf16_autocast_is_finite_trains_and_stays_near_the_fp32_head(capsys):
    pinned, free = _model("hybrid"), _model("hybrid", head_amp=True)
    a, b = _step(pinned, amp=True), _step(free, amp=True)
    assert torch.isfinite(b[0]).all() and torch.isfinite(b[1]) and all(torch.isfinite(g).all() for g in b[2].values() if g is not None)
    assert not torch.equal(a[0], b[0])                                                    # the heads did run differently
    dl = _maxdiff(a[0], b[0]); dg = max(_maxdiff(a[2][n], b[2][n]) for n in a[2] if a[2][n] is not None)
    with capsys.disabled():
        print(f"\n[head-amp, CPU bf16 autocast, tiny hybrid] max |logits - fp32-head logits| = {dl:.4g}, loss {float(b[1]):.5f} vs {float(a[1]):.5f}, max |grad diff| = {dg:.4g}")
    assert dl < 0.5                                                                       # the budget the repo's bf16 forward tests use
    x, y = _batch()
    opt = torch.optim.Adam(free.parameters(), lr=0.01); first = None
    for _ in range(8):                                                                    # and it still learns
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = F.cross_entropy(free(x)[0].float().reshape(-1, 256), y.reshape(-1))
        first = first or float(loss.detach())
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss.detach()) < first


def test_head_amp_keeps_the_tables_and_the_decay_in_fp32():
    code = KeyCode("sparse", 16, 64, 4, 4, 0.5, 0)
    xk = torch.randn(2, 16, 16).bfloat16(); v = torch.randn(2, 16, 16).bfloat16()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out, st = sparse_memory_read(code, xk, v, 8, 0.999, True, None, 0, HeadState(S=torch.zeros(2, 64, 16), Ksum=torch.zeros(2, 64)))
    assert st.S.dtype == torch.float32 and st.Ksum.dtype == torch.float32 and torch.isfinite(out).all() and st.S.abs().max() > 0


# ---------------------------------------------------------------------------------------------------------------------
# the runner: --tf32 --fused --compile --head-amp --detach-state
# ---------------------------------------------------------------------------------------------------------------------

SMALL = ["--task", "code", "--train-bytes", "3000", "--val-bytes", "4000", "--d-model", "16", "--n-layers", "4", "--d-ff", "32",
         "--n-key", "64", "--top-k", "4", "--key-conv", "2", "--normalize", "--chunk", "8", "--batch", "2", "--seq-len", "32",
         "--steps", "4", "--lr", "0.003", "--arms", "mingru_sparse_post", "--gate", "--mem-every", "2", "--save", "--seed", "0",
         "--device", "cpu", "--run-name", "tiny", "--wd", "0.1", "--amp"]


def _corpus(tmp_path):
    path = tmp_path / "tiny.bytes"
    path.write_bytes((b"int main(void) { return 0; }\n" * 5000)[:130_000])
    return path


def _run(monkeypatch, tmp_path, tag, *flags):
    """scripts.run_sparsela.main() in-process with results in tmp_path/tag; (the run's JSON, the checkpoint, the results dir)."""
    import scripts.run_sparsela as sra
    res = tmp_path / tag
    res.mkdir()
    monkeypatch.setattr(sra, "RESULTS", res)
    monkeypatch.setattr(sra, "_CMASK", {})
    monkeypatch.setattr(sys, "argv", ["run_sparsela.py", "--corpus-file", str(_corpus(tmp_path)), *SMALL, *flags])
    sra.main()
    return (json.loads((res / "tiny.json").read_text()),
            torch.load(res / "ckpt" / "tiny_mingru_sparse_post.pt", map_location="cpu", weights_only=False), res)


@pytest.fixture
def restore_tf32():
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def test_the_parser_has_every_switch_and_all_of_them_are_off():
    from scripts.run_sparsela import SPEED_DEFAULTS, parse_args, speed_on
    args = parse_args([])
    assert all(getattr(args, k) == v for k, v in SPEED_DEFAULTS.items()) and not speed_on(args)
    assert set(SPEED_DEFAULTS) == {"tf32", "fused", "compile", "head_amp", "detach_state", "scan"} and args.scan == "closed64"
    on = parse_args(["--tf32", "--fused", "--compile", "--head-amp", "--detach-state", "--scan", "doubling32"])
    assert all(getattr(on, k) is True for k in SPEED_DEFAULTS if k != "scan") and on.scan == "doubling32" and speed_on(on)
    assert speed_on(parse_args(["--scan", "doubling"])) and not speed_on(parse_args(["--scan", "closed64"]))
    with pytest.raises(SystemExit):
        parse_args(["--scan", "associative"])


def test_a_command_line_without_switches_records_the_args_it_always_recorded(monkeypatch, tmp_path):
    from scripts.run_sparsela import SPEED_FLAGS
    legacy, ck, _ = _run(monkeypatch, tmp_path, "a")
    assert not set(SPEED_FLAGS) & set(legacy) and not set(SPEED_FLAGS) & set(ck["args"])
    flagged, ck2, _ = _run(monkeypatch, tmp_path, "b", "--head-amp", "--detach-state", "--fused", "--tf32")
    assert all(flagged[k] is True for k in ("head_amp", "detach_state", "fused", "tf32")) and flagged["compile"] is False
    assert ck2["args"]["head_amp"] is True and set(SPEED_FLAGS) <= set(ck2["args"])


def test_switches_run_end_to_end_and_the_default_is_unchanged_by_their_existence(monkeypatch, tmp_path, restore_tf32):
    plain, ck_plain, _ = _run(monkeypatch, tmp_path, "a")
    again, ck_again, _ = _run(monkeypatch, tmp_path, "b")
    assert all(torch.equal(ck_plain["state"][k], ck_again["state"][k]) for k in ck_plain["state"])            # the run is deterministic: the baseline for the next line
    sw, ck_sw, _ = _run(monkeypatch, tmp_path, "c", "--head-amp", "--detach-state", "--fused", "--tf32")
    assert ck_sw["state"].keys() == ck_plain["state"].keys()                                                  # the same checkpoint layout
    assert torch.backends.cuda.matmul.allow_tf32 is True and torch.backends.cudnn.allow_tf32 is True
    assert math.isfinite(sw["arms"][0]["val_bpb"]) and sw["arms"][0]["val_bpb"] != plain["arms"][0]["val_bpb"]  # a different, finite run


def test_fused_is_adamw_fused_on_cuda_and_the_ordinary_optimizer_elsewhere():
    from scripts.run_sparsela import make_optimizer, parse_args
    m = _model("zeroheads")
    for flags in ([], ["--wd", "0.1"], ["--fused"], ["--fused", "--wd", "0.1"]):
        opt = make_optimizer(m, parse_args(flags))
        assert type(opt) is (torch.optim.AdamW if "--wd" in flags else torch.optim.Adam)
        assert not opt.defaults.get("fused")                                                                  # CPU parameters: never fused


def test_compile_wraps_the_training_step_only_and_the_checkpoint_is_the_plain_module(monkeypatch, tmp_path):
    seen = []
    class Wrapped(torch.nn.Module):                                                                           # what torch.compile returns: a wrapper that owns the module
        def __init__(self, m): super().__init__(); self._orig_mod = m
        def forward(self, *a, **k): seen.append("train"); return self._orig_mod(*a, **k)
    monkeypatch.setattr(torch, "compile", lambda m, *a, **k: Wrapped(m))
    out, ck, _ = _run(monkeypatch, tmp_path, "a", "--compile")
    assert out["compile"] is True and seen == ["train"] * 4                                                   # the 4 training steps, and no eval went through it
    assert not any(k.startswith("_orig_mod") for k in ck["state"]) and "mem.val.weight" in ck["state"]


def test_a_switch_that_finds_no_head_says_so(capsys):
    from scripts.run_sparsela import apply_speed, parse_args
    apply_speed(_model("zeroheads"), parse_args(["--head-amp", "--detach-state"]), "mingru")
    msg = capsys.readouterr().out
    assert "--head-amp has no effect on arm mingru" in msg and "--detach-state has no effect on arm mingru" in msg
    apply_speed(_model("hybrid"), parse_args([]), "x")
    assert capsys.readouterr().out == ""                                                                      # no switch, no note
    apply_speed(_model("hybrid", chunk=0), parse_args(["--detach-state"]), "hybrid")                          # the T x T form carries no tables
    assert "--detach-state needs --chunk > 0" in capsys.readouterr().out


def test_checkpoints_with_and_without_the_switch_args_load_through_cross_eval_rebuild(tmp_path, tiny_checkpoint):
    from scripts.cross_eval import rebuild
    old = torch.load(tiny_checkpoint("mingru_sparse_post"), map_location="cpu", weights_only=False)           # saved before the switches: no such keys
    assert not {"head_amp", "detach_state", "tf32", "fused", "compile"} & set(old["args"])
    m_old = rebuild(old)
    new = dict(old, args=dict(old["args"], head_amp=True, detach_state=True, tf32=True, fused=True, compile=True))   # saved by a run with them
    m_new = rebuild(new)
    x = torch.randint(0, 256, (2, 24))
    with torch.no_grad():
        assert torch.equal(m_old.eval()(x)[0], m_new.eval()(x)[0])                                            # scored the same: rebuild is the reference path
    assert not any(b.head_amp or b.detach_state for b in m_new.all_heads())


# ---------------------------------------------------------------------------------------------------------------------
# scripts/profile_step.py
# ---------------------------------------------------------------------------------------------------------------------

PROFILE = ["--task", "code", "--train-bytes", "20000", "--d-model", "32", "--n-layers", "4", "--d-ff", "64", "--n-key", "128", "--top-k", "4",
           "--key-conv", "2", "--normalize", "--gate", "--chunk", "16", "--mem-every", "2", "--local-attn", "16", "--key-convs", "2,4",
           "--batch", "2", "--seq-len", "64", "--device", "cpu", "--seed", "0", "--warmup", "10", "--cosine", "--steps", "100",
           "--warmup-steps", "1", "--measure-steps", "2", "--profile-steps", "1"]


def _profile(tmp_path, arms, *flags):
    from scripts import profile_step
    out = tmp_path / "p.json"
    profile_step.main(["--corpus-file", str(_corpus(tmp_path)), *PROFILE, "--arms", arms, "--json-out", str(out), *flags])
    return json.loads(out.read_text())["arms"]


def test_profile_step_runs_end_to_end_on_cpu_and_its_parts_sum_to_the_forward(tmp_path, capsys):
    (r,) = _profile(tmp_path, "mingru_sparse_post", "--amp")
    t = r["timing"]; fb = r["forward_breakdown"]
    assert t["plain"]["seconds_per_step"] > 0 and math.isfinite(t["plain"]["final_loss"])
    assert all(t["split"][k]["mean"] > 0 for k in ("forward", "backward", "optimizer")) and t["split"]["total_mean"] > 0
    assert len(t["warmup_seconds"]) == 1
    assert abs(fb["parts_sum_ms"] - fb["total_ms"]) <= 0.10 * fb["total_ms"], (fb["parts_sum_ms"], fb["total_ms"])     # the spec: within 10 %
    assert abs(fb["unattributed_ms"]) <= 0.10 * fb["total_ms"]
    assert fb["plain_forward_ms"] > 0                                                                              # the paired uninstrumented forward
    with capsys.disabled():                                                                                        # printed, not asserted: CPU timings are noise
        print(f"\n[profile_step, CPU, tiny hybrid] parts {fb['parts_sum_ms']:.1f} ms of {fb['total_ms']:.1f} ms instrumented; uninstrumented {fb['plain_forward_ms']:.1f} ms")
    parts = fb["parts"]
    assert {"model", "embed", "layer1", "layer1/scan", "layer4/scan", "local1", "final_ln", "decoder", "loss"} <= set(parts)
    for h in ("head_top", "head_L2"):                                                                              # the 4-layer tiny model: the top head and one inner head
        assert {f"{h}/ln", f"{h}/conv", f"{h}/val", f"{h}/select", f"{h}/recompute", f"{h}/read", f"{h}/norm_gate", f"{h}/out", h} <= set(parts), h
    assert parts["head_top/norm_gate"]["calls"] == 64 // 16 and parts["head_top/select"]["calls"] == 1             # one call per chunk; one selection per head
    assert fb["heads"] == ["head_top", "head_L2"]
    ops = r["operators"]
    assert ops["ops"] and "error" not in ops and ops["by"] == "self CPU time" and sum(o["pct"] for o in ops["ops"]) <= 100.0 + 1e-6
    assert r["memory"]["kind"].startswith("cpu") and r["memory"]["peak_rss_bytes"] > 0
    md = capsys.readouterr().out
    assert "### Seconds per step" in md and "### Forward breakdown" in md and "### Top operators" in md and "address: no-grad matmul + top-k" in md
    assert r["speed_switches"] == []


def test_profile_step_leaves_the_model_as_it_found_it_and_covers_the_other_arms(tmp_path):
    from scripts import profile_step
    rigs = _profile(tmp_path, "mingru,transformer", "--rope")
    assert [r["arm"]["arm"] for r in rigs] == ["mingru", "transformer"]
    zero, tf = rigs
    assert zero["forward_breakdown"]["heads"] == [] and "layer1/scan" in zero["forward_breakdown"]["parts"]
    assert {"layer1/attn", "layer1/ffn", "layer1/ln"} <= set(tf["forward_breakdown"]["parts"])
    for r in rigs:
        fb = r["forward_breakdown"]
        assert abs(fb["unattributed_ms"]) <= 0.10 * fb["total_ms"], r["arm"]["arm"]
    # the wrappers are gone: the instrumented model is a plain model again
    m = _model("hybrid"); timer = profile_step.SectionTimer()
    with profile_step.Instrument(m, timer):
        m(_batch()[0])
    assert timer.acc and "memory" not in m.mem.__dict__ and "recurrence" not in m.core.layers[0].__dict__ and "select" not in m.mem.code.__dict__
    assert not m.core.layers[0]._forward_hooks and sparsela.sparse_memory_read.__module__ == "flybrain.model.sparsela"
    assert _same(_step(m), _step(_model("hybrid")))


def test_profile_step_reports_the_switches_it_was_given_and_takes_the_runners_flags(tmp_path):
    (r,) = _profile(tmp_path, "mingru_sparse_post", "--amp", "--head-amp", "--detach-state", "--save", "--val-fixed", "2")
    assert r["speed_switches"] == ["--head-amp", "--detach-state"] and r["flags"]["amp"] is True
    assert "head_top/select" in r["forward_breakdown"]["parts"]


# ---------------------------------------------------------------------------------------------------------------------
# --scan {closed64, doubling32, doubling}
# ---------------------------------------------------------------------------------------------------------------------

def _ab(B=3, T=100, d=8, seed=0, dtype=torch.float32, trained=False):
    """Random (a, b, h0) for the recurrence: a uniform in [1e-4, 1], or drawn like a trained gate (a = 1 - sigmoid(N(-1, 2)): most gates near
    1, some near the clamp), b normal."""
    g = torch.Generator().manual_seed(seed)
    a = (1 - torch.sigmoid(torch.randn(B, T, d, generator=g) * 2 - 1)).clamp_min(1e-4) if trained else torch.rand(B, T, d, generator=g) * (1 - 1e-4) + 1e-4
    return a.to(dtype), torch.randn(B, T, d, generator=g).to(dtype), torch.randn(B, d, generator=g).to(dtype)


def _near(x, y, tol):
    """Normwise: max |x - y| <= tol * max(1, max |y|), compared in float64."""
    x, y = x.detach().double(), y.detach().double()
    return float((x - y).abs().max()) <= tol * max(1.0, float(y.abs().max()))


def _recur(mode, a, b, h0, w, dtype=None):
    """h, carry and the gradients of sum(h * w) + sum(carry) wrt a, b, h0 for one mode of the recurrence."""
    layer = MinGRULayer(a.size(-1), 8)
    layer.scan_mode = mode
    ls = [t.detach().clone().to(dtype or t.dtype).requires_grad_() for t in (a, b, h0)]
    h, c = layer.recurrence(ls[0], ls[1], ls[2])
    (h.double() * w).sum().add(c.sum()).backward()
    return h.detach(), c.detach(), [t.grad for t in ls]


def test_scan_mode_defaults_and_validation():
    assert MinGRULayer.scan_mode == "closed64" and MinGRULayer(8, 8).scan_mode == "closed64"
    with pytest.raises(ValueError):
        configure_speed(_model("zeroheads"), scan="associative")
    with pytest.raises(ValueError):
        _model("zeroheads", scan="log")
    m = _model("hybrid", scan="doubling32")
    assert all(l.scan_mode == "doubling32" for l in m.core.layers) and configure_speed(m, scan="doubling") == {"head_amp": 0, "detach_state": 0, "scan": 4}
    assert all(l.scan_mode == "doubling" for l in m.core.layers)
    assert configure_speed(_model("tf"), scan="doubling32")["scan"] == 0                  # no minGRU layer on the transformer: nothing to reach


@pytest.mark.parametrize("T", [1, 2, 3, 31, 32, 33, 100, 257])
def test_doubling_is_the_same_recurrence_exactly_in_float64(T):
    """The algebra, with no float32 rounding to hide behind: prefix doubling equals the closed form to 1e-12, values, carry and gradients."""
    a, b, h0 = _ab(B=2, T=T, d=5, dtype=torch.float64, trained=(T % 2 == 0))
    w = torch.randn(2, T, 5, dtype=torch.float64)
    ref = _recur("closed64", a, b, h0, w)
    for mode in ("doubling", "doubling32"):
        got = _recur(mode, a, b, h0, w, dtype=torch.float64)
        tol = 1e-12 if mode == "doubling" else 3e-6                                           # doubling32 casts to float32 first
        assert _near(got[0], ref[0], tol) and _near(got[1], ref[1], tol), (mode, T)
        assert got[1].dtype == torch.float64
        assert all(_near(g, r, tol * 10) for g, r in zip(got[2], ref[2])), (mode, T)


@pytest.mark.parametrize("trained", [False, True])
def test_doubling32_matches_closed64_values_and_gradients_within_1e4_on_random_inputs(trained):
    for seed, T, with_h0 in ((0, 100, True), (1, 64, False), (2, 257, True)):
        a, b, h0 = _ab(T=T, seed=seed, trained=trained)
        if not with_h0:
            h0 = torch.zeros_like(h0)
        w = torch.randn(3, T, 8, dtype=torch.float64)
        ref = _recur("closed64", a, b, h0, w)
        got = _recur("doubling32", a, b, h0, w)
        assert _near(got[0], ref[0], 1e-4) and _near(got[1], ref[1], 1e-4), (seed, T)
        assert all(_near(g, r, 1e-4) for g, r in zip(got[2], ref[2])), (seed, T)
        assert got[0].dtype == torch.float32 and got[1].dtype == torch.float64            # float32 scan, float64 carry like the closed form's


def test_doubling32_on_a_layer_and_on_tiny_models_matches_closed64():
    torch.manual_seed(0)
    layer = MinGRULayer(12, 24); x = torch.randn(2, 70, 12); h0 = torch.randn(2, 12, dtype=torch.float64)
    ref_y, ref_h = layer.scan(x, h0)
    layer.scan_mode = "doubling32"
    y, h = layer.scan(x, h0)
    assert _near(y, ref_y, 1e-4) and _near(h, ref_h, 1e-4) and h.dtype == torch.float64
    for arm in ("hybrid", "zeroheads"):
        ref, new = _step(_model(arm), T=70), _step(_model(arm, scan="doubling32"), T=70)
        assert _near(new[0], ref[0], 1e-4) and abs(float(new[1]) - float(ref[1])) < 1e-4, arm
        assert all(_near(new[2][n], ref[2][n], 1e-4) for n in ref[2] if ref[2][n] is not None), arm


def test_doubling_error_under_bf16_autocast_is_finite_and_printed(capsys):
    """Printed, not asserted: what the ambient-dtype scan costs when a and b are bf16, against the float64 closed form on the same values."""
    a, b, h0 = _ab(T=256, trained=True)
    a, b, h0 = a.bfloat16(), b.bfloat16(), h0.bfloat16()
    w = torch.randn(3, 256, 8, dtype=torch.float64)
    ref = _recur("closed64", a, b, h0, w, dtype=torch.float64)
    lines = []
    for mode, dt in (("doubling32", torch.float32), ("doubling", None)):
        got = _recur(mode, a, b, h0, w, dtype=dt)
        err = float((got[0].double() - ref[0]).abs().max()); peak = float(ref[0].abs().max())
        gerr = max(float((g.double() - r).abs().max()) for g, r in zip(got[2], ref[2]))
        lines.append(f"{mode:10} ({str(got[0].dtype):14}) max|dh| = {err:.3g} (peak |h| {peak:.3g}), max|dgrad| = {gerr:.3g}")
        assert torch.isfinite(got[0].float()).all() and all(torch.isfinite(g.float()).all() for g in got[2])
    ref_m, new_m = _step(_model("hybrid"), amp=True), _step(_model("hybrid", scan="doubling"), amp=True)
    lines.append(f"tiny hybrid under CPU bf16 autocast, doubling vs closed64: max|dlogits| = {_maxdiff(ref_m[0], new_m[0]):.3g}, loss {float(new_m[1]):.5f} vs {float(ref_m[1]):.5f}")
    assert torch.isfinite(new_m[0]).all() and torch.isfinite(new_m[1])
    with capsys.disabled():
        print("\n[--scan error, random bf16 gates, T=256]\n  " + "\n  ".join(lines))


@pytest.mark.parametrize("mode", ["doubling32", "doubling"])
def test_the_streaming_entry_point_works_in_every_scan_mode(mode):
    torch.manual_seed(0)
    layer = MinGRULayer(12, 24).eval(); layer.scan_mode = mode
    x = torch.randn(2, 70, 12)
    with torch.no_grad():
        ref = layer(x)
        for cut in (1, 31, 32, 33):                                                           # around the 32-position chunk of the closed form, and the doubling rounds
            a, h = layer.scan(x[:, :cut])
            b, _ = layer.scan(x[:, cut:], h)
            assert torch.allclose(torch.cat([a, b], 1), ref, atol=1e-5), (mode, cut)
            assert h.dtype == torch.float64 and h.shape == (2, 12) and h.untyped_storage().nbytes() == h.numel() * 8
        h = None; outs = []
        for t in range(70):                                                                   # byte by byte
            y, h = layer.scan(x[:, t : t + 1], h)
            outs.append(y)
        assert torch.allclose(torch.cat(outs, 1), ref, atol=1e-5)
        _, h_keep = layer.scan(x[:, :17]); before = h_keep.clone()
        y0, h_same = layer.scan(x[:, :0], h_keep)                                              # an empty segment: no output, the carry handed back
        assert y0.shape == (2, 0, 12) and h_same.dtype == torch.float64 and torch.equal(h_same, before) and torch.equal(h_keep, before)
        _, h_zero = layer.scan(x[:, :0])
        assert torch.equal(h_zero, torch.zeros(2, 12, dtype=torch.float64))
        ref64 = MinGRULayer(12, 24).eval(); ref64.load_state_dict(layer.state_dict())          # and the same numbers as the closed form
        assert torch.allclose(ref64(x), ref, atol=1e-5)


@pytest.mark.parametrize("mode", ["doubling32", "doubling"])
def test_a_whole_model_streams_like_it_runs_forward_in_every_scan_mode(mode):
    m = _model("hybrid", scan=mode).eval()
    x = torch.randint(0, 256, (2, 61))
    with torch.no_grad():
        ref, _ = m(x)
        for cuts in ([61], [1, 60], [8, 8, 45], [1] * 61):
            st = m.init_state(2); outs = []; c0 = 0
            for n in cuts:
                y, st = m.stream(x[:, c0 : c0 + n], st)
                outs.append(y); c0 += n
            assert torch.allclose(torch.cat(outs, 1), ref, atol=1e-4), (mode, cuts)
        assert torch.allclose(ref, _model("hybrid").eval()(x)[0], atol=1e-4)                   # and it is the closed-form model's forward


def test_scan_through_the_runner_is_recorded_and_runs_end_to_end(monkeypatch, tmp_path, capsys):
    plain, _, _ = _run(monkeypatch, tmp_path, "a")
    out, ck, _ = _run(monkeypatch, tmp_path, "b", "--scan", "doubling32")
    assert out["scan"] == "doubling32" and ck["args"]["scan"] == "doubling32" and "scan" not in plain
    assert abs(out["arms"][0]["val_bpb"] - plain["arms"][0]["val_bpb"]) < 0.05 and math.isfinite(out["arms"][0]["val_bpb"])
    with capsys.disabled():
        print(f"\n[--scan doubling32, 4 tiny steps] val_bpb {out['arms'][0]['val_bpb']:.4f} vs closed64 {plain['arms'][0]['val_bpb']:.4f}")
    from scripts.run_sparsela import apply_speed, parse_args
    capsys.readouterr()
    apply_speed(_model("tf"), parse_args(["--scan", "doubling"]), "transformer")
    assert "--scan has no effect on arm transformer (it has no minGRU layer)" in capsys.readouterr().out


def test_profile_step_times_the_doubling_scan_too(tmp_path):
    (r,) = _profile(tmp_path, "mingru_sparse_post", "--amp", "--scan", "doubling32")
    assert r["speed_switches"] == ["--scan"] and "layer1/scan" in r["forward_breakdown"]["parts"]
    fb = r["forward_breakdown"]
    assert abs(fb["unattributed_ms"]) <= 0.10 * fb["total_ms"]
