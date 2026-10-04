"""The speed switches and the local-learning switches of scripts/run_sparsela.py together (spike/speed-int): --tf32 --fused --compile
--head-amp --detach-state --scan --sparse-claws with --train-only, --freeze-layers and --local-blocks. CPU only, tiny models, in-process
runs of main() with the results in tmp_path; no compiler is needed (torch.compile is faked, or runs the aot_eager backend)."""

from __future__ import annotations

import functools
import json
import math
import sys

import pytest
import torch

CPU = torch.device("cpu")
ARM = "mingru_sparse_post"
# 8 layers so that --freeze-layers 1,3,5,7 is in range; an inner head after every 2nd layer, local attention after layer 4
MODEL = ["--task", "code", "--train-bytes", "3000", "--val-bytes", "4000", "--d-model", "16", "--n-layers", "8", "--d-ff", "32", "--n-key", "64",
         "--top-k", "4", "--key-conv", "2", "--normalize", "--gate", "--chunk", "8", "--mem-every", "2", "--local-attn", "8", "--key-convs", "2,4",
         "--batch", "2", "--seq-len", "32", "--lr", "0.003", "--wd", "0.1", "--arms", ARM, "--seed", "0", "--device", "cpu", "--run-name", "tiny",
         "--save"]
STEPS = 2
NO_COMPILER = ["--tf32", "--fused", "--head-amp", "--sparse-claws", "--scan", "doubling"]   # every speed switch that needs no compiler

# (flags of the local-learning switch, whether it freezes any parameter)
LOCAL = {
    "train-only-readout": (["--train-only", "readout"], True),
    "train-only-readout-embed-heads": (["--train-only", "readout,embed,heads"], True),
    "freeze-layers-1357": (["--freeze-layers", "1,3,5,7"], True),
    "local-blocks-2": (["--local-blocks", "2"], False),
    "local-blocks-2-train-only-readout": (["--local-blocks", "2", "--train-only", "readout"], True),
    "local-blocks-2-freeze-layers-1357": (["--local-blocks", "2", "--freeze-layers", "1,3,5,7"], True),
}


@pytest.fixture
def restore_tf32():
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def _argv(tmp_path, *flags):
    path = tmp_path / "tiny.bytes"
    if not path.exists():
        path.write_bytes((b"int main(void) { return 0; }\n" * 5000)[:130_000])
    return ["--corpus-file", str(path), *MODEL, "--steps", str(STEPS), *flags]


def _run(monkeypatch, tmp_path, tag, *flags):
    """scripts.run_sparsela.main() in-process: (the run's JSON, the checkpoint, the model as main() builds and freezes it before any step)."""
    import scripts.run_sparsela as sra
    res = tmp_path / tag
    res.mkdir()
    monkeypatch.setattr(sra, "RESULTS", res)
    monkeypatch.setattr(sra, "_CMASK", {})
    argv = _argv(tmp_path, *flags)
    args = sra.parse_args(argv)
    ref = sra.make_model(ARM, args, CPU)                                    # the same seed: the random initialisation the run starts from
    sra.apply_freezing(ref, args.train_only, args.freeze_layers)
    monkeypatch.setattr(sys, "argv", ["run_sparsela.py", *argv])
    sra.main()
    return (json.loads((res / "tiny.json").read_text()),
            torch.load(res / "ckpt" / f"tiny_{ARM}.pt", map_location="cpu", weights_only=False), ref)


def _check(out, ck, ref, freezes, blocks=0):
    """Two steps ran with a finite loss, every frozen parameter is bit-identical to its initial value, and the trainable ones moved."""
    arm = out["arms"][0]
    assert len(arm["curve"]) == STEPS and all(math.isfinite(r["train_bpb"]) for r in arm["curve"])
    assert math.isfinite(arm["val_bpb"])
    if blocks:
        assert all(len(r["block_bpb"]) == ref.n_blocks and all(math.isfinite(b) for b in r["block_bpb"]) for r in arm["curve"])
    init = ref.state_dict()
    frozen = [n for n, p in ref.named_parameters() if not p.requires_grad]
    trainable = [n for n, p in ref.named_parameters() if p.requires_grad]
    assert bool(frozen) == freezes and trainable
    for n in frozen:
        assert torch.equal(ck["state"][n], init[n]), f"frozen parameter {n} changed"
    assert any(not torch.equal(ck["state"][n], init[n]) for n in trainable)   # the run did train something
    assert ck["state"].keys() == init.keys()


@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "amp"])
@pytest.mark.parametrize("local", list(LOCAL))
def test_the_no_compiler_speed_switches_run_with_every_local_learning_switch_and_frozen_parameters_stay_put(monkeypatch, tmp_path, restore_tf32, local, amp):
    flags, freezes = LOCAL[local]
    out, ck, ref = _run(monkeypatch, tmp_path, "a", *NO_COMPILER, *flags, *(["--amp"] if amp else []))
    _check(out, ck, ref, freezes, blocks=2 if "--local-blocks" in flags else 0)
    assert out["scan"] == "doubling" and all(out[k] is True for k in ("tf32", "fused", "head_amp", "sparse_claws")) and out["compile"] is False
    assert ("--train-only" in flags) == ("train_only" in out) and ("--local-blocks" in flags) == ("local_blocks" in out)   # recorded when on


def test_the_switches_are_on_in_the_model_the_local_learning_switches_freeze(monkeypatch, tmp_path, restore_tf32):
    from flybrain.model.hybrid import MinGRULayer
    from scripts.run_sparsela import apply_freezing, make_model, parse_args
    args = parse_args(_argv(tmp_path, *NO_COMPILER, "--freeze-layers", "1,3,5,7", "--local-blocks", "2"))
    m = make_model(ARM, args, CPU)
    apply_freezing(m, args.train_only, args.freeze_layers)
    assert all(l.scan_mode == "doubling" for l in m.modules() if isinstance(l, MinGRULayer))
    assert m.all_heads() and all(h.head_amp and h.code.sparse_claws for h in m.all_heads())
    assert any(not p.requires_grad for p in m.parameters()) and m.n_blocks == 4


def test_a_command_line_with_neither_family_of_switches_is_unchanged_by_the_merge(monkeypatch, tmp_path):
    from scripts.run_sparsela import SPEED_FLAGS, SWITCH_FLAGS
    out, ck, _ = _run(monkeypatch, tmp_path, "a")
    assert not set(SPEED_FLAGS) & set(out) and not set(SPEED_FLAGS) & set(ck["args"])
    assert not set(SWITCH_FLAGS) & set(out) and not set(SWITCH_FLAGS) & set(ck["args"])
    assert not any("aux" in k for k in ck["state"])


def test_one_family_of_switches_records_only_its_own_flags(monkeypatch, tmp_path, restore_tf32):
    from scripts.run_sparsela import SPEED_FLAGS, SWITCH_FLAGS
    speed, _, _ = _run(monkeypatch, tmp_path, "a", "--head-amp")
    assert set(SPEED_FLAGS) <= set(speed) and not set(SWITCH_FLAGS) & set(speed)
    local, _, _ = _run(monkeypatch, tmp_path, "b", "--train-only", "readout")
    assert not set(SPEED_FLAGS) & set(local) and local["train_only"] == "readout"
    assert "freeze_layers" not in local and "local_blocks" not in local


# ---------------------------------------------------------------------------------------------------------------------
# --compile: the training step only, with --train-only / --freeze-layers (no code change) and with --local-blocks (BlocksForward)
# ---------------------------------------------------------------------------------------------------------------------

class Wrapped(torch.nn.Module):
    """What torch.compile returns: a wrapper that owns the module it was given (as tests/test_speed_switches.py fakes it)."""

    def __init__(self, m, log):
        super().__init__(); self._orig_mod = m; self.log = log

    def forward(self, *a, **k):
        self.log.append(type(self._orig_mod).__name__)
        return self._orig_mod(*a, **k)


@pytest.mark.parametrize("local", ["train-only-readout", "train-only-readout-embed-heads", "freeze-layers-1357"])
def test_compile_with_train_only_and_freeze_layers_compiles_the_model_for_the_training_step_only(monkeypatch, tmp_path, local):
    flags, freezes = LOCAL[local]
    seen = []
    monkeypatch.setattr(torch, "compile", lambda m, *a, **k: Wrapped(m, seen))
    out, ck, ref = _run(monkeypatch, tmp_path, "a", "--compile", *flags)
    assert out["compile"] is True and seen == ["GRUHybrid"] * STEPS                                          # the training steps, and no eval went through it
    assert not any(k.startswith("_orig_mod") for k in ck["state"]) and "mem.val.weight" in ck["state"]
    _check(out, ck, ref, freezes)


def test_compile_with_local_blocks_compiles_the_block_forward_and_trains_the_same_model(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(torch, "compile", lambda m, *a, **k: Wrapped(m, seen))
    out, ck, ref = _run(monkeypatch, tmp_path, "a", "--compile", "--local-blocks", "2")
    assert seen == ["BlocksForward"] * STEPS                                                                 # the block forward, not the plain one
    assert not any(k.startswith(("_orig_mod", "model.")) for k in ck["state"]) and ck["state"].keys() == ref.state_dict().keys()
    assert any(k.startswith("aux_decoders") for k in ck["state"])
    _check(out, ck, ref, False, blocks=2)
    monkeypatch.undo()                                                                                       # the same run, never compiled
    plain, ck_plain, _ = _run(monkeypatch, tmp_path, "b", "--local-blocks", "2")
    assert all(torch.equal(ck["state"][k], ck_plain["state"][k]) for k in ck["state"])                       # the wrapper changes no number


def test_make_train_fwd_hands_torch_compile_the_plain_model_unless_local_blocks_is_on(monkeypatch):
    from scripts.run_sparsela import BlocksForward, make_model, make_train_fwd, parse_args
    got = []
    monkeypatch.setattr(torch, "compile", lambda m, *a, **k: got.append(m) or m)
    base = ["--arms", ARM, "--d-model", "16", "--n-layers", "4", "--d-ff", "32", "--n-key", "64", "--top-k", "4", "--chunk", "8"]
    m = make_model(ARM, parse_args(base), CPU)
    assert make_train_fwd(m, parse_args(base)) is m and not got                                              # no switch: the module itself
    assert make_train_fwd(m, parse_args([*base, "--compile"])) is m and got == [m]                           # --compile: the model, as before
    got.clear()
    argb = parse_args([*base, "--local-blocks", "2"])
    mb = make_model(ARM, argb, CPU)
    wrapper = make_train_fwd(mb, argb)
    assert isinstance(wrapper, BlocksForward) and wrapper.model is mb and not got                            # no --compile: not compiled
    make_train_fwd(mb, parse_args([*base, "--local-blocks", "2", "--compile"]))
    assert len(got) == 1 and isinstance(got[0], BlocksForward) and got[0].model is mb                        # --compile: the block forward
    assert list(got[0].parameters()) == list(mb.parameters())                                                # it owns no parameter of its own


def _dynamo_or_skip():
    """The aot_eager backend: dynamo traces the step and AOTAutograd builds the backward, with no C++ compiler. Skip where even that fails."""
    real = torch.compile
    try:
        torch._dynamo.reset()
        real(lambda t: t * 2, backend="aot_eager")(torch.ones(2, requires_grad=True)).sum().backward()
    except Exception as e:                                                                                   # no usable dynamo on this Python / build
        pytest.skip(f"torch.compile (dynamo, aot_eager) is not usable here: {type(e).__name__}")
    return real


@pytest.mark.parametrize("flags", [[], ["--head-amp", "--sparse-claws", "--scan", "doubling"]], ids=["plain", "speed"])
def test_the_compiled_local_blocks_loss_and_gradients_agree_with_the_plain_ones(monkeypatch, flags):
    from scripts.run_sparsela import make_model, make_train_fwd, parse_args, train_loss
    real = _dynamo_or_skip()
    monkeypatch.setattr(torch, "compile", functools.partial(real, backend="aot_eager"))
    args = parse_args(["--arms", ARM, "--d-model", "16", "--n-layers", "4", "--d-ff", "32", "--n-key", "64", "--top-k", "4", "--key-conv", "2",
                       "--normalize", "--gate", "--chunk", "8", "--mem-every", "2", "--local-attn", "8", "--key-convs", "2,4",
                       "--local-blocks", "2", "--compile", *flags])
    m = make_model(ARM, args, CPU)
    g = torch.Generator().manual_seed(1)
    x, y = torch.randint(0, 256, (2, 32), generator=g), torch.randint(0, 256, (2, 32), generator=g)

    def run(fwd):
        m.zero_grad(set_to_none=True)
        loss, parts = train_loss(m, x, y, args, CPU, fwd)
        loss.backward()
        return loss.detach(), [p.detach() for p in parts], {n: None if p.grad is None else p.grad.clone() for n, p in m.named_parameters()}

    try:
        plain = run(None)                                                                                    # eager: forward_blocks of the plain module
        compiled = run(make_train_fwd(m, args))                                                              # the compiled block forward
    finally:
        torch._dynamo.reset()
    assert len(plain[1]) == len(compiled[1]) == m.n_blocks == 2 and torch.isfinite(compiled[0])
    assert torch.allclose(plain[0], compiled[0], atol=1e-5) and all(torch.allclose(a, b, atol=1e-5) for a, b in zip(plain[1], compiled[1]))
    for n, gp in plain[2].items():
        gc = compiled[2][n]
        assert (gp is None) == (gc is None), n
        if gp is not None:
            assert torch.allclose(gp, gc, atol=1e-5, rtol=1e-4), n
    assert compiled[2]["aux_decoders.0.weight"].abs().sum() > 0                                              # the earlier block's readout is in the compiled graph
