#!/usr/bin/env python3
"""Long-sequence tests for sparse-key linear attention vs learned keys, Gaussian keys, softmax, GRU."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.baselines.gru import ByteGRU
from flybrain.data.corpus import copyable_mask
from flybrain.data.memfeat import FeatureStore
from flybrain.data.task import copy_batch, multi_copy_batch
from flybrain.data.text import batch_starts, batch_windows, fixed_windows, load_bytes, ngram_bpb
from flybrain.meter import GpuMeter, SplitClock, device_uuid
from flybrain.model.hybrid import SCAN_MODES, GRUHybrid, RopeAttention, configure_speed
from flybrain.model.inputs import channel_kwargs, parse_orders
from flybrain.model.sparsela import Block, SparseKeyLM

RESULTS = ROOT / "results"
WIKI = Path("/games/ailab/lab/scrt-lab/data/wiki.train.raw")


def numel(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def is_hybrid_arm(arm: str) -> bool:
    """True for the arms build() makes a GRUHybrid of: mingru, gla, transformer and <backbone>_<key>_<pre|post>."""
    return arm in ("mingru", "gla", "transformer") or arm.count("_") == 2


def build(arm: str, args) -> torch.nn.Module:
    torch.manual_seed(args.seed)
    local_blocks = getattr(args, "local_blocks", 0)   # getattr: callers with a partial Namespace keep working
    if local_blocks and not is_hybrid_arm(arm):
        raise ValueError(f"--local-blocks needs a GRUHybrid arm, not {arm!r}")
    chan = channel_kwargs(vars(args))  # the extra input channels (--mem-feat, --ngram-emb, --match-feat); empty when none is on
    if chan and not (arm in ("mingru", "gla", "transformer") or (arm != "gru" and arm.count("_") == 2)):
        raise ValueError(f"the extra input channels (--mem-feat, --ngram-emb, --match-feat) are built into the GRUHybrid arms (mingru, gla, transformer, <backbone>_<key>_<pre|post>), not into {arm!r}")
    if arm == "gru":
        return ByteGRU(args.d_model, args.gru_hidden)
    if arm in ("mingru", "gla", "transformer"):
        # local attention is only meaningful on a layered recurrence; the transformer backbone rejects it
        extra = {} if arm == "transformer" else {"local_attn": args.local_attn,
                                                 "local_at": tuple(int(i) for i in args.local_at.split(",")) if args.local_at else None}
        return GRUHybrid(d_model=args.d_model, hidden=args.gru_hidden, key="sparse", where="none", backbone=arm,
                         n_layers=args.n_layers, d_ff=args.d_ff, learned_key=args.learned_key_dim, key_conv=args.key_conv, seed=args.seed,
                         max_len=max(1024, args.seq_len), rope=args.rope, local_blocks=local_blocks, **extra, **chan)
    if arm.count("_") == 2:  # <backbone>_<key>_<pre|post>
        bb, key, where = arm.split("_")
        return GRUHybrid(d_model=args.d_model, hidden=args.gru_hidden, key=key, where=where, n_key=args.n_key, backbone=bb,
                         n_layers=args.n_layers, d_ff=args.d_ff,
                         top_k=args.top_k, claws=args.claws, theta=args.theta, learned_key=args.learned_key_dim,
                         normalize=args.normalize, key_conv=args.key_conv, decay=args.decay, seed=args.seed,
                         shuffle_keys=args.shuffle_keys, gate=args.gate, max_len=max(1024, args.seq_len), rope=args.rope, chunk=args.chunk,
                         mem_every=args.mem_every, key_convs=tuple(int(k) for k in args.key_convs.split(",")) if args.key_convs else None,
                         count_head=args.count_head, learn_decay=args.learn_decay, local_attn=args.local_attn,
                         decays=tuple(float(x) for x in args.decays.split(",")) if args.decays else None,
                         local_at=tuple(int(i) for i in args.local_at.split(",")) if args.local_at else None,
                         local_blocks=local_blocks, **chan)
    n_key = args.n_key if arm in ("sparse", "gausstopk", "gauss", "mixed") else args.learned_key
    return SparseKeyLM(key=arm, d_model=args.d_model, n_key=n_key, claws=args.claws, top_k=args.top_k,
                       theta=args.theta, n_layers=args.n_layers, d_ff=args.d_ff, decay=args.decay, seed=args.seed,
                       key_conv=args.key_conv, freeze_conv=args.freeze_conv,
                       normalize=args.normalize, signed=args.signed, learned_key=args.learned_key_dim)


def lr_lambda(steps: int, warmup: int, cosine: bool):
    def f(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if not cosine:
            return 1.0
        p = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


NGRAM_TABLE = "ngram.table.weight"  # the hashed n-gram rows: no weight decay (decay would pull every row that no window touched to zero), and with --ngram-sparse not in the main optimizer


def make_optimizer(model, args):
    """Only the parameters that learn: a frozen one (--train-only, --freeze-layers) is not in the optimizer, so no step and no weight decay moves it.
    --fused: the one-kernel optimizer, CUDA parameters only (silently the ordinary one on CPU). --ngram-sparse: the n-gram table is left out
    (make_sparse_optimizer trains it); it never gets weight decay, which would pull every row no window touched to zero."""
    kw = {"fused": True} if getattr(args, "fused", False) and next(model.parameters()).is_cuda else {}
    sparse = bool(getattr(args, "ngram_sparse", False))
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not (sparse and n == NGRAM_TABLE)]
    if not args.wd:
        return torch.optim.Adam([p for _, p in named], lr=args.lr, **kw)
    decay = [p for n, p in named if p.dim() >= 2 and n != NGRAM_TABLE]
    no_decay = [p for n, p in named if p.dim() < 2 or n == NGRAM_TABLE]
    groups = [{"params": ps, "weight_decay": wd} for ps, wd in ((decay, args.wd), (no_decay, 0.0)) if ps]   # --train-only norms leaves one group empty
    return torch.optim.AdamW(groups, lr=args.lr, **kw)


def make_sparse_optimizer(model, args):
    """With --ngram-sparse, a SparseAdam on the n-gram table (sparse gradients, no weight decay), same learning rate; None otherwise, and None
    when the table is frozen (--train-only or --freeze-layers leave it out of every group)."""
    if not getattr(args, "ngram_sparse", False):
        return None
    ps = [p for n, p in model.named_parameters() if n == NGRAM_TABLE and p.requires_grad]
    return torch.optim.SparseAdam(ps, lr=args.lr) if ps else None


def make_opt_sched(model, args):
    """The optimizer and the lr schedule exactly as main() trains with them."""
    opt = make_optimizer(model, args)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(args.steps, args.warmup, args.cosine))


def set_precision(args) -> None:
    """--tf32: allow TF32 for float32 matmuls and cuDNN convolutions (CUDA). Off leaves PyTorch's own settings alone."""
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def apply_speed(model, args, arm: str = "") -> dict:
    """Hand the model switches (--head-amp, --detach-state, --scan, --sparse-claws) to every module that has them; say so when a switch finds none.
    --sparse-claws refuses (ValueError) a frozen projection whose rows do not all have exactly --claws nonzeros."""
    touched = configure_speed(model, head_amp=args.head_amp, detach_state=args.detach_state, scan=args.scan, sparse_claws=args.sparse_claws)
    for flag, n in touched.items():
        if getattr(args, flag) != SPEED_DEFAULTS[flag] and not n:
            print(f"note: --{flag.replace('_', '-')} has no effect on arm {arm or type(model).__name__} (it has no {dict(scan='minGRU layer', sparse_claws='sparse-key head').get(flag, 'memory head')})", flush=True)
    if args.detach_state and touched["detach_state"] and not any(m.chunk > 0 for m in model.modules() if isinstance(m, Block)):
        print(f"note: --detach-state needs --chunk > 0 (the T x T form carries no tables between chunks): it does nothing on arm {arm or type(model).__name__}", flush=True)
    return touched


def make_model(arm: str, args, device) -> torch.nn.Module:
    """The arm as main() trains it: built, on the device, with the head state bits and the speed switches applied."""
    model = build(arm, args).to(device)
    if args.state_bits and hasattr(model, "all_heads"):
        for h in model.all_heads(): h.state_bits = int(args.state_bits)
    apply_speed(model, args, arm)
    if getattr(args, "rope_table", False):
        layers = [m for m in model.modules() if isinstance(m, RopeAttention)]
        for m in layers: m.use_rope_table(max(1024, args.seq_len))
        if not layers:
            print(f"note: --rope-table has no effect on arm {arm or type(model).__name__} (it has no rotary attention layer: the transformer with --rope has)", flush=True)
    return model


TRAIN_ONLY_GROUPS = ("readout", "embed", "heads", "norms")
SWITCH_FLAGS = ("train_only", "freeze_layers", "local_blocks")   # the local-learning switches: recorded only when on, so an unswitched run writes the file it always wrote


def parse_train_only(spec: str) -> list[str]:
    """--train-only GROUPS -> the group names, once each, in order. An unknown or empty name is refused."""
    names = [s.strip() for s in spec.split(",")]
    bad = [n for n in names if n not in TRAIN_ONLY_GROUPS]
    if bad:
        raise ValueError(f"unknown --train-only group {', '.join(repr(n) for n in bad)}: the groups are {', '.join(TRAIN_ONLY_GROUPS)}")
    return list(dict.fromkeys(names))


def parse_freeze_layers(spec: str, n_layers: int) -> list[int]:
    """--freeze-layers LIST -> sorted 0-based layer indices, once each. Anything that is not an integer, or lies outside the stack, is refused."""
    out = []
    for s in spec.split(","):
        try:
            i = int(s)
        except ValueError:
            raise ValueError(f"--freeze-layers wants a comma list of integer layer indices, got {s!r} in {spec!r}") from None
        if not 0 <= i < n_layers:
            raise ValueError(f"--freeze-layers {i} is outside the stack of {n_layers} layers (indices 0..{n_layers - 1})")
        out.append(i)
    return sorted(set(out))


def apply_freezing(model, train_only: str = "", freeze_layers: str = "") -> dict | None:
    """--train-only and --freeze-layers: a parameter keeps learning only if both allow it; every other one gets requires_grad_(False) and stays at its
    random initialisation (make_optimizer leaves it out). Gradients still pass through frozen layers. Prints the counts and returns them for the result
    JSON; None when both switches are off. Refuses a bad spec before freezing anything."""
    if not (train_only or freeze_layers):
        return None
    if not hasattr(model, "param_groups"):
        raise ValueError("--train-only and --freeze-layers are for the GRUHybrid arms (mingru, gla, transformer, <backbone>_<key>_<pre|post>)")
    groups = model.param_groups()
    names = parse_train_only(train_only) if train_only else []
    layers, frozen = [], set()
    if freeze_layers:
        units = model.layer_units()
        layers = parse_freeze_layers(freeze_layers, len(units))
        frozen = {id(p) for i in layers for mod in units[i] for p in mod.parameters()}
    how = ([f"--train-only {','.join(names)}"] if names else []) + ([f"--freeze-layers {','.join(map(str, layers))}"] if layers else [])
    allowed = {id(p) for n in names for p in groups[n]}
    keep = {id(p) for p in model.parameters() if p.requires_grad and (not names or id(p) in allowed) and id(p) not in frozen}
    if not keep:
        raise ValueError(f"{' '.join(how)} leaves no parameter to train in this model")
    for p in model.parameters():
        p.requires_grad_(id(p) in keep)
    in_groups = {id(p) for ps in groups.values() for p in ps}
    size = {n: sum(p.numel() for p in ps) for n, ps in groups.items()}
    size["other"] = sum(p.numel() for p in model.parameters() if id(p) not in in_groups)                     # the stack's layers, the local attention
    on = {n: sum(p.numel() for p in ps if p.requires_grad) for n, ps in groups.items()}
    on["other"] = sum(p.numel() for p in model.parameters() if p.requires_grad and id(p) not in in_groups)
    rep = {"trainable_scalars": numel(model), "total_scalars": sum(p.numel() for p in model.parameters()), "trainable_by_group": on}
    if names:
        rep["train_only"] = names
    if layers:
        rep["frozen_layers"] = layers
    print(f"{' '.join(how)}: trainable by group (trainable/size) " + ", ".join(f"{n} {on[n]:,}/{size[n]:,}" for n in on), flush=True)   # no leading spaces: run_*.sh drop lines that start with two
    print(f"trainable {rep['trainable_scalars']:,} of {rep['total_scalars']:,} parameters", flush=True)
    return rep


def _copy(args, seed):
    if args.n_spans > 1:
        return multi_copy_batch(args.batch, seq_len=args.seq_len, span=args.span, n_spans=args.n_spans, seed=seed)
    return copy_batch(args.batch, seq_len=args.seq_len, span=args.span, seed=seed)


def _amp(args, device):
    """bf16 autocast when --amp; a no-op context otherwise."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(getattr(args, "amp", False)))


class BlocksForward(torch.nn.Module):
    """--local-blocks: model.forward_blocks as the forward of a module, so that torch.compile has a forward that covers the block readouts (compiling
    the model alone compiles model.forward, which the readouts of the earlier blocks are never asked of when called from outside). It owns no
    parameters of its own: the model is the one the optimizer, the checkpoints and the evals use."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_blocks(x)


class SkipForward(torch.nn.Module):
    """--skip-frac: model.forward_skip as the forward of a module (as BlocksForward), so that the compiled training forward returns the cheap
    gate head's logits for its own loss. Owns no parameters."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, **kw):
        return self.model.forward_skip(x, **kw)


def make_train_fwd(model, args):
    """The callable the training step runs: model(x) -> (logits, state), or with --local-blocks BlocksForward(model)(x) -> (logits, block logits),
    or with --skip-frac SkipForward(model)(x, **kw) -> (logits, gate logits); torch.compile of it with --compile. The evals, the checkpoints
    and the optimizer keep the plain module."""
    if getattr(args, "skip_frac", 0.0):
        fwd = SkipForward(model)
    else:
        fwd = BlocksForward(model) if getattr(args, "local_blocks", 0) else model
    return torch.compile(fwd) if getattr(args, "compile", False) else fwd


def train_loss(model, x, y, args, device, fwd=None, kw=None):
    """(the loss to backpropagate, the next-byte loss of each block's readout with the model's own output last).
    Plain: the cross-entropy of the output, one part. With --local-blocks: the sum of the block losses, so every block learns from its own
    readout only (no gradient crosses a block boundary); the last part is still the model's output, the only one evaluation uses.
    fwd: what runs the forward, as make_train_fwd returns it (compiled with --compile); None = the plain module, never compiled.
    kw: the keyword tensors of forward for this batch (--mem-feat: mem_pred, mem_conf), passed on to fwd; --local-blocks takes none."""
    blocks = getattr(args, "local_blocks", 0)
    kw = kw or {}
    if blocks and kw:
        raise ValueError("--local-blocks trains through forward_blocks, which takes no extra forward inputs (--mem-feat)")
    skip = getattr(args, "skip_frac", 0.0)
    if fwd is None:
        fwd = model.forward_skip if skip else model.forward_blocks if blocks else model
    with _amp(args, device):
        if blocks:
            logits, aux = fwd(x)
            parts = [F.cross_entropy(l.float().reshape(-1, 256), y.reshape(-1)) for l in (*aux, logits)]
            return sum(parts), parts
        if skip:   # --skip-frac: the gate head's own loss (weighted) plus the model's; the model's output last, as with --local-blocks
            logits, gate = fwd(x, **kw)
            parts = [F.cross_entropy(gate.float().reshape(-1, 256), y.reshape(-1)), F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))]
            return args.skip_aux * parts[0] + parts[1], parts
        logits, _ = fwd(x, **kw)
        loss = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))
    return loss, [loss]


@torch.no_grad()
def eval_copy(model, args, n_batches, seed, device):
    model.eval(); tot = n = hit = 0
    for i in range(n_batches):
        x, y, mask = _copy(args, seed + i)
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="none").view_as(y)
        tot += float(ce[mask].sum()); n += int(mask.sum()); hit += int((logits.argmax(-1)[mask] == y[mask]).sum())
    model.train()
    return {"copy_bpb": tot / n / math.log(2), "copy_acc": hit / n}


def _features_to(device, feats) -> dict:
    """The keyword tensors of forward for a (pred, conf) pair of uint8 feature tensors: mem_pred, mem_conf, on the device."""
    return {"mem_pred": feats[0].to(device), "mem_conf": feats[1].to(device)}


@torch.no_grad()
def eval_wiki(model, val, args, n_batches, seed, device, store=None, val_offset=0):
    """store: a FeatureStore when the model takes table features; val_offset: the absolute byte position val starts at (the features are indexed by it)."""
    model.eval(); T = args.seq_len
    tot = torch.zeros(T, device=device); cnt = 0
    cp_sum = cp_n = ncp_sum = ncp_n = 0.0
    for i in range(n_batches):
        x, y = batch_windows(val, seq_len=T, batch=args.batch, seed=seed + i)
        cm = _CMASK.setdefault((seed + i, T, args.batch), copyable_mask(x, y, n=8)).to(device)
        x, y = x.to(device), y.to(device)
        kw = {} if store is None else _features_to(device, store.windows(val_offset + batch_starts(val.numel(), T, args.batch, seed + i), T))
        with _amp(args, device):
            logits, _ = model(x, **kw)
        ce = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1), reduction="none").view(-1, T)
        tot += ce.sum(0); cnt += ce.size(0)
        cp_sum += float(ce[cm].sum()); cp_n += int(cm.sum())
        ncp_sum += float(ce[~cm].sum()); ncp_n += int((~cm).sum())
    model.train()
    per_pos = (tot / cnt / math.log(2)).tolist()
    q = T // 4
    return {"val_bpb": sum(per_pos) / T, "val_bpb_first_q": sum(per_pos[:q]) / q,
            "val_bpb_last_q": sum(per_pos[-q:]) / q,
            "val_bpb_copyable": cp_sum / max(cp_n, 1) / math.log(2), "copyable_frac": cp_n / max(cp_n + ncp_n, 1),
            "val_bpb_noncopyable": ncp_sum / max(ncp_n, 1) / math.log(2)}


@torch.no_grad()
def eval_fixed(model, xf, yf, args, device, ff=None) -> dict:
    """Score fixed windows (xf, yf: [N, T] from fixed_windows) in batches of args.batch under the training amp context.
    cold = mean bits/byte over every position; warm = mean over positions T // 2 onward. No RNG, any N.
    ff: the windows' table features (pred, conf), uint8 [N, T, orders] (FeatureStore.fixed), for a model that takes them."""
    model.eval(); T = xf.size(1)
    tot = torch.zeros(T, dtype=torch.float64, device=device)
    for i in range(0, xf.size(0), args.batch):
        x, y = xf[i:i + args.batch].to(device), yf[i:i + args.batch].to(device)
        kw = {} if ff is None else _features_to(device, [t[i:i + args.batch] for t in ff])
        with _amp(args, device):
            logits, _ = model(x, **kw)
        tot += F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1), reduction="none").view(-1, T).double().sum(0)
    model.train()
    per_pos = tot / xf.size(0) / math.log(2)
    return {"bpb": float(per_pos.mean()), "bpb_warm": float(per_pos[T // 2:].mean())}


_CMASK: dict = {}

CHECK_EVERY = 50  # steps between GpuMeter.check() calls
YARDSTICK_FLAGS = ("val_fixed", "val_offset", "eval_every", "meter", "ckpt_every")
SPEED_DEFAULTS = {"tf32": False, "fused": False, "compile": False, "head_amp": False, "detach_state": False, "scan": "closed64", "sparse_claws": False}  # the speed switches, all off
SPEED_FLAGS = tuple(SPEED_DEFAULTS)


def yardstick_on(args) -> bool:
    """True when any yardstick flag is set. Only then do the new fields reach the JSON, so a legacy command line writes a legacy file."""
    return bool(args.val_fixed or args.val_offset or args.eval_every or args.meter or args.ckpt_every)


def speed_on(args) -> bool:
    """True when any speed switch is not at its default."""
    return any(getattr(args, k) != v for k, v in SPEED_DEFAULTS.items())


CHANNEL_FLAGS = (("mem_feat", "mem_orders"), ("ngram_emb", "ngram_rows", "ngram_dim", "ngram_sparse"), ("match_feat",))  # each group is recorded only when its first flag is on


def recorded_args(args) -> dict:
    """vars(args) as a run records it (JSON and checkpoints): the yardstick flags appear only when one of them is set, the speed switches
    only when one of them is on, the local-learning switches only when they are (a --local-blocks checkpoint needs its value to be rebuilt),
    and the flags of an extra input channel only when that channel is on, so a command line without them writes the file it always wrote."""
    d = vars(args) if yardstick_on(args) else {k: v for k, v in vars(args).items() if k not in YARDSTICK_FLAGS}
    d = d if speed_on(args) else {k: v for k, v in d.items() if k not in SPEED_FLAGS}
    off = {k for group in CHANNEL_FLAGS if not getattr(args, group[0], "") for k in group}
    return {k: v for k, v in d.items() if k not in off and (k not in SWITCH_FLAGS and k not in ("rope_table", "fused_channels") or v)}   # --rope-table, --fused-channels: recorded only when on


def make_meter(args, device) -> GpuMeter | None:
    """A GpuMeter for the physical card behind device when --meter is set; None on CPU, where --meter does nothing."""
    if not args.meter:
        return None
    if device.type != "cuda":
        print("meter: skipped, device is cpu", flush=True)
        return None
    return GpuMeter(device_uuid(device), os.getpid())


def fixed_row(arm, model, xf, yf, args, device, step: int, train_seconds: float, ff=None) -> dict:
    row = {"step": step, "train_seconds": train_seconds, **eval_fixed(model, xf, yf, args, device, ff)}
    print(f"fixed {arm} {row}", flush=True)  # no leading spaces: run_*.sh drop lines that start with two
    return row


def save_ckpt(path: Path, arm: str, args, step: int, model, opt, sched, sparse_opt=None) -> None:
    """Model, optimizer and scheduler at a step (and the n-gram table's SparseAdam, with --ngram-sparse). Save only: there is no resume path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arm": arm, "args": recorded_args(args), "step": step, "state": model.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(), **({"opt_sparse": sparse_opt.state_dict()} if sparse_opt is not None else {})}, path)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="copy", choices=("copy", "wiki", "code", "terminal"))
    p.add_argument("--corpus-file", default="")
    p.add_argument("--n-spans", type=int, default=1)
    p.add_argument("--save", action="store_true")
    p.add_argument("--learned-key-dim", type=int, default=64)
    p.add_argument("--arms", default="sparse,gausstopk,gauss,learned,softmax,gru")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--span", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-key", type=int, default=1024)
    p.add_argument("--learned-key", type=int, default=64)
    p.add_argument("--claws", type=int, default=4)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--decay", type=float, default=0.98)
    p.add_argument("--gru-hidden", type=int, default=160)
    p.add_argument("--key-conv", type=int, default=1)
    p.add_argument("--freeze-conv", action="store_true")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--signed", action="store_true")
    p.add_argument("--shuffle-keys", action="store_true", help="leakage control for gru_* arms")
    p.add_argument("--gate", action="store_true", help="match-strength gate on the memory head output")
    p.add_argument("--rope", action="store_true", help="rotary positions for the transformer backbone")
    p.add_argument("--chunk", type=int, default=0, help="chunk length for the linear-cost memory head (0 = T x T parallel form)")
    p.add_argument("--amp", action="store_true", help="train and evaluate under bfloat16 autocast (backbone matmuls/attention in bf16; the memory head stays fp32)")
    p.add_argument("--state-bits", type=int, default=0, help=">0: fake-quantise every memory head's carried state to this many bits between chunks while training (needs --chunk)")
    p.add_argument("--mem-every", type=int, default=0, help="extra memory head after every k-th backbone layer (memory that grows with depth)")
    p.add_argument("--key-convs", default="", help="comma list of key widths cycled over heads, e.g. 2,4 (default: --key-conv for all)")
    p.add_argument("--count-head", action="store_true", help="next-byte count memory mixed into the output distribution")
    p.add_argument("--learn-decay", action="store_true", help="learned recency decay per head (init at --decay, which must be in (0,1))")
    p.add_argument("--local-attn", type=int, default=0, help=">0: one sliding-window attention layer of this window mid-stack")
    p.add_argument("--local-at", default="", help="comma list of layer counts after which a local-attention layer sits (default n_layers//2); several = several layers")
    p.add_argument("--decays", default="", help="comma list of per-head decays cycled over heads in all_heads() order (top first), e.g. 1.0,0.999")
    p.add_argument("--warmup", type=int, default=0, help="linear lr warmup steps")
    p.add_argument("--cosine", action="store_true", help="cosine decay to 10%% of lr after warmup")
    p.add_argument("--wd", type=float, default=0.0, help="AdamW weight decay (0 = plain Adam)")
    p.add_argument("--train-bytes", type=int, default=2_000_000)
    p.add_argument("--val-bytes", type=int, default=200_000)
    p.add_argument("--run-name", default="")
    p.add_argument("--val-fixed", type=int, default=0, help=">0: also score this many fixed non-overlapping windows from --val-offset every --eval-every steps and at the end (cold and warm bits/byte)")
    p.add_argument("--val-offset", type=int, default=0, help="byte offset of the fixed-window region (needs --val-fixed)")
    p.add_argument("--eval-every", type=int, default=0, help="steps between fixed-window evals, 0 = steps // 6 (needs --val-fixed); the legacy curve keeps its steps // 6 cadence")
    p.add_argument("--meter", action="store_true", help="sample the card once a second while training: joules, power, temperature, contention; aborts at 85 C (a no-op on CPU)")
    p.add_argument("--ckpt-every", type=int, default=0, help=">0: save model, optimizer and scheduler every this many steps (save only, no resume)")
    # speed switches: all off by default (today's behaviour); scripts/profile_step.py times each one
    p.add_argument("--tf32", action="store_true", help="allow TF32 for float32 matmuls and cuDNN convolutions (CUDA), the memory head's pinned-fp32 matmuls included; off = PyTorch's own settings")
    p.add_argument("--fused", action="store_true", help="AdamW (Adam with --wd 0) with fused=True; CUDA only, the ordinary optimizer on CPU")
    p.add_argument("--compile", action="store_true", help="torch.compile the model for the training step (with --local-blocks, the block forward that also returns each block's readout logits); evals, checkpoints and the state_dict use the plain module, so checkpoints are unchanged")
    p.add_argument("--head-amp", action="store_true", help="the memory heads follow the surrounding autocast instead of being pinned to fp32: with --amp their matmuls, the top-k selection matmul included, run in bf16 (the tables S and Ksum and the decay arithmetic stay fp32); no effect without --amp")
    p.add_argument("--detach-state", action="store_true", help="the memory tables a head carries from earlier chunks are constants for the gradient (detached at every --chunk boundary; needs --chunk); the forward is unchanged. Cut: the gradient from a read in one chunk into the writes (values and key magnitudes) of all earlier chunks. Kept: the gradient through the writes and the reads inside a chunk, and from a read of the carried tables into the query")
    p.add_argument("--scan", default="closed64", choices=SCAN_MODES, help="how the minGRU recurrence h_t = a_t h_{t-1} + b_t is computed: closed64 = the chunked closed form in float64 with a Python loop over chunk carries (today's code); doubling32 = prefix doubling on the affine pairs (a, b) in float32 (no division, no float64, no loop over chunks); doubling = the same in the ambient autocast dtype (bf16 under --amp)")
    p.add_argument("--sparse-claws", action="store_true", help="compute the selected activations of the memory head from the --claws nonzeros of each frozen projection row (buffers of claw indices and weights, built once) instead of gathering whole rows of d; which units fire is still the dense matmul. Refuses a projection whose rows do not all have exactly --claws nonzeros")
    p.add_argument("--skip-frac", type=float, default=0.0, help="bet C: the stack runs only on the positions a cheap 0-layer gate over the input is unsure about; this fraction of positions (by a running entropy threshold) is skipped and carries the last kept position's output (transformer + --rope --rope-table only; 0 = off)")
    p.add_argument("--skip-momentum", type=float, default=0.99, help="EMA momentum of the gate's entropy threshold over training batches (needs --skip-frac)")
    p.add_argument("--skip-aux", type=float, default=1.0, help="weight of the cheap gate head's own next-byte loss (needs --skip-frac)")
    p.add_argument("--fused-channels", action="store_true", help="the table and match channels without their loop over orders: one embedding table per kind read once (FusedFeatureChannel; the same parameters and numbers up to summation order; with 18 orders the loop was 11 %% of a step). Recorded in the checkpoint; not saved with the weights otherwise")
    p.add_argument("--rope-table", action="store_true", help="the transformer's rotary cos and sin as buffers computed once, instead of in every layer of every step (--rope; the same numbers, and not saved with the weights). Compiled, the plain transformer spent 36 ms of a 255 ms step computing them")
    p.add_argument("--train-only", default="", help="comma list of parameter groups that stay trainable; every other parameter is frozen at its random initialisation: readout (final norm and output projection, and the --local-blocks readouts), embed (byte embedding), heads (every parameter of the memory heads), norms (all normalisation layers). Groups overlap")
    p.add_argument("--freeze-layers", default="", help="comma list of 0-based layer indices whose parameters (the layer with its norms, and the memory head placed after it) stay at their random initialisation; gradients still pass through them. The local-attention layer, the head after the stack, the final norm, the readout and the embedding are in no layer. With --train-only a parameter learns only if both allow it")
    p.add_argument("--local-blocks", type=int, default=0, help=">0: cut the layer stack into blocks of this many layers (the last may be shorter); the stream is detached between blocks, each block has its own norm and readout with its own next-byte loss, the training loss is their sum and evaluation uses the last block only. A head belongs to the block of the layer it follows; the local attention to the block of its position; the head after the stack to the last block")
    p.add_argument("--mem-feat", default="", help="DIR of per-position corpus-table features (scripts/build_corpus_memory.py --features): the table's prediction enters every GRUHybrid arm as an input embedding and an output logit bonus (text tasks)")
    p.add_argument("--mem-orders", default="8,16,32", help="comma list of the table's context orders to feed (needs --mem-feat; the directory must hold them)")
    p.add_argument("--ngram-emb", default="", help="comma list of orders, e.g. 2,3,4,6,8: hashed n-gram embedding tables learned by gradient (GRUHybrid arms): a row per order chosen by a fixed hash of the last n bytes of the window, summed, projected to the model width, added to the byte embedding")
    p.add_argument("--ngram-rows", type=int, default=262144, help="hashed rows per order (needs --ngram-emb); an order with 256 ** n <= rows is indexed directly; each order also has one reserved row for positions with fewer than n bytes of history")
    p.add_argument("--ngram-dim", type=int, default=64, help="width of a table row, projected to d_model (needs --ngram-emb)")
    p.add_argument("--ngram-sparse", action="store_true", help="sparse gradients for the n-gram table, updated by torch.optim.SparseAdam (no weight decay) instead of the main optimizer (needs --ngram-emb)")
    p.add_argument("--match-feat", default="", help="comma list of orders, e.g. 8,16,32: in-window match features (GRUHybrid arms): for each order the byte that followed the most recent earlier occurrence, inside the window, of the last n bytes, with a class by how many earlier occurrences; enters like the table features, with its own parameters; nothing is read from disk")
    return p


def check_args(p: argparse.ArgumentParser, args) -> None:
    """The checks across flags; p.error() exits with the parser's usage line."""
    if min(args.val_fixed, args.val_offset, args.eval_every, args.ckpt_every) < 0:
        p.error("--val-fixed, --val-offset, --eval-every and --ckpt-every must be >= 0")
    if args.local_blocks < 0:
        p.error("--local-blocks must be >= 0")
    try:   # the specs first, so a typo is reported as a typo whatever the arms
        if args.train_only:
            parse_train_only(args.train_only)
        if args.freeze_layers:
            parse_freeze_layers(args.freeze_layers, args.n_layers)
    except ValueError as e:
        p.error(str(e))
    if (args.train_only or args.freeze_layers or args.local_blocks) and not all(is_hybrid_arm(a) for a in args.arms.split(",")):
        p.error(f"--train-only, --freeze-layers and --local-blocks are for the GRUHybrid arms (mingru, gla, transformer, <backbone>_<key>_<pre|post>), not --arms {args.arms}")
    if (args.val_offset or args.eval_every) and not args.val_fixed:
        p.error("--val-offset and --eval-every belong to the fixed-window eval and need --val-fixed N")
    if args.val_fixed and args.task == "copy":
        p.error("--val-fixed scores corpus windows; use a text task, not copy")
    if args.mem_orders != p.get_default("mem_orders") and not args.mem_feat:
        p.error("--mem-orders belongs to the table features and needs --mem-feat DIR")
    if args.mem_feat:
        if args.task == "copy":
            p.error("--mem-feat feeds the table's features of corpus windows: it needs a text task, not copy")
        try:
            parse_orders(args.mem_orders, "--mem-orders")
        except ValueError as e:
            p.error(str(e))
    if args.match_feat:
        try:
            parse_orders(args.match_feat, "--match-feat")
        except ValueError as e:
            p.error(str(e))
    if not args.ngram_emb and (args.ngram_rows != p.get_default("ngram_rows") or args.ngram_dim != p.get_default("ngram_dim") or args.ngram_sparse):
        p.error("--ngram-rows, --ngram-dim and --ngram-sparse belong to the n-gram tables and need --ngram-emb ORDERS")
    if args.ngram_emb:
        try:
            parse_orders(args.ngram_emb, "--ngram-emb")
        except ValueError as e:
            p.error(str(e))
        if args.ngram_rows < 1 or args.ngram_dim < 1:
            p.error("--ngram-rows and --ngram-dim must be positive")
    if args.mem_feat and args.local_blocks:
        p.error("--mem-feat and --local-blocks cannot be used together: --local-blocks trains through forward_blocks, which does not take the table features")


def parse_args(argv=None):
    p = make_parser()
    args = p.parse_args(argv)
    check_args(p, args)
    return args


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_precision(args)
    name = args.run_name or f"sla_{args.task}_s{args.seed}"
    extra = {}
    xf = yf = ff = store = None
    if args.task != "copy":
        src = Path(args.corpus_file) if args.corpus_file else WIKI
        if args.mem_feat:  # first as well: features of another corpus, or of a table that saw the training bytes, are refused before any work
            store = FeatureStore(args.mem_feat, parse_orders(args.mem_orders, "--mem-orders"))
            store.check_corpus(src)
            if args.train_bytes > store.train_end:
                raise ValueError(f"--train-bytes {args.train_bytes} reaches past the training bytes of the table ({store.train_end}): "
                                 "the features of the later positions would not be the leak-free ones")
        if args.val_fixed:  # first: a short region or a bad offset fails now, not after the slow baselines below
            xf, yf = fixed_windows(load_bytes(src, max_bytes=args.val_fixed * args.seq_len + 1, offset=args.val_offset), args.seq_len, args.val_fixed)
            ff = store.fixed(args.val_offset, args.val_fixed, args.seq_len) if store is not None else None
        train = load_bytes(src, max_bytes=args.train_bytes)
        val_offset = args.train_bytes + 100_000  # absolute byte position the legacy validation bytes start at
        val = load_bytes(src, max_bytes=args.val_bytes, offset=val_offset)
        extra = {"unigram_bpb": ngram_bpb(train, val, 1), "bigram_bpb": ngram_bpb(train, val, 2)}
        print(f"{args.task} train={train.numel()} val={val.numel()} unigram={extra['unigram_bpb']:.3f} bigram={extra['bigram_bpb']:.3f}", flush=True)
    rows = []
    for arm in args.arms.split(","):
        model = make_model(arm, args, device)
        frozen = apply_freezing(model, args.train_only, args.freeze_layers)   # None unless --train-only or --freeze-layers
        if args.local_blocks:
            starts = [0, *(e + 1 for e in model.block_ends[:-1])]
            spans = " | ".join(str(s) if s == e else f"{s}-{e}" for s, e in zip(starts, model.block_ends))
            print(f"local-blocks {args.local_blocks}: {model.n_blocks} blocks (layers {spans}), {model.n_blocks - 1} extra readouts, +{model.aux_scalars:,} parameters; "
                  f"the loss is the sum of the block losses and evaluation uses the last block only", flush=True)
        fwd = make_train_fwd(model, args)   # the training step runs this (compiled with --compile); everything else the plain module
        opt, sched = make_opt_sched(model, args)
        sopt = make_sparse_optimizer(model, args)  # with --ngram-sparse: the n-gram table's own optimizer, on the same schedule
        ssched = torch.optim.lr_scheduler.LambdaLR(sopt, lr_lambda(args.steps, args.warmup, args.cosine)) if sopt is not None else None
        torch.manual_seed(args.seed)
        timed = yardstick_on(args)
        clock = SplitClock(sync=(lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None))
        boundary = clock.boundary if timed else nullcontext  # every eval and save runs inside one; a no-op unless a yardstick flag is set
        meter = make_meter(args, device)
        curve_fixed = []
        fixed_every = args.eval_every or max(1, args.steps // 6)
        if timed and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter(); curve = []
        if meter:
            meter.start()
        if timed:
            clock.start()
        try:
            for step in range(args.steps):
                if args.task == "copy":
                    x, y, _ = _copy(args, args.seed * 100_000 + step)
                else:
                    x, y = batch_windows(train, seq_len=args.seq_len, batch=args.batch, seed=args.seed * 100_000 + step)
                x, y = x.to(device), y.to(device)
                kw = {}
                if store is not None:  # the features of the byte each input predicts: the table features of the windows' positions + 1
                    kw = _features_to(device, store.windows(batch_starts(train.numel(), args.seq_len, args.batch, args.seed * 100_000 + step), args.seq_len))
                loss, parts = train_loss(model, x, y, args, device, fwd, kw)
                opt.zero_grad()
                if sopt is not None:
                    sopt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
                if sopt is not None:
                    sopt.step(); ssched.step()
                if meter and (step + 1) % CHECK_EVERY == 0:
                    meter.check()
                if (step + 1) % max(1, args.steps // 6) == 0:
                    with boundary():
                        row = {"step": step + 1, "train_bpb": float(parts[-1]) / math.log(2)}   # the model's own output (== loss unless --local-blocks)
                        if len(parts) > 1:   # --local-blocks: every block's own training loss at this step, the model's output last
                            row["block_bpb"] = [float(l) / math.log(2) for l in parts]
                        row.update(eval_copy(model, args, 2, 10_000 + step, device) if args.task == "copy"
                                   else eval_wiki(model, val, args, 4, 10_000 + step, device, store, val_offset))
                        curve.append(row); print(f"  {arm} {row}", flush=True)
                if args.val_fixed and (step + 1) % fixed_every == 0 and step + 1 < args.steps:  # the last step is scored once, below
                    with boundary():
                        curve_fixed.append(fixed_row(arm, model, xf, yf, args, device, step + 1, clock.train_seconds, ff))
                if args.ckpt_every and (step + 1) % args.ckpt_every == 0:
                    with boundary():
                        save_ckpt(RESULTS / "ckpt" / f"{name}_{arm}.step{step + 1}.pt", arm, args, step + 1, model, opt, sched, sopt)
                        print(f"saved {name}_{arm}.step{step + 1}.pt", flush=True)
            with boundary():  # closes train_seconds; everything below is eval_seconds
                rep = eval_copy(model, args, 8, 20_000, device) if args.task == "copy" else eval_wiki(model, val, args, 16, 20_000, device, store, val_offset)
                if args.val_fixed:
                    curve_fixed.append(fixed_row(arm, model, xf, yf, args, device, args.steps, clock.train_seconds, ff))
                if args.save:
                    (RESULTS / "ckpt").mkdir(exist_ok=True)
                    torch.save({"arm": arm, "args": recorded_args(args), "state": model.state_dict()}, RESULTS / "ckpt" / f"{name}_{arm}.pt")
        finally:
            if meter:
                meter.stop()
        rep.update(arm=arm, seconds=time.perf_counter() - t0, trainable_scalars=numel(model), curve=curve,
                   flops_per_token_recurrent=model.flops_per_token_recurrent() if hasattr(model, "flops_per_token_recurrent") else model.flops_per_step(1))
        if frozen:   # train_only, frozen_layers, trainable_by_group, total_scalars; trainable_scalars above is already the trainable count
            rep.update({k: v for k, v in frozen.items() if k != "trainable_scalars"})
        if args.local_blocks:   # the block readouts are training only: the model that gets scored has added_scalars fewer parameters than the model trained
            rep.update(local_blocks=args.local_blocks, n_blocks=model.n_blocks, added_scalars=model.aux_scalars)
        if args.match_feat:
            rep["match_feat"] = {"orders": list(model.match_orders), "added_params": model.match_params()}
        if getattr(model, "skip_frac", 0.0):
            rep["skip"] = {"frac": model.skip_frac, "momentum": model.skip_momentum, "aux": args.skip_aux, "tau_nats": float(model.skip_tau), "added_params": model.skip_params()}
        if args.ngram_emb:
            rep["ngram_emb"] = {"orders": list(model.ngram.orders), "rows": model.ngram.rows, "dim": model.ngram.dim, "sparse": model.ngram.sparse, "added_params": model.ngram_params()}
        if store is not None:
            rep["mem_feat"] = {"dir": args.mem_feat, "orders": list(store.orders), "added_params": model.table_params(), "table_digest": store.info.get("table_digest")}
        metric = {k: round(v, 3) for k, v in rep.items() if k.startswith(("copy", "val"))}
        print(f"{arm:10} {metric} scalars={rep['trainable_scalars']} rflops={rep['flops_per_token_recurrent']} sec={rep['seconds']:.0f}", flush=True)
        if timed:
            rep.update(train_seconds=clock.train_seconds, eval_seconds=clock.eval_seconds, bytes_seen=args.steps * args.batch * args.seq_len,
                       peak_vram_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0)
            if args.val_fixed:
                rep.update(curve_fixed=curve_fixed, val_fixed={"offset": args.val_offset, "n_windows": args.val_fixed,
                                                               "bpb": curve_fixed[-1]["bpb"], "bpb_warm": curve_fixed[-1]["bpb_warm"]})
            if meter:
                rep.update(meter.summary(), gpu_uuid=meter.gpu_uuid)
            print(f"{arm:10} yardstick train_s={rep['train_seconds']:.1f} eval_s={rep['eval_seconds']:.1f} peak_vram_gib={rep['peak_vram_bytes'] / 2**30:.2f}"
                  + (f" joules={rep['joules']:.0f} contended={rep['contended']}" if meter else ""), flush=True)
        rows.append(rep)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps({"run_name": name, **recorded_args(args), **extra, "arms": rows}, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
