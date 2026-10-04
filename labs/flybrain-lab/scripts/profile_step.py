#!/usr/bin/env python3
"""Where does one training step go? Profile scripts/run_sparsela.py's step for the flags it is given (spike, throwaway).

Same flags as the runner (its parser, its model, optimizer, schedule, autocast and batches), plus --warmup-steps and --measure-steps.
For every arm in --arms it reports, as one markdown report and one JSON:
  * seconds per step with no instrumentation (the runner's own loop, one sync at the end), and the same step split into data,
    forward, backward, gradient clipping and optimizer step with a device sync at every boundary;
  * an instrumented forward, synchronised around each part (embedding; per layer the recurrence scan and the rest of the layer;
    per memory head the LayerNorm, key conv, value projection, address selection, recompute of the selected activations,
    write/read, normaliser and gate, output projection; local attention; decoder; loss). Always eager, even with --compile;
  * the top operators by self device time (self CPU time on CPU) over forward + backward, from torch.profiler;
  * peak memory.
Timings from a CPU are for testing the tool only: they say nothing about the GPU.

  python scripts/profile_step.py <the runner's flags> --device cuda:0 --warmup-steps 5 --measure-steps 20 --json-out results/profile_x.json
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.data.text import batch_windows, load_bytes
from flybrain.model import sparsela
from flybrain.model.hybrid import GLABackbone, GRUHybrid, MinGRU, TransformerBackbone
from flybrain.model.sparsela import Block
from scripts import run_sparsela as rs

HEAD_PARTS = ("ln", "conv", "val", "cast", "select", "recompute", "read", "norm_gate", "out", "other")
HEAD_PART_TEXT = {"ln": "LayerNorm", "conv": "key conv", "val": "value projection", "cast": "fp32 pin: casts, glue",
                  "select": "address: no-grad matmul + top-k", "recompute": "address: recompute selected", "read": "write/read (rest of sparse_memory_read)",
                  "norm_gate": "normaliser + gate", "out": "output projection", "other": "other (padding, residual)"}


# ---------------------------------------------------------------------------------------------------------------------
# synchronised section timer
# ---------------------------------------------------------------------------------------------------------------------

class SectionTimer:
    """Exclusive wall time per named section. enter() and exit() sync the device, so a section holds its own work and nothing queued
    before it; a section's time excludes the sections nested inside it, so the times of all sections add up to the time spent inside any."""

    def __init__(self, sync=lambda: None):
        self.sync = sync
        self.reset()

    def reset(self) -> None:
        self.acc: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.stack: list[str] = []
        self.mark = 0.0

    def enter(self, name: str) -> None:
        self.sync()
        now = time.perf_counter()
        if self.stack:
            self.acc[self.stack[-1]] = self.acc.get(self.stack[-1], 0.0) + (now - self.mark)
        self.stack.append(name)
        self.calls[name] = self.calls.get(name, 0) + 1
        self.mark = now

    def exit(self) -> None:
        self.sync()
        now = time.perf_counter()
        name = self.stack.pop()
        self.acc[name] = self.acc.get(name, 0.0) + (now - self.mark)
        self.mark = now

    @contextmanager
    def section(self, name: str):
        self.enter(name)
        try:
            yield
        finally:
            self.exit()


class Instrument:
    """Wrap the parts of a model in SectionTimer sections for the length of a `with` block; every change is undone on exit.
    Sections: model (the whole call: its own time is the glue), embed, layer{i} (the rest of the layer) with layer{i}/scan (or /attn, /ffn, /ln for a transformer), local{j}, final_ln,
    decoder, and for every memory head <head>/<part> with the parts of HEAD_PARTS (<head> alone is the head's own glue). A head is
    head_top, head_pre or head_L{i} (after layer i); a Block of any other model is block:<path>."""

    def __init__(self, model: nn.Module, timer: SectionTimer):
        self.model, self.t = model, timer
        self._undo: list = []
        self._head_stack: list[str] = []
        self.heads: list[str] = []

    def __enter__(self):
        m = self.model
        try:
            self.module(m, "model")   # the whole call: what no part below covers (residual adds, loops over layers) is its own time
            if isinstance(m, GRUHybrid):
                self._hybrid(m)
            else:  # any other arm: one section per top-level child (per element of a ModuleList), and every Block as a head
                for name, child in m.named_children():
                    if isinstance(child, nn.ModuleList):
                        for i, c in enumerate(child):
                            self.module(c, f"{name}{i}")
                    else:
                        self.module(child, name)
                for path, blk in m.named_modules():
                    if isinstance(blk, Block):
                        self._head(blk, f"block:{path}")
            for attr, part in (("sparse_memory_read", "read"), ("_normalize_gate", "norm_gate")):   # module functions, named after the running head
                self.method(sparsela, attr, (lambda part=part: f"{self._head_stack[-1] if self._head_stack else 'other'}/{part}"))
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        for undo in reversed(self._undo):
            undo()
        self._undo.clear()
        return False

    # -- primitives
    def module(self, mod: nn.Module, name: str) -> None:
        h1 = mod.register_forward_pre_hook(lambda mm, a: self.t.enter(name))
        h2 = mod.register_forward_hook(lambda mm, a, o: self.t.exit())
        self._undo += [h1.remove, h2.remove]

    def method(self, obj, attr: str, name, enter_cb=None, exit_cb=None) -> None:
        """Replace obj.attr by a timed wrapper. `name` is a string, or a function returning one (called at every call)."""
        orig = getattr(obj, attr)
        had, prev = attr in obj.__dict__, obj.__dict__.get(attr)

        def wrapped(*a, **k):
            self.t.enter(name() if callable(name) else name)
            if enter_cb:
                enter_cb()
            try:
                return orig(*a, **k)
            finally:
                self.t.exit()
                if exit_cb:
                    exit_cb()

        setattr(obj, attr, wrapped)
        self._undo.append((lambda: setattr(obj, attr, prev)) if had else (lambda: delattr(obj, attr)))

    # -- the parts
    def _hybrid(self, m: GRUHybrid) -> None:
        self.module(m.embed, "embed")
        core = getattr(m, "core", None)
        if isinstance(core, MinGRU):
            for i, layer in enumerate(core.layers, 1):
                self.module(layer, f"layer{i}")
                self.method(layer, "recurrence", f"layer{i}/scan")
        elif isinstance(core, GLABackbone):
            for i, blk in enumerate(core.blocks, 1):
                self.module(blk, f"layer{i}")
        elif isinstance(core, TransformerBackbone):
            for i in range(len(core.attn)):
                self.module(core.attn[i], f"layer{i + 1}/attn")
                self.module(core.ff[i], f"layer{i + 1}/ffn")
                self.module(core.ln1[i], f"layer{i + 1}/ln")
                self.module(core.ln2[i], f"layer{i + 1}/ln")
        elif hasattr(m, "gru"):
            self.module(m.gru, "layer1")
        if core is not None:
            self.module(core.ln, "final_ln")
        if getattr(m, "local_attn", 0):
            self.module(m.local, "local1")
            for j, la in enumerate(m.local_more, 2):
                self.module(la, f"local{j}")
        self.module(m.decoder, "decoder")
        if m.where != "none":
            self._head(m.mem, "head_pre" if m.where == "pre" else "head_top")
        for k, blk in enumerate(m.mems):
            self._head(blk, f"head_L{(k + 1) * m.mem_every}")

    def _head(self, blk: Block, label: str) -> None:
        self.heads.append(label)
        self.method(blk, "memory", label, lambda: self._head_stack.append(label), lambda: self._head_stack.pop())
        self.module(blk.ln1, f"{label}/ln")                 # called outside memory(), by the model
        if getattr(blk, "conv", None) is not None:
            self.module(blk.conv, f"{label}/conv")
        self.module(blk.val, f"{label}/val")
        for o in blk.outs:
            self.module(o, f"{label}/out")
        self.method(blk, "_one_head", f"{label}/cast")      # the fp32 pin: casts and the autocast switch; the read is nested in it
        for code in blk.codes:
            if hasattr(code, "select"):
                self.method(code, "select", f"{label}/select")
                self.method(code, "recompute", f"{label}/recompute")


# ---------------------------------------------------------------------------------------------------------------------
# the step, as the runner runs it
# ---------------------------------------------------------------------------------------------------------------------

class Rig:
    """One arm's model, optimizer, schedule and batches, built the way scripts/run_sparsela.py main() builds them."""

    def __init__(self, arm: str, args, device: torch.device, batch_fn):
        self.arm, self.args, self.device, self.batch_fn = arm, args, device, batch_fn
        self.model = rs.make_model(arm, args, device)
        self.fwd = torch.compile(self.model) if args.compile else self.model
        self.opt, self.sched = rs.make_opt_sched(self.model, args)
        torch.manual_seed(args.seed)
        self.cuda = device.type == "cuda"

    def sync(self) -> None:
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def batch(self, step: int):
        x, y = self.batch_fn(step)
        return x.to(self.device), y.to(self.device)

    def loss(self, x, y):
        with rs._amp(self.args, self.device):
            logits, _ = self.fwd(x)
            return F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))

    def clip(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    def update(self) -> None:
        self.opt.step(); self.sched.step()

    def full_step(self, step: int):
        x, y = self.batch(step)
        loss = self.loss(x, y)
        self.opt.zero_grad(); loss.backward()
        self.clip(); self.update()
        return loss


def _stats(xs: list[float]) -> dict:
    return {"mean": statistics.fmean(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs),
            "stdev": statistics.pstdev(xs) if len(xs) > 1 else 0.0, "n": len(xs)}


def time_plain(rig: Rig, first: int, n: int) -> dict:
    """The runner's loop body n times, with one sync before and one after: seconds per step as the run sees it."""
    meter = rs.make_meter(rig.args, rig.device)
    rig.sync()
    if meter:
        meter.start()
    t0 = time.perf_counter()
    try:
        for i in range(n):
            loss = rig.full_step(first + i)
        rig.sync()
        dt = time.perf_counter() - t0
    finally:
        if meter:
            meter.stop()
    out = {"seconds_per_step": dt / n, "steps": n, "final_loss": float(loss.detach())}
    if meter:
        s = meter.summary()
        out.update(mean_power_w=s["mean_power_w"], power_limit_w=s["power_limit_w"], max_temp_c=s["max_temp_c"], contended=s["contended"],
                   joules_per_step=s["mean_power_w"] * dt / n)
    return out


def time_split(rig: Rig, first: int, n: int) -> dict:
    """The same step with a sync at every boundary: data, forward, backward, clip, optimizer step. On CUDA also the memory the forward
    leaves allocated (the activations the backward will use)."""
    rows = {k: [] for k in ("data", "forward", "backward", "clip", "optimizer")}
    act = []
    for i in range(n):
        rig.sync(); t0 = time.perf_counter()
        x, y = rig.batch(first + i)
        rig.sync(); t1 = time.perf_counter()
        m0 = torch.cuda.memory_allocated(rig.device) if rig.cuda else 0
        loss = rig.loss(x, y)
        rig.sync(); t2 = time.perf_counter()
        if rig.cuda:
            act.append(torch.cuda.memory_allocated(rig.device) - m0)
        rig.opt.zero_grad(); loss.backward()
        rig.sync(); t3 = time.perf_counter()
        rig.clip()
        rig.sync(); t4 = time.perf_counter()
        rig.update()
        rig.sync(); t5 = time.perf_counter()
        for k, v in zip(rows, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
            rows[k].append(v)
    out = {k: _stats(v) for k, v in rows.items()}
    out["total_mean"] = sum(out[k]["mean"] for k in rows)
    if act:
        out["activation_bytes_after_forward"] = int(statistics.fmean(act))
    return out


def time_instrumented(rig: Rig, first: int, n: int) -> dict:
    """The forward and the loss with a sync around every part (eager: the plain module, never the compiled one), n times; each one is
    paired with the same forward uninstrumented (eager, synced at its two ends) so the cost of the instrumentation is a like-for-like
    ratio. Returns the mean time of the whole, the mean of every section, and what no section covers."""
    timer = SectionTimer(rig.sync)
    totals: list[float] = []
    plain: list[float] = []
    per: list[dict] = []
    calls: dict[str, int] = {}
    heads: list[str] = []
    for i in range(n):
        x, y = rig.batch(first + i)
        # [0], not `logits, _ = ...`: the hidden-state view would keep the whole autograd graph alive until the next call rebinds it,
        # and freeing that graph would land inside the next clock window
        with rs._amp(rig.args, rig.device):       # a fresh autocast context for each, entered before the clock starts: both pay for the weight casts
            rig.sync(); t0 = time.perf_counter()
            logits = rig.model(x)[0]
            loss = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))
            rig.sync(); plain.append(time.perf_counter() - t0)
        del logits, loss
        timer.reset()
        with Instrument(rig.model, timer) as ins, rs._amp(rig.args, rig.device):
            rig.sync(); t0 = time.perf_counter()
            logits = rig.model(x)[0]
            with timer.section("loss"):
                loss = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))
            rig.sync(); totals.append(time.perf_counter() - t0)
        per.append(dict(timer.acc)); calls = dict(timer.calls); heads = list(ins.heads)
        del logits, loss
    names = sorted({k for d in per for k in d})
    mean = {k: statistics.fmean(d.get(k, 0.0) for d in per) for k in names}
    total = statistics.fmean(totals)
    parts_sum = sum(mean.values())
    return {"total_ms": total * 1e3, "parts_sum_ms": parts_sum * 1e3, "unattributed_ms": (total - parts_sum) * 1e3,
            "plain_forward_ms": statistics.fmean(plain) * 1e3,
            "parts": {k: {"ms": mean[k] * 1e3, "pct": 100 * mean[k] / total, "calls": calls.get(k, 0)} for k in names},
            "heads": heads, "steps": n}


def _device_time_us(e) -> float:
    for a in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(e, a):
            return float(getattr(e, a))
    return 0.0


def _is(e, kind: str) -> bool:
    return getattr(e.device_type, "name", str(e.device_type)) == kind


def profile_ops(rig: Rig, first: int, n: int, top: int = 25) -> dict:
    """The top operators over n forward + backward passes (no optimizer step): self device time on CUDA, self CPU time on CPU."""
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if rig.cuda else [])
    rig.sync()
    with profile(activities=acts) as prof:
        for i in range(n):
            x, y = rig.batch(first + i)
            loss = rig.loss(x, y)
            rig.opt.zero_grad(); loss.backward()
        rig.sync()
    events = prof.key_averages()
    host = [e for e in events if _is(e, "CPU")]          # the operators; on CUDA their self device time is the kernels they launched
    key = _device_time_us if rig.cuda else (lambda e: float(e.self_cpu_time_total))
    total = sum(key(e) for e in host) or 1.0
    out = {"by": "self device time" if rig.cuda else "self CPU time", "steps": n, "total_self_ms_per_step": total / 1e3 / n,
           "ops": [{"op": e.key, "calls_per_step": e.count / n, "self_ms_per_step": key(e) / 1e3 / n, "pct": 100 * key(e) / total}
                   for e in sorted(host, key=key, reverse=True)[:top]]}
    if rig.cuda:  # the kernels themselves: a second view of the same time
        kern = [e for e in events if _is(e, "CUDA")]
        ktotal = sum(_device_time_us(e) for e in kern) or 1.0
        out["kernels"] = [{"kernel": e.key[:110], "calls_per_step": e.count / n, "ms_per_step": _device_time_us(e) / 1e3 / n,
                           "pct": 100 * _device_time_us(e) / ktotal} for e in sorted(kern, key=_device_time_us, reverse=True)[:15]]
    return out


def memory_report(rig: Rig) -> dict:
    if rig.cuda:
        d = rig.device
        return {"kind": "cuda", "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(d)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(d))}
    return {"kind": "cpu: peak RSS of the whole process, corpus included", "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def md_table(head: list[str], rows: list[list], left: tuple = (0,)) -> str:
    """A markdown table; the columns in `left` are left-aligned text, the others right-aligned numbers."""
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(":---" if i in left else "---:" for i in range(len(head))) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def head_text(label: str) -> str:
    if label == "head_top":
        return "top head (after the final LN)"
    if label == "head_pre":
        return "pre head (on the embedding)"
    return f"head after layer {label[6:]}" if label.startswith("head_L") else label


def summarise(fb: dict) -> dict:
    """Group the sections of an instrumented forward by kind, layer and head. Every section lands in exactly one group."""
    parts, heads = fb["parts"], fb["heads"]
    claimed: set[str] = set()

    def ms(k: str) -> float:
        claimed.add(k)
        return parts.get(k, {}).get("ms", 0.0)

    layers = sorted({int(k[5:].split("/")[0]) for k in parts if k.startswith("layer")})
    per_layer = {i: {"scan": ms(f"layer{i}/scan"), "rest": ms(f"layer{i}"), "attn": ms(f"layer{i}/attn"), "ffn": ms(f"layer{i}/ffn"), "ln": ms(f"layer{i}/ln")}
                 for i in layers}
    per_head = {h: {**{p: ms(f"{h}/{p}") for p in HEAD_PARTS if p != "other"}, "other": ms(h)} for h in heads}
    head_total = {p: sum(v[p] for v in per_head.values()) for p in HEAD_PARTS}
    pl = lambda k: sum(v[k] for v in per_layer.values())
    groups = {"embedding": ms("embed"), "recurrence scan (all layers)": pl("scan"), "layers, rest (LN, projections, FFN)": pl("rest"),
              "transformer attention": pl("attn"), "transformer FFN": pl("ffn"), "transformer LayerNorm": pl("ln"),
              "local attention": sum(ms(k) for k in parts if k.startswith("local")), "memory heads (all)": sum(head_total.values()),
              "final LayerNorm": ms("final_ln"), "decoder": ms("decoder"), "loss": ms("loss"),
              "model glue (residual adds, loops over layers)": ms("model")}
    groups["other modules"] = sum(v["ms"] for k, v in parts.items() if k not in claimed)
    groups["unattributed (outside the model call and the loss)"] = fb["unattributed_ms"]
    return {"per_layer": per_layer, "per_head": per_head, "head_total": head_total, "groups": {k: v for k, v in groups.items() if abs(v) > 1e-9}}


def render(result: dict) -> str:
    a, env, t = result["arm"], result["env"], result["timing"]
    out = [f"## Arm `{a['arm']}` ({a['trainable_scalars']:,} trainable parameters)\n"]
    sw = result["speed_switches"]
    out.append(f"Device `{env['device']}`{' (' + env['device_name'] + ')' if env.get('device_name') else ''}, torch {env['torch']}; "
               f"autocast {'bf16' if result['flags'].get('amp') else 'off'}; speed switches: {', '.join(sw) if sw else 'none'}; "
               f"batch {result['flags']['batch']} x {result['flags']['seq_len']}; warmup {env['warmup_steps']}, measured {env['measure_steps']} steps.\n")
    pl, spl = t["plain"], t["split"]
    rows = [["**plain step, no instrumentation**", f"{pl['seconds_per_step']:.4f}", f"{pl['steps']} steps, one sync at the end, data draw included; final loss {pl['final_loss']:.3f}"]]
    for k, text in (("data", "batch draw on CPU + copy to the device"), ("forward", "forward + loss"), ("backward", "zero_grad + backward"),
                    ("clip", "clip_grad_norm_"), ("optimizer", "optimizer step + schedule")):
        rows.append([f"split: {k}", f"{spl[k]['mean']:.4f}", f"{text}; median {spl[k]['median']:.4f}, min {spl[k]['min']:.4f}, max {spl[k]['max']:.4f}"])
    rows.append(["split: sum", f"{spl['total_mean']:.4f}", f"sync at every boundary: {spl['total_mean'] / pl['seconds_per_step']:.2f} x the plain step"])
    if "joules_per_step" in pl:
        rows.append(["energy per step (J)", f"{pl['joules_per_step']:.1f}", f"mean power {pl['mean_power_w']:.0f} W (limit {pl['power_limit_w']:.0f} W), max {pl['max_temp_c']:.0f} C, contended {pl['contended']}"])
    out.append("### Seconds per step\n")
    out.append(md_table(["measure", "seconds", "note"], rows, left=(0, 2)))
    out.append("\nWarmup steps (s): " + ", ".join(f"{w:.3f}" for w in t["warmup_seconds"]) + (" (the first includes torch.compile)" if result["flags"].get("compile") else "") + "\n")

    fb = result["forward_breakdown"]; sm = fb["summary"]
    out.append("### Forward breakdown (instrumented: synced around every part, eager)\n")
    out.append(f"Instrumented forward + loss {fb['total_ms']:.1f} ms, the parts add up to {fb['parts_sum_ms']:.1f} ms. The same forward + loss without instrumentation "
               f"(eager, synced at its two ends) took {fb['plain_forward_ms']:.1f} ms, so the instrumented one is {fb['total_ms'] / fb['plain_forward_ms']:.2f} x it "
               f"(the syncs remove the overlap of launches with work); the forward in the split run took {spl['forward']['mean'] * 1e3:.1f} ms.\n")
    out.append(md_table(["part", "ms", "% of forward"], [[k, f"{v:.2f}", f"{100 * v / fb['total_ms']:.1f}"] for k, v in sorted(sm["groups"].items(), key=lambda kv: -kv[1])]))
    if sm["per_head"]:
        out.append("\nMemory heads, ms per forward, by part:\n")
        rows = [[head_text(h)] + [f"{v[p]:.2f}" for p in HEAD_PARTS] + [f"{sum(v.values()):.2f}"] for h, v in sm["per_head"].items()]
        rows.append(["**all heads**"] + [f"{sm['head_total'][p]:.2f}" for p in HEAD_PARTS] + [f"{sum(sm['head_total'].values()):.2f}"])
        out.append(md_table(["head"] + [HEAD_PART_TEXT[p] for p in HEAD_PARTS] + ["total"], rows))
    lay = sm["per_layer"]
    if any(v["scan"] or v["rest"] for v in lay.values()):
        out.append("\nLayers, ms per forward:\n")
        out.append(md_table(["layer", "recurrence scan", "rest of the layer"], [[i, f"{v['scan']:.2f}", f"{v['rest']:.2f}"] for i, v in lay.items()]))
    if any(v["attn"] for v in lay.values()):
        out.append("\nLayers, ms per forward:\n")
        out.append(md_table(["layer", "attention", "FFN", "LayerNorm"], [[i, f"{v['attn']:.2f}", f"{v['ffn']:.2f}", f"{v['ln']:.2f}"] for i, v in lay.items()]))
    ops = result["operators"]
    out.append(f"\n### Top operators by {ops['by']}, forward + backward, per step (all operators: {ops['total_self_ms_per_step']:.1f} ms)\n")
    if ops.get("error"):
        out.append(f"the profiler failed: {ops['error']}\n")
    else:
        out.append(md_table(["operator", "calls", "self ms", "% of total"], [[f"`{o['op']}`", f"{o['calls_per_step']:.0f}", f"{o['self_ms_per_step']:.2f}", f"{o['pct']:.1f}"] for o in ops["ops"]]))
        if ops.get("kernels"):
            out.append("\nTop kernels per step:\n")
            out.append(md_table(["kernel", "calls", "ms", "% of kernel time"], [[f"`{k['kernel']}`", f"{k['calls_per_step']:.0f}", f"{k['ms_per_step']:.2f}", f"{k['pct']:.1f}"] for k in ops["kernels"]]))
    mem = result["memory"]
    if mem["kind"] == "cuda":
        act = f"; the forward leaves {spl['activation_bytes_after_forward'] / 2**30:.2f} GiB of activations" if "activation_bytes_after_forward" in spl else ""
        out.append(f"\n### Memory\n\nPeak allocated {mem['peak_allocated_bytes'] / 2**30:.2f} GiB, peak reserved {mem['peak_reserved_bytes'] / 2**30:.2f} GiB over the timed training steps{act}.")
    else:
        out.append(f"\n### Memory\n\n{mem['kind']}: {mem['peak_rss_bytes'] / 2**30:.2f} GiB.")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------------------------------

def make_batches(args):
    """batch_fn(step) -> (x, y) on the CPU, drawn the way the runner draws them: the same corpus slice, the same seeds."""
    if args.task == "copy":
        return lambda step: rs._copy(args, args.seed * 100_000 + step)[:2]
    train = load_bytes(Path(args.corpus_file) if args.corpus_file else rs.WIKI, max_bytes=args.train_bytes)
    return lambda step: batch_windows(train, seq_len=args.seq_len, batch=args.batch, seed=args.seed * 100_000 + step)


def env_info(args, device) -> dict:
    env = {"device": str(device), "torch": torch.__version__, "python": platform.python_version(), "warmup_steps": args.warmup_steps,
           "measure_steps": args.measure_steps, "profile_steps": args.profile_steps, "argv": sys.argv,
           "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}
    if device.type == "cuda":
        env["device_name"] = torch.cuda.get_device_name(device)
        env["cuda"] = torch.version.cuda
        try:  # the card's power limit and top clock as nvidia-smi reports them: a note on what the timings mean, not a measurement
            q = subprocess.run(["nvidia-smi", "--query-gpu=power.limit,clocks.max.sm", "--format=csv,noheader,nounits", "-i", rs.device_uuid(device)],
                               capture_output=True, text=True, timeout=10, check=True).stdout.strip().split(",")
            env["power_limit_w"], env["max_sm_clock_mhz"] = float(q[0]), float(q[1])
        except Exception:
            pass
    return env


def profile_arm(arm: str, args, device, batch_fn) -> dict:
    rig = Rig(arm, args, device, batch_fn)
    w, m = args.warmup_steps, args.measure_steps
    warm = []
    for i in range(w):  # the first step carries lazy init (and torch.compile); every one is timed alone
        rig.sync(); t0 = time.perf_counter()
        rig.full_step(i)
        rig.sync(); warm.append(time.perf_counter() - t0)
    if rig.cuda:
        torch.cuda.reset_peak_memory_stats(device)
    plain = time_plain(rig, w, m)
    split = time_split(rig, w + m, m)
    memory = memory_report(rig)
    inst = time_instrumented(rig, w + 2 * m, max(1, min(m, 10)))
    inst["summary"] = summarise(inst)
    try:
        ops = profile_ops(rig, w + 3 * m, args.profile_steps)
    except Exception as e:  # the profiler API differs between builds; the rest of the report must survive it
        ops = {"by": "n/a", "steps": 0, "total_self_ms_per_step": 0.0, "ops": [], "error": f"{type(e).__name__}: {e}"}
    flags = {k: v for k, v in vars(args).items() if k in ("amp", "compile", "batch", "seq_len", "d_model", "n_layers", "d_ff", "n_key", "top_k", "chunk", "task")}
    switches = [f"--{k.replace('_', '-')}" for k in rs.SPEED_FLAGS if getattr(args, k) != rs.SPEED_DEFAULTS[k]]
    return {"arm": {"arm": arm, "trainable_scalars": rs.numel(rig.model)}, "env": env_info(args, device), "flags": flags, "speed_switches": switches,
            "timing": {"warmup_seconds": warm, "plain": plain, "split": split}, "forward_breakdown": inst, "operators": ops, "memory": memory}


def make_parser() -> argparse.ArgumentParser:
    p = rs.make_parser()
    p.description, p.formatter_class = __doc__, argparse.RawDescriptionHelpFormatter
    p.add_argument("--warmup-steps", type=int, default=5, help="training steps run, and timed one by one, before anything is measured (lazy init, torch.compile)")
    p.add_argument("--measure-steps", type=int, default=20, help="steps in each timed block: the plain loop, and the split into data / forward / backward / clip / optimizer")
    p.add_argument("--profile-steps", type=int, default=3, help="forward + backward passes under torch.profiler for the operator table")
    p.add_argument("--json-out", default="", help="where to write the JSON (default results/profile_step_<run name or arms>.json)")
    return p


def main(argv=None):
    p = make_parser()
    args = p.parse_args(argv)
    rs.check_args(p, args)
    if args.warmup_steps < 0 or args.measure_steps < 1 or args.profile_steps < 1:
        p.error("--warmup-steps must be >= 0; --measure-steps and --profile-steps must be >= 1")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.device.startswith("cpu"):
        print(f"note: {args.device} is not available, profiling on the CPU (its timings say nothing about a GPU)", file=sys.stderr)
    rs.set_precision(args)
    ignored = [f"--{k.replace('_', '-')}" for k in ("save", "val_fixed", "val_offset", "eval_every", "ckpt_every") if getattr(args, k)]
    if ignored:
        print(f"note: the profiler ignores {' '.join(ignored)}", file=sys.stderr)
    batch_fn = make_batches(args)
    results = [profile_arm(arm, args, device, batch_fn) for arm in args.arms.split(",")]
    print("# Step profile\n\n" + "\n".join(render(r) for r in results))
    out = Path(args.json_out) if args.json_out else rs.RESULTS / f"profile_step_{args.run_name or args.arms.replace(',', '+')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arms": results}, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    return results


if __name__ == "__main__":
    main()
