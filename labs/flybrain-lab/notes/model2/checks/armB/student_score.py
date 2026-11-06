#!/usr/bin/env python
"""Arm B (2026-11-06): score a Qwen3.5 student (a Base checkpoint, optionally with a LoRA adapter) on the held-out windows under the same
prompts as the 27B runs: cold (the first half alone), memory (the span retriever's two chunks) or a chooser's choices file (--choices,
--headers). The halves are tokenised with the teacher GGUF's tokeniser (identical ids, verified on 1,694 halves), the prompt is BOS +
memory + first half, left-truncated to n_ctx, and the second half's tokens are scored. Writes tokens.jsonl in Arm A's format (set, w,
date, chunks, bits, ctx, tgt, lp), qwen_memory.json and a one-line summary. Numbers and ids only.
usage: student_score.py --model DIR [--adapter DIR] --cond cold|mem|choices [--choices F --headers] --out DIR [--limit N]"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import Retriever, Tokenizer, load_choices, sep_bytes, windows

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--adapter", default=""); ap.add_argument("--cond", default="mem", choices=["cold", "mem", "choices"])
ap.add_argument("--choices", default=""); ap.add_argument("--headers", action="store_true"); ap.add_argument("--sep", default="\\n\\n"); ap.add_argument("--order", default="top-first")
ap.add_argument("--out", required=True); ap.add_argument("--n-ctx", type=int, default=6144); ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--kyle-every", type=int, default=1); ap.add_argument("--dlg-every", type=int, default=4); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--label", default="")
a = ap.parse_args(); os.umask(0o077); os.makedirs(a.out, exist_ok=True); SEP = sep_bytes(a.sep); T = a.T
tok = Tokenizer(); BOS = tok.bos
r = Retriever() if a.cond != "cold" else None
choices = chooser = None
if a.cond == "choices": choices, chooser = load_choices(a.choices, r)
from transformers import Qwen3_5ForCausalLM
m = Qwen3_5ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda"); m.eval()
if a.adapter:
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, a.adapter); m = m.merge_and_unload(); m.eval()
print(f"student {a.model}{' + ' + a.adapter if a.adapter else ''}, condition {a.cond}" + (f" ({a.label})" if a.label else ""), flush=True)

@torch.no_grad()
def score(prefix: bytes, target: bytes):
    tp = [BOS] + tok(prefix); tt = tok(target); toks = (tp + tt)[-a.n_ctx:]; k = len(tt)
    x = torch.tensor([toks], device="cuda"); logits = m(input_ids=x).logits[0, len(toks) - k - 1:len(toks) - 1].float()
    lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(toks[len(toks) - k:], device="cuda")[:, None])[:, 0].cpu().numpy().astype(np.float64)
    return float(-lp.sum() / math.log(2)), np.asarray(toks[len(toks) - k:]), lp

summary = {}; rec = open(os.path.join(a.out, "tokens.jsonl"), "w"); res = {"kyle": [], "dlg": []}; ntok = {"kyle": 0, "dlg": 0}; mem_used = {"kyle": 0, "dlg": 0}; mem_bytes = {"kyle": 0, "dlg": 0}
n = {"kyle": 0, "dlg": 0}; t0 = time.time()
for name, w, date, first, second, W in windows(T, a.kyle_every, a.dlg_every):
    if a.limit and n[name] >= a.limit: continue
    if a.cond == "cold": mem = b""; nch = 0
    else:
        ids = list(choices[name][n[name]]) if choices is not None else r.retrieve(first, W)
        mem = r.memory(ids, a.headers, SEP, a.order); nch = len(ids)
        if ids: mem_used[name] += 1; mem_bytes[name] += len(mem) - nch * len(SEP)
    b, targ, lp = score(mem + first, second); res[name].append(b / (T // 2)); ntok[name] += len(targ)
    rec.write(json.dumps({"set": name, "w": w, "date": date, "chunks": nch, "bits": round(b, 4), "ctx": [int(t) for t in tok(first)], "tgt": [int(t) for t in targ],
                          "lp": [round(float(x), 5) for x in lp]}) + "\n"); n[name] += 1
    if n[name] % 100 == 0: print(f"  {name}: {n[name]} windows, {np.mean(res[name]):.4f} ({time.time() - t0:.0f} s)", flush=True)
rec.close()
for name in ("kyle", "dlg"):
    summary[name] = {"memory": {"bpb": float(np.mean(res[name])), "n": len(res[name]), "tokens": ntok[name], "per_window": [round(x, 4) for x in res[name]]},
                     "memory_used_windows": mem_used[name], "memory_bytes_mean": (mem_bytes[name] / mem_used[name]) if mem_used[name] else 0}
summary.update({"model": a.model, "adapter": a.adapter, "cond": a.cond, "choices": a.choices, "chooser": chooser, "headers": a.headers, "sep": a.sep, "order": a.order, "T": T, "n_ctx": a.n_ctx,
                "label": a.label, "seconds": round(time.time() - t0)})
json.dump(summary, open(os.path.join(a.out, "qwen_memory.json"), "w"), indent=1)
line = (f"{os.path.basename(a.model.rstrip('/'))}{' + ' + os.path.basename(a.adapter.rstrip('/')) if a.adapter else ''} {a.cond}{' headers' if a.headers else ''}{' ' + a.label if a.label else ''}: "
        f"Kyle's turns ({summary['kyle']['memory']['n']}) {summary['kyle']['memory']['bpb']:.4f}; the dialogue ({summary['dlg']['memory']['n']}) {summary['dlg']['memory']['bpb']:.4f} "
        f"bits/byte ({summary['seconds']} s)")
print(line); open(os.path.join(a.out, "qwen_memory.txt"), "a").write(line + "\n")
