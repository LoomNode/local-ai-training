#!/usr/bin/env python
"""Arm B (2026-11-06): LoRA on a Qwen3.5 Base student over train.npz, with the 27B's sampled targets (teacher_pass.py) or the true tokens.
Loss per loss position = alpha * KD + (1 - alpha) * CE, KD = -mean_j log q(v_j) over the K teacher samples (an unbiased estimate of the
cross-entropy to the teacher, the gradient of the KL), CE = -log q(true). Sequences are packed by length into micro-batches under
--micro-tokens, steps hold --step-tokens loss tokens; AdamW, cosine with warm-up, gradient checkpointing, the embedding frozen. 2% of
the sequences (seeded) are held out and scored every --eval-every steps. Writes adapter/ (peft), base_model, train.json. Numbers only.
usage: train_lora.py --model DIR --data DIR --out DIR [--teacher DIR] [--alpha 1.0] [--r 64] [--lr 1e-4] [--epochs 1]"""
import argparse, json, math, os, sys, time
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--teacher", default="")
ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--r", type=int, default=64); ap.add_argument("--lora-alpha", type=int, default=128); ap.add_argument("--dropout", type=float, default=0.05)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--step-tokens", type=int, default=32768); ap.add_argument("--micro-tokens", type=int, default=8192)
ap.add_argument("--warmup", type=float, default=0.05); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval-every", type=int, default=50); ap.add_argument("--limit-steps", type=int, default=0)
ap.add_argument("--kinds", default="0,1,2", help="sequence kinds to train on (0 files, 1 sessions, 2 replay)"); ap.add_argument("--no-ckpt", action="store_true")
a = ap.parse_args(); os.umask(0o077); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
z = np.load(os.path.join(a.data, "train.npz")); tokens, seq_start, loss_start, kind = z["tokens"], z["seq_start"], z["loss_start"], z["kind"]
L = seq_start[1:] - seq_start[:-1]; nloss = L - loss_start; pos_start = np.concatenate([[0], np.cumsum(nloss)])
if a.teacher:
    T_ids = np.load(os.path.join(a.teacher, "ids.npy"), mmap_mode="r"); K = T_ids.shape[1]; assert T_ids.shape[0] == pos_start[-1]
else: T_ids = None; K = 0; assert a.alpha == 0, "no teacher targets: use --alpha 0"
keep = np.isin(kind, [int(k) for k in a.kinds.split(",")]); idx = np.nonzero(keep)[0]; rng.shuffle(idx)
n_dev = max(8, len(idx) // 50); dev, train = idx[:n_dev], idx[n_dev:]

def batches(seqs, shuffle):
    """micro-batches: sequences of similar length packed under --micro-tokens padded tokens"""
    s = np.asarray(seqs)
    if shuffle: s = s[rng.permutation(len(s))]
    chunks = [s[i:i + 256] for i in range(0, len(s), 256)]; out = []
    for ch in chunks:
        ch = ch[np.argsort(L[ch])]; cur = []
        for j in ch:
            if cur and max(L[cur + [j]]) * (len(cur) + 1) > a.micro_tokens: out.append(cur); cur = []
            cur.append(int(j))
        if cur: out.append(cur)
    if shuffle: rng.shuffle(out)
    return out

from transformers import Qwen3_5ForCausalLM
from peft import LoraConfig, get_peft_model
base = Qwen3_5ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
for p in base.parameters(): p.requires_grad_(False)
if not a.no_ckpt: base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); base.enable_input_require_grads()
cfg = LoraConfig(r=a.r, lora_alpha=a.lora_alpha, lora_dropout=a.dropout, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(base, cfg); inner = model.base_model.model; ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"student {a.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, {ntrain / 1e6:.1f}M trainable (LoRA r={a.r}); {len(train)} train / {len(dev)} dev sequences, "
      f"{int(nloss[train].sum()):,} loss tokens, alpha {a.alpha}, K {K}", flush=True)

def loss_of(mb, grad=True):
    """(sum of loss over the loss positions, their count, sum KD, sum CE) for one micro-batch"""
    Lm = int(L[mb].max()); B = len(mb); x = np.zeros((B, Lm), dtype=np.int64); mask = np.zeros((B, Lm), dtype=np.int64); lossm = np.zeros((B, Lm), dtype=bool)
    tids = []; trues = []
    for b, j in enumerate(mb):
        n = int(L[j]); x[b, :n] = tokens[seq_start[j]:seq_start[j] + n]; mask[b, :n] = 1; ls = int(loss_start[j]); lossm[b, ls:n] = True
        trues.append(x[b, ls:n])
        if T_ids is not None: tids.append(np.asarray(T_ids[pos_start[j]:pos_start[j + 1]]))
    x = torch.from_numpy(x).cuda(); mask = torch.from_numpy(mask).cuda(); lossm = torch.from_numpy(lossm).cuda()
    true = torch.from_numpy(np.concatenate(trues)).cuda(); tid = torch.from_numpy(np.concatenate(tids)).cuda().long() if tids else None
    # the predicting positions are one to the left of the loss positions
    predm = torch.zeros_like(lossm); predm[:, :-1] = lossm[:, 1:]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = inner.model(input_ids=x, attention_mask=mask).last_hidden_state
    hp = h[predm]                                                  # [M, H] in the same order as true / tid
    hd = hp.detach().requires_grad_(grad)                          # the loss in chunks against a detached copy (the logits never all exist at once), one backward through the model
    tot = 0.0; kd_s = 0.0; ce_s = 0.0; M = hp.shape[0]
    for c in range(0, M, 1024):
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = inner.lm_head(hd[c:c + 1024])
        lsm = torch.log_softmax(lg.float(), -1)
        ce = -lsm.gather(1, true[c:c + 1024, None])[:, 0]
        kd = -lsm.gather(1, tid[c:c + 1024]).mean(1) if tid is not None else torch.zeros_like(ce)
        l = (a.alpha * kd + (1 - a.alpha) * ce).sum()
        if grad: (l / a.step_tokens).backward()
        tot += float(l.detach()); kd_s += float(kd.detach().sum()); ce_s += float(ce.detach().sum())
    if grad: hp.backward(hd.grad)
    return tot, M, kd_s, ce_s

@torch.no_grad()
def evaluate():
    model.eval(); kd = ce = 0.0; n = 0
    for mb in batches(dev, False):
        _, M, k, c = loss_of(mb, grad=False); kd += k; ce += c; n += M
    model.train(); return kd / n, ce / n

# steps: groups of micro-batches holding ~step_tokens loss tokens
micro = []; 
for ep in range(math.ceil(a.epochs)): micro += batches(train, True)
steps = []; cur = []; cnt = 0; total_loss = int(nloss[train].sum() * a.epochs)
for mb in micro:
    cur.append(mb); cnt += int(nloss[mb].sum())
    if cnt >= a.step_tokens: steps.append(cur); cur = []; cnt = 0
    if sum(int(nloss[m].sum()) for s in steps for m in s) >= total_loss: break
if cur: steps.append(cur)
if a.limit_steps: steps = steps[:a.limit_steps]
nsteps = len(steps); warm = max(1, int(a.warmup * nsteps))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, nsteps - warm))))))
print(f"{nsteps} steps of ~{a.step_tokens:,} loss tokens, warm-up {warm}, lr {a.lr}", flush=True)
curve = []; seen = 0; dev0 = evaluate(); t0 = time.time(); print(f"  dev before: KD {dev0[0]:.4f} CE {dev0[1]:.4f} nats/token", flush=True); curve.append({"step": 0, "dev_kd": dev0[0], "dev_ce": dev0[1]})
model.train()
for s, grp in enumerate(steps):
    kd = ce = 0.0; n = 0
    for mb in grp:
        _, M, k, c = loss_of(mb); kd += k; ce += c; n += M
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); seen += n
    row = {"step": s + 1, "kd": kd / n, "ce": ce / n, "tokens": seen, "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0)}
    if (s + 1) % a.eval_every == 0 or s + 1 == nsteps: d = evaluate(); row.update({"dev_kd": d[0], "dev_ce": d[1]})
    curve.append(row)
    if (s + 1) % 10 == 0 or "dev_kd" in row:
        print(f"  step {s + 1}/{nsteps}: KD {row['kd']:.4f} CE {row['ce']:.4f}" + (f" | dev KD {row['dev_kd']:.4f} CE {row['dev_ce']:.4f}" if "dev_kd" in row else "")
              + f" | {seen / (time.time() - t0):.0f} loss tok/s, {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
model.save_pretrained(os.path.join(a.out, "adapter")); open(os.path.join(a.out, "base_model"), "w").write(a.model)
json.dump({"args": vars(a), "steps": nsteps, "trainable": ntrain, "curve": curve, "seconds": round(time.time() - t0)}, open(os.path.join(a.out, "train.json"), "w"), indent=1)
open(os.path.join(a.out, "train.done"), "w").close(); print("done", flush=True)
