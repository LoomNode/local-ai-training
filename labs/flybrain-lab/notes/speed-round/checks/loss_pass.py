"""Per-position loss of a plain checkpoint over the training bytes, for gating table writes (the fly's dopamine: write what surprised you).
usage: loss_pass.py <checkpoint> <out.npy> [--device cuda:0]
Windows of 2,048 bytes, non-overlapping, so a position's loss is from a cold window (the first bytes of each window are
predicted from little context; they come out high and get written, which is harmless). Output: uint8 = bits x 16, clipped at 255,
one per corpus position in [0, 200,000,000); position 0 gets 255."""
import math
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

WT = "/games/ailab/lab/flybrain-lab/.worktrees/speed-run4"
sys.path.insert(0, WT)
from scripts.cross_eval import rebuild  # noqa: E402

CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
TRAIN_END, T, B = 200_000_000, 2048, 32
ckpt, out = sys.argv[1], sys.argv[2]
device = torch.device(sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:0")
data = np.fromfile(CORPUS, dtype=np.uint8, count=TRAIN_END + 1)
m = rebuild(torch.load(ckpt, map_location="cpu", weights_only=False)).to(device).eval()
loss = np.full(TRAIN_END, 255, dtype=np.uint8)
t0 = time.time()
starts = list(range(0, TRAIN_END - T, T * B))
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for i, s in enumerate(starts):
        n = min(B, (TRAIN_END - 1 - s) // T)
        if n <= 0:
            break
        x = torch.from_numpy(data[s:s + n * T].astype(np.int64)).view(n, T).to(device)
        y = torch.from_numpy(data[s + 1:s + 1 + n * T].astype(np.int64)).view(n, T).to(device)
        logits, _ = m(x)
        ce = F.cross_entropy(logits.reshape(-1, 256).float(), y.reshape(-1), reduction="none") / math.log(2)
        q = (ce * 16).clamp_(0, 255).round().to(torch.uint8).cpu().numpy()
        loss[s + 1:s + 1 + n * T] = q                       # the loss of predicting position p = s + t + 1
        if i % 500 == 0:
            print(f"{s:,} / {TRAIN_END:,}  {time.time() - t0:.0f} s", flush=True)
np.save(out, loss)
bits = loss[1:].astype(np.float64) / 16
print(f"done in {time.time() - t0:.0f} s: mean {bits.mean():.3f} bits/byte; share of positions at or above 0.5 / 1 / 2 / 4 bits: "
      f"{(bits >= 0.5).mean():.3f} / {(bits >= 1).mean():.3f} / {(bits >= 2).mean():.3f} / {(bits >= 4).mean():.3f}", flush=True)
