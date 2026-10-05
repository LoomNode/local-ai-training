"""How much of the table-fed models' score comes from table lookups whose context starts before the window?
The feature at window index t for order n is made from the n bytes ending at the input of index t, so for t < n - 1 it uses
bytes the model cannot see in a cold window. Score the first 128 dev windows three ways: features as trained, features set
to a miss where t < n - 1, and all features set to a miss. CPU, fp32."""
import math
import sys

import torch
from torch.nn import functional as F

W = "/games/ailab/lab/flybrain-lab/.worktrees/speed-run2"
sys.path.insert(0, W)
torch.set_num_threads(6)

from flybrain.data.memfeat import FeatureStore  # noqa: E402
from flybrain.data.text import fixed_windows, load_bytes  # noqa: E402
from scripts.cross_eval import rebuild  # noqa: E402

CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
FEAT = "/games/ailab/lab/flybrain-lab/.worktrees/speed/results/mem/feat"
CK = "/games/ailab/lab/flybrain-lab/results/ckpt/"
OFF, N, T, ORDERS = 202_117_357, 128, 2048, (8, 16, 32)

data = load_bytes(CORPUS, max_bytes=N * T + 1, offset=OFF)
xs, ys = fixed_windows(data, T, N)
store = FeatureStore(FEAT, ORDERS)
pred, conf = store.fixed(OFF, N, T)
masked_p, masked_c = pred.clone(), conf.clone()
for k, n in enumerate(ORDERS):
    masked_p[:, :n - 1, k] = 0
    masked_c[:, :n - 1, k] = 0
none_p, none_c = torch.zeros_like(pred), torch.zeros_like(conf)


@torch.no_grad()
def per_position(model, p, c):
    tot = torch.zeros(T, dtype=torch.float64)
    for i in range(0, N, 8):
        logits, _ = model(xs[i:i + 8], mem_pred=p[i:i + 8], mem_conf=c[i:i + 8])
        ce = F.cross_entropy(logits.reshape(-1, 256).float(), ys[i:i + 8].reshape(-1), reduction="none").view(-1, T)
        tot += ce.double().sum(0)
    return tot / N / math.log(2)


for name in ("speed_t2_tf_tab_c_3600_s0_transformer.pt", "speed_t2_zh_tab_c_3600_s0_mingru.pt"):
    m = rebuild(torch.load(CK + name, map_location="cpu", weights_only=False)).eval()
    rows = {}
    for label, (p, c) in (("as trained", (pred, conf)), ("miss where the context starts before the window", (masked_p, masked_c)), ("all misses", (none_p, none_c))):
        per = per_position(m, p, c)
        rows[label] = per
        print(f"{name} | {label}: cold {float(per.mean()):.4f}, warm {float(per[T // 2:].mean()):.4f}, "
              f"positions 0-6 {float(per[:7].mean()):.3f}, 7-30 {float(per[7:31].mean()):.3f}, 31-255 {float(per[31:256].mean()):.3f}", flush=True)
    d = rows["miss where the context starts before the window"] - rows["as trained"]
    print(f"  masked minus as trained: cold {float(d.mean()):+.4f}; from positions 0-30 {float(d[:31].sum()) / T:+.4f}, from later positions {float(d[31:].sum()) / T:+.4f}", flush=True)
