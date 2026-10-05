"""Why is the compiled transformer faster per step with --match-feat? One training step as run_sparsela.main() runs it, timed in
its three parts (forward + loss, backward, clip + optimizer) with a sync after each, for several flag sets in turn (A B A B ...).
usage: probe.py [--prof] [--steps N] name=flags ... (flags are appended to the night queues' COMMON)."""
import sys
import time

import torch

import os
WT = os.environ.get("PROBE_WT", "/games/ailab/lab/flybrain-lab/.worktrees/speed-run3")
sys.path.insert(0, WT); sys.path.insert(0, WT + "/scripts")
import run_sparsela as R  # noqa: E402
from flybrain.data.text import batch_windows, load_bytes  # noqa: E402

C = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
COMMON = (f"--task code --corpus-file {C} --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 "
          "--decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 "
          "--chunk 256 --amp --steps 400").split()
prof = "--prof" in sys.argv
n = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 40
specs = [a.split("=", 1) for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
train = load_bytes(C, max_bytes=20_000_000)
device = torch.device("cuda:0")
sync = lambda: torch.cuda.synchronize(device)


def make(flags):
    args = R.parse_args(COMMON + flags.split())
    R.set_precision(args)
    torch.manual_seed(0)
    model = R.make_model(args.arms, args, device)
    fwd = R.make_train_fwd(model, args)
    opt, sched = R.make_opt_sched(model, args)

    def step(i, split=None):
        x, y = batch_windows(train, seq_len=args.seq_len, batch=args.batch, seed=i)
        x, y = x.to(device), y.to(device)
        if split is not None:
            sync(); t0 = time.perf_counter()
        loss, _ = R.train_loss(model, x, y, args, device, fwd, {})
        if split is not None:
            sync(); t1 = time.perf_counter()
        opt.zero_grad(); loss.backward()
        if split is not None:
            sync(); t2 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if split is not None:
            sync(); t3 = time.perf_counter()
            split[0] += t1 - t0; split[1] += t2 - t1; split[2] += t3 - t2
        return loss
    return step


steps = {}
for name, flags in specs:
    t = time.perf_counter()
    steps[name] = make(flags)
    for i in range(8):
        steps[name](i)
    sync()
    print(f"{name}: built, compiled and warmed in {time.perf_counter() - t:.0f} s  [{flags}]", flush=True)
for rnd in range(3):
    for name, _ in specs:
        step = steps[name]
        sync(); t = time.perf_counter()
        for i in range(n):
            loss = step(1000 * rnd + i)
        float(loss); sync(); whole = (time.perf_counter() - t) / n
        split = [0.0, 0.0, 0.0]
        for i in range(n // 2):
            step(1000 * rnd + 500 + i, split)
        f, b, o = (1000 * s / (n // 2) for s in split)
        print(f"round {rnd} {name:14s} {1000 * whole:7.1f} ms/step   forward+loss {f:6.1f}  backward {b:6.1f}  clip+optimizer {o:5.1f}   loss {float(loss):.3f}", flush=True)
if prof:
    from torch.profiler import ProfilerActivity, profile
    for name, _ in specs:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            for i in range(5):
                steps[name](9000 + i)
            sync()
        print(f"\n=== {name}: top device time over 5 steps")
        rows = sorted(p.key_averages(), key=lambda e: -e.self_device_time_total)
        tot = sum(e.self_device_time_total for e in rows)
        print(f"self device time {tot / 5000:.1f} ms/step")
        for e in rows[:34]:
            if e.self_device_time_total > 0:
                print(f"{e.self_device_time_total / 5000:8.2f} ms/step  {e.count // 5:4d} calls/step  {e.self_device_time_total / max(e.count, 1):8.1f} us each  {e.key[:110]}")
