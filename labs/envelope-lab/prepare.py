"""
FROZEN — agent must NOT modify this file.

Hardware envelope, data prep, tokenizer, dataloader, evaluation, metric reporting.
This is the law + oracle for envelope-lab.
"""

from __future__ import annotations

import argparse
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken
import torch

# ---------------------------------------------------------------------------
# HARDWARE ENVELOPE (the constitution)
# ---------------------------------------------------------------------------

# Solo-serve product law (for later serve checks / keep rules).
MAX_VRAM_MB = 22_000  # hard fail above this at end of trial
SOFT_VRAM_MB = 18_000  # prefer staying under when quality is close
MIN_DECODE_TPS = 15.0  # floor for "usable" later; reported every trial
TIME_BUDGET_S = float(os.environ.get("ENVELOPE_TIME_BUDGET", "300"))

# Training geometry (fixed so trials are comparable). Agent may change model
# size in train.py but not these eval constants.
MAX_SEQ_LEN = 512
EVAL_TOKENS = 32_768  # val tokens for bpb (keep modest on 3090)
VOCAB = "gpt2"  # fixed tokenizer — do not change mid-campaign without new baseline

# Data — bulky cache lives on /vault by default (code stays on /games)
ROOT = Path(__file__).resolve().parent
_DEFAULT_CACHE = Path("/vault/ailab/envelope-lab/cache")
CACHE = Path(os.environ.get("ENVELOPE_CACHE", str(_DEFAULT_CACHE)))
RUNS_DIR = Path(os.environ.get("ENVELOPE_RUNS", "/vault/ailab/envelope-lab/runs"))
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
# Fallback corpus if network fails: built-in snippet expanded (still deterministic).
_FALLBACK_TEXT = (
    "Before we proceed any further, hear me speak. "
    "You are all resolved rather to die than to famish? "
    "Resolved. Resolved. First, you know Caius Marcius is chief enemy to the people. "
    "We know it, we know it. Let us kill him, and we'll have corn at our own price. "
    "Is it a verdict? No more talking on it; let it be done.\n"
) * 400

# ---------------------------------------------------------------------------
# Tokenizer + data
# ---------------------------------------------------------------------------


def get_tokenizer():
    return tiktoken.get_encoding(VOCAB)


def _download_text() -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "corpus.txt"
    if path.exists() and path.stat().st_size > 1000:
        return path.read_text(encoding="utf-8", errors="replace")
    try:
        print(f"downloading corpus -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) < 1000:
            raise RuntimeError("corpus too small")
        return text
    except Exception as e:
        print(f"download failed ({e}); using fallback corpus")
        path.write_text(_FALLBACK_TEXT, encoding="utf-8")
        return _FALLBACK_TEXT


def prepare_data() -> dict:
    """Tokenize corpus once; write train.bin / val.bin as uint32 token ids."""
    CACHE.mkdir(parents=True, exist_ok=True)
    train_bin = CACHE / "train.bin"
    val_bin = CACHE / "val.bin"
    meta_path = CACHE / "meta.txt"

    if train_bin.exists() and val_bin.exists() and meta_path.exists():
        print(f"data ready under {CACHE}")
        return {"train": train_bin, "val": val_bin}

    text = _download_text()
    enc = get_tokenizer()
    ids = np.array(enc.encode_ordinary(text), dtype=np.uint32)
    n = len(ids)
    if n < 10_000:
        # repeat to get a usable stream for short experiments
        reps = math.ceil(50_000 / max(n, 1))
        ids = np.tile(ids, reps)

    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    train_ids.tofile(train_bin)
    val_ids.tofile(val_bin)
    meta_path.write_text(
        f"tokenizer={VOCAB}\ntrain_tokens={len(train_ids)}\nval_tokens={len(val_ids)}\n",
        encoding="utf-8",
    )
    print(f"wrote {train_bin} ({len(train_ids)} tok), {val_bin} ({len(val_ids)} tok)")
    return {"train": train_bin, "val": val_bin}


def load_tokens(split: str) -> np.ndarray:
    path = CACHE / f"{split}.bin"
    if not path.exists():
        prepare_data()
    return np.fromfile(path, dtype=np.uint32)


def make_batch(split: str, batch_size: int, seq_len: int, device: torch.device):
    data = load_tokens(split)
    assert len(data) > seq_len + 1
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate_bpb(model, device: torch.device, seq_len: int = MAX_SEQ_LEN) -> float:
    """Validation bits-per-byte proxy: CE loss (nats) -> bits, vocab-fixed.

    With a fixed tokenizer this is comparable across architecture changes.
    Lower is better.
    """
    model.eval()
    data = load_tokens("val")
    # consume ~EVAL_TOKENS tokens
    n_seq = max(1, EVAL_TOKENS // seq_len)
    total_nll = 0.0
    total_tok = 0
    pos = 0
    for _ in range(n_seq):
        if pos + seq_len + 1 >= len(data):
            pos = 0
        chunk = data[pos : pos + seq_len + 1].astype(np.int64)
        pos += seq_len
        x = torch.from_numpy(chunk[:-1][None, :]).to(device)
        y = torch.from_numpy(chunk[1:][None, :]).to(device)
        logits, loss = model(x, y)
        # loss is mean nll per token
        total_nll += float(loss.item()) * seq_len
        total_tok += seq_len
    # nats -> bits
    bpb = (total_nll / max(total_tok, 1)) / math.log(2)
    model.train()
    return bpb


@torch.no_grad()
def measure_decode_tps(model, device: torch.device, seq_len: int = 64, gen_tokens: int = 64) -> float:
    """Rough decode tok/s (single stream) for envelope reporting."""
    model.eval()
    enc = get_tokenizer()
    ids = enc.encode_ordinary("The meaning of life is")
    x = torch.tensor([ids[:seq_len]], dtype=torch.long, device=device)
    # warmup
    for _ in range(3):
        logits, _ = model(x)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, next_id], dim=1)[:, -seq_len:]
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    import time

    t0 = time.perf_counter()
    x = torch.tensor([ids[: min(len(ids), seq_len)]], dtype=torch.long, device=device)
    for _ in range(gen_tokens):
        logits, _ = model(x)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, next_id], dim=1)
        if x.size(1) > seq_len:
            x = x[:, -seq_len:]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    model.train()
    return gen_tokens / max(dt, 1e-6)


def peak_vram_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def envelope_status(peak_mb: float, decode_tps: float) -> str:
    """Return ok | soft_vram | fail_vram | fail_tps for logging."""
    if peak_mb > MAX_VRAM_MB:
        return "fail_vram"
    if decode_tps < MIN_DECODE_TPS:
        return "fail_tps"
    if peak_mb > SOFT_VRAM_MB:
        return "soft_vram"
    return "ok"


def print_report(
    *,
    val_bpb: float,
    training_seconds: float,
    total_seconds: float,
    peak_vram_mb_val: float,
    decode_tps: float,
    num_params_m: float,
    num_steps: int,
    extra: dict | None = None,
) -> None:
    """Canonical trial report — agent greps these keys from run.log."""
    env = envelope_status(peak_vram_mb_val, decode_tps)
    print("---")
    print(f"val_bpb:            {val_bpb:.6f}")
    print(f"training_seconds:   {training_seconds:.1f}")
    print(f"total_seconds:      {total_seconds:.1f}")
    print(f"peak_vram_mb:       {peak_vram_mb_val:.1f}")
    print(f"decode_tps:         {decode_tps:.2f}")
    print(f"envelope:           {env}")
    print(f"max_vram_mb_limit:  {MAX_VRAM_MB}")
    print(f"min_decode_tps:     {MIN_DECODE_TPS}")
    print(f"num_params_M:       {num_params_m:.2f}")
    print(f"num_steps:          {num_steps}")
    if extra:
        for k, v in extra.items():
            print(f"{k}: {v}")


def main_cli() -> None:
    ap = argparse.ArgumentParser(description="envelope-lab data prep (frozen)")
    ap.add_argument("--force", action="store_true", help="rebuild bins")
    args = ap.parse_args()
    if args.force:
        for p in (CACHE / "train.bin", CACHE / "val.bin", CACHE / "meta.txt"):
            if p.exists():
                p.unlink()
    prepare_data()
    print("prepare complete")


if __name__ == "__main__":
    main_cli()
