#!/usr/bin/env python3
"""Prompt a saved checkpoint: bytes in, sampled bytes out.

The recompute path runs the plain forward over the last --window bytes for every new byte. It works for any arm
and needs nothing from the model beyond forward()."""

from __future__ import annotations

import argparse
import codecs
import sys
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cross_eval import rebuild


def sample_next(logits: Tensor, temperature: float, top_k: int, gen: torch.Generator) -> int:
    """One byte from next-byte logits [256]. temperature <= 0 is greedy; top_k > 0 keeps the k most likely bytes."""
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if 0 < top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen))


@torch.no_grad()
def generate(model, prompt: bytes, n: int, temperature: float = 0.8, top_k: int = 40, seed: int = 0, window: int = 2048, on_byte=None) -> bytes:
    """Sample n bytes after `prompt` (an empty prompt starts from a newline). `on_byte` is called with each byte as it is drawn."""
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    buf = list(prompt) or [10]
    out = []
    for _ in range(n):
        logits, _ = model(torch.tensor([buf[-window:]], device=device))
        b = sample_next(logits[0, -1].float().cpu(), temperature, top_k, gen)
        buf.append(b); out.append(b)
        if on_byte is not None:
            on_byte(b)
    return bytes(out)


class Utf8Printer:
    """Writes sampled bytes as text as they arrive: a multi-byte character prints once it is complete,
    and a byte that cannot be part of UTF-8 prints as U+FFFD."""

    def __init__(self, write):
        self._write = write
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, b: int) -> None:
        s = self._dec.decode(bytes([b]))
        if s:
            self._write(s)

    def close(self) -> None:
        s = self._dec.decode(b"", final=True)
        if s:
            self._write(s)


def load_model(path: str, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return rebuild(ck).to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--prompt", default="")
    src.add_argument("--prompt-file", default="")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    prompt = Path(args.prompt_file).read_bytes() if args.prompt_file else args.prompt.encode("utf-8")
    model = load_model(args.ckpt, torch.device(args.device))

    def write(s: str) -> None:
        sys.stdout.write(s); sys.stdout.flush()

    write(prompt.decode("utf-8", errors="replace"))
    printer = Utf8Printer(write)
    generate(model, prompt, args.n, args.temperature, args.top_k, args.seed, args.window, on_byte=printer.feed)
    printer.close()
    write("\n")


if __name__ == "__main__":
    main()
