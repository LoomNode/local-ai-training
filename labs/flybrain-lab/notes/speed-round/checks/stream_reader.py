"""The streaming reader (the use layer, second pass, 2026-10-14): sample_tables.window_logits one byte at a time, with the network's
keys and values kept. The writer's first pass ran the whole window through the twelve layers for every byte (2,048 positions of work
for one byte: 20 bytes a second on a free card, and nothing on a CPU). Here the window's keys and values are kept per layer, so a new
byte runs through the layers as one position attending to the kept ones, the arithmetic of the full pass at that position: the same
logits to the card's float noise (--check measures it). The locked reader's shape (checked at load): byte embedding + the table channel
(18 inputs) + the match channel (orders 8, 16, 32), twelve pre-norm blocks of rotary attention and a feed-forward, the final norm, the
decoder, the two channels' bonuses. Nothing else (no memory head, no n-gram rows, no block cuts).

The window: positions are rotary and the model was trained on 2,048 of them, so the kept keys hold at most `window` bytes. When the
text outgrows it the reader re-encodes the last window - slack bytes from position 0 (one full pass, every `slack` bytes) and streams
on; between re-encodings it reads 1,792 to 2,048 bytes of context instead of exactly 2,048 (the first pass's choice). The table inputs
of the new position come from use_features.text_features over the text so far (recomputed each byte in this version: about 4 ms a
byte at a few KB of text, growing with the text), the match inputs from match_features over the window (also recomputed: a sort over
the window). Those two are the next cache; the network was the big one.

usage (the check):  stream_reader.py --ckpt CKPT --tables DIR --check FILE [--bytes 3000] [--window 2048] [--slack 256] [--device cuda:0]
    feeds FILE byte by byte and compares every step's logits with window_logits over the same window; prints the largest difference,
    the agreement of the argmax, and the bytes per second of each path. sample_tables.py --stream writes with it."""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

CODE = os.environ.get("SCORE_CODE", "/games/ailab/lab/flybrain-lab/.worktrees/big1")
sys.path.insert(0, CODE)
from flybrain.model.hybrid import TransformerBackbone, rope_at  # noqa: E402
from flybrain.model.inputs import FusedFeatureChannel, conf_class, match_features  # noqa: E402

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from use_features import HEADER, Tables, text_features  # noqa: E402


def check_shape(model) -> None:
    """The reader this streams: refuse anything else (the first pass's full forward handles every model)."""
    bad = []
    if not isinstance(model.core, TransformerBackbone) or not model.core.rope: bad.append("a rope transformer core")
    if getattr(model, "where", None) not in (None, "none"): bad.append("no memory head")
    if getattr(model, "count", None) is not None: bad.append("no count head")
    if getattr(model, "ngram", None) is not None: bad.append("no n-gram rows")
    if model.mem_every or model.local_attn or model.local_blocks: bad.append("no interleaved heads, local attention or block cuts")
    if not isinstance(model.table, FusedFeatureChannel) or not isinstance(model.match, FusedFeatureChannel): bad.append("fused table and match channels")
    if model.core.attn[0].rope_base != 10000.0: bad.append("the trained rope base")
    if bad:
        raise ValueError("stream_reader needs " + ", ".join(bad))


class Reader:
    """A text read so far, the kept keys and values of its last window, and the logits of the next byte."""

    def __init__(self, model, orders, tables: Tables, device, window: int = 2048, slack: int = 256):
        check_shape(model)
        self.model, self.orders, self.tables, self.device = model, orders, tables, device
        self.window, self.slack = int(window), int(slack)
        self.core = model.core
        self.text = bytearray()
        self.pos0 = 0                                   # the text index at window position 0
        self.kv = [None] * len(self.core.attn)          # per layer: (K, V) [1, H, t, Dh], rotated, for window positions 0..t-1
        self.logits = None                              # of the next byte, [256], after the last feed
        self.full_passes = 0

    @property
    def length(self) -> int:
        return len(self.text)

    def _inputs(self, lo: int, hi: int):
        """The channels' inputs of the window positions [lo, hi) of the text: the table rows lo+1..hi (row p = the inputs for the byte at
        p; the position at p predicts p+1) and the match features of the window up to hi."""
        data = np.frombuffer(bytes(self.text), dtype=np.uint8)
        pred, conf = text_features(self.tables, data, self.orders)
        tpred = torch.from_numpy(np.ascontiguousarray(pred[lo + 1:hi + 1]))[None].to(self.device).long()
        tcls = conf_class(torch.from_numpy(np.ascontiguousarray(conf[lo + 1:hi + 1]))[None].to(self.device))
        win = torch.from_numpy(data[self.pos0:hi].astype(np.int64))[None].to(self.device)
        mpred, mcls = match_features(win, self.model.match_orders)
        a = lo - self.pos0
        return tpred, tcls, mpred[:, a:], mcls[:, a:]

    @torch.no_grad()
    def _run(self, lo: int, hi: int, fresh: bool) -> torch.Tensor:
        """The text positions [lo, hi) through the network at window positions lo-pos0.., attending to the kept keys (fresh: none kept).
        Returns the logits of every position run, [hi - lo, 256]."""
        m, core = self.model, self.core
        tokens = torch.from_numpy(np.frombuffer(bytes(self.text[lo:hi]), dtype=np.uint8).astype(np.int64))[None].to(self.device)
        tpred, tcls, mpred, mcls = self._inputs(lo, hi)
        e = m.embed(tokens) + m.table.embed(tpred, tcls) + m.match.embed(mpred, mcls)
        t = hi - lo
        pos = torch.arange(lo - self.pos0, hi - self.pos0, device=self.device)
        for i, (ln1, ln2, attn, ff) in enumerate(zip(core.ln1, core.ln2, core.attn, core.ff)):
            u = ln1(e)
            q, k, v = attn.qkv(u).view(1, t, 3, attn.h, attn.dh).permute(2, 0, 3, 1, 4)
            q, k = rope_at(q, pos, attn.rope_base), rope_at(k, pos, attn.rope_base)
            if fresh or self.kv[i] is None:
                K, V = k, v
            else:
                K, V = torch.cat([self.kv[i][0], k], dim=2), torch.cat([self.kv[i][1], v], dim=2)
            self.kv[i] = (K, V)
            if t == 1:
                y = F.scaled_dot_product_attention(q, K, V)
            else:
                L = K.size(2); kept = L - t
                mask = torch.arange(L, device=self.device)[None, :] <= (kept + torch.arange(t, device=self.device))[:, None]
                y = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
            e = e + attn.out(y.transpose(1, 2).reshape(1, t, -1))
            e = e + ff(ln2(e))
        out = core.ln(e)
        logits = m.decoder(out)
        logits = m.table.add_bonus(logits, out, tpred, tcls)
        logits = m.match.add_bonus(logits, out, mpred, mcls)
        return logits[0].float()

    def feed(self, data: bytes) -> torch.Tensor:
        """Append `data`; returns the logits of the byte after it. A long append is one pass over the new bytes (and a re-encoding when
        the window overflows); a single byte is one position of work."""
        if not data:
            return self.logits
        self.text += data
        N = len(self.text)
        if N - self.pos0 > self.window:                 # re-encode the last window - slack bytes from position 0
            self.pos0 = N - (self.window - self.slack)
            self.full_passes += 1
            lo, fresh = self.pos0, True
        else:
            lo, fresh = N - len(data), False
        self.logits = self._run(lo, N, fresh)[-1]
        return self.logits

    def window_lo(self) -> int:
        return self.pos0


def check(model, orders, tables, text: bytes, window: int, slack: int, device, prime: int = 64):
    """Feed `text` byte by byte; at every step compare with window_logits over [reader.pos0, N) (the full pass over the same window)."""
    from sample_tables import window_logits
    r = Reader(model, orders, tables, device, window, slack)
    r.feed(text[:prime])
    worst = 0.0; agree = 0; n = 0; t_stream = 0.0; t_full = 0.0
    data = np.frombuffer(text, dtype=np.uint8)
    pred, conf = text_features(tables, data, orders)
    for i in range(prime, len(text)):
        t0 = time.perf_counter(); ls = r.feed(text[i:i + 1]); ls_c = ls.cpu(); t_stream += time.perf_counter() - t0
        N = i + 1
        t0 = time.perf_counter(); lf = window_logits(model, data[:N], pred, conf, r.pos0, N, device)[-1]; lf_c = lf.cpu(); t_full += time.perf_counter() - t0
        d = float((ls_c - lf_c).abs().max()); worst = max(worst, d)
        agree += int(ls_c.argmax() == lf_c.argmax()); n += 1
        if (i - prime) % 500 == 0:
            print(f"  byte {N}: window from {r.pos0}, max |diff| so far {worst:.2e}, argmax agree {agree}/{n}, "
                  f"stream {n / max(t_stream, 1e-9):.0f} B/s, full {n / max(t_full, 1e-9):.1f} B/s", flush=True)
    print(f"{n} steps, {r.full_passes} re-encodings: max |logit diff| {worst:.2e}, argmax agree {agree}/{n}; "
          f"stream {n / t_stream:.0f} bytes/s, full pass {n / t_full:.1f} bytes/s ({device})")
    return worst, agree, n


def main():
    from sample_tables import load
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tables", required=True); ap.add_argument("--check", required=True)
    ap.add_argument("--bytes", type=int, default=3000); ap.add_argument("--window", type=int, default=2048); ap.add_argument("--slack", type=int, default=256)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    tables = Tables(a.tables)
    model, orders, _ = load(a.ckpt, device)
    text = open(a.check, "rb").read()[:a.bytes]
    if not text.startswith(HEADER):
        text = HEADER + os.path.basename(a.check).encode() + b" ====\n" + text
    check(model, orders, tables, text, a.window, a.slack, device)


if __name__ == "__main__":
    main()
