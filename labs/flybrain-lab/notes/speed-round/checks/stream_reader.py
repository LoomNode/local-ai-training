"""The streaming reader (the use layer, second pass, 2026-10-14): sample_tables.window_logits one byte at a time, with the network's
keys and values kept. The writer's first pass ran the whole window through the twelve layers for every byte (2,048 positions of work
for one byte: 20 bytes a second on a free card, and nothing on a CPU). Here the window's keys and values are kept per layer, so a new
byte runs through the layers as one position attending to the kept ones, the arithmetic of the full pass at that position: the same
logits to the card's float noise (--check measures it). The locked reader's shape (checked at load): byte embedding + the table channel
(18 inputs) + the match channel (orders 8, 16, 32), twelve pre-norm blocks of rotary attention and a feed-forward, the final norm, the
decoder, the two channels' bonuses. Nothing else (no memory head, no n-gram rows, no block cuts).

The window: positions are rotary and the model was trained on 2,048 of them, so the kept keys hold at most `window` bytes. When the
text outgrows it the reader re-encodes the last window - slack bytes from position 0 (one full pass, every `slack` bytes) and streams
on; between re-encodings it reads 1,792 to 2,048 bytes of context instead of exactly 2,048 (the first pass's choice). The inputs are
carried byte to byte (stream_inputs.py, the second version; the first recomputed them over the whole text at every byte, 4 ms and
growing): TextState gives the table inputs of the new position from its state (26 us a byte), MatchState the match features from one
dictionary per order over the window. A long feed takes the vectorised inputs and warms the state from a line start near the end. One
lookahead of the build is honoured by recomputing: a file header line resets the brace and paren depths from its first byte, known
eight bytes later, so a header met at a depth other than 0 reruns its eight positions through the network (feed's `redo`).

usage (the check):  stream_reader.py --ckpt CKPT --tables DIR --check FILE [--bytes 3000] [--prime 1024] [--window 2048] [--slack 256] [--device cuda:0]
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
from stream_inputs import MatchState, TextState  # noqa: E402
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
        self.inp = TextState(tables, orders)            # the table inputs: rows 0..N, row p the inputs of position p
        self.text = self.inp.text
        self.ms = MatchState(model.match_orders, self.text)
        self.pos0 = 0                                   # the text index at window position 0
        self.kv = [None] * len(self.core.attn)          # per layer: (K, V) [1, H, window, Dh], rotated, filled for window positions 0..t-1
        self.filled = 0                                 # how many window positions the kept keys hold
        self.logits = None                              # of the next byte, [256], after the last feed
        self.full_passes = 0
        self.redos = 0                                  # header depth resets that reran positions

    @property
    def length(self) -> int:
        return len(self.text)

    def _inputs(self, lo: int, hi: int, mrows):
        """The channels' inputs of the text positions [lo, hi): the table rows lo+1..hi (row p = the inputs for the byte at p; the
        position at p predicts p+1) and the match features of the same positions (mrows: int64 [hi - lo, orders] pred and cls)."""
        tpred = torch.from_numpy(np.ascontiguousarray(self.inp.pred[lo + 1:hi + 1]))[None].to(self.device).long()
        tcls = conf_class(torch.from_numpy(np.ascontiguousarray(self.inp.conf[lo + 1:hi + 1]))[None].to(self.device))
        mpred, mcls = (torch.from_numpy(np.ascontiguousarray(x))[None].to(self.device) for x in mrows)
        return tpred, tcls, mpred, mcls

    @torch.no_grad()
    def _run(self, lo: int, hi: int, fresh: bool, mrows) -> torch.Tensor:
        """The text positions [lo, hi) through the network at window positions lo-pos0.., attending to the kept keys (fresh: none kept;
        lo below the kept ones: those are recomputed). Returns the logits of every position run, [hi - lo, 256]."""
        m, core = self.model, self.core
        tokens = torch.from_numpy(np.frombuffer(bytes(self.text[lo:hi]), dtype=np.uint8).astype(np.int64))[None].to(self.device)
        tpred, tcls, mpred, mcls = self._inputs(lo, hi, mrows)
        e = m.embed(tokens) + m.table.embed(tpred, tcls) + m.match.embed(mpred, mcls)
        t = hi - lo
        pos = torch.arange(lo - self.pos0, hi - self.pos0, device=self.device)
        a = lo - self.pos0                              # the window positions run: a .. a + t - 1
        if fresh:
            self.filled = 0
        assert a <= self.filled, f"positions {a}.. fed, but the kept keys hold {self.filled}"
        for i, (ln1, ln2, attn, ff) in enumerate(zip(core.ln1, core.ln2, core.attn, core.ff)):
            u = ln1(e)
            q, k, v = attn.qkv(u).view(1, t, 3, attn.h, attn.dh).permute(2, 0, 3, 1, 4)
            q, k = rope_at(q, pos, attn.rope_base), rope_at(k, pos, attn.rope_base)
            if self.kv[i] is None:                      # one buffer per layer for the whole window, written in place (no growing copies)
                self.kv[i] = (torch.empty(1, attn.h, self.window, attn.dh, device=self.device, dtype=k.dtype),
                              torch.empty(1, attn.h, self.window, attn.dh, device=self.device, dtype=v.dtype))
            Kb, Vb = self.kv[i]
            Kb[:, :, a:a + t] = k; Vb[:, :, a:a + t] = v
            K, V = Kb[:, :, :a + t], Vb[:, :, :a + t]
            if t == 1:
                y = F.scaled_dot_product_attention(q, K, V)
            else:
                mask = torch.arange(a + t, device=self.device)[None, :] <= (a + torch.arange(t, device=self.device))[:, None]
                y = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
            e = e + attn.out(y.transpose(1, 2).reshape(1, t, -1))
            e = e + ff(ln2(e))
        self.filled = a + t
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
        fix = None
        if len(data) == 1:
            fix = self.inp.push(data[0])
        else:
            self.inp.extend(data)
        N = len(self.text)
        if N - self.pos0 > self.window:                 # re-encode the last window - slack bytes from position 0
            self.pos0 = N - (self.window - self.slack)
            self.full_passes += 1
            lo, fresh = self.pos0, True
        else:
            lo, fresh = N - len(data), False
        if fix is not None and not fresh:               # a header's depth reset: rerun from the position that read its first row
            self.redos += 1
            lo = max(fix - 1, self.pos0); fresh = lo == self.pos0
        if not fresh and len(data) == 1 and fix is None:
            mrows = [np.array([r], np.int64) for r in self.ms.push(N - 1)]
        else:                                           # a window from scratch, a long append or a rerun: the window's dictionaries anew
            mp, mc = self.ms.rows(self.pos0, N)
            mrows = (mp[lo - self.pos0:], mc[lo - self.pos0:])
        self.logits = self._run(lo, N, fresh, mrows)[-1]
        return self.logits

    def window_lo(self) -> int:
        return self.pos0


def check(model, orders, tables, text: bytes, window: int, slack: int, device, prime: int = 1024):
    """Feed `text` byte by byte; at every step compare with window_logits over [reader.pos0, N) (the full pass over the same window)."""
    from sample_tables import window_logits
    r = Reader(model, orders, tables, device, window, slack)
    r.feed(text[:prime])
    worst = 0.0; agree = 0; n = 0; t_stream = 0.0; t_full = 0.0
    data = np.frombuffer(text, dtype=np.uint8)
    for i in range(prime, len(text)):
        t0 = time.perf_counter(); ls = r.feed(text[i:i + 1]); ls_c = ls.cpu(); t_stream += time.perf_counter() - t0
        N = i + 1
        t0 = time.perf_counter()
        pred, conf = text_features(tables, data[:N], orders)           # from the text so far: the reference cannot know a header before its eighth byte either
        lf = window_logits(model, data[:N], pred, conf, r.pos0, N, device)[-1]; lf_c = lf.cpu(); t_full += time.perf_counter() - t0
        d = float((ls_c - lf_c).abs().max()); worst = max(worst, d)
        agree += int(ls_c.argmax() == lf_c.argmax()); n += 1
        if (i - prime) % 500 == 0:
            print(f"  byte {N}: window from {r.pos0}, max |diff| so far {worst:.2e}, argmax agree {agree}/{n}, "
                  f"stream {n / max(t_stream, 1e-9):.0f} B/s, full {n / max(t_full, 1e-9):.1f} B/s", flush=True)
    print(f"{n} steps, {r.full_passes} re-encodings, {r.redos} header reruns: max |logit diff| {worst:.2e}, argmax agree {agree}/{n}; "
          f"stream {n / t_stream:.0f} bytes/s, full pass {n / t_full:.1f} bytes/s ({device})")
    return worst, agree, n


def main():
    from sample_tables import load
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tables", required=True); ap.add_argument("--check", required=True)
    ap.add_argument("--bytes", type=int, default=3000); ap.add_argument("--window", type=int, default=2048); ap.add_argument("--slack", type=int, default=256)
    ap.add_argument("--prime", type=int, default=1024, help="bytes fed at once first (the warm start), then one at a time")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    tables = Tables(a.tables)
    model, orders, _ = load(a.ckpt, device)
    text = open(a.check, "rb").read()[:a.bytes]
    if not text.startswith(HEADER):
        text = HEADER + os.path.basename(a.check).encode() + b" ====\n" + text
    check(model, orders, tables, text, a.window, a.slack, device, a.prime)


if __name__ == "__main__":
    main()
