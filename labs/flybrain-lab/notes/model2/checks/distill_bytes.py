#!/usr/bin/env python
"""The byte marginalisation of a token model's next-token distribution (T3 of the distillation plan, 2026-10-20; the spec:
docs/superpowers/specs/2026-10-20-distillation-design.md). The reader predicts bytes; the teacher (Qwen3.8-27B, 248,320 tokens of 1 to
32 bytes) predicts tokens. At a token position the teacher's kept tokens (its top K by probability, with their probabilities) give the
next-BYTE distribution at every byte of the text's next token tau = b1..bm:
  at the boundary (prefix b""):      P(b) = sum of P(t) over kept t whose first byte is b;
  inside tau (prefix b1..bj, j < m): P(b) ~ sum of P(t) over kept t with t[:j] = b1..bj, len(t) > j and t[j] = b; tokens that END at j
                                      are dropped (their next byte would come from a next-token distribution the pass does not have).
The result is the top 8 bytes with probabilities renormalised to sum to 1 over those 8, the mass (the probability of the kept tokens
that continue the prefix, before any renormalisation) and the cover (the share of that mass the 8 bytes hold; a diagnostic, expected
near 1 for a peaked teacher); mass 0 means no kept token continues the prefix and the caller writes the hard target. Pure numpy; the vocabulary comes from llama-cpp's detokenize (vocab_only=True).
usage: distill_bytes.py --test [--model GGUF]   (the toy-vocabulary checks; with a model, a check on the real vocabulary)"""
import argparse
import numpy as np

MAXLEN = 32


class ByteVocab:
    """tok [V, MAXLEN] uint8 (the token's bytes, zero-padded), len [V] int16."""

    def __init__(self, token_bytes):
        V = len(token_bytes); self.tok = np.zeros((V, MAXLEN), np.uint8); self.len = np.zeros(V, np.int16)
        for i, b in enumerate(token_bytes):
            b = b[:MAXLEN]; self.len[i] = len(b)
            if b: self.tok[i, :len(b)] = np.frombuffer(b, np.uint8)

    @classmethod
    def from_llama(cls, llm):
        return cls([llm.detokenize([i]) for i in range(llm.n_vocab())])

    def kept(self, ids):
        """The bytes and lengths of the kept token ids: (tok_k [K, MAXLEN], len_k [K])."""
        return self.tok[ids], self.len[ids]


def next_bytes(tok_k, len_k, p_k, prefix: bytes, top: int = 8):
    """(ids [top] uint8, p [top] float32 summing to 1, mass, cover) of the next byte after `prefix` under the kept tokens; see above."""
    j = len(prefix)
    m = len_k > j
    if j: m &= (tok_k[:, :j] == np.frombuffer(prefix, np.uint8)).all(axis=1)
    if not m.any(): return np.zeros(top, np.uint8), np.zeros(top, np.float32), 0.0, 0.0
    w = np.bincount(tok_k[m, j], weights=p_k[m], minlength=256)
    mass = float(w.sum())
    if mass <= 0: return np.zeros(top, np.uint8), np.zeros(top, np.float32), 0.0, 0.0
    ids = np.argsort(-w, kind="stable")[:top]; held = float(w[ids].sum())
    return ids.astype(np.uint8), (w[ids] / held).astype(np.float32), mass, held / mass


def token_targets(tok_k, len_k, p_k, true_token: bytes, top: int = 8):
    """The targets of every byte of the text's next token: a list of (ids, p, mass, cover) for prefixes b"" .. true_token[:-1]; where
    mass is 0 the hard target (the true byte alone) is returned with mass 0, so the caller can count the fallbacks."""
    out = []
    for j in range(len(true_token)):
        ids, p, mass, cover = next_bytes(tok_k, len_k, p_k, true_token[:j], top)
        if mass == 0:
            ids = np.zeros(top, np.uint8); p = np.zeros(top, np.float32); ids[0] = true_token[j]; p[0] = 1.0
        out.append((ids, p, mass, cover))
    return out


def selftest(model=None):
    vb = ByteVocab([b"a", b"ab", b"abc", b"b", b"abd", b""]); ids = np.arange(6); tok_k, len_k = vb.kept(ids)
    p = np.array([0.1, 0.2, 0.3, 0.25, 0.15, 0.0])
    i, q, m, c = next_bytes(tok_k, len_k, p, b"");    assert m == 1.0 and c == 1.0 and i[0] == ord("a") and i[1] == ord("b") and abs(q[0] - 0.75) < 1e-6 and abs(q[1] - 0.25) < 1e-6 and q[2] == 0, (i, q, m)
    i, q, m, c = next_bytes(tok_k, len_k, p, b"a");   assert abs(m - 0.65) < 1e-9 and i[0] == ord("b") and abs(q[0] - 1.0) < 1e-6 and q[1] == 0, (i, q, m)
    i, q, m, c = next_bytes(tok_k, len_k, p, b"ab");  assert abs(m - 0.45) < 1e-9 and i[0] == ord("c") and i[1] == ord("d") and abs(q[0] - 2 / 3) < 1e-6, (i, q, m)
    i, q, m, c = next_bytes(tok_k, len_k, p, b"x");   assert m == 0.0 and q.sum() == 0
    i, q, m, c = next_bytes(tok_k, len_k, p, b"", top=1); assert abs(q[0] - 1.0) < 1e-6 and abs(c - 0.75) < 1e-6
    t = token_targets(tok_k, len_k, p, b"abcq");       assert len(t) == 4 and t[0][2] == 1.0 and abs(t[2][2] - 0.45) < 1e-9 and t[3][2] == 0.0 and t[3][0][0] == ord("q") and t[3][1][0] == 1.0
    print("toy vocabulary: 5 checks pass")
    if model:
        from llama_cpp import Llama
        llm = Llama(model_path=model, vocab_only=True, verbose=False); vb = ByteVocab.from_llama(llm)
        rng = np.random.default_rng(0); ids = rng.choice(llm.n_vocab(), 512, replace=False); pk = rng.dirichlet(np.ones(512) * 0.1)
        tok_k, len_k = vb.kept(ids); i, q, m, c = next_bytes(tok_k, len_k, pk, b"")
        assert abs(q.sum() - 1.0) < 1e-5 and len(set(i[q > 0].tolist())) == int((q > 0).sum()) and 0 < m <= 1.0 + 1e-6 and 0 < c <= 1.0, (i, q, m, c)
        # the true token in the kept set: its bytes are reachable at every prefix
        toks = llm.tokenize(b"The validators restarted after the upgrade.", add_bos=False, special=False)
        hits = 0; n = 0
        for t in toks:
            tb = llm.detokenize([t]); ids2 = np.concatenate([[t], ids[:511]]); pk2 = np.concatenate([[0.5], pk[:511] * 0.5 / pk[:511].sum()])
            tk, lk = vb.kept(ids2)
            for j in range(len(tb)):
                i2, q2, m2, _ = next_bytes(tk, lk, pk2, tb[:j]); n += 1; hits += int(m2 > 0 and tb[j] in set(i2[q2 > 0].tolist()))
        assert hits == n, (hits, n)
        print(f"real vocabulary ({llm.n_vocab():,} tokens): the kept distribution sums to 1; the true byte reachable at {hits}/{n} positions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); ap.add_argument("--model", default=None); a = ap.parse_args()
    if a.test: selftest(a.model)
