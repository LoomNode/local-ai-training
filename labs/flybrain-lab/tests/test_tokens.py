"""Model #2's token stream (notes/model2/checks): the stream round-trips the corpus, the layout lands on file units, the dev windows map to
token spans that cover their bytes."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "notes" / "model2" / "checks"
sys.path.insert(0, str(CHECKS))
pytest.importorskip("tokenizers")
from tok_common import TokenStream, dev_spans, file_units, id_bytes  # noqa: E402

PY = sys.executable


def make_corpus(path: Path, seed: int = 0, dev_seed: int = 0) -> dict:
    """Three files behind headers; tabs, CRLF, a run of 0xFF and a truncated multibyte sequence in the second; the third is the dev file."""
    rng = np.random.default_rng(seed)
    words = [b"int", b"return", b"foo_bar", b"x", b"y", b"if", b"else", b"for", b"\t", b" ", b"\n", b"(", b")", b"{", b"}", b";", b"=", b"+"]
    def text(n):
        return b"".join(words[i] for i in rng.integers(0, len(words), n))
    a = b"// ==== a.c ====\n" + text(20000) + b"\r\nend\n\n"
    b = b"// ==== b/c.py ====\n" + text(15000) + b"\xff\xff\xfe" + text(2000) + "é".encode()[:1] + b"\n" + text(3000) + b"\n\n"
    rng = np.random.default_rng(100 + dev_seed)
    c = b"// ==== dev/d.c ====\n" + text(12000) + b"\n\n"
    data = a + b + c
    path.write_bytes(data)
    layout = {"corpus": str(path), "corpus_bytes": len(data), "train_end": len(a) + len(b), "dev_offset": len(a) + len(b), "dev_bytes": len(c),
              "test_offset": len(data), "test_bytes": 0}
    json.dump(layout, open(path.with_suffix(".layout.json"), "w"))
    return layout


@pytest.fixture(scope="module")
def stream(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    corpus = d / "c.bytes"
    layout = make_corpus(corpus)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "RAYON_NUM_THREADS": "2"}
    subprocess.run([PY, str(CHECKS / "tokenizer_train.py"), "--corpus", str(corpus), "--layout", str(corpus.with_suffix(".layout.json")),
                    "--vocab", "600", "--out", str(d / "bpe.json")], check=True, env=env, capture_output=True)
    subprocess.run([PY, str(CHECKS / "tokenize_corpus.py"), "--corpus", str(corpus), "--layout", str(corpus.with_suffix(".layout.json")),
                    "--tokenizer", str(d / "bpe.json"), "--out", str(d / "stream"), "--chunk", "20000"], check=True, env=env, capture_output=True)
    from tokenizers import Tokenizer
    return TokenStream(str(d / "stream")), Tokenizer.from_file(str(d / "bpe.json")), corpus, layout


def test_stream_roundtrip(stream):
    st, tok, corpus, layout = stream
    data = corpus.read_bytes()
    idb = id_bytes(tok)
    assert st.decode(0, st.n, idb) == data
    assert int(st.tok_off[-1]) == len(data) and int(np.asarray(st.tok_len, dtype=np.int64).sum()) == len(data)
    assert len(set(idb)) == len(idb)                                   # one byte string per id
    assert int(st.tok_off[st.train_end_tok]) == layout["train_end"]     # the dev file's first token
    starts = set(int(x) for x in st.tok_off[:-1])
    for lo, hi in file_units(str(corpus)):
        assert lo in starts                                             # no token spans a file boundary
    assert st.layout["dev_tok"] == [st.train_end_tok, st.n]
    inv = data.index(b"\xff\xff\xfe")
    t0, t1 = st.span(inv, inv + 3)
    assert [int(st.tok_len[t]) for t in range(t0, t1)] == [1, 1, 1]     # invalid bytes are single-byte tokens


def test_dev_spans(stream):
    st, tok, corpus, layout = stream
    T, n = 64, 20
    spans = dev_spans(st, layout["dev_offset"], T, range(n))
    for k, (t0, t1) in enumerate(spans):
        assert t1 > t0
        b = st.bytes_in(t0, t1)
        assert abs(b - T) <= int(np.asarray(st.tok_len[t0:t1]).max())  # within one token of the window's bytes
        assert int(st.tok_off[t0]) >= layout["dev_offset"] + k * T
        if k:
            assert t0 == spans[k - 1][1]                                 # adjacent windows, adjacent spans
    assert sum(st.bytes_in(*s) for s in spans) == pytest.approx(n * T, abs=16)


def build_tables(st_dir: str, out: Path, log2: int = 12) -> dict:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run([PY, str(CHECKS / "tok_tables.py"), "--stream", st_dir, "--out", str(out), "--log2", str(log2), "--chunk", "500", "--jobs", "2",
                    "--save-tables"], check=True, env=env, capture_output=True)
    return json.load(open(out / "features.json"))


def test_tok_tables_prefix_honest(stream, tmp_path):
    """Every stored input of a held-out token equals the same input recomputed from the prefix alone and looked up in the final tables."""
    import tok_tables as tt
    sys.path.insert(0, str(ROOT / "notes" / "speed-round" / "checks")); sys.path.insert(0, str(ROOT))
    from counter_tables import counters
    from flybrain.corpus_memory import context_hashes, slot_and_tag
    from key_tables import compute_keys
    st, tok, corpus, layout = stream
    out = tmp_path / "feat"
    info = build_tables(st.dir, out)
    assert info["value_orders"] == list(tt.COUNTERS) and info["n_tokens"] == st.n
    tables = {n: np.load(out / f"table_n{n}.npy") for n in (*tt.NGRAM, *tt.KEYS)}
    pred = {n: np.load(out / f"pred_n{n}.npy") for n in info["orders"]}
    conf = {n: np.load(out / f"conf_n{n}.npy") for n in info["orders"]}
    assert pred[1].dtype == np.uint16 and conf[1].dtype == np.uint8
    data = np.frombuffer(corpus.read_bytes(), dtype=np.uint8)
    rng = np.random.default_rng(1)
    TE = st.train_end_tok
    hits = 0
    for i in sorted(rng.choice(np.arange(TE + 20, st.n), size=24, replace=False).tolist()):
        prefix = np.append(np.asarray(st.tokens[:i]), 0).astype(np.uint16)     # nothing at or after i
        h = context_hashes(prefix, i, i + 1, tt.NGRAM)
        p = int(st.tok_off[i])
        keys = compute_keys(data[:p + 1])
        for n in (*tt.NGRAM, *tt.KEYS):
            hv = h[n] if n in tt.NGRAM else np.asarray([keys[n][p]], dtype=np.uint64)
            slot, tag = slot_and_tag(hv, info["log2_slots"])
            pp, cc = tt.lookup(tables[n], slot, tag)
            assert (int(pp[0]), int(cc[0])) == (int(pred[n][i]), int(conf[n][i])), f"order {n} at token {i}"
            hits += int(cc[0] > 0)
        vals, _ = counters(data[:p + 1])
        for cid in tt.COUNTERS:
            assert int(pred[cid][i]) == min(int(vals[cid][p]), 127) and int(conf[cid][i]) == 1, f"counter {cid} at token {i}"
    assert hits > 24                                                          # the tables answer something on the held-out file
    # leak-free: another dev file leaves every training feature unchanged
    corpus2 = tmp_path / "c2.bytes"
    make_corpus(corpus2, dev_seed=1)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "RAYON_NUM_THREADS": "2"}
    subprocess.run([PY, str(CHECKS / "tokenize_corpus.py"), "--corpus", str(corpus2), "--layout", str(corpus2.with_suffix(".layout.json")),
                    "--tokenizer", str(Path(st.dir).parent / "bpe.json"), "--out", str(tmp_path / "stream2"), "--chunk", "20000"], check=True, env=env, capture_output=True)
    st2 = TokenStream(str(tmp_path / "stream2"))
    assert st2.train_end_tok == TE and np.array_equal(st2.tokens[:TE], st.tokens[:TE])
    build_tables(st2.dir, tmp_path / "feat2")
    for n in info["orders"]:
        assert np.array_equal(np.load(tmp_path / "feat2" / f"pred_n{n}.npy")[:TE], pred[n][:TE]), f"order {n} leaks"
        assert np.array_equal(np.load(tmp_path / "feat2" / f"conf_n{n}.npy")[:TE], conf[n][:TE]), f"order {n} leaks"
