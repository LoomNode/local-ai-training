"""Arm B (2026-11-06): the prompt pieces shared by the student scorer and the training-set builder, copied from armA/qwen_tokens.py so
that every prompt matches the 27B runs byte for byte: the span retriever over the world chunks, the chooser's choices file, the memory
block (headers, separator, order), the held-out windows with their session dates, and the 27B GGUF's tokeniser (vocab only, bytes in,
ids out) so that halves cut inside a UTF-8 character tokenise exactly as they did for the teacher. Numbers and ids only, never text."""
import codecs, glob, hashlib, json, os
import numpy as np

WORLD = "/games/ailab/corpora/kyle/world/files"; HELDOUT = "/games/ailab/corpora/kyle/heldout.bytes"; HELDJSON = "/games/ailab/corpora/kyle/heldout.json"
ASOF = "/games/ailab/corpora/kyle/asof_dates.json"; TEACHER_GGUF = "/vault/ailab/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf"
M = np.uint64(0x9E3779B97F4A7C15); DAY0 = np.datetime64("2000-01-01")

def day(d) -> int:
    """days since 2000-01-01 of a YYYY-MM-DD string; 0 when undated"""
    return int((np.datetime64(d) - DAY0).astype(int)) if d else 0

def span_hashes(b: bytes, S: int) -> np.ndarray:
    d = np.frombuffer(b, dtype=np.uint8).astype(np.uint64)
    if len(d) < S: return np.zeros(0, dtype=np.uint64)
    h = np.zeros(len(d) - S + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(S): h = h * M + d[j:len(d) - S + 1 + j] + np.uint64(1)
    return h

class Tokenizer:
    """the teacher GGUF's tokeniser: bytes -> ids, as the 27B runs tokenised"""
    def __init__(self, gguf=TEACHER_GGUF):
        from llama_cpp import Llama
        self.llm = Llama(model_path=gguf, vocab_only=True, verbose=False); self.bos = self.llm.token_bos()
    def __call__(self, b: bytes) -> list[int]:
        return self.llm.tokenize(b, add_bos=False, special=False) if b else []

class Retriever:
    """qwen_memory.py's span retriever: the world cut into overlapping chunks, each window's first half queried by 12-byte spans"""
    def __init__(self, world=WORLD, chunk=1500, span=12, top=2, min_hits=3, max_mem=3200, asof=ASOF):
        self.chunk, self.span, self.top, self.min_hits, self.max_mem = chunk, span, top, min_hits, max_mem
        chunks = []; hs = []; ids = []; paths = []; files = []
        for p in sorted(glob.glob(os.path.join(world, "**", "*"), recursive=True)):
            if not os.path.isfile(p): continue
            b = open(p, "rb").read(); rel = os.path.relpath(p, world); fi = len(files); files.append(rel)
            for s in range(0, max(1, len(b) - chunk // 2), chunk // 2 if len(b) > chunk else chunk):
                c = b[s:s + chunk]
                if len(c) < 64: continue
                h = np.unique(span_hashes(c, span)); hs.append(h); ids.append(np.full(len(h), len(chunks), dtype=np.int32)); chunks.append(c); paths.append(rel)
        H = np.concatenate(hs); I = np.concatenate(ids); order = np.argsort(H, kind="stable")
        self.H, self.I, self.chunks, self.paths, self.files = H[order], I[order], chunks, paths, files
        dates = json.load(open(asof))["files"] if asof else {}
        self.file_day = {f: day(dates.get(f, "")) for f in files}
        self.chunk_day = np.array([self.file_day[p] for p in paths], dtype=np.int64); self.chunk_file = np.array([files.index(p) if False else 0 for p in paths])
        fidx = {f: i for i, f in enumerate(files)}; self.chunk_file = np.array([fidx[p] for p in paths], dtype=np.int64)
    def digest(self) -> str:
        return hashlib.sha256(b"".join(hashlib.sha256(c).digest() for c in self.chunks)).hexdigest()
    def retrieve(self, prefix: bytes, exclude_bytes: bytes = b"", before_day: int = 0, exclude_file: int = -1) -> list[int]:
        """chunk ids for a query, as qwen_memory.py chose them; optionally only chunks of files created before `before_day`
        (0 = any date) and never chunks of `exclude_file`"""
        q = np.unique(span_hashes(prefix, self.span))
        lo = np.searchsorted(self.H, q, "left"); hi = np.searchsorted(self.H, q, "right")
        counts = np.zeros(len(self.chunks), dtype=np.int32)
        for l_, h_ in zip(lo, hi):
            if h_ > l_: np.add.at(counts, self.I[l_:h_], 1)
        if before_day: counts[(self.chunk_day >= before_day) & (self.chunk_day > 0)] = 0
        if exclude_file >= 0: counts[self.chunk_file == exclude_file] = 0
        best = np.argsort(-counts)[: self.top * 3]; out = []; total = 0
        for i in best:
            if counts[i] < self.min_hits or len(out) >= self.top: break
            c = self.chunks[i]
            if exclude_bytes and (c[:200] in exclude_bytes or exclude_bytes[:200] in c): continue
            out.append(int(i)); total += len(c)
            if total >= self.max_mem: break
        return out
    def memory(self, ids: list[int], headers: bool, sep: bytes = b"\n\n", order: str = "top-first") -> bytes:
        if order == "top-last": ids = ids[::-1]
        mem = [((b"## " + self.paths[i].encode("utf-8", "replace") + b"\n") if headers else b"") + self.chunks[i] for i in ids]
        return b"".join(c + sep for c in mem)

def load_choices(path: str, r: Retriever):
    ch = json.load(open(path))
    if ch["pool"]["digest"] != r.digest() or ch["pool"]["chunks"] != len(r.chunks): raise SystemExit(f"{path} was chosen from another chunk pool")
    return ch["choices"], ch["chooser"]

def sep_bytes(s: str) -> bytes: return codecs.decode(s, "unicode_escape").encode()

def windows(T=2048, kyle_every=1, dlg_every=4, heldout=HELDOUT, heldjson=HELDJSON, asof=ASOF):
    """yields (set name 'kyle'|'dlg', window index, session date, first half bytes, second half bytes, whole window bytes)"""
    held = open(heldout, "rb").read(); meta = json.load(open(heldjson)); a = json.load(open(asof)) if asof else {}
    for name, key, every, dates in (("kyle", "kyle", kyle_every, a.get("kyle_windows", [])), ("dlg", "dialogue", dlg_every, a.get("dialogue_windows", []))):
        off, nwin = meta[key]["offset"], meta[key]["windows"]
        for w in range(0, nwin, every):
            W = held[off + w * T: off + (w + 1) * T]
            yield name, w, (dates[w] if w < len(dates) else ""), W[:T // 2], W[T // 2:], W
