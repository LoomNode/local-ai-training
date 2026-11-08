"""Bet E, second build: PAQ-style bit tables with a 16-bit check per slot and a width-4 beam (flybrain-lab speed round, 2026-10-08).

The first build (paq_bits.py, results/mem/feat_paqbits) had untagged 2-byte slots: at 2^28 per order every order from 8 up was saturated by
collisions. The coordinator's three changes:

1. Check. A slot is (tag u16, n0 u8, n1 u8), 4 bytes, 2^<log2> per order. With h = corpus_memory._finalise(ctx * 0x9E3779B97F4A7C15 +
   node * 0xC2B2AE3D27D4EB4F): slot = the top <log2> bits of h, tag = its low 16 bits (disjoint bits, as corpus_memory.slot_and_tag). A
   slot is empty when n0 + n1 == 0 (a used slot always holds a count >= 1).
   Update (training positions in corpus order, each position's 8 nodes root first): an empty slot or one with the node's tag is used; one
   with another tag is evicted (tag replaced, counts reset to 0, 0); then the brief's rule: if n_b == 255 halve both; n_b += 1; if
   n_other > 2: n_other = n_other // 2 + 1.
   Lookup: any node the walk reads that is empty or holds another tag makes the whole lookup a miss: pred 0, conf 0 for both candidates.
2. Beam. From the root, each kept partial path (node, product) is extended by the bits OBSERVED at its node (n_bit > 0: an unobserved bit
   has no node of this context below it, so it is not a path), with p(bit) = (n_bit + 0.4) / (n0 + n1 + 0.8); of the extensions the 4 with
   the highest product are kept (ties: generation order = beam order, then bit 0 before bit 1). After the 8 levels the best complete path
   is the top byte and the second best, when there is one, the second byte. The nodes the beam reads are "the walk" of the miss rule. The
   greedy path (the more probable bit at each node, ties -> 0) is walked alongside for the diagnostic "beam top != greedy top" only.
3. Bins. Top: the brief's (1: p < 0.5, 2: < 0.75, 4: < 0.9, 16: >= 0.9). Second, by its own p2 (< 0.5, since p1 + p2 <= 1 and p2 <= p1):
   1: p2 < 0.1, 2: < 0.25, 4: < 0.4, 16: >= 0.4; conf2 = 0 on a miss or when the context has no second observed path.

Everything else as paq_bits.py: the eleven lookup orders, chunks of 2^20 cut at train_end, a chunk's lookups before its writes, positions
p < o skipped, features pred/conf_n<o> and pred/conf_n<5000+o>, counters 1001..1007 linked. The key hashes are read with pread (not
memory-mapped) so their 8.8 GB do not count in the process's RSS.

usage:
  paq_bits2.py <log2> <out dir>     full build; stats to results/tables/stats_<basename>.json (with the false-hit rates of feat_paqbits,
                                    this build and feat_all28 on the same held-out bytes, and the beam-vs-greedy rate)
  paq_bits2.py --selftest           hand check of the tag path (an eviction, a foreign-tag miss), kernel vs scalar reference at chunk
                                    1 / 256 / 1000, leak tests (log2 12)
  paq_bits2.py --seen               cache, per order, which held-out positions' contexts occur in the training bytes (for false hits)
  paq_bits2.py --dry <log2> <MiB>   the first <MiB> MiB (last MiB held out): this build and the first side by side, against feat_all28
At 2^28 the eleven tables take 11 GB of RAM."""
import ctypes
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paq_bits as v1  # noqa: E402  (the first build: constants, helpers, and its kernel for the before/after comparison)
from paq_bits import (BINS, BYTE_ORDERS, C1, C2, CHUNK, CORPUS, COUNTERS, HELDOUT, KEY_IDS, MASK, MEM, ORDERS, SECOND,  # noqa: E402
                      TABLES_DIR, TRAIN_END, WORKERS, FileSink, _rate, avail_gb, count_stats, fin_py, fmt2, peak_rss_gb,
                      refuse_if_in_use, rss_gb, toy_key, toy_words)
from flybrain.corpus_memory import context_hashes, hash_context  # noqa: E402

BEAM = 4
NDIAG = 64
SLOT2 = np.dtype([("tag", "<u2"), ("n0", "u1"), ("n1", "u1")])
SEEN_DIR = f"{HERE}/seen"

C_SRC = r"""
#include <stdint.h>
#include <stddef.h>

#define M1 0xFF51AFD7ED558CCDULL      /* corpus_memory._finalise */
#define M2 0xC4CEB9FE1A85EC53ULL
#define C1 0x9E3779B97F4A7C15ULL      /* h = fin(ctx * C1 + node * C2): slot = top bits, tag = low 16 bits */
#define C2 0xC2B2AE3D27D4EB4FULL
#define G 8                           /* positions walked in lockstep by the lookup */
#define D 8                           /* update prefetch distance, positions */
#define W 4                           /* beam width */

typedef struct { uint16_t tag; uint8_t n0, n1; } slot_t;

static inline uint64_t fin(uint64_t h) {
    h ^= h >> 33; h *= M1; h ^= h >> 33; h *= M2; h ^= h >> 33;
    return h;
}
static inline uint8_t bin1(double p) { return p < 0.5 ? 1 : (p < 0.75 ? 2 : (p < 0.9 ? 4 : 16)); }
static inline uint8_t bin2(double p) { return p < 0.1 ? 1 : (p < 0.25 ? 2 : (p < 0.4 ? 4 : 16)); }
static inline int bidx(uint8_t b) { return b == 1 ? 0 : (b == 2 ? 1 : (b == 4 ? 2 : 3)); }

/* diag, positions i >= stats_lo: [0] positions, [1] misses, [2] misses with the root present (the walk broke below it), [3] hits,
   [4+x] top hits in bin x, [8+x] sum of p1, [12+x] top right; [16] second present, [17+x] second in bin2 x, [21+x] sum of p2,
   [25+x] second right; [29] greedy path broken (an absent node off the beam), [30] beam top != greedy top, [31] beam top right among
   those, [32] greedy top right among those; [33+d] top hits with p1 in decile d, [43+d] right among them.
   Updates (training chunks, all positions): [54] node updates, [55] empty slots claimed, [56] foreign slots evicted. */
static void lookup(const slot_t *tab, int log2, const uint64_t *ctx, const uint8_t *nxt, int64_t lo, int64_t hi, int64_t stats_lo,
                   uint8_t *pred, uint8_t *conf, uint8_t *pred2, uint8_t *conf2, double *p1o, double *p2o, int16_t *gro, double *diag)
{
    const int shift = 64 - log2;
    for (int64_t i0 = lo; i0 < hi; i0 += G) {
        const int g = (hi - i0) < G ? (int)(hi - i0) : G;
        uint64_t base[G], bnode[G][W], gnode[G], addr[G][W + 1];
        uint16_t tg[G][W + 1];
        double bprod[G][W];
        int nb[G], alive[G], gok[G], rootok[G];
        for (int j = 0; j < g; j++) {
            base[j] = ctx[i0 + j] * C1;
            nb[j] = 1; bnode[j][0] = 1; bprod[j][0] = 1.0; gnode[j] = 1;
            alive[j] = 1; gok[j] = 1; rootok[j] = 0;
        }
        for (int lvl = 0; lvl < 8; lvl++) {
            for (int j = 0; j < g; j++) {                  /* this level's addresses: the beam's nodes, then the greedy node */
                if (!alive[j]) continue;
                for (int i = 0; i < nb[j]; i++) {
                    const uint64_t h = fin(base[j] + bnode[j][i] * C2);
                    addr[j][i] = h >> shift; tg[j][i] = (uint16_t)h;
                    __builtin_prefetch(tab + addr[j][i]);
                }
                if (gok[j]) {
                    const uint64_t h = fin(base[j] + gnode[j] * C2);
                    addr[j][W] = h >> shift; tg[j][W] = (uint16_t)h;
                    __builtin_prefetch(tab + addr[j][W]);
                }
            }
            for (int j = 0; j < g; j++) {
                if (!alive[j]) continue;
                uint64_t cn[2 * W];
                double cp[2 * W];
                int nc = 0;
                for (int i = 0; i < nb[j]; i++) {
                    const slot_t e = tab[addr[j][i]];
                    if ((e.n0 | e.n1) == 0 || e.tag != tg[j][i]) { alive[j] = 0; break; }      /* empty or foreign: a miss */
                    const double tot = (double)(e.n0 + e.n1) + 0.8;
                    for (int b = 0; b < 2; b++) {
                        const unsigned c = b ? e.n1 : e.n0;
                        if (!c) continue;                                                   /* unobserved bit: no path */
                        const double p = bprod[j][i] * ((c + 0.4) / tot);
                        int k = nc++;
                        while (k > 0 && cp[k - 1] < p) { cp[k] = cp[k - 1]; cn[k] = cn[k - 1]; k--; }   /* stable, descending */
                        cp[k] = p; cn[k] = 2 * bnode[j][i] + b;
                    }
                }
                if (!alive[j]) continue;
                if (lvl == 0) rootok[j] = 1;
                nb[j] = nc < W ? nc : W;
                for (int i = 0; i < nb[j]; i++) { bnode[j][i] = cn[i]; bprod[j][i] = cp[i]; }
                if (gok[j]) {
                    const slot_t e = tab[addr[j][W]];
                    if ((e.n0 | e.n1) == 0 || e.tag != tg[j][W]) gok[j] = 0;
                    else gnode[j] = 2 * gnode[j] + (e.n1 > e.n0);
                }
            }
        }
        for (int j = 0; j < g; j++) {
            const int64_t i = i0 + j;
            const int st = i >= stats_lo;
            if (!alive[j]) {
                pred[i] = 0; conf[i] = 0; pred2[i] = 0; conf2[i] = 0;
                if (p1o) { p1o[i] = 0.0; p2o[i] = 0.0; gro[i] = -2; }
                if (st) { diag[0] += 1; diag[1] += 1; if (rootok[j]) diag[2] += 1; }
                continue;
            }
            const uint8_t top = (uint8_t)(bnode[j][0] - 256);
            const double p1 = bprod[j][0];
            const int has2 = nb[j] >= 2;
            const uint8_t sec = has2 ? (uint8_t)(bnode[j][1] - 256) : 0;
            const double p2 = has2 ? bprod[j][1] : 0.0;
            const uint8_t b1 = bin1(p1), b2 = has2 ? bin2(p2) : 0;
            const int gb = gok[j] ? (int)(gnode[j] - 256) : -1;
            pred[i] = top; conf[i] = b1; pred2[i] = sec; conf2[i] = b2;
            if (p1o) { p1o[i] = p1; p2o[i] = p2; gro[i] = (int16_t)gb; }
            if (st) {
                const int r1 = top == nxt[i], r2 = has2 && sec == nxt[i], x1 = bidx(b1);
                diag[0] += 1; diag[3] += 1;
                diag[4 + x1] += 1; diag[8 + x1] += p1; diag[12 + x1] += r1;
                if (has2) { const int x2 = bidx(b2); diag[16] += 1; diag[17 + x2] += 1; diag[21 + x2] += p2; diag[25 + x2] += r2; }
                if (gb < 0) diag[29] += 1;
                else if (gb != top) { diag[30] += 1; diag[31] += r1; diag[32] += gb == nxt[i]; }
                int d = (int)(p1 * 10.0); if (d > 9) d = 9;
                diag[33 + d] += 1; diag[43 + d] += r1;
            }
        }
    }
}

static inline void slots8(const slot_t *tab, uint64_t ctx, unsigned byte, int shift, uint64_t *a, uint16_t *t) {
    const uint64_t base = ctx * C1;
    uint64_t node = 1;
    for (int lvl = 0; lvl < 8; lvl++) {
        const uint64_t h = fin(base + node * C2);
        a[lvl] = h >> shift; t[lvl] = (uint16_t)h;
        __builtin_prefetch(tab + a[lvl], 1);
        node = 2 * node + ((byte >> (7 - lvl)) & 1);
    }
}

/* positions lo .. hi-1 one after another, each one's 8 nodes root first: the sequential rule, with claim / evict before the count */
static void update(slot_t *tab, int log2, const uint64_t *ctx, const uint8_t *nxt, int64_t lo, int64_t hi, double *diag)
{
    const int shift = 64 - log2;
    uint64_t ra[D][8];
    uint16_t rt[D][8];
    double claimed = 0, evicted = 0;
    for (int64_t f = lo; f < hi && f < lo + D; f++) slots8(tab, ctx[f], nxt[f], shift, ra[f % D], rt[f % D]);
    for (int64_t i = lo; i < hi; i++) {
        const unsigned byte = nxt[i];
        for (int lvl = 0; lvl < 8; lvl++) {
            const unsigned b = (byte >> (7 - lvl)) & 1;
            slot_t *e = tab + ra[i % D][lvl];
            const uint16_t t = rt[i % D][lvl];
            unsigned n0 = e->n0, n1 = e->n1;
            if ((n0 | n1) == 0) { e->tag = t; claimed += 1; }
            else if (e->tag != t) { e->tag = t; n0 = 0; n1 = 0; evicted += 1; }
            unsigned nb = b ? n1 : n0, no = b ? n0 : n1;
            if (nb == 255) { nb >>= 1; no >>= 1; }
            nb += 1;
            if (no > 2) no = no / 2 + 1;
            if (b) { e->n1 = (uint8_t)nb; e->n0 = (uint8_t)no; } else { e->n0 = (uint8_t)nb; e->n1 = (uint8_t)no; }
        }
        if (i + D < hi) slots8(tab, ctx[i + D], nxt[i + D], shift, ra[i % D], rt[i % D]);
    }
    diag[54] += 8.0 * (double)(hi > lo ? hi - lo : 0); diag[55] += claimed; diag[56] += evicted;
}

void paq2_chunk(slot_t *tab, int log2, const uint64_t *ctx, const uint8_t *nxt, int64_t first, int64_t m, int do_update, int64_t stats_lo,
                uint8_t *pred, uint8_t *conf, uint8_t *pred2, uint8_t *conf2, double *p1, double *p2, int16_t *gr, double *diag)
{
    lookup(tab, log2, ctx, nxt, first, m, stats_lo, pred, conf, pred2, conf2, p1, p2, gr, diag);
    if (do_update) update(tab, log2, ctx, nxt, first, m, diag);
}
"""

_LIB = None


def kernel2():
    global _LIB
    if _LIB is None:
        tag = hashlib.sha1(C_SRC.encode()).hexdigest()[:12]
        so = f"{HERE}/_paq_bits2_{tag}.so"
        if not os.path.exists(so):
            src = f"{HERE}/_paq_bits2_{tag}.c"
            with open(src, "w") as f:
                f.write(C_SRC)
            subprocess.run(["gcc", "-O2", "-fPIC", "-shared", "-ffp-contract=off", "-o", so + ".tmp", src], check=True)
            os.replace(so + ".tmp", so)
        lib = ctypes.CDLL(so)
        vp, i64 = ctypes.c_void_p, ctypes.c_int64
        lib.paq2_chunk.restype = None
        lib.paq2_chunk.argtypes = [vp, ctypes.c_int, vp, vp, i64, i64, ctypes.c_int, i64, vp, vp, vp, vp, vp, vp, vp, vp]
        _LIB = lib
    return _LIB


def _ptr(a):
    return None if a is None else a.ctypes.data


def run_order2(table, log2, h, nxt, first, m, update, stats_lo, out, dd):
    pred, conf, pred2, conf2, p1, p2, gr = out
    kernel2().paq2_chunk(_ptr(table), log2, _ptr(h), _ptr(nxt), first, m, int(update), stats_lo,
                         _ptr(pred), _ptr(conf), _ptr(pred2), _ptr(conf2), _ptr(p1), _ptr(p2), _ptr(gr), _ptr(dd))


class KeyReader:
    """The key orders' precomputed hashes, read with pread into fresh arrays (page cache, not this process's RSS)."""
    def __init__(self, kids):
        self.f = {}
        for kid in kids:
            path = f"{MEM}/feat_keys/key_n{kid}.npy"
            mm = np.load(path, mmap_mode="r")
            assert mm.dtype == np.uint64 and mm.ndim == 1
            off, n = mm.offset, len(mm)
            del mm
            self.f[kid] = (os.open(path, os.O_RDONLY), off, n)

    def get(self, kid, lo, hi):
        fd, off, n = self.f[kid]
        assert 0 <= lo <= hi <= n
        buf = np.empty(hi - lo, np.uint64)
        mv = memoryview(buf).cast("B")
        done = 0
        while done < len(mv):
            k = os.preadv(fd, [mv[done:done + (1 << 28)]], off + 8 * lo + done)
            if k <= 0:
                raise IOError(f"short read of key {kid}")
            done += k
        return buf


def hashes_fn2(data, reader):
    def hashes_of(lo, hi):
        h = context_hashes(data, lo, hi, BYTE_ORDERS)
        for kid in KEY_IDS:
            h[kid] = reader.get(kid, lo, hi)
        return h
    return hashes_of


def build2(data, hashes_of, orders, log2, chunk, train_end, sink, raw=False, stats_from=None, workers=WORKERS, log=None):
    """paq_bits.build with the tagged slots and the beam: same chunks, same leak-free order. Returns (tables, diag)."""
    N = len(data)
    kernel2()
    tables = {n: np.zeros(4 << log2, dtype=np.uint8) for n in orders}       # slot s = (tag u16, n0, n1) at [4s, 4s + 4)
    diag = {n: np.zeros(NDIAG) for n in orders}
    bounds = sorted(set(range(0, N, chunk)) | {train_end, N})
    spans = list(zip(bounds[:-1], bounds[1:]))
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        upcoming = hashes_of(*spans[0])
        for i, (lo, hi) in enumerate(spans):
            hashes, m = upcoming, hi - lo
            nxt = np.ascontiguousarray(data[lo:hi], dtype=np.uint8)
            stats_lo = m if stats_from is None else min(max(stats_from - lo, 0), m)
            out, dd, futs = {}, {}, []
            for n in orders:
                first = min(max(lo, n) - lo, m)
                out[n] = [np.zeros(m, np.uint8) for _ in range(4)] + ([np.zeros(m), np.zeros(m), np.full(m, -2, np.int16)] if raw else [None] * 3)
                dd[n] = np.zeros(NDIAG)
                if first < m:
                    h = np.ascontiguousarray(hashes[n], dtype=np.uint64)
                    assert h.shape == (m,)
                    futs.append(pool.submit(run_order2, tables[n], log2, h, nxt, first, m, hi <= train_end, stats_lo, out[n], dd[n]))
            upcoming = hashes_of(*spans[i + 1]) if i + 1 < len(spans) else None
            for f in futs:
                f.result()
            for n in orders:
                diag[n] += dd[n]
            sink(lo, hi, out)
            if log is not None and (i % 20 == 0 or hi == N):
                log(f"{hi:,} / {N:,}  {time.time() - t0:.0f} s  rss {rss_gb():.1f} GB  available {avail_gb():.1f} GB")
    return tables, diag


class MemSink2:
    def __init__(self, N, orders, raw=False):
        self.f = {n: np.zeros((4, N), np.uint8) for n in orders}
        self.p = {n: np.zeros((2, N)) for n in orders} if raw else None
        self.g = {n: np.full(N, -2, np.int16) for n in orders} if raw else None

    def __call__(self, lo, hi, out):
        for n, arrs in out.items():
            for k in range(4):
                self.f[n][k, lo:hi] = arrs[k]
            if self.p is not None:
                self.p[n][0, lo:hi], self.p[n][1, lo:hi], self.g[n][lo:hi] = arrs[4], arrs[5], arrs[6]


def occupancy2(table):
    v = table.view(SLOT2)
    return float(np.count_nonzero(v["n0"] | v["n1"])) / len(v)


# ---------------------------------------------------------------------------------------------------------------------------------
# scalar reference (Python ints and floats, one position at a time, table as an int64 array [S, 3] = tag, n0, n1)

def ref2_hash(ctx, node, log2):
    h = fin_py((ctx * C1 + node * C2) & MASK)
    return h >> (64 - log2), h & 0xFFFF


def bin1(p):
    return 1 if p < 0.5 else 2 if p < 0.75 else 4 if p < 0.9 else 16


def bin2(p):
    return 1 if p < 0.1 else 2 if p < 0.25 else 4 if p < 0.4 else 16


def ref2_update(tab, ctx, byte, log2, touched=None, events=None):
    node = 1
    for lvl in range(8):
        b = (byte >> (7 - lvl)) & 1
        s, t = ref2_hash(ctx, node, log2)
        tg, n0, n1 = (int(x) for x in tab[s])
        if n0 + n1 == 0:
            tg = t
            if events is not None:
                events.append(("claim", s, ctx, node))
        elif tg != t:
            if events is not None:
                events.append(("evict", s, ctx, node, tg))
            tg, n0, n1 = t, 0, 0
        nb, no = (n1, n0) if b else (n0, n1)
        if nb + 1 > 255:
            nb, no = nb // 2, no // 2
        nb += 1
        if no > 2:
            no = no // 2 + 1
        n0, n1 = (no, nb) if b else (nb, no)
        tab[s] = (tg, n0, n1)
        if touched is not None:
            touched.setdefault(s, []).append((ctx, node))
        node = 2 * node + b


def ref2_lookup(tab, ctx, log2):
    """None for a miss, else (top, p1, second or None, p2, greedy byte or -1). Beam first (with the miss rule), greedy walked after."""
    def read(node):
        s, t = ref2_hash(ctx, node, log2)
        tg, n0, n1 = (int(x) for x in tab[s])
        return None if (n0 + n1 == 0 or tg != t) else (n0, n1)
    beam = [(1, 1.0)]
    for _ in range(8):
        cands = []
        for node, P in beam:
            r = read(node)
            if r is None:
                return None
            n0, n1 = r
            tot = (n0 + n1) + 0.8
            for b, c in ((0, n0), (1, n1)):
                if c > 0:
                    cands.append((2 * node + b, P * ((c + 0.4) / tot)))
        beam = sorted(cands, key=lambda x: -x[1])[:BEAM]                   # stable: ties keep generation order
    top, p1 = beam[0][0] - 256, beam[0][1]
    sec, p2 = (beam[1][0] - 256, beam[1][1]) if len(beam) > 1 else (None, 0.0)
    node = 1
    for _ in range(8):
        r = read(node)
        if r is None:
            return top, p1, sec, p2, -1
        node = 2 * node + (1 if r[1] > r[0] else 0)
    return top, p1, sec, p2, node - 256


def ref2_build(data, H, orders, log2, chunk, train_end):
    """Positions one at a time; a lookup reads a snapshot of its table taken when its chunk began; training updates go to the live table."""
    N = len(data)
    tabs = {n: np.zeros((1 << log2, 3), dtype=np.int64) for n in orders}
    f = {n: np.zeros((4, N), np.uint8) for n in orders}
    p = {n: np.zeros((2, N)) for n in orders}
    g = {n: np.full(N, -2, np.int16) for n in orders}
    bounds = sorted(set(range(0, N, chunk)) | {train_end, N})
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        snaps = {n: tabs[n].copy() for n in orders}
        for pos in range(lo, hi):
            for n in orders:
                if pos < n:
                    continue
                ctx = int(H[n][pos])
                r = ref2_lookup(snaps[n], ctx, log2)
                if r is not None:
                    top, p1, sec, p2, gb = r
                    f[n][:, pos] = (top, bin1(p1), 0 if sec is None else sec, 0 if sec is None else bin2(p2))
                    p[n][:, pos] = (p1, p2)
                    g[n][pos] = gb
                if hi <= train_end:
                    ref2_update(tabs[n], ctx, int(data[pos]), log2)
    return f, p, g, tabs


# ---------------------------------------------------------------------------------------------------------------------------------
# verification

def toy_build2(data, log2, chunk, train_end, orders, raw=True):
    keys = {116: toy_key(data)} if 116 in orders else {}
    sink = MemSink2(len(data), orders, raw=raw)
    tables, diag = build2(data, v1.hashes_fn(data, keys), orders, log2, chunk, train_end, sink, raw=raw)
    return sink, tables, diag


def path_nodes(byte):
    out, node = [], 1
    for lvl in range(8):
        out.append(node)
        node = 2 * node + ((byte >> (7 - lvl)) & 1)
    return out


def selftest2():
    log2, ok = 12, True

    def check(cond, msg):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
        ok = ok and bool(cond)

    # 1a. hand check of the tag path: context A's root shares its slot with a deeper node of context B (other tag)
    print("== 1a: hand check of the tag path (order 4, log2 12)", flush=True)
    a, bb = ord("a"), ord("b")
    pa, pb = path_nodes(a), path_nodes(bb)
    cands_a = [bytes(c) for c in itertools.product(b"KQJVPMRT", repeat=4)]     # A and B from disjoint letters (and neither has a or b),
    cands_b = [bytes(c) for c in itertools.product(b"GLNSWXYZ", repeat=4)]     # so the windows across the joins are never A or B
    deep = {}                                                   # slot -> [(B, node, tag)] for the nodes of 'b' below B's root
    for B in cands_b:
        cb = hash_context(B)
        for node in pb[1:]:
            s, t = ref2_hash(cb, node, log2)
            deep.setdefault(s, []).append((B, node, t))
    found = None
    for A in cands_a:
        ca = hash_context(A)
        sA, tA = ref2_hash(ca, 1, log2)
        for B, nodeB, tB in deep.get(sA, []):
            if tB == tA:
                continue
            seq = (A + b"a") * 3 + B + b"b" + A + b"a"
            held = A + b"a" + B + b"b" + b"ZZZZc"
            data = np.frombuffer(seq + held, dtype=np.uint8).copy()
            te = len(seq)
            H = context_hashes(data, 0, len(data), (4,))
            touched, rt, ev = {}, np.zeros((1 << log2, 3), np.int64), []
            for pos in range(4, te):
                ref2_update(rt, int(H[4][pos]), int(data[pos]), log2, touched, ev)
            cb = hash_context(B)
            want = {ref2_hash(ca, nd, log2)[0]: {(ca, nd)} for nd in pa}
            for nd in pb:
                want.setdefault(ref2_hash(cb, nd, log2)[0], set()).add((cb, nd))
            clean = all(set(touched.get(s, [])) == w for s, w in want.items()) and len(want) == 15
            if clean:
                found = (A, B, nodeB, ca, cb, sA, tA, tB, data, te, rt, ev)
                break
        if found:
            break
    check(found is not None, "found contexts A, B whose slots collide only where intended (A's root = one of B's deeper nodes)")
    A, B, nodeB, ca, cb, sA, tA, tB, data, te, rt, ev = found
    print(f"  A = {A!r} (root slot {sA}, tag {tA:#06x}), B = {B!r} (node {nodeB} of 'b' in the same slot, tag {tB:#06x}); "
          f"training bytes: (A a) x3, B b, A a; held out: A a, B b, ZZZZ c", flush=True)
    evs = [e for e in ev if e[0] == "evict" and e[1] == sA]
    print(f"  evictions of slot {sA} during training (reference): {[('by ' + ('A' if e[2] == ca else 'B') + f' node {e[3]}') for e in evs]}", flush=True)
    sink, tables, diag = toy_build2(data, log2, 256, te, (4,))
    tab = tables[4].view(SLOT2)
    got = lambda s: (int(tab["tag"][s]), int(tab["n0"][s]), int(tab["n1"][s]))
    check(got(sA) == (tA, 1, 0) and len(evs) == 2,
          f"slot {sA}: (tag, n0, n1) = ({got(sA)[0]:#06x}, {got(sA)[1]}, {got(sA)[2]}); by hand (A's tag, 1, 0): A a x3 -> (A, 3, 0), B b evicts "
          f"-> (B, counts of its node {nodeB}), A a evicts again -> (A, 0, 0) + bit 0 -> (A, 1, 0); two evictions")
    for nd in pa[1:]:
        s, t = ref2_hash(ca, nd, log2)
        bit = (a >> (7 - (len(bin(nd)) - 3))) & 1
        check(got(s) == (t, 0 if bit else 4, 4 if bit else 0), f"A's node {nd}: {got(s)[1:]} with A's tag; by hand 4 on bit {bit} (not evicted)")
    for nd in pb:
        s, t = ref2_hash(cb, nd, log2)
        if nd == nodeB:
            check(got(s)[0] == tA and got(s)[0] != t, f"B's node {nd}: the slot now holds A's tag ({got(s)[0]:#06x}), not B's ({t:#06x})")
        else:
            bit = (bb >> (7 - (len(bin(nd)) - 3))) & 1
            check(got(s) == (t, 0 if bit else 1, 1 if bit else 0), f"B's node {nd}: {got(s)[1:]} with B's tag; by hand 1 on bit {bit}")
    posA, posB, posC = te + 4, te + 9, te + 14
    fA, pA, gA = sink.f[4][:, posA], sink.p[4][:, posA], sink.g[4][posA]
    pexp = 1.0
    for fct in [(1 + 0.4) / (1 + 0.8)] + [(4 + 0.4) / (4 + 0.8)] * 7:
        pexp = pexp * fct
    check(int(fA[0]) == a and int(fA[1]) == bin1(pexp) and pA[0] == pexp and int(fA[2]) == 0 and int(fA[3]) == 0 and gA == a,
          f"lookup A (held out): top {chr(fA[0])!r} p = {pA[0]:.12f}, by hand (1.4/1.8)(4.4/4.8)^7 = {pexp:.12f}, conf {fA[1]}; one observed path, "
          f"so no second (pred2 {fA[2]}, conf2 {fA[3]}); greedy {chr(gA)!r}")
    fB = sink.f[4][:, posB]
    sBr, tBr = ref2_hash(cb, 1, log2)
    check(not fB.any() and got(sBr)[0] == tBr and got(sBr)[1] + got(sBr)[2] > 0,
          f"lookup B (held out): B's root is there (slot {sBr} = {got(sBr)}), but its node {nodeB} holds A's tag: the whole lookup "
          f"misses, features {fB.tolist()}")
    fC = sink.f[4][:, posC]
    sC, _ = ref2_hash(hash_context(b"ZZZZ"), 1, log2)
    check(not fC.any(), f"lookup ZZZZ (never seen; its root slot {sC} holds {got(sC)}): miss, features {fC.tolist()}")
    rf, rp, rg, rtab = ref2_build(data, {4: context_hashes(data, 0, len(data), (4,))[4]}, (4,), log2, 256, te)
    check(np.array_equal(sink.f[4], rf[4]) and np.array_equal(sink.p[4], rp[4]) and np.array_equal(sink.g[4], rg[4])
          and np.array_equal(np.stack([tab["tag"], tab["n0"], tab["n1"]], 1).astype(np.int64), rtab[4]),
          "C kernel == scalar reference on this toy (features, raw p, greedy, table)")

    # 1b. chunk size 1 vs 256 vs 1000, kernel vs scalar reference, on the word toy with every byte order and a key order
    print("== 1b: chunk 1 vs 256 vs 1000, against the scalar reference (4000 bytes, train_end 3000, log2 12)", flush=True)
    data = toy_words(4000, 0)
    N, te, orders = len(data), 3000, (*BYTE_ORDERS, 116)
    H = context_hashes(data, 0, N, BYTE_ORDERS)
    H[116] = toy_key(data)
    runs = {}
    for chunk in (1, 256, 1000):
        sink, tables, diag = toy_build2(data, log2, chunk, te, orders)
        rf, rp, rg, rt = ref2_build(data, H, orders, log2, chunk, te)
        same_f = all(np.array_equal(sink.f[n], rf[n]) for n in orders)
        same_p = all(np.array_equal(sink.p[n], rp[n]) and np.array_equal(sink.g[n], rg[n]) for n in orders)
        same_t = all(np.array_equal(np.stack([tables[n].view(SLOT2)[k] for k in ("tag", "n0", "n1")], 1).astype(np.int64), rt[n]) for n in orders)
        hits = sum(int((sink.f[n][1] > 0).sum()) for n in orders)
        sec = sum(int((sink.f[n][3] > 0).sum()) for n in orders)
        ev = sum(int(diag[n][56]) for n in orders)
        diff = sum(int(((sink.g[n] >= 0) & (sink.g[n] != sink.f[n][0])).sum()) for n in orders)
        check(same_f and same_p and same_t, f"chunk {chunk:4d}: C kernel == scalar reference (features, raw p1/p2, greedy byte, final tags and "
              f"counts); {hits} hits, {sec} with a second, {ev} evictions, beam top != greedy top at {diff}")
        runs[chunk] = (sink, tables)
    s1, t1 = runs[1]
    for chunk in (256, 1000):
        s, t = runs[chunk]
        same_t = all(np.array_equal(t[n], t1[n]) for n in orders)
        same_h = all(np.array_equal(s.f[n][:, te:], s1.f[n][:, te:]) and np.array_equal(s.p[n][:, te:], s1.p[n][:, te:]) for n in orders)
        diff_train = sum(int((s.f[n][:, :te] != s1.f[n][:, :te]).any(axis=0).sum()) for n in orders)
        check(same_t and same_h, f"chunk {chunk} vs chunk 1: final tables identical {same_t}, held-out features identical {same_h} "
              f"(training features differ at {diff_train} position-orders, by design)")
    vals = set()
    for n in orders:
        vals |= set(np.unique(s1.f[n][1]).tolist()) | set(np.unique(s1.f[n][3]).tolist())
    check(vals <= {0, 1, 2, 4, 16} and all((s1.p[n][1] < 0.5).all() for n in orders),
          f"conf values {sorted(vals)}, p2 < 0.5 everywhere")

    # 2. leak tests
    print("== 2: leak tests (2 x 1024 bytes, chunk 1024, log2 12)", flush=True)
    data = toy_words(2048, 1)
    A2, tA2, _ = toy_build2(data, log2, 1024, 2048, orders)
    B2, tB2, _ = toy_build2(data, log2, 1024, 1024, orders)
    D2, _, _ = toy_build2(data[:1024].copy(), log2, 1024, 1024, orders)
    e = data.copy()
    e[1500:] = np.random.default_rng(7).integers(0, 256, 548, dtype=np.uint8)
    E2, _, _ = toy_build2(e, log2, 1024, 2048, orders)
    A1, _, _ = toy_build2(data, log2, 1, 2048, orders)
    eq = lambda X, Y, sl: all(np.array_equal(X.f[n][:, sl], Y.f[n][:, sl]) and np.array_equal(X.p[n][:, sl], Y.p[n][:, sl])
                              and np.array_equal(X.g[n][sl], Y.g[n][sl]) for n in orders)
    hits2 = sum(int((A2.f[n][1, 1024:] > 0).sum()) for n in orders)
    check(eq(A2, B2, slice(1024, 2048)), f"chunk 2 written (train_end 2048) == chunk 2 frozen (train_end 1024): identical ({hits2} hits in chunk 2)")
    check(not all(np.array_equal(tA2[n], tB2[n]) for n in orders), "and the tables after the pass differ (chunk 2 was written after its lookups)")
    check(eq(A2, B2, slice(0, 1024)) and all(np.array_equal(A2.f[n][:, :1024], D2.f[n]) and np.array_equal(A2.p[n][:, :1024], D2.p[n]) for n in orders),
          "chunk 1 identical with the data ending after chunk 1")
    check(eq(A2, E2, slice(0, 1501)), "bytes from 1500 on replaced: positions <= 1500 identical")
    nd = sum(int((A1.f[n][:, 1024:] != B2.f[n][:, 1024:]).any(axis=0).sum()) for n in orders)
    check(nd > 0, f"the test has teeth: with chunk 1, chunk 2 differs from the frozen build at {nd} position-orders")
    print("SELFTEST2 " + ("PASSED" if ok else "FAILED"), flush=True)
    return ok


# ---------------------------------------------------------------------------------------------------------------------------------
# which held-out positions' contexts occur in the training bytes (exact 64-bit hash identity), for the false-hit rates

def seen_mask(n, data, reader=None):
    path = f"{SEEN_DIR}/seen_n{n}.npy"
    if os.path.exists(path):
        return np.load(path)
    N = len(data)
    t = time.time()
    if n in BYTE_ORDERS:
        train = np.empty(TRAIN_END - n, np.uint64)
        for lo in range(0, TRAIN_END, CHUNK):
            hi = min(lo + CHUNK, TRAIN_END)
            first = max(lo, n)
            if first < hi:
                train[first - n:hi - n] = context_hashes(data, lo, hi, (n,))[n][first - lo:]
        held = np.empty(N - HELDOUT, np.uint64)
        for lo in range(HELDOUT, N, CHUNK):
            hi = min(lo + CHUNK, N)
            held[lo - HELDOUT:hi - HELDOUT] = context_hashes(data, lo, hi, (n,))[n]
    else:
        reader = reader if reader is not None else KeyReader((n,))
        train = reader.get(n, n, TRAIN_END)                       # positions p < n are skipped by every builder: never written
        held = reader.get(n, HELDOUT, N)
    train.sort()
    idx = np.searchsorted(train, held)
    np.minimum(idx, len(train) - 1, out=idx)
    seen = train[idx] == held
    del train
    os.makedirs(SEEN_DIR, exist_ok=True)
    np.save(path, seen)
    print(f"seen mask of order {n}: {seen.mean():.4f} of held-out positions have a context from the training bytes ({time.time() - t:.0f} s)", flush=True)
    return seen


# ---------------------------------------------------------------------------------------------------------------------------------
# stats

def paq_stats2(d, pred, conf, pred2, conf2):
    hit, has2 = conf > 0, conf2 > 0
    nh = int(hit.sum())
    r1, r2 = pred == d, (pred2 == d) & has2
    wrong = hit & ~r1
    return {"positions": int(len(d)), "hit_rate": nh / max(len(d), 1), "right_given_hit": _rate(r1, hit),
            "bins": {str(b): {"share_of_hits": float((conf == b).sum()) / nh if nh else None, "right": _rate(r1, conf == b)} for b in BINS},
            "second_present_given_hit": _rate(has2, hit),
            "second_right_given_first_wrong": _rate(r2, wrong),
            "second_bins": {str(b): {"share_of_hits": float((conf2 == b).sum()) / nh if nh else None, "right": _rate(r2, conf2 == b)} for b in BINS},
            "top2_right_given_hit": _rate(r1 | r2, hit)}


def false_hits(hit, right, seen):
    """A false hit: the table answers at a position whose exact context never occurred in the training bytes."""
    nh = int(hit.sum())
    fh = hit & ~seen
    return {"seen_share": float(seen.mean()), "false_hit_rate": float(fh.mean()), "false_share_of_hits": float(fh.sum()) / nh if nh else None,
            "hit_rate_on_unseen": _rate(hit, ~seen), "hit_rate_on_seen": _rate(hit, seen),
            "right_given_hit_on_seen": _rate(right, hit & seen), "right_given_false_hit": _rate(right, fh)}


def diag2_stats(dg):
    hits = dg[3]
    q = lambda a, b: (a / b) if b else None
    return {"positions": int(dg[0]), "misses": int(dg[1]), "misses_with_root_present": int(dg[2]), "hits": int(hits),
            "mean_p_per_bin": {str(b): q(dg[8 + x], dg[4 + x]) for x, b in enumerate(BINS)},
            "second_present": int(dg[16]), "second_mean_p_per_bin": {str(b): q(dg[21 + x], dg[17 + x]) for x, b in enumerate(BINS)},
            "greedy_broken_share_of_hits": q(dg[29], hits), "beam_top_differs_from_greedy_share_of_hits": q(dg[30], hits),
            "when_differs": {"beam_top_right": q(dg[31], dg[30]), "greedy_top_right": q(dg[32], dg[30])},
            "deciles": [{"p": [k / 10, (k + 1) / 10], "share_of_hits": q(dg[33 + k], hits), "right": q(dg[43 + k], dg[33 + k])} for k in range(10)],
            "training_node_updates": int(dg[54]), "claims_of_empty_slots": int(dg[55]), "evictions": int(dg[56])}


# ---------------------------------------------------------------------------------------------------------------------------------
# dry run: this build and the first on the same bytes

def dry2(log2, mib):
    B, te = mib << 20, (mib - 1) << 20
    t0 = time.time()
    data = np.fromfile(CORPUS, dtype=np.uint8, count=B)
    keys = {kid: np.load(f"{MEM}/feat_keys/key_n{kid}.npy", mmap_mode="r") for kid in KEY_IDS}
    s2 = MemSink2(B, ORDERS)
    t1 = time.time()
    tab2, d2 = build2(data, v1.hashes_fn(data, keys), ORDERS, log2, CHUNK, te, s2, stats_from=te)
    tb2 = time.time() - t1
    s1 = v1.MemSink(B, ORDERS)
    t1 = time.time()
    tab1, d1 = v1.build(data, v1.hashes_fn(data, keys), ORDERS, log2, CHUNK, te, s1, stats_from=te)
    tb1 = time.time() - t1
    print(f"dry run: {B:,} bytes, train_end {te:,}, 2^{log2} slots per order; tagged + beam built in {tb2:.1f} s, untagged in {tb1:.1f} s; "
          f"peak RSS {peak_rss_gb():.2f} GB", flush=True)
    H = context_hashes(data, 0, B, BYTE_ORDERS)
    for kid in KEY_IDS:
        H[kid] = np.asarray(keys[kid][:B])
    ho, tr = slice(te, B), slice(CHUNK, te)
    d = data[ho]
    res = {}
    print(f"{'order':>5} | {'occ v1/v2':>11} | {'hit v1 / v2 / all28':>20} | {'right v1 / v2 / all28':>22} | {'false-hit rate v1 / v2 / all28':>31} | "
          f"{'hit on seen v1/v2/a28':>21} | {'right on seen v1/v2/a28':>23} | {'2nd':>5} {'beam!=gr':>8} | v2 train hit/right (all28)")
    for n in ORDERS:
        p28 = np.load(f"{MEM}/feat_all28/pred_n{n}.npy", mmap_mode="r")[:B]
        c28 = np.load(f"{MEM}/feat_all28/conf_n{n}.npy", mmap_mode="r")[:B]
        seen = np.isin(H[n][ho], H[n][n:te])
        s = {"occupancy_v1": float(np.count_nonzero(tab1[n].view(np.uint16))) / (1 << log2), "occupancy_v2": occupancy2(tab2[n])}
        s["heldout_v2"] = paq_stats2(d, s2.f[n][0][ho], s2.f[n][1][ho], s2.f[n][2][ho], s2.f[n][3][ho])
        s["heldout_v1"] = v1.paq_stats(d, s1.f[n][0][ho], s1.f[n][1][ho], s1.f[n][2][ho], s1.f[n][3][ho])
        s["heldout_all28"] = count_stats(d, p28[ho], c28[ho])
        s["false_hits"] = {"v1": false_hits(s1.f[n][1][ho] > 0, s1.f[n][0][ho] == d, seen),
                           "v2": false_hits(s2.f[n][1][ho] > 0, s2.f[n][0][ho] == d, seen),
                           "all28": false_hits(c28[ho] > 0, p28[ho] == d, seen)}
        s["train_v2"] = paq_stats2(data[tr], s2.f[n][0][tr], s2.f[n][1][tr], s2.f[n][2][tr], s2.f[n][3][tr])
        s["train_all28"] = count_stats(data[tr], p28[tr], c28[tr])
        s["heldout_diag_v2"] = diag2_stats(d2[n])
        res[n] = s
        h1, h2, h3 = s["heldout_v1"], s["heldout_v2"], s["heldout_all28"]
        f1, f2, f3 = s["false_hits"]["v1"], s["false_hits"]["v2"], s["false_hits"]["all28"]
        dg = s["heldout_diag_v2"]
        print(f"{n:>5} | {s['occupancy_v1']:.3f}/{s['occupancy_v2']:.3f} | {h1['hit_rate']:.3f} / {h2['hit_rate']:.3f} / {h3['hit_rate']:.3f} | "
              f"{fmt2(h1['right_given_hit'])} / {fmt2(h2['right_given_hit'])} / {fmt2(h3['right_given_hit'])}    | "
              f"{f1['false_hit_rate']:.4f} / {f2['false_hit_rate']:.4f} / {f3['false_hit_rate']:.4f}       | "
              f"{fmt2(f1['hit_rate_on_seen'])}/{fmt2(f2['hit_rate_on_seen'])}/{fmt2(f3['hit_rate_on_seen'])}        | "
              f"{fmt2(f1['right_given_hit_on_seen'])}/{fmt2(f2['right_given_hit_on_seen'])}/{fmt2(f3['right_given_hit_on_seen'])}         | "
              f"{fmt2(h2['second_present_given_hit']):>5} {fmt2(dg['beam_top_differs_from_greedy_share_of_hits']):>8} | "
              f"{s['train_v2']['hit_rate']:.3f}/{fmt2(s['train_v2']['right_given_hit'])} ({s['train_all28']['hit_rate']:.3f}/{fmt2(s['train_all28']['right_given_hit'])})",
              flush=True)
    out = f"{HERE}/dry2_{log2}_{mib}.json"
    json.dump({"log2": log2, "bytes": B, "train_end": te, "build_seconds_v2": tb2, "build_seconds_v1": tb1, "peak_rss_gb": peak_rss_gb(),
               "stats": {str(n): v for n, v in res.items()}}, open(out, "w"), indent=1)
    print(f"wrote {out} ({time.time() - t0:.0f} s)", flush=True)


# ---------------------------------------------------------------------------------------------------------------------------------
# the full build

RULES = {"slot": "(tag u16, n0 u8, n1 u8); h = corpus_memory._finalise(ctx * 0x9E3779B97F4A7C15 + node * 0xC2B2AE3D27D4EB4F): slot = top log2 bits, tag = low 16 bits; empty = n0 + n1 == 0",
         "update": "per training position in corpus order, 8 nodes root first: empty or same tag -> use; other tag -> evict (tag replaced, counts 0); then: if n_b == 255 halve both; n_b += 1; if n_other > 2: n_other = n_other // 2 + 1",
         "lookup": "beam of width 4 over the observed bits (n_bit > 0), p(bit) = (n_bit + 0.4) / (n0 + n1 + 0.8), stable ties; any node read that is empty or holds another tag -> miss for the whole lookup; top / second = the two best complete paths",
         "conf_bins": "top: 1 p < 0.5, 2 < 0.75, 4 < 0.9, 16; second (5000+o): 1 p2 < 0.1, 2 < 0.25, 4 < 0.4, 16 >= 0.4; 0 = miss or no second path",
         "false_hit": "a hit at a held-out position whose exact context (64-bit hash) never occurs in the training bytes"}


def full2(log2, out):
    t0 = time.time()
    out = os.path.abspath(out)
    if os.path.exists(f"{out}/features.json"):
        sys.exit(f"{out} exists: not rewriting a feature directory (a trainer may have it mapped)")
    refuse_if_in_use(out)
    os.makedirs(out, exist_ok=True)
    print(f"paq_bits2.py {log2} {out}: {WORKERS} threads, available {avail_gb():.1f} GB, load {os.getloadavg()}", flush=True)
    data = np.fromfile(CORPUS, dtype=np.uint8)
    N = len(data)
    reader = KeyReader(KEY_IDS)
    sink = FileSink(out, N, ORDERS)
    tables, diag = build2(data, hashes_fn2(data, reader), ORDERS, log2, CHUNK, TRAIN_END, sink, stats_from=HELDOUT,
                          log=lambda s: print(s, flush=True))
    sink.close()
    occupancy = {n: occupancy2(tables[n]) for n in ORDERS}
    del tables
    build_s = time.time() - t0
    print(f"pass done in {build_s:.0f} s, peak RSS {peak_rss_gb():.2f} GB", flush=True)
    for n in COUNTERS:
        for kind in ("pred", "conf"):
            dst = f"{out}/{kind}_n{n}.npy"
            if not os.path.exists(dst):
                os.symlink(os.path.realpath(f"{MEM}/feat_counters/{kind}_n{n}.npy"), dst)
    ld = lambda d_, kind, oid, lo: np.asarray(np.load(f"{d_}/{kind}_n{oid}.npy", mmap_mode="r")[lo:N])
    first_build = f"{MEM}/feat_paqbits"
    stats = {}
    for n in ORDERS:
        s = {"occupancy": occupancy[n]}
        for name, lo in (("heldout", HELDOUT), ("from_train_end", TRAIN_END)):
            d = data[lo:N]
            pr, cf, pr2, cf2 = (ld(out, k, o, lo) for k, o in (("pred", n), ("conf", n), ("pred", SECOND + n), ("conf", SECOND + n)))
            p28, c28 = ld(f"{MEM}/feat_all28", "pred", n, lo), ld(f"{MEM}/feat_all28", "conf", n, lo)
            s[name] = paq_stats2(d, pr, cf, pr2, cf2)
            s[name + "_all28"] = count_stats(d, p28, c28)
            if name == "heldout":
                seen = seen_mask(n, data, reader)
                p1_, c1_ = ld(first_build, "pred", n, lo), ld(first_build, "conf", n, lo)
                s["false_hits"] = {"before_feat_paqbits": false_hits(c1_ > 0, p1_ == d, seen), "after_feat_paqbits2": false_hits(cf > 0, pr == d, seen),
                                   "feat_all28": false_hits(c28 > 0, p28 == d, seen)}
                hp, h28 = cf > 0, c28 > 0
                s["heldout_vs_all28"] = {"paq_hit_where_all28_hits": _rate(hp, h28), "paq_hit_where_all28_misses": _rate(hp, ~h28),
                                         "paq_right_where_both_hit": _rate(pr == d, hp & h28), "all28_right_where_both_hit": _rate(p28 == d, hp & h28),
                                         "paq_right_where_only_paq_hits": _rate(pr == d, hp & ~h28)}
                dg = diag2_stats(diag[n])
                assert dg["positions"] == len(d) and dg["hits"] == int(hp.sum()) and dg["second_present"] == int((cf2 > 0).sum()), "diag != features"
                assert all(int(diag[n][4 + x]) == int((cf == b).sum()) for x, b in enumerate(BINS)), "diag bins != features"
                s["heldout_diag"] = dg
        stats[str(n)] = s
        h, a, fb, fa = s["heldout"], s["heldout_all28"], s["false_hits"]["before_feat_paqbits"], s["false_hits"]["after_feat_paqbits2"]
        print(f"order {n}: occ {occupancy[n]:.3f}; held out: hits {h['hit_rate']:.3f} right {fmt2(h['right_given_hit'])} | all28 {a['hit_rate']:.3f} "
              f"{fmt2(a['right_given_hit'])}; bins " + " ".join(f"{b}:{fmt2(h['bins'][str(b)]['right'])}({fmt2(h['bins'][str(b)]['share_of_hits'])})" for b in BINS)
              + f"; second present {fmt2(h['second_present_given_hit'])}, right when first wrong {fmt2(h['second_right_given_first_wrong'])}; "
              f"false-hit rate {fb['false_hit_rate']:.4f} -> {fa['false_hit_rate']:.4f}; beam != greedy {fmt2(s['heldout_diag']['beam_top_differs_from_greedy_share_of_hits'])}",
              flush=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    stats_path = f"{TABLES_DIR}/stats_{os.path.basename(out)}.json" if out.startswith(MEM + "/") else f"{out}.stats.json"
    json.dump({"log2_slots": log2, "corpus_bytes": N, "train_end": TRAIN_END, "heldout_from": HELDOUT, "chunk": CHUNK, "beam": BEAM,
               "build_seconds": build_s, "peak_rss_gb": peak_rss_gb(), "workers": WORKERS, "rules": RULES, "stats": stats}, open(stats_path, "w"), indent=1)
    info = json.load(open(f"{MEM}/feat/features.json"))
    for k in ("feature_digest", "compare", "table_digest", "occupancy", "train_hit_rate_by_tenth", "train_right_given_hit_by_tenth", "train_hit_rate",
              "heldout_hit_rate", "heldout_right_given_hit", "seconds_by_stage", "build_cpu_seconds", "heldout_lookup_seconds",
              "digest_and_compare_seconds", "build_mb_per_second", "loadavg_start", "loadavg_end"):
        info.pop(k, None)
    info.update(orders=[*ORDERS, *(SECOND + n for n in ORDERS), *COUNTERS], log2_slots=log2, slots_per_order=1 << log2, paq_bits2=RULES,
                build_seconds=build_s, stats={str(n): {"occupancy": stats[str(n)]["occupancy"], "heldout_hit_rate": stats[str(n)]["heldout"]["hit_rate"],
                                                       "heldout_right_given_hit": stats[str(n)]["heldout"]["right_given_hit"]} for n in ORDERS},
                stats_file=stats_path,
                note=f"paq_bits2.py: PAQ-style bit tables with a 16-bit check per slot (tag, n0, n1; evict on a foreign tag; any absent node on the "
                     f"walk = miss) of 2^{log2} slots per order for byte orders {list(BYTE_ORDERS)} and keys {list(KEY_IDS)}; <o> = the best path of "
                     f"a width-{BEAM} beam with its probability bin (1 <0.5, 2 <0.75, 4 <0.9, 16), {SECOND}+<o> = the second best path, binned by p2 "
                     f"(1 <0.1, 2 <0.25, 4 <0.4, 16; 0 = none); one leak-free pass; counters linked from feat_counters")
    json.dump(info, open(f"{out}/features.json", "w"), indent=2)
    print(f"wrote {stats_path} and {out}/features.json; done in {time.time() - t0:.0f} s, peak RSS {peak_rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selftest"]:
        sys.exit(0 if selftest2() else 1)
    elif sys.argv[1:2] == ["--seen"]:
        data = np.fromfile(CORPUS, dtype=np.uint8)
        reader = KeyReader(KEY_IDS)
        for n in ORDERS:
            seen_mask(n, data, reader)
        print(f"peak RSS {peak_rss_gb():.2f} GB", flush=True)
    elif sys.argv[1:2] == ["--dry"]:
        dry2(int(sys.argv[2]), int(sys.argv[3]))
    else:
        full2(int(sys.argv[1]), sys.argv[2])
