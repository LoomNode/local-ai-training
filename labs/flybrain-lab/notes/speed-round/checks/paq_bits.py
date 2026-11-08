"""Bet E: PAQ-style bit-level tables as an input channel (flybrain-lab speed round).

For each lookup order o of the combined arm (byte orders 4, 8, 12, 16, 32, 64 from corpus_memory.context_hashes; keys 116, 132, 202, 203,
301 from results/mem/feat_keys/key_n<id>.npy) one table of 2^<log2> slots; a slot is two uint8 counts (n0, n1), no tag (collisions are
accepted, as in PAQ). The byte to predict is walked as a binary tree, MSB first: node 1 is the root, node = 2 * node + bit after each bit,
so the internal nodes are 1..255.

Addressing: slot(ctx, node) = the top <log2> bits of corpus_memory._finalise(ctx * 0x9E3779B97F4A7C15 + node * 0xC2B2AE3D27D4EB4F), wrapping
uint64 arithmetic, ctx = the order's 64-bit context hash at the position (the same hashes the count tables use). Top bits, as slot_and_tag.

Update (training positions only, in corpus order, after the chunk's lookups), at each of the 8 nodes of the actual byte with observed bit b:
    if n_b == 255: n_b, n_other = n_b // 2, n_other // 2        (halve both first)
    n_b += 1
    if n_other > 2: n_other = n_other // 2 + 1                   (PAQ's nonstationary rule, on the possibly halved n_other)
The nodes of one position are updated in tree order (root first) and positions one after another, so a slot touched many times inside a
chunk ends exactly as the one-at-a-time rule leaves it (the C loop below is that rule; --selftest checks it against a scalar reference).

Lookup (from the table state before the position's chunk was written): a root with n0 + n1 == 0 is a miss (pred 0, conf 0 for both
candidates). Otherwise walk greedily: at a node p(bit) = (n_bit + 0.4) / (n0 + n1 + 0.8), take bit 1 when n1 > n0, else bit 0 (a tie
takes 0); the top byte is the path, its probability the product of the 8 chosen-bit probabilities (multiplied root first). Second
candidate: for each level k, flip the choice there (probability of the other bit at that node) and continue greedily below; the best of
the 8 products (ties: the lowest k) is the second byte. Probabilities are binned into the conf byte: 1 if p < 0.5, 2 if p < 0.75, 4 if
p < 0.9, 16 otherwise, which inputs.conf_class turns into the classes 1, 2, 3, 4.

Features (uint8, one per corpus byte, feature[p] predicts byte p from the bytes before it): pred_n<o>, conf_n<o> (top byte, bin) and
pred_n<5000+o>, conf_n<5000+o> (second byte, bin), with the counters 1001..1007 linked from feat_counters. One leak-free pass with the
template's chunks (2^20 positions, cut at train_end; positions past train_end read the frozen tables; positions p < o skipped).

The hot loop is a small C kernel (embedded below, compiled with gcc into this directory on first use, called through ctypes, which drops
the GIL): the eleven tables are independent, so a pool of OMP_NUM_THREADS threads runs one (order, chunk) call each.

usage:
  paq_bits.py <log2> <out dir>          the full build; stats to results/tables/stats_<basename of out dir>.json
  paq_bits.py --selftest                toy checks: hand-computed slot and lookup, chunk 1 vs chunked vs a scalar reference, leak tests
  paq_bits.py --dry <log2> <MiB>        the first <MiB> MiB of the corpus (the last MiB held out), compared with feat_all28
  paq_bits.py --probe <log2>            timing of one 2^20-position chunk on a table of random counts (every lookup walks the whole tree)
At 2^28 the eleven tables take 5.5 GB of RAM."""
import ctypes
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
sys.path.insert(0, SPEED)
from flybrain.corpus_memory import _finalise, context_hashes, hash_context  # noqa: E402

CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
MEM = SPEED + "/results/mem"
TABLES_DIR = SPEED + "/results/tables"
TRAIN_END, CHUNK = 200_000_000, 1 << 20
HELDOUT = 202_117_357                      # first byte of the dev windows: held-out stats are over [HELDOUT, N)
BYTE_ORDERS = (4, 8, 12, 16, 32, 64)
KEY_IDS = (116, 132, 202, 203, 301)
ORDERS = (*BYTE_ORDERS, *KEY_IDS)
COUNTERS = (1001, 1002, 1003, 1004, 1005, 1006, 1007)
SECOND = 5000
BINS = (1, 2, 4, 16)
C1, C2 = 0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F
MASK = (1 << 64) - 1
HERE = os.path.dirname(os.path.abspath(__file__))
WORKERS = max(1, int(os.environ.get("OMP_NUM_THREADS", "4")))
NDIAG = 64

C_SRC = r"""
#include <stdint.h>
#include <stddef.h>

#define M1 0xFF51AFD7ED558CCDULL      /* corpus_memory._finalise */
#define M2 0xC4CEB9FE1A85EC53ULL
#define C1 0x9E3779B97F4A7C15ULL      /* slot = top bits of fin(ctx * C1 + node * C2) */
#define C2 0xC2B2AE3D27D4EB4FULL
#define G 8                           /* positions walked in lockstep by the lookup (their loads overlap) */
#define D 8                           /* update: slots of position i + D are prefetched while position i is written */

static inline uint64_t fin(uint64_t h) {
    h ^= h >> 33; h *= M1; h ^= h >> 33; h *= M2; h ^= h >> 33;
    return h;
}
static inline uint64_t slot_of(uint64_t base, uint64_t node, int shift) { return fin(base + node * C2) >> shift; }
static inline uint8_t bin_of(double p) { return p < 0.5 ? 1 : (p < 0.75 ? 2 : (p < 0.9 ? 4 : 16)); }
static inline int bin_index(uint8_t b) { return b == 1 ? 0 : (b == 2 ? 1 : (b == 4 ? 2 : 3)); }

/* diag (positions i >= stats_lo): [0] positions, [1] misses, [2+x] top hits in bin x, [6+x] sum of top p, [10+x] top right,
   [14+x] second in bin x, [18+x] sum of second p, [22+x] second right, [26] hits where the second's product beats the greedy one,
   [27] top right among those, [28] second right among those, [29+d] top hits with p in decile d, [39+d] top right in decile d */
static void lookup(const uint8_t *tab, int log2, const uint64_t *ctx, const uint8_t *nxt, int64_t lo, int64_t hi, int64_t stats_lo,
                   uint8_t *pred, uint8_t *conf, uint8_t *pred2, uint8_t *conf2, double *p1, double *p2, double *diag)
{
    const int shift = 64 - log2;
    for (int64_t i0 = lo; i0 < hi; i0 += G) {
        const int g = (hi - i0) < G ? (int)(hi - i0) : G;
        uint64_t base[G], addr[G][8], mnode[G], cnode[G][8];
        double P[G], cprod[G][8];
        int alive[G];
        for (int j = 0; j < g; j++) {
            base[j] = ctx[i0 + j] * C1;
            addr[j][0] = slot_of(base[j], 1, shift);
            __builtin_prefetch(tab + 2 * addr[j][0]);
        }
        for (int j = 0; j < g; j++) {
            const uint8_t *c = tab + 2 * addr[j][0];
            alive[j] = (c[0] | c[1]) != 0;
            mnode[j] = 1;
            P[j] = 1.0;
        }
        for (int lvl = 0; lvl < 8; lvl++) {
            if (lvl > 0) {                              /* addresses of this level: the main path's node and every open chain's node */
                for (int j = 0; j < g; j++) {
                    if (!alive[j]) continue;
                    addr[j][0] = slot_of(base[j], mnode[j], shift);
                    __builtin_prefetch(tab + 2 * addr[j][0]);
                    for (int k = 0; k < lvl; k++) {
                        addr[j][1 + k] = slot_of(base[j], cnode[j][k], shift);
                        __builtin_prefetch(tab + 2 * addr[j][1 + k]);
                    }
                }
            }
            for (int j = 0; j < g; j++) {
                if (!alive[j]) continue;
                for (int k = 0; k < lvl; k++) {         /* chain k (flipped at level k) takes its greedy step at this level */
                    const uint8_t *c = tab + 2 * addr[j][1 + k];
                    const unsigned m0 = c[0], m1 = c[1], b = m1 > m0;
                    cprod[j][k] *= ((b ? m1 : m0) + 0.4) / ((double)(m0 + m1) + 0.8);
                    cnode[j][k] = 2 * cnode[j][k] + b;
                }
                const uint8_t *c = tab + 2 * addr[j][0];
                const unsigned n0 = c[0], n1 = c[1], b = n1 > n0;
                const double tot = (double)(n0 + n1) + 0.8;
                const double pc = ((b ? n1 : n0) + 0.4) / tot, q = ((b ? n0 : n1) + 0.4) / tot;
                cnode[j][lvl] = 2 * mnode[j] + (1 - b);  /* chain lvl: the other bit here, greedy below */
                cprod[j][lvl] = P[j] * q;
                P[j] *= pc;
                mnode[j] = 2 * mnode[j] + b;
            }
        }
        for (int j = 0; j < g; j++) {
            const int64_t i = i0 + j;
            if (!alive[j]) {
                pred[i] = 0; conf[i] = 0; pred2[i] = 0; conf2[i] = 0;
                if (p1) { p1[i] = 0.0; p2[i] = 0.0; }
                if (i >= stats_lo) { diag[0] += 1; diag[1] += 1; }
                continue;
            }
            int best = 0;
            for (int k = 1; k < 8; k++) if (cprod[j][k] > cprod[j][best]) best = k;
            const uint8_t top = (uint8_t)(mnode[j] - 256), sec = (uint8_t)(cnode[j][best] - 256);
            const double pt = P[j], ps = cprod[j][best];
            const uint8_t b1 = bin_of(pt), b2 = bin_of(ps);
            pred[i] = top; conf[i] = b1; pred2[i] = sec; conf2[i] = b2;
            if (p1) { p1[i] = pt; p2[i] = ps; }
            if (i >= stats_lo) {
                const int r1 = top == nxt[i], r2 = sec == nxt[i], x1 = bin_index(b1), x2 = bin_index(b2);
                diag[0] += 1;
                diag[2 + x1] += 1; diag[6 + x1] += pt; diag[10 + x1] += r1;
                diag[14 + x2] += 1; diag[18 + x2] += ps; diag[22 + x2] += r2;
                if (ps > pt) { diag[26] += 1; diag[27] += r1; diag[28] += r2; }
                int d = (int)(pt * 10.0); if (d > 9) d = 9;
                diag[29 + d] += 1; diag[39 + d] += r1;
            }
        }
    }
}

static inline void slots8(const uint8_t *tab, uint64_t ctx, unsigned byte, int shift, uint64_t *out) {
    const uint64_t base = ctx * C1;
    uint64_t node = 1;
    for (int lvl = 0; lvl < 8; lvl++) {
        out[lvl] = slot_of(base, node, shift);
        __builtin_prefetch(tab + 2 * out[lvl], 1);
        node = 2 * node + ((byte >> (7 - lvl)) & 1);
    }
}

/* positions lo .. hi-1 one after another, each one's 8 nodes root first: the sequential rule */
static void update(uint8_t *tab, int log2, const uint64_t *ctx, const uint8_t *nxt, int64_t lo, int64_t hi)
{
    const int shift = 64 - log2;
    uint64_t ring[D][8];
    for (int64_t f = lo; f < hi && f < lo + D; f++) slots8(tab, ctx[f], nxt[f], shift, ring[f % D]);
    for (int64_t i = lo; i < hi; i++) {
        const uint64_t *s = ring[i % D];
        const unsigned byte = nxt[i];
        for (int lvl = 0; lvl < 8; lvl++) {
            const unsigned b = (byte >> (7 - lvl)) & 1;
            uint8_t *c = tab + 2 * s[lvl];
            unsigned nb = c[b], no = c[b ^ 1];
            if (nb == 255) { nb >>= 1; no >>= 1; }
            nb += 1;
            if (no > 2) no = no / 2 + 1;
            c[b] = (uint8_t)nb; c[b ^ 1] = (uint8_t)no;
        }
        if (i + D < hi) slots8(tab, ctx[i + D], nxt[i + D], shift, ring[i % D]);
    }
}

/* one (order, chunk): lookups of positions first .. m-1 on the table as it is, then (training chunks) their updates in order */
void paq_chunk(uint8_t *tab, int log2, const uint64_t *ctx, const uint8_t *nxt, int64_t first, int64_t m, int do_update, int64_t stats_lo,
               uint8_t *pred, uint8_t *conf, uint8_t *pred2, uint8_t *conf2, double *p1, double *p2, double *diag)
{
    lookup(tab, log2, ctx, nxt, first, m, stats_lo, pred, conf, pred2, conf2, p1, p2, diag);
    if (do_update) update(tab, log2, ctx, nxt, first, m);
}
"""

_LIB = None


def kernel():
    """The compiled C kernel (built once per source version, next to this script)."""
    global _LIB
    if _LIB is None:
        tag = hashlib.sha1(C_SRC.encode()).hexdigest()[:12]
        so = f"{HERE}/_paq_bits_{tag}.so"
        if not os.path.exists(so):
            src = f"{HERE}/_paq_bits_{tag}.c"
            with open(src, "w") as f:
                f.write(C_SRC)
            # -ffp-contract=off and no -march: plain IEEE double, so the probabilities equal the Python reference's bit for bit
            subprocess.run(["gcc", "-O2", "-fPIC", "-shared", "-ffp-contract=off", "-o", so + ".tmp", src], check=True)
            os.replace(so + ".tmp", so)
        lib = ctypes.CDLL(so)
        vp, i64 = ctypes.c_void_p, ctypes.c_int64
        lib.paq_chunk.restype = None
        lib.paq_chunk.argtypes = [vp, ctypes.c_int, vp, vp, i64, i64, ctypes.c_int, i64, vp, vp, vp, vp, vp, vp, vp]
        _LIB = lib
    return _LIB


def _ptr(a):
    return None if a is None else a.ctypes.data


def run_order(table, log2, h, nxt, first, m, update, stats_lo, out, dd):
    pred, conf, pred2, conf2, p1, p2 = out
    kernel().paq_chunk(_ptr(table), log2, _ptr(h), _ptr(nxt), first, m, int(update), stats_lo,
                       _ptr(pred), _ptr(conf), _ptr(pred2), _ptr(conf2), _ptr(p1), _ptr(p2), _ptr(dd))


def hashes_fn(data, keys):
    """hashes_of(lo, hi): the byte orders' context hashes (corpus_memory.context_hashes, as every builder) plus the precomputed key hashes."""
    def hashes_of(lo, hi):
        h = context_hashes(data, lo, hi, BYTE_ORDERS)
        for kid, arr in keys.items():
            h[kid] = np.ascontiguousarray(arr[lo:hi], dtype=np.uint64)
        return h
    return hashes_of


def build(data, hashes_of, orders, log2, chunk, train_end, sink, raw=False, stats_from=None, workers=WORKERS, log=None):
    """One leak-free pass over data: chunk bounds = multiples of chunk plus train_end; a chunk's lookups read the tables before its writes;
    chunks past train_end only look up. sink(lo, hi, out) receives {order: [pred, conf, pred2, conf2, p1, p2]} (p1, p2 = raw
    probabilities when raw, else None). The C kernel's diag counters cover positions >= stats_from. Returns (tables, diag)."""
    N = len(data)
    kernel()
    tables = {n: np.zeros(2 << log2, dtype=np.uint8) for n in orders}       # slot s = (n0, n1) at [2s, 2s + 1]
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
                first = min(max(lo, n) - lo, m)                                  # positions p < n: no full context (keys: skipped as the template)
                out[n] = [np.zeros(m, np.uint8) for _ in range(4)] + ([np.zeros(m), np.zeros(m)] if raw else [None, None])
                dd[n] = np.zeros(NDIAG)
                if first < m:
                    h = np.ascontiguousarray(hashes[n], dtype=np.uint64)
                    assert h.shape == (m,)
                    futs.append(pool.submit(run_order, tables[n], log2, h, nxt, first, m, hi <= train_end, stats_lo, out[n], dd[n]))
            upcoming = hashes_of(*spans[i + 1]) if i + 1 < len(spans) else None  # overlaps with the kernel calls (numpy drops the GIL too)
            for f in futs:
                f.result()
            for n in orders:
                diag[n] += dd[n]
            sink(lo, hi, out)
            if log is not None and (i % 20 == 0 or hi == N):
                log(f"{hi:,} / {N:,}  {time.time() - t0:.0f} s  rss {rss_gb():.1f} GB  available {avail_gb():.1f} GB")
    return tables, diag


class MemSink:
    def __init__(self, N, orders, raw=False):
        self.f = {n: np.zeros((4, N), np.uint8) for n in orders}
        self.p = {n: np.zeros((2, N)) for n in orders} if raw else None

    def __call__(self, lo, hi, out):
        for n, arrs in out.items():
            for k in range(4):
                self.f[n][k, lo:hi] = arrs[k]
            if self.p is not None:
                self.p[n][0, lo:hi], self.p[n][1, lo:hi] = arrs[4], arrs[5]


class FileSink:
    """Feature files created with open_memmap (header, zero body) and then written chunk by chunk with pwrite: page cache, not process RSS."""
    def __init__(self, out, N, orders):
        self.fds = {}
        for n in orders:
            for k, (kind, oid) in enumerate((("pred", n), ("conf", n), ("pred", SECOND + n), ("conf", SECOND + n))):
                path = f"{out}/{kind}_n{oid}.npy"
                mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(N,))
                off = mm.offset
                del mm
                self.fds[(n, k)] = (os.open(path, os.O_RDWR), off)

    def __call__(self, lo, hi, out):
        for n, arrs in out.items():
            for k in range(4):
                fd, off = self.fds[(n, k)]
                a = np.ascontiguousarray(arrs[k])
                assert os.pwrite(fd, a, off + lo) == a.nbytes

    def close(self):
        for fd, _ in self.fds.values():
            os.fsync(fd)
            os.close(fd)


def rss_gb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def refuse_if_in_use(directory):
    """A feature file that a running process has memory-mapped must not be rewritten (SIGBUS in that process, 2026-10-07)."""
    import glob
    real = {os.path.realpath(f) for f in glob.glob(f"{directory}/*.npy")}
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            maps = open(f"/proc/{pid}/maps").read()
        except OSError:
            continue
        for f in real:
            if f in maps:
                sys.exit(f"refusing to rewrite {f}: mapped by pid {pid} ({open(f'/proc/{pid}/comm').read().strip()})")


# ---------------------------------------------------------------------------------------------------------------------------------
# scalar reference (Python integers and floats, one position at a time): the definition the C kernel is checked against

def fin_py(h):
    h ^= h >> 33
    h = (h * 0xFF51AFD7ED558CCD) & MASK
    h ^= h >> 33
    h = (h * 0xC4CEB9FE1A85EC53) & MASK
    h ^= h >> 33
    return h


def ref_slot(ctx, node, log2):
    return fin_py((ctx * C1 + node * C2) & MASK) >> (64 - log2)


def bin_of(p):
    return 1 if p < 0.5 else 2 if p < 0.75 else 4 if p < 0.9 else 16


def ref_update(tab, ctx, byte, log2, touched=None):
    node = 1
    for lvl in range(8):
        b = (byte >> (7 - lvl)) & 1
        s = ref_slot(ctx, node, log2)
        nb, no = int(tab[s, b]), int(tab[s, 1 - b])
        if nb + 1 > 255:
            nb, no = nb // 2, no // 2
        nb += 1
        if no > 2:
            no = no // 2 + 1
        tab[s, b], tab[s, 1 - b] = nb, no
        if touched is not None:
            touched.setdefault(s, set()).add((ctx, node))
        node = 2 * node + b


def ref_lookup(tab, ctx, log2):
    """(top byte, its probability, second byte, its probability), or None for a miss. Walks the main path, then each flipped chain
    separately (the C kernel walks them level by level)."""
    def counts(node):
        s = ref_slot(ctx, node, log2)
        return int(tab[s, 0]), int(tab[s, 1])
    n0, n1 = counts(1)
    if n0 + n1 == 0:
        return None
    node, P, levels = 1, 1.0, []
    for _ in range(8):
        n0, n1 = counts(node)
        b = 1 if n1 > n0 else 0
        tot = (n0 + n1) + 0.8
        pc, q = ((n1 if b else n0) + 0.4) / tot, ((n0 if b else n1) + 0.4) / tot
        levels.append((node, b, P, q))
        P = P * pc
        node = 2 * node + b
    top, best = node - 256, None
    for node_k, b_k, P_k, q_k in levels:
        prod, x = P_k * q_k, 2 * node_k + (1 - b_k)
        while x < 256:
            m0, m1 = counts(x)
            c = 1 if m1 > m0 else 0
            prod = prod * (((m1 if c else m0) + 0.4) / ((m0 + m1) + 0.8))
            x = 2 * x + c
        if best is None or prod > best[1]:
            best = (x - 256, prod)
    return top, P, best[0], best[1]


def ref_build(data, H, orders, log2, chunk, train_end):
    """Positions one at a time in corpus order: a position's lookup reads a snapshot of its table taken when its chunk began, its update
    (training chunks) goes straight to the live table. Returns (features {n: [4, N] uint8}, raw p {n: [2, N]}, tables {n: [S, 2]})."""
    N = len(data)
    tabs = {n: np.zeros((1 << log2, 2), dtype=np.int64) for n in orders}
    f = {n: np.zeros((4, N), np.uint8) for n in orders}
    p = {n: np.zeros((2, N)) for n in orders}
    bounds = sorted(set(range(0, N, chunk)) | {train_end, N})
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        snaps = {n: tabs[n].copy() for n in orders}
        for pos in range(lo, hi):
            for n in orders:
                if pos < n:
                    continue
                ctx = int(H[n][pos])
                r = ref_lookup(snaps[n], ctx, log2)
                if r is not None:
                    top, pt, sec, ps = r
                    f[n][:, pos] = (top, bin_of(pt), sec, bin_of(ps))
                    p[n][:, pos] = (pt, ps)
                if hi <= train_end:
                    ref_update(tabs[n], ctx, int(data[pos]), log2)
    return f, p, tabs


# ---------------------------------------------------------------------------------------------------------------------------------
# verification 1 and 2

def toy_words(n_bytes, seed):
    rng = np.random.default_rng(seed)
    words = [b"int ", b"return ", b"x = ", b"y;\n", b"if (", b") {\n", b"}\n", b"for (i = 0; ", b"i++", b"struct ", b"->", b"NULL",
             b"\t", b"0x1f", b"err", b"static ", b"(void)", b"len", b" + 1", b"buf["]
    out = b"".join(words[k] for k in rng.integers(0, len(words), n_bytes))
    return np.frombuffer(out[:n_bytes], dtype=np.uint8).copy()


def toy_key(data):
    """A stand-in key order: any 64-bit hash of bytes before the position (here of the 3 bytes before it, rehashed)."""
    return _finalise(context_hashes(data, 0, len(data), (3,))[3] ^ np.uint64(0x5DEECE66D))


def toy_build(data, log2, chunk, train_end, orders, raw=True):
    keys = {116: toy_key(data)} if 116 in orders else {}
    sink = MemSink(len(data), orders, raw=raw)
    tables, diag = build(data, hashes_fn(data, keys), orders, log2, chunk, train_end, sink, raw=raw)
    return sink, tables


def selftest():
    log2, ok = 12, True

    def check(cond, msg):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
        ok = ok and bool(cond)

    # 1a. hand check: a context followed by q, q, q and then a 297 times (training), looked up once more after it (held out)
    print("== 1a: hand check (order 4, log2 12)", flush=True)
    a, q = ord("a"), ord("q")        # a = 0110 0001, q = 0111 0001: they part at level 3 (node 11): a -> node 22, q -> node 23
    hand = {1: (172, 0), 2: (0, 172), 5: (0, 172), 11: (169, 1), 22: (169, 0), 44: (169, 0), 88: (169, 0), 176: (0, 169),
            23: (3, 0), 46: (3, 0), 92: (3, 0), 184: (0, 3)}
    for cand in (b"WXYZ", b"KQJV", b"PMRT", b"BDFH", b"GLNS"):
        seq = cand + b"q"
        seq = seq * 3 + (cand + b"a") * 297
        data = np.frombuffer(seq + cand + b"a", dtype=np.uint8).copy()
        te = len(seq)
        H = context_hashes(data, 0, len(data), (4,))
        ctx = hash_context(cand)
        touched, rt = {}, np.zeros((1 << log2, 2), np.int64)
        for pos in range(4, te):
            ref_update(rt, int(H[4][pos]), int(data[pos]), log2, touched)
        if all(touched.get(ref_slot(ctx, node, log2)) == {(ctx, node)} for node in hand):
            break
        print(f"  context {cand!r}: a slot of interest is shared with another (context, node) at 2^12; trying the next one", flush=True)
    print(f"  context {cand!r}: {te} training bytes, its hash {ctx:#018x} == context_hashes at position 4: {ctx == int(H[4][4])}", flush=True)
    sink, tables = toy_build(data, log2, 256, te, (4,))
    tab = tables[4].reshape(-1, 2)
    for node, want in hand.items():
        s = ref_slot(ctx, node, log2)
        got = (int(tab[s, 0]), int(tab[s, 1]))
        alone = touched.get(s) == {(ctx, node)}
        check(got == want and alone, f"node {node:3d}: slot {s:4d} holds (n0, n1) = {got}, by hand {want}; no other (context, node) in that slot: {alone}")
    # by hand: root..node 5 at 172 (300 observations: 255, halve to 127 + 1 = 128, then 44 more), node 11: q x3 -> n1 = 3, the first a
    # -> n0 = 1, n1 = 3 // 2 + 1 = 2, a up to n0 = 255, the 256th a halves both (127, 1) and counts (128, 1), 41 more a -> (169, 1)
    pt = 1.0
    for f in [(172 + 0.4) / (172 + 0.8)] * 3 + [(169 + 0.4) / (170 + 0.8)] + [(169 + 0.4) / (169 + 0.8)] * 4:
        pt = pt * f
    ps = 1.0
    for f in [(172 + 0.4) / (172 + 0.8)] * 3 + [(1 + 0.4) / (170 + 0.8)] + [(3 + 0.4) / (3 + 0.8)] * 4:
        ps = ps * f
    pos = len(data) - 1
    fp, pp = sink.f[4][:, pos], sink.p[4][:, pos]
    check(int(fp[0]) == a and int(fp[1]) == bin_of(pt) == 16 and abs(pp[0] - pt) < 1e-12,
          f"lookup after training: top {chr(fp[0])!r} p = {pp[0]:.12f} (by hand {pt:.12f}) conf {fp[1]}")
    check(int(fp[2]) == q and int(fp[3]) == bin_of(ps) == 1 and abs(pp[1] - ps) < 1e-12,
          f"second candidate: {chr(fp[2])!r} p = {pp[1]:.12f} (by hand {ps:.12f}: flip at level 3, then (3, 0) x4) conf {fp[3]}")

    # 1b. chunk size 1 vs chunked vs the scalar reference, on a word toy with every byte order and a key order
    print("== 1b: chunk 1 vs 256 vs 1000, against the scalar reference (4000 bytes, train_end 3000, log2 12)", flush=True)
    data = toy_words(4000, 0)
    N, te, orders = len(data), 3000, (*BYTE_ORDERS, 116)
    H = context_hashes(data, 0, N, BYTE_ORDERS)
    H[116] = toy_key(data)
    runs = {}
    for chunk in (1, 256, 1000):
        sink, tables = toy_build(data, log2, chunk, te, orders)
        rf, rp, rt = ref_build(data, H, orders, log2, chunk, te)
        same_f = all(np.array_equal(sink.f[n], rf[n]) for n in orders)
        same_p = all(np.array_equal(sink.p[n], rp[n]) for n in orders)          # raw doubles, bit for bit
        same_t = all(np.array_equal(tables[n].reshape(-1, 2), rt[n]) for n in orders)
        hits = sum(int((sink.f[n][1] > 0).sum()) for n in orders)
        check(same_f and same_p and same_t, f"chunk {chunk:4d}: C kernel == scalar reference (features, raw probabilities, final tables); {hits} hits")
        runs[chunk] = (sink, tables)
    s1, t1 = runs[1]
    for chunk in (256, 1000):
        s, t = runs[chunk]
        same_t = all(np.array_equal(t[n], t1[n]) for n in orders)
        same_h = all(np.array_equal(s.f[n][:, te:], s1.f[n][:, te:]) and np.array_equal(s.p[n][:, te:], s1.p[n][:, te:]) for n in orders)
        diff_train = sum(int((s.f[n][:, :te] != s1.f[n][:, :te]).any(axis=0).sum()) for n in orders)
        check(same_t and same_h, f"chunk {chunk} vs chunk 1: final tables identical {same_t}, held-out features identical {same_h} "
              f"(training features differ at {diff_train} position-orders, by design: a lookup reads the state before its chunk)")
    # updates of one slot many times inside a chunk: the slot most touched in the 1000-chunk run
    tab4 = H[4][4:te]
    s_root = np.array([ref_slot(int(c), 1, log2) for c in tab4])
    cnt = np.bincount(s_root, minlength=1 << log2)
    print(f"  in-chunk repeats: the busiest root slot of order 4 is written {cnt.max()} times in the training bytes; chunked == sequential above", flush=True)

    # 2. leak tests on two chunks of 1024
    print("== 2: leak tests (2 x 1024 bytes, chunk 1024, log2 12)", flush=True)
    data = toy_words(2048, 1)
    A, tA = toy_build(data, log2, 1024, 2048, orders)            # both chunks written; chunk 2 looked up before its writes
    B, tB = toy_build(data, log2, 1024, 1024, orders)            # only chunk 1 written; chunk 2 reads the frozen tables
    Dd, _ = toy_build(data[:1024].copy(), log2, 1024, 1024, orders)   # the data ends after chunk 1
    e = data.copy()
    e[1500:] = np.random.default_rng(7).integers(0, 256, 548, dtype=np.uint8)
    E, _ = toy_build(e, log2, 1024, 2048, orders)                # bytes from 1500 on replaced
    A1, _ = toy_build(data, log2, 1, 2048, orders)               # a build whose lookups see every earlier write (what a leak looks like)
    eq = lambda X, Y, sl: all(np.array_equal(X.f[n][:, sl], Y.f[n][:, sl]) and np.array_equal(X.p[n][:, sl], Y.p[n][:, sl]) for n in orders)
    hits2 = sum(int((A.f[n][1, 1024:] > 0).sum()) for n in orders)
    check(eq(A, B, slice(1024, 2048)), f"chunk 2 written (train_end 2048) == chunk 2 frozen (train_end 1024): features identical ({hits2} hits in chunk 2)")
    check(not all(np.array_equal(tA[n], tB[n]) for n in orders), "and the tables after the pass differ (chunk 2 was written in the first build, after its lookups)")
    check(eq(A, B, slice(0, 1024)) and all(np.array_equal(A.f[n][:, :1024], Dd.f[n]) and np.array_equal(A.p[n][:, :1024], Dd.p[n]) for n in orders),
          "chunk 1 features identical with the data ending after chunk 1")
    check(eq(A, E, slice(0, 1501)), "bytes from 1500 on replaced: features of positions <= 1500 identical")
    nd = sum(int((A1.f[n][:, 1024:] != B.f[n][:, 1024:]).any(axis=0).sum()) for n in orders)
    check(nd > 0, f"the test has teeth: with chunk 1 (lookups after every earlier write) chunk 2 differs from the frozen build at {nd} position-orders")
    print("SELFTEST " + ("PASSED" if ok else "FAILED"), flush=True)
    return ok


# ---------------------------------------------------------------------------------------------------------------------------------
# stats

def _rate(x, m):
    k = int(m.sum())
    return float(x[m].mean()) if k else None


def paq_stats(d, pred, conf, pred2, conf2):
    hit = conf > 0
    nh = int(hit.sum())
    r1, r2 = pred == d, pred2 == d
    wrong = hit & ~r1
    return {"positions": int(len(d)), "hit_rate": nh / max(len(d), 1), "right_given_hit": _rate(r1, hit),
            "bins": {str(b): {"share_of_hits": float((conf == b).sum()) / nh if nh else None, "right": _rate(r1, conf == b)} for b in BINS},
            "second_right_given_first_wrong": _rate(r2, wrong),
            "second_bins": {str(b): {"share_of_hits": float((conf2 == b).sum()) / nh if nh else None, "right": _rate(r2, conf2 == b)} for b in BINS},
            "top2_right_given_hit": _rate(r1 | r2, hit)}


def count_stats(d, pred, conf):
    """The count tables' top-1 numbers, with their confidence in inputs.conf_class's classes 1 | 2-3 | 4-15 | 16+."""
    hit = conf > 0
    nh = int(hit.sum())
    r1 = pred == d
    cls = {"1": conf == 1, "2": (conf >= 2) & (conf < 4), "3": (conf >= 4) & (conf < 16), "4": conf >= 16}
    return {"positions": int(len(d)), "hit_rate": nh / max(len(d), 1), "right_given_hit": _rate(r1, hit),
            "classes": {k: {"share_of_hits": float(m.sum()) / nh if nh else None, "right": _rate(r1, m)} for k, m in cls.items()}}


def diag_stats(dg):
    hits = dg[0] - dg[1]
    out = {"positions": int(dg[0]), "hits": int(hits),
           "mean_p_per_bin": {str(b): (dg[6 + x] / dg[2 + x] if dg[2 + x] else None) for x, b in enumerate(BINS)},
           "second_mean_p_per_bin": {str(b): (dg[18 + x] / dg[14 + x] if dg[14 + x] else None) for x, b in enumerate(BINS)},
           "second_beats_greedy_share_of_hits": dg[26] / hits if hits else None,
           "when_second_beats_greedy": {"top_right": dg[27] / dg[26] if dg[26] else None, "second_right": dg[28] / dg[26] if dg[26] else None},
           "deciles": [{"p": [k / 10, (k + 1) / 10], "share_of_hits": dg[29 + k] / hits if hits else None,
                        "right": dg[39 + k] / dg[29 + k] if dg[29 + k] else None} for k in range(10)]}
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# verification 3: dry run on the first MiB of the corpus

def dry(log2, mib):
    B, te = mib << 20, (mib - 1) << 20
    t0 = time.time()
    data = np.fromfile(CORPUS, dtype=np.uint8, count=B)
    keys = {kid: np.load(f"{MEM}/feat_keys/key_n{kid}.npy", mmap_mode="r") for kid in KEY_IDS}
    sink = MemSink(B, ORDERS)
    t1 = time.time()
    tables, diag = build(data, hashes_fn(data, keys), ORDERS, log2, CHUNK, te, sink, stats_from=te)
    tb = time.time() - t1
    print(f"dry run: {B:,} bytes ({mib} MiB), train_end {te:,}, 2^{log2} slots per order, {WORKERS} threads: built in {tb:.1f} s "
          f"({tb / (B * len(ORDERS)) * 1e9:.0f} ns per position-order wall), peak RSS {peak_rss_gb():.2f} GB", flush=True)
    H = context_hashes(data, 0, B, BYTE_ORDERS)
    for kid in KEY_IDS:
        H[kid] = np.asarray(keys[kid][:B])
    tr, ho = slice(CHUNK, te), slice(te, B)               # training positions after the first chunk (which is all misses for every table); held out
    res = {}
    print(f"{'order':>5} {'occ':>5} | {'PAQ train hit':>13} {'right':>6} | {'all28 train hit':>15} {'right':>6} | {'PAQ held hit':>12} {'right':>6} "
          f"{'2nd|1st wrong':>13} | {'all28 held hit':>14} {'right':>6} | {'seen':>5} {'PAQ hit|unseen':>14} {'right|unseen':>12} {'right|seen':>10}")
    for n in ORDERS:
        f = sink.f[n]
        p28 = np.load(f"{MEM}/feat_all28/pred_n{n}.npy", mmap_mode="r")[:B]
        c28 = np.load(f"{MEM}/feat_all28/conf_n{n}.npy", mmap_mode="r")[:B]
        s = {"occupancy": float(np.count_nonzero(tables[n].view(np.uint16))) / (1 << log2)}
        for name, sl in (("train", tr), ("heldout", ho)):
            s[name] = paq_stats(data[sl], f[0][sl], f[1][sl], f[2][sl], f[3][sl])
            s[name + "_all28"] = count_stats(data[sl], p28[sl], c28[sl])
        # exact context seen among the training positions before the held-out chunk (64-bit hash identity)
        seen = np.isin(H[n][ho], H[n][n:te])
        hit, right = f[1][ho] > 0, f[0][ho] == data[ho]
        s["heldout_seen"] = {"seen_share": float(seen.mean()), "paq_hit_given_unseen": _rate(hit, ~seen), "paq_hit_given_seen": _rate(hit, seen),
                             "paq_right_given_hit_unseen": _rate(right, hit & ~seen), "paq_right_given_hit_seen": _rate(right, hit & seen),
                             "all28_hit_given_seen": _rate(c28[ho] > 0, seen), "all28_right_given_hit_seen": _rate(p28[ho] == data[ho], seen & (c28[ho] > 0))}
        s["heldout_diag"] = diag_stats(diag[n])
        res[n] = s
        a, b_, hs = s["train"], s["train_all28"], s["heldout_seen"]
        c, d_ = s["heldout"], s["heldout_all28"]
        fmt = lambda x: "  -  " if x is None else f"{x:.3f}"
        print(f"{n:>5} {s['occupancy']:5.3f} | {a['hit_rate']:13.3f} {fmt(a['right_given_hit']):>6} | {b_['hit_rate']:15.3f} {fmt(b_['right_given_hit']):>6} | "
              f"{c['hit_rate']:12.3f} {fmt(c['right_given_hit']):>6} {fmt(c['second_right_given_first_wrong']):>13} | {d_['hit_rate']:14.3f} "
              f"{fmt(d_['right_given_hit']):>6} | {hs['seen_share']:5.3f} {fmt(hs['paq_hit_given_unseen']):>14} {fmt(hs['paq_right_given_hit_unseen']):>12} "
              f"{fmt(hs['paq_right_given_hit_seen']):>10}", flush=True)
    print("held-out right rate per bin (share of hits): " + "; ".join(
        f"{n}: " + " ".join(f"{b}:{fmt2(res[n]['heldout']['bins'][str(b)]['right'])}({fmt2(res[n]['heldout']['bins'][str(b)]['share_of_hits'])})" for b in BINS)
        for n in ORDERS), flush=True)
    out = f"{HERE}/dry_{log2}_{mib}.json"
    json.dump({"log2": log2, "bytes": B, "train_end": te, "build_seconds": tb, "workers": WORKERS, "peak_rss_gb": peak_rss_gb(),
               "stats": {str(n): v for n, v in res.items()}}, open(out, "w"), indent=1)
    print(f"wrote {out} ({time.time() - t0:.0f} s)", flush=True)


def fmt2(x):
    return "-" if x is None else f"{x:.2f}"


def probe(log2):
    """Time one chunk of 2^20 random positions on tables of random counts (every root non-empty: whole-tree walks, the worst case)."""
    rng = np.random.default_rng(1)
    m = CHUNK
    tabs = [rng.integers(1, 256, 2 << log2, dtype=np.uint8) for _ in range(WORKERS)]
    hs = [rng.integers(0, 2**63, m, dtype=np.uint64) * np.uint64(2) + np.uint64(1) for _ in range(WORKERS)]
    nxt = rng.integers(0, 256, m, dtype=np.uint8)
    outs = [[np.zeros(m, np.uint8) for _ in range(4)] + [None, None] for _ in range(WORKERS)]
    dd = [np.zeros(NDIAG) for _ in range(WORKERS)]
    kernel()
    wall = {}
    for upd in (0, 1):
        t = time.time()
        run_order(tabs[0], log2, hs[0], nxt, 0, m, upd, m, outs[0], dd[0])
        one = time.time() - t
        t = time.time()
        with ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(lambda k: run_order(tabs[k], log2, hs[k], nxt, 0, m, upd, m, outs[k], dd[k]), range(WORKERS)))
        par = time.time() - t
        wall[upd] = par / (WORKERS * m)                      # wall seconds per position-order with WORKERS tables in flight
        print(f"2^{log2}, {'lookup + update' if upd else 'lookup only'}: one thread {one / m * 1e9:.0f} ns per position; {WORKERS} threads at once "
              f"{par:.2f} s for {WORKERS} chunks = {wall[upd] * 1e9:.0f} ns of wall per position-order", flush=True)
    N = os.path.getsize(CORPUS)
    est = (TRAIN_END * wall[1] + (N - TRAIN_END) * wall[0]) * len(ORDERS)
    print(f"kernel wall for the full build if every lookup walked the whole tree: ~{est / 60:.0f} min", flush=True)


# ---------------------------------------------------------------------------------------------------------------------------------
# the full build

def full(log2, out):
    t0 = time.time()
    out = os.path.abspath(out)
    if os.path.exists(f"{out}/features.json"):
        sys.exit(f"{out} exists: not rewriting a feature directory (a trainer may have it mapped)")
    refuse_if_in_use(out)
    os.makedirs(out, exist_ok=True)
    print(f"paq_bits.py {log2} {out}: {WORKERS} threads, available {avail_gb():.1f} GB, load {os.getloadavg()}", flush=True)
    data = np.fromfile(CORPUS, dtype=np.uint8)
    N = len(data)
    keys = {kid: np.load(f"{MEM}/feat_keys/key_n{kid}.npy", mmap_mode="r") for kid in KEY_IDS}
    sink = FileSink(out, N, ORDERS)
    tables, diag = build(data, hashes_fn(data, keys), ORDERS, log2, CHUNK, TRAIN_END, sink, stats_from=HELDOUT,
                         log=lambda s: print(s, flush=True))
    sink.close()
    occupancy = {n: float(np.count_nonzero(tables[n].view(np.uint16))) / (1 << log2) for n in ORDERS}
    del tables
    build_s = time.time() - t0
    print(f"pass done in {build_s:.0f} s, peak RSS {peak_rss_gb():.2f} GB", flush=True)
    for n in COUNTERS:
        for kind in ("pred", "conf"):
            dst = f"{out}/{kind}_n{n}.npy"
            if not os.path.exists(dst):
                os.symlink(os.path.realpath(f"{MEM}/feat_counters/{kind}_n{n}.npy"), dst)
    # stats: held out = [HELDOUT, N) (the dev windows on), and [TRAIN_END, N) as the older stats files
    stats = {}
    for n in ORDERS:
        ld = lambda kind, oid, d=out: np.load(f"{d}/{kind}_n{oid}.npy", mmap_mode="r")
        s = {"occupancy": occupancy[n]}
        for name, lo in (("heldout", HELDOUT), ("from_train_end", TRAIN_END)):
            d = data[lo:N]
            pr, cf, pr2, cf2 = (np.asarray(ld(k, o)[lo:N]) for k, o in (("pred", n), ("conf", n), ("pred", SECOND + n), ("conf", SECOND + n)))
            p28, c28 = np.asarray(ld("pred", n, f"{MEM}/feat_all28")[lo:N]), np.asarray(ld("conf", n, f"{MEM}/feat_all28")[lo:N])
            s[name] = paq_stats(d, pr, cf, pr2, cf2)
            s[name + "_all28"] = count_stats(d, p28, c28)
            if name == "heldout":
                hp, h28, r1, r28 = cf > 0, c28 > 0, pr == d, p28 == d
                s["heldout_vs_all28"] = {"paq_hit_where_all28_hits": _rate(hp, h28), "paq_hit_where_all28_misses": _rate(hp, ~h28),
                                         "paq_right_where_both_hit": _rate(r1, hp & h28), "all28_right_where_both_hit": _rate(r28, hp & h28),
                                         "paq_right_where_only_paq_hits": _rate(r1, hp & ~h28)}
                dg = diag_stats(diag[n])
                assert dg["positions"] == len(d) and all(int(diag[n][2 + x]) == int((cf == b).sum()) for x, b in enumerate(BINS)), "diag != features"
                s["heldout_diag"] = dg
                top2 = f"{MEM}/feat_top2"
                if os.path.exists(f"{top2}/pred_n{SECOND + n}.npy"):
                    t1, tc1, t2 = (np.asarray(ld(k, o, top2)[lo:N]) for k, o in (("pred", n), ("conf", n), ("pred", SECOND + n)))
                    w = (tc1 > 0) & (t1 != d)
                    s["heldout_feat_top2_27"] = {"hit_rate": float((tc1 > 0).mean()), "right_given_hit": _rate(t1 == d, tc1 > 0),
                                                 "second_right_given_first_wrong": _rate(t2 == d, w)}
        stats[str(n)] = s
        h, a = s["heldout"], s["heldout_all28"]
        print(f"order {n}: occupancy {occupancy[n]:.3f}; held-out from {HELDOUT:,}: PAQ hits {h['hit_rate']:.3f} right {h['right_given_hit']:.3f}, "
              f"right per bin " + " ".join(f"{b}:{fmt2(h['bins'][str(b)]['right'])}({fmt2(h['bins'][str(b)]['share_of_hits'])})" for b in BINS) +
              f", second right when the first is wrong {fmt2(h['second_right_given_first_wrong'])} | all28 hits {a['hit_rate']:.3f} right {a['right_given_hit']:.3f}",
              flush=True)
    rules = {"addressing": "slot = top log2 bits of corpus_memory._finalise(ctx * 0x9E3779B97F4A7C15 + node * 0xC2B2AE3D27D4EB4F), node 1..255 MSB-first bit tree, no tag",
             "update": "per training position in corpus order, 8 nodes root first: if n_b == 255 halve both; n_b += 1; if n_other > 2: n_other = n_other // 2 + 1",
             "lookup": "miss when root n0 + n1 == 0; greedy path with p(bit) = (n_bit + 0.4) / (n0 + n1 + 0.8), tie -> bit 0; second = best of the 8 one-flip paths (tie -> lowest level)",
             "conf_bins": "1: p < 0.5, 2: p < 0.75, 4: p < 0.9, 16: p >= 0.9; 0 = miss"}
    os.makedirs(TABLES_DIR, exist_ok=True)
    stats_path = f"{TABLES_DIR}/stats_{os.path.basename(out)}.json" if out.startswith(MEM + "/") else f"{out}.stats.json"
    json.dump({"log2_slots": log2, "corpus_bytes": N, "train_end": TRAIN_END, "heldout_from": HELDOUT, "chunk": CHUNK, "build_seconds": build_s,
               "peak_rss_gb": peak_rss_gb(), "workers": WORKERS, "rules": rules, "stats": stats}, open(stats_path, "w"), indent=1)
    info = json.load(open(f"{MEM}/feat/features.json"))
    for k in ("feature_digest", "compare", "table_digest", "occupancy", "train_hit_rate_by_tenth", "train_right_given_hit_by_tenth", "train_hit_rate",
              "heldout_hit_rate", "heldout_right_given_hit", "seconds_by_stage", "build_cpu_seconds", "heldout_lookup_seconds",
              "digest_and_compare_seconds", "build_mb_per_second", "loadavg_start", "loadavg_end"):
        info.pop(k, None)                                    # the base build's numbers, about other tables
    info.update(orders=[*ORDERS, *(SECOND + n for n in ORDERS), *COUNTERS], log2_slots=log2, slots_per_order=1 << log2, paq_bits=rules,
                build_seconds=build_s, stats={str(n): {"occupancy": stats[str(n)]["occupancy"], "heldout_hit_rate": stats[str(n)]["heldout"]["hit_rate"],
                                                       "heldout_right_given_hit": stats[str(n)]["heldout"]["right_given_hit"]} for n in ORDERS},
                stats_file=stats_path,
                note=f"paq_bits.py: PAQ-style bit tables (two uint8 counts per slot, addressed by (context hash, bit-tree node), no tag) of 2^{log2} slots "
                     f"per order for byte orders {list(BYTE_ORDERS)} and keys {list(KEY_IDS)}; <o> = greedy top byte with its probability bin "
                     f"(1 <0.5, 2 <0.75, 4 <0.9, 16), {SECOND}+<o> = the best one-flip second byte; one leak-free pass; counters linked from feat_counters")
    json.dump(info, open(f"{out}/features.json", "w"), indent=2)
    print(f"wrote {stats_path} and {out}/features.json; done in {time.time() - t0:.0f} s, peak RSS {peak_rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selftest"]:
        sys.exit(0 if selftest() else 1)
    elif sys.argv[1:2] == ["--dry"]:
        dry(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1:2] == ["--probe"]:
        probe(int(sys.argv[2]))
    else:
        full(int(sys.argv[1]), sys.argv[2])
