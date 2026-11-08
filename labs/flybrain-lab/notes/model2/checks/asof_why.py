"""Why the as-of memory wins on his turns and loses on the dialogue: per window, the full arm vs the as-of arm, split by whether the
memory changed, by how many of the full memory's chunks were dated after the session, by the files' age, by the session's month.
Numbers and file names only."""
import json, glob, os, hashlib, datetime as dt, numpy as np
from collections import Counter, defaultdict
WORLD = "/games/ailab/corpora/kyle/world/files"; CH = 1500; R = "results/tables"; SW = f"{R}/qwen-sweep"
chunks = []; chunk_file = []; files = []
for p in sorted(glob.glob(os.path.join(WORLD, "**", "*"), recursive=True)):
    if not os.path.isfile(p): continue
    b = open(p, "rb").read(); fi = len(files); files.append(p)
    for s in range(0, max(1, len(b) - CH // 2), CH // 2 if len(b) > CH else CH):
        c = b[s:s + CH]
        if len(c) < 64: continue
        chunks.append(c); chunk_file.append(fi)
digest = hashlib.sha256(b"".join(hashlib.sha256(c).digest() for c in chunks)).hexdigest()
D = json.load(open("/games/ailab/corpora/kyle/asof_dates.json")); FD = D["files"]
def rel(i): return os.path.relpath(files[chunk_file[i]], WORLD)
def fdate(i): return FD.get(rel(i)) or None
def day(s): return dt.date.fromisoformat(s) if s else None
A = json.load(open(f"{R}/qwen/qwen_memory.json"))
def load(arm, key):
    M = json.load(open(f"{SW}/{arm}/qwen_memory.json")); C = json.load(open(f"{SW}/{arm}/choices.json"))
    assert C["pool"]["digest"] == digest, arm
    return M, C
pairs = [("headers vs hdrasof", "headers", "dedup200", "choices", "hdrasof", "choices"), ("spans vs spansasof", "spans", "dedup200", "span_choices", "spansasof", "choices")]
for title, full, fullch, fkey, asof, akey in pairs:
    MF = json.load(open(f"{SW}/{full}/qwen_memory.json")); CF = json.load(open(f"{SW}/{fullch}/choices.json")); assert CF["pool"]["digest"] == digest
    MA, CA = load(asof, akey)
    for name, label, wins in [("kyle", "his turns", D["kyle_windows"]), ("dlg", "the dialogue", D["dialogue_windows"][::4])]:
        w = np.array(A[name]["warm"]["per_window"]); bf = np.array(MF[name]["memory"]["per_window"]); ba = np.array(MA[name]["memory"]["per_window"])
        chf = CF[fkey][name]; cha = CA[akey][name]; n = len(w); assert len(chf) == len(cha) == n == len(wins), (len(chf), len(cha), n, len(wins))
        gf = bf - w; ga = ba - w; d = ba - bf
        changed = np.array([set(a) != set(b) for a, b in zip(chf, cha)]); dated = np.array([bool(s) for s in wins])
        n_after = np.array([sum(1 for c in a if fdate(c) and s and day(fdate(c)) > day(s)) for a, s in zip(chf, wins)])
        n_unk = np.array([sum(1 for c in a if not fdate(c)) for a in chf])
        print(f"\n== {title} on {label}: {n} windows, {dated.sum()} dated; full {gf.mean():+.4f}, as-of {ga.mean():+.4f}, delta {d.mean():+.4f} (median {np.median(d):+.4f}; as-of better >0.02 on {(d < -0.02).sum()}, worse on {(d > 0.02).sum()})")
        for lab, m in [("memory unchanged", ~changed), ("memory changed", changed)]:
            if m.sum(): print(f"  {lab:18s} n {m.sum():4d}  full {gf[m].mean():+.4f}  as-of {ga[m].mean():+.4f}  delta {d[m].mean():+.4f}")
        print("  by the full memory's after-dated chunks (of its chunks):")
        for k in sorted(set(n_after)):
            m = n_after == k; print(f"    after {k}: n {m.sum():4d}  full {gf[m].mean():+.4f}  as-of {ga[m].mean():+.4f}  delta {d[m].mean():+.4f}  (undated chunks mean {n_unk[m].mean():.2f})")
        # the files' age: days from the file's date to the session, for the memory's dated chunks
        def ages(ch):
            out = []
            for a, s in zip(ch, wins):
                if not s: continue
                for c in a:
                    f = fdate(c)
                    if f: out.append((day(s) - day(f)).days)
            return np.array(out)
        af, aa = ages(chf), ages(cha)
        print(f"  age of the memory's files (days before the session; negative = after): full median {np.median(af):+.0f}, within 30 days {np.mean(np.abs(af) <= 30):.2f}, after {np.mean(af < 0):.2f}; as-of median {np.median(aa):+.0f}, within 30 days {np.mean(np.abs(aa) <= 30):.2f}, after {np.mean(aa < 0):.2f}")
        # by the session's month
        print("  by the session's month: n, full, as-of, delta")
        bym = defaultdict(list)
        for i, s in enumerate(wins):
            if s: bym[s[:7]].append(i)
        for mo in sorted(bym):
            ix = np.array(bym[mo]); print(f"    {mo}: n {len(ix):4d}  full {gf[ix].mean():+.4f}  as-of {ga[ix].mean():+.4f}  delta {d[ix].mean():+.4f}")
        # top files: what the as-of memory reads instead
        def tops(ch): return Counter(rel(a[0]) for a in ch if a)
        tf, ta = tops(chf), tops(cha)
        print("  most-picked top files, full: " + "; ".join(f"{f} x{k}" for f, k in tf.most_common(6)))
        print("  most-picked top files, as-of: " + "; ".join(f"{f} x{k}" for f, k in ta.most_common(6)))
        ef = Counter(os.path.splitext(rel(a[0]))[1] for a in chf if a); ea = Counter(os.path.splitext(rel(a[0]))[1] for a in cha if a)
        print(f"  top chunk kind, full: {dict(ef.most_common(4))}; as-of: {dict(ea.most_common(4))}")
