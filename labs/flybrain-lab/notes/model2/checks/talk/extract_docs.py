"""Build the docs talk corpus (software documentation from the owner's domains).

Usage: nice -n 19 taskset -c 0-5,8-13 env CUDA_VISIBLE_DEVICES= \
  /games/ailab/lab/flybrain-lab/.venv/bin/python extract_docs.py
Sources (raw under /vault/ailab/corpora-src/talk/docs/raw/, depth-1 clones, skipped if present):
  godot-docs  https://github.com/godotengine/godot-docs (master)   *.rst
  pytorch     https://github.com/pytorch/pytorch (sparse: docs/source) docs/source/**/*.rst,*.md
  cosmos-sdk  https://github.com/cosmos/cosmos-sdk                 docs/**/*.md
  docs        https://github.com/celestiaorg/docs                  **/*.md and **/*.mdx
  archwiki    Arch package arch-wiki-docs 20261003-1 (geo.mirror.pkgbuild.com/extra/os/x86_64/),
              html/en/*.html -> text (<pre> verbatim, content = div.mw-parser-output + h1 title)
Out: ../docs.bytes + docs.json ("\\n// ==== docs/<repo>/<path> ====\\n" + text). Docs > 1 MB, empty
or exact duplicates dropped. Budget 200 MB: pytorch is trimmed first (largest files last dropped).
Idempotent.
"""
import json, datetime, hashlib, re, os, subprocess, urllib.request, collections
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path("/vault/ailab/corpora-src/talk"); RAW = ROOT / "docs" / "raw"
BUDGET = 200_000_000
ARCH_PKG = "arch-wiki-docs-20261003-1-any.pkg.tar.zst"
ARCH_URL = "https://geo.mirror.pkgbuild.com/extra/os/x86_64/" + ARCH_PKG
GIT = [("godot-docs", "https://github.com/godotengine/godot-docs", None),
       ("cosmos-sdk", "https://github.com/cosmos/cosmos-sdk", None),
       ("docs", "https://github.com/celestiaorg/docs", None)]
RUN = lambda *a, **k: subprocess.run(a, check=True, **k)

for n, u, _ in GIT:
    if not (RAW / n / ".git").exists():
        RUN("git", "clone", "-q", "--depth", "1", u, str(RAW / n))
if not (RAW / "pytorch" / ".git").exists():
    RUN("git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
        "https://github.com/pytorch/pytorch", str(RAW / "pytorch"))
    RUN("git", "-C", str(RAW / "pytorch"), "sparse-checkout", "set", "docs/source")
if not (RAW / ARCH_PKG).exists():
    urllib.request.urlretrieve(ARCH_URL, str(RAW / "arch.part")); os.rename(RAW / "arch.part", RAW / ARCH_PKG)
if not (RAW / "arch" / "usr").exists():
    (RAW / "arch").mkdir(exist_ok=True)
    RUN("tar", "--use-compress-program=unzstd", "-xf", str(RAW / ARCH_PKG), "-C", str(RAW / "arch"))

def commit(n): return subprocess.check_output(["git", "-C", str(RAW / n), "rev-parse", "HEAD"], text=True).strip()

def safe(p): return re.sub(r"[^A-Za-z0-9._/+-]", "_", p)

def walk(repo, base, exts):
    b = RAW / repo / base
    for dp, dn, fn in os.walk(b):
        dn[:] = sorted(d for d in dn if d not in (".git", "node_modules", "_build"))
        for f in sorted(fn):
            if f.endswith(exts):
                p = Path(dp) / f
                if p.is_symlink(): continue
                yield f"docs/{repo}/{p.relative_to(RAW / repo)}", p

# ---- arch wiki HTML -> text
SKIP_CLASS = ("mw-editsection", "toc", "navbox", "catlinks", "printfooter", "noprint", "metadata", "mw-empty-elt")
VOID = {"br", "hr", "img", "input", "meta", "link", "wbr", "col", "area", "base", "source"}
BLOCK = {"p", "div", "li", "ul", "ol", "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "dl", "dt", "dd",
         "blockquote", "section", "figure", "figcaption", "caption", "br", "hr"}
class Wiki(HTMLParser):
    def __init__(s):
        super().__init__(convert_charrefs=True)
        s.out = []; s.title = []; s.in_h1 = False; s.stack = []  # stack of (tag, is_content_root/skip)
        s.depth_root = None; s.skip = 0; s.pre = 0; s.in_skipscript = 0
    def start(s, t, a):
        a = dict(a); cl = a.get("class", "") or ""
        if t == "h1" and a.get("id") == "firstHeading": s.in_h1 = True
        if s.depth_root is None:
            if t == "div" and "mw-parser-output" in cl.split(): s.depth_root = len(s.stack)
            else: return False
        return True
    def handle_starttag(s, t, a):
        if t == "h1" and dict(a).get("id") == "firstHeading": s.in_h1 = True
        if t in VOID: 
            if s.depth_root is not None and not s.skip and t in BLOCK: s.out.append("\n")
            return
        a = dict(a); cl = (a.get("class") or "").split()
        if s.depth_root is None:
            if t == "div" and "mw-parser-output" in cl:
                s.depth_root = len(s.stack); s.stack.append((t, False))
            else:
                s.stack.append((t, False))
            return
        skip = t in ("script", "style", "sup" if "reference" in cl else "script") or a.get("id") in ("toc",) \
            or any(c in SKIP_CLASS for c in cl) or "navigation-not-searchable" in cl
        s.stack.append((t, skip)); s.skip += skip
        if not s.skip:
            if t == "pre": s.pre += 1; s.out.append("\n")
            elif t in BLOCK: s.out.append("\n")
            if t == "li": s.out.append("* ")
            if re.fullmatch(r"h[1-6]", t): s.out.append("\n" + "#" * int(t[1]) + " ")
    def handle_endtag(s, t):
        if t == "h1": s.in_h1 = False
        if t in VOID or not s.stack: return
        # pop to the matching tag (tolerate unclosed)
        for i in range(len(s.stack) - 1, -1, -1):
            if s.stack[i][0] == t: break
        else: return
        while len(s.stack) > i:
            tg, sk = s.stack.pop(); s.skip -= sk
            if s.depth_root is not None and not s.skip:
                if tg == "pre": s.pre -= 1; s.out.append("\n")
                elif tg in BLOCK: s.out.append("\n")
            if s.depth_root is not None and len(s.stack) == s.depth_root:
                s.depth_root = -1
    def handle_data(s, d):
        if s.in_h1: s.title.append(d)
        if s.depth_root is None or s.depth_root == -1 or s.skip: return
        if s.pre: s.out.append(d)
        else: s.out.append(re.sub(r"\s+", " ", d))

def wiki_text(path):
    p = Wiki(); p.feed(path.read_text("utf-8", "replace")); p.close()
    body = "".join(p.out)
    lines = [l.rstrip() if "\n" in body else l for l in body.split("\n")]
    body = "\n".join(l.strip() if not l.startswith((" ", "\t")) else l for l in lines)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    title = " ".join("".join(p.title).split())
    return title, body

def arch_docs():
    d = RAW / "arch/usr/share/doc/arch-wiki/html/en"
    used = set()
    for f in sorted(os.listdir(d)):
        if not f.endswith(".html") or re.match(r"[0-9a-f]{32}\.html$", f) or f.startswith("File:"): continue
        title, body = wiki_text(d / f)
        if not body: continue
        name = safe(f[:-5]); 
        if name.lower() in used: name += "_" + hashlib.sha1(f.encode()).hexdigest()[:6]
        used.add(name.lower())
        yield f"docs/archwiki/{name}", ((title + "\n\n") if title else "") + body + "\n"

def lic(repo, names):
    for n in names:
        p = RAW / repo / n
        if p.exists(): return p.read_text("utf-8", "replace").strip().splitlines()[0:3]
    return None

sources = [
    ("godot-docs", lambda: walk("godot-docs", ".", (".rst",))),
    ("cosmos-sdk", lambda: walk("cosmos-sdk", "docs", (".md",))),
    ("docs", lambda: walk("docs", ".", (".md", ".mdx"))),
    ("archwiki", arch_docs),
    ("pytorch", lambda: ((k, p) for k, p in walk("pytorch", "docs/source", (".rst", ".md")))),
]
seen = set(); per = collections.Counter(); perb = collections.Counter(); skipped = collections.Counter()
recs = {}  # repo -> list of (path, bytes)
for repo, gen in sources:
    L = []
    for path, src in gen():
        if isinstance(src, Path):
            try: text = src.read_text("utf-8")
            except UnicodeDecodeError: skipped[repo + ":not_utf8"] += 1; continue
        else: text = src
        text = text.replace("\r\n", "\n").replace("\x00", "")
        if not text.strip(): skipped[repo + ":empty"] += 1; continue
        if not text.endswith("\n"): text += "\n"
        b = text.encode("utf-8")
        if len(b) > 1_000_000: skipped[repo + ":over_1MB"] += 1; continue
        h = hashlib.sha1(b).digest()
        if h in seen: skipped[repo + ":duplicate"] += 1; continue
        seen.add(h); L.append((safe(path), b))
    recs[repo] = L
total = sum(len(b) + len(p) + 20 for L in recs.values() for p, b in L)
if total > BUDGET:  # trim pytorch first (drop from the end, alphabetical)
    L = recs["pytorch"]
    while L and total > BUDGET:
        p, b = L.pop(); total -= len(b) + len(p) + 20; skipped["pytorch:trimmed_budget"] += 1
out = bytearray()
for repo, L in recs.items():
    for p, b in L:
        rec = b"\n// ==== " + p.encode() + b" ====\n" + b
        out += rec; per[repo] += 1; perb[repo] += len(rec)
(ROOT / "docs.bytes").write_bytes(out)
meta = {
 "source": "docs", "urls": [u for _, u, _ in GIT] + ["https://github.com/pytorch/pytorch (docs/source)", ARCH_URL],
 "license": "godot-docs CC BY 3.0 (LICENSE.txt); pytorch BSD-style (LICENSE); cosmos-sdk Apache-2.0 (LICENSE); "
            "celestiaorg/docs has no LICENSE file in the repo (task brief: Apache/CC); Arch wiki GFDL-1.3-or-later (package metadata)",
 "docs": sum(per.values()), "bytes": len(out), "built": datetime.date.today().isoformat(),
 "per_repo": {r: {"docs": per[r], "bytes": perb[r]} for r in per},
 "commits": {n: commit(n) for n in ("godot-docs", "pytorch", "cosmos-sdk", "docs")},
 "notes": "arch package " + ARCH_PKG + " html/en only (hash-named localized main pages and File: pages skipped); "
          "celestia docs include .mdx files; skipped: " + json.dumps(dict(skipped))}
(ROOT / "docs.json").write_text(json.dumps(meta, indent=1))
print(meta["docs"], meta["bytes"], meta["per_repo"], dict(skipped))
