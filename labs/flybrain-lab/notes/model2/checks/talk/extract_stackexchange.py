#!/usr/bin/env python3
"""Build the Stack Exchange part of the flybrain-lab "talk" corpus.

Usage:
  nice -n 19 taskset -c 0-5,8-13 env CUDA_VISIBLE_DEVICES= \
    /games/ailab/lab/flybrain-lab/.venv/bin/python extract_stackexchange.py \
    [--sites unix.stackexchange.com,...] [--limit N] [--out PATH] [--raw DIR] [--force]

Sources: Internet Archive item "stackexchange" data dump,
  https://archive.org/download/stackexchange/<site>.7z  (CC BY-SA 4.0)
Sites/budgets (MB of text, newest threads by CreationDate kept): unix 300,
askubuntu 300, serverfault 200, superuser 150, gamedev 50.

Each .7z is downloaded (wget -c, resumable, size-checked, max 1 at a time) to
RAW/ and Posts.xml is streamed with `7z e -so`. One document per question thread:
title, blank line, question, then answers ("---" / "Answer (score N[, accepted])"),
accepted first then by score. Per-site output parts are cached in RAW/parts/;
a rerun skips existing parts (--force rebuilds). Parts are concatenated into OUT
and a sidecar OUT-with-.json is written. --limit N keeps only N threads per site
(smoke test; with --limit, parts are not cached and OUT defaults to RAW/smoke.bytes).
"""
import argparse, hashlib, html, json, os, re, subprocess, sys, time, urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

BASE = "https://archive.org/download/stackexchange/"
UA = "flybrain-lab corpus build"
SITES = [("unix.stackexchange.com", 300), ("askubuntu.com", 300), ("serverfault.com", 200),
         ("superuser.com", 150), ("gamedev.stackexchange.com", 50)]
ROOT = "/vault/ailab/corpora-src/talk/"
MAXDOC = 1_000_000


class H2T(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []; self.pre = 0; self.bq = 0; self.skip = 0
    def w(self, s): self.out.append(s)
    def handle_starttag(self, t, a):
        if t == "pre": self.w("\n\n```\n"); self.pre += 1
        elif self.pre: pass
        elif t in ("p", "div", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "hr"):
            self.w("\n\n")
            if t == "blockquote": self.bq += 1; self.w("> ")
        elif t == "li": self.w("\n- ")
        elif t == "br": self.w("\n")
        elif t in ("tr",): self.w("\n")
        elif t in ("td", "th"): self.w(" ")
        elif t in ("script", "style"): self.skip += 1
        elif t == "img":
            d = dict(a)
            if d.get("alt"): self.w("[image: %s]" % d["alt"])
    def handle_endtag(self, t):
        if t == "pre":
            self.pre = max(0, self.pre - 1)
            if not self.pre: self.w("\n```\n\n")
        elif self.pre: pass
        elif t in ("p", "div", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"):
            self.w("\n\n")
            if t == "blockquote": self.bq = max(0, self.bq - 1)
        elif t in ("script", "style"): self.skip = max(0, self.skip - 1)
    def handle_data(self, d):
        if self.skip: return
        if self.bq and not self.pre: d = d.replace("\n", "\n> ")
        self.w(d)


def to_text(h):
    p = H2T()
    p.feed(h); p.close()
    s = "".join(p.out)
    # normalise: blank lines outside fences, strip trailing spaces
    parts = re.split(r"(```\n.*?\n```)", s, flags=re.S)
    res = []
    for i, x in enumerate(parts):
        if x.startswith("```\n") and x.endswith("\n```"):
            res.append(x.rstrip("\n") if False else x)
        else:
            x = re.sub(r"[ \t]+\n", "\n", x)
            x = re.sub(r"\n{3,}", "\n\n", x)
            res.append(x)
    s = "".join(res)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def download(site, raw):
    fn = site + ".7z"; dst = os.path.join(raw, fn); url = BASE + fn
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    size = int(urllib.request.urlopen(req).headers["Content-Length"])
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return dst, size
    subprocess.run(["wget", "-c", "-q", "-U", UA, "-O", dst, url], check=True)
    assert os.path.getsize(dst) == size, "incomplete download " + fn
    return dst, size


def rows(path):
    p = subprocess.Popen(["7z", "e", "-so", path, "Posts.xml"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for ev, el in ET.iterparse(p.stdout, events=("end",)):
        if el.tag == "row":
            yield el.attrib
            el.clear()
    p.stdout.close(); p.wait()


def build_site(site, budget_mb, path, limit):
    # pass 1: sizes only
    qs = {}; asz = {}
    for a in rows(path):
        t = a.get("PostTypeId"); n = len(a.get("Body", ""))
        if t == "1":
            if n and a.get("Title") and "Id" in a:
                qs[a["Id"]] = [a["CreationDate"], n + len(a["Title"]), 0, a.get("ClosedDate") is not None]
        elif t == "2":
            asz[a["ParentId"]] = asz.get(a["ParentId"], [0, 0]); asz[a["ParentId"]][0] += n + 30; asz[a["ParentId"]][1] += 1
    for k, v in asz.items():
        if k in qs: qs[k][2] = v[0]; qs[k].append(v[1])
    cand = [(v[0], k, v) for k, v in qs.items() if not (v[3] and v[2] == 0)]
    cand.sort(reverse=True)
    budget = budget_mb * 1_000_000 / 0.85  # html bytes ~ text bytes / 0.85
    sel = set(); tot = 0
    for d, k, v in cand:
        if limit and len(sel) >= limit: break
        s = v[1] + v[2]
        if tot + s > budget and not limit: continue
        sel.add(k); tot += s
    # pass 2
    Q = {}; A = {}
    for a in rows(path):
        t = a.get("PostTypeId")
        if t == "1" and a.get("Id") in sel:
            Q[a["Id"]] = a
        elif t == "2" and a.get("ParentId") in sel:
            A.setdefault(a["ParentId"], []).append(a)
    docs = []
    for qid in sorted(Q, key=int):
        q = Q[qid]
        body = to_text(q.get("Body", ""))
        title = html.unescape(q.get("Title", "")).strip()
        ans = A.get(qid, [])
        if not body or not title: continue
        if not ans and len(body) < 200: continue
        acc = q.get("AcceptedAnswerId")
        ans.sort(key=lambda a: (a["Id"] != acc, -int(a.get("Score", 0))))
        out = [title, "", body]
        for a in ans:
            b = to_text(a.get("Body", ""))
            if not b: continue
            out += ["", "---", "Answer (score %s%s)" % (a.get("Score", 0), ", accepted" if a["Id"] == acc else ""), "", b]
        txt = "\n".join(out) + "\n"
        b = txt.encode("utf-8", "replace")
        if len(b) > MAXDOC: continue
        docs.append((qid, b))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=ROOT + "stackexchange.bytes")
    ap.add_argument("--raw", default=ROOT + "stackexchange/raw")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.raw, exist_ok=True); parts = os.path.join(a.raw, "parts_smoke" if a.limit else "parts"); os.makedirs(parts, exist_ok=True)
    want = a.sites.split(",") if a.sites else None
    if a.limit and a.out == ROOT + "stackexchange.bytes": a.out = os.path.join(a.raw, "smoke.bytes")
    seen = set(); per = {}; total = 0; ndocs = 0; urls = []; t0 = time.time()
    with open(a.out + ".tmp", "wb") as fo:
        for site, mb in SITES:
            if want and site not in want: continue
            part = os.path.join(parts, site + ".bytes")
            cache = os.path.join(parts, site + ".json")
            path, size = download(site, a.raw); urls.append(BASE + site + ".7z")
            if not a.limit and not a.force and os.path.exists(part) and os.path.exists(cache):
                info = json.load(open(cache))
            else:
                docs = build_site(site, mb, path, a.limit)
                n = 0; nb = 0
                with open(part, "wb") as fp:
                    for qid, b in docs:
                        h = hashlib.md5(b).digest()
                        if h in seen: continue
                        seen.add(h)
                        d = b"\n// ==== " + ("stackexchange/%s/%s" % (site, qid)).encode() + b" ====\n" + b
                        fp.write(d); n += 1; nb += len(d)
                info = {"docs": n, "bytes": nb}
                json.dump(info, open(cache, "w"))
            with open(part, "rb") as fp:
                while True:
                    c = fp.read(1 << 24)
                    if not c: break
                    fo.write(c)
            per[site] = info; total += info["bytes"]; ndocs += info["docs"]
            print(site, info, "%.0fs" % (time.time() - t0), flush=True)
    os.replace(a.out + ".tmp", a.out)
    side = {"source": "stackexchange", "urls": urls, "license": "CC BY-SA 4.0 (Stack Exchange user content)",
            "docs": ndocs, "bytes": total, "per_site": per, "built": time.strftime("%Y-%m-%d"),
            "notes": "Internet Archive stackexchange dump; one doc per question thread, answers accepted-first then by score; newest threads by CreationDate up to per-site budget (unix 300MB, askubuntu 300, serverfault 200, superuser 150, gamedev 50); closed questions without answers dropped; unanswered kept if body>=200 chars; dups and >1MB docs dropped."}
    json.dump(side, open(a.out[:-6] + ".json" if a.out.endswith(".bytes") else a.out + ".json", "w"), indent=1)
    print("total", ndocs, total)

main()
