"""Qwen on the completion test's spots (the other side of model #1's viability test, item 2): the same held-out lines complete_test.py
scores, each spot's file prefix sent to a llama-server (the lab's CUDA build of llama.cpp in lab/llama.cpp-qwen4exp, started for the
run on one card, no port kept: not the model server), greedy, stopped at the newline, and scored the same way (exact line, the shared
prefix's share, the first token). The server: see qwen_complete.sh beside this (it starts one on 127.0.0.1:8099, runs this, stops it).

usage: qwen_complete.py --stream heldout.bytes --url http://127.0.0.1:8099 [--spots 60] [--seed 0] [--ext gd] [--min-prefix 256]
                        [--max-line 120] [--max-prefix 6000] [--out results.json] [--show 8]
The spots, seed and filters match complete_test.py's exactly (its spots_of), so the two JSONs compare row by row. The prefix sent is
the file from its header, cut to the last --max-prefix bytes at a line start when longer (Qwen's context, in tokens, is far longer than
our reader's 2,048 bytes; the cut keeps the request short, 6 KB is about 1,500 tokens of code)."""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from complete_test import file_name, files_of, first_ident, spots_of  # noqa: E402


def completion(url: str, prompt: bytes, max_line: int) -> bytes:
    body = json.dumps({"prompt": prompt.decode("latin-1"), "n_predict": max_line, "temperature": 0.0, "top_k": 1, "stop": ["\n"],
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(f"{url}/completion", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["content"].encode("latin-1", "replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stream", required=True); ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--spots", type=int, default=60); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--ext", default="gd")
    ap.add_argument("--min-prefix", type=int, default=256); ap.add_argument("--max-line", type=int, default=120); ap.add_argument("--max-prefix", type=int, default=6000)
    ap.add_argument("--out", default=None); ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    stream = open(a.stream, "rb").read()
    files = files_of(stream)
    spots = spots_of(stream, a.spots, a.seed, a.min_prefix, set(e for e in a.ext.lower().split(",") if e))
    print(f"{len(spots)} spots in {len({s['file'] for s in spots})} files of {os.path.basename(a.stream)}; {a.url}", flush=True)
    rows = []; t0 = time.time()
    for k, sp in enumerate(spots):
        f, _ = files[sp["file"]]
        prefix, real = f[:sp["start"]], f[sp["start"]:sp["end"]]
        if len(prefix) > a.max_prefix:
            cut = prefix.find(b"\n", len(prefix) - a.max_prefix) + 1
            prefix = prefix[cut:]
        got = completion(a.url, prefix, a.max_line)
        real_s, got_s = real.rstrip(), got.rstrip()
        common = 0
        for x, y in zip(real_s, got_s):
            if x != y:
                break
            common += 1
        row = {"file": file_name(f), "line_at": sp["start"], "real": real.decode("latin-1"), "got": got.decode("latin-1"),
               "exact": real_s == got_s, "prefix_share": common / max(len(real_s), 1),
               "first_token": first_ident(real_s.lstrip()) == first_ident(got_s.lstrip()) and bool(first_ident(real_s.lstrip()))}
        rows.append(row)
        if a.show and k < a.show:
            verdict = "exact" if row["exact"] else f"{row['prefix_share']:.0%} prefix"
            print(f"  [{row['file']}] real: {row['real'].strip()[:70]!r}\n      got: {row['got'].strip()[:70]!r}  ({verdict})", flush=True)
    n = len(rows)
    res = {"model": "qwen (llama-server)", "url": a.url, "stream": a.stream, "spots": a.spots, "seed": a.seed, "ext": a.ext, "max_prefix": a.max_prefix,
           "n": n, "exact": sum(r["exact"] for r in rows) / n, "prefix_share": sum(r["prefix_share"] for r in rows) / n,
           "half_or_more": sum(r["prefix_share"] >= 0.5 for r in rows) / n, "first_token": sum(r["first_token"] for r in rows) / n,
           "seconds": round(time.time() - t0, 1), "rows": rows}
    print(f"exact {res['exact']:.0%}, first token right {res['first_token']:.0%}, half the line or more {res['half_or_more']:.0%}, "
          f"mean shared prefix {res['prefix_share']:.0%} ({n} spots, {res['seconds']} s)")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
