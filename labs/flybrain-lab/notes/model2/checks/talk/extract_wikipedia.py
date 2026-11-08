"""Build the wikipedia / wikipedia-simple talk corpora (flybrain-lab).

Sources: Hugging Face dataset wikimedia/wikipedia, configs 20231101.en
(technical/scientific filter, stop at 500 MB of kept text) and 20231101.simple
(all articles >= 300 chars). License CC BY-SA 4.0. Fallback (not implemented,
use only if HF is unavailable): dumps.wikimedia.org pages-articles bz2.

Usage:
  nice -n 19 taskset -c 0-5,8-13 env CUDA_VISIBLE_DEVICES= \
    .venv/bin/python extract_wikipedia.py --config en|simple [--limit N] [--out-dir D]
--limit N: smoke run (first N articles of shard 0, writes <name>.smoke.* in --out-dir).
Idempotent: raw shards are downloaded only if absent; output is rebuilt atomically.
Doc format: b"\\n// ==== wikipedia/<en|simple>/<id> ====\\n" + title + "\\n\\n" + text + "\\n".
"""
import argparse, datetime, hashlib, json, os, re
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "wikimedia/wikipedia"
BASE = "/vault/ailab/corpora-src/talk"
RAW = BASE + "/wikipedia/raw"
TARGET = 500_000_000
MAXDOC = 1_000_000
VOCAB = set("""software algorithm protocol compiler kernel linux unix server database processor circuit voltage electron physics chemistry mathematics theorem equation engineering engine network encryption programming computer hardware semiconductor transistor satellite rocket aircraft telescope molecule enzyme genome neuron electricity magnet turbine reactor bitcoin blockchain cryptography statistics probability geometry calculus quantum relativity thermodynamics signal frequency antenna radar laser optics robot sensor bandwidth gpu cpu firmware filesystem bytecode microcontroller oscillator capacitor amplifier spectrum vector matrix polynomial integer binary hexadecimal simulation emulator virtualization container hypervisor protocol router ethernet wireless modem compiler interpreter syntax recursion iteration dataset neural inference gradient""".split())
WORD = re.compile(r"[a-z]+")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["en", "simple"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=BASE)
    a = ap.parse_args()
    en = a.config == "en"
    name = "wikipedia" if en else "wikipedia-simple"
    if a.limit: name += ".smoke"
    os.makedirs(RAW, exist_ok=True); os.makedirs(a.out_dir, exist_ok=True)
    cfg = "20231101.en" if en else "20231101.simple"
    nsh = 41 if en else 1
    out = f"{a.out_dir}/{name}.bytes"; tmp = out + ".tmp"
    seen = kept = nbytes = 0; used = []; seen_hash = set(); stop = False
    with open(tmp, "wb") as f:
        for i in range(nsh):
            fn = f"{cfg}/train-{i:05d}-of-{nsh:05d}.parquet"
            p = hf_hub_download(REPO, fn, repo_type="dataset", local_dir=RAW)
            used.append(fn)
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=2000, columns=["id", "title", "text"]):
                for aid, title, text in zip(*[batch.column(c).to_pylist() for c in ("id", "title", "text")]):
                    if a.limit and seen >= a.limit: stop = True; break
                    seen += 1
                    if not text: continue
                    if en:
                        if len(text) < 2000: continue
                        if len(VOCAB & set(WORD.findall(text.lower()))) < 4: continue
                    elif len(text) < 300: continue
                    body = f"{title}\n\n{text.rstrip()}\n".encode("utf-8")
                    if len(body) > MAXDOC: continue
                    h = hashlib.sha1(body).digest()
                    if h in seen_hash: continue
                    seen_hash.add(h)
                    d = f"\n// ==== wikipedia/{a.config}/{aid} ====\n".encode() + body
                    f.write(d); kept += 1; nbytes += len(d)
                    if en and nbytes >= TARGET: stop = True; break
                if stop: break
            if stop: break
    os.replace(tmp, out)
    meta = {"source": "wikipedia" if en else "wikipedia-simple",
            "urls": [f"https://huggingface.co/datasets/{REPO}/tree/main/{cfg}"],
            "license": "CC BY-SA 4.0 (Wikipedia text)", "docs": kept, "bytes": nbytes,
            "shards_used": used, "built": datetime.date.today().isoformat(),
            "notes": ("English Wikipedia 20231101, technical slice, stops at 500 MB kept" if en else
                      "Simple English Wikipedia 20231101, all articles >= 300 chars") + "; docs > 1 MB and exact duplicates dropped"
                      + ("; SMOKE RUN" if a.limit else "")}
    if en:
        meta["filter"] = {"rule": ">= 2000 chars and >= 4 distinct whole-word case-insensitive matches from vocabulary",
                          "vocabulary": sorted(VOCAB)}
        meta["articles_seen"] = seen
    else:
        meta["articles_seen"] = seen
    json.dump(meta, open(f"{a.out_dir}/{name}.json", "w"), indent=1)
    print(name, "seen", seen, "kept", kept, "bytes", nbytes, "shards", len(used))

main()
