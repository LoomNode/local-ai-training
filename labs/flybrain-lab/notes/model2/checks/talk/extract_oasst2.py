"""Build the oasst2 talk corpus (English OpenAssistant message trees).

Usage: nice -n 19 taskset -c 0-5,8-13 env CUDA_VISIBLE_DEVICES= \
  /games/ailab/lab/flybrain-lab/.venv/bin/python extract_oasst2.py
Source: HF dataset OpenAssistant/oasst2 (data/train + data/validation parquet), Apache-2.0.
Raw: /vault/ailab/corpora-src/talk/oasst2/raw/ ; out: ../oasst2.bytes + oasst2.json.
English (lang=en) non-deleted messages only; trees whose root is not English dropped.
One doc per tree: depth-first walk, "User: "/"Assistant: " blocks separated by a blank
line; sibling branches separated by a line "---". Idempotent.
"""
import json, datetime, hashlib, collections
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path("/vault/ailab/corpora-src/talk"); RAW = ROOT / "oasst2" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
files = ["data/train-00000-of-00001-88ba0162028a73fc.parquet",
         "data/validation-00000-of-00001-1deeef95c3248fe0.parquet"]
rows = []
for f in files:
    p = RAW / f
    if not p.exists():
        hf_hub_download("OpenAssistant/oasst2", f, repo_type="dataset", local_dir=str(RAW))
    rows += pq.read_table(p, columns=["message_id", "parent_id", "message_tree_id", "role",
                                      "text", "lang", "deleted"]).to_pylist()
for f in ["README.md"]:
    if not (RAW / f).exists():
        hf_hub_download("OpenAssistant/oasst2", f, repo_type="dataset", local_dir=str(RAW))

trees = collections.defaultdict(list)
for r in rows:
    if r["lang"] == "en" and not r["deleted"] and r["text"] and r["text"].strip():
        trees[r["message_tree_id"]].append(r)

def render(m, kids):
    lab = "User" if m["role"] == "prompter" else "Assistant"
    here = f"{lab}: {m['text'].strip()}"
    ch = kids.get(m["message_id"], [])
    if not ch:
        return here
    parts = [render(c, kids) for c in ch]
    return here + "\n\n" + "\n---\n".join(parts)

out = bytearray(); n = 0; seen = set(); skipped = collections.Counter()
for tid in sorted(trees):
    ms = trees[tid]
    ids = {m["message_id"] for m in ms}
    roots = [m for m in ms if m["parent_id"] is None]
    if not roots:
        skipped["root_not_english"] += 1; continue
    kids = collections.defaultdict(list)
    for m in ms:
        # orphans (parent dropped) are unreachable from the root and fall out of the walk
        if m["parent_id"] is not None:
            kids[m["parent_id"]].append(m)
    text = render(roots[0], kids) + "\n"
    b = text.encode("utf-8")
    if len(b) > 1_000_000: skipped["over_1MB"] += 1; continue
    h = hashlib.sha1(b).digest()
    if h in seen: skipped["duplicate"] += 1; continue
    seen.add(h)
    out += b"\n// ==== oasst2/" + tid.encode() + b" ====\n" + b; n += 1
(ROOT / "oasst2.bytes").write_bytes(out)
(ROOT / "oasst2.json").write_text(json.dumps({
    "source": "oasst2", "urls": ["https://huggingface.co/datasets/OpenAssistant/oasst2"],
    "license": "Apache-2.0", "docs": n, "bytes": len(out),
    "built": datetime.date.today().isoformat(),
    "notes": "train+validation parquet; lang==en, deleted dropped, trees with non-English root dropped; "
             "one doc per tree, DFS, siblings separated by '---'; skipped: " + json.dumps(dict(skipped))}, indent=1))
print(n, len(out), dict(skipped))
