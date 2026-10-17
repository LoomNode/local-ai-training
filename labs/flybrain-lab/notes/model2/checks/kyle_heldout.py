#!/usr/bin/env python
"""Kyle's held-out set (R112): his own typed turns from the Claude Code transcripts on this box, one document per session, in the
corpus format ("\\n// ==== path ====\\n" + text), never trained on. Secrets are scrubbed by pattern (counts reported, never the text).
Output: /games/ailab/corpora/kyle/kyle.bytes, layout.json (all of it is dev: offset 0, windows of 2,048 bytes), windows.txt (every window).
usage: kyle_heldout.py [--root ~/.claude/projects] [--out /games/ailab/corpora/kyle]
"""
import argparse, json, os, re, sys, time
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/.claude/projects"))
ap.add_argument("--out", default="/games/ailab/corpora/kyle")
ap.add_argument("--extra", nargs="*", default=[os.path.expanduser("~/TODO.md")], help="his own files to add as documents")
ap.add_argument("--T", type=int, default=2048)
a = ap.parse_args()

SKIP_PREFIX = ("<", "[Request interrupted", "This session is being continued", "Caveat:", "Implement the following", "Base directory for this skill")
TAG_BLOCK = re.compile(r"<(pasted_content|system-reminder|task-notification|command-name|command-message|command-args|local-command-stdout|local-command-caveat)[^>]*>.*?</\1>", re.S)
SECRETS = [  # (name, pattern): replaced by [redacted]
    ("openai/anthropic key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{16,}")),
    ("aws key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("hf token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("long hex", re.compile(r"\b[0-9a-fA-F]{40,}\b")),
    ("long base64", re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{48,}={0,2}(?![A-Za-z0-9+/])")),
    ("password assignment", re.compile(r"(?i)\b(password|passwd|passphrase|secret|token)\s*[:=]\s*\S+")),
]

def text_of(content):
    if isinstance(content, str): return content
    parts = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text": parts.append(b.get("text", ""))
    return "\n".join(parts)

sessions = defaultdict(list)   # (project, sessionId) -> [(ts, text)]
seen = set(); n_lines = n_user = n_kept = 0
for dirpath, _, files in os.walk(a.root):
    project = os.path.basename(dirpath)
    for fn in files:
        if not fn.endswith(".jsonl"): continue
        with open(os.path.join(dirpath, fn), "rb") as f:
            for line in f:
                n_lines += 1
                try: d = json.loads(line)
                except Exception: continue
                if d.get("type") != "user" or d.get("isSidechain") or d.get("isMeta"): continue
                if d.get("userType") not in (None, "external"): continue
                n_user += 1
                t = text_of((d.get("message") or {}).get("content"))
                t = TAG_BLOCK.sub("", t).strip()
                if not t or t.startswith(SKIP_PREFIX): continue
                if t in seen: continue
                seen.add(t); n_kept += 1
                sessions[(project, d.get("sessionId") or fn[:-6])].append((d.get("timestamp") or "", t))

docs = []
for (project, sid), msgs in sessions.items():
    msgs.sort()
    first = msgs[0][0][:10] if msgs else ""
    docs.append((f"kyle/{project.strip('-')}/{first}-{sid[:8]}", "\n\n".join(m for _, m in msgs)))
docs.sort(key=lambda x: x[0])
for p in a.extra:
    if os.path.isfile(p): docs.append((f"kyle/files/{os.path.basename(p)}", open(p, encoding="utf-8", errors="replace").read()))

redactions = defaultdict(int)
def scrub(t):
    for name, pat in SECRETS:
        t, n = pat.subn("[redacted]", t); redactions[name] += n
    return t

os.makedirs(a.out, exist_ok=True)
out = os.path.join(a.out, "kyle.bytes")
total = 0
with open(out, "wb") as f:
    for path, t in docs:
        t = scrub(t)
        if not t.endswith("\n"): t += "\n"
        f.write(b"\n// ==== " + path.encode() + b" ====\n" + t.encode("utf-8", "replace")); total = f.tell()
os.chmod(out, 0o600)
n_win = total // a.T
json.dump({"corpus": out, "built": time.strftime("%F"), "builder": "kyle_heldout.py", "train_end": 0, "dev_offset": 0, "dev_bytes": total,
           "test_offset": total, "test_bytes": 0, "T": a.T, "windows": n_win, "sessions": len(sessions), "messages": n_kept,
           "note": "Kyle's own typed turns from ~/.claude/projects transcripts, one document per session, secrets scrubbed by pattern; held-out only (R112)"},
          open(os.path.join(a.out, "layout.json"), "w"), indent=1)
with open(os.path.join(a.out, "windows.txt"), "w") as f: f.write("".join(f"{i}\n" for i in range(n_win)))
print(f"lines {n_lines:,}; user entries {n_user:,}; kept {n_kept:,} messages in {len(sessions)} sessions + {len(docs) - len(sessions)} files; "
      f"{total:,} bytes = {n_win} windows of {a.T}")
print("redactions:", {k: v for k, v in redactions.items() if v} or "none")
