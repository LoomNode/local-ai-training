#!/usr/bin/env python
"""The right book (test 1 of the weights' pile, 2026-10-18): Kyle's world as a project directory for index_project.py, never his held-out
text. Two kinds of file: (a) the assistant side of every transcript session, one file per session (kyle_heldout.py's readers, copied:
the same stores, local and the /vault copies of the other hosts'); his own turns are ALL in the held-out turns set, so they are never
written; (b) his own files: Markdown, text and shell files under the roots given (notes, READMEs, docs, DONE.md, his scripts), corpora
and results excluded. Every file is then checked against the held-out sets: a file holding any 64-byte span of heldout.bytes is dropped
(the dialogue sessions' assistant turns are verbatim there, so they go; so does ~/TODO.md, which is in the turns set). Prints counts
only, never text. Output mode 0600/0700. Reads Kyle's stores, so Kyle runs it (~/world.sh).
usage: kyle_world.py --out /games/ailab/corpora/kyle/world [--roots /games/ailab/lab /games/ailab/DONE.md ~] [--heldout .../heldout.bytes]"""
import argparse, glob, json, os, sqlite3, time
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/games/ailab/corpora/kyle/world")
ap.add_argument("--heldout", default="/games/ailab/corpora/kyle/heldout.bytes")
ap.add_argument("--roots", nargs="*", default=["/games/ailab/lab", "/games/ailab/DONE.md", os.path.expanduser("~")])
ap.add_argument("--remote", default="/vault/ailab/corpora-src/kyle-remote")
ap.add_argument("--span", type=int, default=64); ap.add_argument("--max-file", type=int, default=2_000_000)
ap.add_argument("--ext", default=".md,.txt,.rst,.sh", help="file extensions to take from the roots")
ap.add_argument("--no-replies", action="store_true", help="his files only: do not read the transcript stores")
a = ap.parse_args()
os.umask(0o077)
USER_ROLES = {"user", "human"}; ASSIST_ROLES = {"assistant", "gemini", "model", "ai", "agent", "grok"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "corpora", "results", "runs", "data", "build", "dist", ".cache", ".worktrees",
             "site-packages", ".superpowers", "ckpt", ".claude", ".codex", ".grok", ".gemini", ".local", ".config", "snap", ".npm", ".cargo", ".rustup", "go"}
EXT = tuple(e if e.startswith(".") else "." + e for e in a.ext.split(","))


def text_of(content):
    if isinstance(content, str): return content
    if isinstance(content, dict): return text_of(content.get("text") or content.get("content") or "")
    parts = []
    for b in content or []:
        if isinstance(b, str): parts.append(b)
        elif isinstance(b, dict) and (b.get("type") in (None, "text", "input_text", "output_text")) and isinstance(b.get("text"), str): parts.append(b["text"])
    return "\n".join(parts)


def norm_role(r):
    r = (r or "").lower()
    return "User" if r in USER_ROLES else "Assistant" if r in ASSIST_ROLES else None


def read_claude(root, host):
    for dirpath, _, files in os.walk(root):
        project = os.path.basename(dirpath).strip("-")
        for fn in files:
            if not fn.endswith(".jsonl"): continue
            with open(os.path.join(dirpath, fn), "rb") as f:
                for line in f:
                    try: d = json.loads(line)
                    except Exception: continue
                    if d.get("isSidechain"): continue
                    sid = d.get("sessionId") or fn[:-6]; ts = d.get("timestamp") or ""
                    if d.get("type") == "assistant":
                        yield "claude", host, project, sid, ts, text_of((d.get("message") or {}).get("content"))


def generic_line(o):
    m = o.get("message") if isinstance(o.get("message"), dict) else {}
    role = o.get("role") or o.get("type") or o.get("sender") or o.get("author") or m.get("role")
    content = o.get("content") if "content" in o else o.get("text") if "text" in o else m.get("content") if m else o.get("message") or o.get("prompt")
    ts = o.get("timestamp") or o.get("ts") or o.get("created_at") or o.get("time") or ""
    return norm_role(role if isinstance(role, str) else None), text_of(content), str(ts)


def read_grok(root, host):
    for p in glob.glob(os.path.join(root, "*", "*", "chat_history.jsonl")):
        project = os.path.basename(os.path.dirname(os.path.dirname(p))).replace("%2F", "/").replace("%20", " ").strip("/").replace("/", "-")
        sid = os.path.basename(os.path.dirname(p))
        with open(p, "rb") as f:
            for i, line in enumerate(f):
                try: o = json.loads(line)
                except Exception: continue
                if not isinstance(o, dict) or o.get("is_bash"): continue
                role, text, ts = generic_line(o)
                if role == "Assistant": yield "grok", host, project, sid, ts or f"{i:08d}", text


def read_codex(db, host):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for tid, ordn, ts, typ, js in c.execute("select thread_id, rollout_ordinal, created_at_ms, item_type, item_json from thread_items "
                                             "where item_type = 'agentMessage' order by thread_id, rollout_ordinal"):
        try: o = json.loads(js)
        except Exception: continue
        yield "codex", host, "codex", tid, f"{ts:015d}", o.get("text") or ""


def read_gemini(root, host):
    for p in glob.glob(os.path.join(root, "*", "chats", "*.json")):
        try: d = json.load(open(p))
        except Exception: continue
        project = os.path.basename(os.path.dirname(os.path.dirname(p))); sid = d.get("sessionId") or os.path.basename(p)[:-5]
        for i, m in enumerate(d.get("messages") or []):
            if norm_role(m.get("type") or m.get("role")) == "Assistant": yield "gemini", host, project, sid, m.get("timestamp") or f"{i:08d}", text_of(m.get("content"))


H = os.path.expanduser; R = a.remote
stores = [(read_claude, H("~/.claude/projects"), "goblin"), (read_claude, f"{R}/wyvern/claude-projects", "wyvern"), (read_claude, f"{R}/fairy/claude-projects", "fairy"),
          (read_claude, f"{R}/genie/claude-projects", "genie"), (read_grok, H("~/.grok/sessions"), "goblin"), (read_grok, f"{R}/fairy/grok-sessions", "fairy"),
          (read_codex, H("~/.codex/thread_history_1.sqlite"), "goblin"), (read_codex, f"{R}/wyvern/codex/thread_history_1.sqlite", "wyvern"), (read_gemini, H("~/.gemini/tmp"), "goblin")]

held = open(a.heldout, "rb").read(); S = a.span
spans = {hash(held[i:i + S]) for i in range(0, len(held) - S + 1)}          # every 64-byte window of the held-out sets


def leaks(b: bytes) -> bool:
    return any(hash(b[i:i + S]) in spans for i in range(0, len(b) - S + 1))


os.makedirs(a.out, exist_ok=True); os.chmod(a.out, 0o700)
stats = Counter(); kept_bytes = Counter()


def put(rel: str, b: bytes, kind: str):
    stats[(kind, "seen")] += 1
    if not b.strip() or len(b) > a.max_file: stats[(kind, "empty or too long")] += 1; return
    if leaks(b): stats[(kind, "dropped: holds held-out text")] += 1; return
    p = os.path.join(a.out, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(b); os.chmod(p, 0o600); stats[(kind, "kept")] += 1; kept_bytes[kind] += len(b)


t0 = time.time(); seen = set()
for fn, path, host in ([] if a.no_replies else stores):
    if not os.path.exists(path): stats[("store", "missing")] += 1; continue
    sessions = {}
    for store, h, project, sid, ts, text in fn(path, host):
        t = text.strip()
        if not t or (store, t) in seen: continue
        seen.add((store, t)); sessions.setdefault((store, h, project, sid), []).append((ts, t))
    for (store, h, project, sid), msgs in sessions.items():
        msgs.sort()
        put(f"replies/{h}/{store}/{project}/{msgs[0][0][:10]}-{sid[:8]}.md", "\n\n".join(m for _, m in msgs).encode("utf-8", "replace") + b"\n", f"replies@{h}")
for root in a.roots:
    if os.path.isfile(root): put(f"files/{os.path.basename(root)}", open(root, "rb").read(), "files"); continue
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for f in files:
            if f.endswith(EXT):
                p = os.path.join(dirpath, f)
                try: b = open(p, "rb").read()
                except Exception: continue
                put("files/" + os.path.relpath(p, "/").replace("..", "_"), b, "files")
print(f"held-out spans: {len(spans):,} of {S} bytes; {time.time() - t0:.0f} s")
for k in sorted(stats): print(f"  {k[0]:16s} {k[1]:28s} {stats[k]:6d}")
print("kept bytes:", {k: f"{v:,}" for k, v in kept_bytes.items()}, "total", f"{sum(kept_bytes.values()):,}")
