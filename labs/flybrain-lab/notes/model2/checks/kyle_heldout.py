#!/usr/bin/env python
"""Kyle's held-out sets (R112, R120): his own typed turns, and the two-sided dialogues, from every assistant transcript store, in the
corpus format ("\\n// ==== path ====\\n" + text), never trained on. Stores (each may be given more than once, with an optional :host tag):
  --claude DIR[:host]   Claude Code  ~/.claude/projects (JSONL per session: type user/assistant, message.content)
  --grok DIR[:host]     grok CLI     ~/.grok/sessions/<project>/<uuid>/chat_history.jsonl (JSONL, generic role/content lines)
  --codex DB[:host]     codex CLI    ~/.codex/thread_history_1.sqlite (thread_items: userMessage / agentMessage)
  --gemini DIR[:host]   gemini CLI   ~/.gemini/tmp/<project>/chats/*.json (messages: type user/gemini)
Sets: kyle.bytes = his turns only (reading him), dialogue.bytes = the same sessions with both sides in order, "User:" / "Assistant:"
(answering him; the assistant side is whichever model answered, capped at --dialogue-windows, sessions drawn with a seed across stores).
heldout.bytes = both back to back, the dev region of every scoring corpus; heldout.json gives each set's offset and window count;
windows.txt lists Kyle's windows. A user turn over --max-user-bytes (2,000) is a paste or injected context, not typing: dropped. Secrets are scrubbed by pattern
(counts reported, never the text). Nothing but counts is printed.
usage: kyle_heldout.py [--claude ~/.claude/projects] [--grok ~/.grok/sessions] [--codex ~/.codex/thread_history_1.sqlite]
                       [--gemini ~/.gemini/tmp] [--extra ~/TODO.md] [--out /games/ailab/corpora/kyle] [--dialogue-windows 2048]
"""
import argparse, glob, json, os, random, re, sqlite3, sys, time
from collections import defaultdict, Counter

ap = argparse.ArgumentParser()
ap.add_argument("--claude", action="append", default=[]); ap.add_argument("--grok", action="append", default=[])
ap.add_argument("--codex", action="append", default=[]); ap.add_argument("--gemini", action="append", default=[])
ap.add_argument("--extra", nargs="*", default=[os.path.expanduser("~/TODO.md")], help="his own files to add as documents (turns set)")
ap.add_argument("--out", default="/games/ailab/corpora/kyle")
ap.add_argument("--T", type=int, default=2048); ap.add_argument("--dialogue-windows", type=int, default=2048); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-user-bytes", type=int, default=2000, help="a user turn longer than this is a paste, an attached file or injected context, not typing: dropped from both sets")
a = ap.parse_args()
if not (a.claude or a.grok or a.codex or a.gemini): a.claude = [os.path.expanduser("~/.claude/projects")]

SKIP_PREFIX = ("<", "[Request interrupted", "This session is being continued", "Caveat:", "Implement the following", "Base directory for this skill")
TAG_BLOCK = re.compile(r"<(pasted_content|system-reminder|task-notification|command-name|command-message|command-args|local-command-stdout|local-command-caveat)[^>]*>.*?</\1>", re.S)
SECRETS = [
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
USER_ROLES = {"user", "human", "prompter"}; ASSIST_ROLES = {"assistant", "model", "agent", "gemini", "grok", "ai"}

def split_host(spec, default="local"):
    if ":" in spec and not os.path.exists(spec): p, h = spec.rsplit(":", 1); return os.path.expanduser(p), h
    return os.path.expanduser(spec), default

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

# every reader yields (store, host, project, session, ts, role, text)
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
                        yield "claude", host, project, sid, ts, "Assistant", text_of((d.get("message") or {}).get("content"))
                    elif d.get("type") == "user" and not d.get("isMeta") and d.get("userType") in (None, "external"):
                        yield "claude", host, project, sid, ts, "User", text_of((d.get("message") or {}).get("content"))

unknown_keys = Counter()
def generic_line(o):
    role = o.get("role") or o.get("type") or o.get("sender") or o.get("author") or (o.get("message") or {}).get("role") if isinstance(o.get("message"), dict) else o.get("role") or o.get("type") or o.get("sender") or o.get("author")
    content = o.get("content") if "content" in o else o.get("text") if "text" in o else (o.get("message") or {}).get("content") if isinstance(o.get("message"), dict) else o.get("message") or o.get("prompt")
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
                if not isinstance(o, dict): continue
                if o.get("is_bash"): continue
                role, text, ts = generic_line(o)
                if role is None: unknown_keys[tuple(sorted(o.keys()))[:8]] += 1; continue
                yield "grok", host, project, sid, ts or f"{i:08d}", role, text

def read_codex(db, host):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for tid, ordn, ts, typ, js in c.execute("select thread_id, rollout_ordinal, created_at_ms, item_type, item_json from thread_items "
                                             "where item_type in ('userMessage','agentMessage') order by thread_id, rollout_ordinal"):
        try: o = json.loads(js)
        except Exception: continue
        if typ == "userMessage": yield "codex", host, "codex", tid, f"{ts:015d}", "User", text_of(o.get("content"))
        else: yield "codex", host, "codex", tid, f"{ts:015d}", "Assistant", o.get("text") or ""

def read_gemini(root, host):
    for p in glob.glob(os.path.join(root, "*", "chats", "*.json")):
        try: d = json.load(open(p))
        except Exception: continue
        project = os.path.basename(os.path.dirname(os.path.dirname(p))); sid = d.get("sessionId") or os.path.basename(p)[:-5]
        for i, m in enumerate(d.get("messages") or []):
            role = norm_role(m.get("type") or m.get("role"))
            if role: yield "gemini", host, project, sid, m.get("timestamp") or f"{i:08d}", role, text_of(m.get("content"))

readers = [(read_claude, s) for s in a.claude] + [(read_grok, s) for s in a.grok] + [(read_codex, s) for s in a.codex] + [(read_gemini, s) for s in a.gemini]
turns = defaultdict(list); dialogue = defaultdict(list)   # key (store, host, project, session) -> [(ts, [role,] text)]
seen = set(); stats = Counter()
for fn, spec in readers:
    path, host = split_host(spec)
    if not os.path.exists(path): print(f"missing: {spec}"); continue
    for store, h, project, sid, ts, role, text in fn(path, host):
        stats[(store, h, role, "seen")] += 1
        t = (TAG_BLOCK.sub("", text) if role == "User" else text).strip()
        if not t or (role == "User" and t.startswith(SKIP_PREFIX)): continue
        if role == "User" and len(t.encode("utf-8", "replace")) > a.max_user_bytes: stats[(store, h, role, "too_long")] += 1; continue
        if (role, t) in seen: continue
        seen.add((role, t)); stats[(store, h, role, "kept")] += 1
        key = (store, h, project, sid)
        dialogue[key].append((ts, role, t))
        if role == "User": turns[key].append((ts, t))

redactions = Counter()
def scrub(t):
    for name, pat in SECRETS:
        t, n = pat.subn("[redacted]", t); redactions[name] += n
    return t
def doc(path, text):
    text = scrub(text)
    if not text.endswith("\n"): text += "\n"
    return b"\n// ==== " + path.encode() + b" ====\n" + text.encode("utf-8", "replace")

os.makedirs(a.out, exist_ok=True)
tdocs = []
for (store, h, project, sid), msgs in turns.items():
    msgs.sort(); tdocs.append((f"kyle/{h}/{store}/{project}/{msgs[0][0][:10]}-{sid[:8]}", "\n\n".join(m for _, m in msgs)))
tdocs.sort()
for p in a.extra:
    if os.path.isfile(p): tdocs.append((f"kyle/local/files/{os.path.basename(p)}", open(p, encoding="utf-8", errors="replace").read()))
out = os.path.join(a.out, "kyle.bytes"); total = 0
with open(out, "wb") as f:
    for path, t in tdocs: f.write(doc(path, t))
    total = f.tell()
os.chmod(out, 0o600); n_win = total // a.T

ddocs = []
for (store, h, project, sid), items in dialogue.items():
    if (store, h, project, sid) not in turns: continue
    items.sort(); ddocs.append((f"kyle/dialogue/{h}/{store}/{project}/{items[0][0][:10]}-{sid[:8]}", "\n\n".join(f"{r}: {t}" for _, r, t in items)))
random.Random(a.seed).shuffle(ddocs)
dout = os.path.join(a.out, "dialogue.bytes"); cap = a.dialogue_windows * a.T; dtotal = 0; n_dd = 0; per_store = Counter()
with open(dout, "wb") as f:
    for path, t in ddocs:
        b = doc(path, t)
        if dtotal + len(b) > cap and n_dd: continue
        f.write(b); dtotal += len(b); n_dd += 1; per_store[path.split("/")[3]] += len(b)
os.chmod(dout, 0o600); n_dwin = dtotal // a.T
hout = os.path.join(a.out, "heldout.bytes")
with open(hout, "wb") as f: f.write(open(out, "rb").read()); f.write(open(dout, "rb").read())
os.chmod(hout, 0o600)
with open(os.path.join(a.out, "windows.txt"), "w") as f: f.write("".join(f"{i}\n" for i in range(n_win)))
by_store = {f"{s}@{h}": {"user_seen": stats[(s, h, "User", "seen")], "user_kept": stats[(s, h, "User", "kept")],
                        "user_too_long": stats[(s, h, "User", "too_long")], "assistant_kept": stats[(s, h, "Assistant", "kept")]} for (s, h, _, _) in stats}
json.dump({"corpus": hout, "built": time.strftime("%F"), "builder": "kyle_heldout.py", "T": a.T, "bytes": total + dtotal, "stores": by_store,
           "kyle": {"offset": 0, "bytes": total, "windows": n_win, "sessions": len(tdocs), "note": "Kyle's own turns, one document per session, + his files"},
           "dialogue": {"offset": total, "bytes": dtotal, "windows": n_dwin, "sessions": n_dd, "of_sessions": len(ddocs), "bytes_per_store": dict(per_store),
                        "note": "both sides of the same sessions in order, 'User:' / 'Assistant:'; the assistant side is whichever model answered"}},
          open(os.path.join(a.out, "heldout.json"), "w"), indent=1)
json.dump({"corpus": out, "built": time.strftime("%F"), "builder": "kyle_heldout.py", "train_end": 0, "dev_offset": 0, "dev_bytes": total,
           "test_offset": total, "test_bytes": 0, "T": a.T, "windows": n_win, "sessions": len(tdocs)}, open(os.path.join(a.out, "layout.json"), "w"), indent=1)
print("stores:", json.dumps(by_store))
print(f"Kyle's turns: {len(tdocs)} documents, {total:,} bytes = {n_win} windows of {a.T}")
print(f"dialogue: {n_dd} of {len(ddocs)} sessions, {dtotal:,} bytes = {n_dwin} windows (per store: {dict(per_store)}); heldout.bytes {total + dtotal:,} bytes")
print("redactions:", {k: v for k, v in redactions.items() if v} or "none")
if unknown_keys: print("grok lines with no recognised role (keys only):", unknown_keys.most_common(3))
