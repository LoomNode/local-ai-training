"""Arm B (2026-11-06): a date for every held-out session document, so that a time split can put his earlier sessions into a table or a
student without touching the windows it is scored on. kyle.bytes and dialogue.bytes hold one document per session behind a
"// ==== kyle/... ====" header (kyle_heldout.py). The date is asof_dates.json's date of any window lying wholly inside the document (the
dating the hindsight check used), else the header: claude and gemini carry an ISO timestamp, codex the first ten digits of a
zero-padded millisecond epoch (local time), grok a line index (undated, so the session file's mtime under ~/.grok/sessions). Writes
session_dates.json next to the corpora (0600): {"kyle": [[start, end, date], ...], "dialogue": [...]}; prints counts only."""
import datetime, glob, json, os, re, sys
import numpy as np

D = "/games/ailab/corpora/kyle"; T = 2048; GROK = os.path.expanduser("~/.grok/sessions")
asof = json.load(open(f"{D}/asof_dates.json"))
grok_mtime = {}
for p in glob.glob(os.path.join(GROK, "*", "*", "chat_history.jsonl")):
    sid = os.path.basename(os.path.dirname(p)); grok_mtime[sid[:8]] = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d")

def header_date(path: str) -> str:
    parts = path.split("/"); store = parts[3] if parts[1] == "dialogue" else parts[2]; last = parts[-1]
    if "-" not in last: return ""
    ds, sid8 = last.rsplit("-", 1)
    if re.match(r"^20\d\d-\d\d-\d\d$", ds): return ds
    if store == "codex" and ds.isdigit() and len(ds) == 10: return datetime.datetime.fromtimestamp(int(ds) * 100).strftime("%Y-%m-%d")
    if store == "grok": return grok_mtime.get(sid8, "")
    return ""

out = {}; os.umask(0o077)
for fn, key, name in (("kyle.bytes", "kyle_windows", "kyle"), ("dialogue.bytes", "dialogue_windows", "dialogue")):
    b = open(f"{D}/{fn}", "rb").read(); hs = [(m.start(), m.group(1).decode("utf-8", "replace")) for m in re.finditer(rb"\n// ==== (.*?) ====\n", b)]
    ends = [h[0] for h in hs[1:]] + [len(b)]; wd = asof[key]; docs = []; src = {"window": 0, "header": 0, "both_agree": 0, "both_differ": 0, "undated": 0}; diffs = []
    for (s, p), e in zip(hs, ends):
        full = sorted({wd[w] for w in range(len(wd)) if w * T >= s and (w + 1) * T <= e and wd[w]})
        hd = header_date(p); d = ""
        if full and hd:
            d = full[-1]; src["both_agree" if hd in full else "both_differ"] += 1
            if hd not in full: diffs.append(abs((np.datetime64(hd) - np.datetime64(full[-1])).astype(int)))
        elif full: d = full[-1]; src["window"] += 1
        elif hd: d = hd; src["header"] += 1
        else: src["undated"] += 1
        docs.append([s, e, d])
    out[name] = docs
    ds = sorted(x[2] for x in docs if x[2])
    print(f"{name}: {len(docs)} sessions, dated {len(ds)} ({ds[0]}..{ds[-1]}), sources {src}, header-vs-window day gaps: "
          f"{'none' if not diffs else f'n={len(diffs)} max={max(diffs)} median={int(np.median(diffs))}'}", flush=True)
json.dump(out, open(f"{D}/session_dates.json", "w"))
print("wrote", f"{D}/session_dates.json")
