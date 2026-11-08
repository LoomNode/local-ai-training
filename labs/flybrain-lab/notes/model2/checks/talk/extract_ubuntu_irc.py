"""Build the ubuntu-irc talk corpus from https://irclogs.ubuntu.com/ (Libera.chat Ubuntu channels).

Usage:
  nice -n 19 taskset -c 0-5,8-13 env CUDA_VISIBLE_DEVICES= \
    /games/ailab/lab/flybrain-lab/.venv/bin/python extract_ubuntu_irc.py [--smoke N] [--budget-mb 300]

Source: https://irclogs.ubuntu.com/YYYY/MM/DD/%23<channel>.txt (public logs, Ubuntu IRC Council).
Channels in priority order: ubuntu, ubuntu-server, ubuntu-devel, ubuntu-offtopic; days walked newest
first from 2026-09-30 back to 2004-10-01 until the cleaned text reaches the budget. One request at a
time, >=0.3 s apart. Raw day files are kept in /vault/ailab/corpora-src/talk/ubuntu-irc/raw/<channel>/YYYY/MM/DD.txt
(a zero-size ".404" marker records a missing day), so reruns skip what is downloaded.
Cleaning: keep "[hh:mm] <nick> msg" -> "<nick> msg" and "[hh:mm] * nick ..." -> "* nick ...";
drop everything else (joins/parts/quits/nick changes/modes/topics).
Output: /vault/ailab/corpora-src/talk/ubuntu-irc.bytes (+ .json sidecar).
"""
import sys, os, re, time, json, hashlib, datetime, urllib.request, urllib.error

ROOT = "/vault/ailab/corpora-src/talk"
RAW = ROOT + "/ubuntu-irc/raw"
OUT = ROOT + "/ubuntu-irc.bytes"
SIDE = ROOT + "/ubuntu-irc.json"
CHANNELS = ["ubuntu", "ubuntu-server", "ubuntu-devel", "ubuntu-offtopic"]
UA = "flybrain-lab corpus build (contact: lab owner)"
START = datetime.date(2026, 9, 30)
END = datetime.date(2004, 10, 1)
MAXDOC = 1_000_000
LINE = re.compile(r"^\[\d\d:\d\d(?::\d\d)?\] (<[^>\s]+> .*|\* \S.*)$")

args = sys.argv[1:]
smoke = int(args[args.index("--smoke") + 1]) if "--smoke" in args else 0
budget = int(float(args[args.index("--budget-mb") + 1]) * 1e6) if "--budget-mb" in args else 300_000_000
if smoke:
    OUT = OUT + ".smoke"; SIDE = SIDE + ".smoke"

last_req = 0.0
bad = 0
nreq = 0
def fetch(url):
    """Return bytes, or None on 404. Polite: serial, >=0.3 s apart."""
    global last_req, bad, nreq
    for attempt in range(4):
        w = 0.35 - (time.time() - last_req)
        if w > 0: time.sleep(w)
        last_req = time.time(); nreq += 1
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            if e.code in (429, 403):
                bad += 1
                if bad >= 5:
                    sys.exit("STOP: repeated %d from site" % e.code)
                time.sleep(30 * bad); continue
            time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    sys.exit("STOP: persistent failure on " + url)

def get_day(ch, d):
    p = "%s/%s/%04d/%02d/%02d" % (RAW, ch, d.year, d.month, d.day)
    if os.path.exists(p + ".txt"):
        return open(p + ".txt", "rb").read()
    if os.path.exists(p + ".404"):
        return None
    os.makedirs(os.path.dirname(p), exist_ok=True)
    data = fetch("https://irclogs.ubuntu.com/%04d/%02d/%02d/%%23%s.txt" % (d.year, d.month, d.day, ch))
    if data is None:
        open(p + ".404", "wb").close(); return None
    open(p + ".tmp", "wb").write(data); os.replace(p + ".tmp", p + ".txt")
    return data

def clean(raw):
    out = []
    for ln in raw.decode("utf-8", "replace").replace("\x00", "").split("\n"):
        ln = ln.rstrip("\r ")
        m = LINE.match(ln)
        if m: out.append(m.group(1))
    return out

seen = set()
total = 0; docs = 0
per = {}; skipped404 = 0; dupes = 0; days_fetched = 0
dmin = dmax = None
t0 = time.time()
tmp = OUT + ".tmp"
with open(tmp, "wb") as f:
    for ch in CHANNELS:
        if total >= budget: break
        d = START; n = 0
        while d >= END and total < budget:
            if smoke and n >= smoke: break
            n += 1
            data = get_day(ch, d)
            day = d; d -= datetime.timedelta(days=1)
            if data is None: skipped404 += 1; continue
            days_fetched += 1
            lines = clean(data)
            if not lines: continue
            # split into <=1 MB parts
            parts = []; cur = []; sz = 0
            for ln in lines:
                b = len(ln.encode("utf-8")) + 1
                if sz + b > MAXDOC - 200 and cur:
                    parts.append(cur); cur = []; sz = 0
                cur.append(ln); sz += b
            if cur: parts.append(cur)
            for i, pl in enumerate(parts):
                text = ("\n".join(pl) + "\n").encode("utf-8")
                h = hashlib.sha1(text).digest()
                if h in seen: dupes += 1; continue
                seen.add(h)
                path = "ubuntu-irc/%s/%s%s" % (ch, day.isoformat(), "" if i == 0 else "-p%d" % (i + 1))
                f.write(b"\n// ==== " + path.encode() + b" ====\n" + text)
                total += len(text); docs += 1
                per.setdefault(ch, {"docs": 0, "bytes": 0})
                per[ch]["docs"] += 1; per[ch]["bytes"] += len(text)
                dmin = day if dmin is None or day < dmin else dmin
                dmax = day if dmax is None or day > dmax else dmax
            if n % 200 == 0:
                print(ch, day, "bytes", total, "req", nreq, flush=True)
os.replace(tmp, OUT)
side = {"source": "ubuntu-irc",
        "urls": ["https://irclogs.ubuntu.com/YYYY/MM/DD/%23<channel>.txt"],
        "license": "irclogs.ubuntu.com public logs (https://irclogs.ubuntu.com/), published publicly by the Ubuntu IRC Council; Libera.chat channels",
        "docs": docs, "bytes": total, "per_channel": per,
        "date_range": "%s..%s" % (dmin, dmax), "built": datetime.date.today().isoformat(),
        "notes": "Message and action lines only, timestamps removed; join/part/quit/nick/mode/topic lines dropped. "
                 "Days walked newest first per channel until budget %d bytes. 404 days skipped: %d; exact duplicate docs dropped: %d; "
                 "docs >1MB split with -pN suffix." % (budget, skipped404, dupes)}
json.dump(side, open(SIDE, "w"), indent=1)
print("done", docs, total, per, side["date_range"], "404:", skipped404, "dupes", dupes, "secs", int(time.time() - t0))
