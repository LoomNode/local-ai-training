"""The chunked table builders (notes/speed-round/checks: key_tables.py --keys-only, counter_tables.py, assoc_all.py) against the original
whole-corpus code from git (abad9ef, the last commit before the chunking), on a slice of the Linux corpus: every key, counter and two-way table
feature must be bit-identical. The originals run with their constants rewritten to the slice; the new code with CHUNK = 1 MB and two
workers, so the slice is cut into several chunks. CPU only (CUDA_VISIBLE_DEVICES="" for every child)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "notes/speed-round/checks"
LINUX = Path("/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes")
SIZE, TRAIN_END = 6_000_000, 5_000_000
ORIGINAL_REV = os.environ.get("TABLE_BUILDERS_ORIGINAL", "abad9ef")     # the last commit with the whole-corpus builders
KEY_IDS = (116, 132, 202, 203, 301)
COUNTER_IDS = (1001, 1002, 1003, 1004, 1005, 1006, 1007)
ASSOC_ORDERS = (4, 8, 12, 16, 32, 64, *KEY_IDS)

pytestmark = pytest.mark.skipif(not LINUX.exists(), reason="needs the Linux corpus")


def run(args, env=None, cwd=None):
    e = {**os.environ, "CUDA_VISIBLE_DEVICES": "", **(env or {})}
    r = subprocess.run([sys.executable, *map(str, args)], env=e, cwd=cwd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{args} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def original(name: str, dst: Path, corpus: Path, mem: Path) -> Path:
    """The original builder from git, its constants pointed at the slice."""
    src = subprocess.run(["git", "show", f"{ORIGINAL_REV}:notes/speed-round/checks/{name}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for pat, new in ((r'^CORPUS = .*$', f'CORPUS = "{corpus}"'), (r'^MEM = .*$', f'MEM = "{mem}"'),
                     (r'^TRAIN_END, CHUNK = .*$', f'TRAIN_END, CHUNK = {TRAIN_END}, 1 << 20'), (r'^TRAIN_END = .*$', f'TRAIN_END = {TRAIN_END}')):
        src = re.sub(pat, new, src, flags=re.M)
    assert f'CORPUS = "{corpus}"' in src and f'MEM = "{mem}"' in src and str(TRAIN_END) in src
    p = dst / name
    p.write_text(src)
    return p


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tables")
    corpus = tmp / "slice.bytes"
    with open(LINUX, "rb") as f:
        corpus.write_bytes(f.read(SIZE))
    assert corpus.read_bytes().count(b"\n\n// ==== ") > 100
    old, new = tmp / "old", tmp / "new"
    (old / "feat").mkdir(parents=True)
    (old / "feat" / "features.json").write_text(json.dumps({"corpus_bytes": SIZE, "train_end": TRAIN_END}))
    new.mkdir()
    code = tmp / "orig"; code.mkdir()
    run([original("key_tables.py", code, corpus, old), "--keep-keys"])
    run([original("counter_tables.py", code, corpus, old)])
    run([original("assoc_all.py", code, corpus, old), 20, old / "all20"])
    env = {"CORPUS": str(corpus), "TRAIN_END": str(TRAIN_END), "MEM": str(new), "CHUNK": str(1 << 20), "JOBS": "2"}
    logs = {"keys": run([CHECKS / "key_tables.py", "--keys-only"], env), "counters": run([CHECKS / "counter_tables.py"], env)}
    logs["perfile"] = run([CHECKS / "counter_tables.py"], {**env, "CHUNK": "1", "OUT": str(new / "counters_perfile")})
    env.pop("CHUNK")
    logs["assoc"] = run([CHECKS / "assoc_all.py", 20, new / "all20"], env)
    run([CHECKS / "assoc_all.py", 20], {**env, "OUT": str(new / "split"), "ORDERS": "4,16,64,132,301"})
    logs["split"] = run([CHECKS / "assoc_all.py", 20], {**env, "OUT": str(new / "split"), "ORDERS": "8,12,32,116,202,203"})
    logs["dump"] = run([CHECKS / "assoc_all.py", 20], {**env, "OUT": str(new / "tables20"), "TABLES_ONLY": "1", "KEYS": str(new / "feat_keys")})
    return old, new, logs


def same(a: Path, b: Path) -> bool:
    x, y = np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r")
    return x.dtype == y.dtype and np.array_equal(x, y)


def test_keys_match_the_original(built):
    old, new, logs = built
    assert "chunks" in logs["keys"] and not list((new / "feat_keys").glob("*.partial.npy"))
    for kid in KEY_IDS:
        assert same(old / f"feat_keys/key_n{kid}.npy", new / f"feat_keys/key_n{kid}.npy"), kid


def test_counters_match_the_original(built):
    old, new, logs = built
    n_chunks = int(re.search(r"([\d,]+) chunks", logs["counters"]).group(1).replace(",", ""))
    assert n_chunks >= 4
    for kid in COUNTER_IDS:
        for kind in ("pred", "conf"):
            assert same(old / f"feat_counters/{kind}_n{kid}.npy", new / f"feat_counters/{kind}_n{kid}.npy"), (kind, kid)
    info = json.loads((new / "feat_counters/features.json").read_text())
    assert info["corpus_bytes"] == SIZE and info["train_end"] == TRAIN_END and info["orders"] == list(COUNTER_IDS)
    for kid in COUNTER_IDS:                  # every file a chunk of its own: the state machine is cut at every header
        assert same(old / f"feat_counters/pred_n{kid}.npy", new / f"counters_perfile/pred_n{kid}.npy"), kid
    print("\n1 MB chunks:", re.search(r"state carried.*", logs["counters"]).group(0))
    print("one file per chunk:", re.search(r"state carried.*", logs["perfile"]).group(0))


def test_block_comment_state_is_carried_across_chunks(tmp_path):
    """Files ending inside an unterminated block comment (or a string): the original's state machine carries the comment on into the
    next files; the chunked code, one file per chunk, must give the same bits."""
    files = [(f"f{i}.c", body) for i, body in enumerate([
        b"int a(void) { return (1); }\n/* open comment with { brace\n",
        b"int b(void) { if (x) { y(\"s\"); } }\n",
        b"char *s = \"unterminated\nint c(void) { return '{'; }\n/* closes */ int d;\n",
        b"}} /* a */ (( x */ y;\n// line\n{ z; /*/ still */ }\n"] * 3)]
    corpus = tmp_path / "c.bytes"
    corpus.write_bytes(b"".join(b"\n// ==== " + p.encode() + b" ====\n" + body for p, body in files))
    n = corpus.stat().st_size
    old = tmp_path / "old"; (old / "feat").mkdir(parents=True)
    (old / "feat/features.json").write_text("{}")
    code = tmp_path / "orig"; code.mkdir()
    src = original("counter_tables.py", code, corpus, old).read_text().replace(f"TRAIN_END = {TRAIN_END}", f"TRAIN_END = {n // 2}")
    (code / "counter_tables.py").write_text(src.replace("q = TRAIN_END +", "q = 0 * TRAIN_END +"))
    run([code / "counter_tables.py"])
    log = run([CHECKS / "counter_tables.py"], {"CORPUS": str(corpus), "TRAIN_END": str(n // 2), "MEM": str(tmp_path / "new"), "CHUNK": "1", "JOBS": "2"})
    carried = int(re.search(r"([\d,]+) string/comment positions", log).group(1).replace(",", ""))
    assert carried > 0
    for kid in COUNTER_IDS:
        assert same(old / f"feat_counters/pred_n{kid}.npy", tmp_path / f"new/feat_counters/pred_n{kid}.npy"), kid


def test_assoc_matches_the_original(built):
    old, new, _ = built
    for out in ("all20", "split"):
        for n in (*ASSOC_ORDERS, *COUNTER_IDS):
            for kind in ("pred", "conf"):
                assert same(old / f"all20/{kind}_n{n}.npy", new / f"{out}/{kind}_n{n}.npy"), (out, kind, n)
        info = json.loads((new / out / "features.json").read_text())
        assert info["orders"] == [*ASSOC_ORDERS, *COUNTER_IDS] and info["log2_slots"] == 20 and info["slots_per_order"] == 1 << 20
        assert info["corpus_bytes"] == SIZE and info["train_end"] == TRAIN_END and info["heldout_positions"] == SIZE - TRAIN_END
        assert sorted(map(int, info["stats"])) == sorted(ASSOC_ORDERS)


def test_feature_store_reads_the_new_dir(built):
    _, new, _ = built
    sys.path.insert(0, str(ROOT))
    from flybrain.data.memfeat import FeatureStore
    fs = FeatureStore(new / "all20", [4, 301, 1004])
    pred, conf = fs.at([TRAIN_END], 64)
    assert pred.shape == (1, 64, 3)


def test_the_table_dump_reproduces_the_held_out_features(built):
    """TABLES_ONLY=1 saves the final tables; a lookup in them at every position past train_end equals the normal build's features."""
    old, new, logs = built
    meta = json.loads((new / "tables20" / "tables.json").read_text())
    assert meta["log2_slots"] == 20 and meta["train_end"] == TRAIN_END and sorted(meta["orders"]) == sorted(ASSOC_ORDERS)
    assert not list((new / "tables20").glob("pred_n*.npy"))                      # no features written in this mode
    t = np.load(new / "tables20" / "table_n8.npy", mmap_mode="r")
    assert t.shape == (1 << 20,) and t.dtype.names == ("tag", "byte", "conf") and (t["conf"] > 0).mean() > 0.1
    out = run([CHECKS / "check_tables.py", new / "tables20", new / "all20", "--ranges", f"{TRAIN_END}:{SIZE}"])
    assert "tables reproduce the features" in out and out.count("0 differences") == len(ASSOC_ORDERS)
