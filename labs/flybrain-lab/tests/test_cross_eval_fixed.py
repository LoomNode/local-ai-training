"""cross_eval.py --fixed: the yardstick's scorer for saved checkpoints. CPU only; the checkpoint and corpus are written here."""

from __future__ import annotations

import json
import math
import sys

import pytest
import torch
from torch.nn import functional as F

from flybrain.data.text import fixed_windows, load_bytes
from flybrain.model.hybrid import GRUHybrid

CK_ARGS = dict(d_model=16, gru_hidden=16, n_layers=1, d_ff=32, learned_key_dim=16, key_conv=2, seed=1, seq_len=64)
LEGACY_ROW_KEYS = ["T", "bpb", "bpb_first_eighth", "bpb_last_eighth", "bpb_copyable", "bpb_noncopyable", "ckpt", "arm"]


def _files(tmp_path):
    torch.manual_seed(1)
    m = GRUHybrid(d_model=16, hidden=16, backbone="mingru", where="none", n_layers=1, d_ff=32, max_len=1024)
    ck, corpus = tmp_path / "tiny_mingru.pt", tmp_path / "c.bytes"
    torch.save({"arm": "mingru", "args": CK_ARGS, "state": m.state_dict()}, ck)
    corpus.write_bytes((b"int main(void) { return 0; }\n" * 400)[:6000])
    return ck, corpus


def _main(monkeypatch, tmp_path, *flags):
    import scripts.cross_eval as ce
    ck, corpus = _files(tmp_path)
    monkeypatch.setattr(ce, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["cross_eval.py", "--ckpts", str(ck), "--corpus-file", str(corpus), "--device", "cpu", "--run-name", "out", *flags])
    ce.main()
    return json.loads((tmp_path / "out.json").read_text())


def test_fixed_scores_the_windows_at_the_offset_cold_and_warm(monkeypatch, tmp_path):
    from scripts.cross_eval import rebuild
    out = _main(monkeypatch, tmp_path, "--offset", "100", "--fixed", "5", "--lens", "16,32")
    assert [r["T"] for r in out["rows"]] == [16, 32] and all(r["n_windows"] == 5 for r in out["rows"])
    m = rebuild(torch.load(tmp_path / "tiny_mingru.pt", weights_only=False)).eval()
    data = load_bytes(tmp_path / "c.bytes", max_bytes=5 * 32 + 1, offset=100)
    for r in out["rows"]:
        x, y = fixed_windows(data, r["T"], 5)                             # independent reference over the same bytes
        with torch.no_grad():
            ce = F.cross_entropy(m(x)[0].reshape(-1, 256), y.reshape(-1), reduction="none").view(5, r["T"])
        assert r["bpb"] == pytest.approx(float(ce.mean()) / math.log(2), abs=1e-5)
        assert r["bpb_warm"] == pytest.approx(float(ce[:, r["T"] // 2:].mean()) / math.log(2), abs=1e-5)
    assert out["args"]["fixed"] == 5 and set(out["rows"][0]) == set(LEGACY_ROW_KEYS) | {"bpb_warm", "n_windows"}


def test_fixed_is_deterministic_and_ignores_the_random_window_flags(monkeypatch, tmp_path):
    a = _main(monkeypatch, tmp_path, "--offset", "100", "--fixed", "5", "--lens", "32")["rows"]
    state = torch.get_rng_state()
    b = _main(monkeypatch, tmp_path, "--offset", "100", "--fixed", "5", "--lens", "32", "--n-batches", "9", "--bytes", "999")["rows"]
    assert a == b and torch.equal(state, torch.get_rng_state())


def test_without_fixed_the_rows_and_the_recorded_args_are_the_legacy_ones(monkeypatch, tmp_path, capsys):
    out = _main(monkeypatch, tmp_path, "--offset", "100", "--lens", "32", "--n-batches", "2")
    assert list(out["rows"][0]) == LEGACY_ROW_KEYS and "fixed" not in out["args"]
    assert " warm=" not in capsys.readouterr().out


def test_fixed_prints_the_warm_number(monkeypatch, tmp_path, capsys):
    _main(monkeypatch, tmp_path, "--offset", "100", "--fixed", "3", "--lens", "32")
    assert " warm=" in capsys.readouterr().out


def test_a_region_too_short_for_the_windows_fails_before_any_checkpoint_loads(monkeypatch, tmp_path):
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("a checkpoint was loaded"))
    with pytest.raises(ValueError, match="need 321 bytes for 10 windows of 32"):
        _main(monkeypatch, tmp_path, "--offset", "5900", "--fixed", "10", "--lens", "16,32")
