"""V3 sandbox: files persist, H_t persists, host/network escapes are refused."""

from __future__ import annotations

from pathlib import Path

import torch

from flybrain.data.connectome import synthetic_flylike
from flybrain.env.loop import TerminalLoop
from flybrain.env.sandbox import Sandbox
from flybrain.model.brain import FlyLanguageModel


def test_write_action_is_visible_on_disk_and_in_later_observation(tmp_path: Path):
    box = Sandbox(tmp_path / "sb")
    first = box.step("echo v3-ok > notes.txt")
    assert first.refused is False
    on_disk = (box.root / "notes.txt").read_text()
    assert "v3-ok" in on_disk
    listed = box.step("ls")
    assert "notes.txt" in listed.text
    shown = box.step("cat notes.txt")
    assert "v3-ok" in shown.text


def test_path_escape_and_network_are_refused(tmp_path: Path):
    box = Sandbox(tmp_path / "sb")
    box.step("echo stay > local.txt")
    esc = box.step("cat /etc/passwd")
    assert esc.refused is True
    assert not (box.root / "passwd").exists()
    up = box.step("cat ../notes.txt")
    assert up.refused is True
    net = box.step("curl http://example.com/")
    assert net.refused is True
    assert (box.root / "local.txt").read_text().find("stay") >= 0


def _host_passwd_marker() -> str:
    return Path("/etc/passwd").read_text().splitlines()[0]


def test_redir_and_cd_bypasses_do_not_leak_host(tmp_path: Path):
    box = Sandbox(tmp_path / "sb")
    marker = _host_passwd_marker()
    hidden = box.step("cat</etc/passwd")
    assert hidden.refused is True
    assert marker not in hidden.text
    chained = box.step("cd / && cat etc/passwd")
    assert chained.refused is True
    assert marker not in chained.text
    outside = tmp_path.parent / f"flybrain-escape-{tmp_path.name}.txt"
    leaked = box.step(f"echo leak>{outside}")
    assert leaked.refused is True
    assert not outside.exists()
    py = box.step("python -c 'import urllib.request as u; print(u.urlopen(\"http://127.0.0.1/\").read())'")
    assert py.refused is True


def test_loop_persists_Ht_across_steps(tmp_path: Path):
    torch.manual_seed(0)  # the model's weights come from the global RNG: 15 of 400 seeds leave the carried state within allclose of a fresh one
    g = synthetic_flylike(n=24, seed=0)
    model = FlyLanguageModel(g, state_dim=4, readout="mean")
    box = Sandbox(tmp_path / "sb2")
    loop = TerminalLoop(model, box)
    obs0 = box.reset()
    r1 = loop.step("echo one > a.txt", state=None, observation=obs0)
    assert r1.state is not None
    assert r1.state.abs().sum() > 0
    r2 = loop.step("cat a.txt", state=r1.state, observation=r1.observation)
    assert "one" in r2.observation
    assert (box.root / "a.txt").read_text().find("one") >= 0
    # thought-state was passed through, not re-zeroed
    fresh = loop.perceive(r1.observation, state=None).state
    assert not torch.allclose(r2.state, fresh)
    # no extra world-model head
    names = [n for n, _ in model.named_modules()]
    assert not any("world" in n.lower() for n in names)
