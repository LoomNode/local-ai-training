"""The yardstick's scorer inside the runner: fixed windows, cold and warm bits/byte. CPU only, no corpus."""

from __future__ import annotations

import argparse
import math

import pytest
import torch
from torch.nn import functional as F

from flybrain.data.text import fixed_windows
from flybrain.model.hybrid import GRUHybrid


def _tiny():
    torch.manual_seed(0)
    return GRUHybrid(d_model=16, backbone="mingru", where="none", n_layers=1, d_ff=32)


def test_eval_fixed_is_the_per_position_mean_cold_and_its_second_half_warm():
    from scripts.run_sparsela import eval_fixed
    m = _tiny()
    xf, yf = fixed_windows(torch.randint(0, 256, (7 * 16 + 1,)), 16, 7)   # 7 windows in batches of 3: the last batch is short
    out = eval_fixed(m, xf, yf, argparse.Namespace(batch=3, amp=False), torch.device("cpu"))
    assert m.training                                                    # left in train mode, like the runner's other evals
    m.eval()
    with torch.no_grad():                                                 # reference: one window at a time
        ce = torch.stack([F.cross_entropy(m(xf[i:i + 1])[0][0], yf[i], reduction="none") for i in range(7)])
    assert out["bpb"] == pytest.approx(float(ce.mean()) / math.log(2), abs=1e-5)
    assert out["bpb_warm"] == pytest.approx(float(ce[:, 8:].mean()) / math.log(2), abs=1e-5)
    assert out["bpb_warm"] != pytest.approx(out["bpb"], abs=1e-6)         # the two numbers really are different cuts
    again = eval_fixed(m, xf, yf, argparse.Namespace(batch=7, amp=False), torch.device("cpu"))
    assert again["bpb"] == pytest.approx(out["bpb"], abs=1e-6)            # the batch size does not change the score


def test_eval_fixed_uses_no_global_rng():
    from scripts.run_sparsela import eval_fixed
    m = _tiny()
    xf, yf = fixed_windows(torch.randint(0, 256, (4 * 16 + 1,)), 16, 4)
    state = torch.get_rng_state()
    eval_fixed(m, xf, yf, argparse.Namespace(batch=2, amp=False), torch.device("cpu"))
    assert torch.equal(state, torch.get_rng_state())
