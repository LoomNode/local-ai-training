"""Model #2 on tokens (notes/model2): the reader with a large vocabulary, the shared-embedding table channel, the value channel for
the counters, the token loader and the feature store on a uint16 directory. The byte path is untouched: a model built without the
switches is the model as it was."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flybrain.data.memfeat import FeatureStore  # noqa: E402
from flybrain.data.text import load_tokens, ngram_bpb  # noqa: E402
from flybrain.model.hybrid import GRUHybrid  # noqa: E402
from flybrain.model.inputs import N_CONF_CLASSES, FeatureChannel, SharedFeatureChannel, channel_kwargs  # noqa: E402
from scripts.run_sparsela import parse_args, train_loss  # noqa: E402

V = 1000


def _model(**kw):
    torch.manual_seed(0)
    return GRUHybrid(d_model=32, hidden=32, key="sparse", where="none", backbone="transformer", n_layers=1, d_ff=64, rope=True, max_len=64, **kw)


def test_shared_channel_matches_feature_channel_with_gate_one():
    """With gate 1 and the same class table and bonus, the shared channel equals a FeatureChannel whose byte tables are the shared embedding."""
    torch.manual_seed(1)
    d, d_out, K = 16, 24, 3
    emb = torch.nn.Embedding(V, d)
    sh = SharedFeatureChannel(d, d_out, K, N_CONF_CLASSES, emb)
    ch = FeatureChannel(d, d_out, K, N_CONF_CLASSES, vocab=V)
    with torch.no_grad():
        sh.gate.fill_(1.0); sh.class_emb.weight.normal_(); sh.bonus.weight.normal_(); sh.bonus.bias.normal_()
        for o in range(K):
            ch.byte_emb[o].weight.copy_(emb.weight)
            ch.class_emb[o].weight.copy_(sh.class_emb.weight[o * N_CONF_CLASSES:(o + 1) * N_CONF_CLASSES])
            ch.bonus[o].weight.copy_(sh.bonus.weight[o * (N_CONF_CLASSES - 1):(o + 1) * (N_CONF_CLASSES - 1)])
            ch.bonus[o].bias.copy_(sh.bonus.bias[o * (N_CONF_CLASSES - 1):(o + 1) * (N_CONF_CLASSES - 1)])
    g = torch.Generator().manual_seed(2)
    pred = torch.randint(0, V, (2, 9, K), generator=g); cls = torch.randint(0, N_CONF_CLASSES, (2, 9, K), generator=g)
    assert torch.allclose(sh.embed(pred, cls), ch.embed(pred, cls), atol=1e-5)
    h = torch.randn(2, 9, d_out); logits = torch.randn(2, 9, V)
    assert torch.allclose(sh.add_bonus(logits, h, pred, cls), ch.add_bonus(logits, h, pred, cls), atol=1e-5)
    assert sum(p.numel() for p in sh.parameters()) == K + K * N_CONF_CLASSES * d + (d_out + 1) * K * (N_CONF_CLASSES - 1)   # no vocab tables


def test_token_model_starts_plain_and_values_add_no_bonus():
    orders = (1, 2, 1001, 1002)
    m = _model(vocab_size=V, mem_orders=orders, mem_value_orders=(1001, 1002), shared_table=True, match_orders=(2, 4), fused_channels=True)
    plain = _model(vocab_size=V)
    g = torch.Generator().manual_seed(3)
    x = torch.randint(0, V, (2, 16), generator=g)
    pred = torch.cat([torch.randint(0, V, (2, 16, 2), generator=g), torch.randint(0, 128, (2, 16, 2), generator=g)], -1).to(torch.int32)
    conf = torch.cat([torch.randint(0, 20, (2, 16, 2), generator=g), torch.ones(2, 16, 2, dtype=torch.long)], -1).to(torch.uint8)
    m.eval(); plain.eval()
    with torch.no_grad():
        lo, _ = m(x, mem_pred=pred, mem_conf=conf)
        lp, _ = plain(x)
    assert lo.shape == (2, 16, V) and torch.allclose(lo, lp, atol=1e-5)          # zero-initialised channels: the plain model
    assert m.table_orders == orders and m.value_orders == (1001, 1002) and m.count_idx.tolist() == [0, 1] and m.value_idx.tolist() == [2, 3]
    assert isinstance(m.match, SharedFeatureChannel) and m.match.vocab == V and m.values.vocab == 128
    plain_match = _model(vocab_size=V, match_orders=(2, 4), fused_channels=True)
    assert plain_match.match.vocab == V and plain_match.match.byte_emb.weight.shape == (2 * V, 32)   # without sharing: the vocabulary wide
    # the value channel reaches the logits only through its input embedding: with that embedding zeroed, values change nothing even when trained
    with torch.no_grad():
        for p in m.values.parameters():
            p.normal_()
        m.values.byte_emb.weight.zero_(); m.values.class_emb.weight.zero_()
        lo2, _ = m(x, mem_pred=pred, mem_conf=conf)
        pred2 = pred.clone(); pred2[..., 2:] = (pred2[..., 2:] + 7) % 128
        lo3, _ = m(x, mem_pred=pred2, mem_conf=conf)
    assert torch.allclose(lo2, lo3, atol=1e-6)                                   # the bonus linear of the value channel is never applied
    # the gate gets a gradient from the shared embedding
    m.train()
    y = torch.randint(0, V, (2, 16), generator=g)
    loss = F.cross_entropy(m(x, mem_pred=pred, mem_conf=conf)[0].reshape(-1, V), y.reshape(-1))
    loss.backward()
    assert m.table.gate.grad is not None and float(m.table.gate.grad.abs().sum()) > 0


def test_train_loss_and_rebuild_kwargs_with_vocab():
    class A:  # the fields train_loss reads
        local_blocks = 0; amp = False; distill_weight = 0.5; distill_temp = 1.0
    m = _model(vocab_size=V)
    g = torch.Generator().manual_seed(4)
    x = torch.randint(0, V, (2, 8), generator=g); y = torch.randint(0, V, (2, 8), generator=g)
    loss, parts = train_loss(m, x, y, A(), torch.device("cpu"))
    assert loss.item() > 0 and len(parts) == 1
    kw = channel_kwargs({"vocab": V, "mem_feat": "d", "mem_orders": "1,2,1001", "mem_value_orders": "1001", "shared_table": True, "fused_channels": True})
    assert kw == {"mem_orders": (1, 2, 1001), "fused_channels": True, "vocab_size": V, "mem_value_orders": (1001,), "shared_table": True}
    assert channel_kwargs({"vocab": 256, "mem_feat": "d"}) == {"mem_orders": (8, 16, 32)}   # a byte checkpoint rebuilds as before
    args = parse_args(["--task", "code", "--corpus-file", "x", "--tokens", "--vocab", str(V), "--arms", "transformer"])
    assert args.tokens and args.vocab == V
    with pytest.raises(SystemExit):
        parse_args(["--task", "code", "--corpus-file", "x", "--tokens", "--arms", "transformer"])       # --tokens without --vocab


def test_load_tokens_and_uint16_feature_store(tmp_path):
    ids = np.array([5, 644, 1, 999, 3, 7], dtype="<u2")
    ids.tofile(tmp_path / "tokens.u16")
    t = load_tokens(tmp_path / "tokens.u16")
    assert t.dtype == torch.uint16 and t.tolist() == ids.tolist()
    assert load_tokens(tmp_path / "tokens.u16", 2, 3).tolist() == [999, 3]
    assert ngram_bpb(t, t[:3], 1, vocab=V) > 0
    with pytest.raises(ValueError):
        ngram_bpb(t, t, 2, vocab=V * 10)
    d = tmp_path / "feat"; d.mkdir()
    n = len(ids)
    np.save(d / "pred_n1.npy", np.arange(n, dtype=np.uint16) * 300); np.save(d / "conf_n1.npy", np.ones(n, dtype=np.uint8))
    np.save(d / "pred_n1001.npy", np.arange(n, dtype=np.uint16)); np.save(d / "conf_n1001.npy", np.ones(n, dtype=np.uint8))
    json.dump({"kind": "features", "n_tokens": n, "train_end_tok": 4, "corpus_bytes": 100, "train_end": 60, "orders": [1, 1001], "value_orders": [1001]},
              open(d / "features.json", "w"))
    st = FeatureStore(d, (1, 1001))
    assert st.unit == "token" and st.size == n and st.train_end == 4 and st.value_orders == (1001,) and st.pred_dtype == np.uint16
    st.check_corpus(tmp_path / "tokens.u16")                                     # 2 bytes per token
    pred, conf = st.windows(torch.tensor([0, 1]), 3)
    assert pred.dtype == torch.uint16 and pred[0, :, 0].tolist() == [300, 600, 900] and pred[1, :, 1].tolist() == [2, 3, 4]
    with pytest.raises(ValueError):
        st.check_corpus(d / "features.json")
