import torch
from torch.nn import functional as F

from flybrain.model.sparsela import KEY_KINDS, SparseKeyLM


def test_all_key_kinds_forward_and_are_causal():
    for kind in KEY_KINDS:
        m = SparseKeyLM(key=kind, d_model=16, n_key=64, top_k=8, n_layers=1, d_ff=32)
        x = torch.randint(0, 256, (2, 12))
        logits, _ = m(x)
        assert logits.shape == (2, 12, 256)
        x2 = x.clone()
        x2[:, 6:] = (x2[:, 6:] + 1) % 256
        logits2, _ = m(x2)
        assert torch.allclose(logits[:, :6], logits2[:, :6], atol=1e-5), kind


def test_sparse_key_is_topk_sparse():
    m = SparseKeyLM(key="sparse", d_model=16, n_key=64, top_k=8, n_layers=1)
    x = torch.randn(2, 5, 16)
    q, k = m.blocks[0].code(x)
    assert torch.all((k != 0).sum(-1) <= 8)


def test_train_step_reduces_loss():
    torch.manual_seed(0)
    m = SparseKeyLM(key="sparse", d_model=16, n_key=64, top_k=8, n_layers=1, d_ff=32)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    x = torch.randint(0, 8, (8, 16))
    y = x.roll(-1, 1)
    first = None
    for _ in range(30):
        logits, _ = m(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        first = first or float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < first


def test_key_conv_is_causal():
    m = SparseKeyLM(key="sparse", d_model=16, n_key=64, top_k=8, n_layers=1, d_ff=32, key_conv=3)
    x = torch.randint(0, 256, (2, 12))
    logits, _ = m(x)
    x2 = x.clone(); x2[:, 6:] = (x2[:, 6:] + 1) % 256
    logits2, _ = m(x2)
    assert torch.allclose(logits[:, :6], logits2[:, :6], atol=1e-5)


def test_mixed_and_tied_kinds_forward_causal():
    for kind in ("mixed", "tied"):
        m = SparseKeyLM(key=kind, d_model=16, n_key=64, top_k=4, n_layers=1, d_ff=32, learned_key=16, normalize=True, key_conv=2)
        x = torch.randint(0, 256, (2, 12))
        logits, _ = m(x)
        assert logits.shape == (2, 12, 256)
        x2 = x.clone(); x2[:, 6:] = (x2[:, 6:] + 1) % 256
        logits2, _ = m(x2)
        assert torch.allclose(logits[:, :6], logits2[:, :6], atol=1e-5), kind
    m = SparseKeyLM(key="mixed", d_model=16, n_key=64, top_k=4, n_layers=1, learned_key=16)
    assert len(m.blocks[0].codes) == 2 and m.blocks[0].codes[1].kind == "sparse"


def test_copyable_mask_and_terminal_corpus():
    from flybrain.data.corpus import copyable_mask, terminal_transcript
    from flybrain.data.task import multi_copy_batch

    x, y, m = multi_copy_batch(3, seq_len=128, span=8, n_spans=4, seed=0)
    assert torch.all(m.sum(1) == 4 * 7)
    cm = copyable_mask(x, y, n=4)
    # every recalled byte after the first 4 of a span is copyable by a 4-gram key
    assert (cm & m).sum() >= 4 * 4 * 3
    a = terminal_transcript(4000, seed=1); b = terminal_transcript(4000, seed=1)
    assert a == b and len(a) == 4000 and b"user@box" in a and b"cat " in a
