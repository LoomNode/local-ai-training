"""--rope-table: the transformer's rotary cos and sin kept as buffers. The same numbers as computing them in every call, nothing new
in the saved weights, and the old path whenever the table does not apply."""
import torch

from flybrain.model.hybrid import GRUHybrid, RopeAttention


def _tf(seed: int = 0) -> GRUHybrid:
    torch.manual_seed(seed)
    return GRUHybrid(d_model=64, key="sparse", where="none", backbone="transformer", n_layers=2, d_ff=128, max_len=64, rope=True)


def _layers(m):
    return [a for a in m.modules() if isinstance(a, RopeAttention)]


def test_table_gives_the_same_logits_and_gradients():
    a, b = _tf(), _tf()
    for l in _layers(b): l.use_rope_table(64)
    x = torch.randint(0, 256, (3, 48))
    la, lb = a(x)[0], b(x)[0]
    assert torch.equal(la, lb)
    la.sum().backward(); lb.sum().backward()
    assert all(torch.equal(p.grad, q.grad) for p, q in zip(a.parameters(), b.parameters()))


def test_table_is_not_saved_and_old_weights_load():
    a, b = _tf(), _tf(1)
    keys = set(a.state_dict())
    for l in _layers(b): l.use_rope_table(64)
    assert set(b.state_dict()) == keys and all(l.rope_cos.shape == (64, l.dh // 2) for l in _layers(b))
    b.load_state_dict(a.state_dict())                      # strict: a table adds no expected key
    x = torch.randint(0, 256, (2, 64))
    assert torch.equal(a(x)[0], b(x)[0])


def test_longer_windows_and_a_scaled_base_take_the_old_path():
    a, b = _tf(), _tf()
    for l in _layers(b): l.use_rope_table(32)
    x = torch.randint(0, 256, (2, 48))                     # longer than the table
    assert torch.equal(a(x)[0], b(x)[0])
    for m in (a, b): m.core.set_rope_scale(2.0)            # the NTK extension changes the base: the table no longer applies
    y = torch.randint(0, 256, (2, 24))
    assert torch.equal(a(y)[0], b(y)[0])
    for l in _layers(b): l.use_rope_table(0)
    assert all(l.rope_cos is None and l.rope_sin is None for l in _layers(b))


def test_runner_flag_is_recorded_only_when_on_and_reaches_the_layers():
    from scripts.run_sparsela import make_model, parse_args, recorded_args
    base = ["--arms", "transformer", "--rope", "--d-model", "64", "--d-ff", "128", "--n-layers", "2", "--seq-len", "32", "--device", "cpu"]
    off, on = parse_args(base), parse_args([*base, "--rope-table"])
    assert "rope_table" not in recorded_args(off) and recorded_args(on)["rope_table"] is True
    m = make_model("transformer", on, torch.device("cpu"))
    assert all(l.rope_cos is not None and l.rope_cos.size(0) == 1024 for l in _layers(m)) and len(_layers(m)) == 2
    assert all(l.rope_cos is None for l in _layers(make_model("transformer", off, torch.device("cpu"))))
    assert not _layers(make_model("mingru", parse_args(["--arms", "mingru", "--d-model", "64", "--d-ff", "128", "--n-layers", "2", "--seq-len", "32", "--device", "cpu", "--rope-table"]), torch.device("cpu")))
