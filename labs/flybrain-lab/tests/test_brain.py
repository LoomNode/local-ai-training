"""V0 brain: 4-d neurons, scalar synapses, sensory/readout bottlenecks, frozen topology."""

from __future__ import annotations

import torch
from torch import nn

from flybrain.data.connectome import degree_matched_shuffle, synthetic_flylike
from flybrain.model.brain import FlyLanguageModel


def _model(n: int = 40, seed: int = 0, state_dim: int = 4) -> FlyLanguageModel:
    g = synthetic_flylike(n=n, seed=seed)
    return FlyLanguageModel(g, state_dim=state_dim)


def test_state_is_four_dimensional_and_not_a_parameter():
    m = _model(n=32, seed=1)
    tokens = torch.randint(0, 256, (2, 5))
    logits, state = m(tokens)
    assert logits.shape == (2, 5, 256)
    assert state.shape == (2, 32, 4)
    for name, p in m.named_parameters():
        assert "state" not in name
        assert list(p.shape) != [32, 32]


def test_encoder_width_tracks_n_sensory():
    g = synthetic_flylike(n=64, seed=2, n_sensory=20)
    m = FlyLanguageModel(g, state_dim=4, readout="mean")
    assert m.encoder.weight.shape == (256, 20)
    tokens = torch.randint(0, 256, (2, 3))
    logits, state = m(tokens)
    assert logits.shape == (2, 3, 256)
    assert state.shape == (2, 64, 4)


def test_encoder_does_not_broadcast_to_every_neuron():
    m = _model(n=48, seed=2)
    n_s = int(m.connectome.sensory_idx.numel())
    n_r = int(m.connectome.readout_idx.numel())
    assert n_s < 48 and n_r < 48
    embed = m.encoder.weight
    assert embed.shape == (256, n_s)
    # decoder reads the designated population, not all neurons
    assert m.decoder.in_features == n_r * m.state_dim
    assert m.decoder.out_features == 256


def test_one_trainable_scalar_per_synapse_regardless_of_state_dim():
    g = synthetic_flylike(n=36, seed=5)
    m4 = FlyLanguageModel(g, state_dim=4)
    m8 = FlyLanguageModel(g, state_dim=8)
    assert m4.n_synapses == g.n_edges
    assert m8.n_synapses == g.n_edges
    assert m4.synapse_param.numel() == g.n_edges
    assert m8.synapse_param.numel() == g.n_edges


def test_unique_program_grows_with_neuron_count_and_is_vectorized():
    g_small = synthetic_flylike(n=24, seed=1)
    g_large = synthetic_flylike(n=72, seed=1)
    small = FlyLanguageModel(g_small, state_dim=4, readout="mean", cell="unique")
    large = FlyLanguageModel(g_large, state_dim=4, readout="mean", cell="unique")
    n_small = sum(p.numel() for n, p in small.named_parameters() if n.startswith("unique"))
    n_large = sum(p.numel() for n, p in large.named_parameters() if n.startswith("unique"))
    assert n_large > n_small
    assert n_small > 0
    gru_cells = [m for m in large.modules() if isinstance(m, nn.GRUCell)]
    assert len(gru_cells) == 0
    tokens = torch.randint(0, 256, (2, 4))
    logits, state = large(tokens)
    assert logits.shape == (2, 4, 256)
    assert state.shape == (2, 72, 4)
    loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 256), tokens[:, 1:].reshape(-1))
    loss.backward()
    for _, p in large.named_parameters():
        assert list(p.shape) != [72, 72]


def test_unique_fly_and_shuffle_match_scalars():
    g = synthetic_flylike(n=40, seed=8)
    sh = degree_matched_shuffle(g, seed=2)
    fly = FlyLanguageModel(g, state_dim=4, readout="mean", cell="unique")
    rnd = FlyLanguageModel(sh, state_dim=4, readout="mean", cell="unique")
    n_fly = sum(p.numel() for p in fly.parameters() if p.requires_grad)
    n_rnd = sum(p.numel() for p in rnd.parameters() if p.requires_grad)
    assert n_fly == n_rnd


def test_universal_is_one_shared_cell_independent_of_n():
    small = FlyLanguageModel(synthetic_flylike(n=24, seed=1), state_dim=4, readout="mean", cell="universal")
    large = FlyLanguageModel(synthetic_flylike(n=72, seed=1), state_dim=4, readout="mean", cell="universal")
    typed = FlyLanguageModel(synthetic_flylike(n=24, seed=1), state_dim=4, readout="mean", cell="type")
    def program(m: FlyLanguageModel) -> int:
        return sum(
            p.numel()
            for name, p in m.named_parameters()
            if name.startswith("cells") or name.startswith("emit")
        )
    assert program(small) == program(large)
    assert program(small) < program(typed)
    gru_cells = [m for m in large.modules() if isinstance(m, nn.GRUCell)]
    assert len(gru_cells) == 1
    tokens = torch.randint(0, 256, (2, 4))
    logits, state = large(tokens)
    assert logits.shape == (2, 4, 256)
    loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 256), tokens[:, 1:].reshape(-1))
    loss.backward()
    for _, p in large.named_parameters():
        assert list(p.shape) != [72, 72]


def test_universal_fly_and_shuffle_match_scalars():
    g = synthetic_flylike(n=40, seed=8)
    sh = degree_matched_shuffle(g, seed=2)
    fly = FlyLanguageModel(g, state_dim=4, readout="mean", cell="universal")
    rnd = FlyLanguageModel(sh, state_dim=4, readout="mean", cell="universal")
    assert sum(p.numel() for p in fly.parameters() if p.requires_grad) == sum(
        p.numel() for p in rnd.parameters() if p.requires_grad
    )


def test_type_program_size_does_not_grow_with_neuron_count():
    small = _model(n=24, seed=1)
    large = _model(n=72, seed=1)
    def type_params(m: FlyLanguageModel) -> int:
        n = 0
        for name, p in m.named_parameters():
            if name.startswith("cells") or name.startswith("emit"):
                n += p.numel()
        return n
    assert type_params(small) == type_params(large)
    assert type_params(small) > 0


def test_sign_constraint_survives_gradient_steps():
    m = _model(n=32, seed=6)
    signs = m.connectome.sign.clone()
    w0 = m.synapse_weights().detach()
    assert torch.equal(torch.sign(w0), signs)
    tokens = torch.randint(0, 256, (4, 8))
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(8):
        logits, _ = m(tokens)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 256),
            tokens[:, 1:].reshape(-1),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    w1 = m.synapse_weights().detach()
    assert torch.equal(torch.sign(w1), signs)
    assert not torch.allclose(w1, w0)


def test_state_persists_across_tokens_and_resets_without_touching_theta():
    m = _model(n=28, seed=3)
    theta0 = {k: v.detach().clone() for k, v in m.state_dict().items()}
    a = torch.tensor([[ord("A"), 0, 0]])
    z = torch.tensor([[0, 0, 0]])
    _, h_a = m(a)
    _, h_z = m(z)
    assert not torch.allclose(h_a, h_z)
    m.reset_state()
    for k, v in m.state_dict().items():
        assert torch.equal(v, theta0[k])


def test_vector_messages_keep_one_scalar_per_edge_and_widen_the_cell():
    g = synthetic_flylike(n=36, seed=5)
    scalar = FlyLanguageModel(g, state_dim=4, message="scalar")
    vector = FlyLanguageModel(g, state_dim=4, message="vector")
    assert scalar.cells[0].input_size == 1
    assert vector.cells[0].input_size == 4
    assert vector.synapse_param.numel() == g.n_edges
    tokens = torch.randint(0, 256, (2, 4))
    logits, state = vector(tokens)
    assert logits.shape == (2, 4, 256)
    assert state.shape == (2, 36, 4)
    for _, p in vector.named_parameters():
        assert list(p.shape) != [36, 36]


def test_mean_readout_head_is_small_and_independent_of_n_readout():
    g = synthetic_flylike(n=40, seed=4)
    m = FlyLanguageModel(g, state_dim=4, readout="mean")
    assert m.decoder.in_features == 4
    tokens = torch.randint(0, 256, (2, 3))
    logits, state = m(tokens)
    assert logits.shape == (2, 3, 256)
    assert state.shape == (2, 40, 4)


def test_emit_readout_is_small_head_not_flatten64():
    g = synthetic_flylike(n=128, seed=0, n_sensory=16, n_readout=8, n_kenyon=96, claws=3)
    m = FlyLanguageModel(
        g,
        state_dim=4,
        vocab_size=32,
        readout="emit",
        update="ordered",
        inject="local",
        kenyon_top_k=12,
        kenyon_theta=1.5,
        n_out=32,
    )
    n_r = int(g.readout_idx.numel())
    assert m.decoder.in_features == n_r
    assert m.decoder.in_features < 64
    assert m.decoder.out_features == 32
    assert m.encoder.weight.requires_grad is False
    tokens = torch.randint(0, 16, (2, 5))
    logits, state = m(tokens)
    assert logits.shape == (2, 5, 32)
    assert state.shape == (2, 128, 4)
    bag = torch.tensor([[0, 1, 2], [3, 4, 5]])
    odor_logits, odor_state = m.forward_odor(bag)
    assert odor_logits.shape == (2, 1, 32)
    assert odor_state.shape == (2, 128, 4)
    kidx = (m.region == 1).nonzero(as_tuple=False).view(-1)
    live_k = (odor_state[:, kidx].abs().sum(dim=-1) > 1e-6).sum(dim=1)
    assert torch.all(live_k <= 12)
    sens = odor_state[0, m.sensory_idx, 0]
    assert float(sens[0].detach()) != 0.0 and float(sens[15].detach()) == 0.0


def test_unknown_decode_pop_is_rejected():
    g = synthetic_flylike(n=32, seed=0)
    try:
        FlyLanguageModel(g, state_dim=4, readout="mean", decode_pop="core")
    except ValueError as exc:
        assert "decode_pop" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_sensory_decode_keeps_mean_head_and_graph():
    g = synthetic_flylike(n=40, seed=4)
    read = FlyLanguageModel(g, state_dim=4, readout="mean", decode_pop="readout")
    sens = FlyLanguageModel(g, state_dim=4, readout="mean", decode_pop="sensory")
    assert sens.decoder.in_features == 4
    assert sens.encoder.weight.shape == read.encoder.weight.shape
    assert sens.n_synapses == read.n_synapses
    n_s = int(g.sensory_idx.numel())
    flat = FlyLanguageModel(g, state_dim=4, readout="flatten", decode_pop="sensory")
    assert flat.decoder.in_features == n_s * 4
    tokens = torch.randint(0, 256, (2, 3))
    logits, state = sens(tokens)
    assert logits.shape == (2, 3, 256)
    assert state.shape == (2, 40, 4)


def test_sensory_decode_sees_current_byte_readout_does_not():
    """Structural lag: current token moves sensory, not fly readout."""
    g = synthetic_flylike(n=40, seed=0)
    read = FlyLanguageModel(g, state_dim=4, readout="mean", decode_pop="readout")
    sens = FlyLanguageModel(g, state_dim=4, readout="mean", decode_pop="sensory")
    base = torch.arange(10, 16).unsqueeze(0)
    alt = base.clone()
    alt[0, -1] = (base[0, -1] + 17) % 256

    def pop_rms(m: FlyLanguageModel) -> float:
        with torch.no_grad():
            _, h0 = m(base)
            _, h1 = m(alt)
            idx = m.sensory_idx if m.decode_pop == "sensory" else m.readout_idx
            return float((h1[:, idx] - h0[:, idx]).pow(2).mean().sqrt())

    assert pop_rms(sens) > 1e-4
    assert pop_rms(read) < 1e-8


def test_unknown_skip_is_rejected():
    g = synthetic_flylike(n=32, seed=0)
    try:
        FlyLanguageModel(g, state_dim=4, readout="mean", skip="dense")
    except ValueError as exc:
        assert "skip" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_sensory_readout_skip_has_no_extra_scalars_when_io_matched():
    g = synthetic_flylike(n=40, seed=4)
    plain = FlyLanguageModel(g, state_dim=4, readout="mean")
    skip = FlyLanguageModel(g, state_dim=4, readout="mean", skip="sensory_readout")
    n_plain = sum(p.numel() for p in plain.parameters() if p.requires_grad)
    n_skip = sum(p.numel() for p in skip.parameters() if p.requires_grad)
    assert n_plain == n_skip
    assert skip.n_synapses == plain.n_synapses
    for _, p in skip.named_parameters():
        assert list(p.shape) != [40, 40]


def test_sensory_readout_skip_puts_current_byte_on_fly_readout():
    g = synthetic_flylike(n=40, seed=0)
    plain = FlyLanguageModel(g, state_dim=4, readout="mean")
    skip = FlyLanguageModel(g, state_dim=4, readout="mean", skip="sensory_readout")
    base = torch.arange(10, 16).unsqueeze(0)
    alt = base.clone()
    alt[0, -1] = (base[0, -1] + 17) % 256

    def readout_rms(m: FlyLanguageModel) -> float:
        with torch.no_grad():
            _, h0 = m(base)
            _, h1 = m(alt)
            return float((h1[:, m.readout_idx] - h0[:, m.readout_idx]).pow(2).mean().sqrt())

    assert readout_rms(plain) < 1e-8
    assert readout_rms(skip) > 1e-4


def test_sensory_readout_skip_fly_shuffle_match_scalars():
    g = synthetic_flylike(n=40, seed=8)
    sh = degree_matched_shuffle(g, seed=2)
    fly = FlyLanguageModel(g, state_dim=4, readout="mean", skip="sensory_readout")
    rnd = FlyLanguageModel(sh, state_dim=4, readout="mean", skip="sensory_readout")
    assert sum(p.numel() for p in fly.parameters() if p.requires_grad) == sum(
        p.numel() for p in rnd.parameters() if p.requires_grad
    )


def test_unknown_update_is_rejected():
    g = synthetic_flylike(n=32, seed=0)
    try:
        FlyLanguageModel(g, state_dim=4, readout="mean", update="async")
    except ValueError as exc:
        assert "update" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_ordered_update_has_no_extra_scalars():
    g = synthetic_flylike(n=40, seed=4)
    plain = FlyLanguageModel(g, state_dim=4, readout="mean")
    ordered = FlyLanguageModel(g, state_dim=4, readout="mean", update="ordered")
    n_plain = sum(p.numel() for p in plain.parameters() if p.requires_grad)
    n_ord = sum(p.numel() for p in ordered.parameters() if p.requires_grad)
    assert n_plain == n_ord
    assert ordered.n_synapses == plain.n_synapses
    for _, p in ordered.named_parameters():
        assert list(p.shape) != [40, 40]


def test_ordered_update_puts_current_byte_on_fly_readout_through_the_graph():
    g = synthetic_flylike(n=40, seed=0)
    plain = FlyLanguageModel(g, state_dim=4, readout="mean", update="parallel")
    ordered = FlyLanguageModel(g, state_dim=4, readout="mean", update="ordered")
    ordered.load_state_dict(plain.state_dict())
    base = torch.arange(10, 16).unsqueeze(0)
    alt = base.clone()
    alt[0, -1] = (base[0, -1] + 17) % 256

    def readout_rms(m: FlyLanguageModel) -> float:
        with torch.no_grad():
            _, h0 = m(base)
            _, h1 = m(alt)
            return float((h1[:, m.readout_idx] - h0[:, m.readout_idx]).pow(2).mean().sqrt())

    d_plain = readout_rms(plain)
    d_ord = readout_rms(ordered)
    assert d_plain < 1e-8
    assert d_ord > 1e-7
    assert d_ord > d_plain


def test_ordered_fly_shuffle_match_scalars():
    g = synthetic_flylike(n=40, seed=8)
    sh = degree_matched_shuffle(g, seed=2)
    fly = FlyLanguageModel(g, state_dim=4, readout="mean", update="ordered")
    rnd = FlyLanguageModel(sh, state_dim=4, readout="mean", update="ordered")
    assert sum(p.numel() for p in fly.parameters() if p.requires_grad) == sum(
        p.numel() for p in rnd.parameters() if p.requires_grad
    )


def test_vector2_messages_widen_the_cell_and_keep_one_p_per_edge():
    g = synthetic_flylike(n=36, seed=5)
    m = FlyLanguageModel(g, state_dim=4, readout="mean", message="vector2")
    assert m.cells[0].input_size == 2
    assert m.synapse_param.numel() == g.n_edges
    tokens = torch.randint(0, 256, (2, 4))
    logits, state = m(tokens)
    assert logits.shape == (2, 4, 256)
    assert state.shape == (2, 36, 4)
    loss = nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 256), tokens[:, 1:].reshape(-1)
    )
    loss.backward()
    assert m.synapse_param.grad is not None
    assert m.synapse_param.grad.abs().sum() > 0
    for _, p in m.named_parameters():
        assert list(p.shape) != [36, 36]


def test_vector2_prefix_channels_are_not_a_single_emit():
    g = synthetic_flylike(n=32, seed=1)
    m = FlyLanguageModel(g, state_dim=4, readout="mean", message="vector2")
    with torch.no_grad():
        m.encoder.weight.zero_()
    token = torch.zeros(1, dtype=torch.long)
    a = torch.zeros(1, 32, 4)
    a[..., 0] = 1.0
    b = torch.zeros(1, 32, 4)
    b[..., 1] = 1.0
    out_a = m.step(a, token)
    out_b = m.step(b, token)
    assert not torch.allclose(out_a, out_b)


def test_projected_synapses_are_per_channel_and_sign_locked():
    g = synthetic_flylike(n=36, seed=5)
    m = FlyLanguageModel(g, state_dim=4, readout="mean", message="projected")
    assert m.cells[0].input_size == 4
    assert tuple(m.synapse_param.shape) == (g.n_edges, 4)
    w0 = m.synapse_weights().detach()
    assert w0.shape == (g.n_edges, 4)
    signs = g.sign.view(-1, 1).expand_as(w0)
    assert torch.equal(torch.sign(w0), signs)
    tokens = torch.randint(0, 256, (2, 4))
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(6):
        logits, _ = m(tokens)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 256), tokens[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    w1 = m.synapse_weights().detach()
    assert torch.equal(torch.sign(w1), signs)
    assert not torch.allclose(w1, w0)
    for _, p in m.named_parameters():
        assert list(p.shape) != [36, 36]


def test_projected_channels_are_not_tied():
    g = synthetic_flylike(n=32, seed=2)
    m = FlyLanguageModel(g, state_dim=4, readout="mean", message="projected")
    with torch.no_grad():
        m.encoder.weight.zero_()
    token = torch.zeros(1, dtype=torch.long)
    state = torch.ones(1, 32, 4)
    w = m.synapse_weights().detach().clone()
    out0 = m.step(state, token, w)
    w_hi = w.clone()
    w_hi[:, 1] = w_hi[:, 1] * 3
    out1 = m.step(state, token, w_hi)
    assert not torch.allclose(out0, out1)


def test_projected_fly_and_shuffle_match_trainable_scalars():
    g = synthetic_flylike(n=40, seed=8)
    sh = degree_matched_shuffle(g, seed=2)
    fly = FlyLanguageModel(g, state_dim=4, readout="mean", message="projected")
    rnd = FlyLanguageModel(sh, state_dim=4, readout="mean", message="projected")
    n_fly = sum(p.numel() for p in fly.parameters() if p.requires_grad)
    n_rnd = sum(p.numel() for p in rnd.parameters() if p.requires_grad)
    assert n_fly == n_rnd
    assert n_fly > FlyLanguageModel(g, state_dim=4, readout="mean", message="vector").synapse_param.numel()


def test_shuffled_topology_matches_parameter_count_not_wiring():
    g = synthetic_flylike(n=40, seed=8)
    sh = degree_matched_shuffle(g, seed=2)
    fly = FlyLanguageModel(g, state_dim=4)
    rnd = FlyLanguageModel(sh, state_dim=4)
    n_fly = sum(p.numel() for p in fly.parameters() if p.requires_grad)
    n_rnd = sum(p.numel() for p in rnd.parameters() if p.requires_grad)
    assert n_fly == n_rnd
    assert not torch.equal(fly.connectome.dst, rnd.connectome.dst)
