import torch
from torch import nn

from local_ai_training.model import ModelConfig, RatchetGPT, build_seeded_model
from local_ai_training.ratchet import DiscreteRatchetLinear, audit_no_master_weights


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=11,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )


def test_seeded_quinary_and_septenary_models_share_logical_initialization() -> None:
    quinary = build_seeded_model(tiny_config(), max_code=2, seed=123)
    septenary = build_seeded_model(tiny_config(), max_code=3, seed=123)

    assert torch.equal(quinary.token_embedding.weight, septenary.token_embedding.weight)
    q_layers = [m for m in quinary.modules() if isinstance(m, DiscreteRatchetLinear)]
    s_layers = [m for m in septenary.modules() if isinstance(m, DiscreteRatchetLinear)]
    assert len(q_layers) == len(s_layers) > 0
    for q_layer, s_layer in zip(q_layers, s_layers, strict=True):
        assert torch.allclose(q_layer.scale * 2, s_layer.scale * 3)


def test_every_dense_transformer_path_uses_ratchet_weights() -> None:
    model = build_seeded_model(tiny_config(), max_code=2, seed=9)

    assert not any(isinstance(module, nn.Linear) for module in model.modules())
    report = audit_no_master_weights(model, raise_on_violation=True)
    assert report.ratchet_layers == 2 * 4 + 1
    assert model.position_encoding.requires_grad is False


def test_forward_backward_and_ratchet_update_clear_all_temporary_weights() -> None:
    model = build_seeded_model(tiny_config(), max_code=3, seed=5)
    model.train()
    tokens = torch.randint(0, tiny_config().vocab_size, (3, tiny_config().block_size))

    logits, loss = model(tokens, tokens)
    assert logits.shape == (3, tiny_config().block_size, tiny_config().vocab_size)
    assert loss is not None
    loss.backward()
    stats = model.ratchet_update()

    ratchet_layers = [m for m in model.modules() if isinstance(m, DiscreteRatchetLinear)]
    assert stats.total_weights == sum(layer.code.numel() for layer in ratchet_layers)
    assert all(not layer.has_pending_gradient for layer in ratchet_layers)
    assert all(layer._effective_weight is None for layer in ratchet_layers)


def test_sequence_longer_than_context_is_rejected() -> None:
    model = RatchetGPT(tiny_config(), max_code=2)
    tokens = torch.zeros((1, tiny_config().block_size + 1), dtype=torch.long)

    try:
        model(tokens)
    except ValueError as error:
        assert "block_size" in str(error)
    else:
        raise AssertionError("expected context-length validation")

