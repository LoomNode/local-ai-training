from dataclasses import replace

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


def test_fp32_model_uses_bias_free_linear_layers_and_matched_support_initialization() -> None:
    ratchet = build_seeded_model(tiny_config(), max_code=2, seed=123)
    fp32 = build_seeded_model(tiny_config(), max_code=None, seed=123)

    assert torch.equal(ratchet.token_embedding.weight, fp32.token_embedding.weight)
    linears = [module for module in fp32.modules() if isinstance(module, nn.Linear)]
    assert len(linears) == 2 * 4 + 1
    assert all(module.bias is None for module in linears)


def test_model_config_rejects_unknown_matmul_mode() -> None:
    try:
        ModelConfig(vocab_size=11, matmul_mode="tf32")
    except ValueError as error:
        assert "matmul_mode" in str(error)
    else:
        raise AssertionError("expected matmul mode validation")


def test_bf16_and_int8_modes_share_identical_initial_state() -> None:
    base = tiny_config()
    bf16 = build_seeded_model(replace(base, matmul_mode="bf16"), max_code=3, seed=123)
    int8 = build_seeded_model(replace(base, matmul_mode="int8"), max_code=3, seed=123)

    assert torch.equal(bf16.token_embedding.weight, int8.token_embedding.weight)
    assert torch.equal(bf16.position_encoding, int8.position_encoding)
    bf16_layers = [m for m in bf16.modules() if isinstance(m, DiscreteRatchetLinear)]
    int8_layers = [m for m in int8.modules() if isinstance(m, DiscreteRatchetLinear)]
    for bf16_layer, int8_layer in zip(bf16_layers, int8_layers, strict=True):
        assert torch.equal(bf16_layer.packed, int8_layer.packed)
        assert torch.equal(bf16_layer.scale, int8_layer.scale)
        assert bf16_layer.matmul_mode == "bf16"
        assert int8_layer.matmul_mode == "int8"
    bf16_support = dict(bf16.named_parameters())
    int8_support = dict(int8.named_parameters())
    assert bf16_support.keys() == int8_support.keys()
    assert all(torch.equal(bf16_support[name], int8_support[name]) for name in bf16_support)
