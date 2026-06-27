import torch
import torch.nn.functional as F

from local_ai_training.ratchet import (
    DiscreteRatchetLinear,
    RatchetEmbedding,
    audit_no_master_weights,
)


def _embedding(seed: int = 0, *, max_code: int = 7) -> RatchetEmbedding:
    torch.manual_seed(seed)
    return RatchetEmbedding(num_embeddings=12, embedding_dim=8, max_code=max_code)


def test_is_a_discrete_ratchet_linear_with_token_rows() -> None:
    embedding = _embedding()
    assert isinstance(embedding, DiscreteRatchetLinear)  # auto-joins existing sweeps
    assert embedding.code.shape == (12, 8)  # rows are tokens, cols are embedding dim
    assert embedding.scale.shape == (12,)


def test_forward_matches_f_embedding_on_effective_weight() -> None:
    embedding = _embedding().eval()
    token_ids = torch.tensor([[0, 3, 11], [5, 5, 1]])
    expected = F.embedding(token_ids, embedding.effective_weight())
    assert torch.equal(embedding(token_ids), expected)


def test_has_no_master_weight_parameter() -> None:
    embedding = _embedding()
    assert list(embedding.parameters()) == []  # buffers only; nothing AdamW would train


def test_forward_captures_and_releases_effective_weight_gradient() -> None:
    embedding = _embedding().train()
    token_ids = torch.tensor([[0, 1, 2, 3]])
    out = embedding(token_ids)
    out.square().sum().backward()
    assert embedding.has_pending_gradient
    assert embedding._effective_weight.grad is not None
    embedding.ratchet_update(validate=True)
    assert embedding._effective_weight is None  # released after update


def test_codes_move_under_a_persistent_gradient() -> None:
    embedding = _embedding(max_code=7)
    before = embedding.code.clone()
    grad = torch.ones_like(embedding.code, dtype=torch.float32)
    # pressure_threshold default 8; bucket gives +2/step for |z|>=high, so a handful of
    # identical applications must move at least one code.
    for _ in range(8):
        embedding.apply_weight_gradient(grad, validate=False)
    assert not torch.equal(embedding.code, before)


def test_audit_reports_no_violation_and_counts_embedding_state() -> None:
    embedding = _embedding()
    report = audit_no_master_weights(embedding, raise_on_violation=True)
    assert report.ratchet_layers == 1
    assert report.ratchet_state_bytes > 0
