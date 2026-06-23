import torch
from torch import nn

from local_ai_training.qat import QATLinear
from local_ai_training.ratchet import DiscreteRatchetLinear


def _ratchet_effective(layer: DiscreteRatchetLinear) -> torch.Tensor:
    return layer.code.to(torch.float32) * layer.scale.to(torch.float32)[:, None]


def test_qat_quantized_weight_matches_ratchet_quantizer_bit_exact() -> None:
    torch.manual_seed(0)
    reference = torch.empty(12, 20, dtype=torch.float32)
    nn.init.kaiming_uniform_(reference, a=5**0.5)
    for max_code in (2, 3, 4):
        qat = QATLinear(20, 12, max_code=max_code, initial_weight=reference)
        ratchet = DiscreteRatchetLinear.from_reference(reference, max_code=max_code)
        assert torch.equal(qat.quantized_weight().detach(), _ratchet_effective(ratchet))


def test_qat_init_matches_kaiming_uniform_like_nn_linear() -> None:
    # Same seed/RNG draw as nn.Linear -> shared logical FP init across arms.
    torch.manual_seed(7)
    qat = QATLinear(20, 12, max_code=2)
    torch.manual_seed(7)
    ref = nn.Linear(20, 12, bias=False)
    assert torch.equal(qat.weight, ref.weight)


def test_qat_ste_gradient_reaches_master_including_saturated() -> None:
    torch.manual_seed(1)
    qat = QATLinear(8, 4, max_code=2)
    # Force saturation: large weights so |weight/scale| > max_code for many entries.
    with torch.no_grad():
        qat.weight.mul_(50.0)
    x = torch.randn(6, 8)
    qat(x).pow(2).sum().backward()
    assert qat.weight.grad is not None
    assert torch.count_nonzero(qat.weight.grad) > 0


def test_qat_forward_equals_linear_with_quantized_weight() -> None:
    torch.manual_seed(2)
    qat = QATLinear(10, 5, max_code=3)
    x = torch.randn(7, 10)
    expected = torch.nn.functional.linear(x, qat.quantized_weight())
    assert torch.equal(qat(x), expected)
