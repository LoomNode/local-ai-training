import torch

from local_ai_training.projection import project_to_codes, reconstruction_mse


def _naive(weight, max_code):
    """The training init's quantization: scale = row_max/max_code, no scale refinement."""
    row_max = weight.abs().amax(dim=1)
    scale = (row_max / max_code).clamp_min(torch.finfo(torch.float32).tiny)
    code = torch.round(weight / scale[:, None]).clamp(-max_code, max_code).to(torch.int8)
    return code, scale.to(torch.float32)


def test_projection_beats_naive_scale_on_reconstruction():
    torch.manual_seed(0)
    w = torch.randn(64, 96)
    for max_code in (2, 3, 4):
        code, scale = project_to_codes(w, max_code)
        proj = reconstruction_mse(w, code, scale)
        naive = reconstruction_mse(w, *_naive(w, max_code))
        assert proj <= naive + 1e-12, f"projection MSE {proj} worse than naive {naive}"


def test_projection_beats_naive_across_all_state_counts():
    # Sweep max_code 1..7 (codes 3..15): projection MSE must never exceed the naive scale,
    # and codes stay in range. Guards the state-count sweep.
    torch.manual_seed(3)
    w = torch.randn(48, 80)
    for max_code in range(1, 8):
        code, scale = project_to_codes(w, max_code)
        assert int(code.abs().max()) <= max_code
        proj_mse = reconstruction_mse(w, code, scale)
        naive_mse = reconstruction_mse(w, *_naive(w, max_code))
        assert proj_mse <= naive_mse + 1e-12


def test_projection_respects_code_range_and_shapes():
    torch.manual_seed(1)
    w = torch.randn(40, 50) * 5.0
    code, scale = project_to_codes(w, 2)
    assert code.shape == w.shape
    assert code.dtype == torch.int8
    assert scale.shape == (w.shape[0],)
    assert int(code.abs().max()) <= 2
    assert torch.all(scale > 0)


def test_projection_is_near_exact_when_weight_is_already_quinary():
    # A weight that IS scale*code should reconstruct with ~zero error.
    torch.manual_seed(2)
    true_scale = torch.rand(30) + 0.1
    true_code = torch.randint(-2, 3, (30, 48)).float()
    w = true_code * true_scale[:, None]
    code, scale = project_to_codes(w, 2)
    assert reconstruction_mse(w, code, scale) < 1e-6
