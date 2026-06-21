# scripts/int8_spike/activation_precision.py
# How much accuracy is lost quantizing ACTIVATIONS to int8 vs int4 (per-token symmetric scales)?
# int8 was ~1% in the earlier spike; int4 (16 levels) is the scary one -- outliers force a wide
# scale that crushes the many small values. Measure on realistic activations (gaussian + heavy
# tail outliers, like real transformer activations) via the matmul output error vs bf16.
import torch

torch.manual_seed(0)
dev = "cuda"
M, K, N = 4096, 4096, 4096


def quant_per_token(x, n_levels):
    qmax = (n_levels - 1) // 2  # symmetric: int8->127, int4->7
    scale = x.abs().amax(dim=1, keepdim=True) / qmax
    q = torch.clamp((x / scale).round(), -qmax, qmax)
    return q * scale  # dequantized (fake-quant), same dtype


def realistic_acts(M, K):
    x = torch.randn(M, K, device=dev)
    # inject transformer-style outlier channels (a few features ~10-30x larger)
    out_ch = torch.randperm(K, device=dev)[: K // 256]
    x[:, out_ch] *= 20.0
    return x


x = realistic_acts(M, K)
w = torch.randn(K, N, device=dev) * 0.02
ref = x @ w  # bf16-ish full-precision reference (fp32 here)


def report(name, xq, wq):
    out = xq @ wq
    rel = (out - ref).norm() / ref.norm()
    print(f"{name:28s} rel L2 err = {100 * rel.item():6.3f}%")


print(f"activations: {M}x{K}, ~{K//256} outlier channels @20x, per-token symmetric quant\n")
report("int8 act / fp w", quant_per_token(x, 256), w)
report("int4 act / fp w", quant_per_token(x, 16), w)
report("int8 act / int8 w(per-col)", quant_per_token(x, 256), quant_per_token(w.t(), 256).t())
report("int4 act / int4 w(per-col)", quant_per_token(x, 16), quant_per_token(w.t(), 16).t())
# the ratchet's real case: weights already low-state (we use the int4/code path), bf16 activations
report("bf16 act (baseline)", x, w)
