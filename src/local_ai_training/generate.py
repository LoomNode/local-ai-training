"""Autoregressive character sampling from a saved ratchet checkpoint.

The training pipeline has no inference path; this module is the read-only
counterpart. It rebuilds the model purely from a checkpoint's metadata
(architecture + vocabulary), loads the persisted weights, and samples one
character at a time. Effective FP weights are materialized exactly as in the
forward pass during training, so no master weights are introduced.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import ModelConfig, RatchetGPT, build_seeded_model


def _metadata_path(base_path: str | Path) -> Path:
    base = Path(base_path)
    if base.suffix in {".safetensors", ".json"}:
        base = base.with_suffix("")
    return base.with_suffix(".json")


def _tensor_path(base_path: str | Path) -> Path:
    base = Path(base_path)
    if base.suffix in {".safetensors", ".json"}:
        base = base.with_suffix("")
    return base.with_suffix(".safetensors")


def load_for_generation(
    base_path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[RatchetGPT, tuple[str, ...]]:
    """Rebuild a model from its checkpoint and return it (in eval mode) plus its vocabulary."""
    from safetensors.torch import load_file

    metadata = json.loads(_metadata_path(base_path).read_text())
    vocabulary = tuple(metadata["vocabulary"])
    config = metadata["experiment_config"]
    model_config = ModelConfig(
        vocab_size=len(vocabulary),
        block_size=int(config["block_size"]),
        n_layer=int(config["n_layer"]),
        n_head=int(config["n_head"]),
        n_embd=int(config["n_embd"]),
        dropout=0.0,
        matmul_mode=config.get("matmul_mode", "fp32"),
    )
    # max_code 0 marks an FP32 control (plain nn.Linear); >=1 is a ratchet model.
    max_code = int(metadata["max_code"]) or None
    model = build_seeded_model(model_config, max_code=max_code, seed=0)

    tensors = load_file(_tensor_path(base_path))
    model_state = {
        key.removeprefix("model::"): value
        for key, value in tensors.items()
        if key.startswith("model::")
    }
    model.load_state_dict(model_state, strict=True)
    return model.to(device).eval(), vocabulary


@torch.no_grad()
def generate(
    model: RatchetGPT,
    vocabulary: tuple[str, ...],
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> str:
    """Sample `max_new_tokens` characters continuing `prompt`. Returns only the new text.

    `temperature == 0` is greedy (deterministic). A non-None `seed` makes sampling reproducible.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    char_to_id = {character: index for index, character in enumerate(vocabulary)}
    unknown = sorted({character for character in prompt if character not in char_to_id})
    if unknown:
        raise ValueError(f"prompt characters not in the model vocabulary: {unknown!r}")

    device = device or next(model.parameters()).device
    block_size = model.config.block_size
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    context = [char_to_id[character] for character in prompt]
    if not context:
        # Seed generation with token 0 so there is always a context window.
        context = [0]
    ids = torch.tensor([context], dtype=torch.long, device=device)

    produced: list[int] = []
    for _ in range(max_new_tokens):
        cropped = ids[:, -block_size:]
        logits, _ = model(cropped)
        next_logits = logits[:, -1, :]
        if temperature == 0.0:
            next_id = int(torch.argmax(next_logits, dim=-1).item())
        else:
            next_logits = next_logits / temperature
            if top_k is not None:
                k = min(top_k, next_logits.shape[-1])
                kth_value = torch.topk(next_logits, k, dim=-1).values[:, -1, None]
                next_logits = next_logits.masked_fill(next_logits < kth_value, float("-inf"))
            probabilities = torch.softmax(next_logits, dim=-1).cpu()
            sampled = torch.multinomial(probabilities, num_samples=1, generator=generator)
            next_id = int(sampled.item())
        produced.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

    return "".join(vocabulary[token] for token in produced)
