"""Tiny GPT whose dense matrices use discrete ratchet storage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .qat import QATLinear
from .ratchet import DiscreteRatchetLinear, RatchetUpdateStats


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    pressure_threshold: int = 8
    bucket_low: float = 0.5
    bucket_high: float = 1.5
    trainable_scale: bool = False
    rms_ema_beta: float = 0.0
    pressure_leak_period: int = 0
    compile_update: bool = False
    gradient_checkpointing: bool = False
    matmul_mode: Literal["fp32", "bf16", "int8"] = "fp32"
    qat: bool = False

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_embd) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.matmul_mode not in {"fp32", "bf16", "int8"}:
            raise ValueError("matmul_mode must be fp32, bf16, or int8")


def _sinusoidal_positions(block_size: int, n_embd: int) -> Tensor:
    positions = torch.arange(block_size, dtype=torch.float32)[:, None]
    even_dimensions = torch.arange(0, n_embd, 2, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10_000.0) * even_dimensions / n_embd)
    encoding = torch.zeros(block_size, n_embd, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if n_embd > 1:
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


def _linear(config: ModelConfig, in_features: int, out_features: int, max_code: int | None):
    if max_code is None:
        return nn.Linear(in_features, out_features, bias=False)
    if config.qat:
        return QATLinear(in_features, out_features, max_code=max_code)
    return DiscreteRatchetLinear(
        in_features,
        out_features,
        max_code=max_code,
        pressure_threshold=config.pressure_threshold,
        bucket_low=config.bucket_low,
        bucket_high=config.bucket_high,
        trainable_scale=config.trainable_scale,
        rms_ema_beta=config.rms_ema_beta,
        pressure_leak_period=config.pressure_leak_period,
        compile_update=config.compile_update,
        matmul_mode=config.matmul_mode,
        fuse_backward_update=True,
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, *, max_code: int | None) -> None:
        super().__init__()
        self.qkv = _linear(config, config.n_embd, 3 * config.n_embd, max_code)
        self.projection = _linear(config, config.n_embd, config.n_embd, max_code)
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, sequence_length, channels = inputs.shape
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)

        def heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch_size, sequence_length, self.n_head, self.head_size).transpose(
                1, 2
            )

        attended = F.scaled_dot_product_attention(
            heads(query),
            heads(key),
            heads(value),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        joined = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.projection(joined)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig, *, max_code: int | None) -> None:
        super().__init__()
        hidden_size = 4 * config.n_embd
        self.expand = _linear(config, config.n_embd, hidden_size, max_code)
        self.contract = _linear(config, hidden_size, config.n_embd, max_code)
        self.dropout = config.dropout

    def forward(self, inputs: Tensor) -> Tensor:
        return F.dropout(
            self.contract(F.gelu(self.expand(inputs))),
            p=self.dropout,
            training=self.training,
        )


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, *, max_code: int | None) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.n_embd)
        self.attention = CausalSelfAttention(config, max_code=max_code)
        self.feed_forward_norm = nn.RMSNorm(config.n_embd)
        self.feed_forward = FeedForward(config, max_code=max_code)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.feed_forward(self.feed_forward_norm(inputs))


class RatchetGPT(nn.Module):
    def __init__(self, config: ModelConfig, *, max_code: int | None) -> None:
        super().__init__()
        self.config = config
        self.max_code = max_code
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.register_buffer(
            "position_encoding", _sinusoidal_positions(config.block_size, config.n_embd)
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config, max_code=max_code) for _ in range(config.n_layer)
        )
        self.final_norm = nn.RMSNorm(config.n_embd)
        self.lm_head = _linear(config, config.n_embd, config.vocab_size, max_code)

    def forward(
        self, tokens: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        sequence_length = tokens.shape[1]
        if sequence_length > self.config.block_size:
            raise ValueError("token sequence exceeds configured block_size")
        hidden = self.token_embedding(tokens) + self.position_encoding[:sequence_length]
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                import torch.utils.checkpoint
                hidden = torch.utils.checkpoint.checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != tokens.shape:
                raise ValueError("targets must have the same shape as tokens")
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    def ratchet_update(self) -> RatchetUpdateStats:
        updates = [
            module.ratchet_update()
            for module in self.modules()
            if isinstance(module, DiscreteRatchetLinear)
        ]
        total_weights = sum(update.total_weights for update in updates)
        return RatchetUpdateStats(
            total_weights=total_weights,
            positive_moves=sum(update.positive_moves for update in updates),
            negative_moves=sum(update.negative_moves for update in updates),
            blocked_positive_moves=sum(update.blocked_positive_moves for update in updates),
            blocked_negative_moves=sum(update.blocked_negative_moves for update in updates),
            gradient_rms_mean=(
                sum(update.gradient_rms_mean for update in updates) / len(updates)
                if updates
                else 0.0
            ),
        )

    def discard_pending_gradients(self) -> None:
        for module in self.modules():
            if isinstance(module, DiscreteRatchetLinear):
                module.discard_pending_gradient()


def build_seeded_model(config: ModelConfig, *, max_code: int | None, seed: int) -> RatchetGPT:
    """Build matched arms without changing the caller's global CPU RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return RatchetGPT(config, max_code=max_code)
