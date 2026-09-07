"""Baseline GPT, looped shared-core GPT, and SCRT (looped + residual slot bank)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from scrt.config import ModelConfig


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ResidualSlotBank(nn.Module):
    """Fixed-capacity LoRA-style residual bank.

    Physical parameter count is constant: ``n_slots`` adapters of rank ``r``.
    Each slot has occupancy flag and a loop-assignment. Consolidation only
    rewrites values / occupancy / assignment — never grows tensors.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_slots = cfg.n_slots
        self.n_embd = cfg.n_embd
        self.rank = cfg.lora_rank
        self.n_loops = cfg.n_loops

        # Down / up projections per slot: x @ A^T @ B^T  (B, T, C)
        self.down = nn.Parameter(torch.empty(cfg.n_slots, cfg.lora_rank, cfg.n_embd))
        self.up = nn.Parameter(torch.empty(cfg.n_slots, cfg.n_embd, cfg.lora_rank))
        # Scale so residuals start small.
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        nn.init.zeros_(self.up)

        # Non-grad bookkeeping buffers (state, not learned tensors that grow).
        occupied = torch.zeros(cfg.n_slots, dtype=torch.bool)
        assignment = torch.full((cfg.n_slots,), -1, dtype=torch.long)
        if getattr(cfg, "init_occupied", None) is not None:
            # Exactly n slots, spread round-robin over loops.
            n_init = max(0, min(int(cfg.init_occupied), cfg.n_slots))
            for idx in range(n_init):
                occupied[idx] = True
                assignment[idx] = idx % cfg.n_loops
        else:
            # Round-robin: first slots_per_loop * n_loops slots assigned.
            idx = 0
            for loop_i in range(cfg.n_loops):
                for _ in range(cfg.slots_per_loop):
                    if idx >= cfg.n_slots:
                        break
                    occupied[idx] = True
                    assignment[idx] = loop_i
                    idx += 1
        self.register_buffer("occupied", occupied)
        self.register_buffer("assignment", assignment)  # loop index or -1
        # Running importance for regrow decisions (EMA of grad energy).
        self.register_buffer("importance", torch.zeros(cfg.n_slots))
        self.register_buffer("loop_pressure", torch.zeros(cfg.n_loops))
        # Consolidation cycles a slot has been occupied. A regrown slot starts with
        # up == 0, so it contributes exactly nothing and carries no importance until
        # up grows away from zero — freeing it on those grounds before it has had a
        # chance to bootstrap is what produced the free/regrow thrash.
        self.register_buffer("age", torch.zeros(cfg.n_slots, dtype=torch.long))

    def n_active(self) -> int:
        return int(self.occupied.sum().item())

    @torch.no_grad()
    def tick_age(self) -> None:
        """Advance the age of every occupied slot by one consolidation cycle."""
        self.age += self.occupied.long()

    def n_free(self) -> int:
        return int((~self.occupied).sum().item())

    def forward(self, x: torch.Tensor, loop_i: int) -> torch.Tensor:
        """Apply sum of residual slots assigned to this loop."""
        mask = self.occupied & (self.assignment == loop_i)
        if not bool(mask.any()):
            return torch.zeros_like(x)
        # Gather active slot params for this loop.
        idxs = mask.nonzero(as_tuple=False).view(-1)
        # x: (B, T, C); down[i]: (r, C); up[i]: (C, r)
        # delta_i = (x @ down[i].T) @ up[i].T
        down = self.down[idxs]  # (S, r, C)
        up = self.up[idxs]  # (S, C, r)
        # (B, T, S, r)
        mid = torch.einsum("btc,src->btsr", x, down)
        # (B, T, S, C)
        out = torch.einsum("btsr,scr->btsc", mid, up)
        return out.sum(dim=2)

    def slot_delta(self, x: torch.Tensor, slot_i: int) -> torch.Tensor:
        """Residual contribution of a single slot (for functional probes)."""
        if not bool(self.occupied[slot_i]):
            return torch.zeros_like(x)
        mid = torch.einsum("btc,rc->btr", x, self.down[slot_i])
        return torch.einsum("btr,cr->btc", mid, self.up[slot_i])

    @torch.no_grad()
    def free_slot(self, slot_i: int) -> None:
        self.occupied[slot_i] = False
        self.assignment[slot_i] = -1
        self.down[slot_i].zero_()
        self.up[slot_i].zero_()
        self.importance[slot_i] = 0.0
        self.age[slot_i] = 0

    @torch.no_grad()
    def allocate_slot(self, slot_i: int, loop_i: int) -> None:
        assert 0 <= loop_i < self.n_loops
        self.occupied[slot_i] = True
        self.assignment[slot_i] = loop_i
        # Small random re-init so regrown slots can learn.
        nn.init.kaiming_uniform_(self.down[slot_i : slot_i + 1], a=math.sqrt(5))
        self.up[slot_i].zero_()
        self.importance[slot_i] = 0.0
        self.age[slot_i] = 0

    def update_importance_from_grads(self, ema: float = 0.9) -> None:
        """Call after backward; uses .grad energy on down/up."""
        if self.down.grad is None or self.up.grad is None:
            return
        with torch.no_grad():
            g = self.down.grad.detach().float().pow(2).mean(dim=(1, 2))
            g = g + self.up.grad.detach().float().pow(2).mean(dim=(1, 2))
            self.importance.mul_(ema).add_(g, alpha=1.0 - ema)
            # Loop pressure = sum importance of slots on that loop.
            pressure = torch.zeros(self.n_loops, device=self.importance.device)
            for s in range(self.n_slots):
                if self.occupied[s] and self.assignment[s] >= 0:
                    pressure[self.assignment[s]] += self.importance[s]
            self.loop_pressure.copy_(pressure)


class GPTBase(nn.Module):
    """Shared embedding / head plumbing."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying.
        self.lm_head.weight = self.wte.weight

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"seq len {T} > block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        return self.drop(self.wte(idx) + self.wpe(pos))

    def head(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.ln_f(x))


class BaselineGPT(GPTBase):
    """Standard stacked transformer — independent layers."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def residual_bank(self) -> Optional[ResidualSlotBank]:
        return None


class LoopedGPT(GPTBase):
    """Shared core reused ``n_loops`` times (no residual bank)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.prelude = Block(cfg)
        self.core = Block(cfg)
        self.coda = Block(cfg)
        self.n_loops = cfg.n_loops

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embed(idx)
        x = self.prelude(x)
        for _ in range(self.n_loops):
            x = self.core(x)
        x = self.coda(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def residual_bank(self) -> Optional[ResidualSlotBank]:
        return None


class SharedLoRA(nn.Module):
    """Single LoRA adapter (born-consolidated). Rank defaults to cfg.lora_rank."""

    def __init__(self, cfg: ModelConfig, rank: Optional[int] = None) -> None:
        super().__init__()
        r = cfg.lora_rank if rank is None else rank
        self.down = nn.Linear(cfg.n_embd, r, bias=False)
        self.up = nn.Linear(r, cfg.n_embd, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class SharedResidualGPT(GPTBase):
    """Looped core + *one* residual LoRA reused on every loop.

    Nipping residual redundancy in the bud: no per-loop adapter copies,
    no merge/free/regrow.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.prelude = Block(cfg)
        self.core = Block(cfg)
        self.coda = Block(cfg)
        self.shared_lora = SharedLoRA(cfg)
        self.n_loops = cfg.n_loops

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embed(idx)
        x = self.prelude(x)
        for _ in range(self.n_loops):
            x = self.core(x)
            x = x + self.shared_lora(x)
        x = self.coda(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def residual_bank(self) -> Optional[ResidualSlotBank]:
        return None


class ModulatedResidualGPT(GPTBase):
    """Looped core + shared LoRA gated by a tiny loop-index embedding.

    Common residual weights once; loop identity only modulates (FiLM-style gate),
    not a second full adapter.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.prelude = Block(cfg)
        self.core = Block(cfg)
        self.coda = Block(cfg)
        self.shared_lora = SharedLoRA(cfg)
        self.loop_emb = nn.Embedding(cfg.n_loops, cfg.n_embd)
        # Gate: e_i -> per-channel scale in (0, 2) around 1.
        self.gate = nn.Linear(cfg.n_embd, cfg.n_embd, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.n_loops = cfg.n_loops

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embed(idx)
        x = self.prelude(x)
        for loop_i in range(self.n_loops):
            x = self.core(x)
            e = self.loop_emb.weight[loop_i]
            # sigmoid(0)=0.5 → scale in ~[0, 2] with zero-init gate → starts at 1.0
            scale = 2.0 * torch.sigmoid(self.gate(e))
            x = x + self.shared_lora(x) * scale
        x = self.coda(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def residual_bank(self) -> Optional[ResidualSlotBank]:
        return None


class SharedPlusTinyGPT(GPTBase):
    """Born-merge form: shared residual + small per-loop residual.

    This is what reactive merge *aims* to produce without running merge:

        residual = shared_LoRA(x) + tiny_LoRA_i(x)

    Shared part holds the common direction; each loop only spends a thin
    low-rank correction (cfg.tiny_lora_rank << cfg.lora_rank typically).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.prelude = Block(cfg)
        self.core = Block(cfg)
        self.coda = Block(cfg)
        self.shared_lora = SharedLoRA(cfg, rank=cfg.lora_rank)
        self.tiny = nn.ModuleList(
            [SharedLoRA(cfg, rank=cfg.tiny_lora_rank) for _ in range(cfg.n_loops)]
        )
        self.n_loops = cfg.n_loops

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embed(idx)
        x = self.prelude(x)
        for loop_i in range(self.n_loops):
            x = self.core(x)
            x = x + self.shared_lora(x) + self.tiny[loop_i](x)
        x = self.coda(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def residual_bank(self) -> Optional[ResidualSlotBank]:
        return None


class SCRTGPT(GPTBase):
    """Looped shared core + fixed residual slot bank (self-consolidating)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.prelude = Block(cfg)
        self.core = Block(cfg)
        self.coda = Block(cfg)
        self.bank = ResidualSlotBank(cfg)
        self.n_loops = cfg.n_loops

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        collect_loop_acts: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[torch.Tensor]]]:
        x = self.embed(idx)
        x = self.prelude(x)
        loop_acts: list[torch.Tensor] = []
        for loop_i in range(self.n_loops):
            x = self.core(x)
            x = x + self.bank(x, loop_i)
            if collect_loop_acts:
                loop_acts.append(x.detach())
        x = self.coda(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, (loop_acts if collect_loop_acts else None)

    def residual_bank(self) -> Optional[ResidualSlotBank]:
        return self.bank


def build_model(arm: str, cfg: ModelConfig) -> nn.Module:
    if arm == "baseline":
        return BaselineGPT(cfg)
    if arm == "looped":
        return LoopedGPT(cfg)
    if arm == "shared_residual":
        return SharedResidualGPT(cfg)
    if arm == "modulated":
        return ModulatedResidualGPT(cfg)
    if arm == "shared_plus_tiny":
        return SharedPlusTinyGPT(cfg)
    if arm in ("scrt", "scrt_static"):
        # Same architecture; consolidate on/off is a training choice.
        return SCRTGPT(cfg)
    raise ValueError(f"unknown arm: {arm}")


@dataclass
class ForwardResult:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]


def model_forward(
    model: nn.Module, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
) -> ForwardResult:
    """Uniform forward across arms (SCRT returns a 3-tuple; may be PaddedModel)."""
    out = model(idx, targets)
    # SCRT (or padded SCRT) returns (logits, loss, loop_acts_or_none)
    if isinstance(out, tuple) and len(out) == 3:
        logits, loss, _ = out
    else:
        logits, loss = out
    return ForwardResult(logits=logits, loss=loss)
