"""
EDITABLE by the research agent / auto-loop.

Lab goal: maximize quality under the hardware envelope (see prepare.py),
with a bias toward size efficiency (better bpb per param / VRAM).

Knobs between the markers are mutated by `ailab loop`. Architecture variants
live in Block / MemoryBlock — the loop can toggle USE_MEMORY_BLOCK and sizes.

Usage:
  ENVELOPE_MODE=solo uv run python train.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab_config import apply_cuda_device, load_lab_config
from prepare import (
    MAX_SEQ_LEN,
    TIME_BUDGET_S,
    evaluate_bpb,
    make_batch,
    measure_decode_tps,
    peak_vram_mb,
    prepare_data,
    print_report,
)

# ---------------------------------------------------------------------------


# AGENT_KNOBS_BEGIN
DEPTH = 6
N_EMBD = 384
N_HEAD = 6
BATCH_SIZE = 16
LR = 0.0003
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1
USE_MEMORY_BLOCK = 0
MEMORY_SLOTS = 16
MLP_MULT = 4
# AGENT_KNOBS_END
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    seq_len: int = MAX_SEQ_LEN
    n_layer: int = DEPTH
    n_head: int = N_HEAD
    n_embd: int = N_EMBD
    use_memory: bool = bool(USE_MEMORY_BLOCK)
    memory_slots: int = MEMORY_SLOTS
    mlp_mult: int = MLP_MULT


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.mlp_mult * cfg.n_embd
        self.fc = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """Plain residual block."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MemoryBlock(nn.Module):
    """
    Weight + memory hypothesis: fixed compute (attn/mlp) plus a small bank of
    learned *and* input-addressed slots that can reuse state cheaply.

    slots: (n_slots, d) parameters + soft read/write from token stream.
    Intent: fewer layers or narrower FFN compensated by durable local memory.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
        self.n_slots = max(1, cfg.memory_slots)
        self.slot = nn.Parameter(torch.zeros(self.n_slots, cfg.n_embd))
        nn.init.normal_(self.slot, std=0.02)
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.out = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.ln_m = nn.LayerNorm(cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        # soft-read memory slots (shared across batch/time via content addressing)
        B, T, C = x.shape
        q = self.q_proj(self.ln_m(x))  # B,T,C
        # slots: S,C
        keys = self.slot  # S,C
        # attention over slots
        scale = C**-0.5
        scores = torch.einsum("btc,sc->bts", q, keys) * scale
        w = scores.softmax(dim=-1)
        read = torch.einsum("bts,sc->btc", w, self.v_proj(keys))
        x = x + self.out(read)
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.seq_len, cfg.n_embd)
        block_cls = MemoryBlock if cfg.use_memory else Block
        self.blocks = nn.ModuleList([block_cls(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.seq_len:
            raise ValueError(f"seq len {T} > {self.cfg.seq_len}")
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def main() -> None:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    lab = load_lab_config()
    apply_cuda_device(lab.experiment_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    t_start = time.perf_counter()
    prepare_data()

    cfg = GPTConfig(
        seq_len=MAX_SEQ_LEN,
        n_layer=DEPTH,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        use_memory=bool(USE_MEMORY_BLOCK),
        memory_slots=MEMORY_SLOTS,
        mlp_mult=MLP_MULT,
    )
    # heads must divide embd
    if cfg.n_embd % cfg.n_head != 0:
        raise SystemExit(f"N_EMBD={cfg.n_embd} not divisible by N_HEAD={cfg.n_head}")

    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
    )

    n_params = model.num_params()
    print(
        f"model params_M={n_params / 1e6:.2f} depth={DEPTH} embd={N_EMBD} "
        f"mem={USE_MEMORY_BLOCK} slots={MEMORY_SLOTS} mlp_mult={MLP_MULT} "
        f"device={device} lab_mode={lab.mode} physical_gpu={lab.experiment_gpu}"
    )

    model.train()
    t_train0 = time.perf_counter()
    steps = 0
    running = 0.0
    while True:
        elapsed = time.perf_counter() - t_train0
        if elapsed >= TIME_BUDGET_S:
            break
        x, y = make_batch("train", BATCH_SIZE, MAX_SEQ_LEN, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        steps += 1
        running = (
            0.9 * running + 0.1 * float(loss.item()) if steps > 1 else float(loss.item())
        )
        if steps % 20 == 0:
            print(f"step {steps} loss {running:.4f} elapsed {elapsed:.1f}s")

    training_seconds = time.perf_counter() - t_train0
    val_bpb = evaluate_bpb(model, device, seq_len=MAX_SEQ_LEN)
    decode_tps = measure_decode_tps(model, device)
    peak_mb = peak_vram_mb(device)
    total_seconds = time.perf_counter() - t_start
    density = (1.0 / max(val_bpb, 1e-6)) / max(n_params / 1e6, 1e-6)  # higher better

    print_report(
        val_bpb=val_bpb,
        training_seconds=training_seconds,
        total_seconds=total_seconds,
        peak_vram_mb_val=peak_mb,
        decode_tps=decode_tps,
        num_params_m=n_params / 1e6,
        num_steps=steps,
        extra={
            "depth": DEPTH,
            "n_embd": N_EMBD,
            "batch_size": BATCH_SIZE,
            "use_memory_block": USE_MEMORY_BLOCK,
            "memory_slots": MEMORY_SLOTS,
            "mlp_mult": MLP_MULT,
            "density": f"{density:.6f}",
            "lab_mode": lab.mode,
            "experiment_gpu": lab.experiment_gpu,
        },
    )


def main_cli() -> None:
    main()


if __name__ == "__main__":
    main()
