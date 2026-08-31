"""Sequential MoE: frozen trunk + resident-only expert FFN bank + token Top-K."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from seqmoe.config import ModelConfig

VALID_MODES = ("pretrain", "expert", "router", "eval")


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def expert_mlp_numel(cfg: ModelConfig) -> int:
    """Parameter count of one expert's FFN bank (all layers, fc+proj)."""
    hidden = cfg.mlp_mult * cfg.n_embd
    per_layer = cfg.n_embd * hidden + hidden * cfg.n_embd
    return per_layer * cfg.n_layer


@dataclass
class ForwardResult:
    logits: torch.Tensor
    loss: torch.Tensor | None
    aux_loss: torch.Tensor | None
    route_frac: torch.Tensor | None  # (E,) token mass per expert
    token_route_weights: torch.Tensor | None = None  # (B, T, E) mean over layers
    gold_agree: torch.Tensor | None = None
    route_ce: torch.Tensor | None = None


@dataclass
class _RouteOpts:
    force_expert: int | None = None
    route: str = "learned"
    hard_id: int | None = None
    top_k: int | None = None
    gold_id: torch.Tensor | None = None


class MLP(nn.Module):
    def __init__(self, n_embd: int, mlp_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, mlp_mult * n_embd, bias=False)
        self.proj = nn.Linear(mlp_mult * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


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


class ExpertBank(nn.Module):
    """Per-layer map logical_id → MLP. Absent id ⇒ not in the module tree."""

    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleDict()

    def install(self, logical_id: int, mlp: MLP) -> None:
        self.experts[str(int(logical_id))] = mlp

    def uninstall(self, logical_id: int) -> None:
        key = str(int(logical_id))
        if key in self.experts:
            del self.experts[key]

    def resident_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(k) for k in self.experts.keys()))

    def get(self, logical_id: int) -> MLP:
        return self.experts[str(int(logical_id))]


class Router(nn.Module):
    def __init__(self, n_embd: int, n_experts: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Linear(n_embd, n_experts, bias=False)
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → logits (B, T, E)
        return self.gate(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp: MLP | None = MLP(cfg.n_embd, cfg.mlp_mult, cfg.dropout)
        self.bank: ExpertBank | None = None
        self.router: Router | None = None

    def forward(
        self, x: torch.Tensor, opts: _RouteOpts | None = None
    ) -> tuple[torch.Tensor, dict]:
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        stats: dict = {}
        if self.mlp is not None:
            x = x + self.mlp(h)
            return x, stats
        if self.bank is None:
            raise RuntimeError("block has neither dense MLP nor expert bank")
        delta, stats = run_expert_ffn(h, self.bank, self.router, self.cfg, opts)
        return x + delta, stats


def run_expert_ffn(
    x: torch.Tensor,
    bank: ExpertBank,
    router: Router | None,
    cfg: ModelConfig,
    opts: _RouteOpts | None,
) -> tuple[torch.Tensor, dict]:
    """Token-level gather/scatter. Runs only selected expert MLPs."""
    opts = opts or _RouteOpts()
    B, T, C = x.shape
    resident = bank.resident_ids()
    if not resident:
        raise RuntimeError("expert bank is empty")

    if opts.force_expert is not None:
        eid = int(opts.force_expert)
        if eid not in resident:
            raise ValueError(f"force_expert={eid} is not resident {resident}")
        y = bank.get(eid)(x)
        w = torch.zeros(B, T, cfg.n_experts, device=x.device, dtype=x.dtype)
        w[:, :, eid] = 1.0
        return y, {"token_weights": w, "aux": None, "logits": None}

    if opts.route == "hard":
        if opts.hard_id is None:
            raise ValueError("route='hard' requires hard_id")
        eid = int(opts.hard_id)
        if eid not in resident:
            raise ValueError(f"hard_id={eid} is not resident {resident}")
        y = bank.get(eid)(x)
        w = torch.zeros(B, T, cfg.n_experts, device=x.device, dtype=x.dtype)
        w[:, :, eid] = 1.0
        return y, {"token_weights": w, "aux": None, "logits": None}

    # Expert mode (one resident, no router): run that MLP. Top-K is a router
    # constraint and must not fire while a specialist is training alone.
    if router is None and len(resident) == 1:
        eid = resident[0]
        y = bank.get(eid)(x)
        w = torch.zeros(B, T, cfg.n_experts, device=x.device, dtype=x.dtype)
        w[:, :, eid] = 1.0
        return y, {"token_weights": w, "aux": None, "logits": None}

    k = cfg.top_k if opts.top_k is None else int(opts.top_k)
    if k < 1:
        raise ValueError("top_k must be >= 1")
    if k > len(resident):
        raise ValueError(f"top_k={k} > resident experts={len(resident)}")

    N = B * T
    x_flat = x.reshape(N, C)

    gate_logits: torch.Tensor | None = None
    if opts.route == "random":
        # Uniform per token: sample k distinct resident experts, equal weight.
        scores = torch.rand(N, len(resident), device=x.device, dtype=x.dtype)
        val, local_idx = scores.topk(k, dim=-1)
        resident_t = torch.tensor(resident, device=x.device, dtype=torch.long)
        idx = resident_t[local_idx]
        w = torch.full((N, k), 1.0 / k, device=x.device, dtype=x.dtype)
    elif opts.route == "learned":
        if router is None:
            if len(resident) == 1:
                eid = resident[0]
                y = bank.get(eid)(x)
                tw = torch.zeros(B, T, cfg.n_experts, device=x.device, dtype=x.dtype)
                tw[:, :, eid] = 1.0
                return y, {"token_weights": tw, "aux": None, "logits": None}
            raise ValueError("learned routing requires a router")
        gate_logits = router(x)  # (B, T, E)
        logits_flat = gate_logits.reshape(N, cfg.n_experts)
        val, idx = logits_flat.topk(k, dim=-1)
        w = torch.softmax(val, dim=-1)
    else:
        raise ValueError(f"unknown route: {opts.route}")

    out = _gather_scatter(x_flat, idx, w, bank, cfg.n_experts)
    token_w = torch.zeros(N, cfg.n_experts, device=x.device, dtype=x.dtype)
    token_w.scatter_(1, idx, w)
    token_w = token_w.view(B, T, cfg.n_experts)

    aux = None
    if opts.route == "learned" and gate_logits is not None:
        aux = _balance_loss(gate_logits, idx)

    return out.view(B, T, C), {
        "token_weights": token_w,
        "aux": aux,
        "logits": gate_logits,
    }


def _gather_scatter(
    x_flat: torch.Tensor,
    idx: torch.Tensor,
    w: torch.Tensor,
    bank: ExpertBank,
    n_experts: int,
) -> torch.Tensor:
    """Run each selected expert only on the tokens that chose it."""
    del n_experts
    N, C = x_flat.shape
    out = torch.zeros(N, C, device=x_flat.device, dtype=x_flat.dtype)
    # (N, k) → per-expert token mask. topk ids are unique along k.
    for eid in bank.resident_ids():
        slot_match = idx == eid  # (N, k)
        token_mask = slot_match.any(dim=-1)
        if not bool(token_mask.any()):
            continue
        w_e = (w * slot_match.to(w.dtype)).sum(dim=-1)  # (N,)
        y = bank.get(eid)(x_flat[token_mask])
        out[token_mask] = out[token_mask] + w_e[token_mask].unsqueeze(-1) * y
    return out


def _balance_loss(gate_logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Switch-Transformer load-balance: n_experts * sum_i f_i P_i."""
    B, T, E = gate_logits.shape
    probs = torch.softmax(gate_logits, dim=-1)
    p_i = probs.mean(dim=(0, 1))
    top1 = idx[:, 0] if idx.ndim == 2 else idx.reshape(B * T, -1)[:, 0]
    f = torch.zeros(E, device=gate_logits.device, dtype=probs.dtype)
    ones = torch.ones(top1.numel(), device=gate_logits.device, dtype=probs.dtype)
    f.scatter_add_(0, top1, ones)
    f = f / float(top1.numel())
    return E * (f * p_i).sum()


class SeqMoE(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = "pretrain"
        self._expert_id: int | None = None
        self._mlp_templates: list[dict[str, torch.Tensor]] | None = None
        self._stored_trunk_sha256: str | None = None

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    @classmethod
    def from_trunk(cls, path: str | Path, map_location: str = "cpu") -> "SeqMoE":
        from seqmoe.assembly import load_trunk

        return load_trunk(path, map_location=map_location)

    def clone_and_install(self, logical_id: int, *, init: str = "clone") -> None:
        """Install one expert from the trunk MLP template (no-op if resident)."""
        self._ensure_left_pretrain()
        assert self._mlp_templates is not None
        for i, blk in enumerate(self.blocks):
            assert blk.bank is not None
            if logical_id in blk.bank.resident_ids():
                continue
            mlp = MLP(self.cfg.n_embd, self.cfg.mlp_mult, self.cfg.dropout)
            if init == "clone":
                mlp.load_state_dict(self._mlp_templates[i])
            elif init != "random":
                raise ValueError(f"expert_init must be clone|random, got {init}")
            blk.bank.install(logical_id, mlp)

    def set_mode(
        self,
        mode: str,
        expert_id: int | None = None,
        *,
        expert_init: str = "clone",
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode: {mode}")
        if mode == "expert" and expert_id is None:
            raise ValueError("set_mode('expert') requires expert_id")

        if mode == "pretrain":
            self._restore_pretrain()
            self.mode = "pretrain"
            self._expert_id = None
            self._apply_grad_policy()
            return

        self._ensure_left_pretrain()

        if mode == "expert":
            eid = int(expert_id)  # type: ignore[arg-type]
            for blk in self.blocks:
                assert blk.bank is not None
                for rid in list(blk.bank.resident_ids()):
                    if rid != eid:
                        blk.bank.uninstall(rid)
                blk.router = None
            self.clone_and_install(eid, init=expert_init)
            self._expert_id = eid
        else:
            # router / eval: keep whatever experts are installed; create routers.
            self._expert_id = None
            for blk in self.blocks:
                if blk.router is None:
                    blk.router = Router(self.cfg.n_embd, self.cfg.n_experts)

        self.mode = mode
        self._apply_grad_policy()

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        force_expert: int | None = None,
        route: str = "learned",
        hard_id: int | None = None,
        top_k: int | None = None,
        gold_id: torch.Tensor | None = None,
    ) -> ForwardResult:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"seq len {T} > block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        opts = _RouteOpts(
            force_expert=force_expert,
            route=route,
            hard_id=hard_id,
            top_k=top_k,
            gold_id=gold_id,
        )
        token_w_sum: torch.Tensor | None = None
        aux_acc: list[torch.Tensor] = []
        route_logits_acc: list[torch.Tensor] = []
        n_moe = 0
        for blk in self.blocks:
            x, stats = blk(x, opts)
            tw = stats.get("token_weights")
            if tw is not None:
                n_moe += 1
                token_w_sum = tw if token_w_sum is None else token_w_sum + tw
            if stats.get("aux") is not None:
                aux_acc.append(stats["aux"])
            if stats.get("logits") is not None:
                route_logits_acc.append(stats["logits"])
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        token_w = None
        route_frac = None
        if token_w_sum is not None and n_moe:
            token_w = token_w_sum / float(n_moe)
            route_frac = token_w.reshape(-1, self.cfg.n_experts).mean(dim=0)

        aux_loss = None
        if aux_acc:
            aux_loss = torch.stack(aux_acc).mean()

        route_ce = None
        if gold_id is not None and route_logits_acc and opts.route == "learned":
            gold = gold_id.to(idx.device)
            if gold.ndim == 0:
                gold = gold.expand(B)
            gold_tok = gold.long().view(B, 1).expand(B, T)
            route_ce = torch.stack(
                [
                    F.cross_entropy(
                        lg.reshape(-1, self.cfg.n_experts), gold_tok.reshape(-1)
                    )
                    for lg in route_logits_acc
                ]
            ).mean()

        gold_agree = None
        if gold_id is not None and token_w is not None:
            gold = gold_id.to(idx.device)
            if gold.ndim == 0:
                gold = gold.expand(B)
            top1 = token_w.argmax(dim=-1)
            gold_agree = (top1 == gold.long().view(B, 1)).to(token_w.dtype).mean()

        return ForwardResult(
            logits=logits,
            loss=loss,
            aux_loss=aux_loss,
            route_frac=route_frac,
            token_route_weights=token_w,
            gold_agree=gold_agree,
            route_ce=route_ce,
        )

    def resident_expert_ids(self) -> tuple[int, ...]:
        ids: set[int] = set()
        for blk in self.blocks:
            if blk.bank is not None:
                ids.update(blk.bank.resident_ids())
        return tuple(sorted(ids))

    def resident_expert_count(self) -> int:
        counts = [
            len(blk.bank.resident_ids()) for blk in self.blocks if blk.bank is not None
        ]
        return int(counts[0]) if counts else 0

    def _ensure_left_pretrain(self) -> None:
        if any(blk.mlp is not None for blk in self.blocks):
            self._snapshot_mlp_templates()
            for blk in self.blocks:
                blk.mlp = None
        for blk in self.blocks:
            if blk.bank is None:
                blk.bank = ExpertBank()

    def _snapshot_mlp_templates(self) -> None:
        if self._mlp_templates is not None:
            return
        tpls: list[dict[str, torch.Tensor]] = []
        for blk in self.blocks:
            if blk.mlp is None:
                raise RuntimeError("cannot snapshot trunk MLP; already dropped")
            tpls.append({k: v.detach().cpu().clone() for k, v in blk.mlp.state_dict().items()})
        self._mlp_templates = tpls

    def _restore_pretrain(self) -> None:
        for i, blk in enumerate(self.blocks):
            blk.bank = None
            blk.router = None
            if blk.mlp is None:
                mlp = MLP(self.cfg.n_embd, self.cfg.mlp_mult, self.cfg.dropout)
                if self._mlp_templates is not None:
                    mlp.load_state_dict(self._mlp_templates[i])
                blk.mlp = mlp
        self._mlp_templates = None

    def _apply_grad_policy(self) -> None:
        if self.mode == "pretrain":
            for p in self.parameters():
                p.requires_grad = True
            return
        if self.mode == "expert":
            eid = str(int(self._expert_id)) if self._expert_id is not None else None
            for n, p in self.named_parameters():
                p.requires_grad = eid is not None and f"bank.experts.{eid}." in n
            return
        if self.mode == "router":
            for n, p in self.named_parameters():
                p.requires_grad = ".router." in n
            return
        # eval
        for p in self.parameters():
            p.requires_grad = False
