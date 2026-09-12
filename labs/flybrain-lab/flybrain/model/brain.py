"""Sparse fly-derived language model: 4-d neurons, scalar synapses, BPTT."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from flybrain.data.connectome import (
    CELL_GENERIC,
    N_CELL_TYPES,
    REGION_CORE,
    REGION_DOPAMINE,
    REGION_KENYON,
    REGION_READOUT,
    REGION_SENSORY,
    Connectome,
)

# scalar: emit→1 channel. vector2/vector: w⊙ prefix/full state, one p per edge.
# projected: per-channel p, m_ij = w_ij ⊙ h_i, still sparse (not N×N).
MESSAGE_KINDS = ("scalar", "vector2", "vector", "projected")
CELL_KINDS = ("type", "unique", "universal")
DECODE_POPS = ("readout", "sensory")
SKIP_KINDS = ("none", "sensory_readout")
UPDATE_KINDS = ("parallel", "ordered")
READOUT_KINDS = ("flatten", "mean", "emit")
INJECT_KINDS = ("embed", "local")
ORDERED_REGIONS = (
    REGION_SENSORY,
    REGION_KENYON,
    REGION_CORE,
    REGION_DOPAMINE,
    REGION_READOUT,
)


def message_dim(kind: str, state_dim: int) -> int:
    if kind == "scalar":
        return 1
    if kind == "vector2":
        return 2
    if kind in ("vector", "projected"):
        return state_dim
    raise ValueError(f"unknown message {kind}")


class FlyLanguageModel(nn.Module):
    """Byte in → sensory population → recurrent graph → readout → 256 logits."""

    def __init__(
        self,
        connectome: Connectome,
        state_dim: int = 4,
        vocab_size: int = 256,
        readout: str = "flatten",
        message: str = "scalar",
        capacity: int | None = None,
        cell: str = "type",
        decode_pop: str = "readout",
        skip: str = "none",
        update: str = "parallel",
        inject: str = "embed",
        kenyon_top_k: int | None = None,
        kenyon_theta: float = 0.0,
        n_out: int | None = None,
    ) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        if readout not in READOUT_KINDS:
            raise ValueError(f"unknown readout {readout}")
        if message not in MESSAGE_KINDS:
            raise ValueError(f"unknown message {message}")
        if message == "vector2" and state_dim < 2:
            raise ValueError("vector2 needs state_dim >= 2")
        if cell not in CELL_KINDS:
            raise ValueError(f"unknown cell {cell}")
        if decode_pop not in DECODE_POPS:
            raise ValueError(f"unknown decode_pop {decode_pop}")
        if skip not in SKIP_KINDS:
            raise ValueError(f"unknown skip {skip}")
        if update not in UPDATE_KINDS:
            raise ValueError(f"unknown update {update}")
        if inject not in INJECT_KINDS:
            raise ValueError(f"unknown inject {inject}")
        cap = int(connectome.n if capacity is None else capacity)
        if cap < connectome.n:
            raise ValueError("capacity must be >= connectome.n")
        self.connectome = connectome
        self.capacity = cap
        self.state_dim = state_dim
        self.vocab_size = vocab_size
        self.readout = readout
        self.message = message
        self.cell = cell
        self.decode_pop = decode_pop
        self.skip = skip
        self.update = update
        self.inject = inject
        self.kenyon_top_k = None if kenyon_top_k is None else int(kenyon_top_k)
        self.kenyon_theta = float(kenyon_theta)
        self.msg_dim = message_dim(message, state_dim)
        n_s = int(connectome.sensory_idx.numel())
        n_r = int(connectome.readout_idx.numel())
        n_decode = n_s if decode_pop == "sensory" else n_r
        out_dim = int(vocab_size if n_out is None else n_out)
        self.n_out = out_dim
        self.encoder = nn.Embedding(vocab_size, n_s)
        if inject == "local":
            with torch.no_grad():
                self.encoder.weight.zero_()
                eye = min(n_s, vocab_size)
                self.encoder.weight[:eye, :eye] = torch.eye(eye)
            self.encoder.weight.requires_grad = False
        if readout == "mean":
            in_features = state_dim
        elif readout == "emit":
            in_features = n_decode
        else:
            in_features = n_decode * state_dim
        self.decoder = nn.Linear(in_features, out_dim)
        if cell in ("type", "universal"):
            n_prog = 1 if cell == "universal" else N_CELL_TYPES
            self.cells = nn.ModuleList(
                [nn.GRUCell(self.msg_dim, state_dim) for _ in range(n_prog)]
            )
            self.register_parameter("unique_weight_ih", None)
            self.register_parameter("unique_weight_hh", None)
            self.register_parameter("unique_bias_ih", None)
            self.register_parameter("unique_bias_hh", None)
            self.register_parameter("unique_emit", None)
            if message == "scalar":
                self.emit = nn.Parameter(0.1 * torch.randn(n_prog, state_dim))
            else:
                self.register_parameter("emit", None)
        else:
            self.cells = nn.ModuleList()
            gates = 3 * state_dim
            std = 1.0 / (state_dim ** 0.5)
            self.unique_weight_ih = nn.Parameter(torch.empty(cap, gates, self.msg_dim))
            self.unique_weight_hh = nn.Parameter(torch.empty(cap, gates, state_dim))
            self.unique_bias_ih = nn.Parameter(torch.empty(cap, gates))
            self.unique_bias_hh = nn.Parameter(torch.empty(cap, gates))
            nn.init.uniform_(self.unique_weight_ih, -std, std)
            nn.init.uniform_(self.unique_weight_hh, -std, std)
            nn.init.uniform_(self.unique_bias_ih, -std, std)
            nn.init.uniform_(self.unique_bias_hh, -std, std)
            self.register_parameter("emit", None)
            if message == "scalar":
                self.unique_emit = nn.Parameter(0.1 * torch.randn(cap, state_dim))
            else:
                self.register_parameter("unique_emit", None)
        if message == "scalar":
            self.register_parameter("sensory_mix", None)
        else:
            self.sensory_mix = nn.Parameter(torch.ones(self.msg_dim) / self.msg_dim)
        target = torch.log1p(connectome.contact_count.float()).clamp(min=1e-4)
        # softplus(p) ≈ target  ⇒  p = log(exp(target) - 1)
        p = torch.log(torch.expm1(target))
        if message == "projected":
            p = p.unsqueeze(-1).expand(-1, self.msg_dim).clone()
        self.synapse_param = nn.Parameter(p)
        self.register_buffer("src", connectome.src.long())
        self.register_buffer("dst", connectome.dst.long())
        self.register_buffer("sign", connectome.sign.float())
        sk_mask = (connectome.region[connectome.src] == REGION_SENSORY) & (
            connectome.region[connectome.dst] == REGION_KENYON
        )
        self.register_buffer("sk_mask", sk_mask)
        self.freeze_sk = inject == "local"
        cell_type = connectome.cell_type.long()
        if cap > connectome.n:
            extra = torch.full((cap - connectome.n,), CELL_GENERIC, dtype=torch.long)
            cell_type = torch.cat([cell_type, extra])
        self.register_buffer("cell_type", cell_type)
        region = connectome.region.long()
        if cap > connectome.n:
            extra_r = torch.full((cap - connectome.n,), REGION_CORE, dtype=torch.long)
            region = torch.cat([region, extra_r])
        self.register_buffer("region", region)
        live = torch.zeros(cap, dtype=torch.bool)
        live[: connectome.n] = True
        self.register_buffer("live_mask", live)
        self.register_buffer("sensory_idx", connectome.sensory_idx.long())
        self.register_buffer("readout_idx", connectome.readout_idx.long())

    @property
    def n_synapses(self) -> int:
        return int(self.src.numel())

    def synapse_weights(self) -> Tensor:
        mag = F.softplus(self.synapse_param)
        if mag.ndim == 2:
            w = self.sign.unsqueeze(-1) * mag
        else:
            w = self.sign * mag
        if self.freeze_sk:
            frozen = self.sign * torch.ones_like(self.sign)
            if w.ndim == 2:
                frozen = frozen.unsqueeze(-1).expand_as(w)
            mask = self.sk_mask
            if w.ndim == 2:
                mask = mask.unsqueeze(-1)
            w = torch.where(mask, frozen, w)
        return w

    @property
    def n_live(self) -> int:
        return int(self.live_mask.sum().item())

    def activate_slot(self, idx: int) -> None:
        if not 0 <= idx < self.capacity:
            raise IndexError(idx)
        self.live_mask[idx] = True

    def deactivate_slot(self, idx: int) -> None:
        if not 0 <= idx < self.capacity:
            raise IndexError(idx)
        self.live_mask[idx] = False

    def persistent_state_scalars(self, batch: int = 1) -> int:
        return int(batch * self.capacity * self.state_dim)

    def flops_per_step(self, batch: int = 1) -> int:
        n, e, d = self.connectome.n, self.n_synapses, self.state_dim
        cell_in = self.msg_dim
        msg = e * (d + 2) if self.message == "scalar" else e * (2 * cell_in)
        passes = len(ORDERED_REGIONS) if self.update == "ordered" else 1
        return int(batch * passes * (msg + n * 3 * ((cell_in + d) * d + d)))

    def reset_state(self) -> None:
        """H is not θ. Clearing thought-state must not touch learned parameters."""
        return None

    def _unique_gru(self, x: Tensor, h: Tensor) -> Tensor:
        """Batched per-neuron GRU; x is [B, N, I], h is [B, N, H]."""
        gi = torch.einsum("bni,noi->bno", x, self.unique_weight_ih) + self.unique_bias_ih
        gh = torch.einsum("bnh,noh->bno", h, self.unique_weight_hh) + self.unique_bias_hh
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h

    def _align_to_pop(self, values: Tensor, n_dst: int) -> Tensor:
        n_src = int(values.shape[1])
        if n_src == n_dst:
            return values
        if n_src > n_dst:
            return values[:, :n_dst]
        pad_n = n_dst - n_src
        if values.ndim == 2:
            return F.pad(values, (0, pad_n))
        return F.pad(values, (0, 0, 0, pad_n))

    def _inject_readout_skip(self, incoming: Tensor, payload: Tensor) -> None:
        if self.skip != "sensory_readout":
            return
        aligned = self._align_to_pop(payload, int(self.readout_idx.numel()))
        batch = incoming.size(0)
        if incoming.ndim == 2:
            index = self.readout_idx.unsqueeze(0).expand(batch, -1)
            incoming.scatter_add_(1, index, aligned)
            return
        cell_in = incoming.size(-1)
        index = self.readout_idx.view(1, -1, 1).expand(batch, -1, cell_in)
        incoming.scatter_add_(1, index, aligned)

    def _incoming(
        self,
        state: Tensor,
        token: Tensor,
        weights: Tensor,
        *,
        inject_sensory: bool,
        inject_skip: bool,
    ) -> Tensor:
        batch, n, _dim = state.shape
        cell_in = self.msg_dim
        src_live = self.live_mask[self.src].to(dtype=state.dtype)
        sensory = self.encoder(token)
        if self.message == "scalar":
            if self.cell == "unique":
                emit_w = self.unique_emit
            elif self.cell == "universal":
                emit_w = self.emit.expand(n, -1)
            else:
                emit_w = self.emit[self.cell_type]
            if self.inject == "local":
                scalar = state[:, :, 0]
            else:
                scalar = (state * emit_w.unsqueeze(0)).sum(-1)
            incoming = torch.zeros(batch, n, device=state.device, dtype=state.dtype)
            if self.inject == "local" and bool(self.sk_mask.any()):
                nsk = ~self.sk_mask
                if bool(nsk.any()):
                    messages = (
                        scalar[:, self.src[nsk]]
                        * weights[nsk].unsqueeze(0)
                        * src_live[nsk].unsqueeze(0)
                    )
                    incoming.scatter_add_(
                        1, self.dst[nsk].unsqueeze(0).expand(batch, -1), messages
                    )
                u_src, u_dst, u_w = self._unique_sk_pairs()
                if u_src.numel() > 0:
                    sk_msg = scalar[:, u_src] * u_w.unsqueeze(0)
                    incoming.scatter_add_(
                        1, u_dst.unsqueeze(0).expand(batch, -1), sk_msg
                    )
            else:
                messages = scalar[:, self.src] * weights.unsqueeze(0) * src_live.unsqueeze(0)
                incoming.scatter_add_(1, self.dst.unsqueeze(0).expand(batch, -1), messages)
            if inject_sensory:
                incoming.scatter_add_(
                    1, self.sensory_idx.unsqueeze(0).expand(batch, -1), sensory
                )
            if inject_skip:
                self._inject_readout_skip(incoming, sensory)
            return incoming
        src_state = state[:, self.src, :cell_in]
        if weights.ndim == 1:
            messages = src_state * weights.view(1, -1, 1)
        else:
            messages = src_state * weights.unsqueeze(0)
        messages = messages * src_live.view(1, -1, 1)
        incoming = torch.zeros(batch, n, cell_in, device=state.device, dtype=state.dtype)
        index = self.dst.view(1, -1, 1).expand(batch, -1, cell_in)
        incoming.scatter_add_(1, index, messages)
        if inject_sensory:
            mix = self.sensory_mix.view(1, 1, cell_in)
            sensory_vec = sensory.unsqueeze(-1) * mix
            s_index = self.sensory_idx.view(1, -1, 1).expand(batch, -1, cell_in)
            incoming.scatter_add_(1, s_index, sensory_vec)
            if inject_skip:
                self._inject_readout_skip(incoming, sensory_vec)
        elif inject_skip:
            mix = self.sensory_mix.view(1, 1, cell_in)
            sensory_vec = sensory.unsqueeze(-1) * mix
            self._inject_readout_skip(incoming, sensory_vec)
        return incoming

    def _apply_update(self, state: Tensor, incoming: Tensor, mask: Tensor) -> Tensor:
        batch, n, dim = state.shape
        cell_in = self.msg_dim
        if incoming.ndim == 2:
            inp = incoming.unsqueeze(-1)
        else:
            inp = incoming
        if self.cell == "unique":
            updated = self._unique_gru(inp, state)
            return torch.where(mask.view(1, n, 1), updated, state)
        new_state = state.clone()
        n_sel = int(mask.sum())
        if n_sel == 0:
            return new_state
        if self.cell == "universal":
            hidden = state[:, mask, :].reshape(batch * n_sel, dim)
            x = incoming[:, mask].reshape(batch * n_sel, cell_in)
            new_state[:, mask, :] = self.cells[0](x, hidden).view(batch, n_sel, dim)
            return new_state
        for type_id, cell in enumerate(self.cells):
            tmask = (self.cell_type == type_id) & mask
            n_c = int(tmask.sum())
            if n_c == 0:
                continue
            hidden = state[:, tmask, :].reshape(batch * n_c, dim)
            x = incoming[:, tmask].reshape(batch * n_c, cell_in)
            new_state[:, tmask, :] = cell(x, hidden).view(batch, n_c, dim)
        return new_state

    def _unique_sk_pairs(self) -> tuple[Tensor, Tensor, Tensor]:
        """One (PN, Kenyon) claw: duplicate shuffle edges cannot 3x-count one PN."""
        sk = self.sk_mask
        if not bool(sk.any()):
            empty = self.src.new_empty(0)
            return empty, empty, self.sign.new_empty(0)
        src = self.src[sk]
        dst = self.dst[sk]
        sgn = self.sign[sk]
        key = src * int(self.capacity) + dst
        uniq, inv = torch.unique(key, return_inverse=True)
        first = torch.full((int(uniq.numel()),), src.numel(), dtype=torch.long, device=src.device)
        first.scatter_reduce_(0, inv, torch.arange(src.numel(), device=src.device), reduce="amin")
        return src[first], dst[first], sgn[first]

    def _write_sensory(self, state: Tensor, incoming: Tensor) -> Tensor:
        """Glomeruli relay the odor pattern; they are not language GRUs."""
        sidx = self.sensory_idx
        val = incoming[:, sidx] if incoming.ndim == 2 else incoming[:, sidx].sum(dim=-1)
        state = state.clone()
        state[:, sidx, :] = 0
        state[:, sidx, 0] = val
        return state

    def _write_kenyon(self, state: Tensor, incoming: Tensor) -> Tensor:
        """Kenyon cells are coincidence units, not language GRUs."""
        kidx = ((self.region == REGION_KENYON) & self.live_mask).nonzero(as_tuple=False).view(-1)
        val = incoming[:, kidx] if incoming.ndim == 2 else incoming[:, kidx].sum(dim=-1)
        state = state.clone()
        state[:, kidx, :] = 0
        state[:, kidx, 0] = torch.tanh(val)
        return state

    def _write_linear_pop(self, state: Tensor, incoming: Tensor, idx: Tensor) -> Tensor:
        val = incoming[:, idx] if incoming.ndim == 2 else incoming[:, idx].sum(dim=-1)
        state = state.clone()
        state[:, idx, :] = 0
        state[:, idx, 0] = torch.tanh(val)
        return state

    def _gate_kenyon_incoming(self, incoming: Tensor) -> Tensor:
        if self.kenyon_theta <= 0:
            return incoming
        kmask = (self.region == REGION_KENYON) & self.live_mask
        if incoming.ndim == 2:
            mag = incoming.abs()
            fire = mag > self.kenyon_theta
            gated = incoming * fire.to(incoming.dtype)
            return torch.where(kmask.unsqueeze(0), gated, incoming)
        mag = incoming.abs().sum(dim=-1)
        fire = mag > self.kenyon_theta
        gated = incoming * fire.unsqueeze(-1).to(incoming.dtype)
        return torch.where(kmask.view(1, -1, 1), gated, incoming)

    def _sparsify_kenyon(self, state: Tensor) -> Tensor:
        if self.kenyon_top_k is None or self.kenyon_top_k <= 0:
            return state
        idx = ((self.region == REGION_KENYON) & self.live_mask).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return state
        h = state[:, idx, :]
        score = h.abs().sum(dim=-1)
        k = min(self.kenyon_top_k, int(idx.numel()))
        winners = score.topk(k, dim=1).indices
        keep = torch.zeros_like(score, dtype=h.dtype)
        keep.scatter_(1, winners, 1.0)
        state = state.clone()
        state[:, idx, :] = h * keep.unsqueeze(-1)
        return state

    def step(self, state: Tensor, token: Tensor, weights: Tensor | None = None) -> Tensor:
        if weights is None:
            weights = self.synapse_weights()
        if self.update == "ordered":
            for region_id in ORDERED_REGIONS:
                incoming = self._incoming(
                    state,
                    token,
                    weights,
                    inject_sensory=region_id == REGION_SENSORY,
                    inject_skip=self.skip != "none" and region_id == REGION_READOUT,
                )
                if region_id == REGION_SENSORY and self.inject == "local":
                    state = self._write_sensory(state, incoming)
                    continue
                if region_id == REGION_KENYON:
                    incoming = self._gate_kenyon_incoming(incoming)
                    if self.inject == "local":
                        state = self._write_kenyon(state, incoming)
                        state = self._sparsify_kenyon(state)
                        continue
                mask = (self.region == region_id) & self.live_mask
                state = self._apply_update(state, incoming, mask)
                if region_id == REGION_KENYON:
                    state = self._sparsify_kenyon(state)
            return state
        incoming = self._incoming(
            state, token, weights, inject_sensory=True, inject_skip=True
        )
        incoming = self._gate_kenyon_incoming(incoming)
        state = self._apply_update(state, incoming, self.live_mask)
        return self._sparsify_kenyon(state)

    def _add_injects(self, incoming: Tensor, token: Tensor) -> Tensor:
        batch = incoming.size(0)
        sensory = self.encoder(token)
        if incoming.ndim == 2:
            incoming.scatter_add_(
                1, self.sensory_idx.unsqueeze(0).expand(batch, -1), sensory
            )
            return incoming
        cell_in = incoming.size(-1)
        mix = self.sensory_mix.view(1, 1, cell_in)
        sensory_vec = sensory.unsqueeze(-1) * mix
        s_index = self.sensory_idx.view(1, -1, 1).expand(batch, -1, cell_in)
        incoming.scatter_add_(1, s_index, sensory_vec)
        return incoming

    def step_odor(self, state: Tensor, bag: Tensor, weights: Tensor | None = None) -> Tensor:
        """One scent: inject every token in the bag, then one ordered graph pass."""
        if weights is None:
            weights = self.synapse_weights()
        incoming = self._incoming(
            state, bag[:, 0], weights, inject_sensory=True, inject_skip=False
        )
        for t in range(1, bag.size(1)):
            incoming = self._add_injects(incoming, bag[:, t])
        if self.update != "ordered":
            incoming = self._gate_kenyon_incoming(incoming)
            state = self._apply_update(state, incoming, self.live_mask)
            return self._sparsify_kenyon(state)
        for region_id in ORDERED_REGIONS:
            if self.inject == "local" and region_id in (REGION_CORE, REGION_DOPAMINE):
                continue
            if region_id == REGION_SENSORY:
                inc = incoming
                if self.inject == "local":
                    state = self._write_sensory(state, inc)
                    continue
            else:
                inc = self._incoming(
                    state,
                    bag[:, 0],
                    weights,
                    inject_sensory=False,
                    inject_skip=self.skip != "none" and region_id == REGION_READOUT,
                )
                if region_id == REGION_KENYON:
                    inc = self._gate_kenyon_incoming(inc)
                    if self.inject == "local":
                        state = self._write_kenyon(state, inc)
                        state = self._sparsify_kenyon(state)
                        continue
                if region_id == REGION_READOUT and self.inject == "local":
                    inc = self._incoming(
                        state,
                        bag[:, 0],
                        weights,
                        inject_sensory=False,
                        inject_skip=False,
                    )
                    state = self._write_linear_pop(state, inc, self.readout_idx)
                    continue
            mask = (self.region == region_id) & self.live_mask
            state = self._apply_update(state, inc, mask)
            if region_id == REGION_KENYON:
                state = self._sparsify_kenyon(state)
        return state

    def forward_odor(
        self, bag: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        batch = bag.size(0)
        n = self.capacity
        if state is None:
            state = bag.new_zeros(
                batch, n, self.state_dim, dtype=self.decoder.weight.dtype
            )
        state = self.step_odor(state, bag)
        idx = self.sensory_idx if self.decode_pop == "sensory" else self.readout_idx
        pop = state[:, idx, :]
        if self.readout == "mean":
            features = pop.mean(dim=1)
        elif self.readout == "emit":
            features = pop.mean(dim=-1)
        else:
            features = pop.reshape(batch, -1)
        logits = self.decoder(features).unsqueeze(1)
        return logits, state

    def forward(
        self,
        tokens: Tensor,
        state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, seq_len = tokens.shape
        n = self.capacity
        if state is None:
            state = tokens.new_zeros(
                batch, n, self.state_dim, dtype=self.decoder.weight.dtype
            )
        weights = self.synapse_weights()
        logits = []
        for t in range(seq_len):
            state = self.step(state, tokens[:, t], weights)
            idx = self.sensory_idx if self.decode_pop == "sensory" else self.readout_idx
            pop = state[:, idx, :]
            if self.readout == "mean":
                features = pop.mean(dim=1)
            elif self.readout == "emit":
                features = pop.mean(dim=-1)
            else:
                features = pop.reshape(batch, -1)
            logits.append(self.decoder(features))
        return torch.stack(logits, dim=1), state
