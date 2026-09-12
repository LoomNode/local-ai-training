"""Perception–action loop: observation bytes → brain → (caller action) → sandbox."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from flybrain.env.sandbox import Sandbox


@dataclass
class Perceive:
    logits: Tensor
    state: Tensor


@dataclass
class LoopResult:
    observation: str
    state: Tensor
    refused: bool
    action: str
    decoded: str


class TerminalLoop:
    """Same readout is the motor channel. No separate world-model head."""

    def __init__(self, model: nn.Module, sandbox: Sandbox, max_obs_bytes: int = 256) -> None:
        self.model = model
        self.sandbox = sandbox
        self.max_obs_bytes = max_obs_bytes

    def perceive(self, text: str, state: Tensor | None) -> Perceive:
        raw = list(text.encode("utf-8")[: self.max_obs_bytes]) or [ord("\n")]
        tokens = torch.tensor(raw, dtype=torch.long).unsqueeze(0)
        logits, new_state = self.model(tokens, state=state)
        return Perceive(logits=logits, state=new_state)

    def decode_action(self, logits: Tensor, n_bytes: int = 24) -> str:
        ids = logits[0, -1].argmax().view(1)
        out = [int(ids.item())]
        # one-shot greedy byte; V0 cannot type commands
        return bytes(out).decode("utf-8", errors="replace")

    def step(
        self,
        action: str,
        state: Tensor | None,
        observation: str,
    ) -> LoopResult:
        seen = self.perceive(observation, state)
        decoded = self.decode_action(seen.logits)
        obs = self.sandbox.step(action)
        return LoopResult(
            observation=obs.text,
            state=seen.state,
            refused=obs.refused,
            action=action,
            decoded=decoded,
        )
