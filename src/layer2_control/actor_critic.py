from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ActorCriticConfig:
    local_state_dim: int = 6
    graph_emb_dim: int = 64
    global_state_dim: int = 2
    action_dim: int = 2
    actor_hidden: tuple[int, int] = (128, 64)
    critic_hidden: tuple[int, int] = (256, 128)
    min_std: float = 1e-4


@dataclass(frozen=True)
class ActorOutput:
    mean: torch.Tensor
    std: torch.Tensor


@dataclass(frozen=True)
class CriticOutput:
    value: torch.Tensor


class ActorCritic(nn.Module):
    def __init__(self, config: ActorCriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or ActorCriticConfig()

        actor_in = self.config.graph_emb_dim + self.config.local_state_dim
        ah1, ah2 = self.config.actor_hidden
        self.actor_net = nn.Sequential(
            nn.Linear(actor_in, ah1),
            nn.ReLU(),
            nn.Linear(ah1, ah2),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(ah2, self.config.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.config.action_dim))

        critic_in = self.config.graph_emb_dim + self.config.global_state_dim
        ch1, ch2 = self.config.critic_hidden
        self.critic_net = nn.Sequential(
            nn.Linear(critic_in, ch1),
            nn.ReLU(),
            nn.Linear(ch1, ch2),
            nn.ReLU(),
            nn.Linear(ch2, 1),
        )

    @staticmethod
    def _to_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)

    def actor(self, node_embedding: np.ndarray | torch.Tensor, local_state: np.ndarray | torch.Tensor) -> ActorOutput:
        node_embedding_t = self._to_tensor(node_embedding)
        local_state_t = self._to_tensor(local_state)

        if node_embedding_t.ndim == 1:
            node_embedding_t = node_embedding_t.unsqueeze(0)
        if local_state_t.ndim == 1:
            local_state_t = local_state_t.unsqueeze(0)

        features = torch.cat([node_embedding_t, local_state_t], dim=-1)
        hidden = self.actor_net(features)
        mean = torch.tanh(self.actor_mean(hidden))
        std = F.softplus(self.log_std).unsqueeze(0).expand_as(mean) + self.config.min_std
        return ActorOutput(mean=mean, std=std)

    def critic(self, graph_embedding: np.ndarray | torch.Tensor, global_state: np.ndarray | torch.Tensor) -> CriticOutput:
        graph_embedding_t = self._to_tensor(graph_embedding)
        global_state_t = self._to_tensor(global_state)

        if graph_embedding_t.ndim == 1:
            graph_embedding_t = graph_embedding_t.unsqueeze(0)
        if global_state_t.ndim == 1:
            global_state_t = global_state_t.unsqueeze(0)

        features = torch.cat([graph_embedding_t, global_state_t], dim=-1)
        value = self.critic_net(features)
        return CriticOutput(value=value.squeeze(-1))

    def evaluate_actions(
        self,
        node_embedding: torch.Tensor,
        local_state: torch.Tensor,
        graph_embedding: torch.Tensor,
        global_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_out = self.actor(node_embedding=node_embedding, local_state=local_state)
        dist = torch.distributions.Normal(actor_out.mean, actor_out.std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(graph_embedding=graph_embedding, global_state=global_state).value
        return log_prob, entropy, value
