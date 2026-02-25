from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.layer2_control.actor_critic import ActorCritic, ActorCriticConfig
from src.layer2_control.gat_encoder import GATEncoder, GATEncoderConfig, GraphObservation


@dataclass(frozen=True)
class MappoPolicyConfig:
    action_low: float = -1.0
    action_high: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256


class RolloutBuffer:
    def __init__(self) -> None:
        self.node_features: list[np.ndarray] = []
        self.adjacency: list[np.ndarray] = []
        self.agent_index: list[int] = []
        self.local_state: list[np.ndarray] = []
        self.global_state: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[float] = []

    def add(
        self,
        *,
        node_features: np.ndarray,
        adjacency: np.ndarray,
        agent_index: int,
        local_state: np.ndarray,
        global_state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.node_features.append(node_features.astype(np.float32))
        self.adjacency.append(adjacency.astype(np.float32))
        self.agent_index.append(int(agent_index))
        self.local_state.append(local_state.astype(np.float32))
        self.global_state.append(global_state.astype(np.float32))
        self.actions.append(action.astype(np.float32))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(float(done))

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        self.__init__()


class MappoPolicy(nn.Module):
    """Trainable MAPPO policy with shared GAT encoder and centralized critic."""

    def __init__(
        self,
        encoder: GATEncoder | None = None,
        actor_critic: ActorCritic | None = None,
        config: MappoPolicyConfig | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder or GATEncoder(GATEncoderConfig())
        self.actor_critic = actor_critic or ActorCritic(ActorCriticConfig())
        self.config = config or MappoPolicyConfig()

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor_critic.parameters()),
            lr=self.config.learning_rate,
        )

    @staticmethod
    def _to_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return torch.nan_to_num(x.float(), nan=0.0, posinf=1e6, neginf=-1e6)
        arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.tensor(arr, dtype=torch.float32)

    def _encode(self, node_features: np.ndarray, adjacency: np.ndarray) -> torch.Tensor:
        obs = GraphObservation(node_features=node_features, adjacency=adjacency)
        return self.encoder.encode(obs)

    def act(self, obs: dict[str, Any]) -> tuple[np.ndarray, float, float]:
        node_features = np.asarray(obs["node_features"], dtype=np.float32)
        adjacency = np.asarray(obs["adjacency"], dtype=np.float32)
        local_state = np.asarray(obs["local_state"], dtype=np.float32)
        global_state = np.asarray(obs["global_state"], dtype=np.float32)
        agent_index = int(obs["agent_index"])

        embeddings = self._encode(node_features, adjacency)
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
        graph_embedding = embeddings.mean(dim=0)
        node_embedding = embeddings[agent_index]

        actor_out = self.actor_critic.actor(node_embedding=node_embedding, local_state=local_state)
        mean = torch.nan_to_num(actor_out.mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std = torch.nan_to_num(actor_out.std, nan=1.0, posinf=10.0, neginf=1e-4).clamp_min(1e-4)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        value = self.actor_critic.critic(graph_embedding=graph_embedding, global_state=global_state).value
        clipped_action = torch.clamp(action, self.config.action_low, self.config.action_high)
        return (
            clipped_action.squeeze(0).detach().cpu().numpy(),
            float(log_prob.item()),
            float(value.squeeze(0).item()),
        )

    @torch.no_grad()
    def act_deterministic(self, obs: dict[str, Any]) -> np.ndarray:
        node_features = np.asarray(obs["node_features"], dtype=np.float32)
        adjacency = np.asarray(obs["adjacency"], dtype=np.float32)
        local_state = np.asarray(obs["local_state"], dtype=np.float32)
        agent_index = int(obs["agent_index"])

        embeddings = self._encode(node_features, adjacency)
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
        node_embedding = embeddings[agent_index]
        actor_out = self.actor_critic.actor(node_embedding=node_embedding, local_state=local_state)
        mean = torch.nan_to_num(actor_out.mean, nan=0.0, posinf=1.0, neginf=-1.0)
        action = torch.clamp(mean, self.config.action_low, self.config.action_high)
        return action.squeeze(0).detach().cpu().numpy()

    def evaluate_value(self, obs: dict[str, Any]) -> float:
        node_features = np.asarray(obs["node_features"], dtype=np.float32)
        adjacency = np.asarray(obs["adjacency"], dtype=np.float32)
        global_state = np.asarray(obs["global_state"], dtype=np.float32)

        embeddings = self._encode(node_features, adjacency)
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
        graph_embedding = embeddings.mean(dim=0)
        value = self.actor_critic.critic(graph_embedding=graph_embedding, global_state=global_state).value
        return float(value.squeeze(0).item())

    def compute_gae(self, buffer: RolloutBuffer, last_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        values = np.asarray(buffer.values + [last_value], dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def _build_batch_tensors(
        self,
        buffer: RolloutBuffer,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        n = len(buffer)
        node_features = torch.tensor(np.stack(buffer.node_features), dtype=torch.float32)
        adjacency = torch.tensor(np.stack(buffer.adjacency), dtype=torch.float32)
        agent_index = torch.tensor(np.array(buffer.agent_index), dtype=torch.long)
        local_state = torch.tensor(np.stack(buffer.local_state), dtype=torch.float32)
        global_state = torch.tensor(np.stack(buffer.global_state), dtype=torch.float32)
        actions = torch.tensor(np.stack(buffer.actions), dtype=torch.float32)
        old_log_probs = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32)
        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        return {
            "node_features": node_features.view(n, *node_features.shape[1:]),
            "adjacency": adjacency.view(n, *adjacency.shape[1:]),
            "agent_index": agent_index,
            "local_state": local_state,
            "global_state": global_state,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "advantages": advantages_t,
            "returns": returns_t,
        }

    def _evaluate_batch(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        agent_index: torch.Tensor,
        local_state: torch.Tensor,
        global_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = node_features.shape[0]
        all_log_probs: list[torch.Tensor] = []
        all_entropy: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        for i in range(batch_size):
            embeddings = self.encoder.encode(
                GraphObservation(node_features=node_features[i], adjacency=adjacency[i])
            )
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
            node_embedding = embeddings[agent_index[i]]
            graph_embedding = embeddings.mean(dim=0)

            actor_out = self.actor_critic.actor(node_embedding=node_embedding, local_state=local_state[i])
            mean = torch.nan_to_num(actor_out.mean, nan=0.0, posinf=1.0, neginf=-1.0)
            std = torch.nan_to_num(actor_out.std, nan=1.0, posinf=10.0, neginf=1e-4).clamp_min(1e-4)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(actions[i].unsqueeze(0)).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            value = self.actor_critic.critic(graph_embedding=graph_embedding, global_state=global_state[i]).value

            all_log_probs.append(log_prob.squeeze(0))
            all_entropy.append(entropy.squeeze(0))
            all_values.append(value.squeeze(0))

        return torch.stack(all_log_probs), torch.stack(all_entropy), torch.stack(all_values)

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> dict[str, float]:
        if len(buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        advantages, returns = self.compute_gae(buffer, last_value=last_value)
        batch = self._build_batch_tensors(buffer, advantages, returns)

        n = len(buffer)
        minibatch = min(self.config.minibatch_size, n)

        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        total_loss_total = 0.0
        updates = 0

        for _ in range(self.config.update_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, minibatch):
                idx = indices[start : start + minibatch]
                idx_t = torch.tensor(idx, dtype=torch.long)

                log_probs, entropy, values = self._evaluate_batch(
                    node_features=batch["node_features"][idx_t],
                    adjacency=batch["adjacency"][idx_t],
                    agent_index=batch["agent_index"][idx_t],
                    local_state=batch["local_state"][idx_t],
                    global_state=batch["global_state"][idx_t],
                    actions=batch["actions"][idx_t],
                )

                old_log_probs = batch["old_log_probs"][idx_t]
                adv = batch["advantages"][idx_t]
                ret = batch["returns"][idx_t]

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = torch.mean((ret - values) ** 2)
                entropy_bonus = entropy.mean()

                total_loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.actor_critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                policy_loss_total += float(policy_loss.item())
                value_loss_total += float(value_loss.item())
                entropy_total += float(entropy_bonus.item())
                total_loss_total += float(total_loss.item())
                updates += 1

        if updates == 0:
            updates = 1

        return {
            "policy_loss": policy_loss_total / updates,
            "value_loss": value_loss_total / updates,
            "entropy": entropy_total / updates,
            "total_loss": total_loss_total / updates,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder_state": self.encoder.state_dict(),
                "actor_critic_state": self.actor_critic.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config.__dict__,
            },
            checkpoint_path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location="cpu")
        self.encoder.load_state_dict(checkpoint["encoder_state"])
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
