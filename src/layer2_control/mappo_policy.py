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
        self.embeddings: list[np.ndarray] = []
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
        embeddings: np.ndarray,
        agent_index: int,
        local_state: np.ndarray,
        global_state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.embeddings.append(embeddings.astype(np.float32))
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

    def encode(self, node_features: np.ndarray, adjacency: np.ndarray) -> torch.Tensor:
        embeddings = self._encode(node_features, adjacency)
        return torch.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)

    def act(self, obs: dict[str, Any]) -> tuple[np.ndarray, float, float]:
        node_features = np.asarray(obs["node_features"], dtype=np.float32)
        adjacency = np.asarray(obs["adjacency"], dtype=np.float32)
        local_state = np.asarray(obs["local_state"], dtype=np.float32)
        global_state = np.asarray(obs["global_state"], dtype=np.float32)
        agent_index = int(obs["agent_index"])

        embeddings = self.encode(node_features, adjacency)
        return self.act_with_embeddings(embeddings, local_state, global_state, agent_index)

    def act_with_embeddings(
        self,
        embeddings: torch.Tensor,
        local_state: np.ndarray,
        global_state: np.ndarray,
        agent_index: int,
    ) -> tuple[np.ndarray, float, float]:
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

    def act_batch(
        self,
        embeddings: np.ndarray | torch.Tensor,
        local_states: np.ndarray | torch.Tensor,
        global_states: np.ndarray | torch.Tensor,
        agent_indices: np.ndarray | torch.Tensor,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        emb = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
        lst = torch.as_tensor(local_states, dtype=torch.float32, device=device)
        gst = torch.as_tensor(global_states, dtype=torch.float32, device=device)

        actor_out = self.actor_critic.actor(node_embedding=emb, local_state=lst)
        mean = torch.nan_to_num(actor_out.mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std = torch.nan_to_num(actor_out.std, nan=1.0, posinf=10.0, neginf=1e-4).clamp_min(1e-4)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        value = self.actor_critic.critic(graph_embedding=emb, global_state=gst).value
        clipped_action = torch.clamp(action, self.config.action_low, self.config.action_high)
        return (
            clipped_action.detach().cpu().numpy(),
            log_prob.detach().cpu(),
            value.detach().cpu(),
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
        embeddings = torch.tensor(np.stack(buffer.embeddings), dtype=torch.float32)
        agent_index = torch.tensor(np.array(buffer.agent_index), dtype=torch.long)
        local_state = torch.tensor(np.stack(buffer.local_state), dtype=torch.float32)
        global_state = torch.tensor(np.stack(buffer.global_state), dtype=torch.float32)
        actions = torch.tensor(np.stack(buffer.actions), dtype=torch.float32)
        old_log_probs = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32)
        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        return {
            "embeddings": embeddings,
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
        embeddings: torch.Tensor,
        agent_index: torch.Tensor,
        local_state: torch.Tensor,
        global_state: torch.Tensor,
        actions: torch.Tensor,
        *,
        profile: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        import time

        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
        graph_embedding = embeddings

        t_eval = 0.0
        t_val = 0.0

        if profile:
            t1 = time.time()
        actor_out = self.actor_critic.actor(node_embedding=embeddings, local_state=local_state)
        mean = torch.nan_to_num(actor_out.mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std = torch.nan_to_num(actor_out.std, nan=1.0, posinf=10.0, neginf=1e-4).clamp_min(1e-4)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        if profile:
            t_eval = time.time() - t1
            t2 = time.time()

        value = self.actor_critic.critic(graph_embedding=graph_embedding, global_state=global_state).value
        if profile:
            t_val = time.time() - t2

        return log_prob, entropy, value, t_eval, t_val

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> dict[str, float]:
        import time

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
        device = next(self.parameters()).device

        total_iters = 0
        t_compute = 0.0
        update_start = time.time()
        printed_minibatch = False

        for _ in range(self.config.update_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, minibatch):
                idx = indices[start : start + minibatch]
                idx_t = torch.tensor(idx, dtype=torch.long)

                t0 = time.time()
                embeddings = batch["embeddings"][idx_t].to(device)
                agent_index = batch["agent_index"][idx_t].to(device)
                local_state = batch["local_state"][idx_t].to(device)
                global_state = batch["global_state"][idx_t].to(device)
                actions = batch["actions"][idx_t].to(device)
                old_log_probs = batch["old_log_probs"][idx_t].to(device)
                adv = batch["advantages"][idx_t].to(device)
                ret = batch["returns"][idx_t].to(device)
                t_transfer = time.time() - t0

                if not printed_minibatch:
                    print(
                        f"[MINIBATCH] yielding batch of size {len(idx)}, "
                        f"embeddings.shape={tuple(embeddings.shape)}"
                    )

                profile = not printed_minibatch
                log_probs, entropy, values, t_eval, t_val = self._evaluate_batch(
                    embeddings=embeddings,
                    agent_index=agent_index,
                    local_state=local_state,
                    global_state=global_state,
                    actions=actions,
                    profile=profile,
                )

                t2 = time.time()
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
                t_ppo = time.time() - t2

                t3 = time.time()
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.actor_critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                t_back = time.time() - t3

                t_compute += time.time() - t0
                total_iters += 1

                if profile:
                    print(f"  [LOSS] data transfer: {t_transfer:.4f}s")
                    print(f"  [LOSS] evaluate_actions: {t_eval:.4f}s")
                    print(f"  [LOSS] get_values: {t_val:.4f}s")
                    print(f"  [LOSS] ppo_loss: {t_ppo:.4f}s")
                    print(f"  [LOSS] backward+step: {t_back:.4f}s")
                    printed_minibatch = True

                policy_loss_total += float(policy_loss.item())
                value_loss_total += float(value_loss.item())
                entropy_total += float(entropy_bonus.item())
                total_loss_total += float(total_loss.item())
                updates += 1

        if updates == 0:
            updates = 1

        total_time = time.time() - update_start
        print(f"[UPDATE] total_iters={total_iters}")
        print(f"[UPDATE] t_compute={t_compute:.3f}s")
        print(f"[UPDATE] t_overhead (total-compute)={total_time - t_compute:.3f}s")

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
