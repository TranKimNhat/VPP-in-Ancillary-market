"""Multi-Agent Twin Delayed DDPG (MATD3) baseline.

Based on: Li & Zhou (2025), "A Robust Large-Scale Multiagent Deep Reinforcement
Learning Method for Coordinated Automatic Generation Control," IEEE/CAA JAS.

Key features:
- CTDE framework: decentralized actors, centralized critics
- Twin Q-networks with clipped double-Q
- Delayed policy updates (every 2 critic updates)
- Target policy smoothing with clipped noise
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MATD3Config:
    """Configuration for MATD3 agent."""
    obs_dim: int = 24
    action_dim: int = 2
    hidden_dim: int = 256
    n_agents: int = 41
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_noise: float = 0.3
    exploration_noise_min: float = 0.05
    exploration_decay: float = 0.9995
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4


class ReplayBuffer:
    """Multi-agent replay buffer."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, n_agents: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idxs]),
            "actions": torch.as_tensor(self.actions[idxs]),
            "rewards": torch.as_tensor(self.rewards[idxs]),
            "next_obs": torch.as_tensor(self.next_obs[idxs]),
            "dones": torch.as_tensor(self.dones[idxs]),
        }

    def __len__(self) -> int:
        return self.size


class Actor(nn.Module):
    """Deterministic actor network."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """Twin Q-network critic (centralized)."""

    def __init__(self, global_obs_dim: int, all_actions_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = global_obs_dim + all_actions_dim

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_obs: torch.Tensor, all_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([global_obs, all_actions], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, global_obs: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([global_obs, all_actions], dim=-1)
        return self.q1(x)


class MATD3Agent:
    """Multi-Agent TD3 for frequency control."""

    def __init__(self, config: MATD3Config, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.n_agents = config.n_agents
        self.action_dim = config.action_dim

        # Per-agent actors
        self.actors = nn.ModuleList([
            Actor(config.obs_dim, config.action_dim, config.hidden_dim)
            for _ in range(config.n_agents)
        ]).to(self.device)

        self.target_actors = nn.ModuleList([
            Actor(config.obs_dim, config.action_dim, config.hidden_dim)
            for _ in range(config.n_agents)
        ]).to(self.device)

        # Copy weights to targets
        for actor, target in zip(self.actors, self.target_actors):
            target.load_state_dict(actor.state_dict())

        # Centralized critic (one for all agents)
        global_obs_dim = config.obs_dim * config.n_agents
        all_actions_dim = config.action_dim * config.n_agents

        self.critic = Critic(global_obs_dim, all_actions_dim, config.hidden_dim).to(self.device)
        self.target_critic = Critic(global_obs_dim, all_actions_dim, config.hidden_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        actor_params = []
        for actor in self.actors:
            actor_params.extend(actor.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        # Replay buffer
        self.buffer = ReplayBuffer(
            config.buffer_size, config.obs_dim, config.action_dim, config.n_agents
        )

        # Training state
        self.total_updates = 0
        self.exploration_noise = config.exploration_noise

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select actions for all agents.

        Args:
            obs: Shape (n_agents, obs_dim)
            explore: Add exploration noise if True

        Returns:
            actions: Shape (n_agents, action_dim)
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            actions = []
            for i, actor in enumerate(self.actors):
                action = actor(obs_t[i])
                actions.append(action)
            actions = torch.stack(actions, dim=0)

            if explore:
                noise = torch.randn_like(actions) * self.exploration_noise
                actions = torch.clamp(actions + noise, -1.0, 1.0)

            return actions.cpu().numpy()

    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        """Select actions without exploration noise."""
        return self.act(obs, explore=False)

    def update(self) -> dict[str, float]:
        """Perform one update step."""
        if len(self.buffer) < self.config.batch_size:
            return {}

        batch = self.buffer.sample(self.config.batch_size)
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device).unsqueeze(-1)

        batch_size = obs.shape[0]

        # Flatten for centralized critic
        obs_flat = obs.reshape(batch_size, -1)
        actions_flat = actions.reshape(batch_size, -1)
        next_obs_flat = next_obs.reshape(batch_size, -1)

        # Compute target actions with smoothing
        with torch.no_grad():
            next_actions = []
            for i, target_actor in enumerate(self.target_actors):
                next_action = target_actor(next_obs[:, i])
                next_actions.append(next_action)
            next_actions = torch.stack(next_actions, dim=1)

            # Target policy smoothing
            noise = torch.randn_like(next_actions) * self.config.policy_noise
            noise = torch.clamp(noise, -self.config.noise_clip, self.config.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)
            next_actions_flat = next_actions.reshape(batch_size, -1)

            # Clipped double-Q
            target_q1, target_q2 = self.target_critic(next_obs_flat, next_actions_flat)
            target_q = torch.min(target_q1, target_q2)
            mean_reward = rewards.mean(dim=-1, keepdim=True)
            target_q = mean_reward + self.config.gamma * (1 - dones) * target_q

        # Critic update
        q1, q2 = self.critic(obs_flat, actions_flat)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.total_updates += 1

        actor_loss = torch.tensor(0.0)

        # Delayed policy update
        if self.total_updates % self.config.policy_delay == 0:
            # Actor update
            current_actions = []
            for i, actor in enumerate(self.actors):
                current_actions.append(actor(obs[:, i]))
            current_actions = torch.stack(current_actions, dim=1)
            current_actions_flat = current_actions.reshape(batch_size, -1)

            actor_loss = -self.critic.q1_forward(obs_flat, current_actions_flat).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update targets
            self._soft_update()

            # Decay exploration noise
            self.exploration_noise = max(
                self.config.exploration_noise_min,
                self.exploration_noise * self.config.exploration_decay,
            )

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "exploration_noise": self.exploration_noise,
        }

    def _soft_update(self) -> None:
        """Soft update target networks."""
        tau = self.config.tau

        for actor, target_actor in zip(self.actors, self.target_actors):
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, path: Path) -> None:
        """Save agent checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actors": self.actors.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config,
            "total_updates": self.total_updates,
            "exploration_noise": self.exploration_noise,
        }, path)

    def load(self, path: Path) -> None:
        """Load agent checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.actors.load_state_dict(ckpt["actors"])
        self.target_actors.load_state_dict(ckpt["target_actors"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.total_updates = ckpt.get("total_updates", 0)
        self.exploration_noise = ckpt.get("exploration_noise", self.config.exploration_noise)

    def eval(self) -> None:
        """Set networks to eval mode."""
        self.actors.eval()
        self.critic.eval()

    def train(self) -> None:
        """Set networks to train mode."""
        self.actors.train()
        self.critic.train()
