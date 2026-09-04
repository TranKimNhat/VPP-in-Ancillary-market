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
    """Configuration for EIE-MATD3 agent.

    Core TD3 (twin-Q, target smoothing, delayed update, soft target) plus the
    EIE machinery: diverse-noise explorers, a demonstrator imitation pool, and
    dual-replay mixed sampling. hidden_dim=128 matches the proposed method's
    capacity for a fair comparison.
    """
    obs_dim: int = 24
    action_dim: int = 2
    hidden_dim: int = 128
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
    # --- EIE (Efficient Integration Exploration) ---
    demo_ratio: float = 0.25          # fraction of each mini-batch drawn from the demonstrator pool
    ou_theta: float = 0.15            # Ornstein-Uhlenbeck mean-reversion
    ou_sigma: float = 0.20            # OU volatility
    eps_greedy_prob: float = 0.20     # prob. of a uniform-random action under eps-greedy explorer
    max_entropy_scale: float = 2.0    # noise-scale multiplier for the max-entropy explorer


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

        # Dual replay pools (EIE): Pool 1 = explorers, Pool 2 = demonstrator.
        self.buffer = ReplayBuffer(
            config.buffer_size, config.obs_dim, config.action_dim, config.n_agents
        )
        self.demo_buffer = ReplayBuffer(
            config.buffer_size, config.obs_dim, config.action_dim, config.n_agents
        )

        # Training state
        self.total_updates = 0
        self.exploration_noise = config.exploration_noise
        # Ornstein-Uhlenbeck state (one explorer strategy); reset per episode.
        self._ou_state = np.zeros((config.n_agents, config.action_dim), dtype=np.float32)

    def reset_noise(self) -> None:
        """Reset per-episode exploration state (Ornstein-Uhlenbeck)."""
        self._ou_state.fill(0.0)

    def act(self, obs: np.ndarray, explore: bool = True, strategy: str = "gaussian") -> np.ndarray:
        """Select actions for all agents (EIE explorers share the leader actors).

        Args:
            obs: Shape (n_agents, obs_dim)
            explore: Add exploration if True.
            strategy: Explorer noise strategy — one of
                "gaussian", "ou", "eps_greedy", "max_entropy". All explorers use
                the SAME (leader) actor weights; only the injected noise differs,
                which is the implicit leader->explorer weight sync.

        Returns:
            actions: Shape (n_agents, action_dim), clipped to [-1, 1].
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            actions = torch.stack([actor(obs_t[i]) for i, actor in enumerate(self.actors)], dim=0)
            a = actions.cpu().numpy()

            if not explore or strategy == "none":
                return np.clip(a, -1.0, 1.0)

            sigma = self.exploration_noise
            if strategy == "gaussian":
                a = a + np.random.randn(*a.shape).astype(np.float32) * sigma
            elif strategy == "ou":
                self._ou_state = (
                    (1.0 - self.config.ou_theta) * self._ou_state
                    + self.config.ou_sigma * np.random.randn(*a.shape).astype(np.float32)
                )
                a = a + self._ou_state
            elif strategy == "eps_greedy":
                if np.random.rand() < self.config.eps_greedy_prob:
                    a = np.random.uniform(-1.0, 1.0, size=a.shape).astype(np.float32)
                else:
                    a = a + np.random.randn(*a.shape).astype(np.float32) * sigma
            elif strategy == "max_entropy":
                a = a + np.random.randn(*a.shape).astype(np.float32) * sigma * self.config.max_entropy_scale
            else:
                raise ValueError(f"Unknown explorer strategy: {strategy}")

            return np.clip(a, -1.0, 1.0).astype(np.float32)

    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        """Select actions without exploration noise (for eval / leader policy)."""
        return self.act(obs, explore=False)

    def update(self, demo_ratio: float | None = None) -> dict[str, float]:
        """Perform one update step (EIE: mixed mini-batch from dual pools).

        Draws (1-demo_ratio) of the batch from the explorer pool (Pool 1) and
        demo_ratio from the demonstrator pool (Pool 2) when the latter has enough
        samples; otherwise falls back to the explorer pool only.
        """
        bs = self.config.batch_size
        if len(self.buffer) < bs:
            return {}

        beta = self.config.demo_ratio if demo_ratio is None else float(demo_ratio)
        n_demo = int(round(bs * beta))
        if n_demo > 0 and len(self.demo_buffer) >= n_demo:
            n_exp = bs - n_demo
            b_exp = self.buffer.sample(n_exp)
            b_demo = self.demo_buffer.sample(n_demo)
            batch = {k: torch.cat([b_exp[k], b_demo[k]], dim=0) for k in b_exp}
        else:
            batch = self.buffer.sample(bs)
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
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
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
