"""
Graph-Attention MAPPO for Ancillary Market (AM) in Islanded Microgrid.

Key features:
- DER-level agents (41) with parameter sharing
- Graph attention for inter-agent communication
- Agent-centered critic for credit assignment
- Focus on fast-loop frequency control (FFR, droop)
- Proper reward design and normalization for AM metrics
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.microgrid_env_dual import MicrogridEnvDual
from src.layer2_control.graph_sage_encoder import GraphSAGEEncoder


# ============================================================================
# Normalization utilities
# ============================================================================

class RunningNormalizer:
    """Running mean/std normalizer for observations."""

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0, epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip, self.clip)


class RewardNormalizer:
    """Running reward normalizer using return-based scaling."""

    def __init__(self, gamma: float = 0.99, clip: float = 10.0, epsilon: float = 1e-8):
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon
        self.ret = 0.0
        self.var = 1.0
        self.count = 1e-4

    def normalize(self, reward: float, done: bool = False) -> float:
        self.ret = self.ret * self.gamma + reward
        # Update running variance
        self.var = self.var + (self.ret ** 2 - self.var) / (self.count + 1)
        self.count += 1
        if done:
            self.ret = 0.0
        return np.clip(reward / np.sqrt(self.var + self.epsilon), -self.clip, self.clip)


# ============================================================================
# AM Reward Design
# ============================================================================

@dataclass
class AMRewardConfig:
    """Reward weights for Ancillary Market metrics."""
    # Frequency deviation penalty (Hz)
    w_delta_f: float = 1.0
    delta_f_target: float = 0.0
    delta_f_deadband: float = 0.02  # No penalty within ±20mHz

    # RoCoF penalty (Hz/s) - increased weight for balance
    w_rocof: float = 0.5  # Was 0.2, now 0.5 for better balance
    rocof_target: float = 0.0
    rocof_deadband: float = 0.05  # No penalty within ±50mHz/s

    # Violation time penalty
    w_violation: float = 0.3
    f_limit: float = 0.5  # ±0.5 Hz violation threshold

    # Control effort penalty (encourage smooth actions)
    w_effort: float = 0.01

    # Nadir bonus (reward for keeping frequency above threshold)
    w_nadir_bonus: float = 0.2
    nadir_threshold: float = 49.5  # Hz

    # Coordination bonus (reward for VPP-level cooperation)
    w_coordination: float = 0.1


def compute_am_reward(
    delta_f: float,
    rocof: float,
    action: np.ndarray,
    prev_action: np.ndarray | None,
    freq_hz: float,
    cfg: AMRewardConfig,
) -> tuple[float, dict[str, float]]:
    """
    Compute AM reward with proper scaling and deadbands.

    Returns:
        reward: Scalar reward
        info: Dict of reward components for logging
    """
    # Frequency deviation penalty with deadband
    delta_f_abs = abs(delta_f)
    if delta_f_abs <= cfg.delta_f_deadband:
        r_delta_f = 0.0
    else:
        r_delta_f = -cfg.w_delta_f * (delta_f_abs - cfg.delta_f_deadband)

    # RoCoF penalty with deadband
    rocof_abs = abs(rocof)
    if rocof_abs <= cfg.rocof_deadband:
        r_rocof = 0.0
    else:
        r_rocof = -cfg.w_rocof * (rocof_abs - cfg.rocof_deadband)

    # Violation penalty (frequency outside limits)
    in_violation = delta_f_abs > cfg.f_limit
    r_violation = -cfg.w_violation if in_violation else 0.0

    # Control effort penalty (action magnitude + smoothness)
    action_norm = np.linalg.norm(action) / max(len(action), 1)
    if prev_action is not None:
        action_diff = np.linalg.norm(action - prev_action) / max(len(action), 1)
    else:
        action_diff = 0.0
    r_effort = -cfg.w_effort * (action_norm + 0.5 * action_diff)

    # Nadir bonus (reward for staying above threshold)
    if freq_hz >= cfg.nadir_threshold:
        r_nadir = cfg.w_nadir_bonus * (freq_hz - cfg.nadir_threshold) / 0.5
    else:
        r_nadir = -cfg.w_nadir_bonus * (cfg.nadir_threshold - freq_hz) / 0.5

    # Total reward
    reward = r_delta_f + r_rocof + r_violation + r_effort + r_nadir

    info = {
        "r_delta_f": r_delta_f,
        "r_rocof": r_rocof,
        "r_violation": r_violation,
        "r_effort": r_effort,
        "r_nadir": r_nadir,
    }

    return float(reward), info


# ============================================================================
# GraphSAGE Encoder Adapter (feeder-level -> agent-level)
# ============================================================================

class FeederGraphSAGEAgentEncoder(nn.Module):
    """Encode full feeder graph and extract DER agent embeddings by bus index."""

    def __init__(self, obs_feat: int, hidden_dim: int = 64, embed_dim: int = 64):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_dim=obs_feat, hidden_dim=hidden_dim, out_dim=embed_dim)

    def forward(self, x_full: torch.Tensor, edge_index: torch.Tensor, agent_bus_idx: torch.Tensor) -> torch.Tensor:
        if x_full.ndim != 2:
            raise ValueError("x_full must have shape (n_bus, obs_feat)")
        node_embeds = self.encoder(x_full, edge_index)
        return node_embeds[agent_bus_idx]


# ============================================================================
# Per-Agent Actor with Parameter Sharing
# ============================================================================

class SharedGaussianActor(nn.Module):
    """Shared actor network for all agents."""

    def __init__(self, embed_dim: int, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embed: Agent embeddings [n_agents, embed_dim] or [batch, n_agents, embed_dim]

        Returns:
            mean: Action means
            std: Action stds
        """
        h = self.net(embed)
        mean = torch.tanh(self.mean_head(h))
        std = torch.clamp(self.log_std.exp(), 0.01, 1.0)
        return mean, std.expand_as(mean)

    def dist(self, embed: torch.Tensor) -> torch.distributions.Normal:
        mean, std = self.forward(embed)
        return torch.distributions.Normal(mean, std)


# ============================================================================
# Agent-Centered Critic
# ============================================================================

class AgentCenteredCritic(nn.Module):
    """
    Agent-centered critic for better credit assignment.
    Each agent has its own value estimate based on local + neighbor info.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Global context from graph pooling
        self.global_proj = nn.Linear(embed_dim * 2, hidden_dim)  # mean + max pool

        # Agent-specific value head
        self.value_net = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, agent_embeds: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            agent_embeds: [n_agents, embed_dim] or [batch * n_agents, embed_dim]
            batch: Batch assignment for each node (for batched graphs)

        Returns:
            values: [n_agents, 1] or [batch * n_agents, 1]
        """
        if batch is None:
            # Single graph
            global_mean = agent_embeds.mean(dim=0, keepdim=True)
            global_max = agent_embeds.max(dim=0, keepdim=True).values
            global_ctx = torch.cat([global_mean, global_max], dim=-1)
            global_ctx = F.relu(self.global_proj(global_ctx))
            global_ctx = global_ctx.expand(agent_embeds.shape[0], -1)
        else:
            # Batched graphs
            global_mean = global_mean_pool(agent_embeds, batch)
            global_max = global_max_pool(agent_embeds, batch)
            global_ctx = torch.cat([global_mean, global_max], dim=-1)
            global_ctx = F.relu(self.global_proj(global_ctx))
            global_ctx = global_ctx[batch]

        # Combine agent embedding with global context
        combined = torch.cat([agent_embeds, global_ctx], dim=-1)
        values = self.value_net(combined)
        return values


# ============================================================================
# GA-MAPPO Agent
# ============================================================================

class GAMAPPOAgent(nn.Module):
    """GraphSAGE MAPPO agent for AM control."""

    def __init__(
        self,
        obs_feat: int,
        n_agents: int,
        n_bus: int,
        agent_bus_indices: np.ndarray,
        action_dim_per_agent: int = 1,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        lr: float = 3e-4,
        max_grad_norm: float = 0.5,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_bus = n_bus
        self.obs_feat = obs_feat
        self.action_dim = action_dim_per_agent
        self.embed_dim = embed_dim
        self.agent_bus_indices = torch.as_tensor(agent_bus_indices, dtype=torch.long)

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Networks
        self.encoder = FeederGraphSAGEAgentEncoder(obs_feat, hidden_dim, embed_dim)
        self.actor = SharedGaussianActor(embed_dim, action_dim_per_agent, hidden_dim)
        self.critic = AgentCenteredCritic(embed_dim, hidden_dim)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)

    def _agent_index_tensor(self, device: torch.device) -> torch.Tensor:
        return self.agent_bus_indices.to(device=device)

    def get_agent_embeddings(self, obs_full: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get per-agent embeddings from full feeder graph."""
        return self.encoder(obs_full, edge_index, self._agent_index_tensor(obs_full.device))

    @torch.no_grad()
    def act(
        self,
        obs_full: np.ndarray,
        edge_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Select actions for all DER agents from full-feeder observation.

        Returns:
            actions: [n_agents, action_dim]
            log_probs: [n_agents]
            values: [n_agents]
            embeddings: [n_agents, embed_dim]
        """
        obs_t = torch.tensor(obs_full, dtype=torch.float32)
        edge_t = torch.tensor(edge_index, dtype=torch.long)

        embeds = self.get_agent_embeddings(obs_t, edge_t)
        dist = self.actor.dist(embeds)
        actions = dist.sample()
        actions = torch.clamp(actions, -1.0, 1.0)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.critic(embeds).squeeze(-1)

        return (
            actions.detach().cpu().numpy(),
            log_probs.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
            embeds.detach().cpu().numpy(),
        )

    @torch.no_grad()
    def act_deterministic(self, obs_full: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        """Deterministic action selection (for evaluation)."""
        obs_t = torch.tensor(obs_full, dtype=torch.float32)
        edge_t = torch.tensor(edge_index, dtype=torch.long)

        embeds = self.get_agent_embeddings(obs_t, edge_t)
        mean, _ = self.actor(embeds)
        return mean.detach().cpu().numpy()

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        obs_batch: torch.Tensor,
        edge_batch: list[torch.Tensor],
        actions_batch: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        n_epochs: int = 4,
        mini_batch_size: int = 64,
    ) -> dict[str, float]:
        """PPO update with mini-batches."""
        n_samples = obs_batch.shape[0]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, mini_batch_size):
                end = min(start + mini_batch_size, n_samples)
                idx = indices[start:end]

                # Get batch data
                obs_mb = obs_batch[idx]
                actions_mb = actions_batch[idx]
                old_lp_mb = old_log_probs[idx]
                returns_mb = returns[idx]
                adv_mb = advantages[idx]

                # Forward pass (for simplicity, use first edge_index - assumes same topology)
                edge_idx = edge_batch[0] if len(edge_batch) > 0 else edge_batch[idx[0]]

                # Process each sample
                policy_loss_acc = 0.0
                value_loss_acc = 0.0
                entropy_acc = 0.0

                for i, sample_idx in enumerate(idx):
                    obs_i = obs_mb[i]
                    action_i = actions_mb[i]
                    old_lp_i = old_lp_mb[i]
                    ret_i = returns_mb[i]
                    adv_i = adv_mb[i]
                    edge_i = edge_batch[sample_idx] if sample_idx < len(edge_batch) else edge_idx

                    embeds = self.get_agent_embeddings(obs_i, edge_i)
                    dist = self.actor.dist(embeds)
                    new_log_prob = dist.log_prob(action_i).sum()
                    entropy = dist.entropy().mean()
                    value = self.critic(embeds).mean()

                    # PPO clipped objective
                    ratio = torch.exp(new_log_prob - old_lp_i)
                    surr1 = ratio * adv_i
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_i
                    policy_loss = -torch.min(surr1, surr2)

                    # Value loss
                    value_loss = F.mse_loss(value, ret_i)

                    policy_loss_acc += policy_loss
                    value_loss_acc += value_loss
                    entropy_acc += entropy

                # Average over batch
                batch_size = len(idx)
                policy_loss = policy_loss_acc / batch_size
                value_loss = value_loss_acc / batch_size
                entropy = entropy_acc / batch_size

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            "loss": total_loss / max(n_updates, 1),
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }


# ============================================================================
# Training Buffer
# ============================================================================

@dataclass
class AMBuffer:
    """Rollout buffer for AM training."""
    obs: list[np.ndarray] = field(default_factory=list)
    edge_index: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    log_probs: list[np.ndarray] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        edge_index: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.obs.append(obs)
        self.edge_index.append(edge_index)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self) -> None:
        self.obs.clear()
        self.edge_index.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ============================================================================
# Environment Wrapper for AM-only
# ============================================================================

def ensure_edge_index(edge_index: np.ndarray | torch.Tensor, n_nodes: int) -> np.ndarray:
    ei = np.asarray(edge_index, dtype=np.int64)
    if ei.ndim == 2 and ei.shape[0] == 2 and ei.shape[1] > 0:
        valid = (ei[0] >= 0) & (ei[0] < n_nodes) & (ei[1] >= 0) & (ei[1] < n_nodes)
        ei = ei[:, valid]
    if ei.ndim != 2 or ei.shape[0] != 2 or ei.shape[1] == 0:
        src = np.arange(0, n_nodes - 1, dtype=np.int64)
        dst = np.arange(1, n_nodes, dtype=np.int64)
        ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
    return ei


def get_am_obs(env: MicrogridEnvDual, obs_fast: np.ndarray) -> np.ndarray:
    """
    Extract AM-relevant observation for each agent.

    Features per agent:
    0. delta_f (Hz) - normalized
    1. rocof (Hz/s) - normalized
    2. p_net (MW) - normalized by p_rated
    3. soc (if battery) - [0, 1]
    4. zone_lmp - normalized
    5. agent_type (one-hot: PV=0, BESS=1, V2G=2, DPV=3)
    6. vpp_membership (one-hot: VPP1=0, VPP2=1, VPP3=2, none=3)
    """
    n_agents = env.n_agents
    obs = np.zeros((n_agents, 10), dtype=np.float32)

    # Global frequency state
    freq_state = env.freq_dyn.get_state()
    delta_f = np.clip(freq_state.delta_f_hz / 0.5, -1.0, 1.0)  # Normalize to [-1, 1]
    rocof = np.clip(freq_state.rocof_hz_s / 1.0, -1.0, 1.0)  # Normalize to [-1, 1]

    obs[:, 0] = delta_f
    obs[:, 1] = rocof

    # Per-agent features from fast obs
    obs[:, 2] = np.clip(obs_fast[:, 2] / 1.0, -1.0, 1.0)  # p_net normalized
    obs[:, 3] = np.clip(obs_fast[:, 3], 0.0, 1.0)  # dcob/soc
    obs[:, 4] = np.clip(obs_fast[:, 4] / 100.0, 0.0, 1.0)  # zone_lmp normalized

    # Agent type encoding
    for i, spec in enumerate(env._agent_specs):
        agent_type = spec.get("type", "")
        if "PV" in agent_type and "DPV" not in agent_type:
            obs[i, 5] = 1.0  # EVCS_PV
        elif "BESS" in agent_type:
            obs[i, 6] = 1.0  # BESS
        elif "V2G" in agent_type:
            obs[i, 7] = 1.0  # V2G
        else:
            obs[i, 8] = 1.0  # DPV

    # VPP membership
    for vpp_idx, (vpp_id, agents) in enumerate(env._vpp_droop_agents.items()):
        for ai in agents:
            if ai < n_agents:
                obs[ai, 9] = (vpp_idx + 1) / 3.0  # Normalized VPP index

    return obs


def _fill_grid_state_background_am(env: MicrogridEnvDual, obs_full: np.ndarray, base_feat: int) -> None:
    n_bus = obs_full.shape[0]
    net = env.net

    vm_pu = np.ones(n_bus, dtype=np.float32)
    if hasattr(net, "res_bus") and not net.res_bus.empty and "vm_pu" in net.res_bus.columns:
        vm_vals = net.res_bus["vm_pu"].values
        if len(vm_vals) == n_bus:
            vm_pu = np.nan_to_num(vm_vals.astype(np.float32), nan=1.0)

    p_load = np.zeros(n_bus, dtype=np.float32)
    q_load = np.zeros(n_bus, dtype=np.float32)
    if hasattr(net, "load") and len(net.load) > 0:
        for _, row in net.load.iterrows():
            bus_idx = int(row.get("bus", -1))
            if 0 <= bus_idx < n_bus:
                p_load[bus_idx] += float(row.get("p_mw", 0.0))
                q_load[bus_idx] += float(row.get("q_mvar", 0.0))

    if base_feat > 2:
        obs_full[:, 2] = p_load / 100.0
    if base_feat > 4:
        obs_full[:, 4] = np.clip(obs_full[:, 4] + (vm_pu - 1.0), -1.0, 1.0)
    if base_feat > 5:
        obs_full[:, 5] = q_load / 100.0


def build_am_full_feeder_obs(env: MicrogridEnvDual, obs_fast: np.ndarray) -> np.ndarray:
    obs_agent = get_am_obs(env, obs_fast)
    n_bus = int(len(env.net.bus.index))
    base_feat = int(obs_agent.shape[1])
    extra_feat = 4
    obs_full = np.zeros((n_bus, base_feat + extra_feat), dtype=np.float32)

    _fill_grid_state_background_am(env, obs_full, base_feat)

    bus_pos = np.asarray(getattr(env, "_agent_bus_pp", np.arange(obs_agent.shape[0])), dtype=np.int64)
    bus_pos = np.clip(bus_pos, 0, max(n_bus - 1, 0))
    counts = np.zeros(n_bus, dtype=np.int32)

    for i in range(min(obs_agent.shape[0], bus_pos.shape[0])):
        b = int(bus_pos[i])
        obs_full[b, :base_feat] += obs_agent[i]
        counts[b] += 1

        p_val = float(obs_fast[i, 2]) if i < obs_fast.shape[0] and obs_fast.shape[1] > 2 else 0.0
        soc_val = float(obs_fast[i, 3]) if i < obs_fast.shape[0] and obs_fast.shape[1] > 3 else 0.0
        obs_full[b, base_feat + 0] += p_val
        obs_full[b, base_feat + 1] += soc_val
        obs_full[b, base_feat + 2] += float(env._agent_specs[i].get("vpp_idx", -1)) if i < len(env._agent_specs) else -1.0

    valid = counts > 0
    if np.any(valid):
        obs_full[valid, :base_feat] = obs_full[valid, :base_feat] / counts[valid, None]
        obs_full[valid, base_feat + 0] = obs_full[valid, base_feat + 0] / counts[valid]
        obs_full[valid, base_feat + 1] = obs_full[valid, base_feat + 1] / counts[valid]
        obs_full[valid, base_feat + 2] = obs_full[valid, base_feat + 2] / counts[valid]
        obs_full[valid, base_feat + 3] = counts[valid].astype(np.float32) / float(max(env.n_agents, 1))

    return np.nan_to_num(obs_full, nan=0.0, posinf=1e3, neginf=-1e3)


# ============================================================================
# Training Loop
# ============================================================================

def train_am_mappo(
    env: MicrogridEnvDual,
    agent: GAMAPPOAgent,
    n_episodes: int = 1000,
    steps_per_episode: int = 300,
    update_freq: int = 10,
    update_epochs: int = 4,
    mini_batch_size: int = 64,
    log_interval: int = 10,
    checkpoint_dir: Path | None = None,
    reward_cfg: AMRewardConfig | None = None,
) -> dict[str, list[float]]:
    """Main training loop for AM-MAPPO."""

    if reward_cfg is None:
        reward_cfg = AMRewardConfig()

    buffer = AMBuffer()
    obs_fast0, _, _ = env.reset()
    obs_full0 = build_am_full_feeder_obs(env, obs_fast0)
    obs_normalizer = RunningNormalizer(obs_full0.shape)
    reward_normalizer = RewardNormalizer(gamma=agent.gamma)

    history = {
        "episode_reward": [],
        "delta_f_mean": [],
        "rocof_mean": [],
        "rocof_max": [],
        "max_abs_delta_f": [],
        "freq_nadir": [],
        "violation_fraction": [],
        "loss": [],
        "entropy": [],
        "r_delta_f": [],
        "r_rocof": [],
        "r_violation": [],
        "r_effort": [],
        "r_nadir": [],
    }

    for ep in range(n_episodes):
        obs_fast, _, _ = env.reset()
        n_bus = int(len(env.net.bus.index))
        edge_index = ensure_edge_index(env.edge_index, n_nodes=n_bus)
        obs_full = build_am_full_feeder_obs(env, obs_fast)
        obs_normalizer.update(obs_full)
        obs_norm = obs_normalizer.normalize(obs_full)

        ep_reward = 0.0
        ep_delta_f = []
        ep_rocof = []
        ep_freq_hz = []
        ep_violation = []
        ep_r_delta_f = []
        ep_r_rocof = []
        ep_r_violation = []
        ep_r_effort = []
        ep_r_nadir = []
        prev_action = None

        for _t in range(steps_per_episode):
            actions, log_probs, values, _ = agent.act(obs_norm, edge_index)

            full_action = np.zeros(44, dtype=np.float32)
            full_action[:41] = actions.flatten()[:41]
            for vpp_idx, (_vpp_id, member_agents) in enumerate(env._vpp_droop_agents.items()):
                vpp_action = np.mean([actions[ai, 0] for ai in member_agents if ai < len(actions)])
                full_action[41 + vpp_idx] = vpp_action

            next_obs_fast, _r_env, done, _trunc, info = env.step_fast(full_action)
            n_bus = int(len(env.net.bus.index))
            edge_index = ensure_edge_index(info.get("edge_index", edge_index), n_nodes=n_bus)

            freq_state = env.freq_dyn.get_state()
            freq_hz = 50.0 + freq_state.delta_f_hz
            reward, reward_info = compute_am_reward(
                delta_f=freq_state.delta_f_hz,
                rocof=freq_state.rocof_hz_s,
                action=actions.flatten(),
                prev_action=prev_action,
                freq_hz=freq_hz,
                cfg=reward_cfg,
            )

            reward_norm = reward_normalizer.normalize(reward, done=done)

            buffer.add(
                obs=obs_norm.copy(),
                edge_index=edge_index.copy(),
                action=actions.copy(),
                log_prob=float(log_probs.mean()),
                value=float(values.mean()),
                reward=reward_norm,
                done=done,
            )

            next_obs = build_am_full_feeder_obs(env, next_obs_fast)
            obs_normalizer.update(next_obs)
            obs_norm = obs_normalizer.normalize(next_obs)
            prev_action = actions.flatten().copy()

            ep_reward += reward
            ep_delta_f.append(abs(freq_state.delta_f_hz))
            ep_rocof.append(abs(freq_state.rocof_hz_s))
            ep_freq_hz.append(freq_hz)
            ep_violation.append(1.0 if abs(freq_state.delta_f_hz) > reward_cfg.f_limit else 0.0)
            ep_r_delta_f.append(reward_info["r_delta_f"])
            ep_r_rocof.append(reward_info["r_rocof"])
            ep_r_violation.append(reward_info["r_violation"])
            ep_r_effort.append(reward_info["r_effort"])
            ep_r_nadir.append(reward_info["r_nadir"])

        # Update policy
        if len(buffer) >= update_freq:
            # Compute advantages
            rewards = np.array(buffer.rewards, dtype=np.float32)
            values = np.array(buffer.values, dtype=np.float32)
            dones = np.array(buffer.dones, dtype=np.float32)
            advantages, returns = agent.compute_gae(rewards, values, dones)

            # Prepare batch
            obs_batch = torch.tensor(np.stack(buffer.obs), dtype=torch.float32)
            edge_batch = [torch.tensor(e, dtype=torch.long) for e in buffer.edge_index]
            actions_batch = torch.tensor(np.stack(buffer.actions), dtype=torch.float32)
            old_log_probs = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32)
            returns_t = torch.tensor(returns, dtype=torch.float32)
            advantages_t = torch.tensor(advantages, dtype=torch.float32)

            # Update
            update_info = agent.update(
                obs_batch,
                edge_batch,
                actions_batch,
                old_log_probs,
                returns_t,
                advantages_t,
                n_epochs=update_epochs,
                mini_batch_size=mini_batch_size,
            )

            history["loss"].append(update_info["loss"])
            history["entropy"].append(update_info["entropy"])

            buffer.clear()

        # Log
        history["episode_reward"].append(ep_reward)
        history["delta_f_mean"].append(np.mean(ep_delta_f) if ep_delta_f else 0.0)
        history["rocof_mean"].append(np.mean(ep_rocof) if ep_rocof else 0.0)
        history["rocof_max"].append(np.max(ep_rocof) if ep_rocof else 0.0)
        history["max_abs_delta_f"].append(np.max(ep_delta_f) if ep_delta_f else 0.0)
        history["freq_nadir"].append(np.min(ep_freq_hz) if ep_freq_hz else 50.0)
        history["violation_fraction"].append(np.mean(ep_violation) if ep_violation else 0.0)
        history["r_delta_f"].append(np.mean(ep_r_delta_f) if ep_r_delta_f else 0.0)
        history["r_rocof"].append(np.mean(ep_r_rocof) if ep_r_rocof else 0.0)
        history["r_violation"].append(np.mean(ep_r_violation) if ep_r_violation else 0.0)
        history["r_effort"].append(np.mean(ep_r_effort) if ep_r_effort else 0.0)
        history["r_nadir"].append(np.mean(ep_r_nadir) if ep_r_nadir else 0.0)

        if (ep + 1) % log_interval == 0:
            recent_reward = np.mean(history["episode_reward"][-log_interval:])
            recent_delta_f = np.mean(history["delta_f_mean"][-log_interval:])
            recent_rocof = np.mean(history["rocof_mean"][-log_interval:])
            recent_vio = np.mean(history["violation_fraction"][-log_interval:])
            recent_nadir = np.mean(history["freq_nadir"][-log_interval:])
            recent_entropy = np.mean(history["entropy"][-log_interval:]) if history["entropy"] else 0.0

            print(
                f"Ep {ep+1:4d} | R={recent_reward:7.2f} | dF={recent_delta_f:.4f} | "
                f"RoCoF={recent_rocof:.4f} | Violation={recent_vio:.3f} | Nadir={recent_nadir:.3f} | H={recent_entropy:.4f}"
            )

        # Checkpoint
        if checkpoint_dir and (ep + 1) % 100 == 0:
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "obs_normalizer": {"mean": obs_normalizer.mean, "var": obs_normalizer.var, "count": obs_normalizer.count},
                "episode": ep + 1,
            }, checkpoint_dir / f"am_mappo_ep{ep+1}.pt")

    return history


# ============================================================================
# Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphSAGE-MAPPO for Ancillary Market")
    parser.add_argument("--n-episodes", type=int, default=1000)
    parser.add_argument("--steps-per-episode", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--placement", type=str, default="artifacts/placement/official_placement_v3.json")
    parser.add_argument("--mpc-path", type=str, default="data/grid_IEEE123_complete.m")
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints_am_mappo")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing environment...")
    env = MicrogridEnvDual(
        placement_path=args.placement,
        mpc_path=args.mpc_path,
        seed=args.seed,
    )

    print(f"Environment: {env.n_agents} agents, 3 VPPs")

    sample_obs_fast, _, _ = env.reset()
    sample_obs_full = build_am_full_feeder_obs(env, sample_obs_fast)
    n_bus = int(sample_obs_full.shape[0])
    obs_feat = int(sample_obs_full.shape[1])
    agent_bus_indices = np.asarray(getattr(env, "_agent_bus_pp", np.arange(env.n_agents)), dtype=np.int64)
    agent_bus_indices = np.clip(agent_bus_indices, 0, max(n_bus - 1, 0))

    # Initialize agent
    agent = GAMAPPOAgent(
        obs_feat=obs_feat,
        n_agents=env.n_agents,
        n_bus=n_bus,
        agent_bus_indices=agent_bus_indices,
        action_dim_per_agent=1,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
    )

    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # Train
    print("\n=== Training GA-MAPPO for AM ===")
    history = train_am_mappo(
        env=env,
        agent=agent,
        n_episodes=args.n_episodes,
        steps_per_episode=args.steps_per_episode,
        update_epochs=args.update_epochs,
        mini_batch_size=args.mini_batch_size,
        log_interval=args.log_interval,
        checkpoint_dir=checkpoint_dir,
    )

    # Save final
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "history": history,
        "args": vars(args),
    }, checkpoint_dir / "am_mappo_final.pt")

    print(f"\nTraining complete. Saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
