from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.microgrid_env_dual import MicrogridEnvDual
from src.layer2_control.graph_sage_encoder import GraphSAGEEncoder
from src.opt.tie_switch_reconfig import TieSwitchReconfiguration


DUAL_PHASES: dict[str, dict[str, Any]] = {
    "A": {
        "n_episodes": 400,
        "event_probs": {"load_step": 0.0, "gen_trip": 0.0, "line_trip": 0.0, "high_ren": 0.0},
        "n_topologies": 1,
        "lr_factor": 1.0,
    },
    "B": {
        "n_episodes": 1000,
        "event_probs": {"load_step": 0.5, "gen_trip": 0.5, "line_trip": 0.0, "high_ren": 0.0},
        "n_topologies": 5,
        "lr_factor": 1.0,
    },
    "C1": {
        "n_episodes": 1000,
        "event_probs": {"load_step": 0.4, "gen_trip": 0.4, "line_trip": 0.0, "high_ren": 0.2},
        "n_topologies": 20,
        "lr_factor": 1.0,
    },
    "C2": {
        "n_episodes": 1000,
        "event_probs": {"load_step": 0.3, "gen_trip": 0.3, "line_trip": 0.1, "high_ren": 0.3},
        "n_topologies": 20,
        "lr_factor": 1.0,
    },
    "D": {
        "n_episodes": 1000,
        "event_probs": {"load_step": 0.3, "gen_trip": 0.3, "line_trip": 0.2, "high_ren": 0.2},
        "n_topologies": 20,
        "lr_factor": 1.0,
    },
    "E": {
        "n_episodes": 1000,
        "event_probs": {"load_step": 0.3, "gen_trip": 0.3, "line_trip": 0.2, "high_ren": 0.2},
        "n_topologies": 20,
        "lr_factor": 0.5,
    },
    "F": {
        "n_episodes": 1000,
        "event_probs": {"load_step": 0.3, "gen_trip": 0.3, "line_trip": 0.2, "high_ren": 0.2},
        "n_topologies": 20,
        "lr_factor": 0.5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-loop MAPPO training")
    parser.add_argument("--mode", choices=["fast", "slow", "dual"], default="dual")
    parser.add_argument("--n-episodes", type=int, default=5400)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--phase-smoke", action="store_true")
    parser.add_argument("--phase-episodes", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--encoder", choices=["sage", "mlp", "gat"], default="sage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--placement", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--update-freq-fast", type=int, default=1)
    parser.add_argument("--update-freq-slow", type=int, default=1)
    parser.add_argument("--mpc-path", type=str, default="data/grid_IEEE123_complete.m")
    return parser.parse_args()


@dataclass
class PPOBuffer:
    obs: list[torch.Tensor]
    edge_index: list[torch.Tensor]
    actions: list[torch.Tensor]
    log_probs: list[torch.Tensor]
    values: list[torch.Tensor]
    rewards: list[float]
    dones: list[float]

    def __init__(self) -> None:
        self.obs = []
        self.edge_index = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(
        self,
        obs: torch.Tensor,
        edge_index: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        self.obs.append(obs.detach())
        self.edge_index.append(edge_index.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(float(reward))
        self.dones.append(float(done))

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.rewards)


class GaussianActor(nn.Module):
    def __init__(self, emb_dim: int, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim + obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, emb: torch.Tensor, obs_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([emb, obs_flat], dim=-1)
        mean = torch.tanh(self.net(x))
        std = torch.nn.functional.softplus(self.log_std).expand_as(mean) + 1e-4
        std = torch.clamp(std, min=1e-4, max=2.0)
        return mean, std

    def dist(self, emb: torch.Tensor, obs_flat: torch.Tensor) -> torch.distributions.Normal:
        mean, std = self.forward(emb, obs_flat)
        return torch.distributions.Normal(mean, std)


class ValueCritic(nn.Module):
    def __init__(self, emb_dim: int, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim + obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, emb: torch.Tensor, obs_flat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb, obs_flat], dim=-1)
        return self.net(x).squeeze(-1)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int = 7, hidden_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: np.ndarray | torch.Tensor, edge_index: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x_t = x.float()
        else:
            x_t = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
        if x_t.ndim == 2:
            return self.net(x_t)
        if x_t.ndim == 3:
            b, n, f = x_t.shape
            out = self.net(x_t.reshape(b * n, f))
            return out.reshape(b, n, -1)
        raise ValueError("MLPEncoder expects x with shape (N, F) or (B, N, F)")


class GATEncoderDual(nn.Module):
    def __init__(self, in_dim: int = 7, hidden_dim: int = 32, out_dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=0.1)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=True, dropout=0.1)

    @staticmethod
    def _to_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(np.asarray(x), dtype=torch.float32)

    def _normalize_edge(self, edge_index: np.ndarray | torch.Tensor, n_nodes: int) -> torch.Tensor:
        ei = edge_index if isinstance(edge_index, torch.Tensor) else torch.as_tensor(edge_index, dtype=torch.long)
        ei = ei.long()
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if ei.numel() > 0:
            valid = (ei[0] >= 0) & (ei[0] < n_nodes) & (ei[1] >= 0) & (ei[1] < n_nodes)
            ei = ei[:, valid]
        if ei.numel() == 0:
            src = torch.arange(0, n_nodes - 1, dtype=torch.long)
            dst = torch.arange(1, n_nodes, dtype=torch.long)
            ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        return ei

    def _forward_single(self, x: torch.Tensor, edge_index: np.ndarray | torch.Tensor) -> torch.Tensor:
        ei = self._normalize_edge(edge_index, n_nodes=x.shape[0]).to(device=x.device)
        h = self.gat1(x, ei)
        h = F.elu(h)
        h = F.dropout(h, p=0.1, training=self.training)
        return self.gat2(h, ei)

    def forward(self, x: np.ndarray | torch.Tensor, edge_index: np.ndarray | torch.Tensor) -> torch.Tensor:
        x_t = self._to_tensor(x).float()
        if x_t.ndim == 2:
            return self._forward_single(x_t, edge_index)
        if x_t.ndim == 3:
            outs = [self._forward_single(x_t[b], edge_index) for b in range(x_t.shape[0])]
            return torch.stack(outs, dim=0)
        raise ValueError("GATEncoderDual expects x with shape (N, F) or (B, N, F)")


class LoopPolicy(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        obs_nodes: int,
        obs_feat: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.003,
        lr: float = 3e-4,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.obs_dim = obs_nodes * obs_feat
        self.emb_dim = 128
        self.action_dim = action_dim

        self.actor = GaussianActor(self.emb_dim, self.obs_dim, action_dim)
        self.critic = ValueCritic(self.emb_dim, self.obs_dim)

        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.last_entropy = 0.0

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

    @staticmethod
    def _to_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)

    def _encode_graph(self, obs: np.ndarray | torch.Tensor, edge_index: np.ndarray | torch.Tensor) -> torch.Tensor:
        obs_t = self._to_tensor(obs)
        h = self.encoder(obs_t, edge_index)
        h_mean = h.mean(dim=0)
        h_max = h.max(dim=0).values
        return torch.cat([h_mean, h_max], dim=0)

    def evaluate_obs(
        self,
        obs: np.ndarray | torch.Tensor,
        edge_index: np.ndarray | torch.Tensor,
    ) -> tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_t = self._to_tensor(obs)
        obs_flat = obs_t.reshape(1, -1)
        emb = self._encode_graph(obs_t, edge_index).reshape(1, -1)
        dist = self.actor.dist(emb, obs_flat)
        value = self.critic(emb, obs_flat)
        return dist, value, emb.squeeze(0), obs_t

    def act(
        self,
        obs: np.ndarray | torch.Tensor,
        edge_index: np.ndarray | torch.Tensor,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value, emb, _ = self.evaluate_obs(obs, edge_index)
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).detach().cpu().numpy(), log_prob.squeeze(0), value.squeeze(0), emb

    @torch.no_grad()
    def act_deterministic(
        self,
        obs: np.ndarray | torch.Tensor,
        edge_index: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        dist, _value, _emb, _obs_t = self.evaluate_obs(obs, edge_index)
        action = torch.clamp(dist.mean, -1.0, 1.0)
        return action.squeeze(0).detach().cpu().numpy()

    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    def update(self, buffer: PPOBuffer) -> dict[str, float]:
        if len(buffer) == 0:
            return {"loss": 0.0, "entropy": 0.0, "value_loss": 0.0}

        obs = torch.stack(buffer.obs)
        edge_idx_list = [ei for ei in buffer.edge_index]
        actions = torch.stack(buffer.actions)
        old_log_probs = torch.stack(buffer.log_probs)
        values = torch.stack(buffer.values).detach().cpu().numpy().astype(np.float32)
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)

        advantages, returns = self._compute_gae(rewards, values, dones)
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        adv_t = torch.tensor((advantages - advantages.mean()) / (advantages.std() + 1e-8), dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)

        n_transitions = len(buffer)
        log_probs = torch.zeros(n_transitions, dtype=torch.float32)
        entropy_each = torch.zeros(n_transitions, dtype=torch.float32)
        value_pred = torch.zeros(n_transitions, dtype=torch.float32)

        topo_groups: dict[bytes, tuple[torch.Tensor, list[int]]] = {}
        for i, edge_i in enumerate(edge_idx_list):
            edge_t = edge_i if isinstance(edge_i, torch.Tensor) else torch.as_tensor(edge_i, dtype=torch.long)
            edge_key = edge_t.detach().cpu().numpy().tobytes()
            if edge_key not in topo_groups:
                topo_groups[edge_key] = (edge_t, [i])
            else:
                topo_groups[edge_key][1].append(i)

        for edge_t, idxs in topo_groups.values():
            idx_t = torch.as_tensor(idxs, dtype=torch.long)
            obs_batch = obs.index_select(0, idx_t)
            action_batch = actions.index_select(0, idx_t)

            h_batch = self.encoder(obs_batch, edge_t)
            emb_batch = torch.cat([h_batch.mean(dim=1), h_batch.max(dim=1).values], dim=1)
            obs_flat_batch = obs_batch.reshape(obs_batch.shape[0], -1)

            dist_batch = self.actor.dist(emb_batch, obs_flat_batch)
            value_batch = self.critic(emb_batch, obs_flat_batch)

            log_probs.index_copy_(0, idx_t, dist_batch.log_prob(action_batch).sum(dim=-1))
            entropy_each.index_copy_(0, idx_t, dist_batch.entropy().sum(dim=-1))
            value_pred.index_copy_(0, idx_t, value_batch)

        entropy = entropy_each.mean()

        if (not torch.isfinite(log_probs).all()) or (not torch.isfinite(value_pred).all()) or (not torch.isfinite(adv_t).all()):
            return {"loss": 0.0, "entropy": float(self.last_entropy), "value_loss": 0.0}

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = torch.mean((ret_t - value_pred) ** 2)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        if not torch.isfinite(loss):
            return {"loss": float("nan"), "entropy": float("nan"), "value_loss": float("nan")}

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()),
            0.5,
        )
        self.optimizer.step()

        self.last_entropy = float(entropy.item())
        return {
            "loss": float(loss.item()),
            "entropy": float(entropy.item()),
            "value_loss": float(value_loss.item()),
        }


class MultiEnvDual:
    def __init__(self, envs: list[MicrogridEnvDual]) -> None:
        self.envs = envs

    def reset(self, seed: int | None = None) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
        obs_f_all: list[np.ndarray] = []
        obs_s_all: list[np.ndarray] = []
        info_all: list[dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            obs_f, obs_s, info = env.reset(seed=None if seed is None else seed + i)
            obs_f_all.append(obs_f)
            obs_s_all.append(obs_s)
            info_all.append(info)
        return obs_f_all, obs_s_all, info_all


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


def build_full_feeder_obs(env: MicrogridEnvDual, obs_agents: np.ndarray, loop: str) -> np.ndarray:
    obs_a = np.asarray(obs_agents, dtype=np.float32)
    n_bus = int(len(env.net.bus.index))
    base_feat = int(obs_a.shape[1]) if obs_a.ndim == 2 else 7
    extra_feat = 7
    obs_full = np.zeros((n_bus, base_feat + extra_feat), dtype=np.float32)

    # Fill all buses with grid state background (avoid dead nodes)
    _fill_grid_state_background(env, obs_full, base_feat, loop)

    if obs_a.ndim != 2 or obs_a.shape[0] == 0:
        return np.nan_to_num(obs_full, nan=0.0, posinf=1e3, neginf=-1e3)

    bus_pos = np.asarray(getattr(env, "_agent_bus_pp", np.arange(obs_a.shape[0])), dtype=np.int64)
    bus_pos = np.clip(bus_pos, 0, max(n_bus - 1, 0))
    specs = list(getattr(env, "_agent_specs", []))

    counts = np.zeros(n_bus, dtype=np.int32)
    for i in range(min(obs_a.shape[0], bus_pos.shape[0])):
        b = int(bus_pos[i])
        row = obs_a[i]
        obs_full[b, :base_feat] += row
        counts[b] += 1

        element = ""
        if i < len(specs):
            element = str(specs[i].get("element", "")).lower()

        p_val = float(row[2]) if row.shape[0] > 2 else 0.0
        q_val = 0.0
        if loop == "slow":
            q_val = float(row[1]) if row.shape[0] > 1 else 0.0

        if element == "sgen":
            obs_full[b, base_feat + 0] += p_val
            obs_full[b, base_feat + 3] += q_val
        elif element == "storage":
            obs_full[b, base_feat + 1] += p_val
            obs_full[b, base_feat + 4] += q_val
        else:
            obs_full[b, base_feat + 2] += p_val
            obs_full[b, base_feat + 5] += q_val

    valid = counts > 0
    if np.any(valid):
        obs_full[valid, :base_feat] = obs_full[valid, :base_feat] / counts[valid, None]
        obs_full[valid, base_feat + 6] = counts[valid].astype(np.float32) / float(max(env.n_agents, 1))

    return np.nan_to_num(obs_full, nan=0.0, posinf=1e3, neginf=-1e3)


def _fill_grid_state_background(env: MicrogridEnvDual, obs_full: np.ndarray, base_feat: int, loop: str) -> None:
    """Fill non-agent buses with grid state features to avoid dead nodes."""
    n_bus = obs_full.shape[0]
    net = env.net

    # Get voltage magnitude from res_bus if available
    vm_pu = np.ones(n_bus, dtype=np.float32)
    if hasattr(net, "res_bus") and not net.res_bus.empty and "vm_pu" in net.res_bus.columns:
        vm_vals = net.res_bus["vm_pu"].values
        if len(vm_vals) == n_bus:
            vm_pu = np.nan_to_num(vm_vals.astype(np.float32), nan=1.0)

    # Aggregate net load per bus
    p_load = np.zeros(n_bus, dtype=np.float32)
    q_load = np.zeros(n_bus, dtype=np.float32)
    if hasattr(net, "load") and len(net.load) > 0:
        for idx, row in net.load.iterrows():
            bus_idx = int(row.get("bus", -1))
            if 0 <= bus_idx < n_bus:
                p_load[bus_idx] += float(row.get("p_mw", 0.0))
                q_load[bus_idx] += float(row.get("q_mvar", 0.0))

    # For slow loop: idx 0 = vm_pu (normalized around 1.0)
    # For fast loop: idx 0 = delta_f (global, fill with 0 as placeholder)
    if loop == "slow" and base_feat > 0:
        obs_full[:, 0] = vm_pu - 1.0  # Center around 0
    if loop == "slow" and base_feat > 1:
        obs_full[:, 1] = q_load / 100.0  # Normalized q
    if base_feat > 2:
        obs_full[:, 2] = p_load / 100.0  # Normalized p


def run_episode(
    env: MicrogridEnvDual,
    mode: str,
    policy_fast: LoopPolicy,
    policy_slow: LoopPolicy,
    buffer_fast: PPOBuffer,
    buffer_slow: PPOBuffer,
) -> tuple[float, float, np.ndarray]:
    obs_f, obs_s, _ = env.reset()
    n_bus = int(len(env.net.bus.index))
    edge_index = ensure_edge_index(env.edge_index, n_nodes=n_bus)
    obs_f_graph = build_full_feeder_obs(env, obs_f, loop="fast")
    obs_s_graph = build_full_feeder_obs(env, obs_s, loop="slow")

    r_fast_list: list[float] = []
    r_slow = 0.0

    act_s, lp_s, val_s, _emb_s = policy_slow.act(obs_s_graph, edge_index)
    env.set_slow_baseline(act_s)

    if mode in {"fast", "dual"}:
        for _t in range(300):
            act_f, lp_f, val_f, _emb_f = policy_fast.act(obs_f_graph, edge_index)
            next_obs_f, r_f, _done, _trunc, info = env.step_fast(act_f)
            n_bus = int(len(env.net.bus.index))
            next_edge = ensure_edge_index(info.get("edge_index", edge_index), n_nodes=n_bus)
            buffer_fast.add(
                obs=torch.tensor(obs_f_graph, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                action=torch.tensor(act_f, dtype=torch.float32),
                log_prob=lp_f,
                value=val_f,
                reward=float(r_f),
                done=False,
            )
            obs_f = next_obs_f
            obs_f_graph = build_full_feeder_obs(env, obs_f, loop="fast")
            edge_index = next_edge
            r_fast_list.append(float(r_f))

    if mode in {"slow", "dual"}:
        obs_s_next, r_slow, done_s, _trunc_s, info_s = env.step_slow(act_s)
        obs_s_graph = build_full_feeder_obs(env, obs_s, loop="slow")
        buffer_slow.add(
            obs=torch.tensor(obs_s_graph, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            action=torch.tensor(act_s, dtype=torch.float32),
            log_prob=lp_s,
            value=val_s,
            reward=float(r_slow),
            done=bool(done_s),
        )
        obs_s = obs_s_next

    r_fast_mean = float(np.mean(r_fast_list)) if r_fast_list else 0.0
    return r_fast_mean, float(r_slow), edge_index


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Precomputing tie-switch topologies (offline)...")
    t0 = time.time()
    base_env = MicrogridEnvDual(
        placement_path=args.placement,
        mpc_path=args.mpc_path,
        seed=args.seed,
    )
    tie_reconfig = TieSwitchReconfiguration(base_env.base_net, seed=args.seed)
    if not tie_reconfig.load_cache("data/tie_switch_cache.pkl"):
        print("Generating tie-switch scenarios...")
        tie_reconfig.generate_scenarios(n=20)
        tie_reconfig.save_cache("data/tie_switch_cache.pkl")
    if not base_env._is_topology_cache_compatible(tie_reconfig._cache):
        print("Regenerating tie-switch scenarios for current DER layout...")
        tie_reconfig.generate_scenarios(n=20)
        tie_reconfig.save_cache("data/tie_switch_cache.pkl")
    topology_cache = tie_reconfig._cache
    print("Warming up topology selector...")
    for ls in np.arange(0.8, 1.3, 0.1):
        for ps in np.arange(0.3, 1.1, 0.1):
            tie_reconfig.select_optimal(round(float(ls), 1), round(float(ps), 1))
    print("Warmup done.")
    print(f"Done: {len(topology_cache)} topologies in {time.time() - t0:.1f}s")

    envs = [
        MicrogridEnvDual(
            placement_path=args.placement,
            mpc_path=args.mpc_path,
            seed=args.seed + i,
            topology_cache=topology_cache,
        )
        for i in range(args.n_envs)
    ]
    menv = MultiEnvDual(envs)
    obs_nodes = int(len(envs[0].net.bus.index)) if envs else int(len(base_env.net.bus.index))

    sample_obs_fast, sample_obs_slow, _ = envs[0].reset(seed=args.seed)
    sample_full_fast = build_full_feeder_obs(envs[0], sample_obs_fast, loop="fast")
    obs_feat = int(sample_full_fast.shape[1])

    if args.encoder == "sage":
        shared_encoder: nn.Module = GraphSAGEEncoder(in_dim=obs_feat, hidden_dim=64, out_dim=64)
    elif args.encoder == "mlp":
        shared_encoder = MLPEncoder(in_dim=obs_feat, hidden_dim=128, out_dim=64)
    else:
        shared_encoder = GATEncoderDual(in_dim=obs_feat, hidden_dim=32, out_dim=64, heads=4)

    policy_fast = LoopPolicy(
        encoder=shared_encoder,
        action_dim=44,
        obs_nodes=obs_nodes,
        obs_feat=obs_feat,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
    )
    policy_slow = LoopPolicy(
        encoder=shared_encoder,
        action_dim=82,
        obs_nodes=obs_nodes,
        obs_feat=obs_feat,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
    )

    buffer_fast = PPOBuffer()
    buffer_slow = PPOBuffer()

    if args.curriculum:
        phase_items = list(DUAL_PHASES.items())
    else:
        phase_items = [
            (
                "FULL",
                {
                    "n_episodes": int(args.n_episodes),
                    "event_probs": {"load_step": 0.3, "gen_trip": 0.3, "line_trip": 0.2, "high_ren": 0.2},
                    "n_topologies": len(topology_cache),
                    "lr_factor": 1.0,
                },
            )
        ]

    base_lr = float(args.lr)
    prev_phase_last_r_fast: float | None = None

    for phase_name, cfg in phase_items:
        if int(args.phase_episodes) > 0:
            n_phase_eps = int(args.phase_episodes)
        else:
            n_phase_eps = int(3 if args.phase_smoke else cfg["n_episodes"])

        if "event_probs" in cfg:
            phase_probs = dict(cfg["event_probs"])
        else:
            phase_probs = {"load_step": 0.3, "gen_trip": 0.3, "line_trip": 0.2, "high_ren": 0.2}
        phase_probs = {str(k): float(v) for k, v in phase_probs.items()}

        phase_topo = int(cfg.get("n_topologies", len(topology_cache)))
        phase_lr = float(base_lr * float(cfg.get("lr_factor", 1.0)))

        print(f"\n=== Phase {phase_name} ({n_phase_eps} ep) ===")
        for env in menv.envs:
            env.event_injector.set_probs(phase_probs)
            env.reconfig.set_active_topologies(phase_topo)

        for pg in policy_fast.optimizer.param_groups:
            pg["lr"] = phase_lr
        for pg in policy_slow.optimizer.param_groups:
            pg["lr"] = phase_lr

        phase_event_count = 0
        phase_line_trip_count = 0

        for episode in range(n_phase_eps):
            ep_r_fast: list[float] = []
            ep_r_slow: list[float] = []

            for env in menv.envs:
                r_f, r_s, _edge = run_episode(
                    env=env,
                    mode=args.mode,
                    policy_fast=policy_fast,
                    policy_slow=policy_slow,
                    buffer_fast=buffer_fast,
                    buffer_slow=buffer_slow,
                )
                ep_r_fast.append(r_f)
                ep_r_slow.append(r_s)
                if env.current_event is not None and bool(getattr(env.current_event, "injected", False)):
                    phase_event_count += 1
                    if str(getattr(env.current_event, "type", "")) == "line_trip":
                        phase_line_trip_count += 1

            r_fast_current = float(np.mean(ep_r_fast)) if ep_r_fast else 0.0
            if episode == 0 and prev_phase_last_r_fast is not None:
                if r_fast_current < prev_phase_last_r_fast - 0.3:
                    print(
                        f"WARNING: r_fast regression detected at phase {phase_name}: "
                        f"{prev_phase_last_r_fast:.3f} -> {r_fast_current:.3f}"
                    )

            loss_f = {"loss": 0.0, "entropy": policy_fast.last_entropy, "value_loss": 0.0}
            loss_s = {"loss": 0.0, "entropy": policy_slow.last_entropy, "value_loss": 0.0}
            entropy_f_per_dim = float(policy_fast.last_entropy / max(policy_fast.action_dim, 1))

            if args.mode in {"fast", "dual"} and ((episode + 1) % max(args.update_freq_fast, 1) == 0):
                loss_f = policy_fast.update(buffer_fast)
                buffer_fast.clear()
                entropy_f_per_dim = float(loss_f.get("entropy", policy_fast.last_entropy) / max(policy_fast.action_dim, 1))

            if args.mode in {"slow", "dual"} and ((episode + 1) % max(args.update_freq_slow, 1) == 0):
                loss_s = policy_slow.update(buffer_slow)
                buffer_slow.clear()

            for k in ["loss", "value_loss", "entropy"]:
                v = loss_f.get(k, 0.0)
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    raise RuntimeError(f"Non-finite fast {k} at phase {phase_name} episode {episode}: {v}")
                v2 = loss_s.get(k, 0.0)
                if isinstance(v2, float) and (np.isnan(v2) or np.isinf(v2)):
                    raise RuntimeError(f"Non-finite slow {k} at phase {phase_name} episode {episode}: {v2}")

            prev_phase_last_r_fast = r_fast_current

            if episode % args.log_interval == 0:
                print(
                    f"phase={phase_name} ep={episode} "
                    f"r_fast={float(np.mean(ep_r_fast)):.3f} "
                    f"r_slow={float(np.mean(ep_r_slow)):.3f} "
                    f"entropy_f={entropy_f_per_dim:.3f} "
                    f"loss_f={float(loss_f.get('loss', 0.0)):.4f} "
                    f"loss_s={float(loss_s.get('loss', 0.0)):.4f}"
                )

        torch.save(
            {
                "encoder": shared_encoder.state_dict(),
                "policy_fast_actor": policy_fast.actor.state_dict(),
                "policy_fast_critic": policy_fast.critic.state_dict(),
                "policy_slow_actor": policy_slow.actor.state_dict(),
                "policy_slow_critic": policy_slow.critic.state_dict(),
                "args": vars(args),
                "encoder_type": str(args.encoder),
                "phase": phase_name,
                "phase_event_count": int(phase_event_count),
                "phase_line_trip_count": int(phase_line_trip_count),
                "phase_lr": float(phase_lr),
            },
            checkpoint_dir / f"phase_{phase_name}_final.pt",
        )
        print(
            f"Phase {phase_name} done: events={phase_event_count} line_trip={phase_line_trip_count} "
            f"lr={phase_lr:.6f} topologies={phase_topo}"
        )

    torch.save(
        {
            "encoder": shared_encoder.state_dict(),
            "policy_fast_actor": policy_fast.actor.state_dict(),
            "policy_fast_critic": policy_fast.critic.state_dict(),
            "policy_slow_actor": policy_slow.actor.state_dict(),
            "policy_slow_critic": policy_slow.critic.state_dict(),
            "args": vars(args),
            "encoder_type": str(args.encoder),
        },
        checkpoint_dir / "dual_final.pt",
    )


if __name__ == "__main__":
    main()
