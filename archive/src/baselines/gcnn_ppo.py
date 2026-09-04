from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.microgrid_env_dual import MicrogridEnvDual

# Import shared Markov process components from main trainer
from src.rl.train_am_mappo import (
    AMRewardConfig,
    compute_am_reward,
    ensure_edge_index,
    build_am_full_feeder_obs,
    RunningNormalizer,
)


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


def ensure_edge_index(edge_index: np.ndarray | torch.Tensor, n_nodes: int = 41) -> np.ndarray:
    ei = np.asarray(edge_index, dtype=np.int64)
    if ei.ndim == 2 and ei.shape[0] == 2 and ei.shape[1] > 0:
        valid = (ei[0] >= 0) & (ei[0] < n_nodes) & (ei[1] >= 0) & (ei[1] < n_nodes)
        ei = ei[:, valid]
    if ei.ndim != 2 or ei.shape[0] != 2 or ei.shape[1] == 0:
        src = np.arange(0, n_nodes - 1, dtype=np.int64)
        dst = np.arange(1, n_nodes, dtype=np.int64)
        ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
    return ei


def build_adj(edge_index: np.ndarray | torch.Tensor, n: int) -> torch.Tensor:
    ei_np = ensure_edge_index(edge_index, n_nodes=n)
    ei = torch.as_tensor(ei_np, dtype=torch.long)
    a = torch.zeros((n, n), dtype=torch.float32)
    if ei.ndim == 2 and ei.shape[0] == 2 and ei.numel() > 0:
        src = ei[0]
        dst = ei[1]
        a[src, dst] = 1.0
    a = a + torch.eye(n, dtype=torch.float32)
    return a


def degree_inv_sqrt(a_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    deg = torch.clamp(a_hat.sum(dim=1), min=eps)
    return torch.pow(deg, -0.5)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: np.ndarray | torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("GCNLayer expects x with shape (N, F)")

        n_nodes = int(x.shape[0])
        a_hat = build_adj(edge_index, n_nodes).to(device=x.device, dtype=x.dtype)
        d_inv_sqrt = degree_inv_sqrt(a_hat)
        norm = d_inv_sqrt.unsqueeze(1) * a_hat * d_inv_sqrt.unsqueeze(0)
        out = norm @ x @ self.weight
        if self.activation is not None:
            out = self.activation(out)
        return out


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int = 10, hidden_dim: int = 64, out_dim: int = 64) -> None:
        super().__init__()
        self.layer1 = GCNLayer(in_dim=in_dim, out_dim=hidden_dim, activation=nn.ReLU())
        self.layer2 = GCNLayer(in_dim=hidden_dim, out_dim=out_dim, activation=None)

    @staticmethod
    def _to_tensor(x: np.ndarray | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype)
        return torch.as_tensor(np.asarray(x), dtype=dtype)

    def encode(self, x: np.ndarray | torch.Tensor, edge_index: np.ndarray | torch.Tensor) -> torch.Tensor:
        x_t = self._to_tensor(x, dtype=torch.float32)
        if x_t.ndim != 2:
            raise ValueError("encode expects x with shape (N, F)")
        h = self.layer1(x_t, edge_index)
        h = self.layer2(h, edge_index)
        return h

    def encode_batch(self, x_batch: np.ndarray | torch.Tensor, edge_index: np.ndarray | torch.Tensor) -> torch.Tensor:
        x_t = self._to_tensor(x_batch, dtype=torch.float32)
        if x_t.ndim != 3:
            raise ValueError("encode_batch expects x_batch with shape (B, N, F)")

        out = []
        for b in range(x_t.shape[0]):
            h = self.layer1(x_t[b], edge_index)
            h = self.layer2(h, edge_index)
            out.append(h)
        return torch.stack(out, dim=0)

    def forward(self, x: np.ndarray | torch.Tensor, edge_index: np.ndarray | torch.Tensor) -> torch.Tensor:
        x_t = self._to_tensor(x, dtype=torch.float32)
        if x_t.ndim == 2:
            return self.encode(x_t, edge_index)
        if x_t.ndim == 3:
            return self.encode_batch(x_t, edge_index)
        raise ValueError("x must have shape (N, F) or (B, N, F)")


class GCNActorPPO(nn.Module):
    def __init__(self, in_dim: int = 10, hidden_dim: int = 64, emb_dim: int = 64, action_dim: int = 123, global_obs_dim: int = 410) -> None:
        super().__init__()
        self.encoder = GCNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=emb_dim)
        # Spec-faithful GCNN-PPO: both heads consume ONLY the global summation-pooled
        # GCN embedding (2 GCN layers -> sum-pool over all nodes -> MLP), so the GCN
        # is the sole state representation (clean GCN-vs-GraphSAGE ablation). The raw
        # flattened global observation is NOT concatenated. `global_obs_dim` is kept
        # in the signature for backward-compatible construction but is unused.
        feature_dim = emb_dim
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Init log_std = -1.0 (std ~ 0.37 via softplus) to mirror train_am_mappo.py.
        # Zero-init gave std=0.693, entropy stuck and slowly growing under value_loss-dominated gradients.
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    @staticmethod
    def _to_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.as_tensor(np.asarray(x), dtype=torch.float32)

    def _features(
        self,
        node_obs: np.ndarray | torch.Tensor,
        edge_index: np.ndarray | torch.Tensor,
        global_obs: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Global summation pooling of the 2-layer GCN embedding (graph -> vector)."""
        dev = self.encoder.layer1.weight.device
        x_t = self._to_tensor(node_obs).to(dev)
        h = self.encoder(x_t, edge_index)
        if h.ndim == 2:        # (N, F) single sample
            return h.sum(dim=0, keepdim=True)
        if h.ndim == 3:        # (B, N, F) batch
            return h.sum(dim=1)
        raise ValueError("Encoder output must have shape (N, F) or (B, N, F)")

    def dist(
        self,
        node_obs: np.ndarray | torch.Tensor,
        edge_index: np.ndarray | torch.Tensor,
        global_obs: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[torch.distributions.Normal, torch.Tensor]:
        z = self._features(node_obs, edge_index)
        mean = torch.tanh(self.actor(z))
        std = torch.nn.functional.softplus(self.log_std).expand_as(mean) + 1e-4
        std = torch.clamp(std, min=1e-4, max=2.0)
        value = self.critic(z).squeeze(-1)
        return torch.distributions.Normal(mean, std), value


class GCNNPPOAgent:
    def __init__(
        self,
        env: MicrogridEnvDual,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.1,
        entropy_coef: float = 0.01,
        lr: float = 3e-4,
        n_epochs: int = 4,
        mini_batch_size: int = 64,
        log_std_min: float = -3.0,
        log_std_max: float = 0.5,
        normalize_returns: bool = True,
        device: str | None = None,
    ) -> None:
        """GCNN-PPO agent.

        Coef defaults mirror train_am_mappo.py: entropy_coef=0.01, value_coef
        lowered to 0.1 because the critic's raw value_loss is on the order
        of 100-400 (vs. policy_loss O(1)). n_epochs=4 / mini_batch_size=64
        mirrors the reference PPO inner loop. log_std clamp bounds prevent
        runaway growth under value-dominated gradients.
        """
        self.env = env

        # Get observation dimension from build_am_full_feeder_obs (same as train_am_mappo.py)
        sample_obs_fast, _, _ = env.reset()
        sample_obs_full = build_am_full_feeder_obs(env, sample_obs_fast)
        obs_feat = int(sample_obs_full.shape[1])  # 20 features from build_am_full_feeder_obs
        n_bus = int(sample_obs_full.shape[0])     # 123 buses
        global_obs_dim = n_bus * obs_feat         # Flattened observation size

        # Action dim = n_agents * 2 (same as train_am_mappo.py: ctrl_p + ctrl_k per agent)
        self.n_agents = env.n_agents  # 41
        action_dim = self.n_agents * 2  # 82

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = GCNActorPPO(
            in_dim=obs_feat, hidden_dim=64, emb_dim=64,
            action_dim=action_dim, global_obs_dim=global_obs_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = PPOBuffer()

        # Observation normalizer (parity with the proposed GraphSAGE-MAPPO method).
        # Updated with raw obs during rollout; the NORMALIZED obs is what gets
        # stored in the buffer and fed to the policy, so the PPO update path is
        # automatically consistent. Frozen and reused at eval (saved in checkpoint).
        self.obs_normalizer = RunningNormalizer(sample_obs_full.shape)

        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.n_epochs = int(n_epochs)
        self.mini_batch_size = int(mini_batch_size)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.normalize_returns = bool(normalize_returns)
        self.last_entropy = 0.0

    @staticmethod
    def _combine_obs(obs_fast: np.ndarray, obs_slow: np.ndarray) -> np.ndarray:
        obs_f = np.asarray(obs_fast, dtype=np.float32)
        obs_s = np.asarray(obs_slow, dtype=np.float32)
        if obs_f.ndim != 2 or obs_s.ndim != 2 or obs_f.shape[0] != 41 or obs_s.shape[0] != 41:
            raise ValueError(f"Expected obs_fast/obs_slow shape (41,F), got {obs_f.shape}/{obs_s.shape}")
        if obs_f.shape[1] < 5 or obs_s.shape[1] < 5:
            raise ValueError(f"Expected at least 5 features, got {obs_f.shape}/{obs_s.shape}")
        return np.concatenate([obs_f[:, :5], obs_s[:, :5]], axis=1)

    def _map_action(self, action123: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(action123, dtype=np.float32).reshape(41, 3)
        p_all = a[:, 0]
        q_all = a[:, 1]
        droop_all = a[:, 2]

        # Aggregate to 3 VPP-level droops
        vpp_droop = np.zeros(3, dtype=np.float32)
        vpp_agents = [list(range(9, 12)) + list(range(18, 21)),
                      list(range(12, 15)) + list(range(21, 24)),
                      list(range(15, 18)) + list(range(24, 27))]
        for i, agents in enumerate(vpp_agents):
            vpp_droop[i] = droop_all[agents].mean()

        action_slow = np.concatenate([p_all, q_all], axis=0).astype(np.float32)
        action_fast = np.concatenate([p_all, vpp_droop], axis=0).astype(np.float32)

        if action_fast.shape != (44,):
            raise ValueError(f"action_fast must have shape (44,), got {action_fast.shape}")
        if action_slow.shape != (82,):
            raise ValueError(f"action_slow must have shape (82,), got {action_slow.shape}")

        return np.clip(action_fast, -1.0, 1.0), np.clip(action_slow, -1.0, 1.0)

    def act(
        self,
        node_obs: np.ndarray,
        edge_index: np.ndarray,
        global_obs: np.ndarray,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, np.ndarray]:
        """Sample an action.

        Returns (action_env, log_prob, value, raw_action):
          - action_env: clamped to [-1, 1] for the environment
          - log_prob:   log_prob of the *raw* sample under the Normal (no clamp bias)
          - value:      critic estimate
          - raw_action: unclamped sample, stored in buffer so updates use the
                        same distribution that produced it (matches PPO theory)
        """
        dist, value = self.model.dist(node_obs, edge_index, global_obs)
        raw = dist.sample()
        log_prob = dist.log_prob(raw).sum(dim=-1)
        action_env = torch.clamp(raw, -1.0, 1.0)
        return (
            action_env.squeeze(0).detach().cpu().numpy(),
            log_prob.squeeze(0),
            value.squeeze(0),
            raw.squeeze(0).detach().cpu().numpy(),
        )

    @torch.no_grad()
    def act_deterministic(
        self,
        node_obs: np.ndarray,
        edge_index: np.ndarray,
        global_obs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Deterministic action (distribution mean, no sampling) for evaluation.

        Normalizes the raw observation with the frozen obs_normalizer, then returns
        the clamped tanh-mean. Used by the eval policy wrapper.
        """
        obs_n = self.obs_normalizer.normalize(np.asarray(node_obs, dtype=np.float32)).astype(np.float32)
        dist, _value = self.model.dist(obs_n, edge_index)
        action_env = torch.clamp(dist.mean, -1.0, 1.0)
        return action_env.squeeze(0).detach().cpu().numpy()

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

    def _update_model(self) -> dict[str, float]:
        """PPO update: n_epochs of mini-batched gradient steps.

        Mirrors train_am_mappo.py:710-713 (4 epochs x mini_batch 64). The
        previous one-step-per-rollout regime meant ratio~1.0 always, clip
        never engaged, and the value loss dominated -> policy never moved.

        Returns are optionally normalized (per call) so that value_loss
        magnitude does not crush policy/entropy gradients under
        clip_grad_norm.
        """
        if len(self.buffer) == 0:
            return {"loss": 0.0, "entropy": 0.0, "value_loss": 0.0, "policy_loss": 0.0}

        dev = self.device
        obs = torch.stack(self.buffer.obs).to(dev)
        edge_idx_list = [ei for ei in self.buffer.edge_index]
        actions = torch.stack(self.buffer.actions).to(dev)
        old_log_probs_all = torch.stack(self.buffer.log_probs).detach().to(dev)
        values_np = torch.stack(self.buffer.values).detach().cpu().numpy().astype(np.float32)
        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        dones = np.asarray(self.buffer.dones, dtype=np.float32)

        advantages_np, returns_np = self._compute_gae(rewards, values_np, dones)
        advantages_np = np.nan_to_num(advantages_np, nan=0.0, posinf=0.0, neginf=0.0)
        returns_np = np.nan_to_num(returns_np, nan=0.0, posinf=0.0, neginf=0.0)
        adv_all = torch.tensor(
            (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8),
            dtype=torch.float32, device=dev,
        )
        if self.normalize_returns:
            ret_mean = float(returns_np.mean())
            ret_std = float(returns_np.std()) + 1e-8
            ret_all = torch.tensor((returns_np - ret_mean) / ret_std, dtype=torch.float32, device=dev)
        else:
            ret_all = torch.tensor(returns_np, dtype=torch.float32, device=dev)

        # Pre-build topo group key per buffer index (edge_index bytes)
        edge_keys: list[bytes] = []
        edge_tensors: dict[bytes, torch.Tensor] = {}
        for edge_i in edge_idx_list:
            edge_t = edge_i if isinstance(edge_i, torch.Tensor) else torch.as_tensor(edge_i, dtype=torch.long)
            key = edge_t.detach().cpu().numpy().tobytes()
            edge_keys.append(key)
            if key not in edge_tensors:
                edge_tensors[key] = edge_t

        n_transitions = len(self.buffer)
        idx_all = np.arange(n_transitions)

        loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        n_updates = 0

        for _epoch in range(self.n_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n_transitions, self.mini_batch_size):
                mb_idx = idx_all[start : start + self.mini_batch_size]
                if mb_idx.size == 0:
                    continue
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=dev)

                # Group this minibatch by topology so we can batch the GCN call
                topo_to_mb_indices: dict[bytes, list[int]] = {}
                for j, buf_i in enumerate(mb_idx):
                    topo_to_mb_indices.setdefault(edge_keys[buf_i], []).append(int(buf_i))

                mb_log_probs = torch.zeros(mb_idx.size, dtype=torch.float32, device=dev)
                mb_entropy_each = torch.zeros(mb_idx.size, dtype=torch.float32, device=dev)
                mb_value_pred = torch.zeros(mb_idx.size, dtype=torch.float32, device=dev)

                # Map original buffer index -> position within the minibatch
                buf_to_mb_pos = {int(buf_i): j for j, buf_i in enumerate(mb_idx)}

                for key, buf_indices in topo_to_mb_indices.items():
                    edge_t = edge_tensors[key]
                    buf_idx_t = torch.as_tensor(buf_indices, dtype=torch.long, device=dev)
                    obs_batch = obs.index_select(0, buf_idx_t)
                    action_batch = actions.index_select(0, buf_idx_t)

                    dist_batch, value_batch = self.model.dist(obs_batch, edge_t, obs_batch)
                    lp = dist_batch.log_prob(action_batch).sum(dim=-1)
                    ent = dist_batch.entropy().sum(dim=-1)

                    target_positions = torch.as_tensor(
                        [buf_to_mb_pos[bi] for bi in buf_indices], dtype=torch.long, device=dev
                    )
                    mb_log_probs.index_copy_(0, target_positions, lp)
                    mb_entropy_each.index_copy_(0, target_positions, ent)
                    mb_value_pred.index_copy_(0, target_positions, value_batch)

                mb_adv = adv_all.index_select(0, mb_idx_t)
                mb_ret = ret_all.index_select(0, mb_idx_t)
                mb_old_log_probs = old_log_probs_all.index_select(0, mb_idx_t)

                if (
                    not torch.isfinite(mb_log_probs).all()
                    or not torch.isfinite(mb_value_pred).all()
                    or not torch.isfinite(mb_adv).all()
                ):
                    continue

                ratio = torch.exp(mb_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = torch.mean((mb_ret - mb_value_pred) ** 2)
                entropy = mb_entropy_each.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                # Hard clamp on raw log_std to prevent runaway growth.
                with torch.no_grad():
                    self.model.log_std.data.clamp_(self.log_std_min, self.log_std_max)

                loss_sum += float(loss.item())
                policy_loss_sum += float(policy_loss.item())
                value_loss_sum += float(value_loss.item())
                entropy_sum += float(entropy.item())
                n_updates += 1

        if n_updates == 0:
            return {"loss": 0.0, "entropy": float(self.last_entropy), "value_loss": 0.0, "policy_loss": 0.0}

        avg_loss = loss_sum / n_updates
        avg_policy = policy_loss_sum / n_updates
        avg_value = value_loss_sum / n_updates
        avg_entropy = entropy_sum / n_updates
        self.last_entropy = float(avg_entropy)
        return {
            "loss": avg_loss,
            "policy_loss": avg_policy,
            "entropy": avg_entropy,
            "value_loss": avg_value,
        }

    def rollout_episode(self) -> tuple[float, float]:
        """Rollout episode synchronized with train_am_mappo.py Markov process.

        Returns (episode_reward_fast, episode_reward_slow).
        episode_reward_fast is the SUM of per-step am_reward across the
        300-step fast episode, matching train_mlp_mappo.py's episode_reward
        (line 449). Previously this was np.mean(), which put GCNN-PPO on a
        different numerical scale than MLP-MAPPO comparison plots.
        """
        reward_cfg = AMRewardConfig()

        # Reset and build full feeder observation (same as train_am_mappo.py)
        obs_fast, obs_slow, info = self.env.reset()
        n_bus = int(len(self.env.net.bus.index))
        edge_index = ensure_edge_index(self.env.edge_index, n_nodes=n_bus)
        obs_full = build_am_full_feeder_obs(self.env, obs_fast)

        r_fast_list: list[float] = []
        prev_action: np.ndarray | None = None
        prev_k_droop = np.zeros(self.env.n_agents, dtype=np.float32)

        for _ in range(300):
            # Update normalizer with raw obs, then feed/store the NORMALIZED obs so
            # rollout and the PPO update path use identical inputs.
            self.obs_normalizer.update(obs_full)
            obs_n = self.obs_normalizer.normalize(obs_full).astype(np.float32)
            global_obs = obs_n.reshape(-1)
            action_env_flat, log_prob, value, raw_action_flat = self.act(
                obs_n, edge_index, global_obs
            )

            # Use clamped action for env stepping; raw action (unclamped) is
            # stored in the buffer so updated log_probs match the sampling
            # distribution (no clamp bias for boundary samples).
            policy_actions = action_env_flat.reshape(self.n_agents, 2)

            # Build full action (same logic as train_am_mappo.py)
            ctrl_p = -policy_actions[:, 0]  # Sign flip for frequency response
            ctrl_k = policy_actions[:, 1]

            n_vpps = len(self.env._vpp_droop_agents)
            if self.env.ffr_mode == "mappo_dual" and ctrl_k is not None:
                full_action = np.zeros(2 * self.env.n_agents + n_vpps, dtype=np.float32)
                full_action[: self.env.n_agents] = ctrl_p[: self.env.n_agents]
                full_action[self.env.n_agents : 2 * self.env.n_agents] = ctrl_k[: self.env.n_agents]
            else:
                full_action = np.zeros(self.env.n_agents + n_vpps, dtype=np.float32)
                full_action[: self.env.n_agents] = ctrl_p[: self.env.n_agents]
                for vpp_idx, (_vpp_id, member_agents) in enumerate(self.env._vpp_droop_agents.items()):
                    vpp_action = np.mean([ctrl_p[ai] for ai in member_agents if ai < len(ctrl_p)])
                    full_action[self.env.n_agents + vpp_idx] = vpp_action

            control_actions = ctrl_p.reshape(-1, 1) if ctrl_k is None else np.stack([ctrl_p, ctrl_k], axis=1)

            next_obs_fast, _r_env, done, _, info_fast = self.env.step_fast(full_action)

            # Compute reward using compute_am_reward (same as train_am_mappo.py),
            # on the COI frequency deviation: the resolvable system-frequency
            # observable at the 1 s control step, identical across all methods.
            post_freq_state = self.env.freq_dyn_lti.get_state()
            post_delta_f = float(post_freq_state.delta_f_hz)
            post_rocof = float(post_freq_state.rocof_hz_s)
            post_freq_hz = 50.0 + post_delta_f

            soc_vec = np.asarray(next_obs_fast[:, 3], dtype=np.float32) if next_obs_fast.ndim == 2 else None
            zone_lmp_vec = (
                np.asarray(next_obs_fast[:, 4], dtype=np.float32) if (next_obs_fast.ndim == 2 and next_obs_fast.shape[1] > 4) else None
            )
            k_droop_now = getattr(self.env, "_k_droop_last", None)
            k_droop_max = getattr(self.env, "_k_droop_max_per_agent", None)
            p_ref_now = getattr(self.env, "_p_ref_last", None)

            am_reward, _ = compute_am_reward(
                delta_f=post_delta_f,
                rocof=post_rocof,
                action=control_actions.flatten(),
                prev_action=prev_action,
                freq_hz=post_freq_hz,
                cfg=reward_cfg,
                soc=soc_vec,
                zone_lmp=zone_lmp_vec,
                k_droop_now=k_droop_now,
                k_droop_prev=prev_k_droop,
                p_ref_now=p_ref_now,
                k_droop_max=k_droop_max,
                lambda_as_ffr=getattr(self.env, "lambda_as_ffr", None),
            )

            if k_droop_now is not None:
                prev_k_droop = np.asarray(k_droop_now, dtype=np.float32).copy()

            n_bus = int(len(self.env.net.bus.index))
            next_edge = ensure_edge_index(info_fast.get("edge_index", edge_index), n_nodes=n_bus)

            self.buffer.add(
                obs=torch.tensor(obs_n, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                action=torch.tensor(raw_action_flat, dtype=torch.float32),
                log_prob=log_prob,
                value=value,
                reward=float(am_reward),
                done=done,
            )

            prev_action = control_actions.flatten().copy()
            obs_fast = next_obs_fast
            edge_index = next_edge
            obs_full = build_am_full_feeder_obs(self.env, obs_fast)
            r_fast_list.append(float(am_reward))

            if done:
                break

        # Slow step (for compatibility, though main focus is fast-timescale FFR)
        self.obs_normalizer.update(obs_full)
        obs_n_slow = self.obs_normalizer.normalize(obs_full).astype(np.float32)
        global_obs = obs_n_slow.reshape(-1)
        action_env_flat_slow, log_prob_slow, value_slow, raw_action_flat_slow = self.act(
            obs_n_slow, edge_index, global_obs
        )

        # set_slow_baseline expects (82,) = n_agents * 2, use clamped action
        self.env.set_slow_baseline(action_env_flat_slow)

        # step_slow expects (41,) or (82,), extract P from clamped action
        policy_actions_slow = action_env_flat_slow.reshape(self.n_agents, 2)
        action_slow_p = policy_actions_slow[:, 0]
        next_obs_slow, r_slow, done_s, _, _ = self.env.step_slow(action_slow_p)

        self.buffer.add(
            obs=torch.tensor(obs_n_slow, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            action=torch.tensor(raw_action_flat_slow, dtype=torch.float32),
            log_prob=log_prob_slow,
            value=value_slow,
            reward=float(r_slow),
            done=bool(done_s),
        )

        # Sum (not mean) to align with train_mlp_mappo.py episode_reward scale.
        r_fast_total = float(np.sum(r_fast_list)) if r_fast_list else 0.0
        return r_fast_total, float(r_slow)

    def update(self, n_episodes: int = 3) -> dict[str, float]:
        fast_rewards: list[float] = []
        slow_rewards: list[float] = []

        self.buffer.clear()
        for _ in range(int(n_episodes)):
            r_f, r_s = self.rollout_episode()
            fast_rewards.append(r_f)
            slow_rewards.append(r_s)

        metrics = self._update_model()
        self.buffer.clear()

        metrics["r_fast"] = float(np.mean(fast_rewards)) if fast_rewards else 0.0
        metrics["r_slow"] = float(np.mean(slow_rewards)) if slow_rewards else 0.0
        # entropy is summed over action_dim by the Normal dist; divide by
        # the actual action_dim, not the hardcoded 123 used previously
        action_dim = int(self.model.actor[-1].out_features)
        metrics["entropy_per_dim"] = float(metrics.get("entropy", 0.0) / max(action_dim, 1))

        for key, value in metrics.items():
            if not np.isfinite(float(value)):
                raise RuntimeError(f"Non-finite metric {key}: {value}")

        return metrics

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        actor_out = self.model.actor[-1]
        critic_in = self.model.critic[0]
        enc_in = self.model.encoder.layer1.weight
        action_dim = int(actor_out.out_features)
        feature_dim = int(critic_in.in_features)
        in_dim = int(enc_in.shape[0])
        emb_dim = int(self.model.encoder.layer2.weight.shape[1])
        global_obs_dim = feature_dim - emb_dim
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": {
                    "in_dim": in_dim,
                    "hidden_dim": 64,
                    "emb_dim": emb_dim,
                    "action_dim": action_dim,
                    "global_obs_dim": global_obs_dim,
                },
                "obs_normalizer": {
                    "mean": self.obs_normalizer.mean,
                    "var": self.obs_normalizer.var,
                    "count": self.obs_normalizer.count,
                },
            },
            path,
        )

    @classmethod
    def load(cls, checkpoint_path: Path) -> "GCNNPPOAgent":
        """Load a GCNNPPOAgent checkpoint.

        Reads model dims from the saved 'config' dict (new checkpoints).
        For legacy checkpoints with no config, auto-infer dims from the
        state_dict shapes so old runs can still be evaluated.
        """
        ckpt = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        sd = ckpt["model"]
        cfg = ckpt.get("config") or {}

        # Infer from state_dict shapes when config absent (legacy ckpts).
        # state_dict layout:
        #   log_std                  : (action_dim,)
        #   encoder.layer1.weight    : (in_dim, hidden_dim)
        #   encoder.layer2.weight    : (hidden_dim, emb_dim)
        #   actor.0.weight           : (256, emb_dim + global_obs_dim)
        def _infer(key: str, default: int) -> int:
            val = cfg.get(key)
            if val is not None:
                return int(val)
            if key == "action_dim":
                return int(sd["log_std"].shape[0])
            if key == "in_dim":
                return int(sd["encoder.layer1.weight"].shape[0])
            if key == "hidden_dim":
                return int(sd["encoder.layer1.weight"].shape[1])
            if key == "emb_dim":
                return int(sd["encoder.layer2.weight"].shape[1])
            if key == "global_obs_dim":
                feature_dim = int(sd["actor.0.weight"].shape[1])
                emb_dim = int(sd["encoder.layer2.weight"].shape[1])
                return feature_dim - emb_dim
            return default

        in_dim = _infer("in_dim", 10)
        hidden_dim = _infer("hidden_dim", 64)
        emb_dim = _infer("emb_dim", 64)
        action_dim = _infer("action_dim", 82)
        global_obs_dim = _infer("global_obs_dim", 0)

        obj = cls.__new__(cls)
        obj.model = GCNActorPPO(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
            action_dim=action_dim,
            global_obs_dim=global_obs_dim,
        )
        obj.model.load_state_dict(sd)
        obj.model.eval()
        obj.env = None
        obj.last_entropy = 0.0

        # Restore obs normalizer (frozen) for deterministic eval; fall back to a
        # per-feature identity normalizer for legacy checkpoints without it.
        norm = ckpt.get("obs_normalizer")
        if norm is not None:
            rn = RunningNormalizer(np.asarray(norm["mean"]).shape)
            rn.mean = np.asarray(norm["mean"], dtype=np.float64)
            rn.var = np.asarray(norm["var"], dtype=np.float64)
            rn.count = float(norm["count"])
            obj.obs_normalizer = rn
        else:
            obj.obs_normalizer = RunningNormalizer((in_dim,))
        return obj


def smoke_gate(
    n_updates: int = 5,
    save_load_check: bool = True,
) -> dict[str, list[float]]:
    """Gate for the GCNN-PPO fixes.

    Verifies:
      1. Module imports + agent constructs on a real env.
      2. n_updates of update() run without NaN / shape errors.
      3. value_loss stays bounded (was 100-400 before fix, < 10 expected).
      4. entropy stays in a sensible range (was stuck at ~86 = 1.06/dim
         before fix; with log_std init=-1.0, expected per-dim ~0.2-0.4).
      5. save() / load() round-trip preserves action_dim (Step 2 fix).

    r_fast improvement is logged but not asserted at n=5 (too noisy for
    a short smoke); the real learning gate is the 200-ep mini-run.

    Returns per-update trace of entropy/value_loss/r_fast.
    """
    import tempfile
    import numpy as np

    env = MicrogridEnvDual(
        placement_path="artifacts/placement/official_placement_v3.json",
        mpc_path="data/grid_IEEE123_complete.m",
        seed=42,
    )
    agent = GCNNPPOAgent(env)

    entropies: list[float] = []
    value_losses: list[float] = []
    r_fasts: list[float] = []
    for k in range(int(n_updates)):
        metrics = agent.update(n_episodes=1)
        assert metrics is not None
        entropies.append(float(metrics["entropy"]))
        value_losses.append(float(metrics["value_loss"]))
        r_fasts.append(float(metrics.get("r_fast", float("nan"))))
        print(
            f"  smoke update {k+1}/{n_updates}: "
            f"r_fast={metrics.get('r_fast', float('nan')):.3f}  "
            f"entropy={metrics['entropy']:.3f}  "
            f"entropy_per_dim={metrics['entropy_per_dim']:.4f}  "
            f"value_loss={metrics['value_loss']:.3f}  "
            f"policy_loss={metrics.get('policy_loss', float('nan')):.4f}"
        )

    # --- assertions (Step 8 gate) ---
    max_value_loss = max(value_losses)
    assert max_value_loss < 50.0, (
        f"value_loss = {max_value_loss:.2f} > 50 -- critic still dominated. "
        f"Pre-fix log showed 100-400; fix is broken."
    )
    last_ent_per_dim = entropies[-1] / max(int(agent.model.actor[-1].out_features), 1)
    assert 0.05 < last_ent_per_dim < 1.0, (
        f"entropy_per_dim = {last_ent_per_dim:.3f} out of sane range (0.05, 1.0). "
        f"Pre-fix run was stuck at 1.06 -- log_std init/clamp may be wrong."
    )
    if n_updates >= 2:
        delta_r = r_fasts[-1] - r_fasts[0]
        print(f"  r_fast first={r_fasts[0]:.3f} last={r_fasts[-1]:.3f} delta={delta_r:+.3f}")

    if save_load_check:
        action_dim_before = int(agent.model.actor[-1].out_features)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
        try:
            agent.save(tmp_path)
            agent2 = GCNNPPOAgent.load(tmp_path)
            action_dim_after = int(agent2.model.actor[-1].out_features)
            assert action_dim_after == action_dim_before, (
                f"action_dim mismatch after load: {action_dim_after} vs {action_dim_before}"
            )
            print(f"  save/load round-trip OK (action_dim={action_dim_after})")
        finally:
            from pathlib import Path as _Path
            try:
                _Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    print("GCNN-PPO GATE: PASS")
    return {"entropy": entropies, "value_loss": value_losses, "r_fast": r_fasts}


def train_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train GCNN-PPO baseline")
    parser.add_argument("--n-episodes", type=int, default=2000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/baseline_gcnn_ppo"))
    parser.add_argument("--placement", type=Path, default=Path("artifacts/placement/official_placement_v3.json"))
    parser.add_argument("--mpc-path", type=Path, default=Path("data/grid_IEEE123_complete.m"))
    parser.add_argument("--precomputed-dir", type=Path, default=Path("data/precomputed_365d_97to67"))
    parser.add_argument("--save-every", type=int, default=200, help="Save checkpoint every N episodes")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N episodes")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        placement_path=str(args.placement),
        mpc_path=str(args.mpc_path),
        precomputed_dir=str(args.precomputed_dir),
    )
    envs = [MicrogridEnvDual(seed=args.seed + i, **env_kwargs) for i in range(args.n_envs)]
    agent = GCNNPPOAgent(envs[0])

    n_envs = args.n_envs
    n_updates = max(1, args.n_episodes // n_envs)
    best_r_fast = -float("inf")

    print(f"[gcnn_ppo] n_episodes={args.n_episodes}  n_envs={n_envs}  updates={n_updates}")

    for update in range(1, n_updates + 1):
        ep_done = update * n_envs
        agent.buffer.clear()
        r_fasts, r_slows = [], []

        for env_i in envs:
            agent.env = env_i
            try:
                r_f, r_s = agent.rollout_episode()
            except Exception as exc:
                print(f"  [ep={ep_done}] rollout error: {exc}")
                continue
            r_fasts.append(r_f)
            r_slows.append(r_s)

        if not r_fasts:
            agent.buffer.clear()
            continue

        try:
            metrics = agent._update_model()
        except Exception as exc:
            print(f"  [ep={ep_done}] update error: {exc}")
            agent.buffer.clear()
            continue
        agent.buffer.clear()

        r_fast_mean = float(np.mean(r_fasts))
        r_slow_mean = float(np.mean(r_slows))

        if r_fast_mean > best_r_fast:
            best_r_fast = r_fast_mean
            agent.save(ckpt_dir / "best.pt")

        if ep_done % args.log_every == 0 or update == 1:
            loss = metrics.get("loss", float("nan"))
            entropy = metrics.get("entropy", float("nan"))
            print(f"  ep={ep_done:5d}  r_fast={r_fast_mean:.4f}  r_slow={r_slow_mean:.4f}"
                  f"  loss={loss:.2f}  entropy={entropy:.2f}")

        if ep_done % args.save_every == 0:
            agent.save(ckpt_dir / f"ep_{ep_done:05d}.pt")
            agent.save(ckpt_dir / "latest.pt")

    agent.save(ckpt_dir / "final.pt")
    print(f"[gcnn_ppo] Saved → {ckpt_dir / 'final.pt'}")


if __name__ == "__main__":
    train_main()
