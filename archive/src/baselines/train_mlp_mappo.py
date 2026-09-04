"""
MLP-MAPPO Ablation Baseline for Ancillary Market.

Identical to GraphSAGE-MAPPO except the encoder:
- Replaces GraphSAGE message-passing with MLP (no graph structure)
- Proves that graph awareness is essential for topology generalization

This is the most important ablation in the paper (Section VI-D):
- Same RL algorithm (MAPPO), same critic, same reward, same training schedule
- Only difference: MLP encoder ignores edge_index (no message passing)

Expected result:
- Train topologies: MLP-MAPPO competitive (memorizes training distribution)
- Unseen topologies: MLP-MAPPO degrades significantly
- IAE degradation vs d_E: steep curve (vs flat for GraphSAGE)
"""
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
from torch_geometric.nn import global_mean_pool, global_max_pool

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.microgrid_env_dual import MicrogridEnvDual

# Import shared components from main MAPPO trainer
from src.rl.train_am_mappo import (
    AM_PHASES,
    AMRewardConfig,
    RunningNormalizer,
    RewardNormalizer,
    compute_am_reward,
    SharedGaussianActor,
    AgentCenteredCritic,
    ensure_edge_index,
    get_am_obs,
    build_am_full_feeder_obs,
)


# ============================================================================
# MLP Encoder (ignores graph structure)
# ============================================================================

class MLPAgentEncoder(nn.Module):
    """MLP encoder that ignores graph structure.

    Same interface as FeederGraphSAGEAgentEncoder but edge_index is ignored.
    This isolates the graph contribution: if MLP-MAPPO degrades on unseen
    topologies while GraphSAGE-MAPPO maintains performance, the graph
    message-passing is proven necessary.
    """

    def __init__(self, obs_feat: int, hidden_dim: int = 64, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_feat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        x_full: torch.Tensor,
        edge_index: torch.Tensor,  # IGNORED
        agent_bus_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_full: Node features [n_bus, obs_feat]
            edge_index: Graph edges [2, E] - IGNORED
            agent_bus_idx: Bus indices for agents [n_agents]

        Returns:
            Agent embeddings [n_agents, embed_dim]
        """
        # Encode ALL nodes with MLP (no message passing)
        node_embeds = self.net(x_full)
        agent_embeds = node_embeds[agent_bus_idx]
        # Frequency skip-connection: MIRRORS FeederGraphSAGEAgentEncoder exactly so
        # the ablation stays clean — both variants feed the actor [embed | delta_f,
        # rocof], differing ONLY in how node_embeds is computed (MLP vs message
        # passing). The MLP's learned nonlinear embedding does not preserve the raw
        # delta_f/rocof, so the shortcut is as meaningful here as in the GNN; keeping
        # it on both sides means any GraphSAGE win is attributable to the graph
        # encoder, not to extra raw-frequency features at the actor input.
        freq_raw = x_full[agent_bus_idx, :2]  # (n_agents, 2): delta_f_norm, rocof_norm
        return torch.cat([agent_embeds, freq_raw], dim=-1)


# ============================================================================
# MLP-MAPPO Agent
# ============================================================================

class MLPMAPPOAgent(nn.Module):
    """MLP-MAPPO agent for AM control (ablation baseline)."""

    def __init__(
        self,
        obs_feat: int,
        n_agents: int,
        n_bus: int,
        agent_bus_indices: np.ndarray,
        action_dim_per_agent: int = 2,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        lr: float = 3e-4,
        max_grad_norm: float = 0.5,
        log_std_init: float = -1.0,
        min_std: float = 0.05,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_bus = n_bus
        self.obs_feat = obs_feat
        self.action_dim = action_dim_per_agent
        self.embed_dim = embed_dim
        self.agent_bus_indices = torch.as_tensor(agent_bus_indices, dtype=torch.long)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # MLP Encoder (key difference from GraphSAGE-MAPPO)
        self.encoder = MLPAgentEncoder(obs_feat, hidden_dim=embed_dim, embed_dim=embed_dim)

        # Actor and Critic (same as GraphSAGE-MAPPO). +2 for the frequency
        # skip-connection (delta_f, rocof) mirrored from FeederGraphSAGEAgentEncoder
        # so the actor/critic architecture is byte-identical across the ablation.
        _actor_in = embed_dim + 2
        self.actor = SharedGaussianActor(
            embed_dim=_actor_in,
            action_dim=action_dim_per_agent,
            hidden_dim=hidden_dim,
            log_std_init=log_std_init,
            min_std=min_std,
        )
        self.critic = AgentCenteredCritic(embed_dim=_actor_in, hidden_dim=hidden_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)

    def get_agent_embeddings(
        self,
        obs: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observations to agent embeddings."""
        agent_bus_idx = self.agent_bus_indices.to(obs.device)
        return self.encoder(obs, edge_index, agent_bus_idx)

    def act(
        self,
        obs: torch.Tensor,
        edge_index: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select actions for all agents."""
        agent_embeds = self.get_agent_embeddings(obs, edge_index)
        dist = self.actor.dist(agent_embeds)

        if deterministic:
            actions = dist.mean
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.critic(agent_embeds).squeeze(-1)

        return actions, log_probs, values

    @torch.no_grad()
    def act_deterministic(
        self,
        obs: np.ndarray | torch.Tensor,
        edge_index: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Deterministic action selection for evaluation."""
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)

        agent_embeds = self.get_agent_embeddings(obs, edge_index)
        mean, _ = self.actor(agent_embeds)
        return mean.cpu().numpy()

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        edge_index: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        agent_embeds = self.get_agent_embeddings(obs, edge_index)
        dist = self.actor.dist(agent_embeds)

        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(agent_embeds).squeeze(-1)

        return log_probs, values, entropy

    def compute_gae(
        self,
        rewards: list[float],
        values: list[torch.Tensor],
        dones: list[bool],
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        device = next_value.device
        T = len(rewards)

        advantages = torch.zeros(T, self.n_agents, device=device)
        returns = torch.zeros(T, self.n_agents, device=device)

        gae = torch.zeros(self.n_agents, device=device)
        next_val = next_value

        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_val = values[t]

        return advantages, returns

    def update(
        self,
        obs_batch: torch.Tensor,
        edge_index_batch: list[torch.Tensor],
        actions_batch: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        n_epochs: int = 4,
    ) -> dict[str, float]:
        """PPO update step."""
        batch_size = obs_batch.shape[0]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(n_epochs):
            for i in range(batch_size):
                obs = obs_batch[i]
                edge_index = edge_index_batch[i]
                actions = actions_batch[i]
                old_lp = old_log_probs[i]
                adv = advantages[i]
                ret = returns[i]

                log_probs, values, entropy = self.evaluate_actions(obs, edge_index, actions)

                ratio = torch.exp(log_probs - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, ret)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        n_updates = n_epochs * batch_size
        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }


# ============================================================================
# Training Loop
# ============================================================================

def train_mlp_mappo(
    env: MicrogridEnvDual,
    agent: MLPMAPPOAgent,
    n_episodes: int = 6000,
    rollout_len: int = 300,  # match proposed (300 steps/episode) for equal step-budget
    n_ppo_epochs: int = 4,
    seed: int = 42,
    save_dir: Path = Path("checkpoints/mlp_mappo"),
    save_every: int = 500,
    log_every: int = 50,
    resume_from: Path | None = None,
) -> None:
    """Train MLP-MAPPO agent with curriculum learning."""
    save_dir.mkdir(parents=True, exist_ok=True)
    device = next(agent.parameters()).device

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get initial observation to determine normalizer shape (same as train_am_mappo.py)
    obs_fast0, _, _ = env.reset()
    obs_full0 = build_am_full_feeder_obs(env, obs_fast0)
    obs_normalizer = RunningNormalizer(shape=obs_full0.shape)
    reward_normalizer = RewardNormalizer(gamma=agent.gamma)
    reward_cfg = AMRewardConfig()

    episode_idx = 0
    phase_keys = list(AM_PHASES.keys())
    current_phase_idx = 0

    # Resume from a checkpoint (load weights+optimizer+normalizer; restore the
    # curriculum position from the saved episode count).
    if resume_from is not None:
        ckpt = torch.load(Path(resume_from), map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model_state_dict"])
        agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        nz = ckpt.get("obs_normalizer")
        if nz is not None:
            obs_normalizer.mean = np.asarray(nz["mean"], dtype=np.float64)
            obs_normalizer.var = np.asarray(nz["var"], dtype=np.float64)
            obs_normalizer.count = float(nz["count"])
        episode_idx = int(ckpt.get("episode", 0))
        cum = 0
        current_phase_idx = len(phase_keys) - 1
        for i, k in enumerate(phase_keys):
            cum += AM_PHASES[k]["n_episodes"]
            if episode_idx < cum:
                current_phase_idx = i
                break
        print(f"[mlp] resumed from {resume_from} at ep {episode_idx}, "
              f"phase {phase_keys[current_phase_idx]}")

    print(f"\n{'='*60}")
    print("MLP-MAPPO Training (Ablation Baseline)")
    print(f"{'='*60}")
    print(f"Total episodes: {n_episodes}")
    print(f"Rollout length: {rollout_len}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    start_time = time.time()

    while episode_idx < n_episodes:
        # Determine current phase
        phase_key = phase_keys[min(current_phase_idx, len(phase_keys) - 1)]
        phase = AM_PHASES[phase_key]

        # Update phase if needed
        phase_episodes = sum(AM_PHASES[k]["n_episodes"] for k in phase_keys[:current_phase_idx + 1])
        if episode_idx >= phase_episodes and current_phase_idx < len(phase_keys) - 1:
            current_phase_idx += 1
            phase_key = phase_keys[current_phase_idx]
            phase = AM_PHASES[phase_key]
            print(f"\n>>> Phase {phase_key}: {phase['description']}")

        # Configure environment for current phase (event_prob is handled internally by event_injector)
        env.event_injector.set_max_delta_p_mw(phase["max_delta_p_mw"])
        env.event_injector.set_probs(phase["event_probs"])

        # Update learning rate
        for pg in agent.optimizer.param_groups:
            pg["lr"] = 3e-4 * phase["lr_factor"]

        # Update entropy coefficient
        agent.entropy_coef = phase["entropy_bonus"]

        # Collect rollout (synchronized with train_am_mappo.py Markov process)
        obs_list, edge_list, action_list = [], [], []
        log_prob_list, value_list, reward_list, done_list = [], [], [], []

        # Reset returns (obs_fast, obs_slow, info) for dual-timescale env
        obs_fast, obs_slow, info = env.reset()
        n_bus = int(len(env.net.bus.index))
        edge_index = ensure_edge_index(env.edge_index, n_nodes=n_bus)

        # Build full feeder observation (same as GraphSAGE-MAPPO)
        obs_full = build_am_full_feeder_obs(env, obs_fast)
        obs_normalizer.update(obs_full[np.newaxis, :])
        obs_norm = obs_normalizer.normalize(obs_full)
        obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)
        edge_t = torch.as_tensor(edge_index, dtype=torch.long, device=device)

        episode_reward = 0.0
        prev_action = None
        prev_k_droop = np.zeros(env.n_agents, dtype=np.float32)

        for step in range(rollout_len):
            with torch.no_grad():
                actions, log_probs, values = agent.act(obs_t, edge_t)

            # Build full action for dual-mode env (same logic as train_am_mappo.py)
            policy_actions = actions.cpu().numpy()
            action_dim = policy_actions.shape[1] if policy_actions.ndim > 1 else 1
            ctrl_p = -policy_actions[:, 0] if action_dim >= 1 else -policy_actions.flatten()
            ctrl_k = policy_actions[:, 1] if action_dim >= 2 else None

            n_vpps = len(env._vpp_droop_agents)
            if env.ffr_mode == "mappo_dual" and ctrl_k is not None:
                full_action = np.zeros(2 * env.n_agents + n_vpps, dtype=np.float32)
                full_action[: env.n_agents] = ctrl_p[: env.n_agents]
                full_action[env.n_agents : 2 * env.n_agents] = ctrl_k[: env.n_agents]
            else:
                full_action = np.zeros(env.n_agents + n_vpps, dtype=np.float32)
                full_action[: env.n_agents] = ctrl_p[: env.n_agents]
                for vpp_idx, (_vpp_id, member_agents) in enumerate(env._vpp_droop_agents.items()):
                    vpp_action = np.mean([ctrl_p[ai] for ai in member_agents if ai < len(ctrl_p)])
                    full_action[env.n_agents + vpp_idx] = vpp_action

            control_actions = ctrl_p.reshape(-1, 1) if ctrl_k is None else np.stack([ctrl_p, ctrl_k], axis=1)

            # Step fast timescale (same as GraphSAGE-MAPPO)
            next_obs_fast, _r_env, done, _trunc, info = env.step_fast(full_action)

            # Reward on the COI frequency deviation (same target as train_am_mappo.py):
            # at the 1 s control step the COI is the resolvable system-frequency
            # observable; identical across all methods, keeping train==eval on COI.
            post_freq_state = env.freq_dyn_lti.get_state()
            post_delta_f = float(post_freq_state.delta_f_hz)
            post_rocof = float(post_freq_state.rocof_hz_s)
            post_freq_hz = 50.0 + post_delta_f

            # Pull dual-product signals from env
            soc_vec = np.asarray(next_obs_fast[:, 3], dtype=np.float32) if next_obs_fast.ndim == 2 else None
            zone_lmp_vec = (
                np.asarray(next_obs_fast[:, 4], dtype=np.float32) if (next_obs_fast.ndim == 2 and next_obs_fast.shape[1] > 4) else None
            )
            k_droop_now = getattr(env, "_k_droop_last", None)
            k_droop_max = getattr(env, "_k_droop_max_per_agent", None)
            p_ref_now = getattr(env, "_p_ref_last", None)

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
                lambda_as_ffr=getattr(env, "lambda_as_ffr", None),
            )

            if k_droop_now is not None:
                prev_k_droop = np.asarray(k_droop_now, dtype=np.float32).copy()

            reward_norm = reward_normalizer.normalize(am_reward, done)
            episode_reward += am_reward

            # Update edge_index from info if topology changed
            n_bus = int(len(env.net.bus.index))
            edge_index = ensure_edge_index(info.get("edge_index", edge_index), n_nodes=n_bus)

            obs_list.append(obs_t.clone())
            edge_list.append(edge_t.clone())
            action_list.append(actions.clone())
            log_prob_list.append(log_probs.clone())
            value_list.append(values.clone())
            reward_list.append(reward_norm)
            done_list.append(done)

            prev_action = control_actions.flatten().copy()

            if done:
                break

            # Build next full feeder observation
            next_obs_full = build_am_full_feeder_obs(env, next_obs_fast)
            obs_normalizer.update(next_obs_full[np.newaxis, :])
            obs_norm = obs_normalizer.normalize(next_obs_full)
            obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)
            edge_t = torch.as_tensor(edge_index, dtype=torch.long, device=device)

        # Compute GAE
        with torch.no_grad():
            _, _, next_value = agent.act(obs_t, edge_t)
            if done_list[-1]:
                next_value = torch.zeros_like(next_value)

        advantages, returns = agent.compute_gae(
            reward_list, value_list, done_list, next_value
        )

        # Stack batches
        obs_batch = torch.stack(obs_list)
        actions_batch = torch.stack(action_list)
        old_log_probs = torch.stack(log_prob_list)

        # PPO update
        update_info = agent.update(
            obs_batch, edge_list, actions_batch,
            old_log_probs, advantages, returns,
            n_epochs=n_ppo_epochs,
        )

        episode_idx += 1

        # Logging
        if episode_idx % log_every == 0:
            elapsed = time.time() - start_time
            print(
                f"[Ep {episode_idx:5d}] Phase {phase_key} | "
                f"Reward: {episode_reward:7.2f} | "
                f"Loss: {update_info['loss']:.4f} | "
                f"Entropy: {update_info['entropy']:.4f} | "
                f"Time: {elapsed/60:.1f}m"
            )

        # Save checkpoint
        if episode_idx % save_every == 0:
            ckpt_path = save_dir / f"mlp_mappo_ep{episode_idx}.pt"
            torch.save({
                "episode": episode_idx,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "obs_normalizer": {
                    "mean": obs_normalizer.mean,
                    "var": obs_normalizer.var,
                    "count": obs_normalizer.count,
                },
                "encoder_type": "mlp",  # Key marker for loading
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = save_dir / "mlp_mappo_final.pt"
    torch.save({
        "episode": episode_idx,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "obs_normalizer": {
            "mean": obs_normalizer.mean,
            "var": obs_normalizer.var,
            "count": obs_normalizer.count,
        },
        "encoder_type": "mlp",
    }, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    print(f"Total time: {(time.time() - start_time) / 3600:.2f}h")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP-MAPPO (ablation baseline)")
    parser.add_argument("--episodes", type=int, default=6000, help="Total training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/mlp_mappo"))
    parser.add_argument("--placement", type=Path, default=Path("artifacts/placement/official_placement_v3.json"))
    parser.add_argument("--mpc-path", type=Path, default=Path("data/grid_IEEE123_complete.m"))
    parser.add_argument("--precompute", type=Path, default=Path("data/precompute_365d.h5"))
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Checkpoint (.pt) to resume from: loads weights+optimizer+normalizer "
                             "and restores the curriculum phase from the saved episode count")
    parser.add_argument("--fixed-base-topology", action="store_true",
                        help="Train only on the nominal base feeder (no reconfiguration). "
                             "Used for the train-on-base, eval-on-all-reconfig generalization protocol.")
    args = parser.parse_args()

    import json
    placement = json.loads(args.placement.read_text(encoding="utf-8"))

    env_config = {
        "placement_path": str(args.placement),
        "mpc_path": str(args.mpc_path),
        "precomputed_dir": str(args.precompute.parent) if args.precompute.exists() else "data/precomputed_365d_97to67",
        "ffr_mode": "mappo_dual",  # same MDP as proposed; exercises the (a_P, a_K) dual action
    }

    env = MicrogridEnvDual(**env_config)
    env.fixed_base_topology = args.fixed_base_topology
    print(f"fixed_base_topology={env.fixed_base_topology}")

    # Get sample observation to determine proper feature dimensions (same as train_am_mappo.py)
    sample_obs_fast, _, _ = env.reset()
    sample_obs_full = build_am_full_feeder_obs(env, sample_obs_fast)
    n_bus = int(sample_obs_full.shape[0])
    obs_feat = int(sample_obs_full.shape[1])

    # Get agent bus indices from placement or env
    agent_bus_indices = np.asarray(getattr(env, "_agent_bus_pp", np.arange(env.n_agents)), dtype=np.int64)
    agent_bus_indices = np.clip(agent_bus_indices, 0, max(n_bus - 1, 0))

    agent = MLPMAPPOAgent(
        obs_feat=obs_feat,
        n_agents=env.n_agents,
        n_bus=n_bus,
        agent_bus_indices=agent_bus_indices,
        action_dim_per_agent=2,
        embed_dim=128,  # match GraphSAGE-MAPPO (ASHA-tuned) so the ablation differs ONLY in encoder
        hidden_dim=128,
    ).to(args.device)

    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    train_mlp_mappo(
        env=env,
        agent=agent,
        n_episodes=args.episodes,
        seed=args.seed,
        save_dir=args.save_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
