"""
GraphSAGE-MAPPO for Ancillary Market (AM) in Islanded Microgrid.

Key features:
- DER-level agents (41) with parameter sharing
- GraphSAGE feeder-level encoder for inter-agent communication
- Agent-centered critic for credit assignment
- MAPPO **replaces classical droop** for FFR control (proposed contribution).
  Each BESS/V2G unit is directly commanded by the RL policy (full P_rated capacity)
  instead of receiving the fixed droop response P_ffr = -k_droop * delta_f.
  See src/baselines/droop.py for the classical droop baseline kept for comparison.
- Proper reward design and normalization for AM metrics
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
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
# Curriculum Phases for AM-MAPPO (FFR training)
# ============================================================================
# Literature-backed progressive training for frequency control in low-inertia MG
#
# References:
#   [1] Zhang et al. (2022, IEEE TPS) - Two-stage curriculum: simple -> complex
#       Key insight: "CL accelerates convergence AND improves local optimum quality"
#   [2] Matavalam et al. (2022, L2RPN) - Domain-aware curriculum with physics
#       Key insight: "Without curriculum, RL agent failed most test scenarios"
#   [3] Benhmidouch et al. (2024) - VSG control focusing on nadir/RoCoF
#       Key insight: "TD3 + proper reward shaping improves frequency transients"
#   [4] Cui et al. (2020, IEEE TPS) - Lyapunov-based stability guarantees
#       Key insight: "Structured learning outperforms unconstrained RL"
#
# Design principles:
#   1. STAGE 1 (A-B): Learn basic droop response with predictable disturbances
#   2. STAGE 2 (C-D): Introduce event variety and topology changes
#   3. STAGE 3 (E-F): Full severity with LR annealing for fine-tuning
#
# Total: 6000 episodes
# S_BASE = 15.7 MW -> contingencies: 10-40% of system

AM_PHASES: dict[str, dict[str, Any]] = {
    # STAGE 1: Foundation - Learn basic frequency response
    "A": {
        "n_episodes": 400,
        "event_prob": 0.5,          # 50% chance - sparse events for stable learning
        "max_delta_p_mw": 2.5,      # ~16% of S_BASE - covers S1 (load_step 2.5 MW)
        "event_probs": {"load_step": 1.0, "gen_trip": 0.0, "line_trip": 0.0, "high_ren": 0.0},
        "lr_factor": 1.0,
        "entropy_bonus": 0.02,      # Extra exploration in early phase
        "description": "Foundation: load-only response, learn basic droop",
    },
    "B": {
        "n_episodes": 1000,
        "event_prob": 0.7,          # Increase event frequency
        "max_delta_p_mw": 3.0,      # ~19% of S_BASE - moderate
        "event_probs": {"load_step": 0.5, "gen_trip": 0.5, "line_trip": 0.0, "high_ren": 0.0},
        "lr_factor": 1.0,
        "entropy_bonus": 0.01,
        "description": "Bidirectional: add gen trips (under-frequency events)",
    },
    # STAGE 2: Complexity - Event variety and renewable uncertainty
    "C": {
        "n_episodes": 1000,
        "event_prob": 0.8,
        "max_delta_p_mw": 4.0,      # ~25% of S_BASE - severe
        "event_probs": {"load_step": 0.35, "gen_trip": 0.35, "line_trip": 0.0, "high_ren": 0.3},
        "lr_factor": 1.0,
        "entropy_bonus": 0.005,
        "description": "Renewable: add high_ren surges (over-frequency)",
    },
    "D": {
        "n_episodes": 1000,
        "event_prob": 0.85,
        "max_delta_p_mw": 5.5,      # ~35% of S_BASE - covers S4 (gen_trip −5.5 MW)
        "event_probs": {"load_step": 0.3, "gen_trip": 0.3, "line_trip": 0.15, "high_ren": 0.25},
        "lr_factor": 0.8,           # Start LR decay
        "entropy_bonus": 0.003,
        "description": "Topology: introduce line trips (hardest events)",
    },
    # STAGE 3: Mastery - Full severity with fine-tuning
    "E": {
        "n_episodes": 1000,
        "event_prob": 0.9,
        "max_delta_p_mw": 5.5,      # ~35% of S_BASE
        "event_probs": {"load_step": 0.3, "gen_trip": 0.3, "line_trip": 0.2, "high_ren": 0.2},
        "lr_factor": 0.5,
        "entropy_bonus": 0.001,
        "description": "N-1 ready: full event mix, significant LR decay",
    },
    "F": {
        "n_episodes": 1600,
        "event_prob": 0.9,
        "max_delta_p_mw": 6.3,      # ~40% of S_BASE - extreme stress test
        "event_probs": {"load_step": 0.3, "gen_trip": 0.3, "line_trip": 0.2, "high_ren": 0.2},
        "lr_factor": 0.25,
        "entropy_bonus": 0.0,       # Pure exploitation in final phase
        "description": "Extreme: 40% contingency, fine-tune policy",
    },
}


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
    """Reward weights for Ancillary Market metrics.

    Thresholds aligned with ENTSO-E Network Codes (RfG, SO GL):
    - FCR deadband: ±10-20 mHz (Continental Europe)
    - FCR activation: ±200 mHz (49.8-50.2 Hz)
    - UFLS Stage 1: 49.0 Hz (Δf = -1.0 Hz)
    - Pre-UFLS warning: 49.5 Hz (Δf = -0.5 Hz)
    """
    # Frequency / safety terms (re-balanced for dual-action)
    w_delta_f: float = 0.25
    w_rocof: float = 0.12
    w_violation: float = 0.22
    w_nadir: float = 0.12
    w_effort: float = 0.05
    w_tracking: float = 0.0   # disabled in dual mode: K_droop handles transient automatically
    # New dual-product terms (revenue + storage health + gain stability)
    w_soc_pen: float = 0.08
    w_market: float = 0.10
    w_commit: float = 0.06

    # Frequency references (ENTSO-E compliant)
    delta_f_target: float = 0.0
    delta_f_deadband: float = 0.02
    delta_f_ref: float = 0.5
    rocof_target: float = 0.0
    rocof_deadband: float = 0.1
    rocof_ref: float = 1.0

    # Safety thresholds (ENTSO-E UFLS standards)
    f_limit: float = 0.5
    nadir_threshold: float = 49.5
    nadir_ref: float = 0.5

    # Control effort and droop-like tracking references
    action_ref_scale: float = 0.75
    effort_smoothness_coef: float = 0.5
    tracking_delta_f_ref: float = 0.5
    tracking_rocof_ref: float = 0.2
    tracking_k_delta_f: float = 0.5
    tracking_k_rocof: float = 0.25

    # SoC band (per-DER comfortable operating range)
    soc_band_lo: float = 0.20
    soc_band_hi: float = 0.85
    # Market signals
    # - lmp_ref: zone_LMP normalisation (€/MWh) for energy revenue
    # - as_ffr_ref: FFR capacity price normalisation (€/MW/h) for capacity revenue.
    #   Replaces the legacy "cap_price_ratio × LMP" proxy; the env now exposes
    #   the real per-step lambda_as_ffr from the precompute.
    lmp_ref: float = 100.0
    as_ffr_ref: float = 20.0
    cap_price_ratio: float = 0.5   # fallback weight when lambda_as_ffr is unavailable


def compute_am_reward(
    delta_f: float,
    rocof: float,
    action: np.ndarray,
    prev_action: np.ndarray | None,
    freq_hz: float,
    cfg: AMRewardConfig,
    soc: np.ndarray | None = None,
    zone_lmp: np.ndarray | None = None,
    k_droop_now: np.ndarray | None = None,
    k_droop_prev: np.ndarray | None = None,
    p_ref_now: np.ndarray | None = None,
    k_droop_max: np.ndarray | None = None,
    lambda_as_ffr: float | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Compute AM reward with proper scaling and deadbands.

    Returns:
        reward: Scalar reward
        info: Dict of reward components for logging
    """
    # Under-frequency-only normalized frequency and RoCoF penalties.
    delta_f_abs = abs(delta_f)
    e_delta_f = float(np.clip(max(-(delta_f + cfg.delta_f_deadband), 0.0) / cfg.delta_f_ref, 0.0, 1.0))
    r_delta_f = -cfg.w_delta_f * e_delta_f

    rocof_abs = abs(rocof)
    e_rocof = float(np.clip(max(-(rocof + cfg.rocof_deadband), 0.0) / cfg.rocof_ref, 0.0, 1.0))
    r_rocof = -cfg.w_rocof * e_rocof

    # Under-frequency violation penalty (binary safety term).
    in_violation = delta_f < -cfg.f_limit
    e_violation = 1.0 if in_violation else 0.0
    r_violation = -cfg.w_violation * e_violation

    # Normalized control effort penalty.
    action_arr = np.asarray(action, dtype=np.float32)
    action_norm = float(np.clip(np.mean(np.square(action_arr / cfg.action_ref_scale)), 0.0, 1.0))
    if prev_action is not None:
        prev_action_arr = np.asarray(prev_action, dtype=np.float32)
        action_diff = float(np.clip(np.mean(np.square((action_arr - prev_action_arr) / cfg.action_ref_scale)), 0.0, 1.0))
    else:
        action_diff = 0.0
    e_effort = float(np.clip(action_norm + cfg.effort_smoothness_coef * action_diff, 0.0, 1.0))
    r_effort = -cfg.w_effort * e_effort

    if delta_f < -cfg.delta_f_deadband:
        delta_f_tracking = delta_f + cfg.delta_f_deadband
        rocof_tracking = min(rocof + cfg.rocof_deadband, 0.0) if rocof < -cfg.rocof_deadband else 0.0
        action_ref = -(
            cfg.tracking_k_delta_f * delta_f_tracking / cfg.tracking_delta_f_ref
            + cfg.tracking_k_rocof * rocof_tracking / cfg.tracking_rocof_ref
        )
        action_ref = float(np.clip(action_ref, 0.0, cfg.action_ref_scale))
        e_tracking = float(np.clip(np.mean(np.square((action_arr - action_ref) / cfg.action_ref_scale)), 0.0, 1.0))
        r_tracking = -cfg.w_tracking * e_tracking
    else:
        action_ref = 0.0
        e_tracking = 0.0
        r_tracking = 0.0

    # Under-frequency-only nadir safety term.
    e_nadir = float(np.clip((cfg.nadir_threshold - freq_hz) / cfg.nadir_ref, 0.0, 1.0))
    r_nadir = -cfg.w_nadir * e_nadir

    # ── Dual-product terms (only active when corresponding inputs provided) ─────────
    # SoC band penalty: quadratic outside [soc_band_lo, soc_band_hi].
    if soc is not None and cfg.w_soc_pen > 0.0:
        soc_arr = np.asarray(soc, dtype=np.float32)
        lo = np.maximum(0.0, cfg.soc_band_lo - soc_arr)
        hi = np.maximum(0.0, soc_arr - cfg.soc_band_hi)
        e_soc = float(np.clip(np.mean(lo * lo + hi * hi), 0.0, 1.0))
        r_soc_pen = -cfg.w_soc_pen * e_soc
    else:
        e_soc = 0.0
        r_soc_pen = 0.0

    # Market revenue: energy (LMP × P_ref) + FFR capacity payment.
    # Energy uses normalised zone_LMP as before.
    # Capacity uses env-exposed per-step lambda_as_ffr when available (from
    # precompute via env._apply_day_context); falls back to the LMP × cap_price_ratio
    # proxy when not provided (e.g. evaluation harness without market data).
    # Sign: positive reward when policy provides genuine product value.
    if (zone_lmp is not None and p_ref_now is not None and cfg.w_market > 0.0):
        lmp = np.asarray(zone_lmp, dtype=np.float32) / max(cfg.lmp_ref, 1e-6)
        p_ref = np.asarray(p_ref_now, dtype=np.float32)
        energy_rev = float(np.mean(lmp * p_ref))
        if k_droop_now is not None and k_droop_max is not None:
            k_now = np.asarray(k_droop_now, dtype=np.float32)
            k_max = np.maximum(np.asarray(k_droop_max, dtype=np.float32), 1e-6)
            k_util = k_now / k_max                               # in [0, 1]
            if lambda_as_ffr is not None:
                ffr_norm = float(lambda_as_ffr) / max(cfg.as_ffr_ref, 1e-6)
                cap_rev = float(np.mean(ffr_norm * k_util))
            else:
                cap_rev = float(np.mean(lmp * cfg.cap_price_ratio * k_util))
        else:
            cap_rev = 0.0
        r_market = cfg.w_market * float(np.clip(energy_rev + cap_rev, -1.0, 1.0))
    else:
        r_market = 0.0

    # K commitment stability: penalize |ΔK| step-to-step (avoid gain chattering).
    if (k_droop_now is not None and k_droop_prev is not None
            and k_droop_max is not None and cfg.w_commit > 0.0):
        k_now = np.asarray(k_droop_now, dtype=np.float32)
        k_prev = np.asarray(k_droop_prev, dtype=np.float32)
        k_max = np.maximum(np.asarray(k_droop_max, dtype=np.float32), 1e-6)
        e_commit = float(np.clip(np.mean(np.square((k_now - k_prev) / k_max)), 0.0, 1.0))
        r_commit = -cfg.w_commit * e_commit
    else:
        r_commit = 0.0

    # Total reward
    reward = (
        r_delta_f + r_rocof + r_violation + r_effort + r_tracking + r_nadir
        + r_soc_pen + r_market + r_commit
    )

    info = {
        "r_delta_f": r_delta_f,
        "r_rocof": r_rocof,
        "r_violation": r_violation,
        "r_effort": r_effort,
        "r_tracking": r_tracking,
        "action_ref": action_ref,
        "r_nadir": r_nadir,
        "r_soc_pen": r_soc_pen,
        "r_market": r_market,
        "r_commit": r_commit,
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
    """Shared actor network for all agents.

    Std parametrization (replacing former hard clamp [0.05, 0.5] which produced
    zero gradient at the boundary): std = softplus(log_std) + min_std. Gradient
    is always non-zero ⇒ std can shrink/grow throughout training.
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int = 1,
        hidden_dim: int = 128,
        action_scale: float = 0.75,
        log_std_init: float = -1.0,
        min_std: float = 0.05,
    ):
        super().__init__()
        self.action_scale = float(action_scale)
        self.min_std = float(min_std)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        # log_std is now a free pre-activation (softplus applied in forward).
        # Initial std = softplus(log_std_init) + min_std. For log_std_init=-1.0
        # this gives std ≈ 0.31 + 0.05 = 0.36 — matches previous behavior.
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def _std(self) -> torch.Tensor:
        return F.softplus(self.log_std) + self.min_std

    def forward(self, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embed: Agent embeddings [n_agents, embed_dim] or [batch, n_agents, embed_dim]

        Returns:
            mean: Action means
            std: Action stds
        """
        h = self.net(embed)
        mean = self.action_scale * torch.tanh(self.mean_head(h))
        std = self._std()
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

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Networks
        self.encoder = FeederGraphSAGEAgentEncoder(obs_feat, hidden_dim, embed_dim)
        self.actor = SharedGaussianActor(
            embed_dim, action_dim_per_agent, hidden_dim,
            log_std_init=log_std_init, min_std=min_std,
        )
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
        next_value: np.ndarray | float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns with per-agent credit assignment.

        Args:
            rewards: [n_steps] shared reward
            values: [n_steps, n_agents] per-agent values
            dones: [n_steps] episode done flags
            next_value: [n_agents] or scalar for bootstrap
        """
        n_steps = len(rewards)

        # Handle both scalar (legacy) and per-agent values
        if values.ndim == 1:
            # Legacy scalar path
            advantages = np.zeros(n_steps, dtype=np.float32)
            gae = 0.0
            nv = float(next_value) if np.isscalar(next_value) else float(np.mean(next_value))
            for t in reversed(range(n_steps)):
                next_val = nv if t == n_steps - 1 else values[t + 1]
                mask = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * next_val * mask - values[t]
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                advantages[t] = gae
            returns = advantages + values
            return advantages, returns

        # Per-agent path: values is [n_steps, n_agents]
        n_agents = values.shape[1]
        advantages = np.zeros((n_steps, n_agents), dtype=np.float32)
        gae = np.zeros(n_agents, dtype=np.float32)

        # Ensure next_value is [n_agents]
        if np.isscalar(next_value):
            next_value = np.full(n_agents, float(next_value), dtype=np.float32)
        else:
            next_value = np.asarray(next_value, dtype=np.float32)

        for t in reversed(range(n_steps)):
            next_val = next_value if t == n_steps - 1 else values[t + 1]
            mask = 1.0 - dones[t]
            # Broadcast shared reward to all agents
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
        """PPO update with mini-batches and per-agent credit assignment.

        Args:
            old_log_probs: [n_samples, n_agents] per-agent log probs
            returns: [n_samples, n_agents] per-agent returns
            advantages: [n_samples, n_agents] per-agent advantages
        """
        n_samples = obs_batch.shape[0]

        # Handle both legacy scalar and per-agent advantages
        if advantages.ndim == 1:
            advantages = advantages.unsqueeze(-1).expand(-1, self.n_agents)
            returns = returns.unsqueeze(-1).expand(-1, self.n_agents)
            old_log_probs = old_log_probs.unsqueeze(-1).expand(-1, self.n_agents)

        # Normalize advantages per-agent
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        # Per-dim accumulators (length = action_dim)
        action_dim = int(self.action_dim)
        total_entropy_per_dim = np.zeros(action_dim, dtype=np.float64)
        total_action_abs_per_dim = np.zeros(action_dim, dtype=np.float64)
        total_grad_mean_head_per_dim = np.zeros(action_dim, dtype=np.float64)
        total_grad_log_std_per_dim = np.zeros(action_dim, dtype=np.float64)
        total_kl_per_dim = np.zeros(action_dim, dtype=np.float64)
        total_ratio = 0.0
        total_clip_fraction = 0.0
        total_approx_kl = 0.0
        total_explained_var = 0.0

        for _ in range(n_epochs):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, mini_batch_size):
                end = min(start + mini_batch_size, n_samples)
                idx = indices[start:end]
                batch_size = len(idx)

                # Accumulate gradients over mini-batch
                policy_losses = []
                value_losses = []
                entropies = []
                entropy_per_dim_batch = []
                action_abs_per_dim_batch = []
                kl_per_dim_batch = []
                ratio_batch: list[float] = []
                clip_batch: list[float] = []
                approx_kl_batch: list[float] = []
                value_var_batch: list[tuple[float, float]] = []

                for i, sample_idx in enumerate(idx):
                    obs_i = obs_batch[sample_idx]
                    action_i = actions_batch[sample_idx]
                    old_lp_i = old_log_probs[sample_idx]  # [n_agents]
                    ret_i = returns[sample_idx]  # [n_agents]
                    adv_i = advantages[sample_idx]  # [n_agents]
                    edge_i = edge_batch[sample_idx] if sample_idx < len(edge_batch) else edge_batch[0]

                    embeds = self.get_agent_embeddings(obs_i, edge_i)
                    dist = self.actor.dist(embeds)
                    new_log_prob_per_dim = dist.log_prob(action_i)              # [n_agents, action_dim]
                    new_log_prob = new_log_prob_per_dim.sum(dim=-1)             # [n_agents]
                    entropy_per_dim_tensor = dist.entropy()                     # [n_agents, action_dim]
                    entropy = entropy_per_dim_tensor.sum(dim=-1).mean()
                    value = self.critic(embeds).squeeze(-1)  # [n_agents]

                    # Per-agent PPO loss
                    ratio = torch.exp(new_log_prob - old_lp_i)  # [n_agents]
                    surr1 = ratio * adv_i
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_i
                    policy_loss = -torch.min(surr1, surr2).mean()  # Mean over agents

                    # Per-agent value loss
                    value_loss = F.mse_loss(value, ret_i)

                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
                    entropies.append(entropy)

                    # Per-dim diagnostics (no grad needed)
                    with torch.no_grad():
                        entropy_per_dim_batch.append(entropy_per_dim_tensor.mean(dim=0).cpu().numpy())
                        action_abs_per_dim_batch.append(action_i.abs().mean(dim=0).cpu().numpy())
                        # Per-dim approx KL: -mean(new - old) but old_log_prob is summed, so we
                        # approximate per-dim KL by allocating proportionally to per-dim entropy ratio.
                        # Simpler: use new_log_prob_per_dim mean minus baseline (-action_dim*old_lp/action_dim).
                        kl_per_dim_batch.append(
                            (-new_log_prob_per_dim.mean(dim=0) + (old_lp_i.mean() / max(action_dim, 1))).cpu().numpy()
                        )
                        ratio_batch.append(float(ratio.mean().item()))
                        clip_batch.append(float(((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()))
                        approx_kl_batch.append(float((old_lp_i - new_log_prob).mean().item()))
                        value_var_batch.append(
                            (float(ret_i.var().item()), float((ret_i - value).var().item()))
                        )

                # Aggregate batch
                policy_loss = torch.stack(policy_losses).mean()
                value_loss = torch.stack(value_losses).mean()
                entropy = torch.stack(entropies).mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                # Capture per-dim gradient norms BEFORE clipping (more interpretable)
                with torch.no_grad():
                    mh_grad = self.actor.mean_head.weight.grad
                    ls_grad = self.actor.log_std.grad
                    if mh_grad is not None:
                        # mean_head.weight shape: [action_dim, hidden_dim]
                        mh_norm = mh_grad.norm(dim=1).cpu().numpy()
                        total_grad_mean_head_per_dim += mh_norm.astype(np.float64)
                    if ls_grad is not None:
                        # log_std shape: [action_dim]
                        ls_norm = ls_grad.abs().cpu().numpy()
                        total_grad_log_std_per_dim += ls_norm.astype(np.float64)
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                # Aggregate per-dim diagnostics
                total_entropy_per_dim += np.mean(entropy_per_dim_batch, axis=0)
                total_action_abs_per_dim += np.mean(action_abs_per_dim_batch, axis=0)
                total_kl_per_dim += np.mean(kl_per_dim_batch, axis=0)
                total_ratio += float(np.mean(ratio_batch))
                total_clip_fraction += float(np.mean(clip_batch))
                total_approx_kl += float(np.mean(approx_kl_batch))
                ret_var = float(np.mean([v[0] for v in value_var_batch]))
                resid_var = float(np.mean([v[1] for v in value_var_batch]))
                total_explained_var += 1.0 - resid_var / max(ret_var, 1e-8)
                n_updates += 1

        with torch.no_grad():
            # log_std is the raw pre-softplus param; real std uses softplus+min_std (no clamp).
            log_std_t = self.actor.log_std
            std_t = self.actor._std()
            actor_log_std = log_std_t.mean().item()
            actor_std = std_t.mean().item()
            log_std_per_dim = log_std_t.cpu().numpy().tolist()
            std_per_dim = std_t.cpu().numpy().tolist()

        nu = max(n_updates, 1)
        per_dim_dict: dict[str, float] = {}
        for d in range(action_dim):
            per_dim_dict[f"entropy_dim{d}"] = float(total_entropy_per_dim[d] / nu)
            per_dim_dict[f"action_abs_dim{d}"] = float(total_action_abs_per_dim[d] / nu)
            per_dim_dict[f"grad_mean_head_dim{d}"] = float(total_grad_mean_head_per_dim[d] / nu)
            per_dim_dict[f"grad_log_std_dim{d}"] = float(total_grad_log_std_per_dim[d] / nu)
            per_dim_dict[f"approx_kl_dim{d}"] = float(total_kl_per_dim[d] / nu)
            per_dim_dict[f"log_std_dim{d}"] = float(log_std_per_dim[d])
            per_dim_dict[f"std_dim{d}"] = float(std_per_dim[d])

        out: dict[str, float] = {
            "loss": total_loss / nu,
            "policy_loss": total_policy_loss / nu,
            "value_loss": total_value_loss / nu,
            "entropy": total_entropy / nu,
            "actor_log_std": actor_log_std,
            "actor_std": actor_std,
            "ratio_mean": total_ratio / nu,
            "clip_fraction": total_clip_fraction / nu,
            "approx_kl": total_approx_kl / nu,
            "explained_var": total_explained_var / nu,
        }
        out.update(per_dim_dict)
        return out


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


def get_am_obs(env: MicrogridEnvDual, obs_fast: np.ndarray, extended: bool = True) -> np.ndarray:
    """
    Extract AM-relevant observation for each agent.

    Base features (10-D):
    0. delta_f (Hz, /0.5)
    1. rocof (Hz/s, /1.0)
    2. p_net (MW, /1.0)
    3. soc / dcob in [0, 1]
    4. zone_lmp / 100
    5-8. agent_type one-hot (EVCS_PV, BESS, V2G, DPV)
    9. vpp_membership (normalized index)

    Extended features (+6 dims when extended=True, total 16-D):
    10. K_droop_prev_norm   — last K_droop / K_max (dual-action memory)
    11. P_ref_prev          — last a_P in [-1, 1]
    12. SoC_band_lo_viol    — max(0, 0.20 - SoC)  (under-band stress)
    13. SoC_band_hi_viol    — max(0, SoC - 0.85)  (over-band stress)
    14. P_forecast_+1       — persistence baseline = current p_net
    15. t_to_depart_norm    — V2G availability proxy (1.0 = abundant, 0.0 = imminent)
    """
    n_agents = env.n_agents
    n_feat = 16 if extended else 10
    obs = np.zeros((n_agents, n_feat), dtype=np.float32)

    # Global frequency state
    freq_state = env.freq_dyn.get_state()
    delta_f = np.clip(freq_state.delta_f_hz / 0.5, -1.0, 1.0)
    rocof = np.clip(freq_state.rocof_hz_s / 1.0, -1.0, 1.0)

    obs[:, 0] = delta_f
    obs[:, 1] = rocof

    # Per-agent features from fast obs
    obs[:, 2] = np.clip(obs_fast[:, 2] / 1.0, -1.0, 1.0)
    obs[:, 3] = np.clip(obs_fast[:, 3], 0.0, 1.0)
    obs[:, 4] = np.clip(obs_fast[:, 4] / 100.0, 0.0, 1.0)

    # Agent type encoding (one-hot)
    for i, spec in enumerate(env._agent_specs):
        agent_type = spec.get("type", "")
        if "PV" in agent_type and "DPV" not in agent_type:
            obs[i, 5] = 1.0
        elif "BESS" in agent_type:
            obs[i, 6] = 1.0
        elif "V2G" in agent_type:
            obs[i, 7] = 1.0
        else:
            obs[i, 8] = 1.0

    # VPP membership
    for vpp_idx, (vpp_id, agents) in enumerate(env._vpp_droop_agents.items()):
        for ai in agents:
            if ai < n_agents:
                obs[ai, 9] = (vpp_idx + 1) / 3.0

    if not extended:
        return obs

    # ── Extended features (dual-action memory + SoC band + forecast + V2G availability)
    k_prev = getattr(env, "_k_droop_last", np.zeros(n_agents, dtype=np.float32))
    p_ref_prev = getattr(env, "_p_ref_last", np.zeros(n_agents, dtype=np.float32))
    k_max = getattr(env, "_k_droop_max_per_agent", np.ones(n_agents, dtype=np.float32))
    # Normalize K_prev by its per-agent max (so type-agnostic in [0, 1])
    k_norm_denom = np.maximum(k_max, 1e-6)
    obs[:, 10] = np.clip(k_prev / k_norm_denom, 0.0, 1.0)
    obs[:, 11] = np.clip(p_ref_prev, -1.0, 1.0)

    soc = np.clip(obs_fast[:, 3], 0.0, 1.0)
    obs[:, 12] = np.maximum(0.0, 0.20 - soc)
    obs[:, 13] = np.maximum(0.0, soc - 0.85)

    # Persistence forecast = current p_net (research framework hook; replace with NWP later)
    obs[:, 14] = obs[:, 2]

    # V2G departure-time proxy: 1.0 for non-V2G assets; 0.5 default for V2G
    # (placeholder until day_context exposes departure_step per agent)
    is_v2g = obs[:, 7] > 0.5
    obs[:, 15] = np.where(is_v2g, 0.5, 1.0).astype(np.float32)

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
    update_epochs: int = 4,
    mini_batch_size: int = 64,
    log_interval: int = 10,
    checkpoint_dir: Path | None = None,
    reward_cfg: AMRewardConfig | None = None,
    _phase_name: str | None = None,
    _max_delta_p_mw: float = 6.3,
) -> dict[str, list[float]]:
    """Main training loop for AM-MAPPO.

    Args:
        _phase_name: Optional phase name for curriculum logging
        _max_delta_p_mw: Maximum event magnitude (MW) for curriculum control
    """

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
        "violation_under_fraction": [],
        "violation_over_fraction": [],
        "violation_post_inject_fraction": [],
        "loss": [],
        "entropy": [],
        "actor_log_std": [],
        "actor_std": [],
        "mean_abs_action": [],
        "action_saturation_fraction": [],
        "r_delta_f": [],
        "r_rocof": [],
        "r_violation": [],
        "r_effort": [],
        "r_tracking": [],
        "r_nadir": [],
        "r_soc_pen": [],
        "r_market": [],
        "r_commit": [],
        # PPO diagnostics
        "ratio_mean": [],
        "clip_fraction": [],
        "approx_kl": [],
        "explained_var": [],
        # Per-dim (populated dynamically below based on action_dim)
        "per_dim": {},
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
        ep_violation_under = []
        ep_violation_over = []
        ep_violation_post_inject = []
        ep_r_delta_f = []
        ep_r_rocof = []
        ep_r_violation = []
        ep_r_effort = []
        ep_r_tracking = []
        ep_r_nadir = []
        ep_r_soc_pen = []
        ep_r_market = []
        ep_r_commit = []
        ep_mean_abs_action = []
        ep_action_saturation = []
        prev_action = None
        k_droop_prev = np.zeros(env.n_agents, dtype=np.float32)

        for _t in range(steps_per_episode):
            policy_actions, log_probs, values, _ = agent.act(obs_norm, edge_index)
            # policy_actions shape: (n_agents, action_dim). For dual mode, dim 0=a_P, dim 1=a_K.
            # Keep historical sign flip for a_P so reward's action_ref (positive when Δf<0) aligns.
            # a_K maps to K_droop bounds [K_min, K_max] and is NOT flipped.
            action_dim = policy_actions.shape[1] if policy_actions.ndim > 1 else 1
            ctrl_p = -policy_actions[:, 0] if action_dim >= 1 else -policy_actions.flatten()
            ctrl_k = policy_actions[:, 1] if action_dim >= 2 else None

            n_vpps = len(env._vpp_droop_agents)
            if env.ffr_mode == "mappo_dual" and ctrl_k is not None:
                full_action = np.zeros(2 * env.n_agents + n_vpps, dtype=np.float32)
                full_action[: env.n_agents] = ctrl_p[: env.n_agents]
                full_action[env.n_agents : 2 * env.n_agents] = ctrl_k[: env.n_agents]
                # Legacy VPP-K slots: zeros (per-DER K replaces VPP aggregation in dual mode)
            else:
                full_action = np.zeros(env.n_agents + n_vpps, dtype=np.float32)
                full_action[: env.n_agents] = ctrl_p[: env.n_agents]
                for vpp_idx, (_vpp_id, member_agents) in enumerate(env._vpp_droop_agents.items()):
                    vpp_action = np.mean([ctrl_p[ai] for ai in member_agents if ai < len(ctrl_p)])
                    full_action[env.n_agents + vpp_idx] = vpp_action

            # Stash post-flip control vector for reward + buffer (1-D for backward-compat reward signature)
            control_actions = ctrl_p.reshape(-1, 1) if ctrl_k is None else np.stack([ctrl_p, ctrl_k], axis=1)

            next_obs_fast, _r_env, done, _trunc, info = env.step_fast(full_action)

            # Compute reward AFTER step to give correct RL signal
            post_freq_state = env.freq_dyn.get_state()
            post_delta_f = float(post_freq_state.delta_f_hz)
            post_rocof = float(post_freq_state.rocof_hz_s)
            post_freq_hz = 50.0 + post_delta_f

            # Pull dual-product signals from env (only populated when ffr_mode == "mappo_dual")
            soc_vec = np.asarray(next_obs_fast[:, 3], dtype=np.float32) if next_obs_fast.ndim == 2 else None
            zone_lmp_vec = (
                np.asarray(next_obs_fast[:, 4], dtype=np.float32) if (next_obs_fast.ndim == 2 and next_obs_fast.shape[1] > 4) else None
            )
            k_droop_now = getattr(env, "_k_droop_last", None)
            k_droop_max = getattr(env, "_k_droop_max_per_agent", None)
            p_ref_now = getattr(env, "_p_ref_last", None)

            reward, reward_info = compute_am_reward(
                delta_f=post_delta_f,
                rocof=post_rocof,
                action=control_actions.flatten(),
                prev_action=prev_action,
                freq_hz=post_freq_hz,
                cfg=reward_cfg,
                soc=soc_vec,
                zone_lmp=zone_lmp_vec,
                k_droop_now=k_droop_now,
                k_droop_prev=k_droop_prev,
                p_ref_now=p_ref_now,
                k_droop_max=k_droop_max,
                lambda_as_ffr=getattr(env, "lambda_as_ffr", None),
            )
            if k_droop_now is not None:
                k_droop_prev = np.asarray(k_droop_now, dtype=np.float32).copy()
            n_bus = int(len(env.net.bus.index))
            edge_index = ensure_edge_index(info.get("edge_index", edge_index), n_nodes=n_bus)

            freq_state = env.freq_dyn.get_state()
            freq_hz = 50.0 + freq_state.delta_f_hz

            reward_norm = reward_normalizer.normalize(reward, done=done)

            buffer.add(
                obs=obs_norm.copy(),
                edge_index=edge_index.copy(),
                action=policy_actions.copy(),
                log_prob=log_probs.copy(),  # Keep per-agent [n_agents]
                value=values.copy(),         # Keep per-agent [n_agents]
                reward=reward_norm,
                done=done,
            )

            next_obs = build_am_full_feeder_obs(env, next_obs_fast)
            obs_normalizer.update(next_obs)
            obs_norm = obs_normalizer.normalize(next_obs)
            prev_action = control_actions.flatten().copy()

            ep_reward += reward
            ep_delta_f.append(abs(freq_state.delta_f_hz))
            ep_rocof.append(abs(freq_state.rocof_hz_s))
            ep_freq_hz.append(freq_hz)

            is_under = float(freq_state.delta_f_hz < -reward_cfg.f_limit)
            is_over = float(freq_state.delta_f_hz > reward_cfg.f_limit)
            is_any = float((freq_state.delta_f_hz < -reward_cfg.f_limit) or (freq_state.delta_f_hz > reward_cfg.f_limit))
            ep_violation_under.append(is_under)
            ep_violation_over.append(is_over)
            ep_violation.append(is_any)

            # Post-injection window metric to avoid dilution by pre-event steps
            inject_step = int(getattr(env.current_event, "t_inject", 30.0)) if env.current_event is not None else 30
            in_post_window = (_t >= inject_step) and (_t < inject_step + 50)
            if in_post_window:
                ep_violation_post_inject.append(is_any)

            ep_r_delta_f.append(reward_info["r_delta_f"])
            ep_r_rocof.append(reward_info["r_rocof"])
            ep_r_violation.append(reward_info["r_violation"])
            ep_r_effort.append(reward_info["r_effort"])
            ep_r_tracking.append(reward_info["r_tracking"])
            ep_r_nadir.append(reward_info["r_nadir"])
            ep_r_soc_pen.append(reward_info.get("r_soc_pen", 0.0))
            ep_r_market.append(reward_info.get("r_market", 0.0))
            ep_r_commit.append(reward_info.get("r_commit", 0.0))
            ep_mean_abs_action.append(float(np.mean(np.abs(control_actions))))
            ep_action_saturation.append(float(np.mean(np.abs(control_actions) > 0.95)))

        # Update policy after each episode
        if len(buffer) > 0:
            # Compute advantages with per-agent values
            rewards = np.array(buffer.rewards, dtype=np.float32)  # [n_steps]
            values = np.stack(buffer.values, axis=0)  # [n_steps, n_agents]
            dones = np.array(buffer.dones, dtype=np.float32)  # [n_steps]
            advantages, returns = agent.compute_gae(rewards, values, dones)

            # Prepare batch - per-agent data
            obs_batch = torch.tensor(np.stack(buffer.obs), dtype=torch.float32)
            edge_batch = [torch.tensor(e, dtype=torch.long) for e in buffer.edge_index]
            actions_batch = torch.tensor(np.stack(buffer.actions), dtype=torch.float32)
            old_log_probs = torch.tensor(np.stack(buffer.log_probs, axis=0), dtype=torch.float32)  # [n_steps, n_agents]
            returns_t = torch.tensor(returns, dtype=torch.float32)  # [n_steps, n_agents]
            advantages_t = torch.tensor(advantages, dtype=torch.float32)  # [n_steps, n_agents]

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
            history["actor_log_std"].append(update_info["actor_log_std"])
            history["actor_std"].append(update_info["actor_std"])
            history["ratio_mean"].append(update_info.get("ratio_mean", 1.0))
            history["clip_fraction"].append(update_info.get("clip_fraction", 0.0))
            history["approx_kl"].append(update_info.get("approx_kl", 0.0))
            history["explained_var"].append(update_info.get("explained_var", 0.0))

            # Per-dim diagnostics (keys like "entropy_dim0", "grad_log_std_dim1", ...)
            for k, v in update_info.items():
                if "_dim" in k:
                    history["per_dim"].setdefault(k, []).append(float(v))

            buffer.clear()

        # Log
        history["episode_reward"].append(ep_reward)
        history["delta_f_mean"].append(np.mean(ep_delta_f) if ep_delta_f else 0.0)
        history["rocof_mean"].append(np.mean(ep_rocof) if ep_rocof else 0.0)
        history["rocof_max"].append(np.max(ep_rocof) if ep_rocof else 0.0)
        history["max_abs_delta_f"].append(np.max(ep_delta_f) if ep_delta_f else 0.0)
        history["freq_nadir"].append(np.min(ep_freq_hz) if ep_freq_hz else 50.0)
        history["violation_fraction"].append(np.mean(ep_violation) if ep_violation else 0.0)
        history["violation_under_fraction"].append(np.mean(ep_violation_under) if ep_violation_under else 0.0)
        history["violation_over_fraction"].append(np.mean(ep_violation_over) if ep_violation_over else 0.0)
        history["violation_post_inject_fraction"].append(
            np.mean(ep_violation_post_inject) if ep_violation_post_inject else 0.0
        )
        history["r_delta_f"].append(np.mean(ep_r_delta_f) if ep_r_delta_f else 0.0)
        history["r_rocof"].append(np.mean(ep_r_rocof) if ep_r_rocof else 0.0)
        history["r_violation"].append(np.mean(ep_r_violation) if ep_r_violation else 0.0)
        history["r_effort"].append(np.mean(ep_r_effort) if ep_r_effort else 0.0)
        history["r_tracking"].append(np.mean(ep_r_tracking) if ep_r_tracking else 0.0)
        history["r_nadir"].append(np.mean(ep_r_nadir) if ep_r_nadir else 0.0)
        history["r_soc_pen"].append(np.mean(ep_r_soc_pen) if ep_r_soc_pen else 0.0)
        history["r_market"].append(np.mean(ep_r_market) if ep_r_market else 0.0)
        history["r_commit"].append(np.mean(ep_r_commit) if ep_r_commit else 0.0)
        history["mean_abs_action"].append(np.mean(ep_mean_abs_action) if ep_mean_abs_action else 0.0)
        history["action_saturation_fraction"].append(np.mean(ep_action_saturation) if ep_action_saturation else 0.0)

        if (ep + 1) % log_interval == 0:
            recent_reward = np.mean(history["episode_reward"][-log_interval:])
            recent_delta_f = np.mean(history["delta_f_mean"][-log_interval:])
            recent_rocof = np.mean(history["rocof_mean"][-log_interval:])
            recent_vio = np.mean(history["violation_fraction"][-log_interval:])
            recent_vio_post = np.mean(history["violation_post_inject_fraction"][-log_interval:])
            recent_nadir = np.mean(history["freq_nadir"][-log_interval:])
            recent_entropy = np.mean(history["entropy"][-log_interval:]) if history["entropy"] else 0.0
            recent_actor_std = np.mean(history["actor_std"][-log_interval:]) if history["actor_std"] else 0.0
            recent_abs_action = np.mean(history["mean_abs_action"][-log_interval:]) if history["mean_abs_action"] else 0.0
            recent_sat = np.mean(history["action_saturation_fraction"][-log_interval:]) if history["action_saturation_fraction"] else 0.0
            recent_kl = np.mean(history["approx_kl"][-log_interval:]) if history["approx_kl"] else 0.0
            recent_clip = np.mean(history["clip_fraction"][-log_interval:]) if history["clip_fraction"] else 0.0
            recent_ev = np.mean(history["explained_var"][-log_interval:]) if history["explained_var"] else 0.0

            print(
                f"Ep {ep+1:4d} | R={recent_reward:7.2f} | dF={recent_delta_f:.4f} | "
                f"RoCoF={recent_rocof:.4f} | Vio={recent_vio:.3f} | VioPost={recent_vio_post:.3f} | Nadir={recent_nadir:.3f} | "
                f"H={recent_entropy:.4f} | std={recent_actor_std:.4f} | |a|={recent_abs_action:.3f} | sat={recent_sat:.3f} | "
                f"KL={recent_kl:+.4f} | clip={recent_clip:.3f} | EV={recent_ev:+.3f}"
            )

            # Per-dim diagnostic line (only when action_dim >= 2)
            pd = history.get("per_dim", {})
            if pd and "entropy_dim1" in pd:
                def _recent(key: str) -> float:
                    vals = pd.get(key, [])
                    return float(np.mean(vals[-log_interval:])) if vals else 0.0
                parts = []
                for d in range(2):
                    parts.append(
                        f"dim{d}[std={_recent(f'std_dim{d}'):.3f} "
                        f"H={_recent(f'entropy_dim{d}'):.3f} "
                        f"|a|={_recent(f'action_abs_dim{d}'):.3f} "
                        f"g_mh={_recent(f'grad_mean_head_dim{d}'):.2e} "
                        f"g_ls={_recent(f'grad_log_std_dim{d}'):.2e}]"
                    )
                print("        " + " | ".join(parts))

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
    # ASHA-optimized defaults (Trial 5, score=-0.219)
    parser.add_argument("--lr", type=float, default=3.13e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.03)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=16)
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=-1.0,
        help="Initial pre-softplus log_std. Effective initial std = softplus(x) + 0.05.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--placement", type=str, default="artifacts/placement/official_placement_v3.json")
    parser.add_argument("--mpc-path", type=str, default="data/grid_IEEE123_complete.m")
    parser.add_argument(
        "--precomputed-dir",
        type=str,
        default="data/precomputed_365d_97to67",
        help="Directory containing day_*.parquet + eval_days.txt",
    )
    parser.add_argument(
        "--day-split",
        type=str,
        default="train",
        choices=["train", "eval", "all"],
        help="Which day partition to draw episodes from",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints_am_mappo")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--result-json", type=str, default=None)
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint (.pt with 'agent_state_dict'); load before training. Used by ASHA promotion.",
    )
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning (6000 episodes)")
    parser.add_argument("--phase-episodes", type=int, default=0, help="Override episodes per phase (0=use defaults)")
    parser.add_argument(
        "--ffr-mode",
        type=str,
        default="mappo_dual",
        choices=["droop", "mappo", "mappo_dual"],
        help=(
            "FFR control mode: "
            "'mappo_dual' (RL outputs P_ref+K_droop per DER, proposed AM dual-product, default), "
            "'mappo' (RL single-action P, replaces droop), "
            "'droop' (classical baseline)"
        ),
    )
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
        precomputed_dir=args.precomputed_dir,
        day_split=args.day_split,
        ffr_mode=args.ffr_mode,
    )

    print(f"Environment: {env.n_agents} agents, 3 VPPs, ffr_mode='{env.ffr_mode}'")

    sample_obs_fast, _, _ = env.reset()
    sample_obs_full = build_am_full_feeder_obs(env, sample_obs_fast)
    n_bus = int(sample_obs_full.shape[0])
    obs_feat = int(sample_obs_full.shape[1])
    agent_bus_indices = np.asarray(getattr(env, "_agent_bus_pp", np.arange(env.n_agents)), dtype=np.int64)
    agent_bus_indices = np.clip(agent_bus_indices, 0, max(n_bus - 1, 0))

    # Initialize agent: 2-D action per DER = (a_P, a_K) for dual-product AM
    agent = GAMAPPOAgent(
        obs_feat=obs_feat,
        n_agents=env.n_agents,
        n_bus=n_bus,
        agent_bus_indices=agent_bus_indices,
        action_dim_per_agent=2,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        log_std_init=args.log_std_init,
    )

    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume-from checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        agent.load_state_dict(ckpt["agent_state_dict"])
        print(f"Resumed agent weights from {resume_path}")

    # Setup curriculum or single-phase training
    if args.curriculum:
        phase_items = list(AM_PHASES.items())
        print("\n=== Curriculum Training (6000 episodes total) ===")
    else:
        phase_items = [
            ("FULL", {
                "n_episodes": args.n_episodes,
                "event_prob": 0.9,
                "max_delta_p_mw": 6.3,
                "event_probs": {"load_step": 0.3, "gen_trip": 0.3, "line_trip": 0.2, "high_ren": 0.2},
                "lr_factor": 1.0,
                "description": "Full difficulty",
            })
        ]
        print("\n=== Training GA-MAPPO for AM ===")

    base_lr = float(args.lr)
    all_history: dict[str, list] = {
        "episode_reward": [], "delta_f_mean": [], "max_abs_delta_f": [],
        "violation_fraction": [], "entropy": [], "phase": [],
    }
    global_ep = 0

    for phase_name, cfg in phase_items:
        n_phase_eps = args.phase_episodes if args.phase_episodes > 0 else cfg["n_episodes"]
        phase_lr = base_lr * cfg.get("lr_factor", 1.0)

        # Update learning rate
        for pg in agent.optimizer.param_groups:
            pg["lr"] = phase_lr

        # Update entropy coefficient for curriculum
        phase_entropy = cfg.get("entropy_bonus", 0.01)
        agent.entropy_coef = phase_entropy

        # Configure event injector for this phase
        env.event_injector.set_probs(cfg["event_probs"])
        env.event_injector.set_max_delta_p_mw(cfg.get("max_delta_p_mw", 6.3))

        print(f"\n=== Phase {phase_name}: {cfg.get('description', '')} ({n_phase_eps} ep, LR={phase_lr:.6f}) ===")

        history = train_am_mappo(
            env=env,
            agent=agent,
            n_episodes=n_phase_eps,
            steps_per_episode=args.steps_per_episode,
            update_epochs=args.update_epochs,
            mini_batch_size=args.mini_batch_size,
            log_interval=args.log_interval,
            checkpoint_dir=checkpoint_dir,
            _phase_name=phase_name,
            _max_delta_p_mw=cfg.get("max_delta_p_mw", 6.3),
        )

        # Accumulate history
        for k in all_history:
            if k == "phase":
                all_history[k].extend([phase_name] * len(history.get("episode_reward", [])))
            elif k in history:
                all_history[k].extend(history[k])

        global_ep += n_phase_eps

        # Save phase checkpoint
        torch.save({
            "agent_state_dict": agent.state_dict(),
            "phase": phase_name,
            "phase_config": cfg,
            "history": history,
            "args": vars(args),
        }, checkpoint_dir / f"phase_{phase_name}_final.pt")
        print(f"Phase {phase_name} complete. Checkpoint saved.")

    # Save final
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "history": all_history,
        "args": vars(args),
        "curriculum": args.curriculum,
    }, checkpoint_dir / "am_mappo_final.pt")

    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_payload = {
            "seed": int(args.seed),
            "n_episodes": int(args.n_episodes),
            "final_reward": float(history["episode_reward"][-1]) if history["episode_reward"] else float("nan"),
            "mean_reward": float(np.mean(history["episode_reward"])) if history["episode_reward"] else float("nan"),
            "mean_delta_f": float(np.mean(history["delta_f_mean"])) if history["delta_f_mean"] else float("nan"),
            "max_abs_delta_f": float(np.mean(history["max_abs_delta_f"])) if history["max_abs_delta_f"] else float("nan"),
            "mean_rocof": float(np.mean(history["rocof_mean"])) if history["rocof_mean"] else float("nan"),
            "violation_fraction": float(np.mean(history["violation_fraction"])) if history["violation_fraction"] else float("nan"),
            "mean_nadir": float(np.mean(history["freq_nadir"])) if history["freq_nadir"] else float("nan"),
            "mean_entropy": float(np.mean(history["entropy"])) if history["entropy"] else float("nan"),
            "mean_actor_log_std": float(np.mean(history["actor_log_std"])) if history["actor_log_std"] else float("nan"),
            "final_actor_log_std": float(history["actor_log_std"][-1]) if history["actor_log_std"] else float("nan"),
            "mean_actor_std": float(np.mean(history["actor_std"])) if history["actor_std"] else float("nan"),
            "final_actor_std": float(history["actor_std"][-1]) if history["actor_std"] else float("nan"),
            "mean_abs_action": float(np.mean(history["mean_abs_action"])) if history["mean_abs_action"] else float("nan"),
            "final_abs_action": float(history["mean_abs_action"][-1]) if history["mean_abs_action"] else float("nan"),
            "mean_action_saturation_fraction": float(np.mean(history["action_saturation_fraction"])) if history["action_saturation_fraction"] else float("nan"),
            "final_action_saturation_fraction": float(history["action_saturation_fraction"][-1]) if history["action_saturation_fraction"] else float("nan"),
            "checkpoint_dir": str(checkpoint_dir),
        }
        result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    print(f"\nTraining complete. Saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
