"""
Evaluation script focused on FFR capability and topology adaptation.

Paper contributions demonstrated:
1. GraphSAGE-MAPPO provides effective Fast Frequency Response (FFR)
2. GraphSAGE enables topology-invariant control (vs MLP baseline)

Key tables:
- Table 1: FFR performance comparison across methods
- Table 2: Topology generalization (seen vs unseen topologies)
- Table 3: Contingency severity scaling
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.events import EventConfig
from src.env.microgrid_env_dual import MicrogridEnvDual
from src.rl.train_am_mappo import (
    GAMAPPOAgent,
    build_am_full_feeder_obs,
    ensure_edge_index,
    RunningNormalizer,
)


# =============================================================================
# FFR Metrics (ENTSO-E / Nordic Grid Code aligned)
# =============================================================================

@dataclass
class FFRMetrics:
    """FFR performance metrics per episode (IEEE 1547 / ENTSO-E aligned)."""
    nadir_hz: float = 50.0
    zenith_hz: float = 50.0
    rocof_max_hz_s: float = 0.0
    delta_f_max_hz: float = 0.0
    settling_time_s: float = 0.0
    iae_total: float = 0.0
    iae_post: float = 0.0
    itae: float = 0.0
    time_in_violation_s: float = 0.0
    ffr_success: bool = True
    f_trace: np.ndarray = field(default_factory=lambda: np.array([]))
    f_trace_hires: np.ndarray = field(default_factory=lambda: np.array([]))
    dt_hires: float = 0.1

    def to_dict(self) -> dict[str, float]:
        return {
            "nadir_hz": self.nadir_hz,
            "zenith_hz": self.zenith_hz,
            "rocof_max_hz_s": self.rocof_max_hz_s,
            "delta_f_max_hz": self.delta_f_max_hz,
            "settling_time_s": self.settling_time_s,
            "iae_total": self.iae_total,
            "iae_post": self.iae_post,
            "itae": self.itae,
            "time_in_violation_s": self.time_in_violation_s,
            "ffr_success": float(self.ffr_success),
        }


def compute_itae(
    f_trace: np.ndarray,
    f_nominal: float,
    dt: float,
    t_event: float,
) -> float:
    """Compute ITAE (Integral of Time-weighted Absolute Error).

    ITAE = ∫ t·|Δf(t)|dt from t_event onwards
    Penalizes slow recovery more than IAE.
    """
    t = np.arange(len(f_trace)) * dt
    delta_f = np.abs(f_trace - f_nominal)
    mask = t >= t_event
    if not np.any(mask):
        return 0.0
    t_rel = t[mask] - t_event
    return float(np.trapezoid(t_rel * delta_f[mask], t_rel))


def compute_jaccard_edge_distance(edge_index_a: np.ndarray, edge_index_b: np.ndarray) -> float:
    """Compute Jaccard edge distance between two graphs.

    d_E = 1 - |E_a ∩ E_b| / |E_a ∪ E_b|

    Returns value in [0, 1] where 0 = identical topology, 1 = disjoint edges.
    """
    def to_edge_set(ei: np.ndarray) -> set[tuple[int, int]]:
        if ei.size == 0:
            return set()
        return set(map(tuple, ei.T))

    edges_a = to_edge_set(edge_index_a)
    edges_b = to_edge_set(edge_index_b)
    intersection = len(edges_a & edges_b)
    union = len(edges_a | edges_b)
    return 1.0 - intersection / union if union > 0 else 0.0


def compute_farthest_split(
    topologies: list[np.ndarray],
    n_test: int = 5,
) -> tuple[list[int], list[int], dict[str, float]]:
    """Compute train/test split maximizing minimum edge distance.

    Uses greedy farthest-point selection:
    1. Start with all topologies in train set
    2. Iteratively move topology with max d_min to test set
    3. Repeat until |test| = n_test

    Returns:
        train_ids: List of train topology indices
        test_ids: List of test topology indices
        stats: Dict with d_min_mean, d_min_min, d_min_max
    """
    n_topos = len(topologies)
    if n_test >= n_topos:
        return [], list(range(n_topos)), {"d_min_mean": 0.0, "d_min_min": 0.0, "d_min_max": 0.0}

    # Compute pairwise distance matrix
    dist_matrix = np.zeros((n_topos, n_topos))
    for i in range(n_topos):
        for j in range(i + 1, n_topos):
            d = compute_jaccard_edge_distance(topologies[i], topologies[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    train_ids = list(range(n_topos))
    test_ids: list[int] = []

    for _ in range(n_test):
        # Find topology in train with max d_min to other train topologies
        best_idx = -1
        best_d_min = -1.0

        for idx in train_ids:
            other_train = [t for t in train_ids if t != idx]
            if not other_train:
                d_min = 0.0
            else:
                d_min = min(dist_matrix[idx, t] for t in other_train)
            if d_min > best_d_min:
                best_d_min = d_min
                best_idx = idx

        if best_idx >= 0:
            train_ids.remove(best_idx)
            test_ids.append(best_idx)

    # Compute stats: distance from each test to nearest train
    d_mins = []
    for test_idx in test_ids:
        if train_ids:
            d_min = min(dist_matrix[test_idx, t] for t in train_ids)
            d_mins.append(d_min)

    stats = {
        "d_min_mean": float(np.mean(d_mins)) if d_mins else 0.0,
        "d_min_min": float(np.min(d_mins)) if d_mins else 0.0,
        "d_min_max": float(np.max(d_mins)) if d_mins else 0.0,
    }

    return train_ids, test_ids, stats


def compute_ffr_metrics(
    f_trace: np.ndarray,
    rocof_trace: np.ndarray,
    f_nominal: float = 50.0,
    f_limit: float = 0.5,
    settle_band: float = 0.02,
    event_step: int = 30,
    post_window: int = 50,
    dt: float = 1.0,
    rocof_limit: float = 2.0,
) -> FFRMetrics:
    """Compute FFR metrics from frequency trajectory (IEEE 1547 Cat III aligned)."""
    delta_f = f_trace - f_nominal

    # Basic metrics
    nadir = float(np.min(f_trace))
    zenith = float(np.max(f_trace))
    rocof_max = float(np.max(np.abs(rocof_trace)))
    delta_f_max = float(np.max(np.abs(delta_f)))

    # IAE (Integral Absolute Error)
    iae_total = float(np.trapezoid(np.abs(delta_f), dx=dt))

    # Post-event metrics
    post_start = min(event_step, len(delta_f) - 1)
    post_end = min(event_step + post_window, len(delta_f))
    post_delta_f = delta_f[post_start:post_end]
    iae_post = float(np.trapezoid(np.abs(post_delta_f), dx=dt)) if len(post_delta_f) > 0 else 0.0

    # ITAE (Integral of Time-weighted Absolute Error)
    itae = compute_itae(f_trace, f_nominal, dt, t_event=event_step * dt)

    # Settling time (time to stay within deadband)
    settling_time = float(post_window * dt)
    abs_post = np.abs(post_delta_f)
    for i in range(len(abs_post)):
        if np.all(abs_post[i:] <= settle_band):
            settling_time = float(i * dt)
            break

    # Time in violation
    in_violation = np.abs(delta_f) > f_limit
    time_violation = float(np.sum(in_violation) * dt)

    # FFR success: continuous violation < 300 ms (IEEE 81 islanded UFLS delay) AND RoCoF ≤ 2.0 Hz/s
    # 300 ms = 3 hi-res samples @ dt=0.1s; time_violation sums all under-threshold samples
    ffr_success = (time_violation <= 0.3) and (rocof_max <= rocof_limit)

    return FFRMetrics(
        nadir_hz=nadir,
        zenith_hz=zenith,
        rocof_max_hz_s=rocof_max,
        delta_f_max_hz=delta_f_max,
        settling_time_s=settling_time,
        iae_total=iae_total,
        iae_post=iae_post,
        itae=itae,
        time_in_violation_s=time_violation,
        ffr_success=ffr_success,
        f_trace=f_trace,
    )


# =============================================================================
# Policy Wrappers
# =============================================================================

class NoFFRPolicy:
    """Baseline: No frequency response."""
    def act(self, obs: np.ndarray, edge_index: np.ndarray, env: Any, obs_fast: np.ndarray | None = None) -> np.ndarray:
        n_agents = env.n_agents
        n_vpps = len(env._vpp_droop_agents)
        return np.zeros(n_agents + n_vpps, dtype=np.float32)


class FixedDroopPolicy:
    """Baseline: Fixed droop control (linear proportional, 5% droop per IEEE/ENTSO-E)."""
    def __init__(self, k_droop: float = 0.05):
        self.k_droop = k_droop

    def act(self, obs: np.ndarray, edge_index: np.ndarray, env: Any, obs_fast: np.ndarray | None = None) -> np.ndarray:
        freq_state = env.freq_dyn.get_state()
        delta_f = freq_state.delta_f_hz
        control = -self.k_droop * np.clip(delta_f / 0.5, -1.0, 1.0)

        n_agents = env.n_agents
        n_vpps = len(env._vpp_droop_agents)
        action = np.full(n_agents + n_vpps, control, dtype=np.float32)
        return np.clip(action, -1.0, 1.0)


class GraphSAGEMAPPOPolicy:
    """Our method: GraphSAGE-MAPPO trained agent."""
    def __init__(self, checkpoint_path: Path, env: MicrogridEnvDual):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract dimensions from checkpoint
        agent_state = ckpt.get("agent_state_dict", ckpt)

        # Infer hidden_dim and embed_dim from checkpoint weights
        hidden_dim = 64
        embed_dim = 64
        if "encoder.encoder.layer2.w_self.weight" in agent_state:
            embed_dim = agent_state["encoder.encoder.layer2.w_self.weight"].shape[0]
        if "encoder.encoder.layer1.w_self.weight" in agent_state:
            hidden_dim = agent_state["encoder.encoder.layer1.w_self.weight"].shape[0]

        # Build agent
        sample_obs_fast, _, _ = env.reset()
        sample_obs_full = build_am_full_feeder_obs(env, sample_obs_fast)
        n_bus = sample_obs_full.shape[0]
        obs_feat = sample_obs_full.shape[1]

        agent_bus_indices = np.asarray(
            getattr(env, "_agent_bus_pp", np.arange(env.n_agents)), dtype=np.int64
        )
        agent_bus_indices = np.clip(agent_bus_indices, 0, max(n_bus - 1, 0))

        self.agent = GAMAPPOAgent(
            obs_feat=obs_feat,
            n_agents=env.n_agents,
            n_bus=n_bus,
            agent_bus_indices=agent_bus_indices,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )
        self.agent.load_state_dict(agent_state)
        self.agent.eval()

        # Observation normalizer
        self.obs_normalizer = RunningNormalizer(sample_obs_full.shape)
        if "obs_normalizer" in ckpt:
            norm_data = ckpt["obs_normalizer"]
            self.obs_normalizer.mean = norm_data["mean"]
            self.obs_normalizer.var = norm_data["var"]
            self.obs_normalizer.count = norm_data["count"]

    def act(self, obs: np.ndarray, edge_index: np.ndarray, env: Any, obs_fast: np.ndarray | None = None) -> np.ndarray:
        obs_norm = self.obs_normalizer.normalize(obs)
        action = self.agent.act_deterministic(obs_norm, edge_index)

        # Negate for droop response convention
        control_actions = -action

        n_agents = env.n_agents
        n_vpps = len(env._vpp_droop_agents)
        full_action = np.zeros(n_agents + n_vpps, dtype=np.float32)
        full_action[:n_agents] = control_actions.flatten()[:n_agents]

        for vpp_idx, (_, member_agents) in enumerate(env._vpp_droop_agents.items()):
            vpp_action = np.mean([control_actions[ai, 0] for ai in member_agents if ai < len(control_actions)])
            full_action[n_agents + vpp_idx] = vpp_action

        return np.clip(full_action, -1.0, 1.0)


class GCNNPPOPolicy:
    """Baseline: GCNN-PPO (Guo et al. 2024).

    Spectral GCN + single centralized PPO. Uses legacy 5-column obs_fast.
    """
    def __init__(self, checkpoint_path: Path, env: MicrogridEnvDual):
        from src.baselines.gcnn_ppo import GCNNPPOAgent
        self._agent = GCNNPPOAgent.load(checkpoint_path)
        self._env = env

    @staticmethod
    def _to_legacy_obs5(obs_fast: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs_fast, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError(f"Expected obs_fast (N,>=5), got {arr.shape}")
        return arr[:, :5]

    def _map_to_env_action(self, raw: np.ndarray, env: Any) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        n_agents = env.n_agents
        p_all = raw[:n_agents]
        droop_all = raw[n_agents:2 * n_agents] if raw.size >= 2 * n_agents else np.zeros(n_agents, dtype=np.float32)
        n_vpps = len(env._vpp_droop_agents)
        vpp_droop = np.zeros(n_vpps, dtype=np.float32)
        for vpp_idx, (_, members) in enumerate(env._vpp_droop_agents.items()):
            members = [m for m in members if m < n_agents]
            vpp_droop[vpp_idx] = float(np.mean(droop_all[members])) if members else 0.0
        return np.clip(np.concatenate([p_all, vpp_droop]).astype(np.float32), -1.0, 1.0)

    def act(self, obs: np.ndarray, edge_index: np.ndarray, env: Any, obs_fast: np.ndarray | None = None) -> np.ndarray:
        # GCNN-PPO is trained on build_am_full_feeder_obs (shape (n_bus, 20))
        # not the legacy 10-feature _combine_obs. Use the same obs builder
        # the trainer's rollout_episode uses, otherwise the encoder's
        # in_dim=20 will mismatch the eval-side 10-feature input.
        from src.rl.train_am_mappo import build_am_full_feeder_obs
        if obs_fast is None:
            raise ValueError("GCNNPPOPolicy requires obs_fast")
        obs_full = build_am_full_feeder_obs(env, obs_fast)
        global_obs = obs_full.reshape(-1)
        action_env, _lp, _v, _raw = self._agent.act(obs_full, edge_index, global_obs)
        return self._map_to_env_action(action_env, env)


class MATD3Policy:
    """Baseline: MATD3 (Li & Zhou 2025, base algorithm without EIE enhancements).

    CTDE multi-agent TD3 with MLP encoder. Builds per-agent obs from env state.
    """
    def __init__(self, checkpoint_path: Path, env: MicrogridEnvDual):
        from src.baselines.matd3 import MATD3Agent, MATD3Config
        config = MATD3Config(obs_dim=24, action_dim=2, n_agents=env.n_agents)
        self._agent = MATD3Agent(config, device="cpu")
        self._agent.load(checkpoint_path)
        self._agent.eval()
        self._env = env

    def _build_obs(self, obs_fast: np.ndarray, env: Any) -> np.ndarray:
        n_agents = env.n_agents
        obs = np.zeros((n_agents, 24), dtype=np.float32)
        freq_state = env.freq_dyn.get_state()
        for i in range(n_agents):
            obs[i, 0] = freq_state.delta_f_hz
            obs[i, 1] = freq_state.rocof_hz_s
            if i < len(obs_fast):
                tail = obs_fast[i][:22]
                obs[i, 2:2 + len(tail)] = tail
        return obs

    def act(self, obs: np.ndarray, edge_index: np.ndarray, env: Any, obs_fast: np.ndarray | None = None) -> np.ndarray:
        if obs_fast is None:
            raise ValueError("MATD3Policy requires obs_fast")
        per_agent_obs = self._build_obs(obs_fast, env)
        actions = self._agent.act_deterministic(per_agent_obs)
        p_actions = actions[:, 0]
        n_vpps = len(env._vpp_droop_agents)
        vpp_droop = np.zeros(n_vpps, dtype=np.float32)
        for vpp_idx, (_, members) in enumerate(env._vpp_droop_agents.items()):
            valid = [m for m in members if m < len(actions)]
            if valid:
                vpp_droop[vpp_idx] = float(np.mean([actions[a, 1] for a in valid]))
        return np.clip(np.concatenate([p_actions, vpp_droop]).astype(np.float32), -1.0, 1.0)


class MLPMAPPOPolicy:
    """Ablation: MAPPO with MLP encoder (no graph message passing).

    Same RL stack as the proposed GraphSAGE-MAPPO but with an MLP encoder so the
    only differing variable is the graph encoder. This is the most important
    ablation for proving GraphSAGE's contribution to topology generalization.
    """
    def __init__(self, checkpoint_path: Path, env: MicrogridEnvDual):
        from src.baselines.train_mlp_mappo import MLPMAPPOAgent

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        agent_state = ckpt.get("model_state_dict", ckpt.get("agent_state_dict", ckpt))

        # Infer dimensions from checkpoint weights
        embed_dim = 64
        hidden_dim = 128
        if "encoder.net.0.weight" in agent_state:
            hidden_dim = agent_state["encoder.net.0.weight"].shape[0]
        if "encoder.net.4.weight" in agent_state:
            embed_dim = agent_state["encoder.net.4.weight"].shape[0]

        # Build agent
        sample_obs_fast, _, _ = env.reset()
        sample_obs_full = build_am_full_feeder_obs(env, sample_obs_fast)
        n_bus = sample_obs_full.shape[0]
        obs_feat = sample_obs_full.shape[1]

        agent_bus_indices = np.asarray(
            getattr(env, "_agent_bus_pp", np.arange(env.n_agents)), dtype=np.int64
        )
        agent_bus_indices = np.clip(agent_bus_indices, 0, max(n_bus - 1, 0))

        self.agent = MLPMAPPOAgent(
            obs_feat=obs_feat,
            n_agents=env.n_agents,
            n_bus=n_bus,
            agent_bus_indices=agent_bus_indices,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )
        self.agent.load_state_dict(agent_state)
        self.agent.eval()

        # Observation normalizer
        self.obs_normalizer = RunningNormalizer(sample_obs_full.shape)
        if "obs_normalizer" in ckpt:
            norm_data = ckpt["obs_normalizer"]
            self.obs_normalizer.mean = norm_data["mean"]
            self.obs_normalizer.var = norm_data["var"]
            self.obs_normalizer.count = norm_data["count"]

    def act(self, obs: np.ndarray, edge_index: np.ndarray, env: Any, obs_fast: np.ndarray | None = None) -> np.ndarray:
        obs_norm = self.obs_normalizer.normalize(obs)
        action = self.agent.act_deterministic(obs_norm, edge_index)

        # Negate for droop response convention
        control_actions = -action

        n_agents = env.n_agents
        n_vpps = len(env._vpp_droop_agents)
        full_action = np.zeros(n_agents + n_vpps, dtype=np.float32)
        full_action[:n_agents] = control_actions.flatten()[:n_agents]

        for vpp_idx, (_, member_agents) in enumerate(env._vpp_droop_agents.items()):
            vpp_action = np.mean([control_actions[ai, 0] for ai in member_agents if ai < len(control_actions)])
            full_action[n_agents + vpp_idx] = vpp_action

        return np.clip(full_action, -1.0, 1.0)


# =============================================================================
# Evaluation Runner
# =============================================================================

class FFRTopologyEvaluator:
    """Evaluator for FFR + topology adaptation experiments."""

    def __init__(
        self,
        env_config: dict[str, Any],
        checkpoint_path: Path | None = None,
        output_dir: Path = Path("results/ffr_topology"),
        gcnn_checkpoint: Path | None = None,
        matd3_checkpoint: Path | None = None,
        mlp_mappo_checkpoint: Path | None = None,
    ):
        self.env = MicrogridEnvDual(**env_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build policies (ladder per Baseline Comparison.md §1)
        self.policies: dict[str, Any] = {
            "No FFR": NoFFRPolicy(),
            "Fixed Droop": FixedDroopPolicy(k_droop=0.05),
        }

        def _try_load(name: str, ctor) -> None:
            try:
                self.policies[name] = ctor()
                print(f"Loaded {name}")
            except Exception as exc:
                print(f"[warn] Failed to load {name}: {exc}")

        if matd3_checkpoint and matd3_checkpoint.exists():
            _try_load("MATD3", lambda: MATD3Policy(matd3_checkpoint, self.env))
        if gcnn_checkpoint and gcnn_checkpoint.exists():
            _try_load("GCNN-PPO", lambda: GCNNPPOPolicy(gcnn_checkpoint, self.env))
        if mlp_mappo_checkpoint and mlp_mappo_checkpoint.exists():
            _try_load("MLP-MAPPO", lambda: MLPMAPPOPolicy(mlp_mappo_checkpoint, self.env))
        if checkpoint_path and checkpoint_path.exists():
            _try_load("GraphSAGE-MAPPO", lambda: GraphSAGEMAPPOPolicy(checkpoint_path, self.env))

        # Contingency scenarios (Section VI - 4 scenarios aligned with IEEE 1547)
        # S_BASE = 15.705 MW, H_SYS = 1.18s
        # Generator locations: bus 67 (3MW), 105 (3MW), 101 (3MW), 98 (3MW), 60 (2MW)
        self.scenarios = {
            "S1_load_step": EventConfig(type="load_step", delta_P_mw=2.5, location=45, t_inject=30.0),
            "S2_gen_trip": EventConfig(type="gen_trip", delta_P_mw=-3.9, location=67, t_inject=30.0),
            "S3_line_trip": EventConfig(type="line_trip", delta_P_mw=-2.4, location=67068, t_inject=30.0),
            "S4_high_ren_surge": EventConfig(type="high_ren", delta_P_mw=4.7, location=105, t_inject=30.0),
        }

        # Topology splits using farthest-point selection
        topo_cache = getattr(self.env.reconfig, "_cache", [])
        n_topos = len(topo_cache)
        if n_topos >= 5:
            # Extract edge_index from each cached topology.
            # Cache entries are 3-tuples (net, edge_index, open_set); legacy
            # dict format is also supported.
            topo_edges = []
            for topo in topo_cache:
                if isinstance(topo, dict):
                    ei = topo.get("edge_index", np.array([[], []]))
                elif isinstance(topo, tuple) and len(topo) >= 2:
                    ei = topo[1]
                else:
                    ei = np.array([[], []])
                topo_edges.append(np.asarray(ei))
            self.train_topologies, self.test_topologies, split_stats = compute_farthest_split(topo_edges, n_test=5)
            print(f"Farthest-point split: {len(self.train_topologies)} train, {len(self.test_topologies)} test")
            print(f"  d_min: mean={split_stats['d_min_mean']:.4f}, min={split_stats['d_min_min']:.4f}, max={split_stats['d_min_max']:.4f}")
            self.split_stats = split_stats
        else:
            self.train_topologies = list(range(min(15, n_topos)))
            self.test_topologies = list(range(15, min(20, n_topos)))
            self.split_stats = {"d_min_mean": 0.0, "d_min_min": 0.0, "d_min_max": 0.0}
            print(f"Topologies: {len(self.train_topologies)} train, {len(self.test_topologies)} test (sequential split)")

    def run_episode(
        self,
        policy: Any,
        event: EventConfig | None = None,
        topology_idx: int | None = None,
        n_steps: int = 300,
    ) -> FFRMetrics:
        """Run single episode and compute FFR metrics."""
        options = {}
        if event is not None:
            options["force_event"] = deepcopy(event)
        if topology_idx is not None:
            options["force_topology"] = topology_idx

        obs_fast, _, _ = self.env.reset(options=options)
        n_bus = len(self.env.net.bus.index)
        edge_index = ensure_edge_index(self.env.edge_index, n_nodes=n_bus)

        obs_full = build_am_full_feeder_obs(self.env, obs_fast)

        f_trace = []
        rocof_trace = []
        # Hi-resolution trace (one sample per ODE sub-step, dt ~= 0.1 s)
        f_hires: list[float] = []
        # Clear the env's hi-res buffer if present; it accumulates over the episode
        self.env._hires_df = []

        for _ in range(n_steps):
            action = policy.act(obs_full, edge_index, self.env, obs_fast=obs_fast)
            obs_fast, _, done, _, info = self.env.step_fast(action)

            # Update graph if topology changed
            new_edge = info.get("edge_index", edge_index)
            edge_index = ensure_edge_index(new_edge, n_nodes=n_bus)
            obs_full = build_am_full_feeder_obs(self.env, obs_fast)

            freq_state = self.env.freq_dyn.get_state()
            f_trace.append(50.0 + freq_state.delta_f_hz)
            rocof_trace.append(freq_state.rocof_hz_s)

        # Snapshot hi-res trace (50.0 + delta_f) — env appended one entry per sub-step
        f_hires = [50.0 + df for df in self.env._hires_df]

        event_step = int(event.t_inject) if event else 30
        metrics = compute_ffr_metrics(
            np.array(f_trace),
            np.array(rocof_trace),
            event_step=event_step,
        )
        # Attach hi-res trace for plotting helpers
        metrics.f_trace_hires = np.asarray(f_hires, dtype=np.float32)
        metrics.dt_hires = float(getattr(self.env, "dt_ode_s", 0.1))
        return metrics

    def build_table1_ffr_comparison(self, n_runs: int = 20) -> pd.DataFrame:
        """Table 1: FFR performance comparison across methods and scenarios."""
        rows = []

        for scenario_name, event in self.scenarios.items():
            for policy_name, policy in self.policies.items():
                metrics_list = []

                for run in range(n_runs):
                    # Random topology for diversity
                    topo_idx = self.train_topologies[run % len(self.train_topologies)] if self.train_topologies else None
                    m = self.run_episode(policy, event=event, topology_idx=topo_idx)
                    metrics_list.append(m.to_dict())

                # Aggregate
                for metric_name in metrics_list[0].keys():
                    vals = [m[metric_name] for m in metrics_list]
                    rows.append({
                        "scenario": scenario_name,
                        "method": policy_name,
                        "metric": metric_name,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                    })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "table1_ffr_comparison.csv", index=False)
        return df

    def build_table2_topology_adaptation(self, n_runs: int = 10) -> pd.DataFrame:
        """Table 2: Topology generalization - train vs unseen topologies."""
        # Use gen_trip as stress test scenario
        event = self.scenarios["S2_gen_trip"]
        rows = []

        for policy_name, policy in self.policies.items():
            for split_name, topo_list in [("train", self.train_topologies), ("unseen", self.test_topologies)]:
                if not topo_list:
                    continue

                metrics_agg = {
                    "ffr_success_rate": [],
                    "nadir_hz": [],
                    "rocof_max_hz_s": [],
                    "settling_time_s": [],
                    "iae_post": [],
                }

                for topo_idx in topo_list:
                    for _ in range(n_runs):
                        m = self.run_episode(policy, event=event, topology_idx=topo_idx)
                        metrics_agg["ffr_success_rate"].append(float(m.ffr_success))
                        metrics_agg["nadir_hz"].append(m.nadir_hz)
                        metrics_agg["rocof_max_hz_s"].append(m.rocof_max_hz_s)
                        metrics_agg["settling_time_s"].append(m.settling_time_s)
                        metrics_agg["iae_post"].append(m.iae_post)

                rows.append({
                    "method": policy_name,
                    "topology_split": split_name,
                    "n_topologies": len(topo_list),
                    "ffr_success_rate": float(np.mean(metrics_agg["ffr_success_rate"])),
                    "ffr_success_std": float(np.std(metrics_agg["ffr_success_rate"])),
                    "nadir_hz_mean": float(np.mean(metrics_agg["nadir_hz"])),
                    "nadir_hz_std": float(np.std(metrics_agg["nadir_hz"])),
                    "rocof_max_mean": float(np.mean(metrics_agg["rocof_max_hz_s"])),
                    "settling_time_mean": float(np.mean(metrics_agg["settling_time_s"])),
                    "iae_post_mean": float(np.mean(metrics_agg["iae_post"])),
                })

        df = pd.DataFrame(rows)

        # Compute generalization gap (only if we collected rows; topology cache may be empty)
        if not df.empty and "method" in df.columns:
            for policy_name in self.policies.keys():
                train_row = df[(df["method"] == policy_name) & (df["topology_split"] == "train")]
                test_row = df[(df["method"] == policy_name) & (df["topology_split"] == "unseen")]

                if not train_row.empty and not test_row.empty:
                    train_success = train_row["ffr_success_rate"].values[0]
                    test_success = test_row["ffr_success_rate"].values[0]
                    gap = train_success - test_success
                    retention = test_success / train_success if train_success > 0 else 0

                    df.loc[(df["method"] == policy_name) & (df["topology_split"] == "unseen"), "generalization_gap"] = gap
                    df.loc[(df["method"] == policy_name) & (df["topology_split"] == "unseen"), "retention_ratio"] = retention
        else:
            print("[warn] Topology cache empty — Table 2 skipped (no train/test topologies)")

        df.to_csv(self.output_dir / "table2_topology_adaptation.csv", index=False)
        return df

    def build_table3_severity_scaling(self, n_runs: int = 10) -> pd.DataFrame:
        """Table 3: Performance across contingency severities."""
        severity_scenarios = {
            "mild_2MW": EventConfig(type="load_step", delta_P_mw=2.0, location=45, t_inject=30.0),
            "moderate_4MW": EventConfig(type="gen_trip", delta_P_mw=-4.0, location=97, t_inject=30.0),
            "severe_6MW": EventConfig(type="gen_trip", delta_P_mw=-6.0, location=97, t_inject=30.0),
        }

        rows = []
        for severity_name, event in severity_scenarios.items():
            for policy_name, policy in self.policies.items():
                metrics_list = []

                for _ in range(n_runs):
                    topo_idx = self.train_topologies[0] if self.train_topologies else None
                    m = self.run_episode(policy, event=event, topology_idx=topo_idx)
                    metrics_list.append(m.to_dict())

                rows.append({
                    "severity": severity_name,
                    "delta_P_mw": abs(event.delta_P_mw),
                    "method": policy_name,
                    "ffr_success_rate": float(np.mean([m["ffr_success"] for m in metrics_list])),
                    "nadir_hz_mean": float(np.mean([m["nadir_hz"] for m in metrics_list])),
                    "rocof_max_mean": float(np.mean([m["rocof_max_hz_s"] for m in metrics_list])),
                    "settling_time_mean": float(np.mean([m["settling_time_s"] for m in metrics_list])),
                })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "table3_severity_scaling.csv", index=False)
        return df

    # ------------------------------------------------------------------ styling
    _METHOD_PALETTE = {
        "GraphSAGE-MAPPO": ("#1f3a93", 2.6, "-"),     # Proposed: thick, bold blue
        "MLP-MAPPO":       ("#e94e77", 1.6, "--"),
        "GCNN-PPO":        ("#f5a623", 1.6, "--"),
        "MATD3":           ("#6a737d", 1.6, "--"),
        "Fixed Droop":     ("#56c596", 1.4, ":"),
        "No FFR":          ("#a04668", 1.4, ":"),
    }

    def _style_for(self, name: str) -> tuple[str, float, str]:
        return self._METHOD_PALETTE.get(name, ("#444444", 1.4, "-"))

    def _annotate_trace(self, ax, f_trace: np.ndarray, dt: float, event_step: int, color: str) -> None:
        """Mark nadir + approximate settling on a freq trace."""
        nadir_idx = int(np.argmin(f_trace))
        ax.scatter([nadir_idx * dt], [f_trace[nadir_idx]], s=22, color=color, zorder=5,
                   edgecolor="black", linewidth=0.6)

    def plot_frequency_traces(
        self,
        scenario_name: str = "S2_gen_trip",
        zoom_post_event_s: float | None = 20.0,
        save_csv: bool = True,
    ) -> None:
        """Plot frequency traces comparing methods (single scenario), hi-res.

        Uses the env sub-step (≈0.1 s) trace if available so the true nadir
        of the underdamped ring-down is captured. Adds the cooperative control
        windows (FFR / Primary / AGC) and a 20 s post-event zoom by default.
        Optionally saves the per-method hi-res trace as a CSV.
        """
        event = self.scenarios.get(scenario_name)
        if event is None:
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        event_t = float(event.t_inject)

        # Collect traces — fall back to no forced topology if cache is empty
        topo_idx = 0 if getattr(self.env.reconfig, "_cache", []) else None
        traces: dict[str, tuple[np.ndarray, float]] = {}
        for policy_name, policy in self.policies.items():
            m = self.run_episode(policy, event=event, topology_idx=topo_idx)
            if m.f_trace_hires.size > 0:
                traces[policy_name] = (np.asarray(m.f_trace_hires, dtype=float), float(m.dt_hires))
            else:
                traces[policy_name] = (np.asarray(m.f_trace, dtype=float), 1.0)

        # Optional CSV dump for reuse / external plotting
        if save_csv and traces:
            csv_rows: list[dict[str, float]] = []
            dt_csv = next(iter(traces.values()))[1]
            L_csv = min(len(arr) for arr, _ in traces.values())
            for k in range(L_csv):
                row = {"t_s": k * dt_csv}
                for name, (arr, _) in traces.items():
                    row[name] = float(arr[k])
                csv_rows.append(row)
            pd.DataFrame(csv_rows).to_csv(
                self.output_dir / f"trace_{scenario_name}.csv", index=False
            )

        # Plot frequency
        ax1 = axes[0]
        ax1.axvspan(event_t,        event_t + 2.0,  color="#3a7bd5", alpha=0.08, label="FFR 0-2s")
        ax1.axvspan(event_t + 2.0,  event_t + 10.0, color="#f5a623", alpha=0.08, label="Primary 2-10s")
        for name, (trace, dt_loc) in traces.items():
            color, lw, ls = self._style_for(name)
            t = np.arange(len(trace)) * dt_loc
            ax1.plot(t, trace, label=name, color=color, linewidth=lw, linestyle=ls)
        ax1.axhline(49.5, ls="--", color="red", alpha=0.7, label="UFLS threshold (49.5 Hz)")
        ax1.axhline(50.0, ls=":", color="gray", alpha=0.5)
        ax1.axvline(event_t, ls="--", color="orange", alpha=0.7, label=f"Event @ t={event_t:g}s")
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_ylim(48.7, 51.0)
        if zoom_post_event_s is not None:
            ax1.set_xlim(max(0.0, event_t - 5.0), event_t + float(zoom_post_event_s))
            ax1.axvspan(event_t + 10.0, event_t + float(zoom_post_event_s),
                        color="#56c596", alpha=0.08, label="AGC ≥10s")
        else:
            ax1.axvspan(event_t + 10.0, ax1.get_xlim()[1],
                        color="#56c596", alpha=0.08, label="AGC ≥10s")
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(alpha=0.3)
        ax1.set_title(f"FFR Performance: {scenario_name}")

        # Plot deviation
        ax2 = axes[1]
        ax2.axvspan(event_t,       event_t + 2.0,  color="#3a7bd5", alpha=0.08)
        ax2.axvspan(event_t + 2.0, event_t + 10.0, color="#f5a623", alpha=0.08)
        for name, (trace, dt_loc) in traces.items():
            color, lw, ls = self._style_for(name)
            t = np.arange(len(trace)) * dt_loc
            delta_f = trace - 50.0
            ax2.plot(t, delta_f, label=name, color=color, linewidth=lw, linestyle=ls)
        ax2.axhline(-0.5, ls="--", color="red", alpha=0.7)
        ax2.axhline(0.0, ls=":", color="gray", alpha=0.5)
        ax2.axvline(event_t, ls="--", color="orange", alpha=0.7)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Δf (Hz)")
        ax2.set_ylim(-1.5, 1.0)
        if zoom_post_event_s is not None:
            ax2.set_xlim(max(0.0, event_t - 5.0), event_t + float(zoom_post_event_s))
            ax2.axvspan(event_t + 10.0, event_t + float(zoom_post_event_s),
                        color="#56c596", alpha=0.08)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.output_dir / f"fig_freq_trace_{scenario_name}.pdf", dpi=300)
        fig.savefig(self.output_dir / f"fig_freq_trace_{scenario_name}.png", dpi=150)
        plt.close(fig)
        print(f"Saved frequency trace plot: {scenario_name}")

    def plot_topology_comparison(self) -> None:
        """Plot FFR success rate: train vs unseen topologies."""
        if "GraphSAGE-MAPPO" not in self.policies:
            print("GraphSAGE-MAPPO not loaded, skipping topology plot")
            return

        event = self.scenarios["S2_gen_trip"]

        # Collect per-topology success rates
        results = {"train": [], "unseen": []}

        for split_name, topo_list in [("train", self.train_topologies), ("unseen", self.test_topologies)]:
            for topo_idx in topo_list:
                successes = []
                for _ in range(5):
                    m = self.run_episode(self.policies["GraphSAGE-MAPPO"], event=event, topology_idx=topo_idx)
                    successes.append(float(m.ffr_success))
                results[split_name].append(np.mean(successes))

        fig, ax = plt.subplots(figsize=(8, 5))

        x_train = np.arange(len(results["train"]))
        x_unseen = np.arange(len(results["unseen"])) + len(results["train"]) + 1

        ax.bar(x_train, results["train"], color="steelblue", label="Train topologies", alpha=0.8)
        ax.bar(x_unseen, results["unseen"], color="coral", label="Unseen topologies", alpha=0.8)

        ax.axhline(np.mean(results["train"]), ls="--", color="steelblue", alpha=0.7)
        ax.axhline(np.mean(results["unseen"]), ls="--", color="coral", alpha=0.7)

        ax.set_xlabel("Topology Index")
        ax.set_ylabel("FFR Success Rate")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.set_title("GraphSAGE-MAPPO: Topology Generalization")
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        fig.savefig(self.output_dir / "fig_topology_generalization.pdf", dpi=300)
        fig.savefig(self.output_dir / "fig_topology_generalization.png", dpi=150)
        plt.close(fig)
        print("Saved topology generalization plot")

    def plot_frequency_grid_all_scenarios(
        self,
        n_runs: int = 3,
        topology_idx: int | None = 0,
        dt: float = 1.0,
        ufls_hz: float = 49.5,
        settle_band: float = 0.1,
        zoom_post_event_s: float | None = 20.0,
        save_suffix: str = "",
    ) -> None:
        """2×2 grid: S1-S4 freq response, 6 methods overlaid with mean ± std band.

        Per-subplot annotations:
          - Mean trace per method, ±1σ shaded band over ``n_runs`` runs.
          - Proposed (GraphSAGE-MAPPO) drawn thicker on top so it pops.
          - Vertical line at event injection.
          - Horizontal lines at f_0 (50 Hz) and UFLS threshold (49.5 Hz).
          - Settling band ``50 ± settle_band`` shaded.
          - Nadir marker per method.
        """
        if not self.scenarios:
            return
        scenario_names = list(self.scenarios.keys())[:4]
        n = len(scenario_names)
        if n == 0:
            return

        # Guard against empty topology cache (force_topology would raise IndexError)
        if not getattr(self.env.reconfig, "_cache", []):
            topology_idx = None

        # Layout: 2x2 if 4 scenarios, else 1xN
        if n == 4:
            fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
            axes_flat = axes.flatten()
        else:
            fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharex=True, sharey=True)
            axes_flat = np.atleast_1d(axes).flatten()

        method_order = sorted(
            self.policies.keys(),
            key=lambda k: (k != "GraphSAGE-MAPPO", k),  # Proposed last (drawn on top)
        )

        for ax, sc_name in zip(axes_flat, scenario_names):
            event = self.scenarios[sc_name]
            event_t = float(event.t_inject)

            # Cooperative-dispatch control windows (FFR / Primary droop / AGC)
            ax.axvspan(event_t,        event_t + 2.0,  color="#3a7bd5", alpha=0.07, label="FFR 0-2s")
            ax.axvspan(event_t + 2.0,  event_t + 10.0, color="#f5a623", alpha=0.07, label="Primary 2-10s")
            # AGC window extends to end of plot; computed after we know xlim below.
            # Settling and UFLS shading
            ax.axhspan(50.0 - settle_band, 50.0 + settle_band, color="green", alpha=0.07,
                       label=f"Settling ±{settle_band:g} Hz")
            ax.axhline(50.0, ls=":", color="gray", alpha=0.6, linewidth=0.8)
            ax.axhline(ufls_hz, ls="--", color="red", alpha=0.65, linewidth=1.0,
                       label=f"UFLS {ufls_hz} Hz")
            ax.axvline(event_t, ls="--", color="orange", alpha=0.7, linewidth=1.0,
                       label=f"Event @ {event_t:g}s")

            for method_name in method_order:
                policy = self.policies[method_name]
                runs: list[np.ndarray] = []
                hires_dt = None
                for _ in range(n_runs):
                    m = self.run_episode(policy, event=event, topology_idx=topology_idx)
                    # Prefer hi-res sub-step trace (dt ≈ 0.1 s) over fast-step trace (1 s)
                    if m.f_trace_hires.size > 0:
                        runs.append(np.asarray(m.f_trace_hires, dtype=float))
                        hires_dt = float(m.dt_hires)
                    else:
                        runs.append(np.asarray(m.f_trace, dtype=float))
                if not runs:
                    continue
                # Align lengths
                L = min(len(r) for r in runs)
                mat = np.stack([r[:L] for r in runs], axis=0)
                dt_local = hires_dt if hires_dt is not None else dt
                t = np.arange(L) * dt_local
                mean = mat.mean(axis=0)
                std = mat.std(axis=0)

                color, lw, ls = self._style_for(method_name)
                zorder = 5 if method_name == "GraphSAGE-MAPPO" else 3
                ax.plot(t, mean, label=method_name, color=color, linewidth=lw, linestyle=ls, zorder=zorder)
                if n_runs > 1:
                    ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.12, zorder=zorder - 1)
                self._annotate_trace(ax, mean, dt, int(round(event_t / dt)), color)

            ax.set_title(f"{sc_name}  (ΔP = {event.delta_P_mw:+.2f} MW)", fontsize=11)
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=0.3)
            ax.set_ylim(48.7, 51.2)
            # Tight x-axis window centred on the transient
            if zoom_post_event_s is not None:
                ax.set_xlim(max(0.0, event_t - 5.0), event_t + float(zoom_post_event_s))
                ax.axvspan(event_t + 10.0, event_t + float(zoom_post_event_s),
                           color="#56c596", alpha=0.07, label="AGC ≥10s")
            else:
                ax.axvspan(event_t + 10.0, ax.get_xlim()[1],
                           color="#56c596", alpha=0.07, label="AGC ≥10s")

        # Single legend at figure level
        handles, labels = axes_flat[0].get_legend_handles_labels()
        # De-dup keeping order
        seen = set()
        uniq: list = []
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq.append((h, l))
                seen.add(l)
        fig.legend([h for h, _ in uniq], [l for _, l in uniq],
                   loc="lower center", ncol=min(len(uniq), 6), fontsize=9,
                   bbox_to_anchor=(0.5, -0.02), frameon=True)
        fig.suptitle("Frequency response across contingency scenarios — Proposed vs baselines",
                     fontsize=13, y=1.0)
        plt.tight_layout(rect=[0, 0.04, 1, 0.98])
        out_name = f"fig_freq_grid_S1_S4{save_suffix}"
        fig.savefig(self.output_dir / f"{out_name}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(self.output_dir / f"{out_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Fig: {out_name} (zoom={zoom_post_event_s}s)")

    def plot_iae_degradation_vs_distance(self, n_runs: int = 5) -> pd.DataFrame:
        """Fig. 6 ★ KEY — IAE degradation vs Jaccard edge distance.

        For each method × each test topology, compute:
          x = d_E(test_topo, nearest train_topo)
          y = (IAE_test - IAE_train_mean) / IAE_train_mean × 100  (degradation %)
        and overlay a linear regression line per method.

        Expected per Baseline Comparison.md §6.3:
          - GraphSAGE-MAPPO: flat line (inductive generalisation)
          - MLP-MAPPO / MATD3: steep slope (no topology awareness)
          - GCNN-PPO: moderate slope (spectral filters topology-specific)
        """
        if not self.train_topologies or not self.test_topologies:
            print("Need both train and test topologies for Fig.6 — skipping")
            return pd.DataFrame()

        # Use S2 (the AM stress scenario) as the canonical event
        event = self.scenarios.get("S2_gen_trip")
        if event is None:
            return pd.DataFrame()

        topo_cache = getattr(self.env.reconfig, "_cache", [])
        topo_edges: list[np.ndarray] = []
        for topo in topo_cache:
            if isinstance(topo, dict):
                ei = topo.get("edge_index", np.array([[], []]))
            elif isinstance(topo, tuple) and len(topo) >= 2:
                ei = topo[1]
            else:
                ei = np.array([[], []])
            topo_edges.append(np.asarray(ei))

        # Distance from each test topo to its nearest train topo
        d_min_per_test: dict[int, float] = {}
        for t_idx in self.test_topologies:
            d_min = min(
                compute_jaccard_edge_distance(topo_edges[t_idx], topo_edges[tr_idx])
                for tr_idx in self.train_topologies
            )
            d_min_per_test[t_idx] = float(d_min)

        rows: list[dict[str, Any]] = []
        for policy_name, policy in self.policies.items():
            # Train baseline
            train_iaes = []
            for tr_idx in self.train_topologies:
                for _ in range(n_runs):
                    m = self.run_episode(policy, event=event, topology_idx=tr_idx)
                    train_iaes.append(m.iae_post)
            iae_train_mean = float(np.mean(train_iaes)) if train_iaes else 0.0

            # Per test topology
            for t_idx in self.test_topologies:
                iaes = []
                for _ in range(n_runs):
                    m = self.run_episode(policy, event=event, topology_idx=t_idx)
                    iaes.append(m.iae_post)
                iae_test = float(np.mean(iaes))
                deg_pct = ((iae_test - iae_train_mean) / max(iae_train_mean, 1e-6)) * 100.0
                rows.append({
                    "method": policy_name,
                    "test_topology_id": int(t_idx),
                    "d_E": d_min_per_test[t_idx],
                    "iae_train_mean": iae_train_mean,
                    "iae_test": iae_test,
                    "iae_degradation_pct": deg_pct,
                })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "fig6_iae_vs_distance.csv", index=False)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        palette = {
            "GraphSAGE-MAPPO": "#3a7bd5",
            "MLP-MAPPO": "#e94e77",
            "GCNN-PPO": "#f5a623",
            "MATD3": "#7a7a7a",
            "Fixed Droop": "#56c596",
            "No FFR": "#a04668",
        }
        for method in sorted(df["method"].unique()):
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            color = palette.get(method, None)
            ax.scatter(sub["d_E"], sub["iae_degradation_pct"], s=55, label=method,
                       color=color, edgecolor="black", linewidth=0.5, alpha=0.85)
            if len(sub) >= 2:
                x = sub["d_E"].to_numpy(dtype=float)
                y = sub["iae_degradation_pct"].to_numpy(dtype=float)
                # Skip regression if x has no spread or y has non-finite values
                if np.ptp(x) > 1e-12 and np.all(np.isfinite(y)):
                    try:
                        slope, intercept = np.polyfit(x, y, 1)
                        x_line = np.linspace(x.min(), x.max(), 50)
                        ax.plot(x_line, slope * x_line + intercept, "--", color=color, alpha=0.7, linewidth=1.5)
                    except np.linalg.LinAlgError:
                        pass

        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel("Jaccard edge distance $d_E$ (test → nearest train)")
        ax.set_ylabel("IAE degradation (%)")
        ax.set_title("Topology generalisation: IAE degradation vs edge distance")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.output_dir / "fig6_iae_vs_distance.pdf", dpi=300)
        fig.savefig(self.output_dir / "fig6_iae_vs_distance.png", dpi=150)
        plt.close(fig)
        print("Saved Fig.6: IAE degradation vs d_E")
        return df

    def run_all(self, n_runs: int = 20) -> dict[str, pd.DataFrame]:
        """Run all evaluations."""
        print("\n" + "="*60)
        print("FFR + TOPOLOGY ADAPTATION EVALUATION")
        print("="*60)

        print("\n[1/6] Building Table 1: FFR Performance Comparison...")
        table1 = self.build_table1_ffr_comparison(n_runs=n_runs)

        print("\n[2/6] Building Table 2: Topology Adaptation...")
        table2 = self.build_table2_topology_adaptation(n_runs=max(n_runs // 2, 5))

        print("\n[3/6] Building Table 3: Severity Scaling...")
        table3 = self.build_table3_severity_scaling(n_runs=max(n_runs // 2, 5))

        print("\n[4/7] Plotting per-scenario frequency traces (S1-S4)...")
        for sc_name in self.scenarios.keys():
            self.plot_frequency_traces(sc_name)

        print("\n[5/7] Plotting multi-scenario freq grid (Proposed vs baselines, mean±std)...")
        # Default zoom: 20s post-event window with control-zone shading
        self.plot_frequency_grid_all_scenarios(
            n_runs=max(n_runs // 5, 3),
            zoom_post_event_s=20.0,
        )
        # Full-window variant for context (no zoom)
        self.plot_frequency_grid_all_scenarios(
            n_runs=max(n_runs // 5, 3),
            zoom_post_event_s=None,
            save_suffix="_full",
        )

        print("\n[6/7] Plotting topology comparison...")
        self.plot_topology_comparison()

        print("\n[7/7] Plotting Fig.6 (IAE degradation vs d_E)...")
        self.plot_iae_degradation_vs_distance(n_runs=max(n_runs // 4, 3))

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        # Table 1 pivot
        pivot1 = table1.pivot_table(index=["scenario", "method"], columns="metric", values="mean", aggfunc="first")
        print("\nTable 1 - FFR Success Rate by Method:")
        print(pivot1[["ffr_success", "nadir_hz", "settling_time_s"]].round(3))

        # Table 2 summary
        print("\nTable 2 - Topology Generalization:")
        print(table2[["method", "topology_split", "ffr_success_rate", "generalization_gap"]].to_string(index=False))

        # Save combined results
        combined = {
            "table1": table1.to_dict(orient="records"),
            "table2": table2.to_dict(orient="records"),
            "table3": table3.to_dict(orient="records"),
        }
        with open(self.output_dir / "results_combined.json", "w") as f:
            json.dump(combined, f, indent=2)

        print(f"\nResults saved to: {self.output_dir}")
        return {"table1": table1, "table2": table2, "table3": table3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FFR + Topology Adaptation Evaluation")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints_am_mappo/am_mappo_final.pt"),
                        help="GraphSAGE-MAPPO (Proposed) checkpoint")
    parser.add_argument("--gcnn-checkpoint", type=Path, default=None,
                        help="GCNN-PPO baseline checkpoint (Guo et al. 2024)")
    parser.add_argument("--matd3-checkpoint", type=Path, default=None,
                        help="MATD3 baseline checkpoint (Li & Zhou 2025)")
    parser.add_argument("--mlp-mappo-checkpoint", type=Path, default=None,
                        help="MLP-MAPPO ablation checkpoint")
    parser.add_argument("--placement", type=Path, default=Path("artifacts/placement/official_placement_v3.json"))
    parser.add_argument("--mpc-path", type=Path, default=Path("data/grid_IEEE123_complete.m"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ffr_topology"))
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_config = {
        "placement_path": str(args.placement),
        "mpc_path": str(args.mpc_path),
        "seed": args.seed,
    }

    evaluator = FFRTopologyEvaluator(
        env_config=env_config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        gcnn_checkpoint=args.gcnn_checkpoint,
        matd3_checkpoint=args.matd3_checkpoint,
        mlp_mappo_checkpoint=args.mlp_mappo_checkpoint,
    )

    evaluator.run_all(n_runs=args.n_runs)


if __name__ == "__main__":
    main()
