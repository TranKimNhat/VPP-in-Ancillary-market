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
import shutil
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
    # COI counterparts (reported in parallel for comparison). The headline fields
    # above are worst-GFM when per-bus traces are available, else identical to COI.
    nadir_coi_hz: float = 50.0
    zenith_coi_hz: float = 50.0
    rocof_max_coi_hz_s: float = 0.0

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
            "nadir_coi_hz": self.nadir_coi_hz,
            "zenith_coi_hz": self.zenith_coi_hz,
            "rocof_max_coi_hz_s": self.rocof_max_coi_hz_s,
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
    settle_band: float = 0.05,  # ENTSO-E SO GL standard frequency range (±50 mHz, CE)
    event_step: int = 30,
    post_window: int = 50,
    dt: float = 1.0,
    rocof_limit: float = 2.0,  # IEEE 1547-2018 Cat III mandatory ride-through (NOT protection trip; ENTSO-E mainland uses 1.0 Hz/s)
    f_trace_nadir: np.ndarray | None = None,
    f_trace_zenith: np.ndarray | None = None,
    rocof_trace_worst: np.ndarray | None = None,
) -> FFRMetrics:
    """Compute FFR metrics from frequency trajectory (IEEE 1547 Cat III aligned).

    When per-timestep worst-GFM traces are supplied (``f_trace_nadir`` = min over
    GFMs, ``f_trace_zenith`` = max over GFMs, ``rocof_trace_worst`` = max |RoCoF|
    over GFMs), the HEADLINE metrics are computed on the worst-GFM quantity and the
    COI counterparts are reported in parallel. Absent those, behaviour is identical
    to the legacy COI computation (back-compat).
    """
    f_trace = np.asarray(f_trace, dtype=float)
    rocof_trace = np.asarray(rocof_trace, dtype=float)

    # COI metrics (always computed; reported in parallel as *_coi fields).
    nadir_coi = float(np.min(f_trace))
    zenith_coi = float(np.max(f_trace))
    rocof_max_coi = float(np.max(np.abs(rocof_trace)))

    use_worst = (f_trace_nadir is not None and f_trace_zenith is not None
                 and rocof_trace_worst is not None)
    if use_worst:
        f_nadir = np.asarray(f_trace_nadir, dtype=float)
        f_zenith = np.asarray(f_trace_zenith, dtype=float)
        nadir = float(np.min(f_nadir))
        zenith = float(np.max(f_zenith))
        # RoCoF headline stays on the COI (backbone-governed): peak RoCoF =
        # ΔP/2H_sys is a system property set by the GFM backbone, ~common to all
        # controllers, and the worst-GFM rate would be inflated by the same single
        # diverging unit that motivates the nadir clamp. Only nadir/zenith use the
        # worst unit (Approach-C / D1). rocof_trace_worst is still reported below.
        rocof_max = rocof_max_coi
        # Per-timestep worst deviation magnitude across BOTH directions. A synthetic
        # trace pinned below nominal by this magnitude makes |Δf| = worst_dev, so the
        # |Δf|-based integrals (IAE/ITAE/violation/settling) reuse the COI code paths.
        worst_dev = np.maximum.reduce([
            f_nominal - f_nadir, f_zenith - f_nominal, np.zeros_like(f_nadir)
        ])
        mag_trace = f_nominal - worst_dev
    else:
        nadir, zenith, rocof_max = nadir_coi, zenith_coi, rocof_max_coi
        mag_trace = f_trace

    delta_f = mag_trace - f_nominal
    delta_f_max = float(np.max(np.abs(delta_f)))

    # IAE (Integral Absolute Error)
    iae_total = float(np.trapezoid(np.abs(delta_f), dx=dt))

    # Post-event metrics
    post_start = min(event_step, len(delta_f) - 1)
    post_end = min(event_step + post_window, len(delta_f))
    post_delta_f = delta_f[post_start:post_end]
    iae_post = float(np.trapezoid(np.abs(post_delta_f), dx=dt)) if len(post_delta_f) > 0 else 0.0

    # ITAE (Integral of Time-weighted Absolute Error)
    itae = compute_itae(mag_trace, f_nominal, dt, t_event=event_step * dt)

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

    # FFR success criterion (frequency-security definition): the controllable
    # GFL fleet shapes the frequency-deviation / nadir margin, so success is a
    # continuous post-event excursion below the 49.5 Hz alarm for < 300 ms — the
    # typical UFLS Stage 1 trip delay (IEEE Std C37.117-2007 §6.2; NOT IEEE 81,
    # the grounding standard). 300 ms = 3 hi-res samples @ dt=0.1s; time_violation
    # sums all under-threshold samples.
    #
    # RoCoF is deliberately NOT part of the gate: peak RoCoF = ΔP_imb/(2H_backbone)
    # is set by the grid-forming backbone inertia, not by the GFL action (Approach-C),
    # so it is reported (rocof_max_hz_s) for completeness and is ~identical across
    # controllers; it does not discriminate them. rocof_limit is retained only for
    # backward-compatible reporting of the ride-through threshold.
    ffr_success = (time_violation <= 0.3)

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
        nadir_coi_hz=nadir_coi,
        zenith_coi_hz=zenith_coi,
        rocof_max_coi_hz_s=rocof_max_coi,
    )


# =============================================================================
# Policy Wrappers
# =============================================================================

class NoFFRPolicy:
    """Baseline: No frequency response from the VPP fleet.

    Runs the SAME mappo_dual interface/coupling as the proposed method so the
    comparison is symmetric. a_P=0 (no power reference) and a_K=-1 → K_droop=0
    (K_min lowered to 0 in the env), i.e. the DERs do not participate in FFR at
    all. Frequency is then governed solely by the GFM backbone + AGC.
    """
    ffr_mode = "mappo_dual"
    nadir_safety = False   # genuine baseline: no proposed nadir safety projection

    def act(self, obs: np.ndarray, edge_index: np.ndarray, env: Any, obs_fast: np.ndarray | None = None) -> np.ndarray:
        n_agents = env.n_agents
        n_vpps = len(env._vpp_droop_agents)
        action = np.zeros(2 * n_agents + n_vpps, dtype=np.float32)
        action[n_agents:2 * n_agents] = -1.0   # a_K = -1 → K_droop = K_min = 0
        return action


class FixedDroopPolicy:
    """Baseline: fixed, uniform droop on every participating DER.

    Same mappo_dual interface as the proposed method: a_P=0 (no learned power
    reference) and a constant a_K so every DER holds a fixed mid-range droop
    gain. This isolates the value of the LEARNED, heterogeneous (P,K) policy
    over a flat classical droop using the identical coupling into A_f.
    """
    ffr_mode = "mappo_dual"
    nadir_safety = False   # genuine baseline: no proposed nadir safety projection

    def __init__(self, k_droop: float = 0.05, a_k_fixed: float = 0.0):
        # a_k_fixed = 0.0 → K = midpoint of [0, K_max] (a moderate fixed droop).
        self.k_droop = k_droop
        self.a_k_fixed = a_k_fixed

    def act(self, obs: np.ndarray, edge_index: np.ndarray, env: Any, obs_fast: np.ndarray | None = None) -> np.ndarray:
        n_agents = env.n_agents
        n_vpps = len(env._vpp_droop_agents)
        action = np.zeros(2 * n_agents + n_vpps, dtype=np.float32)
        action[n_agents:2 * n_agents] = self.a_k_fixed   # uniform fixed droop
        return action


class GraphSAGEMAPPOPolicy:
    """Our method: GraphSAGE-MAPPO trained agent."""
    ffr_mode = "mappo_dual"  # env must run dual (per-DER P,K) to match training

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
        action = self.agent.act_deterministic(obs_norm, edge_index)  # (n_agents, action_dim)
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(-1, 1)

        n_agents = env.n_agents
        n_vpps = len(env._vpp_droop_agents)

        # Build the dual-action vector EXACTLY as training does (train_am_mappo loop):
        #   ctrl_p = -a_P  (the shared action->power sign convention; see P3 note in
        #            train_am_mappo.py — flip must match on both train and eval sides)
        #   ctrl_k =  a_K  (NOT flipped; maps to K_droop bounds in env)
        #   layout = [a_P (n_agents), a_K (n_agents), VPP-K legacy zeros (n_vpps)]
        # env runs in "mappo_dual" mode (set by the evaluator before reset), which
        # expects this 2*n_agents + n_vpps vector. The prior code built a single-mode
        # (n_agents + n_vpps) vector that discarded the K channel and mis-indexed P.
        ctrl_p = -action[:, 0]
        ctrl_k = action[:, 1] if action.shape[1] >= 2 else np.zeros(n_agents, dtype=np.float32)

        full_action = np.zeros(2 * n_agents + n_vpps, dtype=np.float32)
        full_action[:n_agents] = np.clip(ctrl_p[:n_agents], -1.0, 1.0)
        full_action[n_agents:2 * n_agents] = np.clip(ctrl_k[:n_agents], -1.0, 1.0)
        # VPP-K legacy slots stay zero: per-DER K replaces VPP aggregation in dual mode.
        return full_action


class GCNNPPOPolicy:
    """Baseline: GCNN-PPO (Guo et al. 2024), adapted to our POMDP.

    Spectral GCN + centralized PPO. Evaluated under the SAME dual (a_P, a_K)
    mappo_dual interface as the proposed method so the comparison is symmetric.
    """
    ffr_mode = "mappo_dual"  # same MDP/interface as training; dual (P,K) per DER

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

    def act(self, obs: np.ndarray, edge_index: np.ndarray, env: Any, obs_fast: np.ndarray | None = None) -> np.ndarray:
        # GCNN-PPO is trained on build_am_full_feeder_obs (shape (n_bus, 20)).
        from src.rl.train_am_mappo import build_am_full_feeder_obs
        if obs_fast is None:
            raise ValueError("GCNNPPOPolicy requires obs_fast")
        obs_full = build_am_full_feeder_obs(env, obs_fast)
        global_obs = obs_full.reshape(-1)
        # Deterministic eval (distribution mean, no sampling).
        action_env = self._agent.act_deterministic(obs_full, edge_index, global_obs)
        # Dual (a_P, a_K) layout, identical to training and to GraphSAGEMAPPOPolicy.
        a = np.asarray(action_env, dtype=np.float32).reshape(-1)
        n_agents = env.n_agents
        n_vpps = len(env._vpp_droop_agents)
        a = a.reshape(n_agents, 2) if a.size >= 2 * n_agents else a.reshape(-1, 1)
        ctrl_p = -a[:, 0]
        ctrl_k = a[:, 1] if a.shape[1] >= 2 else np.zeros(n_agents, dtype=np.float32)
        full_action = np.zeros(2 * n_agents + n_vpps, dtype=np.float32)
        full_action[:n_agents] = np.clip(ctrl_p[:n_agents], -1.0, 1.0)
        full_action[n_agents:2 * n_agents] = np.clip(ctrl_k[:n_agents], -1.0, 1.0)
        return full_action


class MATD3Policy:
    """Baseline: EIE-MATD3 (Li & Zhou 2025), adapted to our POMDP.

    CTDE multi-agent TD3 with MLP encoder. Builds per-agent obs from env state.
    Evaluated under the SAME dual (a_P, a_K) mappo_dual interface as the proposed
    method so the comparison is symmetric.
    """
    ffr_mode = "mappo_dual"  # same MDP/interface as training; dual (P,K) per DER

    def __init__(self, checkpoint_path: Path, env: MicrogridEnvDual):
        from src.baselines.matd3 import MATD3Agent, MATD3Config
        # Auto-detect obs_dim AND hidden_dim: prefer stored config, else infer from
        # actor weights. hidden_dim MUST be restored — checkpoints trained with
        # hidden_dim=256 fail load_state_dict against the default-128 actor (a latent
        # bug: the prior code restored obs_dim only).
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        obs_dim = 24
        hidden_dim = 128
        stored_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
        if stored_cfg is not None and hasattr(stored_cfg, "obs_dim"):
            obs_dim = int(stored_cfg.obs_dim)
            hidden_dim = int(getattr(stored_cfg, "hidden_dim", hidden_dim))
        else:
            actors_blob = ckpt.get("actors") if isinstance(ckpt, dict) else None
            if isinstance(actors_blob, dict) and "0.net.0.weight" in actors_blob:
                w = actors_blob["0.net.0.weight"].shape
                hidden_dim = int(w[0])
                obs_dim = int(w[1])
        config = MATD3Config(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=2, n_agents=env.n_agents)
        self._agent = MATD3Agent(config, device="cpu")
        self._agent.load(checkpoint_path)
        self._agent.eval()
        self._env = env

    def act(self, obs: np.ndarray, edge_index: np.ndarray, env: Any, obs_fast: np.ndarray | None = None) -> np.ndarray:
        # `obs` IS the full-feeder observation (n_bus, obs_feat) from
        # build_am_full_feeder_obs — the SAME tensor train_matd3.py trains on.
        # Extract per-agent rows by bus index exactly as training does
        # (train_matd3.py: `obs = obs_full[agent_bus_idx]`). The prior _build_obs
        # rebuilt a DIFFERENT vector (raw COI delta_f + raw fast-obs + zero-padded
        # tail), a train/eval mismatch that fed the actor out-of-distribution input
        # (15/20 channels differed; slots 9–19 zeroed) and invalidated the MATD3
        # comparison. obs_fast is no longer needed (kept in signature for the
        # shared policy.act interface).
        agent_bus_idx = np.clip(
            np.asarray(getattr(env, "_agent_bus_pp", np.arange(env.n_agents)), dtype=np.int64),
            0, obs.shape[0] - 1,
        )
        per_agent_obs = obs[agent_bus_idx]  # (n_agents, obs_feat) — identical to training
        actions = self._agent.act_deterministic(per_agent_obs)  # (n_agents, 2)
        # Dual (a_P, a_K) layout, identical to training and to GraphSAGEMAPPOPolicy:
        #   ctrl_p = -a_P (sign convention), ctrl_k = a_K (maps to per-DER K_droop).
        n_agents = env.n_agents
        n_vpps = len(env._vpp_droop_agents)
        ctrl_p = -actions[:, 0]
        ctrl_k = actions[:, 1] if actions.shape[1] >= 2 else np.zeros(n_agents, dtype=np.float32)
        full_action = np.zeros(2 * n_agents + n_vpps, dtype=np.float32)
        full_action[:n_agents] = np.clip(ctrl_p[:n_agents], -1.0, 1.0)
        full_action[n_agents:2 * n_agents] = np.clip(ctrl_k[:n_agents], -1.0, 1.0)
        return full_action


class MLPMAPPOPolicy:
    """Ablation: MAPPO with MLP encoder (no graph message passing).

    Same RL stack as the proposed GraphSAGE-MAPPO but with an MLP encoder so the
    only differing variable is the graph encoder. This is the most important
    ablation for proving GraphSAGE's contribution to topology generalization.
    """
    ffr_mode = "mappo_dual"  # same RL stack as GraphSAGE-MAPPO; dual (P,K) per DER

    def __init__(self, checkpoint_path: Path, env: MicrogridEnvDual):
        from src.baselines.train_mlp_mappo import MLPMAPPOAgent

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        agent_state = ckpt.get("model_state_dict", ckpt.get("agent_state_dict", ckpt))

        # Infer dimensions from checkpoint weights.
        # In MLPMAPPOAgent, the encoder uses embed_dim for ALL of its layers
        # (encoder.net.0 hidden = embed_dim, encoder.net.4 output = embed_dim),
        # while the actor uses a separate hidden_dim. So we need to read the
        # actor's hidden_dim from actor.net.0.weight, not from any encoder layer.
        embed_dim = 64
        hidden_dim = 128
        if "encoder.net.4.weight" in agent_state:
            embed_dim = agent_state["encoder.net.4.weight"].shape[0]
        if "actor.net.0.weight" in agent_state:
            hidden_dim = agent_state["actor.net.0.weight"].shape[0]

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
        action = self.agent.act_deterministic(obs_norm, edge_index)  # (n_agents, action_dim)
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(-1, 1)

        n_agents = env.n_agents
        n_vpps = len(env._vpp_droop_agents)

        # Build the dual-action vector EXACTLY as training does (train_am_mappo loop):
        #   ctrl_p = -a_P  (the shared action->power sign convention; see P3 note in
        #            train_am_mappo.py — flip must match on both train and eval sides)
        #   ctrl_k =  a_K  (NOT flipped; maps to K_droop bounds in env)
        #   layout = [a_P (n_agents), a_K (n_agents), VPP-K legacy zeros (n_vpps)]
        # env runs in "mappo_dual" mode (set by the evaluator before reset), which
        # expects this 2*n_agents + n_vpps vector. The prior code built a single-mode
        # (n_agents + n_vpps) vector that discarded the K channel and mis-indexed P.
        ctrl_p = -action[:, 0]
        ctrl_k = action[:, 1] if action.shape[1] >= 2 else np.zeros(n_agents, dtype=np.float32)

        full_action = np.zeros(2 * n_agents + n_vpps, dtype=np.float32)
        full_action[:n_agents] = np.clip(ctrl_p[:n_agents], -1.0, 1.0)
        full_action[n_agents:2 * n_agents] = np.clip(ctrl_k[:n_agents], -1.0, 1.0)
        # VPP-K legacy slots stay zero: per-DER K replaces VPP aggregation in dual mode.
        return full_action


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
        base_reference: bool = False,
    ):
        self.env = MicrogridEnvDual(**env_config)
        # Train-on-base, eval-on-all-reconfig protocol: when True, the reference is
        # the nominal base feeder (topology sentinel -1) and ALL cached reconfig
        # topologies are treated as unseen. Default False = farthest-split behaviour.
        self.base_reference = base_reference
        self.base_edges: np.ndarray | None = None
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

        if self.base_reference and n_topos >= 1:
            # Capture the base feeder edge set (the ONLY topology models trained on),
            # then mark every cached reconfig topology as unseen.
            _prev = self.env.fixed_base_topology
            self.env.fixed_base_topology = True
            self.env.reset()
            self.base_edges = np.asarray(self.env.edge_index).copy()
            self.env.fixed_base_topology = _prev
            self.train_topologies = [-1]                  # sentinel: base reference
            self.test_topologies = list(range(n_topos))   # all topologies unseen
            print(f"Base-reference mode: reference=base feeder, "
                  f"{len(self.test_topologies)} reconfig topologies ALL unseen")

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
        use_base = topology_idx is not None and int(topology_idx) < 0
        if topology_idx is not None and not use_base:
            options["force_topology"] = int(topology_idx)

        # Each policy is evaluated in its NATIVE control mode (user decision):
        # proposed/ablation MAPPO -> "mappo_dual" (per-DER P,K from the policy);
        # baselines (No-FFR, Fixed Droop) -> "droop" (env's built-in droop law).
        # The frequency physics (event injection, LTI dynamics, AGC) is identical
        # across modes — only the FFR control law differs — so this is a fair
        # like-for-like comparison. step_fast reads self.ffr_mode each call and the
        # per-DER K buffers are always initialised, so switching here is safe.
        self.env.ffr_mode = getattr(policy, "ffr_mode", "droop")
        # Per-policy nadir safety layer: classical references (No-FFR, Fixed-Droop)
        # run WITHOUT the proposed in-the-loop nadir projection so they are genuine
        # baselines (otherwise the layer injects power into the VPP channel and
        # rescues even the No-FFR fleet, pinning nadir at the band edge). Learned
        # methods keep it ON to match their training-time environment.
        self.env.nadir_safety_enabled = bool(getattr(policy, "nadir_safety", True))

        _prev_base_flag = self.env.fixed_base_topology
        if use_base:
            self.env.fixed_base_topology = True
        obs_fast, _, _ = self.env.reset(options=options)
        self.env.fixed_base_topology = _prev_base_flag
        n_bus = len(self.env.net.bus.index)
        edge_index = ensure_edge_index(self.env.edge_index, n_nodes=n_bus)

        obs_full = build_am_full_feeder_obs(self.env, obs_fast)

        f_trace = []
        rocof_trace = []
        # Per-bus traces for spatial analysis (when LTI freq dynamics active)
        f_trace_per_bus: list[np.ndarray] = []
        rocof_trace_per_bus: list[np.ndarray] = []
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

            # Use LTI freq dynamics if active, else legacy scalar COI
            use_lti = getattr(self.env, 'use_lti_freq', False) and getattr(self.env, 'freq_dyn_lti', None) is not None
            if use_lti:
                freq_state = self.env.freq_dyn_lti.get_state()
                f_trace_per_bus.append(50.0 + freq_state.delta_f_per_bus.copy())
                rocof_trace_per_bus.append(freq_state.rocof_per_bus.copy())
            else:
                freq_state = self.env.freq_dyn_lti.get_state()
            # Always append scalar COI for backward-compat metrics
            f_trace.append(50.0 + freq_state.delta_f_hz)
            rocof_trace.append(freq_state.rocof_hz_s)

        # Snapshot hi-res trace (50.0 + delta_f) — env appended one entry per sub-step
        f_hires = [50.0 + df for df in self.env._hires_df]

        event_step = int(event.t_inject) if event else 30
        # Headline metrics are computed on the COI (rating-weighted system frequency).
        # At the 1 s FFR/SFR control step the COI is the resolvable system-frequency
        # observable (Anderson 1990); per-GFM (inter-machine) frequency is a
        # sub-second phenomenon NOT resolvable at this timestep, so it is not used for
        # the headline nadir/IAE/RoCoF/FFR-success -- only as a transient spatial-
        # spread diagnostic (clamped to the model validity band so a divergent unit's
        # leftover trace stays bounded).
        metrics = compute_ffr_metrics(
            np.array(f_trace),
            np.array(rocof_trace),
            event_step=event_step,
        )
        # Attach hi-res trace for plotting helpers. When env.hires_substeps>1 the
        # buffer holds n_sub samples per fast-step, so the true spacing is
        # dt_fast_s / n_sub (NOT dt_ode_s, which is unused by the LTI path and was
        # the source of the collapsed-time-axis blank figure).
        n_sub = int(getattr(self.env, "hires_substeps", 0))
        dt_fast = float(getattr(self.env, "dt_fast_s", 1.0))
        metrics.f_trace_hires = np.asarray(f_hires, dtype=np.float32)
        metrics.dt_hires = dt_fast / n_sub if n_sub > 1 else dt_fast
        # Per-bus traces: transient spatial diagnostic ONLY (not headline metrics).
        if f_trace_per_bus:
            pb = np.clip(np.stack(f_trace_per_bus, axis=0), 45.0, 55.0)  # (T, n_gfm)
            metrics.f_trace_per_bus = pb
            metrics.rocof_trace_per_bus = np.stack(rocof_trace_per_bus, axis=0)
            metrics.nadir_worstbus_hz = float(pb.min())
            metrics.worstbus_spread_hz = float((pb.max(axis=1) - pb.min(axis=1)).max())
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
            for split_name, topo_list in [
                ("base" if self.base_reference else "train", self.train_topologies),
                ("unseen", self.test_topologies),
            ]:
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
                    "iae_post_std": float(np.std(metrics_agg["iae_post"])),
                })

        df = pd.DataFrame(rows)

        # Generalization measured on a CONTINUOUS metric (IAE_post), not the binary
        # ffr_success — for the large design contingencies (gen_trip -3.9 MW → nadir
        # ~47.9 Hz) no controller keeps |Δf|<0.5 Hz within 300 ms, so ffr_success is
        # 0 for every method and cannot discriminate topology generalization. IAE
        # (lower=better) does. gap = relative IAE degradation on unseen topologies;
        # retention = train/unseen (≤1 better, →1 = generalizes perfectly).
        if not df.empty and "method" in df.columns:
            ref_label = "base" if self.base_reference else "train"
            for policy_name in self.policies.keys():
                train_row = df[(df["method"] == policy_name) & (df["topology_split"] == ref_label)]
                test_row = df[(df["method"] == policy_name) & (df["topology_split"] == "unseen")]

                if not train_row.empty and not test_row.empty:
                    iae_train = float(train_row["iae_post_mean"].values[0])
                    iae_unseen = float(test_row["iae_post_mean"].values[0])
                    gap = (iae_unseen - iae_train) / iae_train if iae_train > 1e-9 else float("nan")
                    retention = iae_train / iae_unseen if iae_unseen > 1e-9 else float("nan")

                    df.loc[(df["method"] == policy_name) & (df["topology_split"] == "unseen"), "iae_generalization_gap"] = gap
                    df.loc[(df["method"] == policy_name) & (df["topology_split"] == "unseen"), "iae_retention_ratio"] = retention
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
    # Colours pulled from figures_style.METHOD_COLORS (purple/teal palette).
    # Linewidth + linestyle encode emphasis (Proposed thicker, baselines dashed).
    _METHOD_LW_LS = {
        "GraphSAGE-MAPPO": (2.2, "-"),
        "MLP-MAPPO":       (1.4, "--"),
        "GCNN-PPO":        (1.4, "--"),
        "MATD3":           (1.4, "--"),
        "Fixed Droop":     (1.2, ":"),
        "No FFR":          (1.2, ":"),
    }

    def _style_for(self, name: str) -> tuple[str, float, str]:
        from src.eval.figures_style import color_for_method
        lw, ls = self._METHOD_LW_LS.get(name, (1.4, "-"))
        return color_for_method(name), lw, ls

    def _annotate_trace(self, ax, f_trace: np.ndarray, dt: float, event_step: int, color: str) -> None:
        """Mark nadir + approximate settling on a freq trace."""
        nadir_idx = int(np.argmin(f_trace))
        ax.scatter([nadir_idx * dt], [f_trace[nadir_idx]], s=22, color=color, zorder=5,
                   edgecolor="black", linewidth=0.6)

    def plot_frequency_grid_all_scenarios(
        self,
        n_runs: int = 3,
        topology_idx: int | None = 0,
        dt: float = 1.0,
        ufls_hz: float = 49.5,
        settle_band: float = 0.05,  # match compute_ffr_metrics (ENTSO-E ±50 mHz)
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

        from src.eval.figures_style import (
            apply_style, FIGSIZE_GRID_2x2, FIGSIZE_DOUBLE_COL,
            ZORDER_BAND, ZORDER_MAIN, style_grid, tighten_spines,
        )
        apply_style()

        # Per-method style: proposed = bold dark-indigo solid + ±1σ band;
        # baselines = thinner dashed/dotted in distinct hues; non-learning
        # controls = light gray.
        FREQ_STYLE: dict[str, tuple[str, float, str]] = {
            "GraphSAGE-MAPPO": ("#2c3e7a", 3.0, "-"),    # dark indigo, thick
            "MLP-MAPPO":       ("#9671bd", 1.5, "--"),   # purple, dashed
            "GCNN-PPO":        ("#77b5b6", 1.5, ":"),    # teal, dotted
            "MATD3":           ("#d68f4f", 1.5, "-."),   # amber, dash-dot
            "Fixed Droop":     ("#a0a0a0", 1.0, ":"),    # light gray, dotted
            "No FFR":          ("#555555", 1.0, "--"),   # medium gray, dashed
        }
        def _style(name: str) -> tuple[str, float, str]:
            return FREQ_STYLE.get(name, ("#444444", 1.2, "-"))

        if n == 4:
            fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_GRID_2x2,
                                     sharex=True, sharey=True)
            axes_flat = axes.flatten()
        else:
            fig, axes = plt.subplots(1, n, figsize=(FIGSIZE_DOUBLE_COL[0], 5.0),
                                     sharex=True, sharey=True)
            axes_flat = np.atleast_1d(axes).flatten()

        # All loaded methods (proposed + all baselines). Proposed drawn LAST
        # so it sits on top of the others.
        method_order = sorted(
            self.policies.keys(),
            key=lambda k: k != "GraphSAGE-MAPPO",
        )

        # Short display labels for the inline nadir/zenith summary table.
        SHORT_NAME = {
            "GraphSAGE-MAPPO": "GraphSAGE",
            "MLP-MAPPO":       "MLP",
            "GCNN-PPO":        "GCNN",
            "MATD3":           "MATD3",
            "Fixed Droop":     "Droop",
            "No FFR":          "No FFR",
        }

        # Control-window background zones (subtle, muted, distinct hues).
        ZONE_FFR  = "#fff3d4"  # warm yellow (urgent FFR 0–2 s)
        ZONE_PRIM = "#d9e8f5"  # cool blue (primary 2–10 s)
        ZONE_AGC  = "#dceedc"  # cool green (AGC ≥10 s)

        # Resolve the sub-second transient for the illustrative curve: re-simulate
        # each fast-step at dt_fast/100 = 10 ms (read-only, training-untouched; the
        # textbook RMS/phasor-swing timestep). Restored in the finally block. The
        # nadir/zenith ANNOTATION below is NOT taken from this 10 ms trace -- it uses
        # the 1 s COI metric (compute_ffr_metrics) so the figure agrees with Tables.
        _prev_hires = int(getattr(self.env, "hires_substeps", 0))
        self.env.hires_substeps = 100
        try:
          for sub_i, (ax, sc_name) in enumerate(zip(axes_flat, scenario_names)):
            event = self.scenarios[sc_name]
            event_t = float(event.t_inject)

            # Background zones drawn first (lowest zorder, alpha ~0.4 for
            # visibility without fighting the curves).
            ax.axvspan(event_t,       event_t + 2.0,  facecolor=ZONE_FFR, alpha=0.55,
                       zorder=0, label="FFR (0–2 s)")
            ax.axvspan(event_t + 2.0, event_t + 10.0, facecolor=ZONE_PRIM, alpha=0.55,
                       zorder=0, label="Primary (2–10 s)")
            if zoom_post_event_s is not None:
                ax.axvspan(event_t + 10.0, event_t + float(zoom_post_event_s),
                           facecolor=ZONE_AGC, alpha=0.55, zorder=0,
                           label=r"AGC ($\geq$10 s)")
            ax.axvline(event_t, ls="--", color="#666666", alpha=0.9,
                       linewidth=1.0, label=f"Event @ {event_t:g} s")
            ax.axhline(ufls_hz, ls="--", color="#cc3333", alpha=0.95,
                       linewidth=1.0, label=f"UFLS {ufls_hz} Hz")
            ax.axhline(50.0, ls=":", color="#999999", linewidth=0.6, alpha=0.85)

            # Detect event direction: only the high-renewable surge causes
            # frequency to RISE. Load steps, generation trips and line trips
            # all cause frequency to DROP (report nadir).
            over_freq = event.type == "high_ren"
            nadir_records: list[tuple[str, float, str, float]] = []

            for method_name in method_order:
                policy = self.policies[method_name]
                runs: list[np.ndarray] = []
                coi_runs: list[np.ndarray] = []   # 1 s COI traces (table-timescale)
                metric_vals: list[float] = []     # per-run table nadir/zenith
                hires_dt = None
                for _ in range(n_runs):
                    m = self.run_episode(policy, event=event, topology_idx=topology_idx)
                    # Prefer the sub-step trace (env.hires_substeps>1 set below): it
                    # resolves the true sub-second nadir transient at dt_fast/n_sub.
                    # Falls back to the per-fast-step COI trace otherwise. (dt_hires
                    # is now the correct spacing, not the old dt_ode_s bug that
                    # collapsed the axis and blanked the figure.)
                    if m.f_trace_hires.size > 0:
                        runs.append(np.asarray(m.f_trace_hires, dtype=float))
                        hires_dt = float(m.dt_hires)
                    else:
                        runs.append(np.asarray(m.f_trace, dtype=float))
                    # Keep the 1 s COI trace + the SAME control-timescale extremum the
                    # Tables report (compute_ffr_metrics), so the annotation can never
                    # disagree with the tabulated nadir_hz / zenith_hz.
                    coi_runs.append(np.asarray(m.f_trace, dtype=float))
                    metric_vals.append(float(m.zenith_hz if over_freq else m.nadir_hz))
                if not runs:
                    continue
                L = min(len(r) for r in runs)
                mat = np.stack([r[:L] for r in runs], axis=0)
                dt_local = hires_dt if hires_dt is not None else dt
                t = np.arange(L) * dt_local
                mean = mat.mean(axis=0)
                std = mat.std(axis=0)

                color, lw, ls = _style(method_name)
                z_main = ZORDER_MAIN if method_name == "GraphSAGE-MAPPO" else ZORDER_MAIN - 2
                ax.plot(t, mean, label=method_name, color=color, linewidth=lw,
                        linestyle=ls, zorder=z_main)
                # ±1σ band only for the proposed controller (focus the eye).
                if method_name == "GraphSAGE-MAPPO" and n_runs > 1:
                    ax.fill_between(t, mean - std, mean + std, color=color,
                                    alpha=0.18, linewidth=0, zorder=ZORDER_BAND)

                # Record nadir/zenith — report the SAME control-timescale (1 s COI)
                # quantity as Tables 1/2 (mean per-run nadir_hz / zenith_hz from
                # compute_ffr_metrics), NOT the 10 ms hi-res dip. The smooth hires
                # curve above is illustrative only; pinning the label + marker to the
                # tabulated extremum keeps the figure and the tables in lockstep, and
                # honestly shows the resolved nadir is shallower than the sub-second
                # excursion the controller neither optimises nor guards at 1 s.
                if metric_vals:
                    val = float(np.mean(metric_vals))
                    # Marker time: extremum of the averaged 1 s COI trace over the
                    # post-event window (skip 0.5 s spike); position only, height = val.
                    Lc = min(len(r) for r in coi_runs)
                    coi_mean = np.stack([r[:Lc] for r in coi_runs], axis=0).mean(axis=0)
                    tc = np.arange(Lc) * dt
                    c_mask = (tc >= event_t + 0.5) & (tc <= event_t + 15.0)
                    if c_mask.any():
                        seg, seg_t = coi_mean[c_mask], tc[c_mask]
                        j = int(np.argmax(seg) if over_freq else np.argmin(seg))
                        t_pos = float(seg_t[j])
                    else:
                        t_pos = event_t
                    nadir_records.append((method_name, val, color, t_pos))

            # Mark the extremum point on each curve.
            for _, val, c, t_pos in nadir_records:
                ax.plot(t_pos, val, "o", color=c, markersize=7,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=11)

            # Sort: best controller first (highest nadir for under-freq,
            # lowest zenith for over-freq).
            nadir_records.sort(key=lambda r: r[1], reverse=not over_freq)

            # Summary text box: always TOP-RIGHT so it sits in the empty
            # corner regardless of event direction. For over-freq events the
            # peak occurs briefly at the very start (far-left of axes); the
            # top-right corner (later half × high freq) is empty there too.
            # High bbox alpha so the table fully covers any underlying data.
            title_txt = ("Zenith (Hz)" if over_freq else "Nadir (Hz)") + ":"
            y_anchor, va = 0.98, "top"
            y_step = -0.055
            ax.text(0.985, y_anchor, title_txt, transform=ax.transAxes,
                    ha="right", va=va, fontsize=15, fontweight="bold",
                    color="#222222", zorder=20,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              edgecolor="#999999", alpha=0.97, lw=0.7))
            for i, (name, val, c, _) in enumerate(nadir_records, start=1):
                ax.text(0.985, y_anchor + y_step * i,
                        f"{SHORT_NAME.get(name, name):>9s}  {val:5.2f}",
                        transform=ax.transAxes, ha="right", va=va,
                        fontsize=13, color=c, fontfamily="monospace",
                        zorder=20,
                        bbox=dict(boxstyle="round,pad=0.16",
                                  facecolor="white", edgecolor="none",
                                  alpha=0.97))

            ax.set_title(f"{sc_name.replace('_', ' ')}  ($\\Delta P = {event.delta_P_mw:+.1f}$ MW)")
            # Per-subplot axis labels only on the border subplots of the grid.
            row, col = sub_i // 2, sub_i % 2
            if col == 0:
                ax.set_ylabel("Frequency (Hz)")
            if row == 1 or n != 4:
                ax.set_xlabel("Time (s)")
            style_grid(ax, minor=False)
            tighten_spines(ax)
            ax.set_ylim(47.5, 53.0)
            if zoom_post_event_s is not None:
                ax.set_xlim(max(0.0, event_t - 2.0), event_t + float(zoom_post_event_s))
        finally:
            self.env.hires_substeps = _prev_hires

        # Single top-of-figure legend, 3 cols, no frame.
        handles, labels = axes_flat[0].get_legend_handles_labels()
        seen: set[str] = set()
        uniq: list[tuple] = []
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq.append((h, l))
                seen.add(l)
        fig.legend([h for h, _ in uniq], [l for _, l in uniq],
                   loc="upper center", ncol=min(len(uniq), 4),
                   bbox_to_anchor=(0.5, 1.02), frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out_name = f"fig_freq_grid_S1_S4{save_suffix}"
        fig.savefig(self.output_dir / f"{out_name}.pdf")
        fig.savefig(self.output_dir / f"{out_name}.png")
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

        # Distance from each test topo to the reference. In base-reference mode the
        # reference is the base feeder (sentinel -1, not a cache index); otherwise
        # it is the nearest train topology.
        if self.base_reference and self.base_edges is not None:
            ref_edges_list = [self.base_edges]
        else:
            ref_edges_list = [topo_edges[tr_idx] for tr_idx in self.train_topologies]
        d_min_per_test: dict[int, float] = {}
        for t_idx in self.test_topologies:
            d_min = min(
                compute_jaccard_edge_distance(topo_edges[t_idx], ref_edges)
                for ref_edges in ref_edges_list
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

        from src.eval.figures_style import (
            apply_style, FIGSIZE_DOUBLE_COL, ZORDER_TREND, ZORDER_MARKER,
            style_grid, tighten_spines,
        )
        apply_style()

        # Per-method scatter / trend style. Proposed = bold indigo star;
        # MLP = red (brittle); GCNN = orange; MATD3 = purple.
        SCATTER = {
            "GraphSAGE-MAPPO": dict(color="#2c3e7a", marker="*", s=320, edgecolor="white", lw=1.8, alpha=0.95),
            "MLP-MAPPO":       dict(color="#cc3333", marker="^", s=120, edgecolor="#5a1414", lw=1.3, alpha=0.90),
            "GCNN-PPO":        dict(color="#e98b1e", marker="v", s=120, edgecolor="#7a4500", lw=1.3, alpha=0.90),
            "MATD3":           dict(color="#7a3b9e", marker="s", s=110, edgecolor="#3d1f4f", lw=1.3, alpha=0.90),
        }
        TREND = {
            "GraphSAGE-MAPPO": dict(linestyle="-",  linewidth=2.6),
            "MLP-MAPPO":       dict(linestyle="--", linewidth=1.8),
            "GCNN-PPO":        dict(linestyle="-.", linewidth=1.8),
            "MATD3":           dict(linestyle="--", linewidth=1.8),
        }
        NON_LEARN = ("Fixed Droop", "No FFR")
        Y_LO, Y_HI = -8.0, 95.0

        fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_COL)

        # Three regimes via background shading — no inline text labels.
        ax.axhspan(Y_LO, 0,    facecolor="#dfeaf4", alpha=0.55, zorder=0)
        ax.axhspan(50, Y_HI,   facecolor="#f5dcdc", alpha=0.55, zorder=0)
        ax.axhline(0, color="#222222", linewidth=0.9, zorder=ZORDER_TREND - 1)

        # Non-learning baseline as a horizontal reference line.
        non_learn = df[df["method"].isin(NON_LEARN)]
        non_learn_y = float(non_learn["iae_degradation_pct"].mean()) if not non_learn.empty else 30.0
        ax.axhline(non_learn_y, color="#777777", linewidth=1.4, linestyle=":",
                   zorder=ZORDER_TREND - 1,
                   label=f"No-FFR / Droop baseline (≈{non_learn_y:.0f}%)")

        # Add tiny x-jitter ±0.0005 per (method, topology) to spread the
        # vertically stacked cluster at the dense d_E values.
        rng = np.random.default_rng(7)
        df = df.copy()
        df["_x_jitter"] = df["d_E"] + rng.uniform(-0.00045, 0.00045, len(df))

        # Plot in method order so proposed (star) sits on top.
        method_order = ["MLP-MAPPO", "GCNN-PPO", "MATD3", "GraphSAGE-MAPPO"]
        for method in method_order:
            sub = df[df["method"] == method]
            if sub.empty or method not in SCATTER:
                continue
            style = SCATTER[method]
            tstyle = TREND[method]
            x_actual = sub["d_E"].to_numpy(dtype=float)
            x_jit    = sub["_x_jitter"].to_numpy(dtype=float)
            y = sub["iae_degradation_pct"].to_numpy(dtype=float)
            # Regression line uses the TRUE x values (no jitter) within data range.
            if len(sub) >= 2 and np.ptp(x_actual) > 1e-12 and np.all(np.isfinite(y)):
                try:
                    slope, intercept = np.polyfit(x_actual, y, 1)
                    x_line = np.linspace(x_actual.min(), x_actual.max(), 40)
                    ax.plot(x_line, slope * x_line + intercept,
                            color=style["color"], alpha=0.75,
                            zorder=ZORDER_TREND, **tstyle)
                except np.linalg.LinAlgError:
                    pass
            ax.scatter(x_jit, y, label=method,
                       **{k: v for k, v in style.items() if k != "color"},
                       color=style["color"], zorder=ZORDER_MARKER)

        # Single concise inset explaining the narrative — no ellipse, no
        # per-method callouts, no inline region labels (the colour shading
        # already conveys the regime).
        narrative = (
            "Flat slope → topology-robust (proposed)\n"
            "Steep slope → brittle on unseen topologies"
        )
        ax.text(0.02, 0.97, narrative,
                transform=ax.transAxes, fontsize=16, ha="left", va="top",
                color="#222222", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#888888", lw=0.8, alpha=0.96))

        ax.set_xlabel(r"Jaccard edge distance $d_E$ (test $\to$ nearest train)")
        ax.set_ylabel("IAE degradation (%)")
        ax.set_ylim(Y_LO, Y_HI)
        # Bring the x range in a touch so the scatter doesn't sit on the spine.
        ax.set_xlim(df["_x_jitter"].min() - 0.0005, df["_x_jitter"].max() + 0.0005)
        style_grid(ax, minor=False)
        tighten_spines(ax)

        # Legend at top of axes, single row, no frame — clear and uncluttered.
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                  ncol=5, frameon=False, fontsize=16,
                  handlelength=1.8, columnspacing=1.6)

        fig.tight_layout()
        fig.savefig(self.output_dir / "fig6_iae_vs_distance.pdf")
        fig.savefig(self.output_dir / "fig6_iae_vs_distance.png")
        plt.close(fig)
        print("Saved Fig.6: IAE degradation vs d_E")
        return df

    def build_topology_paired_stats(
        self,
        n_runs: int = 10,
        metrics: tuple[str, ...] = ("iae_post", "nadir_hz"),
        proposed_name: str = "GraphSAGE-MAPPO",
    ) -> pd.DataFrame:
        """Per-topology dispersion + paired significance on the unseen split (M3).

        The held-out topologies are the experimental unit: for each policy we
        average over episodes WITHIN a topology, then report the mean and a 95%
        confidence interval ACROSS topologies (not across rollouts). Significance
        of the proposed controller versus the next-best baseline is a paired
        Wilcoxon signed-rank test over the matched per-topology values (the same
        topologies enter both arms), which is the correct paired design here.
        """
        from scipy import stats as _stats

        event = self.scenarios["S2_gen_trip"]
        topo_list = self.test_topologies
        if not topo_list:
            return pd.DataFrame()

        # per-policy, per-metric: one value per topology (mean over its rollouts)
        per_topo: dict[str, dict[str, list[float]]] = {}
        for policy_name, policy in self.policies.items():
            per_topo[policy_name] = {m: [] for m in metrics}
            for topo_idx in topo_list:
                acc = {m: [] for m in metrics}
                for _ in range(n_runs):
                    d = self.run_episode(policy, event=event, topology_idx=topo_idx).to_dict()
                    for m in metrics:
                        acc[m].append(float(d[m]))
                for m in metrics:
                    per_topo[policy_name][m].append(float(np.mean(acc[m])))

        def _ci95(arr: list[float]) -> float:
            a = np.asarray(arr, dtype=float)
            if a.size < 2:
                return 0.0
            sem = a.std(ddof=1) / np.sqrt(a.size)
            return float(_stats.t.ppf(0.975, a.size - 1) * sem)

        rows = []
        for m in metrics:
            means = {p: float(np.mean(per_topo[p][m])) for p in per_topo}
            others = {p: v for p, v in means.items() if p != proposed_name}
            # nadir: higher is better; iae/rocof/settling: lower is better.
            nb = (max(others, key=others.get) if m == "nadir_hz" else min(others, key=others.get)) \
                if others else None
            for policy_name in per_topo:
                arr = per_topo[policy_name][m]
                row = {
                    "metric": m, "method": policy_name, "n_topologies": len(arr),
                    "mean": float(np.mean(arr)), "ci95": _ci95(arr),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                }
                if policy_name == proposed_name and nb is not None:
                    a = np.asarray(per_topo[proposed_name][m], dtype=float)
                    b = np.asarray(per_topo[nb][m], dtype=float)
                    try:
                        _, p = _stats.wilcoxon(a, b)
                    except ValueError:  # zero-difference / too-few samples
                        p = float("nan")
                    row["paired_vs"] = nb
                    row["wilcoxon_p"] = float(p)
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "table_topology_paired_stats.csv", index=False)
        return df

    def run_all(self, n_runs: int = 20) -> dict[str, pd.DataFrame]:
        """Run the FFR + topology evaluation suite (Sections 1 & 2 of outline).

        Outputs (each computed across the full method roster — proposed plus
        all baselines):
          [1/5] table1_ffr_comparison.csv  -> tab_ffr_main
          [2/5] table2_topology_adaptation.csv -> tab_topo_train_test
          [3/5] table3_severity_scaling.csv -> tab_severity
          [4/5] fig_freq_grid_S1_S4.{pdf,png} -> fig_freq_grid
          [5/5] fig6_iae_vs_distance.{pdf,png,csv} -> fig_iae_vs_distance
        """
        print("\n" + "=" * 60)
        print("FFR + TOPOLOGY ADAPTATION EVALUATION")
        print("=" * 60)

        print("\n[1/5] Building Table 1: FFR Performance Comparison...")
        table1 = self.build_table1_ffr_comparison(n_runs=n_runs)

        print("\n[2/5] Building Table 2: Topology Adaptation...")
        table2 = self.build_table2_topology_adaptation(n_runs=max(n_runs // 2, 5))

        print("\n[3/5] Building Table 3: Severity Scaling...")
        table3 = self.build_table3_severity_scaling(n_runs=max(n_runs // 2, 5))

        print("\n[4/5] Plotting multi-scenario freq grid (Proposed vs baselines, mean±std)...")
        # Show the frequency response on the topology the policies were TRAINED on:
        # in base-reference mode that is the nominal base feeder (sentinel -1);
        # otherwise the first cached topology.
        grid_topology = -1 if self.base_reference else 0
        self.plot_frequency_grid_all_scenarios(
            n_runs=max(n_runs // 5, 3),
            topology_idx=grid_topology,
            zoom_post_event_s=20.0,
        )

        print("\n[5/5] Plotting Fig.6 (IAE degradation vs d_E)...")
        self.plot_iae_degradation_vs_distance(n_runs=max(n_runs // 4, 3))

        print("\n[+] Per-topology dispersion + paired Wilcoxon (unseen split)...")
        paired_stats = self.build_topology_paired_stats(n_runs=max(n_runs // 2, 5))

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        pivot1 = table1.pivot_table(index=["scenario", "method"], columns="metric",
                                   values="mean", aggfunc="first")
        print("\nTable 1 - FFR Performance by Method:")
        _t1cols = [c for c in ["iae_total", "nadir_hz", "settling_time_s"] if c in pivot1.columns]
        print(pivot1[_t1cols].round(3))
        print("\nTable 2 - Topology Generalization (IAE-based):")
        _t2cols = [c for c in ["method", "topology_split", "iae_post_mean",
                               "iae_generalization_gap", "iae_retention_ratio"]
                   if c in table2.columns]
        print(table2[_t2cols].to_string(index=False))

        if not paired_stats.empty:
            print("\nPer-topology paired stats (unseen split):")
            print(paired_stats.round(4).to_string(index=False))

        print(f"\nResults saved to: {self.output_dir}")
        return {"table1": table1, "table2": table2, "table3": table3,
                "paired_stats": paired_stats}


# =============================================================================
# Paper-Section Orchestration (4-section outline: Stability, Topology,
# Economic, Harmonic)
# =============================================================================

def export_paper_sections(
    working_dir: Path,
    sections_root: Path,
    table1_df: pd.DataFrame,
    placement_path: Path,
    mpc_path: Path,
    seed: int,
    n_runs_econ: int = 5,
    gcnn_checkpoint: Path | None = None,
    matd3_checkpoint: Path | None = None,
    mlp_mappo_checkpoint: Path | None = None,
    proposed_checkpoint: Path | None = None,
) -> dict[str, list[str]]:
    """Map working-dir outputs into the 4-section paper layout and add the
    section-3 (economic) and section-4 (harmonic) artefacts that
    eval_ffr_topology does not produce on its own.

    Layout written under ``sections_root``:
      section1_stability/  tab_ffr_main.csv, tab_severity.csv, fig_freq_grid.png
      section2_topology/   tab_topo_train_test.csv, tab_encoder_ablation.csv,
                           fig_iae_vs_distance.png
      section3_economic/   tab_economic_methods.csv, tab_cost_effectiveness.csv,
                           fig_pareto.png
      section4_harmonic/   tab_thd_compliance.csv, fig_thd_bus_heatmap.png
    """
    written: dict[str, list[str]] = {}

    # ------------------------------------------------------- section 1
    sec1 = sections_root / "section1_stability"
    sec1.mkdir(parents=True, exist_ok=True)
    pairs1 = [
        ("table1_ffr_comparison.csv", "tab_ffr_main.csv"),
        ("table3_severity_scaling.csv", "tab_severity.csv"),
        ("fig_freq_grid_S1_S4.png", "fig_freq_grid.png"),
    ]
    written["section1_stability"] = _copy_pairs(working_dir, sec1, pairs1)

    # ------------------------------------------------------- section 2
    sec2 = sections_root / "section2_topology"
    sec2.mkdir(parents=True, exist_ok=True)
    pairs2 = [
        ("table2_topology_adaptation.csv", "tab_topo_train_test.csv"),
        ("fig6_iae_vs_distance.png", "fig_iae_vs_distance.png"),
    ]
    written["section2_topology"] = _copy_pairs(working_dir, sec2, pairs2)

    # Encoder ablation: extract MAPPO-PPO rows with different encoders.
    ablation = _build_encoder_ablation(table1_df)
    ablation.to_csv(sec2 / "tab_encoder_ablation.csv", index=False)
    written["section2_topology"].append("tab_encoder_ablation.csv")

    # ------------------------------------------------------- section 3
    sec3 = sections_root / "section3_economic"
    sec3.mkdir(parents=True, exist_ok=True)
    try:
        from src.eval.eval_economics import (
            EconomicsEvaluator, MarketPriceConfig, _load_placement,
        )
        env_econ = MicrogridEnvDual(
            placement_path=str(placement_path), mpc_path=str(mpc_path), seed=seed,
            ffr_mode="mappo_dual",
        )
        evaluator_econ = EconomicsEvaluator(
            env_econ, MarketPriceConfig(), _load_placement(placement_path), sec3,
        )
        policies = _build_all_policies(
            env_econ,
            proposed_checkpoint=proposed_checkpoint,
            mlp_mappo_checkpoint=mlp_mappo_checkpoint,
            gcnn_checkpoint=gcnn_checkpoint,
            matd3_checkpoint=matd3_checkpoint,
        )
        scenarios = _econ_scenarios()
        print(f"\n  [Section 3] Running economics for {len(policies)} methods × "
              f"{len(scenarios)} scenarios × {n_runs_econ} runs...")
        _ = evaluator_econ.build_table_revenue_breakdown(policies, scenarios, n_runs=n_runs_econ)
        t10 = evaluator_econ.build_table_method_comparison(policies, scenarios, n_runs=n_runs_econ)
        evaluator_econ.plot_pareto(t10)
        # Rename to outline names
        if (sec3 / "table10_method_economics.csv").exists():
            shutil.copy2(sec3 / "table10_method_economics.csv", sec3 / "tab_economic_methods.csv")
            written.setdefault("section3_economic", []).append("tab_economic_methods.csv")
        if (sec3 / "fig13_pareto_profit_vs_ffr.png").exists():
            shutil.copy2(sec3 / "fig13_pareto_profit_vs_ffr.png", sec3 / "fig_pareto.png")
            written.setdefault("section3_economic", []).append("fig_pareto.png")
        # DSO-perspective cost-effectiveness derived from the two tables.
        try:
            from scripts.dso_ffr_cost import aggregate_dso_costs
            dso = aggregate_dso_costs(
                sec3 / "table9_revenue_breakdown.csv",
                sec3 / "table10_method_economics.csv",
                events_per_day=5.0,
            )
            dso.sort_values("gross_payment_per_secured_event_eur").to_csv(
                sec3 / "tab_cost_effectiveness.csv", index=False,
            )
            written.setdefault("section3_economic", []).append("tab_cost_effectiveness.csv")
        except Exception as exc:
            print(f"  [warn] tab_cost_effectiveness.csv skipped: {exc}")
    except Exception as exc:
        print(f"  [warn] Section 3 (economic) skipped: {exc}")

    # ------------------------------------------------------- section 4
    # Per-method THD: run each controller on a single representative
    # scenario, drive the env to steady-state dispatch, then call
    # HarmonicAnalyzer on the resulting network. This produces a
    # 123-bus THD_V curve per controller (line plot input) and a
    # per-method compliance row.
    sec4 = sections_root / "section4_harmonic"
    sec4.mkdir(parents=True, exist_ok=True)
    try:
        from src.eval.harmonic_analysis import HarmonicAnalyzer, IEEE519_THD_V_LIMIT
        env_harm = MicrogridEnvDual(
            placement_path=str(placement_path), mpc_path=str(mpc_path), seed=seed,
            ffr_mode="mappo_dual",
        )
        policies_harm = _build_all_policies(
            env_harm,
            proposed_checkpoint=proposed_checkpoint,
            mlp_mappo_checkpoint=mlp_mappo_checkpoint,
            gcnn_checkpoint=gcnn_checkpoint,
            matd3_checkpoint=matd3_checkpoint,
        )
        import pandapower as pp
        from scripts.eval_thd import commanded_p_mw, agent_rated_mw

        thd_v_per_method: dict[str, np.ndarray] = {}
        rows: list[dict] = []
        for m_name, policy in policies_harm.items():
            obs_fast, _, _ = env_harm.reset()
            n_bus = len(env_harm.net.bus.index)
            edge_index = ensure_edge_index(env_harm.edge_index, n_nodes=n_bus)
            obs_full = build_am_full_feeder_obs(env_harm, obs_fast)
            last_action = None
            # Drive 30 fast steps (~3 s) so the dispatch settles before
            # measuring harmonics. No event injected: this is the steady-state
            # power-quality audit, not a contingency response.
            for _ in range(30):
                try:
                    action = policy.act(obs_full, edge_index, env_harm, obs_fast=obs_fast)
                except TypeError:
                    action = policy.act(obs_full, edge_index, env_harm)
                last_action = action
                obs_fast, _, _, _, info = env_harm.step_fast(action)
                edge_index = ensure_edge_index(info.get("edge_index", edge_index), n_nodes=n_bus)
                obs_full = build_am_full_feeder_obs(env_harm, obs_fast)
            try:
                pp.runpp(env_harm.net, numba=False, algorithm="nr", init="flat")
                if last_action is None:
                    n_act = env_harm.n_agents + len(env_harm._vpp_droop_agents)
                    last_action = np.zeros(n_act, dtype=np.float32)
                p_mw = commanded_p_mw(env_harm, np.asarray(last_action, dtype=float))
                rated = agent_rated_mw(env_harm)
                agent_bus_idx = [int(b) for b in env_harm._agent_bus_pp.tolist()]
                gfm_idx = getattr(env_harm.net, "_gfm_bus_idx", None)
                if gfm_idx is not None and int(gfm_idx) not in set(agent_bus_idx):
                    agent_bus_idx = agent_bus_idx + [int(gfm_idx)]
                    p_mw = np.concatenate([p_mw, np.zeros(1, dtype=float)])
                    rated = np.concatenate([rated, np.ones(1, dtype=float)])
                vm = env_harm.net.res_bus["vm_pu"].values
                bus_mask = np.isfinite(vm) & (np.abs(vm) > 0.05)
                analyzer = HarmonicAnalyzer(env_harm.net)
                result = analyzer.run(p_mw, agent_bus_idx, bus_mask=bus_mask,
                                      agent_p_rated_mw=rated)
            except Exception as exc:
                print(f"  [warn] HarmonicAnalyzer failed for {m_name}: {exc}")
                continue
            thd_v_per_bus = np.asarray(result.get("THD_V_pct", []), dtype=float)
            if thd_v_per_bus.size > 0:
                thd_v_per_method[m_name] = thd_v_per_bus
            rows.append({
                "method": m_name,
                "THD_V_PCC_pct": float(result.get("THD_V_PCC", float("nan"))),
                "THD_V_max_pct": float(result.get("THD_V_max", float("nan"))),
                "Buses_over_5pct": int(result.get("buses_over_limit",
                                                 int(np.sum(thd_v_per_bus > IEEE519_THD_V_LIMIT)))),
                "limit_pct": IEEE519_THD_V_LIMIT,
                "standard": "IEEE Std 519-2014 §5.1 MV (5%)",
            })
        pd.DataFrame(rows).to_csv(sec4 / "tab_thd_compliance.csv", index=False)
        written["section4_harmonic"] = ["tab_thd_compliance.csv"]
        if thd_v_per_method:
            _plot_thd_per_bus_lines(thd_v_per_method, sec4 / "fig_thd_per_bus.png")
            written["section4_harmonic"].append("fig_thd_per_bus.png")
    except Exception as exc:
        print(f"  [warn] Section 4 (harmonic) skipped: {exc}")

    print("\n=== Paper sections written ===")
    for sec, files in written.items():
        print(f"  {sec}: {files}")
    return written


def _copy_pairs(src: Path, dst: Path, pairs: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for src_name, dst_name in pairs:
        p = src / src_name
        if p.exists():
            shutil.copy2(p, dst / dst_name)
            out.append(dst_name)
        else:
            print(f"  [skip] {dst.name}/{dst_name}: source missing ({p.name})")
    return out


def _build_encoder_ablation(table1: pd.DataFrame) -> pd.DataFrame:
    """Encoder ablation: same MAPPO RL stack, different encoders.

    Filter table1 to GraphSAGE-MAPPO (proposed) and MLP-MAPPO (ablation),
    aggregate FFR success rate, IAE post, nadir, RoCoF over all scenarios.
    """
    if table1.empty or "method" not in table1.columns:
        return pd.DataFrame(columns=["encoder", "n_scenarios", "ffr_sr_mean", "iae_post_mean", "nadir_mean", "rocof_max_mean"])
    keep = {"GraphSAGE-MAPPO": "GraphSAGE", "MLP-MAPPO": "MLP"}
    subset = table1[table1["method"].isin(keep.keys())]
    if subset.empty:
        return pd.DataFrame(columns=["encoder", "n_scenarios", "ffr_sr_mean", "iae_post_mean", "nadir_mean", "rocof_max_mean"])
    pivot = subset.pivot_table(index="method", columns="metric", values="mean", aggfunc="mean")
    rows = []
    for method, encoder in keep.items():
        if method not in pivot.index:
            continue
        rows.append({
            "encoder": encoder,
            "n_scenarios": int(subset[subset["method"] == method]["scenario"].nunique()),
            "ffr_sr_mean": float(pivot.loc[method].get("ffr_success", float("nan"))),
            "iae_post_mean": float(pivot.loc[method].get("iae_post", float("nan"))),
            "nadir_mean": float(pivot.loc[method].get("nadir_hz", float("nan"))),
            "rocof_max_mean": float(pivot.loc[method].get("rocof_max_hz_s", float("nan"))),
        })
    return pd.DataFrame(rows)


def _build_all_policies(
    env, *,
    proposed_checkpoint: Path | None,
    mlp_mappo_checkpoint: Path | None,
    gcnn_checkpoint: Path | None,
    matd3_checkpoint: Path | None,
) -> dict[str, Any]:
    policies: dict[str, Any] = {
        "No FFR": NoFFRPolicy(),
        "Fixed Droop": FixedDroopPolicy(k_droop=0.05),
    }
    specs = [
        ("GraphSAGE-MAPPO", proposed_checkpoint, GraphSAGEMAPPOPolicy),
        ("MLP-MAPPO",       mlp_mappo_checkpoint, MLPMAPPOPolicy),
        ("GCNN-PPO",        gcnn_checkpoint, GCNNPPOPolicy),
        ("MATD3",           matd3_checkpoint, MATD3Policy),
    ]
    for name, ckpt, cls in specs:
        if ckpt is None or not Path(ckpt).exists():
            continue
        try:
            policies[name] = cls(ckpt, env)
        except Exception as exc:
            print(f"  [warn] Could not load {name}: {exc}")
    return policies


def _econ_scenarios() -> dict[str, EventConfig]:
    return {
        "S1_load_step":      EventConfig(type="load_step", delta_P_mw=4.5,  location=45,    t_inject=30.0),
        "S2_gen_trip":       EventConfig(type="gen_trip",  delta_P_mw=-5.0, location=67,    t_inject=30.0),
        "S3_line_trip":      EventConfig(type="line_trip", delta_P_mw=-4.5, location=67068, t_inject=30.0),
        "S4_high_ren_surge": EventConfig(type="high_ren",  delta_P_mw=5.5,  location=105,   t_inject=30.0),
    }


def _plot_thd_per_bus_lines(thd_v_per_method: dict[str, np.ndarray], out_path: Path) -> None:
    """Per-bus THD_V line plot: one curve per controller, IEEE 519 limit shown.

    X-axis: bus index (clipped to valid analyzer range, typically 0-114).
    Y-axis: voltage THD (%), auto-zoomed.
    Buses outside the analyzer's bus_mask (vm < 0.05 pu) appear as 0 or NaN
    in the input; both are masked here so the line breaks cleanly.
    The IEEE 519-2014 limit is annotated directly on the dashed line at the
    right edge instead of cluttering the legend.
    """
    from src.eval.figures_style import (
        apply_style, FIGSIZE_DOUBLE_COL, ZORDER_TREND, ZORDER_MAIN,
        color_for_method, style_grid, tighten_spines,
    )
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_COL)
    methods = list(thd_v_per_method.keys())

    # First pass: find the largest LEADING continuous prefix that is fully
    # valid across ALL methods. This robustly excludes the noisy tail buses
    # (GFM / virtual nodes) at the right edge of the IEEE 123 feeder.
    max_valid = 0
    arrays: dict[str, np.ndarray] = {}
    for method in methods:
        y = np.asarray(thd_v_per_method[method], dtype=float).copy()
        if y.size == 0:
            continue
        y[~np.isfinite(y)] = np.nan
        y[y == 0.0] = np.nan
        arrays[method] = y
    if arrays:
        common_len = min(a.size for a in arrays.values())
        # Find the longest prefix where every method has a finite value.
        for idx in range(common_len):
            if any(np.isnan(a[idx]) for a in arrays.values()):
                max_valid = idx - 1
                break
        else:
            max_valid = common_len - 1
        # Require at least 80 buses; otherwise fall back to the loosest
        # criterion (any method's max valid prefix).
        if max_valid < 80:
            for method, y in arrays.items():
                valid = np.isfinite(y)
                if valid.any():
                    max_valid = max(max_valid, int(np.max(np.where(valid)[0])))

    for method in methods:
        y_full = np.asarray(thd_v_per_method[method], dtype=float).copy()
        if y_full.size == 0:
            continue
        # Mask invalid buses then clip to the common valid range.
        y_full[~np.isfinite(y_full)] = np.nan
        y_full[y_full == 0.0] = np.nan
        y = y_full[: max_valid + 1]
        x = np.arange(y.size)
        lw = 2.6 if method.startswith(("GraphSAGE", "GNN-MAPPO")) else 1.8
        zorder = ZORDER_MAIN if method.startswith(("GraphSAGE", "GNN-MAPPO")) else ZORDER_MAIN - 2
        ax.plot(x, y, label=method, color=color_for_method(method),
                linewidth=lw, zorder=zorder)

    # IEEE 519 5% limit drawn as a dashed red horizontal line + inline label.
    ax.axhline(5.0, linestyle="--", color="#b25555", linewidth=2.0,
               alpha=0.95, zorder=ZORDER_TREND)
    ax.text(max_valid * 0.995, 5.05, "IEEE 519-2014 MV limit (5%)",
            ha="right", va="bottom", fontsize=16, color="#7a2828",
            fontstyle="italic", fontweight="bold", zorder=ZORDER_TREND + 1,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#b25555", lw=0.8, alpha=0.96))

    ax.set_xlabel("Bus index (IEEE 123 feeder)", fontsize=18)
    ax.set_ylabel(r"THD$_V$ (%)", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    ax.set_xlim(0, max_valid)
    # Auto-zoom y to the actual data range with a small margin.
    all_y = np.concatenate([
        np.asarray(thd_v_per_method[m], dtype=float)[: max_valid + 1]
        for m in methods if np.asarray(thd_v_per_method[m]).size > 0
    ])
    finite = all_y[np.isfinite(all_y) & (all_y > 0)]
    if finite.size > 0:
        ax.set_ylim(max(0.0, float(finite.min()) - 0.15),
                    max(5.2, float(finite.max()) + 0.15))
    style_grid(ax, minor=False)
    tighten_spines(ax)

    # Legend at bottom, 3 cols, larger font.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False, fontsize=16,
              handlelength=2.2, columnspacing=1.8)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# Backward-compatibility alias (callers still import _plot_thd_heatmap; the
# function now produces a line plot per the outline revision).
_plot_thd_heatmap = _plot_thd_per_bus_lines


# =============================================================================
# CLI
# =============================================================================

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
    parser.add_argument("--base-reference", action="store_true",
                        help="Train-on-base, eval-on-all-reconfig protocol: reference = nominal "
                             "base feeder, ALL cached reconfig topologies treated as unseen.")
    parser.add_argument("--paper-sections-out", type=Path, default=None,
                        help="If set, after run_all also export all 4 paper sections "
                             "(stability/topology/economic/harmonic) into this root dir.")
    parser.add_argument("--n-runs-econ", type=int, default=5,
                        help="Episodes per (method, scenario) for the economic section "
                             "(default 5; only used if --paper-sections-out is set).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_config = {
        "placement_path": str(args.placement),
        "mpc_path": str(args.mpc_path),
        "seed": args.seed,
        # MUST match training: policies were trained with the dual (a_P, a_K) action
        # under mappo_dual; the env defaults to "droop", which would feed the policies
        # the wrong action layout and invalidate the evaluation.
        "ffr_mode": "mappo_dual",
        # Operating points (load/PV/price) drawn from the held-out day partition.
        "day_split": "eval",
    }

    evaluator = FFRTopologyEvaluator(
        env_config=env_config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        gcnn_checkpoint=args.gcnn_checkpoint,
        matd3_checkpoint=args.matd3_checkpoint,
        mlp_mappo_checkpoint=args.mlp_mappo_checkpoint,
        base_reference=args.base_reference,
    )

    results = evaluator.run_all(n_runs=args.n_runs)

    if args.paper_sections_out is not None:
        export_paper_sections(
            working_dir=args.output_dir,
            sections_root=args.paper_sections_out,
            table1_df=results["table1"],
            placement_path=args.placement,
            mpc_path=args.mpc_path,
            seed=args.seed,
            n_runs_econ=args.n_runs_econ,
            gcnn_checkpoint=args.gcnn_checkpoint,
            matd3_checkpoint=args.matd3_checkpoint,
            mlp_mappo_checkpoint=args.mlp_mappo_checkpoint,
            proposed_checkpoint=args.checkpoint,
        )


if __name__ == "__main__":
    main()
