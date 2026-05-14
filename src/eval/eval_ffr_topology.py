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

    # FFR success: nadir ≥ 49.5 Hz AND RoCoF ≤ 2.0 Hz/s (IEEE 1547 Cat III)
    ffr_success = (nadir >= (f_nominal - f_limit)) and (rocof_max <= rocof_limit)

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
    def act(self, obs: np.ndarray, edge_index: np.ndarray, env: Any) -> np.ndarray:
        n_agents = env.n_agents
        n_vpps = len(env._vpp_droop_agents)
        return np.zeros(n_agents + n_vpps, dtype=np.float32)


class FixedDroopPolicy:
    """Baseline: Fixed droop control (linear proportional)."""
    def __init__(self, k_droop: float = 0.5):
        self.k_droop = k_droop

    def act(self, obs: np.ndarray, edge_index: np.ndarray, env: Any) -> np.ndarray:
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
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Extract dimensions from checkpoint
        agent_state = ckpt.get("agent_state_dict", ckpt)

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

    def act(self, obs: np.ndarray, edge_index: np.ndarray, env: Any) -> np.ndarray:
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
    ):
        self.env = MicrogridEnvDual(**env_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build policies
        self.policies = {
            "No FFR": NoFFRPolicy(),
            "Fixed Droop": FixedDroopPolicy(k_droop=0.5),
        }

        if checkpoint_path and checkpoint_path.exists():
            try:
                self.policies["GraphSAGE-MAPPO"] = GraphSAGEMAPPOPolicy(checkpoint_path, self.env)
                print(f"Loaded GraphSAGE-MAPPO from {checkpoint_path}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

        # Contingency scenarios (Section VI - 4 scenarios aligned with IEEE 1547)
        # S_BASE = 15.705 MW, H_SYS = 1.18s
        self.scenarios = {
            "S1_load_step": EventConfig(type="load_step", delta_P_mw=2.5, location=45, t_inject=30.0),   # 16% S_BASE, RoCoF=-3.36
            "S2_gen_trip": EventConfig(type="gen_trip", delta_P_mw=-3.9, location=97, t_inject=30.0),    # 25% S_BASE, RoCoF=+5.24
            "S3_line_trip": EventConfig(type="line_trip", delta_P_mw=-2.4, location=97098, t_inject=30.0),  # 15% S_BASE, topology change
            "S4_gen_trip_severe": EventConfig(type="gen_trip", delta_P_mw=-5.5, location=97, t_inject=30.0),  # 35% S_BASE, extreme
        }

        # Topology splits using farthest-point selection
        topo_cache = getattr(self.env.reconfig, "_cache", [])
        n_topos = len(topo_cache)
        if n_topos >= 5:
            # Extract edge_index from each cached topology
            topo_edges = []
            for topo in topo_cache:
                ei = topo.get("edge_index", np.array([[],[]])) if isinstance(topo, dict) else np.array([[],[]])
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

        for _ in range(n_steps):
            action = policy.act(obs_full, edge_index, self.env)
            obs_fast, _, done, _, info = self.env.step_fast(action)

            # Update graph if topology changed
            new_edge = info.get("edge_index", edge_index)
            edge_index = ensure_edge_index(new_edge, n_nodes=n_bus)
            obs_full = build_am_full_feeder_obs(self.env, obs_fast)

            freq_state = self.env.freq_dyn.get_state()
            f_trace.append(50.0 + freq_state.delta_f_hz)
            rocof_trace.append(freq_state.rocof_hz_s)

        event_step = int(event.t_inject) if event else 30
        return compute_ffr_metrics(
            np.array(f_trace),
            np.array(rocof_trace),
            event_step=event_step,
        )

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
        event = self.scenarios["gen_trip_3MW"]
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

        # Compute generalization gap
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

    def plot_frequency_traces(self, scenario_name: str = "gen_trip_3MW") -> None:
        """Plot frequency traces comparing methods."""
        event = self.scenarios.get(scenario_name)
        if event is None:
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Collect traces
        traces = {}
        for policy_name, policy in self.policies.items():
            m = self.run_episode(policy, event=event, topology_idx=0)
            traces[policy_name] = m.f_trace

        # Plot frequency
        ax1 = axes[0]
        for name, trace in traces.items():
            ax1.plot(trace, label=name, linewidth=1.5)
        ax1.axhline(49.5, ls="--", color="red", alpha=0.7, label="UFLS threshold (49.5 Hz)")
        ax1.axhline(50.0, ls=":", color="gray", alpha=0.5)
        ax1.axvline(event.t_inject, ls="--", color="orange", alpha=0.7, label=f"Event @ t={event.t_inject}s")
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_ylim(49.0, 50.5)
        ax1.legend(loc="lower right", fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.set_title(f"FFR Performance: {scenario_name}")

        # Plot deviation
        ax2 = axes[1]
        for name, trace in traces.items():
            delta_f = trace - 50.0
            ax2.plot(delta_f, label=name, linewidth=1.5)
        ax2.axhline(-0.5, ls="--", color="red", alpha=0.7)
        ax2.axhline(0.0, ls=":", color="gray", alpha=0.5)
        ax2.axvline(event.t_inject, ls="--", color="orange", alpha=0.7)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Δf (Hz)")
        ax2.set_ylim(-0.8, 0.3)
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

        event = self.scenarios["gen_trip_3MW"]

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

    def run_all(self, n_runs: int = 20) -> dict[str, pd.DataFrame]:
        """Run all evaluations."""
        print("\n" + "="*60)
        print("FFR + TOPOLOGY ADAPTATION EVALUATION")
        print("="*60)

        print("\n[1/5] Building Table 1: FFR Performance Comparison...")
        table1 = self.build_table1_ffr_comparison(n_runs=n_runs)

        print("\n[2/5] Building Table 2: Topology Adaptation...")
        table2 = self.build_table2_topology_adaptation(n_runs=max(n_runs // 2, 5))

        print("\n[3/5] Building Table 3: Severity Scaling...")
        table3 = self.build_table3_severity_scaling(n_runs=max(n_runs // 2, 5))

        print("\n[4/5] Plotting frequency traces...")
        self.plot_frequency_traces("gen_trip_3MW")
        self.plot_frequency_traces("load_step_4MW")

        print("\n[5/5] Plotting topology comparison...")
        self.plot_topology_comparison()

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
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints_am_mappo/am_mappo_final.pt"))
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
    )

    evaluator.run_all(n_runs=args.n_runs)


if __name__ == "__main__":
    main()
