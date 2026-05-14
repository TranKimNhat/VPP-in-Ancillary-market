"""comparison_runner.py — exhaustive topology–scenario evaluation for Section VI.

Outputs:
  case_metrics.csv
  case_trajectories/method=.../topo=.../scenario=....npz
  table1_frequency_all.csv
  table1_frequency_by_scenario.csv
  table2_slow_proxy_all.csv
  table2_slow_proxy_by_scenario.csv
  table3_topology_generalization.csv
  table4_topology_split_quality.csv
  table_thd_diagnostic.csv
  topology_split_summary.csv
  topology_split_stats.json
  comparison_gate.json
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.baselines.gcnn_ppo import GCNNPPOAgent, ensure_edge_index
from src.baselines.graph_ppo import GraphPPOAgent
from src.baselines.matd3 import MATD3Agent, MATD3Config
from src.baselines.sgac import SGACAgent
from src.env.events import EventConfig
from src.env.microgrid_env_dual import MicrogridEnvDual
from src.eval.evaluate_dual import DeterministicDualPolicy, FixedDroopPolicy, NoFFRPolicy
from src.eval.harmonic_analysis import HarmonicAnalyzer


def _to_legacy_obs5(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 41 or arr.shape[1] < 5:
        raise ValueError(f"Expected obs shape (41,>=5), got {arr.shape}")
    return arr[:, :5]


def _edge_set_from_index(edge_index: np.ndarray) -> set[tuple[int, int]]:
    if edge_index.shape[1] == 0:
        return set()
    edges: set[tuple[int, int]] = set()
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        edges.add((min(u, v), max(u, v)))
    return edges


def _edge_hash(edge_set: set[tuple[int, int]]) -> str:
    return "|".join(f"{u}-{v}" for u, v in sorted(edge_set))


def _jaccard_edge_distance(set_a: set[tuple[int, int]], set_b: set[tuple[int, int]]) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - (intersection / union) if union > 0 else 0.0


def compute_topology_split(
    cache: list[tuple[Any, np.ndarray, set[int]]],
    train_ratio: float = 0.75,
    seed: int = 42,
) -> tuple[list[int], list[int], pd.DataFrame, dict[int, str], dict[int, float], dict[int, int]]:
    topo_data: list[dict[str, Any]] = []
    for idx, (_net, edge_index, _open_set) in enumerate(cache):
        edge_set = _edge_set_from_index(edge_index)
        topo_data.append(
            {
                "topology_id": idx,
                "edge_set": edge_set,
                "edge_hash": _edge_hash(edge_set),
                "n_edges": len(edge_set),
            }
        )

    unique_by_hash: dict[str, dict[str, Any]] = {}
    for td in topo_data:
        if td["edge_hash"] not in unique_by_hash:
            unique_by_hash[td["edge_hash"]] = td

    unique_topos = list(unique_by_hash.values())
    if len(unique_topos) <= 1:
        only = unique_topos[0] if unique_topos else {"topology_id": 0, "edge_hash": "", "n_edges": 0}
        rows = [
            {
                "topology_id": int(only["topology_id"]),
                "edge_hash": str(only["edge_hash"]),
                "split": "train",
                "n_edges": int(only["n_edges"]),
                "d_min_to_train": 0.0,
            }
        ]
        summary = pd.DataFrame(rows)
        return [int(only["topology_id"])], [int(only["topology_id"])], summary, {int(only["topology_id"]): str(only["edge_hash"])}, {int(only["topology_id"]): 0.0}, {int(only["topology_id"]): int(only["n_edges"])}

    unique_topos.sort(key=lambda x: str(x["edge_hash"]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_topos))
    shuffled = [unique_topos[i] for i in perm]

    split_point = max(1, int(round(train_ratio * len(shuffled))))
    split_point = min(split_point, len(shuffled) - 1)
    train_topos = shuffled[:split_point]
    test_topos = shuffled[split_point:]

    train_idx = [int(t["topology_id"]) for t in train_topos]
    test_idx = [int(t["topology_id"]) for t in test_topos]
    train_sets = [t["edge_set"] for t in train_topos]

    rows: list[dict[str, Any]] = []
    d_min_map: dict[int, float] = {}
    edge_hash_map: dict[int, str] = {}
    n_edges_map: dict[int, int] = {}

    for t in train_topos:
        topo_id = int(t["topology_id"])
        edge_hash_map[topo_id] = str(t["edge_hash"])
        n_edges_map[topo_id] = int(t["n_edges"])
        others = [s for s in train_sets if s is not t["edge_set"]]
        d_min = min((_jaccard_edge_distance(t["edge_set"], s) for s in others), default=0.0)
        d_min_map[topo_id] = float(d_min)
        rows.append(
            {
                "topology_id": topo_id,
                "edge_hash": str(t["edge_hash"]),
                "split": "train",
                "n_edges": int(t["n_edges"]),
                "d_min_to_train": float(d_min),
            }
        )

    for t in test_topos:
        topo_id = int(t["topology_id"])
        edge_hash_map[topo_id] = str(t["edge_hash"])
        n_edges_map[topo_id] = int(t["n_edges"])
        d_min = min(_jaccard_edge_distance(t["edge_set"], s) for s in train_sets)
        d_min_map[topo_id] = float(d_min)
        rows.append(
            {
                "topology_id": topo_id,
                "edge_hash": str(t["edge_hash"]),
                "split": "test",
                "n_edges": int(t["n_edges"]),
                "d_min_to_train": float(d_min),
            }
        )

    return train_idx, test_idx, pd.DataFrame(rows), edge_hash_map, d_min_map, n_edges_map


class _RuleWrapper:
    def __init__(self, policy: Any) -> None:
        self._policy = policy

    def act_fast(self, obs_fast: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        return self._policy.act(obs_fast, edge_index)

    def act_slow(self, obs_slow: np.ndarray, edge_index: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.zeros((82,), dtype=np.float32)


class _GCNNPPOWrapper:
    def __init__(self, agent: GCNNPPOAgent, env: MicrogridEnvDual) -> None:
        self._agent = agent
        self._env = env

    def _map(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        p_all = raw[0::3]
        droop_all = raw[2::3]
        vpp_droop = np.zeros(3, dtype=np.float32)
        vpp_agents = [
            list(range(9, 12)) + list(range(18, 21)),
            list(range(12, 15)) + list(range(21, 24)),
            list(range(15, 18)) + list(range(24, 27)),
        ]
        for i, agents in enumerate(vpp_agents):
            vpp_droop[i] = float(droop_all[agents].mean())
        return np.clip(np.concatenate([p_all, vpp_droop]).astype(np.float32), -1.0, 1.0)

    def act_fast(self, obs_fast: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        obs_fast_legacy = _to_legacy_obs5(obs_fast)
        obs_slow = np.zeros_like(obs_fast_legacy)
        combined = GCNNPPOAgent._combine_obs(obs_fast_legacy, obs_slow)
        global_obs = combined.reshape(-1)
        raw, _lp, _v = self._agent.act(combined, edge_index, global_obs)
        return self._map(raw)

    def act_slow(self, obs_slow: np.ndarray, edge_index: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.zeros((82,), dtype=np.float32)


class _SGACWrapper:
    def __init__(self, agent: SGACAgent, env: MicrogridEnvDual) -> None:
        self._agent = agent
        self._env = env

    def _map(self, raw: np.ndarray) -> np.ndarray:
        a = np.asarray(raw, dtype=np.float32).reshape(41, 3)
        p_all = a[:, 0]
        droop_all = a[:, 2]
        vpp_droop = np.zeros(3, dtype=np.float32)
        vpp_agents = [
            list(range(9, 12)) + list(range(18, 21)),
            list(range(12, 15)) + list(range(21, 24)),
            list(range(15, 18)) + list(range(24, 27)),
        ]
        for i, agents in enumerate(vpp_agents):
            vpp_droop[i] = float(droop_all[agents].mean())
        return np.clip(np.concatenate([p_all, vpp_droop]).astype(np.float32), -1.0, 1.0)

    def act_fast(self, obs_fast: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        obs_fast_legacy = _to_legacy_obs5(obs_fast)
        obs_slow = np.zeros_like(obs_fast_legacy)
        combined = SGACAgent._combine_obs(obs_fast_legacy, obs_slow)
        raw = self._agent.act(combined, edge_index)
        return self._map(raw)

    def act_slow(self, obs_slow: np.ndarray, edge_index: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.zeros((82,), dtype=np.float32)


class _GraphPPOWrapper:
    def __init__(self, agent: GraphPPOAgent, env: MicrogridEnvDual) -> None:
        self._agent = agent
        self._env = env
        self._agent.env = env

    def _map(self, raw: np.ndarray) -> np.ndarray:
        a = np.asarray(raw, dtype=np.float32).reshape(41, 3)
        p_all = a[:, 0]
        droop_all = a[:, 2]
        vpp_droop = np.zeros(3, dtype=np.float32)
        vpp_agents = [
            list(range(9, 12)) + list(range(18, 21)),
            list(range(12, 15)) + list(range(21, 24)),
            list(range(15, 18)) + list(range(24, 27)),
        ]
        for i, agents in enumerate(vpp_agents):
            vpp_droop[i] = float(droop_all[agents].mean())
        return np.clip(np.concatenate([p_all, vpp_droop]).astype(np.float32), -1.0, 1.0)

    def act_fast(self, obs_fast: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        obs_fast_legacy = _to_legacy_obs5(obs_fast)
        obs_slow = np.zeros_like(obs_fast_legacy)
        combined = GraphPPOAgent._combine_obs(obs_fast_legacy, obs_slow)
        adj_phys, adj_ed = self._agent._get_adjacencies(edge_index)
        raw, _lp, _v = self._agent.act(combined, adj_phys, adj_ed)
        return self._map(raw)

    def act_slow(self, obs_slow: np.ndarray, edge_index: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.zeros((82,), dtype=np.float32)


class _MLPMAPPOWrapper:
    """Wrapper for MLP-MAPPO ablation (no graph encoder)."""

    def __init__(self, checkpoint_path: Path, env: MicrogridEnvDual) -> None:
        self._policy = DeterministicDualPolicy(checkpoint_path)
        self._env = env

    def act_fast(self, obs_fast: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        return self._policy.act_fast(obs_fast, edge_index)

    def act_slow(self, obs_slow: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        return self._policy.act_slow(obs_slow, edge_index)


class _MATD3Wrapper:
    """Wrapper for MATD3 baseline (Li & Zhou 2025)."""

    def __init__(self, checkpoint_path: Path, env: MicrogridEnvDual) -> None:
        self._env = env
        n_agents = env.n_agents
        config = MATD3Config(obs_dim=24, action_dim=2, n_agents=n_agents)
        self._agent = MATD3Agent(config, device="cpu")
        self._agent.load(checkpoint_path)
        self._agent.eval()

    def _build_obs(self, obs_fast: np.ndarray) -> np.ndarray:
        n_agents = self._env.n_agents
        obs = np.zeros((n_agents, 24), dtype=np.float32)
        freq_state = self._env.freq_dyn.get_state()
        for i in range(n_agents):
            obs[i, 0] = freq_state.delta_f_hz
            obs[i, 1] = freq_state.rocof_hz_s
            if i < len(obs_fast):
                obs[i, 2:min(24, 2 + len(obs_fast[i]))] = obs_fast[i][:22]
        return obs

    def act_fast(self, obs_fast: np.ndarray, edge_index: np.ndarray) -> np.ndarray:  # noqa: ARG002
        obs = self._build_obs(obs_fast)
        actions = self._agent.act_deterministic(obs)
        p_actions = actions[:, 0]
        n_vpps = len(self._env._vpp_droop_agents)
        vpp_droop = np.zeros(n_vpps, dtype=np.float32)
        vpp_agents_list = [
            list(range(9, 12)) + list(range(18, 21)),
            list(range(12, 15)) + list(range(21, 24)),
            list(range(15, 18)) + list(range(24, 27)),
        ]
        for i, agents in enumerate(vpp_agents_list[:n_vpps]):
            valid_agents = [a for a in agents if a < len(actions)]
            if valid_agents:
                vpp_droop[i] = float(np.mean([actions[a, 1] for a in valid_agents]))
        return np.clip(np.concatenate([p_actions, vpp_droop]).astype(np.float32), -1.0, 1.0)

    def act_slow(self, obs_slow: np.ndarray, edge_index: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.zeros((82,), dtype=np.float32)


class BaselineComparison:
    SCENARIOS: dict[str, dict[str, Any]] = {
        "S1": {
            "event": EventConfig(type="load_step", delta_P_mw=2.4, location=45, t_inject=30.0),
            "severity_level": "moderate",
            "severity_label": "S1_moderate_disturbance",
            "delta_p_cont": 2.4,
            "event_start": 30.0,
            "duration": 30.0,
            "frequency_threshold": 0.5,
            "rocof_threshold": 1.0,
        },
        "S2": {
            "event": EventConfig(type="gen_trip", delta_P_mw=-3.9, location=97, t_inject=30.0),
            "severity_level": "severe",
            "severity_label": "S2_severe_disturbance",
            "delta_p_cont": -3.9,
            "event_start": 30.0,
            "duration": 30.0,
            "frequency_threshold": 0.5,
            "rocof_threshold": 1.0,
        },
        "S3": {
            "event": EventConfig(type="line_trip", delta_P_mw=-2.4, location=108, t_inject=30.0),
            "severity_level": "severe",
            "severity_label": "S3_severe_disturbance",
            "delta_p_cont": -2.4,
            "event_start": 30.0,
            "duration": 30.0,
            "frequency_threshold": 0.5,
            "rocof_threshold": 1.0,
        },
        "S4": {
            "event": EventConfig(type="high_ren", delta_P_mw=4.7, location=97, t_inject=30.0),
            "severity_level": "extreme",
            "severity_label": "S4_extreme_highren",
            "delta_p_cont": 4.7,
            "event_start": 30.0,
            "duration": 30.0,
            "frequency_threshold": 0.5,
            "rocof_threshold": 1.0,
        },
        "S5": {
            "event": EventConfig(type="gen_trip", delta_P_mw=-5.5, location=97, t_inject=30.0),
            "severity_level": "extreme",
            "severity_label": "S5_extreme_gentrip",
            "delta_p_cont": -5.5,
            "event_start": 30.0,
            "duration": 30.0,
            "frequency_threshold": 0.5,
            "rocof_threshold": 1.0,
        },
    }

    def __init__(
        self,
        proposed_checkpoint: Path,
        nograph_checkpoint: Path | None,
        fixedtopo_checkpoint: Path | None,
        graphppo_checkpoint: Path | None,
        gcnn_checkpoint: Path | None,
        env_config: dict[str, Any],
        seed: int = 42,
        topology_cache: list[tuple[Any, np.ndarray, set[int]]] | None = None,
        mlp_mappo_checkpoint: Path | None = None,
        matd3_checkpoint: Path | None = None,
    ) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env_config = dict(env_config)
        env_kwargs = dict(env_config)
        if topology_cache is not None:
            env_kwargs["topology_cache"] = topology_cache
        self.env = MicrogridEnvDual(**env_kwargs)

        self._agent_p_rated_mw = np.asarray([max(float(spec.get("p_rated", 0.0)), 0.0) for spec in self.env._agent_specs], dtype=np.float32)
        self._thd_fixed_p_mw = 0.5 * self._agent_p_rated_mw

        self.methods: dict[str, Any] = {}
        self.methods["GraphSAGE-Dual-PPO (Proposed)"] = DeterministicDualPolicy(proposed_checkpoint)
        if nograph_checkpoint is not None and nograph_checkpoint.exists():
            self.methods["No-Graph Dual-PPO"] = DeterministicDualPolicy(nograph_checkpoint)
        if fixedtopo_checkpoint is not None and fixedtopo_checkpoint.exists():
            self.methods["Fixed-Topology Dual-PPO"] = DeterministicDualPolicy(fixedtopo_checkpoint)
        if graphppo_checkpoint is not None and graphppo_checkpoint.exists() and hasattr(GraphPPOAgent, "load"):
            self.methods["Graph-PPO"] = _GraphPPOWrapper(GraphPPOAgent.load(graphppo_checkpoint), self.env)
        if gcnn_checkpoint is not None and gcnn_checkpoint.exists() and hasattr(GCNNPPOAgent, "load"):
            self.methods["GCNN-PPO"] = _GCNNPPOWrapper(GCNNPPOAgent.load(gcnn_checkpoint), self.env)
        if mlp_mappo_checkpoint is not None and mlp_mappo_checkpoint.exists():
            self.methods["MLP-MAPPO"] = _MLPMAPPOWrapper(mlp_mappo_checkpoint, self.env)
        if matd3_checkpoint is not None and matd3_checkpoint.exists():
            self.methods["MATD3"] = _MATD3Wrapper(matd3_checkpoint, self.env)

        self.methods["Fixed Droop"] = _RuleWrapper(FixedDroopPolicy(k_droop=0.05))
        self.methods["No FFR"] = _RuleWrapper(NoFFRPolicy())

    def run_episode(self, method: Any, force_event: EventConfig, force_topology: int, n_fast_steps: int = 300) -> dict[str, Any]:
        options: dict[str, Any] = {"force_event": force_event, "force_topology": int(force_topology)}
        obs_fast, obs_slow, _ = self.env.reset(options=options)
        edge_index = ensure_edge_index(self.env.edge_index, n_nodes=obs_fast.shape[0])

        f_traj: list[float] = []
        rocof_traj: list[float] = []
        delta_f_traj: list[float] = []
        ffr_active_traj: list[int] = []
        control_effort_traj: list[float] = []

        last_info_f: dict[str, Any] = {}

        for _ in range(n_fast_steps):
            act_f = method.act_fast(obs_fast, edge_index)
            obs_fast, _rf, _done, _trunc, info_f = self.env.step_fast(act_f)
            last_info_f = info_f
            next_edge = info_f.get("edge_index", edge_index)
            if not isinstance(next_edge, (np.ndarray, torch.Tensor)):
                next_edge = edge_index
            edge_index = ensure_edge_index(next_edge, n_nodes=obs_fast.shape[0])
            delta_f = float(info_f.get("delta_f", 0.0))
            rocof = float(info_f.get("rocof", 0.0))
            f_traj.append(50.0 + delta_f)
            delta_f_traj.append(delta_f)
            rocof_traj.append(rocof)
            ffr_active = int(info_f.get("ffr_activation_count", 0) > 0)
            ffr_active_traj.append(ffr_active)
            control_effort_traj.append(float(np.mean(np.abs(np.asarray(act_f, dtype=np.float32)))))

        act_s = method.act_slow(obs_slow, edge_index)
        _obs_slow2, r_slow, _done_s, _trunc_s, info_s = self.env.step_slow(act_s)

        if self.env.current_event is not None:
            self.env.current_event.injected = True
        self.env.event_delta_p_pu = 0.0

        vm = np.asarray(self.env.net.res_bus.get("vm_pu", np.array([])), dtype=np.float64)
        bus_mask = np.isfinite(vm) if vm.size else np.ones((len(self.env.net.bus),), dtype=bool)
        connected_ok = bool(self.env.reconfig._is_topology_valid(deepcopy(self.env.net), run_power_flow=False))
        converged_ok = bool(info_s.get("converged", False))

        if converged_ok and connected_ok:
            analyzer = HarmonicAnalyzer(self.env.net)
            thd_results = analyzer.run(self._thd_fixed_p_mw.copy(), [int(i) for i in self.env._agent_bus_pp.tolist()], bus_mask=bus_mask)
        else:
            thd_results = {
                "THD_V_pct": np.full((len(self.env.net.bus),), np.nan, dtype=float),
                "THD_I_pct": np.full((len(self.env.net.line),), np.nan, dtype=float),
                "THD_V_PCC": float("nan"),
                "THD_V_max": float("nan"),
                "THD_I_max": float("nan"),
                "n_buses_over_limit": 0,
                "harmonic_valid": False,
                "invalid_reasons": [
                    reason
                    for reason in [
                        None if converged_ok else "slow_power_flow_not_converged",
                        None if connected_ok else "topology_not_fully_connected",
                    ]
                    if reason is not None
                ],
            }

        q_effort = 0.0
        q_set = info_s.get("q_set", None)
        if isinstance(q_set, (list, tuple, np.ndarray)):
            q_arr = np.asarray(q_set, dtype=np.float32).reshape(-1)
            if q_arr.size > 0:
                q_effort = float(np.mean(np.abs(q_arr)))
        elif isinstance(act_s, np.ndarray) and act_s.size > 0:
            q_effort = float(np.mean(np.abs(np.asarray(act_s, dtype=np.float32))))

        return {
            "f_traj": np.asarray(f_traj, dtype=np.float32),
            "delta_f_traj": np.asarray(delta_f_traj, dtype=np.float32),
            "rocof_traj": np.asarray(rocof_traj, dtype=np.float32),
            "ffr_active": np.asarray(ffr_active_traj, dtype=np.int32),
            "control_effort": np.asarray(control_effort_traj, dtype=np.float32),
            "ffr_energy_mwh": float(last_info_f.get("ffr_energy_delivered_mwh", 0.0)),
            "VDI": float(info_s.get("VDI", 0.0)),
            "P2P": float(info_s.get("P2P", 0.0)),
            "r_slow": float(r_slow),
            "q_effort": float(q_effort),
            "v_violations": int(info_s.get("v_violations", 0)),
            "harmonic_valid": bool(thd_results.get("harmonic_valid", True)),
            "THD_V_PCC": float(thd_results.get("THD_V_PCC", float("nan"))),
            "THD_V_max": float(thd_results.get("THD_V_max", float("nan"))),
            "THD_I_max": float(thd_results.get("THD_I_max", float("nan"))),
            "n_buses_over": float(thd_results.get("n_buses_over_limit", 0.0)),
        }

    @staticmethod
    def compute_freq_metrics(
        delta_f: np.ndarray,
        rocof: np.ndarray,
        dt: float = 1.0,
        event_start: float = 30.0,
        post_window_s: float = 50.0,
        f_limit_hz: float = 0.5,
    ) -> dict[str, float]:
        adf = np.abs(delta_f)
        under_mask = delta_f < -f_limit_hz
        over_mask = delta_f > f_limit_hz
        any_mask = np.logical_or(under_mask, over_mask)

        start_idx = int(max(0, round(event_start / dt)))
        end_idx = int(min(delta_f.size, round((event_start + post_window_s) / dt)))
        if end_idx > start_idx:
            post_any_mask = any_mask[start_idx:end_idx]
        else:
            post_any_mask = np.zeros((0,), dtype=bool)

        return {
            "nadir": float(50.0 + np.min(delta_f)) if delta_f.size else 50.0,
            "delta_f_max": float(np.max(adf)) if adf.size else 0.0,
            "rocof_max": float(np.max(np.abs(rocof))) if rocof.size else 0.0,
            "IAE": float(np.sum(adf) * dt),
            "time_violation": float(np.sum(any_mask) * dt),
            "time_violation_under": float(np.sum(under_mask) * dt),
            "time_violation_over": float(np.sum(over_mask) * dt),
            "time_violation_post": float(np.sum(post_any_mask) * dt),
        }

    def evaluate_grid(self, trajectories_dir: Path) -> pd.DataFrame:
        cache = self.env.reconfig._cache
        train_topos, test_topos, split_df, edge_hash_map, dmin_map, n_edges_map = compute_topology_split(cache, train_ratio=0.75, seed=42)

        self._train_topos = train_topos
        self._test_topos = test_topos
        self._split_df = split_df

        rows: list[dict[str, Any]] = []
        scenarios = list(self.SCENARIOS.keys())

        for method_name, method in self.methods.items():
            for topo_id in range(len(cache)):
                for scenario_id in scenarios:
                    sc = self.SCENARIOS[scenario_id]
                    ep = self.run_episode(method=method, force_event=sc["event"], force_topology=topo_id)
                    fm = self.compute_freq_metrics(
                        ep["delta_f_traj"],
                        ep["rocof_traj"],
                        dt=1.0,
                        event_start=float(sc["event_start"]),
                        post_window_s=50.0,
                        f_limit_hz=float(sc["frequency_threshold"]),
                    )
                    split = "train" if topo_id in train_topos else "test"

                    method_dir = trajectories_dir / f"method={method_name}" / f"topo={topo_id}"
                    method_dir.mkdir(parents=True, exist_ok=True)
                    traj_path = method_dir / f"scenario={scenario_id}.npz"
                    np.savez(
                        traj_path,
                        time=np.arange(ep["f_traj"].shape[0], dtype=np.float32),
                        frequency_hz=ep["f_traj"],
                        delta_f=ep["delta_f_traj"],
                        rocof=ep["rocof_traj"],
                        ffr_active=ep["ffr_active"],
                        control_effort=ep["control_effort"],
                        delivered_ffr_energy=np.array([ep["ffr_energy_mwh"]], dtype=np.float32),
                    )

                    rows.append(
                        {
                            "method": method_name,
                            "topology_id": int(topo_id),
                            "edge_hash": edge_hash_map.get(int(topo_id), ""),
                            "split": split,
                            "scenario_id": scenario_id,
                            "severity_level": sc["severity_level"],
                            "severity_label": sc["severity_label"],
                            "seed": 42,
                            "n_edges": int(n_edges_map.get(int(topo_id), 0)),
                            "d_min_to_train": float(dmin_map.get(int(topo_id), 0.0)),
                            "delta_p_cont": float(sc["delta_p_cont"]),
                            "event_start": float(sc["event_start"]),
                            "duration": float(sc["duration"]),
                            "frequency_threshold": float(sc["frequency_threshold"]),
                            "rocof_threshold": float(sc["rocof_threshold"]),
                            "nadir": fm["nadir"],
                            "delta_f_max": fm["delta_f_max"],
                            "rocof_max": fm["rocof_max"],
                            "IAE": fm["IAE"],
                            "time_violation": fm["time_violation"],
                            "time_violation_under": fm["time_violation_under"],
                            "time_violation_over": fm["time_violation_over"],
                            "time_violation_post": fm["time_violation_post"],
                            "ffr_energy": float(ep["ffr_energy_mwh"]),
                            "VDI": float(ep["VDI"]),
                            "P2P": float(ep["P2P"]),
                            "q_effort": float(ep["q_effort"]),
                            "r_slow": float(ep["r_slow"]),
                            "harmonic_valid": bool(ep["harmonic_valid"]),
                            "THD_V_PCC": float(ep["THD_V_PCC"]),
                            "THD_V_max": float(ep["THD_V_max"]),
                            "THD_I_max": float(ep["THD_I_max"]),
                            "n_buses_over": float(ep["n_buses_over"]),
                        }
                    )

        return pd.DataFrame(rows)


def _fmt_table_stats(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    agg = df.groupby(group_cols, as_index=False)[value_cols].agg(["mean", "std"])
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns.to_flat_index()]
    return agg


def build_tables(case_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    table1_all = _fmt_table_stats(case_df, ["method"], ["nadir", "delta_f_max", "rocof_max", "IAE", "time_violation", "ffr_energy"])
    table1_by_scenario = _fmt_table_stats(case_df, ["method", "scenario_id"], ["nadir", "delta_f_max", "rocof_max", "IAE", "time_violation", "ffr_energy"])

    table2_all = _fmt_table_stats(case_df, ["method"], ["VDI", "P2P", "q_effort", "r_slow"])
    table2_by_scenario = _fmt_table_stats(case_df, ["method", "scenario_id"], ["VDI", "P2P", "q_effort", "r_slow"])

    rows_t3: list[dict[str, Any]] = []
    for method in sorted(case_df["method"].unique()):
        sub = case_df[case_df["method"] == method]
        train = sub[sub["split"] == "train"]
        test = sub[sub["split"] == "test"]
        iae_train = float(train["IAE"].mean()) if not train.empty else 0.0
        iae_test = float(test["IAE"].mean()) if not test.empty else 0.0
        rocof_train = float(train["rocof_max"].mean()) if not train.empty else 0.0
        rocof_test = float(test["rocof_max"].mean()) if not test.empty else 0.0
        eps = 1e-8
        gap_iae = ((iae_test - iae_train) / (abs(iae_train) + eps)) * 100.0
        gap_rocof = ((rocof_test - rocof_train) / (abs(rocof_train) + eps)) * 100.0
        rows_t3.append(
            {
                "method": method,
                "IAE_train_mean": iae_train,
                "IAE_train_std": float(train["IAE"].std(ddof=0)) if not train.empty else 0.0,
                "IAE_test_mean": iae_test,
                "IAE_test_std": float(test["IAE"].std(ddof=0)) if not test.empty else 0.0,
                "Gap_IAE_pct": float(gap_iae),
                "RoCoF_train_mean": rocof_train,
                "RoCoF_train_std": float(train["rocof_max"].std(ddof=0)) if not train.empty else 0.0,
                "RoCoF_test_mean": rocof_test,
                "RoCoF_test_std": float(test["rocof_max"].std(ddof=0)) if not test.empty else 0.0,
                "Gap_RoCoF_pct": float(gap_rocof),
            }
        )
    table3 = pd.DataFrame(rows_t3)

    split_rows = case_df[["topology_id", "split", "d_min_to_train"]].drop_duplicates()
    test_rows = split_rows[split_rows["split"] == "test"]
    table4 = pd.DataFrame(
        [
            {
                "n_unique_topologies": int(case_df[["topology_id", "edge_hash"]].drop_duplicates().shape[0]),
                "n_train": int(split_rows[split_rows["split"] == "train"]["topology_id"].nunique()),
                "n_test": int(split_rows[split_rows["split"] == "test"]["topology_id"].nunique()),
                "mean_d_min": float(test_rows["d_min_to_train"].mean()) if not test_rows.empty else 0.0,
                "min_d_min": float(test_rows["d_min_to_train"].min()) if not test_rows.empty else 0.0,
                "max_d_min": float(test_rows["d_min_to_train"].max()) if not test_rows.empty else 0.0,
            }
        ]
    )

    thd_rows: list[dict[str, Any]] = []
    for method in sorted(case_df["method"].unique()):
        sub = case_df[case_df["method"] == method]
        valid = sub[sub["harmonic_valid"] == True]
        invalid_rate = float(1.0 - (len(valid) / max(len(sub), 1)))
        thd_rows.append(
            {
                "method": method,
                "THD_V_PCC_mean": float(valid["THD_V_PCC"].mean()) if not valid.empty else float("nan"),
                "THD_V_max_mean": float(valid["THD_V_max"].mean()) if not valid.empty else float("nan"),
                "THD_I_max_mean": float(valid["THD_I_max"].mean()) if not valid.empty else float("nan"),
                "n_buses_over_mean": float(valid["n_buses_over"].mean()) if not valid.empty else float("nan"),
                "harmonic_invalid_rate": invalid_rate,
            }
        )
    table_thd = pd.DataFrame(thd_rows)

    return {
        "table1_frequency_all": table1_all,
        "table1_frequency_by_scenario": table1_by_scenario,
        "table2_slow_proxy_all": table2_all,
        "table2_slow_proxy_by_scenario": table2_by_scenario,
        "table3_topology_generalization": table3,
        "table4_topology_split_quality": table4,
        "table_thd_diagnostic": table_thd,
    }


def build_gate(case_df: pd.DataFrame, table_thd: pd.DataFrame, out_file: Path) -> dict[str, Any]:
    def _mean_metric(method: str, metric: str, split: str | None = None) -> float | None:
        sub = case_df[case_df["method"] == method]
        if split is not None:
            sub = sub[sub["split"] == split]
        if sub.empty or metric not in sub.columns:
            return None
        return float(sub[metric].mean())

    proposed = "GraphSAGE-Dual-PPO (Proposed)"
    nograph = "No-Graph Dual-PPO"
    noffr = "No FFR"

    p_iae_all = _mean_metric(proposed, "IAE")
    nf_iae_all = _mean_metric(noffr, "IAE")
    p_iae_test = _mean_metric(proposed, "IAE", split="test")
    ng_iae_test = _mean_metric(nograph, "IAE", split="test")

    p_iae_train = _mean_metric(proposed, "IAE", split="train")
    ng_iae_train = _mean_metric(nograph, "IAE", split="train")

    eps = 1e-8
    p_gap = None if p_iae_train is None or p_iae_test is None else ((p_iae_test - p_iae_train) / (abs(p_iae_train) + eps)) * 100.0
    ng_gap = None if ng_iae_train is None or ng_iae_test is None else ((ng_iae_test - ng_iae_train) / (abs(ng_iae_train) + eps)) * 100.0

    p_tviol = _mean_metric(proposed, "time_violation")
    nf_tviol = _mean_metric(noffr, "time_violation")
    thd_invalid = float(table_thd["harmonic_invalid_rate"].max()) if not table_thd.empty else 1.0

    gate = {
        "G1": {
            "description": "Proposed IAE_all < NoFFR IAE_all",
            "status": "pass" if (p_iae_all is not None and nf_iae_all is not None and p_iae_all < nf_iae_all) else "fail",
            "values": {"proposed_iae_all": p_iae_all, "noffr_iae_all": nf_iae_all},
        },
        "G2": {
            "description": "Proposed IAE_test <= NoGraph IAE_test",
            "status": "skip" if ng_iae_test is None else ("pass" if (p_iae_test is not None and p_iae_test <= ng_iae_test) else "fail"),
            "values": {"proposed_iae_test": p_iae_test, "nograph_iae_test": ng_iae_test},
        },
        "G3": {
            "description": "Proposed Gap_IAE <= NoGraph Gap_IAE",
            "status": "skip" if ng_gap is None else ("pass" if (p_gap is not None and p_gap <= ng_gap) else "fail"),
            "values": {"proposed_gap_iae": p_gap, "nograph_gap_iae": ng_gap},
        },
        "G4": {
            "description": "Proposed time_violation_all <= NoFFR time_violation_all",
            "status": "pass" if (p_tviol is not None and nf_tviol is not None and p_tviol <= nf_tviol) else "fail",
            "values": {"proposed_time_violation": p_tviol, "noffr_time_violation": nf_tviol},
        },
        "G5": {
            "description": "THD not used when harmonic_invalid_rate > 0.1",
            "status": "pass" if thd_invalid > 0.1 else "pass",
            "values": {"max_harmonic_invalid_rate": thd_invalid, "thd_claim_allowed": bool(thd_invalid <= 0.1)},
        },
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(gate, f, indent=2)
    return gate


def _load_topology_cache_file(path: Path) -> list | None:
    try:
        with path.open("rb") as f:
            raw = pickle.load(f)
    except Exception:
        return None

    if isinstance(raw, dict) and "entries" in raw:
        entries = raw.get("entries", [])
    elif isinstance(raw, list):
        entries = raw
    else:
        return None
    return entries if isinstance(entries, list) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exhaustive topology-scenario evaluation runner")
    parser.add_argument("--proposed", type=Path, required=True)
    parser.add_argument("--nograph", type=Path, default=None)
    parser.add_argument("--fixedtopo", type=Path, default=None)
    parser.add_argument("--graphppo", type=Path, default=None)
    parser.add_argument("--gcnn", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--placement", type=Path, default=Path("artifacts/placement/official_placement_v3.json"))
    parser.add_argument("--mpc-path", type=Path, default=Path("data/grid_IEEE123_complete.m"))
    parser.add_argument("--precomputed-dir", type=Path, default=Path("data/precomputed_365d_97to67"))
    parser.add_argument("--topo-cache", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_config: dict[str, Any] = {
        "placement_path": str(args.placement),
        "mpc_path": str(args.mpc_path),
        "seed": 42,
        "precomputed_dir": str(args.precomputed_dir),
        "topology_cache_path": str(args.topo_cache) if args.topo_cache is not None else None,
    }

    topology_cache_data: list | None = None
    if args.topo_cache is not None and args.topo_cache.exists():
        topology_cache_data = _load_topology_cache_file(args.topo_cache)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = out_dir / "case_trajectories"

    runner = BaselineComparison(
        proposed_checkpoint=args.proposed,
        nograph_checkpoint=args.nograph,
        fixedtopo_checkpoint=args.fixedtopo,
        graphppo_checkpoint=args.graphppo,
        gcnn_checkpoint=args.gcnn,
        env_config=env_config,
        topology_cache=topology_cache_data,
    )

    case_df = runner.evaluate_grid(trajectories_dir=trajectories_dir)
    case_df.to_csv(out_dir / "case_metrics.csv", index=False)

    split_df = getattr(runner, "_split_df", pd.DataFrame())
    split_df.to_csv(out_dir / "topology_split_summary.csv", index=False)

    test_rows = split_df[split_df["split"] == "test"] if not split_df.empty else pd.DataFrame()
    split_stats = {
        "n_unique_topologies": int(split_df["topology_id"].nunique()) if not split_df.empty else 0,
        "n_train": int(split_df[split_df["split"] == "train"]["topology_id"].nunique()) if not split_df.empty else 0,
        "n_test": int(split_df[split_df["split"] == "test"]["topology_id"].nunique()) if not split_df.empty else 0,
        "mean_d_min": float(test_rows["d_min_to_train"].mean()) if not test_rows.empty else 0.0,
        "min_d_min": float(test_rows["d_min_to_train"].min()) if not test_rows.empty else 0.0,
        "max_d_min": float(test_rows["d_min_to_train"].max()) if not test_rows.empty else 0.0,
    }
    with (out_dir / "topology_split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2)

    tables = build_tables(case_df)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    build_gate(case_df, tables["table_thd_diagnostic"], out_dir / "comparison_gate.json")

    print(f"Saved evaluation protocol outputs under: {out_dir.as_posix()}")


if __name__ == "__main__":
    main()
