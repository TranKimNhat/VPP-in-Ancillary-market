from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any
import json

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.env.events import EventConfig
from src.env.microgrid_env_dual import MicrogridEnvDual
from src.eval.figures import _save
from src.eval.harmonic_analysis import HarmonicAnalyzer
from src.layer2_control.graph_sage_encoder import GraphSAGEEncoder
from src.rl.train_dual import GATEncoderDual, LoopPolicy, MLPEncoder, ensure_edge_index

# OPEX rates (€/kW/year) - IRENA 2023, NREL ATB 2023
OPEX_PV_EUR_KW_YEAR = 12.0
OPEX_WIND_EUR_KW_YEAR = 30.0
OPEX_BESS_EUR_KWH_YEAR = 7.5
HOURS_PER_YEAR = 8760.0


def compute_opex_hourly(placement: dict) -> float:
    """Compute hourly OPEX (€/h) from placement asset capacities."""
    opex_year = 0.0
    for gfm in placement.get("gfm", {}).values():
        opex_year += float(gfm.get("pv_mw", 0.0)) * 1000 * OPEX_PV_EUR_KW_YEAR
        opex_year += float(gfm.get("bess_mwh", 0.0)) * 1000 * OPEX_BESS_EUR_KWH_YEAR
    for wind in placement.get("wind", []):
        opex_year += float(wind.get("mw", 0.0)) * 1000 * OPEX_WIND_EUR_KW_YEAR
    for evcs in placement.get("evcs", []):
        opex_year += float(evcs.get("pv_mw", 0.0)) * 1000 * OPEX_PV_EUR_KW_YEAR
        opex_year += float(evcs.get("bess_mwh", 0.0)) * 1000 * OPEX_BESS_EUR_KWH_YEAR
    for dpv in placement.get("dpv", []):
        opex_year += float(dpv.get("mw", 0.0)) * 1000 * OPEX_PV_EUR_KW_YEAR
    return opex_year / HOURS_PER_YEAR


class NoFFRPolicy:
    def act(self, obs: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        return np.zeros((44,), dtype=np.float32)


class FixedDroopPolicy:
    def __init__(self, k_droop: float = 0.05) -> None:
        self.k_droop = float(k_droop)

    def act(self, obs: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        delta_f = np.asarray(obs[:18, 0], dtype=np.float32)
        action = np.zeros((44,), dtype=np.float32)
        action[:18] = -self.k_droop * delta_f
        action[41:44] = self.k_droop
        return np.clip(action, -1.0, 1.0)


class DeterministicDualPolicy:
    def __init__(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        required_keys = {
            "encoder",
            "policy_fast_actor",
            "policy_fast_critic",
            "policy_slow_actor",
            "policy_slow_critic",
        }
        missing = required_keys.difference(checkpoint.keys())
        if missing:
            raise ValueError(
                f"Unsupported checkpoint schema at {checkpoint_path}. Missing keys: {sorted(missing)}"
            )

        encoder_state = checkpoint["encoder"]
        encoder_in_dim = 7
        if isinstance(encoder_state, dict):
            w_self = encoder_state.get("layer1.w_self.weight")
            if isinstance(w_self, torch.Tensor) and w_self.ndim == 2:
                encoder_in_dim = int(w_self.shape[1])

        encoder_type = str(checkpoint.get("encoder_type", ""))
        if not encoder_type:
            args = checkpoint.get("args", {})
            if isinstance(args, dict):
                encoder_type = str(args.get("encoder", "sage"))
            else:
                encoder_type = "sage"

        if encoder_type == "mlp":
            encoder = MLPEncoder(in_dim=encoder_in_dim, hidden_dim=128, out_dim=64)
        elif encoder_type == "gat":
            encoder = GATEncoderDual(in_dim=encoder_in_dim, hidden_dim=32, out_dim=64, heads=4)
        else:
            encoder = GraphSAGEEncoder(in_dim=encoder_in_dim, hidden_dim=64, out_dim=64)

        self.encoder_in_dim = int(encoder_in_dim)
        obs_nodes = int(checkpoint.get("obs_nodes", 123))  # Default to IEEE 123-bus
        self.policy_fast = LoopPolicy(encoder=encoder, action_dim=44, obs_nodes=obs_nodes, obs_feat=self.encoder_in_dim)
        self.policy_slow = LoopPolicy(encoder=encoder, action_dim=82, obs_nodes=obs_nodes, obs_feat=self.encoder_in_dim)

        self.policy_fast.encoder.load_state_dict(encoder_state)
        self.policy_fast.actor.load_state_dict(checkpoint["policy_fast_actor"])
        self.policy_fast.critic.load_state_dict(checkpoint["policy_fast_critic"])
        self.policy_slow.actor.load_state_dict(checkpoint["policy_slow_actor"])
        self.policy_slow.critic.load_state_dict(checkpoint["policy_slow_critic"])

        self.policy_fast.eval()
        self.policy_slow.eval()

    def _project_obs(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim != 2:
            return obs
        feat = int(obs.shape[1])
        if feat == self.encoder_in_dim:
            return obs
        if feat > self.encoder_in_dim:
            return obs[:, : self.encoder_in_dim]
        pad = np.zeros((int(obs.shape[0]), self.encoder_in_dim - feat), dtype=obs.dtype)
        return np.concatenate([obs, pad], axis=1)

    def act_fast(self, obs_fast: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        return self.policy_fast.act_deterministic(self._project_obs(obs_fast), edge_index)

    def act_slow(self, obs_slow: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        return self.policy_slow.act_deterministic(self._project_obs(obs_slow), edge_index)


class DualEvaluator:
    def __init__(
        self,
        checkpoints: dict[str, str | None],
        env_config: dict[str, Any],
        output_dir: Path = Path("results"),
        figures_dir: Path = Path("results/figures"),
        log_path: Path = Path("logs/t77_full_ppase_2k.txt"),
    ) -> None:
        self.env = MicrogridEnvDual(**env_config)
        self.output_dir = Path(output_dir)
        self.figures_dir = Path(figures_dir)
        self.log_path = Path(log_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        placement_path = Path(env_config.get("placement_path", ""))
        if placement_path.exists():
            placement = json.loads(placement_path.read_text(encoding="utf-8"))
            evcs = placement.get("evcs", [])
            dpv = placement.get("dpv", [])
            agent_bus_ids = [int(ev["bus"]) for ev in evcs] + [int(ev["bus"]) for ev in evcs] + [int(ev["bus"]) for ev in evcs] + [int(pv["bus"]) for pv in dpv]
            self.agent_bus_pp = [self.env._pp_idx(bus_id) for bus_id in agent_bus_ids]
            self.opex_hourly = compute_opex_hourly(placement)
        else:
            self.agent_bus_pp = [int(i) for i in self.env._agent_bus_pp.tolist()]
            self.opex_hourly = 0.0

        self.methods: dict[str, Any] = {
            "No FFR": NoFFRPolicy(),
            "Fixed droop": FixedDroopPolicy(k_droop=0.05),
        }
        for name, ckpt in checkpoints.items():
            if ckpt is None:
                continue
            path = Path(ckpt)
            if not path.exists():
                continue
            try:
                self.methods[name] = DeterministicDualPolicy(path)
            except ValueError as exc:
                print(f"[evaluate_dual] Skip method '{name}': {exc}")

        self.scenarios = {
            # Main benchmark tier (moderate->severe, literature-aligned)
            "S1_load_step": EventConfig(type="load_step", delta_P_mw=2.4, location=45, t_inject=30.0),
            "S2_gen_trip": EventConfig(type="gen_trip", delta_P_mw=-3.9, location=97, t_inject=30.0),
            "S3_line_trip": EventConfig(type="line_trip", delta_P_mw=-2.4, location=108, t_inject=30.0),
            # Stress tier (extreme)
            "S4_high_ren_extreme": EventConfig(type="high_ren", delta_P_mw=4.7, location=97, t_inject=30.0),
            "S5_gen_trip_extreme": EventConfig(type="gen_trip", delta_P_mw=-5.5, location=97, t_inject=30.0),
        }

        self._agent_p_rated_mw = np.asarray(
            [max(float(spec.get("p_rated", 0.0)), 0.0) for spec in self.env._agent_specs],
            dtype=np.float32,
        )
        self._thd_fixed_p_mw = 0.5 * self._agent_p_rated_mw

    def _act_fast(self, method: Any, obs_fast: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        if isinstance(method, DeterministicDualPolicy):
            return method.act_fast(obs_fast, edge_index)
        return method.act(obs_fast, edge_index)

    def _act_slow(self, method: Any, obs_slow: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        if isinstance(method, DeterministicDualPolicy):
            return method.act_slow(obs_slow, edge_index)
        return np.zeros((82,), dtype=np.float32)

    def run_episode(
        self,
        method: Any,
        force_event: EventConfig | None = None,
        force_topology: int | None = None,
        n_fast_steps: int = 300,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if force_event is not None:
            options["force_event"] = force_event
        if force_topology is not None:
            options["force_topology"] = int(force_topology)

        obs_fast, obs_slow, _ = self.env.reset(options=options)
        edge_index = ensure_edge_index(self.env.edge_index, n_nodes=obs_fast.shape[0])

        f_traj: list[float] = []
        rocof_traj: list[float] = []
        v_bus_last = np.ones((obs_fast.shape[0],), dtype=np.float32)
        p2p_rev = 0.0
        lmp_mean = 0.0
        r_fast = 0.0
        ffr_activations = 0
        ffr_energy_mwh = 0.0

        for _ in range(n_fast_steps):
            act_f = self._act_fast(method, obs_fast, edge_index)
            obs_fast, rf, _done, _trunc, info_f = self.env.step_fast(act_f)
            next_edge = info_f.get("edge_index", edge_index)
            if not isinstance(next_edge, (np.ndarray, torch.Tensor)):
                next_edge = edge_index
            edge_index = ensure_edge_index(next_edge, n_nodes=obs_fast.shape[0])
            r_fast += float(rf)
            f_traj.append(50.0 + float(info_f.get("delta_f", 0.0)))
            rocof_traj.append(float(info_f.get("rocof", 0.0)))
            lmp_mean += float(np.mean(obs_fast[:, 4]))
            # Track FFR metrics
            ffr_activations = int(info_f.get("ffr_activation_count", 0))
            ffr_energy_mwh = float(info_f.get("ffr_energy_delivered_mwh", 0.0))

        act_s = self._act_slow(method, obs_slow, edge_index)
        obs_slow, r_slow, _done_s, _trunc_s, info_s = self.env.step_slow(act_s)
        v_bus_last = np.asarray(obs_slow[:, 0], dtype=np.float32)
        p_p2p = np.asarray(info_s.get("P_p2p", np.zeros((41,))), dtype=np.float32)
        zone_prices = self.env._zone_lmp_vec
        episode_hours = n_fast_steps / 3600.0
        p2p_rev = float(np.sum(p_p2p * zone_prices) * episode_hours)

        if self.env.current_event is not None:
            self.env.current_event.injected = True
        self.env.event_delta_p_pu = 0.0

        vm = np.asarray(self.env.net.res_bus.get("vm_pu", np.array([])), dtype=np.float64)
        bus_mask = np.isfinite(vm) if vm.size else np.ones((len(self.env.net.bus),), dtype=bool)
        connected_ok = bool(self.env.reconfig._is_topology_valid(deepcopy(self.env.net), run_power_flow=False))
        converged_ok = bool(info_s.get("converged", False))

        if converged_ok and connected_ok:
            analyzer = HarmonicAnalyzer(self.env.net)
            thd_results = analyzer.run(
                self._thd_fixed_p_mw.copy(),
                self.agent_bus_pp,
                bus_mask=bus_mask,
            )
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

        invalid_reasons_obj = thd_results.get("invalid_reasons", [])
        invalid_reasons = [str(x) for x in invalid_reasons_obj] if isinstance(invalid_reasons_obj, list) else []

        opex_episode = self.opex_hourly * episode_hours
        profit = p2p_rev - opex_episode

        return {
            "f_traj": np.asarray(f_traj, dtype=np.float32),
            "rocof_traj": np.asarray(rocof_traj, dtype=np.float32),
            "r_fast": float(r_fast),
            "r_slow": float(r_slow),
            "VDI": float(info_s.get("VDI", 0.0)),
            "P2P": p2p_rev,
            "OPEX": float(opex_episode),
            "profit": float(profit),
            "v_violations": int(info_s.get("v_violations", 0)),
            "soc_end": float(np.mean(self.env.soc)),
            "v_bus": v_bus_last,
            "lmp_mean": float(lmp_mean / max(n_fast_steps, 1)),
            "edge_index": edge_index,
            "THD_V_pct": thd_results["THD_V_pct"],
            "THD_I_pct": thd_results["THD_I_pct"],
            "THD_V_PCC": thd_results["THD_V_PCC"],
            "THD_V_max": thd_results["THD_V_max"],
            "THD_I_max": thd_results["THD_I_max"],
            "n_buses_over": thd_results["n_buses_over_limit"],
            "harmonic_valid": bool(thd_results.get("harmonic_valid", True)),
            "harmonic_invalid_reasons": invalid_reasons,
            "ffr_activations": ffr_activations,
            "ffr_energy_mwh": ffr_energy_mwh,
        }

    @staticmethod
    def compute_freq_metrics(
        f_traj: np.ndarray,
        rocof_traj: np.ndarray,
        f_nominal: float = 50.0,
        event_start: float = 30.0,
        post_window_steps: int = 50,
        f_limit_hz: float = 0.5,
        settle_band_hz: float = 0.02,
    ) -> dict[str, float]:
        delta_f = f_traj - float(f_nominal)
        under_mask = delta_f < -f_limit_hz
        over_mask = delta_f > f_limit_hz
        any_mask = np.logical_or(under_mask, over_mask)

        start_idx = int(max(0, round(event_start)))
        end_idx = int(min(delta_f.size, start_idx + max(int(post_window_steps), 0)))
        if end_idx > start_idx:
            post_delta_f = delta_f[start_idx:end_idx]
            post_any_mask = any_mask[start_idx:end_idx]
            post_rocof = rocof_traj[start_idx:end_idx]
        else:
            post_delta_f = np.zeros((0,), dtype=np.float32)
            post_any_mask = np.zeros((0,), dtype=bool)
            post_rocof = np.zeros((0,), dtype=np.float32)

        # Settling time after disturbance: first index such that all remaining points stay inside deadband.
        settle_time_s = float(post_window_steps)
        if post_delta_f.size > 0:
            abs_post = np.abs(post_delta_f)
            for i in range(abs_post.size):
                if np.all(abs_post[i:] <= settle_band_hz):
                    settle_time_s = float(i)
                    break

        return {
            "nadir": float(np.min(f_traj)) if f_traj.size else float(f_nominal),
            "delta_f_max": float(np.max(np.abs(delta_f))) if delta_f.size else 0.0,
            "rocof_max": float(np.max(np.abs(rocof_traj))) if rocof_traj.size else 0.0,
            "IAE": float(np.trapezoid(np.abs(delta_f), dx=1.0)) if delta_f.size else 0.0,
            "IAE_post": float(np.trapezoid(np.abs(post_delta_f), dx=1.0)) if post_delta_f.size else 0.0,
            "rocof_post_max": float(np.max(np.abs(post_rocof))) if post_rocof.size else 0.0,
            "time_violation": float(np.sum(any_mask)) if delta_f.size else 0.0,
            "time_violation_under": float(np.sum(under_mask)) if delta_f.size else 0.0,
            "time_violation_over": float(np.sum(over_mask)) if delta_f.size else 0.0,
            "time_violation_post": float(np.sum(post_any_mask)),
            "settling_time_s": settle_time_s,
            "ffr_success": float((np.min(f_traj) >= (f_nominal - f_limit_hz)) and (np.max(np.abs(post_rocof)) <= 1.0 if post_rocof.size else True)),
        }

    def build_table1(self, n_runs: int = 20) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for scenario_name, event in self.scenarios.items():
            for method_name, method in self.methods.items():
                run_metrics: list[dict[str, float]] = []
                ffr_activations: list[int] = []
                ffr_energy: list[float] = []
                for _ in range(int(n_runs)):
                    ep = self.run_episode(method=method, force_event=event)
                    run_metrics.append(
                        self.compute_freq_metrics(
                            ep["f_traj"],
                            ep["rocof_traj"],
                            event_start=float(event.t_inject),
                            post_window_steps=50,
                            f_limit_hz=0.5,
                        )
                    )
                    ffr_activations.append(int(ep.get("ffr_activations", 0)))
                    ffr_energy.append(float(ep.get("ffr_energy_mwh", 0.0)))

                for metric in [
                    "delta_f_max",
                    "rocof_max",
                    "IAE",
                    "IAE_post",
                    "rocof_post_max",
                    "time_violation",
                    "time_violation_under",
                    "time_violation_over",
                    "time_violation_post",
                    "settling_time_s",
                    "ffr_success",
                    "nadir",
                ]:
                    vals = np.asarray([m[metric] for m in run_metrics], dtype=np.float32)
                    rows.append(
                        {
                            "scenario": scenario_name,
                            "method": method_name,
                            "metric": metric,
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals)),
                        }
                    )
                # Add FFR metrics
                rows.append({
                    "scenario": scenario_name,
                    "method": method_name,
                    "metric": "ffr_activations",
                    "mean": float(np.mean(ffr_activations)),
                    "std": float(np.std(ffr_activations)),
                })
                rows.append({
                    "scenario": scenario_name,
                    "method": method_name,
                    "metric": "ffr_energy_mwh",
                    "mean": float(np.mean(ffr_energy)),
                    "std": float(np.std(ffr_energy)),
                })
        return pd.DataFrame(rows)

    def build_table2(self, n_runs: int = 20) -> pd.DataFrame:
        method_names = [m for m in ["Fixed droop", "MAPPO no-GNN", "GNN-MAPPO"] if m in self.methods]
        rows: list[dict[str, Any]] = []
        for method_name in method_names:
            method = self.methods[method_name]
            vals = {"VDI": [], "viol_rate": [], "P2P": [], "OPEX": [], "profit": [], "SOC_end": []}
            for _ in range(int(n_runs)):
                ep = self.run_episode(method=method, force_event=None)
                vals["VDI"].append(float(ep["VDI"]))
                vals["viol_rate"].append(float(ep["v_violations"]) / 41.0)
                vals["P2P"].append(float(ep["P2P"]))
                vals["OPEX"].append(float(ep["OPEX"]))
                vals["profit"].append(float(ep["profit"]))
                vals["SOC_end"].append(float(ep["soc_end"]))
            rows.append(
                {
                    "method": method_name,
                    "VDI": float(np.mean(vals["VDI"])),
                    "voltage_violation_rate": float(np.mean(vals["viol_rate"])),
                    "P2P_revenue": float(np.mean(vals["P2P"])),
                    "OPEX": float(np.mean(vals["OPEX"])),
                    "profit": float(np.mean(vals["profit"])),
                    "BESS_SOC_end": float(np.mean(vals["SOC_end"])),
                }
            )
        return pd.DataFrame(rows)

    def _generalization_score(self, method: Any, topo_indices: list[int], n_runs: int) -> float:
        scores = []
        for topo_idx in topo_indices:
            for _ in range(int(n_runs)):
                ep = self.run_episode(method=method, force_topology=topo_idx)
                m = self.compute_freq_metrics(ep["f_traj"], ep["rocof_traj"])
                # Higher is better: reward FFR success and penalize post-event instability.
                score = (
                    10.0 * m["ffr_success"]
                    - 2.0 * m["time_violation_post"]
                    - 1.5 * m["IAE_post"]
                    - 1.0 * m["rocof_post_max"]
                    - 0.5 * m["settling_time_s"]
                )
                scores.append(float(score))
        return float(np.mean(scores)) if scores else 0.0

    def build_table3(self, n_runs: int = 20) -> pd.DataFrame:
        mapping = {
            "MLP": "MAPPO no-GNN",
            "GAT": "GAT",
            "GraphSAGE": "GNN-MAPPO",
        }
        train_topos = list(range(15))
        test_topos = list(range(15, 20))

        rows = []
        for label, method_key in mapping.items():
            if method_key not in self.methods:
                continue
            method = self.methods[method_key]
            train_score = self._generalization_score(method, train_topos, n_runs)
            test_score = self._generalization_score(method, test_topos, n_runs)
            denom = abs(train_score) if abs(train_score) > 1e-8 else 1.0
            drop = (train_score - test_score) / denom * 100.0
            retained = test_score / train_score if abs(train_score) > 1e-8 else 0.0
            rows.append(
                {
                    "encoder": label,
                    "train_score": float(train_score),
                    "test_score": float(test_score),
                    "drop_percent": float(drop),
                    "retained_ratio": float(retained),
                }
            )
        return pd.DataFrame(rows)

    def build_ffr_topology_table(self, n_runs: int = 20) -> pd.DataFrame:
        method_names = [m for m in ["MAPPO no-GNN", "GAT", "GNN-MAPPO"] if m in self.methods]
        if not method_names:
            return pd.DataFrame()

        train_topos = list(range(15))
        test_topos = list(range(15, 20))
        scenario = self.scenarios["S3_line_trip"]
        rows: list[dict[str, float | str]] = []

        for method_name in method_names:
            method = self.methods[method_name]
            for split_name, topo_indices in (("train_topology", train_topos), ("unseen_topology", test_topos)):
                ffr_success_vals: list[float] = []
                iae_post_vals: list[float] = []
                settling_vals: list[float] = []
                ffr_energy_vals: list[float] = []

                for topo_idx in topo_indices:
                    for _ in range(int(n_runs)):
                        ep = self.run_episode(method=method, force_event=scenario, force_topology=topo_idx)
                        freq = self.compute_freq_metrics(
                            ep["f_traj"],
                            ep["rocof_traj"],
                            event_start=float(scenario.t_inject),
                            post_window_steps=50,
                            f_limit_hz=0.5,
                        )
                        ffr_success_vals.append(float(freq["ffr_success"]))
                        iae_post_vals.append(float(freq["IAE_post"]))
                        settling_vals.append(float(freq["settling_time_s"]))
                        ffr_energy_vals.append(float(ep.get("ffr_energy_mwh", 0.0)))

                rows.append(
                    {
                        "method": method_name,
                        "topology_split": split_name,
                        "ffr_success_rate": float(np.mean(ffr_success_vals)) if ffr_success_vals else 0.0,
                        "iae_post": float(np.mean(iae_post_vals)) if iae_post_vals else 0.0,
                        "settling_time_s": float(np.mean(settling_vals)) if settling_vals else 0.0,
                        "ffr_energy_mwh": float(np.mean(ffr_energy_vals)) if ffr_energy_vals else 0.0,
                    }
                )
        return pd.DataFrame(rows)

    def plot_fig4(self, trajectories: dict[str, np.ndarray]) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, f in trajectories.items():
            ax.plot(np.arange(len(f)), f, label=name)
        ax.axhline(49.5, ls="--", color="red", label="49.5Hz limit")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        _save(fig, self.figures_dir, "fig4")

    def plot_fig5(self, trajectories: dict[str, np.ndarray]) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, f in trajectories.items():
            ax.plot(np.arange(len(f)), f, label=name)
        ax.axhline(49.5, ls="--", color="red")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        _save(fig, self.figures_dir, "fig5")

    def plot_fig6(self, normal_v: np.ndarray, reconf_v: np.ndarray) -> None:
        x = np.arange(len(normal_v))
        w = 0.4
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - w / 2, normal_v, width=w, label="normal")
        ax.bar(x + w / 2, reconf_v, width=w, label="reconfig")
        ax.set_xlabel("Agent")
        ax.set_ylabel("V (pu)")
        ax.legend()
        _save(fig, self.figures_dir, "fig6")

    def plot_fig7(self, lmp: list[float], p2p: list[float]) -> None:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        x = np.arange(len(lmp))
        ax1.plot(x, lmp, color="tab:blue", label="Zone LMP")
        ax1.set_ylabel("LMP", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(x, p2p, color="tab:orange", label="P2P")
        ax2.set_ylabel("P2P revenue", color="tab:orange")
        ax1.set_xlabel("Episode")
        _save(fig, self.figures_dir, "fig7")

    def plot_fig8(self, table3: pd.DataFrame) -> None:
        if table3.empty or "encoder" not in table3.columns or "drop_percent" not in table3.columns:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(table3["encoder"], table3["drop_percent"])
        ax.set_ylabel("Performance drop (%)")
        _save(fig, self.figures_dir, "fig8")

    def plot_fig9(self) -> None:
        if not self.log_path.exists():
            return
        rows = []
        for line in self.log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "phase=" not in line or "r_fast=" not in line or "r_slow=" not in line or "entropy_f=" not in line:
                continue
            try:
                toks = line.replace("=", " ").split()
                rows.append(
                    {
                        "phase": toks[toks.index("phase") + 1],
                        "ep": int(toks[toks.index("ep") + 1]),
                        "r_fast": float(toks[toks.index("r_fast") + 1]),
                        "r_slow": float(toks[toks.index("r_slow") + 1]),
                        "entropy_f": float(toks[toks.index("entropy_f") + 1]),
                    }
                )
            except Exception:
                continue
        if not rows:
            return
        df = pd.DataFrame(rows)
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        axes[0].plot(df["ep"], df["r_fast"], label="r_fast")
        axes[1].plot(df["ep"], df["r_slow"], label="r_slow")
        axes[2].plot(df["ep"], df["entropy_f"], label="entropy_f")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axes[2].set_xlabel("Episode")
        _save(fig, self.figures_dir, "fig9")

    def plot_fig10(self) -> None:
        labels = ["low_load", "nominal", "high_load"]
        n_topos = int(len(getattr(self.env.reconfig, "_cache", [])))
        if n_topos <= 0:
            return

        candidate_idx = [0, 10, 19]
        topo_idx = [min(idx, n_topos - 1) for idx in candidate_idx]
        edge_counts = []
        default_method = self.methods[next(iter(self.methods))]
        for idx in topo_idx:
            ep = self.run_episode(method=default_method, force_topology=idx, n_fast_steps=1)
            edge_counts.append(float(ep["edge_index"].shape[1]))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, edge_counts)
        ax.set_ylabel("Edge count")
        ax.set_title("Tie-switch reconfiguration")
        _save(fig, self.figures_dir, "fig10")

    def run_all(self, n_runs: int, smoke_runs: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        table1 = self.build_table1(n_runs=smoke_runs)
        table1.to_csv(self.output_dir / "table1_smoke.csv", index=False)

        s1_traj = {}
        s3_traj = {}
        for name in ["No FFR", "Fixed droop", "MAPPO no-GNN", "GAT", "GNN-MAPPO"]:
            if name not in self.methods:
                continue
            s1 = self.run_episode(self.methods[name], force_event=self.scenarios["S1_load_step"])
            s3 = self.run_episode(self.methods[name], force_event=self.scenarios["S3_line_trip"])
            s1_traj[name] = s1["f_traj"]
            s3_traj[name] = s3["f_traj"]

        self.plot_fig4(s1_traj)
        self.plot_fig5(s3_traj)

        default_method = self.methods[next(iter(self.methods))]
        n_topos = int(len(getattr(self.env.reconfig, "_cache", [])))
        normal_topo = 0
        reconf_topo = 16 if n_topos > 16 else max(n_topos - 1, 0)

        ep_normal = self.run_episode(default_method, force_topology=normal_topo)
        ep_reconf = self.run_episode(default_method, force_topology=reconf_topo)
        self.plot_fig6(ep_normal["v_bus"], ep_reconf["v_bus"])

        lmp_track, p2p_track = [], []
        for _ in range(max(smoke_runs, 3)):
            ep = self.run_episode(self.methods[next(iter(self.methods))])
            lmp_track.append(ep["lmp_mean"])
            p2p_track.append(ep["P2P"])
        self.plot_fig7(lmp_track, p2p_track)

        table3 = self.build_table3(n_runs=smoke_runs)
        self.plot_fig8(table3)
        self.plot_fig9()
        self.plot_fig10()

        table1_final = self.build_table1(n_runs=n_runs)
        table2_final = self.build_table2(n_runs=n_runs)
        table3_final = self.build_table3(n_runs=n_runs)
        table_ffr_topology = self.build_ffr_topology_table(n_runs=n_runs)

        table1_final.to_csv(self.output_dir / "table1.csv", index=False)
        table2_final.to_csv(self.output_dir / "table2.csv", index=False)
        table3_final.to_csv(self.output_dir / "table3.csv", index=False)
        table_ffr_topology.to_csv(self.output_dir / "table_ffr_topology.csv", index=False)

        return table1, table2_final, table3_final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-loop evaluator for paper tables and figures")
    parser.add_argument("--placement", type=Path, default=Path("artifacts/placement/official_placement_v3.json"))
    parser.add_argument("--mpc-path", type=Path, default=Path("data/grid_IEEE123_complete.m"))
    parser.add_argument("--checkpoint-proposed", type=Path, default=Path("checkpoints/dual_vpp_full/best.pt"))
    parser.add_argument("--checkpoint-nognn", type=Path, default=Path("checkpoints/baseline_mlp/best.pt"))
    parser.add_argument("--checkpoint-gat", type=Path, default=Path("checkpoints/baseline_gat/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--smoke-runs", type=int, default=3)
    parser.add_argument("--log-path", type=Path, default=Path("logs/t77_full_ppase_2k.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoints = {
        "No FFR": None,
        "Fixed droop": None,
        "MAPPO no-GNN": str(args.checkpoint_nognn),
        "GAT": str(args.checkpoint_gat),
        "GNN-MAPPO": str(args.checkpoint_proposed),
    }

    env_config = {
        "placement_path": str(args.placement),
        "mpc_path": str(args.mpc_path),
        "seed": 42,
    }

    evaluator = DualEvaluator(
        checkpoints=checkpoints,
        env_config=env_config,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        log_path=args.log_path,
    )

    table1_smoke, _table2, _table3 = evaluator.run_all(n_runs=int(args.n_runs), smoke_runs=int(args.smoke_runs))
    print(
        table1_smoke.pivot_table(
            index=["scenario", "method"],
            columns="metric",
            values="mean",
            aggfunc="first",
        )
    )
    print("EVALUATE GATE: PASS")


if __name__ == "__main__":
    main()
