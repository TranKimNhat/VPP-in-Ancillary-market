from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pandapower as pp

from src.environment.pandapower_backend import PandapowerBackend
from src.environment.topology_manager import TopologySnapshot, build_topology_snapshot
from src.environment.vpp_mapping import CanonicalMappings, load_canonical_mappings, resolve_der_vpp
from src.layer2_control.reward import RewardWeights, compute_reward
from src.layer2_control.safety_layer import SafetyLimits, enforce_safety


CONTROLLABLE_TYPES = {"pv", "wind", "bess", "storage"}


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 96
    voltage_tolerance: float = 0.05
    action_scale_p: float = 0.2
    action_scale_q: float = 0.2
    reward_weights: RewardWeights = RewardWeights()
    zoning_mode: str = "static"
    vpp_mode: bool = False
    mapping_config: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class GridState:
    topology: TopologySnapshot
    node_features: np.ndarray


@dataclass(frozen=True)
class AgentMapping:
    agent_id: str
    sgen_idx: int
    bus_idx: int
    zone_id: str
    vpp_id: str | None


class GridEnvironment:
    """Multi-agent environment contract for Layer 2 training and evaluation."""

    def __init__(
        self,
        net: pp.pandapowerNet,
        backend: PandapowerBackend | None = None,
        config: EnvConfig | None = None,
        layer1_pref_csv: str | Path | None = None,
        market_signal_csv: str | Path | None = None,
        mapping_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.net = net
        self.backend = backend or PandapowerBackend()
        self.config = config or EnvConfig()
        if str(self.config.zoning_mode).lower() != "static":
            raise NotImplementedError(
                f"zoning_mode='{self.config.zoning_mode}' is not supported. GridEnvironment requires static zoning."
            )

        self._base_p: dict[int, float] = {}
        self._base_q: dict[int, float] = {}
        self._agent_map: list[AgentMapping] = []
        self._agent_to_index: dict[str, int] = {}
        self._current_step = 0
        self._last_voltage_violation = 0.0
        self._last_tracking_error = 0.0
        self._curtailment_count = 0

        self._p_ref_series = np.zeros(self.config.max_steps, dtype=float)
        self._reserve_price_series = np.zeros(self.config.max_steps, dtype=float)
        self._mappings: CanonicalMappings = load_canonical_mappings(
            self.net,
            mapping_config if mapping_config is not None else self.config.mapping_config,
        )
        self._p_ref_by_vpp: dict[str, np.ndarray] = {}

        self._build_agent_mapping()
        self._load_layer1_signal(layer1_pref_csv)
        self._load_market_signals(market_signal_csv)

    @property
    def agents(self) -> list[str]:
        return [item.agent_id for item in self._agent_map]

    @property
    def num_agents(self) -> int:
        return len(self._agent_map)

    def _build_agent_mapping(self) -> None:
        self._agent_map = []
        self._agent_to_index = {}
        if self.net.sgen.empty:
            return

        for sgen_idx, row in self.net.sgen.iterrows():
            if row.get("in_service") is False:
                continue
            sgen_type = str(row.get("type", "")).lower()
            if sgen_type not in CONTROLLABLE_TYPES:
                continue
            bus_idx = int(row["bus"])
            vpp_id = resolve_der_vpp(self._mappings, der_idx=int(sgen_idx), bus_idx=bus_idx)
            zone_id = str(self._mappings.bus_to_zone.get(bus_idx, 1))
            agent_id = f"agent_{len(self._agent_map)}"
            self._agent_to_index[agent_id] = len(self._agent_map)
            self._agent_map.append(
                AgentMapping(
                    agent_id=agent_id,
                    sgen_idx=int(sgen_idx),
                    bus_idx=bus_idx,
                    zone_id=zone_id,
                    vpp_id=vpp_id,
                )
            )
            self._base_p[int(sgen_idx)] = float(row.get("p_mw", 0.0))
            self._base_q[int(sgen_idx)] = float(row.get("q_mvar", 0.0))

    def _load_layer1_signal(self, path: str | Path | None) -> None:
        if path is None:
            return
        df = pd.read_csv(path)
        if "P_ref" not in df.columns:
            return

        if self.config.vpp_mode and {"vpp_id", "hour"}.issubset(df.columns):
            self._p_ref_by_vpp = {}
            for vpp_id, group in df.groupby("vpp_id"):
                ordered = group.sort_values("hour")
                arr = np.zeros(self.config.max_steps, dtype=float)
                values = ordered["P_ref"].to_numpy(dtype=float)
                n = min(len(values), self.config.max_steps)
                arr[:n] = values[:n]
                self._p_ref_by_vpp[str(vpp_id)] = arr

            if self._p_ref_by_vpp:
                stacked = np.stack(list(self._p_ref_by_vpp.values()), axis=0)
                self._p_ref_series = np.mean(stacked, axis=0)
            return

        values = df["P_ref"].to_numpy(dtype=float)
        n = min(len(values), self.config.max_steps)
        self._p_ref_series[:n] = values[:n]

    def _load_market_signals(self, path: str | Path | None) -> None:
        if path is None:
            return
        df = pd.read_csv(path)
        if "reserve_price" not in df.columns:
            return
        reserve = (
            df.groupby("hour", as_index=False)["reserve_price"]
            .mean()
            .sort_values("hour")["reserve_price"]
            .to_numpy(dtype=float)
        )
        n = min(len(reserve), self.config.max_steps)
        self._reserve_price_series[:n] = reserve[:n]

    def observe(self, net: pp.pandapowerNet | None = None) -> GridState:
        active_net = net or self.net
        topology = build_topology_snapshot(active_net)

        if not hasattr(active_net, "res_bus") or active_net.res_bus.empty:
            vm = np.ones(len(topology.bus_index), dtype=float)
            va = np.zeros(len(topology.bus_index), dtype=float)
        else:
            vm = np.array(
                [
                    float(active_net.res_bus.at[bus, "vm_pu"]) if bus in active_net.res_bus.index else 1.0
                    for bus in topology.bus_index
                ]
            )
            va = np.array(
                [
                    float(active_net.res_bus.at[bus, "va_degree"]) if "va_degree" in active_net.res_bus.columns and bus in active_net.res_bus.index else 0.0
                    for bus in topology.bus_index
                ]
            )

        load_p = np.zeros(len(topology.bus_index), dtype=float)
        load_q = np.zeros(len(topology.bus_index), dtype=float)
        if not active_net.load.empty:
            for _, row in active_net.load.iterrows():
                bus = int(row["bus"])
                if bus in topology.bus_index:
                    idx = topology.bus_index.index(bus)
                    load_p[idx] += float(row.get("p_mw", 0.0))
                    load_q[idx] += float(row.get("q_mvar", 0.0))

        gen_p = np.zeros(len(topology.bus_index), dtype=float)
        gen_q = np.zeros(len(topology.bus_index), dtype=float)
        if not active_net.sgen.empty:
            for _, row in active_net.sgen.iterrows():
                bus = int(row["bus"])
                if bus in topology.bus_index:
                    idx = topology.bus_index.index(bus)
                    gen_p[idx] += float(row.get("p_mw", 0.0))
                    gen_q[idx] += float(row.get("q_mvar", 0.0))

        node_features = np.stack([vm, va, load_p, load_q, gen_p, gen_q], axis=1)
        node_features = np.nan_to_num(node_features, nan=0.0, posinf=1e6, neginf=-1e6)
        return GridState(topology=topology, node_features=node_features)

    def _agent_obs(self, state: GridState) -> dict[str, dict[str, np.ndarray | float | int | str | None]]:
        obs: dict[str, dict[str, np.ndarray | float | int | str | None]] = {}
        p_ref = float(self._p_ref_series[min(self._current_step, len(self._p_ref_series) - 1)])
        reserve_price = float(self._reserve_price_series[min(self._current_step, len(self._reserve_price_series) - 1)])
        global_state = np.array([p_ref, reserve_price], dtype=float)

        for mapping in self._agent_map:
            if mapping.bus_idx in state.topology.bus_index:
                bus_local_idx = state.topology.bus_index.index(mapping.bus_idx)
            else:
                bus_local_idx = 0

            local_state = state.node_features[bus_local_idx].astype(float)
            p_ref_agent = p_ref
            if self.config.vpp_mode and mapping.vpp_id is not None:
                series = self._p_ref_by_vpp.get(mapping.vpp_id)
                if series is not None and len(series) > 0:
                    p_ref_agent = float(series[min(self._current_step, len(series) - 1)])
            obs[mapping.agent_id] = {
                "node_features": state.node_features.astype(float),
                "adjacency": state.topology.adjacency.astype(float),
                "global_state": global_state,
                "local_state": local_state,
                "agent_index": int(bus_local_idx),
                "p_ref": p_ref_agent,
                "zone_id": mapping.zone_id,
                "vpp_id": mapping.vpp_id,
            }
        return obs

    def _action_to_setpoint(self, mapping: AgentMapping, action: np.ndarray) -> tuple[float, float, bool]:
        base_p = self._base_p.get(mapping.sgen_idx, 0.0)
        base_q = self._base_q.get(mapping.sgen_idx, 0.0)

        scaled = np.asarray(action, dtype=float)
        if scaled.size < 2:
            scaled = np.pad(scaled, (0, 2 - scaled.size), mode="constant")

        p_target = base_p + self.config.action_scale_p * float(scaled[0])
        q_target = base_q + self.config.action_scale_q * float(scaled[1])

        limits = SafetyLimits(
            p_min=base_p - abs(self.config.action_scale_p),
            p_max=base_p + abs(self.config.action_scale_p),
            q_min=base_q - abs(self.config.action_scale_q),
            q_max=base_q + abs(self.config.action_scale_q),
            s_max=max(np.sqrt((abs(base_p) + self.config.action_scale_p) ** 2 + (abs(base_q) + self.config.action_scale_q) ** 2), 1e-6),
        )
        safe = enforce_safety(np.array([p_target, q_target], dtype=float), limits)
        return safe.p_safe, safe.q_safe, safe.curtailed

    def reset(self, seed: int | None = None) -> tuple[dict[str, dict[str, np.ndarray | float | int]], dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        self._current_step = 0
        self._curtailment_count = 0
        self._last_tracking_error = 0.0
        self._last_voltage_violation = 0.0

        for mapping in self._agent_map:
            self.net.sgen.at[mapping.sgen_idx, "p_mw"] = self._base_p.get(mapping.sgen_idx, 0.0)
            self.net.sgen.at[mapping.sgen_idx, "q_mvar"] = self._base_q.get(mapping.sgen_idx, 0.0)

        self.step_power_flow(self.net)
        state = self.observe(self.net)
        obs = self._agent_obs(state)
        info = {
            "step": self._current_step,
            "tracking_error": self._last_tracking_error,
            "voltage_violation": self._last_voltage_violation,
            "zoning_mode": self.config.zoning_mode,
            "mapping_scope": "vpp" if self.config.vpp_mode else "legacy",
            "legacy_mapping_fallback": bool(self._mappings.legacy_mode),
        }
        return obs, info

    def step(
        self,
        action_dict: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, dict[str, np.ndarray | float | int]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, float | bool]],
    ]:
        if not self._agent_map:
            raise RuntimeError("No controllable agents found in network sgen table.")

        curtailed_flags: dict[str, bool] = {}
        for mapping in self._agent_map:
            action = np.asarray(action_dict.get(mapping.agent_id, np.zeros(2, dtype=float)), dtype=float)
            p_safe, q_safe, curtailed = self._action_to_setpoint(mapping, action)
            self.net.sgen.at[mapping.sgen_idx, "p_mw"] = p_safe
            self.net.sgen.at[mapping.sgen_idx, "q_mvar"] = q_safe
            curtailed_flags[mapping.agent_id] = curtailed

        converged = self.step_power_flow(self.net)
        state = self.observe(self.net)

        p_ref = float(self._p_ref_series[min(self._current_step, len(self._p_ref_series) - 1)])
        p_actual = float(self.net.sgen["p_mw"].sum()) if not self.net.sgen.empty else 0.0
        tracking_error = abs(p_ref - p_actual)

        vm = state.node_features[:, 0]
        voltage_violations = np.maximum(np.abs(vm - 1.0) - self.config.voltage_tolerance, 0.0)
        voltage_violation = float(np.mean(voltage_violations)) if voltage_violations.size else 0.0

        rewards: dict[str, float] = {}
        infos: dict[str, dict[str, float | bool | str | None]] = {}
        for mapping in self._agent_map:
            curtailed = bool(curtailed_flags.get(mapping.agent_id, False))
            if curtailed:
                self._curtailment_count += 1

            p_ref_agent = p_ref
            p_actual_agent = p_actual
            if self.config.vpp_mode and mapping.vpp_id is not None:
                series = self._p_ref_by_vpp.get(mapping.vpp_id)
                if series is not None and len(series) > 0:
                    p_ref_agent = float(series[min(self._current_step, len(series) - 1)])
                agent_rows = [m for m in self._agent_map if m.vpp_id == mapping.vpp_id]
                if agent_rows:
                    p_actual_agent = float(sum(float(self.net.sgen.at[m.sgen_idx, "p_mw"]) for m in agent_rows))

            reward = compute_reward(
                p_ref=p_ref_agent,
                p_actual=p_actual_agent,
                voltage_violation=voltage_violation,
                curtailed=curtailed,
                weights=self.config.reward_weights,
            )
            rewards[mapping.agent_id] = reward
            infos[mapping.agent_id] = {
                "converged": bool(converged),
                "tracking_error": float(abs(p_ref_agent - p_actual_agent)),
                "voltage_violation": float(voltage_violation),
                "curtailed": curtailed,
                "p_ref": p_ref_agent,
                "p_actual": p_actual_agent,
                "zone_id": mapping.zone_id,
                "vpp_id": mapping.vpp_id,
                "zoning_mode": self.config.zoning_mode,
                "mapping_scope": "vpp" if self.config.vpp_mode else "legacy",
                "safety_mode": "clip_project",
            }

        self._last_tracking_error = float(tracking_error)
        self._last_voltage_violation = float(voltage_violation)

        self._current_step += 1
        terminated = (not converged) or (self._current_step >= self.config.max_steps)
        truncated = self._current_step >= self.config.max_steps

        obs = self._agent_obs(state)
        terminated_dict = {agent: terminated for agent in self.agents}
        terminated_dict["__all__"] = terminated
        truncated_dict = {agent: truncated for agent in self.agents}
        truncated_dict["__all__"] = truncated

        return obs, rewards, terminated_dict, truncated_dict, infos

    def step_power_flow(self, net: pp.pandapowerNet) -> bool:
        result = self.backend.run_power_flow(net)
        return result.converged

    def metrics(self) -> dict[str, float]:
        total_agent_steps = max(self._current_step * max(self.num_agents, 1), 1)
        return {
            "steps": float(self._current_step),
            "tracking_error": float(self._last_tracking_error),
            "voltage_violation": float(self._last_voltage_violation),
            "curtailment_ratio": float(self._curtailment_count / total_agent_steps),
        }
