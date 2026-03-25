from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import random
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.converter.matpower import from_mpc

from src.env.evcs_model import EVCSModel
from src.env.freq_model import AnalyticalFrequencyModel, compute_nadir
from src.env.safety_layer import apply_safety_layer
from src.env.IEEE123bus import (
    IEEE123_ZONE_BUS_MAP,
    _build_bus_name_to_index,
    _normalize_bus_label,
)


def _read_matpower_bus_numbers(mpc_path: Path) -> list[int]:
    lines = mpc_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("mpc.bus"):
            start = idx
            break
    if start is None:
        raise ValueError(f"mpc.bus section not found in {mpc_path}")

    bus_numbers: list[int] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if stripped.startswith("];"):
            break
        if stripped.endswith(";"):
            stripped = stripped[:-1]
        parts = stripped.split()
        if not parts:
            continue
        try:
            bus_no = int(float(parts[0]))
        except ValueError:
            continue
        bus_numbers.append(bus_no)

    if not bus_numbers:
        raise ValueError(f"No MATPOWER bus IDs parsed from {mpc_path}")
    return bus_numbers


def _fix_mpc_file(mpc_path: str) -> str:
    """
    Pre-process MATPOWER .m file to fix pandapower parsing issues.
    Writes a fixed copy to <original_stem>_ppfix.m and returns its path.

    Fixes applied:
      1. Branch expressions like "0.001010139*5" → evaluated float "0.005050695"
      2. Wide whitespace in near-zero impedance rows → single tab delimiter
      3. MATLAB gencost array expression → plain numeric matrix
    """
    import re

    with open(mpc_path, encoding="utf-8") as f:
        content = f.read()

    content = content.encode("ascii", "ignore").decode("ascii")

    def eval_mult(match: re.Match[str]) -> str:
        return f"{float(match.group(1)) * float(match.group(2)):.10g}"

    content = re.sub(r"([\.\d]+)\*([\.\d]+)", eval_mult, content)

    lines = content.split("\n")
    in_branch = False
    result: List[str] = []
    for line in lines:
        raw = line
        if re.search(r"mpc\.branch\s*=\s*\[", raw):
            in_branch = True
        if in_branch and raw.strip() == "];":
            in_branch = False
        if in_branch and raw.strip() and not raw.strip().startswith("%"):
            if re.match(r"\s*\d", raw):
                raw = re.sub(r"[ \t]+", "\t", raw.strip())
        result.append(raw)
    content = "\n".join(result)

    gen_match = re.search(r"mpc\.gen\s*=\s*\[(.*?)\];", content, re.DOTALL)
    n_gen = 0
    if gen_match:
        n_gen = sum(
            1
            for l in gen_match.group(1).split("\n")
            if l.strip() and not l.strip().startswith("%")
        )
    n_gen = max(n_gen, 86)
    gc_new = "mpc.gencost = [\n"
    gc_new += "2\t0\t0\t3\t0\t1\t0\t;\n"
    for _ in range(n_gen - 1):
        gc_new += "2\t0\t0\t3\t0\t0\t0\t;\n"
    gc_new += "];\n"
    content = re.sub(
        r"mpc\.gencost\s*=\s*\.\.\..*?(?=\n\n|\Z)",
        gc_new.rstrip("\n"),
        content,
        flags=re.DOTALL,
    )

    stem = Path(mpc_path).stem
    fixed_path = str(Path(mpc_path).parent / f"{stem}_ppfix.m")
    with open(fixed_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[MicrogridEnv] MPC file fixed: {fixed_path}")
    return fixed_path


@dataclass
class AgentSpec:
    agent_type: str
    vpp: str
    zone: int
    mpc_bus: int
    element: str
    element_idx: int
    p_rated: float
    q_max: float


class MicrogridEnv(gym.Env):
    metadata = {"render.modes": []}

    @staticmethod
    def pre_fix_mpc(mpc_path: str | Path) -> str:
        mpc_path = Path(mpc_path)
        fixed_path = mpc_path.with_name(f"{mpc_path.stem}_ppfix.m")
        if not fixed_path.exists():
            _fix_mpc_file(str(mpc_path))
        return str(fixed_path)

    @classmethod
    def load_all_days(cls, precomputed_dir: str | Path) -> list[pd.DataFrame]:
        precomputed_path = Path(precomputed_dir)
        parquet_files = sorted(precomputed_path.glob("day_*.parquet"))
        return [pd.read_parquet(path) for path in parquet_files]

    def __init__(
        self,
        placement_path: str | Path,
        mpc_path: str | Path,
        precomputed_dir: str | Path | None = None,
        seed: int = 42,
        preloaded_days: list[pd.DataFrame] | None = None,
    ) -> None:
        super().__init__()
        self.rng = random.Random(seed)

        placement_path = Path(placement_path)
        mpc_path = Path(mpc_path)
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir else None
        self.use_precomputed = self.precomputed_dir is not None

        fixed_mpc = self.pre_fix_mpc(mpc_path)
        net = from_mpc(fixed_mpc)

        bus_numbers = _read_matpower_bus_numbers(Path(fixed_mpc))
        if len(bus_numbers) != len(net.bus.index):
            raise ValueError(
                f"MATPOWER bus count {len(bus_numbers)} does not match pandapower bus count {len(net.bus.index)}"
            )
        net.bus["bus_id"] = pd.Series(bus_numbers, index=net.bus.index, dtype="Int64")
        self._bus_map: Dict[int, int] = {
            int(bus_id): int(idx)
            for idx, bus_id in net.bus["bus_id"].items()
            if not pd.isna(bus_id)
        }
        self.pp_idx = lambda mpc_id: self._bus_map[int(mpc_id)]

        net.load["p_mw"] = net.load["p_mw"] * 8.5
        net.load["q_mvar"] = net.load["q_mvar"] * 8.5
        net.bus["vn_kv"] = 22.0

        with open(placement_path, encoding="utf-8") as f:
            placement = json.load(f)

        self._placement = placement
        self._evcs_configs = placement.get("evcs", [])
        self._dpv_configs = placement.get("dpv", [])
        self._n_evcs = len(self._evcs_configs)
        self._n_dpv = len(self._dpv_configs)
        self._n_agents = self._n_evcs * 3 + self._n_dpv

        self._agent_specs = []
        self._agent_bus_map = {}
        self.sgen_pv_map: dict[int, float] = {}
        self.sgen_wind_map: dict[int, float] = {}

        self._inject_assets(net, placement)
        self._add_capacitor_banks(net)
        self._order_agents(placement)
        self.edge_index = self._build_edge_index(net)
        self.agent_bus_map = self._agent_bus_map

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n_agents, 24), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(
                self._n_evcs * 2
                + self._n_evcs * 2
                + self._n_evcs * 1
                + self._n_dpv * 2,
            ),
            dtype=np.float32,
        )

        self.base_net = deepcopy(net)
        self.net = deepcopy(net)
        self.step_count = 0
        self._day_data: pd.DataFrame | None = None

        self.parquet_files: list[Path] = []
        self.eval_files: list[Path] = []
        self.current_day_data: pd.DataFrame | None = None
        self.current_row: pd.Series | None = None
        self.zone_load_indices: dict[int, list[int]] = {1: [], 2: [], 3: [], 4: []}
        self.zone_base_loads: dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

        self._all_days: list[pd.DataFrame] = []
        self._n_days = 0
        self._data: pd.DataFrame | None = None
        self._data_checksum: object | None = None
        self._current_step = 0

        if self.precomputed_dir and self.precomputed_dir.exists():
            self.parquet_files = sorted(self.precomputed_dir.glob("day_*.parquet"))
            eval_days_path = self.precomputed_dir / "eval_days.txt"
            if eval_days_path.exists():
                eval_days = [
                    line.strip()
                    for line in eval_days_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.eval_files = [
                    path for path in self.parquet_files if path.stem in set(eval_days)
                ]

            if self.use_precomputed:
                if preloaded_days is not None:
                    self._all_days = preloaded_days
                else:
                    self._all_days = self.load_all_days(self.precomputed_dir)
                self._n_days = len(self._all_days)

        self.zone_load_indices, self.zone_base_loads = self._build_zone_load_mappings()

        self._evcs_models = [EVCSModel(str(idx), cfg) for idx, cfg in enumerate(self._evcs_configs)]
        self._freq_model = AnalyticalFrequencyModel()
        self._freq_model.reset()
        self._last_freq_state = None
        self.last_freq_event = 0.0
        self._freq_override_prob: float | None = None
        self._last_safety_info = {
            "stage1_activations": 0,
            "stage2_activations": 0,
            "stage3_activations": 0,
            "stage4_activations": 0,
            "stage5_activations": 0,
            "total_safety_activations": 0,
            "s_margin_pen": 0.0,
        }

    def _inject_assets(self, net: pp.pandapowerNet, placement: Dict[str, Any]) -> None:
        evcs = placement.get("evcs", [])
        dpv = placement.get("dpv", [])
        wind = placement.get("wind", [])
        gfm = placement.get("gfm", {})

        for ev in evcs:
            bus = self.pp_idx(ev["bus"])
            pv_idx = pp.create_sgen(
                net, bus=bus, p_mw=ev["pv_mw"], q_mvar=0.0, controllable=True
            )
            self.sgen_pv_map[pv_idx] = float(ev["pv_mw"])
            bess_idx = pp.create_storage(
                net,
                bus=bus,
                p_mw=0.0,
                max_p_mw=ev["bess_mw"],
                min_p_mw=-ev["bess_mw"],
                max_e_mwh=ev["bess_mwh"],
                soc_percent=50.0,
            )
            v2g_idx = pp.create_load(
                net, bus=bus, p_mw=0.0, q_mvar=0.0, controllable=True
            )

            inv_mva = float(ev.get("inverter_mva", ev["pv_mw"]))
            q_max_pv = inv_mva * math.sin(math.acos(0.9))
            q_max_bess = float(ev.get("inverter_mva", ev["bess_mw"])) * math.sin(
                math.acos(0.9)
            )
            q_max_v2g = 0.0
            zone = int(ev.get("zone", 1))
            self._append_agent(
                "EVCS_PV",
                ev["vpp"],
                zone,
                ev["bus"],
                "sgen",
                pv_idx,
                ev["pv_mw"],
                q_max_pv,
            )
            self._append_agent(
                "EVCS_BESS",
                ev["vpp"],
                zone,
                ev["bus"],
                "storage",
                bess_idx,
                ev["bess_mw"],
                q_max_bess,
            )
            self._append_agent(
                "EVCS_V2G",
                ev["vpp"],
                zone,
                ev["bus"],
                "load",
                v2g_idx,
                ev["v2g_mw"],
                q_max_v2g,
            )

        for pv in dpv:
            bus = self.pp_idx(pv["bus"])
            pv_idx = pp.create_sgen(
                net, bus=bus, p_mw=pv["mw"], q_mvar=0.0, controllable=True
            )
            self.sgen_pv_map[pv_idx] = float(pv["mw"])
            inv_mva = float(pv.get("inverter_mva", pv.get("sn_mva", pv["mw"])))
            q_max = inv_mva * math.sin(math.acos(0.9))
            zone = int(pv.get("zone", 1))
            self._append_agent(
                "DPV",
                pv["vpp"],
                zone,
                pv["bus"],
                "sgen",
                pv_idx,
                pv["mw"],
                q_max,
            )

        for w in wind:
            bus = self.pp_idx(w["bus"])
            wind_idx = pp.create_sgen(
                net,
                bus=bus,
                p_mw=w["mw"],
                q_mvar=0.0,
                controllable=False,
                type="WP",
            )
            self.sgen_wind_map[wind_idx] = float(w["mw"])

        if "G2" in gfm:
            g2 = gfm["G2"]
            bus = self.pp_idx(g2["bus"])
            pp.create_storage(
                net,
                bus=bus,
                p_mw=0.0,
                max_p_mw=g2["bess_mw"],
                min_p_mw=-g2["bess_mw"],
                max_e_mwh=g2["bess_mwh"],
                soc_percent=50.0,
            )
            pp.create_sgen(
                net, bus=bus, p_mw=g2["pv_mw"], q_mvar=0.0, controllable=False
            )

        if "G1" in gfm:
            g1_bus = self.pp_idx(gfm["G1"]["bus"])
            net.ext_grid.loc[:, "bus"] = g1_bus

    def _add_capacitor_banks(self, net: pp.pandapowerNet) -> None:
        def _bus_idx(mpc_bus: int) -> int:
            matches = net.bus.index[net.bus.bus_id == mpc_bus]
            if len(matches) == 0:
                raise ValueError(f"Cap bank bus {mpc_bus} not found in net.bus")
            return int(matches[0])

        pp.create_shunt(net, bus=_bus_idx(83), q_mvar=-1.7, p_mw=0.0, name="CB1")
        pp.create_shunt(net, bus=_bus_idx(88), q_mvar=-0.4, p_mw=0.0, name="CB2")
        pp.create_shunt(net, bus=_bus_idx(90), q_mvar=-0.4, p_mw=0.0, name="CB3")
        pp.create_shunt(net, bus=_bus_idx(92), q_mvar=-0.4, p_mw=0.0, name="CB4")

        for mpc_bus, q_mvar, name in [
            (18, -1.0, "CB-A"),
            (47, -0.8, "CB-B"),
            (76, -0.6, "CB-C"),
        ]:
            pp.create_shunt(net, bus=_bus_idx(mpc_bus), q_mvar=q_mvar, p_mw=0.0, name=name)

    def _build_zone_load_mappings(self) -> tuple[dict[int, list[int]], dict[int, float]]:
        zone_bus_indices: dict[int, set[int]] = {1: set(), 2: set(), 3: set(), 4: set()}
        bus_name_to_index = {
            _normalize_bus_label(name).lower(): idx
            for name, idx in _build_bus_name_to_index(self.base_net).items()
        }
        for zone, bus_names in IEEE123_ZONE_BUS_MAP.items():
            indices = set()
            for name in bus_names:
                normalized = _normalize_bus_label(name).lower()
                if normalized in {"total", "s73c", ""}:
                    continue
                bus_idx = bus_name_to_index.get(normalized)
                if bus_idx is not None:
                    indices.add(int(bus_idx))
            zone_bus_indices[int(zone)] = indices

        zone_load_indices: dict[int, list[int]] = {1: [], 2: [], 3: [], 4: []}
        zone_base_loads: dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for load_idx in self.base_net.load.index:
            bus_idx = int(self.base_net.load.at[load_idx, "bus"])
            for zone, bus_indices in zone_bus_indices.items():
                if bus_idx in bus_indices:
                    zone_load_indices[zone].append(int(load_idx))
                    zone_base_loads[zone] += float(self.base_net.load.at[load_idx, "p_mw"])
                    break

        return zone_load_indices, zone_base_loads

    def _append_agent(
        self,
        agent_type: str,
        vpp: str,
        zone: int,
        mpc_bus: int,
        element: str,
        element_idx: int,
        p_rated: float,
        q_max: float,
    ) -> None:
        spec = AgentSpec(
            agent_type=agent_type,
            vpp=vpp,
            zone=int(zone),
            mpc_bus=int(mpc_bus),
            element=element,
            element_idx=int(element_idx),
            p_rated=float(p_rated),
            q_max=float(q_max),
        )
        self._agent_specs.append(spec)
        agent_idx = len(self._agent_specs) - 1
        self._agent_bus_map[agent_idx] = (spec.mpc_bus, spec.agent_type, spec.vpp)
        self.agent_bus_map = self._agent_bus_map

    def _build_edge_index(self, net: pp.pandapowerNet) -> np.ndarray:
        if net.line.empty:
            return np.zeros((2, 0), dtype=np.int64)
        return net.line[["from_bus", "to_bus"]].to_numpy().T.astype(np.int64)

    def _order_agents(self, placement: Dict[str, Any]) -> None:
        evcs = placement.get("evcs", [])
        dpv = placement.get("dpv", [])
        ordered: List[AgentSpec] = []

        def find_spec(agent_type: str, bus: int) -> AgentSpec:
            for spec in self._agent_specs:
                if spec.agent_type == agent_type and spec.mpc_bus == int(bus):
                    return spec
            raise KeyError(f"Agent spec not found for {agent_type} bus {bus}")

        for ev in evcs:
            ordered.append(find_spec("EVCS_PV", ev["bus"]))
        for ev in evcs:
            ordered.append(find_spec("EVCS_BESS", ev["bus"]))
        for ev in evcs:
            ordered.append(find_spec("EVCS_V2G", ev["bus"]))
        for pv in dpv:
            ordered.append(find_spec("DPV", pv["bus"]))

        self._agent_specs = ordered
        self._agent_bus_map = {
            idx: (spec.mpc_bus, spec.agent_type, spec.vpp)
            for idx, spec in enumerate(self._agent_specs)
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)

        self.net = deepcopy(self.base_net)
        self.step_count = 0
        self._day_data = None
        self.current_day_data = None
        self.current_row = None
        self._data_checksum = None

        if self.parquet_files:
            mode = None
            if options:
                mode = options.get("mode")

            if self.use_precomputed and self._all_days:
                if mode == "eval" and self.eval_files:
                    eval_set = set(self.eval_files)
                    eval_indices = [
                        idx
                        for idx, path in enumerate(self.parquet_files)
                        if path in eval_set
                    ]
                    day_idx = int(self.rng.choice(eval_indices)) if eval_indices else int(self.rng.randrange(self._n_days))
                elif self.eval_files:
                    eval_set = set(self.eval_files)
                    train_indices = [
                        idx
                        for idx, path in enumerate(self.parquet_files)
                        if path not in eval_set
                    ]
                    day_idx = int(self.rng.choice(train_indices)) if train_indices else int(self.rng.randrange(self._n_days))
                else:
                    day_idx = int(self.rng.randrange(self._n_days))
                self._data = self._all_days[day_idx].copy(deep=False)
                if __debug__ and not self._data.empty:
                    self._data_checksum = self._data.iloc[0, 0]
                else:
                    self._data_checksum = None
                self._current_step = 0
                self._day_data = self._data
                self.current_day_data = self._data
                if not self._data.empty:
                    self.current_row = self._data.iloc[0]
            else:
                if mode == "eval" and self.eval_files:
                    day_file = self.rng.choice(self.eval_files)
                elif self.eval_files:
                    eval_set = set(self.eval_files)
                    train_files = [path for path in self.parquet_files if path not in eval_set]
                    day_file = self.rng.choice(train_files) if train_files else self.rng.choice(self.parquet_files)
                else:
                    day_file = self.rng.choice(self.parquet_files)
                self._day_data = pd.read_parquet(day_file)
                self.current_day_data = self._day_data
                if self.current_day_data is not None and not self.current_day_data.empty:
                    self.current_row = self.current_day_data.iloc[0]

        if not self.net.storage.empty:
            self.net.storage["soc_percent"] = 50.0

        for evcs in self._evcs_models:
            evcs.ev_fleet = []
            evcs.bess_soc = 0.5
            evcs.step_count = 0

        self._freq_model.reset()
        self._last_freq_state = None
        self.last_freq_event = 0.0
        self._last_safety_info = {
            "stage1_activations": 0,
            "stage2_activations": 0,
            "stage3_activations": 0,
            "stage4_activations": 0,
            "stage5_activations": 0,
            "total_safety_activations": 0,
            "s_margin_pen": 0.0,
        }

        obs = self._build_observation(converged=False)
        obs = self._update_obs_from_parquet(self.current_row, obs)
        info = {
            "converged": False,
            "v_min": 0.0,
            "v_max": 0.0,
            "v_violations": 0,
            "p_loss_mw": 0.0,
            "step": self.step_count,
            "reward_breakdown": {
                "r_track": 0.0,
                "r_volt": 0.0,
                "r_freq": 0.0,
                "r_as": 0.0,
                "r_deg": 0.0,
                "r_oblig": 0.0,
                "r_p2p": 0.0,
                "r_s_margin": 0.0,
                "r_q_revenue": 0.0,
            },
            "P_p2p": np.zeros(self._n_agents, dtype=np.float32),
            "delta_f": float(self._freq_model.delta_f),
            "rocof": 0.0,
            "freq_violated": False,
            "last_freq_event": float(self.last_freq_event),
            **self._last_safety_info,
        }
        return obs, info

    def step(self, action: np.ndarray):
        if self.use_precomputed:
            return self._step_precomputed(action)
        return self._step_live(action)

    def _step_live(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_dim = int(self.action_space.shape[0])
        if action.shape[0] != expected_dim:
            raise ValueError(f"Expected action shape ({expected_dim},), got {action.shape}")

        if self.current_day_data is not None and not self.current_day_data.empty:
            row_idx = min(self.step_count, len(self.current_day_data) - 1)
            row = self.current_day_data.iloc[row_idx]
            self.current_row = row
            self._apply_parquet_row(row)

        actions_dict = self._decode_actions(action)
        env_state = self._build_env_state(converged=self.step_count > 0)
        safe_actions, safety_info = apply_safety_layer(
            actions_dict, env_state, self._evcs_configs
        )
        self._last_safety_info = safety_info
        self._apply_actions(safe_actions)

        p_p2p = np.zeros(self._n_agents, dtype=np.float32)
        n_evcs = self._n_evcs
        for idx, spec in enumerate(self._agent_specs[0:n_evcs]):
            p_p2p[idx] = spec.p_rated - float(safe_actions["evcs_pv"][idx][0])
        for idx, spec in enumerate(self._agent_specs[n_evcs : 2 * n_evcs]):
            p_p2p[idx + n_evcs] = float(safe_actions["evcs_bess"][idx][0])
        for idx, spec in enumerate(self._agent_specs[2 * n_evcs : 3 * n_evcs]):
            p_p2p[idx + 2 * n_evcs] = -float(safe_actions["evcs_v2g"][idx])
        for idx, spec in enumerate(self._agent_specs[3 * n_evcs :]):
            p_p2p[idx + 3 * n_evcs] = spec.p_rated - float(safe_actions["dpv"][idx][0])

        converged = True
        try:
            pp.runpp(
                self.net,
                algorithm="nr",
                numba=False,
                max_iteration=50,
                tolerance_mva=1e-4,
            )
            if not bool(self.net["converged"]):
                converged = False
        except Exception:
            converged = False

        if not converged:
            reward = -100.0
            obs = self._build_observation(converged=False)
            obs = self._update_obs_from_parquet(self.current_row, obs)
            self.last_freq_event = 0.0
            self._last_freq_state = None
            info = {
                "converged": False,
                "v_min": 0.0,
                "v_max": 0.0,
                "v_violations": 0,
                "p_loss_mw": 0.0,
                "step": self.step_count,
                "reward_breakdown": {
                    "r_track": 0.0,
                    "r_volt": 0.0,
                    "r_freq": 0.0,
                    "r_as": 0.0,
                    "r_deg": 0.0,
                    "r_oblig": 0.0,
                    "r_p2p": 0.0,
                    "r_s_margin": 0.0,
                    "r_q_revenue": 0.0,
                },
                "P_p2p": p_p2p,
                "delta_f": float(self._freq_model.delta_f),
                "rocof": 0.0,
                "freq_violated": False,
                "last_freq_event": float(self.last_freq_event),
                **self._last_safety_info,
            }
        else:
            v = self.net.res_bus.vm_pu
            v_min = float(v.min()) if not v.empty else 0.0
            v_max = float(v.max()) if not v.empty else 0.0
            v_violations = int(((v < 0.95) | (v > 1.05)).sum()) if not v.empty else 0
            p_loss = float(self.net.res_line.pl_mw.sum()) if not self.net.res_line.empty else 0.0

            day_row = self._get_day_row()
            delta_p_cont = float(day_row.get("delta_p_cont", 0.0)) if day_row else 0.0
            freq_flag = bool(day_row.get("freq_event_flag", 0)) if day_row else False
            delta_p_net = -delta_p_cont if freq_flag else 0.0
            self.last_freq_event = float(delta_p_net)
            self._last_freq_state = self._freq_model.step(delta_p_net)

            env_state = self._build_env_state(converged=True)
            reward, breakdown = self._compute_reward(env_state, safe_actions, safety_info, day_row)

            obs = self._build_observation(converged=True)
            obs = self._update_obs_from_parquet(self.current_row, obs)
            info = {
                "converged": True,
                "v_min": v_min,
                "v_max": v_max,
                "v_violations": v_violations,
                "p_loss_mw": p_loss,
                "step": self.step_count,
                "reward_breakdown": breakdown,
                "P_p2p": p_p2p,
                "delta_f": float(self._freq_model.delta_f),
                "rocof": float(self._last_freq_state.rocof) if self._last_freq_state else 0.0,
                "freq_violated": bool(self._last_freq_state.freq_violated)
                if self._last_freq_state
                else False,
                "last_freq_event": float(self.last_freq_event),
                **safety_info,
            }

        self.step_count += 1
        terminated = self.step_count >= 96
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def set_freq_prob(self, prob: float | None) -> None:
        """
        None  = dùng freq_event_flag từ parquet (Phase A, B)
        float = override với xác suất mới (Phase C: 0.15, Phase D: 0.073)
        """
        self._freq_override_prob = prob

    def _step_precomputed(self, action: np.ndarray):
        if __debug__ and self._data is not None and self._data_checksum is not None:
            assert self._data.iloc[0, 0] == self._data_checksum, (
                "DataFrame mutated! Check for in-place writes."
            )

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_dim = int(self.action_space.shape[0])
        if action.shape[0] != expected_dim:
            raise ValueError(f"Expected action shape ({expected_dim},), got {action.shape}")

        if self._data is None or self._data.empty:
            raise RuntimeError("Precomputed data not loaded. Call reset() first.")

        row_idx = min(self._current_step, len(self._data) - 1)
        row = self._data.iloc[row_idx]

        if self._freq_override_prob is not None:
            if np.random.random() < self._freq_override_prob:
                freq_flag = 1
                delta_p = float(np.random.choice([-0.3, -0.5, -1.0, -1.5, +0.3, +0.5]))
                freq = compute_nadir(delta_p)
                f_nadir = float(freq["f_nadir"])
                rocof = float(freq["rocof"])
            else:
                freq_flag = 0
                delta_p, f_nadir, rocof = 0.0, 50.0, 0.0
        else:
            freq_flag = int(row["freq_event_flag"])
            delta_p = float(row["delta_p_cont"])
            f_nadir = float(row["f_nadir"])
            rocof = float(row["rocof"])

        row = row.copy(deep=True)
        row["freq_event_flag"] = freq_flag
        row["delta_p_cont"] = delta_p
        row["f_nadir"] = f_nadir
        row["rocof"] = rocof
        self.current_row = row
        self.last_freq_event = float(delta_p)
        self._last_freq_state = self._freq_model.step(delta_p if freq_flag else 0.0)

        actions_dict = self._decode_actions(action)
        safe_actions = self._clip_actions(actions_dict)

        p_p2p = np.zeros(self._n_agents, dtype=np.float32)
        n_evcs = self._n_evcs
        for idx, spec in enumerate(self._agent_specs[0:n_evcs]):
            p_p2p[idx] = spec.p_rated - float(safe_actions["evcs_pv"][idx][0])
        for idx, spec in enumerate(self._agent_specs[n_evcs : 2 * n_evcs]):
            p_p2p[idx + n_evcs] = float(safe_actions["evcs_bess"][idx][0])
        for idx, spec in enumerate(self._agent_specs[2 * n_evcs : 3 * n_evcs]):
            p_p2p[idx + 2 * n_evcs] = -float(safe_actions["evcs_v2g"][idx])
        for idx, spec in enumerate(self._agent_specs[3 * n_evcs :]):
            p_p2p[idx + 3 * n_evcs] = spec.p_rated - float(safe_actions["dpv"][idx][0])

        reward, breakdown = self._compute_reward_from_row(safe_actions, row)

        obs = self._build_observation(converged=False)
        obs = self._update_obs_from_parquet(row, obs)
        obs[:, 22] = float(freq_flag)
        obs[:, 23] = float(delta_p) / 15.705

        info = {
            "converged": True,
            "v_min": 0.0,
            "v_max": 0.0,
            "v_violations": 0,
            "p_loss_mw": 0.0,
            "step": self._current_step,
            "reward_breakdown": breakdown,
            "P_p2p": p_p2p,
            "delta_f": float(self._freq_model.delta_f),
            "rocof": float(rocof),
            "freq_violated": bool(freq_flag),
            "freq_event_flag": int(freq_flag),
            "f_nadir": float(f_nadir),
            "delta_p_cont": float(delta_p),
            "last_freq_event": float(self.last_freq_event),
            **self._last_safety_info,
        }

        self._current_step += 1
        self.step_count = self._current_step
        terminated = self._current_step >= 96
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def _decode_actions(self, action: np.ndarray) -> Dict[str, List[Any]]:
        idx = 0
        evcs_pv: List[List[float]] = []
        evcs_bess: List[List[float]] = []
        evcs_v2g: List[float] = []
        dpv: List[List[float]] = []

        n_evcs = self._n_evcs

        for spec in self._agent_specs[0:n_evcs]:
            p_curt_norm = float(action[idx])
            q_norm = float(action[idx + 1])
            idx += 2
            curtail_fraction = (p_curt_norm + 1.0) / 2.0
            p_curt = float(np.clip(curtail_fraction * spec.p_rated, 0.0, spec.p_rated))
            q_set = float(np.clip(q_norm, -1.0, 1.0) * spec.q_max)
            evcs_pv.append([p_curt, q_set])

        for spec in self._agent_specs[n_evcs : 2 * n_evcs]:
            p_norm = float(action[idx])
            q_norm = float(action[idx + 1])
            idx += 2
            p_set = float(np.clip(p_norm, -1.0, 1.0) * spec.p_rated)
            q_set = float(np.clip(q_norm, -1.0, 1.0) * spec.q_max)
            evcs_bess.append([p_set, q_set])

        for spec in self._agent_specs[2 * n_evcs : 3 * n_evcs]:
            p_norm = float(action[idx])
            idx += 1
            p_set = float((p_norm + 1.0) / 2.0 * spec.p_rated)
            evcs_v2g.append(p_set)

        for spec in self._agent_specs[3 * n_evcs :]:
            p_curt_norm = float(action[idx])
            q_norm = float(action[idx + 1])
            idx += 2
            p_curt = float((p_curt_norm + 1.0) / 2.0 * spec.p_rated)
            p_curt = float(np.clip(p_curt, 0.0, spec.p_rated))
            q_set = float(np.clip(q_norm, -1.0, 1.0) * spec.q_max)
            dpv.append([p_curt, q_set])

        return {
            "evcs_pv": evcs_pv,
            "evcs_bess": evcs_bess,
            "evcs_v2g": evcs_v2g,
            "dpv": dpv,
        }

    def _clip_actions(self, actions_dict: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        safe_actions = {
            "evcs_pv": [list(item) for item in actions_dict.get("evcs_pv", [])],
            "evcs_bess": [list(item) for item in actions_dict.get("evcs_bess", [])],
            "evcs_v2g": [float(item) for item in actions_dict.get("evcs_v2g", [])],
            "dpv": [list(item) for item in actions_dict.get("dpv", [])],
        }

        evcs_cfgs = self._evcs_configs or []
        n_evcs = len(evcs_cfgs)
        dpv_p_rated = [spec.p_rated for spec in self._agent_specs[3 * n_evcs :]]
        dpv_s_rated = [spec.q_max for spec in self._agent_specs[3 * n_evcs :]]

        s_margin_pen = 0.0

        for i, (p_curt, q) in enumerate(safe_actions["evcs_pv"]):
            cfg = evcs_cfgs[i]
            p_curt_new = float(np.clip(p_curt, 0.0, float(cfg["pv_mw"])))
            s_rated = float(cfg.get("inverter_mva", cfg["pv_mw"]))
            q_max = 0.436 * s_rated
            q_new = float(np.clip(q, -q_max, q_max))
            safe_actions["evcs_pv"][i] = [p_curt_new, q_new]

        for i, (p, q) in enumerate(safe_actions["evcs_bess"]):
            cfg = evcs_cfgs[i]
            p_new = float(np.clip(p, -float(cfg["bess_mw"]), float(cfg["bess_mw"])))
            s_rated = float(cfg.get("inverter_mva", cfg["bess_mw"]))
            q_max = 0.436 * s_rated
            q_new = float(np.clip(q, -q_max, q_max))
            safe_actions["evcs_bess"][i] = [p_new, q_new]

        for i, p in enumerate(safe_actions["evcs_v2g"]):
            cfg = evcs_cfgs[i]
            p_cap = min(float(cfg["v2g_mw"]), 0.75)
            p_new = float(np.clip(p, 0.0, p_cap))
            safe_actions["evcs_v2g"][i] = p_new

        for i, (p_curt, q) in enumerate(safe_actions["dpv"]):
            p_rated = float(dpv_p_rated[i])
            p_curt_new = float(np.clip(p_curt, 0.0, p_rated))
            s_rated = float(dpv_s_rated[i])
            q_max = 0.436 * s_rated
            q_new = float(np.clip(q, -q_max, q_max))
            safe_actions["dpv"][i] = [p_curt_new, q_new]

        for i, (p_curt, q) in enumerate(safe_actions["evcs_pv"]):
            cfg = evcs_cfgs[i]
            s_rated = float(cfg.get("inverter_mva", cfg["pv_mw"]))
            p_set = float(cfg["pv_mw"]) - p_curt
            s_sq = p_set * p_set + q * q
            if s_sq > s_rated * s_rated:
                q_allowed = float(np.sqrt(max(0.0, s_rated * s_rated - p_set * p_set)))
                q_new = float(np.clip(q, -q_allowed, q_allowed))
                s_margin_pen += max(0.0, (s_sq - s_rated * s_rated) / (s_rated * s_rated))
                safe_actions["evcs_pv"][i][1] = q_new

        for i, (p, q) in enumerate(safe_actions["evcs_bess"]):
            cfg = evcs_cfgs[i]
            s_rated = float(cfg.get("inverter_mva", cfg["bess_mw"]))
            s_sq = p * p + q * q
            if s_sq > s_rated * s_rated:
                q_allowed = float(np.sqrt(max(0.0, s_rated * s_rated - p * p)))
                q_new = float(np.clip(q, -q_allowed, q_allowed))
                s_margin_pen += max(0.0, (s_sq - s_rated * s_rated) / (s_rated * s_rated))
                safe_actions["evcs_bess"][i][1] = q_new

        for i, (p_curt, q) in enumerate(safe_actions["dpv"]):
            s_rated = float(dpv_s_rated[i])
            p_set = float(dpv_p_rated[i]) - p_curt
            s_sq = p_set * p_set + q * q
            if s_sq > s_rated * s_rated:
                q_allowed = float(np.sqrt(max(0.0, s_rated * s_rated - p_set * p_set)))
                q_new = float(np.clip(q, -q_allowed, q_allowed))
                s_margin_pen += max(0.0, (s_sq - s_rated * s_rated) / (s_rated * s_rated))
                safe_actions["dpv"][i][1] = q_new

        self._last_safety_info = {
            "stage1_activations": 0,
            "stage2_activations": 0,
            "stage3_activations": 0,
            "stage4_activations": 0,
            "stage5_activations": 0,
            "total_safety_activations": 0,
            "s_margin_pen": float(s_margin_pen),
        }

        return safe_actions

    def _apply_actions(self, actions_dict: Dict[str, List[Any]]) -> None:
        n_evcs = self._n_evcs

        for idx, spec in enumerate(self._agent_specs[0:n_evcs]):
            p_curt, q_set = actions_dict["evcs_pv"][idx]
            p_set = max(spec.p_rated - float(p_curt), 0.0)
            self.net.sgen.at[spec.element_idx, "p_mw"] = p_set
            self.net.sgen.at[spec.element_idx, "q_mvar"] = float(q_set)

        for idx, spec in enumerate(self._agent_specs[n_evcs : 2 * n_evcs]):
            p_set, q_set = actions_dict["evcs_bess"][idx]
            self.net.storage.at[spec.element_idx, "p_mw"] = float(p_set)
            if "q_mvar" in self.net.storage.columns:
                self.net.storage.at[spec.element_idx, "q_mvar"] = float(q_set)

        for idx, spec in enumerate(self._agent_specs[2 * n_evcs : 3 * n_evcs]):
            p_set = float(actions_dict["evcs_v2g"][idx])
            self.net.load.at[spec.element_idx, "p_mw"] = -p_set

        for idx, spec in enumerate(self._agent_specs[3 * n_evcs :]):
            p_curt, q_set = actions_dict["dpv"][idx]
            p_set = max(spec.p_rated - float(p_curt), 0.0)
            self.net.sgen.at[spec.element_idx, "p_mw"] = p_set
            self.net.sgen.at[spec.element_idx, "q_mvar"] = float(q_set)

    def _build_observation(self, converged: bool) -> np.ndarray:
        obs = np.zeros((self._n_agents, 24), dtype=np.float32)
        if not converged:
            self._encode_types(obs)
            return obs

        v = self.net.res_bus.vm_pu
        for i, spec in enumerate(self._agent_specs):
            bus_idx = self.pp_idx(spec.mpc_bus)
            if bus_idx in v.index:
                obs[i, 0] = float(v.at[bus_idx])

            if spec.element == "sgen":
                if spec.element_idx in self.net.res_sgen.index:
                    obs[i, 1] = float(self.net.res_sgen.at[spec.element_idx, "p_mw"])
            elif spec.element == "storage":
                if spec.element_idx in self.net.res_storage.index:
                    obs[i, 1] = float(self.net.res_storage.at[spec.element_idx, "p_mw"])
            elif spec.element == "load":
                if spec.element_idx in self.net.res_load.index:
                    obs[i, 1] = float(self.net.res_load.at[spec.element_idx, "p_mw"])

        self._encode_types(obs)
        return obs

    def _apply_parquet_row(self, row: pd.Series) -> None:
        for zone, col in zip([1, 2, 3, 4], ["load_z1", "load_z2", "load_z3", "load_z4"]):
            base = float(self.zone_base_loads.get(zone, 0.0))
            if base <= 0.0:
                continue
            scale = float(row.get(col, base)) / base
            for load_idx in self.zone_load_indices.get(zone, []):
                self.net.load.at[load_idx, "p_mw"] = (
                    float(self.base_net.load.at[load_idx, "p_mw"]) * scale
                )

        pv_pu = float(row.get("pv_pu", 1.0))
        for sgen_idx, rated in self.sgen_pv_map.items():
            self.net.sgen.at[sgen_idx, "p_mw"] = float(rated) * pv_pu

        wind_norm = float(row.get("wind_mw", 0.0)) / 12.0
        for sgen_idx, rated in self.sgen_wind_map.items():
            self.net.sgen.at[sgen_idx, "p_mw"] = float(rated) * wind_norm

    def _update_obs_from_parquet(
        self, row: pd.Series | None, obs: np.ndarray
    ) -> np.ndarray:
        if row is None:
            return obs

        def vpp_key(spec_vpp: str) -> int:
            value = spec_vpp.replace("_", "").upper()
            if value.endswith("1"):
                return 1
            if value.endswith("2"):
                return 2
            if value.endswith("3"):
                return 3
            return 1

        for i, spec in enumerate(self._agent_specs):
            vpp_idx = vpp_key(spec.vpp)
            obs[i, 3] = float(row.get(f"p_ref_vpp{vpp_idx}", 0.0))
            obs[i, 4] = float(row.get(f"r_as_vpp{vpp_idx}", 0.0))

            zone = int(getattr(spec, "zone", 1))
            obs[i, 11] = float(
                row.get(f"lambda_p2p_z{zone}", row.get("lambda_p2p", 0.0))
            )
            obs[i, 12] = float(
                row.get(f"lambda_as_z{zone}", row.get("lambda_as_ffr", 0.0))
            )

        hour = float(row.get("hour", 0.0))
        obs[:, 13] = float(math.sin(hour * math.pi / 12.0))
        obs[:, 14] = float(math.cos(hour * math.pi / 12.0))
        obs[:, 22] = float(row.get("freq_event_flag", 0.0))
        obs[:, 23] = float(row.get("delta_p_cont", 0.0)) / 15.705
        return obs

    def _build_env_state(self, converged: bool) -> Dict[str, Any]:
        v_bus = (
            self.net.res_bus.vm_pu.to_numpy(copy=True)
            if converged and not self.net.res_bus.empty
            else np.ones(len(self.net.bus))
        )
        agent_buses = [self.pp_idx(spec.mpc_bus) for spec in self._agent_specs]

        evcs_p_ch_min = []
        evcs_oblig = []
        evcs_features = []
        for model in self._evcs_models:
            features = model.get_obs_features()
            evcs_features.append(features)
            evcs_p_ch_min.append(float(features[3] * model.bess_mw))
            evcs_oblig.append(False)

        n_evcs = self._n_evcs
        p_flex_up = np.zeros(self._n_agents, dtype=np.float32)
        p_flex_down = np.zeros(self._n_agents, dtype=np.float32)

        for i, spec in enumerate(self._agent_specs[0:n_evcs]):
            p_set = float(self.net.sgen.at[spec.element_idx, "p_mw"])
            p_flex_up[i] = max(spec.p_rated - p_set, 0.0)
            p_flex_down[i] = max(p_set, 0.0)

        for i, spec in enumerate(self._agent_specs[n_evcs : 2 * n_evcs]):
            p_set = float(self.net.storage.at[spec.element_idx, "p_mw"])
            p_flex_up[i + n_evcs] = max(spec.p_rated - max(p_set, 0.0), 0.0)
            p_flex_down[i + n_evcs] = max(spec.p_rated + min(p_set, 0.0), 0.0)

        for i, spec in enumerate(self._agent_specs[2 * n_evcs : 3 * n_evcs]):
            p_set = -float(self.net.load.at[spec.element_idx, "p_mw"])
            p_flex_up[i + 2 * n_evcs] = max(spec.p_rated - p_set, 0.0)
            p_flex_down[i + 2 * n_evcs] = max(p_set, 0.0)

        for i, spec in enumerate(self._agent_specs[3 * n_evcs :]):
            p_set = float(self.net.sgen.at[spec.element_idx, "p_mw"])
            p_flex_up[i + 3 * n_evcs] = max(spec.p_rated - p_set, 0.0)
            p_flex_down[i + 3 * n_evcs] = max(p_set, 0.0)

        dpv_p_rated = [spec.p_rated for spec in self._agent_specs[3 * n_evcs :]]
        dpv_s_rated = [spec.q_max for spec in self._agent_specs[3 * n_evcs :]]

        evcs_s_rated = [spec.q_max for spec in self._agent_specs[0:n_evcs]]
        evcs_bess_s_rated = [spec.q_max for spec in self._agent_specs[n_evcs : 2 * n_evcs]]

        return {
            "v_bus": v_bus,
            "delta_f": float(self._freq_model.delta_f),
            "evcs_states": evcs_features,
            "evcs_p_ch_min": evcs_p_ch_min,
            "p_flex_up": p_flex_up,
            "p_flex_down": p_flex_down,
            "agent_buses": agent_buses,
            "dpv_p_rated": dpv_p_rated,
            "dpv_s_rated": dpv_s_rated,
            "evcs_s_rated": evcs_s_rated,
            "evcs_bess_s_rated": evcs_bess_s_rated,
        }

    def _get_day_row(self) -> Dict[str, float]:
        if self._day_data is None or self._day_data.empty:
            return {}
        idx = min(self.step_count, len(self._day_data) - 1)
        row = self._day_data.iloc[idx]
        return row.to_dict()

    def _compute_reward(
        self,
        env_state: Dict[str, Any],
        safe_actions: Dict[str, List[Any]],
        safety_info: Dict[str, Any],
        day_row: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        weights = {
            "track": 1.0,
            "volt": 10.0,
            "freq": 15.0,
            "as": 0.5,
            "deg": 0.3,
            "oblig": 50.0,
            "p2p": 0.2,
            "s_margin": 2.0,
            "q_revenue": 0.3,
        }

        def get_price(key: str, default: float = 1.0) -> float:
            value = day_row.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        vpp_totals = {"VPP1": 0.0, "VPP2": 0.0, "VPP3": 0.0}
        q_total = 0.0
        s_margin_pen = float(safety_info.get("s_margin_pen", 0.0))

        def normalize_vpp(vpp: str) -> str:
            return vpp.replace("_", "")

        n_evcs = self._n_evcs
        evcs_s_rated = env_state.get(
            "evcs_s_rated", [spec.q_max for spec in self._agent_specs[0:n_evcs]]
        )
        evcs_bess_s_rated = env_state.get(
            "evcs_bess_s_rated", [spec.q_max for spec in self._agent_specs[n_evcs : 2 * n_evcs]]
        )

        for idx, spec in enumerate(self._agent_specs[0:n_evcs]):
            p_curt, q = safe_actions["evcs_pv"][idx]
            p_set = spec.p_rated - float(p_curt)
            vpp_totals[normalize_vpp(spec.vpp)] += p_set
            q_total += abs(float(q))

        for idx, spec in enumerate(self._agent_specs[n_evcs : 2 * n_evcs]):
            p_set, q = safe_actions["evcs_bess"][idx]
            vpp_totals[normalize_vpp(spec.vpp)] += float(p_set)
            q_total += abs(float(q))

        for idx, spec in enumerate(self._agent_specs[2 * n_evcs : 3 * n_evcs]):
            p_set = safe_actions["evcs_v2g"][idx]
            vpp_totals[normalize_vpp(spec.vpp)] -= float(p_set)

        for idx, spec in enumerate(self._agent_specs[3 * n_evcs :]):
            p_curt, q = safe_actions["dpv"][idx]
            p_set = spec.p_rated - float(p_curt)
            vpp_totals[normalize_vpp(spec.vpp)] += p_set
            q_total += abs(float(q))

        r_track = 0.0
        for vpp_idx, vpp in enumerate(["VPP1", "VPP2", "VPP3"], start=1):
            p_ref = get_price(f"p_ref_vpp{vpp_idx}", 0.0)
            r_track -= weights["track"] * (vpp_totals[vpp] - p_ref) ** 2

        v = self.net.res_bus.vm_pu
        if v.empty:
            r_volt = 0.0
        else:
            violations = np.maximum(0.0, np.abs(v.to_numpy() - 1.0) - 0.05)
            r_volt = -weights["volt"] * float(np.sum(violations**2))

        freq_event_flag = bool(day_row.get("freq_event_flag", 0))
        if freq_event_flag:
            delta_p_cont = float(day_row.get("delta_p_cont", 0.0))
            bess_v2g_dispatch = 0.0
            for idx in range(self._n_evcs):
                bess_v2g_dispatch += float(safe_actions["evcs_bess"][idx][0])
                bess_v2g_dispatch += float(safe_actions["evcs_v2g"][idx])
            required = abs(delta_p_cont) * 0.8
            shortfall = max(0.0, required - bess_v2g_dispatch)
            r_freq = -15.0 * (shortfall / (required + 1e-6))
        else:
            r_freq = 0.0

        r_as = 0.0
        p_flex_up = env_state.get("p_flex_up", np.zeros(self._n_agents))
        lambda_as = get_price("lambda_as_ffr", 0.0)
        for vpp_idx, vpp in enumerate(["VPP1", "VPP2", "VPP3"], start=1):
            r_as_req = get_price(f"r_as_vpp{vpp_idx}", 0.0)
            vpp_agents = [
                i for i, spec in enumerate(self._agent_specs) if normalize_vpp(spec.vpp) == vpp
            ]
            p_flex_available = float(np.sum(p_flex_up[vpp_agents])) if vpp_agents else 0.0
            r_deliver = min(p_flex_available, r_as_req)
            r_as += weights["as"] * lambda_as * r_deliver

        r_deg = 0.0
        dT = 0.25
        n_evcs = self._n_evcs
        for idx, spec in enumerate(self._agent_specs[n_evcs : 2 * n_evcs]):
            p_set = float(safe_actions["evcs_bess"][idx][0])
            r_deg -= weights["deg"] * 0.02 * abs(p_set) * dT
        for idx, spec in enumerate(self._agent_specs[2 * n_evcs : 3 * n_evcs]):
            p_set = float(safe_actions["evcs_v2g"][idx])
            r_deg -= weights["deg"] * 0.05 * p_set * dT

        r_oblig = 0.0

        for idx, model in enumerate(self._evcs_models):
            p_bess = float(safe_actions["evcs_bess"][idx][0])
            p_v2g = float(safe_actions["evcs_v2g"][idx])
            p_pv = float(self._agent_specs[idx].p_rated - safe_actions["evcs_pv"][idx][0])
            evcs_result = model.step(p_bess, p_v2g, p_pv)
            if evcs_result.get("obligation_violated", False):
                r_oblig -= weights["oblig"]

        vpp_zone_price = {
            "VPP1": get_price("lambda_p2p_z1", get_price("lambda_p2p", 0.0)),
            "VPP2": get_price("lambda_p2p_z2", get_price("lambda_p2p", 0.0)),
            "VPP3": get_price("lambda_p2p_z4", get_price("lambda_p2p", 0.0)),
        }
        r_p2p = 0.0
        for vpp, total in vpp_totals.items():
            r_p2p += weights["p2p"] * vpp_zone_price[vpp] * max(0.0, total)

        r_s_margin = -weights["s_margin"] * s_margin_pen

        lambda_q = get_price("lambda_q", 1.0)
        r_q_revenue = weights["q_revenue"] * lambda_q * q_total

        total = (
            r_track
            + r_volt
            + r_freq
            + r_as
            + r_deg
            + r_oblig
            + r_p2p
            + r_s_margin
            + r_q_revenue
        )
        total = float(np.clip(total / 100.0, -10.0, 2.0))

        breakdown = {
            "r_track": float(r_track),
            "r_volt": float(r_volt),
            "r_freq": float(r_freq),
            "r_as": float(r_as),
            "r_deg": float(r_deg),
            "r_oblig": float(r_oblig),
            "r_p2p": float(r_p2p),
            "r_s_margin": float(r_s_margin),
            "r_q_revenue": float(r_q_revenue),
        }
        return total, breakdown

    def _compute_reward_from_row(
        self,
        safe_actions: Dict[str, List[Any]],
        row: pd.Series,
    ) -> Tuple[float, Dict[str, float]]:
        day_row = row.to_dict()
        dummy_env_state = {
            "p_flex_up": np.zeros(self._n_agents),
            "p_flex_down": np.zeros(self._n_agents),
            "evcs_s_rated": [spec.q_max for spec in self._agent_specs[0 : self._n_evcs]],
            "evcs_bess_s_rated": [spec.q_max for spec in self._agent_specs[self._n_evcs : 2 * self._n_evcs]],
        }
        safety_info = {"s_margin_pen": self._last_safety_info.get("s_margin_pen", 0.0)}

        reward, breakdown = self._compute_reward(dummy_env_state, safe_actions, safety_info, day_row)
        breakdown["r_volt"] = 0.0
        reward = (
            breakdown["r_track"]
            + breakdown["r_freq"]
            + breakdown["r_as"]
            + breakdown["r_deg"]
            + breakdown["r_oblig"]
            + breakdown["r_p2p"]
            + breakdown["r_s_margin"]
            + breakdown["r_q_revenue"]
        )
        reward = float(np.clip(reward / 100.0, -10.0, 2.0))
        return reward, breakdown

    def _encode_types(self, obs: np.ndarray) -> None:
        for i, spec in enumerate(self._agent_specs):
            if spec.agent_type in {"EVCS_PV"}:
                obs[i, 17:20] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif spec.agent_type in {"EVCS_BESS", "EVCS_V2G"}:
                obs[i, 17:20] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                obs[i, 17:20] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
