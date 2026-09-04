from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import math
import random
from typing import Any

import gymnasium as gym
import numpy as np
import pandapower as pp
from pandapower.converter.matpower import from_mpc

from src.env.day_context import DayContextLoader
from src.env.events import EventConfig, EventInjector
from src.env.evcs_model import EVBatteryConfig, EVBatteryModel, mpc_correction
from src.env.freq_dynamics_lti import LTITopologyFreqDynamics, FrequencyStateLTI
from src.env.IEEE123bus import convert_near_zero_branches_to_switches
from src.env.microgrid_env import _read_matpower_bus_numbers, _fix_mpc_file, _from_mpc_compat
from src.opt.tie_switch_reconfig import TieSwitchReconfiguration

# Tier 3 (secondary AGC) is owned by the VSG frequency model
# (LTITopologyFreqDynamics, via its AGC integral). The distributed
# VPP/BESS AGC is preserved as code but disabled by default to enforce
# single-integrator semantics (see agc.md for rationale).
USE_DISTRIBUTED_AGC = False


class VPPSecondaryControl:
    """AGC (secondary control) for each VPP using integral action."""

    def __init__(self, n_agents: int, K_i: float = 0.1) -> None:
        self.n_agents = n_agents
        self.K_i = K_i
        self.integral = 0.0

    def step(self, frequency: float, dt: float) -> float:
        """AGC power command from integral action.

        Args:
            frequency: Current system frequency (Hz)
            dt: Time step (s)

        Returns:
            p_agc: Power adjustment (pu) for entire VPP
        """
        error = frequency - 50.0
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -20, 20))
        p_agc = -self.K_i * self.integral
        p_agc = float(np.clip(p_agc, -1.0, 1.0))
        return p_agc


class BatterySecondaryControl:
    """AGC specifically for battery units (faster integral action)."""

    def __init__(self, n_bess: int, K_i: float = 0.5) -> None:
        self.n_bess = n_bess
        self.K_i = K_i
        self.integral = 0.0

    def step(self, frequency: float, dt: float) -> float:
        """AGC power command from integral action (higher gain than VPP).

        Args:
            frequency: Current system frequency (Hz)
            dt: Time step (s)

        Returns:
            p_agc: Power adjustment (pu) for battery fleet
        """
        error = frequency - 50.0
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -10, 10))
        p_agc = -self.K_i * self.integral
        p_agc = float(np.clip(p_agc, -1.0, 1.0))
        return p_agc


class MicrogridEnvDual(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        placement_path: str | Path,
        mpc_path: str | Path,
        seed: int = 42,
        topology_cache: list[tuple[Any, np.ndarray, set[int]]] | None = None,
        topology_cache_path: str | Path | None = None,
        precomputed_dir: str | Path = "data/precomputed_365d_97to67",
        day_split: str = "train",
        ffr_mode: str = "droop",
    ) -> None:
        super().__init__()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # FFR control mode:
        #   "droop":      Classical per-DER droop FFR (baseline). RL action provides minor adjustment.
        #   "mappo":      RL directly commands per-DER FFR power (single-action, replaces droop).
        #   "mappo_dual": RL outputs (P_ref, K_droop) per DER (proposed AM dual-product).
        if ffr_mode not in ("droop", "mappo", "mappo_dual"):
            raise ValueError(f"ffr_mode must be 'droop'|'mappo'|'mappo_dual', got '{ffr_mode}'")
        self.ffr_mode = ffr_mode

        placement_path = Path(placement_path)
        mpc_path = Path(mpc_path)

        with open(placement_path, encoding="utf-8") as f:
            self.placement = json.load(f)

        fixed_mpc = self._pre_fix_mpc(mpc_path)
        base_net = _from_mpc_compat(fixed_mpc)
        bus_numbers = _read_matpower_bus_numbers(Path(fixed_mpc))
        base_net.bus["bus_id"] = bus_numbers
        convert_near_zero_branches_to_switches(base_net, bus_numbers=bus_numbers)

        self._bus_map = {
            int(bus_id): int(idx)
            for idx, bus_id in base_net.bus["bus_id"].items()
            if not np.isnan(bus_id)
        }

        base_net.load["p_mw"] = base_net.load["p_mw"] * 8.5
        base_net.load["q_mvar"] = base_net.load["q_mvar"] * 8.5
        base_net.bus["vn_kv"] = 22.0

        self._add_reconfiguration_ties(base_net)

        self._agent_specs: list[dict[str, Any]] = []
        self._inject_assets(base_net)
        self._order_agents()

        self.n_agents = len(self._agent_specs)
        if self.n_agents != 41:
            raise ValueError(f"Expected 41 agents from placement, got {self.n_agents}")

        self._evcs_count = len(self.placement.get("evcs", []))

        # VPP-level FFR: 3 VPPs instead of 18 individual droop agents
        # VPP_1: E1-E3 (agents 0-2 for EVCS, 9-11 for BESS, 18-20 for V2G)
        # VPP_2: E4-E6 (agents 3-5 for EVCS, 12-14 for BESS, 21-23 for V2G)
        # VPP_3: E7-E9 (agents 6-8 for EVCS, 15-17 for BESS, 24-26 for V2G)
        self._vpp_ids = ["VPP_1", "VPP_2", "VPP_3"]
        self._vpp_droop_agents: dict[str, list[int]] = {
            "VPP_1": list(range(9, 12)) + list(range(18, 21)),   # BESS + V2G for E1-E3
            "VPP_2": list(range(12, 15)) + list(range(21, 24)),  # BESS + V2G for E4-E6
            "VPP_3": list(range(15, 18)) + list(range(24, 27)),  # BESS + V2G for E7-E9
        }
        self._n_vpps = len(self._vpp_ids)

        self.observation_space_fast = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents, 7),
            dtype=np.float32,
        )
        self.observation_space_slow = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents, 7),
            dtype=np.float32,
        )

        self.action_space_fast = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(44,),  # 41 agents + 3 VPP k_droop
            dtype=np.float32,
        )
        self.action_space_slow = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(82,),
            dtype=np.float32,
        )

        self.base_net = deepcopy(base_net)
        self.net = deepcopy(base_net)

        self.event_injector = EventInjector(seed=seed)
        # Time-step constants (must precede LTI init since dt_fast is passed in)
        self.dt_fast_s = 1.0
        self.dt_ode_s = 0.01
        self.n_ode_substeps = max(1, int(round(self.dt_fast_s / self.dt_ode_s)))
        # VSG topology-aware frequency dynamics: the single source of truth for
        # frequency/RoCoF/inertia (legacy synchronous-generator model removed).
        self.use_lti_freq = True
        self.freq_dyn_lti = LTITopologyFreqDynamics(
            placement=self.placement,
            base_net=self.base_net,
            bus_map=self._bus_map,
            f0=50.0,
            dt_fast=self.dt_fast_s,
            tau_default=0.1,
            agc_ki=0.05,
            cache_k_bins=5,
            use_pseudoinverse=True,
        )
        self._current_topology_id: int = 0
        # When True, reset() ignores reconfiguration and always returns the nominal
        # base feeder (no tie-switch changes). Used by the "train-on-base, eval-on-
        # all-reconfig" generalization protocol so every cached reconfig topology is
        # genuinely unseen at eval time. Default False = normal reconfig behaviour.
        self.fixed_base_topology: bool = False
        self.reconfig = TieSwitchReconfiguration(self.base_net, seed=seed)
        self.context_loader = DayContextLoader(
            precomputed_dir=precomputed_dir, seed=seed, split=day_split
        )
        self.day_ctx = None
        self.day_step = 0
        self._last_topology_fallback_reason: str | None = None

        cache_path = Path(topology_cache_path) if topology_cache_path is not None else Path("data/tie_switch_cache.pkl")
        if topology_cache is not None:
            sanitized_cache, _ = self.reconfig._sanitize_cache_entries(topology_cache)
            self.reconfig._cache = sanitized_cache
            self.reconfig._active_topologies = len(sanitized_cache)
            self.reconfig._optimal_cache.clear()
            if (not self.reconfig._cache) or (not self._is_topology_cache_compatible(self.reconfig._cache)):
                self.reconfig.generate_scenarios(n=24)
                self.reconfig.save_cache(cache_path)
        else:
            loaded = self.reconfig.load_cache(cache_path)
            if (not loaded) or (not self._is_topology_cache_compatible(self.reconfig._cache)):
                self.reconfig.generate_scenarios(n=24)
                self.reconfig.save_cache(cache_path)

        self.fast_step_count = 0
        self.slow_step_count = 0
        self.episode_done = False
        self.current_event = None
        self.edge_index = np.zeros((2, 0), dtype=np.int64)
        self.current_open_set: set[int] = set()
        self.event_delta_p_pu = 0.0
        self._event_location_pp: int | None = None
        self._topology_just_changed = False

        # GFM bus mapping for topology-aware frequency dynamics
        self._gfm_bus_map: dict[str, int] = {}
        for gfm_id, gfm_data in self.placement.get("gfm", {}).items():
            bus_id = int(gfm_data.get("bus", 0))
            pp_idx = self._bus_map.get(bus_id, -1)
            if pp_idx >= 0:
                self._gfm_bus_map[gfm_id] = pp_idx

        self.soc = np.full((self.n_agents,), 0.5, dtype=np.float32)
        self.p_set = np.zeros((self.n_agents,), dtype=np.float32)
        self.q_set = np.zeros((self.n_agents,), dtype=np.float32)
        self.delta_p_set = np.zeros((self.n_agents,), dtype=np.float32)
        self.k_droop_vpp = np.zeros((self._n_vpps,), dtype=np.float32)  # 3 VPP-level droop coefficients
        # Slew-rate buffers (reset() refreshes them per episode).
        self._prev_a_p_fast = np.zeros((self.n_agents,), dtype=np.float32)
        self._prev_k_droop_fast = np.zeros((self._n_vpps,), dtype=np.float32)

        # FFR activation trigger thresholds (ENTSO-E compliant)
        # Ref: ENTSO-E Network Code RfG, SO GL, Nordic FFR specifications
        # FCR activation: ±200 mHz (Continental Europe)
        # FFR for low-inertia/islanded: stricter RoCoF trigger (Nordic: 1.0 Hz/s)
        self.ffr_threshold_df = 0.2      # Hz - ENTSO-E FCR activation threshold
        self.ffr_threshold_rocof = 1.0   # Hz/s - Nordic FFR RoCoF trigger
        self.ffr_deactivation_df = 0.1   # Hz - hysteresis for deactivation (50% of threshold)
        self.ffr_active = False          # Current FFR activation state
        self.ffr_activation_count = 0    # Number of activations in episode
        self.ffr_energy_delivered = 0.0  # MWh delivered during FFR

        # UFLS thresholds per ENTSO-E emergency standards
        self.ufls_stage1 = 49.0  # Hz - first load shedding stage
        self.ufls_stage2 = 48.5  # Hz - second load shedding stage
        self.ufls_stage3 = 48.0  # Hz - critical stage

        # AGC (Secondary Control) - Distributed integral action for frequency recovery
        # Total system capacity ≈ 15.7 MW (3 VPPs × 5 MW + Battery × 2 MW)
        # K_i distributed proportionally by capacity for effective secondary control
        # Higher K_i provides faster recovery: K_i=0.3-0.5 for ~40-60s recovery time
        total_capacity_mw = 15.7
        k_i_sys = 0.4  # Increased from 0.15 for stronger recovery
        vpp_capacity_mw = 5.0

        self.vpp_agc = [
            VPPSecondaryControl(n_agents=6, K_i=k_i_sys * (vpp_capacity_mw / total_capacity_mw))
            for _ in range(self._n_vpps)
        ]
        self.bess_agc = BatterySecondaryControl(n_bess=18, K_i=k_i_sys * (2.0 / total_capacity_mw))

        # Inter-VPP coordination: SOC-weighted FFR contribution
        self._vpp_soc_weight = np.ones(self._n_vpps, dtype=np.float32)

        # Feasibility feedback: track constraint violations
        self.feasibility_violations = {
            "voltage": 0,      # Count of voltage violations
            "soc": 0,          # Count of SOC limit violations
            "power": 0,        # Count of power limit violations
            "thermal": 0,      # Count of thermal/line limit violations
        }
        self.feasibility_ok = True  # Overall feasibility flag

        self.zone_lmp = {1: 45.0, 2: 50.0, 3: 55.0, 4: 60.0}
        self._slow_baseline = np.zeros((82,), dtype=np.float32)
        # Hi-resolution swing-eq trace (one sample per ODE sub-step)
        self._hires_df: list[float] = []
        # Sub-steps per fast-step for the optional hi-res plotting trace (0/1 = off:
        # one COI sample per fast-step). Eval sets this to e.g. 50 to resolve the
        # true sub-second nadir transient; training leaves it at 0 (untouched).
        self.hires_substeps: int = 0
        # AM-side prices and L1 commitments (populated by _apply_day_context)
        self.lambda_as_ffr: float = 10.0
        self.lambda_as_pfr: float = 5.0
        self.lambda_as_sfr: float = 3.0
        self.zone_lambda_as: dict[int, float] = {1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0}
        self.r_as_target_vpp = np.zeros(3, dtype=np.float32)
        self.p_ref_target_vpp = np.zeros(3, dtype=np.float32)

        self._agent_elements = np.asarray([str(spec["element"]) for spec in self._agent_specs], dtype=object)
        self._agent_idx_labels = np.asarray([int(spec["idx"]) for spec in self._agent_specs], dtype=np.int64)
        self._agent_p_rated = np.asarray([max(float(spec["p_rated"]), 1e-6) for spec in self._agent_specs], dtype=np.float32)
        self._agent_bus_pp = np.asarray([self._pp_idx(int(spec["bus"])) for spec in self._agent_specs], dtype=np.int64)

        self._mask_sgen_agents = self._agent_elements == "sgen"
        self._mask_storage_agents = self._agent_elements == "storage"
        self._mask_load_agents = self._agent_elements == "load"

        # Multi-DER type indices for FFR aggregation
        self._bess_indices: list[int] = []
        self._v2g_indices: list[int] = []
        self._pv_indices: list[int] = []
        # Mask: agent is a PV/wind injection (sgen-side, generation-only).
        # PV/Wind can only DOWN-regulate (curtail tạm thời) — their delta_p must
        # be non-positive, since they already run at MPPT under baseline dispatch.
        _is_pv = np.zeros(self.n_agents, dtype=bool)
        for i, spec in enumerate(self._agent_specs):
            agent_type = spec.get("type", "")
            if "BESS" in agent_type:
                self._bess_indices.append(i)
            elif "V2G" in agent_type:
                self._v2g_indices.append(i)
            elif "PV" in agent_type or "DPV" in agent_type or "Wind" in agent_type or "wind" in agent_type:
                self._pv_indices.append(i)
                _is_pv[i] = True
        self._mask_pv_agents = _is_pv
        # Subset of sgen positions that are PV/wind (for non-positive delta clipping).
        self._mask_pv_among_sgen = _is_pv[self._mask_sgen_agents]

        # Per-DER droop coefficients for FFR
        # k_droop_i: droop gain for DER i (pu power per Hz deviation)
        # P_ffr_i = -k_droop_i * delta_f * P_rated_i / S_BASE
        self._k_droop_bess = np.zeros(len(self._bess_indices), dtype=np.float32)
        self._k_droop_v2g = np.zeros(len(self._v2g_indices), dtype=np.float32)
        for i, idx in enumerate(self._bess_indices):
            # BESS: higher droop gain (faster response)
            p_rated = self._agent_p_rated[idx]
            self._k_droop_bess[i] = 0.15 * p_rated  # Base droop scaled by capacity
        for i, idx in enumerate(self._v2g_indices):
            # V2G: moderate droop gain
            p_rated = self._agent_p_rated[idx]
            self._k_droop_v2g[i] = 0.10 * p_rated  # Lower than BESS

        # Per-DER K_droop bounds for "mappo_dual" mode (units: MW/Hz)
        # Mapping per proposal §2: K_i = R_i * P_rated_i with R_i in [R_min, R_max] per type.
        # PV (non-dispatchable) cannot provide FFR -> bounds collapse to 0.
        # R_min = 0 for ALL types: the lower bound is no-participation (K=0), not a
        # mandated droop floor. This (a) lets a_K=-1 express a TRUE no-FFR baseline,
        # and (b) gives the dual policy the full [0, K_max] range (more freedom, and
        # symmetric with the No-FFR / Fixed-Droop baselines that now share this exact
        # action space and coupling). r_max unchanged.
        self._k_droop_min_per_agent = np.zeros(self.n_agents, dtype=np.float32)
        self._k_droop_max_per_agent = np.zeros(self.n_agents, dtype=np.float32)
        for i, spec in enumerate(self._agent_specs):
            atype = spec.get("type", "")
            p_rated = float(self._agent_p_rated[i])
            if "BESS" in atype:
                r_min, r_max = 0.0, 0.30
            elif "V2G" in atype:
                r_min, r_max = 0.0, 0.20
            elif "DPV" in atype:
                r_min, r_max = 0.0, 0.15
            else:  # EVCS_PV (non-dispatchable PV) — no FFR
                r_min, r_max = 0.0, 0.0
            self._k_droop_min_per_agent[i] = r_min * p_rated
            self._k_droop_max_per_agent[i] = r_max * p_rated

        # State buffers exposed to obs builder (set each step in mappo_dual path)
        self._p_ref_last = np.zeros(self.n_agents, dtype=np.float32)        # a_P_i in [-1, 1]
        self._k_droop_last = np.zeros(self.n_agents, dtype=np.float32)      # MW/Hz, post SoC mask

        self._sgen_agent_pos = np.zeros((0,), dtype=np.int64)
        self._storage_agent_pos = np.zeros((0,), dtype=np.int64)
        self._load_agent_pos = np.zeros((0,), dtype=np.int64)
        self._zone_lmp_vec = np.zeros((self.n_agents,), dtype=np.float32)
        self._k_p_evcs = 0.1
        self._k_i_evcs = 0.05
        self._k_p_dpv = 0.05
        self._k_i_dpv = 0.02

        self._batt_agent_indices = list(range(9, 27))
        self.batt_models = [
            EVBatteryModel(
                EVBatteryConfig(
                    E_cap_kwh=50.0,
                    P_rated_kw=50.0,
                    SoC_min=0.20,
                    SoC_max=0.90,
                )
            )
            for _ in range(len(self._batt_agent_indices))
        ]
        self._batt_soc = np.full((len(self._batt_agent_indices),), 0.5, dtype=np.float32)
        self._batt_vbat = np.full((len(self._batt_agent_indices),), 400.0, dtype=np.float32)
        self._batt_ib = np.zeros((len(self._batt_agent_indices),), dtype=np.float32)
        self.w_soc = 5.0
        self.w_dcob = 1.0
        self.w_v2g = 0.1

    def _is_topology_cache_compatible(self, cache: list[tuple[Any, np.ndarray, set[int]]]) -> bool:
        if not cache:
            return False
        # Reconfiguration ties present in the current base_net (e.g. tie_54_94) must
        # also be present in every cached net — otherwise the cache predates the tie
        # augmentation and must be regenerated so the reroute capability is exercised.
        base_ties = {
            str(self.base_net.line.at[i, "name"])
            for i in self.base_net.line.index
            if str(self.base_net.line.at[i, "name"]).lower().startswith("tie")
        }
        for item in cache:
            try:
                net_i = item[0]
                sgen_idx = {int(i) for i in net_i.sgen.index.to_numpy(dtype=np.int64, copy=False)}
                storage_idx = {int(i) for i in net_i.storage.index.to_numpy(dtype=np.int64, copy=False)}
                load_idx = {int(i) for i in net_i.load.index.to_numpy(dtype=np.int64, copy=False)}
                cached_ties = {
                    str(net_i.line.at[i, "name"])
                    for i in net_i.line.index
                    if str(net_i.line.at[i, "name"]).lower().startswith("tie")
                }
            except Exception:
                return False

            if base_ties - cached_ties:
                return False

            if not self.reconfig._is_topology_valid(deepcopy(net_i), run_power_flow=True):
                return False

            for spec in self._agent_specs:
                elm = str(spec["element"])
                idx = int(spec["idx"])
                if elm == "sgen" and idx not in sgen_idx:
                    return False
                if elm == "storage" and idx not in storage_idx:
                    return False
                if elm == "load" and idx not in load_idx:
                    return False
        return True

    def _find_connected_gfms(self) -> set[str]:
        """Determine which GFMs are electrically connected to main network.

        Uses NetworkX to find connected components and checks which GFM buses
        are in the same component as the main slack bus (G1 @ bus 114).
        """
        import networkx as nx

        G = nx.Graph()
        for _, row in self.net.bus.iterrows():
            G.add_node(int(row.name))

        for _, row in self.net.line.iterrows():
            if row.get("in_service", True):
                G.add_edge(int(row["from_bus"]), int(row["to_bus"]))

        if len(self.net.switch) > 0:
            for _, row in self.net.switch.iterrows():
                if row.get("closed", False) and row.get("et") == "b":
                    G.add_edge(int(row["bus"]), int(row["element"]))

        # Find component containing G1 (main slack)
        g1_pp_idx = self._gfm_bus_map.get("G1", -1)
        if g1_pp_idx < 0 or g1_pp_idx not in G:
            return set(self._gfm_bus_map.keys())

        main_component = set(nx.node_connected_component(G, g1_pp_idx))

        connected_gfms = set()
        for gfm_id, pp_idx in self._gfm_bus_map.items():
            if pp_idx in main_component:
                connected_gfms.add(gfm_id)

        return connected_gfms

    def _update_freq_dyn_topology(self) -> None:
        """Update frequency dynamics model based on current network topology."""
        connected_gfms = self._find_connected_gfms()
        self.freq_dyn_lti.update_topology(connected_gfms)

    @staticmethod
    def _pre_fix_mpc(mpc_path: Path) -> str:
        fixed = mpc_path.with_name(f"{mpc_path.stem}_ppfix.m")
        if not fixed.exists():
            _fix_mpc_file(str(mpc_path))
        return str(fixed)

    def _pp_idx(self, mpc_id: int) -> int:
        return self._bus_map[int(mpc_id)]

    def _add_reconfiguration_ties(self, net: pp.pandapowerNet) -> None:
        """Add the IEEE-123 normally-open reconfiguration tie 54-94 as a real line.

        The standard IEEE-123 switch table lists 54-94 as a normally-open tie — the
        genuine loop-forming reconfiguration switch of the single feeder (verified
        against feeder123/switch data.xls). The reduced MATPOWER source omits it.
        It is modeled as a real LINE (nodes 54 and 94 span physical distance, so it
        cannot be a zero-Z et="b" bus-bus switch), normally OUT of service. The
        reconfiguration engine toggles its in_service flag, and the radiality (forest)
        gate pairs closing it with opening a sectionalizer on the resulting loop.
        151-300 (the other genuine IEEE-123 tie) already exists as an et="b" switch
        and is handled by the engine, so both real single-feeder ties are covered.
        Adds no buses, so DER placement / agent indices are unchanged.
        """
        # Standard IEEE-123 switch/tie set. Four of these (18-135, 60-160,
        # 97-197, 151-300) already exist as et="b" sectionalizer switches and are
        # engine candidates; 250-251/95-195/300-350/450-451 reference buses absent
        # from the reduced MATPOWER model. So this adds the remaining genuine ties
        # (54-94, 13-152) as real lines. Pairs that lack a bus, or that already
        # exist as a bus-bus switch, are skipped so candidates are not duplicated.
        ieee123_ties = [(13, 152), (18, 135), (250, 251), (95, 195), (54, 94),
                        (151, 300), (300, 350), (60, 160), (97, 197), (450, 451)]
        existing_sw = set()
        if hasattr(net, "switch") and len(net.switch) > 0:
            for _i in net.switch.index:
                if str(net.switch.at[_i, "et"]) == "b":
                    existing_sw.add(frozenset((int(net.switch.at[_i, "bus"]),
                                               int(net.switch.at[_i, "element"]))))
        ties = []
        for a_id, b_id in ieee123_ties:
            if a_id not in self._bus_map or b_id not in self._bus_map:
                continue
            if frozenset((self._bus_map[a_id], self._bus_map[b_id])) in existing_sw:
                continue
            ties.append((a_id, b_id))
        if "r_ohm_per_km" in net.line.columns and len(net.line) > 0:
            r_rep = max(float(net.line["r_ohm_per_km"].median()), 0.05)
            x_rep = max(float(net.line["x_ohm_per_km"].median()), 0.10)
        else:
            r_rep, x_rep = 0.05, 0.10
        for a_id, b_id in ties:
            if a_id in self._bus_map and b_id in self._bus_map:
                pp.create_line_from_parameters(
                    net,
                    from_bus=self._bus_map[a_id],
                    to_bus=self._bus_map[b_id],
                    length_km=0.2,
                    r_ohm_per_km=r_rep,
                    x_ohm_per_km=x_rep,
                    c_nf_per_km=0.0,
                    max_i_ka=0.4,
                    name=f"tie_{a_id}_{b_id}",
                    in_service=False,
                )

    def _inject_assets(self, net: pp.pandapowerNet) -> None:
        gfm_buses: list[int] = []

        for ev in self.placement.get("evcs", []):
            bus = self._pp_idx(ev["bus"])
            pv_idx = pp.create_sgen(net, bus=bus, p_mw=ev["pv_mw"], q_mvar=0.0, controllable=True)
            bess_idx = pp.create_storage(
                net,
                bus=bus,
                p_mw=0.0,
                max_p_mw=ev["bess_mw"],
                min_p_mw=-ev["bess_mw"],
                max_e_mwh=ev["bess_mwh"],
                soc_percent=50.0,
            )
            v2g_idx = pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0, controllable=True)

            inv_mva = float(ev.get("inverter_mva", ev["pv_mw"]))
            q_max_pv = inv_mva * math.sin(math.acos(0.9))
            q_max_bess = float(ev.get("inverter_mva", ev["bess_mw"])) * math.sin(math.acos(0.9))

            self._agent_specs.append({"type": "EVCS_PV", "zone": int(ev.get("zone", 1)), "bus": int(ev["bus"]), "element": "sgen", "idx": int(pv_idx), "p_rated": float(ev["pv_mw"]), "q_max": float(q_max_pv)})
            self._agent_specs.append({"type": "EVCS_BESS", "zone": int(ev.get("zone", 1)), "bus": int(ev["bus"]), "element": "storage", "idx": int(bess_idx), "p_rated": float(ev["bess_mw"]), "q_max": float(q_max_bess)})
            self._agent_specs.append({"type": "EVCS_V2G", "zone": int(ev.get("zone", 1)), "bus": int(ev["bus"]), "element": "load", "idx": int(v2g_idx), "p_rated": float(ev["v2g_mw"]), "q_max": 0.0})

        for pv in self.placement.get("dpv", []):
            bus = self._pp_idx(pv["bus"])
            pv_idx = pp.create_sgen(net, bus=bus, p_mw=pv["mw"], q_mvar=0.0, controllable=True)
            inv_mva = float(pv.get("inverter_mva", pv.get("sn_mva", pv["mw"])))
            q_max = inv_mva * math.sin(math.acos(0.9))
            self._agent_specs.append({"type": "DPV", "zone": int(pv.get("zone", 1)), "bus": int(pv["bus"]), "element": "sgen", "idx": int(pv_idx), "p_rated": float(pv["mw"]), "q_max": float(q_max)})

        for gfm_data in self.placement.get("gfm", {}).values():
            if not isinstance(gfm_data, dict):
                continue
            bus_id = int(gfm_data.get("bus", 0))
            if bus_id <= 0:
                continue
            pp_bus = self._bus_map.get(bus_id)
            if pp_bus is None:
                continue
            gfm_buses.append(int(pp_bus))

        # from_mpc() auto-creates an ext_grid at the MATPOWER REF bus (bus 149,
        # the substation node) with +-200 MW/MVAr limits -- an unconstrained
        # infinite-bus artifact that contradicts the islanded, GFM-only design
        # (this feeder has no grid tie; G1 at bus 114 is the intended slack
        # anchor). Drop any non-GFM ext_grid before wiring up the real GFM
        # sources so only the placement's finite-capacity GFM units can supply
        # power in the pandapower model used for reconfiguration validity, PF
        # severity checks, and the LTI operating-point linearization.
        gfm_bus_set = set(gfm_buses)
        if not net.ext_grid.empty:
            net.ext_grid = net.ext_grid[net.ext_grid["bus"].astype(int).isin(gfm_bus_set)].reset_index(drop=True)

        existing_ext_grid_buses = set(net.ext_grid["bus"].astype(int).tolist()) if not net.ext_grid.empty else set()
        for gfm_bus in gfm_buses:
            if gfm_bus in existing_ext_grid_buses:
                continue
            pp.create_ext_grid(net, bus=gfm_bus, vm_pu=1.0, name=f"gfm_{gfm_bus}")

    def _order_agents(self) -> None:
        evcs = self.placement.get("evcs", [])
        dpv = self.placement.get("dpv", [])

        def find_spec(agent_type: str, bus: int) -> dict[str, Any]:
            for spec in self._agent_specs:
                if spec["type"] == agent_type and int(spec["bus"]) == int(bus):
                    return spec
            raise KeyError(f"Agent spec not found for {agent_type} at bus {bus}")

        ordered: list[dict[str, Any]] = []
        for ev in evcs:
            ordered.append(find_spec("EVCS_PV", ev["bus"]))
        for ev in evcs:
            ordered.append(find_spec("EVCS_BESS", ev["bus"]))
        for ev in evcs:
            ordered.append(find_spec("EVCS_V2G", ev["bus"]))
        for pv in dpv:
            ordered.append(find_spec("DPV", pv["bus"]))
        self._agent_specs = ordered

    def _sample_zone_lmp(self) -> None:
        self.zone_lmp = {
            1: float(self.np_rng.uniform(40.0, 60.0)),
            2: float(self.np_rng.uniform(45.0, 65.0)),
            3: float(self.np_rng.uniform(50.0, 70.0)),
            4: float(self.np_rng.uniform(42.0, 62.0)),
        }
        self._zone_lmp_vec = np.asarray(
            [self.zone_lmp.get(int(spec["zone"]), 50.0) for spec in self._agent_specs],
            dtype=np.float32,
        )

    def set_slow_baseline(self, action_slow: np.ndarray) -> None:
        a = np.asarray(action_slow, dtype=np.float32).reshape(-1)
        if a.shape[0] != 82:
            raise ValueError(f"action_slow baseline must have shape (82,), got {a.shape}")
        self._slow_baseline = a.copy()

    def _zone_prices_vector(self) -> np.ndarray:
        return self._zone_lmp_vec

    def _agent_p_net(self) -> np.ndarray:
        p_net = np.zeros((self.n_agents,), dtype=np.float32)
        if np.any(self._mask_sgen_agents):
            p_net[self._mask_sgen_agents] = self.net.sgen["p_mw"].to_numpy(dtype=np.float32, copy=False)[self._sgen_agent_pos]
        if np.any(self._mask_storage_agents):
            p_net[self._mask_storage_agents] = self.net.storage["p_mw"].to_numpy(dtype=np.float32, copy=False)[self._storage_agent_pos]
        if np.any(self._mask_load_agents):
            p_net[self._mask_load_agents] = -self.net.load["p_mw"].to_numpy(dtype=np.float32, copy=False)[self._load_agent_pos]
        return p_net

    def _agent_q_net(self) -> np.ndarray:
        q_net = np.zeros((self.n_agents,), dtype=np.float32)
        for i, spec in enumerate(self._agent_specs):
            elm = spec["element"]
            idx = spec["idx"]
            if elm == "sgen":
                q_net[i] = float(self.net.sgen.at[idx, "q_mvar"])
            elif elm == "storage":
                q_net[i] = float(self.net.storage.at[idx, "q_mvar"]) if "q_mvar" in self.net.storage.columns else 0.0
            else:
                q_net[i] = -float(self.net.load.at[idx, "q_mvar"])
        return q_net

    def _agent_v_bus(self) -> np.ndarray:
        if hasattr(self.net, "res_bus") and not self.net.res_bus.empty and "vm_pu" in self.net.res_bus.columns:
            v = np.asarray(
                [float(self.net.res_bus.at[self._pp_idx(spec["bus"]), "vm_pu"]) for spec in self._agent_specs],
                dtype=np.float32,
            )
            return np.nan_to_num(v, nan=1.0, posinf=1.1, neginf=0.9)
        return np.ones((self.n_agents,), dtype=np.float32)

    def _compute_virtual_md(self, agent_idx: int) -> tuple[float, float]:
        if agent_idx < self._evcs_count:
            kp = self._k_p_evcs
            ki = self._k_i_evcs
        else:
            kp = self._k_p_dpv
            ki = self._k_i_dpv
        m_j = 1.0 / kp
        d_j = ki / kp
        return float(m_j), float(d_j)

    def _build_obs_fast(self) -> np.ndarray:
        # Frequency state from the VSG model (per-bus Δf/RoCoF).
        st = self.freq_dyn_lti.get_state()
        use_per_bus = True

        p_net = np.nan_to_num(self._agent_p_net(), nan=0.0, posinf=1e3, neginf=-1e3)
        obs = np.zeros((self.n_agents, 7), dtype=np.float32)

        # P0 fix (Bug 1): rescale Δf/RoCoF refs to match observed event magnitudes
        # (nadir up to ~3 Hz, RoCoF up to ~3.5 Hz/s under severe contingencies).
        # Prior /0.5 and /1.0 saturated obs for any meaningful event, making
        # policy blind to severity above those thresholds.
        OBS_DELTA_F_REF = 3.0   # Hz
        OBS_ROCOF_REF = 3.5     # Hz/s
        if use_per_bus and hasattr(st, 'delta_f_per_bus'):
            # Per-bus frequency: each agent gets its local GFM's Δω
            for i in range(self.n_agents):
                agent_pp_idx = self._agent_specs[i].get("pp_bus_idx", 0)
                gfm_idx = self.freq_dyn_lti.get_gfm_bus_idx(agent_pp_idx)
                obs[i, 0] = np.float32(np.clip(st.delta_f_per_bus[gfm_idx] / OBS_DELTA_F_REF, -1.0, 1.0))
                obs[i, 1] = np.float32(np.clip(st.rocof_per_bus[gfm_idx] / OBS_ROCOF_REF, -1.0, 1.0))
        else:
            # Legacy scalar broadcast
            obs[:, 0] = np.float32(np.clip(float(st.delta_f_hz) / OBS_DELTA_F_REF, -1.0, 1.0))
            obs[:, 1] = np.float32(np.clip(float(st.rocof_hz_s) / OBS_ROCOF_REF, -1.0, 1.0))

        # P2 fix (Bug 5): normalize p_net per-agent by P_rated (was clipped raw MW
        # spanning [-1000, 1000] which dominated other [-1, 1] channels before
        # RunningNormalizer catches up).
        p_rated_safe = np.maximum(self._agent_p_rated.astype(np.float32), 0.1)
        obs[:, 2] = np.clip(p_net.astype(np.float32) / p_rated_safe, -2.0, 2.0)
        # P2 fix (Bug 5): scale zone_lmp by 100 €/MWh (typical price magnitude)
        obs[:, 4] = np.clip(self._zone_lmp_vec.astype(np.float32) / 100.0, 0.0, 5.0)
        for i in range(self.n_agents):
            if i in self._batt_agent_indices:
                batt_idx = i - self._batt_agent_indices[0]
                ev = self.batt_models[batt_idx]
                dcob_norm = min(ev.dcob() / ev.cfg.P_rated_kw, 1.0)
            else:
                dcob_norm = 0.0
            obs[i, 3] = np.float32(dcob_norm)
            m_j, d_j = self._compute_virtual_md(i)
            obs[i, 5] = np.float32(m_j / 20.0)
            obs[i, 6] = np.float32(d_j / 2.0)
        return np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)

    def _apply_day_context(self, day_ctx, step: int) -> None:
        if day_ctx is None or len(day_ctx) == 0:
            return
        step_idx = int(np.clip(step, 0, len(day_ctx) - 1))
        row = day_ctx.iloc[step_idx]
        self.zone_lmp = {
            1: float(row.get("lambda_p2p_z1", row.get("lambda_p2p", 50.0))),
            2: float(row.get("lambda_p2p_z2", row.get("lambda_p2p", 50.0))),
            3: float(row.get("lambda_p2p", 50.0)),
            4: float(row.get("lambda_p2p_z4", row.get("lambda_p2p", 50.0))),
        }
        self._zone_lmp_vec = np.asarray(
            [self.zone_lmp.get(int(spec["zone"]), 50.0) for spec in self._agent_specs],
            dtype=np.float32,
        )

        # AM-side prices and L1 commitments (used by reward / eval economics)
        self.lambda_as_ffr = float(row.get("lambda_as_ffr", 10.0))
        self.lambda_as_pfr = float(row.get("lambda_as_pfr", 5.0))
        self.lambda_as_sfr = float(row.get("lambda_as_sfr", 3.0))
        self.zone_lambda_as = {
            1: float(row.get("lambda_as_z1", self.lambda_as_ffr)),
            2: float(row.get("lambda_as_z2", self.lambda_as_ffr)),
            3: float(row.get("lambda_as_z2", self.lambda_as_ffr)),
            4: float(row.get("lambda_as_z4", self.lambda_as_ffr)),
        }
        self.r_as_target_vpp = np.asarray(
            [
                float(row.get("r_as_vpp1", 0.0)),
                float(row.get("r_as_vpp2", 0.0)),
                float(row.get("r_as_vpp3", 0.0)),
            ],
            dtype=np.float32,
        )
        self.p_ref_target_vpp = np.asarray(
            [
                float(row.get("p_ref_vpp1", 0.0)),
                float(row.get("p_ref_vpp2", 0.0)),
                float(row.get("p_ref_vpp3", 0.0)),
            ],
            dtype=np.float32,
        )

    def _build_obs_slow(self) -> np.ndarray:
        v_bus = np.nan_to_num(self._agent_v_bus(), nan=1.0, posinf=1.1, neginf=0.9)
        p_net = np.nan_to_num(self._agent_p_net(), nan=0.0, posinf=1e3, neginf=-1e3)
        q_net = np.nan_to_num(self._agent_q_net(), nan=0.0, posinf=1e3, neginf=-1e3)
        price = np.nan_to_num(self._zone_prices_vector(), nan=50.0, posinf=100.0, neginf=0.0)
        obs = np.zeros((self.n_agents, 7), dtype=np.float32)
        for i in range(self.n_agents):
            if i in self._batt_agent_indices:
                batt_idx = i - self._batt_agent_indices[0]
                ev = self.batt_models[batt_idx]
                dcob_norm = min(ev.dcob() / ev.cfg.P_rated_kw, 1.0)
            else:
                dcob_norm = 0.0
            m_j, d_j = self._compute_virtual_md(i)
            obs[i, 0] = np.float32(v_bus[i] - 1.0)
            obs[i, 1] = np.float32(q_net[i])
            obs[i, 2] = np.float32(p_net[i])
            obs[i, 3] = np.float32(dcob_norm)
            obs[i, 4] = np.float32(price[i])
            obs[i, 5] = np.float32(m_j / 20.0)
            obs[i, 6] = np.float32(d_j / 2.0)
        return np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        self.net = deepcopy(self.base_net)
        self.fast_step_count = 0
        self.slow_step_count = 0
        self.episode_done = False
        self.event_delta_p_pu = 0.0
        self._event_location_pp = None
        self._hires_df = []

        self.soc.fill(0.5)
        self.p_set.fill(0.0)
        self.q_set.fill(0.0)
        self.delta_p_set.fill(0.0)
        self.k_droop_vpp.fill(0.0)
        # Slew-rate-limited previous-action buffers (units: normalized action ∈ [-1, 1]).
        # Reset every episode so first step has a zero baseline.
        self._prev_a_p_fast.fill(0.0)
        self._prev_k_droop_fast.fill(0.0)

        # Reset FFR activation state
        self.ffr_active = False
        self.ffr_activation_count = 0
        self.ffr_energy_delivered = 0.0

        # Reset Inter-VPP coordination and feasibility state
        self._vpp_soc_weight = np.ones(self._n_vpps, dtype=np.float32)
        self.feasibility_violations = {"voltage": 0, "soc": 0, "power": 0, "thermal": 0}
        self.feasibility_ok = True

        for ev in self.batt_models:
            soc_init = float(self.np_rng.uniform(0.3, 0.7))
            t_dep_h = float(self.np_rng.uniform(2.0, 8.0))
            ev.reset(SoC_init=soc_init, t_dep_h=t_dep_h)
        self._batt_soc[:] = np.asarray([ev.SoC for ev in self.batt_models], dtype=np.float32)

        self._sample_zone_lmp()
        self.day_ctx = self.context_loader.sample_day()
        self.day_step = int(self.np_rng.integers(0, 96))
        self._apply_day_context(self.day_ctx, self.day_step)
        self.event_injector.reset_cache()
        self._last_topology_fallback_reason = None

        force_topology = options.get("force_topology", None)
        if self.fixed_base_topology and force_topology is None:
            # Base-only training: always the nominal feeder, no reconfiguration.
            # (force_topology, when set by eval, still takes precedence below.)
            self.net = deepcopy(self.base_net)
            self.edge_index = self.event_injector.rebuild_edge_index(self.net)
            self.current_open_set = set()
            self._current_topology_id = -1
        elif force_topology is not None:
            topo_idx = int(force_topology)
            if topo_idx < 0 or topo_idx >= len(self.reconfig._cache):
                raise IndexError(f"force_topology index {topo_idx} out of range [0, {len(self.reconfig._cache) - 1}]")
            sampled_net, sampled_edge, selected_open_set = self.reconfig._cache[topo_idx]
            self.net = deepcopy(sampled_net)
            self.edge_index = np.asarray(sampled_edge, dtype=np.int64).copy()
            self.current_open_set = set(selected_open_set)
            self._current_topology_id = topo_idx
        else:
            try:
                row = self.day_ctx.iloc[int(np.clip(self.day_step, 0, len(self.day_ctx) - 1))] if self.day_ctx is not None else None
                load_cols = ["load_z1", "load_z2", "load_z3", "load_z4"]
                if row is not None and all(col in row for col in load_cols):
                    load_scale = float(np.mean([float(row[col]) for col in load_cols]))
                else:
                    load_scale = 1.0
                if row is not None and ("pv_pu" in row):
                    pv_scale = float(row["pv_pu"])
                else:
                    pv_scale = 0.8

                sampled_net, sampled_edge, selected_open_set = self.reconfig.select_optimal(load_scale=load_scale, pv_scale=pv_scale)
                self.net = deepcopy(sampled_net)
                self.edge_index = np.asarray(sampled_edge, dtype=np.int64)
                self.current_open_set = set(selected_open_set)
                # Find topology_id from cache by matching open_set
                self._current_topology_id = 0
                for idx, (_, _, open_set) in enumerate(self.reconfig._cache):
                    if set(open_set) == self.current_open_set:
                        self._current_topology_id = idx
                        break
            except Exception as exc:
                self._last_topology_fallback_reason = str(exc)
                self.current_open_set = set()
                self.edge_index = self.event_injector.rebuild_edge_index(self.net)
                self._current_topology_id = 0

        # Update frequency dynamics based on topology (which GFMs are connected)
        self._update_freq_dyn_topology()

        # Bind LTI freq_dyn to current operating point (requires converged PF)
        if self.use_lti_freq and self.freq_dyn_lti is not None:
            try:
                pp.runpp(self.net, algorithm="nr", init="auto", calculate_voltage_angles=True)
                self.freq_dyn_lti.bind_operating_point(self.net, self._current_topology_id)
                self.freq_dyn_lti.reset(f0=50.0)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(f"LTI bind_operating_point failed: {exc}")

        force_event = options.get("force_event", None)
        if force_event is not None:
            if isinstance(force_event, EventConfig):
                self.current_event = deepcopy(force_event)
            else:
                self.current_event = EventConfig(
                    type=str(force_event["type"]),
                    delta_P_mw=float(force_event["delta_P_mw"]),
                    location=int(force_event["location"]),
                    t_inject=float(force_event.get("t_inject", 30.0)),
                    injected=bool(force_event.get("injected", False)),
                )
        else:
            self.current_event = self.event_injector.sample()
            if not bool(getattr(self.event_injector, "_events_disabled", False)):
                # P1 fix Bug 2: randomize event injection time in [10, 50] s to
                # prevent the policy from overfitting to event timing at exactly
                # step 30. Eval still uses force_event with t_inject=30 for
                # repeatable comparison.
                self.current_event.t_inject = float(self.event_injector.rng.uniform(10.0, 50.0))
        self.current_event.injected = False

        sgen_index_map = {int(idx): pos for pos, idx in enumerate(self.net.sgen.index.to_numpy(dtype=np.int64, copy=False))}
        storage_index_map = {int(idx): pos for pos, idx in enumerate(self.net.storage.index.to_numpy(dtype=np.int64, copy=False))}
        load_index_map = {int(idx): pos for pos, idx in enumerate(self.net.load.index.to_numpy(dtype=np.int64, copy=False))}

        self._sgen_agent_pos = np.asarray(
            [sgen_index_map[int(idx)] for idx in self._agent_idx_labels[self._mask_sgen_agents]],
            dtype=np.int64,
        )
        self._storage_agent_pos = np.asarray(
            [storage_index_map[int(idx)] for idx in self._agent_idx_labels[self._mask_storage_agents]],
            dtype=np.int64,
        )
        self._load_agent_pos = np.asarray(
            [load_index_map[int(idx)] for idx in self._agent_idx_labels[self._mask_load_agents]],
            dtype=np.int64,
        )

        self._topology_just_changed = False
        obs_fast = self._build_obs_fast()
        obs_slow = self._build_obs_slow()
        info = {
            "edge_index": np.asarray(self.edge_index, dtype=np.int64),
            "topology_changed": False,
            "event_type": None,
            "topology_fallback_reason": self._last_topology_fallback_reason,
        }
        return obs_fast, obs_slow, info

    def _apply_fast_actions(self, action_fast: np.ndarray) -> None:
        af = np.asarray(action_fast, dtype=np.float32).reshape(-1)
        if af.shape[0] != 44:
            raise ValueError(f"action_fast must have shape (44,), got {af.shape}")

        self.delta_p_set = af[: self.n_agents].copy()
        self.k_droop_vpp = af[self.n_agents :].copy()  # 3 VPP-level droop coefficients

        delta_mw = (self.delta_p_set * 0.1 * self._agent_p_rated).astype(np.float32)

        # ----- sgen (PV/Wind injections) -----
        # Semantic convention (Section 6 design):
        #   delta_p > 0 = UP-reg intent  (cannot apply to PV/Wind — already at MPPT)
        #   delta_p < 0 = DOWN-reg / curtailment  (allowed)
        # → For PV/Wind sgen agents we force delta_mw <= 0 before applying.
        if self._sgen_agent_pos.size > 0:
            sgen_p = self.net.sgen["p_mw"].to_numpy(dtype=np.float32, copy=False)
            sgen_delta = delta_mw[self._mask_sgen_agents].copy()
            if self._mask_pv_among_sgen.any():
                sgen_delta[self._mask_pv_among_sgen] = np.minimum(
                    sgen_delta[self._mask_pv_among_sgen], 0.0
                )
            sgen_rated = self._agent_p_rated[self._mask_sgen_agents]
            sgen_p[self._sgen_agent_pos] = np.clip(
                sgen_p[self._sgen_agent_pos] + sgen_delta,
                0.0,
                2.0 * sgen_rated,
            )

        # ----- storage (BESS) -----
        # Bidirectional: delta_p > 0 = discharge UP-reg, delta_p < 0 = charge DOWN-reg.
        if self._storage_agent_pos.size > 0:
            storage_p = self.net.storage["p_mw"].to_numpy(dtype=np.float32, copy=False)
            storage_delta = delta_mw[self._mask_storage_agents]
            storage_rated = self._agent_p_rated[self._mask_storage_agents]
            storage_p[self._storage_agent_pos] = np.clip(
                storage_p[self._storage_agent_pos] + storage_delta,
                -2.0 * storage_rated,
                2.0 * storage_rated,
            )

        # ----- load (EVCS demand response, V2G discharge into grid as negative load) -----
        # Semantic convention: delta_p > 0 = reduce charging or discharge (UP-reg DR/V2G),
        # delta_p < 0 = increase charging (DOWN-reg). EVCS load is NEVER turned off
        # entirely (lower bound 0 keeps booking demand visible to power flow), but the
        # RL policy can ramp it down or to V2G discharge equivalent during a frequency
        # event. The "-load_delta" sign flip below converts UP-reg intent into load
        # decrease at the bus.
        if self._load_agent_pos.size > 0:
            load_p = self.net.load["p_mw"].to_numpy(dtype=np.float32, copy=False)
            load_delta = delta_mw[self._mask_load_agents]
            load_rated = self._agent_p_rated[self._mask_load_agents]
            load_p[self._load_agent_pos] = np.clip(
                load_p[self._load_agent_pos] - load_delta,
                0.0,
                2.0 * load_rated,
            )

        mean_abs = float(np.mean(np.abs(self.delta_p_set)))
        self.soc = np.clip(self.soc - 0.002 * mean_abs, 0.05, 0.95)

    def _severity_gamma(
        self,
        rocof: float,
        alpha_roc: float = 0.5,
        rocof_ref: float = 0.5,
        gamma_max: float = 3.0,
    ) -> float:
        severity = np.clip(abs(rocof) / rocof_ref, 0.0, gamma_max)
        return float(1.0 + alpha_roc * severity)

    def _compute_vpp_soc_weights(self) -> np.ndarray:
        """Inter-VPP Coordination: VPPs with higher SOC contribute more FFR.

        Weight formula: w_vpp = clip(avg_soc / 0.5, 0.2, 2.0)
        - SOC=50% → weight=1.0 (baseline)
        - SOC=80% → weight=1.6 (contribute more)
        - SOC=20% → weight=0.4 (contribute less, preserve capacity)
        """
        weights = np.ones(self._n_vpps, dtype=np.float32)
        for vpp_idx, vpp_id in enumerate(self._vpp_ids):
            agent_indices = self._vpp_droop_agents[vpp_id]
            # Get SOC from BESS models for this VPP's agents
            soc_values = []
            for agent_idx in agent_indices:
                if agent_idx in self._batt_agent_indices:
                    batt_idx = self._batt_agent_indices.index(agent_idx)
                    soc_values.append(float(self.batt_models[batt_idx].SoC))
            if soc_values:
                avg_soc = float(np.mean(soc_values))
                # Higher SOC → higher weight (more FFR contribution)
                weights[vpp_idx] = float(np.clip(avg_soc / 0.5, 0.2, 2.0))
        return weights

    def _check_feasibility(self) -> dict[str, int]:
        """Check constraint violations for feasibility feedback.

        Returns dict with violation counts for each constraint type.
        """
        violations = {"voltage": 0, "soc": 0, "power": 0, "thermal": 0}

        # SOC violations
        for ev in self.batt_models:
            if ev.SoC < ev.cfg.SoC_min:
                violations["soc"] += 1
            elif ev.SoC > ev.cfg.SoC_max:
                violations["soc"] += 1

        # Power limit violations
        for idx, agent_i in enumerate(self._batt_agent_indices):
            p_cmd_kw = abs(self.delta_p_set[agent_i]) * self._agent_p_rated[agent_i] * 1000.0
            p_rated_kw = float(self.batt_models[idx].cfg.P_rated_kw)
            if p_cmd_kw > p_rated_kw * 1.05:  # 5% tolerance
                violations["power"] += 1

        # Voltage violations (simplified - check if bus voltages exist)
        if hasattr(self.net, 'res_bus') and len(self.net.res_bus) > 0:
            v_pu = self.net.res_bus['vm_pu'].values
            v_violations = np.sum((v_pu < 0.95) | (v_pu > 1.05))
            violations["voltage"] = int(v_violations)

        return violations

    def step_fast(self, action_fast: np.ndarray):
        if self.episode_done:
            obs = self._build_obs_fast()
            return obs, 0.0, True, False, {"error": "episode done; call reset"}

        af = np.asarray(action_fast, dtype=np.float32).reshape(-1)
        n_ag = self.n_agents
        n_vp = len(self._vpp_droop_agents)
        expected_dual = 2 * n_ag + n_vp  # mappo_dual: a_P, a_K, VPP-K
        expected_single = n_ag + n_vp    # droop / mappo: a_P, VPP-K

        if self.ffr_mode == "mappo_dual":
            if af.shape[0] != expected_dual:
                raise ValueError(
                    f"action_fast (mappo_dual) must have shape ({expected_dual},), got {af.shape}"
                )
            delta_p_raw = af[:n_ag].copy()                            # a_P_i in [-1, 1]
            a_k_raw = af[n_ag : 2 * n_ag].copy()                      # a_K_i in [-1, 1]
            k_droop_vpp_raw = af[2 * n_ag : 2 * n_ag + n_vp].copy()   # legacy VPP K (RL sends zeros)
        else:
            if af.shape[0] != expected_single:
                raise ValueError(
                    f"action_fast must have shape ({expected_single},), got {af.shape}"
                )
            delta_p_raw = af[:n_ag].copy()
            a_k_raw = None
            k_droop_vpp_raw = af[n_ag:].copy()

        # Slew-rate limit on the fast actuator command (normalized ∈ [-1, 1]).
        # Constrains |a_t - a_{t-1}| ≤ slew_norm per fast step (dt_fast_s = 1.0 s),
        # i.e. ≤ 20 % of rated P per second. Prevents step-like setpoint jumps
        # from spiking |df/dt| above the IEEE-1547 Cat-III 2 Hz/s eval bound
        # while still allowing full ramp within ~5 s — fast enough for FFR.
        slew_norm = 0.2
        delta_p_raw = np.clip(
            delta_p_raw,
            self._prev_a_p_fast - slew_norm,
            self._prev_a_p_fast + slew_norm,
        )
        k_droop_vpp_raw = np.clip(
            k_droop_vpp_raw,
            self._prev_k_droop_fast - slew_norm,
            self._prev_k_droop_fast + slew_norm,
        )

        for idx, agent_i in enumerate(self._batt_agent_indices):
            p_kw = delta_p_raw[agent_i] * self._agent_p_rated[agent_i] * 1000.0
            p_rated_kw = float(self.batt_models[idx].cfg.P_rated_kw)
            p_kw_clipped = np.clip(p_kw, -p_rated_kw, p_rated_kw)
            delta_p_raw[agent_i] = p_kw_clipped / (self._agent_p_rated[agent_i] * 1000.0)

        # Persist slew-limited commands as the new baseline for next step.
        self._prev_a_p_fast = delta_p_raw.astype(np.float32).copy()
        self._prev_k_droop_fast = k_droop_vpp_raw.astype(np.float32).copy()

        action_fast = np.concatenate([delta_p_raw, k_droop_vpp_raw], axis=0)
        self._apply_fast_actions(action_fast)

        t_current = float(self.fast_step_count)
        event_now = False
        if self.current_event is None:
            self.current_event = self.event_injector.sample()
            if not bool(getattr(self.event_injector, "_events_disabled", False)):
                self.current_event.t_inject = 30.0
            self.current_event.injected = False

        prev_edge = np.asarray(self.edge_index, dtype=np.int64)
        self.net, self.edge_index, delta_p_pu = self.event_injector.inject(
            self.net,
            self.current_event,
            t_current=t_current,
        )
        curr_edge = np.asarray(self.edge_index, dtype=np.int64)
        self._topology_just_changed = not np.array_equal(prev_edge, curr_edge)
        if abs(delta_p_pu) > 0.0:
            self.event_delta_p_pu = float(delta_p_pu)
            # Pandapower bus index of the disturbance (resolved in inject); persists
            # with event_delta_p_pu so located J_L injection re-applies each step.
            self._event_location_pp = getattr(self.current_event, "location_pp", None)
            event_now = True

        _prev_state = self.freq_dyn_lti.get_state()
        prev_df = float(_prev_state.delta_f_hz)
        current_rocof = float(_prev_state.rocof_hz_s)

        # FFR Activation Trigger: activate when |Δf| > threshold OR |RoCoF| > threshold
        ffr_should_activate = (
            abs(prev_df) > self.ffr_threshold_df or
            abs(current_rocof) > self.ffr_threshold_rocof
        )

        if ffr_should_activate and not self.ffr_active:
            # Transition to ACTIVE state
            self.ffr_active = True
            self.ffr_activation_count += 1

        if self.ffr_active:
            # FFR is ACTIVE: apply full droop response with severity scaling
            gamma = self._severity_gamma(current_rocof)
            # Inter-VPP Coordination: weight by SOC (high SOC → more FFR contribution)
            self._vpp_soc_weight = self._compute_vpp_soc_weights()
            k_droop_vpp_eff = np.clip(self.k_droop_vpp * gamma * self._vpp_soc_weight, 0.0, 5.0)
        else:
            # FFR is NOT ACTIVE: minimal response to conserve BESS capacity
            gamma = 1.0
            k_droop_vpp_eff = np.zeros(self._n_vpps, dtype=np.float32)

        # Deactivate FFR when frequency stabilized (hysteresis to prevent chattering)
        if self.ffr_active and abs(prev_df) < self.ffr_deactivation_df:
            self.ffr_active = False

        if abs(prev_df) > 0.1:
            delta_p_mpc_kw = mpc_correction(
                evcs_list=self.batt_models,
                delta_f=prev_df,
                rocof=current_rocof,
                H_sys=self.freq_dyn_lti.h_sys,
            )
            for idx, agent_i in enumerate(self._batt_agent_indices):
                extra_mw = delta_p_mpc_kw[idx] / 1000.0
                delta_p_raw[agent_i] = np.clip(
                    delta_p_raw[agent_i] + extra_mw,
                    -self._agent_p_rated[agent_i],
                    self._agent_p_rated[agent_i],
                )
                self.delta_p_set[agent_i] = delta_p_raw[agent_i]

        # VPP-level droop: each VPP aggregates FFR from its member assets
        droop_support = 0.0
        for vpp_idx, vpp_id in enumerate(self._vpp_ids):
            vpp_k_droop = float(k_droop_vpp_eff[vpp_idx])
            for agent_idx in self._vpp_droop_agents[vpp_id]:
                local_dispatch = float(self.delta_p_set[agent_idx]) * 0.02
                droop_term = vpp_k_droop * max(0.0, -prev_df) * 0.02
                droop_support += local_dispatch + droop_term

        event_term = float(np.clip(self.event_delta_p_pu, -0.3, 0.3))
        bess_term = float(np.clip(droop_support, -0.25, 0.25))

        # TẦNG 3: Secondary Control (AGC) - Integral action for frequency recovery
        # Distributed VPP/BESS AGC is disabled by default; AGC is handled
        # internally by the VSG model (its AGC integral) to enforce
        # single-integrator semantics.
        f_current = float(_prev_state.delta_f_hz) + 50.0

        if USE_DISTRIBUTED_AGC:
            agc_term = 0.0
            for vpp_idx in range(self._n_vpps):
                agc_term += self.vpp_agc[vpp_idx].step(f_current, self.dt_fast_s)
            agc_term += self.bess_agc.step(f_current, self.dt_fast_s)
            agc_term = float(np.clip(agc_term, -0.05, 0.05))  # match pref_max_pu
        else:
            agc_term = 0.0

        # Combine droop (Tầng 2) + AGC (Tầng 3) as support power
        support_term = bess_term + agc_term

        # Per-DER FFR control (mode-dependent)
        delta_f = prev_df  # Use previous delta_f for droop calculation
        S_BASE = 15.7  # System base MVA

        if self.ffr_mode == "droop":
            # BASELINE: Classical per-DER droop FFR
            # P_ffr_i = -k_droop_i * delta_f + small RL adjustment
            p_bess_agg = 0.0
            for i, idx in enumerate(self._bess_indices):
                p_der_ffr = -self._k_droop_bess[i] * delta_f / S_BASE
                p_der_ffr += float(self.delta_p_set[idx]) * 0.02
                p_bess_agg += p_der_ffr

            p_v2g_agg = 0.0
            for i, idx in enumerate(self._v2g_indices):
                p_der_ffr = -self._k_droop_v2g[i] * delta_f / S_BASE
                p_der_ffr += float(self.delta_p_set[idx]) * 0.02
                p_v2g_agg += p_der_ffr
        elif self.ffr_mode == "mappo_dual":
            # PROPOSED (DUAL): RL outputs (a_P_i, a_K_i) per DER.
            # K_droop_i = K_min_i + (1+a_K_i)/2 * (K_max_i - K_min_i)   [MW/Hz]
            # P_ffr_i   = (a_P_i * P_rated_i  -  K_droop_i * delta_f) / S_BASE
            # FFR-eligibility SoC band [0.25, 0.85]: zero K for batteries whose
            # SoC leaves this band. It is strictly NESTED inside the physical
            # battery band [SoC_min, SoC_max] = [0.20, 0.90] (EVBatteryModel),
            # leaving a 0.05 headroom margin so a committed DER can always deliver
            # the bidirectional FFR response without hitting its energy limits.
            a_k_clipped = np.clip(a_k_raw, -1.0, 1.0)
            k_span = self._k_droop_max_per_agent - self._k_droop_min_per_agent
            k_droop_per_agent = self._k_droop_min_per_agent + 0.5 * (1.0 + a_k_clipped) * k_span

            # SoC-aware hard mask for battery-backed DERs (BESS + V2G).
            FFR_SOC_LO, FFR_SOC_HI = 0.25, 0.85  # nested in physical [0.20, 0.90]
            for idx_b, agent_i in enumerate(self._batt_agent_indices):
                if agent_i < self.n_agents:
                    soc = float(self._batt_soc[idx_b])
                    if soc < FFR_SOC_LO or soc > FFR_SOC_HI:
                        k_droop_per_agent[agent_i] = 0.0
            self._k_droop_last = k_droop_per_agent.astype(np.float32)
            self._p_ref_last = delta_p_raw.astype(np.float32).copy()

            p_bess_agg = 0.0
            for i, idx in enumerate(self._bess_indices):
                p_ref_mw = float(self.delta_p_set[idx]) * float(self._agent_p_rated[idx])
                p_der_ffr = (p_ref_mw - float(k_droop_per_agent[idx]) * delta_f) / S_BASE
                p_bess_agg += p_der_ffr

            p_v2g_agg = 0.0
            for i, idx in enumerate(self._v2g_indices):
                p_ref_mw = float(self.delta_p_set[idx]) * float(self._agent_p_rated[idx])
                p_der_ffr = (p_ref_mw - float(k_droop_per_agent[idx]) * delta_f) / S_BASE
                p_v2g_agg += p_der_ffr
        else:
            # PROPOSED (MAPPO single-action): RL agent directly controls per-DER FFR power
            # P_ffr_i = action_i * P_rated_i / S_BASE (full capacity, no droop floor)
            p_bess_agg = 0.0
            for i, idx in enumerate(self._bess_indices):
                p_der_ffr = float(self.delta_p_set[idx]) * float(self._agent_p_rated[idx]) / S_BASE
                p_bess_agg += p_der_ffr

            p_v2g_agg = 0.0
            for i, idx in enumerate(self._v2g_indices):
                p_der_ffr = float(self.delta_p_set[idx]) * float(self._agent_p_rated[idx]) / S_BASE
                p_v2g_agg += p_der_ffr

        # FFR energy delivered during activation: the actual dual-product / droop
        # reserve response that drives the LTI model — net (P_ref + K·Δf) power of the
        # controllable fleet (p_bess_agg + p_v2g_agg, pu of S_BASE), captured BEFORE the
        # support-clamp. Replaces the legacy VPP-droop channel (self.k_droop_vpp), which
        # mappo_dual never uses and which left ffr_energy_delivered identically zero.
        if self.ffr_active:
            ffr_power_mw = (abs(p_bess_agg) + abs(p_v2g_agg)) * S_BASE
            self.ffr_energy_delivered += ffr_power_mw * self.dt_fast_s / 3600.0

        # PV: no FFR (curtailment only, controlled by RL)
        p_pv_agg = 0.0
        for idx in self._pv_indices:
            p_pv_agg += float(self.delta_p_set[idx]) * 0.02

        # Add support term and clamp
        p_bess_agg = float(np.clip(p_bess_agg + support_term * 0.5, -0.20, 0.20))
        p_v2g_agg = float(np.clip(p_v2g_agg + support_term * 0.3, -0.15, 0.15))
        p_pv_agg = float(np.clip(p_pv_agg + support_term * 0.2, -0.10, 0.10))

        # Hi-resolution swing-eq trace: record delta_f after each ODE sub-step.
        # Sample period = self.dt_ode_s (0.1 s with default 10 sub-steps per 1 s
        # fast step) which is fine enough to capture the true nadir of the
        # underdamped low-inertia response (occurs ~0.3-0.5 s after event).
        if self.use_lti_freq and self.freq_dyn_lti is not None:
            # LTI model: single matrix-exp step for full fast-step
            # Build per-GFM power reference from delta_p_set (mapped via bus)
            n_gfm = self.freq_dyn_lti.n_gfm
            # Map each VPP agent to its nearest GFM via get_gfm_bus_idx — the SAME
            # mapping the obs builder uses to assign per-bus frequency to agents.
            # Prior code matched on spec["pp_bus_idx"] (a key that does not exist →
            # always -1) against gfm pandapower indices with an |idx diff|<5 rule,
            # so 0/41 agents ever matched: delta_P_ref stayed 0 and K_droop_gfm
            # stayed at its init for ALL GFMs, making the LTI frequency trajectory
            # completely policy-invariant. This routed the VPP control into the
            # frequency model for the first time.
            # Couple BOTH VPP channels into the per-GFM frequency model, mapping
            # each agent to its nearest GFM via get_gfm_bus_idx (same mapping the
            # obs builder uses). Two unit systems must be reconciled:
            #
            #  P-ref channel: delta_P_ref = Σ ΔP_set·P_rated / S_BASE  [pu on S_BASE].
            #    Direct power injection; units consistent, numerically safe.
            #
            #  K (droop) channel: the model's K_droop is a DIMENSIONLESS per-unit
            #    droop (K = 1/R; Δω_pu = -(1/K)·ΔP_pu), with backbone default 1.0.
            #    The per-DER _k_droop_last is ABSOLUTE [MW/Hz]. Conversion:
            #        ΔP_MW = K_vpp·(-Δf),  Δf = f0·Δω_pu,  ΔP_pu = ΔP_MW/S_BASE
            #      ⇒ K_model_contrib = K_vpp · f0 / S_BASE   (dimensionless).
            #    Parallel droop sources add, and the VPP droop ADDS to the backbone:
            #        K_droop_gfm = K_backbone + (f0/S_BASE)·Σ K_vpp.
            #    (Feeding raw K_vpp [MW/Hz] straight into M_p=1/K gave M_p≈33-100 and
            #    blew the dynamics up to ~-200 Hz; this conversion keeps K_gfm≳1 so
            #    M_p≲1 stays bounded while the learned droop still modulates A_f.)
            K_BACKBONE = 1.0
            k_to_pu = self.freq_dyn_lti.f0 / S_BASE
            delta_P_ref = np.zeros(n_gfm, dtype=float)
            k_vpp_sum_per_gfm = np.zeros(n_gfm, dtype=float)
            for agent_i in range(self.n_agents):
                agent_pp = int(self._agent_bus_pp[agent_i])
                gfm_i = int(self.freq_dyn_lti.get_gfm_bus_idx(agent_pp))
                delta_P_ref[gfm_i] += self.delta_p_set[agent_i] * self._agent_p_rated[agent_i]
                if hasattr(self, '_k_droop_last') and agent_i < len(self._k_droop_last):
                    k_vpp_sum_per_gfm[gfm_i] += float(self._k_droop_last[agent_i])
            delta_P_ref /= S_BASE
            K_droop_gfm = K_BACKBONE + k_to_pu * k_vpp_sum_per_gfm

            # Nadir Safety Layer (in-the-loop): closed-form minimal-perturbation
            # projection of delta_P_ref so the predicted next-step COI Δf stays
            # inside the nadir/zenith band. Active during BOTH training and eval so
            # the policy learns aware of the layer (reduced distribution shift).
            # RoCoF is NOT projected here — it is backbone-governed for the GFL
            # fleet (see Approach-C); the layer guards nadir/zenith only.
            if getattr(self, "nadir_safety_enabled", True):
                df_lim = float(getattr(self, "nadir_margin_hz", 0.5))
                delta_P_ref, _ns_act, _ns_dist, _ns_pred = self.freq_dyn_lti.nadir_safe_projection(
                    delta_P_ref=delta_P_ref,
                    delta_P_L=event_term,
                    K_droop=K_droop_gfm,
                    topology_id=self._current_topology_id,
                    delta_f_under=df_lim,
                    delta_f_over=df_lim,
                    event_location_pp=getattr(self, "_event_location_pp", None),
                )
                self._nadir_safety_active = bool(_ns_act)
                self._nadir_safety_dist = float(_ns_dist)
            else:
                self._nadir_safety_active = False
                self._nadir_safety_dist = 0.0

            freq_state = self.freq_dyn_lti.step(
                dt=self.dt_fast_s,
                delta_P_ref=delta_P_ref,
                delta_P_L=event_term,
                K_droop=K_droop_gfm,
                topology_id=self._current_topology_id,
                ffr_active=self.ffr_active,
                event_location_pp=getattr(self, "_event_location_pp", None),
            )
            if hasattr(self, "_hires_df"):
                # Optional sub-step trace for plotting the true nadir transient.
                # step() above already did the real (training-identical) propagation;
                # simulate_hires is read-only and only adds intermediate observation
                # points. Its last sample == freq_state (group property), so the
                # micro-trace joins seamlessly. hires_substeps=0 (default) keeps the
                # legacy 1-sample-per-fast-step behaviour (training untouched).
                n_sub = int(getattr(self, "hires_substeps", 0))
                if n_sub > 1:
                    self._hires_df.extend(
                        self.freq_dyn_lti.simulate_hires(
                            dt=self.dt_fast_s,
                            delta_P_ref=delta_P_ref,
                            delta_P_L=event_term,
                            K_droop=K_droop_gfm,
                            topology_id=self._current_topology_id,
                            n_sub=n_sub,
                            event_location_pp=getattr(self, "_event_location_pp", None),
                        )
                    )
                else:
                    self._hires_df.append(float(freq_state.delta_f_hz))

        for idx, agent_i in enumerate(self._batt_agent_indices):
            p_cmd_kw = self.delta_p_set[agent_i] * self._agent_p_rated[agent_i] * 1000.0
            _, _, soc, vbat, ib = self.batt_models[idx].step(self.dt_fast_s, p_cmd_kw)
            self._batt_soc[idx] = soc
            self._batt_vbat[idx] = vbat
            self._batt_ib[idx] = ib

        self.fast_step_count += 1
        done = False
        truncated = False

        obs_fast = self._build_obs_fast()
        df_clip = float(np.clip(np.nan_to_num(freq_state.delta_f_hz, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0))
        rocof_clip = float(np.clip(np.nan_to_num(freq_state.rocof_hz_s, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0))
        event_mag = abs(event_term)
        if event_mag > 1e-6:
            control_effect = float(np.clip((abs(prev_df) - abs(df_clip)) / event_mag, -2.0, 2.0))
        else:
            control_effect = 0.0
        effort = float(np.clip(np.mean(np.abs(self.delta_p_set)), 0.0, 1.0))
        if not np.isfinite(control_effect):
            control_effect = 0.0
        r_fast = float(-2.0 * abs(df_clip) - 0.2 * abs(rocof_clip) + 0.5 * control_effect - 0.02 * effort)
        r_fast = float(np.clip(np.nan_to_num(r_fast, nan=-10.0, posinf=10.0, neginf=-10.0), -10.0, 10.0))

        soc_violations = 0.0
        dcob_violations = []
        for idx, ev in enumerate(self.batt_models):
            if ev.SoC < ev.cfg.SoC_min + 0.02:
                soc_violations += (ev.cfg.SoC_min + 0.02 - ev.SoC)
            agent_i = self._batt_agent_indices[idx]
            p_cmd_kw = float(self.delta_p_set[agent_i]) * float(self._agent_p_rated[agent_i]) * 1000.0
            p_min_soft_kw, p_max_soft_kw = ev.power_bounds()
            excess_hi = max(0.0, p_cmd_kw - p_max_soft_kw)
            excess_lo = max(0.0, p_min_soft_kw - p_cmd_kw)
            excess_kw = excess_hi + excess_lo
            dcob_violations.append(excess_kw / max(float(ev.cfg.P_rated_kw), 1e-6))
        dcob_violation = float(np.mean(dcob_violations)) if dcob_violations else 0.0
        dcob_violation = float(np.clip(np.nan_to_num(dcob_violation, nan=0.0, posinf=10.0, neginf=0.0), 0.0, 10.0))
        r_fast -= self.w_soc * soc_violations
        r_fast -= self.w_dcob * dcob_violation

        # Feasibility Feedback: check constraint violations
        self.feasibility_violations = self._check_feasibility()
        total_violations = sum(self.feasibility_violations.values())
        self.feasibility_ok = (total_violations == 0)

        freq_flag = float(1.0 if (self.current_event is not None and self.current_event.injected) else 0.0)
        # Expose worst-bus metrics for reward (plan Task A decision: worst-bus for reward)
        if self.use_lti_freq and hasattr(freq_state, 'delta_f_worst'):
            delta_f_worst = float(freq_state.delta_f_worst)
            rocof_worst = float(freq_state.rocof_worst)
            delta_f_per_bus = freq_state.delta_f_per_bus.copy() if hasattr(freq_state, 'delta_f_per_bus') else None
            rocof_per_bus = freq_state.rocof_per_bus.copy() if hasattr(freq_state, 'rocof_per_bus') else None
        else:
            delta_f_worst = abs(float(freq_state.delta_f_hz))
            rocof_worst = abs(float(freq_state.rocof_hz_s))
            delta_f_per_bus = None
            rocof_per_bus = None
        info = {
            "event_injected": bool(event_now),
            "freq_event_flag": freq_flag,
            "delta_f": float(np.nan_to_num(freq_state.delta_f_hz, nan=0.0, posinf=5.0, neginf=-5.0)),
            "rocof": float(np.nan_to_num(freq_state.rocof_hz_s, nan=0.0, posinf=10.0, neginf=-10.0)),
            "nadir_safety_active": bool(getattr(self, "_nadir_safety_active", False)),
            "nadir_safety_dist": float(getattr(self, "_nadir_safety_dist", 0.0)),
            "delta_f_worst": delta_f_worst,
            "rocof_worst": rocof_worst,
            "delta_f_per_bus": delta_f_per_bus,
            "rocof_per_bus": rocof_per_bus,
            "ffr_active": bool(self.ffr_active),
            "ffr_activation_count": int(self.ffr_activation_count),
            "ffr_energy_delivered_mwh": float(self.ffr_energy_delivered),
            "vpp_soc_weights": self._vpp_soc_weight.copy(),  # Inter-VPP coordination weights
            "feasibility_ok": bool(self.feasibility_ok),
            "feasibility_violations": dict(self.feasibility_violations),
            "edge_count": int(self.edge_index.shape[1] if self.edge_index.ndim == 2 else 0),
            "edge_index": np.asarray(self.edge_index, dtype=np.int64),
            "topology_changed": bool(self._topology_just_changed),
            "event_type": self.current_event.type if (self.current_event is not None and self.current_event.injected) else None,
            "k_droop_vpp_eff": k_droop_vpp_eff.astype(np.float32),  # 3 VPP-level droop coefficients
            "evcs_soc": self._batt_soc.copy(),
            "evcs_dcob": np.asarray([ev.dcob() for ev in self.batt_models], dtype=np.float32),
            "step": int(self.fast_step_count),
            "phase": "fast",
        }

        return obs_fast, r_fast, done, truncated, info

    def _apply_slow_actions(self, action_slow: np.ndarray) -> None:
        """Slow (EM) action: per-agent P dispatch only (Q dropped — AM-only build)."""
        a = np.asarray(action_slow, dtype=np.float32).reshape(-1)
        # Accept either (n_agents,) or (2*n_agents,) for backward compatibility
        if a.shape[0] not in (self.n_agents, 2 * self.n_agents):
            raise ValueError(f"action_slow must have shape ({self.n_agents},) or ({2*self.n_agents},), got {a.shape}")

        self.p_set = a[: self.n_agents].copy()
        self.q_set.fill(0.0)  # Q product dropped in AM-only architecture

        for i, spec in enumerate(self._agent_specs):
            p_rated = max(float(spec["p_rated"]), 1e-6)
            p_cmd = float(self.p_set[i]) * p_rated

            elm = spec["element"]
            idx = spec["idx"]
            if elm == "sgen":
                self.net.sgen.at[idx, "p_mw"] = p_cmd
                self.net.sgen.at[idx, "q_mvar"] = 0.0
            elif elm == "storage":
                self.net.storage.at[idx, "p_mw"] = p_cmd
                if "q_mvar" in self.net.storage.columns:
                    self.net.storage.at[idx, "q_mvar"] = 0.0
            else:
                self.net.load.at[idx, "p_mw"] = max(0.0, -p_cmd)
                self.net.load.at[idx, "q_mvar"] = 0.0

        self.soc = np.clip(self.soc - 0.01 * np.abs(self.p_set), 0.05, 0.95)

    def step_slow(self, action_slow: np.ndarray):
        if self.fast_step_count < 300:
            obs = self._build_obs_slow()
            return obs, 0.0, False, False, {"warning": "run 300 fast steps before slow step", "phase": "slow"}

        self.day_step = (int(self.day_step) + 1) % 96
        self._apply_day_context(self.day_ctx, self.day_step)
        self._apply_slow_actions(action_slow)

        converged = True
        try:
            pp.runpp(self.net, numba=False, init="flat")
        except Exception:
            converged = False

        obs_slow = self._build_obs_slow()
        v = obs_slow[:, 0]
        p2p = float(np.sum(np.maximum(0.0, self.p_set)))
        vdi = float(np.mean(np.abs(v - 1.0)))

        # AM-only build: Q product removed; slow reward depends only on voltage
        # discipline (VDI), P2P market participation, and P dispatch.
        r_slow = float(-5.0 * vdi + 0.02 * p2p)

        self.slow_step_count += 1
        self.episode_done = True

        info = {
            "phase": "slow",
            "converged": bool(converged),
            "VDI": float(vdi),
            "P2P": float(p2p),
            "P_p2p": np.maximum(0.0, self.p_set).astype(np.float32),
            "v_min": float(np.min(v)),
            "v_max": float(np.max(v)),
            "v_violations": int(np.sum((v < 0.95) | (v > 1.05))),
            "step": int(self.slow_step_count),
        }

        done = True
        truncated = False
        return obs_slow, r_slow, done, truncated, info
