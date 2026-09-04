from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# System base and topology constants
S_BASE_MW = 15.705  # IEEE 123-bus modified system base (MW)

# Contingency sizing rationale (based on literature):
# - O'Sullivan et al. (2014, 275 citations): "largest infeed loss" as design contingency
# - Seneviratne et al. (2016, 131 citations): frequency response deteriorates with RES
# - Knap et al. (2016, 368 citations): ESS sizing for inertial response
# - Javadi et al. (2021, 60 citations): RoCoF and nadir as critical constraints
#
# Contingency magnitude guidelines for islanded microgrid:
#   Normal:  5-10% of S_BASE = 0.8-1.6 MW
#   N-1:     10-20% of S_BASE = 1.6-3.1 MW (dimensioning incident)
#   Severe:  20-30% of S_BASE = 3.1-4.7 MW
#   Extreme: 30-40% of S_BASE = 4.7-6.3 MW (stress testing FFR)

WIND_BUSES = (97, 98, 101, 105)
HIGH_LOAD_BUSES = (2, 3, 8, 11, 20, 45, 52, 65)
CRITICAL_LINES = ((97, 98), (98, 101), (1, 2), (2, 3), (52, 65))


@dataclass
class EventConfig:
    type: str  # 'load_step','gen_trip','line_trip','high_ren'
    delta_P_mw: float
    location: int  # bus or line index
    t_inject: float = 30.0
    injected: bool = False
    # Pandapower bus index of the disturbance, resolved during inject(). Enables
    # located J_L injection in the frequency model. None for line_trip (no single
    # bus) or when the location can't be mapped.
    location_pp: int | None = None


class EventInjector:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self._edge_cache: np.ndarray | None = None
        self._net_hash: tuple | None = None
        self._event_types = np.asarray(["load_step", "gen_trip", "line_trip", "high_ren"], dtype=object)
        self._event_probs = np.asarray([0.3, 0.3, 0.2, 0.2], dtype=np.float64)
        self._events_disabled = False
        self.max_delta_p_mw: float = 6.3  # Max event magnitude for curriculum

    def set_max_delta_p_mw(self, max_mw: float) -> None:
        """Set maximum event magnitude for curriculum learning."""
        self.max_delta_p_mw = float(max_mw)

    def set_probs(self, probs_dict: dict[str, float]) -> None:
        probs = np.asarray([float(probs_dict.get(str(ev), 0.0)) for ev in self._event_types], dtype=np.float64)
        probs = np.clip(probs, 0.0, None)
        total = float(np.sum(probs))
        if total <= 0.0:
            self._events_disabled = True
            self._event_probs = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            return
        self._events_disabled = False
        self._event_probs = probs / total

    def _topology_hash(self, net) -> tuple:
        line_active = np.asarray(net.line["in_service"].to_numpy(dtype=bool), dtype=np.bool_)
        if len(net.switch) > 0:
            sw = net.switch
            bb_mask = (sw["et"].to_numpy() == "b") & sw["closed"].to_numpy(dtype=bool)
            sw_bus = sw["bus"].to_numpy(dtype=np.int64, copy=False)[bb_mask]
            sw_elm = sw["element"].to_numpy(dtype=np.int64, copy=False)[bb_mask]
            return (id(net), line_active.tobytes(), sw_bus.tobytes(), sw_elm.tobytes())
        return (id(net), line_active.tobytes())

    def _get_cached_edge_index(self, net, force_rebuild: bool = False) -> np.ndarray:
        topo_hash = self._topology_hash(net)
        if force_rebuild or self._edge_cache is None or self._net_hash != topo_hash:
            self._edge_cache = self.rebuild_edge_index(net)
            self._net_hash = topo_hash
        return self._edge_cache

    def reset_cache(self) -> None:
        self._edge_cache = None
        self._net_hash = None

    def _busnum_to_pp_idx(self, net, bus_number: int) -> int | None:
        for idx, row in net.bus.iterrows():
            name = str(row.get("name", "")).strip()
            if name.isdigit() and int(name) == int(bus_number):
                return int(idx)
        if int(bus_number) in set(int(x) for x in net.bus.index):
            return int(bus_number)
        return None

    def _find_line_idx_by_bus_pair(self, net, bus_pair: tuple[int, int]) -> int | None:
        a_pp = self._busnum_to_pp_idx(net, bus_pair[0])
        b_pp = self._busnum_to_pp_idx(net, bus_pair[1])
        if a_pp is None or b_pp is None:
            return None
        mask = (
            ((net.line["from_bus"] == a_pp) & (net.line["to_bus"] == b_pp))
            | ((net.line["from_bus"] == b_pp) & (net.line["to_bus"] == a_pp))
        )
        candidates = net.line.index[mask]
        if len(candidates) == 0:
            return None
        return int(candidates[0])

    def _pick_bus(self, net, candidates: Sequence[int]) -> int:
        selected = int(self.rng.choice(np.asarray(candidates, dtype=np.int64)))
        pp_idx = self._busnum_to_pp_idx(net, selected)
        if pp_idx is None:
            raise ValueError(f"Cannot map bus number {selected} to pandapower index")
        return int(pp_idx)

    def sample(self) -> EventConfig:
        if self._events_disabled:
            return EventConfig(type="load_step", delta_P_mw=0.0, location=int(HIGH_LOAD_BUSES[0]), t_inject=1e9)

        event_type = str(self.rng.choice(self._event_types, p=self._event_probs))

        # Contingency magnitudes for FFR evaluation in 100% renewable islanded microgrid
        # Literature basis:
        #   - O'Sullivan et al. (2014): Ireland system uses largest infeed loss
        #   - MacIver et al. (2021): GB Aug 2019 blackout from ~1GW loss (~3% of 30GW system)
        #   - Hamanah et al. (2024): Different disturbances for uncertain load/RES behavior
        #   - Javadi et al. (2021): RoCoF and frequency nadir as stability constraints
        #
        # For 15.7 MW islanded microgrid (S_BASE):
        #   10% = 1.57 MW, 20% = 3.14 MW, 30% = 4.71 MW, 40% = 6.28 MW

        if event_type == "load_step":
            location = int(self.rng.choice(np.asarray(HIGH_LOAD_BUSES, dtype=np.int64)))
            # Sudden load increase: 10-25% of S_BASE (N-1 to severe)
            # Ref: Bitew et al. (2025) - 500kW MG tested with load surges
            delta = float(self.rng.choice(np.asarray([1.6, 2.5, 3.9], dtype=float)))
            delta = min(delta, self.max_delta_p_mw)  # Clamp to curriculum limit

        elif event_type == "gen_trip":
            location = int(self.rng.choice(np.asarray(WIND_BUSES, dtype=np.int64)))
            # Generator/wind farm trip: 15-35% of S_BASE (severe to extreme)
            # Ref: Seneviratne et al. (2016) - large generator loss with RES penetration
            # Ref: Knap et al. (2016) - contingency event sizing for ESS
            delta = -float(self.rng.choice(np.asarray([2.4, 3.9, 5.5], dtype=float)))
            delta = max(delta, -self.max_delta_p_mw)  # Clamp (negative)

        elif event_type == "line_trip":
            pair = CRITICAL_LINES[int(self.rng.integers(0, len(CRITICAL_LINES)))]
            location = int(pair[0] * 1000 + pair[1])
            # Line trip: ~15% equivalent power island/imbalance
            # Ref: Farooq et al. (2022) - transmission line tripping causes gen-load mismatch
            delta = -min(2.4, self.max_delta_p_mw)  # Clamp to curriculum limit

        else:  # high_ren
            location = int(self.rng.choice(np.asarray(WIND_BUSES, dtype=np.int64)))
            # Sudden renewable surge (over-generation): 20-40% of S_BASE
            # Ref: Kerdphol et al. (2019, 186 citations) - high RES penetration events
            # Causes over-frequency, tests upward FFR capability
            delta = float(self.rng.choice(np.asarray([3.1, 4.7, 6.3], dtype=float)))
            delta = min(delta, self.max_delta_p_mw)  # Clamp to curriculum limit

        return EventConfig(type=event_type, delta_P_mw=delta, location=location, t_inject=30.0)

    def rebuild_edge_index(self, net) -> np.ndarray:
        lines = net.line
        active_mask = lines["in_service"].to_numpy(dtype=bool)
        from_buses = lines["from_bus"].to_numpy(dtype=np.int64, copy=False)[active_mask]
        to_buses = lines["to_bus"].to_numpy(dtype=np.int64, copy=False)[active_mask]

        if len(net.switch) > 0:
            sw = net.switch
            bb_mask = (sw["et"].to_numpy() == "b") & sw["closed"].to_numpy(dtype=bool)
            if np.any(bb_mask):
                sw_from = sw["bus"].to_numpy(dtype=np.int64, copy=False)[bb_mask]
                sw_to = sw["element"].to_numpy(dtype=np.int64, copy=False)[bb_mask]
                from_buses = np.concatenate([from_buses, sw_from])
                to_buses = np.concatenate([to_buses, sw_to])

        if from_buses.size == 0:
            return np.zeros((2, 0), dtype=np.int64)

        return np.asarray(
            [
                np.concatenate([from_buses, to_buses]),
                np.concatenate([to_buses, from_buses]),
            ],
            dtype=np.int64,
        )

    def inject(self, net, event: EventConfig, t_current: float | None = None, **kwargs):
        t_val = float(t_current if t_current is not None else kwargs.get("t", 0.0))
        if event.injected or t_val < float(event.t_inject):
            return net, self._get_cached_edge_index(net), 0.0

        delta_p = float(event.delta_P_mw)
        topology_changed = False

        if event.type == "load_step":
            bus_pp = self._busnum_to_pp_idx(net, int(event.location))
            if bus_pp is None:
                raise ValueError(f"load_step bus {event.location} not found")
            event.location_pp = int(bus_pp)
            load_mask = net.load["bus"] == bus_pp
            load_idx = list(net.load.index[load_mask])
            if load_idx:
                add_each = delta_p / float(len(load_idx))
                for idx in load_idx:
                    net.load.at[idx, "p_mw"] = float(net.load.at[idx, "p_mw"]) + add_each

        elif event.type == "gen_trip":
            bus_pp = self._busnum_to_pp_idx(net, int(event.location))
            if bus_pp is None:
                raise ValueError(f"gen_trip bus {event.location} not found")
            event.location_pp = int(bus_pp)
            trip_amount = abs(delta_p)
            remaining = trip_amount
            sgen_idx = list(net.sgen.index[net.sgen["bus"] == bus_pp])
            gen_idx = list(net.gen.index[net.gen["bus"] == bus_pp]) if hasattr(net, "gen") else []

            for idx in sgen_idx:
                p = float(net.sgen.at[idx, "p_mw"])
                cut = min(p, remaining)
                net.sgen.at[idx, "p_mw"] = p - cut
                remaining -= cut
                if net.sgen.at[idx, "p_mw"] <= 1e-6:
                    net.sgen.at[idx, "in_service"] = False
                if remaining <= 1e-9:
                    break

            for idx in gen_idx:
                if remaining <= 1e-9:
                    break
                p = float(net.gen.at[idx, "p_mw"])
                cut = min(p, remaining)
                net.gen.at[idx, "p_mw"] = p - cut
                remaining -= cut
                if net.gen.at[idx, "p_mw"] <= 1e-6:
                    net.gen.at[idx, "in_service"] = False

            # Gen trip = loss of generation = equivalent to load increase = positive delta_p
            delta_p = +trip_amount

        elif event.type == "line_trip":
            line_idx: int | None
            if int(event.location) in set(int(x) for x in net.line.index):
                line_idx = int(event.location)
            else:
                bus_a = int(event.location) // 1000
                bus_b = int(event.location) % 1000
                line_idx = self._find_line_idx_by_bus_pair(net, (bus_a, bus_b))
                if line_idx is None:
                    pair = CRITICAL_LINES[0]
                    line_idx = self._find_line_idx_by_bus_pair(net, pair)
            if line_idx is None:
                raise ValueError("No line found for line_trip event")
            net.line.at[line_idx, "in_service"] = False
            topology_changed = True
            # Line trip causing power loss = equivalent to load increase = positive delta_p
            delta_p = +abs(delta_p)

        elif event.type == "high_ren":
            bus_pp = self._busnum_to_pp_idx(net, int(event.location))
            if bus_pp is None:
                raise ValueError(f"high_ren bus {event.location} not found")
            event.location_pp = int(bus_pp)
            sgen_idx = list(net.sgen.index[net.sgen["bus"] == bus_pp])
            if sgen_idx:
                add_each = abs(delta_p) / float(len(sgen_idx))
                for idx in sgen_idx:
                    net.sgen.at[idx, "p_mw"] = float(net.sgen.at[idx, "p_mw"]) + add_each
            # Surplus generation → freq rises → negative power imbalance (gen > load)
            delta_p = -abs(delta_p)

        else:
            raise ValueError(f"Unknown event type: {event.type}")

        event.injected = True
        edge_index = self._get_cached_edge_index(net, force_rebuild=topology_changed)
        delta_p_pu = float(delta_p / S_BASE_MW)
        return net, edge_index, delta_p_pu
