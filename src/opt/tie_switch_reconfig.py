from __future__ import annotations

from collections import OrderedDict, deque
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import pickle
from typing import Any, Iterable, cast

import numpy as np
import pandapower as pp


class TieSwitchReconfiguration:
    TIE_LINES = [108, 110, 112, 114, 116]
    CACHE_VERSION = 2
    CACHE_POLICY = "strict_connected_extgrid"

    def __init__(self, net_base, seed: int | None = None) -> None:
        self.net_base = deepcopy(net_base)
        self._rng = np.random.default_rng(seed)
        self._cache: list[tuple[Any, np.ndarray, set[int]]] = []
        self._optimal_cache: OrderedDict[tuple[float, float, int], int] = OrderedDict()
        self._optimal_cache_maxsize = 128
        self._active_topologies = 20

    def _build_edge_index(self, net) -> np.ndarray:
        lines = net.line
        active = lines["in_service"].to_numpy(dtype=bool, copy=False)
        src = lines["from_bus"].to_numpy(dtype=np.int64, copy=False)[active]
        dst = lines["to_bus"].to_numpy(dtype=np.int64, copy=False)[active]
        return np.stack(
            [
                np.concatenate([src, dst]),
                np.concatenate([dst, src]),
            ],
            axis=0,
        )

    def _reachable_mask(self, net) -> np.ndarray:
        bus_labels = net.bus.index.to_numpy(dtype=np.int64, copy=False)
        n_bus = int(len(bus_labels))
        pos_map = {int(lbl): i for i, lbl in enumerate(bus_labels.tolist())}

        adj: list[list[int]] = [[] for _ in range(n_bus)]
        lines = net.line
        active = lines["in_service"].to_numpy(dtype=bool, copy=False)
        src_lbl = lines["from_bus"].to_numpy(dtype=np.int64, copy=False)[active]
        dst_lbl = lines["to_bus"].to_numpy(dtype=np.int64, copy=False)[active]

        for u_lbl, v_lbl in zip(src_lbl.tolist(), dst_lbl.tolist()):
            u = pos_map.get(int(u_lbl), -1)
            v = pos_map.get(int(v_lbl), -1)
            if u >= 0 and v >= 0:
                adj[u].append(v)
                adj[v].append(u)

        if hasattr(net, "switch") and (len(net.switch) > 0):
            sw = net.switch
            closed = sw["closed"].to_numpy(dtype=bool, copy=False)
            et = sw["et"].astype(str).to_numpy(copy=False)
            bus_sw = sw["bus"].to_numpy(dtype=np.int64, copy=False)
            elem_sw = sw["element"].to_numpy(dtype=np.int64, copy=False)
            for is_closed, et_i, a_lbl, b_lbl in zip(closed.tolist(), et.tolist(), bus_sw.tolist(), elem_sw.tolist()):
                if (not bool(is_closed)) or (str(et_i) != "b"):
                    continue
                u = pos_map.get(int(a_lbl), -1)
                v = pos_map.get(int(b_lbl), -1)
                if u >= 0 and v >= 0:
                    adj[u].append(v)
                    adj[v].append(u)

        if len(net.ext_grid) > 0:
            root_lbls = net.ext_grid["bus"].to_numpy(dtype=np.int64, copy=False).tolist()
            primary_roots = [pos_map.get(int(lbl), -1) for lbl in root_lbls]
        else:
            primary_roots = []

        fallback_root = pos_map.get(0, 0)

        def _bfs(roots: list[int]) -> np.ndarray:
            mask_i = np.zeros((n_bus,), dtype=bool)
            q_i: deque[int] = deque()
            for r in roots:
                if 0 <= int(r) < n_bus and not mask_i[int(r)]:
                    mask_i[int(r)] = True
                    q_i.append(int(r))
            while q_i:
                u = q_i.popleft()
                for v in adj[u]:
                    if not mask_i[v]:
                        mask_i[v] = True
                        q_i.append(v)
            return mask_i

        mask = _bfs(primary_roots if primary_roots else [fallback_root])
        if int(np.sum(mask)) <= 1 and 0 <= int(fallback_root) < n_bus:
            alt = _bfs([int(fallback_root)])
            if int(np.sum(alt)) > int(np.sum(mask)):
                mask = alt
        return mask

    def _is_topology_valid(self, net, run_power_flow: bool = False) -> bool:
        try:
            if run_power_flow:
                pp.runpp(net, numba=False, algorithm="nr", max_iteration=cast(Any, 10))
            if not bool(getattr(net, "converged", False)):
                return False
            reachable = self._reachable_mask(net)
            if int(np.sum(reachable)) != int(len(net.bus)):
                return False
            if not hasattr(net, "res_bus") or ("vm_pu" not in net.res_bus.columns):
                return False
            vm = net.res_bus["vm_pu"].to_numpy(dtype=np.float64, copy=False)
            if vm.shape[0] != int(len(net.bus)):
                return False
            if not np.isfinite(vm).all():
                return False
            return True
        except Exception:
            return False

    def _sanitize_cache_entries(self, entries: Iterable[Any]) -> tuple[list[tuple[Any, np.ndarray, set[int]]], int]:
        clean: list[tuple[Any, np.ndarray, set[int]]] = []
        dropped = 0
        for item in entries:
            try:
                if not isinstance(item, tuple):
                    dropped += 1
                    continue
                if len(item) == 3:
                    net_i, _, open_set_raw = item
                elif len(item) == 2:
                    net_i, _ = item
                    open_set_raw = set()
                else:
                    dropped += 1
                    continue

                net_copy = deepcopy(net_i)
                if not self._is_topology_valid(net_copy, run_power_flow=True):
                    dropped += 1
                    continue

                open_set = {int(x) for x in set(open_set_raw) if int(x) in self.TIE_LINES and int(x) in net_copy.line.index}
                edge_index = self._build_edge_index(net_copy)
                clean.append((net_copy, edge_index, open_set))
            except Exception:
                dropped += 1
        return clean, dropped

    def _apply_operating_condition(self, net_eval, load_scale: float, pv_scale: float) -> None:
        if hasattr(net_eval, "load") and not net_eval.load.empty:
            p_load = net_eval.load["p_mw"].to_numpy(dtype=np.float64, copy=True)
            net_eval.load["p_mw"] = p_load * float(load_scale)
            if "q_mvar" in net_eval.load.columns:
                q_load = net_eval.load["q_mvar"].to_numpy(dtype=np.float64, copy=True)
                net_eval.load["q_mvar"] = q_load * float(load_scale)

        if hasattr(net_eval, "sgen") and not net_eval.sgen.empty:
            if "type" in net_eval.sgen.columns:
                type_arr = net_eval.sgen["type"].astype(str).str.lower().to_numpy(copy=False)
                wind_mask = np.isin(type_arr, ["wp", "wind"])
                pv_mask = ~wind_mask
            else:
                pv_mask = np.ones((len(net_eval.sgen),), dtype=bool)
            if np.any(pv_mask):
                p_sgen = net_eval.sgen["p_mw"].to_numpy(dtype=np.float64, copy=True)
                p_sgen[pv_mask] = p_sgen[pv_mask] * float(pv_scale)
                net_eval.sgen["p_mw"] = p_sgen

    def generate_scenarios(self, n: int = 20) -> list[tuple[Any, np.ndarray, set[int]]]:
        scenarios: list[tuple[Any, np.ndarray, set[int]]] = []

        all_subsets: list[set[int]] = []
        for r in range(len(self.TIE_LINES) + 1):
            for subset in combinations(self.TIE_LINES, r):
                all_subsets.append(set(subset))

        order = self._rng.permutation(len(all_subsets))
        for idx in order:
            if len(scenarios) >= int(n):
                break

            open_set = all_subsets[int(idx)]
            net_copy = deepcopy(self.net_base)

            valid_open = [line_idx for line_idx in open_set if line_idx in net_copy.line.index]
            if valid_open:
                net_copy.line.loc[valid_open, "in_service"] = False

            if not self._is_topology_valid(net_copy, run_power_flow=True):
                continue
            ei = self._build_edge_index(net_copy)
            scenarios.append((net_copy, ei, open_set))

        if not scenarios:
            net_base = deepcopy(self.net_base)
            if self._is_topology_valid(net_base, run_power_flow=True):
                scenarios.append((net_base, self._build_edge_index(net_base), set()))

        self._cache = scenarios
        self._active_topologies = len(scenarios)
        self._optimal_cache.clear()
        print(f"Generated {len(scenarios)} valid topologies")
        return scenarios

    def set_active_topologies(self, n: int) -> None:
        self._active_topologies = int(max(1, min(int(n), len(self._cache) if self._cache else int(n))))
        self._optimal_cache.clear()

    def select_optimal(self, load_scale: float, pv_scale: float) -> tuple[Any, np.ndarray, set[int]]:
        if not self._cache:
            net = deepcopy(self.net_base)
            return net, self._build_edge_index(net), set()

        active_n = int(max(1, min(self._active_topologies, len(self._cache))))
        load_key = round(float(load_scale), 1)
        pv_key = round(float(pv_scale), 1)
        cache_key = (load_key, pv_key, active_n)

        cached_idx = self._optimal_cache.get(cache_key)
        if cached_idx is not None and 0 <= int(cached_idx) < active_n:
            self._optimal_cache.move_to_end(cache_key)
            net_c, ei_c, open_set_c = self._cache[int(cached_idx)]
            return net_c, np.asarray(ei_c, dtype=np.int64), set(open_set_c)

        best_obj = float("inf")
        best_idx: int | None = None

        for i, item in enumerate(self._cache[:active_n]):
            net_copy = item[0]
            open_set = item[2]
            net_eval = deepcopy(net_copy)
            try:
                self._apply_operating_condition(net_eval, load_scale=load_key, pv_scale=pv_key)
                if not self._is_topology_valid(net_eval, run_power_flow=True):
                    continue

                if hasattr(net_eval, "res_line") and len(net_eval.res_line) == len(net_eval.line):
                    active_line = net_eval.line["in_service"].to_numpy(dtype=bool, copy=False)
                    pl_arr = net_eval.res_line["pl_mw"].to_numpy(dtype=np.float64, copy=False)
                    p_loss = float(np.nansum(np.abs(pl_arr[active_line])))
                else:
                    p_loss = 0.0

                vm = net_eval.res_bus["vm_pu"].to_numpy(dtype=np.float64, copy=False)
                vdi = float(np.mean(np.abs(vm - 1.0))) if vm.size > 0 else 0.0
                v_viol = float(np.mean(np.maximum(0.0, np.abs(vm - 1.0) - 0.05))) if vm.size > 0 else 1.0
                obj = p_loss + 5.0 * vdi + 50.0 * v_viol + 0.01 * float(len(open_set))

                if obj < best_obj:
                    best_obj = obj
                    best_idx = int(i)
            except Exception:
                continue

        if best_idx is None:
            raise RuntimeError("No valid connected topology available under current operating condition")

        self._optimal_cache[cache_key] = int(best_idx)
        self._optimal_cache.move_to_end(cache_key)
        if len(self._optimal_cache) > int(self._optimal_cache_maxsize):
            self._optimal_cache.popitem(last=False)

        net, ei, open_set = self._cache[int(best_idx)]
        return deepcopy(net), np.asarray(ei, dtype=np.int64).copy(), set(open_set)

    def sample(self) -> tuple[Any, np.ndarray, set[int]]:
        if not self._cache:
            raise RuntimeError("Call generate_scenarios() first")
        idx = int(self._rng.integers(0, len(self._cache)))
        net, ei, open_set = self._cache[idx]
        return deepcopy(net), np.asarray(ei, dtype=np.int64).copy(), set(open_set)

    def save_cache(self, path: str | Path = "data/tie_switch_cache.pkl") -> None:
        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": int(self.CACHE_VERSION),
            "policy": str(self.CACHE_POLICY),
            "n_entries": int(len(self._cache)),
            "entries": self._cache,
        }
        with cache_path.open("wb") as f:
            pickle.dump(payload, f)
        print(f"Saved {len(self._cache)} scenarios to {cache_path.as_posix()}")

    def load_cache(self, path: str | Path = "data/tie_switch_cache.pkl") -> bool:
        cache_path = Path(path)
        if not cache_path.exists():
            return False

        try:
            with cache_path.open("rb") as f:
                raw = pickle.load(f)
        except Exception:
            return False

        if isinstance(raw, dict) and "entries" in raw:
            entries = raw.get("entries", [])
        elif isinstance(raw, list):
            entries = raw
        else:
            return False

        sanitized, dropped = self._sanitize_cache_entries(entries)
        if not sanitized:
            self._cache = []
            self._active_topologies = 0
            self._optimal_cache.clear()
            print(f"Loaded 0 valid scenarios from {cache_path.as_posix()} (dropped={dropped})")
            return False

        self._cache = sanitized
        self._active_topologies = len(self._cache)
        self._optimal_cache.clear()
        print(f"Loaded {len(self._cache)} valid scenarios from {cache_path.as_posix()} (dropped={dropped})")
        return True
