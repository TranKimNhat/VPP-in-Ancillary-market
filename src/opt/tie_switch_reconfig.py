from __future__ import annotations

from collections import OrderedDict, deque
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import json
import pickle
from typing import Any, Iterable, cast

import numpy as np
import pandapower as pp

from src.environment.topology_manager import build_edge_index as _build_edge_index_ext


class TieSwitchReconfiguration:
    # MATPOWER branch indices (1-based) of the near-zero impedance branches that
    # represent closed sectionalizing / tie switches in the IEEE 123-bus feeder.
    # convert_near_zero_branches_to_switches() turns these into pandapower Switch
    # objects (et="b") before TieSwitchReconfiguration is constructed, so the
    # primary code path uses switch_candidates from net.switch, not this list.
    # This list serves as the documented source-of-truth and fallback only.
    #
    # Branch 109: 18→135   Branch 113: 13→152   Branch 115: 60→160
    # Branch 117: 97→197   Branch 123: 151→300
    TIE_LINES = [109, 113, 115, 117, 123]
    CACHE_VERSION = 3  # bumped: TIE_LINES corrected from [108,110,112,114,116]
    CACHE_POLICY = "strict_connected_extgrid"

    def __init__(self, net_base, seed: int | None = None) -> None:
        self.net_base = deepcopy(net_base)
        self._rng = np.random.default_rng(seed)
        self._cache: list[tuple[Any, np.ndarray, set[int]]] = []
        self._optimal_cache: OrderedDict[tuple[float, float, int], int] = OrderedDict()
        self._optimal_cache_maxsize = 128
        self._active_topologies = 20
        # Baseline reachable count: the modified IEEE 123 feeder has a small
        # number of permanently-isolated buses (e.g. unused reference nodes),
        # so the validity gate uses "no worse than baseline" instead of
        # "exactly all buses reachable".
        try:
            self._base_reachable_count = int(np.sum(self._reachable_mask(self.net_base)))
        except Exception:
            self._base_reachable_count = int(len(self.net_base.bus))

    def _build_edge_index(self, net) -> np.ndarray:
        """Delegate to topology_manager.build_edge_index (includes switches)."""
        return _build_edge_index_ext(net)

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
        # Primary check: connectivity not worse than baseline. The modified
        # IEEE-123 feeder retains a couple of unused reference / orphan buses
        # that never appear in any path from the slack; we accept any candidate
        # that preserves reachability for the active part of the network.
        try:
            reachable = self._reachable_mask(net)
            if int(np.sum(reachable)) < int(self._base_reachable_count):
                return False
            return True
        except Exception:
            return False

    def _is_acyclic(self, net) -> bool:
        """True iff the active network graph is a forest (no cycles).

        This is the radiality invariant for THIS feeder: it has multiple
        grid-forming inverters (one per island), so a valid reconfiguration is
        not a single spanning tree but a FOREST of GFM-rooted trees — each
        connected component must be loop-free. Closing a normally-open tie that
        forms a loop within an island is rejected unless a switch on that loop is
        also opened (the standard close-tie / open-sectionalizer pairing). Uses
        union-find on the edge_index used everywhere else (build_edge_index), so
        the check is consistent with training/eval graphs.
        """
        ei = self._build_edge_index(net)
        edges = {tuple(sorted((int(ei[0, k]), int(ei[1, k])))) for k in range(ei.shape[1])}
        parent: dict[int, int] = {}

        def find(a: int) -> int:
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for u, v in edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                return False  # an edge joins two already-connected nodes => cycle
            parent[ru] = rv
        return True

    def _pf_objective(self, net) -> float | None:
        """Run PF and return scalar loss objective; return None if PF fails."""
        try:
            pp.runpp(net, numba=False, algorithm="nr", max_iteration=cast(Any, 50), init="flat")
            if not bool(getattr(net, "converged", False)):
                return None
            if not hasattr(net, "res_bus") or "vm_pu" not in net.res_bus.columns:
                return None
            vm = net.res_bus["vm_pu"].to_numpy(dtype=np.float64, copy=False)
            if not np.isfinite(vm).all():
                return None
            vdi = float(np.mean(np.abs(vm - 1.0)))
            v_viol = float(np.mean(np.maximum(0.0, np.abs(vm - 1.0) - 0.05)))
            if hasattr(net, "res_line") and len(net.res_line) == len(net.line):
                active = net.line["in_service"].to_numpy(dtype=bool, copy=False)
                p_loss = float(np.nansum(np.abs(net.res_line["pl_mw"].to_numpy(dtype=np.float64, copy=False)[active])))
            else:
                p_loss = 0.0
            return p_loss + 5.0 * vdi + 50.0 * v_viol
        except Exception:
            return None

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

                line_ids = set(int(x) for x in net_copy.line.index.to_numpy(dtype=np.int64, copy=False))
                switch_ids: set[int] = set()
                if hasattr(net_copy, "switch") and len(net_copy.switch) > 0:
                    sw = net_copy.switch
                    et = sw["et"].astype(str).to_numpy(copy=False)
                    switch_ids = set(int(idx) for idx, et_i in zip(sw.index.tolist(), et.tolist()) if str(et_i) == "b")

                open_set = {int(x) for x in set(open_set_raw) if int(x) in line_ids or int(x) in switch_ids}
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
        diagnostics: list[dict[str, Any]] = []

        # Reconfigurable elements = et="b" bus-bus switches (the sectionalizers,
        # tagged ("s", idx)) PLUS tie LINES whose name starts with "tie" (added by
        # the model builder with realistic impedance, tagged ("l", idx)). Both kinds
        # participate as binary open/closed candidates. Tie lines carry impedance
        # (54-94 etc. span real distance), so they cannot be zero-Z et="b" switches.
        switch_candidates: list[tuple[str, int]] = []
        if hasattr(self.net_base, "switch") and len(self.net_base.switch) > 0:
            sw = self.net_base.switch
            et = sw["et"].astype(str).to_numpy(copy=False)
            switch_candidates = [("s", int(idx)) for idx, et_i in zip(sw.index.tolist(), et.tolist()) if str(et_i) == "b"]

        tie_line_candidates: list[tuple[str, int]] = []
        if hasattr(self.net_base, "line") and len(self.net_base.line) > 0 and "name" in self.net_base.line.columns:
            for idx in self.net_base.line.index.tolist():
                nm = str(self.net_base.line.at[idx, "name"])
                if nm.lower().startswith("tie"):
                    tie_line_candidates.append(("l", int(idx)))

        if switch_candidates or tie_line_candidates:
            candidates: list[tuple[str, int]] = switch_candidates + tie_line_candidates
        else:
            # Legacy fallback: toggle the documented TIE_LINES by line index.
            candidates = [("l", int(x)) for x in self.TIE_LINES if int(x) in self.net_base.line.index]

        all_subsets: list[set[tuple[str, int]]] = []
        for r in range(len(candidates) + 1):
            for subset in combinations(candidates, r):
                all_subsets.append(set(subset))

        order = self._rng.permutation(len(all_subsets))
        seen_edge_hashes: set[str] = set()
        # Exclude the base topology G_0 itself: the empty open-set (and any subset
        # that collapses back to base) is edge-identical to G_0 and must not appear
        # in the unseen hold-out. Seed the dedup set with G_0's edge hash.
        try:
            _bei = self._build_edge_index(self.net_base)
            if _bei.shape[1] > 0:
                _bedges = sorted({tuple(sorted((int(_bei[0, k]), int(_bei[1, k]))))
                                  for k in range(_bei.shape[1])})
                seen_edge_hashes.add("|".join(f"{u}-{v}" for u, v in _bedges))
        except Exception:
            pass
        for idx in order:
            if len(scenarios) >= int(n):
                break

            open_tagged = all_subsets[int(idx)]
            net_copy = deepcopy(self.net_base)

            # NON-SUBTRACTIVE: set EVERY candidate's state explicitly — open iff it is
            # in this subset, otherwise CLOSED/active. This is the key change from the
            # old subtractive logic (which only ever opened switches and so left a
            # normally-open tie open in every config). Closing a tie that is absent
            # from open_tagged adds its reroute edge; the forest gate below then
            # requires a switch on the resulting loop to also be open.
            for kind, cidx in candidates:
                is_open = (kind, cidx) in open_tagged
                if kind == "s" and cidx in net_copy.switch.index:
                    net_copy.switch.at[cidx, "closed"] = (not is_open)
                elif kind == "l" and cidx in net_copy.line.index:
                    net_copy.line.at[cidx, "in_service"] = (not is_open)
            # open_set stored as plain int ids (advisory; net_copy is authoritative)
            open_set = {int(cidx) for _, cidx in open_tagged}
            valid_open = [cidx for kind, cidx in open_tagged
                          if (kind == "s" and cidx in net_copy.switch.index)
                          or (kind == "l" and cidx in net_copy.line.index)]

            reachable_count = 0
            pf_ok = False
            vm_min = None
            vm_max = None
            line_loading_max = None
            reject_reason = ""

            try:
                reachable = self._reachable_mask(net_copy)
                reachable_count = int(np.sum(reachable))
                if reachable_count < int(self._base_reachable_count):
                    reject_reason = "connectivity_worse_than_base"
                elif not self._is_acyclic(net_copy):
                    # Radiality: each GFM island must stay loop-free. A closed tie
                    # that forms a cycle without a paired sectionalizer open is rejected.
                    reject_reason = "has_cycle"
                else:
                    try:
                        pp.runpp(net_copy, numba=False, algorithm="nr", max_iteration=cast(Any, 50), init="flat")
                        pf_ok = bool(getattr(net_copy, "converged", False))
                        if hasattr(net_copy, "res_bus") and ("vm_pu" in net_copy.res_bus.columns):
                            vm = net_copy.res_bus["vm_pu"].to_numpy(dtype=np.float64, copy=False)
                            if vm.size > 0 and np.isfinite(vm).any():
                                vm_min = float(np.nanmin(vm))
                                vm_max = float(np.nanmax(vm))
                        if hasattr(net_copy, "res_line") and ("loading_percent" in net_copy.res_line.columns):
                            lp = net_copy.res_line["loading_percent"].to_numpy(dtype=np.float64, copy=False)
                            if lp.size > 0 and np.isfinite(lp).any():
                                line_loading_max = float(np.nanmax(lp))
                    except Exception:
                        pf_ok = False

                    if not pf_ok:
                        reject_reason = "pf_not_converged"
            except Exception:
                reject_reason = "mask_check_error"

            edge_hash = ""
            try:
                ei = self._build_edge_index(net_copy)
                if ei.shape[1] > 0:
                    edges_set = set(tuple(sorted((int(ei[0, k]), int(ei[1, k])))) for k in range(ei.shape[1]))
                    edge_hash = "|".join(f"{u}-{v}" for u, v in sorted(edges_set))
                else:
                    edge_hash = "empty"
            except Exception:
                edge_hash = "hash_error"

            # Deduplicate by edge_hash: different open-sets can yield the same graph
            # (e.g. opening a redundant switch), which would otherwise inflate the
            # cache with identical topologies (the min-Jaccard=0 artifact).
            is_duplicate = edge_hash in seen_edge_hashes
            accepted = (
                reject_reason == ""
                and (not is_duplicate)
                and self._is_topology_valid(net_copy, run_power_flow=True)
            )
            if reject_reason == "" and is_duplicate:
                reject_reason = "duplicate_edge_set"
            diagnostics.append({
                "candidate_id": int(idx),
                "open_set": sorted([int(x) for x in open_set]),
                "n_open": int(len(valid_open)),
                "reachable_count": int(reachable_count),
                "n_bus": int(len(net_copy.bus)),
                "pf_ok": bool(pf_ok),
                "vm_min": vm_min,
                "vm_max": vm_max,
                "line_loading_max": line_loading_max,
                "edge_hash": edge_hash,
                "accepted": bool(accepted),
                "reject_reason": reject_reason,
            })

            if not accepted:
                continue

            seen_edge_hashes.add(edge_hash)
            ei = self._build_edge_index(net_copy)
            scenarios.append((net_copy, ei, open_set))

        # ---- N-1 line-contingency phase ------------------------------------
        # Per the evaluation protocol the topology set is generated by tie switch
        # operations AND N-1 line contingencies. The tie phase above covers the
        # former; here a single in-service feeder line is tripped and a normally
        # open tie is reclosed to restore a radial, fully energized network --
        # reconfigurations that tie toggling alone cannot reach. Reuses the same
        # reachability / radiality / PF gates and the shared dedup set, and fills
        # only up to the remaining budget n.
        if len(scenarios) < int(n) and tie_line_candidates:
            removable_lines = [
                int(li) for li in self.net_base.line.index.tolist()
                if bool(self.net_base.line.at[li, "in_service"])
                and not str(self.net_base.line.at[li, "name"]).lower().startswith("tie")
            ]
            for line_idx in removable_lines:
                if len(scenarios) >= int(n):
                    break
                for _, tie_idx in tie_line_candidates:
                    if len(scenarios) >= int(n):
                        break
                    net_copy = deepcopy(self.net_base)
                    net_copy.line.at[line_idx, "in_service"] = False  # N-1 trip
                    net_copy.line.at[tie_idx, "in_service"] = True    # reclose tie
                    try:
                        if int(np.sum(self._reachable_mask(net_copy))) < int(self._base_reachable_count):
                            continue
                        if not self._is_acyclic(net_copy):
                            continue
                        pp.runpp(net_copy, numba=False, algorithm="nr",
                                 max_iteration=cast(Any, 50), init="flat")
                        if not bool(getattr(net_copy, "converged", False)):
                            continue
                    except Exception:
                        continue
                    ei = self._build_edge_index(net_copy)
                    if ei.shape[1] == 0:
                        continue
                    edges_set = set(tuple(sorted((int(ei[0, k]), int(ei[1, k]))))
                                    for k in range(ei.shape[1]))
                    edge_hash = "|".join(f"{u}-{v}" for u, v in sorted(edges_set))
                    if edge_hash in seen_edge_hashes:
                        continue
                    seen_edge_hashes.add(edge_hash)
                    scenarios.append((net_copy, ei, {int(line_idx)}))
                    diagnostics.append({
                        "candidate_id": -1, "phase": "n_minus_1",
                        "open_set": [int(line_idx)], "n_open": 1,
                        "tripped_line": int(line_idx), "reclosed_tie": int(tie_idx),
                        "n_bus": int(len(net_copy.bus)), "edge_hash": edge_hash,
                        "accepted": True, "reject_reason": "",
                    })

        if not scenarios:
            net_base = deepcopy(self.net_base)
            if self._is_topology_valid(net_base, run_power_flow=True):
                scenarios.append((net_base, self._build_edge_index(net_base), set()))

        try:
            diag_path = Path("artifacts") / "topology_generation_diagnostics.json"
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            diag_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            print(f"Saved topology diagnostics: {diag_path.as_posix()} ({len(diagnostics)} candidates)")
        except Exception:
            pass

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
                obj_pf = self._pf_objective(net_eval)
                # If PF doesn't converge, use connectivity-only score (open_set penalty only).
                obj = (obj_pf + 0.01 * float(len(open_set))) if obj_pf is not None else (1e6 + 0.01 * float(len(open_set)))
                if obj < best_obj:
                    best_obj = obj
                    best_idx = int(i)
            except Exception:
                continue

        # Fallback: if nothing ranked, use index 0 (already connectivity-validated).
        if best_idx is None:
            best_idx = 0

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
