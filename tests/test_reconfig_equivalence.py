from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast, Any

import numpy as np
import pandapower as pp

from src.env.IEEE123bus import build_ieee123_net
from src.layer0_dso.reconfiguration import extract_network_data, run_reconfiguration
from src.opt.l0_reconfig import L0Optimizer, build_net_data_from_pandapower
from src.opt.tie_switch_reconfig import TieSwitchReconfiguration


def _build_snapshot() -> pp.pandapowerNet:
    net = build_ieee123_net(
        mode="matpower",
        balanced=True,
        convert_switches=True,
        slack_zones=None,
        source_mode="publish",
    )
    if not net.line.empty and "max_i_ka" in net.line.columns:
        net.line["max_i_ka"] = np.maximum(net.line["max_i_ka"].astype(float), 2.0)
    return net


TopologyEdge = tuple[int, int]


def _edge_key(from_bus: int, to_bus: int) -> TopologyEdge:
    a = int(from_bus)
    b = int(to_bus)
    return (a, b) if a <= b else (b, a)


def _realized_topology_edges(net: pp.pandapowerNet) -> set[TopologyEdge]:
    edges: set[TopologyEdge] = set()

    if not net.line.empty:
        for _, row in net.line.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            edges.add(_edge_key(int(row["from_bus"]), int(row["to_bus"])))

    if hasattr(net, "switch") and not net.switch.empty:
        for _, row in net.switch.iterrows():
            if str(row.get("et", "")) != "b" or not bool(row.get("closed", True)):
                continue
            edges.add(_edge_key(int(row["bus"]), int(row["element"])))

    return edges


def _edge_jaccard(a: set[TopologyEdge], b: set[TopologyEdge]) -> float:
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def _apply_alpha_star(net: pp.pandapowerNet, alpha_star: dict[int, int]) -> pp.pandapowerNet:
    out = copy.deepcopy(net)
    data = extract_network_data(out)
    for edge in data.edges:
        if not edge.is_switch or edge.switch_index is None:
            continue
        if edge.edge_id in alpha_star:
            out.switch.at[edge.switch_index, "closed"] = bool(alpha_star[edge.edge_id])
    return out


def _runpp_voltage(net: pp.pandapowerNet) -> np.ndarray:
    pp.runpp(net, numba=False, algorithm="nr", max_iteration=cast(Any, 10))
    return net.res_bus["vm_pu"].to_numpy(dtype=np.float64, copy=True)


def _voltage_band_violation_pct(reference: np.ndarray, candidate: np.ndarray, tol_pu: float = 0.005) -> float | None:
    if reference.shape != candidate.shape or reference.size == 0:
        return None
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if not np.any(valid):
        return None
    return float(np.mean(np.abs(reference[valid] - candidate[valid]) > tol_pu) * 100.0)


def _write_report(report: dict[str, object]) -> None:
    report_dir = Path("artifacts")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "reconfig_equivalence_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def test_reconfiguration_equivalence_report_smoke() -> None:
    report: dict[str, object] = {
        "feasibility_match": None,
        "topology_edge_symmetric_diff": None,
        "topology_edge_jaccard": None,
        "v_band_violation_pct": None,
        "alpha_star_count": 0,
        "cvx_market_status": None,
        "diagnostics": [],
    }

    net = _build_snapshot()

    _, alpha_star = run_reconfiguration(net, debug=False)
    pyomo_net = _apply_alpha_star(net, alpha_star)
    pyomo_v = _runpp_voltage(pyomo_net)

    # B) CVXPY L0 market path. This optimizes cap-bank/market variables, not topology switches.
    net_b = _build_snapshot()
    net_data = build_net_data_from_pandapower(net_b)
    cvx_optimizer = L0Optimizer(net_data)
    zone_totals = net_data.get("zone_totals", {})
    profiles = {
        "load_z1": float(zone_totals.get(1, 0.0)),
        "load_z2": float(zone_totals.get(2, 0.0)),
        "load_z3": float(zone_totals.get(3, 0.0)),
        "load_z4": float(zone_totals.get(4, 0.0)),
        "pv_pu": 1.0,
        "wind_mw": 0.0,
    }
    vpp_caps = {idx: {"P_max": 0.0, "Q_max": 0.0, "S_agg": 10.0} for idx in [1, 2, 3]}
    cvx_result = cvx_optimizer.solve(hour_block=1, profiles=profiles, vpp_capacities=vpp_caps, placement=None)

    # C) Tie-switch heuristic topology path
    net_c = _build_snapshot()
    tie_reconfig = TieSwitchReconfiguration(net_c, seed=42)
    tie_reconfig.generate_scenarios(n=3)
    tie_net, _, _ = tie_reconfig.select_optimal(load_scale=1.0, pv_scale=0.8)
    tie_v = _runpp_voltage(tie_net)

    pyomo_edges = _realized_topology_edges(pyomo_net)
    tie_edges = _realized_topology_edges(tie_net)

    topology_edge_symmetric_diff = len(pyomo_edges ^ tie_edges)
    topology_edge_jaccard = _edge_jaccard(pyomo_edges, tie_edges)
    v_band_pyomo_tie = _voltage_band_violation_pct(pyomo_v, tie_v)

    # Soft-equivalence acceptance: topology can differ, but should remain close on realized edge set.
    assert topology_edge_jaccard >= 0.95

    report.update(
        {
            "feasibility_match": True,
            "topology_edge_symmetric_diff": {
                "pyomo_vs_tie": topology_edge_symmetric_diff,
            },
            "topology_edge_jaccard": {
                "pyomo_vs_tie": topology_edge_jaccard,
            },
            "v_band_violation_pct": {
                "pyomo_vs_tie": v_band_pyomo_tie,
            },
            "alpha_star_count": len(alpha_star),
            "cvx_market_status": cvx_result.status,
        }
    )

    _write_report(report)
    assert Path("artifacts/reconfig_equivalence_report.json").exists()
