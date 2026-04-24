from __future__ import annotations

from collections import defaultdict, deque

import cvxpy as cp

from src.env.IEEE123bus import build_ieee123_net
from src.opt.l0_reconfig import L0Optimizer, build_net_data_from_pandapower


def test_build_net_data_exposes_solver_metadata() -> None:
    net = build_ieee123_net(
        mode="matpower",
        balanced=True,
        convert_switches=True,
        slack_zones=None,
        source_mode="publish",
    )
    net_data = build_net_data_from_pandapower(net)

    assert "slack_bus" in net_data
    assert net_data["slack_bus"] is not None
    assert "zone_totals" in net_data
    zone_totals = net_data["zone_totals"]
    assert set(zone_totals.keys()) == {1, 2, 3, 4}
    assert all(float(v) >= 0.0 for v in zone_totals.values())


def _build_default_caps() -> dict[int, dict[str, float]]:
    return {
        1: {"P_max": 10.3, "Q_max": 3.5, "S_agg": 7.01},
        2: {"P_max": 10.3, "Q_max": 3.5, "S_agg": 7.01},
        3: {"P_max": 4.9, "Q_max": 2.0, "S_agg": 5.17},
    }


def test_build_net_data_includes_closed_bus_switch_connectivity() -> None:
    net = build_ieee123_net(
        mode="matpower",
        balanced=True,
        convert_switches=True,
        slack_zones=None,
        source_mode="publish",
    )
    net_data = build_net_data_from_pandapower(net)

    branches = net_data["branches"]
    bus_ids = {int(bus[0]) for bus in net_data["buses"]}
    slack_bus = int(net_data["slack_bus"])
    adjacency: dict[int, list[int]] = defaultdict(list)
    for from_bus, to_bus, *_ in branches:
        adjacency[int(from_bus)].append(int(to_bus))
        adjacency[int(to_bus)].append(int(from_bus))

    reachable = {slack_bus}
    queue: deque[int] = deque([slack_bus])
    while queue:
        bus = queue.popleft()
        for neighbor in adjacency[bus]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)

    assert reachable == bus_ids


def test_l0_result_has_strict_elastic_labels() -> None:
    net = build_ieee123_net(
        mode="matpower",
        balanced=True,
        convert_switches=True,
        slack_zones=None,
        source_mode="publish",
    )
    net_data = build_net_data_from_pandapower(net)
    optimizer = L0Optimizer(net_data)

    profiles = {
        "load_z1": float(net_data["zone_totals"].get(1, 0.0)),
        "load_z2": float(net_data["zone_totals"].get(2, 0.0)),
        "load_z3": float(net_data["zone_totals"].get(3, 0.0)),
        "load_z4": float(net_data["zone_totals"].get(4, 0.0)),
        "pv_pu": 0.7,
        "wind_mw": 8.0,
    }

    result = optimizer.solve(1, profiles, _build_default_caps())
    assert result.solution_mode in {"strict", "elastic"}
    assert isinstance(result.elastic_used, bool)
    assert isinstance(result.strict_status, str)
    if result.solution_mode == "elastic":
        assert result.elastic_used is True


def test_l0_elastic_fallback_when_strict_infeasible(monkeypatch) -> None:
    net = build_ieee123_net(
        mode="matpower",
        balanced=True,
        convert_switches=True,
        slack_zones=None,
        source_mode="publish",
    )
    net_data = build_net_data_from_pandapower(net)
    optimizer = L0Optimizer(net_data)

    original_solve = cp.Problem.solve
    call_count = {"n": 0}

    def fake_solve(problem, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            problem._status = "infeasible"
            return None
        if call_count["n"] == 2:
            problem._status = "optimal"
            return None
        return original_solve(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", fake_solve)

    profiles = {
        "load_z1": float(net_data["zone_totals"].get(1, 0.0)),
        "load_z2": float(net_data["zone_totals"].get(2, 0.0)),
        "load_z3": float(net_data["zone_totals"].get(3, 0.0)),
        "load_z4": float(net_data["zone_totals"].get(4, 0.0)),
        "pv_pu": 0.7,
        "wind_mw": 8.0,
    }

    result = optimizer.solve(1, profiles, _build_default_caps())
    assert result.solution_mode == "elastic"
    assert result.elastic_used is True
    assert result.strict_status == "infeasible"
    assert result.status in {"optimal", "optimal_inaccurate"}
