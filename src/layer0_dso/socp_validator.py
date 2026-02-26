from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
import pandapower as pp

from src.layer0_dso.reconfiguration import extract_network_data


@dataclass(frozen=True)
class SocpAcValidationResult:
    socp_ac_gap_max: float
    socp_ac_gap_p95: float
    socp_ac_gap_p50: float
    worst_bus: int | None
    compared_bus_count: int
    ac_valid: bool
    converged: bool


def _apply_switch_decisions(net: pp.pandapowerNet, alpha_star: dict[int, int]) -> pp.pandapowerNet:
    net_copy = copy.deepcopy(net)
    data = extract_network_data(net_copy)
    for edge in data.edges:
        if not edge.is_switch or edge.switch_index is None:
            continue
        if edge.edge_id not in alpha_star:
            continue
        net_copy.switch.at[edge.switch_index, "closed"] = bool(alpha_star[edge.edge_id])
    return net_copy


def _socp_vm_pu(
    net: pp.pandapowerNet,
    socp_voltage_squared: dict[int, float],
) -> dict[int, float]:
    vm_pu: dict[int, float] = {}
    for bus, v_sq in socp_voltage_squared.items():
        if bus not in net.bus.index:
            continue
        vn_kv = float(net.bus.at[bus, "vn_kv"])
        if not np.isfinite(vn_kv) or vn_kv <= 0.0:
            continue
        if not np.isfinite(v_sq):
            continue
        vm_kv = float(np.sqrt(max(v_sq, 0.0)))
        vm = vm_kv / vn_kv
        if not np.isfinite(vm):
            continue
        vm_pu[bus] = vm
    return vm_pu


def validate_socp_against_ac(
    net: pp.pandapowerNet,
    alpha_star: dict[int, int],
    socp_voltage_squared: dict[int, float],
    tolerance: float = 0.01,
) -> SocpAcValidationResult:
    ac_net = _apply_switch_decisions(net, alpha_star)
    try:
        pp.runpp(ac_net, algorithm="nr", init="auto", calculate_voltage_angles=False)
    except Exception:
        return SocpAcValidationResult(
            socp_ac_gap_max=float("inf"),
            socp_ac_gap_p95=float("inf"),
            socp_ac_gap_p50=float("inf"),
            worst_bus=None,
            compared_bus_count=0,
            ac_valid=False,
            converged=False,
        )

    socp_vm = _socp_vm_pu(ac_net, socp_voltage_squared)
    if ac_net.res_bus.empty or not socp_vm:
        return SocpAcValidationResult(
            socp_ac_gap_max=float("inf"),
            socp_ac_gap_p95=float("inf"),
            socp_ac_gap_p50=float("inf"),
            worst_bus=None,
            compared_bus_count=0,
            ac_valid=False,
            converged=False,
        )

    gap_by_bus: dict[int, float] = {}
    for bus, vm_socp in socp_vm.items():
        if bus not in ac_net.res_bus.index:
            continue
        vm_ac = float(ac_net.res_bus.at[bus, "vm_pu"])
        if not np.isfinite(vm_ac):
            continue
        gap = abs(vm_socp - vm_ac)
        if not np.isfinite(gap):
            continue
        gap_by_bus[int(bus)] = float(gap)

    if not gap_by_bus:
        return SocpAcValidationResult(
            socp_ac_gap_max=float("inf"),
            socp_ac_gap_p95=float("inf"),
            socp_ac_gap_p50=float("inf"),
            worst_bus=None,
            compared_bus_count=0,
            ac_valid=False,
            converged=True,
        )

    gaps = np.asarray(list(gap_by_bus.values()), dtype=float)
    finite_mask = np.isfinite(gaps)
    finite_gaps = gaps[finite_mask]
    if finite_gaps.size == 0:
        return SocpAcValidationResult(
            socp_ac_gap_max=float("inf"),
            socp_ac_gap_p95=float("inf"),
            socp_ac_gap_p50=float("inf"),
            worst_bus=None,
            compared_bus_count=0,
            ac_valid=False,
            converged=True,
        )

    worst_bus = max(gap_by_bus, key=gap_by_bus.get)
    gap_max = float(np.max(finite_gaps))
    gap_p95 = float(np.percentile(finite_gaps, 95))
    gap_p50 = float(np.percentile(finite_gaps, 50))
    return SocpAcValidationResult(
        socp_ac_gap_max=gap_max,
        socp_ac_gap_p95=gap_p95,
        socp_ac_gap_p50=gap_p50,
        worst_bus=int(worst_bus),
        compared_bus_count=int(gaps.size),
        ac_valid=bool(gap_max <= tolerance),
        converged=True,
    )
