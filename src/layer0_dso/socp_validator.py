from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
import pandapower as pp

from src.layer0_dso.reconfiguration import extract_network_data


@dataclass(frozen=True)
class SocpAcValidationResult:
    socp_ac_gap_max: float
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
        if vn_kv <= 0.0:
            continue
        vm_kv = float(np.sqrt(max(v_sq, 0.0)))
        vm_pu[bus] = vm_kv / vn_kv
    return vm_pu


def validate_socp_against_ac(
    net: pp.pandapowerNet,
    alpha_star: dict[int, int],
    socp_voltage_squared: dict[int, float],
    tolerance: float = 0.2,
) -> SocpAcValidationResult:
    ac_net = _apply_switch_decisions(net, alpha_star)
    try:
        pp.runpp(ac_net, algorithm="nr", init="auto", calculate_voltage_angles=False)
    except Exception:
        return SocpAcValidationResult(
            socp_ac_gap_max=float("inf"),
            ac_valid=False,
            converged=False,
        )

    socp_vm = _socp_vm_pu(ac_net, socp_voltage_squared)
    if ac_net.res_bus.empty or not socp_vm:
        return SocpAcValidationResult(
            socp_ac_gap_max=float("inf"),
            ac_valid=False,
            converged=False,
        )

    gaps: list[float] = []
    for bus, vm_socp in socp_vm.items():
        if bus not in ac_net.res_bus.index:
            continue
        vm_ac = float(ac_net.res_bus.at[bus, "vm_pu"])
        gaps.append(abs(vm_socp - vm_ac))

    gap_max = float(max(gaps)) if gaps else float("inf")
    return SocpAcValidationResult(
        socp_ac_gap_max=gap_max,
        ac_valid=bool(gap_max < tolerance),
        converged=True,
    )
