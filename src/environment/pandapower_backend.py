from __future__ import annotations

from dataclasses import dataclass

import pandapower as pp


@dataclass(frozen=True)
class PowerFlowResult:
    converged: bool


class PandapowerBackend:
    def run_power_flow(self, net: pp.pandapowerNet) -> PowerFlowResult:
        try:
            pp.runpp(net, algorithm="nr", init="auto", calculate_voltage_angles=False)
            return PowerFlowResult(converged=True)
        except Exception:
            return PowerFlowResult(converged=False)
