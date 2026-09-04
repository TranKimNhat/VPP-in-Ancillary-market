from __future__ import annotations

import numpy as np

from src.env.IEEE123bus import build_ieee123_net
from src.layer0_dso.zonal_pricing import compute_dlmp


def test_compute_dlmp_returns_finite_per_bus() -> None:
    net = build_ieee123_net(mode="matpower", balanced=True, convert_switches=True, slack_zones=None, source_mode="publish")
    lambda_dlmp = {int(bus): 10.0 + 0.01 * i for i, bus in enumerate(net.bus.index)}

    out = compute_dlmp(net, lambda_dlmp)

    assert len(out) == len(net.bus.index)
    for bus in net.bus.index:
        comp = out[int(bus)]
        vals = np.array(
            [
                comp.lambda_p_total,
                comp.lambda_q_total,
                comp.lambda_p_energy,
                comp.lambda_p_loss,
                comp.lambda_p_congestion,
                comp.lambda_p_voltage,
            ],
            dtype=float,
        )
        assert np.isfinite(vals).all()
        assert np.isclose(
            comp.lambda_p_total,
            comp.lambda_p_energy + comp.lambda_p_loss + comp.lambda_p_congestion + comp.lambda_p_voltage,
            atol=1e-9,
        )
