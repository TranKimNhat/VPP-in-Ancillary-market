from __future__ import annotations

import numpy as np

from src.layer1_vpp.dro_bidding import DroConfig, solve_wasserstein_dro
from src.layer1_vpp.scenario_generator import PriceScenarioSet
from src.layer1_vpp.virtual_battery import VirtualBatteryConfig


def _scenarios() -> PriceScenarioSet:
    hours = np.arange(4, dtype=int)
    energy = np.array([[10.0, 20.0, 30.0, 40.0], [12.0, 18.0, 28.0, 42.0]], dtype=float)
    reserve = np.array([[1.0, 1.0, 1.0, 1.0], [1.5, 1.5, 1.5, 1.5]], dtype=float)
    weights = np.array([0.5, 0.5], dtype=float)
    return PriceScenarioSet(
        scenario_names=("s1", "s2"),
        scenario_weights=weights,
        hours=hours,
        energy_prices=energy,
        reserve_prices=reserve,
    )


def test_layer1_dro_defaults_include_q_ref_and_lambda_q() -> None:
    result = solve_wasserstein_dro(_scenarios(), VirtualBatteryConfig(), DroConfig())

    assert result.q_ref.shape == (4,)
    assert result.lambda_q_expected.shape == (4,)
    assert np.allclose(result.q_ref, 0.0)
    assert np.allclose(result.lambda_q_expected, 0.0)


def test_layer1_dro_q_opf_enabled_produces_finite_q_outputs() -> None:
    result = solve_wasserstein_dro(
        _scenarios(),
        VirtualBatteryConfig(),
        DroConfig(q_opf_enabled=True, q_opf_sample_hours=2),
    )

    assert result.q_ref.shape == (4,)
    assert result.lambda_q_expected.shape == (4,)
    assert np.isfinite(result.q_ref).all()
    assert np.isfinite(result.lambda_q_expected).all()
    assert "q_opf_failed" not in result.solver_status
