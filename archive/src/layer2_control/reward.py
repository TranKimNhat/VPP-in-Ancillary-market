from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketPrices:
    lambda_e: float        # active DLMP at agent bus (€/MWh)
    lambda_q: float        # reactive DLMP at agent bus (€/MVAR/h)
    lambda_cap: float      # FFR capacity price (€/MW/h, paid always for committed reserve)
    lambda_act: float      # FFR activation price (€/MWh, paid only when activated)
    c_undersupply: float = 3.0  # Nordic imbalance penalty multiplier


def compute_reward(
    p_actual: float,
    q_actual: float,
    r_commit: float,
    r_delivered: float,
    voltage_violation: float,
    scenario_type: str,
    prices: MarketPrices,
) -> float:
    """Market-linked reward: EM energy+Q payment + AM reserve payment - voltage penalty.

    EM: paid per MWh delivered (active) and MVAR delivered (reactive).
    AM: capacity payment always + activation payment only when scenario is am_*.
        Undersupply incurs Nordic imbalance penalty (c_undersupply × lambda_act).
    Voltage: penalized 2.0 × violation magnitude regardless of market mode.
    """
    is_activated = scenario_type.startswith("am_")

    em = prices.lambda_e * p_actual + prices.lambda_q * q_actual

    am = (
        prices.lambda_cap * r_commit
        + prices.lambda_act * r_delivered * float(is_activated)
        - prices.c_undersupply * prices.lambda_act * max(0.0, r_commit - r_delivered)
    )

    return float(em + am - 2.0 * max(voltage_violation, 0.0))
