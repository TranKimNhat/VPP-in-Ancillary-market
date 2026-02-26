from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    tracking: float = 1.0
    voltage: float = 2.0
    curtailment: float = 1.0


def compute_reward(
    p_ref: float,
    p_actual: float,
    voltage_violation: float,
    curtailed: bool,
    weights: RewardWeights | None = None,
) -> float:
    w = weights or RewardWeights()
    tracking_penalty = abs(p_ref - p_actual)
    voltage_penalty = max(voltage_violation, 0.0)
    curtailment_penalty = 1.0 if curtailed else 0.0

    reward = -(
        w.tracking * tracking_penalty
        + w.voltage * voltage_penalty
        + w.curtailment * curtailment_penalty
    )
    return float(reward)
