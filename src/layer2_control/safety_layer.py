from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyLimits:
    p_min: float
    p_max: float
    q_min: float
    q_max: float
    s_max: float


@dataclass(frozen=True)
class SafetyResult:
    p_safe: float
    q_safe: float
    curtailed: bool
    safety_mode: str = "clip_project"


def enforce_safety(action: np.ndarray, limits: SafetyLimits) -> SafetyResult:
    if action.size < 2:
        raise ValueError("Action must include P and Q components.")

    p = float(np.clip(action[0], limits.p_min, limits.p_max))
    q = float(np.clip(action[1], limits.q_min, limits.q_max))

    apparent = float(np.sqrt(p**2 + q**2))
    curtailed = False
    if apparent > limits.s_max > 0.0:
        scale = limits.s_max / apparent
        p *= scale
        q *= scale
        curtailed = True

    return SafetyResult(p_safe=p, q_safe=q, curtailed=curtailed, safety_mode="clip_project")
