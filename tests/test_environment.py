from __future__ import annotations

import numpy as np

from src.env.IEEE123bus import build_ieee123_net
from src.environment.grid_env import EnvConfig, GridEnvironment


def test_environment_reset_and_step_smoke() -> None:
    net = build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True, slack_zones=None)
    env = GridEnvironment(net=net, config=EnvConfig(max_steps=20))

    obs, info = env.reset(seed=1)
    assert isinstance(obs, dict)
    assert len(obs) > 0
    assert "step" in info

    done = False
    steps = 0
    while not done and steps < 20:
        actions = {agent: np.random.uniform(-1.0, 1.0, size=2).astype(np.float32) for agent in obs}
        obs, rewards, terminated, truncated, _ = env.step(actions)
        assert isinstance(rewards, dict)
        assert "__all__" in terminated
        assert "__all__" in truncated
        done = bool(terminated["__all__"] or truncated["__all__"])
        steps += 1

    metrics = env.metrics()
    assert "tracking_error" in metrics
    assert "voltage_violation" in metrics
