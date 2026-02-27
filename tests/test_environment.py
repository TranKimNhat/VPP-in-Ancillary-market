from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.env.IEEE123bus import build_ieee123_net
from src.environment.grid_env import EnvConfig, GridEnvironment


def test_environment_reset_and_step_smoke(tmp_path: Path) -> None:
    net = build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True, slack_zones=None)
    buses = [int(b) for b in net.bus.index]
    bus_to_zone_csv = tmp_path / "bus_to_zone.csv"
    vpp_to_zone_csv = tmp_path / "vpp_to_zone.csv"
    pd.DataFrame({"bus": buses, "zone_id": [1] * len(buses)}).to_csv(bus_to_zone_csv, index=False)
    pd.DataFrame(columns=["vpp_id", "zone_id"]).to_csv(vpp_to_zone_csv, index=False)

    env = GridEnvironment(
        net=net,
        config=EnvConfig(
            max_steps=20,
            mapping_config={
                "bus_to_zone_csv": str(bus_to_zone_csv),
                "vpp_to_zone_csv": str(vpp_to_zone_csv),
            },
        ),
    )

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


def test_environment_static_zoning_guardrail() -> None:
    net = build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True, slack_zones=None)
    with pytest.raises(NotImplementedError):
        GridEnvironment(net=net, config=EnvConfig(zoning_mode="dynamic"))


def test_environment_vpp_mode_with_unassigned_buses(tmp_path: Path) -> None:
    net = build_ieee123_net(mode="feeder123", balanced=True, convert_switches=True, slack_zones=None)

    all_buses = [int(b) for b in net.bus.index]
    assert len(all_buses) >= 6

    bus_to_zone = pd.DataFrame({"bus": all_buses, "zone_id": [1 if i < len(all_buses) // 2 else 2 for i in range(len(all_buses))]})
    vpp_to_zone = pd.DataFrame(
        {
            "vpp_id": ["vpp_a", "vpp_b", "vpp_c"],
            "zone_id": [1, 1, 2],
        }
    )

    bus_to_vpp_rows: list[dict[str, object]] = []
    for i, bus in enumerate(all_buses):
        zone_id = int(bus_to_zone.loc[bus_to_zone["bus"] == bus, "zone_id"].iloc[0])
        if zone_id == 1:
            if i % 3 == 0:
                owner: str | None = "vpp_a"
            elif i % 3 == 1:
                owner = "vpp_b"
            else:
                owner = None
        else:
            owner = "vpp_c" if i % 3 == 1 else None
        bus_to_vpp_rows.append({"bus": bus, "vpp_id": owner})
    bus_to_vpp = pd.DataFrame(bus_to_vpp_rows)

    bus_to_zone_csv = tmp_path / "bus_to_zone.csv"
    vpp_to_zone_csv = tmp_path / "vpp_to_zone.csv"
    bus_to_vpp_csv = tmp_path / "bus_to_vpp.csv"
    bus_to_zone.to_csv(bus_to_zone_csv, index=False)
    vpp_to_zone.to_csv(vpp_to_zone_csv, index=False)
    bus_to_vpp.to_csv(bus_to_vpp_csv, index=False)

    layer1_csv = tmp_path / "layer1_vpp.csv"
    pd.DataFrame(
        {
            "day": ["expected", "expected", "expected", "expected"],
            "hour": [0, 0, 1, 1],
            "zone_id": ["1", "1", "1", "1"],
            "vpp_id": ["vpp_a", "vpp_b", "vpp_a", "vpp_b"],
            "P_ref": [0.4, 0.6, 0.5, 0.7],
            "R_commit": [0.1, 0.2, 0.1, 0.2],
        }
    ).to_csv(layer1_csv, index=False)

    env = GridEnvironment(
        net=net,
        config=EnvConfig(
            max_steps=4,
            vpp_mode=True,
            mapping_config={
                "bus_to_zone_csv": str(bus_to_zone_csv),
                "vpp_to_zone_csv": str(vpp_to_zone_csv),
                "bus_to_vpp_csv": str(bus_to_vpp_csv),
            },
        ),
        layer1_pref_csv=layer1_csv,
        mapping_config={
            "bus_to_zone_csv": str(bus_to_zone_csv),
            "vpp_to_zone_csv": str(vpp_to_zone_csv),
            "bus_to_vpp_csv": str(bus_to_vpp_csv),
        },
    )

    obs, _ = env.reset(seed=3)
    assert len(obs) > 0

    seen_unassigned = False
    for agent_obs in obs.values():
        assert "zone_id" in agent_obs
        assert "vpp_id" in agent_obs
        if agent_obs["vpp_id"] is None:
            seen_unassigned = True

    actions = {agent: np.zeros(2, dtype=np.float32) for agent in obs}
    next_obs, rewards, terminated, truncated, infos = env.step(actions)

    assert isinstance(next_obs, dict)
    assert isinstance(rewards, dict)
    assert "__all__" in terminated
    assert "__all__" in truncated
    assert seen_unassigned
    assert any("vpp_id" in info for info in infos.values())
