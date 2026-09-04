"""Phase B guardrail — verifies build_edge_index reflects switch state changes.

If this test fails the topology_manager assumption is wrong and Phase D cannot
proceed until topology_manager.build_edge_index is fixed.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.env.IEEE123bus import build_ieee123_net
from src.environment.topology_manager import build_edge_index


def test_build_edge_index_reflects_switch_toggle() -> None:
    """Toggling one bus-bus switch open removes exactly 2 entries (undirected)."""
    net = build_ieee123_net(mode="matpower", balanced=True, convert_switches=True)
    base = build_edge_index(net)

    b_mask = (net.switch["et"].astype(str) == "b") & (net.switch["closed"].astype(bool))
    open_candidates = net.switch.index[b_mask]
    if len(open_candidates) == 0:
        pytest.skip("No closed bus-bus switches in default net — cannot test toggle")

    sw_idx = open_candidates[0]
    net.switch.at[sw_idx, "closed"] = False
    after = build_edge_index(net)

    assert after.shape[1] == base.shape[1] - 2, (
        f"Expected {base.shape[1] - 2} edges after opening switch {sw_idx}, "
        f"got {after.shape[1]}"
    )
