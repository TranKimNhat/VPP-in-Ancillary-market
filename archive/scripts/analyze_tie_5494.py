"""Exploratory: add the IEEE-123 normally-open tie 54-94 and recompute the
topology distribution. Standalone — does NOT modify the production env.

Adds 54-94 as an et='b' bus-bus switch (consistent with the 4 existing
sectionalizer switches), regenerates reconfiguration scenarios, and reports:
  - number of distinct topologies (now over 5 switches -> up to 2^5)
  - Jaccard edge-distance distribution (vs the current 4-switch mean 0.018)
  - radiality of each config (closing the tie + all sectionalizers => loop)
"""
from __future__ import annotations

import sys
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pandapower as pp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.microgrid_env_dual import MicrogridEnvDual
from src.opt.tie_switch_reconfig import TieSwitchReconfiguration


def edge_set(ei: np.ndarray) -> set[tuple[int, int]]:
    ei = np.asarray(ei)
    return {tuple(sorted((int(ei[0, k]), int(ei[1, k])))) for k in range(ei.shape[1])}


def jaccard_stats(esets):
    ds = []
    for i in range(len(esets)):
        for j in range(i + 1, len(esets)):
            u = len(esets[i] | esets[j])
            ds.append(1 - len(esets[i] & esets[j]) / u if u else 0.0)
    return np.array(ds) if ds else np.array([0.0])


def regen(net, seed=42, n=60):
    """Use the REAL reconfiguration validity logic on the given net."""
    rc = TieSwitchReconfiguration(net, seed=seed)
    rc._cache = []
    rc.generate_scenarios(n=n)
    esets, edge_counts = [], []
    for item in rc._cache:
        es = edge_set(item[1])
        esets.append(es)
        edge_counts.append(len(es))
    return esets, edge_counts


def main() -> None:
    env = MicrogridEnvDual(
        placement_path="artifacts/placement/official_placement_v3.json",
        mpc_path="data/grid_IEEE123_complete.m",
    )
    pp54, pp94 = env._bus_map[54], env._bus_map[94]
    print(f"bus 54 -> pp idx {pp54}, bus 94 -> pp idx {pp94}")

    # ---- BEFORE: current 4-switch model ----
    base4 = deepcopy(env.base_net)
    sw = base4.switch
    existing = [int(i) for i, et in zip(sw.index, sw["et"].astype(str)) if et == "b"]
    print(f"existing et='b' switches: {len(existing)} (indices {existing})")
    es4, ec4 = regen(base4)
    d4 = jaccard_stats(es4)
    print(f"\n[BEFORE] 4 switches -> {len(es4)} topologies")
    print(f"  Jaccard: min={d4.min():.4f} mean={d4.mean():.4f} max={d4.max():.4f} | edge-count {min(ec4)}..{max(ec4)}")

    # ---- AFTER: add 54-94 as a normally-open et='b' tie switch ----
    base5 = deepcopy(env.base_net)
    new_sw = pp.create_switch(base5, bus=pp54, element=pp94, et="b", closed=False, type="LBS")
    print(f"\nadded tie 54-94 as switch idx {new_sw} (normally open)")
    es5, ec5 = regen(base5)
    d5 = jaccard_stats(es5)
    print(f"[AFTER]  5 switches -> {len(es5)} topologies")
    print(f"  Jaccard: min={d5.min():.4f} mean={d5.mean():.4f} max={d5.max():.4f} | edge-count {min(ec5)}..{max(ec5)}")

    print("\n=== SUMMARY ===")
    print(f"  topologies:   {len(es4)} -> {len(es5)}")
    print(f"  mean Jaccard: {d4.mean():.4f} -> {d5.mean():.4f}  ({d5.mean()/max(d4.mean(),1e-9):.2f}x)")
    print(f"  max Jaccard:  {d4.max():.4f} -> {d5.max():.4f}")


if __name__ == "__main__":
    main()
