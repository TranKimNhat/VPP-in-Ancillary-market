"""Check whether bus 67 (S2 gen_trip event location) is islanded — i.e. lands
in a connected component WITHOUT any GFM / ext_grid root — on the reconfig
topologies (focus: topo 4 and 6, the d_E~0.18 cases that gave ITAE=0)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import networkx as nx

from src.env.microgrid_env_dual import MicrogridEnvDual

env = MicrogridEnvDual(
    placement_path="artifacts/placement/official_placement_v3.json",
    mpc_path="data/grid_IEEE123_complete.m",
    seed=42, ffr_mode="mappo_dual",
)

# --- Resolve key pp-bus indices --------------------------------------------
bus_map = env._bus_map                       # external id -> pp index
bus67_pp = bus_map.get(67, None)
gfm_pp = dict(env._gfm_bus_map)              # {"G1": idx, "G2": idx}
ext_roots = set()
if not env.net.ext_grid.empty:
    ext_roots = set(int(b) for b in env.net.ext_grid["bus"].tolist())

print(f"bus 67 -> pp index: {bus67_pp}")
print(f"GFM bus map (pp): {gfm_pp}")
print(f"ext_grid root buses (pp): {sorted(ext_roots)}")
roots = set(gfm_pp.values()) | ext_roots
print(f"All freq-reference roots (pp): {sorted(roots)}")
print(f"# topologies in cache: {len(env.reconfig._cache)}\n")


def edge_index_for(topo) -> np.ndarray:
    if isinstance(topo, tuple) and len(topo) >= 2:
        return np.asarray(topo[1])
    if isinstance(topo, dict):
        return np.asarray(topo.get("edge_index", np.array([[], []])))
    return np.array([[], []])


def analyze(topo_id: int) -> None:
    topo = env.reconfig._cache[topo_id]
    ei = edge_index_for(topo)
    G = nx.Graph()
    G.add_nodes_from(range(len(env.net.bus.index)))
    if ei.size:
        G.add_edges_from([(int(ei[0, k]), int(ei[1, k])) for k in range(ei.shape[1])])
    comps = list(nx.connected_components(G))
    # Which component holds bus 67?
    comp67 = next((c for c in comps if bus67_pp in c), None)
    roots_in_comp67 = roots & comp67 if comp67 else set()
    n_islands = sum(1 for c in comps if len(c) > 1)
    print(f"--- topo {topo_id} ---")
    print(f"  edges={ei.shape[1] if ei.size else 0}, components={len(comps)} "
          f"(non-trivial islands={n_islands})")
    if comp67 is None:
        print(f"  bus67 (pp {bus67_pp}) NOT in graph?!")
        return
    print(f"  bus67 component size: {len(comp67)}")
    print(f"  freq-reference roots in bus67's component: {sorted(roots_in_comp67)}")
    if roots_in_comp67:
        print(f"  => bus67 HAS a frequency reference (NOT islanded)  [OK]")
    else:
        print(f"  => bus67 ISLANDED (no GFM/ext_grid root) -> flat freq trace -> ITAE=0  [BAD]")
    # also report any GFM-less island
    for c in comps:
        if len(c) > 1 and not (roots & c):
            print(f"  [island w/o root] size={len(c)} buses (sample): {sorted(list(c))[:8]}")


for tid in [4, 6, 0, 1]:   # 4,6 are the suspect; 0,1 as healthy reference
    analyze(tid)
    print()

# ---- Full 24-topology summary --------------------------------------------
print("=" * 60)
print("FULL SCAN: bus67 islanding + rootless-island coverage")
print("=" * 60)
bus67_islanded = []
rootless_island_topos = []
for tid in range(len(env.reconfig._cache)):
    ei = edge_index_for(env.reconfig._cache[tid])
    G = nx.Graph(); G.add_nodes_from(range(len(env.net.bus.index)))
    if ei.size:
        G.add_edges_from([(int(ei[0, k]), int(ei[1, k])) for k in range(ei.shape[1])])
    comps = list(nx.connected_components(G))
    comp67 = next((c for c in comps if bus67_pp in c), set())
    if not (roots & comp67):
        bus67_islanded.append(tid)
    # any non-trivial island lacking a root?
    rootless = [len(c) for c in comps if len(c) > 1 and not (roots & c)]
    if rootless:
        rootless_island_topos.append((tid, max(rootless)))

print(f"\nbus67 ISLANDED (S2 event invalid) on {len(bus67_islanded)}/24 topos: {bus67_islanded}")
print(f"topos with >=1 rootless island (any dead buses): {len(rootless_island_topos)}/24")
print(f"  max rootless-island size per topo: "
      f"{[(t, s) for t, s in rootless_island_topos]}")
print(f"\n# freq-reference roots = {len(roots)} (2 GFM + 1 slack). "
      f"With 6 GFMs, every reconfig island would contain a root.")
