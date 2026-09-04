"""What sits at buses 27/36/67/101/60/114/149? Source (BESS/PV/GFM/agent) vs load?"""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from src.env.microgrid_env_dual import MicrogridEnvDual

TARGETS = [27, 36, 67, 101, 60, 114, 149]

pl = json.loads((ROOT / "artifacts/placement/official_placement_v3.json").read_text())
print("=" * 60)
print("Placement file entries by category")
for cat in ["gfm", "der", "bess", "pv", "wind", "evcs"]:
    block = pl.get(cat, {})
    if isinstance(block, dict):
        buses = [(k, v.get("bus")) for k, v in block.items()]
    elif isinstance(block, list):
        buses = [(i, b.get("bus")) for i, b in enumerate(block)]
    else:
        buses = []
    if buses:
        print(f"  {cat}: {buses}")

env = MicrogridEnvDual(
    placement_path="artifacts/placement/official_placement_v3.json",
    mpc_path="data/grid_IEEE123_complete.m", seed=42, ffr_mode="mappo_dual",
)
bus_map = env._bus_map
agent_ext = set(int(b) for b in getattr(env, "_agent_bus_pp", []))
gfm_pp = set(env._gfm_bus_map.values())
ext_roots = set(int(b) for b in env.net.ext_grid["bus"]) if not env.net.ext_grid.empty else set()

# sgen (static generators = inverter sources incl. BESS/PV), load, gen
sgen_buses = set(int(b) for b in env.net.sgen["bus"]) if not env.net.sgen.empty else set()
load_buses = set(int(b) for b in env.net.load["bus"]) if not env.net.load.empty else set()
storage_buses = set(int(b) for b in env.net.storage["bus"]) if not env.net.storage.empty else set()

print("\n" + "=" * 60)
print("Per-target-bus inventory (external id -> pp index)")
print(f"{'busID':>6} {'pp':>4} {'GFM?':>5} {'sgen?':>6} {'storage?':>9} {'load?':>6} {'agentDER?':>10}")
for ext_id in TARGETS:
    pp = bus_map.get(ext_id, None)
    if pp is None:
        print(f"{ext_id:>6}  N/A  (bus not in reduced model)")
        continue
    print(f"{ext_id:>6} {pp:>4} "
          f"{('YES' if pp in gfm_pp or pp in ext_roots else '-'):>5} "
          f"{('YES' if pp in sgen_buses else '-'):>6} "
          f"{('YES' if pp in storage_buses else '-'):>9} "
          f"{('YES' if pp in load_buses else '-'):>6} "
          f"{('YES' if pp in agent_ext else '-'):>10}")

print("\nNet summary: #sgen=%d, #storage=%d, #load=%d, #ext_grid=%d, #agents=%d"
      % (len(env.net.sgen), len(env.net.storage), len(env.net.load),
         len(env.net.ext_grid), env.n_agents))
print("agent (DER) buses pp:", sorted(agent_ext))
