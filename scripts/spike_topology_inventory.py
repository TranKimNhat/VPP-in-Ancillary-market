"""Phase A diagnostic spike — read-only, no src/ changes.

Prints structured diagnostics to stdout to answer:
  WHY is layer0_switches.csv empty after run_layer0_pipeline?

Decision gate (see plan):
  C2 : alpha_star keys do not intersect switch_map keys → edge_id mismatch
  C1a: MISOCP infeasible with radiality, feasible without → radiality target wrong
  C3 : MISOCP feasible + DOF=0 (all alpha==1) → force_switch_closed or fix_switch_status locking switches
  STOP: MISOCP infeasible both ways → escalate to user
"""
from __future__ import annotations

import sys
import pathlib
import traceback

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import copy
import pandapower as pp
from src.env.IEEE123bus import build_ieee123_net
from src.layer0_dso.reconfiguration import (
    extract_network_data,
    run_reconfiguration_detailed,
    switch_edge_map,
)

SEP = "=" * 72


def section(title: str) -> None:
    print(f"\n{SEP}\n## {title}\n{SEP}")


# ── 1. Switch inventory ──────────────────────────────────────────────────────

section("1. net.switch inventory  (net built like run_layer0_pipeline)")
net = build_ieee123_net(
    mode="matpower", balanced=True, convert_switches=True,
    slack_zones={1}, source_mode="publish",
)
sw = net.switch
print(f"  len(net.switch) = {len(sw)}")
for et_val, grp in sw.groupby("et"):
    print(f"    et={et_val!r}: {len(grp)} rows")

b_switches = sw[sw["et"].astype(str) == "b"]
print(f"\n  Bus-bus (et='b') switches ({len(b_switches)} total):")
for idx, row in b_switches.iterrows():
    bus_name = net.bus.at[row["bus"], "name"] if row["bus"] in net.bus.index else "?"
    elem_name = net.bus.at[row["element"], "name"] if row["element"] in net.bus.index else "?"
    print(f"    switch_idx={idx:3d}  bus={row['bus']}({bus_name}) -> elem={row['element']}({elem_name})"
          f"  closed={row['closed']}  type={row.get('type','?')}")


# ── 2. extract_network_data → switch edge_ids ────────────────────────────────

section("2. extract_network_data(net) — switch edge assignments")
data = extract_network_data(net)
print(f"  Total edges in NetworkData: {len(data.edges)}")
switch_edges = [e for e in data.edges if e.is_switch]
non_switch_edges = [e for e in data.edges if not e.is_switch]
print(f"  Non-switch edges: {len(non_switch_edges)}")
print(f"  Switch edges    : {len(switch_edges)}")
for e in switch_edges:
    print(f"    edge_id={e.edge_id}  switch_index={e.switch_index}"
          f"  from_bus={e.from_bus}  to_bus={e.to_bus}")


# ── 3. switch_edge_map → keys ────────────────────────────────────────────────

section("3. switch_edge_map (used in export_layer0_csvs)")
sw_map = switch_edge_map(net)
print(f"  switch_map keys: {sorted(sw_map.keys())}")
for eid, info in sorted(sw_map.items()):
    print(f"    edge_id={eid}  from_bus={info['from_bus']}  to_bus={info['to_bus']}")


# ── 4. run_reconfiguration_detailed — attempt 1 (full radiality) ─────────────

section("4a. run_reconfiguration_detailed  enforce_radiality=True  force_switch_closed=False")
try:
    result_r = run_reconfiguration_detailed(
        net=copy.deepcopy(net),
        enforce_radiality=True,
        force_switch_closed=False,
        debug=True,
    )
    alpha = result_r.alpha_star
    print(f"  solver_status   : OK (no exception)")
    print(f"  len(alpha_star) : {len(alpha)}")
    print(f"  alpha keys      : {sorted(alpha.keys())}")
    print(f"  alpha==1 count  : {sum(1 for v in alpha.values() if int(v)==1)}")
    print(f"  alpha==0 count  : {sum(1 for v in alpha.values() if int(v)==0)}")
    print(f"  soc_slack_max   : {result_r.soc_slack_max:.6f}")
    print(f"  soc_slack_sum   : {result_r.soc_slack_sum:.6f}")
    print(f"  voltage_drop_slack_max: {result_r.voltage_drop_slack_max:.6f}")
    overlap = set(alpha.keys()) & set(sw_map.keys())
    print(f"\n  INTERSECTION alpha_star & switch_map keys: {sorted(overlap)}")
    if not overlap:
        print("  *** C2 CANDIDATE: alpha_star keys do NOT match switch_map keys ***")
    elif all(int(v) == 1 for v in alpha.values()):
        print("  *** C3 CANDIDATE: all alpha==1, DOF=0 ***")
    else:
        print("  MISOCP feasible with real DOF — root cause elsewhere")
except Exception as exc:
    print(f"  EXCEPTION: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    result_r = None


# ── 5. run_reconfiguration_detailed — attempt 2 (no radiality) ──────────────

section("4b. run_reconfiguration_detailed  enforce_radiality=False  force_switch_closed=False")
try:
    result_nr = run_reconfiguration_detailed(
        net=copy.deepcopy(net),
        enforce_radiality=False,
        force_switch_closed=False,
        debug=False,
    )
    alpha_nr = result_nr.alpha_star
    print(f"  solver_status   : OK (no exception)")
    print(f"  len(alpha_star) : {len(alpha_nr)}")
    print(f"  alpha==1 count  : {sum(1 for v in alpha_nr.values() if int(v)==1)}")
    print(f"  alpha==0 count  : {sum(1 for v in alpha_nr.values() if int(v)==0)}")
except Exception as exc:
    print(f"  EXCEPTION: {type(exc).__name__}: {exc}")
    traceback.print_exc()


# ── 6. force_switch_closed=True (run_layer0_pipeline default) ───────────────

section("4c. run_reconfiguration_detailed  force_switch_closed=True  (run_layer0_pipeline default)")
try:
    result_fc = run_reconfiguration_detailed(
        net=copy.deepcopy(net),
        enforce_radiality=True,
        force_switch_closed=True,
        debug=False,
    )
    alpha_fc = result_fc.alpha_star
    print(f"  solver_status   : OK (no exception)")
    print(f"  len(alpha_star) : {len(alpha_fc)}")
    print(f"  alpha keys      : {sorted(alpha_fc.keys())}")
    print(f"  alpha==1 count  : {sum(1 for v in alpha_fc.values() if int(v)==1)}")
    print(f"  alpha==0 count  : {sum(1 for v in alpha_fc.values() if int(v)==0)}")
    if alpha_fc and all(int(v) == 1 for v in alpha_fc.values()):
        print("  *** C3 CONFIRMED: force_switch_closed=True locks all alpha==1 ***")
except Exception as exc:
    print(f"  EXCEPTION: {type(exc).__name__}: {exc}")
    traceback.print_exc()


# ── 7. Replay _run_day (generate_signals pipeline) ──────────────────────────

section("5. Replay generate_signals._run_day  (day_index=0)")
try:
    from scripts.generate_signals import _run_day
    from src.layer0_dso.layer0_dso import build_hourly_profiles

    gs_net = build_ieee123_net(
        mode="matpower", balanced=True, convert_switches=True,
        slack_zones=None, source_mode="publish",
    )
    print(f"  generate_signals base_net: {len(gs_net.switch)} switches, {len(gs_net.line)} lines")

    load_profiles, pv_profiles, wind_profiles = build_hourly_profiles()
    results = _run_day(0, gs_net, load_profiles, pv_profiles, wind_profiles, "load_weighted")
    if results is None:
        print("  _run_day returned None — both attempts failed")
    else:
        r0 = results[0]
        print(f"  _run_day succeeded: {len(results)} hourly results")
        print(f"  alpha_star (hour 0): {r0.alpha_star}")
        alpha0 = r0.alpha_star
        overlap_gs = set(alpha0.keys()) & set(sw_map.keys())
        print(f"  INTERSECTION with switch_map: {sorted(overlap_gs)}")
        if not overlap_gs and alpha0:
            print("  *** C2 in generate_signals path: edge_id mismatch ***")
except Exception as exc:
    print(f"  EXCEPTION in _run_day replay: {type(exc).__name__}: {exc}")
    traceback.print_exc()


# ── 8. Decision gate summary ─────────────────────────────────────────────────

section("6. Decision gate summary")
print("""
  Read the sections above and apply:

  C2  → alpha_star keys ≠ switch_map keys (empty intersection)
  C1a → MISOCP infeasible with radiality, feasible without
  C3  → MISOCP feasible, all alpha==1 (force_switch_closed or fix_switch_status)
  STOP→ MISOCP infeasible both with and without radiality
""")
