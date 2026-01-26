from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.converter import from_mpc


def _patch_matpowercaseframes_encoding(encoding: str = "latin-1") -> bool:
    try:
        import matpowercaseframes.core as mpc_core
    except Exception:
        return False

    if getattr(mpc_core, "_patched_encoding", None) == encoding:
        return True

    def _read_matpower(self, filepath, allow_any_keys=False):
        with open(filepath, encoding=encoding, errors="replace") as handle:
            string = handle.read()

        self.name = mpc_core.find_name(string)
        self._attributes = []

        for attribute in mpc_core.find_attributes(string):
            if attribute not in mpc_core.ATTRIBUTES and not allow_any_keys:
                continue

            list_ = mpc_core.parse_file(attribute, string)
            if list_ is not None:
                if attribute in {"version", "baseMVA"}:
                    value = list_[0][0]
                elif attribute in {"bus_name", "branch_name", "gen_name"}:
                    value = mpc_core.pd.Index([name[0] for name in list_], name=attribute)
                else:
                    n_cols = max(len(row) for row in list_)
                    value = self._get_dataframe(attribute, list_, n_cols)

                self.setattr(attribute, value)

    mpc_core.CaseFrames._read_matpower = _read_matpower
    mpc_core._patched_encoding = encoding
    return True

BASE_DIR = Path(__file__).resolve().parent
MPC_PATH = BASE_DIR / "grid_IEEE123.m"
SWITCH_CSV_PATH = BASE_DIR / "tie_switch.csv"


def _build_mv_oberrhein_net() -> pp.pandapowerNet:
    try:
        return pn.mv_oberrhein()
    except ImportError as exc:
        raise ImportError(
            "SimBench is required for mv_oberrhein(). Install simbench and retry."
        ) from exc


def _build_bus_map(net: pp.pandapowerNet, bus_numbers: list[int]) -> dict[int, int]:
    if len(bus_numbers) != len(net.bus):
        raise ValueError("Bus count mismatch between MATPOWER file and pandapower net")
    return {bus_no: int(idx) for bus_no, idx in zip(bus_numbers, net.bus.index)}


def _iter_switch_rows(csv_path: Path) -> Iterable[tuple[int, int, str, bool]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 5:
                continue
            _, node_a, node_b, switch_type, closed = (cell.strip() for cell in row[:5])
            if not (node_a.isdigit() and node_b.isdigit() and closed.isdigit()):
                continue
            yield int(node_a), int(node_b), switch_type or "LBS", bool(int(closed))


def _read_bus_numbers_from_mpc(mpc_path: Path) -> list[int]:
    data = mpc_path.read_text(encoding="latin-1")
    lines = data.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("mpc.bus"):
            start = idx
            break
    if start is None:
        raise ValueError("mpc.bus section not found")

    bus_numbers: list[int] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("%") or not stripped:
            continue
        if stripped.startswith("];"):
            break
        if stripped.endswith(";"):
            stripped = stripped[:-1]
        parts = stripped.split()
        if not parts:
            continue
        if parts[0].isdigit():
            bus_numbers.append(int(parts[0]))
    if not bus_numbers:
        raise ValueError("No bus numbers parsed from mpc.bus")
    return bus_numbers


def add_switches_from_csv(net: pp.pandapowerNet, csv_path: Path, bus_numbers: list[int]) -> int:
    bus_map = _build_bus_map(net, bus_numbers)
    missing = []
    added = 0
    for node_a, node_b, switch_type, closed in _iter_switch_rows(csv_path):
        if node_a not in bus_map or node_b not in bus_map:
            missing.append((node_a, node_b))
            continue
        pp.create_switch(
            net,
            bus=bus_map[node_a],
            element=bus_map[node_b],
            et="b",
            closed=closed,
            type=switch_type or "LBS",
        )
        added += 1
    if missing:
        print(f"Warning: missing buses for switch pairs (skipped): {missing}")
    if added:
        print(f"Added {added} bus-bus switches from {csv_path}")
    return added


def convert_near_zero_branches_to_switches(
    net: pp.pandapowerNet,
    bus_numbers: list[int],
    r_threshold: float = 2e-7,
    x_threshold: float = 2e-6,
) -> int:
    bus_map = _build_bus_map(net, bus_numbers)
    index_to_bus = {int(idx): int(bus_no) for bus_no, idx in zip(bus_numbers, net.bus.index)}
    to_drop = []
    added = 0
    for line_idx, line in net.line.iterrows():
        if abs(line["r_ohm_per_km"]) <= r_threshold and abs(line["x_ohm_per_km"]) <= x_threshold:
            from_bus = index_to_bus.get(int(line["from_bus"]))
            to_bus = index_to_bus.get(int(line["to_bus"]))
            if from_bus is None or to_bus is None:
                continue
            if from_bus not in bus_map or to_bus not in bus_map:
                continue
            pp.create_switch(net, bus=bus_map[from_bus], element=bus_map[to_bus], et="b", closed=True, type="CB")
            to_drop.append(line_idx)
            added += 1
    if to_drop:
        net.line.drop(index=to_drop, inplace=True)
        net.line.reset_index(drop=True, inplace=True)
    print(f"Converted {added} near-zero impedance lines to closed switches")
    return added


def build_ieee123_net(
    mpc_path: Path = MPC_PATH,
    tie_switch_csv: Path = SWITCH_CSV_PATH,
    r_threshold: float = 2e-7,
    x_threshold: float = 2e-6,
    mpc_encoding: str = "latin-1",
) -> pp.pandapowerNet:
    return _build_mv_oberrhein_net()


def _build_edge_index(net: pp.pandapowerNet) -> np.ndarray:
    edges: list[tuple[int, int]] = []
    for _, line in net.line.iterrows():
        edges.append((int(line["from_bus"]), int(line["to_bus"])))
        edges.append((int(line["to_bus"]), int(line["from_bus"])))
    for _, sw in net.switch.iterrows():
        if str(sw.get("et", "")) != "b":
            continue
        if not bool(sw.get("closed", True)):
            continue
        edges.append((int(sw["bus"]), int(sw["element"])))
        edges.append((int(sw["element"]), int(sw["bus"])))
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(edges, dtype=np.int64).T


def _build_node_features(
    voltage: np.ndarray,
    p_load: np.ndarray,
    q_load: np.ndarray,
) -> np.ndarray:
    return np.column_stack([voltage, p_load, q_load]).astype(np.float32)


def _collect_bus_loads(net: pp.pandapowerNet) -> tuple[np.ndarray, np.ndarray]:
    p_load = np.zeros(len(net.bus), dtype=np.float32)
    q_load = np.zeros(len(net.bus), dtype=np.float32)
    if net.load.empty:
        return p_load, q_load
    for _, row in net.load.iterrows():
        bus = int(row["bus"])
        p_load[bus] += float(row.get("p_mw", 0.0))
        q_load[bus] += float(row.get("q_mvar", 0.0))
    return p_load, q_load


def build_ieee123_gnn_inputs(net: pp.pandapowerNet) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    voltage = np.ones(len(net.bus), dtype=np.float32)
    if hasattr(net, "res_bus") and not net.res_bus.empty and "vm_pu" in net.res_bus:
        voltage = net.res_bus["vm_pu"].to_numpy(dtype=np.float32)

    p_load, q_load = _collect_bus_loads(net)
    switch_closed = net.switch["closed"].astype(bool).to_numpy() if not net.switch.empty else np.array([], dtype=bool)

    obs = {
        "V": voltage,
        "P": p_load,
        "Q": q_load,
        "switch_status": switch_closed,
    }

    pyg_data = {
        "edge_index": _build_edge_index(net),
        "x": _build_node_features(voltage, p_load, q_load),
    }

    return obs, pyg_data


def export_ieee123_json(net: pp.pandapowerNet, output_path: Path) -> None:
    pp.to_json(net, str(output_path))


def _print_summary(net: pp.pandapowerNet) -> None:
    print(net)
    print(f"bus={len(net.bus)} line={len(net.line)} load={len(net.load)} switch={len(net.switch)}")


def validate_net(net: pp.pandapowerNet) -> dict[str, int]:
    suspicious_lines = net.line[(net.line.r_ohm_per_km < 1e-4) & (net.line.x_ohm_per_km < 1e-4)]
    near_zero_count = int(len(suspicious_lines))

    n_components = 0
    try:
        import networkx as nx

        mg = pp.topology.create_nxgraph(net, include_lines=True, include_switches=True)
        n_components = int(nx.number_connected_components(mg))
    except Exception:
        n_components = 0

    open_switches = int(len(net.switch[net.switch.closed == False]))

    summary = {
        "near_zero_lines": near_zero_count,
        "islands": n_components,
        "open_switches": open_switches,
    }

    if near_zero_count == 0:
        print("OK: Switch check passed (no near-zero lines)")
    else:
        print(f"WARN: Remaining {near_zero_count} near-zero lines not converted to switches")

    if n_components == 1:
        print("OK: Network is fully connected (1 island)")
    elif n_components > 1:
        print(f"WARN: Network split into {n_components} islands. Check switch status.")

    if open_switches > 0:
        print(f"OK: Found {open_switches} open tie-switches")
    else:
        print("WARN: No open tie-switches found")

    return summary


if __name__ == "__main__":
    net = build_ieee123_net()
    _print_summary(net)
    validate_net(net)
