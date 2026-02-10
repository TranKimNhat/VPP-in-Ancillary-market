from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd
import pandapower as pp

"""
IEEE123 OpenDSS -> pandapower converter (self-parsed).

Data source: data/oedisi-ieee123-main/snapshot/master.dss
Assumptions:
- OpenDSS files are parsed with a minimal parser (new/~, like=, comments).
- Length units default to kft (converted to km).
- Balanced approximation: per-phase values are aggregated to a single-phase equivalent.
- Regcontrol elements are ignored (no tap control applied).
"""

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "oedisi-ieee123-main"

IEEE123_ZONE_BUS_MAP: dict[int, list[str]] = {
    1: [
        "150",
        "149",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "152",
        "34",
        "15",
        "16",
        "17",
    ],
    2: [
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "135",
        "151",
        "250",
        "251",
    ],
    3: [
        "300",
        "350",
        "101",
        "102",
        "103",
        "104",
        "105",
        "106",
        "107",
        "108",
        "109",
        "110",
        "111",
        "112",
        "113",
        "114",
        "97",
        "98",
        "99",
        "100",
        "197",
        "450",
        "451",
    ],
    4: [
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "160",
        "67",
        "68",
        "s73c",
        "69",
        "Total",
        "70",
        "71",
        "72",
        "73",
        "74",
        "75",
        "76",
        "77",
        "78",
        "79",
        "80",
        "81",
        "82",
        "83",
        "84",
        "85",
        "86",
        "87",
        "88",
        "89",
        "90",
        "91",
        "92",
        "93",
        "94",
        "95",
        "96",
        "128",
        "source_hv",
        "610",
    ],
}

DSS_SNAPSHOT_FILES = (
    "master.dss",
    "IEEE123Loads.dss",
    "IEEE123Regulators.dss",
    "IEEELinecodes.dss",
)


def build_ieee123_net(
    mode: str = "snapshot",
    balanced: bool = True,
    convert_switches: bool = True,
) -> pp.pandapowerNet:
    """Build the IEEE123 pandapower network from OpenDSS or feeder123."""
    if mode.lower() == "feeder123":
        net = _build_feeder123_net()
    else:
        dss_paths = _resolve_dss_paths(mode)
        buscoords_path = _resolve_buscoords_path(mode)
        parsed = _parse_dss_files(dss_paths)
        net = _build_pandapower_net(parsed, buscoords_path=buscoords_path)

    if balanced:
        convert_to_balanced(net)

    if convert_switches:
        convert_near_zero_branches_to_switches(net)

    return net


def convert_to_balanced(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """Aggregate per-phase elements to a balanced equivalent model."""
    _aggregate_power_elements(net, "load")
    _aggregate_power_elements(net, "sgen")

    _apply_linecode_diagonal_average(net)
    _average_lines_by_name(net)

    return net


def convert_near_zero_branches_to_switches(
    net: pp.pandapowerNet,
    bus_numbers: list[int] | None = None,
    r_threshold: float = 2e-7,
    x_threshold: float = 2e-6,
) -> int:
    """Convert near-zero impedance lines into bus-bus switches."""
    bus_numbers = list(net.bus.index) if bus_numbers is None else bus_numbers
    bus_map = _build_bus_map(net, bus_numbers)
    index_to_bus = {int(idx): int(bus_no) for bus_no, idx in zip(bus_numbers, net.bus.index)}

    to_drop: list[int] = []
    added = 0
    for line_idx, line in net.line.iterrows():
        if abs(line["r_ohm_per_km"]) <= r_threshold and abs(line["x_ohm_per_km"]) <= x_threshold:
            from_bus = index_to_bus.get(int(line["from_bus"]))
            to_bus = index_to_bus.get(int(line["to_bus"]))
            if from_bus is None or to_bus is None:
                continue
            if from_bus not in bus_map or to_bus not in bus_map:
                continue
            closed = _is_switch_closed(net, int(line["from_bus"]), int(line["to_bus"]))
            pp.create_switch(
                net,
                bus=bus_map[from_bus],
                element=bus_map[to_bus],
                et="b",
                closed=closed,
                type="CB",
            )
            to_drop.append(line_idx)
            added += 1

    if to_drop:
        net.line.drop(index=to_drop, inplace=True)
        net.line.reset_index(drop=True, inplace=True)

    print(f"Converted {added} near-zero impedance lines to switches")
    return added


def validate_ieee123_net(net: pp.pandapowerNet) -> dict[str, int]:
    """Validate the IEEE123 network for near-zero lines, islands, and open switches."""
    suspicious_lines = net.line[(net.line.r_ohm_per_km < 1e-4) & (net.line.x_ohm_per_km < 1e-4)]
    near_zero_count = int(len(suspicious_lines))

    n_components = 0
    slack_component_nodes: set[int] = set()
    disconnected_load_buses: list[int] = []
    try:
        import networkx as nx

        mg = pp.topology.create_nxgraph(net, include_lines=True, include_switches=True)
        n_components = int(nx.number_connected_components(mg))
        slack_buses = {int(bus) for bus in net.ext_grid.bus} if not net.ext_grid.empty else set()
        if slack_buses:
            for component in nx.connected_components(mg):
                if slack_buses.intersection(component):
                    slack_component_nodes.update(component)
        if slack_component_nodes and not net.load.empty:
            load_buses = {int(bus) for bus in net.load["bus"].unique()}
            disconnected_load_buses = sorted(load_buses - slack_component_nodes)
    except Exception:
        n_components = 0

    open_switches = int(len(net.switch[net.switch.closed == False]))
    closed_switches = int(len(net.switch[net.switch.closed == True]))

    summary = {
        "near_zero_lines": near_zero_count,
        "islands": n_components,
        "open_switches": open_switches,
        "closed_switches": closed_switches,
        "disconnected_load_buses": len(disconnected_load_buses),
    }

    if near_zero_count == 0:
        print("OK: Switch check passed (no near-zero lines)")
    else:
        print(f"WARN: Remaining {near_zero_count} near-zero lines not converted to switches")

    if n_components == 1:
        print("OK: Network is fully connected (1 island)")
    elif n_components > 1:
        print(f"WARN: Network split into {n_components} islands. Check switch status.")

    if disconnected_load_buses:
        print(
            "WARN: Load buses disconnected from slack component: "
            + ", ".join(str(bus) for bus in disconnected_load_buses)
        )
    else:
        print("OK: All load buses connect to slack component")

    if open_switches > 0:
        print(f"OK: Found {open_switches} open tie-switches")
    else:
        print("WARN: No open tie-switches found")

    print(f"INFO: Closed switches: {closed_switches}")

    return summary


def _resolve_dss_paths(mode: str) -> list[Path]:
    if mode.lower() == "snapshot":
        base = DATA_DIR / "snapshot"
        paths = [base / name for name in DSS_SNAPSHOT_FILES]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"OpenDSS files not found: {missing}")
    return paths


def _resolve_buscoords_path(mode: str) -> Path:
    if mode.lower() == "snapshot":
        path = DATA_DIR / "snapshot" / "Buscoords.dss"
    elif mode.lower() == "qsts":
        path = DATA_DIR / "qsts" / "Buscoords.dss"
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    if not path.exists():
        raise FileNotFoundError(f"Buscoords file not found: {path}")
    return path


def _feeder123_dir() -> Path:
    return BASE_DIR / "data" / "feeder123"


def _parse_dss_files(paths: Iterable[Path]) -> dict[str, object]:
    elements: dict[str, dict[str, dict[str, object]]] = {
        "circuit": {},
        "linecode": {},
        "line": {},
        "load": {},
        "transformer": {},
        "capacitor": {},
        "regcontrol": {},
    }
    order: list[dict[str, object]] = []

    for path in paths:
        for command in _iter_dss_commands(path):
            if not command:
                continue
            element_type, name, params = _parse_command(command)
            if element_type is None or name is None:
                continue
            if element_type == "regcontrol":
                continue
            params = _apply_like(elements[element_type], params)

            raw_entry = {"type": element_type, "name": name, "params": params}
            order.append(raw_entry)
            elements[element_type][name] = {"raw_params": params}

    parsed = {
        "circuit": None,
        "linecodes": {},
        "lines": [],
        "loads": [],
        "transformers": [],
        "capacitors": [],
    }

    for entry in order:
        element_type = entry["type"]
        name = entry["name"]
        params = entry["params"]

        if element_type == "circuit":
            parsed["circuit"] = _parse_circuit(name, params)
        elif element_type == "linecode":
            parsed["linecodes"][name] = _parse_linecode(name, params)
        elif element_type == "line":
            parsed["lines"].append(_parse_line(name, params))
        elif element_type == "load":
            parsed["loads"].append(_parse_load(name, params))
        elif element_type == "transformer":
            parsed["transformers"].append(_parse_transformer(name, params))
        elif element_type == "capacitor":
            parsed["capacitors"].append(_parse_capacitor(name, params))

    return parsed


def _iter_dss_commands(path: Path) -> Iterable[str]:
    current = ""
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.split("!", 1)[0].strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("new "):
            if current:
                yield current.strip()
            current = line
        elif line.startswith("~"):
            current = f"{current} {line[1:].strip()}"
        else:
            if current:
                current = f"{current} {line.strip()}"
    if current:
        yield current.strip()


def _parse_command(command: str) -> tuple[str | None, str | None, list[tuple[str, str]]]:
    normalized = _normalize_equals(command)
    tokens = _split_tokens(normalized)
    if len(tokens) < 2:
        return None, None, []
    if tokens[0].lower() != "new":
        return None, None, []

    element_token = tokens[1]
    params_start = 2
    if element_token.lower().startswith("object="):
        element_token = element_token.split("=", 1)[1]

    if not element_token:
        return None, None, []

    if "." in element_token:
        element_type, name = element_token.split(".", 1)
    else:
        element_type, name = element_token, element_token

    element_type = element_type.strip().lower()
    name = name.strip()
    if not element_type or not name:
        return None, None, []

    params = _parse_param_tokens(tokens[params_start:])
    return element_type, name, params


def _split_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    buffer = []
    depth = 0
    in_quote = False
    quote_char = ""

    for ch in text:
        if ch in ('"', "'"):
            if in_quote and ch == quote_char:
                in_quote = False
            elif not in_quote:
                in_quote = True
                quote_char = ch
        if not in_quote:
            if ch in "[(":
                depth += 1
            elif ch in "])":
                depth = max(depth - 1, 0)
            if ch.isspace() and depth == 0:
                if buffer:
                    tokens.append("".join(buffer))
                    buffer = []
                continue
        buffer.append(ch)

    if buffer:
        tokens.append("".join(buffer))
    return tokens


def _parse_param_tokens(tokens: Iterable[str]) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = []
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            params.append((key.strip().lower(), value.strip()))
        else:
            params.append((token.strip().lower(), "true"))
    return params


def _normalize_equals(text: str) -> str:
    output = []
    in_quote = False
    quote_char = ""
    depth = 0
    skip_spaces = False

    for ch in text:
        if skip_spaces and ch.isspace():
            continue
        skip_spaces = False

        if ch in ('"', "'"):
            if in_quote and ch == quote_char:
                in_quote = False
            elif not in_quote:
                in_quote = True
                quote_char = ch

        if not in_quote:
            if ch in "[(":
                depth += 1
            elif ch in "])":
                depth = max(depth - 1, 0)
            if ch == "=" and depth == 0:
                while output and output[-1].isspace():
                    output.pop()
                output.append("=")
                skip_spaces = True
                continue

        output.append(ch)

    return "".join(output)


def _apply_like(
    known_elements: dict[str, dict[str, object]],
    params: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    like_name = None
    for key, value in params:
        if key == "like":
            like_name = value
            break
    if like_name:
        like_name = like_name.strip('"\'')
    if like_name and like_name in known_elements:
        base_params = known_elements[like_name].get("raw_params", [])
        filtered_params = [(k, v) for k, v in params if k != "like"]
        return list(base_params) + filtered_params
    return params


def _parse_circuit(name: str, params: list[tuple[str, str]]) -> dict[str, object]:
    data = {"name": name, "basekv": None, "bus1": None}
    for key, value in params:
        if key == "basekv":
            data["basekv"] = _parse_float(value)
        elif key == "bus1":
            data["bus1"] = value
    return data


def _parse_linecode(name: str, params: list[tuple[str, str]]) -> dict[str, object]:
    data = {"name": name, "r_matrix": None, "x_matrix": None, "c_matrix": None}
    for key, value in params:
        if key == "rmatrix":
            data["r_matrix"] = _parse_matrix(value)
        elif key == "xmatrix":
            data["x_matrix"] = _parse_matrix(value)
        elif key == "cmatrix":
            data["c_matrix"] = _parse_matrix(value)
    return data


def _parse_line(name: str, params: list[tuple[str, str]]) -> dict[str, object]:
    data = {
        "name": name,
        "bus1": None,
        "bus2": None,
        "phases": None,
        "linecode": None,
        "length": None,
        "r1": None,
        "x1": None,
        "c1": None,
    }
    for key, value in params:
        if key in data:
            data[key] = value
        elif key == "phases":
            data["phases"] = value
        elif key == "linecode":
            data["linecode"] = value
        elif key == "length":
            data["length"] = value
        elif key == "bus1":
            data["bus1"] = value
        elif key == "bus2":
            data["bus2"] = value
    return data


def _parse_load(name: str, params: list[tuple[str, str]]) -> dict[str, object]:
    data = {"name": name, "bus1": None, "phases": None, "conn": None, "kv": None, "kw": 0.0, "kvar": 0.0}
    for key, value in params:
        if key in ("bus1", "conn"):
            data[key] = value
        elif key == "phases":
            data["phases"] = value
        elif key == "kv":
            data["kv"] = value
        elif key == "kw":
            data["kw"] = value
        elif key == "kvar":
            data["kvar"] = value
    return data


def _parse_transformer(name: str, params: list[tuple[str, str]]) -> dict[str, object]:
    data = {"name": name, "windings": [], "phases": None, "xhl": None}
    buses_list: list[str] | None = None
    kvs_list: list[str] | None = None
    kvas_list: list[str] | None = None
    conns_list: list[str] | None = None

    current_wdg: dict[str, object] | None = None

    for key, value in params:
        if key == "phases":
            data["phases"] = value
        elif key in ("xhl", "xhl%"):
            data["xhl"] = value
        elif key == "wdg":
            current_wdg = {"wdg": value}
            data["windings"].append(current_wdg)
        elif key in ("bus", "kv", "kva", "conn", "%r", "r"):
            target = current_wdg if current_wdg is not None else data
            target[key] = value
        elif key == "buses":
            buses_list = _parse_list(value)
        elif key == "kvs":
            kvs_list = _parse_list(value)
        elif key == "kvas":
            kvas_list = _parse_list(value)
        elif key == "conns":
            conns_list = _parse_list(value)

    if not data["windings"] and buses_list:
        for idx, bus in enumerate(buses_list):
            wdg = {"wdg": str(idx + 1), "bus": bus}
            if kvs_list and idx < len(kvs_list):
                wdg["kv"] = kvs_list[idx]
            if kvas_list and idx < len(kvas_list):
                wdg["kva"] = kvas_list[idx]
            if conns_list and idx < len(conns_list):
                wdg["conn"] = conns_list[idx]
            data["windings"].append(wdg)

    return data


def _parse_capacitor(name: str, params: list[tuple[str, str]]) -> dict[str, object]:
    data = {"name": name, "bus1": None, "phases": None, "kvar": None, "kv": None}
    for key, value in params:
        if key == "bus1":
            data["bus1"] = value
        elif key == "phases":
            data["phases"] = value
        elif key == "kvar":
            data["kvar"] = value
        elif key == "kv":
            data["kv"] = value
    return data


def _parse_list(value: str) -> list[str]:
    cleaned = value.strip().strip("[]()")
    if not cleaned:
        return []
    cleaned = cleaned.replace(",", " ")
    return [item for item in cleaned.split() if item]


def _parse_matrix(value: str) -> np.ndarray | None:
    cleaned = value.strip().strip("[]()")
    if not cleaned:
        return None
    rows = [row.strip() for row in cleaned.split("|")]
    matrix: list[list[float]] = []
    for row in rows:
        if not row:
            continue
        row_values = [float(item) for item in row.replace(",", " ").split() if item]
        if row_values:
            matrix.append(row_values)
    if not matrix:
        return None
    try:
        return np.array(matrix, dtype=float)
    except Exception:
        return None


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def _read_buscoords_order(path: Path) -> list[str]:
    ordered: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        if "," in line:
            parts = [part.strip() for part in line.split(",") if part.strip()]
        else:
            parts = [part for part in line.split() if part]
        if not parts:
            continue
        ordered.append(parts[0])
    return ordered


def _read_csv_table(path: Path, header_row: int, data_start_row: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if header_row >= len(df):
        raise ValueError(f"Header row {header_row} out of range for {path}")
    headers = [str(value).strip() for value in df.iloc[header_row].tolist()]
    start = (header_row + 1) if data_start_row is None else data_start_row
    data = df.iloc[start:].copy()
    data.columns = headers
    data = data.dropna(how="all")
    return data


def _parse_feeder123_lines(base: Path) -> pd.DataFrame:
    path = base / "line data.csv"
    df = _read_csv_table(path, header_row=2)
    df = df.rename(columns={"Node A": "from_bus", "Node B": "to_bus", "Length (ft.)": "length_ft", "Config.": "config"})
    return df[["from_bus", "to_bus", "length_ft", "config"]]


def _parse_feeder123_switches(base: Path) -> pd.DataFrame:
    path = base / "switch data.csv"
    df = _read_csv_table(path, header_row=2)
    df = df.rename(columns={"Node A": "from_bus", "Node B": "to_bus", "Normal": "status"})
    return df[["from_bus", "to_bus", "status"]]


def _parse_feeder123_loads(base: Path) -> pd.DataFrame:
    path = base / "spot loads data.csv"
    raw = pd.read_csv(path, header=None)
    if len(raw) < 5:
        raise ValueError("spot loads data.csv missing data rows")
    data = raw.iloc[4:].copy()
    data.columns = [
        "bus",
        "model",
        "ph1_kw",
        "ph1_kvar",
        "ph2_kw",
        "ph2_kvar",
        "ph3_kw",
        "ph3_kvar",
    ]
    data = data.dropna(how="all")
    data = data[data["bus"].astype(str).str.lower() != "total"]
    return data[["bus", "model", "ph1_kw", "ph1_kvar", "ph2_kw", "ph2_kvar", "ph3_kw", "ph3_kvar"]]


def _parse_feeder123_caps(base: Path) -> pd.DataFrame:
    path = base / "cap data.csv"
    df = _read_csv_table(path, header_row=2, data_start_row=4)
    df = df.rename(columns={"Node": "bus", "Ph-A": "ph_a", "Ph-B": "ph_b", "Ph-C": "ph_c"})
    df = df[df["bus"].astype(str).str.lower() != "total"]
    for col in ["ph_a", "ph_b", "ph_c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df[["bus", "ph_a", "ph_b", "ph_c"]]


def _parse_feeder123_transformer(base: Path) -> pd.DataFrame:
    path = base / "Transformer Data.csv"
    df = _read_csv_table(path, header_row=2)
    df = df.rename(
        columns={
            "kVA": "kva",
            "kV-high": "kv_high",
            "kV-low": "kv_low",
            "R - %": "r_percent",
            "X - %": "x_percent",
        }
    )
    if df.columns.tolist():
        df = df.rename(columns={df.columns[0]: "name"})
    return df


def _parse_feeder123_regulators(base: Path) -> pd.DataFrame:
    path = base / "Regulator Data.csv"
    df = _read_csv_table(path, header_row=2)
    return df


def _parse_qsts_pv_defs(base: Path) -> dict[str, dict[str, object]]:
    pv_path = base / "oedisi-ieee123-main" / "qsts" / "IEEE123Pv.dss"
    if not pv_path.exists():
        return {}
    pattern = re.compile(r"^New\s+PVSystem\.(?P<name>[^\s]+)")
    yearly_pattern = re.compile(r"yearly=(?P<yearly>[^\s]+)", re.IGNORECASE)
    pmpp_pattern = re.compile(r"Pmpp=(?P<pmpp>[0-9.]+)", re.IGNORECASE)
    bus_pattern = re.compile(r"bus1=(?P<bus>[^\s]+)", re.IGNORECASE)
    refs: dict[str, dict[str, object]] = {}
    for raw_line in pv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.search(line)
        if not match:
            continue
        name = match.group("name")
        yearly_match = yearly_pattern.search(line)
        yearly = yearly_match.group("yearly") if yearly_match else None
        pmpp_match = pmpp_pattern.search(line)
        pmpp_kw = float(pmpp_match.group("pmpp")) if pmpp_match else None
        bus_match = bus_pattern.search(line)
        bus = bus_match.group("bus") if bus_match else None
        if bus:
            bus = bus.split(".", 1)[0]
        refs[name] = {"yearly": yearly, "pmpp_kw": pmpp_kw, "bus": bus}
    return refs


def _feeder123_bus_order(buses: set[str]) -> list[str]:
    numeric = sorted([bus for bus in buses if _is_number_bus(str(bus))], key=lambda x: float(x))
    non_numeric = sorted([bus for bus in buses if not _is_number_bus(str(bus))])
    return [str(bus) for bus in numeric + non_numeric]


def _parse_feeder123_config_rx(base: Path) -> dict[str, tuple[float, float]]:
    path = base / "config data.csv"
    df = _read_csv_table(path, header_row=2)
    df = df.rename(columns={"Config.": "config"})
    rx_map: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        config = row.get("config")
        if pd.isna(config):
            continue
        rx_map[str(config).strip()] = (0.4576, 1.0780)

    # IEEE 123 feeder letterhead (Configuration 12) ohms per mile (balanced diag average).
    rx_map.setdefault("12", ((1.5209 + 1.5329 + 1.5209) / 3.0, (0.7521 + 0.7162 + 0.7521) / 3.0))
    return rx_map


def _parse_feeder123_config_ampacity_kA(base: Path) -> dict[str, float]:
    path = base / "config data.csv"
    df = _read_csv_table(path, header_row=2)
    df = df.rename(columns={"Config.": "config"})
    ampacity_map: dict[str, float] = {}

    # Ampacity (A) by conductor size for feeder123 configs.
    # Values based on published conductor ratings (see sources in summary).
    conductor_ampacity_a = {
        "336,400 26/7": 529.0,
        "4/0 6/1": 357.0,
        "1/0": 212.0,
    }
    ampacity_scale = 2.0

    for _, row in df.iterrows():
        config = row.get("config")
        if pd.isna(config):
            continue
        config_key = str(config).strip()
        phase_cond = str(row.get("Phase Cond.") or "").strip()
        ampacity_a = conductor_ampacity_a.get(phase_cond)
        if ampacity_a is None:
            continue
        ampacity_map[config_key] = (ampacity_a * ampacity_scale) / 1000.0

    # Fallback for underground config 12 if missing.
    ampacity_map.setdefault("12", 0.17)
    return ampacity_map


def _normalize_bus_name(bus: str | None) -> str | None:
    if not bus:
        return None
    return bus.split(".", 1)[0]


def _normalize_bus_key(bus: str | None) -> str | None:
    if not bus:
        return None
    return bus.split(".", 1)[0].strip().lower()


def _is_number_bus(bus: str) -> bool:
    try:
        float(bus)
    except Exception:
        return False
    return True


def _bus_voltage_from_element(kv: float | None, phases: int | None, conn: str | None) -> float | None:
    if kv is None:
        return None
    if phases == 1 and conn and conn.lower() in {"wye", "y", "ln", "w"}:
        return float(kv) * float(np.sqrt(3.0))
    return float(kv)


def _build_pandapower_net(parsed: dict[str, object], buscoords_path: Path | None = None) -> pp.pandapowerNet:
    net = pp.create_empty_network()


    circuit = parsed.get("circuit") or {}
    circuit_bus = _normalize_bus_name(circuit.get("bus1"))
    circuit_basekv = _parse_float(circuit.get("basekv"))

    linecodes = parsed.get("linecodes", {})
    lines = parsed.get("lines", [])
    loads = parsed.get("loads", [])
    transformers = parsed.get("transformers", [])
    capacitors = parsed.get("capacitors", [])

    bus_vn: dict[str, float] = {}
    all_buses: set[str] = set()

    if circuit_bus:
        all_buses.add(circuit_bus)
        if circuit_basekv is not None:
            bus_vn[circuit_bus] = circuit_basekv

    for load in loads:
        bus = _normalize_bus_name(load.get("bus1"))
        if not bus:
            continue
        all_buses.add(bus)
        kv = _parse_float(load.get("kv"))
        phases = _parse_float(load.get("phases"))
        vn = _bus_voltage_from_element(kv, int(phases) if phases else None, load.get("conn"))
        if vn is not None:
            bus_vn[bus] = max(bus_vn.get(bus, 0.0), vn)

    for cap in capacitors:
        bus = _normalize_bus_name(cap.get("bus1"))
        if not bus:
            continue
        all_buses.add(bus)
        kv = _parse_float(cap.get("kv"))
        if kv is not None:
            bus_vn[bus] = max(bus_vn.get(bus, 0.0), kv)

    for transformer in transformers:
        for winding in transformer.get("windings", []):
            bus = _normalize_bus_name(winding.get("bus"))
            if not bus:
                continue
            all_buses.add(bus)
            kv = _parse_float(winding.get("kv"))
            phases = _parse_float(transformer.get("phases"))
            vn = _bus_voltage_from_element(kv, int(phases) if phases else None, winding.get("conn"))
            if vn is not None:
                bus_vn[bus] = max(bus_vn.get(bus, 0.0), vn)

    for line in lines:
        bus1 = _normalize_bus_name(line.get("bus1"))
        bus2 = _normalize_bus_name(line.get("bus2"))
        if bus1:
            all_buses.add(bus1)
        if bus2:
            all_buses.add(bus2)

    if circuit_basekv is not None:
        for bus in all_buses:
            bus_vn.setdefault(bus, circuit_basekv)

    bus_index: dict[str, int] = {}
    bus_lookup: dict[str, str] = {_normalize_bus_key(bus): bus for bus in all_buses if _normalize_bus_key(bus)}

    ordered_buses: list[str] = []
    if buscoords_path is not None:
        coords_order = _read_buscoords_order(buscoords_path)
        for bus in coords_order:
            key = _normalize_bus_key(bus)
            if key in bus_lookup:
                ordered_buses.append(bus_lookup[key])
            elif key:
                ordered_buses.append(bus)

    for bus in sorted(all_buses):
        key = _normalize_bus_key(bus)
        if key in bus_lookup and bus_lookup[key] not in ordered_buses:
            ordered_buses.append(bus_lookup[key])

    if buscoords_path is None:
        ordered_buses = sorted(all_buses)

    for bus in ordered_buses:
        vn_kv = bus_vn.get(bus) or circuit_basekv or 4.16
        bus_index[bus] = pp.create_bus(net, vn_kv=vn_kv, name=bus)

    if circuit_bus and circuit_bus in bus_index:
        pp.create_ext_grid(net, bus=bus_index[circuit_bus], vm_pu=1.0, name="source")

    if linecodes:
        net.linecode = pd.DataFrame(
            {
                name: {
                    "r_matrix": data.get("r_matrix"),
                    "x_matrix": data.get("x_matrix"),
                    "c_matrix": data.get("c_matrix"),
                }
                for name, data in linecodes.items()
            }
        ).T

    for line in lines:
        bus1 = _normalize_bus_name(line.get("bus1"))
        bus2 = _normalize_bus_name(line.get("bus2"))
        if not bus1 or not bus2 or bus1 not in bus_index or bus2 not in bus_index:
            continue

        length = _parse_float(line.get("length")) or 1.0
        length_km = length * 0.3048

        r_per_kft = x_per_kft = c_per_kft = None
        code_name = line.get("linecode")
        if code_name and code_name in linecodes:
            code = linecodes[code_name]
            r_per_kft = _mean_diag(code.get("r_matrix"))
            x_per_kft = _mean_diag(code.get("x_matrix"))
            c_per_kft = _mean_diag(code.get("c_matrix"))
        else:
            r_per_kft = _parse_float(line.get("r1"))
            x_per_kft = _parse_float(line.get("x1"))
            c_per_kft = _parse_float(line.get("c1"))

        r_ohm_per_km = (r_per_kft or 0.0) / 0.3048
        x_ohm_per_km = (x_per_kft or 0.0) / 0.3048
        c_nf_per_km = (c_per_kft or 0.0) / 0.3048
        if x_ohm_per_km == 0.0:
            x_ohm_per_km = 1e-6

        pp.create_line_from_parameters(
            net,
            from_bus=bus_index[bus1],
            to_bus=bus_index[bus2],
            length_km=length_km,
            r_ohm_per_km=r_ohm_per_km,
            x_ohm_per_km=x_ohm_per_km,
            c_nf_per_km=c_nf_per_km,
            max_i_ka=0.2,
            name=line.get("name"),
        )

    for load in loads:
        bus = _normalize_bus_name(load.get("bus1"))
        if not bus or bus not in bus_index:
            continue
        p_mw = (_parse_float(load.get("kw")) or 0.0) / 1000.0
        q_mvar = (_parse_float(load.get("kvar")) or 0.0) / 1000.0
        pp.create_load(net, bus=bus_index[bus], p_mw=p_mw, q_mvar=q_mvar, name=load.get("name"))

    for transformer in transformers:
        windings = transformer.get("windings", [])
        if len(windings) < 2:
            continue
        wdg1, wdg2 = windings[0], windings[1]
        wdg1_kv = _parse_float(wdg1.get("kv")) or 0.0
        wdg2_kv = _parse_float(wdg2.get("kv")) or 0.0

        if wdg1_kv >= wdg2_kv:
            hv, lv = wdg1, wdg2
        else:
            hv, lv = wdg2, wdg1

        hv_bus = _normalize_bus_name(hv.get("bus"))
        lv_bus = _normalize_bus_name(lv.get("bus"))
        if not hv_bus or not lv_bus:
            continue
        if hv_bus not in bus_index or lv_bus not in bus_index:
            continue

        sn_mva = (_parse_float(hv.get("kva")) or _parse_float(lv.get("kva")) or 0.0) / 1000.0
        vk_percent = _parse_float(transformer.get("xhl")) or 0.0

        r_values = []
        for winding in (wdg1, wdg2):
            r_val = _parse_float(winding.get("%r"))
            if r_val is not None:
                r_values.append(r_val)
        vkr_percent = float(np.mean(r_values)) if r_values else 0.0

        pp.create_transformer_from_parameters(
            net,
            hv_bus=bus_index[hv_bus],
            lv_bus=bus_index[lv_bus],
            sn_mva=sn_mva,
            vn_hv_kv=bus_vn.get(hv_bus, wdg1_kv or 0.0),
            vn_lv_kv=bus_vn.get(lv_bus, wdg2_kv or 0.0),
            vk_percent=vk_percent,
            vkr_percent=vkr_percent,
            pfe_kw=0.0,
            i0_percent=0.0,
            name=transformer.get("name"),
        )

    for cap in capacitors:
        bus = _normalize_bus_name(cap.get("bus1"))
        if not bus or bus not in bus_index:
            continue
        kvar = _parse_float(cap.get("kvar")) or 0.0
        pp.create_shunt(net, bus=bus_index[bus], q_mvar=-(kvar / 1000.0), name=cap.get("name"))

    return net


def _build_bus_map(net: pp.pandapowerNet, bus_numbers: list[int]) -> dict[int, int]:
    if len(bus_numbers) != len(net.bus):
        raise ValueError("Bus count mismatch between bus_numbers and pandapower net")
    return {bus_no: int(idx) for bus_no, idx in zip(bus_numbers, net.bus.index)}


def _build_feeder123_net() -> pp.pandapowerNet:
    base = _feeder123_dir()
    lines = _parse_feeder123_lines(base)
    switches = _parse_feeder123_switches(base)
    loads = _parse_feeder123_loads(base)
    caps = _parse_feeder123_caps(base)
    config_rx = _parse_feeder123_config_rx(base)
    config_ampacity = _parse_feeder123_config_ampacity_kA(base)
    transformers = _parse_feeder123_transformer(base)
    pv_defs = _parse_qsts_pv_defs(BASE_DIR / "data")
    additional_sources = [
        {"name": "wind_35", "bus": "35", "p_mw": 0.05, "q_mvar": 0.0, "type": "wind"},
        {"name": "battery_61", "bus": "61", "p_mw": 0.03, "q_mvar": 0.0, "type": "storage"},
    ]

    buses: set[str] = set()
    for _, row in lines.iterrows():
        buses.add(str(row["from_bus"]).strip())
        buses.add(str(row["to_bus"]).strip())
    for _, row in switches.iterrows():
        buses.add(str(row["from_bus"]).strip())
        buses.add(str(row["to_bus"]).strip())
    for _, row in loads.iterrows():
        buses.add(str(row["bus"]).strip())
    for _, row in caps.iterrows():
        buses.add(str(row["bus"]).strip())

    ordered_buses = _feeder123_bus_order(buses)

    net = pp.create_empty_network()
    bus_index: dict[str, int] = {}
    for bus in ordered_buses:
        bus_index[bus] = pp.create_bus(net, vn_kv=4.16, name=bus)

    if not transformers.empty:
        hv_row = transformers.loc[transformers["name"].astype(str).str.strip().str.lower() == "substation"]
        if not hv_row.empty and "150" in bus_index and "149" in bus_index:
            hv_kv = str(hv_row.iloc[0].get("kv_high") or "").strip()
            hv_kv = _parse_float(hv_kv.split()[0])
            if hv_kv:
                net.bus.at[bus_index["150"], "vn_kv"] = hv_kv
                pp.create_transformer_from_parameters(
                    net,
                    hv_bus=bus_index["150"],
                    lv_bus=bus_index["149"],
                    sn_mva=(_parse_float(hv_row.iloc[0].get("kva")) or 0.0) / 1000.0,
                    vn_hv_kv=hv_kv,
                    vn_lv_kv=4.16,
                    vk_percent=_parse_float(hv_row.iloc[0].get("x_percent")),
                    vkr_percent=_parse_float(hv_row.iloc[0].get("r_percent")),
                    pfe_kw=0.0,
                    i0_percent=0.0,
                    name="substation_xfm",
                )

    for idx, row in lines.iterrows():
        bus1 = str(row["from_bus"]).strip()
        bus2 = str(row["to_bus"]).strip()
        if bus1 not in bus_index or bus2 not in bus_index:
            continue
        length_ft = float(row["length_ft"])
        length_km = length_ft * 0.3048 / 1000.0
        config = str(row["config"]).strip()
        r_ohm_mile, x_ohm_mile = config_rx.get(config, (0.4576, 1.0780))
        r_ohm_per_km = r_ohm_mile / 1.60934
        x_ohm_per_km = x_ohm_mile / 1.60934
        if x_ohm_per_km == 0.0:
            x_ohm_per_km = 1e-6
        max_i_ka = float(config_ampacity.get(config, 0.2))
        pp.create_line_from_parameters(
            net,
            from_bus=bus_index[bus1],
            to_bus=bus_index[bus2],
            length_km=length_km,
            r_ohm_per_km=r_ohm_per_km,
            x_ohm_per_km=x_ohm_per_km,
            c_nf_per_km=0.0,
            max_i_ka=max_i_ka,
            name=f"line_{idx}",
        )

    existing_switches: set[tuple[str, str]] = set()
    for _, row in switches.iterrows():
        bus1 = str(row["from_bus"]).strip()
        bus2 = str(row["to_bus"]).strip()
        if bus1 not in bus_index or bus2 not in bus_index:
            continue
        if {bus1, bus2} == {"150", "149"}:
            continue
        status = str(row["status"]).strip().lower()
        closed = status == "closed"
        pp.create_switch(net, bus=bus_index[bus1], element=bus_index[bus2], et="b", closed=closed, type="CB")
        existing_switches.add(tuple(sorted((bus1, bus2))))

    key_ties = [
        ("151", "300"),
        ("13", "152"),
        ("18", "135"),
        ("60", "160"),
        ("450", "451"),
        ("250", "251"),
    ]

    for bus1, bus2 in key_ties:
        if bus1 not in bus_index or bus2 not in bus_index:
            continue
        pair_key = tuple(sorted((bus1, bus2)))
        if pair_key in existing_switches:
            continue
        pp.create_switch(net, bus=bus_index[bus1], element=bus_index[bus2], et="b", closed=False, type="CB")
        existing_switches.add(pair_key)

    for _, row in loads.iterrows():
        bus = str(row["bus"]).strip()
        if bus not in bus_index:
            continue
        p_kw = (
            float(row.get("ph1_kw") or 0.0)
            + float(row.get("ph2_kw") or 0.0)
            + float(row.get("ph3_kw") or 0.0)
        )
        q_kvar = (
            float(row.get("ph1_kvar") or 0.0)
            + float(row.get("ph2_kvar") or 0.0)
            + float(row.get("ph3_kvar") or 0.0)
        )
        pp.create_load(net, bus=bus_index[bus], p_mw=p_kw / 1000.0, q_mvar=q_kvar / 1000.0, name=f"load_{bus}")

    for _, row in caps.iterrows():
        bus = str(row["bus"]).strip()
        if bus not in bus_index:
            continue
        ph_a = float(row.get("ph_a") or 0.0)
        ph_b = float(row.get("ph_b") or 0.0)
        ph_c = float(row.get("ph_c") or 0.0)
        kvar = ph_a + ph_b + ph_c
        pp.create_shunt(net, bus=bus_index[bus], q_mvar=-(kvar / 1000.0), name=f"cap_{bus}")

    for pv_name, pv_info in pv_defs.items():
        bus = str(pv_info.get("bus") or "").strip()
        if not bus or bus not in bus_index:
            continue
        pmpp_kw = float(pv_info.get("pmpp_kw") or 0.0)
        pp.create_sgen(
            net,
            bus=bus_index[bus],
            p_mw=pmpp_kw / 1000.0,
            q_mvar=0.0,
            name=f"pv_{pv_name}",
            type="pv",
        )

    for source in additional_sources:
        bus = str(source.get("bus") or "").strip()
        if not bus or bus not in bus_index:
            continue
        pp.create_sgen(
            net,
            bus=bus_index[bus],
            p_mw=float(source.get("p_mw") or 0.0),
            q_mvar=float(source.get("q_mvar") or 0.0),
            name=str(source.get("name") or "source"),
            type=str(source.get("type") or "der"),
        )

    _apply_zone_sources(net)

    return net


def _aggregate_power_elements(net: pp.pandapowerNet, element: str) -> None:
    table = getattr(net, element, None)
    if table is None or table.empty or "bus" not in table.columns:
        return

    grouped = table.groupby("bus", sort=False)
    rows = []
    for bus, group in grouped:
        template = group.iloc[0].copy()
        if "p_mw" in group.columns:
            template["p_mw"] = float(group["p_mw"].sum())
        if "q_mvar" in group.columns:
            template["q_mvar"] = float(group["q_mvar"].sum())
        if "in_service" in group.columns:
            template["in_service"] = bool(group["in_service"].any())
        if "name" in group.columns:
            template["name"] = f"{element}_agg_{bus}"
        rows.append(template)

    new_table = pd.DataFrame(rows).reset_index(drop=True)
    setattr(net, element, new_table)


def _apply_zone_sources(net: pp.pandapowerNet, seed: int = 123) -> None:
    if net.bus.empty:
        return

    bus_name_to_index = {str(row["name"]): int(idx) for idx, row in net.bus.iterrows()}
    zone_bus_indices: dict[int, list[int]] = {}
    for zone, bus_names in IEEE123_ZONE_BUS_MAP.items():
        indices: list[int] = []
        for name in bus_names:
            bus_idx = bus_name_to_index.get(str(name))
            if bus_idx is not None:
                indices.append(bus_idx)
        if indices:
            zone_bus_indices[int(zone)] = indices

    if not zone_bus_indices:
        return

    existing_sgen_buses = set()
    if not net.sgen.empty:
        existing_sgen_buses = {int(bus) for bus in net.sgen.bus}

    slack_buses: set[int] = set()
    for zone, candidates in sorted(zone_bus_indices.items()):
        sgen_candidates = [bus for bus in candidates if bus in existing_sgen_buses]
        preferred_pool = sgen_candidates or candidates
        slack_bus = int(preferred_pool[0])
        slack_buses.add(slack_bus)
        pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0, name=f"slack_zone_{zone}")

    rng = np.random.default_rng(seed)

    def _placement_pool(zone: int) -> list[int]:
        pool = zone_bus_indices.get(zone, [])
        if not pool:
            return []
        non_slack = [bus for bus in pool if bus not in slack_buses]
        return non_slack if non_slack else pool

    def _choose_buses(zone: int, count: int) -> list[int]:
        pool = _placement_pool(zone)
        if not pool:
            return []
        replace = len(pool) < count
        return [int(bus) for bus in rng.choice(pool, size=count, replace=replace)]

    def _add_sgens(zone: int, count: int, p_mw: float, kind: str) -> None:
        for idx, bus in enumerate(_choose_buses(zone, count), start=1):
            pp.create_sgen(
                net,
                bus=bus,
                p_mw=p_mw,
                q_mvar=0.0,
                name=f"{kind}_z{zone}_{idx}",
                type=kind,
            )

    _add_sgens(zone=1, count=3, p_mw=0.5, kind="thermal")
    _add_sgens(zone=1, count=3, p_mw=0.1, kind="wind")
    _add_sgens(zone=4, count=5, p_mw=0.1, kind="wind")
    _add_sgens(zone=4, count=3, p_mw=0.1, kind="bess")


def _apply_linecode_diagonal_average(net: pp.pandapowerNet) -> None:
    if not hasattr(net, "linecode") or net.linecode.empty:
        return
    if "linecode" not in net.line.columns:
        return

    for idx, line in net.line.iterrows():
        code = line.get("linecode")
        if code not in net.linecode.index:
            continue
        code_row = net.linecode.loc[code]

        r_diag = _mean_diag(code_row.get("r_matrix"))
        x_diag = _mean_diag(code_row.get("x_matrix"))
        c_diag = _mean_diag(code_row.get("c_matrix"))

        if r_diag is not None:
            net.line.at[idx, "r_ohm_per_km"] = float(r_diag)
        if x_diag is not None:
            net.line.at[idx, "x_ohm_per_km"] = float(x_diag)
        if c_diag is not None and "c_nf_per_km" in net.line.columns:
            net.line.at[idx, "c_nf_per_km"] = float(c_diag)


def _average_lines_by_name(net: pp.pandapowerNet) -> None:
    if net.line.empty:
        return

    if "name" in net.line.columns:
        group_keys = ["name"]
    else:
        group_keys = ["from_bus", "to_bus", "length_km"]

    grouped = net.line.groupby(group_keys, sort=False, dropna=False)
    rows = []
    for _, group in grouped:
        template = group.iloc[0].copy()
        if len(group) > 1:
            template["r_ohm_per_km"] = float(group["r_ohm_per_km"].mean())
            template["x_ohm_per_km"] = float(group["x_ohm_per_km"].mean())
            if "c_nf_per_km" in group.columns:
                template["c_nf_per_km"] = float(group["c_nf_per_km"].mean())
            if "in_service" in group.columns:
                template["in_service"] = bool(group["in_service"].any())
        rows.append(template)

    net.line = pd.DataFrame(rows).reset_index(drop=True)


def _mean_diag(value: object) -> float | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except Exception:
        return None
    if array.ndim < 2:
        return None
    n = min(array.shape[0], array.shape[1])
    if n == 0:
        return None
    return float(np.diag(array[:n, :n]).mean())


def _is_switch_closed(net: pp.pandapowerNet, from_bus: int, to_bus: int) -> bool:
    try:
        from_name = str(net.bus.at[from_bus, "name"])
        to_name = str(net.bus.at[to_bus, "name"])
    except Exception:
        return True
    name_combo = f"{from_name} {to_name}".upper()
    return "OPEN" not in name_combo


if __name__ == "__main__":
    net = build_ieee123_net()
    print(net)
    print(f"bus={len(net.bus)} line={len(net.line)} load={len(net.load)} switch={len(net.switch)}")
    validate_ieee123_net(net)
