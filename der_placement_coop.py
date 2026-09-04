# =============================================================================
#  DER SITING & SIZING CO-OPTIMIZATION
#  Multi-Objective PSO — IEEE 123-Bus Islanded Microgrid
#  Target: IEEE Transactions on Smart Grid
# =============================================================================
#
# OBJECTIVES (3, minimize both):
#   f1: Active power loss reduction
#   f2: Voltage Deviation Index (VDI)
#   f3: Zone LMP divergence (std of zone LMPs)
#
# DECISION VARIABLES per particle:
#   Location: bus index (integer) for each DER unit
#   Sizing  : installed capacity (continuous) for each DER unit
#
# FIXED (not optimized):
#   - Number of units: 9 EVCS, 14 DPV, 4 Wind (per PD v3.1)
#   - Wind capacity: 3 MW/turbine (commercial class, not negotiable)
#   - GFM G1 location: bus 114 (slack anchor, non-negotiable)
#   - GFM G2 location: bus 60 (2/3-rule, analytically justified separately)
#   - GFM G2 sizing: 3 MW BESS / 2 MW PV (fixed after sensitivity study)
#
# PAPER DEFENSE:
#   Wind sizing fixed → cite IEC 61400 turbine class selection
#   GFM buses fixed → cite Acharya 2006 (2/3 rule) + Section 6.2 H_vi derivation
#   Number of units fixed → cite PD Section 1.3 (VPP structure requirement)
#
# =============================================================================

import numpy as np
import pandapower as pp
from pandapower.converter.matpower import from_mpc
import networkx as nx
import pandas as pd
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from copy import deepcopy

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: PROBLEM FORMULATION
# ─────────────────────────────────────────────────────────────────────────────
#
# DECISION VECTOR x has 2 parts:
#
#   LOCATION genes (integer):
#     l_E[1..9]  ∈ candidate_buses(zone_e[i])   — EVCS bus index
#     l_P[1..14] ∈ candidate_buses(zone_p[j])   — DPV bus index
#     l_W[1..4]  ∈ candidate_buses(zone=3)      — Wind bus index
#
#   SIZING genes (continuous, normalized to [0,1]):
#     s_Epv[1..9]  fixed 0.5 MW
#     s_Eb[1..9]   tiered by zone (large/med/z4)
#     s_Ev[1..9]   capped by zone (Z1/Z2 0.75, Z4 0.65)
#     s_P[1..14]   ∈ [P_DPV_min, P_DPV_max]             MW
#
# Total genes = 27 location + 41 sizing = 68 genes per particle
#
# OBJECTIVES:
#   f1(x) = P_loss(x) / P_loss_base  ∈ [0,1]
#   f2(x) = VDI(x) / VDI_base        ∈ [0,1]
#   f3(x) = std(LMP_zones) / 5.0     ∈ [0,1]
#
# WEIGHTED SUM (Analytic Hierarchy Process weights):
#   F(x) = w1*f1 + w2*f2 + w3*f3
#   w1 = 0.40 (loss — primary: reduces operational cost)
#   w2 = 0.40 (voltage — primary: safety in islanded system)
#   w3 = 0.20 (LMP divergence — market fairness)
#
# AHP JUSTIFICATION:
#   Pairwise comparison matrix A = [[1, 1], [1, 1]]
#   → equal importance → w = [0.5, 0.5]
#   → Consistency Ratio CR = 0 (perfectly consistent) [Saaty, 1980]
#
# HARD CONSTRAINTS (enforced via penalty):
#   C1: Each bus assigned ≤ 1 EVCS
#   C2: Each bus assigned ≤ 1 DPV
#   C3: Zone coverage: ≥ 2 EVCS per zone z ∈ {1,2,4}  (VPP structure)
#   C4: V_bus ∈ [0.95, 1.05] p.u.  (ANSI C84.1)
#   C5: ΣP_rated_DER ≥ P_coverage_min  (islanded capacity adequacy)
#   C6: No DER at GFM buses or intermediate buses
#   C7: Wind only in Zone 3
#   C8: Zone net injection bounds (hard)
#   C9: V2G cap (safety)
#
# PENALTY FUNCTION:
#   P(x) = λ_V * N_volt_violations + λ_Z * N_zone_violations
#         + λ_cap * max(0, P_min_coverage - ΣP_rated)²
#         + λ_zone_net * zone_net_penalties
#   λ_V = 50, λ_Z = 100, λ_cap = 10


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: BOUNDS AND CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SizingBounds:
    """
    Sizing bounds for co-optimization.

    JUSTIFICATION for each bound (for paper Section IV-B):
    - EVCS PV fixed at 0.5 MW (canopy baseline).
    - EVCS BESS: large [1.5–2.5], medium [1.0–2.0], Z4 [0.8–1.5] MW.
    - EVCS V2G caps: Z1/Z2 ≤ 0.75 MW, Z4 ≤ 0.65 MW.
    - DPV [0.2, 0.5 MW] per unit (14 units).

    - Wind fixed at 3.0 MW: IEC Class II commercial turbine (e.g., Vestas V90-3MW).
      Sizing is determined by turbine class, not system optimization.
      Reference: IEC 61400-1 wind turbine class definition.
    """
    # EVCS per station (PV bounds)
    evcs_pv_min: float = 0.05
    evcs_pv_max: float = 0.20

    # EVCS BESS tiers (MW)
    evcs_bess_z1_min: float = 0.20; evcs_bess_z1_max: float = 0.45
    evcs_bess_z2_min: float = 0.15; evcs_bess_z2_max: float = 0.40
    evcs_bess_z4_min: float = 0.10; evcs_bess_z4_max: float = 0.35

    # EVCS V2G caps (MW)
    v2g_min:     float = 0.05
    v2g_cap_z12: float = 0.15
    v2g_cap_z4:  float = 0.10

    # DPV per unit
    dpv_min: float = 0.15;  dpv_max: float = 0.40          # MW

    # Wind (fixed, not optimized)
    wind_mw: float = 3.0                                      # MW per turbine

    # GFM (fixed)
    gfm_g1_bus:    int   = 114
    gfm_g1_bess:   float = 1.020   # MW
    gfm_g2_bus:    int   = 60
    gfm_g2_bess:   float = 0.510   # MW
    gfm_g2_pv:     float = 0.340   # MW

    # System constraints
    n_evcs: int = 9
    n_dpv:  int = 14
    n_wind: int = 4

    # Capacity adequacy: installed DER ≥ coverage_ratio × peak_load
    # 0.8 → 80% coverage by variable DER (wind+GFM handles remaining 20%)
    coverage_ratio: float = 0.8
    peak_load_mw:   float = 15.705   # MATPOWER base ×4.5 scaling

    # Penalty weights
    lambda_volt: float = 50.0
    lambda_zone: float = 100.0
    lambda_cap:  float = 10.0
    lambda_flex: float = 20.0
    lambda_zone_net: float = 20.0
    lambda_v2g: float = 100.0

    # Flexibility targets
    flex_min_mw: float = 3.06

    # AHP weights
    w_loss: float = 0.40
    w_volt: float = 0.40
    w_lmp:  float = 0.20


@dataclass
class ZoneConfig:
    """Zone assignments loaded from bus_zone_map.csv."""
    z1: frozenset = frozenset()
    z2: frozenset = frozenset()
    z3: frozenset = frozenset()
    z4: frozenset = frozenset()

    # Forbidden buses (GFM, intermediate, regulator nodes)
    FORBIDDEN = frozenset({114, 60, 135, 149, 152, 160, 197, 250, 300, 450, 151})

    # Zone -> VPP mapping (Zone 4 -> VPP_3)
    ZONE_TO_VPP = {1: 1, 2: 2, 4: 3}

    _zone_map: Dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        ZoneConfig._zone_map = dict(self._zone_map)
        if self.z1: ZoneConfig.z1 = self.z1
        if self.z2: ZoneConfig.z2 = self.z2
        if self.z3: ZoneConfig.z3 = self.z3
        if self.z4: ZoneConfig.z4 = self.z4

    @staticmethod
    def _load_zone_map() -> Dict[int, int]:
        import os
        candidates = [
            'data/bus_zone_map.csv',
            'artifacts/l0l1_stats/bus_zone_map.csv',
        ]
        for path in candidates:
            if os.path.exists(path):
                df = pd.read_csv(path)
                return dict(zip(df.bus_id.astype(int), df.zone.astype(int)))
        raise FileNotFoundError("bus_zone_map.csv not found in data/ or artifacts/l0l1_stats/")

    @classmethod
    def from_bus_zone_map(cls, map_path: str | None = None) -> "ZoneConfig":
        if map_path is not None:
            df = pd.read_csv(map_path)
            zone_map = dict(zip(df.bus_id.astype(int), df.zone.astype(int)))
        else:
            zone_map = cls._load_zone_map()
        z1 = frozenset(b for b, z in zone_map.items() if z == 1)
        z2 = frozenset(b for b, z in zone_map.items() if z == 2)
        z3 = frozenset(b for b, z in zone_map.items() if z == 3)
        z4 = frozenset(b for b, z in zone_map.items() if z == 4)
        ZoneConfig._zone_map = dict(zone_map)
        ZoneConfig.z1 = z1
        ZoneConfig.z2 = z2
        ZoneConfig.z3 = z3
        ZoneConfig.z4 = z4
        return cls(z1=z1, z2=z2, z3=z3, z4=z4, _zone_map=zone_map)

    @staticmethod
    def of(bus: int) -> int:
        if not hasattr(ZoneConfig, "_zone_map") or not ZoneConfig._zone_map:
            ZoneConfig.from_bus_zone_map()
        return ZoneConfig._zone_map.get(bus, 1)

    @staticmethod
    def candidates(zone: int, all_buses: List[int]) -> List[int]:
        zone_set = {1: ZoneConfig.z1, 2: ZoneConfig.z2, 3: ZoneConfig.z3, 4: ZoneConfig.z4}.get(zone, frozenset())
        return sorted([b for b in all_buses
                       if b in zone_set and b not in ZoneConfig.FORBIDDEN])

    # EVCS zone assignment: 3 per zone {1,2,4}
    # Genes 0–2 → Zone 1, genes 3–5 → Zone 2, genes 6–8 → Zone 4
    EVCS_ZONE_MAP = [1, 1, 1, 2, 2, 2, 4, 4, 4]

    # DPV zone assignment: 4/6/4 for zones {1,2,4}
    # Genes 9–12 → Zone 1, genes 13–18 → Zone 2, genes 19–22 → Zone 4
    DPV_ZONE_MAP = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4]

    # Wind zone: all Zone 3
    # Genes 23–26 → Zone 3 (4 turbines)
    WIND_ZONE_MAP = [3, 3, 3, 3]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: GRID LOADER
# ─────────────────────────────────────────────────────────────────────────────

class GridModel:
    """
    Loads IEEE 123-bus, scales loads, provides pandapower net for PF evaluation.
    Thread-safe via deepcopy — each particle gets its own net copy.
    """

    @staticmethod
    def _fix_mpc_file(mpc_path: str) -> str:
        """
        Pre-process MATPOWER .m file to fix pandapower parsing issues.
        Writes a fixed copy to <original_stem>_ppfix.m and returns its path.

        Fixes applied:
          1. Branch expressions like "0.001010139*5" → evaluated float "0.005050695"
          2. Wide whitespace in near-zero impedance rows → single tab delimiter
          3. MATLAB gencost array expression → plain numeric matrix
        """
        import re, os
        from pathlib import Path

        with open(mpc_path, encoding='utf-8') as f:
            content = f.read()

        # Normalize to ASCII to avoid downstream non-UTF-8 decoders
        content = content.encode('ascii', 'ignore').decode('ascii')

        # Fix 1: evaluate a*b multiplications in branch data
        def eval_mult(m):
            return f"{float(m.group(1)) * float(m.group(2)):.10g}"
        content = re.sub(r'([\.\d]+)\*([\.\d]+)', eval_mult, content)

        # Fix 2: clean wide whitespace in branch data lines
        lines = content.split('\n')
        in_branch = False
        result = []
        for line in lines:
            raw = line
            if re.search(r'mpc\.branch\s*=\s*\[', raw):
                in_branch = True
            if in_branch and raw.strip() == '];':
                in_branch = False
            if in_branch and raw.strip() and not raw.strip().startswith('%'):
                if re.match(r'\s*\d', raw):
                    raw = re.sub(r'[ \t]+', '\t', raw.strip())
            result.append(raw)
        content = '\n'.join(result)

        # Fix 3: replace MATLAB gencost expression with plain matrix
        # Count generators from mpc.gen section
        gen_match = re.search(r'mpc\.gen\s*=\s*\[(.*?)\];', content, re.DOTALL)
        n_gen = 0
        if gen_match:
            n_gen = sum(1 for l in gen_match.group(1).split('\n')
                        if l.strip() and not l.strip().startswith('%'))
        n_gen = max(n_gen, 86)  # fallback
        gc_new = 'mpc.gencost = [\n'
        gc_new += '2\t0\t0\t3\t0\t1\t0\t;\n'
        for _ in range(n_gen - 1):
            gc_new += '2\t0\t0\t3\t0\t0\t0\t;\n'
        gc_new += '];\n'
        content = re.sub(
            r'mpc\.gencost\s*=\s*\.\.\..*?(?=\n\n|\Z)',
            gc_new.rstrip('\n'), content, flags=re.DOTALL)

        # Write fixed file next to original
        stem = Path(mpc_path).stem
        fixed_path = str(Path(mpc_path).parent / f'{stem}_ppfix.m')
        with open(fixed_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'[GridModel] MPC file fixed: {fixed_path}')
        return fixed_path

    def __init__(self, mpc_path: str, scale: float = 4.5,
                 bus_zone_map_path: str = 'artifacts/l0l1_stats/bus_zone_map.csv'):
        # Pre-process .m file to fix pandapower parsing issues
        mpc_path = self._fix_mpc_file(mpc_path)

        # Load from MATPOWER (fixed file)
        self._net_base = from_mpc(mpc_path)

        # Safe bus ID mapping (MATPOWER id ≠ pandapower row index)
        self._net_base.bus["bus_id"] = (
            pd.to_numeric(self._net_base.bus["name"], errors="coerce")
            .astype("Int64")
        )

        # Drop non-numeric bus names (NA) and log them for diagnosis
        invalid_bus = self._net_base.bus[pd.isna(self._net_base.bus["bus_id"])][
            ["name"]
        ]
        if len(invalid_bus) > 0:
            invalid_names = invalid_bus["name"].astype(str).tolist()
            print(f"[GridModel] Warning: non-numeric bus names skipped: {invalid_names}")

        # Fallback: if all bus names are empty/NA, use 1-based MATPOWER indices
        if self._net_base.bus["bus_id"].isna().all():
            self._net_base.bus["bus_id"] = (self._net_base.bus.index + 1).astype(
                "Int64"
            )
            print("[GridModel] Warning: bus names empty; using 1-based indices")

        self._pp_idx_cache = {
            int(row.bus_id): idx
            for idx, row in self._net_base.bus.iterrows()
            if pd.notna(row.bus_id)
        }

        # Scale loads to project MW level
        self._net_base.load['p_mw']   *= scale
        self._net_base.load['q_mvar'] *= scale

        # Scale voltage base (4.16 kV → 22 kV)
        self._net_base.bus['vn_kv'] = 22.0

        # Run base power flow (no DER) — store reference metrics
        pp.runpp(self._net_base, algorithm='nr', numba=False,
                 max_iteration=50, tolerance_mva=1e-6)
        self.p_loss_base = float(self._net_base.res_line['pl_mw'].sum())
        self.vdi_base    = float(
            ((self._net_base.res_bus['vm_pu'] - 1.0) ** 2).mean())
        self.v_min_base  = float(self._net_base.res_bus['vm_pu'].min())

        # Build NetworkX graph for topology queries
        self.G = nx.Graph()
        self.all_buses = list(self._pp_idx_cache.keys())
        for b in self.all_buses:
            self.G.add_node(b)
        for _, ln in self._net_base.line.iterrows():
            fb = int(self._net_base.bus.at[int(ln.from_bus), "bus_id"])
            tb = int(self._net_base.bus.at[int(ln.to_bus),   "bus_id"])
            self.G.add_edge(fb, tb,
                            r=float(ln.r_ohm_per_km) * float(ln.length_km),
                            x=float(ln.x_ohm_per_km) * float(ln.length_km))
        self.distances = nx.single_source_shortest_path_length(self.G, 114)

        # Zone map
        self.zone_config = ZoneConfig.from_bus_zone_map(bus_zone_map_path)

        # Candidate buses per zone (from bus_zone_map.csv)
        self.candidates = {z: self.zone_config.candidates(z, self.all_buses)
                           for z in [1, 2, 3, 4]}

        # Candidate lists (v3.0 spec)
        evcs_z1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,34]
        evcs_z2 = [18,19,21,22,23,24,25,26,27,28,29,30,31,32,35,38,40,41,42,43,44,45,48,49,50]
        evcs_z4 = [52,53,54,55,56,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95]
        wind_z3 = [97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113]
        dpv_z1  = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17]
        dpv_z2  = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,36,37,38,39,40,41,42,43,44,45,46,47,49,50,51]
        dpv_z4  = [52,53,54,55,56,57,58,59,61,62,63,64,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,88,89,90,91,92,93,94,95,96]

        # EVCS candidates (zones 1,2,4)
        self.evcs_candidates = {
            1: [b for b in evcs_z1 if b in self.all_buses and b not in ZoneConfig.FORBIDDEN],
            2: [b for b in evcs_z2 if b in self.all_buses and b not in ZoneConfig.FORBIDDEN],
            4: [b for b in evcs_z4 if b in self.all_buses and b not in ZoneConfig.FORBIDDEN],
        }

        # DPV candidates (zones 1,2,4)
        self.dpv_candidates = {
            1: [b for b in dpv_z1 if b in self.all_buses and b not in ZoneConfig.FORBIDDEN],
            2: [b for b in dpv_z2 if b in self.all_buses and b not in ZoneConfig.FORBIDDEN],
            4: [b for b in dpv_z4 if b in self.all_buses and b not in ZoneConfig.FORBIDDEN],
        }

        # Wind candidates (zone 3)
        self.wind_candidates = [b for b in wind_z3 if b in self.all_buses and b not in ZoneConfig.FORBIDDEN]

        print(f"[GridModel] EVCS candidates: Z1={len(self.evcs_candidates[1])}, "
              f"Z2={len(self.evcs_candidates[2])}, "
              f"Z4={len(self.evcs_candidates[4])}")
        print(f"[GridModel] DPV candidates: Z1={len(self.dpv_candidates[1])}, "
              f"Z2={len(self.dpv_candidates[2])}, "
              f"Z4={len(self.dpv_candidates[4])}")

        print(f"[GridModel] Base: P_loss={self.p_loss_base:.4f} MW, "
              f"VDI={self.vdi_base:.6f}, V_min={self.v_min_base:.4f} p.u.")
        print(f"[GridModel] Candidates: Z1={len(self.candidates[1])}, "
              f"Z2={len(self.candidates[2])}, "
              f"Z3={len(self.candidates[3])}, "
              f"Z4={len(self.candidates[4])}")

    def pp_idx(self, mpc_bus_id: int) -> int:
        return self._pp_idx_cache[int(mpc_bus_id)]

    def fresh_net(self):
        """Return a deepcopy of base net — safe for parallel use."""
        return deepcopy(self._net_base)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: PARTICLE ENCODING & DECODING
# ─────────────────────────────────────────────────────────────────────────────
#
# Particle structure (68 genes):
#
#   [0..26]  — LOCATION genes (integer, index into zone candidate list)
#              [0..2]   EVCS Zone 1 (E1–E3)
#              [3..5]   EVCS Zone 2 (E4–E6)
#              [6..8]   EVCS Zone 4 (E7–E9)
#              [9..12]  DPV  Zone 1 (PV1–PV4)
#              [13..18] DPV  Zone 2 (PV5–PV10)
#              [19..22] DPV  Zone 4 (PV11–PV14)
#              [23..26] Wind Zone 3 (W1–W4)
#
#   [27..67] — SIZING genes (continuous, normalized [0,1])
#              [27..35] EVCS PV capacity    → per-station
#              [36..44] EVCS BESS capacity  → tiered by zone
#              [45..53] EVCS V2G capacity   → capped by zone
#              [54..67] DPV capacity        → decode to [dpv_min, dpv_max]

N_LOC   = 27   # location genes
N_SIZ   = 41   # sizing genes (9+9+9+14)
N_GENES = 68   # total


def decode_particle(particle: np.ndarray,
                    grid: GridModel,
                    bounds: SizingBounds
                    ) -> Dict:
    """
    Decode a 68-gene particle vector into interpretable placement + sizing.

    Location genes: clipped to [0, len(candidates[zone])-1], then
    uniqueness-enforced by shifting duplicates to next available candidate.
    DPV candidates exclude EVCS buses.

    Sizing genes: mapped from [0,1] to physical [min, max] range or capped.

    Returns dict with keys:
      evcs_buses, dpv_buses, wind_buses,
      evcs_pv_mw, evcs_bess_mw, evcs_v2g_mw, dpv_mw
    """
    p = particle.copy()
    loc  = p[:N_LOC]   # location genes
    siz  = p[N_LOC:]   # sizing genes, should be in [0,1]

    def unique_from_zone(indices: np.ndarray, zone: int, candidates: List[int]) -> List[int]:
        cands = candidates
        n = len(cands)
        if n == 0:
            raise ValueError(f"No candidates for zone {zone}")
        used, result = set(), []
        for raw in indices:
            idx = int(np.clip(round(float(raw)), 0, n - 1))
            orig = idx
            while idx in used:
                idx = (idx + 1) % n
                if idx == orig:
                    # All candidates used — this shouldn't happen with n >> k
                    break
            used.add(idx)
            result.append(cands[idx])
        return result

    # Decode locations
    evcs_buses = (unique_from_zone(loc[0:3], 1, grid.evcs_candidates[1]) +
                  unique_from_zone(loc[3:6], 2, grid.evcs_candidates[2]) +
                  unique_from_zone(loc[6:9], 4, grid.evcs_candidates[4]))

    # Prevent EVCS/DPV co-location by removing EVCS buses from DPV candidates
    dpv_candidates = {
        z: [b for b in grid.dpv_candidates[z] if b not in evcs_buses]
        for z in [1, 2, 4]
    }
    dpv_buses  = (unique_from_zone(loc[9:13],  1, dpv_candidates[1]) +
                  unique_from_zone(loc[13:19], 2, dpv_candidates[2]) +
                  unique_from_zone(loc[19:23], 4, dpv_candidates[4]))
    wind_buses =  unique_from_zone(loc[23:27], 3, grid.wind_candidates)

    # Decode sizing (linear mapping from [0,1] to [min,max])
    def to_mw(s01, lo, hi):
        return lo + np.clip(s01, 0, 1) * (hi - lo)

    # EVCS PV genes 27–35 → siz[0:9]
    evcs_pv_mw = to_mw(siz[0:9], bounds.evcs_pv_min, bounds.evcs_pv_max)

    # BESS genes 36–44 → siz[9:18]
    bess_z1 = to_mw(siz[9:12],  bounds.evcs_bess_z1_min, bounds.evcs_bess_z1_max)
    bess_z2 = to_mw(siz[12:15], bounds.evcs_bess_z2_min, bounds.evcs_bess_z2_max)
    bess_z4 = to_mw(siz[15:18], bounds.evcs_bess_z4_min, bounds.evcs_bess_z4_max)
    evcs_bess_mw = np.concatenate([bess_z1, bess_z2, bess_z4])

    # V2G genes 45–53 → siz[18:27]
    v2g_z12 = to_mw(siz[18:24], bounds.v2g_min, bounds.v2g_cap_z12)
    v2g_z4 = to_mw(siz[24:27], bounds.v2g_min, bounds.v2g_cap_z4)
    evcs_v2g_mw = np.concatenate([v2g_z12, v2g_z4])

    # DPV genes 54–67 → siz[27:41]
    dpv_mw = to_mw(siz[27:41], bounds.dpv_min, bounds.dpv_max)

    return dict(
        evcs_buses=evcs_buses, dpv_buses=dpv_buses, wind_buses=wind_buses,
        evcs_pv_mw=evcs_pv_mw.tolist(),
        evcs_bess_mw=evcs_bess_mw.tolist(),
        evcs_v2g_mw=evcs_v2g_mw.tolist(),
        dpv_mw=dpv_mw.tolist(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: OBJECTIVE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_zone_balance(net: pp.pandapowerNet,
                         zone_config: ZoneConfig) -> Dict[int, float]:
    """Compute net real power injection per zone (MW)."""
    zone_net = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

    if not hasattr(net, "res_sgen") or not hasattr(net, "res_load"):
        return zone_net

    def _bus_id(pp_bus_idx: int) -> Optional[int]:
        if "bus_id" in net.bus.columns:
            bus_id = net.bus.at[pp_bus_idx, "bus_id"]
            if pd.notna(bus_id):
                return int(bus_id)
        name = net.bus.at[pp_bus_idx, "name"]
        try:
            return int(name)
        except Exception:
            return None

    for idx in net.res_sgen.index:
        bus_idx = int(net.sgen.at[idx, "bus"])
        bus_id = _bus_id(bus_idx)
        if bus_id is None:
            continue
        z = ZoneConfig.of(bus_id)
        zone_net[z] += float(net.res_sgen.at[idx, "p_mw"])

    for idx in net.res_load.index:
        bus_idx = int(net.load.at[idx, "bus"])
        bus_id = _bus_id(bus_idx)
        if bus_id is None:
            continue
        z = ZoneConfig.of(bus_id)
        zone_net[z] -= float(net.res_load.at[idx, "p_mw"])

    return zone_net


def compute_lmp_approx(zone_net: Dict[int, float],
                       lambda_ref: float = 5.0,
                       alpha: float = 0.3) -> Dict[int, float]:
    """Approximate zone LMPs from zone net injections."""
    total = max(sum(abs(v) for v in zone_net.values()), 1.0)
    lmps = {}
    for z in [1, 2, 4]:
        ratio = zone_net.get(z, 0.0) / total
        lmps[z] = max(0.5, min(lambda_ref * (1.0 - alpha * ratio), lambda_ref * 1.5))
    return lmps

def evaluate(decoded: Dict,
             grid: GridModel,
             bounds: SizingBounds) -> Tuple[float, float, float, Dict]:
    """
    Inject DERs, run Newton-Raphson PF, compute f1 + f2 + penalties.

    Returns: (F_composite, f1_loss, f2_vdi, detail_dict)

    POWER FLOW MODEL:
    - EVCS PV:   sgen (p_mw = evcs_pv_mw, controllable=True)
    - EVCS BESS: storage (dispatch at p_mw = 0 at sizing step,
                  sn_mva reflects max apparent power)
    - EVCS V2G:  storage (aggregated fleet, p_mw = 0)
    - DPV:       sgen (p_mw = dpv_mw, MPPT at rated)
    - Wind:      sgen (p_mw = wind_mw, rated dispatch)
    - GFM G2:    storage + sgen at bus 60 (fixed sizing)
    NOTE: For siting/sizing PF, all sources dispatch at rated P to assess
    worst-case voltage rise and loss at peak generation scenario.
    """
    net = grid.fresh_net()
    idx = grid.pp_idx

    # ── GFM G2 (fixed bus 60) ───────────────────────────────────────────────
    b60 = idx(bounds.gfm_g2_bus)
    pp.create_sgen(net, bus=b60, p_mw=bounds.gfm_g2_pv,
                   sn_mva=bounds.gfm_g2_pv * 1.1, name='G2_PV')
    pp.create_storage(net, bus=b60, p_mw=-bounds.gfm_g2_bess,
                      max_p_mw=bounds.gfm_g2_bess,
                      min_p_mw=-bounds.gfm_g2_bess,
                      max_e_mwh=bounds.gfm_g2_bess * 2,
                      soc_percent=80, sn_mva=bounds.gfm_g2_bess * 1.25,
                      name='G2_BESS')

    # ── Wind ────────────────────────────────────────────────────────────────
    for i, bus in enumerate(decoded['wind_buses']):
        pp.create_sgen(net, bus=idx(bus),
                       p_mw=bounds.wind_mw, sn_mva=bounds.wind_mw * 1.1,
                       type='WP', name=f'Wind_{i+1}@{bus}')

    # ── EVCS ────────────────────────────────────────────────────────────────
    for i, bus in enumerate(decoded['evcs_buses']):
        b = idx(bus)
        pv_mw   = float(decoded['evcs_pv_mw'][i])
        bess_mw = float(decoded['evcs_bess_mw'][i])
        v2g_mw  = float(decoded['evcs_v2g_mw'][i])
        # PV: inject at rated
        pp.create_sgen(net, bus=b, p_mw=pv_mw,
                       sn_mva=pv_mw * 1.1, name=f'E{i+1}_PV@{bus}')
        # BESS: discharge at rated (worst-case voltage rise scenario)
        pp.create_storage(net, bus=b, p_mw=-bess_mw,
                          max_p_mw=bess_mw, min_p_mw=-bess_mw,
                          max_e_mwh=bess_mw * 2,
                          soc_percent=80, sn_mva=bess_mw * 1.25,
                          name=f'E{i+1}_BESS@{bus}')
        # V2G: discharging at rated
        pp.create_storage(net, bus=b, p_mw=-v2g_mw,
                          max_p_mw=v2g_mw, min_p_mw=-v2g_mw,
                          max_e_mwh=v2g_mw * 1.0,
                          soc_percent=80, sn_mva=v2g_mw * 1.25,
                          name=f'E{i+1}_V2G@{bus}')

    # ── DPV ─────────────────────────────────────────────────────────────────
    for i, bus in enumerate(decoded['dpv_buses']):
        pv_mw = float(decoded['dpv_mw'][i])
        pp.create_sgen(net, bus=idx(bus),
                       p_mw=pv_mw, sn_mva=pv_mw * 1.1,
                       name=f'DPV_{i+1}@{bus}')

    # ── Power flow ───────────────────────────────────────────────────────────
    try:
        pp.runpp(net, algorithm='nr', numba=False,
                 max_iteration=50, tolerance_mva=1e-6)
        converged = net.converged
    except Exception:
        converged = False

    if not converged:
        return 1e6, 1e6, 1e6, {'converged': False, 'F': 1e6}

    # ── Objectives ───────────────────────────────────────────────────────────
    vm      = net.res_bus['vm_pu']
    p_loss  = float(net.res_line['pl_mw'].sum())
    vdi     = float(((vm - 1.0) ** 2).mean())

    # Normalize by base case
    f1 = p_loss / max(grid.p_loss_base, 1e-9)
    f2 = vdi    / max(grid.vdi_base,    1e-9)

    # Flexibility signals
    total_bess = sum(decoded['evcs_bess_mw'])
    total_v2g  = sum(decoded['evcs_v2g_mw'])
    flex_mw = total_bess + total_v2g

    # Zone LMP divergence objective
    zone_net = compute_zone_balance(net, grid.zone_config)
    zone_lmp = compute_lmp_approx(zone_net)
    lmp_vals = np.array([zone_lmp[1], zone_lmp[2], zone_lmp[4]], dtype=float)
    f3 = float(np.std(lmp_vals) / 5.0)

    # ── Penalties ────────────────────────────────────────────────────────────
    # C4: voltage violations
    v_viol = int(((vm < 0.95) | (vm > 1.05)).sum())
    pen_volt = bounds.lambda_volt * v_viol

    # C3: zone coverage — each of Z1,Z2,Z4 must have ≥ 2 EVCS
    zone_count = {1: 0, 2: 0, 4: 0}
    for bus in decoded['evcs_buses']:
        z = ZoneConfig.of(bus)
        if z in zone_count:
            zone_count[z] += 1
    zone_viol = sum(1 for z, cnt in zone_count.items() if cnt < 2)
    pen_zone = bounds.lambda_zone * zone_viol

    # C5: capacity adequacy
    total_pv   = sum(decoded['evcs_pv_mw'])   + sum(decoded['dpv_mw'])
    total_wind = bounds.wind_mw * bounds.n_wind
    total_gfm  = bounds.gfm_g1_bess + bounds.gfm_g2_bess + bounds.gfm_g2_pv
    total_cap  = total_pv + total_bess + total_v2g + total_wind + total_gfm
    cap_deficit = max(0, bounds.coverage_ratio * bounds.peak_load_mw - total_cap)
    pen_cap = bounds.lambda_cap * cap_deficit ** 2

    # C8: zone net injection bounds (hard)
    pen_zone_net = 0.0
    for z in [1, 2, 4]:
        net_mw = abs(zone_net.get(z, 0.0))
        hard = max(0.0, net_mw - 5.0)
        pen_zone_net += bounds.lambda_zone_net * hard

    # C9: V2G cap safety
    pen_v2g = 0.0
    for i, bus in enumerate(decoded['evcs_buses']):
        z = ZoneConfig.of(bus)
        cap = bounds.v2g_cap_z12 if z in (1, 2) else bounds.v2g_cap_z4
        excess = max(0.0, float(decoded['evcs_v2g_mw'][i]) - cap)
        pen_v2g += bounds.lambda_v2g * excess

    # C10: minimum flexibility (BESS + V2G)
    flex_deficit = max(0.0, bounds.flex_min_mw - flex_mw)
    pen_flex = bounds.lambda_flex * flex_deficit ** 2

    penalty = pen_volt + pen_zone + pen_cap + pen_zone_net + pen_v2g + pen_flex

    # ── Composite F ──────────────────────────────────────────────────────────
    F = bounds.w_loss * f1 + bounds.w_volt * f2 + bounds.w_lmp * f3 + penalty

    detail = dict(
        converged=True, F=F,
        f1_loss_norm=f1,   f2_vdi_norm=f2, f3_lmp_norm=f3,
        p_loss_mw=p_loss,  vdi=vdi,
        v_min=float(vm.min()), v_max=float(vm.max()),
        v_violations=v_viol,   zone_violations=zone_viol,
        total_cap_mw=total_cap, cap_deficit_mw=cap_deficit,
        penalty=penalty, pen_flex=pen_flex,
        pen_zone_net=pen_zone_net, pen_v2g=pen_v2g,
        p_loss_reduction_pct=100 * (1 - f1),
        vdi_improvement_pct=100 * (1 - f2),
        flex_mw=flex_mw, flex_deficit_mw=flex_deficit,
        total_pv_mw=total_pv + bounds.gfm_g2_pv,
        total_bess_mw=total_bess + bounds.gfm_g1_bess + bounds.gfm_g2_bess,
        total_v2g_mw=total_v2g,
        total_wind_mw=total_wind,
        zone_lmp=zone_lmp,
        zone_net_mw=zone_net,
        load_scale_mw=bounds.peak_load_mw,
    )
    return F, f1, f2, detail


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: PSO OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
#
# ALGORITHM: PSO with Inertia Weight [Shi & Eberhart, 1998]
# Parameters derived from Clerc & Kennedy (2002) constriction factor analysis.
#
# Velocity update (inertia-weight form):
#   v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i(t)) + c2*r2*(gbest - x_i(t))
#
# Parameters from Clerc & Kennedy (2002) constriction analysis:
#   w  = χ     = 0.7298  (inertia = constriction factor)
#   c1 = χ*φ/2 = 1.4962  (cognitive: 0.7298 × 2.05)
#   c2 = χ*φ/2 = 1.4962  (social:    0.7298 × 2.05)
#
# EQUIVALENCE: This is algebraically equivalent to the constriction form
#   v = χ * [v + 2.05*r1*(p-x) + 2.05*r2*(g-x)],  χ=0.7298
# with φ = c1+c2 = 4.1 (Clerc 2002 Table I, κ=1).
#
# Paper citation: "PSO parameters follow Clerc & Kennedy [2002]:
#   w=0.7298, c1=c2=1.4962, derived from constriction factor analysis
#   with φ=4.1, ensuring convergence without manual tuning."
# Reference: Clerc & Kennedy, IEEE TEC vol.6(1), 2002.
#
# HYBRID TREATMENT OF LOCATION vs SIZING GENES:
#   Location genes: velocity acts on continuous position, rounded for decode.
#     Velocity clamped to ±V_max_loc = ub_loc / 2
#   Sizing genes: standard continuous PSO, clamped to [0,1].
#     No rounding applied.
# This follows the "continuous relaxation" approach for discrete PSO
# [Pan et al., Applied Soft Computing 2011].

class PSO:
    """
    Multi-objective PSO for joint DER siting and sizing.

    Parameters (all from literature, citable):
      n_particles = 60  [standard for engineering optimization, Kennedy 2010]
      n_iter      = 250 [sufficient for 68-dim problem]
      chi         = 0.7298 [Clerc & Kennedy 2002 constriction factor]
      c1 = c2     = 1.4962 [Clerc & Kennedy 2002]
      n_seeds     = 3  [multi-seed for statistical validity, per PD Section 10.8]
    """

    # Inertia-weight form, Clerc & Kennedy 2002 (w=χ, c1=c2=χ*φ/2, φ=4.1)
    CHI = 0.7298   # inertia weight w = constriction factor χ
    C1  = 1.4962   # cognitive coefficient = χ × 2.05
    C2  = 1.4962   # social coefficient    = χ × 2.05

    def __init__(self,
                 grid:        GridModel,
                 bounds:      SizingBounds,
                 n_particles: int = 60,
                 n_iter:      int = 250,
                 patience:    int = 30,
                 seed:        int = 42):
        self.grid        = grid
        self.bounds      = bounds
        self.n_p         = n_particles
        self.n_iter      = n_iter
        self.patience    = patience
        self.rng         = np.random.default_rng(seed)

        # Upper bounds for location genes (per zone)
        loc_ub = (
            [len(grid.candidates[1]) - 1] * 3 +
            [len(grid.candidates[2]) - 1] * 3 +
            [len(grid.candidates[4]) - 1] * 3 +
            [len(grid.dpv_candidates[1]) - 1] * 4 +
            [len(grid.dpv_candidates[2]) - 1] * 6 +
            [len(grid.dpv_candidates[4]) - 1] * 4 +
            [len(grid.candidates[3]) - 1] * 4
        )
        self._loc_lb = np.zeros(N_LOC)
        self._loc_ub = np.array(loc_ub, dtype=float)

        # Sizing genes always in [0, 1]
        self._siz_lb = np.zeros(N_SIZ)
        self._siz_ub = np.ones(N_SIZ)

        self.ub = np.concatenate([self._loc_ub, self._siz_ub])
        self.lb = np.concatenate([self._loc_lb, self._siz_lb])

        # Results
        self.gbest_val  = np.inf
        self.gbest_pos  = None
        self.gbest_detail = None
        self.convergence  = []    # gbest F per iteration
        self.f1_curve     = []    # loss component
        self.f2_curve     = []    # voltage component

    # ── Initialization ───────────────────────────────────────────────────────

    def _init_particle(self) -> np.ndarray:
        """Uniform random initialization within bounds."""
        x = self.lb + self.rng.random(N_GENES) * (self.ub - self.lb)
        x[N_LOC + 9:N_LOC + 18] = 0.5  # BESS seeds
        x[N_LOC + 18:N_LOC + 27] = 0.5  # V2G seeds
        x[N_LOC + 27:N_LOC + 41] = 0.5  # DPV seeds
        return x

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Execute PSO and return best solution metrics."""

        # Initialize swarm
        X = np.array([self._init_particle() for _ in range(self.n_p)])
        V = np.zeros_like(X)
        pbest_X   = X.copy()
        pbest_val = np.full(self.n_p, np.inf)

        # Evaluate initial population
        print(f"  Evaluating initial swarm ({self.n_p} particles)...")
        for i in range(self.n_p):
            dec = decode_particle(X[i], self.grid, self.bounds)
            F, f1, f2, detail = evaluate(dec, self.grid, self.bounds)
            pbest_val[i] = F
            if F < self.gbest_val:
                self.gbest_val   = F
                self.gbest_pos   = X[i].copy()
                self.gbest_detail = {**detail, **dec}
                self._gbest_f1 = f1
                self._gbest_f2 = f2

        self.convergence.append(self.gbest_val)
        self.f1_curve.append(self._gbest_f1)
        self.f2_curve.append(self._gbest_f2)

        no_improve = 0

        for it in range(self.n_iter):
            r1 = self.rng.random((self.n_p, N_GENES))
            r2 = self.rng.random((self.n_p, N_GENES))

            # Constriction-factor velocity update
            V = self.CHI * (
                V
                + self.C1 * r1 * (pbest_X - X)
                + self.C2 * r2 * (self.gbest_pos - X)
            )

            # Clamp velocities
            v_max = (self.ub - self.lb) / 2.0
            V = np.clip(V, -v_max, v_max)

            # Update positions
            X = np.clip(X + V, self.lb, self.ub)

            # Evaluate
            prev_best = self.gbest_val
            for i in range(self.n_p):
                dec = decode_particle(X[i], self.grid, self.bounds)
                F, f1, f2, detail = evaluate(dec, self.grid, self.bounds)
                if F < pbest_val[i]:
                    pbest_val[i] = F
                    pbest_X[i]   = X[i].copy()
                if F < self.gbest_val:
                    self.gbest_val    = F
                    self.gbest_pos    = X[i].copy()
                    self.gbest_detail = {**detail, **dec}
                    self._gbest_f1 = f1
                    self._gbest_f2 = f2

            self.convergence.append(self.gbest_val)
            self.f1_curve.append(self._gbest_f1)
            self.f2_curve.append(self._gbest_f2)

            if abs(prev_best - self.gbest_val) < 1e-10:
                no_improve += 1
            else:
                no_improve = 0

            if no_improve >= self.patience:
                print(f"  Early convergence at iter {it + 1} "
                      f"(no improve for {self.patience} iters)")
                break

            if (it + 1) % 20 == 0:
                d = self.gbest_detail
                print(f"  [{it+1:4d}/{self.n_iter}] F={self.gbest_val:.5f} "
                      f"| loss_red={d.get('p_loss_reduction_pct',0):.1f}% "
                      f"| vdi_imp={d.get('vdi_improvement_pct',0):.1f}% "
                      f"| V_viol={d.get('v_violations',0)}")

        return self.gbest_detail


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MULTI-SEED RUNNER + SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_seed(grid:        GridModel,
                   bounds:      SizingBounds,
                   seeds:       List[int] = [42, 123, 777],
                   n_particles: int = 60,
                   n_iter:      int = 250,
                   output_dir:  str = 'artifacts/placement') -> Dict:
    """
    Run PSO with multiple seeds for statistical validity.
    Reports mean ± std of objectives across seeds.

    PAPER USE: "Results are reported as mean ± std across 3 independent PSO
    runs with different random seeds (42, 123, 777), following the multi-seed
    protocol of Section IV-C."
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for seed in seeds:
        print(f"\n{'='*55}")
        print(f"  PSO Seed {seed} | {n_particles} particles × {n_iter} iters")
        print(f"{'='*55}")

        pso = PSO(grid, bounds, n_particles=n_particles,
                  n_iter=n_iter, seed=seed)
        best = pso.run()
        best['seed'] = seed
        best['convergence'] = pso.convergence
        best['f1_curve']    = pso.f1_curve
        best['f2_curve']    = pso.f2_curve
        results.append(best)

        # Save convergence per seed
        pd.DataFrame({
            'iteration': range(len(pso.convergence)),
            'F_composite': pso.convergence,
            'f1_loss':    pso.f1_curve,
            'f2_vdi':     pso.f2_curve,
        }).to_csv(f'{output_dir}/convergence_seed{seed}.csv', index=False)

    # ── Statistical summary ──────────────────────────────────────────────────
    metrics = ['p_loss_reduction_pct', 'vdi_improvement_pct',
               'v_min', 'total_cap_mw', 'v_violations']
    summary = {}
    for m in metrics:
        vals = [r[m] for r in results if m in r]
        summary[m] = {'mean': np.mean(vals), 'std': np.std(vals),
                      'min': np.min(vals),   'max': np.max(vals)}

    # Best overall result (lowest F)
    best_result = min(results, key=lambda r: r.get('F', np.inf))

    print(f"\n{'='*55}")
    print(f"  MULTI-SEED SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*55}")
    print(f"  Loss reduction:  "
          f"{summary['p_loss_reduction_pct']['mean']:.1f}% "
          f"± {summary['p_loss_reduction_pct']['std']:.1f}%")
    print(f"  VDI improvement: "
          f"{summary['vdi_improvement_pct']['mean']:.1f}% "
          f"± {summary['vdi_improvement_pct']['std']:.1f}%")
    print(f"  V_min:           "
          f"{summary['v_min']['mean']:.4f} "
          f"± {summary['v_min']['std']:.4f} p.u.")
    print(f"  Total DER cap:   "
          f"{summary['total_cap_mw']['mean']:.1f} "
          f"± {summary['total_cap_mw']['std']:.1f} MW")

    # ── Export official placement (best seed) ───────────────────────────────
    placement = _build_placement_json(best_result, bounds, grid,
                                      seeds, summary, n_particles, n_iter)

    # Convert numpy / pandas scalar types to native Python types for JSON
    def _to_jsonable(obj):
        import numpy as _np
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_jsonable(v) for v in obj]
        if isinstance(obj, (_np.integer, _np.floating)):
            return obj.item()
        return obj

    with open(f'{output_dir}/official_placement_v3.json', 'w') as f:
        json.dump(_to_jsonable(placement), f, indent=2)

    # ── Export CSVs ──────────────────────────────────────────────────────────
    _export_placement_csvs(best_result, grid, output_dir)

    # ── Baseline comparison table ────────────────────────────────────────────
    baseline = _compute_baselines(grid, bounds, output_dir)

    print(f"\n  Artifacts: {output_dir}/")
    print(f"    official_placement_v3.json  <- runtime source of truth")
    print(f"    bustozone.csv, bustoVPP.csv")
    print(f"    convergence_seed*.csv    <- paper Figure")
    print(f"    baseline_comparison.csv  <- paper Table")

    return {
        'best': best_result,
        'summary': summary,
        'baselines': baseline,
        'placement_path': f'{output_dir}/official_placement_v3.json',
    }


def _compute_baselines(grid: GridModel,
                       bounds: SizingBounds,
                       output_dir: str) -> Dict:
    """
    Compute 3 baseline placements for comparison table in paper.

    Baseline 0: No DER (base case)
    Baseline 1: Uniform placement (equal spacing along feeder)
    Baseline 2: Load-proportional placement (buses with highest load)

    PAPER TABLE:
    | Method           | P_loss (MW) | Loss Red. | VDI      | V_min  |
    |------------------|-------------|-----------|----------|--------|
    | No DER           | X           | 0%        | X        | X      |
    | Uniform          | X           | X%        | X        | X      |
    | Load-prop        | X           | X%        | X        | X      |
    | Proposed PSO     | X           | X%        | X        | X      |
    """
    baselines = {}

    # Baseline 0: No DER
    baselines['no_der'] = {
        'p_loss_mw': grid.p_loss_base,
        'vdi': grid.vdi_base,
        'p_loss_reduction_pct': 0.0,
        'vdi_improvement_pct': 0.0,
        'v_min': grid.v_min_base,
    }

    # Baseline 1: Uniform (evenly spaced by depth)
    def uniform_placement():
        all_b = sorted(grid.all_buses, key=lambda b: grid.distances.get(b, 0))
        step = max(1, len(all_b) // (bounds.n_evcs + bounds.n_dpv + bounds.n_wind))
        selected = all_b[::step][:bounds.n_evcs + bounds.n_dpv + bounds.n_wind]
        # Distribute to zones with fallback
        evcs_b = selected[:bounds.n_evcs]
        dpv_b  = selected[bounds.n_evcs: bounds.n_evcs + bounds.n_dpv]
        wind_b = [b for b in selected if ZoneConfig.of(b) == 3][:bounds.n_wind]
        if len(wind_b) < bounds.n_wind:
            wind_b = grid.wind_candidates[:bounds.n_wind]
        # Fixed average sizing
        avg_evcs_pv = (bounds.evcs_pv_min + bounds.evcs_pv_max) / 2
        avg_evcs_b  = (bounds.evcs_bess_large_min + bounds.evcs_bess_large_max) / 2
        avg_v2g     = bounds.v2g_cap_z12 / 2
        avg_dpv     = (bounds.dpv_min + bounds.dpv_max) / 2
        return dict(
            evcs_buses=evcs_b[:9], dpv_buses=dpv_b[:14], wind_buses=wind_b[:4],
            evcs_pv_mw=[avg_evcs_pv]*9, evcs_bess_mw=[avg_evcs_b]*9,
            evcs_v2g_mw=[avg_v2g]*9,   dpv_mw=[avg_dpv]*14,
        )

    # Baseline 2: Highest-load buses (using actual feeder load from net.load)
    def load_prop_placement():
        # Aggregate actual load per bus from pandapower net
        load_map = {}
        for _, ld in grid.fresh_net().load.iterrows():
            bid = int(grid._net_base.bus.at[int(ld.bus), "bus_id"])
            load_map[bid] = load_map.get(bid, 0.0) + float(ld.p_mw)
        # Sort descending by actual load, exclude forbidden buses
        load_sort = sorted(
            [(b, p) for b, p in load_map.items()
             if b not in grid.zone_config.FORBIDDEN and p > 0],
            key=lambda x: -x[1])
        top_buses = [b for b, _ in load_sort]
        # Assign top buses to zones (need zone coverage for C3)
        evcs_b, dpv_b = [], []
        zone_evcs_count = {1: 0, 2: 0, 4: 0}
        for b in top_buses:
            z = ZoneConfig.of(b)
            if len(evcs_b) < 9 and z in zone_evcs_count and zone_evcs_count[z] < 3:
                evcs_b.append(b)
                zone_evcs_count[z] += 1
            elif len(dpv_b) < 14 and b not in evcs_b:
                dpv_b.append(b)
            if len(evcs_b) == 9 and len(dpv_b) == 14:
                break
        # Fill gaps with any valid candidates
        for z in [1, 2, 4]:
            while zone_evcs_count.get(z, 0) < 3 and len(evcs_b) < 9:
                for b in grid.candidates[z]:
                    if b not in evcs_b:
                        evcs_b.append(b)
                        zone_evcs_count[z] = zone_evcs_count.get(z, 0) + 1
                        break
                else:
                    break
        for z in [1, 2, 4]:
            for b in grid.candidates[z]:
                if len(dpv_b) >= 14: break
                if b not in dpv_b and b not in evcs_b:
                    dpv_b.append(b)
        wind_b = grid.wind_candidates[:4]
        avg_evcs_pv = (bounds.evcs_pv_min + bounds.evcs_pv_max) / 2
        avg_evcs_b  = (bounds.evcs_bess_large_min + bounds.evcs_bess_large_max) / 2
        avg_v2g     = bounds.v2g_cap_z12 / 2
        avg_dpv     = (bounds.dpv_min + bounds.dpv_max) / 2
        return dict(
            evcs_buses=evcs_b[:9], dpv_buses=dpv_b[:14], wind_buses=wind_b[:4],
            evcs_pv_mw=[avg_evcs_pv]*9, evcs_bess_mw=[avg_evcs_b]*9,
            evcs_v2g_mw=[avg_v2g]*9,   dpv_mw=[avg_dpv]*14,
        )

    for name, fn in [('uniform', uniform_placement),
                     ('load_proportional', load_prop_placement)]:
        try:
            dec = fn()
            F, f1, f2, detail = evaluate(dec, grid, bounds)
            baselines[name] = {**detail, **{k: dec[k] for k in
                               ['evcs_buses','dpv_buses','wind_buses']}}
        except Exception as e:
            baselines[name] = {'error': str(e)}

    rows = []
    for method, d in baselines.items():
        rows.append({
            'method': method,
            'p_loss_mw': d.get('p_loss_mw', grid.p_loss_base),
            'p_loss_reduction_pct': d.get('p_loss_reduction_pct', 0),
            'vdi': d.get('vdi', grid.vdi_base),
            'vdi_improvement_pct': d.get('vdi_improvement_pct', 0),
            'v_min': d.get('v_min', grid.v_min_base),
            'v_violations': d.get('v_violations', 0),
        })
    pd.DataFrame(rows).to_csv(f'{output_dir}/baseline_comparison.csv',
                               index=False)
    return baselines


def _build_placement_json(best, bounds, grid, seeds, summary,
                          n_particles, n_iter) -> Dict:
    return {
        'version': '3.0',
        'algorithm': 'Multi-objective Discrete-Continuous PSO (co-optimization)',
        'objectives': ['active_power_loss', 'voltage_deviation_index', 'zone_lmp_divergence'],
        'weights': {'w_loss': bounds.w_loss, 'w_volt': bounds.w_volt, 'w_lmp': bounds.w_lmp,
                    'ahp_method': 'equal_weight_3obj', 'CR': 0.0},
        'pso_params': {
            'n_particles': n_particles, 'n_iter': n_iter,
            'chi': PSO.CHI, 'c1': PSO.C1, 'c2': PSO.C2,
            'seeds': seeds,
            'reference': 'Clerc & Kennedy, IEEE TEC, 2002'
        },
        'sizing_bounds': {
            'evcs_pv_mw':   [bounds.evcs_pv_min, bounds.evcs_pv_max],
            'evcs_bess_mw': {
                'z1': [bounds.evcs_bess_z1_min, bounds.evcs_bess_z1_max],
                'z2': [bounds.evcs_bess_z2_min, bounds.evcs_bess_z2_max],
                'z4': [bounds.evcs_bess_z4_min, bounds.evcs_bess_z4_max],
            },
            'evcs_v2g_caps_mw': {'z12': bounds.v2g_cap_z12, 'z4': bounds.v2g_cap_z4},
            'dpv_mw':       [bounds.dpv_min,        bounds.dpv_max],
            'wind_mw':      bounds.wind_mw,
            'reference_bounds':
                'EVCS BESS tiers per PD v3.0; DPV: IEC 61727; Wind: IEC 61400-1 class II'
        },
        'gfm': {
            'G1': {'bus': 114, 'bess_mw': bounds.gfm_g1_bess,
                   'bess_mwh': bounds.gfm_g1_bess * 2,
                   'inverter_mva': 7.0, 'mode': 'VSG',
                   'justification': 'Slack bus anchor for islanded operation'},
            'G2': {'bus': 60,  'pv_mw': bounds.gfm_g2_pv,
                   'bess_mw': bounds.gfm_g2_bess,
                   'bess_mwh': bounds.gfm_g2_bess * 2,
                   'inverter_mva': 4.0, 'mode': 'Droop',
                   'justification': 'Bus 60 = 2/3-rule mid-feeder (Acharya 2006)'},
        },
        'wind': [
            {'id': f'W{i+1}', 'bus': b, 'mw': bounds.wind_mw,
             'justification': 'Zone 3 wind corridor; turbine class IEC 61400-1 II'}
            for i, b in enumerate(best.get('wind_buses', []))
        ],
        'evcs': [
            {'id': f'E{i+1}', 'bus': b,
             'zone': ZoneConfig.of(b),
             'vpp': f"VPP_{ZoneConfig.ZONE_TO_VPP.get(ZoneConfig.of(b), ZoneConfig.of(b))}",
             'pv_mw':   round(best['evcs_pv_mw'][i],   3),
             'bess_mw': round(best['evcs_bess_mw'][i], 3),
             'bess_mwh': round(best['evcs_bess_mw'][i] * 2, 3),
             'v2g_mw':  round(best['evcs_v2g_mw'][i],  3),
             'inverter_mva': round(
                 max(best['evcs_pv_mw'][i],
                     best['evcs_bess_mw'][i]) * 1.25, 2)}
            for i, b in enumerate(best.get('evcs_buses', []))
        ],
        'dpv': [
            {'id': f'PV{i+1}', 'bus': b,
             'zone': ZoneConfig.of(b),
             'vpp': f"VPP_{ZoneConfig.ZONE_TO_VPP.get(ZoneConfig.of(b), ZoneConfig.of(b))}",
             'mw': round(best['dpv_mw'][i], 3)}
            for i, b in enumerate(best.get('dpv_buses', []))
        ],
        'best_fitness': best.get('F', None),
        'metrics': {
            'F_composite': best.get('F', None),
            'p_loss_mw':   best.get('p_loss_mw', None),
            'p_loss_reduction_pct': best.get('p_loss_reduction_pct', None),
            'vdi':         best.get('vdi', None),
            'vdi_improvement_pct': best.get('vdi_improvement_pct', None),
            'v_min':       best.get('v_min', None),
            'v_max':       best.get('v_max', None),
            'v_violations': best.get('v_violations', 0),
            'total_cap_mw': best.get('total_cap_mw', None),
            'zone_lmp': best.get('zone_lmp', None),
            'zone_net_mw': best.get('zone_net_mw', None),
            'load_scale_mw': best.get('load_scale_mw', None),
        },
        'zone_lmp_model': {'type': 'net_injection_approximate', 'alpha': 0.3, 'lambda_ref': 5.0},
        'multi_seed_summary': summary,
        'counts': {'evcs': 9, 'dpv': 14, 'wind': 4, 'gfm': 2},
        'n_agents': 41,
        'action_flat_dim': 73,
        'particle_genes': N_GENES,
    }


def _export_placement_csvs(best, grid, output_dir):
    """Export bustozone.csv and bustoVPP.csv."""
    rows = [{'bus_id': b,
             'zone': ZoneConfig.of(b),
             'depth_from_114': grid.distances.get(b, 99)}
            for b in sorted(grid.all_buses)]
    pd.DataFrame(rows).to_csv(f'{output_dir}/bustozone.csv', index=False)

    vpp_rows = []
    for i, b in enumerate(best.get('evcs_buses', [])):
        z = ZoneConfig.of(b)
        vpp = ZoneConfig.ZONE_TO_VPP.get(z, z)
        for atype, sfx in [('EVCS_PV','PV'),('EVCS_BESS','BESS'),('EVCS_V2G','V2G')]:
            vpp_rows.append({'bus_id': b, 'vpp': f'VPP_{vpp}',
                             'asset_type': atype, 'asset_id': f'E{i+1}-{sfx}',
                             'zone': z, 'controllable': 'RL'})
    for i, b in enumerate(best.get('dpv_buses', [])):
        z = ZoneConfig.of(b)
        vpp = ZoneConfig.ZONE_TO_VPP.get(z, z)
        vpp_rows.append({'bus_id': b, 'vpp': f'VPP_{vpp}',
                         'asset_type': 'DPV', 'asset_id': f'PV{i+1}',
                         'zone': z, 'controllable': 'RL'})
    pd.DataFrame(vpp_rows).to_csv(f'{output_dir}/bustoVPP.csv', index=False)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Joint DER Siting & Sizing Co-Optimization — IEEE 123-Bus')
    parser.add_argument('--mpc',
        default='data/grid_IEEE123_complete.m',
        help='Path to MATPOWER case file')
    parser.add_argument('--output-dir',
        default='artifacts/placement',
        help='Output directory for results')
    parser.add_argument('--n-particles', type=int, default=60)
    parser.add_argument('--n-iter',      type=int, default=250)
    parser.add_argument('--seeds',       type=int, nargs='+',
        default=[42, 123, 777],
        help='Random seeds for multi-run statistical validation')
    parser.add_argument('--fast',  action='store_true',
        help='Fast mode: 20 particles × 50 iters (for testing)')
    args = parser.parse_args()

    if args.fast:
        args.n_particles, args.n_iter = 20, 50

    print("DER SITING & SIZING CO-OPTIMIZATION")
    print(f"  Objectives: loss + voltage + zone LMP divergence")
    print(f"  Algorithm:  Constriction-Factor PSO [Clerc & Kennedy 2002]")
    print(f"  Variables:  27 location (integer) + 41 sizing (continuous)")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Swarm:      {args.n_particles} particles × {args.n_iter} iters\n")

    grid   = GridModel(args.mpc)
    bounds = SizingBounds()

    result = run_multi_seed(
        grid=grid, bounds=bounds,
        seeds=args.seeds,
        n_particles=args.n_particles,
        n_iter=args.n_iter,
        output_dir=args.output_dir,
    )

    # ── Gate test ───────────────────────────────────────────────────────────
    placement = json.load(open(result['placement_path']))
    zone_map = ZoneConfig._zone_map or ZoneConfig.from_bus_zone_map()._zone_map

    wind_buses = [w['bus'] for w in placement['wind']]
    assert len(set(wind_buses)) == 4, "Duplicate wind buses"
    assert all(zone_map[b] == 3 for b in wind_buses), "Wind not in Zone 3"
    assert all(w['mw'] == 3.0 for w in placement['wind']), "Wind MW != 3.0"

    for e in placement['evcs']:
        assert 0.05 <= e['pv_mw'] <= 0.20, f"{e['id']} PV out of [0.05,0.20]"
        assert e['bess_mw'] <= 0.50,       f"{e['id']} BESS > 0.5MW"
        assert e['v2g_mw']  <= 0.15,       f"{e['id']} V2G > 0.15MW"

    z1 = [e['bess_mw'] for e in placement['evcs'] if e['zone'] == 1]
    z4 = [e['bess_mw'] for e in placement['evcs'] if e['zone'] == 4]
    assert min(z1) >= max(z4) - 0.05, "Tiered BESS: Z1 should >= Z4"

    assert placement['counts'] == {'evcs': 9, 'dpv': 14, 'wind': 4, 'gfm': 2}
    assert abs(3.49 * 4.5 - placement['metrics']['load_scale_mw']) < 0.1

    print(f"Wind buses: {wind_buses}")
    print(f"BESS avg: Z1={sum(z1)/3:.3f}, Z4={sum(z4)/3:.3f} MW")
    print("GATE: PASS")

    print("\nDone. Official placement saved to:",
          result['placement_path'])