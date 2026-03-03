# Tri-Layer DSO–VPP Coordination on IEEE 123-bus: Current Implementation Description

**Version:** Evidence-backed snapshot (2026-03-03)
**Purpose:** Publication-ready technical description aligned to the current repository state, with explicit separation between implemented, partial, and planned work.

---

## 1) Executive Summary (No Over-Claim)

This repository implements a **tri-layer coordination pipeline** for distribution operation:

- **Layer 0 (DSO):** reconfiguration + zonal pricing pipeline with AC quality-gate outputs.
- **Layer 1 (VPP):** DRO-style dispatch on Layer-0 price signals, with optional per-VPP mode.
- **Layer 2 (Control):** GAT-MAPPO-style multi-agent control environment and training/eval scripts.

Current runtime is configured for a **stable baseline**:

- **Static zoning only** (`zoning_mode: static`), enforced by runtime guardrails.
- **Single static topology mode** by default (`topology.mode: single_static`).
- **Reserve pricing default = `legacy_zero`**, so reserve prices in current artifacts are zero.
- **EVCS integration is available behind config toggle** (`configs/evcs_config.yaml::enabled`), default OFF.

This document classifies claims as:

- **I (Implemented)**: code + evidence in tests/artifacts.
- **P (Partial/Scaffolded)**: implemented hooks/tests, but incomplete end-to-end evidence.
- **L (Planned)**: roadmap items not yet experimentally validated in current artifacts.

---

## 2) Contributions Table (I / P / L)

| ID | Contribution | Status | Evidence |
|---|---|---|---|
| C1 | Tri-layer pipeline (Layer0→Layer1→Layer2 scripts/modules) | **I** | `src/layer0_dso/layer0_dso.py`, `src/layer1_vpp/layer1_vpp.py`, `experiments/train_mappo.py`, `experiments/eval_policy.py` |
| C2 | Static zoning runtime guardrails | **I** | `configs/env_config.yaml`, `src/environment/grid_env.py`, `experiments/train_mappo.py`, `tests/test_training_smoke.py` |
| C3 | Canonical mapping contracts + fallback behavior | **I** | `src/environment/vpp_mapping.py`, `src/environment/grid_env.py` |
| C4 | Reserve pricing modes (`legacy_zero`, `proxy_ratio`, `duals_if_available`) | **I** | `configs/training_config.yaml`, `src/layer0_dso/zonal_pricing.py`, `tests/test_reserve_pricing.py` |
| C5 | Current artifact reserve signal is all-zero under default mode | **I** | `configs/training_config.yaml` (`legacy_zero`), `data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv` |
| C6 | Stochastic Layer1 scenario generation mode | **I** | `src/layer1_vpp/scenario_generator.py`, `src/layer1_vpp/layer1_vpp.py`, `tests/test_scenario_generator_stochastic.py` |
| C7 | Topology dataset enumeration + split tooling | **I** | `src/layer0_dso/topology_enumerator.py`, `experiments/generate_topology_dataset.py`, `tests/test_topology_enumerator.py` |
| C8 | End-to-end dynamic-topology runtime generalization in training/eval | **P** | Tooling exists, but runtime still enforces static zoning; default topology mode is static |
| C9 | EVCS runtime hooks in network/env/reward/safety | **P** | `configs/evcs_config.yaml`, `src/environment/grid_env.py`, `src/layer2_control/reward.py`, `src/layer2_control/safety_layer.py`, `tests/test_evcs_integration_runtime.py` |
| C10 | EVCS publication-grade benchmark outcomes (obligation fulfillment/revenue gains) | **L** | Not evidenced by current artifacts |
| C11 | Non-zero reserve commitment outcomes in Layer1 | **L** | Current `layer1_pref.csv` has `R_commit=0` throughout |

---

## 3) Architecture and Data Flow (Current State)

## 3.1 Layer structure

1. **Layer 0 (DSO):**
   - Builds/validates IEEE123 network and computes zonal energy/reserve prices.
   - Exports diagnostics including AC validity and SOCP-AC gap.

2. **Layer 1 (VPP):**
   - Consumes Layer0 prices.
   - Runs DRO scheduling in aggregated or per-VPP mode.

3. **Layer 2 (Control):**
   - Multi-agent environment over grid state.
   - Applies safety clipping/projection constraints and computes reward components.

## 3.2 Runtime defaults and switches

- **Environment:** `configs/env_config.yaml`
  - `zoning_mode: static`
  - `vpp_mode: false`
  - `topology.mode: single_static`
- **Training:** `configs/training_config.yaml`
  - `layer0.reserve_pricing.mode: legacy_zero`
  - `layer1.scenario_mode: deterministic_3day`
- **EVCS:** `configs/evcs_config.yaml`
  - `enabled: false` (default)

## 3.3 Guardrails and compatibility

- Non-static zoning is explicitly rejected in runtime/training entry points.
- Mapping behavior supports canonical CSV contracts and legacy fallback with warnings.
- Layer1 has compatibility path for legacy aggregate output when per-VPP mode is enabled.

---

## 4) Current Verified Implementation Snapshot

## 4.1 What is implemented now (evidence-backed)

- Layer0 quality-gate pipeline and CSV exports:
  - `src/layer0_dso/layer0_dso.py`
  - `tests/test_layer0_quality_gate.py`
- Layer1 DRO scheduling + scenario mode support:
  - `src/layer1_vpp/layer1_vpp.py`
  - `src/layer1_vpp/scenario_generator.py`
  - `tests/test_layer0_layer1_io.py`
  - `tests/test_scenario_generator_stochastic.py`
- Layer2 environment/training loop:
  - `src/environment/grid_env.py`
  - `experiments/train_mappo.py`
  - `experiments/eval_policy.py`
  - `tests/test_training_smoke.py`
- Reserve pricing mode implementation and tests:
  - `src/layer0_dso/zonal_pricing.py`
  - `tests/test_reserve_pricing.py`
- Topology tooling and tests:
  - `src/layer0_dso/topology_enumerator.py`
  - `experiments/generate_topology_dataset.py`
  - `tests/test_topology_enumerator.py`
- EVCS hooks and smoke/runtime tests:
  - `src/evcs/*`
  - `experiments/generate_evcs_profiles.py`
  - `tests/test_evcs_model.py`
  - `tests/test_evcs_integration_runtime.py`

## 4.2 What is partial/scaffolded

- **Dynamic topology generalization claim:**
  - Dataset/enumerator tooling is implemented.
  - Full end-to-end dynamic-topology runtime evidence is not established in current baseline artifacts.
- **EVCS maturity:**
  - Runtime hooks and tests exist, but default remains OFF and no publication-grade benchmark table is present in current artifacts.
- **Reserve pricing economics in baseline artifacts:**
  - Multiple reserve modes are implemented, but default baseline artifacts use `legacy_zero`.

---

## 5) Observed Results (Artifact-Backed Only)

All values below are derived from repository artifacts at the paths listed.

## 5.1 Layer 0 observed outputs

**Source:** `data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv`

- Rows: **1152**
- Day set: **offpeak, median, peak**
- Zones: **1,2,3,4**
- `energy_price`: min **0.0541**, max **1.1037**, mean **0.3061**
- `reserve_price`: **all zero** in current artifact
- AC gate flags: `ac_valid=True` and `ac_converged=True` throughout current file
- `socp_ac_gap_max` observed max: **0.00482**

## 5.2 Layer 1 observed outputs

**Source:** `data/oedisi-ieee123-main/profiles/layer1_vpp/layer1_pref.csv`

- Rows: **96**
- `solver_status`: **optimal**
- Sum(`P_ref`) = **0.0**
- Sum(`R_commit`) = **0.0**
- `price_reserve_expected`: **0.0** throughout current artifact

## 5.3 Layer 2 observed outputs

**Source:** `artifacts/logs/train_metrics.csv`

- Current file rows: **2 updates**
- Last recorded update fields include:
  - `reward_mean`: **-2.9644**
  - `tracking_error`: **2.6396**
  - `voltage_violation`: **0.06085**
  - `curtailment_ratio`: **0.0**

> Note: `Result.md` reports a different historical snapshot (10 updates). This document prioritizes **current artifact files** as ground truth.

---

## 6) Planned Experiments / Expected Outcomes (Future Tense)

The following are **not** claimed as current results:

1. Enable non-default reserve pricing modes (`proxy_ratio` / `duals_if_available`) and regenerate Layer0/Layer1 artifacts to assess non-zero reserve behavior.
2. Run EVCS-enabled end-to-end experiments (`configs/evcs_config.yaml::enabled=true`) with benchmark tables for obligation satisfaction and dispatch impact.
3. Integrate topology dataset mode more broadly into evaluation scripts and report generalization gaps on explicit split sets.
4. Calibrate EVCS degradation economics and reward coupling for publication-grade sensitivity studies.

---

## 7) Risks and Limitations

1. **Default reserve mode bias:** Baseline with `legacy_zero` suppresses reserve-price signal, limiting Layer1 reserve commitment behavior.
2. **Dynamic-topology claim scope:** Tooling exists, but baseline runtime evidence remains centered on static configuration.
3. **EVCS benchmark gap:** Presence of hooks/tests does not yet substitute for full comparative benchmarks.
4. **Artifact drift risk:** Narrative metrics must always be re-synced with current artifact files before publication.

---

## 8) Evidence & Reproducibility Appendix

## 8.1 Core configs

- `configs/env_config.yaml`
- `configs/training_config.yaml`
- `configs/evcs_config.yaml`

## 8.2 Core pipeline code

- `experiments/train_mappo.py`
- `experiments/eval_policy.py`
- `src/environment/grid_env.py`
- `src/environment/vpp_mapping.py`
- `src/layer0_dso/layer0_dso.py`
- `src/layer0_dso/zonal_pricing.py`
- `src/layer1_vpp/layer1_vpp.py`
- `src/layer0_dso/topology_enumerator.py`

## 8.3 Verification tests

- `tests/test_layer0_quality_gate.py`
- `tests/test_layer0_layer1_io.py`
- `tests/test_training_smoke.py`
- `tests/test_topology_enumerator.py`
- `tests/test_evcs_integration_runtime.py`
- `tests/test_reserve_pricing.py`
- `tests/test_scenario_generator_stochastic.py`
- `tests/test_evcs_model.py`

## 8.4 Artifact files used for observed metrics

- `data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv`
- `data/oedisi-ieee123-main/profiles/layer1_vpp/layer1_pref.csv`
- `artifacts/logs/train_metrics.csv`
- `artifacts/results/ieee123_setup_report.md`
- Cross-check context: `Result.md`, `check.md`

---

## 9) Claim Ledger (Detailed)

| Claim | Status | Evidence | Caveat |
|---|---|---|---|
| Static zoning enforced in runtime | **I** | `configs/env_config.yaml`; guards in `src/environment/grid_env.py` and `experiments/train_mappo.py`; `tests/test_training_smoke.py` | Dynamic zoning currently rejected by design |
| Reserve-pricing modes implemented | **I** | `src/layer0_dso/zonal_pricing.py`; `tests/test_reserve_pricing.py`; toggle in `configs/training_config.yaml` | Baseline artifact still uses `legacy_zero` |
| Layer0 AC quality-gate is active | **I** | `src/layer0_dso/layer0_dso.py`; `tests/test_layer0_quality_gate.py`; diagnostics columns in Layer0 CSV | Requires checking diagnostics each run |
| Layer1 supports stochastic scenarios | **I** | `src/layer1_vpp/scenario_generator.py`; `src/layer1_vpp/layer1_vpp.py`; `tests/test_scenario_generator_stochastic.py` | Baseline default remains deterministic mode |
| Topology dataset generator and splits are available | **I** | `src/layer0_dso/topology_enumerator.py`; `experiments/generate_topology_dataset.py`; `tests/test_topology_enumerator.py` | Not yet fully reflected in baseline training/eval reporting |
| EVCS assets can be integrated via runtime flag | **P** | `configs/evcs_config.yaml`; `src/environment/grid_env.py`; `tests/test_evcs_integration_runtime.py` | Default disabled; full benchmark evidence pending |
| End-to-end topology generalization has been demonstrated | **P** | Tooling exists (enumerator/splits) | Current baseline artifacts do not yet prove broad generalization outcomes |
| Current baseline shows non-zero reserve dispatch outcomes | **L** | N/A | Current Layer0/Layer1 artifacts show zero reserve signal/commit |
| EVCS obligation fulfillment >99% currently demonstrated | **L** | N/A | Not reported in current artifacts |

---

## 10) Terminology Consistency Notes

- **Observed results** = values present in current artifact files.
- **Expected/planned outcomes** = future experiment targets, not current evidence.
- **Implemented vs Partial vs Planned** follows I/P/L taxonomy above and is mandatory for all major claims.

---

## 11) Minimal Reproducibility Runbook (Current Baseline)

Run from repository root.

```bash
# 1) Generate Layer0 artifacts (default reserve mode from config is legacy_zero)
python experiments/train_mappo.py --bootstrap-tri-layer --training-config configs/training_config.yaml --env-config configs/env_config.yaml

# 2) (Optional) Evaluate a trained policy checkpoint
python experiments/eval_policy.py --checkpoint artifacts/checkpoints/mappo_final.pt --training-config configs/training_config.yaml --env-config configs/env_config.yaml

# 3) (Optional) Generate EVCS scaffold profiles (does not imply full benchmark)
python experiments/generate_evcs_profiles.py --config configs/evcs_config.yaml --output-dir data/evcs_profiles --seed 42

# 4) (Optional) Generate topology dataset/splits
python experiments/generate_topology_dataset.py --output-dir data/topologies --n-samples 16 --seed 42 --ac-tolerance 0.01
```

Expected key outputs to verify:

- Layer0: `data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv`
- Layer1: `data/oedisi-ieee123-main/profiles/layer1_vpp/layer1_pref.csv`
- Layer2 logs: `artifacts/logs/train_metrics.csv`

---

## 12) Acceptance Criteria to Promote Claims (P/L → I)

| Claim ID | Current Status | Promotion gate to **Implemented (I)** |
|---|---|---|
| C8 (dynamic-topology runtime generalization) | P | End-to-end training/eval executed on topology split files with reported metrics per split (`train/interpolation/extrapolation/extreme`) and reproducible artifact table in `artifacts/results/` |
| C9 (EVCS runtime hooks) | P | EVCS-enabled (`enabled=true`) end-to-end run produces stable Layer0→Layer1→Layer2 artifacts and test pass set including `tests/test_evcs_integration_runtime.py` |
| C10 (EVCS benchmark outcomes) | L | Publish artifact-backed KPI table (obligation fulfillment, dispatch impact, cost/reward deltas) with config snapshot and at least one baseline comparison |
| C11 (non-zero reserve commitment outcomes) | L | Use non-legacy reserve mode (`proxy_ratio` or `duals_if_available`) and produce Layer0 reserve_price > 0 plus Layer1 `R_commit` > 0 in at least part of horizon |

---

## 13) Artifact Provenance & Known Discrepancies

## 13.1 Provenance snapshot (latest local files referenced)

- `configs/env_config.yaml` (exists; local mtime captured)
- `configs/training_config.yaml` (exists; local mtime captured)
- `configs/evcs_config.yaml` (exists; local mtime captured)
- `data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv` (exists; local mtime captured)
- `data/oedisi-ieee123-main/profiles/layer1_vpp/layer1_pref.csv` (exists; local mtime captured)
- `artifacts/logs/train_metrics.csv` (exists; local mtime captured)

> For manuscript release, freeze these with an explicit artifact manifest (file hash + config snapshot + git commit SHA).

## 13.2 Known discrepancy table

| Topic | Historical note | Current artifact-grounded status |
|---|---|---|
| Layer2 update count | `Result.md` references a 10-update snapshot | Current `artifacts/logs/train_metrics.csv` has 2 rows in present workspace snapshot |
| Reserve behavior | Prior narrative discussed reserve-pricing enhancement path | Current default config is `legacy_zero`; current Layer0/Layer1 artifacts show reserve signal/commit at zero |
| EVCS maturity wording | Earlier docs mixed scaffold and full-integration language | Current doc keeps EVCS runtime support as partial unless benchmark artifacts are present |

## 13.3 Artifact manifest (frozen reference)

**Manifest generated from local workspace snapshot**

- **Git HEAD:** `06a52a2699a44bace1346d4c41ae04b71fe7a9c4`

| File | SHA256 |
|---|---|
| `configs/env_config.yaml` | `27cd07fdd803e22c85ae345fdb4d5af85ac2f5753ae5f570719e5025a8deefaf` |
| `configs/training_config.yaml` | `d305311b6af3cbe3efa70be0173121d6fff7da32966736e31ad465d03128feae` |
| `configs/evcs_config.yaml` | `52bbfcf0cb48a8f411b82220f35fb79dd252fe19a4891ae0597b3d999ea84b3b` |
| `data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv` | `6a2085050e845b597c54df302a19a9a0a78e50da3464a7e59f25cd5e0419eb85` |
| `data/oedisi-ieee123-main/profiles/layer1_vpp/layer1_pref.csv` | `6c9a1172b6fa5181236936ea7edceb89076beae391c9ebb4e2b494964ab7ff38` |
| `artifacts/logs/train_metrics.csv` | `6be90be2861d36e6b80c6d41125aeda11ceb430f18de57b9f01305821126db55` |
