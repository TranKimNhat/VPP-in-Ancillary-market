# Evaluation Investigation Report (Smoke Run)

**Date:** 2026-05-07  
**Scope:** Full diagnosis of current Section VI eval behavior after refactor of `src/eval/comparison_runner.py`  
**Data source:** `results/eval_smoke_protocol/*`

---

## 1) Executive summary

### What is working well
- Proposed (**GraphSAGE-Dual-PPO**) beats **No FFR** on key fast-control utility:
  - `IAE`: **129.21 < 140.99**
  - `time_violation`: **117.45 < 126.02**
- Proposed beats **Fixed Droop** on:
  - `IAE`: **129.21 < 134.32**
  - `time_violation`: **117.45 < 126.02**
- Proposed has among best frequency deviation control:
  - `delta_f_max`: **0.515** (best overall among evaluated methods)

### Main gaps found
- **GCNN-PPO has lower overall IAE than Proposed**:
  - `126.25 < 129.21`
- Proposed is **not best on RoCoF**:
  - Proposed `rocof_max = 0.179`, while No FFR / Fixed Droop around `0.125–0.127`
- Nadir is not safe as primary claim metric right now:
  - Proposed nadir lower than several baselines in aggregate
- THD currently unusable for performance claim:
  - `harmonic_invalid_rate = 1.0` for all methods

### Key root-cause conclusion
The “GCNN beats Proposed on overall IAE” is primarily a **scenario composition artifact**: GCNN dominates in **S1 mild**, while Proposed dominates in **S2/S3 disturbance**. Overall average masks this tradeoff.

---

## 2) Run integrity and protocol status

### Output integrity
All protocol outputs were generated:
- `case_metrics.csv`
- `case_trajectories/...`
- `table1_frequency_all.csv`, `table1_frequency_by_scenario.csv`
- `table2_slow_proxy_all.csv`, `table2_slow_proxy_by_scenario.csv`
- `table3_topology_generalization.csv`
- `table4_topology_split_quality.csv`
- `table_thd_diagnostic.csv`
- `topology_split_summary.csv`, `topology_split_stats.json`
- `comparison_gate.json`

### Grid coverage
- Cases: **300 = 5 methods × 20 topologies × 3 scenarios**
- Methods observed:
  - GraphSAGE-Dual-PPO (Proposed)
  - GCNN-PPO
  - Graph-PPO
  - Fixed Droop
  - No FFR

### Gate status
From `comparison_gate.json`:
- **G1 PASS**: Proposed `IAE_all < NoFFR IAE_all`
- **G2 SKIP**: No-Graph Dual-PPO not present
- **G3 SKIP**: No-Graph Dual-PPO not present
- **G4 PASS**: Proposed `time_violation_all <= NoFFR`
- **G5 PASS**: THD excluded since invalid rate too high

---

## 3) Metric diagnosis (what actually drives the outcomes)

## 3.1 Overall means (all topologies × all scenarios)

- **Proposed**: `IAE=129.21`, `delta_f_max=0.515`, `rocof_max=0.179`, `nadir=49.762`, `time_violation=117.45`
- **GCNN-PPO**: `IAE=126.25`, `delta_f_max=0.579`, `rocof_max=0.199`, `nadir=50.069`, `time_violation=138.30`
- **Fixed Droop**: `IAE=134.32`, `delta_f_max=0.519`, `rocof_max=0.127`, `nadir=49.869`, `time_violation=126.02`
- **No FFR**: `IAE=141.00`, `delta_f_max=0.531`, `rocof_max=0.125`, `nadir=49.858`, `time_violation=126.02`

Interpretation:
- Proposed is strong on deviation and violation duration, but not on RoCoF extreme.
- GCNN has lower IAE overall but clearly worse `delta_f_max` and `time_violation`.

## 3.2 Scenario decomposition (critical)

### S1 (mild)
- Proposed `IAE ≈ 94.45`
- GCNN `IAE ≈ 31.37` (very strong advantage)

### S2 (moderate)
- Proposed `IAE ≈ 133.72`
- GCNN `IAE ≈ 162.79` (Proposed better)

### S3 (severe)
- Proposed `IAE ≈ 159.47`
- GCNN `IAE ≈ 184.60` (Proposed better)

**Conclusion:** overall IAE gap is driven by **S1-only behavior**, not by disturbance robustness.

## 3.3 Case-level win rates (Proposed vs GCNN)

Over 60 topology–scenario cases:
- IAE wins: **40/60**
- `delta_f_max` wins: **42/60**
- `rocof_max` wins: **30/60**
- `time_violation` wins: **25/60**
- nadir wins: **0/60**

Important nuance:
- Proposed beats GCNN on most cases for IAE and `delta_f_max`, but loses large-magnitude IAE in specific S1 cases, which skews means.

## 3.4 Where Proposed loses hardest vs GCNN

Top worst IAE gaps all occur in **S1** (mild), with per-case deltas around **+57 to +74** IAE (Proposed minus GCNN). This is the dominant source of overall deficit.

## 3.5 Train/test split behavior

- Proposed:
  - `IAE_train=128.66`, `IAE_test=130.89`, `Gap≈+1.73%`
- GCNN:
  - `IAE_train=126.04`, `IAE_test=126.88`, `Gap≈+0.66%`

Interpretation:
- Both generalize reasonably under current split.
- Proposed has larger OOD degradation than GCNN in this smoke run.

---

## 4) Topology-split quality diagnosis

From `topology_split_stats.json`:
- `n_unique_topologies=20`, `n_train=15`, `n_test=5`
- `mean_d_min≈0.00828`, `min≈0.00820`, `max≈0.00833`

Interpretation:
- Split is valid by edge hash uniqueness, but **distance is very small**.
- This supports claim wording as:
  - **"robustness to local feeder reconfiguration"**
  - not large structural topology transfer.

---

## 5) THD branch status

From `table_thd_diagnostic.csv`:
- `harmonic_invalid_rate = 1.0` for all methods
- THD statistics are NaN

Conclusion:
- THD must remain **diagnostic only**.
- Do not make THD performance claims until harmonic validity is restored.

---

## 6) Root-cause analysis (code-level)

## 6.1 Fast-loop reward bias explains RoCoF behavior
In `src/env/microgrid_env_dual.py`:

```python
r_fast = -2.0 * |delta_f| - 0.2 * |rocof| + 0.5 * control_effect - 0.02 * effort
```

Implication:
- |delta_f| penalty is weighted **10× stronger** than |rocof|.
- Expected policy outcome: prioritize frequency deviation / violation reduction over RoCoF minimization.
- This matches observed results: Proposed strong at `delta_f_max` and `time_violation`, not best at `rocof_max`.

## 6.2 S1 configuration likely over-favors low-intervention policy behavior
Current scenario set in `comparison_runner.py` uses a mild S1 disturbance (`load_step`, `delta_P_mw=0.8`).

Observed effect:
- GCNN gets very low IAE in S1 while losing in S2/S3.
- This can dominate the unweighted overall mean.

## 6.3 Nadir is unstable as a headline metric under mixed event directions
Aggregate nadir can be misleading when event sign/type mix differs (or controllers bias above/below nominal differently). Nadir should be secondary unless sign-conditioned reporting is added.

---

## 7) Risks to paper claims (current)

1. **"Best overall frequency quality"** is not defensible if only IAE mean is shown (GCNN currently lower overall IAE).
2. **RoCoF superiority claim** is not defensible.
3. **THD claim** is not defensible.
4. **Strong OOD transfer claim** is too strong given tiny topology distance.

Safe claims now:
- Proposed improves disturbance execution quality over No FFR / Fixed Droop on key operational metrics (`IAE`, `time_violation`, `delta_f_max`).
- Proposed remains competitive under held-out local reconfigurations.
- Slow-loop results are operational utility proxies, not full monetary profit.

---

## 8) Recommended improvement plan (prioritized)

## Priority A — immediate (high impact, low risk)
1. **Report scenario-stratified results as primary view**
   - Promote `table1_frequency_by_scenario.csv` to main analysis.
   - Avoid single unqualified overall ranking.
2. **Add No-Graph Dual-PPO baseline**
   - Needed to unskip G2/G3 and validate graph-encoder contribution.
3. **Use claim-safe metric hierarchy**
   - Primary: IAE (per scenario), `time_violation`, `delta_f_max`
   - Secondary: RoCoF
   - Diagnostic: THD

## Priority B — method tuning (medium effort)
4. **Retune fast reward weights**
   - Increase RoCoF penalty weight from `0.2` upward (sweep candidates) while monitoring `IAE/time_violation` regression.
   - Objective: improve RoCoF without losing Proposed edge on violation duration.
5. **Rebalance S1 definition**
   - Ensure mild scenario remains informative but not disproportionately easy for low-control policies.
   - Validate with per-scenario variance and win-rate stability.

## Priority C — evaluation robustness (medium-high effort)
6. **Add sign-conditioned frequency diagnostics**
   - Track both under-frequency and over-frequency burden (e.g., signed area components), not nadir alone.
7. **Strengthen topology challenge (if claiming broader OOD)**
   - Increase structural distance between train/test sets if feasible.
   - If not feasible, keep claim wording restricted to local feeder reconfiguration.

## Priority D — THD recovery (separate track)
8. **Fix harmonic validity path**
   - Investigate causes of `harmonic_invalid_rate=1.0`
   - Target `<10%` before enabling THD claims.

---

## 9) Concrete next actions for your team

1. Run **full** (non-smoke) eval with added `No-Graph Dual-PPO` checkpoint.  
2. Keep current protocol outputs and regenerate all tables/figures from full run.  
3. Apply reward sweep for RoCoF tradeoff and compare Pareto frontier:
   - `(IAE, time_violation, rocof_max)`
4. Lock paper wording to:
   - disturbance-quality + local-topology robustness + slow utility proxy
5. Keep THD in appendix as diagnostic only until branch is valid.

---

## 10) Appendix: key file references

- Eval runner: `src/eval/comparison_runner.py`
- Fast/slow reward logic: `src/env/microgrid_env_dual.py`
- Results used in this report: `results/eval_smoke_protocol/`

---

## Final assessment

The current protocol refactor is functioning and already supports defensible claims against No FFR and Fixed Droop. The main technical blocker for stronger novelty defense is not pipeline correctness, but **comparative behavior under S1 and missing No-Graph baseline for gate completeness**. Addressing those two points first will give the highest return for a precise and credible improvement strategy.
