# Hyperparameter Optimization for Frequency Control

## Overview

This document describes the optimization methodology for tuning frequency control parameters in the 100% renewable islanded microgrid. The control hierarchy consists of:

1. **Primary Control (Droop)**: Grid-forming (GFM) units with droop coefficient R
2. **Secondary Control (AGC)**: PI controller with gains Kp, Ki
3. **Fast Frequency Response (FFR)**: VPP DERs (BESS, V2G) with droop-based gain k_ffr

## Optimization Framework

### Algorithm: Differential Evolution (DE)

We use `scipy.optimize.differential_evolution` for global optimization due to:
- Non-convex objective function with multiple local minima
- Multi-dimensional parameter space (4 parameters)
- Bounded constraints on each parameter

**Configuration:**
```python
bounds = [
    (0.02, 0.10),  # R: Primary droop
    (0.10, 0.80),  # k_ffr: FFR gain
    (0.10, 0.80),  # Kp: AGC proportional
    (0.05, 0.40),  # Ki: AGC integral
]
differential_evolution(objective, bounds, maxiter=50, seed=42, polish=True)
```

The `polish=True` option runs L-BFGS-B local optimization after DE converges to refine the solution.

### Objective Function

Multi-objective cost function balancing:

```python
cost = (
    IAE * 1.0 +              # Integral Absolute Error (area under deviation)
    settling_time * 0.3 +     # Time to reach 49.95 Hz
    oscillation * 3.0 +       # Frequency range after t=5s
    overshoot * 5.0 +         # Penalty for exceeding 50.05 Hz
    nadir_penalty +           # Heavy penalty if nadir < 49.5 Hz
    rocof_penalty +           # Penalty if RoCoF > 2 Hz/s
    (50.0 - nadir) * 2.0      # Reward for better nadir
)
```

**Rationale for weights:**
| Component | Weight | Justification |
|-----------|--------|---------------|
| IAE | 1.0 | Baseline metric for overall deviation |
| Settling time | 0.3 | Secondary priority after stability |
| Oscillation | 3.0 | Higher weight to ensure smooth response |
| Overshoot | 5.0 | Prevent over-correction by AGC |
| Nadir penalty | 100× | Critical constraint (must stay > 49.5 Hz) |
| RoCoF penalty | 5× | Protection relay constraint |

### Parameter Bounds

| Parameter | Min | Max | Physical Meaning |
|-----------|-----|-----|------------------|
| R (droop) | 0.02 | 0.10 | 2-10% droop (industry standard) |
| k_ffr | 0.10 | 0.80 | FFR gain relative to delta_f |
| Kp | 0.10 | 0.80 | AGC proportional response |
| Ki | 0.05 | 0.40 | AGC integral action rate |

## Optimization Results

### Test Scenario
- **Disturbance**: 15% generation trip (2.36 MW) at Bus 97 (wind farm)
- **System**: IEEE 123-bus modified, S_BASE = 15.7 MW
- **Inertia**: H_sys = 1.185 s (low inertia islanded microgrid)

### Optimal Parameters

| Parameter | Baseline | Optimized | Change |
|-----------|----------|-----------|--------|
| R | 0.05 | **0.0204** | -59% (more aggressive droop) |
| Kp | 0.50 | **0.7774** | +55% |
| Ki | 0.30 | **0.3979** | +33% |

### Performance Comparison (15% Gen Trip)

| Metric | Droop Only | Proposed (Droop+AGC+per-DER FFR) | Improvement |
|--------|------------|----------------------------------|-------------|
| Nadir | 49.659 Hz | **49.661 Hz** | +2.8 mHz |
| Steady-state | 49.845 Hz | **49.985 Hz** | +140 mHz |
| Range (t>5s) | - | 0.055 Hz | Stable |
| Max RoCoF | 3.165 Hz/s | 3.165 Hz/s | - |
| Recovery time | Never | **~12s** | Full recovery |

### Per-DER FFR Contribution

| DER Type | Units | Total Capacity | Time Constant | Peak FFR |
|----------|-------|----------------|---------------|----------|
| BESS | 9 | 2.475 MW | 100 ms | 0.071 pu |
| V2G | 9 | 0.825 MW | 300 ms | 0.009 pu |
| **Total** | **18** | **3.3 MW** | - | **0.08 pu** |

### Stress Tests

| Disturbance | Nadir | Steady-state | Status |
|-------------|-------|--------------|--------|
| 10% gen trip (1.57 MW) | 49.78 Hz | 49.985 Hz | Stable |
| 15% gen trip (2.36 MW) | 49.66 Hz | 49.985 Hz | Stable |
| 20% gen trip (3.14 MW) | 49.57 Hz | 49.948 Hz | Stable (p_ref saturated) |
| 30% gen trip (4.71 MW) | 49.35 Hz | - | Marginal (< 49.5 Hz) |

## Per-DER Droop Control for FFR

### DER Configuration (from official_placement_v3.json)

The VPP consists of 9 EVCS units, each with co-located BESS and V2G:

| Zone | EVCS Units | BESS (MW) | V2G (MW) | k_droop_BESS | k_droop_V2G |
|------|------------|-----------|----------|--------------|-------------|
| 1 | E1, E2, E3 | 0.325 each | 0.10 each | 1.14 MW/Hz | 0.20 MW/Hz |
| 2 | E4, E5, E6 | 0.275 each | 0.10 each | 0.96 MW/Hz | 0.20 MW/Hz |
| 4 | E7, E8, E9 | 0.225 each | 0.075 each | 0.79 MW/Hz | 0.15 MW/Hz |
| **Total** | **9** | **2.475 MW** | **0.825 MW** | - | - |

### Per-DER Droop Formula

Each DER provides FFR proportional to frequency deviation:

```python
# Per-DER droop control with first-order lag
P_ffr_i = -k_droop_i * delta_f / S_BASE  # Target power (pu)

# First-order lag filter for realistic response time
alpha = dt / (T_der + dt)
P_actual_i = (1 - alpha) * P_prev_i + alpha * P_ffr_i
```

**Droop coefficients:**
- BESS: `k_droop_i = 3.5 × P_rated_i` (MW/Hz) - aggressive response
- V2G: `k_droop_i = 2.0 × P_rated_i` (MW/Hz) - moderate response

**Time constants:**
- BESS: T = 100 ms (inverter-limited, fastest)
- V2G: T = 300 ms (EV battery management system delay)

### Aggregate FFR Capacity

At delta_f = -0.15 Hz (15% gen trip):
- BESS aggregate: 0.071 pu peak (9 units)
- V2G aggregate: 0.009 pu peak (9 units)
- Total FFR: ~0.08 pu = 1.26 MW

## Control Coordination

### Temporal Hierarchy

```
t=0s    : Event occurs (gen trip at Bus 97)
t=0.1s  : FFR activates when |delta_f| > 0.15 Hz
t=0.2s  : FFR ramp-up complete (per-DER with T_bess=100ms, T_v2g=300ms)
t=1-3s  : FFR at full power (droop-based, proportional to delta_f)
t=2s    : AGC activates (PI controller)
t=3-5s  : FFR ramps down, AGC takes over
t>5s    : AGC restores frequency to nominal (~49.985 Hz)
```

### Anti-Coupling Mechanism

To prevent oscillation during FFR-AGC handoff:
- AGC integral is **frozen** while FFR is active
- AGC proportional term continues to compute target
- Smooth handoff when FFR ramps down

```python
# In freq_dynamics.py
if self._agc_armed and not ffr_active:
    self._agc_integral += delta_f_for_agc * dt_val  # Only integrate when FFR inactive
```

## Sensitivity Analysis

### Parameter Sensitivity

| Parameter | Sensitivity | Effect of Increase |
|-----------|-------------|-------------------|
| R (droop) | High | More aggressive primary response, risk of oscillation |
| k_ffr | Medium | Better nadir support, risk of overshoot |
| Kp | Medium | Faster AGC response, risk of oscillation |
| Ki | High | Faster steady-state recovery, risk of overshoot |

### Robustness to Disturbance Size

| Disturbance | Nadir | Stability |
|-------------|-------|-----------|
| 10% gen trip | 49.78 Hz | Stable |
| 15% gen trip | 49.67 Hz | Stable |
| 20% gen trip | 49.57 Hz | Stable |
| 30% gen trip | 49.35 Hz | Marginal (< 49.5 Hz threshold) |

## Implementation Notes

### Code Location
- Optimization script: `scripts/tune_all_control.py`
- Frequency dynamics: `src/env/freq_dynamics.py`
- Event injection: `src/env/events.py`
- Per-DER droop (RL env): `src/env/microgrid_env_dual.py`
- Frequency plot: `scripts/plot_freq_comparison.py`
- DER placement: `artifacts/placement/official_placement_v3.json`

### Reproducibility
```bash
python scripts/tune_all_control.py  # Run optimization
python scripts/plot_freq_comparison.py  # Visualize results
```

### Dependencies
- scipy >= 1.10.0 (differential_evolution)
- numpy >= 1.24.0

## References

1. O'Sullivan et al. (2014) - Largest infeed loss as design contingency
2. Seneviratne et al. (2016) - Frequency response with RES penetration
3. Knap et al. (2016) - ESS sizing for inertial response
4. Javadi et al. (2021) - RoCoF and nadir constraints
5. ENTSO-E guidelines - aFRR activation delay (30s for interconnected, reduced for islanded)
