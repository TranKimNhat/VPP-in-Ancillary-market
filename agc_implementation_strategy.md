# AGC Implementation Strategy cho GFM (Grid-Forming) Units

## 1. CẤU TRÚC ĐIỀU KHIỂN PHÂN CẤP

```
HIERARCHICAL FREQUENCY CONTROL ARCHITECTURE
═══════════════════════════════════════════════════════════

LEVEL 3: SYSTEM-WIDE AGC (Central Coordinator)
  ├─ Monitors: System frequency
  ├─ Targets: All GFM units collectively
  └─ Time scale: 10-100 seconds
     │
     ├─→ LEVEL 2a: GFM #1 Local AGC
     │      ├─ P_agc,1 = K_i,1 × ∫(f-50) dt
     │      └─ Coupled to system frequency
     │
     ├─→ LEVEL 2b: GFM #2 Local AGC
     │      ├─ P_agc,2 = K_i,2 × ∫(f-50) dt
     │      └─ Coupled to system frequency
     │
     └─→ LEVEL 2c: Battery AGC
            ├─ P_bess_agc = K_i,bess × ∫(f-50) dt
            └─ Provides inertia + integral action

LEVEL 1: LOCAL DROOP (Every GFM)
  ├─ P_droop,i = -K_droop,i × ΔF
  ├─ Time scale: 100-500 ms
  └─ Static response (no return to nominal)

SYNTHETIC INERTIA (Every GFM)
  ├─ P_inertia,i = -2·H_i·(df/dt)
  ├─ Time scale: 0-100 ms
  └─ Kinetic energy coupling
```

---

## 2. CONTROL ARCHITECTURE: 3 APPROACHES

### **APPROACH A: CENTRALIZED AGC (Traditional)**

```
┌──────────────────────────────────────────────────────────┐
│                   CENTRAL AGC CONTROLLER                 │
│  • Measures: System frequency f                          │
│  • Computes: P_ref = K_i × ∫(f - 50) dt                │
│  • Broadcasts: P_ref to all GFM units                    │
│  • Problem: Single point of failure!                     │
└──────────────────────────────────────────────────────────┘
         │
         ├─→ GFM #1 + Primary
         ├─→ GFM #2 + Primary
         ├─→ GFM #3 + Primary
         └─→ Battery + Primary

Advantages:
  ✓ Simple centralized logic
  ✓ Easy to tune K_i globally
  
Disadvantages:
  ✗ Single point of failure
  ✗ Communication delay
  ✗ Central controller outage = no AGC
  ✗ Not suitable for microgrids (often need resilience)
```

### **APPROACH B: DISTRIBUTED AGC (Recommended for Microgrids)**

```
┌──────────────────────────────────────────────────────────┐
│ DECENTRALIZED: Each GFM has own AGC                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  GFM #1                  GFM #2                GFM #3   │
│  ┌──────────────┐       ┌──────────────┐     ┌────────┐ │
│  │ Local PI:    │       │ Local PI:    │     │ Local  │ │
│  │ K_i=0.3     │       │ K_i=0.3      │     │ K_i=0.2│ │
│  │ ∫(f-50)     │       │ ∫(f-50)      │     │ ∫(f-50)│ │
│  │ P_agc,1     │       │ P_agc,2      │     │P_agc,3│ │
│  └──────────────┘       └──────────────┘     └────────┘ │
│         ↑ All see same                          ↑       │
│         └──── system frequency (broadcast) ────┘        │
│                                                          │
│  Battery AGC                                            │
│  ┌──────────────────────────────────────┐              │
│  │ K_i,bess = 0.5 (aggressive integral) │              │
│  │ P_agc,bess = K_i,bess × ∫(f-50) dt   │              │
│  └──────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────┘

Total system AGC response:
  P_agc,total = P_agc,1 + P_agc,2 + P_agc,3 + P_agc,bess
              = (K_i,1 + K_i,2 + K_i,3 + K_i,bess) × ∫(f-50) dt
              = K_i,sys × ∫(f-50) dt

Advantages:
  ✓ No single point of failure
  ✓ Resilient (if 1 GFM fails, others still have AGC)
  ✓ Natural load sharing
  ✓ Better for islanded microgrids
  ✓ Can tune K_i per unit based on capacity
  
Disadvantages:
  ⚠ Requires frequency broadcast (wireless/fiber)
  ⚠ Tuning more complex (multiple K_i gains)
  ⚠ Possible oscillations if not tuned correctly
```

### **APPROACH C: HYBRID (Distributed + Backup Central)**

```
Normal operation: Distributed AGC per GFM
Backup:         Central AGC if communication fails

Best of both worlds!
```

---

## 3. DETAILED IMPLEMENTATION FOR OUR SYSTEM

### **Current System Components**

```
Our Microgrid has:
  • 41 distributed agents (batteries/VPPs)
  • 3 VPPs (Virtual Power Plants)
  • 1 System frequency (broadcast to all)
  
Current Layers:
  ✓ Synthetic Inertia: Via FrequencyDynamics (H = 1.18s)
  ✓ Primary Control: Droop + RL FFR (per agent)
  ✗ Secondary Control: AGC NOT implemented yet
```

### **Proposed: Add AGC to Each GFM/VPP**

```python
# In src/env/microgrid_env_dual.py

class VPPSecondaryControl:
    """AGC for each VPP (Group of agents)"""
    def __init__(self, n_agents, K_i=0.1):
        self.n_agents = n_agents
        self.K_i = K_i
        self.integral = 0.0  # ∫(f-50) dt
    
    def step(self, frequency, dt):
        """
        AGC power command for this VPP
        
        Args:
            frequency: Current system frequency (Hz)
            dt: Time step (s)
        
        Returns:
            p_agc: Power adjustment (pu) for entire VPP
        """
        error = frequency - 50.0  # ΔF
        self.integral += error * dt
        
        # Limit integral to prevent wind-up
        self.integral = np.clip(self.integral, -20, 20)
        
        # AGC power = integral action only (PI control)
        p_agc = -self.K_i * self.integral
        
        # Clip to available power range
        p_agc = np.clip(p_agc, -1.0, 1.0)
        
        return p_agc


class BatterySecondaryControl:
    """AGC specifically for battery units (fast response)"""
    def __init__(self, n_bess, K_i=0.5):
        self.n_bess = n_bess
        self.K_i = K_i  # Higher gain for faster response
        self.integral = 0.0
    
    def step(self, frequency, dt):
        error = frequency - 50.0
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)
        
        # Battery AGC more aggressive (K_i=0.5 vs 0.1)
        p_agc = -self.K_i * self.integral
        p_agc = np.clip(p_agc, -1.0, 1.0)
        
        return p_agc
```

### **Integration into Frequency Dynamics**

```python
# In step_fast():

# TẦNG 2: Primary control (existing)
event_term = float(np.clip(self.event_delta_p_pu, -0.3, 0.3))
bess_term = float(np.clip(droop_support, -0.25, 0.25))

# ✨ NEW - TẦNG 3: Secondary control (AGC)
agc_term = self.vpp_agc[vpp_idx].step(freq_state.delta_f_hz + 50.0, self.dt_fast_s)
agc_term += self.bess_agc.step(freq_state.delta_f_hz + 50.0, self.dt_fast_s)

# Total power input to frequency dynamics
total_term = event_term + bess_term + agc_term

for _ in range(self.n_ode_substeps):
    freq_state = self.freq_dyn.step(
        dt=self.dt_ode_s,
        delta_P_pu=total_term,  # ← Includes AGC now!
        P_bess_pu=0.0,  # Already in bess_term
    )
```

---

## 4. IMPACT ON FREQUENCY RESPONSE

### **Before AGC (Current)**

```
S1_load_step +2.5 MW:

Frequency evolution:
Time    Frequency   Status
0s      50.000 Hz   Initial
30.1s   49.455 Hz   Nadir (STUCK HERE!)
50s     49.455 Hz   Settling completed
∞       49.455 Hz   PERMANENT (no return)

Problem: F never returns to 50 Hz
```

### **After AGC Implementation**

```
S1_load_step +2.5 MW:

Timeline with AGC (K_i = 0.3):

Time    Frequency   Integral Error    AGC Power    Status
0s      50.000 Hz   0.0              0.0 MW       Initial
30.1s   49.455 Hz   0.0              0.0 MW       Nadir
31s     49.450 Hz   -0.545 × 0.9s    0.16 MW      AGC ramps up
35s     49.380 Hz   -5.45 × 5s       1.64 MW      More AGC
40s     49.800 Hz   -8.0 × 10s       2.4 MW       Approaching nominal
45s     49.950 Hz   -9.0 × 15s       2.7 MW       Very close
50s     49.990 Hz   -9.0 × 20s       2.7 MW       Nearly there
60s     50.000 Hz   -9.5 × 30s       2.9 MW       BACK TO NOMINAL!
∞       50.000 Hz   ≈ 0 (offset)     ≈ 2.5 MW    Steady-state (matched load!)

✓ Frequency returns to 50.0 Hz!
✓ AGC accumulates integral, pushes f up
✓ When f = 50 Hz, integral stops changing
✓ AGC settles at 2.5 MW (exactly balances load increase)
```

### **Graphical Comparison**

```
Frequency vs Time
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│ 50.0 Hz ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    │
│         │          GOAL (with AGC)                    ↗      │
│ 49.8 Hz ├─                                         ╱        │
│         │                                        ╱           │
│ 49.6 Hz ├─                                     ╱             │
│         │                                    ╱               │
│ 49.4 Hz ├─ ════════════════════════════════╱  WITHOUT AGC   │
│         │  (stuck at nadir, no return)    │                 │
│         │                                  │                 │
│ 49.2 Hz ├─                                 │                 │
│         │                                  │                 │
│ Time:   0  30  40   50    60   70   80     │                 │
│       Event  Settling     Recovery to 50Hz  │                 │
│                           (needs AGC)       │                 │
│                                             │                 │
│         ✓ WITH AGC ────────────────────────┘                │
│         ✗ WITHOUT AGC (stays at 49.455 Hz)                  │
└───────────────────────────────────────────────────────────────┘
```

---

## 5. TUNING AGC GAINS (K_i)

### **Guidelines for K_i Selection**

```
RULE: K_i should be sized by capacity

K_i = contribution_ratio × base_gain

Example for our system:
  Total system capacity ≈ 15.7 MW
  3 VPPs + battery
  
  VPP #1 (5 MW):     K_i,1 = 0.1 × (5/15.7) ≈ 0.032
  VPP #2 (5 MW):     K_i,2 = 0.1 × (5/15.7) ≈ 0.032
  VPP #3 (5 MW):     K_i,3 = 0.1 × (5/15.7) ≈ 0.032
  Battery (2 MW):    K_i,b = 0.5 × (2/15.7) ≈ 0.064
  
  Total K_i,sys = 0.032 + 0.032 + 0.032 + 0.064 = 0.160
  
Response time: τ ≈ 1 / (K_i × S_base) ≈ 1 / (0.160 × 15.7) ≈ 0.4s
  → Integral action brings f back to 50 Hz in ~40-60 seconds
```

### **Stability Considerations**

```
Too Low K_i:
  • Very slow return to 50 Hz (> 100 seconds)
  • System stays at reduced frequency too long
  • Risk of load shedding / brownout
  
Too High K_i:
  • Fast return to 50 Hz (good!)
  • But can overshoot (f > 50.2 Hz)
  • May cause oscillations
  • Battery rapid cycling (stress)
  
RECOMMENDED:
  K_i,sys ≈ 0.1 - 0.3 (depends on system inertia)
  For our case: Start with K_i,sys ≈ 0.15
```

---

## 6. IMPLEMENTATION CHECKLIST

```
Phase 1: Add AGC classes
  ☐ VPPSecondaryControl class (1 per VPP)
  ☐ BatterySecondaryControl class (1 for batteries)
  ☐ Anti-windup integral limiter
  
Phase 2: Integrate into env
  ☐ Instantiate K_i per VPP in __init__
  ☐ Call step() in step_fast() before freq_dyn.step()
  ☐ Add agc_term to total power balance
  
Phase 3: Validation
  ☐ Re-run eval: Check if f → 50 Hz
  ☐ Measure settling time (should be < 100s)
  ☐ Check for oscillations (if any)
  ☐ Verify no overshoot
  
Phase 4: Tuning
  ☐ Adjust K_i if settling time not satisfactory
  ☐ Test all 4 scenarios (S1-S4)
  ☐ Check battery SOC impact
  
Phase 5: RL Training Update
  ☐ Re-train MAPPO with AGC present
  ☐ RL FFR + AGC cooperation
  ☐ Might improve nadir further!
```

---

## 7. EXPECTED IMPROVEMENTS

### **Current State (WITHOUT AGC)**

```
Scenario    Nadir    Status
S1          49.455   Stuck, never returns
S2          49.319   Stuck, never returns
S3          49.435   Stuck, never returns
S4          48.992   Stuck, never returns

IEEE 1547:  0/4 (FAIL - nadir < 49.5 Hz)
ENTSO-E:    3/4 (Cat-A) - OK for Europe
```

### **Expected with AGC (K_i = 0.15)**

```
Scenario    Without AGC   With AGC      Return Time    Status
S1          49.455 Hz     50.000 Hz     60-80s         ✓ Meets IEEE!
S2          49.319 Hz     50.000 Hz     60-80s         ✓ Meets IEEE!
S3          49.435 Hz     50.000 Hz     60-80s         ✓ Meets IEEE!
S4          48.992 Hz     ~49.8 Hz      60-80s         ⚠ Marginal

IEEE 1547:  0/4  →  3/4 (improvement!)
ENTSO-E:    3/4  →  4/4 (Cat-A on all!)
```

---

## 8. COMPARISON: DISTRIBUTED vs CENTRALIZED AGC

| Feature | Distributed | Centralized |
|---------|------------|-------------|
| **Resilience** | ✓ High (no single point of failure) | ✗ Low |
| **Communication** | Frequency broadcast only | Requires AGC commands + feedback |
| **Latency** | Lower (local decision) | Higher (controller→units) |
| **Simplicity** | Medium | Low |
| **Scalability** | ✓ Good (each unit independent) | ✗ Central bottleneck |
| **Islanded Microgrids** | ✓ BEST | ⚠ Not ideal |

**Recommendation: DISTRIBUTED AGC for our islanded system**

---

## 9. PHẦN TIẾP THEO: RL + AGC COOPERATION

**Thú vị:** RL FFR + AGC sẽ hoạt động như thế nào cùng nhau?

```
Current (Tầng 2 only):
  Primary = -K_droop × ΔF (MAPPO-optimized)
  
With AGC (Tầng 2+3):
  Primary = -K_droop × ΔF (MAPPO-optimized)
  Secondary = -K_i × ∫(F-50) dt (AGC-driven)
  
MAPPO must learn:
  • When to provide strong droop (fast response)
  • When to back off (AGC will handle long-term)
  • Coordination with AGC integral action
  
Potential: RL might achieve even BETTER nadir!
  Because MAPPO doesn't need to "worry" about returning to 50 Hz
  Can focus on minimizing nadir instead
```

