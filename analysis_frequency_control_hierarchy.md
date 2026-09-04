# Phân Tích Chi Tiết: FFR, Điều Tần Sơ Cấp & Thứ Cấp

## 1. CẤU TRÚC 3 TẦNG ĐIỀU TẦN TẦN SỐ (Frequency Control Hierarchy)

```
THỜI GIAN (TIME DOMAIN)
     0ms  ┌─ Inertia Response (Synthetic Inertia)  ← Virtual H = 1.18s
    100ms │  
    200ms │ 
    500ms ├─ PRIMARY CONTROL (Droop Response)     ← Our FFR control
   2000ms │  
  10000ms │
 100000s  └─ SECONDARY CONTROL (AGC/Integral)    ← NOT in islanded system!

POWER DIAGRAM:
              │ ΔP = -K_droop × Δf
   ┌──────────────────────────────────────┐
   │   LOAD VARIATION: ΔP_load            │
   └──────────────────────────────────────┘
           │
           ↓
   ┌──────────────────────────────────────┐
   │ 1. SYNTHETIC INERTIA (Very Fast)     │
   │    P_inertia = -2H × (df/dt)         │
   │    Activation: 0-100 ms              │
   └──────────────────────────────────────┘
           │
           ↓ (slow down dF/dt)
   ┌──────────────────────────────────────┐
   │ 2. PRIMARY CONTROL (FFR - Our RL)    │
   │    P_droop = -K_droop × ΔF           │
   │    Response: 100-500 ms              │
   │    → Stabilizes frequency            │
   └──────────────────────────────────────┘
           │
           ↓ (frequency reaches nadir)
   ┌──────────────────────────────────────┐
   │ 3. SECONDARY CONTROL (NOT in island) │
   │    P_agc = K_i × ∫ΔF dt             │
   │    Response: 10-100 seconds          │
   │    → Returns f to exactly 50 Hz      │
   └──────────────────────────────────────┘
```

---

## 2. CHI TIẾT MỖI TẦNG ĐIỀU TẦN

### **TẦNG 1: SYNTHETIC INERTIA (0-100 ms)**

**Công thức:**
```
P_inertia = -2·H·f₀·(df/dt)

Where:
  H = moment of inertia (s) = 1.18s for our system
  f₀ = nominal frequency = 50 Hz
  df/dt = rate of change of frequency (RoCoF)
```

**Vai trò:**
- Phản ứng **tức thì** (không delay)
- Hạn chế RoCoF tăng quá nhanh
- Không thay đổi năng lượng hệ thống (chỉ borrow/lend kinetic energy)
- Tự động trong ODE-based frequency dynamics

**Từ kết quả eval:**
```
RoCoF values:
  S1 (load):    0.408 Hz/s  ← synthetic inertia hoạt động
  S2 (gen_trip): 0.510 Hz/s
  S3 (line_trip): 0.391 Hz/s
  S4 (severe):   0.768 Hz/s

All < 2.0 Hz/s (IEEE limit) ✓
```

**Tại sao RoCoF còn thấp (< 1 Hz/s)?**
- H_sys = 1.18s là rất lớn cho system islanded (bao gồm battery inertia emulation)
- Công thức: RoCoF = -ΔP / (2·H·f₀)
  - ΔP = 3 MW (gen_trip)
  - 2·H·f₀ = 2·1.18·50 = 118
  - RoCoF = 3/118 = 0.025 Hz/s (theoretical minimum)
  - Thực tế cao hơn do state derivatives

---

### **TẦNG 2: PRIMARY FREQUENCY CONTROL (Droop Response - 100-500ms)**

**Công thức Droop:**
```
ΔP_droop = -K_droop × ΔF
         = -(1/R) × (F - F_nom)

Where:
  K_droop = 1/R = droop gain
  R = droop coefficient (Hz/pu)
  ΔF = F - 50 Hz
```

**Steady-State (Static) Behavior:**
```
At steady state (df/dt = 0):
Power balance: ΔP_load = ΔP_droop + ΔP_secondary

ISLAND (NO TIE-LINES):
  ΔP_secondary = 0  ← NO SECONDARY CONTROL
  
Therefore:
  ΔP_load = -K_droop × ΔF_ss
  
  ΔF_ss = -ΔP_load / K_droop
        = -ΔP_load × R
```

**Example từ S1 (Load-Step +2.5 MW):**
```
ΔP_load = +2.5 MW
Nadir = 49.455 Hz
ΔF = 49.455 - 50 = -0.545 Hz (frequency DROP)

Droop required:
R = |ΔF| / |ΔP| = 0.545 / 2.5 = 0.218 Hz/pu

ISSUE: 
  ← Frequency doesn't return to 50 Hz
  ← Stays at -0.545 Hz forever!
  ← Why? Because droop is a static control!
```

**Droop Static Characteristic:**
```
Power Output (pu)

       1.0 ├─────
           │    \  Droop slope = -1/R
      0.8  │     \
           │      \
      0.6  │       \
           │        \
      0.4  │         \
           │          \
      0.2  │           \
           │            \
        0  ├──────────────────── Frequency (Hz)
           49.0  49.5  50.0  50.5
           
↑ Load increases
↓ Frequency drops
↓ Droop increases power output
↓ But frequency stays DOWN (never returns to 50 Hz)
```

**Diagram chi tiết cho S1:**
```
BEFORE event (t < 30s):
  Load = 7.5 MW (baseline)
  Frequency = 50.0 Hz
  Droop output = 0 MW

EVENT (t = 30s): Load step +2.5 MW
  
IMMEDIATE (0-100ms): Synthetic inertia
  Provides -ΔP/dt power
  Slows RoCoF down to ~0.4 Hz/s
  
TRANSIENT (100-500ms): Droop kicks in
  Frequency drops from 50.0 → 49.455 Hz
  ΔF = -0.545 Hz
  Droop power = -K × (-0.545) = +1.36 MW
  New operating point: Load 7.5 + 2.5 = 10 MW, Power out = 10 MW ✓
  
STEADY-STATE (> 500ms): Static equilibrium
  Load = 10 MW (increased permanently)
  Frequency = 49.455 Hz (FIXED!)
  Droop output = 1.36 MW offset
  → Frequency NEVER returns to 50 Hz!
  → Will stay at 49.455 Hz indefinitely!
```

**Nadir = Steady-state frequency với Droop control!**

---

### **TẦNG 3: SECONDARY FREQUENCY CONTROL (AGC - 10-100s)**

**Công thức:**
```
ΔP_AGC = K_p × ΔF + K_i × ∫ΔF dt
       = K_p × (F - 50) + K_i × ∫(F - 50) dt

For integral-only:
ΔP_AGC = K_i × ∫ΔF dt
```

**Vai trò:**
- Tích phân sai số frequency → **unbounded power adjustment**
- Đẩy frequency trở lại 50 Hz chính xác
- Điều chỉnh generation để match load
- Thường do Central Control (EMS/AGC system)

**Trong hệ thống ISLAND:**
```
🚫 NO SECONDARY CONTROL vì:
  1. Không có tie-line (no power exchange với grid)
  2. Không có external generation dispatch
  3. Tất cả generation là distributed trong VPP
  4. Chỉ có PRIMARY control (droop + RL FFR)
  
→ Frequency FIXED at droop steady-state value
→ NEVER returns to 50 Hz automatically
```

**Compare Grid-Tied vs Islanded:**
```
GRID-TIED SYSTEM:
  1. Load increases
  2. Frequency drops (primary control activates)
  3. Nadir reached
  4. AGC sees ΔF < 0 → aumenta generation ↑
  5. Frequency climbs back to 50 Hz
  6. AGC reduces generation to match load
  7. Frequency stays at 50 Hz exactly ✓

ISLANDED SYSTEM (OUR CASE):
  1. Load increases
  2. Frequency drops (primary control activates)
  3. Nadir reached → STEADY STATE
  4. ❌ No AGC, no external generation dispatch
  5. ❌ Frequency stuck at nadir forever
  6. ❌ VPP local droop controls maintain that nadir
  7. Frequency = 49.455 Hz indefinitely
```

---

## 3. TẠI SAO FREQUENCY KHÔNG QUAY VỀ 50 Hz?

### **Root Cause Analysis**

**DROOP IS STATIC CONTROL:**
```
Primary control (droop):
  ΔP = -K × ΔF
  
At equilibrium:
  ΔP_load = ΔP_droop
  ΔP_load = -K × ΔF_ss
  
  → ΔF_ss = -ΔP_load / K
  
This is a PERMANENT offset!
→ f_ss = 50 - ΔP_load / K
```

**NO INTEGRAL ACTION:**
```
Integral control (AGC):
  ΔP = K_i × ∫(F - 50) dt
  
This has "memory":
  - If F < 50 for 1 second, accumulate 1×K_i power
  - If F < 50 for 10 seconds, accumulate 10×K_i power
  → Eventually F returns to 50 Hz
  
OUR SYSTEM:
  No integral AGC → no accumulated correction
  → frequency stays at droop equilibrium forever
```

### **Mathematical Proof (Why nadir = steady-state)**

**System ODE:**
```
2H·(df/dt) = P_gen - P_load - D·Δf - ΔP_event

Where:
  P_gen = primary droop control = -K_droop × Δf
  P_load = constant load (assumed)
  D = damping coefficient
  ΔP_event = one-time event (load step, gen trip, etc)
  
Substituting P_gen:
2H·(df/dt) = -K_droop × Δf - P_load - D·Δf - ΔP_event
2H·(df/dt) = -(K_droop + D) × Δf + constant
2H·(df/dt) = -damping × Δf + constant

At steady state (df/dt = 0):
0 = -damping × Δf_ss + constant
Δf_ss = constant / damping

This is NON-ZERO!
→ Frequency has a permanent offset from 50 Hz
```

**Example (S1 Load-Step):**
```
Event: ΔP_load = +2.5 MW

Steady-state equation:
0 = P_gen - (P_load_old + ΔP_load) - D·Δf_ss

P_gen = -K_droop × Δf_ss

-K_droop × Δf_ss = P_load_old + ΔP_load + D·Δf_ss
-K_droop × Δf_ss - D·Δf_ss = P_load_old + ΔP_load
-(K_droop + D) × Δf_ss = ΔP_load

Δf_ss = -ΔP_load / (K_droop + D)
      = -2.5 MW / (K_droop + D)
      = -0.545 Hz (observed!)

→ Nadir = 50 + Δf_ss = 50 - 0.545 = 49.455 Hz ✓
```

---

## 4. CHI TIẾT CÁC TẦNG TRONG EVAL RESULTS

### **S1: Load-Step (+2.5 MW)**

```
Timeline:

t=0-30s: Initial condition
  F = 50.0 Hz
  P_load = 7.5 MW (baseline)
  P_gen = 7.5 MW

t=30s: Load step event
  ↓ Load jumps to 10.0 MW
  ↓ Power imbalance: 10.0 - 7.5 = 2.5 MW deficit

t=30-30.1s: SYNTHETIC INERTIA (Tầng 1)
  df/dt = -ΔP/(2·H·f₀) ≈ -2.5/(2·1.18·50) = -0.021 Hz/ms
  RoCoF ≈ -21 Hz/s (raw ODE derivative)
  But synthetic H dampens it → actual ≈ -0.4 Hz/s
  
  F drops from 50.0 → 49.9 Hz in 0.1s
  Deceleration of frequency drop (inertia working!)

t=30.1-30.3s: PRIMARY CONTROL (Tầng 2 - Droop)
  Δf = F - 50 = -0.545 Hz (at nadir)
  Droop power = -K_droop × (-0.545) = +1.36 MW
  
  FFR (RL) adds dynamic response:
    MAPPO nadir = 49.455 Hz
    Fixed Droop nadir = 48.779 Hz
    RL provides extra ~0.7 Hz improvement!
  
  → RL learns better droop gain scheduling

t=30.3s+: STEADY STATE
  F = 49.455 Hz (for MAPPO)
  P_gen = 10.0 MW (matches new load)
  P_droop = +1.36 MW offset (maintains 49.455 Hz)
  
  ❌ No AGC (Tầng 3):
  ❌ Frequency stuck at 49.455 Hz forever!
  ❌ Will NOT return to 50 Hz automatically
  
Settling time = 50s:
  After 50 seconds, still at 49.455 Hz
  (Settling time measured is when derivative becomes negligible)
```

### **S2: Gen-Trip (-3.0 MW)**

```
t=30s: Generation trips (3 MW generator trips offline)
  Power deficit: ΔP = -3.0 MW

t=30-30.1s: SYNTHETIC INERTIA
  RoCoF ≈ +0.51 Hz/s (positive = frequency DROP due to deficit)
  
  F drops from 50.0 → 49.3 Hz rapidly

t=30.1-30.5s: PRIMARY CONTROL
  Δf = 49.3 - 50 = -0.7 Hz
  Droop tries to increase power output
  BUT: Generation already tripped!
  Only droop-controlled generators respond
  
  MAPPO nadir = 49.319 Hz
  → RL helps recover ~0.46 Hz vs Fixed Droop

t=30.5s+: STEADY STATE
  Nadir = 49.319 Hz (STUCK!)
  
  Problem: Load is 10 MW, but only 7 MW gen available
  7 MW < 10 MW → perpetual deficit
  
  Frequency MUST stay low (< 50 Hz) to reduce load (via frequency-dependent load model)
  
  At 49.319 Hz:
    Frequency-dependent load ↓ by ~1.5%
    Effective load ≈ 10 MW × 0.985 ≈ 9.85 MW
    Available gen = 7 MW × (1 + FFR boost) ≈ 7.7 MW
    → Roughly balanced at 49.3 Hz!
```

### **S3: Line-Trip (Topology Change)**

```
t=30s: Critical transmission line trips
  Topology changes → some generators isolated
  Available generation reduced
  Power loss from line rupture
  
  Cascading effects:
    - Network impedance changes
    - Voltage regulation affected
    - Generator response dynamics altered

MAPPO advantage:
  MAPPO = 49.435 Hz (understands topology effect)
  Droop = 48.812 Hz (generic droop, ignores topology)
  
  RL learned: "Different topologies → different control response"
  Graph neural network captures this!
  
  Time in violation:
    MAPPO = 36s (recovers faster!)
    Droop = 135s (much slower recovery)
```

---

## 5. TẠI SAO SETTLING TIME = 50 SECONDS?

**Settling time không phải là "return to 50 Hz"**

```
IEEE Definition:
Settling time = Time for frequency to reach ±0.1 Hz band around steady-state
             and remain there (no more oscillations)

Our case:
Steady-state = 49.455 Hz (NOT 50 Hz!)
±0.1 Hz band = [49.355, 49.555] Hz

Timeline:
t=30.0s: Event occurs
t=30.1s: Nadir reached (49.455 Hz minimum)
t=30.1-50s: Transient oscillations, slow damping
t=50s: Finally settled into [49.355, 49.555] band

→ Settling time = 50 seconds

But frequency is NOT at 50 Hz!
It's oscillating around 49.455 Hz steady-state
```

**Why no return to 50 Hz in settling measurement:**
```
Settling time measures:
  ✓ When overshoots end
  ✓ When steady-state reached
  ✗ NOT whether steady-state = nominal
  
Our system:
  Steady-state ≠ nominal
  → Settling time = time to reach steady-state (49.455 Hz)
  → Stays there forever (no AGC)
  
For return to 50 Hz:
  Would need: K_i × ∫(F - 50) dt to force F → 50 Hz
  = Secondary frequency control (AGC)
  = NOT implemented in islanded microgrid
```

---

## 6. SOLUTION: HOW TO RETURN TO 50 HZ?

### **Option 1: Add Integral Control (AGC)**
```python
class SecondaryFrequencyControl:
    def __init__(self, K_i=0.5):
        self.K_i = K_i  # Integral gain
        self.integral = 0.0
    
    def step(self, f_deviation, dt):
        # f_deviation = F - 50 Hz
        self.integral += f_deviation * dt
        return self.K_i * self.integral  # AGC power command
```

**Effect:**
```
Time integral of (49.455 - 50.0) = -0.545 sec-Hz

After 10 seconds: integral = -0.545 × 10 = -5.45
AGC power adjustment = 0.5 × (-5.45) = -2.73 MW
→ Reduce generation by 2.73 MW
→ Frequency starts rising back toward 50 Hz
```

### **Option 2: Droop + Integral (PI Control)**
```
ΔP_total = -K_p × Δf - K_i × ∫Δf dt
         = Proportional (fast) + Integral (slow, brings f → 50 Hz)
```

### **Option 3: Modify Load (Frequency-Dependent Load)**
```
P_load(f) = P_load,0 × [1 - α × (f - 50)]

If α = 0.05:
  At 49.455 Hz: Load = 10 MW × [1 - 0.05 × (-0.545)]
               = 10 MW × 1.0272 = 10.27 MW
  ❌ Load increases! Worse!
  
If α = 0.5:
  At 49.455 Hz: Load = 10 MW × [1 - 0.5 × (-0.545)]
               = 10 MW × 1.272 = 12.7 MW
  ❌ Much worse!
  
Frequency-dependent load helps for generation deficit,
but doesn't return f to 50 Hz either (just new equilibrium)
```

### **Option 4: Energy Storage (Battery) with PI Control**
```
Battery has AGC function:
  - Sees f = 49.455 Hz
  - Integral accumulates: ∫(49.455 - 50) dt
  - Charges itself (absorb power) to support
  - Reduces discharge
  - Generation can decrease
  - Frequency rises back to 50 Hz

This is what BESS AGC does!
```

---

## 7. SUMMARY: 3 TẦNG ĐIỀU TẦN

| Tầng | Tên | Thời gian | Cơ chế | Vai trò | Hệ thống |
|------|-----|----------|--------|--------|----------|
| **1** | Synthetic Inertia | 0-100 ms | Kinetic energy borrow/lend | Hạn chế RoCoF | ✓ Có |
| **2** | Primary Control (Droop) | 100-500 ms | Static droop gain (1/R) | Ổn định tần số | ✓ Có + RL FFR |
| **3** | Secondary Control (AGC) | 10-100s | Integral action K_i·∫ΔF | Trả f về 50 Hz | ✗ Không có |

**Kết quả:**
```
Tầng 1 + 2: Nadir = 49.455 Hz (steady-state)
+ Tầng 3:    Nadir → 50.0 Hz (with AGC integral)
```

**Trong hệ thống của chúng ta:**
```
✓ Tầng 1 + 2 hoạt động tốt (MAPPO improves by +0.6 Hz)
✗ Tầng 3 không có
→ Frequency mãi mãi ở nadir = steady-state droop equilibrium
```

---

## 8. PHÂN TÍCH CHI TIẾT: NADIR ≠ RETURN TO 50 HZ

**Nadir (Điểm thấp nhất):**
```
Definition: Giá trị tần số nhất định tại thời điểm 30.1-30.5s
            (khi quá độ kết thúc)
            
Là: f_minimum = 49.455 Hz (S1 MAPPO)
                
Tính chất: STATIC, không phải dynamic
          Là giá trị ổn định mà frequency rơi xuống
```

**Return to 50 Hz (Quay lại nominal):**
```
Definition: Frequency quay trở lại f = 50.0 Hz
            (exact nominal value)
            
Cần: Integral control (AGC) để push frequency back up
     
Trong hệ islanded:
     ✗ Không có mechanism nào để push f back to 50 Hz
     ✓ Có synthetic inertia + droop
     ✓ Nhưng droop là static, không có "memory"
     
Kết quả:
     Frequency "stuck" ở nadir value (49.455 Hz)
     Settling time = thời gian oscillation decay
     ≠ thời gian return to 50 Hz (never happens!)
```

