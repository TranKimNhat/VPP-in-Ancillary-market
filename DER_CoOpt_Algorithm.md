# DER Siting & Sizing Co-Optimization — Reviewer-Facing Method Justification
# IEEE 123-Bus Islanded Microgrid (Repo-Consistent)

**Version:** 3.1 (code/artifact aligned)  
**Date:** 2026-05-12  
**Primary code:** `der_placement_coop.py`  
**DER optimization result artifact:** `artifacts/placement/official_placement_v3.json`

---

## 1) Scope and legitimacy boundary

Tài liệu này dùng để giải trình tính chính danh cho reviewer với ranh giới rõ ràng:

1. **Bài toán DER co-optimization** (vị trí + công suất DER) được định nghĩa và giải bởi `der_placement_coop.py`.
2. **`official_placement_v3.json`** là kết quả tối ưu cho các DER (EVCS/DPV/Wind) theo bài toán trên.
3. **Vị trí và số lượng GFM** thuộc **bài toán reconfiguration/operational stabilization**, không phải biến quyết định của bài toán DER co-optimization.

---

## 2) Mathematical formulation implemented in `der_placement_coop.py`

### 2.1 Decision vector (DER only)

Particle có **68 genes**:
- **27 location genes**
  - EVCS: 9 vị trí (Z1:3, Z2:3, Z4:3)
  - DPV: 14 vị trí (Z1:4, Z2:6, Z4:4)
  - Wind: 4 vị trí (Zone 3)
- **41 sizing genes**
  - EVCS PV: 9
  - EVCS BESS: 9
  - EVCS V2G: 9
  - DPV: 14

Code anchors: `N_LOC=27`, `N_SIZ=41`, `N_GENES=68`, `decode_particle(...)`.

### 2.2 Objectives

Ba mục tiêu tối thiểu hóa:
- \(f_1\): normalized active power loss
- \(f_2\): normalized VDI
- \(f_3\): normalized zonal LMP divergence

Composite objective:
\[
F = w_{loss} f_1 + w_{volt} f_2 + w_{lmp} f_3 + \mathcal{P}
\]

Weights trong code:
- `w_loss = 0.40`
- `w_volt = 0.40`
- `w_lmp = 0.20`

### 2.3 Constraint-to-penalty mapping

Trong `evaluate(...)`:
- Voltage envelope [0.95, 1.05] p.u. → `pen_volt`
- Zone EVCS coverage (Z1,Z2,Z4) → `pen_zone`
- Capacity adequacy (`coverage_ratio * peak_load`) → `pen_cap`
- Zone net injection bound → `pen_zone_net`
- V2G cap by zone → `pen_v2g`
- Flexibility floor (`flex_min_mw`) → `pen_flex`

Penalty weights:
- `lambda_volt=50`, `lambda_zone=100`, `lambda_cap=10`,
- `lambda_zone_net=20`, `lambda_v2g=100`, `lambda_flex=20`.

---

## 3) Engineering legitimacy of fixed assumptions

### 3.1 Fixed DER counts are structural

Bài toán cố định số lượng: 9 EVCS, 14 DPV, 4 Wind để giữ cấu trúc VPP và fairness khi so sánh.

### 3.2 Wind sizing fixed at 3 MW/turbine

`wind_mw=3.0` là giả định theo turbine class (IEC-grade selection), không thả nổi cho optimizer.

### 3.3 GFM is outside DER optimization variable set

Trong phạm vi `der_placement_coop.py`, bài toán tối ưu DER **không tối ưu vị trí/số lượng GFM**.  
Do đó, mọi GFM placement/capacity dùng trong runtime được xem là **đầu vào từ bài toán reconfiguration/stability design**.

---

## 4) Zone/candidate legitimacy

- Zone map lấy từ dữ liệu `bus_zone_map.csv` qua `ZoneConfig.from_bus_zone_map(...)`.
- Candidate buses có lọc `FORBIDDEN` để tránh bus không hợp lệ vận hành.
- Tách candidate theo loại tài sản (`evcs_candidates`, `dpv_candidates`, `wind_candidates`) để đảm bảo feasibility.

---

## 5) Optimization engine legitimacy

PSO params (constriction-style):
- `chi = 0.7298`
- `c1 = c2 = 1.4962`

Multi-seed protocol:
- seeds `[42, 123, 777]`
- lưu `convergence_seed*.csv`
- chọn nghiệm tốt nhất theo composite fitness, đồng thời báo cáo `multi_seed_summary`.

---

## 6) Official DER optimization result (`official_placement_v3.json`)

### 6.1 Metadata

- `version`: 3.1
- `algorithm`: Multi-objective Discrete-Continuous PSO (co-optimization)
- `n_agents`: 41
- `action_flat_dim`: 73
- `particle_genes`: 68

### 6.2 EVCS optimal placement/sizing

| EVCS | Zone | VPP | Bus | PV (MW) | BESS (MW/MWh) | V2G (MW) |
|---|---:|---|---:|---:|---:|---:|
| E1 | 1 | VPP_1 | 34 | 0.200 | 0.325 / 0.650 | 0.100 |
| E2 | 1 | VPP_1 | 1  | 0.050 | 0.325 / 0.650 | 0.100 |
| E3 | 1 | VPP_1 | 2  | 0.050 | 0.325 / 0.650 | 0.100 |
| E4 | 2 | VPP_2 | 49 | 0.200 | 0.275 / 0.550 | 0.100 |
| E5 | 2 | VPP_2 | 50 | 0.200 | 0.275 / 0.550 | 0.100 |
| E6 | 2 | VPP_2 | 48 | 0.200 | 0.275 / 0.550 | 0.100 |
| E7 | 4 | VPP_3 | 52 | 0.200 | 0.225 / 0.450 | 0.075 |
| E8 | 4 | VPP_3 | 53 | 0.158 | 0.225 / 0.450 | 0.075 |
| E9 | 4 | VPP_3 | 66 | 0.200 | 0.225 / 0.450 | 0.075 |

### 6.3 DPV + Wind optimal placement

DPV buses:
- Zone 1: 6, 17, 3, 4
- Zone 2: 43, 27, 47, 30, 18, 45
- Zone 4: 54, 96, 55, 56
- DPV mỗi unit: 0.275 MW

Wind buses:
- 67, 105, 98, 101
- 3.0 MW mỗi turbine

### 6.4 Quantitative metrics

- `best_fitness`: 0.2559572830
- `p_loss_mw`: 0.0545029197
- `p_loss_reduction_pct`: 51.99%
- `vdi`: 2.2579e-05
- `vdi_improvement_pct`: 85.56%
- `v_min/v_max`: 0.9937 / 1.0000
- `v_violations`: 0
- `zone_lmp`: z1=5.0328, z2=5.2271, z4=5.4103
- `zone_net_mw`: z1=-0.4, z2=-2.7675, z3=10.11, z4=-5.0
- `load_scale_mw`: 15.705

---

## 7) GFM note for reviewer (reconfiguration responsibility)

Nếu trong runtime artifact/env xuất hiện GFM bổ sung (ví dụ G4/G5/G6), các phần tử đó phải được đọc là:
- thành phần của **bài toán reconfiguration / post-contingency stabilization**,
- không phải kết quả của DER siting/sizing PSO.

Vì vậy, khi đánh giá tính đúng đắn của `der_placement_coop.py`, reviewer nên tập trung vào:
- biến quyết định DER,
- objective/constraint mapping,
- convergence và metrics của DER placement.

---

## 8) Reproducibility artifacts

Pipeline xuất ra:
- `official_placement_v3.json` (DER optimization result)
- `convergence_seed*.csv`
- `baseline_comparison.csv`
- `bustozone.csv`, `bustoVPP.csv`

Gate checks cuối script đảm bảo tính hợp lệ cấu hình DER trước khi downstream train/eval.

---

## 9) Conclusion

Hệ thống IEEE 123-bus của nhóm có tính chính danh rõ ràng theo phân tách bài toán:
1. **DER co-optimization**: do `der_placement_coop.py` giải và kết quả thể hiện ở `official_placement_v3.json`.
2. **GFM placement/count**: thuộc bài toán reconfiguration để đảm bảo ổn định vận hành sau tái cấu hình.

Cách phân tách này giúp reviewer kiểm tra đúng tầng mô hình, tránh trộn vai trò optimization DER với operational reconfiguration.