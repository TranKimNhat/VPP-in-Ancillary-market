# Result Report

## 1) Mục tiêu đã thực hiện
Đồng bộ codebase theo `Project_Description.md` mới, tập trung vào:
- Static zoning (không hỗ trợ dynamic runtime).
- VPP static/structured, có non-VPP buses.
- Layer1 per-VPP dispatch + legacy compatibility.
- Làm rõ contract Layer2 safety theo implementation hiện tại.

---

## 2) Kết quả chính đã hoàn thành

### A. Static zoning guardrails + mapping contracts
- Enforce `zoning_mode: static` tại runtime:
  - `experiments/train_mappo.py`: reject `zoning_mode != static` bằng `NotImplementedError`.
  - `src/environment/grid_env.py`: guardrail tương tự trong `GridEnvironment`.
- Chuẩn hóa precedence `vpp_mode`:
  - `env_config` là source chính.
  - `training_config.layer1.vpp_mode` có thể override (kèm warning khi conflict).
- Chuẩn hóa contract mapping CSV canonical:
  - `bus_to_zone_csv`, `vpp_to_zone_csv`, `bus_to_vpp_csv`, `der_to_vpp_csv`.
- Thêm warning rõ ràng khi rơi vào legacy fallback mapping (tránh silent drift).

### B. VPP formation utility + artifacts
- Thêm module mới: `src/layer0_dso/vpp_formation.py`.
- Sinh tự động artifacts:
  - `bus_to_vpp.csv`
  - `vpp_to_zone.csv`
  - `vpp_summary.csv`
- Có CLI/helper để chạy trước training/bootstrap.
- Rule-set deterministic, zone-contained, có non-VPP buses.

### C. Layer1/Environment alignment
- Layer1 dùng thực tế `mapping_bus_to_vpp_csv` để tính metadata `vpp_bus_count`.
- Output per-VPP vẫn giữ tương thích legacy aggregate output.
- Environment info bổ sung metadata phục vụ phân tích:
  - `zoning_mode`, `mapping_scope`, `legacy_mapping_fallback`, `safety_mode`.

### D. Layer2 framing alignment (minimal, không over-scope)
- `SafetyResult` bổ sung field `safety_mode="clip_project"` để phản ánh đúng semantics đang chạy.

### E. Tests & verification
- Thêm test mới: `tests/test_vpp_formation.py`.
- Cập nhật regression tests:
  - `tests/test_environment.py`
  - `tests/test_layer0_layer1_io.py`
  - `tests/test_training_smoke.py`
- Thêm test guardrail non-static zoning fail rõ ràng.
- Cleanup warning footprint test path: truyền canonical mappings để loại warning legacy fallback.

---

## 3) Kết quả test hiện có

### Targeted suite
Đã pass:
- `tests/test_vpp_formation.py`
- `tests/test_layer0_layer1_io.py`
- `tests/test_environment.py`
- `tests/test_training_smoke.py`

Kết quả cuối cùng cho nhóm regression chính:
- **8 passed, 3 warnings**
- 3 warnings còn lại là **DeprecationWarning từ dependency (SWIG/pandapower stack)**, không phải warning logic nghiệp vụ của project.

---

## 4) Kết quả tính toán hiện có theo Layer

Số liệu được tổng hợp từ artifacts hiện tại trong repo:
- Layer0: `data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_zone_prices.csv`, `layer0_diagnostics.csv`
- Layer1: `data/oedisi-ieee123-main/profiles/layer1_vpp/layer1_pref.csv`
- Layer2: `artifacts/logs/train_metrics.csv`

### Layer 0 (DSO + Zonal Pricing)
- Tổng số dòng tín hiệu zone: **1152**
- Tập day đại diện: **offpeak, median, peak**
- Zone IDs: **1, 2, 3, 4**
- `energy_price`:
  - min: **0.0541**
  - max: **1.1037**
  - mean: **0.3061**
- AC quality gate:
  - `ac_valid` toàn bộ: **True**
  - `ac_converged` toàn bộ: **True**
  - `socp_ac_gap_max` lớn nhất: **0.00482**

### Layer 1 (DRO Bidding)
- Tổng số dòng output: **96** (96 bước thời gian)
- `solver_status`: **optimal**
- Tổng `P_ref`: **0.0**
- Tổng `R_commit`: **0.0**

### Layer 2 (GAT-MAPPO)
- Nguồn log: `artifacts/logs/train_metrics.csv`
- Số update ghi nhận: **10**
- Snapshot update cuối:
  - `reward_mean`: **-3.8335**
  - `tracking_error`: **2.4954**
  - `voltage_violation`: **0.05818**
  - `curtailment_ratio`: **0.0**

---

## 5) Commit đã tạo
- **Commit SHA**: `23a14d2`
- **Message**: `Align static zoning runtime with VPP artifact generation and stronger contracts.`

Các file chính trong commit:
- `configs/env_config.yaml`
- `configs/experiment_config.yaml`
- `configs/training_config.yaml`
- `experiments/train_mappo.py`
- `src/environment/grid_env.py`
- `src/environment/vpp_mapping.py`
- `src/layer0_dso/vpp_formation.py` (new)
- `src/layer1_vpp/layer1_vpp.py`
- `src/layer2_control/safety_layer.py`
- `tests/test_environment.py`
- `tests/test_layer0_layer1_io.py`
- `tests/test_training_smoke.py`
- `tests/test_vpp_formation.py` (new)

---

## 6) Đánh giá mức độ khớp với mô tả mới

Đạt được các tiêu chí trọng yếu:
- Static zoning đã explicit + enforced.
- VPP formation artifacts được tạo tự động bởi utility chính thức.
- Layer1 dual mode (per-VPP + legacy) giữ tương thích.
- Env/training smoke pass ở cả legacy/vpp paths đã test.
- Claims và implementation nhất quán hơn ở các điểm chính.

---

## 7) Tồn tại/khuyến nghị tiếp theo
- Repo vẫn còn nhiều thay đổi ngoài phạm vi commit này (data/artifacts/deletions cũ), cần dọn riêng trước khi release/push chính thức.
- Có thể thêm filter warning dependency ở `pytest` để log gọn hơn (không che warning nghiệp vụ).
- Nếu cần mở rộng theo paper narrative, có thể tăng chiều sâu Layer2 (critic scope/feature contract) ở một đợt riêng.
