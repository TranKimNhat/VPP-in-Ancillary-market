# Plan sửa lỗi sau các lần code review

**Dự án**: VPP in Ancillary Market for 100% Renewable Islanded Microgrid
**Ngày cập nhật**: 2026-04-22 (revision sau review patch `floating-snuggling-giraffe.md`)
**Scope**: Fix các vấn đề đã được **verify** qua 3 lần review (Layer architecture + Harmonic + L0 strict/elastic metadata)
**Dành cho**: Vibe coding, sau đó review lại bởi Codex

---

## CHANGELOG

**Rev 4 (2026-04-22)** — tiến độ sau khi user triển khai batch §10 + §Task 1.3 + §14:
- Mark DONE: **Task 10.1 → 10.6**, **Task 10.11**, **Task 1.3**.
- §14 acceptance checklist: 4/5 deliverables tick ✅ (pending chỉ còn mục ablation JSON cần chạy training thật).
- §14.2 note: `scripts/compare_methods.py` đã đổi sang **Welch t-distribution CDF** (không còn normal approximation) — fix P2 statistical issue đã flag.
- Verify: `pytest tests/test_l0_reconfig_metadata.py` 3 passed; `compare_methods.py` chạy OK trên toy output.
- **Progress**: Đợt 1 + Đợt 2 đóng hẳn; §Task 1.3 đóng; §14 infrastructure sẵn sàng. Tiếp theo: §Task 1.1 harmonic hoặc chạy multi-seed training thật.

**Rev 3 (2026-04-22)** — chỉnh theo **góc research impact** (user feedback sau rev 2):
- Reorder §11: §10.1–10.5 (L0 pipeline live) đẩy lên **trước tuyệt đối** mọi task harmonic lớn.
- Task 1.2 (reconfig consolidation): soften tiêu chí từ "alpha giống 100%" → **tương đương** (feasible cùng status, objective gap ≤ tol, switch-Hamming distance nhỏ).
- Task 1.3 (seeding): thêm **flag bật/tắt deterministic** với tradeoff rõ (reproducibility vs GPU throughput).
- Thêm §10.11: **elastic_used ratio** như KPI bắt buộc với ngưỡng cảnh báo 5–10%.
- Thêm §14: **Research Protocol Gate** — mọi thay đổi thuật toán chạy ≥ 3–5 seeds, báo cáo mean±std.

**Rev 2 (2026-04-22)** — sau khi user triển khai patch `floating-snuggling-giraffe.md` và request re-review:
- Thêm section **"DONE from floating-snuggling-giraffe.md"** (§9) — đánh dấu các task đã complete + test pass.
- Thêm section **"Follow-up từ re-review patch L0 metadata"** (§10) — 10 findings mới (1 critical, 2 correctness, 7 hygiene).
- Giữ nguyên các task cũ (§2–§4) vì chúng thuộc scope khác (harmonic + layer architecture, chưa làm).

**Rev 1 (2026-04-22)** — plan ban đầu sau 2 review (layer architecture + harmonic).

---

## 0. Nguyên tắc thực thi (đọc trước khi bắt đầu)

Plan này tuân thủ `CLAUDE.md` của repo:
- **Surgical changes**: mỗi task chỉ sửa đúng phạm vi đã nêu, không refactor code kế bên.
- **Goal-driven**: mỗi task có **success criteria** dạng test hoặc assertion cụ thể.
- **Simplicity first**: không thêm abstraction mới nếu không được yêu cầu.
- **Test trước / sau**: với fix bug → viết test tái hiện trước, sau đó fix để test pass.

Mỗi task được đánh dấu priority:
- **P1** = blocker khoa học/correctness, phải sửa trước khi chạy experiment lớn.
- **P2** = ảnh hưởng độ chính xác/reproducibility, nên sửa trước khi viết paper.
- **P3** = code hygiene, refactor, nice-to-have.

---

## 1. RETRACTIONS — các claim đã bị RÚT LẠI (Codex không cần flag lại)

Ở lần review trước mình (Claude) đã nêu một số "bug" mà **sau khi verify qua Holmes-Lipo textbook, MDPI Energies paper về skin effect, White Rose thesis, IEEE TPE 2018, và các nguồn khác** thì phát hiện là **false positive**. Liệt kê đây để Codex khỏi raise lại:

| Claim ban đầu | Trạng thái | Lý do rút lại |
|---|---|---|
| "Bessel factor phải là `4·Vdc/(mπ)`, code dùng `2·Vdc/(mπ)` là sai" | **RETRACTED** | Code đúng cho **half-bridge convention** (Vdc = total DC bus). Factor 4 chỉ áp dụng cho full H-bridge. 3-phase grid-tied PV inverter với `Vdc_v=800` là half-bridge per-phase → factor 2 là **chuẩn Holmes-Lipo**. |
| "`V_h[slack] = 0` phải được áp đặt, thiếu boundary condition" | **RETRACTED** | Project là **islanded microgrid** → slack bus chính là **GFM inverter** và là nguồn sóng hài, **không phải** harmonic short. Code hiện tại đúng hơn cho islanded operation. |
| "DistFlow voltage drop sai dấu" (Review 1) | **RETRACTED** | Đọc lại `reconfiguration.py`: `drop = -2(rP+xQ) + (r²+x²)I²` đúng với **Baran-Wu**. |
| "DRO worst-case margin đảo dấu" (Review 1) | **RETRACTED** | `c[p_idx] = -μ_e + ρ·MAD` đúng với **góc seller** của inverter (discharge = bán điện). |
| "GAT mask sau softmax" (Review 1) | **RETRACTED** | Đọc lại `gat_encoder.py`: mask được apply **trước** softmax, đúng chuẩn. |
| "Virtual battery efficiency sai dấu" (Review 1) | **RETRACTED** | `delta = (η_ch·p_ch − p_dis/η_dis)·dt` đúng convention standard. |

**Lưu ý cho Codex**: nếu Codex review plan này và thấy "tại sao không đụng tới các phần trên" — câu trả lời là các phần đó **không có bug**. Đừng re-flag.

---

## 2. Phase 1 — P1 Blocker (fix trước khi chạy experiment)

### Task 1.1 — Fix unit inconsistency trong `_compute_branch_THD_I`

**File**: `src/eval/harmonic_analysis.py`, lines 189–230.

**Vấn đề**:
- `V_h_all` ở dạng **per-unit** (solve từ `Y_h` pu).
- `R, X` lấy từ `ppc["branch"][:, 2], [:, 3]` ở dạng **pu**.
- `I_h_br = (V_from − V_to) / Z_h` → kết quả **pu**.
- `I1_A = 1000 * net.res_line["i_from_ka"]` → **Ampere**.
- Tỷ số `sqrt(I_h_sq) / I1_A` lệch **factor I_base_A** → THD_I sai hệ số.

**Success criteria** (viết test trước khi fix):
```python
# tests/test_harmonic_units.py
def test_thd_i_unit_consistency_with_pure_resistive_net():
    # Mạng 2-bus resistive, biết trước analytical THD_I
    # THD_I phải nằm trong [expected*0.9, expected*1.1]
    ...
```

**Fix options** (chọn 1, document rõ):
- **Option A** (recommended): Convert `I_h_br` sang Ampere bằng `I_base_A` từ `baseMVA` và `vn_kv` của **from_bus**, rồi chia cho `I1_A`.
- **Option B**: Convert `I1_A` sang pu rồi chia.

**Acceptance**: test pure-resistive mạng 2-bus với 1 harmonic source cho THD_I analytical = `I_h/I1`, sai số ≤ 1%.

---

### Task 1.2 — Consolidate / document 3 implementations của reconfiguration

**Files liên quan**:
- `src/layer0_dso/reconfiguration.py` (984 dòng, Pyomo-MISOCP)
- `src/opt/l0_reconfig.py` (702 dòng, CVXPY)
- `src/opt/tie_switch_reconfig.py` (310 dòng, dùng trong `rl/train_dual.py` và `env/microgrid_env_dual.py`)

**Vấn đề**: 3 implementation song song, không rõ khi nào dùng cái nào. Risk: kết quả khác nhau giữa các experiment → reviewer paper có thể hỏi "tại sao 3 solver cho 3 kết quả khác nhau trên cùng instance?"

**Lưu ý thực tế** (đã update từ rev 1): **không yêu cầu `alpha_star` giống hệt 100%** vì:
- Pyomo-MISOCP vs CVXPY có thể tie-break branching khác nhau → 2 optimal solution hợp lệ với cùng objective.
- Tie-switch heuristic là **baseline**, không phải solver chính xác → luôn có gap so với MISOCP.
- Yêu cầu bit-exact là over-constraint và không reflect research reality.

**Tiêu chí tương đương mềm** (preferred):
1. **Feasibility parity**: nếu instance feasible trong A, thì feasible trong B. Nếu infeasible trong A, B cũng infeasible hoặc fall về elastic.
2. **Objective gap**: `|obj_A − obj_B| / max(|obj_A|, 1e-6) ≤ 1%` (hoặc tol tunable).
3. **Switch-Hamming distance**: `|{sw : alpha_A[sw] ≠ alpha_B[sw]}| ≤ K` với `K` nhỏ (ví dụ 2/3 switches). Không bắt buộc = 0.
4. **Downstream invariant**: voltage profile `v_pu` của A vs B phải trong band ±0.005 pu trên ≥ 95% bus (đây là thứ ảnh hưởng RL env).

**Fix**:
1. Viết `tests/test_reconfig_equivalence.py`:
   - Input: 1 canonical IEEE 123-bus snapshot (fixture trong `tests/fixtures/`).
   - Chạy cả 3 implementation với cùng input.
   - Assert 4 tiêu chí trên.
   - **Log diff** (không FAIL nếu vượt tolerance) → ghi vào `artifacts/reconfig_equivalence_report.json` để đưa vào paper appendix.
2. Viết `src/layer0_dso/README.md` (≤ 30 dòng):
   - File nào là canonical (MISOCP → ground truth).
   - Tie-switch là heuristic baseline (dùng cho RL env để chạy nhanh).
   - CVXPY là alternative solver (compatibility check).
3. **Không xoá** file nào (CLAUDE.md rule #3).

**Acceptance**:
- Test chạy được, output report có cột `hamming_dist`, `obj_gap_pct`, `v_band_violation_pct`.
- Nếu tiêu chí vượt tol → plan riêng để diagnose (không block paper).
- README giải thích lý do tồn tại song song cho reviewer.

---

### ✅ Task 1.3 — Centralize seeding cho reproducibility (với flag bật/tắt) — **DONE 2026-04-22**

**Implementation**:
- `src/layer2_control/mappo_policy.py`: `RolloutBuffer` nhận `seed`, dùng `self.rng = np.random.default_rng(seed)`; `get_minibatches()` đổi sang `self.rng.permutation`; `clear()` giữ seed cũ.
- `experiments/train_mappo.py`: `set_all_seeds(seed, deterministic=False)`, CLI flags `--seed`, `--deterministic`, `--benchmark-steps`; `run_training(..., seed_override, deterministic)`; `RolloutBuffer(seed=seed + update)` để shuffle reproducible qua update boundary.
- Verify: benchmark throughput toy hiển thị đúng; reproducibility qua cùng seed.

---

### Task 1.3 (original spec) — Centralize seeding cho reproducibility (với flag bật/tắt) ✅ DONE

**Files**: `experiments/train_mappo.py` (entry point), `src/layer2_control/mappo_policy.py` line 75.

**Vấn đề**:
- `mappo_policy.py:75` dùng `np.random.permutation(n)` **không seed** trong minibatch shuffle → run khác nhau mỗi lần.
- Không có central seed setup → torch, numpy, random có thể bị drift.

**Tradeoff cần acknowledge** (rev 3):
- `torch.use_deterministic_algorithms(True)` **giảm GPU throughput** ~10–30% (tùy GPU + ops).
- Một số CUDA kernel (e.g., `atomicAdd` trong scatter) sẽ raise lỗi vì không có deterministic variant → phải fallback CPU hoặc disable.
- Reproducibility bit-exact không phải lúc nào cũng cần. Paper mean±std qua N seeds có giá trị hơn single-run bit-exact.

**Fix** (với flag):
1. Ở đầu `experiments/train_mappo.py`:
   ```python
   def set_all_seeds(seed: int, deterministic: bool = False) -> None:
       import random, numpy as np, torch
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       if deterministic:
           torch.use_deterministic_algorithms(True, warn_only=True)
           torch.backends.cudnn.deterministic = True
           torch.backends.cudnn.benchmark = False
           # Cảnh báo: throughput GPU có thể giảm 10-30%
       else:
           torch.backends.cudnn.benchmark = True  # Faster, non-deterministic
   ```
2. Thêm CLI flag: `--deterministic` (default `False` cho training speed, `True` cho debug/ablation).
3. Trong `mappo_policy.py`: inject RNG `self.rng = np.random.default_rng(seed)`, thay `np.random.permutation` → `self.rng.permutation`. **Luôn làm** (không flag) vì không tốn throughput.

**Chính sách khuyến nghị**:
- **Training lớn / multi-seed ablation**: `--deterministic=False` + set seed khác nhau per run → tốc độ cao, reproducibility qua seed diversity.
- **Debug bug cụ thể / unit test**: `--deterministic=True` + single seed → bit-exact replay.
- **Paper results**: N=5 seeds, mean±std (xem §14 Research Protocol Gate).

**Acceptance**:
- `--deterministic=True` + cùng seed → reward curve identical tới float precision qua ≥ 100 steps.
- `--deterministic=False` + cùng seed → reward curve có variance nhỏ nhưng **trend giống** (không drift).
- Benchmark tốc độ: deterministic vs non-deterministic trên 1000 steps → log tỉ số throughput vào README.

---

## 3. Phase 2 — P2 Correctness (fix trước khi viết paper)

### Task 2.1 — Multi-voltage `V_base_kv` trong harmonic injection

**File**: `src/eval/harmonic_analysis.py`, lines 172–178.

**Vấn đề**: `V_base_kv = np.median(net.bus.vn_kv)` — sai cho mạng MV/LV hỗn hợp (IEEE 123 có 4.16 kV, 480 V, v.v.).

**Fix**: compute `I_base_A` **per-bus** dựa trên `net.bus.vn_kv[bus_k]`:
```python
for P_mw, bus_k in zip(agent_powers_mw, agent_bus_idx):
    if bus_k >= n_bus:
        continue
    V_base_kv_k = float(self.net.bus.at[bus_k, "vn_kv"])
    I_base_A_k = 1e6 * baseMVA / (np.sqrt(3.0) * max(V_base_kv_k, 1e-6) * 1000.0)
    I_pu_per_amp_k = 1.0 / max(I_base_A_k, 1e-9)
    ...
```

**Acceptance**: test với mạng 2 voltage level cho injection đúng pu per bus. Injection tổng theo pu không thay đổi nếu cùng `P_mw` và cùng voltage level.

---

### Task 2.2 — Bus shunt elements trong `_build_Yh`

**File**: `src/eval/harmonic_analysis.py`, lines 146–160.

**Vấn đề**: chỉ scale `branch[:, 3]` (X) và `branch[:, 4]` (B) theo `h`, không scale **bus shunt** (`bus[:, 5] = GS`, `bus[:, 6] = BS`). Với mạng có capacitor bank / shunt compensator → sai Y_h ở các bus đó.

**Fix**:
```python
bus_h = ppc["bus"].copy()
# Scale shunt susceptance BS (cột 6) theo h, giữ GS (cột 5) nếu neglect skin effect
bus_h[:, 6] = bus_h[:, 6] * h
```
**Chú ý**: scale này chỉ đúng nếu shunt là tụ điện (Y ∝ jωC). Nếu có reactor (L) hoặc resistor, cần xử lý riêng. Với IEEE 123 chủ yếu là capacitor → scale `*h` chấp nhận được, **document assumption này**.

**Acceptance**: test mạng với 1 shunt capacitor → Y_h tại bus đó tăng gấp `h` lần baseline Y_1.

---

### Task 2.3 — Tách true DLMP vs zone LMP shortcut

**Files**: `src/layer0_dso/dlmp_calculator.py` (true DLMP từ SOCP dual), `src/opt/l0_reconfig.py::L0Optimizer.compute_zone_lmp` (shortcut bypass MOSEK).

**Vấn đề**: `compute_zone_lmp` bypass SOCP dual → không phải true DLMP, chỉ là approximation. Tên hàm gây hiểu lầm.

**Fix**:
1. Rename `compute_zone_lmp` → `compute_approx_zone_lmp` (hoặc tên rõ ràng hơn).
2. Thêm docstring:
   ```
   APPROX zone LMP, KHÔNG phải DLMP đúng. Dùng khi không thể gọi MOSEK.
   Với true DLMP per-bus, dùng `dlmp_calculator.run_dlmp_calculation`.
   ```
3. Grep toàn repo tìm caller, kiểm tra có ai đang dùng nó như true DLMP không. Nếu có → fix caller hoặc raise warning.

**Acceptance**: grep `compute_zone_lmp` → 0 hit trừ definition; tất cả caller đổi tên; không ai nhầm lẫn.

---

### Task 2.4 — Safety layer: document heuristic hoặc upgrade QP

**File**: `src/layer2_control/safety_layer.py` (40 dòng).

**Vấn đề**: clip-then-uniform-scale **không phải** proper QP projection. Có thể violate box constraint sau khi scale.

**Fix** (chọn 1):
- **Option A** (minimum effort, P2): thêm assertion sau scale:
  ```python
  assert np.all(a_safe >= a_min - 1e-6) and np.all(a_safe <= a_max + 1e-6), \
      "Safety scaling violated box constraints"
  ```
  và viết docstring nói rõ đây là **heuristic**, không phải QP.
- **Option B** (proper fix, P2+): implement QP projection với `cvxpy` hoặc `osqp`:
  ```
  min ‖a − a_proposed‖²  s.t.  a_min ≤ a ≤ a_max,  Σ a · power_i ≤ P_limit
  ```

**Acceptance**: nếu Option A → assert không bao giờ fire trong 10k steps random. Nếu Option B → test so sánh với analytical projection trên case 2D đơn giản.

---

### Task 2.5 — Document GFM inverter trong `agent_bus_idx`

**File**: `src/eval/harmonic_analysis.py` docstring + `src/env/microgrid_env*.py` caller.

**Vấn đề**: Islanded microgrid → GFM inverter là **nguồn** sóng hài chính (không phải external grid). Code hiện tại ĐÚNG **chỉ nếu** `agent_bus_idx` bao gồm GFM bus.

**Fix**:
1. Thêm docstring vào `HarmonicAnalyzer.run`:
   ```
   IMPORTANT (islanded microgrid): `agent_bus_idx` MUST include the
   grid-forming (GFM) inverter bus. In islanded mode, the slack bus
   is itself a PWM harmonic source, not a harmonic short. Not including
   GFM will underestimate THD.
   ```
2. Thêm assertion (nếu metadata cho phép):
   ```python
   if getattr(self.net, '_gfm_bus_idx', None) is not None:
       assert self.net._gfm_bus_idx in agent_bus_idx, \
           "GFM bus must be in agent_bus_idx for islanded harmonic analysis"
   ```

**Acceptance**: docstring rõ ràng; assertion (nếu thêm) không fire trong test hiện tại.

---

## 4. Phase 3 — P3 Hygiene / nice-to-have

### Task 3.1 — Document các assumption trong PD / paper methodology

Tạo `docs/harmonic_assumptions.md` (≤ 1 trang) liệt kê:
- **Skin effect neglected**: `R(h) = R(1)` thay vì `R(h) = R(1)·√h`. Citation: MDPI Energies 2022 cho thấy skin effect nhỏ tại `h < 150` cho aluminum conductor đường kính nhỏ.
- **Phase-coherent worst-case**: injection tại line 185 `complex(I, 0.0)` — tất cả trong pha. Đây là **upper bound** THD, không phải expected value. Reference: IEEE TPE 2018 phase diversity paper.
- **Linear loading proxy**: `loading = |P|/P_rated` thay vì computing `ma` từ modulation depth động. Chấp nhận được với constant-power IBR.
- **Half-bridge PWM convention**: `V_mn = 2·Vdc/(mπ)·J_n(·)·sin(·)` với `Vdc = total DC bus`. Reference: Holmes & Lipo, "Pulse Width Modulation for Power Converters" (2003), Chapter 3.

**Acceptance**: file tồn tại, citations đầy đủ.

---

### Task 3.2 — Move VPP_PARAMS từ code sang YAML

**File**: `src/opt/l1_dispatch.py` — hardcoded dict cho VPP 1/2/3.

**Fix**:
1. Tạo `configs/vpp_params.yaml`:
   ```yaml
   vpp_1:
     bess_mw: ...
     bess_mwh: ...
     v2g_mw: ...
     S_agg: ...
   vpp_2: {...}
   vpp_3: {...}
   ```
2. Load bằng `yaml.safe_load` trong module init.
3. Xoá hardcoded dict.

**Acceptance**: test load YAML cho kết quả giống hardcoded dict cũ.

---

### Task 3.3 — Xoá hoặc document `zonalpricing.py` shim

**File**: `src/layer0_dso/zonalpricing.py` (5 dòng re-export shim của `zonal_pricing.py`).

**Fix**: grep xem có ai import `from src.layer0_dso.zonalpricing` không.
- Nếu 0 caller → xoá file.
- Nếu có caller → comment ở top: `# DEPRECATED shim, prefer zonal_pricing`.

**Acceptance**: hoặc file biến mất, hoặc có warning clear.

---

### Task 3.4 — Remove private `net._ppc` access

**File**: `src/eval/harmonic_analysis.py` line 51.

**Vấn đề**: `self.net._ppc` truy cập attribute private của pandapower, có thể break khi upgrade pandapower.

**Fix**: dùng public API `pandapower.pd2ppc._pd2ppc(net)` hoặc gọi `pp.runpp(net)` rồi `net._ppc` (chấp nhận) với docstring note.

**Acceptance**: vẫn hoạt động với pandapower >= 2.13 (test trong CI nếu có).

---

### Task 3.5 — Configurable `pcc_idx`

**File**: `src/eval/harmonic_analysis.py` line 113 (`pcc_idx = 0`).

**Fix**: thêm `pcc_bus_idx: int | None = None` vào `run()` signature; nếu None → default 0 (backward compat).

**Acceptance**: test với `pcc_bus_idx=5` cho `THD_V_PCC = THD_V[5]`.

---

### Task 3.6 — Named constants thay magic thresholds

**File**: `src/eval/harmonic_analysis.py` lines 98, 223.

**Fix**:
```python
MIN_V_PU_FOR_THD = 0.05  # below this, bus considered de-energized
MIN_I_A_FOR_THD = 1.0    # below this, branch considered idle
```
Dùng thay cho `0.05` và `1.0` trực tiếp.

**Acceptance**: code đọc rõ ràng hơn.

---

### Task 3.7 — Property-based tests cho harmonic analyzer

**File mới**: `tests/test_harmonic_properties.py`.

**Tests**:
1. **Pure-resistive network** → THD phải bằng 0 (vì Z_h không phụ thuộc h cho R thuần, nhưng `X=0` → trivial; verify logic).
2. **Single-bus single-inverter**: compare với analytical `V_h = I_h · Z_source`.
3. **Linearity**: nếu scale tất cả `agent_powers_mw` lên 2x, `V_h_all` scale 2x (với loading chưa bão hoà), `THD_V` không đổi (vì V1 cũng scale).
4. **Symmetry**: đổi `agent_bus_idx` order không đổi kết quả.

**Acceptance**: 4 tests pass.

---

### Task 3.8 — Vectorize nested loop branch×harmonic

**File**: `src/eval/harmonic_analysis.py` lines 198–210 (`_compute_branch_THD_I`).

**Fix**: thay double loop bằng `np.einsum` hoặc broadcasting:
```python
# Z_h[b, h] = R[b] + j*X[b]*HARMONICS[h]
# V_diff[b, h] = V_h_all[from_bus[b], h] - V_h_all[to_bus[b], h]
# I_h_br[b, h] = V_diff / Z_h
# I_h_sq[b] = sum over h of |I_h_br|²
```

**Acceptance**: cho cùng kết quả như loop cũ (sai số ≤ 1e-10). Benchmark gate theo tier: tốc độ ≥ 3× ở medium scale và ≥ 2× ở large scale; ghi rõ trend hiệu năng trong báo cáo.

---

### Task 3.9 — Reward shaping / baseline cho MAPPO

**File**: `src/layer2_control/reward.py` (30 dòng).

**Vấn đề**: reward luôn `≤ 0` → optimum là 0, khó tạo gradient signal mạnh.

**Fix** (optional, P3): thêm baseline `r = r_raw − r_baseline` hoặc shift `r = r_raw + offset`. Cần ablation trước khi áp dụng.

**Acceptance**: ablation study chứng minh convergence nhanh hơn, hoặc skip task này.

---

### Task 3.10 — Entropy coefficient ablation

**File**: `src/layer2_control/mappo_policy.py` line 23 (`entropy_coef=0.01`).

**Fix**: chạy ablation với `[0.001, 0.01, 0.1]`, report trong appendix paper.

**Acceptance**: 3 run với seed cố định, so sánh reward curve.

---

## 5. Testing & CI gates

Trước khi merge bất kỳ task nào vào main:

1. `pytest tests/` pass toàn bộ (bao gồm test mới viết).
2. `pytest tests/test_reconfig_consistency.py` pass (Task 1.2).
3. `pytest tests/test_harmonic_units.py` pass (Task 1.1).
4. Chạy 1 mini-experiment 100 steps với seed=42 → reward curve reproducible.

---

## 6. Thứ tự đề xuất cho vibe coding (rev 1 — giữ lại để tham khảo)

> **Lưu ý**: sau rev 2, thứ tự mới đặt trong §11. Giữ section này làm context lịch sử.

Từ dễ tới khó, nhưng vẫn respect priority:

1. **Task 1.3** (seeding) — quick win, nhỏ gọn, fix ngay.
2. **Task 1.1** (unit fix harmonic) — P1 quan trọng nhất về science.
3. **Task 1.2** (reconfig consolidation) — cần code reading nhiều.
4. **Task 2.1, 2.2, 2.5** (harmonic polish).
5. **Task 2.3, 2.4** (DLMP rename, safety layer).
6. **Task 3.x** (hygiene, làm cuối).

---

## 7. Ghi chú cho Codex reviewer

- **Section 1 (Retractions)** liệt kê các claim đã rút lại. Codex đừng re-raise.
- Nếu Codex tìm thấy bug MỚI ngoài plan này → welcome, add vào new section "Codex additions".
- Nếu Codex disagree với fix option đã chọn (ví dụ Task 1.1 Option A vs B) → comment trong PR, không block.
- **Không yêu cầu** Codex verify lại Holmes-Lipo factor, Baran-Wu DistFlow, hoặc các claim trong section 1 — đã verify bằng WebSearch + textbook citation.

---

## 9. DONE — đã hoàn thành qua patch `floating-snuggling-giraffe.md`

User đã triển khai và verify pass toàn bộ các item sau (2026-04-22).

### 9.1 — L0Result metadata labeling ✅
- **File**: `src/opt/l0_reconfig.py:18`
- **Done**: thêm `solution_mode: str = "strict"`, `strict_status: str = "unknown"`, `elastic_used: bool = False` (default values → backward-compat).
- **Verify**: `tests/test_l0_reconfig_metadata.py::test_l0_result_has_strict_elastic_labels` pass.

### 9.2 — Strict-first → elastic-second pattern ✅
- **File**: `src/opt/l0_reconfig.py:112` (`L0Optimizer.solve`)
- **Done**: `_solve_problem(slack_weight=1e6)` strict → nếu fail thì `_solve_problem(slack_weight=1e4)` elastic. `solution_mode` chỉ set "elastic" khi strict fail **AND** elastic succeed → không có nhãn sai.
- **Verify**: test pass, smoke test in đúng `mode=strict|elastic` và `strict_status`.

### 9.3 — `build_net_data_from_pandapower` trả `slack_bus` + `zone_totals` ✅
- **File**: `src/opt/l0_reconfig.py:416`
- **Done**: zone mapping 2-pass (bus_idx + bus_id) tránh fallback 0 giả; `slack_bus` extract từ `ext_grid.iloc[0]` qua `bus_id` mapping.
- **Verify**: `tests/test_l0_reconfig_metadata.py::test_build_net_data_exposes_solver_metadata` assert `slack_bus is not None` và `zone_totals.keys() == {1,2,3,4}`.

### 9.4 — `NetworkReconfiguration._run_l0_bfm_socp` dynamic inputs ✅
- **File**: `src/opt/l0_reconfig.py:593`
- **Done**: `profiles["load_z*"]` lấy từ `net_data["zone_totals"]`; `vpp_caps` tính từ `placement_totals` (pattern giống `_build_vpp_caps`).
- **Verify**: `tests/test_layer0_quality_gate.py` pass, smoke test thấy `zone_totals` có giá trị thật.

### 9.5 — `precompute.py` status helper + zonal field direct access ✅
- **File**: `src/opt/precompute.py:176` (`_build_l0_status`), `:274` + `:333` (caller), `:282` (zonal fields)
- **Done**: 3 state rõ ràng `feasible_strict` / `feasible_elastic` / `infeasible`. Zonal `lambda_p2p_zN` lấy trực tiếp từ `L0Result`.
- **Verify**: `tests/test_day6b_optimizers.py`, `tests/test_layer0_layer1_io.py`, `tests/test_offline_phase_pipeline.py` pass.

### 9.6 — Smoke test in rõ metadata ✅
- **File**: `scripts/smoke_test.py:246`
- **Done**: in `slack_bus`, `zone_totals`, `solution_mode`, `strict_status`.
- **Verify**: `python scripts/smoke_test.py` L0/L1 GATE PASS.

**Tổng verify**: 13 tests pass, smoke PASS, test_env_basic PASS.

---

## 10. Follow-up từ re-review patch `floating-snuggling-giraffe.md`

Plan re-review phát hiện các vấn đề **mới** trong patch đã commit. Không phải rework toàn bộ, chỉ là fine-tuning.

### ✅ [P1] Task 10.1 — Silent fallback khi `slack_bus=None` ⚠️ BLOCKER — **DONE 2026-04-22**

**Implementation**: `src/opt/l0_reconfig.py:142-151` — bỏ fallback `self.bus_idx.get(slack_bus, 0)`; raise `ValueError` khi `slack_bus=None`, raise `KeyError` khi slack không map được vào `bus_idx`.
**Verify**: `pytest tests/test_l0_reconfig_metadata.py` 3 passed; smoke test PASS.

---

#### [Original spec] Task 10.1 — Silent fallback khi `slack_bus=None` ⚠️ BLOCKER ✅ DONE

**File**: `src/opt/l0_reconfig.py:142–144`

**Vấn đề**:
```python
slack_bus = self.net_data.get("slack_bus")
slack_idx = self.bus_idx.get(slack_bus, 0)  # ← fallback 0 nếu None
constraints.append(v[slack_idx] == 1.0)
```
Nếu `build_net_data_from_pandapower` trả `slack_bus=None` (khi `net.ext_grid.empty`), ta **silent fallback về bus index 0** mà không warning. Nếu bus 0 không phải slack thật (islanded microgrid có thể đặt GFM ở bus khác), voltage profile sẽ sai toàn mạng mà không debug được.

**Fix**:
```python
slack_bus = self.net_data.get("slack_bus")
if slack_bus is None:
    raise ValueError(
        "net_data missing slack_bus; cannot enforce v[slack]=1.0. "
        "Check that net.ext_grid is populated or pass slack_bus explicitly."
    )
slack_idx = self.bus_idx.get(slack_bus)
if slack_idx is None:
    raise KeyError(f"slack_bus={slack_bus} not in bus_idx={list(self.bus_idx.keys())[:5]}...")
constraints.append(v[slack_idx] == 1.0)
```

**Acceptance**:
- Test mới `test_l0_raises_on_missing_slack_bus`: build `net_data` không có `slack_bus` → expect `ValueError`.
- Các test hiện tại vẫn pass (vì IEEE 123 có ext_grid).

---

### ✅ [P2] Task 10.2 — Hardcode `wind_mw=8.0` còn sót trong `_run_l0_bfm_socp` — **DONE 2026-04-22**

**Implementation**:
- `src/opt/l0_reconfig.py:600` — `_run_l0_bfm_socp(self, net, pv_scale, wind_mw=8.0)` thêm param.
- `src/opt/l0_reconfig.py:611` — `profiles["wind_mw"] = float(wind_mw)` thay hardcode.
- `src/opt/l0_reconfig.py:688` — caller `precompute()` truyền `wind_mw` động (sample uniform 0..12).

---

#### [Original spec] Task 10.2 — Hardcode `wind_mw=8.0` còn sót trong `_run_l0_bfm_socp` ✅ DONE

**File**: `src/opt/l0_reconfig.py:604`

**Vấn đề**: Claim "bỏ hardcode" trong patch chỉ đúng cho `load_z*`. `wind_mw` vẫn hardcoded 8.0 trong `_run_l0_bfm_socp`, không phản ánh profile ngày (precompute.py đã có `wind_mw` động per step).

**Fix**:
```python
def _run_l0_bfm_socp(self, net, pv_scale: float, wind_mw: float = 8.0) -> tuple[...]:
    ...
    profiles = {
        ...
        "wind_mw": float(wind_mw),
    }
```
Caller trong `precompute._generate_day` hiện tại gọi `optimizer.solve` trực tiếp nên đã dùng wind_mw động. Chỉ fix path qua `NetworkReconfiguration.precompute` (dùng để random-sample topology cho RL env).

**Acceptance**: unit test verify `wind_mw` param được propagate xuống `profiles`.

---

### ✅ [P2] Task 10.3 — Scale overflow khi `zone_totals[zone]` rất nhỏ — **DONE 2026-04-22**

**Implementation**: `src/opt/l0_reconfig.py:154-160` — thêm `MIN_ZONE_LOAD_MW = 1e-3`, đổi điều kiện `if total <= 0.0` sang `if total < MIN_ZONE_LOAD_MW`.

---

#### [Original spec] Task 10.3 — Scale overflow khi `zone_totals[zone]` rất nhỏ ✅ DONE

**File**: `src/opt/l0_reconfig.py:147–151`

**Vấn đề**:
```python
if total <= 0.0:
    load_scales[zone] = 1.0
else:
    load_scales[zone] = float(profiles.get(f"load_z{zone}", total)) / total
```
Nếu `total = 1e-6` (zone gần-như-rỗng nhưng không hẳn 0), ratio nổ lên → `P_scaled` vượt DER capacity → trigger elastic fallback **giả**. Metadata sẽ report `solution_mode=elastic` trong khi thực ra là numerical artifact.

**Fix**: đổi ngưỡng
```python
MIN_ZONE_LOAD_MW = 1e-3  # 1 kW
if total < MIN_ZONE_LOAD_MW:
    load_scales[zone] = 1.0
else:
    load_scales[zone] = float(profiles.get(f"load_z{zone}", total)) / total
```

**Acceptance**: unit test với `zone_totals = {1: 1e-6, 2: 5.0, ...}` không gây elastic fallback.

---

### ✅ [P3] Task 10.4 — `getattr` dư thừa cho field dataclass — **DONE 2026-04-22**

**Implementation**:
- `src/opt/precompute.py:176-179` — `_build_l0_status()` dùng trực tiếp `result.elastic_used`.
- `scripts/smoke_test.py:262-265` — in trực tiếp `l0_result.solution_mode`, `l0_result.strict_status`.

---

#### [Original spec] Task 10.4 — `getattr` dư thừa cho field dataclass ✅ DONE

**Files**:
- `src/opt/precompute.py:179`: `getattr(result, 'elastic_used', False)`
- `scripts/smoke_test.py:262–264`: `getattr(l0_result, 'solution_mode', 'strict')`, `getattr(l0_result, 'strict_status', l0_result.status)`

**Vấn đề**: `L0Result` là frozen dataclass với field bắt buộc có default. `getattr` với fallback che lỗi nếu ai refactor xoá field → chương trình chạy tiếp với value sai thay vì crash rõ ràng.

**Fix**: thay bằng truy cập trực tiếp `result.elastic_used`, `l0_result.solution_mode`, `l0_result.strict_status`.

**Acceptance**: grep `getattr.*elastic_used\|solution_mode\|strict_status` → 0 hit.

---

### ✅ [P3] Task 10.5 — Test elastic branch coverage — **DONE 2026-04-22**

**Implementation**: `tests/test_l0_reconfig_metadata.py` — thêm `test_l0_elastic_fallback_when_strict_infeasible(...)` (monkeypatch solve để ép strict fail rồi elastic pass), assert:
- `solution_mode == "elastic"`
- `elastic_used is True`
- `strict_status == "infeasible"`
- `status in {"optimal", "optimal_inaccurate"}`

**Verify**: `pytest tests/test_l0_reconfig_metadata.py` 3 passed.

---

#### [Original spec] Task 10.5 — Test elastic branch coverage ✅ DONE

**File**: `tests/test_l0_reconfig_metadata.py`

**Vấn đề**: `test_l0_result_has_strict_elastic_labels` chỉ test happy-path (strict success). Không assert được branch `solution_mode == "elastic"` → regression có thể lọt.

**Fix**: thêm test thứ 2
```python
def test_l0_elastic_fallback_when_strict_infeasible() -> None:
    net = build_ieee123_net(...)
    net_data = build_net_data_from_pandapower(net)
    optimizer = L0Optimizer(net_data)
    
    # Force infeasible strict bằng profile quá lớn
    profiles = {f"load_z{z}": 9999.0 for z in [1,2,3,4]}
    profiles.update({"pv_pu": 0.0, "wind_mw": 0.0})
    
    result = optimizer.solve(1, profiles, vpp_caps_minimal)
    assert result.solution_mode == "elastic"
    assert result.elastic_used is True
    assert result.strict_status not in {"optimal", "optimal_inaccurate"}
```
**Lưu ý**: nếu elastic cũng fail → test sẽ FAIL. Cần điều chỉnh profile sao cho strict infeasible nhưng elastic feasible (slack trọng số 1e4 < 1e6).

**Acceptance**: test mới pass, cover branch elastic.

---

### ✅ [P3] Task 10.6 — Dead code: no-op cast `zone_totals` — **DONE 2026-04-22**

**Implementation**: `src/opt/l0_reconfig.py:565` — xóa dòng `zone_totals = {zone: float(total) for zone, total in zone_totals.items()}`.

---

#### [Original spec] Task 10.6 — Dead code: no-op cast `zone_totals` ✅ DONE

**File**: `src/opt/l0_reconfig.py:557`

**Vấn đề**: `zone_totals = {zone: float(total) for zone, total in zone_totals.items()}` — dict đã là float, cast này vô nghĩa.

**Fix**: xoá dòng đó.

**Acceptance**: test vẫn pass.

---

### [P3] Task 10.7 — Observability: tách `strict_solve_time` / `elastic_solve_time`

**File**: `src/opt/l0_reconfig.py:332–343`

**Vấn đề**: `solve_time` hiện là tổng thời gian strict + elastic. Khi elastic được trigger, không biết strict tốn bao lâu mới fail — khó debug performance bottleneck.

**Fix** (optional): thêm field vào `L0Result`:
```python
strict_solve_time: float = 0.0
elastic_solve_time: float = 0.0
```
Set trong `solve()` sau mỗi branch.

**Acceptance**: smoke test in ra 2 time riêng; không break test hiện tại (field có default).

---

### [P3] Task 10.8 — Document zone 3 không có P2P market

**File**: `src/opt/l0_reconfig.py:100–108`, docstring

**Vấn đề**: `compute_zone_lmp` chỉ tạo LMP cho `[1, 2, 4]` (zone 3 là wind zone, không có P2P). Nhưng `zone_net_mw` lại có zone 3 (L264). Inconsistency đã tồn tại trước patch — không phải regression. Giờ metadata rõ → nên document.

**Fix**: docstring `compute_zone_lmp`:
```
Zone 3 (wind zone) không có P2P market trong thiết kế hiện tại.
Chỉ zone 1, 2, 4 tham gia P2P clearing. zone_net_mw vẫn track zone 3
cho phân tích flow, nhưng không có lambda_p2p_z3.
```

**Acceptance**: docstring tồn tại.

---

### [P3] Task 10.9 — Refactor `build_net_data_from_pandapower` (>150 dòng)

**File**: `src/opt/l0_reconfig.py:416`

**Vấn đề**: function đã lớn trước patch, giờ thêm zone mapping + slack bus logic → 150+ dòng, khó test unit.

**Fix** (nice-to-have): tách
- `_extract_zone_mapping(net) -> (bus_idx_to_zone, bus_id_to_zone)`
- `_extract_buses_with_zones(net, bus_idx_to_zone, bus_id_to_zone) -> (buses, zone_totals)`
- `_extract_slack_bus(net) -> int | None`
- `_extract_branches(net, Z_base) -> list[tuple]`
- `_extract_ders(net) -> list[tuple]`

**Acceptance**: function `build_net_data_from_pandapower` giảm xuống ≤ 50 dòng, các helper unit-testable.

---

### ✅ [P1] Task 10.11 — `elastic_used_ratio` như KPI bắt buộc — **DONE 2026-04-22**

**Implementation**:
- `src/opt/precompute.py:182` — đổi return `_generate_day(...)` thành `(df, l0_status, l0_stats)`.
- `src/opt/precompute.py:257-276` + `319-346` — cộng dồn `strict_success` / `elastic_success` / `infeasible` theo từng lần gọi L0 solve thực tế.
- `src/opt/precompute.py:453-507` — aggregate toàn bộ ngày, compute `elastic_used_ratio = elastic_success / (strict + elastic)`, `health = healthy|warn|fail` theo ngưỡng 5%/10%, ghi vào `metadata.json → solver_stats`.
- `scripts/smoke_test.py:294-314` — đọc metadata, in `L0 solver health: strict=... elastic=... infeasible=... [HEALTHY/WARN/FAIL]`, fail gate khi `elastic_used_ratio > 10%`.

**Verify**: smoke test in `L0 solver health: strict=100.0%, elastic=0.0%, infeasible=0.0% [HEALTHY]`.

---

#### [Original spec] Task 10.11 — `elastic_used_ratio` như KPI bắt buộc (không chỉ log đẹp)

**Files**: `src/opt/precompute.py` (aggregate), `scripts/smoke_test.py` (gate), paper methodology section.

**Vấn đề**: Hiện tại `elastic_used` chỉ là label, không có ngưỡng hay action. Nếu trong 1000 steps có 500 lần elastic → **tín hiệu rõ ràng** rằng:
- Profile data vượt khả năng DER (data issue), hoặc
- Placement chưa đủ DER capacity (model issue), hoặc
- Hyperparam chưa đúng (slack weight quá cao, `vpp_caps` quá chặt).

Nếu bỏ qua, paper sẽ có pipeline "always falling back to elastic" mà vẫn report results → **reviewer sẽ reject**.

**Định nghĩa KPI**:
```
elastic_used_ratio = n_elastic_success / (n_strict_success + n_elastic_success)
```

**Ngưỡng cảnh báo**:
- `≤ 5%`: healthy — strict dominates, elastic chỉ là safety net.
- `5–10%`: **WARN** — log cảnh báo, chạy diagnostic (check profile outliers, DER capacity margin).
- `> 10%`: **FAIL** — pipeline coi là không hợp lệ cho paper. Phải root-cause trước khi generate dataset.

**Fix**:
1. Trong `src/opt/precompute.py::generate_all_days`: accumulate counter `n_strict`, `n_elastic`, `n_infeasible` qua toàn bộ days.
2. Cuối `generate_all_days`, compute ratio và ghi vào `metadata.json`:
   ```json
   "solver_stats": {
     "elastic_used_ratio": 0.023,
     "strict_success": 950,
     "elastic_success": 22,
     "infeasible": 28,
     "total_calls": 1000,
     "health": "healthy"  // hoặc "warn" / "fail"
   }
   ```
3. Trong `scripts/smoke_test.py`: **gate failure** nếu `elastic_used_ratio > 0.10` trên 1-day smoke:
   ```python
   ratio = metadata["solver_stats"]["elastic_used_ratio"]
   if ratio > 0.10:
       _fail(f"elastic_used_ratio {ratio:.1%} > 10% — pipeline invalid")
   elif ratio > 0.05:
       print(f"WARN: elastic_used_ratio {ratio:.1%} > 5% — check data/model")
   ```
4. Trong paper methodology: **báo cáo** `elastic_used_ratio` cho cả train và eval set (thường trong appendix hoặc table "Solver Health Metrics").

**Acceptance**:
- `metadata.json` có field `solver_stats`.
- Smoke test fail nếu ratio > 10%.
- Log console có 1 dòng tóm tắt: `L0 solver health: strict=95.0%, elastic=2.2%, infeasible=2.8% [HEALTHY]`.
- Paper appendix có row table: `elastic_used_ratio_train`, `elastic_used_ratio_eval`.

**Thực tế trong research**: nếu thấy ratio cao trong prototype → **đây là data**, không che giấu. Show reviewers: "out of 1000 instances, 23 hit elastic mode" → chứng minh rigor chứ không phải xấu.

---

### [P3] Task 10.10 — Warning filter repeat trong `_solve_problem`

**File**: `src/opt/l0_reconfig.py:297–306`

**Vấn đề**: `warnings.filterwarnings(...)` nằm trong `_solve_problem`, được gọi 2 lần (strict + elastic). Warning filter là stack-based → OK về mặt correctness, nhưng dư thừa.

**Fix**: move `warnings.filterwarnings` ra đầu `solve()` hoặc module-level. Nhỏ, low priority.

**Acceptance**: chỉ còn 2 dòng `filterwarnings` trong module, không duplicate.

---

## 11. Thứ tự đề xuất cho vibe coding (rev 3 — research-impact ordered)

**Nguyên tắc rev 3**: §10 là fix live pipeline L0 (đang chạy thực tế) → **đẩy lên tuyệt đối đầu**, trước mọi task harmonic (scope khác). §14 (Research Protocol Gate) được áp dụng song song mọi lúc.

### Đợt 1 — Fix live pipeline L0 (BẮT BUỘC trước khi generate dataset cho paper)

1. **Task 10.1** (slack_bus raise ValueError) — P1 Critical, ~15 phút.
2. **Task 10.11** (elastic_used_ratio KPI + gate) — P1, ~30 phút. **Quan trọng nhất cho paper.**
3. **Task 10.2** (wind_mw propagate) — P2, ~10 phút.
4. **Task 10.3** (scale overflow guard) — P2, ~10 phút.
5. **Task 10.5** (test elastic branch) — P3 nhưng cần cho CI gate, ~30 phút.

→ **Gate**: chạy `scripts/smoke_test.py` 1 day. Nếu `elastic_used_ratio ≤ 5%` thì pipeline healthy, tiến tiếp.

### Đợt 2 — Cleanup L0 (song song hoặc sau Đợt 1)

6. **Task 10.4, 10.6** (getattr + dead code) — P3, ~10 phút.
7. **Task 10.7, 10.8, 10.9, 10.10** — optional.

### Đợt 3 — Harmonic + Layer architecture (scope khác, chưa làm)

8. **Task 1.1** (harmonic unit fix) — P1 cho harmonic correctness.
9. **Task 1.3** (seeding với deterministic flag) — cần trước khi multi-seed training (§14).
10. **Task 1.2** (reconfig equivalence test) — dùng mềm hóa tiêu chí.
11. **Task 2.1, 2.2, 2.5** (harmonic polish).
12. **Task 2.3, 2.4** (DLMP rename, safety layer).
13. **Task 3.x** (hygiene, làm cuối).

### Đợt 4 — Research protocol (song song với Đợt 3)

14. Setup **Research Protocol Gate (§14)** — N=5 seeds, mean±std cho mọi thay đổi thuật toán.

---

## 14. Research Protocol Gate — ≥ 3–5 seeds, mean±std

**Mục đích**: tách "code đúng" khỏi "kết quả có ý nghĩa thống kê". Mọi claim trong paper phải pass gate này.

### 14.1 Gate rule

Mọi thay đổi thuật toán (hyperparameter, reward shaping, architecture) trước khi được report trong paper hoặc thesis:

1. **Số seeds tối thiểu**:
   - Ablation / screening: **N = 3** (tiết kiệm compute).
   - Paper main results: **N = 5** (đủ cho t-test, ít variance).
   - Critical claim (e.g., beat SOTA): **N ≥ 10** (narrow CI).
2. **Seed selection**: deterministic list từ `configs/seeds.yaml` — ví dụ `[42, 1337, 2024, 777, 3141]`. **Không cherry-pick** seed tốt.
3. **Report format**: `mean ± std` (không phải `min–max` hay `best`).
4. **Statistical test**: khi so 2 method, dùng **Welch's t-test** (unequal variance) hoặc **Mann-Whitney U** (non-parametric). Report p-value.
5. **Wall-clock tracking**: log wall-clock per seed → cảnh báo nếu variance tốc độ > 2× (có thể là resource contention).

### 14.2 Implementation

1. Viết `experiments/run_multi_seed.py`:
   ```python
   def run_sweep(config_path, seeds: list[int], deterministic: bool = False):
       results = []
       for seed in seeds:
           result = run_single(config_path, seed, deterministic=deterministic)
           results.append(result)
       mean, std = np.mean(results, 0), np.std(results, 0)
       save_to(f"artifacts/multi_seed/{config_hash}.json", {
           "config": config_path, "seeds": seeds,
           "mean": mean.tolist(), "std": std.tolist(),
           "individual": results,
       })
   ```
2. Viết `scripts/compare_methods.py`:
   - Input: 2 JSON results từ multi-seed run.
   - Output: table `method | mean | std | N | p-value vs baseline`.
   - Note: p-value đã được cập nhật sang Welch t-distribution (SciPy `stats.t.sf`), không dùng normal approximation.
3. Tạo `docs/paper_protocol.md` (≤ 1 trang):
   - Checklist bắt buộc trước mỗi paper submission.
   - Template cho caption table: `"Results are mean ± std over N=5 seeds [seed list]. p-values from Welch's t-test."`

### 14.3 Negative results protocol

Nếu ablation cho thấy method mới **không** beat baseline (hoặc chỉ beat trong 2/5 seeds):
- **Không** ẩn kết quả.
- Report trong paper: "Method X improved mean reward in 2/5 seeds (not statistically significant at p=0.05). We report this as a negative result to prevent publication bias."
- Đây là practice chuẩn (RL reproducibility crisis — Henderson et al. 2018).

### 14.4 Acceptance cho gate

- [x] `configs/seeds.yaml` tồn tại với 10 seeds.
- [x] `experiments/run_multi_seed.py` implement và test pass trên toy problem.
- [x] `scripts/compare_methods.py` output đúng format table.
- [x] `docs/paper_protocol.md` đã có template caption và checklist gate.
- [ ] Mỗi ablation trong paper có file `.json` trong `artifacts/multi_seed/` (pending khi chạy training thật).

### 14.5 Compute budget warning

N=5 seeds × 10 ablations × 1M steps ≈ 50M steps. Với MAPPO trên IEEE 123 + harmonic eval, giả sử 100 steps/sec → 500k sec = **~140 GPU-hours**. Plan compute trước khi nhận deadline.

**Khuyến nghị**: screen hyperparam với N=3, seeds `[42, 1337, 2024]`. Khi pick được config cuối → chạy N=5 full với seeds `[42, 1337, 2024, 777, 3141]`.

---

## 12. Ghi chú cho Codex reviewer (rev 4)

- **Đừng re-flag** các item trong §1 (Retractions), §9 (Done patch `floating-snuggling-giraffe.md`), và các task đã mark ✅ DONE trong §2, §10.
- **DONE status snapshot 2026-04-22**:
  - §Task 1.3 ✅ (seeding + `--deterministic` flag + RolloutBuffer RNG injection)
  - §Task 10.1 → 10.6 ✅ (slack guard, wind_mw propagate, scale guard, getattr cleanup, elastic test, dead code)
  - §Task 10.11 ✅ (elastic_used_ratio KPI + smoke gate)
  - §14.1 → 14.4 infrastructure ✅ (seeds.yaml, run_multi_seed, compare_methods Welch t-dist, paper_protocol.md)
- **Pending** (không phải bug, là roadmap):
  - §Task 1.1, 1.2, 2.x, 3.x (harmonic + layer architecture scope)
  - §10.7, 10.8, 10.9, 10.10 (hygiene cleanup batch)
  - §14: ablation JSON chạy training thật (phụ thuộc compute)
- **Nếu Codex phát hiện thêm bug** trong patch đã commit → thêm section "Codex additions to §10".
- **Không yêu cầu** Codex verify lại formula harmonic (đã có citation Holmes-Lipo, MDPI, v.v.).
- **Codex đánh giá theo 2 lens** (rev 3):
  - *Code lens*: correctness, security, performance → cho từng task.
  - *Research lens*: mỗi thay đổi thuật toán có pass §14 Research Protocol Gate không? (≥ 3–5 seeds, mean±std, Welch t-test p-value).
- **Task 1.2 đã soften** (rev 3): đừng flag "test chưa assert alpha giống 100%" — đó là intentional. Xem §Task 1.2 "Tiêu chí tương đương mềm".
- **Welch t-test correctness**: `scripts/compare_methods.py` dùng t-distribution CDF (fix P2 issue đã flag ở rev 3 audit). Không còn normal approximation.

---

## 13. Sources đã verify

Các claim formula-level trong plan này dựa trên:
- Holmes & Lipo, *Pulse Width Modulation for Power Converters*, IEEE Press, 2003 (Ch. 3 — half-bridge convention).
- Baran & Wu, "Optimal capacitor placement on radial distribution systems," IEEE TPWRD 1989 (DistFlow equations).
- MDPI Energies 2022 — skin effect tại frequency range `h < 150`.
- IEEE Trans. Power Ele