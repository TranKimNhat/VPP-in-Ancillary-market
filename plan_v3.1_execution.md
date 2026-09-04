# Kế hoạch thực thi — Concept v3.1

**Bài:** *When Do Reduced Frequency Models Overestimate Security in 100% Inverter-Based Islanded Microgrids?*
**Tài liệu nguồn:** `concept_v2_100pct_ibr_frequency_security.md` (v3.1, đã đóng băng)
**Phạm vi tài liệu này:** cách thực hiện — không thay đổi bất kỳ mục nào đã frozen ở §25 của concept.
**Ngày lập:** 2026-09-02 · **Trạng thái:** dự thảo, chờ phê duyệt Phần 0

---

## Nguyên tắc làm việc

| # | Nguyên tắc |
|---|---|
| N-1 | **Không có bước nào đi tiếp khi cổng của bước trước chưa đạt.** Mỗi phase có tiêu chí nghiệm thu định lượng ở §4. |
| N-2 | **Mọi con số trong bài truy vết được về một tệp kết quả tái chạy được.** Không con số nào chỉ tồn tại trong bản thảo. Lược đồ artifact ở §5. |
| N-3 | **Không tuyên bố vượt quá điều mô phỏng làm.** Mọi giới hạn phạm vi phải ghi rõ trong bài. |
| N-4 | **Mô hình chuẩn EMT không được tự chế.** Dùng thư viện/mô hình đã công bố; nếu buộc phải tự dựng thì phải kiểm chứng riêng. Đây chính là chỗ bản thảo cũ đã ngã. |
| N-5 | **Con người giữ toàn quyền thẩm định.** Mọi tham số, ngưỡng và diễn giải vật lý do nhóm nghiên cứu xác nhận. |

---

# Phần 0 — Bốn quyết định chặn, cần chốt trước khi viết dòng mã nào

Concept liệt kê những mục này ở §25 "open" nhưng không giao chủ sở hữu. Chúng chặn đường găng.

### D1 — Nền tảng EMT

| Phương án | Ưu | Nhược |
|---|---|---|
| **MATLAB/Simulink + Specialized Power Systems** *(khuyến nghị)* | SG + governor + AVR là **khối thư viện có sẵn** — riêng điều này tiết kiệm nhiều tuần ở Task 7. Kịch bản hoá được từ MATLAB nên tự động hoá campaign dễ. Phổ biến trong bài EMT microgrid trên TPWRS. | Chậm hơn PSCAD trên mạng lớn. |
| PSCAD/EMTDC | Cùng nền với Trujillo (M1/M2) và Kenyon (M3); nếu lấy được bộ mô hình inverter mã nguồn mở của Kenyon thì so sánh đặt trên cùng cơ sở. | Bản quyền; tự động hoá campaign khó hơn; cần xác minh bộ mô hình Kenyon có truy cập được không. |
| PLECS | Nhanh cho converter. | Yếu cho mạng 123 nút và cho máy đồng bộ. |
| Tự viết bằng Python | Tích hợp thẳng với vòng lấy mẫu thích nghi. | **Loại.** Vi phạm N-4 — "chúng tôi tự viết EMT" mời lại đúng chỉ trích R1-2 mà bài này vừa thoát ra. |

**Quyết định phụ D1b — độ trung thực VSC: dùng mô hình trung bình (switching-averaged), không phải đóng cắt đầy đủ.** Đây là mức mà mô hình chuẩn PSCAD của Trujillo dùng và là chuẩn cho nghiên cứu cấp hệ thống. Nhanh hơn 10–100 lần. Đóng cắt đầy đủ chỉ dùng cho 2–3 ca kiểm tra tính hợp lý. **Đây là đòn bẩy khả thi lớn nhất của cả dự án** và concept chưa nêu.

### D2 — Nguồn tham số

Không được để "controller gains" ở trạng thái mở — đó là rủi ro tiến độ. Nguồn đề xuất:

| Tham số | Nguồn | Giá trị ứng viên |
|---|---|---|
| $R$ (P–f droop) | M1, M2 | 5% |
| $T_c$ (lọc đo công suất) | M2 §VI-A | **0,0628 s** |
| $R_q$, $T_q$, $K_v$ | M1 + Supplemental Note | $K_v \in [4;5]$ |
| Vòng trong áp/dòng, LCL | M3 (Kenyon) | theo kiến trúc multiloop droop |
| $K_l$ (hệ số tổn thất biên) | M1 §II-A | 1,035–1,05 |
| $I_{\max}$ | Nhóm D [9] Baeckeland | *cần xác nhận* — thông thường 1,1–1,2 pu |
| Máy đồng bộ diesel + governor + AVR | thư viện SPS; họ mô hình governor diesel chuẩn | *cần chốt ở D2b* |

**D2b:** chọn họ mô hình governor/AVR cụ thể và ghi vào concept §25. Không để mở.

### D3 — Feeder IEEE 123 "matched-assumption"

§11 Level 4 yêu cầu một phiên bản cân bằng/thứ tự thuận để tách sai số bậc mô hình khỏi sai số mất cân bằng pha. Cần chốt cách dựng: dùng biến thể cân bằng đã công bố, hay tự suy ra bằng trung bình hoá tổng trở pha. Phải ghi rõ trong bài — nếu không, phản biện sẽ nói kết quả chính chạy trên một feeder không có thật.

### D4 — Nhân lực và song song hoá

Đường găng là EMT. Nếu có hai người, Task 8 (mô hình giải tích, Python thuần) chạy song song từ tuần 1 và rút ngắn tổng tiến độ khoảng một tháng. Nếu một người, phải chấp nhận thứ tự ở §3.

---

# Phần 1 — Phân loại mã nguồn hiện có

Repo hiện khoảng 700 KB mã Python. Sau khi bỏ MAPPO, phần lớn không còn dùng. Phân loại rõ để tránh vừa kéo theo nợ kỹ thuật vừa mất thứ còn giá trị.

### 1.1 Giữ nguyên — dùng lại trực tiếp

| Đường dẫn | Vai trò mới |
|---|---|
| `src/env/IEEE123bus.py` (65 KB) | Dựng feeder. Nền cho D3 và cho xuất mạng sang EMT. |
| `src/opt/tie_switch_reconfig.py` (27 KB) | **Sinh 47 cấu hình khả thi** — kết quả §9 của concept đến từ đây. Tài sản lõi. |
| `src/opt/l0_reconfig.py`, `src/layer0_dso/reconfiguration.py` | Kiểm tra radial/connectivity. |
| `der_placement_coop.py` (61 KB), `artifacts/placement/` | Bố trí GFM 2/5/6 — trục A của §18. |
| `data/`, `tie_switch.csv` | Dữ liệu lưới. |
| `src/eval/figures_style.py` | Định dạng hình cho bản thảo. |

### 1.2 Viết lại — giữ ý tưởng, bỏ mã

| Đường dẫn | Lý do |
|---|---|
| `src/env/freq_dynamics_lti.py` (35 KB) | Chứa **cả bốn lỗi** đã phát hiện: trộn ba base per-unit, cache Φ theo `mean(\|K\|)`, cấu trúc VSG bậc hai ép lên đơn vị droop, `_tau_c` khai báo mà không dùng. **Không vá — viết mới** thành `src/analytical/hybrid_sg_gfm.py` theo M2. Nhưng **giữ lại** phần dựng Jacobian và Kron reduction (`_build_jacobian`, `_kron_reduce`) — đó là mã đúng và tái dùng được, chỉ cần sửa base. |
| `src/eval/eval_ffr_topology.py` (100 KB) | Khung đánh giá theo topology dùng lại được về mặt cấu trúc, nhưng mọi chỉ tiêu phải đổi sang bộ mới ($\kappa$, $P_{\text{head}}^{\min}$, $\mu_I$, $\mu_P$). |

### 1.3 Lưu trữ — chuyển sang `archive/`, không xoá

| Đường dẫn | Kích thước |
|---|---|
| `src/rl/train_am_mappo.py` | 86 KB |
| `src/baselines/` (gcnn_ppo, matd3, mlp_mappo, train_*) | ~117 KB |
| `src/layer2_control/` (mappo_policy, graph_sage/gat/mlp encoder, actor_critic) | ~36 KB |
| `src/env/microgrid_env_dual.py`, `microgrid_env.py` | ~133 KB |
| `experiments/train_mappo.py`, `run_asha_*`, `run_ablation.py`, `run_multi_seed.py` | — |
| `experiments/lyapunov_certificate.py` | — |
| `configs/training_config_*.yaml`, `configs/seeds.yaml` | — |

Lý do lưu chứ không xoá: bài follow-on ở §33 của concept v2 (safe-MARL giữ quỹ đạo trong $\Omega_{\text{sec}}$) sẽ cần lại chúng, và `gat_encoder.py` cần cho phương án surrogate ở §19.4 nếu được kích hoạt.

### 1.4 Xoá trước khi công bố mã

| Đường dẫn | Lý do |
|---|---|
| `artifacts/_lyap_vsg.txt` | **Dương tính giả**: in "CLF FOUND … QED" với `eta = -0.0055` (âm) và `status=optimal_inaccurate`. Nếu phản biện mở repo và thấy tệp này, uy tín toàn bộ phần chứng chỉ mất. |
| `configs/training_config_baseline.yaml` / `_method.yaml` | `entropy_coef: 0.0` cho baseline vs `0.01` cho method; không trainer nào đọc chúng, nhưng phản biện mở repo sẽ kết luận ngay là gian lận tuning. |

---

# Phần 2 — Kiến trúc thư mục mới

```
src/
  analytical/          # Level 1 — mô hình sàng lọc
    network.py           # B_r(G), B_L(G) — tái dùng Kron từ freq_dynamics_lti
    hybrid_sg_gfm.py     # M2: tập chỉ số S (3 trạng thái) / I (2 trạng thái)
    voltage_live.py      # M1: mô hình LIVE bậc nhất
    estimators.py        # Level 2 — Î^ana = |S|/|V|, μ̂_I^ana, μ̂_P^ana
  emt/
    export_network.py    # IEEE123 (pandapower) → định dạng nền tảng EMT
    scenario_builder.py  # sinh tệp ca chạy
    runner.py            # gọi EMT, thu kết quả, ghi artifact
    metrics.py           # f_nadir, RoCoF (theo §13), V_min, I_peak, μ_I^EMT
  campaign/
    sampling.py          # lấy mẫu thích nghi bám biên
    boundary.py          # P_head^min, κ, P_excess
  grid/                  # giữ nguyên: IEEE123bus, tie_switch_reconfig, placement
archive/                 # toàn bộ mục 1.3
emt_models/              # mô hình nền tảng EMT (Simulink/PSCAD)
artifacts/
  T01_agfm/  T02_soc/  T03_device/  ...   # một thư mục / một task, xem §5
```

---

# Phần 3 — Kế hoạch theo giai đoạn

Ánh xạ 14 nhiệm vụ ở §26 của concept vào 6 phase có cổng.

### P0 — Kiểm tra bàn giấy *(Task 1–2; không cần EMT; ~1 tuần)*

Hai nhiệm vụ này **không phụ thuộc bất cứ thứ gì** và de-risk hai mục của concept. Làm ngay tuần này.

| Task | Nội dung | Đầu ra |
|---|---|---|
| **T1** | Đối chiếu mã `Ā_GFM` với định nghĩa chuẩn ở §8 concept ($w_k$ theo tải, $\pi_g$ theo headroom, không thứ nguyên). Tính lại bảng 2/5/6 GFM và các % (+121,2% / +16,5%). | `artifacts/T01_agfm/agfm_recomputed.csv` + ghi chú sai lệch so với số cũ |
| **T2** | Sweep độ nhạy SoC ở headroom, nhiễu, topology, bố trí GFM cố định. Quyết định SoC thuộc $\Omega_E$ hay có ảnh hưởng nhanh. | `artifacts/T02_soc/soc_sweep.csv` + kết luận một dòng đưa vào concept §25 |

**Cổng G0:** $\bar A_{\text{GFM}}$ khớp định nghĩa chuẩn; câu hỏi SoC đã đóng.

> ⚠️ T2 có một vướng mắc: sweep SoC cần một mô hình động — nhưng EMT chưa có và mô hình giải tích không có SoC. **Giải pháp:** chạy T2 bằng quan hệ derating $P_{\max}(\text{SoC})$ đưa vào $P_{\text{head}}$, rồi kiểm tra xem $f_{\text{nadir}}$ giải tích có thay đổi qua $P_{\text{head}}$ hay không. Nếu đó là kênh duy nhất thì kết luận "SoC ∈ $\Omega_E$" là đúng và rẻ. Xác nhận lại bằng EMT ở P1.

---

### P1 — Thiết bị đơn EMT + kiểm chứng bộ ước lượng *(Task 3–6; đường găng; ~4–6 tuần)*

Đây là phase quyết định. Nếu nó trượt, cả dự án trượt.

| Task | Nội dung |
|---|---|
| **T3** | Dựng và kiểm chứng **một** GFM-BESS multiloop droop: BESS/DC → DC link → VSC (mô hình trung bình) → LCL → mạng, kèm lọc P/Q, P–f droop, Q–V droop, PI áp/dòng, giới hạn dòng, anti-windup. |
| **T4** | **Kiểm chứng bộ ước lượng dòng giải tích** $\hat I^{\text{ana}}=\lvert S\rvert/\lvert V\rvert$ so với EMT, trên thiết bị đơn và một mạng nhiều GFM nhỏ. |
| **T5** | Tái lập một đáp ứng tần số/điện áp giải tích-vs-EMT ở **chế độ không ràng buộc** (dòng và headroom xa giới hạn). |
| **T6** | **Kích hoạt giới hạn dòng** và xác nhận đi vào chế độ ràng buộc converter. |

**Tiêu chí nghiệm thu — định lượng:**

| Kiểm tra | Ngưỡng đề xuất |
|---|---|
| T3 — độ dốc P–f droop | sai lệch ≤ 2% so với $R$ đặt |
| T3 — hằng số thời gian đáp ứng công suất | nhất quán với $T_c$ trong ±10% |
| T3 — điện áp DC-link khi bước tải | trong ±5% danh định |
| **T4 — bộ ước lượng dòng** | **p95 của $\lvert\hat\mu_I^{\text{ana}}-\mu_I^{\text{EMT}}\rvert \le 0{,}05$ pu với $\mu_I^{\text{EMT}}\le 0{,}9$** |
| T5 — nadir | $\lvert\Delta f_{\text{nadir}}\rvert \le 0{,}01$ Hz |
| T5 — RoCoF | $\lvert\Delta \text{RoCoF}\rvert \le 0{,}05$ Hz/s |
| T6 — vào ràng buộc | $\mu_I^{\text{EMT}}$ chạm 1,0 và limiter hoạt động quan sát được trên $i_d,i_q$ |

*Cơ sở ngưỡng T5:* Bảng II của M2 (39-bus, 7 GFM + 3 SG vs PSCAD) cho sai số trung bình nadir 0,0013 Hz và RoCoF 0,0315 Hz/s. Ngưỡng trên nới khoảng 3–8 lần, hợp lý cho feeder phân phối. **Nếu T5 không đạt ngay cả ở chế độ không ràng buộc thì vấn đề nằm ở cài đặt, không phải ở giới hạn converter** — phải dừng và truy nguyên, không đi tiếp.

**Cổng G1:** T3–T6 đạt toàn bộ ngưỡng. Đặc biệt **T4 là cổng cứng**: nếu bộ ước lượng dòng không đạt, xem R2 ở §6.

---

### P2 — Tầng giải tích *(Task 8; chạy song song với P1; ~1,5 tuần)*

| Task | Nội dung |
|---|---|
| **T8** | Cài mô hình giải tích hỗn hợp SG/GFM theo công thức tập chỉ số của M2. SG: 3 trạng thái $(\Delta\delta,\Delta\omega,\Delta P_M)$. GFM droop: 2 trạng thái. Mạng qua $B_r(G), B_L(G)$. Bổ sung LIVE cho điện áp. |

**Đây là Python thuần, không phụ thuộc EMT — phải khởi động từ tuần 1.** Nếu để tuần tự sau P1 thì lãng phí khoảng một tháng.

**Ba việc bắt buộc khi viết lại** (tránh lặp lỗi cũ):
1. $\alpha_i = S_B/S_i$ tường minh cho **mọi** thiết bị.
2. Một base công suất duy nhất cho toàn hệ; khẳng định bằng assertion khi khởi tạo.
3. $T_c$ là tham số **độc lập**, không suy ra từ $H$ và $K$.

**Cổng G2:** mô hình chạy được trên $G_0$ với 2/5/6 GFM và một SG; trị riêng nằm trong dải hợp lý về vật lý (không còn mode 37–345 Hz với $\zeta\approx 0{,}002$); kiểm tra base bằng một ca có nghiệm giải tích đã biết.

---

### P3 — Máy diesel và chuyển tiếp *(Task 7, 9–11; ~3–4 tuần)*

| Task | Nội dung |
|---|---|
| **T7** | Dựng phân hệ diesel EMT: máy đồng bộ + governor/prime mover + AVR/exciter + logic máy cắt tại $t_{\text{off}}$. |
| **T9** | Kiểm chứng ca vận hành hỗn hợp SG/GFM **trước** khi thử ngắt diesel. |
| **T10** | Diesel-off có kiểm soát, $P_{DG}(t_{\text{off}}^-)\approx 0$. |
| **T11** | Trip diesel mang tải, quét vài mức $P_{DG}(t_{\text{off}}^-)$. |

**Cổng G3:** trạng thái hỗn hợp ổn định trước $t_{\text{off}}$; diesel-off có kiểm soát cho quá độ bị chặn; trip mang tải tạo ra một quá độ nặng hơn có thể phân biệt được. Đây cũng là dữ liệu cho Figure F.

---

### P4 — Benchmark thời gian chạy và thiết kế campaign *(Task 12–13; ~1 tuần)*

| Task | Nội dung |
|---|---|
| **T12** | Đo thời gian tường trên phần cứng thật, ≥10–20 lần chạy đại diện cho: thiết bị đơn, mạng nhiều GFM nhỏ, IEEE 123 matched-assumption, IEEE 123 ba pha không cân bằng, một ca chuyển tiếp SG+GFM. Ghi $t_{50}, t_{95}$, chân trời mô phỏng, $\Delta t$, solver, CPU/lõi. |
| **T13** | Chạy tập con cân bằng-vs-không-cân-bằng để sai số mô hình không bị lẫn với sai số mất cân bằng pha. |

**Cổng G4:** ngân sách campaign được viết lại **hoàn toàn từ số đo**. Quyết định quy mô tầng không cân bằng. Quyết định surrogate theo §19.4.

*Tham chiếu để đối chiếu:* Bảng I của M2 cho hệ 9 nút / 3 thiết bị: PSCAD 42,00 s cho 20 s mô phỏng → ~2,1 s tường / 1 s mô phỏng. IEEE 123 với ~10 converter và một máy đồng bộ, ba pha không cân bằng, có thể nặng hơn nhiều lần. Nếu $t_{50}$ vượt 10 phút/lần chạy, kích hoạt phương án descope ở §6-R1.

---

### P5 — Campaign biên *(Task 14; ~4–6 tuần)*

Bốn trục theo §18 của concept:

| Trục | Nội dung | Cố định |
|---|---|---|
| A | 2 → 5 → 6 GFM | $G_0$ |
| B | $G_0,\ldots,G_{46}$ | 6 GFM |
| C | Quét $\Delta P$ | — |
| D | Quét $P_{DG}(t_{\text{off}}^-)$ | — |

Quy trình mỗi điểm vận hành:

```
sàng lọc giải tích  →  cổng μ̂_I^ana, μ̂_P^ana  →  chọn ca biên  →  EMT  →  đo μ_I^EMT, ε_f, ε_V
                                                                           ↓
                                                    P_head^min → κ = P_head^min/|ΔP| → P_excess
```

**Cổng G5:** dựng được biên đủ dày để vẽ Figure B, D, E; sai số nội suy trong dung sai đặt trước.

---

### P6 — Viết *(~4 tuần)*

Theo §22 của concept. Chỉ bắt đầu khi G5 đạt. Thứ tự viết: Results → Methods → Introduction → Conclusion → Abstract.

---

# Phần 4 — Bảng tổng hợp cổng

| Cổng | Sau | Điều kiện qua | Nếu trượt |
|---|---|---|---|
| **G0** | P0 | $\bar A_{\text{GFM}}$ khớp §8; SoC đã phân loại | Sửa định nghĩa/mã, không đi tiếp |
| **G1** | P1 | T3–T6 đạt ngưỡng §3; **T4 là cổng cứng** | Xem R2 |
| **G2** | P2 | Base nhất quán; phổ trị riêng hợp lý | Truy nguyên base trước khi dùng |
| **G3** | P3 | Trạng thái hỗn hợp ổn định; hai ca diesel-off phân biệt được | Xem lại tham số SG/governor |
| **G4** | P4 | Ngân sách từ số đo, không từ giả định | Descope theo R1 |
| **G5** | P5 | Biên đủ dày cho Figure B/D/E | Kích hoạt surrogate theo §19.4 |

---

# Phần 5 — Lược đồ artifact và truy vết

Thực thi N-2. Mỗi task ghi vào `artifacts/T<NN>_<slug>/`:

```
artifacts/T05_ana_vs_emt/
  manifest.json      # git commit, ngày, phiên bản nền tảng EMT, hash cấu hình
  config.yaml        # toàn bộ tham số đầu vào
  raw/               # đầu ra EMT thô
  metrics.csv        # một hàng / một lần chạy: các trường ở dưới
  figures/
  README.md          # ca này trả lời câu hỏi nào, kết luận một dòng
```

**Trường bắt buộc trong `metrics.csv`:**

```
run_id, config_id(G), gfm_deployment, event_type, dP_mw, P_dg_pre,
P_head_mw, soc, mu_I_ana, mu_I_emt, mu_P_ana,
f_nadir_ana, f_nadir_emt, rocof_ana, rocof_emt, v_min_emt, i_peak_pu,
eps_f, eps_V, secure_flag, wallclock_s, solver, dt
```

Một tệp duy nhất này sinh ra được Figure B, C, D, E. Mọi con số trong bản thảo phải trỏ về `run_id`.

---

# Phần 6 — Sổ rủi ro

| ID | Rủi ro | Xác suất | Tác động | Giảm thiểu / tiêu chí dừng |
|---|---|---|---|---|
| **R1** | **Dựng mô hình EMT vượt tiến độ** — rủi ro lớn nhất | Cao | Chặn toàn bộ | Mô hình trung bình (D1b); khối thư viện `SM Control` của Simscape Electrical cho AVR (SM DC4C), governor GGOV1 dựng từ sơ đồ chuẩn — xem D2b đã chốt; thiết bị đơn trước. **Tiêu chí dừng: nếu T3 chưa xong sau 6 tuần, descope xuống feeder IEEE 13 hoặc 34 nút** và ghi rõ trong bài. Kết quả về tính khả dụng mô hình vẫn công bố được trên hệ nhỏ hơn. |
| **R2** *(sửa lần 2, 2026-09-03, sau T00-rescale)* | **Bộ ước lượng dòng giải tích không đạt ngưỡng T4** | Trung bình | Mất chức năng *sàng lọc* của C1 | **Phương án lùi tồn tại — nhưng chỉ sau khi đội GFM được đưa về cỡ thật.** Bản sửa ngày 02-09 kết luận $\hat\mu_P^{\text{ana}}$ là cổng rỗng vì T2 (trên hồ sơ v3) đo $\Delta P_{\text{critical}} = 17{,}25$ MW trên feeder 3,49 MW. **Kết luận đó sai vì tiền đề sai**: $P_{\text{head}}$ là *biến quyết định* của campaign chứ không phải hằng số của bài toán. Sau khi rescale đội GFM về 1,25 lần tải đỉnh (`artifacts/T00_rescale/`), T2 chạy lại cho $\Delta P_{\text{critical}} = 3{,}272$ MW, $\kappa$ cắt 1 tại $\Delta P = 3{,}414$ MW, và $\mu_P = 0{,}92$ ngay tại $\Delta P = 3$ MW — **biên nằm trong dải nhiễu khả tín**, nên $\hat\mu_P^{\text{ana}}$ *có* phân biệt được ca nào cần EMT và dùng làm cổng sàng lọc dự phòng được. Thứ tự giảm thiểu: (a) dồn ngân sách de-risk vào T4 sớm, không để cuối P1; (b) nếu T4 trượt, dùng $\hat\mu_P^{\text{ana}}$ làm cổng sàng lọc và phát biểu bao khả dụng bằng $\mu_I^{\text{EMT}}$ hậu nghiệm (Outcome B ở §15) — **có mất độ chặt, không mất chức năng**; (c) chỉ descope C1 xuống mô tả hậu nghiệm thuần tuý nếu cả (b) cũng vượt ngân sách. **Cảnh báo kèm theo:** ở cỡ v4, ràng buộc *tần số* siết trước ràng buộc headroom **4,5 lần** ($\Delta f_{ss} \le 0{,}5$ Hz cho $\Delta P \le 0{,}727$ MW, so với 3,272 MW của headroom, với $R=5\%$). Cổng sàng lọc thực dụng nhất do đó có thể là $\Delta f_{ss}$ chứ không phải $\mu_P$ — cần T3/T4 xác nhận, và cần chốt lại $R$ cho đội thiết bị đã rescale. Bằng chứng: `artifacts/T02_soc_v4/`, `artifacts/T00_rescale/`. |
| **R3** | $\kappa - 1 \approx 0$ — không có dự trữ vượt | Trung bình | Làm yếu C2 | Vẫn báo cáo biến thiên $\kappa(G)$ và $\kappa$ theo bố trí GFM. Sự kiện diesel-off nhiều khả năng cho $\kappa>1$ ngay cả khi nhiễu ổn định không cho. Báo cáo trung thực kết quả null. |
| **R4** | 47 cấu hình quá compact để dịch chuyển $\kappa$ | Trung bình | Làm yếu C3 | Đã lường trước ở §9 concept ($d_J \le 0{,}049$). C3 vốn là contribution hỗ trợ. Báo cáo null kèm $\bar A_{\text{GFM}}$ làm giải thích. |
| **R5** | Feeder cân bằng bị chê là nhân tạo | Trung bình | Tấn công vào kết quả chính | Tầng không cân bằng ở §11 Level 4 tồn tại chính vì điều này — **không được descope nó đi** dù ngân sách eo hẹp. Tối thiểu 5–8 ca biên đại diện. |
| **R6** | Jiang et al. [2] được bình duyệt và đăng trước | Thấp–TB | Thu hẹp C2 | Theo dõi arXiv 2512.01814. C1 không bị ảnh hưởng — đó là lý do đưa C1 lên dẫn đầu. |
| **R8** *(mới 2026-09-03)* | **Biên tìm được trên ANDES là biên của một quyết định chưa ai ra** | Cao | Mọi con số của C1/C2 phải tính lại | Biên `ΔP_max` đo được (0,534 MW) do `μ_I` của đơn vị nhỏ nhất đặt, và **96,9% dòng của đơn vị đó là phản kháng** — lượng phản kháng ấy là nghiệm mà phân bố công suất trả về khi sáu GFM cùng đặt $V = 1{,}00$ pu, không phải một chiến lược điều độ. Độ nhạy $\mathrm{d}Q/\mathrm{d}V$ đo được $1{,}3\cdot10^{3}$ pu/pu sau khi đã đưa mỗi converter ra sau nhánh ghép nối ($3{,}7\cdot10^{4}$ trước đó), nên nghiệm này cực nhạy với setpoint. **Giảm thiểu: chốt chiến lược điều độ $Q$ cho đội GFM trước khi bất kỳ biên nào lên bản thảo** — setpoint theo định mức, hay Q–V droop có điểm làm việc xác định. Đây là toạ độ mà tài liệu về ca 100% GFM báo là nguyên nhân đổ vỡ, nên nó là quyết định vật lý chứ không phải chi tiết cài đặt. Bằng chứng: `artifacts/T20_andes_bisect/`. |
| **R7** | Tham số SG/governor/AVR không có nguồn chắc | Trung bình | Trễ P3 | Đóng ở D2b **trước** khi bắt đầu P3, không phải trong lúc làm. |

---

# Phần 7 — Đường găng và tiến độ

```
Tuần  1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16
      ├─P0─┤
      ├────── P2 (T8, Python, song song) ──────┤
           ├──────────── P1 (T3→T4→T5→T6) ────────────┤        ← ĐƯỜNG GĂNG
                                                 ├──── P3 (T7,T9,T10,T11) ────┤
                                                                    ├─P4─┤
                                                                         ├──── P5 ────
```

**Đường găng: T3 → T4 → T5 → T6 → T7 → T9 → T10/T11 → T12 → T14.**

T1, T2, T8 **không** nằm trên đường găng — khởi động ngay để không lãng phí thời gian chờ EMT.

Ước lượng tổng: **~5–6 tháng cho một người**, khoảng **4 tháng nếu hai người** (một làm EMT, một làm tầng giải tích + campaign). Cộng ~4 tuần viết.

---

# Phần 8 — Việc của tuần này

Bốn việc, không việc nào cần EMT:

1. **Chốt D1** (nền tảng EMT) và **D2b** (họ mô hình SG/governor/AVR). Không có hai quyết định này thì P1 và P3 không khởi động được.
2. **T1** — đối chiếu mã `Ā_GFM` với §8, tính lại bảng 2/5/6 GFM.
3. **T2** — sweep SoC theo cách ở §3-P0 (qua đường derating $P_{\max}(\text{SoC})\to P_{\text{head}}$).
4. **Dọn repo** theo Phần 1: tạo `archive/`, di chuyển mục 1.3, **xoá** `artifacts/_lyap_vsg.txt` và hai `configs/training_config_*.yaml`.

Và một việc song song nếu có người thứ hai: **khởi động T8**.

## Đã làm, 2026-09-02 và 03

| # | Trạng thái | Hiện vật |
|---|---|---|
| 1 | ✅ D1, D1b, D2b chốt | concept §25 |
| 2 | ✅ T1 | `artifacts/T01_agfm/`, `artifacts/T01_agfm_v4/` |
| 3 | ✅ T2 | `artifacts/T02_soc/`, `artifacts/T02_soc_v4/` |
| 4 | ✅ Dọn repo (105 tệp, 4 đợt, bao đóng import tới điểm bất động) | `archive/README.md` |

Ba việc **ngoài kế hoạch tuần**, phát sinh từ chính kết quả trên:

**T00 — rescale đội GFM** về 1,25 lần tải đỉnh. Bắt buộc: ở cỡ v3, vùng khan hiếm headroom
là bất khả đạt, campaign sẽ trả về "an toàn ở mọi nơi". Xem `artifacts/T00_rescale/`.

**D5 — thêm tầng workhorse pha (ANDES 2.0)** giữa tầng giải tích và EMT. Không thay EMT;
nó chỉ *tìm* biên để EMT xác nhận. Bốn tính chất của nền tảng phải đo mới biết (gain vòng
dòng của `REGF1` làm đội thiết bị bất ổn tuyến tính; dQ/dV ≈ 3,7·10⁴; droop quy về system
base; tích phân giới hạn P làm cứng droop 75%) — chi tiết ở concept §25.

**T20 — vòng bisection** trên nền tảng đó, thay cho quét lưới. `experiments/t20_andes_bisect.py`,
`src/phasor/`, `src/campaign/boundary.py`. Đây là bài kiểm tra sống-chết của §3-P5 và
**nó đạt**: bisection kẹp được biên hữu hạn trong 13 lần chạy, đơn điệu, ~12 s một lần chạy.

---

*Tài liệu này do trợ lý nghiên cứu AI soạn dựa trên concept v3.1, kết quả kiểm chứng mã nguồn ngày 2026-09-02, và đối chiếu literature trên Consensus. Mọi tham số, ngưỡng nghiệm thu và diễn giải vật lý cần được nhóm nghiên cứu thẩm định độc lập trước khi thực thi. Các ngưỡng ở §3 là đề xuất khởi điểm, không phải giá trị đã hiệu chuẩn.*
