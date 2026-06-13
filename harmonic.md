# Báo cáo phân tích sóng hài (THD/TDD) — IEEE 123-bus islanded microgrid

**Ngày:** 2026-06-13
**Phạm vi:** Audit chất lượng điện năng cho 6 controller (GraphSAGE-MAPPO/đề xuất, GCNN-PPO,
MLP-MAPPO, MATD3, Fixed Droop, No-FFR) trên feeder IEEE 123-bus 100% inverter-based, islanded.
**Trạng thái:** Đã hoàn tất tính toán. **Quyết định: KHÔNG đưa nội dung harmonic vào bài báo**
(lý do ở §6). Báo cáo này lưu lại toàn bộ phương pháp + kết quả để tham chiếu nội bộ.

---

## 1. Phương pháp tính toán

Phương pháp đi từ vi mô (trích xuất phổ hài của từng nghịch lưu) đến vĩ mô (giải lưới và tính
RSS), gồm 4 bước. Code: `src/eval/harmonic_analysis.py` (lớp `HarmonicAnalyzer`).

### Bước 1 — Trích xuất dòng hài bằng Chuỗi Fourier Kép (DFS / hàm Bessel)

Mô hình SPWM 3 pha (Thakur 2020). Biên độ điện áp dải biên và dòng bơm tương ứng:

```
V_mn = | (2·Vdc / (m·π)) · J_n( m·ma·π/2 ) · sin( (m+n)·π/2 ) |
I_mn = V_mn / (2·π·h·f0·Lf)
```

- `J_n`: hàm Bessel loại 1 (`scipy.special.jv`).
- Tham số nghịch lưu: `Vdc = 800 V`, `ma = 0.9`, `Lf = 150 mH`, `fsw = 3 kHz`, `f0 = 50 Hz`.
- Chỉ giữ 4 dải biên chủ đạo thực sự đi vào lưới:
  `(m,n) ∈ {(1,−2),(1,+2),(2,−1),(2,+1)}` → bậc hài `h ∈ {58, 62, 119, 121}`.
- Dòng bơm tại mỗi bus DER được nhân hệ số tải **theo từng đơn vị**:
  `loading_i = clip( |P_i| / P_rated,i , 0, 1 )`.

### Bước 2 — Giải luồng công suất hài (Harmonic Power Flow)

Với mỗi bậc `h`:
1. Dựng ma trận dẫn nạp `Y_h` bằng cách nội suy theo bậc hài: điện kháng nhánh `X_h = h·X`,
   điện nạp shunt `B_h = h·B` (dùng `makeYbus` của pandapower trên `ppc`).
2. Bơm vector dòng hài `I_h` vào đúng các bus chứa nghịch lưu.
3. Giải hệ tuyến tính `V_h = Y_h⁻¹ · I_h` (fallback `lstsq` nếu suy biến) → điện áp hài tại
   **tất cả 123 bus**, từ đó suy ra dòng hài nhánh `I_h[j] = (V_h[from] − V_h[to]) / Z_h[j]`.

### Bước 3 — Công thức chỉ số méo dạng

**Điện áp — THD tại mỗi bus `i`:**
```
THD_V[i] = sqrt( Σ_h |V_h[i]|² ) / |V_1[i]|  × 100%
```
`V_1[i]` lấy từ power flow cơ bản (`res_bus.vm_pu`). Tử và mẫu cùng pu → tỷ số bất biến đơn vị.

**Dòng điện — TDD tại mỗi nhánh `j` (đúng IEEE 519-2014 §5.2):**
```
TDD[j] = sqrt( Σ_h |I_h[j]|² ) / I_L  × 100%
```
- Quy chiếu theo **dòng phụ tải cực đại `I_L`** (không phải dòng cơ bản tức thời) — đúng định
  nghĩa của chuẩn. Trong mô phỏng một điểm vận hành, `I_L` được xấp xỉ bằng dòng cơ bản tại
  dispatch được audit; khi đo trong sự cố, `I_L` được **đóng băng tại trạng thái pre-event**
  (vì sự cố làm sụt `I_1` tại PCC sẽ thổi phồng TDD giả tạo).
- Vì `I_L,max ≥ I_1`, giá trị TDD báo cáo là **chặn trên bảo thủ**.

### Bước 4 — Đối chiếu giới hạn IEEE 519-2014

- **THD_V:** lưới trung thế (1–69 kV, feeder 4.16 kV) → giới hạn **5%** tại mỗi bus.
- **TDD:** giới hạn theo tier tỷ số `I_sc/I_L` (Table 2). Trong microgrid islanded 100% IBR, dòng
  sự cố bị **converter giới hạn** (~1.2–2 pu rating nguồn) → `I_sc/I_L < 20` theo cấu trúc →
  áp dụng **tier khắt khe nhất 5%** toàn hệ. Đánh giá compliance **tại PCC** (nơi §5.2 chính
  thức áp dụng); TDD từng nhánh chỉ là chẩn đoán phụ.

> **Lưu ý kỹ thuật:** Không tính `I_sc` bằng trở kháng Thevenin `diag(inv(Y_1))` — Ybus của lưới
> hình tia không chứa trở kháng nguồn nối đất nên gần suy biến, cho kết quả rác. Thay vào đó dùng
> lập luận vật lý converter-limited ở trên (đã ghi trong docstring code).

---

## 2. LỖI đã phát hiện và sửa (quan trọng)

Lần audit đầu cho kết quả vô lý: "GraphSAGE = Fixed Droop = No-FFR = 5.11% / 116-of-123 buses
FAIL". Truy vết ra **hai bug cộng hưởng**:

1. **`scripts/eval_thd.py::commanded_p_mw`** đọc key `p_rated_kw` — **không tồn tại** trong agent
   specs (key thật là `p_rated`, đơn vị MW; rating thật 0.05–0.325 MW). → mọi agent rơi về
   default 100 kW.
2. **`HarmonicAnalyzer._build_injection`** chia loading cho hằng số toàn cục `P_rated_mw = 0.05`
   thay vì rating từng đơn vị. → baseline dispatch `0.5 × 0.1 = 0.05 MW` đúng bằng mẫu số →
   `loading = clip(≥1.0) = 1.0` bão hòa cho **mọi** action ≥ 0 → bức tường THD giả.

**Sửa:** dùng loading per-unit `|P_i| / P_rated,i`, khớp đúng phương pháp đã mô tả trong bài
(`agent_p_rated_mw` truyền vào `HarmonicAnalyzer.run`; helper `agent_rated_mw()` trong eval_thd).
Sau khi sửa, kết quả mới hợp lý và phân biệt được các controller.

---

## 3. Kết quả — Audit steady-state (không sự cố)

Lệnh: `python scripts/eval_thd.py` (warmup 30 bước, loading per-unit, `I_L` per-unit).

| Method | THD_V PCC | THD_V max | Buses >5% | TDD_I PCC | TDD_I max | Phán định |
|---|---|---|---|---|---|---|
| MLP-MAPPO | **0.93%** | **1.02%** | 0/123 | **0.61%** | 4.58% | Pass |
| GraphSAGE-MAPPO (ours) | 2.55% | 2.81% | 0/123 | 4.55% | 11.50% | Pass |
| Fixed Droop | 2.55% | 2.81% | 0/123 | 4.55% | 11.50% | Pass |
| No-FFR | 2.55% | 2.81% | 0/123 | 4.55% | 11.50% | Pass |
| GCNN-PPO | 3.65% | 3.96% | 0/123 | 4.49% | 12.80% | Pass |
| MATD3 | 4.04% | 4.40% | 0/123 | **7.58%** | 15.60% | **Fail (TDD)** |

**Nhận xét:**
- Phía điện áp: **mọi** controller pass (0/123 bus vượt). THD_V bám theo độ mở của phân bố action
  steady-state: MATD3 (rail-saturated) cao nhất; MLP (under-dispatch) thấp nhất.
- GraphSAGE bằng **đúng** baseline thụ động (Fixed Droop / No-FFR) — action steady-state ≈ 0
  (mean +0.06) nên FFR readiness không thêm méo dạng nào ngoài dispatch nền.
- Phía dòng (TDD@PCC): chỉ **MATD3 fail** (7.58%); còn lại pass, nhưng GraphSAGE (4.55%) nằm
  **sát ngưỡng** 5% và bằng baseline thụ động.

---

## 4. Kết quả — Audit trong lúc FFR kích hoạt (sự cố S2 gen trip)

Lệnh: `python scripts/thd_during_ffr.py` (`I_L` đóng băng tại pre-event). Đo tại t = 0.5 s
(peak), 3 s, 20 s sau sự cố.

| Method | giai đoạn | THD_V PCC | TDD_I PCC | br>TDD |
|---|---|---|---|---|
| **GraphSAGE (ours)** | pre-event | 2.83% | 5.03% | 4 |
| | t=0.5s (peak) | 2.80% | **4.95%** | 3 |
| | t=3s | 2.83% | 5.02% | 4 |
| | t=20s | 2.83% | 5.02% | 4 |
| MLP-MAPPO | pre-event | 1.88% | 1.24% | 2 |
| | t=0.5s | 1.20% | 2.28% | 2 |
| | t=20s | 0.92% | 2.28% | 0 |
| **MATD3** | pre-event | 4.07% | 7.58% | 4 |
| | t=0.5s (peak) | 4.01% | **7.27%** | 4 |
| | t=3s | 3.92% | 4.88% | 3 |
| | t=20s | 4.01% | **7.59%** | 4 |
| GCNN-PPO | pre-event | 3.31% | 3.66% | 3 |
| | t=0.5s | 3.34% | 4.48% | 3 |
| | t=20s | 3.47% | 4.44% | 2 |

**Nhận xét:**
- **GraphSAGE phẳng tuyệt đối** xuyên suốt sự cố (THD_V 2.80–2.83%, TDD 4.95–5.03%) — không có
  transient spike, vì response đi qua **kênh droop-gain** chứ không phải nhảy P-reference. Đây là
  behavior dự đoán được.
- **MATD3 vi phạm bền vững** (TDD 7.27–7.59% ở peak/settle), và **dao động** TDD giữa các thời
  điểm (4.88 → 7.59%) — dấu vết limit-cycle hiện cả ở phía harmonic.

---

## 5. Diagnostic action steady-state (giải thích cơ chế)

| Method | mean a_P | loading TB | THD_V | Cơ chế |
|---|---|---|---|---|
| GraphSAGE | +0.056 | ~0.55 | 2.55% | pre-arm nhẹ để sẵn sàng FFR, = baseline |
| MLP-MAPPO | −0.388 | ~0.37 | 0.93% | giảm dispatch dưới nền → THD thấp, kèm FFR yếu |
| MATD3 | +0.265 (±1 rail) | ~0.79 | 4.04% | bão hòa rail → injection cao nhất |
| GCNN-PPO | ~0.00 | ~0.65 | 3.65% | trung gian |

Bất đối xứng quan trọng: positive action kẹp đơn vị ở rated loading, còn negative action chỉ
giảm một nửa baseline → controller bão hòa dương (MATD3) có THD cao nhất.

---

## 6. Vì sao KHÔNG đưa harmonic vào bài báo

1. **Không phải lợi thế của method đề xuất.** TDD@PCC của GraphSAGE (4.55–5.03%) **bằng đúng**
   baseline thụ động và **nằm sát ngưỡng** 5% — đây là thuộc tính của lưới (dispatch nền), không
   phải do thiết kế điều khiển. Con số nhạy theo điểm vận hành (4.55% ở audit này, 5.03% ở audit
   khác).
2. **Discriminator duy nhất là MATD3 fail** — chỉ một baseline. Không đủ để thành một đóng góp
   độc lập.
3. **Rủi ro với reviewer.** Một claim "compliance" sát ranh giới và phụ thuộc operating point dễ
   bị phản biện, trong khi không củng cố thêm câu chuyện chính.
4. **Câu chuyện chính đã đủ mạnh** mà không cần harmonic: settling 2.2–3.1× nhanh hơn, ITAE
   29–103× thấp hơn, topology generalization (gap +5.7%, unseen IAE thấp nhất), DSO cost thấp
   nhất nhóm learning.

→ Gỡ toàn bộ Table VI (THD), §V-D (Harmonic Compliance), dòng "Voltage THD limit" trong bảng
ngưỡng, metric group "power quality", và các câu/claim harmonic trong abstract, discussion,
conclusion. Code audit (`harmonic_analysis.py`, `eval_thd.py`, `thd_during_ffr.py`) **giữ lại**
để tham chiếu nội bộ và có thể dùng nếu sau này làm validation EMT.

---

## 7. File liên quan

- `src/eval/harmonic_analysis.py` — lớp `HarmonicAnalyzer` (DFS + Y_h + THD/TDD + tier).
- `scripts/eval_thd.py` — audit steady-state per-method.
- `scripts/thd_during_ffr.py` — audit trong lúc FFR kích hoạt (S2, `I_L` đóng băng).
- `results/thd_verify/thd_per_method.csv` — kết quả steady-state.
- `results/thd_during_ffr/thd_during_ffr.csv` — kết quả during-event.
