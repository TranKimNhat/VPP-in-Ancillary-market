# T32 — bản đồ trị riêng 2-D: có tồn tại cấu hình vừa đúng đặc tả vừa ổn định?

**Câu hỏi (nối tiếp T31):** T31 chốt cơ chế (`KPplim`, không phải vòng trong) nhưng để lại
một câu quyết định giá trị của cả kết quả: **`KPplim` ≤ 0,05 không chạy được** ở cách chỉnh
vòng trong hiện tại. `solve()` báo *"time step reduced to zero"* cho cả sụp đổ vật lý lẫn bất
khả thi số học, nên miền thời gian không phân biệt được.

Nếu vùng đó **bất ổn vật lý**, đường cong nhạy $\kappa_{os}(K_{Pplim})$ là kết quả về mô hình
GFM chuẩn hoá. Nếu chỉ hỏng vì số học, reviewer trả lời *"các anh quét ngoài dải hợp lệ, hỏng
là đương nhiên"* và đường cong vô giá trị.

> **Lưu ý đọc tài liệu này.** T31 dán nhãn `KPplim` ≤ 0,05 là "dải đặc tả REGFM_A1". Nhãn đó
> **sai thứ nguyên** và được sửa ở §1 dưới đây: cửa sổ đúng đặc tả là `KPplim` ∈ [0,1 – 1,0].
> Lưới trị riêng gốc (§1) vẫn giữ nguyên vì nó phủ cả hai vùng; §1b thêm các ô còn thiếu.

Chỉ tính trị riêng: power flow → `TDS.init()` → ma trận trạng thái rút gọn tại điểm cân bằng
trước sự cố. **Không tích phân miền thời gian**, nên nhiễu loạn không liên quan và lưới không
có trục $\Delta P$.

**Kiểm công cụ (bắt buộc, đã đạt):** gains vòng trong **gốc** của REGF1 (0,5; 20) phải trả về
bất ổn. Kết quả: **đúng 10 mode** $\mathrm{Re}\lambda>0$, mode xấu nhất **136,9 Hz** —
`build_case.py:154` ghi *"ten eigenvalues with Re > 0 at 92–260 Hz"*. Linearisation đúng.

---

## 1. Bản đồ — $\max\mathrm{Re}\,\lambda$

108 mode/điểm; 13 mode có $|\mathrm{Re}|<10^{-6}$ là **góc tham chiếu tự do** (đảo không có
nút slack tần số ⇒ mọi `delta` xác định sai khác một phép quay chung) nên loại khỏi phán
quyết, không tính là bất ổn cận biên. Cột `max_re_all` trong CSV giữ giá trị chưa loại.

| (KPi, KIi) | 0,005 | 0,01 | 0,05 | 0,5 | 1 | 2 | 5 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (0,50; 20) ← REGF1 gốc | 34,67 | 34,67 | 34,66 | 34,55 | 34,42 | 34,16 | 33,40 | 32,23 |
| (0,35; 12) | 26,51 | 26,50 | 26,38 | 25,12 | 23,78 | 21,28 | 14,98 | 7,53 |
| **(0,20; 5,0) ← đang ship** | **9,45** | **9,40** | **8,95** | **4,15** | −0,75 | −2,05 | −1,82 | −1,61 |
| (0,10; 3,0) | **−2,151** | **−2,151** | **−2,151** | −2,142 | −2,125 | −2,065 | −1,827 | −1,612 |

### ⚠️ Nhãn conformance của lưới gốc sai thứ nguyên — đã sửa

Lưới trên được dựng với giả định `KPplim` so trực tiếp được với `kppmax`. **Sai.** REGFM_A1
cộng nhánh overload `kppmax + kipmax/s` **thẳng vào ω** (cùng điểm với `mp`, và đặc tả cho hai
tham số này *cùng* dải 0,005–0,05), còn REGF1 cộng vào **công suất** rồi mới nhân `w0·wdrp`.
Đại lượng so được:

$$k_{ppmax}^{eff}=m_p\cdot K_{Pplim}=R\cdot K_{Pplim}=0{,}05\,K_{Pplim}$$

| `KPplim` | 0,005 | 0,05 | **0,1** | **0,2** | **0,5** | **1,0** | 2,0 | **5,0** |
|---|---|---|---|---|---|---|---|---|
| $k_{ppmax}^{eff}$ | 0,00025 | 0,0025 | 0,005 | **0,010** | 0,025 | 0,050 | 0,100 | **0,250** |
| vs 0,005–0,05 | 20× dưới | 2× dưới | đáy ✅ | **ví dụ** ✅ | ✅ | đỉnh ✅ | 2× trên | **5× trên** |

**Cửa sổ đúng đặc tả: `KPplim` ∈ [0,1 – 1,0]; ví dụ đặc tả = 0,2; đang ship 5,0, lệch 5×**
(không phải 100×). Ba cột đầu của bảng trên nằm **dưới** dải, không phải trong nó.

## 1b. Bản đồ trên cửa sổ đặc tả thật

| (KPi, KIi) | 0,1 | **0,2** | 0,5 | 1,0 | 2,0 | 5,0 |
|---|---:|---:|---:|---:|---:|---:|
| **(0,20; 5,0) đang ship** | +8,40 ✗ | +7,31 ✗ | +4,15 ✗ | −0,75 ✅ ζ=0,0020 | −2,05 ✅ ζ=0,0227 | −1,82 ✅ ζ=0,0532 |
| (0,10; 3,0) | −2,150 ✅ ζ=0,0517 | **−2,149 ✅ ζ=0,0679** | −2,142 ✅ ζ=0,1099 | −2,125 ✅ ζ=0,1559 | −2,065 ✅ ζ=0,1874 | −1,827 ✅ ζ=0,1779 |

Ở vòng trong hiện tại, chỉ **đỉnh** cửa sổ đặc tả (1,0) ổn định, và cận biên (ζ = 0,0020). Ở
(0,10; 3,0) **toàn bộ cửa sổ ổn định, và damping tốt hơn cấu hình đang ship** (ζ 0,052–0,156
so với 0,0532).

## 2. Ba câu trả lời

**① Bất ổn vật lý, không phải bất khả số học.** Tại cấu hình đang ship (0,20; 5,0), dải đặc
tả cho $\max\mathrm{Re}\lambda$ = **+9,45 / +9,40 / +8,95** với **6 mode bất ổn**. Đây là mất
ổn định tuyến tính thật. Sụp đổ của T31 được giải thích, không phải lỗi solver.

**② CÓ tồn tại vùng đặc tả ổn định.** Ở vòng trong (0,10; 3,0) **toàn bộ cửa sổ đặc tả**
`KPplim` ∈ [0,1 – 1,0] ổn định với $\max\mathrm{Re}\lambda \approx -2{,}15$ và
$\zeta_{\min}$ = 0,052–0,156 — **tốt hơn cấu hình đang ship** (0,0532). Ở vòng trong hiện
tại (0,20; 5,0) chỉ đỉnh cửa sổ (1,0) ổn định, và cận biên ($\zeta$ = 0,0020).

**Hai bản vá của ta đang đánh đổi với nhau**, đúng như nghi ngờ. Ta chưa từng đổi hai thứ
cùng lúc nên chưa từng thấy.

**③ Ngưỡng overshoot không nằm ở mode tần số thấp.** $\zeta$ của mode $<5$ Hz gần như không
đổi qua `KPplim` (0,64 → 0,69 trên hàng đang ship). Mode siết $\zeta_{\min}$ toàn cục nằm ở
**77 Hz** — miền vòng trong. Nên ngưỡng $\kappa_{os}$ giữa `KPplim` = 2 và 5 là tương tác
vòng trong × đường droop, và **hàm truyền hở mạch của T30 không thể nói được** vì nó không
mang vòng trong. Giới hạn đã ghi ở `regf1_droop.tf()`.

---

## 3. `KPv` — không phải sai lệch, mà là **không so được**

`regfm_a1_mapping.md` §3 từng gắn cờ "`KPv` = 3,0 vs đặc tả 0–0,01 (300×), cần kiểm". Cờ đó
**ill-posed**:

| | REGFM_A1 `kpv` | REGF1 `KPv` |
|---|---|---|
| đầu vào | sai lệch điện áp | sai lệch điện áp |
| **đầu ra** | **`E_droop`** (điện áp trong, clamp `[Emin,Emax]`) | **`Idref`** (tham chiếu dòng) |
| thứ nguyên | vô thứ nguyên (áp/áp) | **dẫn nạp** (dòng/áp) |

REGFM_A1 là biến thể **không có vòng dòng trong**; `kpv`/`kiv` tham số hoá một plant khác.
Không có phép quy đổi số học nào hợp lệ. Cờ này khép lại là **không so được**, không phải sai
lệch — khác hẳn `kppmax`, nơi phép quy đổi $m_p K_{Pplim}$ tồn tại và cho lệch 5× thật.

**Nhưng độ nhạy thì có thật và đáng ghi** (quét tại `KPplim`=0,2, vòng trong (0,10; 3,0)):

| `KPv` | 0 | 0,01 | 0,1 | 0,5 | **1,0** | 2,0 | **3,0 (đang dùng)** | 5,0 |
|---|---|---|---|---|---|---|---|---|
| $\max\mathrm{Re}\lambda$ (KIv=10) | +11,46 | +11,34 | +10,22 | +5,15 | −0,23 | −1,44 | **−2,15** | −1,64 |
| $\zeta_{\min}$ | −0,594 | −0,587 | −0,522 | −0,246 | 0,056 | 0,094 | **0,068** | 0,029 |

`KPv` là tham số **quyết định ổn định**: dưới ~1,0 hệ bất ổn dữ dội. `KIv` gần như không ảnh
hưởng trong 5,86–15 khi `KPv` ≥ 2. Giá trị đang dùng (3,0; 10) nằm ở vùng tốt. Không cần đổi.

---

## 4. Xác minh miền thời gian — trị riêng ổn định ≠ tích phân chạy được

| # | `KPplim` | $k_{ppmax}^{eff}$ | vòng trong | $\max\mathrm{Re}$ | $\zeta_{\min}$ | $\kappa_{os}$ | RoCoF | $\mu_I$ | tái thẩm định |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| hiện tại | 5,0 | 0,250 ❌5× | (0,20; 5,0) | −1,82 | 0,0532 | **1,2304** | **1,9996** | 0,764 | — |
| **A** | 1,0 | 0,050 ✅đỉnh | (0,20; 5,0) | −0,75 | **0,0020** | 1,0046 | 1,5832 | 0,718 | **không cần** |
| **B** | **0,2** | **0,010 ✅ví dụ** | (0,10; 3,0) | −2,15 | **0,0679** | 1,0057 | 1,5685 | 0,728 | cần |
| **C** | 1,0 | 0,050 ✅đỉnh | (0,10; 3,0) | −2,12 | **0,1559** | 1,0046 | 1,5811 | 0,725 | cần |

Cả bốn hội tụ, settled, secure tại ΔP = 0,64 và 1,1793 MW. $\Delta f_{ss}$ = 0,8111 Hz **giống
hệt** ở cả bốn — `KPplim` không có trong DC gain.

Ba phương án conformant **không phân biệt được ở miền thời gian** (κ 1,005–1,006, RoCoF
1,568–1,583). Lựa chọn do biên trị riêng và chi phí tái thẩm định quyết định.

**Ngoại suy** $\Delta P_{\max}$ tại B: $1{,}1793 \times 2{,}0 / 1{,}5685 \approx \mathbf{1{,}50}$ MW
so với **1,1851 MW**, **+27%**. Ngoại suy, không phải bisect; ở mức đó nadir hoặc $\mu_I$ có
thể siết trước RoCoF.

---

## 5. Phán quyết theo tiêu chí đã chốt trước

Nhánh **"tồn tại ô conformant ổn định"** kích hoạt.

**Đề nghị ship: B — `KPplim` = 0,2 (giá trị ví dụ REGFM_A1), vòng trong (KPi, KIi) = (0,10; 3,0),
`KPv`/`KIv` giữ nguyên (3,0; 10).**

Lý do chọn B thay vì A hoặc C:

- **A** không cần tái thẩm định vòng trong, nhưng $\zeta_{\min}$ = 0,0020 — kém cấu hình đang
  ship 26 lần. Không đủ biên để chịu các trục robustness. **Loại.**
- **C** damping tốt nhất (0,156) nhưng nằm ở **mép** dải đặc tả. **Giữ làm phương án dự phòng**
  nếu tái thẩm định cho thấy B quá sát biên trên trục nào đó.
- **B** là **giá trị ví dụ của chính bản đặc tả** — vị trí mạnh nhất để bảo vệ — và
  $\zeta_{\min}$ = 0,0679 đã **tốt hơn cấu hình đang ship 28%**.

### Chi phí phải trả

1. **Tái thẩm định vòng trong là bắt buộc.** `build_case.py:152–166` ghi (0,20; 5,0) đã xác
   minh giữ $\mathrm{Re}\lambda\le0$ trên **mọi deployment (2/5/6 GFM), mọi headroom,
   xf ∈ [0,05; 0,20], R ∈ [0,02; 0,05], cả hai mô hình tải**. Chuyển sang (0,10; 3,0) **huỷ
   toàn bộ xác minh đó**. T32 chỉ phủ case danh định tại một điểm vận hành.
2. **Chạy lại T21–T26**, và kiểm lại $I_{dev}(\Delta P)$ / $I_{\max F}^{crit}$: $\mu_I$ đổi
   0,764 → 0,728 tại cùng ΔP nên khớp cũ dịch.
3. **Damping KHÔNG mất** — điểm này đảo so với bản đầu của tài liệu, vốn so ở `KPplim` = 0,01
   (dưới dải). Ở cửa sổ đặc tả, (0,10; 3,0) damping *tốt hơn* cấu hình đang ship.

---

`experiments/t32_eig_map.py` · `artifacts/T32_eig_map/eig_map.csv` (32 điểm)
