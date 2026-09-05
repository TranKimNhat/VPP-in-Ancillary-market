# T31 — cơ chế của overshoot tần số: `KPplim`, không phải vòng trong

**Câu hỏi (nối tiếp T30):** $\kappa_{os}=(f_0-f_{nadir})/(f_0-f_{ss})=1{,}227$ là đại lượng
mọi mô hình GFM rút gọn dự đoán bằng **1,000** theo cấu trúc (Ducoin Eq. (33) là bậc một cho
hệ toàn IBR ⇒ đáp ứng bước đơn điệu). Hai ứng viên: (a) zero lead do `KPplim` đưa vào đường
droop, (b) vòng trong — giả thiết Ducoin phát biểu ngay trên Eq. (2).

Sweep `KPplim` ∈ {0; 0,01; 0,05; 1; 2; 5; 10} × ΔP ∈ {0,64; 1,1793; 1,82} MW, $P_{head}$ =
3,414 MW (chưa bão hoà), mọi tham số khác giữ nguyên — **vòng trong không đổi**.

## Kết luận một dòng

**(a) đúng, (b) bị loại.** $\kappa_{os}\to1{,}003$ tại `KPplim` = 1–2 **trong khi vòng trong
giữ nguyên** — một hiệu ứng biến mất khi giữ cố định vòng trong thì không thể do vòng trong.
Nhưng **toàn bộ dải đặc tả REGFM_A1 (0,005–0,05) không chạy được**, và đó là phát hiện lớn
hơn câu hỏi ban đầu.

---

## 1. Kết quả — chỉ đọc dòng đã hội tụ

Ba cột ΔP; ΔP = 1,82 MW loại khỏi mọi kết luận (T23 đã cho thấy nó cận biên: 2/6 vị trí
không settle).

| `KPplim` | $\kappa_{os}$ @0,64 | $\kappa_{os}$ @1,1793 | $\tau_{eq}$ [s] | RoCoF @1,1793 | trạng thái |
|---:|---:|---:|---:|---:|---|
| 0 | — | — | — | — | ❌ **không hội tụ** |
| **0,01** ← ví dụ đặc tả | — | — | — | — | ❌ **không hội tụ** |
| **0,05** ← đỉnh dải đặc tả | — | — | — | — | ❌ **không hội tụ** |
| 1,0 | 1,0030 | 1,0046 | 0,368 | 1,5832 | ✅ |
| 2,0 | 1,0026 | 1,0035 | 0,370 | 1,5949 | ✅ |
| **5,0** ← mặc định ANDES, đang dùng | 1,2236 | 1,2304 | 0,263 | 1,9996 | ✅ |
| 10,0 | 1,9007 | 1,9139 | 0,079 | 3,1297 | ✅ |

**Kiểm nhất quán nội bộ:** `df_ss` giống hệt nhau (0,4402 / 0,8111 Hz) ở **mọi** dòng hội tụ
— đúng như hàm truyền nói, vì `KPplim` không xuất hiện trong DC gain. Và `KPplim` = 5 tái
lập T23 chính xác (1,2304 vs 1,2276; τ 0,2610 vs 0,2611).

### ⚠️ Dòng không hội tụ: số trong CSV là rác, không được đọc

Chín run tại `KPplim` ≤ 0,05 đều kết thúc ở t ≈ 1,3–2,0 s (sự cố tại t = 1,0 s) với
`tds_converged = False`. `kappa_os`/`tau_eq`/`df_ss_coi_hz` của chúng tính trên vệt bị cắt
cụt nên vô nghĩa. Dấu hiệu nhận biết: `df_ss` = 0,2375 Hz ở dòng đáng lẽ phải là 0,4402 Hz;
và `KPplim` = 0,01 / ΔP = 0,64 cho `df_ss` = **−2,57 Hz** (tần số *tăng* 2,57 Hz) với RoCoF
23,1 Hz/s — sụp đổ, không phải đáp ứng.

Cột `tds_converged` và `settled` trong `metrics.csv` là bộ lọc bắt buộc khi dùng lại file này.

---

## 2. Phán quyết theo ba nhánh đã đăng ký

Nhánh đăng ký trước: *"$\kappa_{os}\to1{,}00\pm0{,}03$ tại `KPplim` = 0 ⇒ (a) là cơ chế."*

**Đạt, nhưng ở vị trí khác:** $\kappa_{os}$ = 1,003 ± 0,001 tại `KPplim` = 1 và 2, không phải
tại 0 (ở 0 case không chạy). Kết luận cơ chế không đổi và mạnh hơn dự kiến:

- **(a) `KPplim` là cơ chế.** $\kappa_{os}$ bám `KPplim` đơn điệu trên khoảng [2, 10]:
  1,003 → 1,230 → 1,914. Hệ số nhân gần gấp đôi khi `KPplim` đi từ 5 lên 10.
- **(b) vòng trong bị loại.** Vòng trong giữ nguyên suốt sweep, mà $\kappa_{os}$ vẫn về 1,003.

### Giới hạn của hàm truyền T30 — phải nói rõ

Mô hình hở mạch `regf1_droop.tf()` dự đoán zero tại $-1/(T_{pm}(1+K_{Pplim}))$ vẫn nằm dưới
pole $-40$ tại `KPplim` = 1 (zero $-20$ rad/s) ⇒ *vẫn phải* overshoot. Đo được 1,003. Nên hàm
truyền **định danh đúng tham số chịu trách nhiệm nhưng không dự đoán được ngưỡng định lượng**
— ngưỡng đó là tính chất vòng kín (mạng + vòng trong), không phải vị trí zero hở mạch. Đã
ghi giới hạn này vào docstring của module.

---

## 3. ⚠️ SỬA SAU T32: nhãn "dải đặc tả" của lưới này sai thứ nguyên

Bản đầu của mục này ghi rằng `KPplim` ∈ {0,005; 0,01; 0,05} là "dải đặc tả REGFM_A1" và rằng
`KPplim` = 5 lệch 100×. **Cả hai sai**, vì so `KPplim` (vô thứ nguyên) trực tiếp với `kppmax`
(thứ nguyên ω-trên-công-suất). Quy đổi đúng — REGFM_A1 cộng nhánh overload thẳng vào ω, REGF1
cộng vào công suất rồi mới nhân `w0·wdrp`:

$$k_{ppmax}^{eff}=m_p\cdot K_{Pplim}=0{,}05\,K_{Pplim}$$

| `KPplim` | 0,005–0,05 | **0,1 – 1,0** | 2,0 | **5,0** |
|---|---|---|---|---|
| $k_{ppmax}^{eff}$ | 0,00025–0,0025 | **0,005 – 0,050** | 0,10 | **0,25** |
| vs dải 0,005–0,05 | **2–20× DƯỚI** | **ĐÚNG ĐẶC TẢ** (ví dụ = 0,2) | 2× trên | **5× trên** |

Nên phát biểu đúng là:

> Cửa sổ đúng đặc tả là `KPplim` ∈ [0,1 – 1,0]; giá trị đang ship (5,0) lệch **5×**. Vùng
> không hội tụ trong sweep này (≤ 0,05) nằm **dưới** dải đặc tả, không phải trong nó.

Vùng ≤ 0,05 vẫn là một kết quả thật — T32 xác nhận nó **bất ổn tuyến tính** ($\max\mathrm{Re}
\lambda$ = +8,95…+9,45, 6 mode) chứ không phải hỏng số học — nhưng nó **không** chứng minh
điều mục này từng khẳng định. Cửa sổ đặc tả thật thì chạy được: xem T32 §4.

---

## 4. Tác động lên biên an ninh — biên **bị hạ thấp**, không phải nâng khống

Biên bị siết bởi **RoCoF** (T23), không phải nadir, nên đường lan truyền đi qua cột RoCoF
chứ không qua công thức $\kappa_{os}$ ở nadir. Tại ΔP = 1,1793 MW:

| `KPplim` | RoCoF [Hz/s] | ngưỡng | dự phòng |
|---:|---:|---:|---|
| 2 | 1,5949 | 2,0 | còn 20% |
| **5 (đang dùng)** | **1,9996** | 2,0 | **hết** |

Ngoại suy tuyến tính trong vùng chưa bão hoà, tại giá trị ví dụ đặc tả `KPplim` = 0,2 với
vòng trong (0,10; 3,0) (RoCoF = 1,5685 — xem T32 §4):
$\Delta P_{\max}\approx 1{,}1793\times2{,}0/1{,}5685\approx\textbf{1,50 MW}$, so với
**1,1851 MW** đang công bố — **+27%**.

Đây là ngoại suy, **không phải bisect**, và ở mức đó nadir hoặc $\mu_I$ có thể siết trước
RoCoF. Con số thật cần chạy lại bisect.

Hệ quả: $\Delta P_{\max}$ = 1,1851 MW là hàm của một mặc định công cụ lệch 5× khỏi đặc tả, và
sai theo hướng **bảo thủ** (đánh giá thấp năng lực đội). T22–T26 đều thừa hưởng.

---

## 5. Việc tiếp theo — chưa chạy, chờ duyệt

| # | việc | vì sao |
|---|---|---|
| 1 | Trị riêng tại `KPplim` ∈ {0,01; 0,05; 1; 2; 5} | Tách "mất ổn định vật lý" khỏi "bất khả thi số học". Quyết định xem §3 là phát hiện về mô hình hay về solver |
| 2 | Bisect $\Delta P_{\max}$ tại `KPplim` = 2 (và 1) | Thay ngoại suy +25% bằng số đo |
| 3 | Quyết định giá trị nào ship | Phụ thuộc (1). Nếu dải đặc tả mất ổn định thật thì đó là kết quả về mô hình GFM chuẩn hoá, không phải lỗi cấu hình |

`experiments/t31_kpplim_mechanism.py` · `artifacts/T31_kpplim_mechanism/metrics.csv`
