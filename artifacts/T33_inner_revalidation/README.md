# T33 — cổng tái thẩm định vòng trong: B **trượt**, C **đạt và tốt hơn hiện trạng**

**Cổng:** `build_case.py:152–166` tuyên bố vòng trong (0,20; 5,0) giữ $\mathrm{Re}\lambda\le0$
*"cho mọi deployment (2/5/6 GFM), mọi headroom, xf ∈ [0,05; 0,20], R ∈ [0,02; 0,05] và cả hai
mô hình tải"*. Đó là **toàn bộ** mệnh đề ổn định small-signal của bài. T32 vô hiệu hoá nó vì
cấu hình conformant đòi đổi vòng trong. Không được đụng default cho tới khi mệnh đề được
kiếm lại.

81 ô mỗi cấu hình = deployment(3) × xf(4) × R(3) × tải(2) ở headroom danh định, cộng trục
headroom (3 deployment × 3 giá trị). Chỉ trị riêng, không tích phân.

## Kết quả

| cấu hình | `KPplim` | $k_{ppmax}^{eff}$ | vòng trong | **ổn định** | worst $\max\mathrm{Re}$ | $\zeta_{\min}$ danh định |
|---|---|---|---|---|---:|---:|
| hiện tại | 5,0 | 0,250 ❌ 5× | (0,20; 5,0) | 75/81 | +5,05 | 0,0532 |
| **A** | 1,0 | 0,050 ✅ đỉnh | (0,20; 5,0) | **24/81** | +13,67 | 0,0020 |
| **B** ← đã chọn | 0,2 | 0,010 ✅ ví dụ | (0,10; 3,0) | **40/81** | +11,35 | 0,0679 |
| **C** | 1,0 | 0,050 ✅ đỉnh | (0,10; 3,0) | **77/81** | **+3,45** | **0,1559** |

**B trượt cổng.** Bất ổn trên toàn cạnh R = 0,02 (mọi xf) và phần lớn R = 0,03.

**A là tệ nhất**, ngược hẳn trực giác "giữ vòng trong thì giữ được mệnh đề robustness": mệnh
đề cũ đến từ **tổ hợp** (`KPplim`, vòng trong), không từ vòng trong một mình. Hạ `KPplim`
xuống 1,0 mà giữ vòng trong nhanh thì mất ổn định trên gần cả hộp.

**C đạt, và bao trùm hiện trạng.** Ô hỏng của C là **tập con thực sự** của ô hỏng hiện tại:

| | ô bất ổn |
|---|---|
| hiện tại | 2gfm, 5gfm, 6gfm × xf=0,05 × R=0,02 (6 ô) |
| **C** | 5gfm, 6gfm × xf=0,05 × R=0,02 (**4 ô**) |

C tốt hơn hiện trạng ở **mọi** thước đo: nhiều ô ổn định hơn (77 vs 75), worst $\max\mathrm{Re}$
thấp hơn (+3,45 vs +5,05), $\zeta_{\min}$ danh định tốt hơn **2,9×** (0,1559 vs 0,0532) — và
nó **đúng đặc tả** còn hiện trạng thì không. Trục headroom: cả 9 ô ổn định cho mọi cấu hình.

Miền thời gian tại danh định (T32 §4): $\kappa_{os}$ = 1,0046, RoCoF = 1,5811, $\mu_I$ = 0,725,
hội tụ, settled, secure.

## ⚠️ Mệnh đề đang có trong `build_case.py` bị nói quá — độc lập với mọi lựa chọn trên

Cấu hình **hiện tại** bất ổn tại xf = 0,05; R = 0,02 trên cả ba deployment
($\max\mathrm{Re}$ = +1,88…+5,05). Góc đó **nằm trong** hộp mà comment tuyên bố phủ. Nên câu
*"restores Re(lambda) <= 0 for … xf in [0.05, 0.20], R in [0.02, 0.05]"* **sai ở góc đó**, và
đã sai từ trước công việc này. Phải sửa comment dù chốt cấu hình nào.

C thu hẹp góc hỏng chứ không xoá: vẫn hỏng ở 5gfm/6gfm, xf = 0,05, R = 0,02.

## Ghi chú harness

Lần chạy đầu có 36/81 ô "hỏng" ở `load_p2z = 1,0` với `ValueError: load_step alters PQ.Ppf`.
Đó là **bug của kịch bản này**, không phải vật lý: `build_case` chặn `load_step` khi
`load_p2z > 0` (một `Alter` trên `Ppf` là no-op dưới tải trở kháng hằng). Đã sửa sang
`gen_loss` với `step_mw = 0`, hợp lệ cho cả hai mô hình tải và không làm lệch điểm cân bằng.
Sau khi sửa: **0 lỗi power flow**. Mọi số trong tài liệu này là từ lần chạy đã sửa.

## Còn lại: một sai lệch cấu trúc không tham số nào chữa được

Sơ đồ P của REGFM_A1 kẹp nhánh overload về **0** dưới ngưỡng (khối `0` trên cả hai nhánh
`Pmax`/`Pmin`), tức nó **ngủ** trong vận hành bình thường. REGF1 thì `PIplim` nhận
`Psig − Psen` với `Psig` là lag anti-windup của `Psen`, nên **luôn** khác 0 trong quá độ —
`KPplim` nằm trên đường droop kể cả khi không bão hoà (chính là tử số của hàm truyền T30).

Đây là khác biệt **cấu trúc**, không phải tham số: không giá trị `KPplim` nào làm REGF1 ngủ.
Ở C nó nhỏ ($\kappa_{os}$ = 1,005), nhưng phải khai báo. *Bằng chứng là sơ đồ khối trích từ
PDF — nên xác nhận lại với bản triển khai tham chiếu trước khi dựa vào.*

---

`experiments/t33_inner_revalidation.py` · `artifacts/T33_inner_revalidation/revalidation.csv`
(4 cấu hình × 81 ô)
