# T41 — chạy lại các biên còn lại ở dải an ninh đã chốt

Dải đổi từ ±1,0 sang ±0,5 Hz và $V_{\min}$ 0,90 → 0,88 dịch $\Delta P_{\max}$ 1,450 → 0,724 MW
(−50%), lớn hơn cả bản sửa `KPplim` trước đó, nên mọi số đo ở dải cũ là tạm. Nguồn gốc ngưỡng:
`reference/security_band_provenance.md`. Cấu hình converter: ship C.

`kappa_dp1p1` (điểm công bố thứ hai của T26) **không chạy**: ΔP = 1,1 nay vượt $\Delta P_{\max}$,
nên không headroom nào sống sót và bisection không có gì để kẹp. Đó là hệ quả của dải, không
phải hỏng.

---

## 1. Đối chứng âm **đạt chính xác**

| | dải cũ | dải cuối |
|---|---:|---:|
| $P_{head}^{\min}$ @ ΔP = 0,6 | 0,595336 | **0,595336** |
| $\kappa$ | 1,007834 | **1,007834** |

Giống hệt **sáu chữ số** qua một nhiễu loạn 50% ở biên. `run_dp_max` docstring nói
$P_{head}^{\min}$ là biên khả thi `P_head >= dP` và không mang động học; đây là xác nhận, và nó
cũng chứng minh harness không trôi. Thêm điểm @ ΔP = 0,7: 0,700461, $\kappa$ = 0,999342.

## 2. Bất biến topology/vị trí — **lần đầu được phân giải thật**

| | spread $\Delta P_{\max}$ | siết | `margin_gap` |
|---|---:|---|---:|
| topology (4 case) | **0,000%** | nadir 4/4 | **0,476** |
| vị trí sự cố (6 case) | **0,000%** | nadir 6/6 | **0,476** |

Đây là kết quả chính của T41. Ở các dải trước, `margin_gap` là 0,002–0,007 — **dưới** dung sai
bisection 0,02 — nên T35 buộc phải tự cảnh báo rằng spread 0,000% có thể chỉ là trùng hợp của
lưới bisection chứ không phải bằng chứng bất biến. Nay á quân ($\mu_I$) cách tiêu chí siết
**47,6 điểm phần trăm** và spread vẫn 0,000%.

$$\textbf{Bất biến topology/vị trí nay là phép đo, không phải hiện vật ngưỡng.}$$

Và nó đã đứng qua: bản sửa `KPplim` (+21,4%), việc truy nguồn dải (−50%), và **ba toạ độ siết
khác nhau** — `rocof`+nadir (cấu hình công bố) → `v` (ngưỡng 0,90) → `nadir` (dải cuối). Không
kết quả nào khác trong dự án chịu được chừng đó.

## 3. Diesel-off — **phân giải được ở tol 0,002, và đảo dấu kết quả công bố**

Ở tol 0,02 hiệu là +0,45%, **dưới độ phân giải** (khoảng kẹp ~1,6% quanh 0,72). Siết tol xuống
0,002 (`artifacts/T42_tight/`, 17 + 16 lần dò):

| | $\Delta P_{\max}$ | $P_{DG,off}^{\max}$ | hiệu |
|---|---|---|---:|
| tol 0,02 | 0,724121 | 0,727344 | +0,45% *(không phân giải)* |
| **tol 0,002** | **0,724841** [0,724121; 0,725562] | **0,722559** [0,721875; 0,723242] | **−0,315%** |

Hai khoảng kẹp **rời nhau** — 0,723242 < 0,724121 — và bề rộng kẹp 0,199% nhỏ hơn hiệu 0,315%.
**Phân giải được: mất diesel cuối khó hơn mất nguồn phi đồng bộ cùng cỡ, 0,315%.**

Sổ cái đầy đủ của mệnh đề này:

| | hiệu | trạng thái |
|---|---:|---|
| công bố (`KPplim`=5, dải ±1,0, V 0,90) | **+1,99%** diesel *dễ hơn* | ❌ hiện vật |
| ship C, dải cũ | −9,90% diesel khó hơn | ❌ hiện vật |
| **dải cuối, tol 0,002** | **−0,315%** diesel khó hơn | ✅ **phân giải** |

Kết luận công bố *"mất diesel cuối dễ hơn 2,0% — ngược dấu với dự đoán"* **sai dấu**. Hiệu thật
cùng chiều trực giác và **nhỏ hơn 6,3 lần** con số đã công bố. Đóng góp III được xây trên tính
phản trực giác đó; tính đó không tồn tại.

## 4. $I_{\max F}^{crit}$ — không cần run mới

Khớp lại $I_{dev}(\Delta P)$ trên `T40_band_final/dpmax/metrics.csv` (9 điểm chưa bão hoà,
ΔP 0,05–0,787, sai số 0,0007):

$$I_{dev} = 0{,}5748\,\Delta P + 0{,}6170 \quad\Rightarrow\quad I_{\max F}^{crit} = \mathbf{1{,}0332}\ \text{pu}$$

| | $I^{crit}$ | hệ số an toàn tới sàn đặc tả 1,5 |
|---|---:|---:|
| công bố | 1,3555 | 1,107× |
| ship C, dải cũ | 1,4330 | 1,047× |
| **dải cuối** | **1,0332** | **1,452×** |

Và một cảnh báo **bỏ được**: ở $I_{\max F}$ = 1,20 (định mức **liên tục**, dưới sàn đặc tả) dòng
siết tại ΔP = 1,0143 > 0,7241, nên tại biên cuối $\mu_{I,cont}$ = 0,861 < 1. Định mức liên tục
không còn bị vượt; ghi chú duty-cycle nhiệt trong `CaseSpec` không còn cần cho ca này.

§28 N3 (*"không limiter hợp lệ nào siết"*) nay mạnh nhất từ trước tới nay.

---

`experiments/t41_rerun_final_band.sh` · 5 chiến dịch · `artifacts/T42_tight/` cho §3
