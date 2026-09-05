# T39 — $\Delta P_{\max}$ tại ngưỡng $V_{\min}$ có nguồn (IEEE 1547 Cat III, 0,88 pu)

`SecurityBand.v_min_pu` = 0,90 là **khiếm khuyết nền tảng thứ năm**: một ngưỡng trần trụi,
không comment, không trích dẫn, đang quyết định toạ độ siết. Nguồn gốc và quyết định chọn
0,88 pu: `reference/security_band_provenance.md`.

Chạy lại `dp_max` ở cấu hình ship C với `--v-min 0.88`. Không đổi gì khác.

---

## Kết quả

$$\Delta P_{\max} = \mathbf{1{,}4501\ MW}\quad\text{kẹp }[1{,}4443;\ 1{,}4559],\ 14\text{ lần dò, đơn điệu}$$

| | $V_{\min}$ = 0,90 (T34) | $V_{\min}$ = 0,88 (ship) |
|---|---:|---:|
| $\Delta P_{\max}$ | 1,438574 | **1,450098** |
| đổi | — | **+0,80%** — *đúng một khoảng kẹp* |
| toạ độ siết | `v_min` | **`nadir`** |

**Ngưỡng dịch cơ chế, không dịch con số.** +0,80% nằm dưới ngưỡng phân giải của bisection
(tol = 0,02), cùng độ lớn với mọi nhiễu loạn không phân giải được ở T38. Nhưng toạ độ siết đổi
hẳn: điểm mất an ninh đầu tiên giờ hỏng vì `f_nadir 58,9931 < 59,0` và **không** kèm khiếu nại
điện áp.

## Lề tại biên mới

| tiêu chí | giá trị | ngưỡng | dùng hết | còn lại |
|---|---:|---:|---:|---:|
| **nadir** | 59,0011 | 59,0 | 0,9989 | **0,11%** |
| RoCoF | 1,9179 | 2,0 | 0,9589 | 4,11% |
| $V_{\min}$ | 0,8996 | 0,88 | 0,8370 | 16,30% |
| $\mu_I$ | 0,7173 | 1,0 | 0,7173 | 28,27% |

Thứ tự: **nadir siết, RoCoF cách 4,1%, điện áp cách 16,3%.** So với ở ngưỡng 0,90, nơi điện áp
và nadir chỉ cách nhau 0,56% — thế chân vạc biến mất, biên trở lại **giới hạn tần số** một cách
dứt khoát.

## Sửa một phát biểu của tôi ở lượt trước

Tôi nói ngưỡng $V_{\min}$ "kéo $\Delta P_{\max}$ qua dải 45%". **Sai** — 45% là dải của *"ΔP
khi V siết"*, không phải của $\Delta P_{\max}$. $\Delta P_{\max}$ = min trên mọi tiêu chí, nên
nadir chặn trên nó ở 1,4460:

| $V_{\min}$ | ΔP khi V siết | $\Delta P_{\max}$ thật |
|---|---:|---:|
| 0,9167 | 1,195 | **1,195** (V siết) |
| 0,90 | 1,438 | **1,439** (V siết, sát nút) |
| 0,88 | 1,729 | **1,450** (nadir siết) |

Dải thật là **21%**, và **bất đối xứng**: siết ngưỡng thì dịch mạnh, nới ngưỡng dưới ~0,90 thì
gần như không dịch vì nadir tiếp quản.

## Một quan sát, chưa phải mệnh đề

$\kappa_{os}$ = 1,0046 nghĩa là $f_{nadir} \approx f_0 - \Delta f_{ss}$, tức ràng buộc nadir
**rút gọn về ràng buộc đại số của droop**. Ở ngưỡng có nguồn, chính ràng buộc đó là thứ siết.

Nên cách đọc *"ràng buộc nadir suy biến, toạ độ siết chuyển sang điện áp"* **sai chiều**: ràng
buộc tần số không biến mất — nó **đơn giản hoá thành dạng đóng không có động học**, và vẫn siết.
Với một bài toán điều độ, đó là tin tốt hơn: ràng buộc siết nhúng thẳng được vào bài toán tối
ưu mà không cần mô hình động.

**Chưa phát biểu đây là kết quả.** Ba khung trước đã chết vì phát biểu trước khi kiểm. Điều kiện
để khung này đứng, chưa cái nào kiểm:
1. Nó dựa trên **một** mô hình converter (REGF1). $\kappa_{os}\approx1$ chưa có xác nhận độc lập.
2. `f_min_hz` = 59,0 **cũng chưa có nguồn** — xem dưới.

## ⚠️ Việc mở ngay lập tức

Với $V_{\min}$ đã có nguồn, **nadir thành toạ độ siết, nên `f_min_hz` = 59,0 giờ là con số trực
tiếp đặt $\Delta P_{\max}$** — và nó vẫn trần trụi, không trích dẫn, hệt như 0,90 vừa bị thay.
Vấn đề không được giải quyết; nó **chuyển sang tham số bên cạnh**. `rocof_max_hz_s` = 2,0 cách
biên 4,1%, cũng chưa có nguồn.

## Ghi chú về T35

T35 báo `binding = v` ở 10/10 case — đúng **tại ngưỡng 0,90**. Ở 0,88 cả 10 sẽ là `nadir`.
Kết luận bất biến của T35 **không đổi** (nó đã đứng dưới `rocof` ở cấu hình cũ và `v` ở 0,90,
tức bất biến với việc cái gì siết), nhưng nhãn tiêu chí trong tài liệu đó là điều kiện theo
ngưỡng. Chạy lại không đổi kết luận nên chưa chạy.

---

`experiments/t20_andes_bisect.py --v-min 0.88` (nay là mặc định) · 14 lần dò
