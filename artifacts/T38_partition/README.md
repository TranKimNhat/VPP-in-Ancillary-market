# T38 — phép kiểm phân hoạch robust/fragile trên trục thứ hai: **không đứng**

**Mệnh đề được kiểm.** T34–T37 quan sát rằng qua *một* lần đổi cấu hình
(`KPplim` 5 → 1 cùng cặp vòng trong), các đại lượng an ninh tách đôi: những đại lượng do cân
bằng năng lượng và tôpô quyết định thì đứng yên, những đại lượng do **tạo hình quá độ của
converter** quyết định thì dịch. Một điểm không thành một phân hoạch, nên phải kiểm trên ít
nhất một trục nhiễu loạn thứ hai.

Hai đại diện, mỗi vế một cái: $\Delta P_{\max}$ (vế mong manh) và $P_{head}^{\min}$ (vế bền).
Hai trục: $x_f$ (vật lý) và vòng trong $(K_{Pi},K_{Ii})$ (điều khiển — trục kiểm đúng mệnh đề).
Mọi điểm nằm trong hộp ổn định T33; trị riêng kiểm trước, không điểm nào bị bỏ.

---

## 1. Kết quả

| trục | điểm | $\zeta_{\min}$ | $\Delta P_{\max}$ | vs ship | $P_{head}^{\min}$ | vs ship |
|---|---|---:|---:|---:|---:|---:|
| — | ship (0,15 / 0,10 / 3,0) | 0,1559 | 1,438574 | — | 1,094680 | — |
| $x_f$ | 0,10 | | 1,427051 | −0,80% | 1,094680 | 0,00% |
| $x_f$ | 0,20 | | 1,438574 | 0,00% | 1,094680 | 0,00% |
| vòng trong | (0,12; 3,5) | 0,0947 | 1,450098 | **+0,80%** | 1,094680 | 0,00% |
| vòng trong | (0,14; 3,0) | 0,0693 | 1,450098 | **+0,80%** | 1,094680 | 0,00% |

## 2. ❌ Phép kiểm **không đứng**, và lý do quan trọng

**Không trục nào dịch được $\Delta P_{\max}$ quá ngưỡng phân giải.** Cả bốn nhiễu loạn cho
$|\Delta| \le 0{,}80\%$, mà 0,80% là **đúng một khoảng kẹp** ở tol = 0,02 quanh 1,44. Hơn nữa
1,450098 chính là giá trị G2 trả ở T35 — một điểm mút bracket lặp lại, tức hiện vật rời rạc
hoá, không phải độ nhạy.

Một phép kiểm mà nhiễu loạn **không sinh đáp ứng** thì không tách được bền khỏi mong manh.

**Nhưng nó không phải vô thông tin — nó bác mệnh đề.** So sánh:

| nhiễu loạn | $\Delta P_{\max}$ dịch |
|---|---:|
| `KPplim` 5 → 1 (rời khỏi/vào lại dải đặc tả, một tham số) | **+21,4%** |
| vòng trong, trong vùng ổn định đã xác minh | ≤ 0,80% (không phân giải được) |
| $x_f$, dải gấp đôi | ≤ 0,80% (không phân giải được) |

$\Delta P_{\max}$ **không mong manh với tham số điều khiển converter nói chung.** Nó mong manh
với **một** tham số cụ thể — gain giảm quá tải, thứ nằm trên đường thuận của droop và thứ mà
các mô hình rút gọn không mang — và chỉ khi tham số ấy ra **ngoài dải đặc tả**.

Điều này khớp chính xác với đường cong T36: $\kappa_{os}$ phẳng ở 1,003–1,006 suốt dải 20× của
$k_{ppmax}^{eff}$ trong cửa sổ conformant, rồi mới nhảy khi ra 5× ngoài dải. Độ nhạy là một
**ngưỡng ở một tham số**, không phải tính mong manh của cả một lớp đại lượng.

## 3. Phát biểu thay thế, ngược hẳn mệnh đề gốc

> $\Delta P_{\max}$ **bền** với tham số điều khiển converter bên trong vùng vừa đúng đặc tả vừa
> đã xác minh ổn định — vòng trong và điện kháng ghép nối không dịch nó quá ngưỡng phân giải
> của bisection. Nó dịch 21,4% **chỉ vì** một tham số nằm 5× ngoài dải đặc tả. Cái mong manh
> không phải đại lượng an ninh động; cái mong manh là **sự tuân thủ đặc tả**.

Yếu hơn về tham vọng, mạnh hơn về bằng chứng, và có ích hơn cho vận hành: nó nói biên an ninh
động **đáng tin** khi mô hình conformant, chứ không nói nó ngẫu nhiên.

## 4. Vế "bền" phần lớn bền **theo cấu tạo** — lỗ hổng độc lập

Kể cả nếu phép kiểm đã đứng, vế bền vẫn thiếu chân:

| thành viên | vì sao bất biến |
|---|---|
| $\Delta f_{ss}$ | đồng nhất đại số, `KPplim` không có trong DC gain — **theo cấu tạo** |
| $P_{head}^{\min}$, bao $\kappa$ | `t20_andes_bisect.py:262–270` đã ghi sẵn: *"the feasibility bound `P_head >= dP` and nothing more — the droop dynamics never get a say"*. Kiểm: $\kappa$ = 1,1/1,09468 = 1,00486 — **theo cấu tạo** |
| $I_{\max F}$ không siết | **đã dịch**: hệ số an toàn 1,107× → 1,047× |
| bất biến topology/vị trí | **đo được, không theo cấu tạo** ✅ |

Chỉ **một** thành viên là quan sát độc lập. Trong tài liệu này $P_{head}^{\min}$ vì thế được
dùng làm **đối chứng âm** — đại lượng *phải* đứng yên nếu harness chạy đúng — chứ không phải
bằng chứng. Nó đứng yên ở cả 5/5 điểm: harness đúng.

## 5. Muốn kiểm thật thì cần gì

Phân giải hiệu ứng dưới 1% cần **tol ≲ 0,002** thay vì 0,02, tức khoảng 10× số lần dò cho mỗi
biên (~50 phút/biên thay vì ~5). Đây là cùng giới hạn T35 đã gặp ở phép kiểm bất biến. Chưa
làm, và chỉ nên làm nếu có mệnh đề cần con số dưới 1% — mệnh đề ở §3 thì không cần.

---

`experiments/t38_partition_second_axis.sh` (trục $x_f$) ·
`experiments/t38_inner_axis.py` (trục điều khiển) · `inner_axis.csv`
