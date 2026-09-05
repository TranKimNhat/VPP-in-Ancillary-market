# T43 — $\Lambda$ và cấu trúc lưới: feeder không phải thứ sai, **cỡ đội mới là**

**Câu hỏi.** $\Lambda = X_{conv}/X_{feeder}$ là lý do mọi đại lượng an ninh bất biến với topology
và vị trí (T41 §2). Bài toán phân bổ chỉ tồn tại ở hệ có $\Lambda \lesssim 1$. $X_{conv}$ bị khoá
bởi $x_f$ = 0,15 (Bảng 1 REGFM_A1), nên câu hỏi là: **đặt lại sáu GFM ở ly cách điện tối đa trên
chính IEEE 123 có hạ được $\Lambda$ xuống ~1 không?**

Chỉ dùng đồ thị trở kháng (`accessibility.build_branch_graph`); **không chạy ANDES**.

$Z_{base}$ = 17,3056 Ω (4,16 kV, 1 MVA). $X_{conv,i} = (x_f + x_{tr})/S_i$ trên base hệ:
0,1582 pu (G1, 1,3277 MVA) … 0,5537 pu (G4/G6, 0,3793 MVA). $X_{feeder}$ = $|Z|$ đường dẫn
liên-GFM **trung bình**.

---

## 1. Ba kết quả

### (a) $\Lambda \propto 1/S_i$ — **bản rescale D6 là nguyên nhân trực tiếp**

$X_{conv,\Omega} = x_f V^2/S_i$, còn ohm của feeder cố định, nên

$$\Lambda = \frac{x_f V^2}{S_i\,Z_{feeder,\Omega}}$$

Rescale D6 nhân $S_i$ với 0,18967 ⇒ **nhân $\Lambda$ với 5,27×**.

$\Lambda \gg 1$ **không phải tính chất của IEEE 123.** Nó là hệ quả trực tiếp của quyết định thu
nhỏ đội GFM để làm vùng khan headroom chạm tới được. Hai yêu cầu xung đột theo cấu tạo:

| yêu cầu | đòi hỏi |
|---|---|
| an ninh: biên nằm trong dải vận hành (D6) | $\sum S_g$ **nhỏ** |
| phân bổ: $\Lambda \lesssim 1$ | $\sum S_g$ **lớn** |

### (b) Đặt lại ở ly cách tối đa giúp 2,06×, **không đủ**

| cách đặt | $|Z|$ liên-GFM tb | max | min | $\Lambda$ |
|---|---:|---:|---:|---|
| hiện tại `114,60,105,47,67,1` | 0,04183 | 0,09164 | 0,00449 | **3,78 – 13,24** |
| max-min sep. `6,85,114,66,151,96` | 0,08263 | 0,11086 | 0,06376 | 1,91 – 6,70 |
| **max-mean sep. `85,151,114,33,96,66`** | **0,08614** | 0,11086 | 0,05800 | **1,84 – 6,43** |

Vẫn $> 1$ với **mọi** máy.

### (c) Dứt điểm: **đường kính feeder nhỏ hơn $X_{conv}$ của mọi máy**

$|Z|$ lớn nhất giữa **bất kỳ hai bus nào** trên feeder = **0,11086 pu**.

| | $X_{conv}$ | so với đường kính feeder |
|---|---:|---:|
| G1 (lớn nhất, 1,3277 MVA) | 0,1582 | **1,43×** lớn hơn |
| G4/G6 (nhỏ nhất, 0,3793 MVA) | 0,5537 | **5,0×** lớn hơn |

> Đặt hai GFM ở hai đầu xa nhất của feeder, điện kháng converter **vẫn** lớn hơn toàn bộ trở
> kháng feeder giữa chúng.

Ở cỡ đội hiện tại, $\Lambda > 1$ **không sửa được bằng vị trí**. Cấu trúc lưới không phải thứ sai.

---

## 2. Nhưng có cửa sổ sizing, và nó rộng

$\Lambda \le 1$ cho máy **lớn nhất** cần $\sum S_g \ge 4{,}3624\Lambda_{\min}$; biên còn nằm trong
dải vận hành cần $\Delta P_{\max} < 3{,}49$ MW tức $\sum S_g < 21{,}03$ MVA (vì
$\Delta P_{\max} = 0{,}7241\sum S_g/4{,}3624$).

| cách đặt | $\Lambda\le1$ (máy lớn nhất) | $\Lambda\le1$ (mọi máy) | cửa sổ |
|---|---:|---:|---|
| hiện tại | $\ge$ 16,49 MVA | $\ge$ 57,76 MVA | 16,5 – 21,0 (1,3×) |
| **ly cách tối đa** | **$\ge$ 8,03 MVA** | $\ge$ 28,05 MVA | **8,0 – 21,0 (2,6×)** |

Ở đội **v3 gốc 23 MVA** với ly cách tối đa: $\Lambda$ = **0,35 – 1,22**, tức dưới 1 với bốn trên
sáu máy — nhưng $\Delta P_{\max}$ = 3,82 MW > tải 3,49, tức bão hoà, **đúng lý do D6 rescale**.

**Điểm thoả cả hai: ly cách tối đa + đội ~15–20 MVA.** Tại 20 MVA: $\Lambda$ = 0,40 – 1,40,
$\Delta P_{\max}$ = 3,32 MW < 3,49 ✓.

---

## 3. Phán quyết

Không rơi vào ngả nào trong hai ngả đã đặt trước ("$\Lambda$ xuống 1–2 ở đặt lại" / "IEEE 123
không chứa được"). Ngả thứ ba:

> **IEEE 123 @ 4,16 kV *có thể* chứa bài toán phân bổ — nhưng không ở cỡ đội hiện tại và không ở
> cách đặt hiện tại. Cần đồng thời (i) đặt lại sáu GFM ở ly cách tối đa và (ii) nâng đội từ
> 4,362 lên ~15–20 MVA.**

### Chi phí, nói thẳng

Điều đó **vô hiệu toàn bộ T20–T42**. Mọi biên, mọi sweep, mọi con số trong bảy commit gần nhất
đo trên đội 4,362 MVA ở cách đặt hiện tại. Đội mới ⇒ $\Delta P_{\max}$ mới, $\kappa$ mới,
$I^{crit}$ mới, và bất biến phải kiểm lại từ đầu.

### Và một hệ quả ngược chưa kiểm

Ở $\sum S_g$ = 20 MVA thì $\Delta P_{\max} \approx 3{,}3$ MW ≈ **95% tải feeder** — biên an ninh
gần trùng "mất toàn bộ tải". Vùng khan headroom mà D6 đặt ra để chạm tới sẽ **lại** khó chạm.
Cửa sổ tồn tại về số học; **chưa kiểm** liệu ở 20 MVA còn kịch bản nào để nghiên cứu. Đó là phép
thử tiếp theo và nó rẻ.

---

## 4. Ghi chú phương pháp

`build_branch_graph` đặt trở kháng ở thuộc tính cạnh **`z`**, không phải `weight`. Lần tính đầu
của phân tích này dùng `weight='weight'` nên networkx mặc định 1 và kết quả là **số hop**, cho
$\Lambda$ = 0,11–0,38 — sai và mâu thuẫn 37× với `d_elec_pu` của T23. Đã sửa; mọi số ở trên dùng
`weight='z'`. Đồ thị có bus cô lập nên mọi phép tính giới hạn ở thành phần liên thông lớn nhất.

Ly cách tối đa tìm bằng greedy max-min và greedy max-mean, thử **cả 128 bus làm hạt giống**, lấy
kết quả tốt nhất. Không phải tối ưu toàn cục, nhưng cả hai tiêu chí hội tụ về cùng đường kính
0,11086 pu, vốn là chặn trên tuyệt đối — nên khoảng cách tới tối ưu không đổi kết luận.
