# T36 — $\kappa_{os}(K_{Pplim})$ tại vòng trong ship C, và một mệnh đề phải hạ xuống

**Vì sao đo lại.** T31 đo đường cong này ở vòng trong cũ (0,20; 5,0). T33 cho thấy ổn định —
và do đó hình dạng đường cong — là tính chất của **cặp** (`KPplim`, vòng trong), nên đường cong
cũ không mô tả cấu hình đang ship. 8 giá trị `KPplim` × 2 ΔP, vòng trong (0,10; 3,0).

Quy đổi conformance (`src/analytical/regf1_droop.py`):
$k_{ppmax}^{eff} = m_p K_{Pplim} = 0{,}05\,K_{Pplim}$; dải REGFM_A1 0,005–0,05.

---

## 1. Đường cong

| $K_{Pplim}$ | $k_{ppmax}^{eff}$ | vs dải | $\kappa_{os}$ @0,64 | $\kappa_{os}$ @1,1793 | RoCoF @1,1793 |
|---:|---:|---|---:|---:|---:|
| 0,05 | 0,0025 | 2× dưới | 1,0033 | 1,0060 | 1,5668 |
| 0,10 | 0,0050 | **đáy dải ✅** | 1,0033 | 1,0059 | 1,5682 |
| **0,20** | **0,0100** | **ví dụ đặc tả ✅** | 1,0032 | 1,0057 | 1,5685 |
| 0,50 | 0,0250 | ✅ | 1,0031 | 1,0053 | 1,5717 |
| **1,00** | **0,0500** | **đỉnh dải ✅ ← ship** | 1,0029 | **1,0046** | 1,5811 |
| 2,00 | 0,1000 | 2× trên | 1,0026 | 1,0037 | 1,5931 |
| 5,00 | 0,2500 | **5× trên ← cũ** | **1,2281** | **1,2328** | **2,0048** |
| 10,00 | 0,5000 | 10× trên | 1,9053 | 1,9144 | 3,1352 |

Cả 8 điểm hội tụ và settled — kể cả `KPplim` = 0,05, thứ từng phân kỳ ở vòng trong cũ. Xác
nhận bản đồ T32.

**Hai vùng, một ngưỡng sắc.** $\kappa_{os}$ **phẳng ở 1,003–1,006 suốt từ 0,05 đến 2,0** — dải
20× của $k_{ppmax}^{eff}$ làm $\kappa$ đổi 0,0014. Rồi nhảy giữa 2 và 5: 1,0037 → 1,2328
(+23%), và 1,91 ở 10.

**Vị trí ngưỡng không đổi giữa hai cách chỉnh vòng trong.** T31 ở (0,20; 5,0) cũng cho nhảy
giữa 2 và 5 (1,0035 → 1,2304). Nên **ngưỡng là tính chất của `KPplim`**, dù *tính ổn định*
dưới ngưỡng lại phụ thuộc cặp. Đây là kết quả robustness thật.

---

## 2. ⚠️ Mệnh đề "$\kappa_{os}$ đo mức vi phạm giả thiết vòng trong" phải hạ xuống

Khung trước đây (T30 §5) là:

> $\kappa_{os}-1 = 0{,}2275$ là phép đo định lượng mức vi phạm giả thiết *"inner control loops
> track their references perfectly"* của mô hình rút gọn chuẩn.

**Đường cong bác khung đó.** Toàn bộ cửa sổ đúng đặc tả nằm trong vùng phẳng: ở cấu hình
conformant, $\kappa_{os}-1 \approx \mathbf{0{,}005}$, không phải 0,2275. Nghĩa là **dự đoán
$\kappa \equiv 1$ của các mô hình rút gọn về cơ bản là đúng cho một GFM đúng đặc tả**, và
22,7% đo được trước đây gần như toàn bộ là hiện vật của một mặc định lệch 5× khỏi dải.

Cái còn lại sau khi trừ, và nó vẫn thật:

$$\kappa_{os}-1 \to 0{,}0060 \text{ khi } K_{Pplim}\to 0 \quad (\Delta P = 1{,}1793)$$

**0,6%**, không phải 22,7%. Đó mới là mức vi phạm giả thiết vòng trong lý tưởng ở cấu hình
conformant — nhỏ, đo được, và đúng chiều.

### Mệnh đề nào còn đứng

Không còn *"có một hiệu ứng vật lý mà mọi mô hình GFM tổng hợp bỏ sót"*. Còn lại là mệnh đề
hẹp hơn nhưng vẫn đúng, và là mệnh đề về **phạm vi phủ của mô hình**:

> Các mô hình GFM rút gọn tham số hoá converter bằng cặp quán tính–damping và **không có chỗ
> cho gain của bộ giới hạn quá tải**. Trong dải đặc tả điều đó vô hại — $\kappa_{os}$ lệch 1
> dưới 0,6%. Ngoài dải, chính tham số ấy chi phối nadir và RoCoF (23% ở 5×, 91% ở 10×), và
> không mô hình rút gọn nào biểu diễn hay cảnh báo được. Sai lệch tham số ngoài dải là chế độ
> hỏng mà lớp mô hình đó mù.

Yếu hơn khung cũ. Nhưng nó đúng, và nó chính là câu chuyện đã thực sự xảy ra ở chiến dịch này.

---

## 3. Vì sao vẫn phải công bố dưới dạng đường cong

$\kappa_{os}$ và RoCoF **không phải hằng số của plant** — chúng là hàm của một tham số điều
khiển. Công bố một con số duy nhất mời người đọc hiểu nhầm là thuộc tính vật lý. Bảng §1 là
hiện vật; con số cũ 1,2275 phải luôn kèm *"tại $K_{Pplim}$ = 5, tức 5× ngoài dải REGFM_A1"*.

---

`experiments/t36_kappa_curve.py` · `kappa_curve.csv` (16 run, cả 16 hội tụ)
