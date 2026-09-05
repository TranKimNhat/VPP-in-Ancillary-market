# T37 (bước 4) — khớp lại $I_{dev}(\Delta P)$ và $I_{\max F}^{crit}$ tại ship C

**Vì sao phải làm lại.** T24 kết luận *"không `ImaxF` hợp lệ nào siết trước tần số"* bằng cách
khớp dòng thiết bị theo ΔP rồi đánh giá tại biên. Ship C dịch biên +21,4% **và** đổi tiêu chí
siết từ tần số sang điện áp (T34), nên cả điểm đánh giá lẫn khung câu hỏi đều đổi. Ở tin nhắn
trước tôi mới ước lượng bằng **một điểm** ($\mu_I$ tại biên); đây là phép khớp đầy đủ theo
đúng phương pháp T24.

Không có run mới: dùng `metrics.csv` của `T34_rerun_shipC/dpmax_q060` và
`T21_genloss_constP_regfm_q060`. Lọc: hội tụ, settled, chưa bão hoà, ΔP ≤ 1,60 (vùng tuyến
tính), 11 điểm mỗi bên — cùng bộ lọc T24 dùng.

$I_{dev}$ [pu thiết bị] $=\mu_I\times I_{\max F}$, vì `metrics` định nghĩa
$\mu_I = I/(I_{\max F}S_n)$.

---

## 1. Kiểm phương pháp — tái lập bản gốc chính xác

| | khớp | tại $\Delta P_{\max}$ | $I_{\max F}^{crit}$ |
|---|---|---:|---:|
| công bố, T24 ghi | $0{,}6247\,\Delta P + 0{,}6151$ | 1,18506 | **1,3555** |
| công bố, khớp lại ở đây | $0{,}6247\,\Delta P + 0{,}6151$ | 1,18506 | **1,3555** |

Trùng tới bốn chữ số, sai số khớp lớn nhất 0,0008 pu. Phương pháp đúng.

## 2. Ship C

$$I_{dev}(\Delta P) = 0{,}5646\,\Delta P + 0{,}6208 \quad [\text{pu thiết bị}]$$

11 điểm, ΔP = 0,050–1,525, sai số lớn nhất **0,0038 pu**. Tại $\Delta P_{\max}$ = 1,43857:

$$I_{\max F}^{crit} = \mathbf{1{,}4330}\ \text{pu}$$

| | công bố | ship C | đổi |
|---|---:|---:|---:|
| hệ số góc [pu/MW] | 0,6247 | **0,5646** | **−9,6%** |
| chặn [pu] | 0,6151 | 0,6208 | +0,9% |
| $I_{\max F}^{crit}$ | 1,3555 | **1,4330** | **+5,7%** |

Chặn gần như không đổi — nó là điểm vận hành trước sự cố, mà `KPplim` không đụng tới. Hệ số
góc giảm 9,6%: ở ship C đội **rút dòng chậm hơn trên mỗi MW**, nên $I_{\max F}^{crit}$ tăng
chậm hơn nhiều so với mức biên dịch (+5,7% so với +21,4%).

## 3. Kết luận T24 sống — nhưng hệ số an toàn **co lại một nửa**

| | $I_{\max F}^{crit}$ | đáy dải REGFM_A1 | hệ số an toàn |
|---|---:|---:|---:|
| công bố | 1,3555 | 1,50 | **1,107×** |
| ship C | 1,4330 | 1,50 | **1,047×** |

Phát biểu *"không `ImaxF` nào trong dải đặc tả (1,5–3,0) siết trước"* **vẫn đúng**, nhưng biên
dự phòng còn **4,7%** thay vì 10,7%. Phải viết kèm con số này, không được giữ nguyên giọng
"không bao giờ được".

**Xác nhận thực nghiệm, không chỉ ngoại suy:** `T34_rerun_shipC/dpmax_imaxf15` chạy ở
$I_{\max F}$ = 1,5 cho **đúng** 1,438574 — trùng biên ở $I_{\max F}$ = 2,0. Đúng như
1,5 > 1,4330 dự đoán.

$\Delta P$ mà giới hạn dòng sẽ siết, theo từng $I_{\max F}$:

| $I_{\max F}$ | ΔP khi dòng siết [MW] | siết trước biên 1,4386? | trong dải đặc tả? |
|---:|---:|---|---|
| 1,20 (định mức liên tục) | 1,0258 | có | không |
| **1,4330** | **1,4386** | *đúng điểm hoà* | không |
| **1,50** (đáy dải) | 1,5572 | **không** (+8,2%) | có |
| 2,00 (ví dụ) | 2,4427 | không | có |
| 3,00 (đỉnh dải) | 4,2137 | không | có |

Khoảng cách tới đáy dải thu từ +19,5% (công bố: 1,4164 so với 1,1851) xuống **+8,2%**.

Định mức **liên tục** 1,20 pu vẫn bị vượt trước biên (1,0258 < 1,4386), như trước. Đó là câu
hỏi thiết kế nhiệt theo chu kỳ làm việc, báo cáo kèm chứ không phải tiêu chí an ninh — nguyên
văn lập luận trong `build_case.CaseSpec` giữ nguyên hiệu lực.

## 4. Một chỗ khung câu hỏi phải đổi

T24 hỏi *"dòng có siết trước **tần số** không?"* và trả lời bằng bảng dự trữ/tốc-độ-tiêu-hao
đối chiếu `μ_I` với RoCoF. Ở ship C thứ siết là **`v_min`**, không phải RoCoF (T34 §3), nên
bảng đó so sai đối tượng. Câu hỏi đúng bây giờ là *"dòng có siết trước **thứ đang siết**
không?"* — trả lời ở §3: không, với biên 4,7%.

---

Không có run ANDES mới. Nguồn: `artifacts/T34_rerun_shipC/dpmax_q060/metrics.csv`,
`artifacts/T21_genloss_constP_regfm_q060/metrics.csv`.
