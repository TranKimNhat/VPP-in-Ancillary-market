# T24 — `ImaxF = 1,5` (đáy dải REGFM_A1): dòng có siết trước tần số không?

**Câu hỏi (lối ra số 1 của T23):** T23 kết luận không toạ độ nào của Ω_dyn mang thông tin
topology, vì biên do RoCoF quyết định và RoCoF là đại lượng khối. Hạ `ImaxF` từ ví dụ 2,0
xuống đáy dải đặc tả 1,5 có đưa `μ_I` lên làm toạ độ siết không?

## Kết luận một dòng

**Không, và không bao giờ được — với bất kỳ `ImaxF` nào trong dải đặc tả.**
`ΔP_max = 1,1851 MW`, **giống hệt** T21/T22/T23. Tại điểm an toàn cuối cùng (ΔP = 1,1793):
`μ_I = 0,901` (dự trữ 0,099) so với RoCoF dự trữ 0,0005 — **tần số vẫn siết trước 200 lần.**

## Ngưỡng tới hạn: `ImaxF ≤ 1,3555`, nằm **dưới** đáy dải

Dòng converter theo ΔP là đường thẳng gần như hoàn hảo (11 điểm, ΔP = 0,05–1,53 MW, sai số
lớn nhất 0,0008 pu):

$$I_{\text{dev}}(\Delta P) = 0{,}6247\,\Delta P + 0{,}6151 \quad [\text{pu thiết bị}]$$

Tại biên RoCoF ΔP = 1,18506 → `I = 1,3555` pu. Vậy dòng chỉ siết trước nếu **`ImaxF ≤ 1,3555`**
— thấp hơn đáy dải REGFM_A1 (1,5) **1,107 lần**.

| `ImaxF` | ΔP_max nếu dòng siết [MW] | siết trước RoCoF (1,1851)? | trong dải đặc tả? |
|---:|---:|---|---|
| 1,20 | 0,9362 | có | **không** |
| 1,3555 | 1,1851 | *đúng điểm hoà* | **không** |
| **1,50** | **1,4164** | **không** | có (đáy dải) |
| 2,00 | 2,2168 | không | có (ví dụ) |
| 3,00 | 3,8175 | không | có (đỉnh dải) |

Kiểm chứng: mô hình tuyến tính cho `μ_I(1,1793) = 1,3517/1,5 = 0,9011`; đo được **0,901**.

## Vì sao — dòng xuất phát gần trần hơn nhưng tiến chậm hơn một nửa

Đọc bằng dự trữ chuẩn hoá và tốc độ tiêu hao của nó:

| toạ độ | dự trữ tại ΔP = 0 | tốc độ tiêu hao [/MW] | cắt 0 tại [MW] |
|---|---:|---:|---:|
| RoCoF | 1,000 | **0,856** | **1,168** |
| `μ_I` (ImaxF = 1,5) | 0,590 | 0,417 | 1,416 |
| `μ_I` (ImaxF = 2,0) | 0,692 | 0,312 | 2,217 |

`μ_I` **xuất phát gần trần hơn** (0,59 so với 1,00) nhưng **tiêu hao chậm bằng một nửa**, nên
tần số vẫn về đích trước. Nguyên nhân nằm ở hệ số chặn 0,6151 pu: ở ΔP = 0 các converter đã
mang sẵn 0,615 pu dòng, gần như toàn bộ là **dòng phản kháng** (đội thiết bị cấp 1,92 MVAr
trên 4,36 MVA = 0,44 pu, cộng phần đang nạp). Thành phần đó **không phụ thuộc ΔP**, nên ΔP chỉ
điều biến dòng với độ dốc 0,625 pu/MW — bằng 73% độ dốc của RoCoF sau khi chuẩn hoá.

## Hệ quả

**Lối ra số 1 của T23 đóng.** Không phải "1,5 chưa đủ thấp" mà là **cả dải đặc tả đều không
đủ thấp**: muốn dòng siết trước thì phải khai báo `ImaxF` ngoài REGFM_A1, tức tự phá bỏ chính
lập luận tuân thủ đã dựng ở T21.

Còn lại hai lối:

- **Lối 2 — đổi điểm vận hành.** Tăng tải hoặc giảm đội GFM để hệ số chặn 0,6151 và độ dốc
  0,6247 dịch lên. Đây là thay đổi *thật* của bài toán, phải chạy lại toàn bộ.
- **Lối 3 — phát biểu kết quả rỗng như một kết quả.** Giờ nó không còn là quan sát mà là
  **định lượng**: biên an ninh của hệ này do tần số quyết định với mọi tham số giới hạn dòng
  hợp lệ theo REGFM_A1, và khoảng cách tới chế độ dòng-siết là 1,107 lần trên `ImaxF`.

Ghi chú tách bạch định mức: tại biên, `μ_I`(liên tục 1,2 pu) = **1,127** — dòng vượt định mức
*liên tục* 13% ở điểm vận hành an ninh. Đó là bài toán chu trình nhiệt của converter, báo cáo
riêng, không phải tiêu chí an ninh (xem `src/phasor/metrics.py`).

## Sinh lại

```
uv run python experiments/t20_andes_bisect.py --what dp_max --event gen_loss \
    --load-p2z 0.0 --q-max 0.60 --i-max 1.50 --out artifacts/T24_imaxf15
```
