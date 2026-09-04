# T23 — quét **vị trí nhiễu** thay vì quét tie

**Câu hỏi (nối tiếp T22):** T22 cho `ΔP_max` đứng yên qua bốn cấu hình tie, và RoCoF cũng
đứng yên. Giả thuyết khi đó: vì điểm nhiễu bị ghim ở bus 76, thứ duy nhất đổi là trở kháng
*giữa* các nhánh. Vậy dời điểm nhiễu thì sao?

Sáu vị trí, trải đều theo phân vị khoảng cách điện tới GFM gần nhất (đường đi ngắn nhất có
trọng số điện kháng, tie đóng = trở kháng không), cộng bus 76 làm mốc nối về T21.

## Kết luận một dòng

**Không cứu được.** Năm trong sáu vị trí cho **đúng `1,1851 MW`**; một vị trí (bus 102) cho
`1,1735 MW`. Phân tán 0,0115 MW = **0,97%** — và đó là **hiệu ứng ngưỡng tiếp tuyến**, không
phải hiệu ứng vật lý cỡ 1%.

| Vị trí | d_elec [pu] | tải bus [MW] | bậc | ΔP_max [MW] | siết bởi |
|---|---:|---:|---:|---:|---|
| E1 (bus GFM) | 0,0000 | 0,040 | 4 | **1,1851** | RoCoF |
| E76 (mốc) | 0,0056 | 0,245 | 3 | **1,1851** | RoCoF |
| E102 | 0,0059 | 0,020 | 2 | **1,1735** | RoCoF |
| E41 | 0,0121 | 0,020 | 1 | **1,1851** | RoCoF |
| E88 | 0,0212 | 0,040 | 1 | **1,1851** | RoCoF |
| E33 (xa nhất) | 0,0386 | 0,040 | 1 | **1,1851** | RoCoF |

Khoảng cách điện **không** xếp hạng được kết quả: ngoại lệ là E102 (gần thứ ba), không phải
E33 (xa nhất, 6,5 lần xa hơn E102).

## ① Ngoại lệ E102 là khuếch đại ngưỡng, không phải vật lý

So sánh ở **cùng một ΔP** (chứ không ở mốc riêng của từng lần bisection — mốc khác nhau là
nguỵ biện) tại ΔP = 1,1793 MW:

| | E88 | E1 | E41 | E33 | E76 | E102 | phân tán |
|---|---:|---:|---:|---:|---:|---:|---:|
| RoCoF [Hz/s] | 1,99869 | 1,99886 | 1,99897 | 1,99873 | 1,99887 | **2,00010** | **0,07%** |

E102 vượt ngưỡng 2,0 đúng **0,0001 Hz/s**. Mọi vị trí khác nằm dưới. Một chênh lệch vật lý
**0,07%** biến thành chênh lệch biên **0,97%** vì tại điểm góc tiêu chí gần như tiếp tuyến với
biên. Đây là tính chất của **ngưỡng 2,0 Hz/s**, phải phát biểu như vậy trong bài — không được
đọc thành "vị trí nhiễu dịch biên 1%".

## ② Cấu trúc lập luận đúng, nhưng biên độ nhỏ hơn hai bậc

Độ nhạy tương đối theo vị trí nhiễu, đo tại ΔP = 1,1793 MW:

| toạ độ | phân tán tương đối | so với nadir |
|---|---:|---:|
| `f_nadir` | **0,0012%** | 1× |
| `V_min` | 0,053% | 44× |
| RoCoF | 0,070% | 58× |
| `μ_I` | **0,63%** | **530×** |

Thứ tự đúng như dự đoán: **nadir là đại lượng khối, `μ_I` là đại lượng cục bộ.** Nhưng ngay
cả `μ_I` cũng chỉ phân tán 0,63% trong khi dự trữ của nó là 0,32 — cách ngưỡng 50 lần. Không
toạ độ nào mang đủ thông tin topology để dịch biên.

Độ nhạy **tăng theo ΔP**: phân tán `μ_I` đi 0,02% (ΔP = 0,05) → 0,63% (1,18) → 3,0% (1,53).
Từ ΔP ≈ 1,82 MW trở lên các vị trí **phân kỳ định tính**: E76/E88/E102 sụp đổ (nadir 56,6,
`μ_I` 2,21) trong khi E1/E41/E33 không (nadir 58,45, `μ_I` 0,86). Khác biệt topology *thật*
và *lớn* — nhưng nằm sâu trong miền mất an ninh, nên không chạm tới `ΔP_max`.

## ③ Cơ chế: feeder này về mặt điện là một điểm

Điện kháng phía converter, quy về base hệ 1 MVA:

| | G1 | G2/G3/G5 | G4/G6 |
|---|---:|---:|---:|
| `x_f + x_tr` [pu] | 0,158 | 0,277 | **0,554** |

Toàn bộ **độ trải điện của feeder** tới GFM gần nhất là **0,0386 pu**. Giao diện của một
converter lớn hơn cả feeder **4 đến 14 lần**. Lưới 4,16 kV này là một tấm đồng so với các bộ
biến đổi treo trên nó — nên mọi toạ độ của Ω_dyn đều do đáp ứng khối quyết định, bất kể mở
tie ở đâu hay đánh sự cố ở đâu.

Đây cũng là cùng một cơ chế đã giải thích phát hiện ở T21 §5 (chia Q do `x_tr` quyết định chứ
không do `x_f`) và kết quả rỗng của T22.

## Hệ quả cho C3 — và ba lối ra

**Ở điểm vận hành này, không toạ độ nào của Ω_dyn mang thông tin topology dùng được.** C3
không sống ở toạ độ nadir *và cũng không* sống ở toạ độ RoCoF. Muốn có phụ thuộc topology thì
phải đưa hệ vào chế độ mà **`μ_I` siết** — đúng chế độ mà T20 cũ vô tình ở trong, nhưng vì lý
do sai (dùng định mức liên tục làm tiêu chí an ninh). Ba cách hợp lệ:

1. **`ImaxF = 1,5`** — vẫn trong dải REGFM_A1 (1,5–3,0), là đáy dải thay vì ví dụ. Dự trữ
   `μ_I` rơi từ 0,32 xuống ~0,10; cần kiểm xem nó có siết trước RoCoF không.
2. **Tăng tải / giảm đội thiết bị** để `μ_I` tiến gần 1 ở cùng ΔP.
3. **Chấp nhận kết quả rỗng và phát biểu nó** — "trên feeder 4,16 kV với đội GFM cỡ này, biên
   an ninh tần số là bất biến cấu hình vì lưới nhỏ hơn giao diện converter một bậc". Đây là
   một *kết quả*, không phải một thất bại, và nó đo được: bảng ② là bằng chứng.

## Sinh lại

```
uv run python experiments/t22_topology_sweep.py --event-buses auto:5 \
    --out artifacts/T23_event_location_sweep
```

`auto:5` chọn 5 bus theo phân vị khoảng cách điện rồi luôn thêm bus mốc (`--event-bus`, mặc
định 76). Cũng nhận danh sách tên bus tường minh: `--event-buses "76,33,102"`.
