# T22 — biên an ninh có dịch theo cấu hình lưới không?

**Câu hỏi:** T21 cho `ΔP_max` do tần số quyết định. Trạng thái dừng droop đặt ra nadir chỉ phụ
thuộc `ΣS_g`, `R`, `f0` — không cái nào mang tính topology. RoCoF (phụ thuộc vị trí nhiễu) và
`μ_I` (phụ thuộc trở kháng tới điểm nhiễu) thì có. Vậy biên có đứng yên không?

## Kết luận một dòng

**Đứng yên: `ΔP_max = 1,1851 MW` ở cả bốn cấu hình, độ phân tán 0,0000 MW.** Nhưng lý do
*rộng hơn* giả thuyết: toạ độ siết là **RoCoF**, không phải nadir — và ở hệ này RoCoF cũng
bất biến theo topology. Đóng/mở tie không dịch được biên; **muốn RoCoF nhạy topology thì
phải dời điểm nhiễu, không phải dời tie.**

## Bảng

| Cấu hình | nhánh mở | ΔP_max [MW] | siết bởi | á quân | khe hở | dự trữ nadir | dự trữ RoCoF | dự trữ V | dự trữ μ_I | dự trữ μ_P |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| G0 | s5 s6 s7 s8 s9 | **1,1851** | RoCoF | nadir | 0,0039 | 0,0044 | 0,0006 | 0,1962 | 0,3241 | 0,6851 |
| G1 | s0 s2 s5 s6 | **1,1851** | RoCoF | nadir | 0,0039 | 0,0064 | 0,0026 | 0,2023 | 0,3299 | 0,6857 |
| G2 | s0 s7 s9 | **1,1851** | RoCoF | nadir | 0,0039 | 0,0066 | 0,0028 | 0,2030 | 0,3305 | 0,6858 |
| G3 | s1 s2 s5 s6 s9 | **1,1851** | RoCoF | nadir | 0,0039 | 0,0052 | 0,0013 | 0,1993 | 0,3292 | 0,6851 |

Dự trữ đã chuẩn hoá: 1 = hệ chưa nhiễu, 0 = đúng trên ngưỡng. Đọc ở **điểm an toàn cuối cùng**
của bisection (ΔP = 1,1793 MW), không phải ở biên — biên không bao giờ được chạy.

## Ba điều bảng này nói mà con số biên không nói

**① Toạ độ siết là RoCoF, không phải nadir.** Ở cả bốn cấu hình, dự trữ RoCoF nhỏ hơn dự trữ
nadir đúng **0,0039** — hằng số đến bốn chữ số. Hai tiêu chí tần số không chỉ gặp nhau, chúng
**song song**: cùng sinh ra từ một đáp ứng droop, nên tỉ lệ giữa chúng không đổi theo ΔP.
Đây là lý do điểm góc bền chứ không phải ngẫu nhiên.

**② Topology *có* tác dụng, nhưng dưới ngưỡng phân giải hai bậc.** Dự trữ RoCoF chạy từ 0,0006
(G0) tới 0,0028 (G2) — thật và có thứ tự vật lý hợp lý: càng ít nhánh mở, lưới càng khoẻ, dự
trữ càng lớn (G2 mở 3, G0 mở 5). Quy ra biên bằng độ dốc cục bộ `d(RoCoF)/d(ΔP) ≈ 1,71 Hz/s
trên MW`:

| | G0 | G1 | G2 | G3 |
|---|---:|---:|---:|---:|
| dịch biên suy ra so với G0 [MW] | 0,00000 | +0,00235 | +0,00258 | +0,00089 |

Tức **0,08–0,22% của biên**, và **nhỏ hơn dung sai bisection (0,02 MW) tám lần**. Nói "bất
biến" là đúng ở mọi thang đo có ý nghĩa kỹ thuật, nhưng phải nói kèm con số này chứ không
tuyên bố bất biến tuyệt đối.

**③ `μ_I` và `μ_P` không ở gần đâu cả.** Dự trữ 0,32–0,33 và 0,685–0,686 ở mọi cấu hình. Ở
vùng này bản đồ chi phối không có gì để lật: chỉ có một cặp tiêu chí cạnh tranh, và chúng
song song.

## Hệ quả cho C3

Giả thuyết "biên nadir bất biến theo topology" **được xác nhận**, nhưng chưa đủ để kết luận
C3 sống ở toạ độ RoCoF: **RoCoF ở đây cũng bất biến**. Lý do là điểm nhiễu cố định (bus 76) ở
cả bốn lần chạy, nên thứ duy nhất thay đổi là trở kháng *giữa* các nhánh, và đáp ứng droop
tổng hợp của đội thiết bị át nó.

**Thí nghiệm tiếp theo phải là quét vị trí nhiễu, không phải quét tie.** Nếu `ΔP_max` dịch
theo bus sự cố nhưng không dịch theo cấu hình tie, thì phát biểu đúng là *"biên phụ thuộc
**vị trí** nhiễu chứ không phụ thuộc **cấu trúc** lưới"* — mạnh hơn và kiểm được.

## Về điểm góc

Hai tiêu chí chạm ngưỡng trong ±0,006 MW. Đây là tính chất của **định nghĩa dải an ninh**
(59,0 Hz và 2,0 Hz/s), không phải của vật lý — đổi một trong hai thì góc dịch. Khe hở dự trữ
0,0039 là hằng số nên có thể phát biểu chính xác: với dải hiện tại, **RoCoF siết trước nadir
một khoảng tương đương 0,0045 MW**; đổi ngưỡng RoCoF lên 2,01 Hz/s là đủ để nadir thành toạ
độ siết.

## Sinh lại

```
uv run python experiments/t22_topology_sweep.py --n 3 --seed 7
```

Cấu hình sinh bởi `TieSwitchReconfiguration` (`src/opt/tie_switch_reconfig.py`), cùng bộ sinh
mà phần RL dùng. Ghi đè `artifacts/topology_generation_diagnostics.json` như tác dụng phụ.
`CaseSpec.open_elements` giữ tập nhánh mở tuyệt đối (`"s<i>"` = mở switch, `"l<i>"` = cắt
line), nên mỗi hàng tái lập được từ `topologies.json`.
