# T22b — `P_head^min` tại ΔP = 1,1 MW: **không kết luận được**, và lý do đáng ghi

Chạy này nhằm lấy con số `κ` thật đầu tiên gần biên. Nó **thất bại**, nhưng thất bại có chẩn
đoán rõ ràng và cùng họ với phát hiện `Qmax` ở T21.

## Điều xảy ra

Bisection trả về **"an toàn ở cả hai đầu"**: headroom 0,05 MW và 3,414 MW cho nadir, RoCoF,
V_min **giống hệt nhau đến bốn chữ số**. Trần công suất tác dụng không hề tác động.

## Vì sao

Feeder này có tải 3,49 MW và DER 2,88 MW → **tải ròng chỉ 0,61 MW**. Với nhiễu `gen_loss`
1,1 MW, `build_system` đặt điều độ trước sự cố của đội GFM là

```
p_gfm = p_net - p_diesel - p_lost = 0,61 - 0 - 1,1 = -0,49 MW
```

tức **đội thiết bị đang nạp 0,49 MW trước sự cố** — đúng về mặt cân bằng công suất, vì nhiễu
lớn hơn tải ròng. Nhưng trần khi đó là

```
Pmax = (p0 + p_head)/S_n = (-0,49 + 0,05)/4,3624 < 0
```

**`Pmax` âm.** Và `REGF1.Pmax` khai báo `non_negative=True`, nên ANDES **thay giá trị không
dương bằng mặc định của nó là 1,0** rồi ghi một dòng log:

```
REGF1: 6 device(s) had non_negative param <Pmax> corrected to 1.0
```

Trần biến mất hoàn toàn: đo được `ΣPmax = 4,3624` pu hệ (= ΣS_n, tức 1,0 pu thiết bị) ở
head = 0,05, so với 2,9240 ở head = 3,414. **Trần đi ngược chiều headroom.** Đó là lý do cả
hai đầu đều "an toàn".

Đây là **cùng một họ lỗi với phát hiện `Qmax` ở T21**: một giới hạn được khai báo, bị ANDES
âm thầm thay thế, và mô hình chạy tiếp như không có gì. Khác ở chỗ lần này cơ chế là
`non_negative` chứ không phải `adjust_upper`, nên `enforce_q_limits` không bắt được.

## Phạm vi ảnh hưởng

- **`ΔP_max` của T21/T22 không bị ảnh hưởng.** Ở đó `p_head_mw = None` (trần đầy), nên
  `Pmax = (-0,575 + 3,414)/4,3624 = 0,651 > 0` — hợp lệ, không bị sửa. Kiểm được trong
  `metrics.csv`: không có dòng log correction nào, và `μ_P` biến thiên trơn theo ΔP.
- **`P_head^min` cũ (T20, 0,5099 MW)** dùng nhiễu `load_step`, khi đó `p_lost = 0` và
  `p0 = +0,61 MW` → `Pmax > 0`. Kết quả đó vẫn đứng.
- **Chỉ tổ hợp `gen_loss` + quét `p_head` với ΔP > 0,61 MW là hỏng.**

## Ba lối ra — cần bạn chọn, không nên chọn hộ

1. **Đổi nhiễu sang `load_step`** cho `P_head^min`. Đội GFM giữ điều độ dương, `Pmax` hợp lệ.
   Mất tính so sánh trực tiếp với `ΔP_max` (đo trên `gen_loss`).
2. **Nâng `load_scale`** để tải ròng vượt nhiễu. Giữ `gen_loss`, nhưng đổi điểm vận hành nên
   `ΔP_max` phải chạy lại.
3. **Sửa công thức trần** để chịu được `p0 < 0`: headroom *hướng lên* từ một điểm đang nạp
   vẫn có nghĩa, nhưng `Pmax` phải biểu diễn khả năng phát tuyệt đối chứ không phải
   `p0 + head`. Đây là lối ra đúng về vật lý và cũng là lối tốn công nhất.

Cho tới khi chọn: **không có con số `κ`**, và `P_head^min` không trích dẫn được ở chế độ
`gen_loss`.
