# T35 — bất biến topology và vị trí sự cố tại cấu hình ship C

**Vì sao bắt buộc chứ không tuỳ chọn.** T22/T23 kết luận topology và vị trí nhiễu không dịch
được biên, và lý do đưa ra là: *đại lượng siết là đại lượng **khối** (RoCoF, nadir), không phải
cục bộ*. T34 cho thấy ở ship C biên chuyển sang bị siết bởi **`v_min`** — một đại lượng cục
bộ. Tiền đề của lập luận cũ mất, nên kết luận phải được dẫn lại chứ không dùng lại.

Cùng seed (7), cùng số case, cùng mọi cờ. Khác duy nhất là ba default bộ điều khiển (xem T34).

---

## 1. Kết quả thô

| | công bố | ship C |
|---|---|---|
| **topology** (4 case) | 1,185059 × 4 → spread **0,000%** | 1,438574 × 3 + G2 = 1,450098 → spread **0,801%** |
| **vị trí sự cố** (6 case) | 1,185059 × 5 + E102 = 1,173535 → spread **0,982%** | 1,438574 × 6 → spread **0,000%** |
| tiêu chí siết | `rocof` ở mọi case | `v` ở mọi case |

Hai sweep **đổi chỗ cho nhau**: cái từng phẳng thì giờ có một ngoại lệ, cái từng có ngoại lệ
thì giờ phẳng tuyệt đối.

## 2. Cả bốn con số đều là **khuếch đại ngưỡng**, không phải vật lý

Đo độ tản của **chính đại lượng đang siết**, tại biên riêng của từng cấu hình (margin = phần
ngưỡng còn lại):

| | siết bởi | tản margin tại biên | → spread $\Delta P_{\max}$ |
|---|---|---:|---:|
| vị trí, công bố | rocof | 0,00932 | 0,982% |
| vị trí, ship C | v | 0,00426 | **0,000%** |
| topology, công bố | rocof | 0,00220 | **0,000%** |
| topology, ship C | v | 0,00723 | 0,801% |

Spread của biên bám theo tản margin: tản lớn thì phép tiếp tuyến lật một case qua ngưỡng và
hiện thành spread; tản nhỏ thì cả nhóm rơi vào cùng một khoảng bisection. **Với dung sai
bisection 2% và margin dưới 1%, không sweep nào ở đây *phân giải* được bất biến** — 0,000%
không phải bằng chứng bất biến mạnh hơn, và 0,801% không phải bằng chứng vật lý.

Đây đúng là lập luận `T23_event_location_sweep/README.md` §① đã dùng cho ngoại lệ E102
(*"0,07% vật lý biến thành 0,97% biên vì tại điểm góc tiêu chí gần như tiếp tuyến với biên"*).
Giờ có thêm một thể hiện độc lập thứ hai của chính hiệu ứng đó, ở G2 (margin_v = 0,00012 —
gần như tiếp tuyến chính xác), và lần này nó lệch về phía **rộng hơn** (1,450 > 1,439).

## 3. Kết luận sống, nhưng **lời giải thích thì sai**

**Kết luận sống, và mạnh hơn trước.** Cùng một kết luận — topology và vị trí nhiễu không dịch
biên quá mức khuếch đại ngưỡng của một chênh lệch margin dưới 1% — nay đứng vững dưới **hai
tiêu chí siết hoàn toàn khác nhau** (`rocof` khối, `v` cục bộ). Bất biến với việc *cái gì đang
siết* là bằng chứng mạnh hơn hẳn một lần đo dưới một tiêu chí.

**Lời giải thích cũ bị bác.** T22/T23 quy bất biến cho việc đại lượng siết là **khối**. Ở ship
C đại lượng siết là **cục bộ** và biên vẫn bất biến ở cùng mức. Nên cơ chế không phải "đại
lượng siết thuộc loại khối", mà: đáp ứng của đội bị chi phối bởi **chia tải droop tổng hợp**,
và điều đó đúng bất kể tiêu chí nào chạm ngưỡng trước.

Phải viết lại phần giải thích trong bài. Kết luận giữ nguyên; lý do thì không.

## 4. Cái này không trả lời

Không sweep nào phân giải được bất biến dưới mức khuếch đại ngưỡng. Muốn có phát biểu định
lượng thật thì phải **siết dung sai bisection xuống dưới tản margin** (tol ≲ 0,002 thay vì
0,02), tốn khoảng 10× số lần dò. Chưa làm, và **không nên làm trước khi có lý do**: kết luận
định tính đã đủ cho vai trò nó đảm nhận trong bài.

---

`experiments/t35_rerun_sweeps_ship_c.sh` · `topology/` (4 case) · `event_loc/` (6 case)

---

## Phụ chú (T39, 2026-09-05) — nhãn `binding = v` là điều kiện theo ngưỡng

Mọi số trong tài liệu này chạy ở `v_min_pu` = 0,90, một ngưỡng **không có nguồn** — xem
`reference/security_band_provenance.md`. Ngưỡng ship nay là **0,88 pu** (IEEE 1547-2018
Cat III Continuous Operation), và ở đó cả 10 case đổi `binding` từ `v` sang **`nadir`**.

**Kết luận của tài liệu này không đổi**, và §3 mạnh thêm: bất biến topology/vị trí nay đã
đứng dưới **ba** tiêu chí siết khác nhau — `rocof` (cấu hình công bố), `v` (ngưỡng 0,90),
`nadir` (ngưỡng 0,88). Đó chính là luận điểm §3: bất biến với *việc cái gì đang siết*.

Chưa chạy lại vì kết luận không đổi; chỉ nhãn tiêu chí là điều kiện.
