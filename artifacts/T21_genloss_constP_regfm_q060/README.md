# T21 — chạy lại `ΔP_max` (gen_loss / P không đổi) theo tham số REGFM_A1

Thay thế `T20_genloss_constP/`. Cùng bố trí, cùng nhiễu, cùng dải an ninh; khác **ba tham số
GFM** và **cách đọc giới hạn dòng**.

## Kết luận một dòng

**ΔP_max: 0,5801 → 1,1851 MW (+104%), và toạ độ siết chuyển từ `μ_I` sang tần số.**
Biên cũ **không phải** biên ổn định — nó là ngưỡng vượt định mức dòng *liên tục*.

## Thay đổi

| Tham số | T20 | T21 | Nguồn |
|---|---:|---:|---|
| `x_f_pu` (X_L) | 0,05 | **0,15** | PNNL-35110 Bảng 1, ví dụ 0,15 (dải 0,05–0,25) |
| `q_max_pu` (Qmax/Qmin) | ±1,00 | **±0,60** | dải 0,44–1,0; xem §3 về vì sao không lấy 0,44 |
| `i_max_f_pu` (ImaxF) | — | **2,00** | ví dụ 2,0 (dải 1,5–3,0) — tiêu chí an ninh |
| `i_cont_pu` | 1,20 (là tiêu chí) | **1,20 (chỉ báo cáo)** | định mức liên tục, tách khỏi Ω_dyn |

## 1. Biên và toạ độ siết

| ΔP [MW] | nadir | RoCoF | V_min | `μ_I`(ImaxF) | `μ_I`(liên tục) | `μ_P` | siết bởi |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1,1793 | 59,004 | 1,999 | 0,9196 | 0,676 | 1,127 | 0,315 | — (an ninh) |
| **1,1851** | — | — | — | — | — | — | **biên** |
| 1,1908 | 58,995 | 2,019 | 0,9188 | 0,680 | 1,133 | 0,317 | nadir + RoCoF |
| 1,5250 | 58,708 | 2,593 | 0,8959 | 0,784 | 1,307 | 0,391 | nadir + RoCoF + V |
| 3,0000 | 54,967 | 10,10 | 0,6311 | 2,278 | 3,796 | 8,719 | tất cả |

Tại biên, **nadir chạm 59,00 Hz và RoCoF chạm 2,00 Hz/s cùng lúc** — hai ràng buộc tần số
gặp nhau trong khoảng ±0,006 MW. `μ_I` = 0,68 và `μ_P` = 0,32: cả dòng lẫn headroom đều còn
xa. `μ_I` chỉ vượt 1,0 từ ΔP ≈ 1,8 MW, tức **sau** biên tần số 52%.

## 2. Vì sao biên cũ là 0,58 MW

Ở T20, `i_max = 1,20` pu vừa là định mức liên tục vừa là tiêu chí an ninh. Cột `μ_I`(liên tục)
ở trên cho thấy nó cắt 1,0 quanh ΔP ≈ 1,0–1,2 MW *sau khi* đã sửa phân bổ Q; ở T20 (Q lệch,
G4 mang 0,82 pu) nó cắt sớm hơn nữa, tại 0,58 MW. Nói cách khác biên T20 đo **quá tải nhiệt
liên tục của một máy bị chia Q lệch**, và gọi nó là biên an ninh tần số.

## 3. Qmax: 0,44 hay 0,60?

Chạy cả hai. **Biên giống hệt nhau: 1,18506 MW** (`T21_..._q044/` và `T21_..._q060/`) — vì
biên do tần số quyết định, không do Q. Chọn 0,60 làm bản chính vì:

- Nhu cầu Q thường trực của feeder ở điểm điều độ này là **1,921 MVAr trên 4,362 MVA** thiết
  bị = **0,4403 pu** — trùng gần như đúng trần ví dụ 0,44 của đặc tả.
- Đặt 0,44 ép **cả sáu máy nằm đúng trên trần** (độ lệch chia Q 0,569 → 0,001 pu, nhưng bằng
  cách bão hoà mọi máy), và GFM Slack kết thúc **cao hơn trần 0,0011 pu** → mỗi lần chạy đều
  in `** Initialization FAILED **` với thặng dư ~1e-4.
- 0,60 vẫn ép phân bổ lại (G4: 0,82 → 0,60), vẫn trong dải đặc tả, và khởi tạo sạch.

## 4. Hai bẫy của ANDES phải vô hiệu hoá thì `Qmax` mới có tác dụng

Khai báo `Qmax` **không** đồng nghĩa với thực thi. Mặc định:

1. `PV.pv2pq = 0` → power flow bỏ qua `qmax`, ghim cả sáu đầu cực ở v = 1,0 bất kể tốn bao
   nhiêu Q.
2. `REGF1.config.adjust_upper = 1` → lúc khởi tạo TDS, ANDES **nới** `Qmax` lên đúng giá trị
   Q mà power flow đưa vào — im lặng, theo từng máy. Đo được: khai báo 0,44 biến thành
   `[0,44 0,49 0,44 0,82 0,46 0,66]`.

`CaseSpec.enforce_q_limits = True` tắt cả hai. GFM Slack cố tình không bị kẹp: nó là nút cân
bằng Q của lưới đảo.

## 5. `x_f` không liên quan đến chia Q

Nâng `x_f_pu` 0,05 → 0,15 là **thay đổi tuân thủ đặc tả, không phải bản vá phân bổ Q**. Đo:
ở cả 0,05 và 0,15, Q ban đầu giống nhau đến bốn chữ số
(`[0,253 0,494 0,389 0,822 0,464 0,663]` pu). Lý do: đầu cực converter là nút PV/Slack ghim
ở v = 1,0, nên chia Q do mạng *phía ngoài* đầu cực quyết định — `x_tr_pu` = 0,06 và feeder —
chứ không do một điện kháng nằm *sau* đầu cực. Nguyên nhân (b) trong `reference/GFM/regfm_a1_mapping.md`
§2.2 sai ở điểm này; nguyên nhân thật là (a), khởi tạo ghim v = 1,0.

## Sinh lại

```
uv run python experiments/t20_andes_bisect.py --what dp_max --event gen_loss --load-p2z 0.0 \
    --q-max 0.60 --out artifacts/T21_genloss_constP_regfm_q060
uv run python experiments/t20_andes_bisect.py --what dp_max --event gen_loss --load-p2z 0.0 \
    --q-max 0.44 --out artifacts/T21_genloss_constP_regfm_q044
```

## Còn thiếu

`μ_I` vẫn là **hậu nghiệm**: `REGF1` không có bộ giới hạn dòng, nên các điểm ΔP ≥ 1,8 MW với
`μ_I` > 1 nằm **ngoài miền hợp lệ** của nền tảng (`beyond_platform = True`) — nhưng chúng đã
nằm ngoài biên tần số rồi, nên không ảnh hưởng tới `ΔP_max`. Khối limiter (bước 5 của
`regfm_a1_mapping.md` §5) vẫn cần cho T6.
