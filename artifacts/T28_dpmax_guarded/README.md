# T28 — chạy lại `ΔP_max` với guard trần P bật: **tái lập bit-exact**

Bản vá `ceiling_representable` (T26 §0) bắt con dấu bạc thứ ba của ANDES — `Pmax` âm bị thay
bằng mặc định 1,0. T20–T25 đã chạy **trước** bản vá. Lần chạy này kiểm chứng bằng phép đo
điều mà trước đó chỉ là lập luận: `ΔP_max` không bị ảnh hưởng.

## Kết luận

`ΔP_max = 1,1850585938 MW` — **giống T21 đến từng bit.**

| kiểm chứng | kết quả |
|---|---|
| Biên | `1,1850585938` ở cả hai, sai lệch **0,00e+00** |
| Khoảng kẹp | `[1,1793 ; 1,1908]` ở cả hai, 14 lần chạy ở cả hai |
| 9 chỉ tiêu × 14 điểm dò | sai lệch lớn nhất **0,000e+00** |
| `secure_flag`, `n_units_saturated`, `tds_converged` | trùng hoàn toàn |
| Dòng `non_negative param <Pmax> corrected to 1.0` trong log | **0** |

Chỉ tiêu đối chiếu: `f_nadir`, `rocof_window`, `v_min`, `v_max`, `μ_I`, `μ_I` liên tục, `μ_P`,
`df_ss`, `f_ss`.

## Vì sao trùng khít

Trong sweep `ΔP_max`, `p_head_mw = None` → headroom bằng toàn bộ định mức BESS (3,414 MW), nên
`Pmax = (p₀ + head)/S_n = (−0,575 + 3,414)/4,3624 = 0,651 > 0` ở mọi điểm dò. Guard không bao
giờ kích hoạt, và ANDES không bao giờ có gì để sửa — **số 0 ở hàng cuối bảng trên là bằng
chứng trực tiếp, không phải suy luận.**

Lỗi chỉ chạm tổ hợp `gen_loss` **cộng** quét `p_head` xuống dưới `−p₀`, tức đúng một hạng mục
là `P_head^min` (T26), và hạng mục đó đã chạy *sau* bản vá.

## Phạm vi đã kiểm lại

| hiện vật | `p_head` | guard có kích hoạt? | cần chạy lại? |
|---|---|---|---|
| T21 `ΔP_max` | None | không | **không — T28 xác nhận** |
| T22 quét tie | None | không | không |
| T23 quét vị trí nhiễu | None | không | không |
| T24 `ImaxF` = 1,5 | None | không | không |
| T25/T27 tắt diesel | None | không | không |
| T26 `P_head^min` | quét | **có** | đã chạy sau bản vá |

## Sinh lại

```
uv run python experiments/t20_andes_bisect.py --what dp_max --event gen_loss \
    --load-p2z 0.0 --q-max 0.60 --out artifacts/T28_dpmax_guarded
```
