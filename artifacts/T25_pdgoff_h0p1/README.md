# T25b — biến thể `H = 0,1 s`

Bản chính: `artifacts/T25_pdgoff_h1p0/`.

Chỉ khác quán tính diesel: `h_sec = 0.1` thay vì `1.0` (`GENROU.M = 0,3` so với `3,0`, đã xác
minh trong mô hình dựng, `Sn = 1,5` ở cả hai).

**Cả 13 điểm dò giống hệt bản chính đến bốn chữ số. Biên y hệt: `P_DG,off^max = 1,2086 MW`.**

Lý do: máy bị ngắt tại `t_event`, rotor rời hệ cùng lúc, và trước đó hệ ở trạng thái dừng —
quán tính không có gì để tác động. Phép chạy này **không** phân tách được "mất nguồn áp đồng
bộ" khỏi "mất quán tính"; thiết kế sai. Muốn đo đóng góp quán tính: giữ diesel online, đánh
`gen_loss` chỗ khác, quét `H`.

Giá trị của nó: biên tắt diesel **không phụ thuộc lựa chọn `H`** — loại được phản biện "kết
quả phụ thuộc một hằng số quán tính tuỳ chọn" bằng phép đo thay vì bằng lập luận.

```
uv run python experiments/t20_andes_bisect.py --what p_dg_off --load-p2z 0.0 --q-max 0.60 \
    --diesel-bus 76 --diesel-mva 1.5 --diesel-h 0.1 --dg-lo 0.0 --dg-hi 1.4 \
    --out artifacts/T25_pdgoff_h0p1
```
