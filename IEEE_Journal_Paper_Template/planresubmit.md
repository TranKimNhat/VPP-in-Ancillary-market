# Kế hoạch tái cấu trúc và nộp lại — TSG-01727-2026

**Bản thảo gốc:** *Safe Multi-Agent Graph Learning for Fast Frequency Response of Virtual Power Plants in Inverter-Based Islanded Microgrids*
**Quyết định:** Reject (IEEE Transactions on Smart Grid, 22/08/2026)
**Lộ trình đã chọn:** T-PSC — *Topology-Parameterized Safety Certificate*, nhắm lại IEEE TSG
**Ngày lập:** 23/08/2026 · **Trạng thái:** dự thảo, chờ phê duyệt từng giai đoạn

---

## 0. Tóm tắt điều hành

Bản thảo bị từ chối với **16/17 nhận xét là chính xác về mặt kỹ thuật**. Rà soát mã nguồn phát hiện thêm **bốn khiếm khuyết mà cả hai phản biện đều bỏ sót**, trong đó một khiếm khuyết (C1) làm vô hiệu hoá khung FFR của bài. Đồng thời, rà soát mã nguồn cũng cho thấy **ba nguồn lực đã có sẵn** (chứng chỉ Lyapunov mạnh hơn bài đang báo cáo, bộ mã hoá GAT, cấu hình 10 seed) đủ để trả lời một phần đáng kể các nhận xét mà không cần công việc mới.

Nguyên tắc chủ đạo của kế hoạch: **không viết một dòng nào của bản thảo mới trước khi nền mô hình và số liệu được sửa xong.** Bài cũ thất bại một phần vì phần viết chạy trước phần bằng chứng.

### Định vị đóng góp mới

Thay vì học *hành động*, GNN học **ánh xạ từ đồ thị lưới sang tham số chứng chỉ an toàn**:

```
G_t  --f_θ-->  ( K(G_t),   P(G_t),   τ_a(G_t) )
               hộp gain    ma trận    cận
               droop       Lyapunov   dwell-time
```

Chính sách MARL chỉ được phép hành động **bên trong** `K(G_t)` đã được chứng nhận. Ba đóng góp dự kiến:

1. **Chứng chỉ an toàn phụ thuộc topology** — điều kiện đủ để hộp gain `K(G)` giữ Hurwitz trên toàn họ cấu hình reconfiguration, kèm cận dwell-time theo cấu trúc đồ thị.
2. **Chứng minh khả thi vật lý của projection** — phân rã từ không gian GFM về không gian DER thoả SoC / rated power / ramp rate / ràng buộc mạng.
3. **Học GNN cho chứng chỉ, có kiểm chứng chuyển giao liên feeder** — 123-bus → 34/85-bus.

Lý do định vị này thoát được phê phán "lắp ghép kỹ thuật" (R1-1): đóng góp là một kết quả **lý thuyết điều khiển**, learning chỉ là công cụ thực thi. Không công trình nào trong khảo sát §5 phủ nội dung này.

---

## 1. Nguyên tắc làm việc

| # | Nguyên tắc |
|---|---|
| N-1 | **Sửa nền trước, viết sau.** Không soạn văn bản mục nào trước khi số liệu của mục đó đã ổn định. |
| N-2 | **Mỗi tuyên bố trong bài phải truy vết được về một tệp kết quả có thể tái chạy.** Không có con số nào chỉ tồn tại trong bản thảo. |
| N-3 | **Không tuyên bố vượt quá điều code làm.** Mọi giới hạn phạm vi (saturation, delay, tuyến tính hoá) phải được ghi rõ trong bài. |
| N-4 | **Khi chưa chắc chắn về một lựa chọn kỹ thuật, đọc tài liệu trước khi cài đặt.** Mục §5 liệt kê tài liệu cho từng điểm chưa chắc. |
| N-5 | **Con người giữ toàn quyền thẩm định.** Mọi công thức, chứng minh, và diễn giải vật lý phải do nhóm nghiên cứu xác nhận trước khi đưa vào bản thảo. |

---

## 2. Ma trận truy vết: nhận xét → hạng mục công việc

Ký hiệu độ chắc chắn: **[C]** đã chắc, chỉ cần thực thi · **[R]** cần nghiên cứu tài liệu trước · **[D]** cần quyết định của nhóm.

### 2.1 Nhận xét của phản biện

| ID | Nội dung | Workstream | Hạng mục | Chắc chắn |
|---|---|---|---|---|
| R1-1 | Novelty là tổ hợp kỹ thuật có sẵn | WS-D | D1, D2 | [D] |
| R1-2 | Swing + Kron không hợp lệ cho lưới 100% IBR | WS-A | A1, A4, A5 | [R] |
| R1-3 | LMI đỉnh, tính không lồi, dwell-time yếu | WS-B | B1, B2 | [C] |
| R1-4 | Zero-shot không phải chuyển giao topology thật | WS-D | D3 | [C] |
| R1-5a | GCNN-PPO là centralized single-agent → so sánh lệch | WS-D | D4 | [C] |
| R1-5b | Thiếu baseline GAT / Graph Transformer | WS-D | D5 | [C] |
| R1-5c | Fixed droop có retune theo topology hay không | WS-E | E1 | [C] |
| R1-5d | Bất thường MLP-MAPPO (128,7 vs 57,2 €/MWh) | WS-E | E2 | [D] |
| R1-6 | Thiếu λ_imp, λ_cap; thiếu kết quả 3 mức VOLL | WS-E | E3 | [C] |
| R1-7 | Thiếu tham số tải/đường dây/thời điểm nhiễu; thiếu số seed | WS-E | E4 | [C] |
| R1-8 | Lặp luận điểm; §III-A quá dài; thiếu diễn giải embedding | WS-F | F1, F2 | [C] |
| R2-1 | Số liệu abstract ≠ main text ≠ conclusion | WS-F | F3 | [C] |
| R2-2a | Eq (4) chỉ là hàm trừu tượng, không có mô hình tính toán | WS-C | C1, C2 | [R] |
| R2-2b | Không giải thích vì sao chọn GraphSAGE thay vì MPNN khác | WS-D | D5 | [R] |
| R2-3a | ±10% power channel không có biện luận | WS-A | A6 | [D] |
| R2-3b | 1 s không đủ cho FFR dưới giây; 100 ms chỉ là làm mịn | WS-A | **A1** | [C] |
| R2-4 | Projection có khả thi vật lý theo SoC/rating/ramp/mạng? | WS-B | B3, B4 | [R] |
| R2-5 | 96 vertices được dựng thế nào | WS-B | B5 | [C] |
| R2-6 | Giá trị số của (17)–(26) không có trong bảng | WS-E | E5 | [C] |
| R2-7 | 3 seed mà báo cáo 95% CI và p-value rất nhỏ | WS-E | E6 | [R] |
| R2-8 | Ablation: bỏ safety projection; tách riêng hai kênh action | WS-E | E7 | [C] |
| R2-9 | Lỗi ngữ pháp, câu quá dài | WS-F | F4 | [C] |

### 2.2 Khiếm khuyết tự phát hiện (phản biện chưa nêu — sẽ bị bắt ở lần nộp sau)

| ID | Nội dung | Workstream | Hạng mục | Chắc chắn |
|---|---|---|---|---|
| N1 | Bất tương xứng chiều: n_gfm = 2 → plant chỉ 3 trạng thái, phụ thuộc topology chỉ 3 vô hướng, trong khi encoder xử lý ma trận 123×20 | WS-C | C3 | [C] |
| N2 | Trị riêng "38–345 Hz" bị aliasing hoàn toàn ở bước 1 s / 100 ms; đơn vị hoặc tên gọi mode nghi ngờ sai | WS-A | A7 | [R] |
| N3 | Kết luận trình bày "RoCoF thấp hơn 2,3–3,1 lần so với cận vô hướng" như một ưu điểm — trong khi Trujillo (TPWRS 2025) chứng minh mô hình bậc thấp kiểu SG **đánh giá thấp** nadir/RoCoF một cách hệ thống | WS-A | A5 | [C] |
| N4 | §III-D tự mâu thuẫn: dòng 451–452 khẳng định có P chung, dòng 453 thừa nhận LMI đó infeasible | WS-B | B1 | [C] |

---

## 3. Phát hiện từ rà soát mã nguồn

Đường dẫn tương đối tính từ gốc repo.

### 3.1 Khiếm khuyết chặn (blocking)

**C1 — Mô hình bỏ qua toàn bộ transient dưới giây.**
`src/env/freq_dynamics_lti.py:622–628`, docstring `simulate_hires`:

> *"step() advances the state with a single ZOH jump over dt (=dt_fast), which lands on the quasi-steady value and **SKIPS the sub-second nadir transient**. For figures we re-simulate the SAME fast-step at micro_dt = dt/n_sub … This is **READ-ONLY** … step() still performs the real (training-identical) propagation; this only produces extra observation points."*

Hệ quả: agent chưa từng quan sát hay được thưởng trên động học dưới giây; nadir trong Bảng V đến từ đường vẽ hình; tuyên bố "dual-rate … continuous sub-second damping" (`main.tex:455`) không được code hỗ trợ. Đây là câu trả lời **ĐÚNG** cho câu hỏi R2-3b, do chính docstring của dự án đưa ra.
→ **Hạng mục A1. Chặn toàn bộ WS-A, WS-B, WS-E.**

**C2 — Φ được cache theo trung bình vô hướng đã lượng tử hoá của K.**
`src/env/freq_dynamics_lti.py:444–447` và `455–459`:

```python
def _get_k_bin(self, K_droop):
    k_avg = float(np.mean(np.abs(K_droop)))                  # 6 gain → 1 số
    return int(np.clip(np.round(k_avg * self.cache_k_bins), 0, ...))   # cache_k_bins = 5

key = (topology_id, k_bin)
if key in self._phi_cache:
    return self._phi_cache[key]        # trả Φ CŨ, không dựng lại theo K hiện tại
```

Độ phân giải theo `k_avg` là 0,2 (`configs/env_config.yaml: cache_k_droop_bins: 5`). Vector K đầu tiên rơi vào một bin cố định vĩnh viễn Φ cho bin đó. Tính không đồng nhất droop theo từng GFM — nền tảng của `A_f(G_t, K_t)` và của Contribution 2 — bị lượng tử hoá mất khỏi plant. Kênh droop vẫn tác động qua đường injection `ΔP^ffr = ΔP_m − K_m·Δf_t` nhưng gần như không tác động qua đường damping `D(K_t)`.
→ **Hạng mục A2. Chặn WS-B và hạng mục E7.**

### 3.2 Phát hiện có lợi

**C3 — Bài đang công bố nhánh yếu nhất của chính chứng chỉ mình có.**
`experiments/lyapunov_certificate.py` chặt chẽ hơn hẳn §III-D:

- Dòng 281–283: ghi rõ frozen-time Hurwitz là *"PRELIMINARY, necessary not sufficient — the Hurwitz set is non-convex"*. Nhóm **đã biết** đúng điểm R1-3 nêu; chỉ là bản thảo diễn đạt ngược lại.
- Dòng 304–321 (**Level 2**): bổ sung trạng thái tích phân AGC vào `A_f`; mode COI cận biên trở nên có damping; nếu common-P khả thi → *"exponentially stable under ARBITRARY topology AND gain switching; **NO dwell-time needed**"*.
- `τ_a* = 41,6 s` và `μ = 1,69×10¹⁰` đến từ `dwell_time_fallback` (dòng 189–234) — **nhánh dự phòng**. Trong đó `μ = cond(P)` của nghiệm Lyapunov trên hệ dịch chuyển `A + 5×10⁻⁴·I` có mode cận biên → **μ khổng lồ là hiện vật số học**, không phải thước đo rủi ro chuyển mạch.
- `dwell_time_fallback` chỉ dùng **một đỉnh `k_hi` cho mỗi topology** (dòng 209). Do đó câu *"μ bounds the Lyapunov conditioning over the 96 frozen-time vertices"* (`main.tex:453`) **sai về sự kiện**: μ tính từ 24 ma trận, không phải 96.

→ **Hạng mục B1.** Chạy lại và báo cáo Level 2. Nếu khả thi, toàn bộ đoạn ADT biến mất và R1-3 được giải quyết bằng kết quả đã có sẵn.

**C4 — Nguồn gốc "96 vertices" được xác nhận.**
`experiments/lyapunov_certificate.py:66–67, 74–82`: `itertools.product` trên `2^n_gfm` đỉnh hộp gain × số topology. Với `n_gfm = 2`: `24 × 2² = 96`. Đây là câu trả lời trực tiếp cho R2-5.
Với `n_gfm = 6`: `2⁶ = 64` → `24 × 64 = 1536` ma trận; `MAX_VERTEX_GFM = 8` nên vẫn liệt kê đầy đủ. LMI thành 1536 ràng buộc trên `P ∈ ℝ^{12×12}` (`n_state = 11` + AGC).
→ **Hạng mục B5** (viết ra trong bài) và **B6** (kiểm tra ngân sách tính toán).

**C5 — Ablation MLP thực sự đã cân bằng dung lượng; R1-5d đoán sai nguyên nhân.**
`src/baselines/train_mlp_mappo.py:626`: `embed_dim=128,  # match GraphSAGE-MAPPO (ASHA-tuned) so the ablation differs ONLY in encoder`. Đếm tham số với F = 20: MLP encoder ≈ 35 712, GraphSAGE encoder ≈ 38 400 — lệch dưới 8%. Cả hai dùng chung lịch curriculum `AM_PHASES` kể cả hệ số entropy (`train_mlp_mappo.py:399`; `train_am_mappo.py:1685`).
→ Giả thuyết "tuning bất lợi cho MLP" **không có cơ sở trong code**. Nhưng điều đó khiến khoảng cách 2,25× ở Bảng VII **khó giải thích hơn**. → **Hạng mục E2.**
Còn một bất đối xứng nhỏ nên san bằng: MLP = 3 Linear / 2 ReLU; GraphSAGE = 2 lớp message-passing / 1 ReLU.

**C6 — File config cũ sẽ khiến phản biện kết luận ngược lại.**
`configs/training_config_baseline.yaml` có `entropy_coef: 0.0` trong khi `training_config_method.yaml` có `0.01`; cả hai ghi `in_dim: 6, hidden_dim: 32` còn bài nói 20/128. Các YAML này **không được trainer sử dụng**. **Phải xoá hoặc đồng bộ trước khi công bố code** — phản biện mở repo thấy `entropy 0.0` cho baseline sẽ kết luận ngay là gian lận tuning.
→ **Hạng mục E8.**

**C7 — Bộ mã hoá GAT đã tồn tại nhưng chưa báo cáo.** `src/layer2_control/gat_encoder.py`. → **Hạng mục D5**, chỉ cách một lần train.

**C8 — 10 seed đã cấu hình sẵn.** `configs/seeds.yaml`. Bài chỉ dùng 3. → **Hạng mục E6.**

**C9 — Cấu hình 6 GFM đã có.** `src/env/IEEE123bus.py:619` xử lý `g1…g6`; `freq_dynamics_lti` dựng `n_gfm` từ placement, chỉ yêu cầu ≥ 2 (dòng 125). Với `n_gfm = 6`: `n_state = 5 + 6 = 11`, phụ thuộc topology trở thành `J̃_r ∈ ℝ^{6×5}` và `B_net ∈ ℝ^{6×1}` — **36 số thay vì 3**. Đây là thứ làm encoder đồ thị trở nên bảo vệ được, và giải quyết N1.
→ **Hạng mục A3, C3.**

### 3.3 Lỗi nhỏ cần dọn trước khi release code

**C10** — Hằng số ma thuật nhân bản: `experiments/lyapunov_certificate.py:40` `K_BACKBONE = 1.0  # must match env step_fast`; dòng 50 `S_BASE = 15.705`. Đồng bộ bằng tay giữa các module, **sẽ hỏng âm thầm khi chuyển sang 6 GFM**. → **Hạng mục A8.**

**C11** — `src/env/freq_dynamics_lti.py:431`: `D = np.maximum(np.abs(K_droop), 1e-6)`. Bài lập luận `K_min = 0` đảm bảo `D ⪰ 0` *by construction* (`main.tex:441`); code lại đảm bảo bằng `abs()`, tức che lỗi dấu ở thượng nguồn thay vì cưỡng chế ràng buộc. → **Hạng mục B7.**

**C12** — `src/env/freq_dynamics_lti.py:19–22` trích dẫn Trujillo et al. nhưng `main.tex` và `Ref.bib` **không có**. Nguy hiểm gấp đôi vì kết luận trung tâm của Trujillo chính là R1-2. → **Hạng mục A5.**

---

## 4. Kế hoạch theo giai đoạn

### P0 — Sửa nền *(chặn mọi thứ khác)*

| ID | Hạng mục | Ghi chú |
|---|---|---|
| **A1** | Đưa động học dưới giây vào **vòng mô phỏng thật**, không phải đường vẽ hình. `step()` tích phân ở 10–20 ms và trả nadir/RoCoF thật; macro-step điều khiển đặt ở **100 ms** (1 s mâu thuẫn với định nghĩa FFR). | Chặn tất cả. Xem §5.1 trước khi chọn bước tích phân. |
| **A2** | Bỏ cache theo `mean(\|K\|)`. Thay bằng cache theo hash của **vector** K với dung sai chặt, hoặc bỏ cache hoàn toàn nếu chi phí `expm` chấp nhận được với n_state = 11. | Đo chi phí `expm` trước khi quyết định. |
| **A3** | Chuyển sang cấu hình **6 GFM**; kiểm tra `placement` và `J̃_r` dựng đúng. | |
| **A8** | Dọn hằng số nhân bản (`S_BASE`, `K_BACKBONE`) về một nguồn config duy nhất. | Bắt buộc trước A3, nếu không sẽ hỏng âm thầm. |
| **A7** | Kiểm tra lại đơn vị và tên gọi của các mode "38–345 Hz"; xác định chúng có bị aliasing ở bước lấy mẫu mới không. | Xem §5.1. |
| **A6** | Quyết định và biện luận `η_P = ±10%`. | [D] — xem §6. |

**Tiêu chí nghiệm thu P0:** một sự kiện S2 chạy qua `step()` mới cho ra quỹ đạo nadir dưới giây trùng khớp (sai số < 2%) với đường `simulate_hires` cũ, và agent quan sát được quỹ đạo đó.

---

### P1 — Thu hồi chứng chỉ mạnh *(rẻ, hiệu quả cao — nên làm ngay sau P0, hoặc song song nếu muốn thăm dò sớm)*

| ID | Hạng mục | Ghi chú |
|---|---|---|
| **B1** | Chạy `experiments/lyapunov_certificate.py` ở cấu hình 6 GFM; báo cáo **Level 2** (augmented + AGC). Nếu khả thi → xoá toàn bộ đoạn ADT; R1-3 và N4 được giải quyết. | Có thể chạy thử ngay ở 2 GFM để thăm dò trước khi hoàn tất P0. |
| **B2** | Viết lại §III-D: nêu rõ tính không lồi của tập Hurwitz, lý do đối số đỉnh hợp lệ (`A_f` affine theo K), và **báo cáo P tường minh** kèm `λ_min(P)`, `λ_max(P)`, `η`. | |
| **B5** | Viết ra cách dựng đỉnh: `2^n_gfm` đỉnh hộp gain × số topology. Trả lời R2-5. | |
| **B6** | Kiểm tra ngân sách tính toán LMI ở 1536 ràng buộc / `P ∈ ℝ^{12×12}`; nếu SCS quá chậm, cân nhắc MOSEK hoặc giảm số topology trong bước chứng nhận. | [D] |
| **B7** | Thay `np.abs()` bằng cưỡng chế `K ≥ 0` ở thượng nguồn + assertion. | |
| **B3** | **Xây ánh xạ phân rã** `ΔP_ref ∈ ℝ^6 → {ΔP_m}_{m=1}^{41}` thoả SoC / rated / ramp / ràng buộc mạng. | [R] — xem §5.2. Trả lời R2-4. |
| **B4** | Chứng minh (hoặc chứng nhận số) rằng projection (32) luôn cho ra điểm **disaggregable**. | [R] — xem §5.2. |

**Tiêu chí nghiệm thu P1:** tệp `artifacts/lyapunov_certificate_table.csv` ở cấu hình 6 GFM, cùng một mệnh đề duy nhất về ổn định (không "double-claiming"), và một kiểm chứng số cho thấy 100% các action sau projection là khả thi vật lý.

---

### P2 — Mô hình deliverability *(Contribution 1 hiện đang rỗng)*

| ID | Hạng mục | Ghi chú |
|---|---|---|
| **C1** | **Viết ra Eq (4)** — công thức thật cho `R^ffr_{i,t} = f(G_t, SoC, V, S, P, Q)`. Không có nó, Contribution 1 không có phương trình nào. | [R] — xem §5.3. |
| **C2** | Ràng buộc `\|ΔP_{m,t}\| ≤ R^ffr_{m,t}(G_t)` phải được cưỡng chế trong env, có log tỉ lệ ràng buộc hoạt động. | |
| **C3** | Kiểm chứng rằng ở 6 GFM, phụ thuộc topology của plant (36 số) đủ lớn để encoder đồ thị có nội dung mã hoá. Báo cáo định lượng: độ biến thiên của `J̃_r`, `B_net` qua 24 topology. | Trả lời trước N1. |

**Tiêu chí nghiệm thu P2:** Eq (4) là một công thức tính được, có tham chiếu tài liệu, và một hình cho thấy `R^ffr` thay đổi thế nào qua các topology.

---

### P3 — Thực nghiệm

| ID | Hạng mục | Ghi chú |
|---|---|---|
| **E2** | Truy nguyên bất thường MLP-MAPPO. Nếu là lỗi → sửa và chạy lại. Nếu không → giải thích được về mặt vật lý, hoặc rút khỏi bài. | [D] — không được để nguyên. |
| **D5** | Chạy baseline **GAT / GATv2**. Viết lại biện luận chọn encoder: hiện §III-C chỉ biện minh cho *tính inductive*, không cho *GraphSAGE cụ thể*. | [R] — xem §5.4. |
| **D4** | Thay hoặc bổ sung GCNN-PPO bằng một biến thể **CTDE** để so sánh cùng kiến trúc; hoặc nêu rõ trong bài rằng chênh lệch phản ánh kiến trúc chứ không phải encoder. | |
| **D3** | **Cross-feeder**: train trên IEEE 123-bus, test trên 34-bus và 85-bus. Bỏ tuyên bố "zero-shot" nếu không đạt; thay bằng "topology-robust under N-1 and tie-switch reconfiguration". | [D] |
| **E6** | ≥ 20 seed. Thay 95% CI trên 3 seed bằng **interquartile mean + bootstrap CI + performance profile**. Kiểm định lại trên mẫu độc lập, không phải 24 topology phụ thuộc. | [R] — xem §5.5. |
| **E7** | Ablation đầy đủ: (a) bỏ safety projection; (b) chỉ kênh `a^P`; (c) chỉ kênh `a^K`; (d) cả hai. Trả lời R2-8. | Phụ thuộc A2. |
| **E1** | Xác định rõ fixed-droop có retune theo từng topology hay không; báo cáo cả hai biến thể. | |
| **E3** | Công bố λ_imp, λ_cap; chạy sweep VOLL ∈ {1500, 3000, 8000} và báo cáo. Lập luận VOLL-invariance chỉ hợp lệ cho nhóm learning controller (ENS → 0), không cho fixed-droop / no-FFR. | |
| **E4** | Công bố tham số tải, đường dây, thời điểm và thời lượng nhiễu; cách phối hợp contingency × topology trong episode; số seed cho từng test case. | |
| **E5** | Bỏ comment bảng `tab:reward_weights` (`main.tex:350–371`); sửa cross-reference treo ở dòng 345. | Sửa mất vài phút. |
| **E8** | Xoá hoặc đồng bộ `configs/training_config_*.yaml`. | Bắt buộc trước khi release code. |
| **A4/A5** | Spot-check EMT trên Simulink SPS cho 3–4 kịch bản xấu nhất. Trích dẫn Trujillo (TPWRS 2025) và **đối diện** kết luận của họ, không chỉ mượn khung. Viết lại câu ở `main.tex:920`. | [R] — xem §5.1. |

**Tiêu chí nghiệm thu P3:** mọi con số trong bài truy vết được về một tệp kết quả tái chạy được; không còn ô nào trong ma trận §2 ở trạng thái chưa xử lý.

---

### P4 — Viết lại

| ID | Hạng mục |
|---|---|
| **D1** | Đổi tiêu đề — loại bỏ trùng lặp với Yan et al. (TSG 2024). |
| **D2** | Viết lại Introduction quanh đóng góp T-PSC. **Xoá câu ở `main.tex:111`** (*"the novelty lies in the problem, architecture, and evaluation, not in GraphSAGE itself"*) — chính câu này kích hoạt R1-1. Bổ sung trích dẫn Ikram et al. (SEGAN 2026) và các bài TSG trong §5.4. |
| **F1** | Gộp luận điểm "topology-blind" (hiện lặp ở dòng 109, 167, 182, 851) về một chỗ duy nhất. |
| **F2** | Rút §III-A xuống; chuyển các cấu trúc chuẩn (dead-zone, projection, normalisation) sang phụ lục. Bổ sung **diễn giải vật lý của embedding GraphSAGE** và **đường cong hội tụ huấn luyện** — cả hai hiện đều thiếu. |
| **F3** | Thống nhất số liệu: abstract "1,45–1,75×" ↔ conclusion "1,14–1,75×" (`main.tex:920`). Truy nguyên hoặc xoá con số **€1,87/event** trong abstract — hiện không xuất hiện ở bất kỳ đâu khác trong bài. |
| **F4** | Rà soát ngữ pháp toàn bài; sửa câu vỡ ở dòng 288. |
| **F5** | Bổ sung **mục giới hạn phạm vi** nêu rõ điều chứng chỉ *không* bao phủ: anti-windup saturation, độ trễ một bước, chế độ phi tuyến — theo đúng ghi chú trung thực trong `lyapunov_certificate.py:133–137`. |

---

## 5. Danh mục tài liệu cần nghiên cứu

Chỉ liệt kê những điểm **chưa chắc chắn**. Các mục còn lại trong §4 đã đủ thông tin để thực thi.

### 5.1 Mô hình động học và kiểm chứng EMT *(hạng mục A1, A5, A7)*

**Câu hỏi mở:** Mô hình bậc thấp nào là hợp lệ cho lưới 100% GFM? Giao thức kiểm chứng EMT tối thiểu là gì? Bước tích phân bao nhiêu là đủ?

| Tài liệu | Vì sao cần |
|---|---|
| [Analytical Models of Frequency and Voltage in Large-Scale All-Inverter Power Systems](https://consensus.app/papers/details/72d1305ccb395c459e97b7e3e5599fa7/?utm_source=claude_desktop) — Trujillo et al., **IEEE TPWRS 2025**, DOI: 10.1109/tpwrs.2025.3650698 | **Tài liệu quan trọng nhất của toàn kế hoạch.** Chứng minh mô hình bậc thấp kiểu SG *đánh giá thấp nadir và RoCoF một cách hệ thống* trong hệ GFM-dominated; đề xuất mô hình bậc thấp cho cả f và V, **đã validate bằng EMT**. Vừa là lời phê phán chính xác bài của ta, vừa là đường ra. Code đã trích dẫn nhưng bài chưa. |
| [Computationally Efficient Analytical Models of Frequency and Voltage in Low-Inertia Systems](https://consensus.app/papers/details/d5931968269953158b1271091c9d293a/?utm_source=claude_desktop) — Trujillo et al., 2025, DOI: 10.48550/arxiv.2506.06620 | Bản chi tiết hơn về phương pháp và quy trình đối chiếu với EMT cấp công nghiệp. Dùng làm mẫu cho giao thức kiểm chứng của ta. |
| [Interactive Power to Frequency Dynamics Between Grid-Forming Inverters and Synchronous Generators](https://consensus.app/papers/details/ae7bd2f4b78a578aafafcf266bffcf13/?utm_source=claude_desktop) — Kenyon et al., IEEE Systems Journal, DOI: 10.1109/jsyst.2023.3257284 | GFM droop có quan hệ P–f **đảo chiều** và **giảm bậc**; ở tỷ lệ IBR ~96% đáp ứng tiến về bậc nhất. Cần để xác định cấu trúc phương trình đúng, và để xử lý N2 (tên gọi/đơn vị của các mode). |
| [On the Impact of Fault Ride-Through on Transient Stability of Autonomous Microgrids](https://consensus.app/papers/details/25bfa92fd4b65bd7b45f8b8462f7fdcb/?utm_source=claude_desktop) — Eskandari et al., **IEEE TSG 2021**, DOI: 10.1109/tsg.2020.3030015 | Giới hạn dòng của GFM inverter có thể làm microgrid đảo **mất ổn định** (Chetaev instability theorem). Đây là phi tuyến mà mô hình tuyến tính hoàn toàn mù — phải hoặc mô hình hoá, hoặc tuyên bố rõ là ngoài phạm vi (F5). |
| [Modeling and stability issues of voltage-source converter dominated power systems: A review](https://consensus.app/papers/details/519a23ed056f5f0eaf06d8dcb8667c66/?utm_source=claude_desktop) — Xiong et al., CSEE JPES 2020, DOI: 10.17775/cseejpes.2020.03590 | Khảo sát nền: các miền mô hình hoá (time/frequency/energy), khi nào dùng state-space vs impedance vs EMT. Dùng để định vị lựa chọn của ta. |
| [Review of RoCoF Estimation Techniques for Low-Inertia Power Systems](https://consensus.app/papers/details/987643a922b15978bad537c3ec4507a4/?utm_source=claude_desktop) — Deng et al., Energies 2023, DOI: 10.3390/en16093708 | Định nghĩa "center RoCoF", cửa sổ đo, và bản chất số học của RoCoF. Cần để định nghĩa lại chỉ số RoCoF sau khi đổi bước tích phân (A1, A7). |

**Câu hỏi cần trả lời sau khi đọc:** (i) ta giữ mô hình Kron-reduced và sửa, hay thay bằng mô hình Trujillo? (ii) bước tích phân tối thiểu là bao nhiêu để không bỏ sót mode nào ta tuyên bố? (iii) EMT spot-check cần bao nhiêu kịch bản và đo chỉ tiêu nào để được coi là đủ?

### 5.2 Khả thi vật lý của projection và phân rã *(hạng mục B3, B4)*

**Câu hỏi mở:** Làm sao chứng minh một lệnh tổng hợp ở mức GFM luôn phân rã được về 41 DER thoả SoC / rated / ramp / ràng buộc mạng?

| Tài liệu | Vì sao cần |
|---|---|
| [Ramping-aware Enhanced Flexibility Aggregation of Distributed Generation with Energy Storage in Power Distribution Networks](https://consensus.app/papers/details/acdada79383e56878733fae81a456d16/?utm_source=claude_desktop) — Park et al., 2026, DOI: 10.48550/arxiv.2601.14689 | **Trực tiếp nhất.** Xây dựng flexibility envelope **provably disaggregable**, có tính đến ramp limit, kèm chứng minh hình thức. Đây chính là công cụ để trả lời R2-4. |
| [Optimal aggregation and disaggregation for coordinated operation of virtual power plant with distribution network operator](https://consensus.app/papers/details/280e0b95300c53d0ba0706136df762b0/?utm_source=claude_desktop) — Liu et al., Applied Energy 2024, DOI: 10.1016/j.apenergy.2024.124142 | Quy trình aggregation ↔ disaggregation cho VPP với DSO; báo cáo sai số phân rã định lượng (0,63%). Mẫu cho cách ta báo cáo. |
| [Aggregated Feasible Active Power Region for DERs With a Distributionally Robust Joint Probabilistic Guarantee](https://consensus.app/papers/details/2a6cac2b75295f92b8d3d5a3185bbd7c/?utm_source=claude_desktop) — Zhou et al., **IEEE TPWRS 2025**, DOI: 10.1109/tpwrs.2024.3392622 | Xấp xỉ trong (inner approximation) của miền khả thi tổng hợp, đảm bảo cho **cả ràng buộc DER lẫn ràng buộc mạng tuyến tính**. Nếu ta chiếu vào một inner approximation, tính khả thi được đảm bảo theo cấu trúc. |
| [Safe Reinforcement Learning for Strategic Bidding of Virtual Power Plants in Day-Ahead Markets](https://consensus.app/papers/details/309fdfb92c0c5186aecef437dae80047/?utm_source=claude_desktop) — Stanojev et al., 2023, DOI: 10.1109/smartgridcomm57358.2023.10333971 | Projection-based safety shield giới hạn action vào miền khả thi xác định bởi phương trình power flow phi tuyến + ràng buộc DER. Tiền lệ gần nhất về mặt phương pháp. |
| [Data Driven Decentralized Control of IBR Using Safe Guaranteed Multi-Agent DRL](https://consensus.app/papers/details/ebf5d9f145fe5c4abf4a650380344fa2/?utm_source=claude_desktop) — Zhang et al., **IEEE TSTE 2024**, DOI: 10.1109/tste.2023.3341632 | Projection nhúng trong MADRL với tuyên bố "100% safety". Cần đọc kỹ để định vị: họ làm cho ràng buộc **tĩnh** (điện áp); ta làm cho ràng buộc **động** dưới chuyển mạch topology — đây là ranh giới đóng góp của ta. |
| [A Review of Safe Reinforcement Learning Methods for Modern Power Systems](https://consensus.app/papers/details/0758f8a9ae195002b10c364c92dbe69c/?utm_source=claude_desktop) — Su et al., **Proc. IEEE 2024**, DOI: 10.1109/jproc.2025.3584656 | Khảo sát nền cho 4 nhánh safe-RL (Lagrangian, safety layer, CBF, Lyapunov). Dùng để định vị T-PSC trong bản đồ này ở Introduction. |

**Câu hỏi cần trả lời sau khi đọc:** (i) ta chiếu vào miền chính xác hay vào inner approximation? (ii) tính disaggregable chứng minh được hình thức hay chỉ chứng nhận số? (iii) chi phí tính toán của phép chiếu ở mỗi bước 100 ms có chấp nhận được không?

### 5.3 Mô hình deliverability — viết ra Eq (4) *(hạng mục C1)*

**Câu hỏi mở:** Công thức nào cho `R^ffr = f(G_t, SoC, V, S, P, Q)` vừa tính được nhanh, vừa có cơ sở vật lý?

| Tài liệu | Vì sao cần |
|---|---|
| [Modelling and Characterisation of Flexibility From Distributed Energy Resources](https://consensus.app/papers/details/092ca2ea866559879e6cf4e10d07eb8a/?utm_source=claude_desktop) — Riaz et al., **IEEE TPWRS 2021**, DOI: 10.1109/tpwrs.2021.3096971 | Khung *nodal operating envelope* (NOE): capacity, ramp, duration, cost là bốn chỉ số flexibility; dựng qua OPF. **Ứng viên số một cho dạng của Eq (4).** |
| [Robust Dynamic Operating Envelopes for DER Integration in Unbalanced Distribution Networks](https://consensus.app/papers/details/8fe624bd3f225b04ae591b1d12d74d1a/?utm_source=claude_desktop) — Liu et al., **IEEE TPWRS 2022**, DOI: 10.1109/tpwrs.2023.3308104 | DOE cho lưới **ba pha không cân bằng** — đúng đặc thù IEEE 123-bus. Có xử lý bất định tải/phát. |
| [Topology-Constrained Flexibility Assessment of Adjustable Resources in the Regional Electricity Spot Market](https://consensus.app/papers/details/0b6ed322ae365020b13d5b319d16aca4/?utm_source=claude_desktop) — Zhan et al., Energies 2026, DOI: 10.3390/en19112501 | Định lượng "capacity restriction effect": ràng buộc topology làm giảm dự trữ khả dụng thực tế rất mạnh. **Đây chính là luận điểm vật lý của Contribution 1 của ta**, đã được chứng minh độc lập — nên trích dẫn để củng cố. |
| [Quantifying the effects of MV–LV distribution network constraints and DER reactive power capabilities on aggregators](https://consensus.app/papers/details/a038c5f3cf095d7eac8065bdb7454658/?utm_source=claude_desktop) — Gutierrez-Lagos et al., IET GTD 2021, DOI: 10.1049/gtd2.12152 | Chứng minh định lượng rằng bỏ qua ràng buộc mạng làm **đánh giá quá cao** dịch vụ mà aggregator cung cấp được. Bằng chứng trực tiếp cho tiền đề của bài. |
| [Distribution System Flexibility Characterization: A Network-Informed Data-Driven Approach](https://consensus.app/papers/details/980cfaa149015ee7abbb2c10957e87aa/?utm_source=claude_desktop) — Li et al., **IEEE TSG 2023**, DOI: 10.1109/tsg.2023.3328159 | Phương pháp data-driven ước lượng miền công suất khả thi nhanh. Nếu OPF quá chậm cho vòng lặp 100 ms, đây là lối thoát. |
| [Feasible operation region estimation of virtual power plant considering heterogeneity and uncertainty of DERs](https://consensus.app/papers/details/177441dc503854418fc19937b988d667/?utm_source=claude_desktop) — Chen et al., Applied Energy 2024, DOI: 10.1016/j.apenergy.2024.123000 | FOR cho VPP với DER dị thể, dạng polytope. Phù hợp cấu trúc 3 VPP zone của ta. |

**Câu hỏi cần trả lời sau khi đọc:** (i) `R^ffr` là một envelope tính trước theo topology, hay tính online? (ii) dùng độ nhạy tuyến tính hoá hay OPF đầy đủ? (iii) chi phí tính toán ở 24 topology × 41 DER là bao nhiêu?

### 5.4 Lựa chọn encoder và chuẩn tổng quát hoá topology *(hạng mục D3, D5, D2)*

**Câu hỏi mở:** Vì sao GraphSAGE chứ không phải GAT/GATv2/Graph Transformer? Chuẩn hiện tại của "topology generalization" là gì?

| Tài liệu | Vì sao cần |
|---|---|
| [Generalization of Graph Neural Network Models for Distribution Grid Fault Detection](https://consensus.app/papers/details/1e2845622b6350ad9015ae1db27bd5f8/?utm_source=claude_desktop) — Karabulut et al., 2025, DOI: 10.1109/smartgridcomm65349.2025.11204594 | Benchmark GraphSAGE **vs GATv2** trên **IEEE 123-node**, báo cáo suy giảm F1 định lượng khi đổi topology, và kết luận **GATv2 tổng quát hoá tốt hơn**. Bất lợi cho lựa chọn hiện tại của ta — phải đọc kỹ và hoặc phản biện có căn cứ, hoặc đổi encoder. |
| [Universal Graph Learning for Power System Reconfigurations: Transfer Across Topology Variations](https://consensus.app/papers/details/df13fd43a5a95ffea6b3b0bcd4146e4e/?utm_source=claude_desktop) — Wu et al., 2025, DOI: 10.48550/arxiv.2509.08672 | Chuyển giao sang topology khác hẳn về cấu trúc **và số chiều**, không retrain. Đây là chuẩn mà tuyên bố "zero-shot" hiện được đo. |
| [Towards Generalization of Graph Neural Networks for AC Optimal Power Flow](https://consensus.app/papers/details/12d5fa9b740559e08974df4948b89576/?utm_source=claude_desktop) — Arowolo et al., 2025, DOI: 10.48550/arxiv.2510.06860 | **Zero-shot N-1** với gap < 3%, size generalization 14 → 2000 bus. Mẫu cho giao thức cross-feeder của ta (D3). |
| [Graph reinforcement learning for power grids: A comprehensive survey](https://consensus.app/papers/details/3272e2a2379d5d678573b66eb3fa09bd/?utm_source=claude_desktop) — Hassouna et al., Energy and AI 2024, DOI: 10.1016/j.egyai.2025.100671 | Khảo sát nền để định vị đóng góp trong Introduction và tránh lặp lại tuyên bố novelty sai. |

**Bài bắt buộc phải trích dẫn và phân biệt trong Introduction (hiện đều thiếu):**

| Bài | Vì sao bắt buộc |
|---|---|
| [Multi-Agent Safe Graph Reinforcement Learning for PV Inverters-Based Real-Time Decentralized Volt/Var Control](https://consensus.app/papers/details/790c6306bfc15c9ba65c2b2f50e096b3/?utm_source=claude_desktop) — Yan et al., **IEEE TSG 2024**, DOI: 10.1109/tsg.2023.3277087 | Tiêu đề va chạm gần như từng chữ với bản thảo của ta, trên cùng tạp chí. Gần như chắc chắn là bài mà R1 nghĩ tới. |
| [A novel multi agent deep reinforcement learning framework for fast frequency response in inverter based hybrid power plants](https://consensus.app/papers/details/13d2b582cff854d9b184cae2c5e9963f/?utm_source=claude_desktop) — Ikram et al., SEGAN 2026, DOI: 10.1016/j.segan.2026.102282 | **Nguy hiểm nhất.** MAPPO điều chỉnh thích nghi droop gain, virtual inertia, damping cho **FFR** trong hệ IBR-dominated. Rất gần "joint action space" mà ta tuyên bố là mới. |
| [Graph Multi-Agent Reinforcement Learning for Inverter-Based Active Voltage Control](https://consensus.app/papers/details/01d604bafff352f2859ca2b14698327b/?utm_source=claude_desktop) — Mu et al., **IEEE TSG 2024**, DOI: 10.1109/tsg.2023.3298807 | GCN + MARL + barrier function trên cùng tạp chí. |
| [An emergency control strategy for undervoltage load shedding: A graph deep reinforcement learning method](https://consensus.app/papers/details/759c1d6d6aca5d1e9edc9f7742bf22e9/?utm_source=claude_desktop) — Pei et al., 2023, DOI: 10.1049/gtd2.12795 | **GraphSAGE-D3QN** với tuyên bố tường minh về "unseen topology variation". Bài đã trích dẫn (`pei2023graphsage`) nhưng chưa phân biệt đủ. |
| [A Graph-based Deep RL Framework for Autonomous Power Dispatch on Power Systems with Changing Topologies](https://consensus.app/papers/details/4f1cb1ff3268597ea84793b138904cea/?utm_source=claude_desktop) — Zhao et al., 2022, DOI: 10.1109/ispec54162.2022.10033001 | **GraphSAGE nhúng vào PPO** cho topology thay đổi. Tiền lệ trực tiếp cho lựa chọn kiến trúc. |
| [Optimized Unsymmetrical Per-Phase Droop for Soft Line Switching of Reconfigurable Unbalanced Inverter-Based Islanded Microgrid](https://consensus.app/papers/details/1e73f779c3325817a2343e64d15f600b/?utm_source=claude_desktop) — Yousri et al., **IEEE TPWRS 2024**, DOI: 10.1109/tpwrs.2023.3312406 | Bài gần nhất về droop dưới line-switching trong microgrid đảo — nhưng là tối ưu hoá, **không** learning. Đây là ranh giới đóng góp của ta. |

### 5.5 Giao thức thống kê *(hạng mục E6)*

**Câu hỏi mở:** Bao nhiêu seed là đủ? Báo cáo chỉ số nào? Kiểm định nào hợp lệ khi 24 topology phụ thuộc lẫn nhau?

| Tài liệu | Vì sao cần |
|---|---|
| [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://consensus.app/papers/details/211124009c2e53cea6dd07d6d77b65b1/?utm_source=claude_desktop) — Agarwal et al., 2021, DOI: 10.48550/arxiv.2108.13264 | **Tiêu chuẩn de facto.** Interquartile mean, performance profile, stratified bootstrap CI. Kèm thư viện `rliable`. Đây là câu trả lời trực tiếp và có thẩm quyền cho R2-7. |
| [How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments](https://consensus.app/papers/details/dcf47fd9ff51592891d57177752fdccf/?utm_source=claude_desktop) — Colas et al., 2018, DOI: 10.48550/arxiv.1806.08295 | Hướng dẫn lý thuyết để chọn **số seed** theo effect size và mức sai lầm mong muốn. Dùng để biện minh con số seed cuối cùng trong bài, thay vì chọn tuỳ ý. |
| [A Hitchhiker's Guide to Statistical Comparisons of Reinforcement Learning Algorithms](https://consensus.app/papers/details/a32c1c9a442756f998dd8029d04debce/?utm_source=claude_desktop) — Colas et al., 2019, DOI: 10.48550/arxiv.1904.06979 | So sánh thực nghiệm các kiểm định về false-positive rate và power; **độ bền khi vi phạm giả thiết** (chuẩn, cùng phân phối, cùng phương sai). Cần vì 24 topology của ta phụ thuộc mạnh. |
| [Empirical Design in Reinforcement Learning](https://consensus.app/papers/details/e58a02c14bf35f21826d9662dc7d5792/?utm_source=claude_desktop) — Patterson et al., 2023, DOI: 10.48550/arxiv.2304.01315 | Tài liệu tham chiếu toàn diện: giả thiết thống kê sau các chỉ số phổ biến, cách đặc trưng hoá biến thiên, so sánh nhiều agent, xử lý hyperparameter. |
| [Hyperparameters in Reinforcement Learning and How To Tune Them](https://consensus.app/papers/details/fc8618ea498c535e897982029c769221/?utm_source=claude_desktop) — Eimer et al., 2023, DOI: 10.48550/arxiv.2306.01324 | **Tách seed tuning khỏi seed test.** Bài hiện dùng ASHA nhưng không nói rõ tách hay không — phản biện sau sẽ hỏi. |

**Ghi chú về R2-7:** kiểm định Wilcoxon hiện chạy paired trên **24 topology**, không phải 3 seed; với n = 24, p tối thiểu two-sided là 1,19×10⁻⁷ và bài báo cáo đúng 1,2×10⁻⁷ — **con số hợp lệ**. Vấn đề thật nằm ở hai chỗ khác: (a) 95% CI trên 3 seed không bào chữa được; (b) 24 topology đều sinh từ G₀ với Jaccard ≤ 0,18 nên **phụ thuộc mạnh, vi phạm giả thiết mẫu độc lập**. Cần nêu rõ cả hai trong response letter thay vì im lặng.

---

## 6. Quyết định còn treo

| # | Quyết định | Ảnh hưởng | Thông tin cần |
|---|---|---|---|
| Q1 | Giữ mô hình Kron-reduced và sửa, hay thay bằng khung Trujillo? | Toàn bộ §II và mọi số liệu | Đọc §5.1 trước |
| Q2 | Macro-step điều khiển: 100 ms hay nhỏ hơn? | A1, chi phí huấn luyện | Đọc §5.1; đo chi phí |
| Q3 | `η_P = ±10%` — giữ, đổi, hay biến thành tham số quét độ nhạy? | R2-3a | Xem tài liệu grid code (IEEE 2800-2022, ENTSO-E) |
| Q4 | Chiếu vào miền chính xác hay inner approximation? | B3, B4, chi phí online | Đọc §5.2 |
| Q5 | Nếu MLP-MAPPO không truy được nguyên nhân — giữ có chú thích, hay rút khỏi bài? | E2, Bảng VII | Sau khi chạy E2 |
| Q6 | Cross-feeder 34/85-bus: dựng mới hay dùng bộ dữ liệu có sẵn? | D3, khối lượng công việc | Khảo sát dữ liệu |
| Q7 | Nếu Level 2 (augmented CLF) infeasible ở 6 GFM — kế hoạch dự phòng? | B1, Contribution 1 | Sau khi chạy B1 |
| Q8 | Số seed cuối cùng (≥ 20 là tối thiểu; power analysis có thể đòi hơn) | E6, chi phí tính toán | Đọc §5.5 |

---

## 7. Thứ tự thực hiện đề xuất

```
P0  A8 → A3 → A1 → A2 → A7 → (A6 chờ Q3)
        │
        ├─ có thể chạy B1 thăm dò song song ở cấu hình 2 GFM
        │
P1  B1 → B2 → B5 → B6 → B7   [song song: đọc §5.2 → B3 → B4]
        │
P2  đọc §5.3 → C1 → C2 → C3
        │
P3  E2, E8, E5 (độc lập, làm sớm)
    D5, D4, D3, E1, E3, E4 (sau P2)
    E6, E7 (cuối, cần mọi thứ ổn định)
    A4/A5 (EMT spot-check, sau P0)
        │
P4  D1, D2 → F1…F5
```

**Nguyên tắc chặn:** không bắt đầu P3 trước khi P0 và P1 đạt tiêu chí nghiệm thu. Không bắt đầu P4 trước khi mọi ô trong ma trận §2 đã xử lý.

---

## 8. Việc có thể làm ngay, chi phí thấp, không phụ thuộc gì

Bốn hạng mục sau độc lập với P0 và nên hoàn thành trong tuần đầu:

- **E5** — bỏ comment bảng `tab:reward_weights` (`main.tex:350–371`), sửa cross-reference treo ở dòng 345.
- **E8** — xoá hoặc đồng bộ `configs/training_config_*.yaml`.
- **F3** — truy nguyên hoặc xoá con số €1,87/event; thống nhất 1,45 vs 1,14.
- **B1 (thăm dò)** — chạy `lyapunov_certificate.py` ở cấu hình hiện tại để biết Level 2 có khả thi hay không, trước khi đầu tư vào P0.

---

*Tài liệu này do trợ lý nghiên cứu AI soạn dựa trên phản biện, bản thảo `main.tex`, và mã nguồn dự án. Mọi công thức, chứng minh, diễn giải vật lý và quyết định kỹ thuật cần được nhóm nghiên cứu thẩm định độc lập trước khi đưa vào bản thảo.*
