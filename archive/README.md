# archive/

Mã của giai đoạn MAPPO, đưa ra khỏi đường chạy chính theo **plan v3.1 §Phần 1.3**.
**Lưu chứ không xoá**: bài follow-on ở §33 của concept v2 (safe-MARL giữ quỹ đạo trong
Ω_sec) sẽ cần lại, và `gat_encoder.py` cần cho phương án surrogate ở §19.4.

Đường dẫn bên trong giữ nguyên cấu trúc cũ (`archive/src/...`, `archive/experiments/...`)
nên khôi phục chỉ là `git mv` ngược lại.

## Nội dung

| Đường dẫn | Nguồn |
|---|---|
| `src/rl/train_am_mappo.py` | `src/rl/` |
| `src/baselines/` | gcnn_ppo, matd3, mlp_mappo, train_* |
| `src/layer2_control/` | mappo_policy, graph_sage/gat/mlp encoder, actor_critic |
| `src/env/microgrid_env.py`, `microgrid_env_dual.py` | `src/env/` |
| `experiments/train_mappo.py`, `run_asha_*`, `run_ablation.py`, `run_multi_seed.py` | `experiments/` |
| `experiments/lyapunov_certificate.py` | `experiments/` |
| `configs/seeds.yaml`, `configs/training_config.yaml` | `configs/` |

## `_quarantine/` — **không được publish** (đã vào `.gitignore`)

Ba tệp ở §Phần 1.4. Cả ba vốn **untracked**, nên di chuyển vào đây thay vì `rm` để không
mất không hoàn tác được; xoá hẳn khi nhóm nghiên cứu xác nhận.

| Tệp | Lý do |
|---|---|
| `_lyap_vsg.txt` | **Dương tính giả**: in `"CLF FOUND … QED"` trong khi `eta = -0.0055` (âm) và `status=optimal_inaccurate`. Phản biện mở repo thấy tệp này là mất toàn bộ uy tín phần chứng chỉ. |
| `training_config_baseline.yaml` | `entropy_coef: 0.0` cho baseline so với `0.01` cho method. Không trainer nào đọc chúng, nhưng đọc repo sẽ kết luận ngay là gian lận tuning. |
| `training_config_method.yaml` | như trên |

## Đợt 2 — bao đóng truyền ứng (2026-09-02)

Di chuyển mục 1.3 làm gãy import ở tầng trên. Thay vì vá từng tệp, đã tính **bao đóng truyền
ứng** của đồ thị import: mọi tệp phụ thuộc (trực tiếp hoặc gián tiếp) vào các module đã
archive ở đợt 1. Kết quả **50 tệp**, không phải 10 như ước lượng ban đầu:

| Nhóm | Số tệp |
|---|---:|
| `src/` — `env/make_env`, `environment/grid_env`, `eval/{comparison_runner, eval_economics, eval_ffr_topology}`, `opt/precompute` | 6 |
| `experiments/` — `eval_policy`, `test_curriculum_ab` | 2 |
| `scripts/` — chẩn đoán MAPPO/baseline, RoCoF, THD, DSO cost | 29 |
| `tests/` — `test_environment`, `test_day6b_optimizers`, `test_gat_encoder`, `test_graphsage_wiring`, `test_market_reward` | 5 |
| gốc repo — `test_agc`, `test_agc_eval`, `test_env_basic`, `stress_test_day5` | 4 |
| còn lại (bậc 2+) | 4 |

### Đợt 3 — mồ côi `torch` (4 tệp)

Không vào bao đóng vì chúng import `torch` trực tiếp chứ không qua module đã archive, nhưng
cùng một nhánh: `src/rl/networks.py`, `scripts/forensic_ckpt.py`,
`scripts/patch_checkpoint_action_dim.py`, `tests/test_day6_gat_network.py`. Sau đợt này
`src/rl/` rỗng và đã xoá. **Không tệp nào trên đường chạy còn import `torch`**, nên `torch`,
`gymnasium`, `stable-baselines3` đã ra khỏi dependency lõi (xem `pyproject.toml`, nhóm
`mappo`).

### Đợt 4 — import nội bộ không qua `src.` (11 tệp)

Bao đóng đợt 2 chỉ quét tiền tố `src.` nên bỏ sót import kiểu `experiments.*` / `scripts.*`:
`experiments/{eval_generalization, run_benchmarks}`, `scripts/{dso_cost_wilcoxon,
plot_freq_comparison, recompute_settling, regen_table3_severity, test_freq_agc,
tune_all_control, tune_pi_agc}`, `tests/{test_training_smoke, test_eval_topology_selector}`.

## Trạng thái nghiệm thu

- Bao đóng import chạy lại cho **0**.
- Kiểm tĩnh 79 tệp Python còn lại (khớp chính xác, mọi tiền tố nội bộ): **không import nào gãy**.
- `pytest tests --collect-only`: **62 test, 0 lỗi thu thập**.

Danh sách tái sinh được bằng cách quét đồ thị import tới điểm bất động, seed là bốn module
đợt 1 (`src.rl.train_am_mappo`, `src.baselines.*`, `src.layer2_control.*`,
`src.env.microgrid_env*`).

### Đáng lấy lại khi viết `src/emt/metrics.py`

`archive/scripts/measure_rocof_windows.py` — cài đặt RoCoF cửa sổ trượt 100 ms / 500 ms / 1 s
kiểu grid-code, cộng RoCoF đỉnh tức thời. Concept §13 và mục "RoCoF measurement window" ở §25
vẫn đang mở; **phương pháp** ở đây dùng lại được, dù mã buộc vào `microgrid_env_dual` và
`freq_dynamics_lti` (cả hai thuộc diện viết lại ở 1.2).

Đường chạy mới (`src/analytical/`, `src/env/IEEE123bus.py`, `src/opt/tie_switch_reconfig.py`,
`src/opt/l0_reconfig.py`, `src/layer0_dso/reconfiguration.py`, `der_placement_coop.py`,
`src/eval/figures_style.py`) **không** phụ thuộc bất cứ thứ gì ở đây — đã kiểm.
