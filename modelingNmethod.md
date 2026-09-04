# Modeling & Method — Detailed Specification

> Working reference for completing the paper. Equations/parameters transcribed from the
> current code (post VSG-unification, 2026-06-01). File:line pointers given so each claim is
> checkable. Where two implementations exist, the **active** one is marked.

---

## 0. System overview

- **Test system:** modified islanded **IEEE 123-bus** distribution feeder, **100 % inverter-based**
  (no synchronous machines, no bulk-grid slack). `S_base = 15.705` MW, `f_0 = 50` Hz.
- **Two inverter roles** (see `architecture_gfm_backbone_vs_gfl_vpp`):
  - **GFM backbone (6 units):** virtual-synchronous-generator (VSG)-controlled grid-forming
    inverters that *form* the grid (set V and f). They carry the **virtual inertia** → govern RoCoF.
    Placement: `official_placement_v3.json["gfm"]` (G1 = VSG, others = droop-GFM).
  - **VPP fleet (41 agents):** grid-**following** (GFL) DERs — BESS, V2G, DPV. They inject power
    and a learned droop term; they do **not** emulate inertia. These are the RL agents.
- **Control objective:** fast frequency response (FFR) on a 1 s fast step, optimizing
  post-event **nadir / IAE / settling** within a safe envelope; RoCoF is backbone-governed
  (**Approach-C**, `mdp_reward_design_approach_c`).
- **Frequency-security indices:** nadir, max RoCoF, IAE, FFR success rate vs ENTSO-E UFLS
  (pre-UFLS 49.5 Hz, UFLS-1 49.0 Hz; RoCoF ride-through 2.0 Hz/s, IEEE 1547-2018 Cat. III).

---

## 1. Frequency simulation model (VSG-LTI) — ACTIVE

File: `src/env/freq_dynamics_lti.py` (class `LTITopologyFreqDynamics`). The legacy scalar-COI
SG model (`freq_dynamics.py`) was **deleted**; this is now the single source of truth.

### 1.1 Per-unit VSG primitive (building block)
Each GFM unit *i* is VSG-controlled, emulating the swing equation of a synchronous machine
`[hou2020adaptive, fathi2018extended, shadoul2022vsg, gurski2024aid]`:
```
2 H_i · dΔω_i/dt = ΔP_set,i − ΔP_e,i − D_i · Δω_i
dΔδ_i/dt        = ω0 · Δω_i ,      ω0 = 2π f0
```
- `H_i` = **virtual inertia** (programmable control parameter, not rotating mass).
- `D_i` = damping = the (learned) droop gain `K_i` (droop realised as swing damping).
- `ΔP_e,i` = angle-dependent electrical-power deviation set by the network (`= [J̃_r Δδ]_i`).
- **Consequence (i):** immediate post-disturbance RoCoF `= ΔP / (2H)` — inertia-set, not
  alterable by any downstream power command in the first step. **Consequence (ii):** `D_i`
  shapes post-nadir settling.

### 1.2 Topology-aware reduced state-space

State `x = [Δδ_rel (n_g−1), Δω (n_g)]ᵀ`, relative angles w.r.t. the largest-rated GFM.
```
dx/dt = A_f(G_t, K_t) x + B_ref ΔP_ref + B_net ΔP_imb
A_f = [[ 0,                 ω0 · 1⁻¹       ],
       [ −diag(1/2H)·J̃_r,   −diag(1/2H)·D  ]]      D = diag(K_eff_i)
B_ref|_ω = diag(1/2H)
```
- `1⁻¹` = reference-subtraction (relative-angle) map; `J̃_r(G_t)` = **Kron-reduced** active-power
  Jacobian of the energized feeder graph (eliminate passive buses)
  `[luo2014kronmicrogrid, ajala2023gfmreduction, abdolmaleki2020onlinekron]`:
  `J̃_r = J_II − J_IL J_LL⁻¹ J_LI` (`_kron_reduce`, freq_dynamics_lti.py).
  **Topology enters through `J̃_r`**; the **learned droop enters through `D`**.
- `H_i` default by `mode`: VSG → 2.0 s, Droop → floor 0.5 s (`_parse_gfm_placement`); aggregate
  `h_sys = Σ(H_i S_i)/S_base ≈ 1.34 s` (property `h_sys`, consumed by the MPC correction).

### 1.3 Discretization, read-out, secondary control
- **ZOH** (Δt = 1 s ≫ inverter τ): `x_{t+1} = Φ x_t + Γ u_t`, `Φ = e^{A_f Δt}`,
  `Γ = ∫₀^Δt e^{A_f τ}dτ · B`, computed by the augmented-matrix exponential and **cached per
  (topology, K-bin)**.
- **COI read-out:** `Δf_t = f0 · Σ S_r Δω_r / Σ S_r`, `ḟ_t = (Δf_t − Δf_{t−1})/Δt`.
- **Secondary (AGC):** slow integral `ζ_{t+1} = sat(ζ_t + k_agc Δf_t Δt)`, fed back to ΔP_ref by
  rating share, runs **concurrently** with primary (removes steady-state offset).
- **Assumptions (state explicitly in paper):** (i) small-signal linearity around the operating
  point (`J̃_r` at converged PF); (ii) GFM voltage regulation fast vs Δt → inner V/I loops and EMT
  out of scope; (iii) relative-angle form leaves one near-marginal COI mode, AGC-stabilized.

### 1.4 Nadir safety layer (in-the-loop) — see §7
Closed-form minimal-perturbation projection of `ΔP_ref` so the **predicted** next-step COI Δf
stays in `[−δ, +δ]` (δ = `nadir_margin_hz` = 0.5 Hz). Affine predictor `Δf̂_{t+1} = a + bᵀ ΔP_ref`
built to mirror `step()` exactly; single half-space ⇒ closed form, no QP. `nadir_safe_projection`.

---

## 2. DER source models

Files: `src/env/evcs_model.py`, `src/layer1_vpp/virtual_battery.py`. Two battery models exist; the
**simple SoC model** governs the env BESS/V2G dispatch, the **EVBatteryModel (ECM+VSG)** is the
per-EV electrochemical model used by the MPC correction.

### 2.1 BESS (simple SoC model — env dispatch)  `evcs_model.py:44-61`
```
discharge (P>0): SoC ← SoC − (P · dT) / (η_dis · E_mwh)
charge   (P<0):  SoC ← SoC + (|P| · η_ch · dT) / E_mwh
SoC ← clip(SoC, 0.10, 0.90)
```
η_ch = η_dis = 0.95, dT = 0.25 h.

### 2.2 EV battery (ECM + VSG)  `evcs_model.py:465-577`
Equivalent-circuit model: two RC branches (diffusion), terminal voltage
`V_bat = E0 − I_b R0 − Uc − Uc1`; SoC update with efficiency on `E_cap`. VSG params `J_vsg=0.5`,
`Kd_vsg=20`. SoC band `[0.20, 0.90]`. Power bounds from SoC headroom & departure time:
`P_max = min((SoC−SoC_min)E_cap/dt_rem, P_rated)`; DCoB/CCoB = dischargeable/chargeable power
until departure.

### 2.3 V2G  `evcs_model.py:78-96`
Like BESS plus **availability**: EV must be present (`departure_step > step`) and `SoC ≥ 0.25` to
discharge; usable capacity capped at 50 % of stored energy; first-order lag `T_v2g = 0.3 s`
(vs BESS 0.1 s). EVs leave the fleet at their departure step.

### 2.4 DPV  `microgrid_env_dual.py:505-510, 886-902`
Curtailment-**only** (already at MPPT): up-regulation forced to ≤ 0; `P ∈ [0, 2·P_rated]`. No lag.

### 2.5 MPC correction (battery feed-forward)  `evcs_model.py:592-656`
`mpc_correction(evcs_list, Δf, ḟ, H_sys=1.432, S_base=15.7, p=3, dt=1, α=1, β=0.1, deadband=0.1)`.
Predictive horizon `p=3`, swing prediction `Δf_{k+1}=Δf_k + (ΣΔP·dt)/(2 H_sys S_base)`, objective
`J = α Σ Δf_k² + β Σ ΔP_i²`, SLSQP over per-agent power bounds. **Note:** `H_sys` now sourced from
the VSG model's `h_sys` (≈1.34) in the env, not the hardcoded default.

---

## 3. VPP structure

File: `src/env/microgrid_env_dual.py`, `src/layer1_vpp/virtual_battery.py`.

- **3 VPP zones** (`_vpp_droop_agents`, microgrid_env_dual.py:140-150):
  - VPP_1 = BESS{9,10,11} + V2G{18,19,20}; VPP_2 = BESS{12-14}+V2G{21-23}; VPP_3 = BESS{15-17}+V2G{24-26}.
- **Fleet (41 agents)**: EVCS_PV 0-8 (9), EVCS_BESS 9-17 (9, → `_bess_indices`), EVCS_V2G 18-26
  (9, → `_v2g_indices`), DPV 27-40 (14, → `_pv_indices`). Per-agent `_agent_p_rated`, bus
  `_agent_bus_pp` from placement JSON.
- **SoC-weighted inter-VPP coordination** (`_compute_vpp_soc_weights`): `w_vpp = clip(avg_SoC/0.5,
  0.2, 2.0)` — higher SoC ⇒ more FFR.
- **Virtual battery** (`virtual_battery.py`): VPP-level aggregate SoC sim + reserve limits
  (power/SoC-headroom/inverter-headroom), used by Layer-1 reserve accounting.

---

## 4. MDP formulation

### 4.1 Agents & timescale
41 GFL DER agents (CTDE). Fast step Δt = 1 s, 300 steps/episode. (Slow step exists for
compatibility; FFR is the fast-timescale focus.)

### 4.2 Observation  `get_am_obs` + `build_am_full_feeder_obs` (train_am_mappo.py)
Per-agent **16 features**: [0] Δf (/0.5*), [1] RoCoF (/1.0*), [2] p_net, [3] SoC/DCoB,
[4] zone_LMP/100, [5-8] type one-hot (PV/BESS/V2G/DPV), [9] VPP membership (vpp+1)/3,
[10] K_droop_prev/K_max, [11] P_ref_prev, [12] SoC-lo-violation, [13] SoC-hi-violation,
[14] P_forecast (persistence), [15] departure proxy. *Note: the **active** obs uses per-bus Δf/RoCoF
from `freq_dyn_lti` (per-GFM), broadcast/averaged onto buses.
Placed on the **123-bus graph** at `_agent_bus_pp`; **+4 grid-background** features per bus
[16] p_load/100, [17] q_load/100, [18] vm_pu(+Δ), [19] agent-density ⇒ `obs_full ∈ (123, 20)`.

### 4.3 Action (dual, per-DER)
Policy emits `a_i = [a^P_i, a^K_i] ∈ [−1,1]²` per DER.
- **Power:** `ctrl_p = −a^P` (sign flip; same on train & eval). Env power map: `ΔP_MW =
  a^P · 0.1 · P_rated` (±10 % rated per unit). κ-clipped to device ratings/SoC headroom.
- **Droop:** `K_i = K_min,i + (1+a^K_i)/2 · (K_max,i − K_min,i)`, with **K_min = 0** (monotone).
  Per-type max: BESS `0.30·P_rated`, V2G `0.20·P_rated`, DPV `0.15·P_rated`, EVCS-PV `0`.
  SoC-mask: `K_i = 0` if SoC ∉ **[0.25, 0.85]** (FFR-eligibility band, nested inside physical [0.20, 0.90]).
- **Env action vector (mappo_dual):** layout `[a^P×41, a^K×41, VPP×3] = 85` (the legacy VPP-K
  slots are 0 in dual mode). (`action_space_fast` is a nominal `(44,)` Box; the dual contract is 85.)
- Slew-rate limit 0.2/step on a^P.

### 4.4 Reward  `compute_am_reward` + `AMRewardConfig` (train_am_mappo.py)
`r = r_Δf + r_rocof + r_violation + r_effort + r_tracking + r_nadir + r_soc + r_market + r_commit + r_ffrbonus`
(all error terms normalized to [0,1] then weighted; penalties negative, bonus/market positive):

| term | weight | form (normalized error e, ref) |
|---|---|---|
| Δf deviation | `w_Δf=0.9` | `e=clip((|Δf|−0.02)/3.0,0,1)` |
| RoCoF | `w_rocof=0.0` | **disabled (Approach-C)** |
| hard violation | `w_violation=0.4` | `e=clip((|Δf|−0.5)/0.5,0,2)` |
| nadir/zenith | `w_nadir=0.6` | excursions below 49.5 / above 50.5 Hz, /2.5 |
| control effort | `w_effort=0.05` | mean(a/0.75)² + 0.3·Δa² |
| tracking | `w_tracking=0.0` | disabled in dual mode |
| FFR bonus | `w_ffr_bonus=0.4` | `+ clip(1−|Δf|/0.5,0,1)` |
| SoC penalty | `w_soc=0.08` | quadratic outside [0.20,0.85] |
| market | `w_market=0.05` | `+clip(energy_rev + cap_rev,−1,1)`; LMP/100, AS-FFR/20 |
| K commitment | `w_commit=0.06` | mean((ΔK/K_max)²) |

Key: refs scaled to **event magnitudes** (Δf_ref=3.0, rocof_ref=3.5, nadir_ref=2.5). `w_rocof=0`
because peak RoCoF is backbone-governed. Reward passed through a running `RewardNormalizer`
(γ=0.99, clip ±10).

### 4.5 Transition & events
Fast step → event injection (`EventInjector`): load_step / gen_trip / line_trip / high_ren, with
per-phase magnitude/probability (curriculum). FFR activation gate: `|Δf|>0.2` or `|ḟ|>1.0`,
hysteresis off at `|Δf|<0.1`; severity scaling `γ = 1 + 0.5·clip(|ḟ|/0.5, 0, 3)`.

---

## 5. GraphSAGE-MAPPO architecture  (proposed method)

File: `src/rl/train_am_mappo.py` (encoder/agent), obs builder above.

- **Encoder (inductive GraphSAGE, 2 layers) `[hamilton2017graphsage]`:** `H_j^k = σ(W_self H_j^{k-1}
  + W_neigh · mean_{u∈N(j)} H_u^{k-1})`; layer1 in_dim(20)→hidden(128) ReLU, layer2 hidden→embed(128) linear.
  Edge index refreshed **per step** (inductive → zero-shot topology). Extract embeddings at the 41
  DER buses.
- **Actor (parameter-shared diagonal Gaussian):** embed→MLP(128,128)→(mean,log_std);
  `mean = 0.75·tanh(·)`, `std = softplus(log_std)+0.05`; `action_dim=2`. log_std clamp [−5,−1],
  `log_std_init=−1.0`.
- **Critic (agent-centered + global pooling):** per-agent embed concatenated with global
  mean+max pooled context → MLP → per-agent value.
- **GAE+PPO `[yu2022mappoeffective, schulman2017ppo]`:** γ=0.99, λ=0.95, clip=0.2, value_coef=0.5, entropy_coef=0.03 (curriculum-modulated);
  per-agent advantage (shared scalar reward broadcast); clipped surrogate
  `−min(rÂ, clip(r,1±0.2)Â)`; loss `= policy + 0.5·MSE_value − ent·H`.
- **Normalizers:** obs `RunningNormalizer((123,20))` (saved in checkpoint); reward normalizer.
- **HPs (ASHA-tuned, Trial 9):** lr `2.14e-4`, entropy 0.001, embed/hidden 128, update_epochs 8,
  mini_batch 32, log_std_init −2.0. (See `reportasha.md`.)

---

## 6. Training protocol

- **Curriculum** `AM_PHASES` (6000 ep total): Phase A (400, foundation/load-only) → B,C,D,E
  (1000 each, progressive event magnitude/probability) → F (1600). 300 steps/ep, LR/entropy
  annealed per phase.
- Single-phase mode: `--n-episodes` (Section-6 runs used **1000**, seed 42; current full retrain
  uses **6000 curriculum**).
- Entry: `python -m src.rl.train_am_mappo --curriculum --ffr-mode mappo_dual ...`.

---

## 7. Safety & stability (FRAMEWORK-LEVEL — shared by all controllers)

These are properties of the **MDP/env**, inherited by the proposed policy **and** all baselines
(see `safety_story`, and the framing note added to the paper's safety subsection). The comparison
therefore isolates *performance/generalization within the safe envelope*, not safety itself.

1. **Stability by construction (monotone-K)** `[feng2022stabilityrl, alduaij2025partialmonotone]`:
   `a^K → K ∈ [0,K_max]` ⇒ damping `D=diag(K)⪰0` ⇒ droop `−K·Δf` monotone non-increasing
   (non-negative dissipation). Env-side (K_min=0 map). Monotone-in-Δf with topology/SoC-modulated
   magnitude follows the partially-monotonic design of `alduaij2025partialmonotone`.
2. **Lyapunov certificate** (`experiments/lyapunov_certificate.py`) `[cui2020lyapunov]`: swing energy
   `V = ½Δωᵀ M Δω + ½Δδ_relᵀ J̃_r Δδ_rel`, `V̇ = −Δωᵀ D Δω ≤ 0`. Numerically: `A_f` Hurwitz over
   all (topology × K-box vertices) = 1024 matrices (0 non-Hurwitz; worst Re λ ≈ −1.6e-5).
   CLF/dwell-time on the new VSG model **to be re-confirmed after retrain** (current CLF marginal).
   `A_f` affine in **K** (D=diag(K)) ⇒ box vertices are extremal ⇒ finite vertex LMI suffices.
3. **RoCoF by system design `[kenyon2021gfmfreqorder]`:** `ḟ = ΔP_imb/(2 H_backbone)`, set by VSG
   backbone inertia, not actionable by the GFL VPP in the first step (Approach-C) ⇒ `w_rocof=0`.
4. **Nadir online projection (§1.4) `[tabas2021safefilter]`:** closed-form min-perturbation onto the nadir half-space,
   in-the-loop (train + eval), `nadir_margin_hz=0.5`. *Caveat:* prediction-based on the LTV-affine
   model (not hard CBF forward-invariance); model-level certificate (no EMT/HIL yet).

---

## 8. Baselines (Option B: same MDP, only encoder/algorithm differs)

All run in **mappo_dual** (same env/obs/reward/dual-action/safety envelope); cite "adapted from".

- **MLP-MAPPO** (ablation): identical RL stack, MLP encoder (no graph) → isolates GraphSAGE value.
- **GCNN-PPO** (`guo2024gcnnppo`, adapted; GCN operator `kipf2017gcn`): spectral GCN
  `D̂^{-½}ÂD̂^{-½}XW` (2 layers) + global sum-pool critic + PPO; transductive (fixed-adjacency per
  step) vs our inductive.
- **EIE-MATD3** (`li2025matd3`, adapted; TD3 core `fujimoto2018td3`): twin-Q, target smoothing,
  delayed update, soft target + EIE machinery — demonstrator (classical-droop expert → per-DER
  a^K), 4 diverse-noise explorers (Gaussian/OU/ε-greedy/max-entropy), dual replay pools,
  mixed-batch (demo_ratio 0.25).
- **Fixed droop** & **No-FFR**: non-learning references (also run in the shared safe envelope).

(See `baselines_fairness_fix`. Open: a safety-guaranteed comparator — the closest is the
TSG-2026 competitor `zhang2026vppsafety` (or a CMDP/PPO-Lagrangian baseline) — pending reviewer
feedback on whether the current lineup suffices.)

---

## 9. Key parameters & file map

| Quantity | Value | Source |
|---|---|---|
| S_base, f0 | 15.705 MW, 50 Hz | env / setup |
| H_sys (VSG aggregate) | ≈ 1.34 s | `freq_dyn_lti.h_sys` |
| H_virt (VSG / droop GFM) | 2.0 / 0.5 s | `_parse_gfm_placement` |
| nadir band δ | 0.5 Hz | `nadir_margin_hz` |
| RoCoF ride-through | 2.0 Hz/s | IEEE 1547-2018 Cat. III |
| K_max per type | BESS 0.30 / V2G 0.20 / DPV 0.15 ·P_rated | env init |
| n_agents / n_VPP / n_GFM | 41 / 3 / 6 | env |
| episode / step | 300 steps × 1 s | trainer |

**Files:** freq model `src/env/freq_dynamics_lti.py`; env/MDP `src/env/microgrid_env_dual.py`;
DERs `src/env/evcs_model.py`, `src/layer1_vpp/virtual_battery.py`; method `src/rl/train_am_mappo.py`;
Lyapunov `experiments/lyapunov_certificate.py`; eval `src/eval/eval_ffr_topology.py`;
baselines `src/baselines/`.

---

## 10. Honest caveats (put in Limitations)
- Control-design abstraction: LTV-affine VSG model; inner V/I loops, inverter current saturation,
  EMT not modeled → certificates are model-level; EMT/HIL is future work.
- One near-marginal COI mode is AGC-stabilized (not in the strictly-damped Lyapunov subspace).
- Section-6 numbers + Lyapunov CLF/dwell-time must be **regenerated on the VSG model** (retrain in
  progress) — old droop-model values are stale.
- SoC bands (reconciled 2026-06-02): **physical battery band [0.20, 0.90]** (EVBatteryModel, the
  env-active model) and **FFR-eligibility band [0.25, 0.85]** strictly nested inside it (was the
  inconsistent [0.15, 0.95] which exceeded the physical limits — fixed). Report these two in the
  paper (physical + FFR-gate). The simple-SoC clip [0.10, 0.90] in `evcs_model.py` is a separate
  legacy `EVChargingStation` path not used for the env dispatch — omit from the paper.
- **Consistency note:** GCNN-PPO finished (6000 ep) under the OLD FFR band [0.15,0.95]; MLP /
  EIE-MATD3 / Proposed were resumed under the corrected [0.25,0.85]. Practical impact is small
  (SoC rarely reaches 0.85–0.95 during a short FFR transient), but for strict Section-6 parity
  consider re-running GCNN-PPO on the corrected env.

---

## 11. Citations by modeling choice (keys in `paper/ref.bib`)

Use these `\cite{}` keys when writing each part. All keys verified present in `ref.bib`
(new ones added 2026-06-01 are flagged ★).

| Modeling choice / claim | Citation keys |
|---|---|
| **VSG swing emulating SG inertia+damping** (§1.1) | `hou2020adaptive`, `fathi2018extended`, `shadoul2022vsg`, `gurski2024aid`, `waskito2025review`, `sati2025economic` |
| GFM-VSG can stabilize 100% IBR MG | `salem2025gfmreview`, `arevalo2025gfmreview`, `anttila2022gfmreview`, `perez2023adaptive`, `bakeer2024bidirectional`, `lin2022field` |
| **Kron-reduced / topology-aware reduced-order freq model** (§1.2) | ★`luo2014kronmicrogrid`, ★`ajala2023gfmreduction`, ★`abdolmaleki2020onlinekron` |
| **Nadir/RoCoF decoupling, GFM order reduction** (§1.3, §7-RoCoF) | ★`kenyon2021gfmfreqorder` |
| Virtual-inertia control via ESS (DER) (§2) | `skiparev2021vic`, `amiri2023cascaded`, `kerdphol2017robust`, `saxena2020derivative` |
| VPP aggregation / FFR / EV (§3) | `fernandopulle2024vppffr`, `mohyuddin2022dsovpp`, `wang2023vppev`, `alden2026vppev`, `nazariheris2022evaggregators` |
| MARL FFR in microgrids (§4) | `afifi2024rl`, `dolatyabi2025happo`, `hu2022multitimescale`, `qiu2023hierarchical`, `benhmidouch2024vsg` |
| **GraphSAGE inductive encoder** (§5) | `hamilton2017graphsage`; applications `pei2023graphsage`, `wang2025graphsage`, `zhai2024graphsage`, `karabulut2025generalization` |
| **MAPPO / PPO** (§5) | `yu2022mappoeffective`, `schulman2017ppo`; graph-PPO `wang2023graphppo` |
| Topology generalization in grid GNN (§5) | `liu2022topology`, `jacob2024outage`, `wu2025ugcn`, `yang2024pggnn` |
| **Stability by construction (monotone-K)** (§7.1) | ★`feng2022stabilityrl`, ★`alduaij2025partialmonotone` |
| **Lyapunov-based stable RL frequency control** (§7.2) | `cui2020lyapunov` |
| **Closest competitor (safety-guaranteed VPP-FR)** (§7, Related Work) | ★`zhang2026vppsafety` |
| **Closed-form safety filter (parallels nadir layer)** (§7.4) | ★`tabas2021safefilter` |
| **GCNN-PPO baseline** — spectral GCN (§8) | `guo2024gcnnppo`, ★`kipf2017gcn` |
| **EIE-MATD3 baseline** — TD3 core (§8) | `li2025matd3`, ★`fujimoto2018td3`, `benhmidouch2024vsg` |
| FFR/AGC standards (RoCoF, UFLS, FCR) | `ieee_1547_2018`, `ieee_c37117_2007`, `entsoe_sogl_2017`, `entsoe_rfg_2016`, `mota2024fcri`, `papakonstantinou2021bessagc` |

**Bib hygiene:** fixed a duplicate `li2025matd3` entry; added the ★ keys above. Foundational
ML method papers (`hamilton2017graphsage`, `yu2022mappoeffective`, `schulman2017ppo`,
`fujimoto2018td3`, `kipf2017gcn`) are the canonical originals for the encoder/algorithm cores —
cite them where each baseline/method is introduced and label baselines "adapted from" the
power-systems application keys (`guo2024gcnnppo`, `li2025matd3`).
