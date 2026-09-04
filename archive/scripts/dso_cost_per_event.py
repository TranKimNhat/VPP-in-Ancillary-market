"""DSO ancillary cost per contingency event (system-operator perspective).

The DSO does NOT want to maximise procured power (that is the perverse VPP-revenue
view). It wants frequency security at least total cost. We therefore cost each
event as:

    C_DSO = W_deliv * ( lambda_cap * R_armed * T       (reserve procurement)
                      + lambda_act * E_FFR_delivered )  (activation payment)
          + VOLL * E_shed                               (UFLS load shedding externality)

Energy (DLMP) and FFR reserve are DISTINCT co-optimised products: in a 100%-
renewable islanded microgrid the marginal energy cost is ~0 (DLMP energy ~0), so
the ancillary tariff sets the FFR price LEVEL, while the Layer-0 DLMP *congestion*
component sets its locational deliverability SHAPE. W_deliv (Option D) is a
commitment-weighted DLMP-congestion factor over the DER buses, mean-normalised to
1 so the fleet-average level stays on the tariff; a fleet leaning on congested
buses pays > tariff, one on well-connected buses pays < tariff.

E_shed is energy-not-served from a staged ENTSO-E UFLS model applied to the COI
frequency trace (the env DEFINED ufls thresholds 49.0/48.5/48.0 Hz but never used
them — we implement the shedding here). A good controller pays some reserve/
activation but AVOIDS the very expensive shed term; a non-responsive baseline pays
~0 for service but triggers large shedding -> highest total cost.

Per-policy safety layer matches the official eval (baselines OFF, learned ON).
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import pandas as pd

from src.eval.eval_ffr_topology import FFRTopologyEvaluator
from src.rl.train_am_mappo import build_am_full_feeder_obs, ensure_edge_index

# ---- Prices (MarketPriceConfig defaults) + literature-grounded VOLL ----
# VOLL central = EUR 3000/MWh: matches the EU operational/reserve-repricing
# convention (Swinand 2019, ~EUR 3500) and system-wide empirical estimates
# (Her 2026, ~3.3 USD/kWh), which is the correct use-case for FFR/ancillary
# (NOT the higher reliability-planning VoLL). Sweep shows ranking invariance.
LAMBDA_CAP = 50.0          # EUR/MW/h  reserve reservation
LAMBDA_ACT = 100.0         # EUR/MWh   activation
VOLL_CENTRAL = 3000.0      # EUR/MWh   value of lost load (reserve context)
VOLL_SWEEP = [1500.0, 3000.0, 8000.0]
DF_BAND = 0.5              # Hz        committed-reserve band edge for R_armed

# Precomputed Layer-0 DLMP (same SOCP duals the training env consumed)
DLMP_CSV = Path("data/oedisi-ieee123-main/profiles/layer0_hourly/layer0_dlmp_per_bus.csv")

# ENTSO-E staged UFLS shed fractions (cumulative) keyed by frequency floor
def shed_fraction(f_hz: float) -> float:
    if f_hz >= 49.0:
        return 0.0
    if f_hz >= 48.5:
        return 0.05
    if f_hz >= 48.0:
        return 0.15
    return 0.30


def load_deliverability_weights(env, csv_path: Path) -> np.ndarray:
    """Per-DER deliverability weight w_i = |lambda_p_congestion(bus_i)| / ref.

    ref = mean |congestion| over the DER buses (a FIXED, method-independent
    reference), so the per-DER weights average ~1 and only their dispersion —
    i.e. WHERE a controller commits droop — shifts cost. Buses absent from the
    DLMP table (or a degenerate all-zero congestion field) get weight 1.0.
    """
    n = int(env.n_agents)
    if not csv_path.exists():
        print(f"[warn] DLMP CSV not found ({csv_path}); deliverability weights = 1")
        return np.ones(n)
    df = pd.read_csv(csv_path)
    g_by_bus = (
        df.assign(_g=df["lambda_p_congestion"].abs())
        .groupby("bus_id")["_g"].mean().to_dict()
    )
    g_by_bus = {int(k): float(v) for k, v in g_by_bus.items()}
    buses = np.asarray(env._agent_bus_pp, dtype=int)
    gi = np.array([g_by_bus.get(int(b), np.nan) for b in buses], dtype=float)
    found = ~np.isnan(gi)
    w = np.ones(n, dtype=float)
    if found.sum() == 0:
        print("[warn] no DER bus matched DLMP CSV; deliverability weights = 1")
        return w
    ref = float(np.mean(gi[found]))
    if ref > 0:
        w[found] = gi[found] / ref
    print(f"[deliverability] matched {int(found.sum())}/{n} DER buses; "
          f"ref|cong|={ref:.3e}, w in [{w.min():.3f}, {w.max():.3f}]")
    return w

def run_one(pol, event, env, dt_h, P_load_total, DER_W, topology_idx=None):
    """One costed rollout. topology_idx=None -> base topology (per-scenario use);
    an int -> force that cached reconfiguration (per-topology Wilcoxon use)."""
    env.ffr_mode = getattr(pol, "ffr_mode", "droop")
    env.nadir_safety_enabled = bool(getattr(pol, "nadir_safety", True))
    options = {"force_event": copy.deepcopy(event)}
    use_base = topology_idx is None
    if not use_base:
        options["force_topology"] = int(topology_idx)
    _prev_base = env.fixed_base_topology
    if use_base:
        env.fixed_base_topology = True
    obs_fast, _, _ = env.reset(options=options)
    env.fixed_base_topology = _prev_base
    nb = len(env.net.bus.index)
    edge = ensure_edge_index(env.edge_index, n_nodes=nb)
    of = build_am_full_feeder_obs(env, obs_fast)
    f_tr, k_peak = [], 0.0
    k_vec_peak = np.zeros(env.n_agents, dtype=float)
    for t in range(300):
        a = pol.act(of, edge, env, obs_fast=obs_fast)
        obs_fast, _, d, _, info = env.step_fast(a)
        edge = ensure_edge_index(info.get("edge_index", edge), n_nodes=nb)
        of = build_am_full_feeder_obs(env, obs_fast)
        f_tr.append(50.0 + float(env.freq_dyn_lti.get_state().delta_f_hz))
        ksum = float(np.sum(env._k_droop_last))                 # MW/Hz committed
        if ksum >= k_peak:                                      # capture per-DER mix at peak
            k_peak = ksum
            k_vec_peak = np.asarray(env._k_droop_last, dtype=float).copy()
        e_ffr = float(info.get("ffr_energy_delivered_mwh", 0.0))
    f_tr = np.array(f_tr)
    # Energy-not-served from staged UFLS over the trace
    e_shed = float(np.sum([shed_fraction(f) for f in f_tr]) * P_load_total * dt_h)
    # Deliverability factor (Option D): commitment-weighted DLMP-congestion over the
    # committed droop mix. Method-dependent through where each controller arms droop.
    ktot = float(k_vec_peak.sum())
    w_deliv = float((DER_W * k_vec_peak).sum() / ktot) if ktot > 0 else 1.0
    r_armed = k_peak * DF_BAND                              # MW committed at band edge
    t_event_h = len(f_tr) * dt_h
    c_cap = LAMBDA_CAP * r_armed * t_event_h * w_deliv     # FFR cost: tariff level × deliverability shape
    c_act = LAMBDA_ACT * e_ffr * w_deliv
    out = dict(nadir=float(f_tr.min()), e_ffr=e_ffr, e_shed=e_shed,
               w_deliv=w_deliv, c_cap=c_cap, c_act=c_act)
    # VOLL-dependent shed cost + total, per sweep level
    for v in VOLL_SWEEP:
        out[f"c_total@{int(v)}"] = c_cap + c_act + v * e_shed
    out["c_shed"] = VOLL_CENTRAL * e_shed
    out["c_total"] = c_cap + c_act + VOLL_CENTRAL * e_shed   # central
    return out

def build_evaluator():
    return FFRTopologyEvaluator(
        env_config={"placement_path": "artifacts/placement/official_placement_v3.json",
                    "mpc_path": "data/grid_IEEE123_complete.m", "seed": 42},
        checkpoint_path=Path("artifacts/ckpt_proposed_s42/am_mappo_final.pt"),
        mlp_mappo_checkpoint=Path("artifacts/ckpt_mlp_mappo/mlp_mappo_final.pt"),
        gcnn_checkpoint=Path("artifacts/ckpt_gcnn_ppo/final.pt"),
        matd3_checkpoint=Path("artifacts/ckpt_matd3/matd3_ep5700.pt"),
        output_dir=Path("results/_dso"),
        base_reference=True,
    )


if __name__ == "__main__":
    ev = build_evaluator()
    env = ev.env
    dt_h = env.dt_fast_s / 3600.0
    P_load_total = float(env.net.load["p_mw"].sum())
    DER_W = load_deliverability_weights(env, DLMP_CSV)
    N = 10

    rows = []
    for scen, event in ev.scenarios.items():
        for name, pol in ev.policies.items():
            accs = [run_one(pol, event, env, dt_h, P_load_total, DER_W) for _ in range(N)]
            agg = {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}
            agg.update(scenario=scen, method=name)
            rows.append(agg)

    sweep_cols = [f"c_total@{int(v)}" for v in VOLL_SWEEP]
    df = pd.DataFrame(rows)[["scenario", "method", "nadir", "e_ffr", "e_shed",
                             "w_deliv", "c_cap", "c_act", "c_shed", "c_total"] + sweep_cols]
    out = Path("results/ffr_topology_baseref_final")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "dso_cost_per_event.csv", index=False)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print(df.round(2).to_string(index=False))
    print(f"\nSaved -> {out/'dso_cost_per_event.csv'}")
    print(f"(P_load_total={P_load_total:.2f} MW, VOLL_central={VOLL_CENTRAL}, sweep={VOLL_SWEEP}, "
          f"lambda_cap={LAMBDA_CAP}, lambda_act={LAMBDA_ACT})")
