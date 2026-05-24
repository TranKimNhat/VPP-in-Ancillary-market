"""Economic effectiveness evaluation for the dual-market VPP framework.

Aggregates per-VPP revenue, OPEX, and net profit across an episode using the same
``MarketPrices`` formulation as ``src/layer2_control/reward.py`` and produces the
revenue/profit tables and figures referenced in Section 6 (subsection VI-G).

Building blocks:
  - EM revenue   : λ_e · P + λ_q · |Q|      (per agent, summed per VPP)
  - AM revenue   : λ_cap · R_commit + λ_act · R_delivered · 1[event]
  - Undersupply  : c_undersupply · λ_act · max(0, R_commit - R_delivered)
  - OPEX         : hourly OPEX from placement (IRENA 2023 / NREL ATB 2023)

R_commit per VPP is proxied from the running ``|delta_p_set|`` per VPP (the actual
FFR dispatch); R_delivered per VPP is the system-wide ``ffr_energy_delivered_mwh``
allocated proportionally to that share. This matches Wen Chen et al. (2021,
IEEE TSG, doi:10.1109/TSG.2021.3115062) where activation payout is split by
delivered MWh.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.events import EventConfig
from src.env.microgrid_env_dual import MicrogridEnvDual
from src.eval.eval_ffr_topology import FixedDroopPolicy, NoFFRPolicy


# OPEX rates (€/kW/year) — IRENA 2023, NREL ATB 2023 (same as evaluate_dual.py)
OPEX_PV_EUR_KW_YEAR = 12.0
OPEX_WIND_EUR_KW_YEAR = 30.0
OPEX_BESS_EUR_KWH_YEAR = 7.5
HOURS_PER_YEAR = 8760.0


@dataclass(frozen=True)
class MarketPriceConfig:
    """Market clearing prices for the AM-only EM/AM evaluation.

    Defaults are fallback values; when the env exposes per-step prices
    (``env.lambda_as_ffr``, ``env.zone_lambda_as``) those are preferred.
    """
    lambda_e: float = 50.0          # €/MWh — active energy DLMP (P2P)
    lambda_cap: float = 50.0        # €/MW/h — FFR capacity reservation
    lambda_act: float = 100.0       # €/MWh — FFR activation payment
    c_undersupply: float = 3.0      # Nordic imbalance penalty multiplier


def compute_opex_hourly(placement: dict[str, Any]) -> float:
    """Total hourly OPEX (€/h) for the placement (system-wide)."""
    opex_year = 0.0
    for gfm in placement.get("gfm", {}).values():
        opex_year += float(gfm.get("pv_mw", 0.0)) * 1000.0 * OPEX_PV_EUR_KW_YEAR
        opex_year += float(gfm.get("bess_mwh", 0.0)) * 1000.0 * OPEX_BESS_EUR_KWH_YEAR
    for wind in placement.get("wind", []):
        opex_year += float(wind.get("mw", 0.0)) * 1000.0 * OPEX_WIND_EUR_KW_YEAR
    for evcs in placement.get("evcs", []):
        opex_year += float(evcs.get("pv_mw", 0.0)) * 1000.0 * OPEX_PV_EUR_KW_YEAR
        opex_year += float(evcs.get("bess_mwh", 0.0)) * 1000.0 * OPEX_BESS_EUR_KWH_YEAR
    for dpv in placement.get("dpv", []):
        opex_year += float(dpv.get("mw", 0.0)) * 1000.0 * OPEX_PV_EUR_KW_YEAR
    return opex_year / HOURS_PER_YEAR


def opex_hourly_per_vpp(placement: dict[str, Any]) -> dict[str, float]:
    """Per-VPP hourly OPEX from placement. Keys: 'VPP_1','VPP_2','VPP_3','SYSTEM'."""
    opex: dict[str, float] = {"VPP_1": 0.0, "VPP_2": 0.0, "VPP_3": 0.0, "SYSTEM": 0.0}
    gfm = placement.get("gfm", {})
    for vpp_key in ("VPP_1", "VPP_2", "VPP_3"):
        node = gfm.get(vpp_key, {}) if isinstance(gfm, dict) else {}
        if isinstance(node, dict):
            opex[vpp_key] += float(node.get("pv_mw", 0.0)) * 1000.0 * OPEX_PV_EUR_KW_YEAR / HOURS_PER_YEAR
            opex[vpp_key] += float(node.get("bess_mwh", 0.0)) * 1000.0 * OPEX_BESS_EUR_KWH_YEAR / HOURS_PER_YEAR
    # Wind, EVCS, DPV are system-level shared costs
    opex["SYSTEM"] = compute_opex_hourly(placement) - sum(opex[v] for v in ("VPP_1", "VPP_2", "VPP_3"))
    opex["SYSTEM"] = max(0.0, opex["SYSTEM"])
    return opex


@dataclass
class EpisodeEconomics:
    """Per-episode economic breakdown, system + per-VPP (AM-only build).

    EM revenue is driven by active-power dispatch only (energy P at lambda_e);
    the reactive Q product was removed from the framework.
    """
    em_p_revenue: dict[str, float] = field(default_factory=dict)   # €
    am_cap_revenue: dict[str, float] = field(default_factory=dict)
    am_act_revenue: dict[str, float] = field(default_factory=dict)
    undersupply_pen: dict[str, float] = field(default_factory=dict)
    opex: dict[str, float] = field(default_factory=dict)
    duration_h: float = 0.0
    ffr_success: bool = True
    nadir_hz: float = 50.0

    @property
    def gross_revenue(self) -> dict[str, float]:
        keys = set(self.em_p_revenue) | set(self.am_cap_revenue)
        return {
            k: self.em_p_revenue.get(k, 0.0)
            + self.am_cap_revenue.get(k, 0.0)
            + self.am_act_revenue.get(k, 0.0)
            for k in keys
        }

    @property
    def net_profit(self) -> dict[str, float]:
        gross = self.gross_revenue
        return {
            k: gross[k] - self.undersupply_pen.get(k, 0.0) - self.opex.get(k, 0.0)
            for k in gross
        }

    def to_rows(self, method: str, scenario: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        gross = self.gross_revenue
        net = self.net_profit
        for vpp in sorted(self.em_p_revenue.keys()):
            rows.append({
                "method": method,
                "scenario": scenario,
                "vpp": vpp,
                "em_p_eur": self.em_p_revenue.get(vpp, 0.0),
                "am_cap_eur": self.am_cap_revenue.get(vpp, 0.0),
                "am_act_eur": self.am_act_revenue.get(vpp, 0.0),
                "undersupply_eur": self.undersupply_pen.get(vpp, 0.0),
                "opex_eur": self.opex.get(vpp, 0.0),
                "gross_revenue_eur": gross.get(vpp, 0.0),
                "net_profit_eur": net.get(vpp, 0.0),
                "duration_h": self.duration_h,
            })
        return rows


class EconomicsEvaluator:
    """Run policies through an episode and book per-VPP cashflows."""

    def __init__(
        self,
        env: MicrogridEnvDual,
        prices: MarketPriceConfig,
        placement: dict[str, Any] | None = None,
        output_dir: Path = Path("results/economics"),
    ) -> None:
        self.env = env
        self.prices = prices
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.opex_per_vpp = opex_hourly_per_vpp(placement or {})
        self.dt_fast_h = float(getattr(env, "dt_fast_s", 1.0)) / 3600.0
        # Per-VPP agent index map
        self.vpp_agents: dict[str, list[int]] = {
            str(k): [int(i) for i in v] for k, v in env._vpp_droop_agents.items()
        }
        # Agent P-rated (MW) used for capacity attribution
        self._p_rated = np.asarray(
            [max(float(spec.get("p_rated", 0.0)), 0.0) for spec in env._agent_specs],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------ helpers
    def _vpp_p_rated(self, vpp: str) -> float:
        idx = self.vpp_agents[vpp]
        if not idx:
            return 0.0
        return float(np.sum(self._p_rated[idx]))

    # ----------------------------------------------------------------- episode
    def run_episode(
        self,
        policy: Any,
        event: EventConfig | None = None,
        topology_idx: int | None = None,
        n_steps: int = 300,
    ) -> EpisodeEconomics:
        from src.rl.train_am_mappo import build_am_full_feeder_obs, ensure_edge_index  # local to avoid heavy import at module load

        options: dict[str, Any] = {}
        if event is not None:
            options["force_event"] = deepcopy(event)
        if topology_idx is not None:
            options["force_topology"] = int(topology_idx)

        obs_fast, _, _ = self.env.reset(options=options)
        n_bus = len(self.env.net.bus.index)
        edge_index = ensure_edge_index(self.env.edge_index, n_nodes=n_bus)
        obs_full = build_am_full_feeder_obs(self.env, obs_fast)

        is_event = bool(event is not None)

        # Accumulators per VPP (Q product dropped in AM-only build)
        em_p = {v: 0.0 for v in self.vpp_agents}
        cap = {v: 0.0 for v in self.vpp_agents}
        share_acc = {v: 0.0 for v in self.vpp_agents}  # for R_delivered split
        r_commit_total = 0.0
        ffr_energy_total_mwh = 0.0
        nadir = 50.0

        for _ in range(n_steps):
            action = policy.act(obs_full, edge_index, self.env)
            obs_fast, _r, _done, _trunc, info = self.env.step_fast(action)
            new_edge = info.get("edge_index", edge_index)
            edge_index = ensure_edge_index(new_edge, n_nodes=n_bus)
            obs_full = build_am_full_feeder_obs(self.env, obs_fast)

            freq_state = self.env.freq_dyn.get_state()
            nadir = min(nadir, 50.0 + float(freq_state.delta_f_hz))

            # Per-step prices: prefer env-exposed (from precompute), else fallback constants.
            lambda_e_step = float(getattr(self.env, "_zone_lmp_vec", np.array([self.prices.lambda_e])).mean()) \
                if hasattr(self.env, "_zone_lmp_vec") else self.prices.lambda_e
            lambda_cap_step = float(getattr(self.env, "lambda_as_ffr", self.prices.lambda_cap))

            # ----- EM cashflow: energy P × lambda_e
            # Prefer env-exposed L1 LP base dispatch p_ref_target_vpp (set during
            # slow context) since RL fast steps do not write to env.p_set.
            # Fall back to env.p_set if precompute targets are unavailable.
            p_ref_vpp = getattr(self.env, "p_ref_target_vpp", None)
            if p_ref_vpp is not None and len(p_ref_vpp) >= len(self.vpp_agents):
                for j, vpp in enumerate(self.vpp_agents):
                    p_mw = float(p_ref_vpp[j])
                    em_p[vpp] += lambda_e_step * p_mw * self.dt_fast_h
            else:
                p_set = np.asarray(self.env.p_set, dtype=np.float64)
                for vpp, idx in self.vpp_agents.items():
                    if not idx:
                        continue
                    p_mw = float(np.sum(p_set[idx] * self._p_rated[idx]))
                    em_p[vpp] += lambda_e_step * p_mw * self.dt_fast_h

            # ----- AM capacity & delivery proxy
            delta_p = np.asarray(self.env.delta_p_set, dtype=np.float64)
            for vpp, idx in self.vpp_agents.items():
                if not idx:
                    continue
                # Convert |delta_p_set| (pu of rating) to MW commitment proxy
                commit_mw = float(np.sum(np.abs(delta_p[idx]) * self._p_rated[idx]))
                cap[vpp] += lambda_cap_step * commit_mw * self.dt_fast_h
                r_commit_total += commit_mw * self.dt_fast_h
                share_acc[vpp] += commit_mw * self.dt_fast_h

            ffr_energy_total_mwh = float(info.get("ffr_energy_delivered_mwh", ffr_energy_total_mwh))

        # ----- Activation payout split: proportional to accumulated commit share
        total_share = sum(share_acc.values()) or 1.0
        act = {v: 0.0 for v in self.vpp_agents}
        under = {v: 0.0 for v in self.vpp_agents}
        if is_event and ffr_energy_total_mwh > 0.0:
            for vpp in self.vpp_agents:
                w = share_acc[vpp] / total_share
                delivered_mwh = w * ffr_energy_total_mwh
                act[vpp] = self.prices.lambda_act * delivered_mwh
                commit_mwh = share_acc[vpp]
                shortfall = max(0.0, commit_mwh - delivered_mwh)
                under[vpp] = self.prices.c_undersupply * self.prices.lambda_act * shortfall

        duration_h = n_steps * self.dt_fast_h
        opex = {v: self.opex_per_vpp.get(v, 0.0) * duration_h for v in self.vpp_agents}
        opex["SYSTEM"] = self.opex_per_vpp.get("SYSTEM", 0.0) * duration_h
        # Pad SYSTEM keys for completeness
        for d in (em_p, cap, act, under):
            d.setdefault("SYSTEM", 0.0)

        ffr_success = (nadir >= 49.5)

        return EpisodeEconomics(
            em_p_revenue=em_p,
            am_cap_revenue=cap,
            am_act_revenue=act,
            undersupply_pen=under,
            opex=opex,
            duration_h=duration_h,
            ffr_success=ffr_success,
            nadir_hz=nadir,
        )

    # ------------------------------------------------------------------- tables
    def build_table_revenue_breakdown(
        self,
        policies: dict[str, Any],
        scenarios: dict[str, EventConfig],
        n_runs: int = 5,
        topology_idx: int | None = None,
    ) -> pd.DataFrame:
        """Table IX: per-VPP daily revenue breakdown × method × scenario."""
        rows: list[dict[str, Any]] = []
        for sc_name, event in scenarios.items():
            for m_name, policy in policies.items():
                acc: list[EpisodeEconomics] = []
                for _ in range(n_runs):
                    acc.append(self.run_episode(policy, event=event, topology_idx=topology_idx))
                # Average across runs per VPP
                agg = self._average(acc)
                rows.extend(agg.to_rows(m_name, sc_name))
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "table9_revenue_breakdown.csv", index=False)
        return df

    def build_table_method_comparison(
        self,
        policies: dict[str, Any],
        scenarios: dict[str, EventConfig],
        n_runs: int = 5,
        topology_idx: int | None = None,
    ) -> pd.DataFrame:
        """Table X: method-level economic summary across all scenarios."""
        rows: list[dict[str, Any]] = []
        for m_name, policy in policies.items():
            per_method: list[EpisodeEconomics] = []
            ffr_success_runs: list[float] = []
            for sc_name, event in scenarios.items():
                for _ in range(n_runs):
                    ep = self.run_episode(policy, event=event, topology_idx=topology_idx)
                    per_method.append(ep)
                    ffr_success_runs.append(float(ep.ffr_success))
            agg = self._average(per_method)
            gross_total = sum(agg.gross_revenue.get(v, 0.0) for v in self.vpp_agents)
            net_total = sum(agg.net_profit.get(v, 0.0) for v in self.vpp_agents)
            # Add SYSTEM net (gross has no SYSTEM, only opex)
            net_total -= agg.opex.get("SYSTEM", 0.0)
            # Scale episode-window cashflows to a daily view so the per-row
            # numbers reflect operational economics rather than a 5-min snapshot.
            scale_day = (24.0 / max(agg.duration_h, 1e-6))
            rows.append({
                "method": m_name,
                "gross_revenue_eur": gross_total * scale_day,
                "net_profit_eur": net_total * scale_day,
                "em_p_eur": sum(agg.em_p_revenue.get(v, 0.0) for v in self.vpp_agents) * scale_day,
                "am_cap_eur": sum(agg.am_cap_revenue.get(v, 0.0) for v in self.vpp_agents) * scale_day,
                "am_act_eur": sum(agg.am_act_revenue.get(v, 0.0) for v in self.vpp_agents) * scale_day,
                "undersupply_eur": sum(agg.undersupply_pen.get(v, 0.0) for v in self.vpp_agents) * scale_day,
                "opex_eur": sum(agg.opex.values()) * scale_day,
                "ffr_success_rate": float(np.mean(ffr_success_runs)),
                "profit_per_hour": net_total / max(agg.duration_h, 1e-6),
                "duration_h_episode": float(agg.duration_h),
                "scale_to_daily": float(scale_day),
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "table10_method_economics.csv", index=False)
        return df

    @staticmethod
    def _average(episodes: list[EpisodeEconomics]) -> EpisodeEconomics:
        """Mean across episodes preserving per-VPP keys."""
        if not episodes:
            return EpisodeEconomics()
        keys = set(episodes[0].em_p_revenue.keys())
        def _avg(field_name: str) -> dict[str, float]:
            return {
                k: float(np.mean([getattr(e, field_name).get(k, 0.0) for e in episodes]))
                for k in keys
            }
        return EpisodeEconomics(
            em_p_revenue=_avg("em_p_revenue"),
            am_cap_revenue=_avg("am_cap_revenue"),
            am_act_revenue=_avg("am_act_revenue"),
            undersupply_pen=_avg("undersupply_pen"),
            opex=_avg("opex"),
            duration_h=float(np.mean([e.duration_h for e in episodes])),
            ffr_success=bool(np.mean([e.ffr_success for e in episodes]) >= 0.5),
            nadir_hz=float(np.mean([e.nadir_hz for e in episodes])),
        )

    # -------------------------------------------------------------------- plots
    def plot_revenue_decomposition(self, table_x: pd.DataFrame) -> None:
        """Fig 12: stacked bar of revenue components per method."""
        if table_x.empty:
            return
        comp_cols = ["em_p_eur", "am_cap_eur", "am_act_eur"]
        neg_cols = ["undersupply_eur", "opex_eur"]
        labels = ["EM (energy)", "AM (capacity)", "AM (activation)", "− Undersupply", "− OPEX"]
        colors = ["#3a7bd5", "#f5a623", "#e94e77", "#7a7a7a", "#a04668"]

        methods = table_x["method"].tolist()
        x = np.arange(len(methods))
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(methods)), 5))

        bottom_pos = np.zeros(len(methods))
        bottom_neg = np.zeros(len(methods))
        for col, lbl, c in zip(comp_cols, labels[:4], colors[:4]):
            vals = table_x[col].to_numpy(dtype=float)
            ax.bar(x, vals, bottom=bottom_pos, label=lbl, color=c, edgecolor="white")
            bottom_pos += vals
        for col, lbl, c in zip(neg_cols, labels[4:], colors[4:]):
            vals = -table_x[col].to_numpy(dtype=float)
            ax.bar(x, vals, bottom=bottom_neg, label=lbl, color=c, edgecolor="white", alpha=0.85)
            bottom_neg += vals

        ax.plot(x, table_x["net_profit_eur"].to_numpy(dtype=float), "k_", markersize=20, markeredgewidth=2.5, label="Net profit")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("Cashflow per episode (€)")
        ax.set_title("Revenue decomposition by method")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(self.output_dir / "fig12_revenue_decomposition.pdf", dpi=300)
        fig.savefig(self.output_dir / "fig12_revenue_decomposition.png", dpi=150)
        plt.close(fig)

    def plot_pareto(self, table_x: pd.DataFrame) -> None:
        """Fig 13: Pareto scatter of FFR success rate vs net profit."""
        if table_x.empty:
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        x = table_x["ffr_success_rate"].to_numpy(dtype=float)
        y = table_x["net_profit_eur"].to_numpy(dtype=float)
        ax.scatter(x, y, s=80, c="#3a7bd5", edgecolor="black")
        for xi, yi, name in zip(x, y, table_x["method"].tolist()):
            ax.annotate(name, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel("FFR success rate")
        ax.set_ylabel("Net profit per episode (€)")
        ax.set_title("Profitability vs frequency-security tradeoff")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        fig.savefig(self.output_dir / "fig13_pareto_profit_vs_ffr.pdf", dpi=300)
        fig.savefig(self.output_dir / "fig13_pareto_profit_vs_ffr.png", dpi=150)
        plt.close(fig)


# =========================================================================== CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Economic evaluation of dual-market VPP policies")
    parser.add_argument("--placement", type=Path, default=Path("artifacts/placement/official_placement_v3.json"))
    parser.add_argument("--mpc-path", type=Path, default=Path("data/grid_IEEE123_complete.m"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints_am_mappo/am_mappo_final.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/economics"))
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-e", type=float, default=50.0)
    parser.add_argument("--lambda-cap", type=float, default=50.0)
    parser.add_argument("--lambda-act", type=float, default=100.0)
    return parser.parse_args()


def _load_placement(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    placement = _load_placement(args.placement)

    env = MicrogridEnvDual(
        placement_path=str(args.placement),
        mpc_path=str(args.mpc_path),
        seed=args.seed,
    )

    prices = MarketPriceConfig(
        lambda_e=args.lambda_e,
        lambda_cap=args.lambda_cap,
        lambda_act=args.lambda_act,
    )

    evaluator = EconomicsEvaluator(env, prices, placement, args.output_dir)

    policies: dict[str, Any] = {
        "No FFR": NoFFRPolicy(),
        "Fixed Droop": FixedDroopPolicy(k_droop=0.05),
    }
    if args.checkpoint.exists():
        from src.eval.eval_ffr_topology import GraphSAGEMAPPOPolicy
        try:
            policies["GraphSAGE-MAPPO"] = GraphSAGEMAPPOPolicy(args.checkpoint, env)
        except Exception as exc:
            print(f"[warn] Could not load GraphSAGE-MAPPO checkpoint: {exc}")

    scenarios = {
        "S1_load_step": EventConfig(type="load_step", delta_P_mw=2.5, location=45, t_inject=30.0),
        "S2_gen_trip": EventConfig(type="gen_trip", delta_P_mw=-3.9, location=67, t_inject=30.0),
        "S3_line_trip": EventConfig(type="line_trip", delta_P_mw=-2.4, location=67068, t_inject=30.0),
        "S4_gen_trip_severe": EventConfig(type="gen_trip", delta_P_mw=-4.5, location=105, t_inject=30.0),
    }

    print("\n[1/3] Building Table IX (per-VPP revenue breakdown)...")
    t9 = evaluator.build_table_revenue_breakdown(policies, scenarios, n_runs=args.n_runs)

    print("\n[2/3] Building Table X (method economic comparison)...")
    t10 = evaluator.build_table_method_comparison(policies, scenarios, n_runs=args.n_runs)

    print("\n[3/3] Plotting Fig 12 (revenue decomposition) & Fig 13 (Pareto)...")
    evaluator.plot_revenue_decomposition(t10)
    evaluator.plot_pareto(t10)

    print("\n=== Summary ===")
    print(t10[["method", "gross_revenue_eur", "net_profit_eur", "ffr_success_rate"]].to_string(index=False))
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
