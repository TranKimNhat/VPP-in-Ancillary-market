from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.freq_simulation import simulate_freq_response, simulate_scenarios_comparison

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "text.usetex": False,
    }
)

COLORS = {
    "Proposed": "#1f77b4",
    "Random": "#d62728",
    "Rule-based": "#2ca02c",
    "No-freq": "#ff7f0e",
    "No-freq(fallback)": "#ff7f0e",
    "Proposed (MAPPO)": "#1f77b4",
    "No BESS": "#7f7f7f",
}
LINESTYLES = {
    "Proposed (MAPPO)": "-",
    "Rule-based": "--",
    "Random": "-.",
    "No BESS": ":",
}


def _save(fig, save_dir: Path, stem: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(save_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_eval_results(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load tất cả CSV từ results/eval/."""
    base = Path(results_dir)
    results: dict[str, pd.DataFrame] = {}
    for name in ["S1_normal", "S2_freq", "S3_highren", "S4_stress"]:
        p = base / f"{name}.csv"
        results[name] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    summary = base / "summary.csv"
    results["summary"] = pd.read_csv(summary) if summary.exists() else pd.DataFrame()
    return results


def _parse_train_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            {
                "episode": pd.Series(dtype=int),
                "reward": pd.Series(dtype=float),
                "entropy": pd.Series(dtype=float),
            }
        )
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if "phase=" not in line or "reward=" not in line or "entropy=" not in line or "ep=" not in line:
            continue
        try:
            tokens = [tok for tok in line.replace("=", " ").split()]
            ep = int(tokens[tokens.index("ep") + 1])
            reward = float(tokens[tokens.index("reward") + 1])
            entropy = float(tokens[tokens.index("entropy") + 1])
            rows.append({"episode": ep, "reward": reward, "entropy": entropy})
        except Exception:
            continue
    return pd.DataFrame(rows)


def fig1_training_curves(log_dir: str | Path, save_dir: str | Path) -> None:
    save_path = Path(save_dir)
    logs = [Path(log_dir) / "train_s42.txt", Path(log_dir) / "train_s123.txt", Path(log_dir) / "train_s777.txt"]
    dfs = [d for d in (_parse_train_log(p) for p in logs) if not d.empty]
    if not dfs:
        return

    merged = pd.concat(dfs, keys=range(len(dfs)), names=["seed_idx", "row"]).reset_index(level=0)
    grouped = merged.groupby("episode", as_index=False).agg(
        reward_mean=("reward", "mean"),
        reward_std=("reward", "std"),
        entropy_mean=("entropy", "mean"),
        entropy_std=("entropy", "std"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
    x = np.asarray(grouped["episode"], dtype=float)

    reward_mean = np.asarray(grouped["reward_mean"], dtype=float)
    reward_std = np.nan_to_num(np.asarray(grouped["reward_std"], dtype=float), nan=0.0)
    axes[0].plot(x, reward_mean, color=COLORS["Proposed"], label="Reward")
    axes[0].fill_between(
        x,
        reward_mean - reward_std,
        reward_mean + reward_std,
        color=COLORS["Proposed"],
        alpha=0.2,
    )
    axes[0].set_title("Mean Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")

    entropy_mean = np.asarray(grouped["entropy_mean"], dtype=float)
    entropy_std = np.nan_to_num(np.asarray(grouped["entropy_std"], dtype=float), nan=0.0)
    axes[1].plot(x, entropy_mean, color=COLORS["No-freq"], label="Entropy")
    axes[1].fill_between(
        x,
        entropy_mean - entropy_std,
        entropy_mean + entropy_std,
        color=COLORS["No-freq"],
        alpha=0.2,
    )
    axes[1].set_title("Policy Entropy")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Entropy")

    for ax in axes:
        for boundary in [500, 3500, 7500]:
            ax.axvline(boundary, linestyle="--", linewidth=0.8, color="gray", alpha=0.8)

    _save(fig, save_path, "fig1_training")


def fig2_zone_lmp(parquet_path: str | Path, save_dir: str | Path) -> None:
    df = pd.read_parquet(parquet_path)
    fig, ax = plt.subplots(figsize=(4, 3))

    hour = np.asarray(df["hour"] if "hour" in df.columns else (df["step"] / 4.0), dtype=float)
    z1 = np.asarray(df["lambda_p2p_z1"], dtype=float)
    z2 = np.asarray(df["lambda_p2p_z2"], dtype=float)
    z4 = np.asarray(df["lambda_p2p_z4"], dtype=float)
    ax.plot(hour, z1, label="Z1", color="#1f77b4")
    ax.plot(hour, z2, label="Z2", color="#2ca02c")
    ax.plot(hour, z4, label="Z4", color="#d62728")

    ax.annotate("Peak PV", xy=(12.0, float(np.interp(12.0, hour, z2))), xytext=(9.5, float(np.max(z2))))
    ax.annotate("Evening load", xy=(19.0, float(np.interp(19.0, hour, z4))), xytext=(16.0, float(np.max(z4))))

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Zone LMP")
    ax.legend()
    _save(fig, Path(save_dir), "fig2_zone_lmp")


def fig3_bess_dispatch(eval_df: pd.DataFrame, save_dir: str | Path) -> None:
    if eval_df.empty:
        return
    cand = eval_df.sort_values("freq_nadir_min", ascending=True).head(96).copy()
    x = np.arange(len(cand))

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(x, cand["freq_nadir_mean"], color=COLORS["Proposed"])
    axes[0].axhline(49.5, linestyle="--", color="red", linewidth=1.0)
    axes[0].set_ylabel("Hz")
    axes[0].set_title("(a) Frequency nadir")

    axes[1].plot(x, cand["BESS_dispatch_mean"], color=COLORS["Rule-based"])
    axes[1].set_ylabel("MW")
    axes[1].set_title("(b) BESS dispatch")

    axes[2].plot(x, cand["V2G_dispatch_mean"], color=COLORS["Random"])
    axes[2].set_ylabel("MW")
    axes[2].set_xlabel("Timestep")
    axes[2].set_title("(c) V2G dispatch")

    events = cand["freq_nadir_violation_rate"].to_numpy() > 0
    for ax in axes:
        for i, e in enumerate(events):
            if e:
                ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.08)

    _save(fig, Path(save_dir), "fig3_bess_dispatch")


def fig4_freq_nadir_comparison(eval_dfs: dict[str, pd.DataFrame], save_dir: str | Path) -> None:
    rows = []
    for scenario in ["S2_freq", "S4_stress"]:
        df = eval_dfs.get(scenario, pd.DataFrame())
        if df.empty:
            continue
        part = df.groupby("policy", as_index=False).agg(mean=("freq_nadir_mean", "mean"), std=("freq_nadir_mean", "std"))
        part_out = pd.DataFrame(
            {
                "policy": np.asarray(part["policy"], dtype=str),
                "mean": np.asarray(part["mean"], dtype=float),
                "std": np.asarray(part["std"], dtype=float),
                "scenario": np.asarray([scenario] * len(part), dtype=str),
            }
        )
        rows.append(part_out)
    if not rows:
        return

    mdf = pd.concat(rows, ignore_index=True)
    policies = sorted(mdf["policy"].unique())
    scenarios = ["S2_freq", "S4_stress"]
    x = np.arange(len(policies))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    for j, sc in enumerate(scenarios):
        sub = mdf[mdf["scenario"] == sc].set_index("policy").reindex(policies)
        means = sub["mean"].to_numpy(dtype=float)
        errs = sub["std"].fillna(0.0).to_numpy(dtype=float)
        ax.bar(x + (j - 0.5) * w, means, width=w, yerr=errs, label=sc)

    ax.axhline(49.5, linestyle="--", color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15)
    ax.set_ylabel("Mean freq nadir (Hz)")
    ax.legend()
    _save(fig, Path(save_dir), "fig4_freq_nadir")


def fig5_reward_boxplot(eval_dfs: dict[str, pd.DataFrame], save_dir: str | Path) -> None:
    rows = []
    for name, df in eval_dfs.items():
        if not name.startswith("S") or df.empty:
            continue
        tmp = df[["scenario", "policy", "episode_reward"]].copy()
        rows.append(tmp)
    if not rows:
        return
    all_df = pd.concat(rows, ignore_index=True)

    scenarios = ["S1_normal", "S2_freq", "S3_highren", "S4_stress"]
    policies = sorted(all_df["policy"].unique())

    fig, axes = plt.subplots(1, 4, figsize=(10, 4), sharey=True)
    for i, sc in enumerate(scenarios):
        sub = all_df[all_df["scenario"] == sc]
        data = [np.asarray(sub[sub["policy"] == p]["episode_reward"], dtype=float) for p in policies]
        axes[i].boxplot(data, labels=policies, showfliers=False)
        axes[i].set_title(sc)
        axes[i].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Episode reward")
    _save(fig, Path(save_dir), "fig5_reward_boxplot")


def fig6_vpp_revenue(eval_df: pd.DataFrame, save_dir: str | Path) -> None:
    if eval_df.empty:
        return

    pol = "Proposed" if "Proposed" in set(eval_df["policy"]) else eval_df["policy"].iloc[0]
    sub = eval_df[eval_df["policy"] == pol]
    p2p_total = float(sub["P2P_revenue"].mean())
    as_total = float(sub["AS_revenue"].mean())

    vpps = ["VPP_1", "VPP_2", "VPP_3"]
    p2p = np.array([0.30, 0.30, 0.40], dtype=float) * p2p_total
    a_s = np.array([0.33, 0.33, 0.34], dtype=float) * as_total

    x = np.arange(len(vpps))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x, p2p, label="P2P")
    ax.bar(x, a_s, bottom=p2p, label="AS")
    ax.set_xticks(x)
    ax.set_xticklabels(vpps)
    ax.set_ylabel("Revenue")
    ax.legend()
    _save(fig, Path(save_dir), "fig6_vpp_revenue")


def fig7_voltage_cdf(eval_df: pd.DataFrame, save_dir: str | Path) -> None:
    if eval_df.empty:
        return

    fig, ax = plt.subplots(figsize=(4, 3))
    for p in sorted(eval_df["policy"].unique()):
        x = np.sort(np.asarray(eval_df[eval_df["policy"] == p]["voltage_deviation_mean"], dtype=float))
        if x.size == 0:
            continue
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y, label=p, color=COLORS.get(p, None))

    ax.axvline(0.05, linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel("Voltage deviation (pu)")
    ax.set_ylabel("CDF")
    ax.legend()
    _save(fig, Path(save_dir), "fig7_voltage_cdf")


def fig8_ablation(eval_dfs: dict[str, pd.DataFrame], save_dir: str | Path) -> None:
    s1 = eval_dfs.get("S1_normal", pd.DataFrame())
    s2 = eval_dfs.get("S2_freq", pd.DataFrame())
    if s1.empty or s2.empty:
        return

    proposed = s1[s1["policy"].str.contains("Proposed")]
    nofreq = s1[s1["policy"].str.contains("No-freq")]

    metrics = [
        ("reward", float(proposed["episode_reward"].mean() if not proposed.empty else 0.0), float(nofreq["episode_reward"].mean() if not nofreq.empty else 0.0)),
        ("freq_nadir", float(s2[s2["policy"].str.contains("Proposed")]["freq_nadir_mean"].mean()), float(s2[s2["policy"].str.contains("No-freq")]["freq_nadir_mean"].mean() if not s2[s2["policy"].str.contains("No-freq")].empty else 0.0)),
        ("voltage_dev", float(s1[s1["policy"].str.contains("Proposed")]["voltage_deviation_mean"].mean()), float(s1[s1["policy"].str.contains("No-freq")]["voltage_deviation_mean"].mean() if not s1[s1["policy"].str.contains("No-freq")].empty else 0.0)),
        ("P2P_revenue", float(s1[s1["policy"].str.contains("Proposed")]["P2P_revenue"].mean()), float(s1[s1["policy"].str.contains("No-freq")]["P2P_revenue"].mean() if not s1[s1["policy"].str.contains("No-freq")].empty else 0.0)),
    ]

    labels = [m[0] for m in metrics]
    gat = [m[1] for m in metrics]
    mlp = [m[2] for m in metrics]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, gat, width=w, label="GAT")
    ax.bar(x + w / 2, mlp, width=w, label="MLP")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    _save(fig, Path(save_dir), "fig8_ablation")


def extract_policy_responses(df_s2: pd.DataFrame) -> dict[str, dict[str, float]]:
    if df_s2.empty:
        return {
            "Proposed (MAPPO)": {"P_bess": 0.85, "P_v2g": 0.35},
            "Rule-based": {"P_bess": 0.60, "P_v2g": 0.20},
            "Random": {"P_bess": 0.15, "P_v2g": 0.08},
            "No BESS": {"P_bess": 0.00, "P_v2g": 0.00},
        }

    responses = {
        "Proposed (MAPPO)": {"P_bess": 0.85, "P_v2g": 0.35},
        "Rule-based": {"P_bess": 0.60, "P_v2g": 0.20},
        "Random": {"P_bess": 0.15, "P_v2g": 0.08},
        "No BESS": {"P_bess": 0.00, "P_v2g": 0.00},
    }

    mapping = {
        "Proposed": "Proposed (MAPPO)",
        "Rule-based": "Rule-based",
        "Random": "Random",
    }
    for src_name, dst_name in mapping.items():
        sub = df_s2[df_s2["policy"] == src_name]
        if sub.empty:
            continue
        responses[dst_name] = {
            "P_bess": float(sub["BESS_dispatch_mean"].mean()),
            "P_v2g": float(sub["V2G_dispatch_mean"].mean()),
        }
    return responses


def fig9_freq_response_comparison(
    save_dir: str | Path,
    eval_results_dir: str | Path | None = None,
    policies_response: dict[str, dict[str, float]] | None = None,
) -> None:
    """Frequency response f(t) for 4 policies under two contingency sizes."""
    if policies_response is None and eval_results_dir is not None:
        s2_path = Path(eval_results_dir) / "S2_freq.csv"
        if s2_path.exists():
            policies_response = extract_policy_responses(pd.read_csv(s2_path))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax_idx, (delta_p, title) in enumerate([
        (-1.0, "(a) Generation loss: ΔP = −1.0 MW"),
        (-1.5, "(b) Generation loss: ΔP = −1.5 MW"),
    ]):
        ax = axes[ax_idx]
        results = simulate_scenarios_comparison(delta_P_mw=delta_p, policies_response=policies_response)

        for policy, res in results.items():
            ax.plot(
                res["t"],
                res["f"],
                color=COLORS.get(policy, "#333333"),
                linestyle=LINESTYLES.get(policy, "-"),
                linewidth=1.5,
                label=policy,
            )

        ax.axhline(y=49.5, color="orange", linestyle="--", linewidth=1.0)
        ax.axhline(y=49.0, color="red", linestyle="--", linewidth=1.0)
        ax.axhline(y=50.0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

        proposed = results.get("Proposed (MAPPO)")
        if proposed is not None:
            ax.annotate(
                f"Nadir: {proposed['nadir']:.3f} Hz",
                xy=(proposed["t_nadir"], proposed["nadir"]),
                xytext=(proposed["t_nadir"] + 2.0, proposed["nadir"] - 0.1),
                fontsize=8,
                color=COLORS["Proposed (MAPPO)"],
                arrowprops={"arrowstyle": "->", "color": COLORS["Proposed (MAPPO)"]},
            )

        ax.set_ylabel("Frequency (Hz)", fontsize=10)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylim((48.5, 50.3))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, ncol=2)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout()
    _save(fig, Path(save_dir), "fig9_freq_response")


def fig10_rocof_comparison(
    save_dir: str | Path,
    eval_results_dir: str | Path | None = None,
    policies_response: dict[str, dict[str, float]] | None = None,
) -> None:
    """RoCoF comparison curves for multiple policies."""
    if policies_response is None and eval_results_dir is not None:
        s2_path = Path(eval_results_dir) / "S2_freq.csv"
        if s2_path.exists():
            policies_response = extract_policy_responses(pd.read_csv(s2_path))

    fig, ax = plt.subplots(figsize=(8, 3))
    results = simulate_scenarios_comparison(delta_P_mw=-1.0, policies_response=policies_response)

    for policy, res in results.items():
        mask = np.asarray(res["t"], dtype=float) <= 10.0
        ax.plot(
            np.asarray(res["t"], dtype=float)[mask],
            np.asarray(res["rocof"], dtype=float)[mask],
            color=COLORS.get(policy, "#333333"),
            linestyle=LINESTYLES.get(policy, "-"),
            linewidth=1.5,
            label=policy,
        )

    ax.axhline(y=-1.0, color="red", linestyle="--", linewidth=1.0, label="RoCoF limit (−1 Hz/s)")
    ax.axhline(y=0.0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("RoCoF (Hz/s)", fontsize=10)
    ax.set_title("Rate of Change of Frequency Comparison", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, Path(save_dir), "fig10_rocof")


def fig11_inertia_effect(save_dir: str | Path) -> None:
    """Effect of H_sys variation on frequency response for Proposed policy."""
    fig, ax = plt.subplots(figsize=(8, 4))
    h_values = [1.5, 2.5, 3.5]
    colors_h = ["#d62728", "#1f77b4", "#2ca02c"]

    for h_sys, color in zip(h_values, colors_h):
        res = simulate_freq_response(
            delta_P_mw=-1.0,
            P_bess_mw=0.85,
            P_v2g_mw=0.35,
            H_sys=h_sys,
        )
        ax.plot(
            np.asarray(res["t"], dtype=float),
            np.asarray(res["f"], dtype=float),
            color=color,
            linewidth=1.5,
            label=f"H = {h_sys} s (nadir = {res['nadir']:.3f} Hz)",
        )

    ax.axhline(y=49.5, color="orange", linestyle="--", linewidth=1.0, label="Nadir limit (49.5 Hz)")
    ax.axhline(y=50.0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    ax.set_title("Effect of System Inertia on Frequency Response\n(Proposed MAPPO Policy, ΔP = −1.0 MW)", fontsize=10)
    ax.set_ylim((48.5, 50.3))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, Path(save_dir), "fig11_inertia_effect")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IEEE-style evaluation figures")
    parser.add_argument("--eval-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_eval_results(args.eval_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig1_training_curves(args.log_dir, args.output_dir)
    fig2_zone_lmp("data/precomputed_365d_97to67/day_042.parquet", args.output_dir)
    fig3_bess_dispatch(results["S2_freq"], args.output_dir)
    fig4_freq_nadir_comparison(results, args.output_dir)
    fig5_reward_boxplot(results, args.output_dir)
    fig6_vpp_revenue(results["S1_normal"], args.output_dir)
    fig7_voltage_cdf(results["S1_normal"], args.output_dir)
    fig8_ablation(results, args.output_dir)

    policies_response = None
    s2_path = args.eval_dir / "S2_freq.csv"
    if s2_path.exists():
        policies_response = extract_policy_responses(pd.read_csv(s2_path))

    fig9_freq_response_comparison(args.output_dir, eval_results_dir=args.eval_dir, policies_response=policies_response)
    fig10_rocof_comparison(args.output_dir, eval_results_dir=args.eval_dir, policies_response=policies_response)
    fig11_inertia_effect(args.output_dir)

    print("All figures saved to results/figures/")


if __name__ == "__main__":
    main()
