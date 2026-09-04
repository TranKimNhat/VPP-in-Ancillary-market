"""
Plot THD_V per bus and THD_I per branch for multiple methods.
Input: NPZ with dicts thd_v_per_method / thd_i_per_method.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 6,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
})

COLORS = {
    "GNN-MAPPO (Proposed)": "#D62728",
    "GCNN-PPO": "#1F77B4",
    "AGCN-Decentralized": "#FF7F0E",
    "Graph-PPO": "#2CA02C",
    "Fixed Droop": "#9467BD",
    "No FFR": "#7F7F7F",
}
IEEE519_LIMIT = 5.0


def plot_thd_per_bus(thd_data: dict, output_path: str) -> None:
    methods = list(thd_data.keys())
    n_methods = len(methods)
    n_bus = len(next(iter(thd_data.values())))
    bus_indices = np.arange(n_bus)

    fig, ax = plt.subplots(figsize=(7.16, 3.5))

    bar_w = 0.8 / max(n_methods, 1)
    for mi, method in enumerate(methods):
        thd_v = thd_data[method]
        offset = (mi - (n_methods - 1) / 2) * bar_w
        ax.bar(
            bus_indices + offset,
            thd_v,
            width=bar_w * 0.9,
            color=COLORS.get(method, "#333333"),
            alpha=0.75,
            label=method,
        )

    ax.axhline(
        IEEE519_LIMIT,
        color="red",
        linewidth=1.2,
        linestyle="--",
        label="IEEE 519 limit (5%)",
    )

    ax.set_xlabel("Bus index")
    ax.set_ylabel("THD_V (%)")
    ax.set_title(
        "Voltage THD at All 123 Buses — Method Comparison",
        fontsize=9,
        fontweight="bold",
        pad=4,
    )
    ax.set_xlim(-1, n_bus)
    ax.set_xticks(np.arange(0, n_bus, 10))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6, ncol=3, loc="upper right", framealpha=0.85, edgecolor="lightgrey")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", format="pdf")
    plt.close(fig)


def plot_thd_i_per_branch(thd_data: dict, output_path: str) -> None:
    methods = list(thd_data.keys())
    n_methods = len(methods)
    n_branch = len(next(iter(thd_data.values())))
    branch_indices = np.arange(n_branch)

    fig, ax = plt.subplots(figsize=(7.16, 3.5))

    bar_w = 0.8 / max(n_methods, 1)
    for mi, method in enumerate(methods):
        thd_i = thd_data[method]
        offset = (mi - (n_methods - 1) / 2) * bar_w
        ax.bar(
            branch_indices + offset,
            thd_i,
            width=bar_w * 0.9,
            color=COLORS.get(method, "#333333"),
            alpha=0.75,
            label=method,
        )

    ax.axhline(
        IEEE519_LIMIT,
        color="red",
        linewidth=1.2,
        linestyle="--",
        label="IEEE 519 limit (5%)",
    )

    ax.set_xlabel("Branch index")
    ax.set_ylabel("THD_I (%)")
    ax.set_title(
        "Current THD at All Branches — Method Comparison",
        fontsize=9,
        fontweight="bold",
        pad=4,
    )
    ax.set_xlim(-1, n_branch)
    ax.set_xticks(np.arange(0, n_branch, 10))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6, ncol=3, loc="upper right", framealpha=0.85, edgecolor="lightgrey")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", format="pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/thd_results.npz")
    parser.add_argument("--outdir", default="artifacts/figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data = np.load(args.input, allow_pickle=True)
    thd_v_data = data["thd_v_per_method"].item()
    thd_i_data = data["thd_i_per_method"].item()

    plot_thd_per_bus(thd_v_data, os.path.join(args.outdir, "fig_thd_v_per_bus.png"))
    plot_thd_i_per_branch(thd_i_data, os.path.join(args.outdir, "fig_thd_i_per_branch.png"))


if __name__ == "__main__":
    main()
