"""Parse the 3 completed base-only training logs and assess learning curves.

Reward/entropy scales differ across methods (different reward defs + policy
parameterisations), so we assess convergence PER METHOD, not cross-method.
Cross-method comparison is only valid on the final eval IAE.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path("artifacts")


def parse_am(path: Path):
    """GraphSAGE: per-phase 'Ep K | R= V | ... Vio=.. VioPost=.. Nadir=.. H=..'."""
    g, R, H, vio, viopost, nadir, bounds = [], [], [], [], [], [], []
    offset, last_len = 0, 0
    ph = re.compile(r"=== Phase (\w):.*\((\d+) ep")
    dl = re.compile(r"Ep\s+(\d+)\s*\|\s*R=\s*([-\d.]+).*?Vio=([-\d.]+)\s*\|\s*VioPost=([-\d.]+)\s*\|\s*Nadir=([-\d.]+).*?H=([-\d.]+)")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = ph.search(line)
        if m:
            offset += last_len
            last_len = int(m.group(2))
            bounds.append(offset)
            continue
        d = dl.search(line)
        if d:
            ep = int(d.group(1))
            g.append(offset + ep)
            R.append(float(d.group(2)))
            vio.append(float(d.group(3)))
            viopost.append(float(d.group(4)))
            nadir.append(float(d.group(5)))
            H.append(float(d.group(6)))
    return dict(g=g, R=R, H=H, vio=vio, viopost=viopost, nadir=nadir, bounds=bounds)


def parse_mlp(path: Path):
    """MLP: '[Ep K] Phase P | Reward: V | Loss: .. | Entropy: E'."""
    g, R, H = [], [], []
    dl = re.compile(r"\[Ep\s+(\d+)\]\s*Phase\s*\w+\s*\|\s*Reward:\s*([-\d.]+).*?Entropy:\s*([-\d.]+)")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        d = dl.search(line)
        if d:
            g.append(int(d.group(1))); R.append(float(d.group(2))); H.append(float(d.group(3)))
    return dict(g=g, R=R, H=H, bounds=[])


def parse_gcnn(path: Path):
    """GCNN: 'phase=P ep=.. total=T r_fast=V r_slow=.. loss=.. entropy=E'."""
    g, R, H = [], [], []
    dl = re.compile(r"total=\s*(\d+)\s+r_fast=([-\d.]+).*?entropy=([-\d.]+)")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        d = dl.search(line)
        if d:
            g.append(int(d.group(1))); R.append(float(d.group(2))); H.append(float(d.group(3)))
    return dict(g=g, R=R, H=H, bounds=[])


def smooth(y, k=9):
    if len(y) < k:
        return np.asarray(y, float)
    return np.convolve(y, np.ones(k) / k, mode="same")


runs = {
    "GraphSAGE (proposed)": parse_am(ART / "fullretrain_am_base.log"),
    "MLP (ablation)":       parse_mlp(ART / "fullretrain_mlp_base.log"),
    "GCNN-PPO (baseline)":  parse_gcnn(ART / "fullretrain_gcnn_base.log"),
}

# ---- figure: rows=method, cols=[reward, entropy] ----
fig, axes = plt.subplots(3, 2, figsize=(13, 10))
for i, (name, d) in enumerate(runs.items()):
    g = np.asarray(d["g"], float)
    for j, (key, lab) in enumerate([("R", "reward"), ("H", "entropy")]):
        ax = axes[i, j]
        y = np.asarray(d[key], float)
        ax.plot(g, y, lw=0.6, alpha=0.35, color="tab:blue")
        ax.plot(g, smooth(y), lw=1.8, color="tab:blue")
        for b in d["bounds"]:
            ax.axvline(b, color="grey", ls=":", lw=0.7)
        ax.set_title(f"{name} — {lab}", fontsize=10)
        ax.set_xlabel("global episode")
        ax.grid(alpha=0.3)
fig.suptitle("Base-only training curves (scales differ per method — assess convergence, not cross-method level)", fontsize=11)
fig.tight_layout()
out = ART / "base_training_curves.png"
fig.savefig(out, dpi=110)
print(f"Saved figure: {out}")

# ---- quantitative summary ----
def seg(y, frac0, frac1):
    y = np.asarray(y, float); n = len(y)
    if n == 0: return float("nan")
    a, b = int(n * frac0), max(int(n * frac1), int(n * frac0) + 1)
    return float(np.mean(y[a:b]))

print("\n%-22s %8s %8s %8s | %8s %8s | %s" % ("method", "R_start", "R_end", "R_best", "H_start", "H_end", "n_pts"))
for name, d in runs.items():
    R, H = d["R"], d["H"]
    print("%-22s %8.2f %8.2f %8.2f | %8.3f %8.3f | %d" % (
        name, seg(R, 0, 0.05), seg(R, 0.95, 1.0), (max(R) if R else float('nan')),
        seg(H, 0, 0.05), seg(H, 0.95, 1.0), len(R)))

# am-specific safety trends
am = runs["GraphSAGE (proposed)"]
if am["vio"]:
    print("\nGraphSAGE safety trend (start -> end):")
    print("  Vio       %.4f -> %.4f" % (seg(am["vio"], 0, 0.05), seg(am["vio"], 0.95, 1.0)))
    print("  VioPost   %.4f -> %.4f" % (seg(am["viopost"], 0, 0.05), seg(am["viopost"], 0.95, 1.0)))
    print("  Nadir(Hz) %.4f -> %.4f" % (seg(am["nadir"], 0, 0.05), seg(am["nadir"], 0.95, 1.0)))
