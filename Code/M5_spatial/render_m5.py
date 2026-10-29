"""Render M5 English figures (PNG + SVG, 300 dpi) from saved CSV evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE = "#286187"
GOLD = "#BA8B33"
PINK = "#B96583"
GREEN = "#7A8560"
GRAY = "#999999"


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "svg.fonttype": "path"})


def main():
    setup()
    before = pd.read_csv(OUT / "M5_allocation_before.csv")
    after = pd.read_csv(OUT / "M5_allocation_after.csv")
    flow = pd.read_csv(OUT / "M5_network_flow.csv")
    transfer = pd.read_csv(OUT / "M5_transfer_matrix.csv", index_col=0)
    entries = []
    node_ids = before.node_id.tolist()
    names = after.node_name.tolist()
    x = np.arange(len(names))
    w = 0.38

    def save(fig, name, title, note):
        fig.suptitle(title, y=0.99, fontweight="bold", fontsize=15)
        fig.text(0.015, 0.014, note, color="#555555", fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        fig.savefig(FIG / f"{name}.png", dpi=300)
        fig.savefig(FIG / f"{name}.svg")
        plt.close(fig)
        entries.append(dict(file=name, title=title, note=note))

    # 01 Gravity allocation before/after.
    fig, ax = plt.subplots(figsize=(10, 5.6))
    b1 = ax.bar(x - w/2, before.visitors_10k, w, label="Before investment", color=GRAY)
    b2 = ax.bar(x + w/2, after.visitors_10k, w, label="After 5 million USD investment", color=GREEN)
    ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Visitors (10k, season)")
    ax.legend()
    save(fig, "M5_01_Gravity_allocation", "Gravity allocation before and after cold-point investment",
         "The cold nodes (whale watching, rainforest) absorb visitors diverted from the hot nodes.")

    # 02 Utilization with target.
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(x - w/2, before.utilization, w, label="Before", color=GOLD)
    ax.bar(x + w/2, after.utilization, w, label="After", color=BLUE)
    ax.axhline(0.95, color=PINK, ls="--", lw=1.5, label="Glacier baseline 0.95")
    ax.axhline(0.80, color=GREEN, ls="--", lw=1.5, label="Target 0.80")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Utilization (visitors / capacity)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    save(fig, "M5_02_Utilization", "Node utilization: glacier pressure 0.95 -> 0.80",
         "Investment lowers glacier utilization from 0.95 below the 0.80 target without capping total visitors.")

    # 03 Network max flow.
    fig, ax = plt.subplots(figsize=(9, 5.4))
    scen = ["before", "after"]
    xs = np.arange(2)
    ax.bar(xs - w/2, flow.max_flow_10k, w, label="Max flow (road capacity)", color=BLUE)
    ax.bar(xs + w/2, flow.outgoing_demand_10k, w, label="Outgoing peak demand", color=GOLD)
    for i, row in flow.iterrows():
        ax.text(i - w/2, row.max_flow_10k + 0.01, f"{row.max_flow_10k:.2f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(["Before investment", "After investment"])
    ax.set_ylabel("Daily flow (10k visitors)")
    ax.legend()
    save(fig, "M5_03_Network_maxflow", "Max-flow cut-set: road capacity vs peak demand",
         "Before investment the road cut is the bottleneck; transit spending raises max flow to meet demand.")

    # 04 Transfer matrix heatmap.
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    mat = transfer.to_numpy()
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(np.arange(len(node_ids))); ax.set_xticklabels(node_ids)
    ax.set_yticks(np.arange(len(node_ids))); ax.set_yticklabels(node_ids)
    ax.set_xlabel("To node"); ax.set_ylabel("From node")
    for i in range(len(node_ids)):
        for j in range(len(node_ids)):
            if mat[i, j] > 0.01:
                ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() * 0.55 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "M5_04_Transfer_matrix", "Hot-to-cold visitor transfer matrix (10k visitors)",
         "Rows are origin nodes, columns are destination nodes; about 190k visitors shift to the cold nodes.")

    # 05 Coefficient sensitivity (calibration transparency).
    sens = pd.read_csv(OUT / "M5_coefficient_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    labels = ["Calibrated\n(target)", "Neutral\nround", "Conservative", "Optimistic"]
    colors = [GREEN if v else PINK for v in sens.glacier_below_080]
    bars = ax.bar(np.arange(len(sens)), sens.glacier_utilization, 0.6, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.axhline(0.80, color="#333333", ls="--", lw=1.5, label="Target 0.80")
    ax.set_xticks(np.arange(len(sens))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Glacier utilization after investment")
    ax.set_ylim(0, 0.95)
    ax.legend()
    save(fig, "M5_05_Coefficient_sensitivity",
         "Sensitivity to assumed investment-response coefficients",
         "The primary beta set is target-calibrated; neutral coefficients place the glacier at ~0.80 (the boundary), so the exact margin is assumption-dependent.")

    pd.DataFrame(entries).to_csv(OUT / "figure_index_en.csv", index=False, encoding="utf-8-sig")
    print("M5 English figures written to figures/")


if __name__ == "__main__":
    main()
