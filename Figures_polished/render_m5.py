"""M5 polished figures: gradient gravity allocation, utilization vs target,
and a smooth transfer heatmap. Reads frozen M5 results; values not recomputed."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (setup, save, vbar, cmap_teal, cmap_coral, cmap_heat,
                    TEAL_SOFT, CORAL_SOFT, WARM_GRAY, TEAL, CORAL, INK)

M5 = Path(__file__).resolve().parent.parent / "Code" / "M5_spatial" / "results"
OUT = Path(__file__).resolve().parent / "M5_spatial"


def fig_gravity():
    before = pd.read_csv(M5 / "M5_allocation_before.csv")
    after = pd.read_csv(M5 / "M5_allocation_after.csv")
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    x = np.arange(4)
    w = 0.36
    for i in range(4):
        hot = before.iloc[i].type == "hot"
        soft = TEAL_SOFT if hot else CORAL_SOFT
        cmap = cmap_teal if hot else cmap_coral
        ax.bar(x[i] - w / 2, before.iloc[i].visitors_10k, w, color=soft,
               zorder=2, label="Before" if i == 0 else None)
        vbar(ax, x[i] + w / 2, after.iloc[i].visitors_10k, w, cmap,
             rounding=0.12)
        ax.text(x[i] + w / 2, after.iloc[i].visitors_10k + 0.9,
                f"{after.iloc[i].visitors_10k:.1f}", ha="center", fontsize=9.4,
                color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.node_id}\n{r.node_name}" for _, r in before.iterrows()],
                       fontsize=10)
    ax.set_ylabel("Visitors (10k, season)")
    ax.set_ylim(0, before.visitors_10k.max() * 1.16)
    ax.set_title("Gravity allocation before vs after cold-point investment",
                 fontweight="bold", fontsize=14, pad=12)
    ax.legend(handles=[Patch(color=TEAL_SOFT, label="Before investment"),
                       Patch(color="#6FA6AE", label="After 5M USD (hot node)"),
                       Patch(color="#DE8868", label="After 5M USD (cold node)")],
              frameon=False, loc="upper right", fontsize=9.4)
    ax.grid(axis="y", color="#E4DED4", linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, OUT, "M5P_01_Gravity_allocation")


def fig_utilization():
    before = pd.read_csv(M5 / "M5_allocation_before.csv")
    after = pd.read_csv(M5 / "M5_allocation_after.csv")
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    x = np.arange(4)
    w = 0.36
    ax.bar(x - w / 2, before.utilization, w, color=WARM_GRAY, zorder=2,
           label="Before")
    colors = [TEAL if u < 0.80 else CORAL for u in after.utilization]
    bars = ax.bar(x + w / 2, after.utilization, w, color=colors, zorder=2,
                  label="After")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.axhline(0.80, color="#B96583", linestyle="--", linewidth=1.4,
               label="Target utilization 0.80")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.node_id} {r.node_name}" for _, r in before.iterrows()],
                       fontsize=9.8)
    ax.set_ylabel("Capacity utilization")
    ax.set_ylim(0, 1.05)
    ax.set_title("Node utilization against the 0.80 pressure target",
                 fontweight="bold", fontsize=14, pad=12)
    ax.legend(frameon=False, fontsize=9.4, loc="upper right")
    ax.grid(axis="y", color="#E4DED4", linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, OUT, "M5P_02_Utilization_target")


def fig_heatmap():
    t = pd.read_csv(M5 / "M5_transfer_matrix.csv", index_col=0)
    data = t.values
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    im = ax.imshow(data, cmap=cmap_heat, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(t.columns)
    ax.set_yticklabels(t.index)
    for i in range(4):
        for j in range(4):
            v = data[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=10, color="white" if v > data.max() * 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Visitors (10k, season)", fontsize=9.6)
    ax.set_xlabel("Destination node")
    ax.set_ylabel("Origin node")
    ax.set_title("Visitor transfer matrix after investment",
                 fontweight="bold", fontsize=14, pad=12)
    save(fig, OUT, "M5P_03_Transfer_heatmap")


if __name__ == "__main__":
    setup()
    fig_gravity()
    fig_utilization()
    fig_heatmap()
    print("M5 polished figures written.")
