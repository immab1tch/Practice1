"""M4 polished figures: combined feature weights, TOPSIS priorities,
and a NEW combined WTP + gross-revenue decision chart.
Reads frozen M4 results; values are not recomputed."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (setup, save, hbar, cmap_teal, cmap_gold, cmap_heat,
                    cmap_rose, TEAL, GOLD, INK)

M4 = Path(__file__).resolve().parent.parent / "Code" / "M4_migration" / "results"
OUT = Path(__file__).resolve().parent / "M4_migration"


def fig_weights():
    df = pd.read_csv(M4 / "M4_feature_weights.csv").sort_values(
        "combined", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    y = np.arange(len(df))
    h = 0.60
    for i, r in df.iterrows():
        hbar(ax, y[i] - h / 2, r.combined, h, cmap_teal, rounding=0.10)
        ax.text(r.combined + 0.004, y[i], f"{r.combined:.3f}", va="center",
                fontsize=9.5, color=INK)
        ax.scatter(r.ahp, y[i], color=GOLD, s=34, zorder=4, edgecolor="white")
        ax.scatter(r.entropy, y[i], color="#B96583", s=34, marker="D",
                   zorder=4, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.feature_id}  {r.feature_name}"
                        for _, r in df.iterrows()], fontsize=9.6)
    ax.set_xlabel("Weight (combined AHP + entropy)")
    all_max = max(df.combined.max(), df.ahp.max(), df.entropy.max())
    ax.set_xlim(0, all_max * 1.12)
    ax.set_title("Combined feature weights with component sources",
                 fontweight="bold", fontsize=14, pad=12)
    ax.legend(handles=[Patch(color=TEAL, label="Combined weight"),
                       Patch(color=GOLD, label="AHP component"),
                       Patch(color="#B96583", label="Entropy component")],
              frameon=False, loc="lower right", fontsize=9.5)
    ax.grid(axis="x", color="#E4DED4", linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, OUT, "M4P_01_Combined_feature_weights")


def fig_topsis():
    df = pd.read_csv(M4 / "M4_policy_ranking.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), sharex=True)
    for ax, dest, cmap in zip(axes, ["Juneau", "Venice"], [cmap_teal, cmap_gold]):
        sub = df[df.destination == dest].sort_values("closeness",
                                                     ascending=True).reset_index(drop=True)
        y = np.arange(len(sub))
        h = 0.62
        for i, r in sub.iterrows():
            hbar(ax, y[i] - h / 2, r.closeness, h, cmap, rounding=0.10)
            ax.text(r.closeness + 0.012, y[i], f"{r.closeness:.2f}", va="center",
                    fontsize=9.8, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(sub.policy_name, fontsize=10)
        ax.set_xlabel("TOPSIS closeness coefficient")
        ax.set_xlim(0, sub.closeness.max() * 1.20)
        ax.set_title(dest, fontweight="bold", fontsize=13)
        ax.grid(axis="x", color="#E4DED4", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("TOPSIS policy-priority ranking", fontweight="bold",
                 fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, OUT, "M4P_02_TOPSIS_policy_priorities")


def fig_wtp_revenue():
    df = pd.read_csv(M4 / "M4_real_fee_scenarios.csv")
    fig, ax1 = plt.subplots(figsize=(10.4, 6.0))
    x = np.arange(len(df))
    # gross revenue bars (right axis), gold gradient emulation via solid Morandi
    gross = df.gross_fee_revenue_eur / 1e6
    bars = ax1.bar(x, gross, 0.60, color=GOLD, alpha=0.85, zorder=2,
                   label="Gross fee revenue (left axis)")
    ax1.bar_label(bars, fmt="%.2f", padding=3, fontsize=8.6)
    ax1.set_ylabel("Gross fee revenue (million EUR)")
    ax1.set_ylim(0, gross.max() * 1.20)
    ax1.set_xlabel("Entrance fee (EUR)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df.fee_eur.astype(str))
    # WTP survival step line (left axis)
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.step(x, df.wtp_survival, where="mid", color=TEAL, linewidth=2.2,
             marker="o", ms=6, zorder=4, label="WTP survival (right axis)")
    ax2.set_ylabel("Share willing to pay (survival)")
    ax2.set_ylim(0, 0.82)
    ax1.set_title("Willingness to pay and conditional gross revenue",
                  fontweight="bold", fontsize=14, pad=12)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h2 + h1, l2 + l1, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9.6)
    ax1.grid(axis="y", color="#E4DED4", linewidth=0.8)
    ax1.set_axisbelow(True)
    save(fig, OUT, "M4P_03_WTP_revenue_combined")


if __name__ == "__main__":
    setup()
    fig_weights()
    fig_topsis()
    fig_wtp_revenue()
    print("M4 polished figures written.")
