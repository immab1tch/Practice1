"""Overview NEW polished figure: executive dashboard.
A single multi-panel summary built only from frozen M1-M5 results."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (setup, save, TEAL, GOLD, CORAL, SAGE, ROSE, INK,
                    TEAL_SOFT, WARM_GRAY)

PROJ = Path(__file__).resolve().parent.parent
M1 = PROJ / "Code" / "M1_equilibrium" / "results"
M4 = PROJ / "Code" / "M4_migration" / "results"
M5 = PROJ / "Code" / "M5_spatial" / "results"
OUT = Path(__file__).resolve().parent / "Overview"


def kpi(ax, value, label, color):
    ax.axis("off")
    box = FancyBboxPatch((0.06, 0.14), 0.88, 0.72,
                         boxstyle="round,pad=0.02,rounding_size=0.06",
                         facecolor=color, edgecolor="none", alpha=0.16,
                         transform=ax.transAxes)
    ax.add_patch(box)
    ax.text(0.5, 0.58, value, ha="center", va="center", fontsize=21,
            fontweight="bold", color=color)
    ax.text(0.5, 0.26, label, ha="center", va="center", fontsize=10.5,
            color=INK)


def fig_dashboard():
    rec = pd.read_csv(M1 / "recommended.csv").iloc[0]
    fig = plt.figure(figsize=(13.6, 8.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.05], hspace=0.22,
                             left=0.05, right=0.97, top=0.90, bottom=0.07)
    # ---- KPI row ----
    kgs = outer[0].subgridspec(1, 4, wspace=0.22)
    kpis = [
        (f"{rec.x/100:.3f}M", "Annual visitors", TEAL),
        (f"{rec.R*100:.1f}M", "Tourism revenue (USD)", GOLD),
        (f"{rec.tau*100:.2f}%", "Tax rate", CORAL),
        (f"{rec.S:.3f}", "Resident satisfaction", SAGE),
    ]
    for i, (v, l, c) in enumerate(kpis):
        kpi(fig.add_subplot(kgs[i]), v, l, c)

    # ---- Mini charts row ----
    cgs = outer[1].subgridspec(1, 3, wspace=0.30)

    # 1. New investment
    ax1 = fig.add_subplot(cgs[0])
    cats = ["Infra.", "Ecology", "Community"]
    vals = [rec.spend_I * 100, rec.spend_E * 100, rec.spend_C * 100]
    bars = ax1.bar(cats, vals, color=[TEAL, GOLD, CORAL], width=0.62)
    ax1.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax1.set_title("New investment (million USD)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(vals) * 1.2)
    ax1.tick_params(labelsize=9)
    ax1.grid(axis="y", color="#E4DED4", linewidth=0.7)
    ax1.set_axisbelow(True)

    # 2. Utilization before vs after
    ax2 = fig.add_subplot(cgs[1])
    before = pd.read_csv(M5 / "M5_allocation_before.csv")
    after = pd.read_csv(M5 / "M5_allocation_after.csv")
    x = np.arange(4)
    ax2.bar(x - 0.2, before.utilization, 0.38, color=WARM_GRAY, label="Before")
    ax2.bar(x + 0.2, after.utilization, 0.38, color=TEAL, label="After")
    ax2.axhline(0.80, color=ROSE, ls="--", lw=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(before.node_id, fontsize=9)
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Node utilization (target 0.80)", fontsize=11, fontweight="bold")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax2.grid(axis="y", color="#E4DED4", linewidth=0.7)
    ax2.set_axisbelow(True)

    # 3. WTP survival
    ax3 = fig.add_subplot(cgs[2])
    fee = pd.read_csv(M4 / "M4_real_fee_scenarios.csv")
    ax3.step(fee.fee_eur, fee.wtp_survival, where="mid", color=CORAL, lw=2,
             marker="o", ms=4.5)
    ax3.set_xlabel("Entrance fee (EUR)", fontsize=9.5)
    ax3.set_ylabel("WTP survival", fontsize=9.5)
    ax3.set_ylim(0, 0.78)
    ax3.set_title("Willingness to pay (Venice evidence)", fontsize=11,
                  fontweight="bold")
    ax3.tick_params(labelsize=9)
    ax3.grid(color="#E4DED4", linewidth=0.7)
    ax3.set_axisbelow(True)

    fig.suptitle("Sustainable tourism management: model overview",
                 fontweight="bold", fontsize=17, y=0.975)
    save(fig, OUT, "OVP_01_Executive_dashboard")


if __name__ == "__main__":
    setup()
    fig_dashboard()
    print("Overview dashboard written.")
