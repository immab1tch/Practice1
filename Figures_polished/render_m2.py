"""M2 NEW polished figure: fund-allocation donut.
Reads the frozen M2 ledger; values are not recomputed."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import setup, save, TEAL, GOLD, CORAL, ROSE, SAGE, SLATE, INK

M2 = Path(__file__).resolve().parent.parent / "Code" / "M2_budget" / "results"
OUT = Path(__file__).resolve().parent / "M2_budget"


def fig_donut():
    df = pd.read_csv(M2 / "M2_budget_ledger.csv")
    sub = df[(df.case == "default") & (df.million_usd > 0)].copy()
    labels = {
        "infrastructure_new": "Infrastructure (new)",
        "ecology_new": "Ecology (new)",
        "community_new": "Community (new)",
        "measures": "Measures (annualized)",
        "maintenance_charged": "Maintenance charged",
        "available_unallocated": "Available, unallocated",
    }
    colors = [TEAL, GOLD, CORAL, ROSE, SAGE, SLATE]
    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    wedges, _ = ax.pie(sub.million_usd, colors=colors, startangle=90,
                       counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    total = sub.million_usd.sum()
    ax.text(0, 0.08, f"{total:.2f}", ha="center", va="center",
            fontsize=19, fontweight="bold", color=INK)
    ax.text(0, -0.16, "million USD / year", ha="center", va="center",
            fontsize=10, color="#6b645c")
    legend_labels = [f"{labels[c]}  ({v:.2f}, {v / total * 100:.1f}%)"
                     for c, v in zip(sub.category, sub.million_usd)]
    ax.legend(wedges, legend_labels, frameon=False, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=10)
    ax.set_title("Allocation of annual tax revenue", fontweight="bold",
                 fontsize=14, pad=16)
    save(fig, OUT, "M2P_01_Fund_allocation_donut")


if __name__ == "__main__":
    setup()
    fig_donut()
    print("M2 polished donut written.")
