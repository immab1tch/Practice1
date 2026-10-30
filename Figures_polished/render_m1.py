"""M1 polished figures: gradient investment bars + color-mapped Pareto front.
Reads frozen M1 results; values are not recomputed."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (setup, save, vbar, cmap_teal, cmap_coral, cmap_gold,
                    cmap_heat, TEAL, CORAL, ROSE, INK)

M1 = Path(__file__).resolve().parent.parent / "Code" / "M1_equilibrium" / "results"
OUT = Path(__file__).resolve().parent / "M1_equilibrium"


def fig_investment():
    rec = pd.read_csv(M1 / "recommended.csv").iloc[0]
    cats = ["Infrastructure", "Ecology", "Community"]
    # stored in 100M USD -> million USD
    vals = [rec.spend_I * 100, rec.spend_E * 100, rec.spend_C * 100]
    cmaps = [cmap_teal, cmap_gold, cmap_coral]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    x = np.arange(3)
    width = 0.56
    for i, (v, c) in enumerate(zip(vals, cmaps)):
        vbar(ax, x[i] - width / 2, v, width, c, rounding=0.14)
        ax.text(x[i], v + 0.5, f"{v:.2f}", ha="center", fontsize=12,
                fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=12)
    ax.set_ylabel("New investment (million USD / year)")
    ax.set_ylim(0, max(vals) * 1.16)
    ax.set_title("Round-2 new investment allocation", fontweight="bold",
                 fontsize=14, pad=12)
    ax.grid(axis="y", color="#E4DED4", linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, OUT, "M1P_01_New_investment_allocation")


def fig_pareto():
    p = pd.read_csv(M1 / "pareto.csv")
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    sc = ax.scatter(p.R * 100, p.C_env, c=p.S, cmap=cmap_heat,
                    s=46, alpha=0.88, edgecolor="white", linewidth=0.5, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Resident satisfaction S", fontsize=10)
    # recommended point
    rec = pd.read_csv(M1 / "recommended.csv").iloc[0]
    ax.scatter(rec.R * 100, rec.C_env, marker="*", s=420, color=ROSE,
               edgecolor="white", linewidth=1.1, zorder=5, label="Recommended")
    ax.annotate("Recommended\n1.472M visitors", (rec.R * 100, rec.C_env),
                xytext=(16, 18), textcoords="offset points", fontsize=10,
                color=INK, arrowprops=dict(arrowstyle="-", color="#8a8278", lw=0.9))
    ax.set_xlabel("Tourism revenue (million USD / year)")
    ax.set_ylabel("Environmental pressure (lower is better)")
    ax.set_title("Near-nondominated set: revenue vs environmental pressure",
                 fontweight="bold", fontsize=14, pad=12)
    ax.grid(color="#E4DED4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    save(fig, OUT, "M1P_02_Pareto_front_color_mapped")


if __name__ == "__main__":
    setup()
    fig_investment()
    fig_pareto()
    print("M1 polished figures written.")
