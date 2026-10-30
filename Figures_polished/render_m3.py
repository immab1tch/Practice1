"""M3 polished: gradient diverging sensitivity ranking.
Reads frozen M3 sensitivity ranking; values are not recomputed."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import setup, save, hbar, cmap_teal, cmap_coral, INK

M3 = Path(__file__).resolve().parent.parent / "Code" / "M3_robustness" / "results"
OUT = Path(__file__).resolve().parent / "M3_robustness"


def fig_sensitivity():
    df = pd.read_csv(M3 / "M3_sensitivity_ranking.csv").sort_values(
        "max_abs_delta_J", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    y = np.arange(len(df))
    h = 0.62
    for i, r in df.iterrows():
        # negative arm (teal, extends left from 0 to delta_J_minus)
        hbar(ax, y[i] - h / 2, 0.0, h, cmap_teal,
             x0=r.delta_J_minus, rounding=0.10)
        # positive arm (coral, extends right)
        hbar(ax, y[i] - h / 2, r.delta_J_plus, h, cmap_coral,
             x0=0.0, rounding=0.10)
        ax.text(r.delta_J_minus - 0.006, y[i], f"{r.delta_J_minus:+.3f}",
                ha="right", va="center", fontsize=8.6, color=INK)
        ax.text(r.delta_J_plus + 0.006, y[i], f"{r.delta_J_plus:+.3f}",
                ha="left", va="center", fontsize=8.6, color=INK)
    ax.axvline(0, color="#9a9288", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df.parameter, fontsize=10.5)
    ax.set_xlabel("Score change after re-optimization (parameter x0.9 vs x1.1)")
    ax.set_xlim(df.delta_J_minus.min() * 1.22, df.delta_J_plus.max() * 1.28)
    ax.set_title("Parameter sensitivity ranking (diverging impact)",
                 fontweight="bold", fontsize=14, pad=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#6FA6AE", label="Parameter x 0.9"),
                       Patch(color="#DE8868", label="Parameter x 1.1")],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, fontsize=10)
    ax.grid(axis="x", color="#E4DED4", linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, OUT, "M3P_01_Sensitivity_ranking")


if __name__ == "__main__":
    setup()
    fig_sensitivity()
    print("M3 polished figure written.")
