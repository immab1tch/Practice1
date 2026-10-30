"""Premium upgraded figures that move closer to the O-award visual standard.

All data-driven figures still read the FROZEN result CSVs; no model number is
recomputed. Smoothing (PCHIP) and gradient fills are purely visual, and the
real sample points are kept visible. The conceptual figure is explicitly a
schematic with normalized axes and no fabricated measurements.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (setup, save, TEAL, GOLD, CORAL, SAGE, ROSE, SLATE, INK,
                    cmap_heat, cmap_teal, cmap_gold)

PROJ = Path(__file__).resolve().parent.parent
M1 = PROJ / "Code" / "M1_equilibrium" / "results"
M4 = PROJ / "Code" / "M4_migration" / "results"
M1OUT = Path(__file__).resolve().parent / "M1_equilibrium"
M4OUT = Path(__file__).resolve().parent / "M4_migration"
OVOUT = Path(__file__).resolve().parent / "Overview"


def _style_3d(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#D8D2C8")
        axis._axinfo["grid"]["color"] = "#E6E0D6"
        axis._axinfo["grid"]["linewidth"] = 0.6
    ax.tick_params(labelsize=9, colors="#6b645c")


def fig_pareto_3d():
    df = pd.read_csv(M1 / "pareto.csv")
    rec = pd.read_csv(M1 / "recommended.csv").iloc[0]
    ax = plt.figure(figsize=(10.6, 7.6)).add_subplot(projection="3d")
    sc = ax.scatter(df.x, df.R, df.C_env, c=df.S, cmap=cmap_heat,
                    s=26, alpha=0.9, depthshade=True)
    ax.scatter(rec.x, rec.R, rec.C_env, marker="*", s=420, color=ROSE,
               edgecolor="white", linewidth=0.8, zorder=10)
    ax.set_xlabel("Visitors (10k)", fontsize=10, labelpad=8)
    ax.set_ylabel("Revenue (100M USD)", fontsize=10, labelpad=8)
    ax.set_zlabel("Environmental pressure", fontsize=10, labelpad=8)
    ax.view_init(elev=20, azim=-58)
    _style_3d(ax)
    cb = plt.gcf().colorbar(sc, ax=ax, pad=0.10, shrink=0.75)
    cb.set_label("Resident satisfaction", fontsize=9.5)
    ax.set_title("3D Pareto front: revenue, environment and satisfaction",
                 fontweight="bold", fontsize=14, pad=18)
    save(plt.gcf(), M1OUT, "M1P_03_Pareto_3D")


def _gradient_fill(ax, xs, ys, ymin, cmap):
    poly = ax.fill_between(xs, ys, ymin, facecolor="none")
    grad = np.linspace(0.10, 1.0, 256).reshape(-1, 1)
    im = ax.imshow(grad, extent=(xs.min(), xs.max(), ymin, ys.max()),
                   aspect="auto", cmap=cmap, origin="lower", zorder=1)
    im.set_clip_path(poly.get_paths()[0], transform=ax.transData)


def fig_revenue_curve():
    df = pd.read_csv(M1 / "pareto.csv").sort_values("x")
    rec = pd.read_csv(M1 / "recommended.csv").iloc[0]
    visitors = df.x.values / 100.0                      # million
    revenue = df.R.values * 100.0                       # million USD
    # PCHIP needs strictly increasing x; collapse repeated visitor counts
    # (mean revenue) for the smooth line only. Real points stay visible.
    agg = pd.DataFrame({"v": visitors, "r": revenue}).groupby("v").mean()
    uv, ur = agg.index.values, agg.r.values
    xs = np.linspace(uv.min(), uv.max(), 400)
    ys = PchipInterpolator(uv, ur)(xs)
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    _gradient_fill(ax, xs, ys, 0, cmap_teal)
    ax.plot(visitors, revenue, "o", ms=3.2, color=SLATE, alpha=0.55, zorder=3)
    ax.plot(xs, ys, color=TEAL, lw=2.2, zorder=4)
    rx, ry = rec.x / 100, rec.R * 100
    ax.scatter(rx, ry, marker="*", s=300, color=CORAL, edgecolor="white",
               linewidth=0.7, zorder=6)
    ax.annotate("recommended optimum", xy=(rx, ry),
                xytext=(rx - 0.06, ry + 30), ha="center",
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.4),
                color=CORAL, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Annual visitors (million)")
    ax.set_ylabel("Tourism revenue (million USD)")
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(0, revenue.max() * 1.16)
    ax.grid(axis="y", color="#E4DED4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Revenue response with gradient feasibility region",
                 fontweight="bold", fontsize=14, pad=12)
    save(fig, M1OUT, "M1P_04_Revenue_feasibility")


def fig_radar():
    pr = pd.read_csv(M4 / "M4_policy_ranking.csv")
    j = pr[pr.destination == "Juneau"].set_index("policy_id")
    v = pr[pr.destination == "Venice"].set_index("policy_id")
    ids = j.index.tolist()
    short = ["Capacity", "Shore power", "Visitor cap", "Fee/tax",
             "Rent subsidy", "Transit", "Alcohol control"]
    angles = np.linspace(0, 2 * np.pi, len(ids), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8.8, 8.2), subplot_kw=dict(polar=True))
    for sub, color, name in [(j, TEAL, "Juneau"), (v, GOLD, "Venice")]:
        vals = sub.closeness.loc[ids].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=color, lw=2.2, label=name)
        ax.fill(angles, vals, color=color, alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short, fontsize=10)
    ax.set_ylim(0, 0.8)
    ax.set_rlabel_position(18)
    ax.tick_params(labelsize=8.5, colors="#8a8378")
    ax.spines["polar"].set_color("#D8D2C8")
    ax.grid(color="#E4DED4", linewidth=0.8)
    ax.set_title("Policy-priority profiles (TOPSIS closeness)",
                 fontweight="bold", fontsize=14, pad=24)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.18, 1.10),
              fontsize=10)
    save(fig, M4OUT, "M4P_04_Policy_radar")


def fig_schematic():
    t = np.linspace(0, 1, 300)
    natural = np.exp(-((t - 0.5) ** 2) / 0.055)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    for ax, shaped, title in [
        (axes[0], natural, "(a) Natural season distribution"),
        (axes[1], np.minimum(natural, 0.62), "(b) Policy-shaped distribution"),
    ]:
        ax.fill_between(t, shaped, 0, color=TEAL, alpha=0.22)
        ax.plot(t, shaped, color=TEAL, lw=2.2)
        ax.axhline(0.62, color=ROSE, ls="--", lw=1.4)
        ax.text(0.02, 0.66, "carrying capacity", color=ROSE, fontsize=9.5)
        ax.set_xticks([0.12, 0.5, 0.88])
        ax.set_xticklabels(["off-season", "peak season", "off-season"],
                           fontsize=9.5)
        ax.set_yticks([])
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#C9C2B8")
    fig.text(0.5, 0.02, "Schematic illustration - normalized axes, not to scale",
             ha="center", fontsize=9, color="#8a8378", style="italic")
    fig.suptitle("Visitor flow before and after demand management",
                 fontweight="bold", fontsize=14, y=0.99)
    save(fig, OVOUT, "OVP_02_Distribution_schematic")


if __name__ == "__main__":
    setup()
    fig_pareto_3d()
    fig_revenue_curve()
    fig_radar()
    fig_schematic()
    print("Premium figures written.")
