"""Render M4 English figures (PNG + SVG, 300 dpi) from saved CSV evidence."""
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
GRAY = "#888888"


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "svg.fonttype": "path"})


def main():
    setup()
    features = pd.read_csv(OUT / "M4_features.csv")
    weights = pd.read_csv(OUT / "M4_feature_weights.csv")
    thresholds = pd.read_csv(OUT / "M4_thresholds.csv")
    ranking = pd.read_csv(OUT / "M4_policy_ranking.csv")
    entries = []

    def save(fig, name, title, note):
        fig.suptitle(title, y=0.99, fontweight="bold", fontsize=15)
        fig.text(0.015, 0.014, note, color="#555555", fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        fig.savefig(FIG / f"{name}.png", dpi=300)
        fig.savefig(FIG / f"{name}.svg")
        plt.close(fig)
        entries.append(dict(file=name, title=title, note=note))

    fnames = weights.feature_name.tolist()
    n = len(fnames)

    # 01 Radar: feature profiles.
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(111, polar=True)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    for dest, color in [("Juneau", BLUE), ("Venice", GOLD)]:
        vals = features[features.destination == dest].sort_values("feature_id", key=lambda s: s.map({f"F{i+1}": i for i in range(n)})).intensity.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=color, lw=2.2, label=dest)
        ax.fill(angles, vals, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(fnames, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8, color="#777777")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.1))
    save(fig, "M4_01_Feature_profiles", "Destination pressure profiles (normalized 0-1)",
         "Semi-quantitative expert coding; Juneau is nature-dominated, Venice is urban/social-dominated.")

    # 02 Feature weights.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(n)
    w = 0.26
    ax.bar(x - w, weights.ahp, w, label="AHP", color=BLUE)
    ax.bar(x, weights.entropy, w, label="Entropy", color=GOLD)
    ax.bar(x + w, weights.combined, w, label="Combined (theta=0.5)", color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(fnames, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Feature weight")
    ax.legend()
    save(fig, "M4_02_Feature_weights", "AHP, entropy and combined feature weights",
         "Entropy downweights features shared by both destinations; combined weights retain the AHP structural importance.")

    # 03 Threshold recalibration (dumbbell).
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = thresholds.name.tolist()
    y = np.arange(len(thresholds))
    ax.hlines(y, thresholds.juneau_base, thresholds.venice_value, color=GRAY, lw=1.5)
    ax.scatter(thresholds.juneau_base, y, color=BLUE, s=70, label="Juneau")
    ax.scatter(thresholds.venice_value, y, color=PINK, s=70, label="Venice (recalibrated)")
    for i, row in thresholds.iterrows():
        ax.annotate(f"{row.venice_value:.2f}", (row.venice_value, i), xytext=(6, -2),
                    textcoords="offset points", fontsize=9, color=PINK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Threshold value")
    ax.legend(loc="lower right")
    save(fig, "M4_03_Threshold_recalibration", "Threshold recalibration from Juneau to Venice",
         "Urban/social features tighten crowding and acceptance thresholds; the absolute revenue floor needs Venice's own baseline.")

    # 04 Policy priorities.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=False)
    for ax, dest, color in [(axes[0], "Juneau", BLUE), (axes[1], "Venice", GOLD)]:
        sub = ranking[ranking.destination == dest].sort_values("closeness")
        bars = ax.barh(sub.policy_name, sub.closeness, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_xlabel("TOPSIS closeness coefficient")
        ax.set_title(dest, fontsize=12)
        ax.set_xlim(0, max(sub.closeness) * 1.18)
    save(fig, "M4_04_Policy_priorities", "TOPSIS policy-priority ranking",
         "Juneau: shore power + capacity expansion; Venice: daily cap + rent subsidy + high tourism fee.")

    pd.DataFrame(entries).to_csv(OUT / "figure_index_en.csv", index=False, encoding="utf-8-sig")
    print("M4 English figures written to figures/")


if __name__ == "__main__":
    main()
