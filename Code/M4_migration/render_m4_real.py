"""Render M4 REAL-data English figures (PNG + SVG, 300 dpi) from evidence CSVs."""
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
    daily = pd.read_csv(OUT / "M4_real_voucher_daily.csv")
    scenarios = pd.read_csv(OUT / "M4_real_fee_scenarios.csv")
    feature_map = pd.read_csv(OUT / "M4_real_feature_evidence.csv")
    entries = []

    def save(fig, name, title, note):
        fig.suptitle(title, y=0.99, fontweight="bold", fontsize=15)
        fig.text(0.015, 0.014, note, color="#555555", fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        fig.savefig(FIG / f"{name}.png", dpi=300)
        fig.savefig(FIG / f"{name}.svg")
        plt.close(fig)
        entries.append(dict(file=name, title=title, note=note))

    # 06 Daily vouchers (29 observed chargeable days).
    fig, ax = plt.subplots(figsize=(12, 5.6))
    x = np.arange(len(daily))
    colors = [GOLD if w else BLUE for w in daily.is_weekend]
    ax.bar(x, daily.paid_vouchers, 0.72, color=colors)
    mean = daily.paid_vouchers.mean()
    ax.axhline(mean, color=PINK, ls="--", lw=1.5, label=f"Mean {mean:,.0f}")
    ax.set_xticks(x)
    ax.set_xticklabels([d[5:] for d in daily.date], rotation=90, fontsize=7.5)
    ax.set_ylabel("Paid vouchers (fee-eligible visits)")
    ax.legend()
    save(fig, "M4_05_Real_daily_vouchers",
         "Observed 2024 chargeable days: paid vouchers (5 EUR)",
         "Blue = weekday, gold = weekend. Administrative counts include repeat visits and exclude exemptions; not total city visitors.")

    # 07 WTP survival curve (stated preference).
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.step(scenarios.fee_eur, scenarios.wtp_survival, where="mid", color=BLUE, lw=2)
    ax.scatter(scenarios.fee_eur, scenarios.wtp_survival, color=BLUE, s=35, zorder=3)
    for _, r in scenarios.iterrows():
        ax.annotate(f"{r.wtp_survival:.3f}", (r.fee_eur, r.wtp_survival),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("Entrance fee (EUR)")
    ax.set_ylabel("Share willing to pay (survival)")
    ax.set_ylim(0, 0.8)
    save(fig, "M4_06_WTP_survival",
         "Stated-preference willingness-to-pay survival curve",
         "Rubbi (2024) survey, n=153. Hypothetical stated preference, not observed demand elasticity.")

    # 08 Gross fee revenue vs fee level (real anchor at 5 EUR).
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(scenarios.fee_eur.astype(str), scenarios.gross_fee_revenue_eur / 1e6,
                  0.62, color=[GREEN if f == 5 else BLUE for f in scenarios.fee_eur])
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5)
    ax.set_xlabel("Entrance fee (EUR)")
    ax.set_ylabel("Gross fee revenue (million EUR)")
    save(fig, "M4_07_Fee_gross_revenue",
         "Conditional gross fee revenue (5 EUR is the observed anchor)",
         "Absolute level anchored to the real 485,062 vouchers; relative response from WTP. The 20 EUR peak is a conditional scenario, not an observed optimum.")

    # 09 Feature evidence mapping.
    fig, ax = plt.subplots(figsize=(10, 5.6))
    sub = feature_map.sort_values("expert_score")
    bars = ax.barh(sub.feature, sub.expert_score, color=GOLD)
    ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=10)
    ax.set_xlim(0, 5.6)
    ax.set_xlabel("Expert score (1-5)")
    for y, (_, r) in enumerate(sub.iterrows()):
        ax.text(0.08, y, r.real_evidence, va="center", fontsize=8, color="white")
    save(fig, "M4_08_Feature_evidence",
         "Feature scores grounded in real evidence (source IDs)",
         "Scores remain 1-5 expert codings; each is tied to a documented primary-source fact rather than fitted.")

    pd.DataFrame(entries).to_csv(OUT / "figure_index_real_en.csv", index=False, encoding="utf-8-sig")
    print("M4 real-data English figures written to figures/")


if __name__ == "__main__":
    main()
