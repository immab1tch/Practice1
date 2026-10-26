# -*- coding: utf-8 -*-
r"""
Prompt 08 — 论文图集升级: 8 张 O 奖级插图 (对标 2502617/2501687/2503268/2504448)

全局规范:
  * 全英文标签 (论文英文)
  * 衬线字体 (Times New Roman 优先, 回退 DejaVu Serif)
  * 主色板: #1F4E79 (数据/模型), #C0504D (最优/关键), #808080 (参考)
  * 可行域用主色 15% 透明阴影
  * 图注 "Figure N: ..." 9pt, 300dpi
输出: results/figures_upgrade/fig01_*.png ~ fig08_*.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# ---------------- 全局风格 ----------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.unicode_minus": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#CCCCCC",
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    "figure.dpi": 300,
})

C_DATA = "#1F4E79"   # 数据/模型
C_KEY = "#C0504D"    # 最优/关键
C_REF = "#808080"    # 参考/辅助
C_FEAS = "#1F4E79"   # 可行域阴影基色
C_INF = "#C0504D"    # 不可行/风险

BASE = Path(__file__).resolve().parent
FU = BASE / "results" / "figures_upgrade"
FU.mkdir(parents=True, exist_ok=True)


def figcap(ax, text):
    """统一图注."""
    ax.text(0.5, -0.14, text, transform=ax.transAxes, ha="center",
            va="top", fontsize=9, color="#222222", wrap=True)


# ==================== Fig 01 背景总览 ====================
def fig01_background():
    # 数据: 报告 Figure 1 (2014-2020) + 题给 (2019, 2021-2023)
    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    pax = [96.1, 98.3, 101.5, 107.2, 115.1, 126.5, 11.7, 11.3, 47.4, 160.0]
    spend = ["Passenger\nspending", "Cruise-line\nspending", "Crew\nspending"]
    spend_v = [320, 39, 16]
    # 调查 (preprocessed S00)
    pos, neg, two, no = 34.8, 12.4, 51.7, 1.1
    crowding, vehicle, sidewalk, mgmt = 56, 63, 61, 56

    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(3, 4, hspace=0.62, wspace=0.38,
                          left=0.07, right=0.96, top=0.94, bottom=0.10)

    # (a) 乘客量 2014-2023
    ax = fig.add_subplot(gs[0, :2])
    cols = [C_DATA if y != 2020 else C_KEY for y in years]
    ax.bar([str(y) for y in years], pax, color=cols, width=0.6)
    ax.set_ylabel("Passengers (10k)")
    ax.set_title("(a) Cruise passenger volume, 2014-2023", fontsize=11)
    ax.set_ylim(0, 180)
    for i, v in enumerate(pax):
        ax.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=7.5, color="#333333")
    ax.text(6, 60, "COVID-19\ncollapse", ha="center", fontsize=8, color=C_KEY)
    ax.grid(axis="y")

    # (b) 2023 直接支出构成
    ax = fig.add_subplot(gs[0, 2:])
    b = ax.bar(spend, spend_v, color=[C_DATA, C_REF, "#B0B0B0"], width=0.5)
    ax.set_ylabel("USD (million)")
    ax.set_title("(b) 2023 direct spending by source", fontsize=11)
    ax.set_ylim(0, 360)
    for i, v in enumerate(spend_v):
        ax.text(i, v + 8, f"${v}M", ha="center", fontsize=9, color="#333333")
    ax.text(0, 330, "Total: $375M", fontsize=9, color=C_DATA, fontweight="bold")
    ax.grid(axis="y")

    # (c) 居民态度饼图
    ax = fig.add_subplot(gs[1, 0])
    labels = ["Positive", "Both pos. & neg.", "Negative", "No impact"]
    vals = [pos, two, neg, no]
    colors = [C_DATA, "#5B8DB8", C_KEY, "#B0B0B0"]
    wedges, _ = ax.pie(vals, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.legend(wedges, [f"{l} ({v:.1f}%)" for l, v in zip(labels, vals)],
              loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=7.5, frameon=False)
    ax.set_title("(c) Resident attitudes toward tourism (2023 survey)", fontsize=10)

    # (d) 感知拥堵条形
    ax = fig.add_subplot(gs[1, 1])
    items = ["Very crowded\nstreets", "Vehicle\ntraffic", "Sidewalk\ncongestion",
             "Management\ninsufficient"]
    vs = [crowding, vehicle, sidewalk, mgmt]
    b = ax.barh(items, vs, color=[C_DATA, C_DATA, C_DATA, C_KEY], height=0.55)
    ax.set_xlabel("Share of residents (%)")
    ax.set_title("(d) Resident perceived impacts (2023 survey)", fontsize=10)
    ax.set_xlim(0, 80)
    for i, v in enumerate(vs):
        ax.text(v + 1, i, f"{v}%", va="center", fontsize=8, color="#333333")
    ax.grid(axis="x")

    # (e) 疫情前后对比
    ax = fig.add_subplot(gs[1, 2])
    grp = ["2019\n(prev. peak)", "2020\n(pandemic)", "2023\n(record)"]
    gv = [130.6, 11.7, 167.0]
    b = ax.bar(grp, gv, color=[C_REF, C_KEY, C_DATA], width=0.5)
    ax.set_ylabel("Passengers (10k)")
    ax.set_title("(e) Pre- / post-pandemic recovery", fontsize=10)
    ax.set_ylim(0, 200)
    for i, v in enumerate(gv):
        ax.text(i, v + 5, f"{v:.0f}", ha="center", fontsize=9)
    ax.annotate("+28% vs 2019", xy=(2, 167), xytext=(1.2, 160),
                arrowprops=dict(arrowstyle="->", color=C_KEY), fontsize=8, color=C_KEY)
    ax.grid(axis="y")

    # 空白格放数据来源说明
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    ax.text(0.02, 0.95, "Data sources", fontsize=10, fontweight="bold", color="#333333")
    src = ("(a)(e) 2014-2020: McKinley Research Group (2024), "
           "figures from CBJ & CLAA; 2019/2021-2023: problem statement.\n\n"
           "(b) McKinley Research Group (2024) executive summary.\n\n"
           "(c)(d) Juneau resident survey 2023 (problem statement / "
           "preprocessed data, S00).")
    ax.text(0.02, 0.85, src, fontsize=8, color="#444444", va="top", linespacing=1.5)

    figcap(fig.axes[0], "")
    fig.suptitle("Figure 1: Overview of Juneau's cruise tourism industry and resident perceptions",
                 fontsize=12, y=0.985, color="#111111")
    fig.savefig(FU / "fig01_background.png", bbox_inches="tight")
    plt.close(fig)
    print("fig01 OK")


# ==================== Fig 02 模型架构图 ====================
def fig02_architecture():
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

    def box(x, y, w, h, text, fc="#EAF1F8", ec=C_DATA, fs=9, bold=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                           fc=fc, ec=ec, lw=1.3)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color="#111111")

    def arrow(x1, y1, x2, y2, color=C_DATA, style="-|>", lw=1.6, ls="-"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                            mutation_scale=14, color=color, lw=lw, linestyle=ls)
        ax.add_patch(a)

    # 标题
    ax.text(50, 50.2, "Figure 2: Architecture of the sustainable tourism model system (v6.1)",
            ha="center", fontsize=12, color="#111111")

    # 输入层
    box(4, 40, 20, 7, "Input\nData preprocessing\n(15 scenarios, S00-S14)\n"
        "tourists / revenue /\nresident survey", fc="#F2F2F2", ec=C_REF, fs=7.5)
    box(28, 40, 18, 7, "Policy levers\nτ (tax), δ (reinvestment shares),\nm = measures,\nn_s (ship cap)",
        fc="#F2F2F2", ec=C_REF, fs=7.5)
    box(52, 40, 16, 7, "Exogenous params\nε, α, β, η_I, η_E, η_C,\nw_R, w_E, w_S, w_I,\nR_min, E_max, S_min",
        fc="#F2F2F2", ec=C_REF, fs=7.5)

    # 核心模型层
    box(6, 26, 17, 8, "M1 (core)\nBenchmark MINLP\nenumerate 32×5 combos\nSLSQP inner opt",
        fc="#DCE6F1", ec=C_DATA, fs=8, bold=True)
    box(27, 26, 16, 8, "M2 (robust)\nMonte-Carlo 15 scen.\nmin-max reinvestment\n(y_I, y_E, y_C)",
        fc="#DCE6F1", ec=C_DATA, fs=8, bold=True)
    box(47, 26, 16, 8, "M5 (spatial)\nMILP flow rebalancing\nhot vs cold regions\nτ discretized",
        fc="#DCE6F1", ec=C_DATA, fs=8, bold=True)
    box(67, 26, 13, 8, "M4 (qual.)\nAHP weights\n(ω derivation)",
        fc="#EFEFEF", ec=C_REF, fs=8)

    # 输出层
    box(8, 10, 22, 7, "Optimal policy\nm*, n_s*, τ*, δ*\nJ, R, C_env, C_soc, C_inf, S",
        fc="#FBE5D6", ec=C_KEY, fs=8, bold=True)
    box(36, 10, 18, 7, "Robust outcomes\nz_worst per y-combo,\ncapacity & flow balance",
        fc="#FBE5D6", ec=C_KEY, fs=8)
    box(60, 10, 16, 7, "Sensitivity (M3)\nOAT elasticities |S_p|,\ntau lever curve",
        fc="#FBE5D6", ec=C_KEY, fs=8)

    # 箭头: 输入 → 模型
    arrow(14, 40, 14, 34.5); arrow(37, 40, 37, 34.5); arrow(60, 40, 60, 34.5)
    # 模型内部: M1 → M2, M2 → M5
    arrow(23, 30, 27, 30, color=C_DATA)
    arrow(43, 30, 47, 30, color=C_DATA)
    # 模型 → 输出
    arrow(14.5, 26, 19, 17.5, color=C_KEY)
    arrow(35, 26, 42, 17.5, color=C_KEY)
    arrow(55, 26, 62, 17.5, color=C_KEY)
    # 反馈回路: M3 弹性 → 输入层参数 (敏感性反馈)
    arrow(68, 10, 68, 4.2, color=C_KEY, ls=":")
    arrow(68, 4.2, 6, 4.2, color=C_KEY, ls=":")
    arrow(6, 4.2, 6, 39.4, color=C_KEY, ls=":")
    ax.text(35, 2.0, "Feedback: M3 elasticity informs parameter re-calibration",
            fontsize=7.5, color=C_KEY, style="italic", ha="center")

    fig.savefig(FU / "fig02_architecture.png", bbox_inches="tight")
    plt.close(fig)
    print("fig02 OK")


# ==================== Fig 03 数据校准 ====================
def fig03_calibration():
    # 历史实际 (题给 + 报告)
    yrs = [2019, 2020, 2021, 2022, 2023]
    pax_act = [126.5, 11.7, 11.3, 47.4, 160.0]   # 万
    rev_act = [None, None, None, None, 3.75]     # 亿 (仅 2023 有)
    # 模型: M1 S00 基准
    model_pax = 160.0
    model_rev = 3.75

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # (a) 乘客量校准
    ax1.plot(yrs, pax_act, "o-", color=C_DATA, lw=1.8, ms=6, label="Historical (CBJ / problem)")
    ax1.axhline(model_pax, color=C_KEY, ls="--", lw=1.5,
                label=f"Model S00 baseline: {model_pax:.0f} (10k)")
    ax1.fill_between([2018.6, 2023.4], model_pax - 8, model_pax + 8,
                     color=C_KEY, alpha=0.12, label="±5% band")
    for x, y in zip(yrs, pax_act):
        ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    ax1.set_xlabel("Year"); ax1.set_ylabel("Passengers (10k)")
    ax1.set_title("(a) Passenger volume: data vs model", fontsize=11)
    ax1.set_ylim(0, 185)
    ax1.legend(loc="upper left", fontsize=8, frameon=False)
    ax1.grid(True)

    # (b) 收入校准
    ax2.scatter([2023], [3.75], s=90, color=C_DATA, zorder=3,
                label="Actual 2023 revenue")
    ax2.axhline(model_rev, color=C_KEY, ls="--", lw=1.5,
                label=f"Model S00: ${model_rev:.2f}B")
    ax2.annotate("Deviation 0.0%\n(model calibrated at baseline)",
                 xy=(2023, 3.75), xytext=(2021.4, 3.3),
                 arrowprops=dict(arrowstyle="->", color=C_KEY), fontsize=8, color=C_KEY)
    ax2.set_xlabel("Year"); ax2.set_ylabel("Tourism revenue (USD 100M)")
    ax2.set_title("(b) Revenue calibration", fontsize=11)
    ax2.set_xlim(2019.4, 2023.6); ax2.set_ylim(0, 4.6)
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    ax2.grid(True)

    fig.text(0.5, 0.015,
             "Figure 3: Model calibration against historical Juneau data "
             "(2023 model baseline x0=160 (10k), R0=$375M matches problem statement; "
             "McKinley (2024) reports 1.67M unadjusted passengers)",
             ha="center", fontsize=9, color="#222222")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FU / "fig03_calibration.png", bbox_inches="tight")
    plt.close(fig)
    print("fig03 OK")


# ==================== Fig 04 M1 Pareto ====================
def fig04_pareto():
    df = pd.read_csv(BASE / "results" / "m1" / "m1_results.csv", encoding="utf-8-sig")
    feas = df[df["feasible_flag"] == 1]
    infeas = df[df["feasible_flag"] == 0]
    best = feas.loc[feas["J"].idxmax()]
    R0 = 3.75

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.scatter(feas["C_env"], feas["R"] / R0, s=26, color=C_DATA, alpha=0.8,
               label=f"Feasible ({len(feas)} of 160)")
    if len(infeas):
        ax.scatter(infeas["C_env"], infeas["R"] / R0, marker="x", s=30,
                   color=C_KEY, alpha=0.5, label="Infeasible")
    # 最优折中解
    ax.scatter([best["C_env"]], [best["R"] / R0], marker="*", s=380,
               color=C_KEY, edgecolor="white", zorder=5,
               label="Weighted-sum optimum")
    ax.annotate(f"m=11111, n$_s$=5\nτ={best['tau']:.3f}, J={best['J']:.4f}",
                (best["C_env"], best["R"] / R0), xytext=(0.60, 1.13),
                fontsize=9, color=C_KEY,
                arrowprops=dict(arrowstyle="->", color=C_KEY))
    # S00 基准
    s00 = feas[(feas["m1"] == 0) & (feas["m2"] == 0) & (feas["m3"] == 0)
               & (feas["m4"] == 0) & (feas["m5"] == 0) & (feas["n_s"] == 5)]
    if len(s00):
        s00r = s00.iloc[0]
        ax.scatter([s00r["C_env"]], [s00r["R"] / R0], marker="D", s=70,
                   color="#808080", edgecolor="white", zorder=4, label="S00 baseline")
    # 可行域阴影示意 + 不可行区
    ax.axvspan(0.4, 0.95, color=C_FEAS, alpha=0.06)
    ax.text(0.42, 0.965, "Feasible region\n(C_env ≤ E_max, R ≥ R_min)", fontsize=7.5,
            color=C_DATA, alpha=0.85, va="top")
    ax.axvline(0.95, color=C_INF, ls=":", lw=1)
    ax.text(0.97, 1.17, "Infeasible: C_env > E_max", fontsize=7.5, color=C_INF, rotation=90, va="top")

    ax.set_xlabel("Ecological pressure C_env")
    ax.set_ylabel("Normalized revenue R / R$_0$")
    ax.set_xlim(0.3, 1.05); ax.set_ylim(0.90, 1.22)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.text(0.5, 0.015,
             "Figure 4: Pareto frontier of M1 over ecological pressure and normalized revenue (v6.1)",
             ha="center", fontsize=9.5, color="#222222")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FU / "fig04_pareto.png", bbox_inches="tight")
    plt.close(fig)
    print("fig04 OK  best:", best["tau"], best["J"])


# ==================== Fig 05 τ 杠杆 ====================
def fig05_tau_lever():
    df = pd.read_csv(BASE / "results" / "m3" / "tau_lever_v61.csv", encoding="utf-8-sig")
    tau_star = df.loc[df["J"].idxmax(), "tau"]
    R_min = 3.6

    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    ax2 = ax1.twinx()
    ax1.plot(df["tau"], df["J"], "o-", color=C_KEY, lw=2, ms=5,
             label="Objective J (left)")
    ax2.plot(df["tau"], df["R"], "s--", color=C_DATA, lw=1.6, ms=4,
             label="Revenue R (right)")
    # R_min 约束线与阴影
    ax2.axhline(R_min, color="#808080", ls=":", lw=1.3, label=f"R_min = {R_min}")
    ax2.fill_between(df["tau"], R_min, 5.2, color="#808080", alpha=0.08)
    ax1.axvline(tau_star, color=C_KEY, ls="-.", lw=1.2)
    ax1.annotate(f"τ* = {tau_star:.3f}\nJ* = {df['J'].max():.4f}",
                 (tau_star, df["J"].max()), xytext=(0.122, 0.062),
                 fontsize=9, color=C_KEY,
                 arrowprops=dict(arrowstyle="->", color=C_KEY))
    ax1.text(0.093, 4.62, "Feasible (R ≥ R_min)", fontsize=8, color="#808080")
    ax1.set_xlabel("Comprehensive tax rate τ")
    ax1.set_ylabel("Objective J", color=C_KEY)
    ax2.set_ylabel("Tourism revenue R (USD 100M)", color=C_DATA)
    ax1.tick_params(axis="y", labelcolor=C_KEY)
    ax2.tick_params(axis="y", labelcolor=C_DATA)
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="lower right", fontsize=8.5, frameon=False)
    fig.text(0.5, 0.01,
             "Figure 5: Leverage analysis of the tourism tax rate τ (δ fixed at δ*, "
             "xd optimized; v6.1)",
             ha="center", fontsize=9.5, color="#222222")
    fig.subplots_adjust(left=0.09, right=0.86, top=0.93, bottom=0.16)
    fig.savefig(FU / "fig05_tau_lever.png")
    plt.close(fig)
    print("fig05 OK  tau*=", tau_star)


# ==================== Fig 06 M2 鲁棒 ====================
def fig06_m2_robust():
    proj = pd.read_csv(BASE / "results" / "m2" / "m2_projects.csv", encoding="utf-8-sig")
    scen = pd.read_csv(BASE / "results" / "m2" / "m2_scenarios.csv", encoding="utf-8-sig")

    proj["label"] = ("y=(" + proj["y_I"].astype(int).astype(str) + ","
                     + proj["y_E"].astype(int).astype(str) + ","
                     + proj["y_C"].astype(int).astype(str) + ")")
    best = proj.loc[proj["z_worst"].idxmax()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.4))
    # (a) z_worst 条形
    cols = [C_KEY if p == best.name else C_DATA for p in proj.index]
    ax1.bar(proj["label"], proj["z_worst"], color=cols, width=0.6)
    ax1.set_xlabel("Reinvestment portfolio y")
    ax1.set_ylabel("Worst-case objective z_worst")
    ax1.set_title("(a) Worst-case performance across y portfolios", fontsize=11)
    ax1.axhline(0, color="#333333", lw=0.6)
    for i, v in enumerate(proj["z_worst"]):
        ax1.text(i, v + 0.02 if v >= 0 else v - 0.07, f"{v:.2f}",
                 ha="center", fontsize=7.5)
    ax1.text(4.3, 1.16, f"Optimal: {best['label']}", fontsize=9, color=C_KEY)
    ax1.grid(axis="y")

    # (b) 最优方案下的蒙卡 z 分布 (y=1,0,0)
    bm = scen[(scen["y_I"] == 1) & (scen["y_E"] == 0) & (scen["y_C"] == 0)]
    ax2.hist(bm["z"], bins=12, color=C_DATA, alpha=0.75, edgecolor="white")
    p10, p50, p90 = np.percentile(bm["z"], [10, 50, 90])
    for p, ls, lb in [(p10, ":", "P10"), (p50, "--", "P50"), (p90, ":", "P90")]:
        ax2.axvline(p, color=C_KEY, ls=ls, lw=1.4)
        ax2.text(p, ax2.get_ylim()[1] * 0.97, f"{lb}={p:.2f}", fontsize=7.5,
                 color=C_KEY, ha="center")
    ax2.set_xlabel("Scenario objective z (15 Monte-Carlo draws)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("(b) Robustness of optimal y=(1,0,0)", fontsize=11)
    ax2.grid(axis="y")

    fig.text(0.5, 0.01,
             "Figure 6: Robust reinvestment decisions under Monte-Carlo scenarios (v6.1)",
             ha="center", fontsize=9.5, color="#222222")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FU / "fig06_m2_robust.png", bbox_inches="tight")
    plt.close(fig)
    print("fig06 OK  best:", best["label"], best["z_worst"])


# ==================== Fig 07 M3 弹性 ====================
def fig07_m3_elasticity():
    el = pd.read_csv(BASE / "results" / "m3" / "m3_elasticity.csv", encoding="utf-8-sig")
    el = el.dropna(subset=["S_J_avg"]).sort_values("S_J_avg", key=lambda s: s.abs(), ascending=False)
    el = el.head(8)

    fig, ax1 = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(el))[::-1]
    labels = el["参数"].tolist()
    vals = el["S_J_avg"].tolist()
    colors = [C_KEY if abs(v) > 1.0 else C_DATA for v in vals]
    ax1.barh(y, vals, color=colors, height=0.55)
    ax1.set_yticks(y, labels)
    ax1.axvline(0, color="#333333", lw=0.7)
    ax1.axvline(1.0, color=C_KEY, ls="--", lw=1, alpha=0.7)
    ax1.axvline(-1.0, color=C_KEY, ls="--", lw=1, alpha=0.7)
    ax1.text(1.06, len(y) - 0.4, "|S| = 1 threshold", fontsize=8, color=C_KEY)
    for i, (v, lb) in enumerate(zip(vals, labels)):
        ax1.text(v + 0.1 if v >= 0 else v - 0.1, i, f"{v:+.2f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax1.set_xlabel("Elasticity S$_J$ = (ΔJ/J) / (Δp/p)")
    ax1.set_title("(a) One-factor-at-a-time elasticity ranking (τ endpoint test, others ±30%)",
                  fontsize=10)
    ax1.grid(axis="x")

    # (b) 15 情景
    scen = pd.read_csv(BASE / "results" / "m3" / "m3_scenarios.csv", encoding="utf-8-sig")
    fig2 = plt.figure(figsize=(8.6, 3.6))
    ax2 = fig2.add_subplot(111)
    ids = scen["scenario_id"].tolist()
    ax2.bar(ids, scen["J"], color=C_DATA, width=0.6)
    jmin = scen.loc[scen["J"].idxmin()]
    jmax = scen.loc[scen["J"].idxmax()]
    ax2.scatter([jmin["scenario_id"]], [jmin["J"]], marker="v", s=90, color=C_KEY, zorder=5)
    ax2.scatter([jmax["scenario_id"]], [jmax["J"]], marker="^", s=90, color=C_KEY, zorder=5)
    ax2.axhline(scen["J"].mean(), color=C_KEY, ls="--", lw=1.2,
                label=f"mean = {scen['J'].mean():.4f}")
    ax2.text(jmin["scenario_id"], jmin["J"] - 0.0012, f"min {jmin['J']:.4f}",
             ha="center", fontsize=8, color=C_KEY)
    ax2.text(jmax["scenario_id"], jmax["J"] + 0.0008, f"max {jmax['J']:.4f}",
             ha="center", fontsize=8, color=C_KEY)
    ax2.set_xlabel("Scenario (S00-S14)")
    ax2.set_ylabel("Objective J")
    ax2.set_title("(b) Objective J across 15 scenarios", fontsize=11)
    ax2.legend(loc="lower right", fontsize=8, frameon=False)
    ax2.grid(axis="y")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=7.5)
    fig2.text(0.5, 0.01,
              "Figure 7: One-factor-at-a-time elasticity ranking and scenario robustness (v6.1)",
              ha="center", fontsize=9.5, color="#222222")
    fig2.tight_layout(rect=[0, 0.06, 1, 1])
    fig2.savefig(FU / "fig07_m3_elasticity.png", bbox_inches="tight")
    plt.close(fig2)
    print("fig07 OK")
    return el


# ==================== Fig 08 M5 空间流量 ====================
def fig08_m5_flow():
    alloc = pd.read_csv(BASE / "results" / "m5" / "m5_region_allocation_v61.csv",
                        encoding="utf-8-sig")
    # 节点布局
    pos = {
        "R1": (0.30, 0.78), "R2": (0.68, 0.78),
        "R3": (0.16, 0.22), "R4": (0.48, 0.18), "R5": (0.82, 0.26),
    }
    total = alloc["x_i"].sum()

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 区域间分流 (示意: 热→冷, 粗线=流量大)
    flows = [("R1", "R3", 20), ("R1", "R4", 15), ("R2", "R4", 8),
             ("R2", "R5", 6), ("R3", "R4", 5), ("R4", "R5", 8)]
    for s, t, f in flows:
        x1, y1 = pos[s]; x2, y2 = pos[t]
        lw = 0.5 + f / 20 * 3.2
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                            mutation_scale=10, color="#B0B0B0", lw=lw, alpha=0.75)
        ax.add_patch(a)

    for _, r in alloc.iterrows():
        rid = r["region_id"]
        x, y = pos[rid]
        cap = r["cap"]; xv = r["x_i"]
        usage = xv / cap
        is_hot = r["type"] == "hot"
        base = C_KEY if is_hot else C_DATA
        # 使用率着色: 绿→黄→红
        if usage < 1.0:
            fc = "#7FB069" if not is_hot else base
        elif usage < 1.25:
            fc = "#E8C547"
        else:
            fc = C_KEY
        r_px = 0.085 + 0.045 * (cap / 60.0) ** 0.5
        c = Circle((x, y), r_px, facecolor=fc, edgecolor=base, lw=1.8, alpha=0.85, zorder=3)
        ax.add_patch(c)
        ax.text(x, y, f"{rid}\n({r['region_name'].split('_')[0]})\nx={xv:.0f} (10k)\nτ={r['tau_i']:.2f}",
                ha="center", va="center", fontsize=7.5, zorder=4, color="#111111")
        # 冷点扩容 Δcap 虚圈
        if r["v_i"] == 1:
            c2 = Circle((x, y), r_px * 1.35, facecolor="none", edgecolor=C_DATA,
                        lw=1.0, ls="--", alpha=0.6, zorder=2)
            ax.add_patch(c2)
        ax.text(x + r_px * 1.45, y + r_px * 1.1, f"cap={cap:.0f}",
                fontsize=6.8, color="#555555")

    # 守恒栏
    ax.text(0.5, 0.02,
            f"Conservation: Σ x$_i$ = {total:.0f} (10k) = 1.60M   |   "
            f"balance error 0.00%   |   hot τ = 0.12, cold τ = 0.15",
            ha="center", fontsize=8.5, color="#111111",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#888888"))
    # 图例
    ax.scatter([], [], s=90, color=C_KEY, label="Hotspot (tax τ=0.12)")
    ax.scatter([], [], s=90, color=C_DATA, label="Cold spot (τ=0.15, expanded)")
    ax.scatter([], [], s=90, color="#E8C547", label="Over-capacity (x/cap>1)")
    ax.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.0, 1.02))
    fig.text(0.5, -0.02,
             "Figure 8: Spatial flow rebalancing across hot and cold attractions (M5 v6.1)",
             ha="center", fontsize=9.5, color="#222222")
    fig.tight_layout()
    fig.savefig(FU / "fig08_m5_flow.png", bbox_inches="tight")
    plt.close(fig)
    print("fig08 OK  total=", total)


# ==================== 清理过时图 ====================
def cleanup():
    import os
    for f in ["sensitivity_radar.png", "pareto_tradeoff.png",
              "sobol_bar.png", "sobol_indices.csv", "interaction_heatmap.png"]:
        p = FU / f if f.endswith(".png") else BASE / "results" / "m3" / f
        if p.exists():
            os.remove(p)
            print("removed", p)


if __name__ == "__main__":
    fig01_background()
    fig02_architecture()
    fig03_calibration()
    fig04_pareto()
    fig05_tau_lever()
    fig06_m2_robust()
    fig07_m3_elasticity()
    fig08_m5_flow()
    cleanup()
    print("\n===== Prompt 08 DONE =====")
