# -*- coding: utf-8 -*-
r"""M1/M3 图表升级 v1.0 (O 奖风格) —— 依据 Prompts/04_图表升级.md

数据来源铁律 (禁止编造)
----------------------
* 图 1/2/3/5: 直接调用 m1_solver.solve_one_combo / m1_solver.evaluate 现算,
  不查表、不插值、不手填任何数值
* 图 4: 读 results/m3/sobol_indices.csv + results/m3/m3_elasticity.csv 原始输出
* 图 6: 流程图, 无数据

输出 (results/figures_upgrade/, 300 dpi)
----------------------------------------
1. pareto_tradeoff.png       S00 32 组合 3 联两两权衡 + Pareto 前沿 + ★ 加权和最优
2. lever_scan.png            固定最优组合, τ 与 ε 双杠杆扫描 (evaluate 现算)
3. feasibility_heatmap.png   (x_d × τ) 60×60 可行域离散热力图
4. sensitivity_radar.png     4 目标 Sobol 雷达图 (|S1| 与 |ST-S1|)
5. sensitivity_lines.png     5 参数 ±20% 连续扫描 (重求解) 折线组图
6. workflow_framework.png    M1→M3→M2→M5→M4 建模流程框图

已知退化 (如实呈现, 不改图)
--------------------------
v2.0 M1-4' 的 C_env_min=0.5 地板使 S00 全部 32 个组合的 C_env 几乎恒等 0.5,
因此图 1 的 (R, C_env) 与 (C_env, C_soc) 两个面板的 Pareto 前沿退化为单点,
图 4 的 C_env 面板 Sobol 指数方差退化。脚本会在图上直接标注该退化, 不强行修饰。
"""

from __future__ import annotations

import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.gridspec import GridSpec

BASE_DIR: Path = Path(__file__).resolve().parent
OUT_DIR: Path = BASE_DIR / "results" / "figures_upgrade"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import m1_solver as m1

# ==================== 全局视觉规范 (全套图强制一致) ====================
DPI: int = 300
C_MAIN = "#4C72B0"    # 主色
C_EMPH = "#C44E52"    # 强调
C_SEC = "#DD8452"     # 次色
C_ALT = "#55A868"     # 第四色
C_GREY = "#7F7F7F"
ALPHA_FILL = 0.12
GRID_ALPHA = 0.25


def setup_style() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
        "axes.labelsize": 10, "axes.titlesize": 10.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "grid.alpha": GRID_ALPHA, "grid.linestyle": "--", "grid.linewidth": 0.6,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def _silent(_msg: str) -> None:
    pass


def _finish(fig, n: int, title_en: str, fname: str, tight: bool = True,
            bottom: float = 0.05) -> Path:
    """统一图注 'Figure N: <英文标题>' + 保存。
    tight=False 时不调用 tight_layout（调用方已手工排版）。bottom 为图注预留的底部比例。"""
    if tight:
        fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.text(0.5, 0.008, f"Figure {n}: {title_en}", ha="center", va="bottom",
             fontsize=9, style="italic", color="#444444")
    path = OUT_DIR / fname
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _s00_row() -> pd.Series:
    scenarios = m1.load_scenarios()
    return scenarios[scenarios["scenario_id"] == "S00_baseline"].iloc[0]


def _s00_result() -> Dict[str, Any]:
    """results/m1/m1_results.csv 的 S00 行 (图 1 自检基准)。"""
    df = pd.read_csv(m1.RESULTS_DIR / "m1_results.csv", encoding="utf-8-sig")
    return df[df["scenario_id"] == "S00_baseline"].iloc[0].to_dict()


# =====================================================================================
# 图 1: Pareto 3 联权衡
# =====================================================================================
def _pareto_front(pts: List[Dict[str, Any]], kx: str, ky: str,
                  sx: int, sy: int) -> List[Dict[str, Any]]:
    """2 维非支配前沿。sx/sy=+1 表示该目标越大越好, -1 表示越小越好。"""
    ordered = sorted(pts, key=lambda r: (sx * r[kx], sy * r[ky]), reverse=True)
    front: List[Dict[str, Any]] = []
    best_y = -np.inf
    for r in ordered:
        yv = sy * r[ky]
        if yv > best_y + 1e-12:
            front.append(r)
            best_y = yv
    return front


def fig1_pareto_tradeoff(params: Dict[str, Any]) -> Path:
    row = _s00_row()
    ctx = m1.build_context(row, params)
    results: List[Dict[str, Any]] = []
    for combo in itertools.product([0, 1], repeat=5):
        results.append(m1.solve_one_combo(ctx, combo, params, _silent))
    feas = [r for r in results if r["feasible_flag"] == 1]
    best = max(feas, key=lambda r: (-r["relax_level"], r["objective_value"]))

    panels = [
        ("R", "C_env", +1, -1, "旅游总收入 $R$ (亿美元)", "环境成本 $C_{env}$", "(a) 收入–环境"),
        ("R", "C_soc", +1, -1, "旅游总收入 $R$ (亿美元)", "社会成本 $C_{soc}$", "(b) 收入–社会"),
        ("C_env", "C_soc", -1, -1, "环境成本 $C_{env}$", "社会成本 $C_{soc}$", "(c) 环境–社会"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 3.5))
    handles: Optional[Sequence[Any]] = None
    labels: Optional[Sequence[str]] = None
    for ax, (kx, ky, sx, sy, xlab, ylab, title) in zip(axes, panels):
        ax.scatter([r[kx] for r in feas], [r[ky] for r in feas], s=26, color=C_MAIN,
                   alpha=0.9, edgecolor="white", linewidth=0.5, zorder=3,
                   label=f"可行组合 (n={len(feas)}/32)")
        front = _pareto_front(feas, kx, ky, sx, sy)
        fx = [r[kx] for r in front]
        fy = [r[ky] for r in front]
        ax.plot(fx, fy, color=C_SEC, linewidth=1.6, marker="o", markersize=4.0,
                zorder=4, label="Pareto 前沿")
        if len(front) >= 2:
            y_hi = max(r[ky] for r in feas) * 1.03
            ax.fill_between(fx, fy, y_hi, color=C_SEC, alpha=ALPHA_FILL, zorder=1)
        ax.scatter([best[kx]], [best[ky]], marker="*", s=190, color=C_EMPH,
                   edgecolor="white", linewidth=0.6, zorder=5, label="加权和最优解")
        ax.annotate(f"({best[kx]:.3f}, {best[ky]:.3f})", (best[kx], best[ky]),
                    textcoords="offset points", xytext=(7, 7), fontsize=8, color=C_EMPH)
        if len(front) <= 3:
            if "C_env" in (kx, ky):
                spread = max(r["C_env"] for r in feas) - min(r["C_env"] for r in feas)
                note = (f"前沿退化：$C_{{env}}$ 幅值仅 {spread:.2f}\n（M1-4' $C_{{env,min}}$ 地板）"
                        if spread > 0.02 else
                        "前沿退化：$C_{env}$ 恒钉地板 0.5\n（M1-4' $C_{env,min}$）")
            else:
                note = f"前沿仅 {len(front)} 点（近退化）"
            # 面板 (c) 右上角有数据点, 注记改放左上角
            nx, nha = (0.03, "left") if kx == "C_env" else (0.97, "right")
            ax.text(nx, 0.97, note, transform=ax.transAxes, fontsize=6.2,
                    color=C_GREY, ha=nha, va="top", linespacing=1.3)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    fig.subplots_adjust(left=0.075, right=0.985, top=0.905, bottom=0.36, wspace=0.44)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.10),
               ncol=3, frameon=False, fontsize=8.2)

    # ---- 自检: ★ 坐标必须与 m1_results.csv 的 S00 行一致 ----
    ref = _s00_result()
    for key in ("R", "C_env", "C_soc"):
        d = abs(float(best[key]) - float(ref[key]))
        print(f"  [自检] ★ {key}: 现算 {best[key]:.6f} vs m1_results {float(ref[key]):.6f} "
              f"|Δ|={d:.2e} -> {'PASS' if d < 1e-6 else 'FAIL'}")
    print(f"  [自检] 选中组合 m={''.join(map(str, best['combo']))} "
          f"vs csv m={''.join(str(int(ref[f'm{i}'])) for i in range(1, 6))}")
    print(f"  [数据来源] m1_solver.solve_one_combo × 32 组合现算 (S00_baseline)")

    return _finish(fig, 1, "Pareto trade-off frontiers across the 32 measure portfolios "
                           "(S00 baseline, M1 v2.0)", "pareto_tradeoff.png", tight=False)


# =====================================================================================
# 图 2: 杠杆扫描 (τ 与 ε)
# =====================================================================================
def fig2_lever_scan(params: Dict[str, Any]) -> Path:
    row = _s00_row()
    ctx = m1.build_context(row, params)
    res: List[Dict[str, Any]] = [m1.solve_one_combo(ctx, c, params, _silent)
                                 for c in itertools.product([0, 1], repeat=5)]
    feas = [r for r in res if r["feasible_flag"] == 1]
    best = max(feas, key=lambda r: (-r["relax_level"], r["objective_value"]))
    combo = best["combo"]
    v0 = np.array([best["tau"], best["delta_I"], best["delta_E"], best["delta_C"], best["xd"]])
    ref = {"R": best["R"], "C_env": best["C_env"], "C_soc": best["C_soc"],
           "J": best["objective_value"]}

    def _rel(y: np.ndarray, key: str) -> np.ndarray:
        return (y / ref[key] - 1.0) * 100.0

    # --- (a) 税率 τ 扫描 ---
    taus = np.linspace(params["tau_min"], params["tau_max"], 61)
    s_tau: Dict[str, List[float]] = {k: [] for k in ("R", "C_env", "C_soc", "J")}
    for tau in taus:
        vv = v0.copy()
        vv[0] = tau
        r = m1.evaluate(ctx, vv, combo, params)
        if r is None:
            for k in s_tau:
                s_tau[k].append(np.nan)
            continue
        s_tau["R"].append(r["R"]); s_tau["C_env"].append(r["C_env"])
        s_tau["C_soc"].append(r["C_soc"]); s_tau["J"].append(r["objective_value"])

    # --- (b) 需求弹性 ε 扫描 ---
    eps_grid = np.linspace(-0.8, -0.2, 61)
    s_eps: Dict[str, List[float]] = {k: [] for k in ("R", "C_env", "C_soc", "J")}
    for eps in eps_grid:
        pe = dict(params); pe["epsilon"] = float(eps)
        r = m1.evaluate(ctx, v0, combo, pe)
        if r is None:
            for k in s_eps:
                s_eps[k].append(np.nan)
            continue
        s_eps["R"].append(r["R"]); s_eps["C_env"].append(r["C_env"])
        s_eps["C_soc"].append(r["C_soc"]); s_eps["J"].append(r["objective_value"])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.5))
    specs = [(axes[0], taus, s_tau, "综合税率 $\\tau$", "(a) 税率杠杆"),
             (axes[1], eps_grid, s_eps, "需求价格弹性 $\\varepsilon$", "(b) 需求弹性杠杆")]
    style = {"R": (C_MAIN, "-", "$R$"), "C_env": (C_SEC, "--", "$C_{env}$"),
             "C_soc": (C_ALT, "-.", "$C_{soc}$")}
    handles: Optional[Sequence[Any]] = None
    labels: Optional[Sequence[str]] = None
    for ax, xs, sd, xlab, title in specs:
        hs: List[Any] = []
        for k, (col, ls, lab) in style.items():
            ys = _rel(np.array(sd[k], dtype=float), k)
            ln, = ax.plot(xs, ys, color=col, linestyle=ls, linewidth=1.5,
                          label=f"{lab}（左轴，相对变化 %）")
            hs.append(ln)
            ax.fill_between(xs, ys, np.nanmin(ys) if np.isfinite(ys).any() else 0.0,
                            color=col, alpha=ALPHA_FILL)
        ax.axhline(0.0, color=C_GREY, linewidth=0.7, linestyle=":")
        ax.set_ylabel("相对变化 (%)", fontsize=9.5)
        ax.set_xlabel(xlab)
        ax.grid(True)
        ax.tick_params(labelsize=8.5)

        # J 用右轴绝对值：J 在低税区间会趋零/为负，相对变化不可用
        ax2 = ax.twinx()
        j_arr = np.array(sd["J"], dtype=float)
        lnj, = ax2.plot(xs, j_arr, color=C_EMPH, linewidth=1.9,
                        label="$J$（右轴，绝对值）")
        hs.append(lnj)
        ax2.axhline(0.0, color=C_EMPH, linewidth=0.7, linestyle=":")
        ax2.margins(y=0.10)
        ax2.set_ylabel("$J$（绝对值）", color=C_EMPH, fontsize=9.5)
        ax2.tick_params(axis="y", colors=C_EMPH, labelsize=8)
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
        j_neg = bool(np.isfinite(j_arr).any() and np.nanmin(j_arr) < 0)
        ax.set_title(title + ("（$J<0$ 段：收入罚项激活）" if j_neg else ""))

        i_best = int(np.nanargmax(j_arr))
        ax.axvline(xs[i_best], color=C_EMPH, linewidth=0.9, linestyle="--", alpha=0.7)
        ax2.scatter([xs[i_best]], [j_arr[i_best]], marker="*", s=150, color=C_EMPH,
                    edgecolor="white", linewidth=0.6, zorder=6)
        label = f"最优 {xs[i_best]:.3f}\n$J$={j_arr[i_best]:.4f}"
        frac = (i_best / max(len(xs) - 1, 1))
        if 0.3 <= frac <= 0.7:
            # 峰值居中时曲线会压住紧邻标注, 改为面板右下角 + 引线
            ax2.annotate(label, (xs[i_best], j_arr[i_best]), xycoords="data",
                         xytext=(0.96, 0.06), textcoords="axes fraction",
                         fontsize=7.2, color=C_EMPH, ha="right", va="bottom",
                         arrowprops=dict(arrowstyle="-", linewidth=0.6,
                                         color=C_EMPH, alpha=0.75))
        else:
            ax2.annotate(label, (xs[i_best], j_arr[i_best]), textcoords="offset points",
                         xytext=(-8, -6), fontsize=7.2, color=C_EMPH, ha="right", va="top")
        cenv = np.array(sd["C_env"], dtype=float)
        if np.nanmax(cenv) - np.nanmin(cenv) < 1e-4:
            ax.text(0.03, 0.93, f"$C_{{env}}$ 恒钉地板 {params['C_env_min']:.2f}（变化 0%）",
                    transform=ax.transAxes, fontsize=7.2, color=C_GREY, va="top")
        if handles is None:
            handles, labels = hs, [h.get_label() for h in hs]

    fig.subplots_adjust(left=0.075, right=0.925, top=0.915, bottom=0.30, wspace=0.62)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.125),
               ncol=4, frameon=False, fontsize=7.8)

    print(f"  [数据来源] m1_solver.evaluate × {len(taus)} (τ) + {len(eps_grid)} (ε) 点现算, "
          f"固定组合 m={''.join(map(str, combo))}")
    return _finish(fig, 2, "Policy-lever scans of tax rate and demand elasticity "
                           "(S00, fixed optimum portfolio)", "lever_scan.png", tight=False)


# =====================================================================================
# 图 3: 可行性热力图 (x_d × τ)
# =====================================================================================
def fig3_feasibility_heatmap(params: Dict[str, Any]) -> Path:
    row = _s00_row()
    ctx = m1.build_context(row, params)
    res: List[Dict[str, Any]] = [m1.solve_one_combo(ctx, c, params, _silent)
                                 for c in itertools.product([0, 1], repeat=5)]
    feas = [r for r in res if r["feasible_flag"] == 1]
    best = max(feas, key=lambda r: (-r["relax_level"], r["objective_value"]))
    combo = best["combo"]
    dI, dE, dC = best["delta_I"], best["delta_E"], best["delta_C"]

    n = 60
    tau_g = np.linspace(params["tau_min"], params["tau_max"], n)
    xd_max = max(2.0 - params["Delta_cap"] * combo[0], 0.0)
    xd_g = np.linspace(0.0, xd_max, n)

    code = np.full((n, n), 4, dtype=int)   # 4 = 不动点未收敛
    for i, xd in enumerate(xd_g):
        for j, tau in enumerate(tau_g):
            v = np.array([tau, dI, dE, dC, xd])
            r = m1.evaluate(ctx, v, combo, params)
            if r is None:
                code[i, j] = 4
            elif r["C_env"] > params["E_max"] + 1e-9:          # M1-9 硬: 生态阈值
                code[i, j] = 1
            elif r["x"] > r["I_cap"] + 1e-9:                    # M1-9 硬: 容量
                code[i, j] = 2
            elif r["R"] < ctx["R_min"] - 1e-9:                  # M1-11 软: 收入下限
                code[i, j] = 3
            else:
                code[i, j] = 0

    labels = [
        "可行（硬约束全过且 $R \\geq R_{min}$）",
        "违反生态阈值 $C_{env}>E_{max}$（硬，M1-9）",
        "违反容量 $x>I_{cap}$（硬，M1-9）",
        "$R<R_{min}$（M1-11 软罚项，非硬约束）",
        "不动点未收敛（不可评估）",
    ]
    colors = [C_ALT, C_EMPH, C_SEC, C_MAIN, "#BBBBBB"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(3.7, 4.0))
    ax.imshow(code, origin="lower", cmap=cmap, norm=norm, aspect="auto",
              extent=[tau_g[0], tau_g[-1], xd_g[0], xd_g[-1]], interpolation="nearest")
    ax.set_xlim(tau_g[0], tau_g[-1])
    ax.set_ylim(xd_g[0], xd_g[-1])
    ax.scatter([best["tau"]], [best["xd"]], marker="*", s=190, color="white",
               edgecolor="black", linewidth=0.8, zorder=6, clip_on=False)
    ax.set_xlabel("综合税率 $\\tau$")
    ax.set_ylabel("日峰值上限 $x_d$ (万人/日)")
    ax.set_title("S00 可行域（$\\delta$ 固定于最优）")
    ax.grid(False)
    present = [c for c in range(5) if (code == c).any()]
    handles = [Patch(facecolor=colors[c], edgecolor="#666666", linewidth=0.4, label=labels[c])
               for c in present]
    handles.append(plt.Line2D([], [], marker="*", color="white", markeredgecolor="black",
                              linestyle="none", markersize=13, label="S00 最优解"))
    fig.subplots_adjust(left=0.21, right=0.97, top=0.90, bottom=0.40)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.07),
               ncol=1, fontsize=6.8, frameon=False, handlelength=1.2, labelspacing=0.7)

    cnt = {c: int((code == c).sum()) for c in range(5)}
    print(f"  [数据来源] m1_solver.evaluate × {n}×{n}={n * n} 网格点现算 (δ 固定于 S00 最优)")
    print(f"  [网格统计] 可行 {cnt[0]} | 违反生态 {cnt[1]} | 违反容量 {cnt[2]} | "
          f"收入软约束未达 {cnt[3]} | 未收敛 {cnt[4]}")

    return _finish(fig, 3, "Feasibility map over tax rate and daily peak cap "
                           "(S00, investment fixed at optimum)", "feasibility_heatmap.png",
                   tight=False)


# =====================================================================================
# 图 4: Sobol 雷达图 (读 M3 结果)
# =====================================================================================
def fig4_sensitivity_radar() -> Path:
    sobol_path = BASE_DIR / "results" / "m3" / "sobol_indices.csv"
    elast_path = BASE_DIR / "results" / "m3" / "m3_elasticity.csv"
    sob = pd.read_csv(sobol_path, encoding="utf-8-sig")
    names = ["epsilon", "eta_I", "eta_E", "eta_C", "E_max", "S_min", "tau_max"]
    disp = [r"$\varepsilon$", r"$\eta_I$", r"$\eta_E$", r"$\eta_C$",
            r"$E_{max}$", r"$S_{min}$", r"$\tau_{max}$"]
    targets = ["R", "C_env", "C_soc", "J"]
    titles = {"R": "旅游总收入 $R$", "C_env": "环境成本 $C_{env}$",
              "C_soc": "社会成本 $C_{soc}$", "J": "综合效益 $J$"}

    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.4), subplot_kw={"polar": True},
                             gridspec_kw={"hspace": 0.46, "wspace": 0.34})
    clipped = []
    handles: Optional[Sequence[Any]] = None
    labels: Optional[Sequence[str]] = None
    for ax, t in zip(axes.ravel(), targets):
        sub = sob[sob["target"] == t].set_index("param").loc[names]
        v1 = np.clip(np.abs(sub["S1"].to_numpy(dtype=float)), 0.0, 1.0)
        vi = np.clip(np.abs(sub["interaction_ST_minus_S1"].to_numpy(dtype=float)), 0.0, 1.0)
        if (np.abs(sub["S1"]) > 1.0).any() or (np.abs(sub["interaction_ST_minus_S1"]) > 1.0).any():
            clipped.append(t)
        hs: List[Any] = []
        for vals, col, ls, lab in ((v1, C_MAIN, "-", "$|S_1|$ 一阶指数"),
                                   (vi, C_EMPH, "--", "$|S_T-S_1|$ 交互项")):
            vv = np.concatenate([vals, vals[:1]]).tolist()
            ln, = ax.plot(angles, vv, color=col, linestyle=ls, linewidth=1.4, label=lab)
            hs.append(ln)
            ax.fill(angles, vv, color=col, alpha=ALPHA_FILL)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(disp, fontsize=9.5)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=6.5, color=C_GREY)
        ax.grid(alpha=GRID_ALPHA)
        note = "\n（数值受地板约束主导，指数已截断至 1.0）" if t in clipped else ""
        ax.set_title(titles[t] + note, fontsize=9.8, pad=16)
        if handles is None:
            handles, labels = hs, [h.get_label() for h in hs]

    fig.subplots_adjust(left=0.13, right=0.88, top=0.90, bottom=0.16)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.045),
               ncol=2, frameon=False, fontsize=8)

    print(f"  [数据来源] {sobol_path.relative_to(BASE_DIR)} (Sobol S1/ST) "
          f"+ {elast_path.relative_to(BASE_DIR)} (单因素弹性, 仅核对)")
    if clipped:
        print(f"  [提示] {clipped} 存在 >1 的指数 (C_env 因地板致方差退化), 图中已截断并标注")

    return _finish(fig, 4, "Global sensitivity radar: Sobol first-order and interaction "
                           "indices (M3 v2.0, S00)", "sensitivity_radar.png", tight=False)


# =====================================================================================
# 图 5: ±20% 连续扫描 (重求解, 并行)
# =====================================================================================
def _solve_point(task: Tuple[str, float, Dict[str, Any], Tuple[int, ...], Dict[str, Any]]
                 ) -> Dict[str, Any]:
    """子进程入口: 基准参数随任务传入 (全局变量不会跨进程传递)。"""
    name, value, row_dict, combo, base_params = task
    row = pd.Series(row_dict)
    params = dict(base_params)
    params[name] = float(value)
    ctx = m1.build_context(row, params)          # eta_I 等会改变 ctx 的标定量, 必须重建
    r = m1.solve_one_combo(ctx, combo, params, _silent)
    if r.get("feasible_flag") == 1:
        return {"param": name, "value": float(value), "R": r["R"],
                "J": r["objective_value"], "C_soc": r["C_soc"],
                "C_env": r["C_env"], "feasible": 1, "relax": r["relax_name"]}
    return {"param": name, "value": float(value), "R": np.nan, "J": np.nan,
            "C_soc": np.nan, "C_env": np.nan, "feasible": 0, "relax": r["relax_name"]}


def fig5_sensitivity_lines(params: Dict[str, Any]) -> Path:
    row = _s00_row()
    ctx = m1.build_context(row, params)
    res: List[Dict[str, Any]] = [m1.solve_one_combo(ctx, c, params, _silent)
                                 for c in itertools.product([0, 1], repeat=5)]
    feas = [r for r in res if r["feasible_flag"] == 1]
    best = max(feas, key=lambda r: (-r["relax_level"], r["objective_value"]))
    combo = best["combo"]
    ref = {"R": best["R"], "J": best["objective_value"], "C_soc": best["C_soc"]}

    scan_params = ["epsilon", "eta_I", "tau_max", "S_min", "E_max"]
    n_pts = 101
    row_dict = row.to_dict()
    base_params = dict(params)
    tasks: List[Tuple[str, float, Dict[str, Any], Tuple[int, ...], Dict[str, Any]]] = []
    grid_map: Dict[str, np.ndarray] = {}
    for name in scan_params:
        base_v = float(params[name])
        g = np.linspace(base_v * 0.8, base_v * 1.2, n_pts)
        grid_map[name] = g
        for v in g:
            tasks.append((name, float(v), row_dict, combo, base_params))

    t0 = time.perf_counter()
    workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(_solve_point, tasks))
    dt = time.perf_counter() - t0
    df = pd.DataFrame(out)
    df.to_csv(OUT_DIR / "sensitivity_lines_data.csv", index=False, encoding="utf-8-sig")
    print(f"  [扫描] {len(tasks)} 次重求解 (5 参数 × {n_pts} 点), 并行 {workers} 进程, "
          f"耗时 {dt:.1f}s; 不可行点 {int((df['feasible'] == 0).sum())} 个")

    disp = {"epsilon": r"$\varepsilon$", "eta_I": r"$\eta_I$", "tau_max": r"$\tau_{max}$",
            "S_min": r"$S_{min}$", "E_max": r"$E_{max}$"}
    full = {"epsilon": r"$\varepsilon$（需求弹性）", "eta_I": r"$\eta_I$（基建效率）",
            "tau_max": r"$\tau_{max}$（税率上限）", "S_min": r"$S_{min}$（接受度下限）",
            "E_max": r"$E_{max}$（生态阈值）"}

    fig = plt.figure(figsize=(6.8, 5.8))
    gs = GridSpec(2, 3, figure=fig)
    style = {"R": (C_MAIN, "-", "$R$"), "C_soc": (C_ALT, "-.", "$C_{soc}$")}
    handles: Optional[Sequence[Any]] = None
    labels: Optional[Sequence[str]] = None
    for k, name in enumerate(scan_params):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        d = df[df["param"] == name].sort_values("value")
        xs = d["value"].to_numpy(dtype=float)
        base_v = float(params[name])
        hs: List[Any] = []
        flat = True
        for key, (col, ls, lab) in style.items():
            ys = (d[key].to_numpy(dtype=float) / ref[key] - 1.0) * 100.0
            if np.isfinite(ys).any() and np.nanmax(np.abs(ys)) > 1e-6:
                flat = False
            ln, = ax.plot(xs, ys, color=col, linestyle=ls, linewidth=1.4,
                          label=f"{lab}（左轴，% 变化）")
            hs.append(ln)
            if np.isfinite(ys).any():
                ax.fill_between(xs, ys, np.nanmin(ys), color=col, alpha=ALPHA_FILL)
        ax.axhline(0.0, color=C_GREY, linewidth=0.7, linestyle=":")
        ax.axvline(base_v, color=C_GREY, linewidth=0.8, linestyle="--", alpha=0.8)
        if flat:
            ax.set_ylim(-1.0, 1.0)
            ax.text(0.5, 0.55, "全部目标变化 = 0\n（非紧约束，无影子价格）",
                    transform=ax.transAxes, fontsize=7, color=C_GREY, ha="center")

        # J 走右轴绝对值（J 可趋零/为负，相对变化不可读）
        ax2 = ax.twinx()
        j = d["J"].to_numpy(dtype=float)
        lnj, = ax2.plot(xs, j, color=C_EMPH, linewidth=1.7, label="$J$（右轴，绝对值）")
        hs.append(lnj)
        ax2.axhline(0.0, color=C_EMPH, linewidth=0.6, linestyle=":")
        j_span = float(np.nanmax(j) - np.nanmin(j)) if np.isfinite(j).any() else 0.0
        if j_span < 1e-9:
            ax2.set_ylim(float(np.nanmin(j)) * 0.95, float(np.nanmax(j)) * 1.05)
        else:
            ax2.margins(y=0.22)          # 顶部留白, 供 J 峰标注
        ax2.tick_params(axis="y", colors=C_EMPH, labelsize=7.0, pad=2)
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
        if np.isfinite(j).any() and j_span >= 1e-9:
            # 仅在 J 真实变化时标注峰值 (E_max 面板 J 恒定, 标注无意义)
            i = int(np.nanargmax(j))
            frac = i / max(len(xs) - 1, 1)
            ha = "left" if frac < 0.15 else ("right" if frac > 0.85 else "center")
            dx = 7 if ha == "left" else (-7 if ha == "right" else 0)
            ax2.scatter([xs[i]], [j[i]], marker="o", s=24, color=C_EMPH, zorder=5)
            ax2.annotate(f"$J$ 峰 {xs[i]:.3f}", (xs[i], j[i]), textcoords="offset points",
                         xytext=(dx, 6), fontsize=6.6, color=C_EMPH, ha=ha, va="bottom")
        ax.set_title(full[name], fontsize=9.5)
        ax.set_xlabel(f"参数值（基准 {base_v:.3f}）", fontsize=8.5)
        ax.grid(True)
        ax.tick_params(labelsize=8)
        if k % 3 == 0:                      # 仅最左列标左轴
            ax.set_ylabel("相对 S00 最优解变化 (%)", fontsize=8.5)
        if k % 3 == 2 or k == 4:            # 仅每行最右侧标 J 轴, 避免与邻面板标签互压
            ax2.set_ylabel("$J$（绝对值）", color=C_EMPH, fontsize=8.5)
        if handles is None:
            handles, labels = hs, [h.get_label() for h in hs]

    ax_note = fig.add_subplot(gs[1, 2])
    ax_note.axis("off")
    n_bad = int((df["feasible"] == 0).sum())
    txt = ("扫描设置\n"
           f"· 参数：{len(scan_params)} 个 × {n_pts} 点 = {len(tasks)} 次重求解\n"
           "· 区间：基准值 ±20%\n"
           f"· 不可行点：{n_bad}\n"
           "· 组合固定为 S00 最优 m="
           f"{''.join(map(str, combo))}\n"
           "· 左轴：相对 S00 v2.0 最优解 % 变化\n"
           "· 右轴：综合效益 $J$ 绝对值\n"
           f"· 基准：$R$={ref['R']:.4f}, $J$={ref['J']:.5f},\n"
           f"  $C_{{soc}}$={ref['C_soc']:.4f}\n"
           "· 极端参数处 $R_{min}$/$S_{min}$ 罚项激活,\n"
           "  解跳变, 属真实求解结果")
    ax_note.text(0.0, 0.96, txt, va="top", ha="left", fontsize=7.2, color="#333333",
                 linespacing=1.65)

    fig.subplots_adjust(left=0.075, right=0.905, top=0.915, bottom=0.175,
                        hspace=0.62, wspace=0.68)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.065),
               ncol=3, frameon=False, fontsize=8)
    print(f"  [数据来源] m1_solver.solve_one_combo × {len(tasks)} 次重求解 "
          f"(参数按基准值 ±20% 扫描), 明细 {OUT_DIR.name}/sensitivity_lines_data.csv")

    return _finish(fig, 5, "Continuous sensitivity of R, J and C_soc to ±20% parameter "
                           "perturbations (S00, re-optimised)", "sensitivity_lines.png",
                   tight=False)


# =====================================================================================
# 图 6: 建模流程框图
# =====================================================================================
def fig6_workflow_framework() -> Path:
    stages = [
        ("M1", "基准模型", "MINLP 32 组合\n+ 不动点迭代", C_MAIN),
        ("M3", "敏感性分析", "单因素弹性 + Sobol\n全局敏感性", C_SEC),
        ("M2", "闭环反馈", "鲁棒优化\n+ 蒙特卡洛", C_ALT),
        ("M5", "空间均衡", "MILP · 区域\n流量分配", C_MAIN),
        ("M4", "跨目的地迁移", "AHP-TOPSIS\n可迁移性评价", C_SEC),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0.95, 4.35)          # 收紧下方留白 (反馈弧最低约 y=1.7)
    ax.axis("off")

    w, h, y = 1.62, 1.34, 2.30
    xs = [0.35 + i * 1.95 for i in range(len(stages))]
    for x, (tag, name, desc, col) in zip(xs, stages):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=1.1, edgecolor=col, facecolor=col,
                             alpha=0.16, zorder=2)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.24, tag, ha="center", va="center",
                fontsize=12, color=col, zorder=3)
        ax.text(x + w / 2, y + h - 0.60, name, ha="center", va="center",
                fontsize=9.2, color="#222222", zorder=3)
        ax.text(x + w / 2, y + 0.30, desc, ha="center", va="center",
                fontsize=6.8, color="#444444", zorder=3, linespacing=1.45)

    for i in range(len(stages) - 1):
        x0 = xs[i] + w
        x1 = xs[i + 1]
        ax.annotate("", xy=(x1 - 0.03, y + h / 2), xytext=(x0 + 0.03, y + h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", linewidth=1.1,
                                    shrinkA=0, shrinkB=0), zorder=1)

    # 闭环反馈 M2 -> M1
    ax.annotate("", xy=(xs[0] + w / 2, y - 0.04), xytext=(xs[2] + w / 2, y - 0.04),
                arrowprops=dict(arrowstyle="-|>", color=C_EMPH, linewidth=1.1,
                                connectionstyle="arc3,rad=-0.26", linestyle="--"), zorder=1)
    ax.text((xs[0] + xs[2]) / 2 + w / 2, y - 1.05,
            "闭环反馈：鲁棒策略回灌基准模型", ha="center", va="center",
            fontsize=7.8, color=C_EMPH)

    # 阶段标签与分隔
    ax.text(0.35, 3.92, "阶段 I：现状诊断与政策杠杆", fontsize=8.4, color=C_GREY)
    ax.text(4.30, 3.92, "阶段 II：不确定性与空间/迁移拓展", fontsize=8.4, color=C_GREY)
    ax.plot([4.18, 4.18], [3.80, 1.75], color="#CCCCCC", linewidth=0.7, zorder=0)

    return _finish(fig, 6, "Modelling workflow of the Juneau sustainable-tourism project "
                           "(M1 \u2192 M3 \u2192 M2 \u2192 M5 \u2192 M4)",
                   "workflow_framework.png", tight=False)


# =====================================================================================
def main() -> None:
    setup_style()
    params = dict(m1.DEFAULT_PARAMS)
    t_all = time.perf_counter()
    print("=" * 78)
    print("图表升级 (O 奖风格) | 输出目录:", OUT_DIR)
    print("=" * 78)

    produced: List[Tuple[str, str, Path]] = []

    print("[图 1] pareto_tradeoff ...")
    t0 = time.perf_counter()
    p = fig1_pareto_tradeoff(params)
    produced.append(("pareto_tradeoff", "m1_solver.solve_one_combo (S00 × 32)", p))
    print(f"   -> {p.name}  {time.perf_counter() - t0:.1f}s")

    print("[图 2] lever_scan ...")
    t0 = time.perf_counter()
    p = fig2_lever_scan(params)
    produced.append(("lever_scan", "m1_solver.evaluate (τ/ε 扫描)", p))
    print(f"   -> {p.name}  {time.perf_counter() - t0:.1f}s")

    print("[图 3] feasibility_heatmap ...")
    t0 = time.perf_counter()
    p = fig3_feasibility_heatmap(params)
    produced.append(("feasibility_heatmap", "m1_solver.evaluate (60×60 网格)", p))
    print(f"   -> {p.name}  {time.perf_counter() - t0:.1f}s")

    print("[图 4] sensitivity_radar ...")
    t0 = time.perf_counter()
    p = fig4_sensitivity_radar()
    produced.append(("sensitivity_radar", "results/m3/sobol_indices.csv", p))
    print(f"   -> {p.name}  {time.perf_counter() - t0:.1f}s")

    print("[图 5] sensitivity_lines ...")
    t0 = time.perf_counter()
    p = fig5_sensitivity_lines(params)
    produced.append(("sensitivity_lines", "m1_solver.solve_one_combo (5×101 重求解)", p))
    print(f"   -> {p.name}  {time.perf_counter() - t0:.1f}s")

    print("[图 6] workflow_framework ...")
    t0 = time.perf_counter()
    p = fig6_workflow_framework()
    produced.append(("workflow_framework", "无数据 (流程图)", p))
    print(f"   -> {p.name}  {time.perf_counter() - t0:.1f}s")

    print("-" * 78)
    print("产出清单 (数据来源 -> 路径):")
    for name, src, path in produced:
        print(f"  {name:<22} <- {src}")
        print(f"  {'':<22} -> {path}")
    print(f"总耗时 {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
