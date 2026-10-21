# -*- coding: utf-8 -*-
r"""M2 税收支出与反哺模型 —— 鲁棒项目选择 + 蒙特卡洛情景

依据
----
`Docs/模型建立v2.0.tex` 小问 2 (M2-1 ~ M2-9):
  * M2-1  B_s = τ·p0·x_s                         情景税收
  * M2-2  Σ_k c_k y_k ≤ δ_total·B_min            总支出约束 (鲁棒: 用最坏情景税收)
  * M2-3  ΔI_s = Σ_{k∈I} η_k c_k y_k / B_0       各类项目反哺强度
  * M2-4  I_s = I_0(1+ΔI_s), C_env,s = max(C_env_min, C_env,0(1-ρ_E·ΔE_s)),
          S_s = clip(S_0+ΔS_s)
  * M2-5' x_s^new = x_0·(τ/τ_0)^ε·(I_s/I_0)^α·(S_s/S_0)^β
          ·(C_env,0/(C_env,s+0.1))^γ·Π_j(1+κ_j m_j)
  * M2-6  U_s = λ_IΔI_s+λ_EΔE_s+λ_CΔS_s − λ_L·max(0,x_0−x_s^new)
          − λ_B·max(0,C_env,s−E_max)
  * M2-7  max z, s.t. z ≤ U_s ∀s         鲁棒 max-min
  * M2-9  y_k∈{0,1}, y_a+y_b ≤ 1 (a,b)∈M_excl

实现口径 (与公式逐条对应, 偏离处在此显式声明)
---------------------------------------------
1. y_k 只有 8 个项目 (juno_projects_budget.csv), 直接枚举 2^8=256 个组合求全局最优,
   不依赖 MILP 求解器; z = min_s U_s 可直接比较, 枚举即精确解。
2. δ_I,δ_E,δ_C 在 M2-2 中只进"和 ≤ δ_total"约束, 不进目标 (M2-3 用 c_k y_k);
   因此取 M1 S00 最优解的 δ* 作为常数并复核约束, 不引入无依据的重新分配。
3. τ 取 M1 S00 最优税率 0.15 (政策变量), 情景不确定性只作用于 x_s 与 τ_s 的抽样。
4. τ_0, x_0, I_0, S_0, C_env,0 均为 M1 基准态标定值 (160 万人 / 0.11 / 160 / 1.0 / 1.0),
   故 x_s^new 与 M1-2' 同构, 不发生"从最优解再乘一次措施乘子"的重复计数。

编程手默认值 (CSV/PDF 均无数值, 待建模手确认)
----------------------------------------------
* cost_unit_100m = 0.05 亿/成本等级  (c_k = cost_level × 0.05)
* rho_E = 0.5        (生态项目对 C_env 的折减强度)
* λ_I=λ_E=λ_C=1.0, λ_L=0.05, λ_B=1.0
* M_excl = ∅        (juno_projects_budget.csv 无互斥关系列; 代码保留约束接口)
* σ_x=10%, σ_τ=5%, 情景取 B_s 的 P10/P50/P90 (蒙特卡洛 N=20000)

输出 (results/m2/)
------------------
* m2_projects.csv      项目清单 + 选中标志 + 分项反哺贡献
* m2_allocation.csv    支出分配 (按 I/E/C 分类)
* m2_scenarios.csv     低/中/高情景: B_s, Δ, I_s, C_env,s, S_s, x_new, U_s
* m2_mc_summary.csv    蒙特卡洛 P5/P50/P95 + 预算可行比例
* m2_closed_loop.png   闭环反馈图
* m2_scenario_ci.png   情景对比与置信区间图
* run_log.txt          自检 / 参数口径 / 待确认清单
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE_DIR: Path = Path(__file__).resolve().parent
OUT_DIR: Path = BASE_DIR / "results" / "m2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import m1_solver as m1

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
DPI: int = 300

C_MAIN, C_SEC, C_EMPH, C_ALT, C_GREY = "#4C72B0", "#DD8452", "#C44E52", "#55A868", "#7F7F7F"

M2_CONFIG: Dict[str, Any] = {
    "scenario_id": "S00_baseline",
    # ---- 蒙特卡洛情景 ----
    "mc_n": 20000,
    "mc_seed": 42,
    "sigma_x": 0.10,
    "sigma_tau": 0.05,
    "quantiles": {"low": 0.10, "mid": 0.50, "high": 0.90},
    # ---- 项目成本换算 (编程手假设) ----
    "cost_unit_100m": 0.05,
    # ---- 反哺/效用参数 (编程手假设) ----
    "rho_E": 0.5,
    "lambda_I": 1.0, "lambda_E": 1.0, "lambda_C": 1.0,
    "lambda_L": 0.05, "lambda_B": 1.0,
    # ---- 互斥项目对 (CSV 无数据, 默认空集) ----
    "excl_pairs": [],
}


def _silent(_msg: str) -> None:
    pass


# =====================================================================================
# 输入装载: M1 基准标定 + M1 S00 最优政策 + 项目表
# =====================================================================================
def load_inputs() -> Dict[str, Any]:
    params = dict(m1.DEFAULT_PARAMS)
    scenarios = m1.load_scenarios()
    row = scenarios[scenarios["scenario_id"] == M2_CONFIG["scenario_id"]].iloc[0]
    ctx = m1.build_context(row, params)

    m1_path = m1.RESULTS_DIR / "m1_results.csv"
    if not m1_path.exists():
        raise FileNotFoundError(f"未找到 {m1_path}, 请先运行 m1_solver.py")
    m1_df = pd.read_csv(m1_path, encoding="utf-8-sig").set_index("scenario_id")
    best = m1_df.loc[M2_CONFIG["scenario_id"]]

    projects = pd.read_csv(m1.PREPROCESSED_DIR / "juno_projects_budget.csv",
                           encoding="utf-8-sig")
    projects["cost_100m"] = (projects["cost_level"].astype(float)
                             * M2_CONFIG["cost_unit_100m"]).round(6)
    projects["eta"] = projects["eta"].astype(float)

    combo = tuple(int(best[f"m{i}"]) for i in range(1, 6))
    prod_measures = 1.0
    for j, mj in enumerate(combo, start=1):
        prod_measures *= (1.0 + params["kappa_measure"][f"m{j}"] * mj)

    return {
        "params": params,
        "ctx": ctx,                       # x0=160, tau0=0.11, p0, B0, I_ref=160, S_ref
        "projects": projects,
        "combo": combo,
        "prod_measures": prod_measures,
        "tau_policy": float(best["tau"]),
        "delta": (float(best["delta_I"]), float(best["delta_E"]), float(best["delta_C"])),
        "m1_R": float(best["R"]), "m1_x": float(best["x"]),
        "m1_Cenv": float(best["C_env"]),
        "B0": ctx["B0"],
    }


# =====================================================================================
# M2-1 蒙特卡洛情景: B_s = τ_s·p0·x_s, 取 B 的 P10/P50/P90 作为低/中/高
# =====================================================================================
def make_scenarios(ins: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    cfg, ctx = M2_CONFIG, ins["ctx"]
    rng = np.random.default_rng(cfg["mc_seed"])
    x_s = ctx["x0"] * (1.0 + cfg["sigma_x"] * rng.standard_normal(cfg["mc_n"]))
    tau_s = ins["tau_policy"] * (1.0 + cfg["sigma_tau"] * rng.standard_normal(cfg["mc_n"]))
    x_s = np.clip(x_s, 1e-6, None)
    tau_s = np.clip(tau_s, 1e-6, None)
    B_s = tau_s * ctx["p0"] * x_s                                  # M2-1

    rows = []
    picks = {}
    for name, q in cfg["quantiles"].items():
        target = float(np.quantile(B_s, q))
        i = int(np.argmin(np.abs(B_s - target)))                   # 取真实抽样点, 不插值
        picks[name] = i
        rows.append({"scenario": name, "quantile": q,
                     "x_s": float(x_s[i]), "tau_s": float(tau_s[i]), "B_s": float(B_s[i])})
    scen = pd.DataFrame(rows)
    mc = {"x_s": x_s, "tau_s": tau_s, "B_s": B_s}
    return scen, mc


# =====================================================================================
# M2-3 ~ M2-7: 给定项目组合 y, 计算三情景效用与鲁棒目标 z
# =====================================================================================
def evaluate_selection(y: np.ndarray, scen: pd.DataFrame,
                       ins: Dict[str, Any]) -> Dict[str, Any]:
    cfg, ctx, params = M2_CONFIG, ins["ctx"], ins["params"]
    proj = ins["projects"]
    cost = float(np.dot(proj["cost_100m"].to_numpy(), y))
    B_min = float(scen["B_s"].min())
    budget = ctx["delta_total"] * B_min                            # M2-2 右端

    d = {}
    for cat in ("I", "E", "C"):
        mask = (proj["category"].to_numpy() == cat)
        contribute = float(np.dot((proj["eta"] * proj["cost_100m"]).to_numpy() * y, mask))
        d[cat] = contribute / ins["B0"]                            # M2-3 (除以 B_0)

    I_s = ctx["I_ref"] * (1.0 + d["I"])                            # M2-4 (I_0 = I_ref = x_0)
    Cenv_s = max(params["C_env_min"],
                 params["C_env_0"] * (1.0 - cfg["rho_E"] * d["E"]))
    S_s = float(np.clip(params["S0"] + d["C"], 0.0, 1.0))

    out = {"cost_100m": cost, "budget_100m": budget,
           "feasible_budget": int(cost <= budget + 1e-12),
           "dI": d["I"], "dE": d["E"], "dS": d["C"],
           "I_s": I_s, "C_env_s": Cenv_s, "S_s": S_s}

    U = []
    xs = []
    for _, s in scen.iterrows():
        x_new = (ctx["x0"] * (s["tau_s"] / ctx["tau0"]) ** params["epsilon"]
                 * (I_s / ctx["I_ref"]) ** params["alpha"]
                 * (S_s / params["S0"]) ** params["beta"]
                 * ((params["C_env_0"] / (Cenv_s + 0.1)) ** params["gamma"])
                 * ins["prod_measures"])                           # M2-5'
        u = (cfg["lambda_I"] * d["I"] + cfg["lambda_E"] * d["E"]
             + cfg["lambda_C"] * d["C"]
             - cfg["lambda_L"] * max(0.0, ctx["x0"] - x_new)       # M2-6
             - cfg["lambda_B"] * max(0.0, Cenv_s - params["E_max"]))
        U.append(float(u))
        xs.append(float(x_new))
    out["U"] = U
    out["x_new"] = xs
    out["z"] = float(min(U))                                       # M2-7
    return out


def enumerate_selections(scen: pd.DataFrame, ins: Dict[str, Any]) -> pd.DataFrame:
    proj = ins["projects"]
    n = len(proj)
    combos = list(itertools.product((0, 1), repeat=n))
    excl = M2_CONFIG["excl_pairs"]
    id2idx = {pid: i for i, pid in enumerate(proj["project_id"])}

    rows = []
    for y in combos:
        if any(y[id2idx[a]] == 1 and y[id2idx[b]] == 1 for a, b in excl):   # M2-9
            continue
        ev = evaluate_selection(np.asarray(y), scen, ins)
        if not ev["feasible_budget"]:
            continue
        rows.append({"y": "".join(str(v) for v in y),
                     **{k: v for k, v in ev.items() if k != "U"},
                     "U_low": ev["U"][0], "U_mid": ev["U"][1], "U_high": ev["U"][2],
                     "x_low": ev["x_new"][0], "x_mid": ev["x_new"][1], "x_high": ev["x_new"][2]})
    return pd.DataFrame(rows)


# =====================================================================================
# 绘图
# =====================================================================================
def setup_style() -> None:
    plt.rcParams.update({"axes.edgecolor": "#888888", "axes.linewidth": 0.8,
                         "xtick.color": "#444444", "ytick.color": "#444444"})


def fig_closed_loop(ins: Dict[str, Any], best: Dict[str, Any]) -> Path:
    """M2 闭环反馈图: 税收→预算→项目→反哺→状态→需求→效用→鲁棒目标."""
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    ax.set_xlim(0, 9.9)
    ax.set_ylim(0.75, 4.62)
    ax.axis("off")

    boxes = [
        (0.95, "M2-1", "情景税收 $B_s$\n$=\\tau p_0 x_s$", C_MAIN),
        (2.95, "M2-2", "鲁棒预算\n$\\delta_{tot}\\,B_{min}$", C_MAIN),
        (4.95, "M2-9", "项目选择 $y_k$\n(0-1, 互斥)", C_SEC),
        (6.95, "M2-3", "反哺强度\n$\\Delta I,\\Delta E,\\Delta S$", C_SEC),
        (8.95, "M2-4", "状态更新\n$I_s,C_{env,s},S_s$", C_ALT),
    ]
    w, h, y0 = 1.62, 1.05, 3.05
    for x, tag, name, col in boxes:
        ax.add_patch(FancyBboxPatch((x - w / 2, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                    facecolor="white", edgecolor=col, linewidth=1.2))
        ax.text(x, y0 + h - 0.24, tag, fontsize=6.6, color=col, ha="center", va="center")
        ax.text(x, y0 + h - 0.57, name, fontsize=8.2, color="#333333", ha="center", va="center",
                linespacing=1.25)
        ax.text(x, y0 + 0.26, "", fontsize=6.0, ha="center")

    # 第二行: 需求反馈 → 情景效用 → 鲁棒目标
    y1 = 1.35
    for x, tag, name, col in [
        (2.95, "M2-5'", "需求反馈 $x_s^{new}$\n含环境吸引力", C_MAIN),
        (5.95, "M2-6", "情景效用 $U_s$\n(减惩罚项)", C_SEC),
        (8.95, "M2-7", "鲁棒目标 $z$\n$\\max z,\\ z\\leq U_s$", C_EMPH),
    ]:
        ax.add_patch(FancyBboxPatch((x - w / 2, y1), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                    facecolor="white", edgecolor=col, linewidth=1.2))
        ax.text(x, y1 + h - 0.24, tag, fontsize=6.6, color=col, ha="center", va="center")
        ax.text(x, y1 + h - 0.57, name, fontsize=8.2, color="#333333", ha="center", va="center",
                linespacing=1.25)

    arrow = dict(arrowstyle="-|>", lw=1.1, color="#666666", shrinkA=2, shrinkB=2,
                 mutation_scale=10)
    for c in (0.95, 2.95, 4.95, 6.95):
        ax.annotate("", xy=(c + 1.19, y0 + h / 2), xytext=(c + 0.83, y0 + h / 2), arrowprops=arrow)
    # 右端折回下行: M2-4 -> M2-7
    ax.annotate("", xy=(8.95, y1 + h + 0.04), xytext=(8.95, y0 - 0.04), arrowprops=arrow)
    # 下行左移: M2-7 -> M2-6 -> M2-5'
    ax.annotate("", xy=(5.95 + w / 2 + 0.04, y1 + h / 2), xytext=(8.95 - w / 2 - 0.04, y1 + h / 2),
                arrowprops=arrow)
    ax.annotate("", xy=(2.95 + w / 2 + 0.04, y1 + h / 2), xytext=(5.95 - w / 2 - 0.04, y1 + h / 2),
                arrowprops=arrow)

    # 反馈回路: x_s^new -> B_s (蒙特卡洛情景闭环, 走第 1 列左侧空白区)
    ax.annotate("", xy=(0.95, y0 - 0.03), xytext=(2.95 - w / 2, y1 + h / 2),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color=C_EMPH, alpha=0.85,
                                shrinkA=2, shrinkB=2, mutation_scale=10,
                                connectionstyle="arc3,rad=0.28"))
    # 标签放在弧线左侧空白区, 避免与弧线交叉 (弧最左点约 x=1.15)
    ax.text(0.66, 2.58, "闭环反馈", fontsize=6.2, color=C_EMPH,
            ha="center", va="center")

    # 输入注解
    ax.text(4.95, 4.46, "M1 最优政策输入: $\\tau^*=%.3f$, $\\delta^*=(%.2f,%.2f,%.2f)$, $m=%s$"
            % (ins["tau_policy"], *ins["delta"], "".join(map(str, ins["combo"]))),
            fontsize=7.4, color="#333333", ha="center", va="center")

    ax.text(4.95, 0.97, "最优: $z=%.4f$, 支出 %.3f/%.3f 亿, $x_{mid}=%.2f$ 万人"
            % (best["z"], best["cost_100m"], best["budget_100m"], best["x_mid"]),
            fontsize=7.4, color=C_EMPH, ha="center", va="center")

    path = OUT_DIR / "m2_closed_loop.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_scenario_ci(scen: pd.DataFrame, mc: Dict[str, np.ndarray],
                    best: Dict[str, Any], mc_stats: Dict[str, Dict[str, float]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.30, wspace=0.28)

    # (a) 效用 U 的蒙特卡洛分布 + 情景点
    ax = axes[0]
    u = mc_stats["U_draws"]
    p5, p50, p95 = np.percentile(u, [5, 50, 95])
    ax.hist(u, bins=60, color=C_MAIN, alpha=0.55, edgecolor="white", linewidth=0.3)
    ax.axvspan(p5, p95, color=C_SEC, alpha=0.12, zorder=0,
               label="90%% 置信区间 [%.3f, %.3f]" % (p5, p95))
    share_max = float(np.mean(u >= u.max() - 1e-9))
    ax.text(0.03, 0.62, "%.0f%% 样本无惩罚\n($U_s$ 达上界 %.4f)" % (share_max * 100, u.max()),
            transform=ax.transAxes, fontsize=6.4, color="#444444", ha="left", va="top",
            linespacing=1.3)
    u_lo = float(np.percentile(u, 1.0))          # 极端左尾 (2 个样本) 截出坐标范围
    ax.set_xlim(u_lo - 0.02, u.max() + 0.06)     # 右侧留白, 避免尖峰贴轴
    out_share = float(np.mean(u < u_lo))
    ax.text(0.03, 0.06, "%.1f%% 样本 $U<%.3f$ (坐标外)" % (out_share * 100, u_lo),
            transform=ax.transAxes, fontsize=6.0, color=C_GREY, ha="left", va="bottom")
    # 情景线: low/mid 的 U 相同, 改用一处文字块说明, 避免标签在右端相互重叠
    groups: Dict[float, List[str]] = {}
    for _, s in scen.iterrows():
        uu = float(best["U"][list(scen["scenario"]).index(s["scenario"])])
        groups.setdefault(round(uu, 6), []).append(str(s["scenario"]))
        ax.axvline(uu, color=C_EMPH, lw=1.0, ls="--", alpha=0.85)
    lines = ["虚线 = 情景点 $U_s$:"]
    for val, names in groups.items():
        lines.append("%s %.3f" % ("/".join(names), val))
    ax.text(0.035, 0.46, "\n".join(lines), transform=ax.transAxes, fontsize=6.2,
            color=C_EMPH, ha="left", va="top", linespacing=1.35)
    ax.set_xlabel("鲁棒解的效用 $U_s$", fontsize=9)
    ax.set_ylabel("抽样频数", fontsize=9)
    ax.set_title("(a) $U_s$ 蒙特卡洛分布 ($N$=%d)" % len(u), fontsize=9.5)
    ax.legend(fontsize=6.4, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7.5)

    # (b) 税收 B 的分布 + 鲁棒预算线
    ax = axes[1]
    b = mc["B_s"]
    ax.hist(b, bins=44, color=C_ALT, alpha=0.55, edgecolor="white", linewidth=0.3)
    budget = best["budget_100m"]
    ax.axvline(budget, color=C_EMPH, lw=1.4, ls="--",
               label="鲁棒预算 $\\delta_{tot}B_{min}$=%.3f" % budget)
    ax.axvline(float(scen["B_s"].min()), color=C_MAIN, lw=1.0, ls=":",
               label="$B_{min}$=%.3f (P10)" % float(scen["B_s"].min()))
    ax.set_xlabel("税收 $B_s$ (亿美元)", fontsize=9)
    ax.set_ylabel("抽样频数", fontsize=9)
    ax.set_title("(b) 情景税收分布与预算约束", fontsize=9.5)
    ax.legend(fontsize=6.2, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7.5)

    fig.text(0.5, 0.075, "Figure 7: M2 robust project selection under low / mid / high scenarios",
             ha="center", fontsize=8.5, color="#333333")

    path = OUT_DIR / "m2_scenario_ci.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# =====================================================================================
# 主流程
# =====================================================================================
def main() -> None:
    t0 = time.perf_counter()
    setup_style()
    log_lines: List[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    ins = load_inputs()
    ctx = ins["ctx"]
    params = ins["params"]
    scen, mc = make_scenarios(ins)

    log("=" * 86)
    log("M2 税收支出与反哺模型 (鲁棒项目选择 + 蒙特卡洛情景)")
    log("=" * 86)
    log(f"[基准标定] x0={ctx['x0']:.1f} 万, tau0={ctx['tau0']:.3f}, p0={ctx['p0']:.6f}, "
        f"B0={ins['B0']:.4f} 亿, I_ref={ctx['I_ref']:.2f}, S0={params['S0']:.2f}, "
        f"C_env0={params['C_env_0']:.2f}")
    log(f"[M1 输入] tau*={ins['tau_policy']:.4f}, delta*=({ins['delta'][0]:.4f},"
        f"{ins['delta'][1]:.4f},{ins['delta'][2]:.4f}), m={''.join(map(str, ins['combo']))}, "
        f"sum(delta)={sum(ins['delta']):.4f} <= delta_total={ctx['delta_total']:.2f} -> "
        f"{sum(ins['delta']) <= ctx['delta_total'] + 1e-9}")
    log(f"[项目表] {len(ins['projects'])} 个项目, 成本合计 "
        f"{ins['projects']['cost_100m'].sum():.3f} 亿 (cost_unit={M2_CONFIG['cost_unit_100m']} 亿/级)")
    log("[情景] 蒙特卡洛 N=%d, sigma_x=%.0f%%, sigma_tau=%.0f%%, 取 B_s 的 P10/P50/P90:"
        % (M2_CONFIG["mc_n"], 100 * M2_CONFIG["sigma_x"], 100 * M2_CONFIG["sigma_tau"]))
    for _, s in scen.iterrows():
        log(f"  {s['scenario']:<5s} x_s={s['x_s']:.2f} 万, tau_s={s['tau_s']:.4f}, B_s={s['B_s']:.4f} 亿")

    # ---- 自检 1: M2 锚点与 M1 基准标定一致 ----
    cenv_base = m1.compute_cenv(ctx["x0"], ctx["x0"], 0.0, ctx["xd0"], ins["B0"], ins["B0"],
                                (0, 0, 0, 0, 0), params)
    log(f"[自检 1] 锚点: I_ref={ctx['I_ref']:.4f} (应=x0={ctx['x0']:.1f}), "
        f"S_ref={ctx['S_ref']:.4f} (应=S0={params['S0']:.2f}), "
        f"C_env(基准态)={cenv_base:.4f} (应=C_env0={params['C_env_0']:.2f}) -> "
        f"{'PASS' if abs(ctx['I_ref'] - ctx['x0']) < 1e-6 and abs(cenv_base - params['C_env_0']) < 1e-9 else 'CHECK'}")
    x0_formula = (ctx["x0"]
                  * ((params["C_env_0"] / (params["C_env_0"] + 0.1)) ** params["gamma"]))
    dev = abs(x0_formula - ctx["x0"]) / ctx["x0"] * 100
    log(f"[自检 1b] 不含措施乘子时 M2-5' 复现: x={x0_formula:.2f} 万 vs x0={ctx['x0']:.1f} 万, "
        f"偏差 {dev:.2f}% (与 M1 v2.0 基准偏移 0.2%-1.7% 同源) -> "
        f"{'PASS(<2%)' if dev < 2.0 else 'CHECK'}")
    log(f"[输入] 措施乘子 Π(1+κ_j m_j)={ins['prod_measures']:.4f} (m={''.join(map(str, ins['combo']))})")

    # ---- 自检 2: 情景 B_min 与预算 ----
    log(f"[自检 2] B_min={scen['B_s'].min():.4f} 亿, 鲁棒预算=δ_tot·B_min="
        f"{ctx['delta_total'] * scen['B_s'].min():.4f} 亿; 全部项目成本="
        f"{ins['projects']['cost_100m'].sum():.3f} 亿 -> 预算"
        f"{'紧' if ins['projects']['cost_100m'].sum() > ctx['delta_total'] * scen['B_s'].min() else '松'}")

    # ---- 枚举 ----
    table = enumerate_selections(scen, ins)
    if table.empty:
        raise RuntimeError("无预算可行组合, 请检查 cost_unit_100m / 预算")
    table = table.sort_values(["z", "cost_100m"], ascending=[False, True]).reset_index(drop=True)
    best = table.iloc[0].to_dict()
    best["U"] = [best["U_low"], best["U_mid"], best["U_high"]]
    best["x_new"] = [best["x_low"], best["x_mid"], best["x_high"]]

    log(f"[枚举] 预算可行组合 {len(table)} / 256, 最优 y*={best['y']}, "
        f"z={best['z']:.6f}, J_equiv(支出)={best['cost_100m']:.4f} 亿")
    log(f"[最优解] dI={best['dI']:.4f}, dE={best['dE']:.4f}, dS={best['dS']:.4f}, "
        f"I_s={best['I_s']:.2f}, C_env_s={best['C_env_s']:.4f}, S_s={best['S_s']:.4f}")
    for i, s in scen.iterrows():
        log(f"  {s['scenario']:<5s} U_s={best['U'][i]:.6f}, x_new={best['x_new'][i]:.2f} 万")

    # ---- 选中的项目清单 ----
    proj = ins["projects"].copy()
    yv = np.array([int(c) for c in best["y"]])
    proj["selected"] = yv
    proj["spend_100m"] = (proj["cost_100m"] * proj["selected"]).round(6)
    proj["contribution"] = (proj["eta"] * proj["cost_100m"] * proj["selected"]
                            / ins["B0"]).round(6)
    proj[["project_id", "category", "cost_level", "cost_100m", "eta", "selected",
          "spend_100m", "contribution"]].to_csv(OUT_DIR / "m2_projects.csv",
                                                index=False, encoding="utf-8-sig")

    alloc = []
    for cat in ("I", "E", "C"):
        sub = proj[proj["category"] == cat]
        alloc.append({"category": cat,
                      "selected_projects": ",".join(sub.loc[sub["selected"] == 1, "project_id"])
                      if (sub["selected"] == 1).any() else "",
                      "spend_100m": float(sub["spend_100m"].sum()),
                      "share_of_budget": float(sub["spend_100m"].sum() / best["budget_100m"])
                      if best["budget_100m"] > 0 else np.nan,
                      "delta_from_M1": ins["delta"][("I", "E", "C").index(cat)]})
    alloc_df = pd.DataFrame(alloc)
    alloc_df.loc[len(alloc_df)] = {"category": "TOTAL", "selected_projects": "",
                                   "spend_100m": best["cost_100m"],
                                   "share_of_budget": best["cost_100m"] / best["budget_100m"],
                                   "delta_from_M1": sum(ins["delta"])}
    alloc_df.to_csv(OUT_DIR / "m2_allocation.csv", index=False, encoding="utf-8-sig")

    # ---- 情景明细 ----
    rows = []
    for i, s in scen.iterrows():
        rows.append({"scenario": s["scenario"], "x_s": s["x_s"], "tau_s": s["tau_s"],
                     "B_s": s["B_s"], "budget_100m": best["budget_100m"],
                     "dI": best["dI"], "dE": best["dE"], "dS": best["dS"],
                     "I_s": best["I_s"], "C_env_s": best["C_env_s"], "S_s": best["S_s"],
                     "x_new": best["x_new"][i], "U_s": best["U"][i]})
    pd.DataFrame(rows).to_csv(OUT_DIR / "m2_scenarios.csv", index=False, encoding="utf-8-sig")

    # ---- 蒙特卡洛置信区间 (固定最优 y*, 遍历全部抽样) ----
    U_draws, x_draws = [], []
    for xv, tv in zip(mc["x_s"], mc["tau_s"]):
        xn = (ctx["x0"] * (tv / ctx["tau0"]) ** params["epsilon"]
              * (best["I_s"] / ctx["I_ref"]) ** params["alpha"]
              * (best["S_s"] / params["S0"]) ** params["beta"]
              * ((params["C_env_0"] / (best["C_env_s"] + 0.1)) ** params["gamma"])
              * ins["prod_measures"])
        x_draws.append(xn)
        U_draws.append(M2_CONFIG["lambda_I"] * best["dI"] + M2_CONFIG["lambda_E"] * best["dE"]
                       + M2_CONFIG["lambda_C"] * best["dS"]
                       - M2_CONFIG["lambda_L"] * max(0.0, ctx["x0"] - xn))
    U_draws = np.array(U_draws)
    x_draws = np.array(x_draws)
    need_b = best["cost_100m"] / ctx["delta_total"]
    feas_share = float(np.mean(mc["B_s"] >= need_b))

    stats = {
        "B_s": {"p5": np.percentile(mc["B_s"], 5), "p50": np.percentile(mc["B_s"], 50),
                "p95": np.percentile(mc["B_s"], 95)},
        "x_new": {"p5": np.percentile(x_draws, 5), "p50": np.percentile(x_draws, 50),
                  "p95": np.percentile(x_draws, 95)},
        "U_s": {"p5": np.percentile(U_draws, 5), "p50": np.percentile(U_draws, 50),
                "p95": np.percentile(U_draws, 95)},
        "U_draws": U_draws,
        "budget_feasible_share": feas_share,
    }
    mc_rows = [{"metric": k, "p5": v["p5"], "p50": v["p50"], "p95": v["p95"]}
               for k, v in stats.items() if k in ("B_s", "x_new", "U_s")]
    mc_rows.append({"metric": "budget_feasible_share", "p5": feas_share, "p50": feas_share,
                    "p95": feas_share})
    pd.DataFrame(mc_rows).to_csv(OUT_DIR / "m2_mc_summary.csv", index=False, encoding="utf-8-sig")

    log(f"[蒙卡] x_new 90%CI=[{stats['x_new']['p5']:.2f}, {stats['x_new']['p95']:.2f}] 万, "
        f"U_s 90%CI=[{stats['U_s']['p5']:.4f}, {stats['U_s']['p95']:.4f}], "
        f"U 范围=[{U_draws.min():.4f}, {U_draws.max():.4f}], "
        f"预算可行比例={feas_share*100:.1f}%")
    log("[口径] M2 中 δ_I/δ_E/δ_C 仅进 M2-2 求和约束, 不进目标: 取 M1 最优 δ* 为常数")
    log("[口径观察 1] M2-4 的 S_s=S_0+ΔS 不含 M1-7 的 s_x·(x/x0) 项, 且 S_0=1.0 后 clip 使 "
        f"S_s≡{best['S_s']:.2f} -> 社会成本对需求无反馈 (仅经 λ_C·ΔS 进 U_s); "
        "与 M1 中 S 随游客量下降的口径不一致, 待建模手确认")
    log("[口径观察 2] M2-4 的 C_env,s 只扣减项目反哺, 不含游客规模压力 (M1-4'), "
        f"得 C_env,s={best['C_env_s']:.3f}; M1 最优 C_env={ins['m1_Cenv']:.3f} (地板), "
        "两模型环境口径不一致, 待建模手确认")
    log("[待建模手确认] cost_unit=0.05 亿/级, rho_E=0.5, λ=(1,1,1,0.05,1), M_excl=∅, "
        "sigma_x=10%, sigma_tau=5%, P10/P50/P90")

    # ---- 图 ----
    p1 = fig_closed_loop(ins, best)
    p2 = fig_scenario_ci(scen, mc, best, stats)
    log(f"[输出] {p1.name}, {p2.name}")

    log(f"[耗时] {time.perf_counter() - t0:.2f}s")
    (OUT_DIR / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
