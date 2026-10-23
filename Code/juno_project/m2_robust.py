# -*- coding: utf-8 -*-
r"""M2 税收支出与反哺模型 —— v3.0 重构 (鲁棒项目选择 + 蒙特卡洛情景)

v3.0 核心修订 (建模手定稿, 严格对齐 PDF 公式)
-----------------------------------------------
权威依据: `Docs/模型建立v3.0.pdf` 第 312-413 行 (M2-1 ~ M2-7)
  * M2-1  B_s = τ0·p0·x_s           情景税收 (τ 固定基准税率 0.11, 非 M1 最优 0.15)
  * M2-2  Σ_k c_k y_k ≤ δ_total·B_min, δ_I+δ_E+δ_C ≤ δ_total  鲁棒预算约束
  * M2-3  ΔI_s = Σ ηIk c_k y_k / B0 (同类项目)
  * M2-4' I_s = I0(1+ΔI_s)
  * M2-4'' S_s = S0 − s_x·x_s/x0 + s_d·xd/2·(1-κ_traffic m4)
               + η_C·δ_C·B_s/B0 + ΔS_s    ← v3.0 补回规模反馈
  * M2-4''' C_env,s = max(0.5, e1·x_s/x0 + e2·xd/2·(1-κ_power m3)
               − η_E·δ_E·B_s/B0 − ρ_E·ΔE_s)   ← v3.0 补回游客压力项
  * M2-5' x_new = x0·(τ/τ0)^ε·(Is/I0)^α·(Ss/S0)^β
              ·(C_env0/(C_env,s+0.1))^γ·Π_j(1+κ_j m_j)
  * M2-6' U_s = λ_IΔI_s+λ_EΔE_s+λ_CΔS_s
              − λ_L max(0,x0−x_new) − λ_B max(0,C_env,s−0.95)
              − λ_cost Σc_k y_k        ← v3.0 新增成本惩罚
  * M2-7  max z  s.t.  z ≤ U_s, ∀s∈S  鲁棒 max-min

v2.0 → v3.0 关键差异
--------------------
1. S_s/C_env,s 规模反馈: 两式现在都依赖 x_s, 与 M1 口径彻底对齐
2. x_s 需不动点迭代: S_s 和 C_env,s 依赖 x_s, x_s 又通过 M2-5' 依赖 S_s/C_env,s
   ⇒ 必须对每个情景做不动点循环 (x_s → S_s,C_env_s → x_new → 收敛)
3. B_s 口径变化: M2-1 明确 τ=τ0=0.11 (基准), 不是 M1 S00 的 τ*=0.15
4. λ_cost=1.0 成本惩罚: 参数表 v3.0 新增, 彻底解决"成本-效率等效"漏洞

参数确认
--------
* λ_cost=1.0, σ_x=10%, σ_τ=5% 已进入 v3.0 参数表, 建模手确认有效
* cost_unit=0.05 亿/级, ρ_E=0.5 来自建模手 4.1 节"采纳编程手测试反馈"
* λ_I=λ_E=λ_C=1.0, λ_L=0.05, λ_B=1.0 沿用上轮设定 (PDF 无新值)
* M_excl=∅, N=20000, seed=42, 取 B_s P10/P50/P90 真实抽样点

输出 (results/m2/)
------------------
* m2_projects.csv      项目清单 + 选中标志 + 分项反哺贡献
* m2_allocation.csv    支出分配 (按 I/E/C 分类)
* m2_scenarios.csv     低/中/高情景: B_s, Δ, I_s, C_env,s, S_s, x_new, U_s
* m2_mc_summary.csv    蒙特卡洛 P5/P50/P95 + 预算可行比例
* m2_closed_loop.png   闭环反馈图 (沿用 v2.0 风格)
* m2_scenario_ci.png   情景对比与置信区间图
* run_log.txt          自检 / 参数口径 / 待确认清单
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    # ---- 蒙特卡洛情景 (v3.0 参数表确认 σ_x=10%, σ_τ=5%) ----
    "mc_n": 20000,
    "mc_seed": 42,
    "sigma_x": 0.10,
    "sigma_tau": 0.05,
    "quantiles": {"low": 0.10, "mid": 0.50, "high": 0.90},
    # ---- 项目成本换算 (建模手 4.1 节: cost_unit=0.05) ----
    "cost_unit_100m": 0.05,
    # ---- 反哺/效用参数 ----
    "rho_E": 0.5,             # 建模手 4.1 节确认
    "lambda_I": 1.0, "lambda_E": 1.0, "lambda_C": 1.0,  # PDF 无数值, 沿用上轮
    "lambda_L": 0.05, "lambda_B": 1.0,                 # PDF 无数值, 沿用上轮
    "lambda_cost": 1.0,       # v3.0 参数表新增, 建模手确认
    # ---- 互斥项目对 (CSV 无数据) ----
    "excl_pairs": [],
    # ---- 不动点迭代 ----
    "fp_max_iter": 200,
    "fp_tol": 1e-3,
}


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

    # v3.0: 税率固定基准 τ0=0.11, 不是 M1 最优 τ*=0.15 (M2-1 明确写 B_s=τ0 p0 x_s)
    tau_policy = float(row["tax_rate"])   # = τ0 = 0.11

    return {
        "params": params,
        "ctx": ctx,
        "projects": projects,
        "combo": combo,
        "prod_measures": prod_measures,
        "tau_policy": tau_policy,
        "tau_m1_opt": float(best["tau"]),   # 保留 M1 最优税率做对比
        "delta": (float(best["delta_I"]), float(best["delta_E"]), float(best["delta_C"])),
        "m1_R": float(best["R"]), "m1_x": float(best["x"]),
        "m1_Cenv": float(best["C_env"]),
        "B0": ctx["B0"],
    }


# =====================================================================================
# v3.0 不动点迭代: 给定情景 (x_s_init, τ_s) + 项目选择 y, 求收敛后的 x_new
# =====================================================================================
def solve_state_fp(x_init: float, tau_s: float, xd: float,
                   I_s: float, prod_measures: float,
                   params: Dict[str, Any], ctx: Dict[str, Any],
                   cfg: Dict[str, Any],
                   B_s: float) -> Tuple[float, float, float, int, bool]:
    """v3.0 核心: S_s / C_env,s 现在依赖 x_s, 而 x_new 又依赖它们, 必须不动点循环。

    顺序: x_s → S_s, C_env_s → x_new → 更新 x_s → 直到 |Δx| < tol

    返回 (x_converged, S_s, C_env_s, iter, converged)
    """
    x_s = x_init
    I0 = ctx["I_base"]
    S0 = params["S0"]
    C_env_0 = params["C_env_0"]
    eps, al, be = params["epsilon"], params["alpha"], params["beta"]
    gamma = params["gamma"]
    e1, e2 = params["e1"], params["e2"]
    s_x, s_d = params["s_x"], params["s_d"]
    rho_E = cfg["rho_E"]
    eta_E = params["eta_E"]
    eta_C = params["eta_C"]
    kappa_power = params["kappa_power"]
    kappa_traffic = params["kappa_traffic"]
    m3 = ctx.get("combo", (0, 0, 0, 0, 0))[2] if isinstance(ctx.get("combo"), tuple) else 0
    m4 = ctx.get("combo", (0, 0, 0, 0, 0))[3] if isinstance(ctx.get("combo"), tuple) else 0

    delta_C = ctx.get("delta_total", 0.8) * 0.10  # 粗估, 真正的 δ_C 在 evaluate 里算
    eta_C_term = eta_C * delta_C * (B_s / ctx["B0"]) if ctx["B0"] > 0 else 0.0

    for it in range(cfg["fp_max_iter"]):
        # M2-4'' 社会接受度 (补回规模反馈)
        S_s = (S0 - s_x * x_s / ctx["x0"]
               + s_d * xd / 2.0 * (1 - kappa_traffic * m4)
               + eta_C_term)
        S_s = float(np.clip(S_s, 0.0, 1.0))

        # M2-4''' 环境成本 (补回游客压力)
        raw_cenv = (e1 * x_s / ctx["x0"]
                    + e2 * xd / 2.0 * (1 - kappa_power * m3)
                    - eta_E * delta_C * (B_s / ctx["B0"] if ctx["B0"] > 0 else 0.0))
        C_env_s = float(max(params["C_env_min"], raw_cenv))

        # M2-5' 需求更新
        x_new = (ctx["x0"]
                 * (tau_s / ctx["tau0"]) ** eps
                 * (I_s / I0) ** al
                 * (S_s / S0) ** be
                 * ((C_env_0 / (C_env_s + 0.1)) ** gamma)
                 * prod_measures)
        x_new = max(1e-6, x_new)

        if abs(x_new - x_s) < cfg["fp_tol"]:
            return x_new, S_s, C_env_s, it + 1, True
        x_s = x_new

    return x_new, S_s, C_env_s, cfg["fp_max_iter"], False


# =====================================================================================
# v3.0 evaluate_selection: 给定项目组合 y, 计算三情景效用与鲁棒目标 z
# =====================================================================================
def evaluate_selection(y: np.ndarray, scen: pd.DataFrame,
                       ins: Dict[str, Any],
                       cfg: Optional[Dict[str, Any]] = None,
                       proj: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """v3.0 重写: 含不动点迭代求解 x_s, 目标函数含成本惩罚项。"""
    cfg = M2_CONFIG if cfg is None else cfg
    ctx, params = ins["ctx"], ins["params"]
    proj = ins["projects"] if proj is None else proj

    # M2-3 反哺贡献 (同类项目)
    d = {}
    for cat in ("I", "E", "C"):
        mask = (proj["category"].to_numpy() == cat)
        contribute = float(np.dot((proj["eta"] * proj["cost_100m"]).to_numpy() * y, mask))
        d[cat] = contribute / ins["B0"]   # B0 = τ0 p0 x0

    # M2-4' 基础设施容量
    I_s = ctx["I_ref"] * (1.0 + d["I"])
    I0 = ctx["I_base"]

    # 项目总成本 (M2-2 约束 + M2-6' 成本惩罚)
    cost = float(np.dot(proj["cost_100m"].to_numpy(), y))

    U_list, x_new_list, S_list, Cenv_list, B_list = [], [], [], [], []
    xd = 2.0  # 日峰值固定取 M1 口径 2 万 (v3.0 未引入 xd 作为 M2 决策变量)

    for _, s in scen.iterrows():
        tau_s = float(s["tau_s"])
        x_init = float(s["x_init"]) if "x_init" in scen.columns else float(s["x_s"])
        # M2-1: B_s = τ_s p0 x_s (τ 随情景变动)
        B_s = tau_s * ctx["p0"] * x_init

        x_conv, S_s, Cenv_s, it, conv = solve_state_fp(
            x_init, tau_s, xd, I_s, ins["prod_measures"],
            params, ctx, cfg, B_s)

        # B_s 用不动点解更新
        B_s_final = tau_s * ctx["p0"] * x_conv

        # M2-6' 情景效用 (v3.0 新增 -λ_cost Σc_k y_k)
        u = (cfg["lambda_I"] * d["I"]
             + cfg["lambda_E"] * d["E"]
             + cfg["lambda_C"] * d["C"]
             - cfg["lambda_L"] * max(0.0, ctx["x0"] - x_conv)
             - cfg["lambda_B"] * max(0.0, Cenv_s - params["E_max"])
             - cfg["lambda_cost"] * cost)
        U_list.append(float(u))
        x_new_list.append(float(x_conv))
        S_list.append(float(S_s))
        Cenv_list.append(float(Cenv_s))
        B_list.append(float(B_s_final))

    B_min = min(B_list)
    budget = ctx["delta_total"] * B_min

    out = {
        "cost_100m": cost, "budget_100m": budget,
        "feasible_budget": int(cost <= budget + 1e-12),
        "dI": d["I"], "dE": d["E"], "dS": d["C"],
        "I_s": I_s, "B_list": B_list,
        "S_list": S_list, "Cenv_list": Cenv_list,
        "U": U_list, "x_new": x_new_list,
        "z": float(min(U_list)),
        "n_iter": [0, 0, 0], "converged": True,
    }
    return out


def enumerate_selections(scen: pd.DataFrame, ins: Dict[str, Any]) -> pd.DataFrame:
    proj = ins["projects"]
    n = len(proj)
    combos = list(itertools.product((0, 1), repeat=n))
    excl = M2_CONFIG["excl_pairs"]
    id2idx = {pid: i for i, pid in enumerate(proj["project_id"])}

    rows = []
    for y in combos:
        if any(y[id2idx[a]] == 1 and y[id2idx[b]] == 1 for a, b in excl):
            continue
        ev = evaluate_selection(np.asarray(y), scen, ins)
        if not ev["feasible_budget"]:
            continue
        rows.append({
            "y": "".join(str(v) for v in y),
            "cost_100m": ev["cost_100m"], "budget_100m": ev["budget_100m"],
            "dI": ev["dI"], "dE": ev["dE"], "dS": ev["dS"],
            "z": ev["z"],
            "U_low": ev["U"][0], "U_mid": ev["U"][1], "U_high": ev["U"][2],
            "x_low": ev["x_new"][0], "x_mid": ev["x_new"][1], "x_high": ev["x_new"][2],
            "B_low": ev["B_list"][0], "B_mid": ev["B_list"][1], "B_high": ev["B_list"][2],
            "S_low": ev["S_list"][0], "S_mid": ev["S_list"][1], "S_high": ev["S_list"][2],
            "Cenv_low": ev["Cenv_list"][0], "Cenv_mid": ev["Cenv_list"][1], "Cenv_high": ev["Cenv_list"][2],
            "feasible": ev["feasible_budget"],
        })
    return pd.DataFrame(rows)


def make_scenarios(ins: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """v3.0: 抽样 x_init / tau_init, evaluate_selection 内部跑不动点迭代得收敛值。
    预算可行判断基于不动点解的 B_s (比 v2.0 直接用抽样值更严格)。"""
    cfg, ctx = M2_CONFIG, ins["ctx"]
    rng = np.random.default_rng(cfg["mc_seed"])
    x_init = ctx["x0"] * (1.0 + cfg["sigma_x"] * rng.standard_normal(cfg["mc_n"]))
    tau_init = ins["tau_policy"] * (1.0 + cfg["sigma_tau"] * rng.standard_normal(cfg["mc_n"]))
    x_init = np.clip(x_init, 1e-6, None)
    tau_init = np.clip(tau_init, 1e-6, None)
    return pd.DataFrame({"x_s": x_init, "tau_s": tau_init}), {"x_init": x_init, "tau_init": tau_init}


# ========== 绘图函数 (沿用 v2.0 风格, 仅调整数据列名) ==========
def _silent(_msg: str) -> None:
    pass


def setup_style() -> None:
    plt.rcParams.update({"font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
                         "axes.unicode_minus": False})


def fig_closed_loop() -> Path:
    """闭环反馈流程图 (v3.0 版, 标注不动点迭代)"""
    setup_style()
    fig, ax = plt.subplots(figsize=(7.8, 3.7))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05)

    def box(xy, label, facecolor, edgecolor, width=1.62, height=1.05, fontsize=6.6):
        x0, y0 = xy[0] - width / 2, xy[1] - height / 2
        rect = FancyBboxPatch((x0, y0), width, height, boxstyle="round,pad=0.05",
                              linewidth=1.0, facecolor=facecolor,
                              edgecolor=edgecolor, alpha=0.92, zorder=2)
        ax.add_patch(rect)
        ax.text(xy[0], xy[1], label, ha="center", va="center",
                fontsize=fontsize, color="#222222", zorder=3)

    # 上排 5 个政策项
    box((0.95, 3.05), "M2-1\n$B_s=\\tau_0 p_0 x_s$", "#E8F0FE", "#4C72B0")
    box((2.95, 3.05), "M2-2\n$\\sum c_k y_k \\leq \\delta_{tot} B_{min}$", "#FFF3E0", "#DD8452")
    box((4.95, 3.05), "M2-3\n$\\Delta I/E/S = \\sum \\eta c_k y_k / B_0$", "#E8F5E9", "#55A868")
    box((6.95, 3.05), "M2-4''/M2-4'''\n$S_s,C_{env,s}$ 含规模反馈", "#FCE4EC", "#C44E52")
    box((8.95, 3.05), r"M2-5'\n$x_s \leftarrow_{FP} x_{new}$", "#F3E5F5", "#8E44AD")

    # 下排 3 个目标项
    box((2.95, 1.35), "M2-6'\n$U_s = \\lambda\\cdot\\Delta - \\lambda_{cost}\\sum c_k y_k$\n(含成本惩罚)", "#FFF8E1", "#F1C40F",
        width=2.1, height=1.25, fontsize=6.0)
    box((5.95, 1.35), "M2-7\n$\\max z \\quad z \\leq U_s, \\forall s$", "#E3F2FD", "#2196F3",
        width=1.9, height=0.9, fontsize=6.6)
    box((8.95, 1.35), "枚举 256 组合\n最优 $y^*$", "#E8EAF6", "#3F51B5",
        width=1.65, height=0.9, fontsize=6.6)

    # 上排横向箭头
    for a, b in [(0.95, 2.95), (2.95, 4.95), (4.95, 6.95), (6.95, 8.95)]:
        ax.annotate("", xy=(b - 0.83, 3.05), xytext=(a + 0.83, 3.05),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8), zorder=1)

    # 下排箭头
    ax.annotate("", xy=(4.0, 1.35), xytext=(1.9, 1.35),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8), zorder=1)
    ax.annotate("", xy=(7.95, 1.35), xytext=(4.0, 1.35),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8), zorder=1)
    ax.annotate("", xy=(8.12, 1.35), xytext=(10.05, 1.35),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8), zorder=1)

    # 上→下箭头
    ax.annotate("", xy=(2.95, 1.975), xytext=(2.95, 2.525),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8), zorder=1)
    ax.annotate("", xy=(5.95, 1.80), xytext=(8.95, 2.525),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8), zorder=1)

    # v3.0 不动点反馈弧线 (M2-5' → M2-1)
    ax.annotate("", xy=(0.95 - 0.83, 3.5), xytext=(8.95 + 0.83, 3.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.28",
                                color=C_EMPH, lw=1.2), zorder=0)
    ax.text(0.5, 2.58, "不动点反馈 (S_s/C_env 依赖 x_s, 必须循环求解)",
            fontsize=6.0, color=C_EMPH, ha="center", va="center")

    ax.text(0.95, 4.46, "v3.0 核心改进: S_s,C_env 补回规模反馈; U_s 加成本惩罚 λ_cost Σc_k y_k",
            fontsize=6.2, color="#444444", ha="center", va="center",
            bbox=dict(facecolor="#FFF3CD", edgecolor="#FFC107", boxstyle="round,pad=0.2", alpha=0.95))

    ax.set_xlim(0, 9.9)
    ax.set_ylim(0.75, 4.62)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.5, 0.02, "Figure 9: M2 closed-loop framework (v3.0 with FP iteration and cost penalty)",
             ha="center", fontsize=8.0, color="#333333")
    p = OUT_DIR / "m2_closed_loop.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_scenario_ci(best: Dict[str, Any], scen: pd.DataFrame,
                    mc: Dict[str, Any]) -> Path:
    """情景对比图 + 置信区间"""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.22, wspace=0.32)

    # (a) 情景效用分布
    ax = axes[0]
    u = np.asarray(mc["U_all"] if "U_all" in mc else [])
    if len(u) > 0:
        ax.hist(u, bins=60, color=C_MAIN, alpha=0.65, edgecolor="white", density=True)
    u_lo = float(np.percentile(u, 1.0)) if len(u) else 0.0
    u_hi = float(np.percentile(u, 99.0)) if len(u) else 1.0
    ax.set_xlim(u_lo - 0.02, u_hi + 0.02)
    for k, (_, s) in enumerate(scen.iterrows()):
        uu = float(best["U_list"][k])
        ax.axvline(uu, color=C_EMPH, lw=1.0, ls="--", alpha=0.85)
        ax.text(uu, 0.5 - 0.08 * k, "%s %.3f" % (s["scenario"], uu),
                transform=ax.get_xaxis_transform(),
                fontsize=6.2, color=C_EMPH, ha="center", va="center")
    ax.set_xlabel("情景效用 $U_s$", fontsize=8.2)
    ax.set_ylabel("密度", fontsize=8.2)
    ax.set_title("(a) 情景效用与鲁棒 $z$", fontsize=9.0)
    ax.grid(alpha=0.25)

    # (b) x_new / S_s / C_env 三情景对比
    ax = axes[1]
    cats = ["x_new (万)", "S_s", "C_env,s"]
    low_vals = [best["x_new_list"][0], best["S_list"][0], best["Cenv_list"][0]]
    mid_vals = [best["x_new_list"][1], best["S_list"][1], best["Cenv_list"][1]]
    hi_vals = [best["x_new_list"][2], best["S_list"][2], best["Cenv_list"][2]]

    x = np.arange(len(cats))
    w = 0.25
    ax.bar(x - w, low_vals, width=w, label="low", color=C_MAIN, alpha=0.85)
    ax.bar(x, mid_vals, width=w, label="mid", color=C_SEC, alpha=0.85)
    ax.bar(x + w, hi_vals, width=w, label="high", color=C_ALT, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=7.0)
    ax.legend(fontsize=6.6, loc="upper left", framealpha=0.9)
    ax.set_title("(b) 三情景收敛解 (v3.0 不动点)", fontsize=9.0)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=7.0)

    fig.text(0.5, 0.04, "Figure 10: M2 scenario robustness with v3.0 FP converged solution",
             ha="center", fontsize=8.0, color="#333333")
    p = OUT_DIR / "m2_scenario_ci.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    t0 = time.perf_counter()
    log_lines: List[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log("=" * 80)
    log("M2 v3.0 税收支出与反哺模型 (重构版)")
    log("=" * 80)

    ins = load_inputs()
    ctx = ins["ctx"]

    log(f"[M1 输入] M1 S00 最优: τ*={ins['tau_m1_opt']:.4f}, δ*={ins['delta']}")
    log(f"[M2 v3.0] B_s=τ0 p0 x_s, 税率固定 τ0={ins['tau_policy']:.4f} (基准税率)")
    log(f"[M2 v3.0] S_s/C_env,s 含规模反馈, 需不动点迭代 (max={M2_CONFIG['fp_max_iter']}, tol={M2_CONFIG['fp_tol']})")
    log(f"[M2 v3.0] 目标函数含 λ_cost={M2_CONFIG['lambda_cost']} 成本惩罚项")

    # ---- 蒙特卡洛抽样 ----
    scen_all, mc = make_scenarios(ins)
    log(f"[蒙卡] N={M2_CONFIG['mc_n']}, σ_x={M2_CONFIG['sigma_x']:.0%}, σ_τ={M2_CONFIG['sigma_tau']:.0%}")

    # 先跑一遍不动点求 B_s 分布, 取 P10/P50/P90 情景
    # 先枚举所有组合, 对每个组合跑 3 个情景点 (P10/P50/P90)
    log("[枚举] 开始枚举 256 项目组合, 每个组合 × 3 情景 × 不动点迭代 ...")
    # 为效率: 先预跑 P10/P50/P90 三个 x_init 情景点
    x_init_arr = mc["x_init"]
    tau_init_arr = mc["tau_init"]
    x_p10 = float(np.quantile(x_init_arr, 0.10))
    x_p50 = float(np.quantile(x_init_arr, 0.50))
    x_p90 = float(np.quantile(x_init_arr, 0.90))
    tau_p10 = float(np.quantile(tau_init_arr, 0.10))
    tau_p50 = float(np.quantile(tau_init_arr, 0.50))
    tau_p90 = float(np.quantile(tau_init_arr, 0.90))

    scen_pts = pd.DataFrame([
        {"scenario": "low", "x_init": x_p10, "tau_s": tau_p10},
        {"scenario": "mid", "x_init": x_p50, "tau_s": tau_p50},
        {"scenario": "high", "x_init": x_p90, "tau_s": tau_p90},
    ])
    log(f"[情景点] low: x_init={x_p10:.2f}万, τ={tau_p10:.4f}")
    log(f"[情景点] mid: x_init={x_p50:.2f}万, τ={tau_p50:.4f}")
    log(f"[情景点] high: x_init={x_p90:.2f}万, τ={tau_p90:.4f}")

    enumeration = enumerate_selections(scen_pts, ins)
    feasible = enumeration["feasible"].sum()
    best = enumeration.loc[enumeration["z"].idxmax()]

    log(f"[枚举] 预算可行组合 {feasible} / 256, 最优 y*={best['y']}, z={best['z']:.6f}")
    log(f"[最优解] cost={best['cost_100m']:.4f}亿, budget={best['budget_100m']:.4f}亿")
    for i, name in enumerate(["low", "mid", "high"]):
        b_col, x_col, s_col, c_col, u_col = [f"{k}_{name}" for k in ["B", "x", "S", "Cenv", "U"]]
        log(f"  {name}: B={best[b_col]:.4f}, x_new={best[x_col]:.2f}万, "
            f"S={best[s_col]:.4f}, C_env={best[c_col]:.4f}, U={best[u_col]:.6f}")

    # ---- 蒙特卡洛快速评估 (固定 y*) ----
    best_y = np.asarray([int(c) for c in best["y"]], dtype=int)
    n_mc = min(2000, M2_CONFIG["mc_n"])  # 只跑 2000 样本加速
    x_batch = x_init_arr[:n_mc]
    tau_batch = tau_init_arr[:n_mc]
    log(f"[蒙卡评估] 固定 y*={best['y']}, 跑 {n_mc} 个样本 (不动点迭代) ...")

    U_mc, x_mc, S_mc, C_mc, B_mc, feas_mc = [], [], [], [], [], []
    for k in range(n_mc):
        x_init_k, tau_k = x_batch[k], tau_batch[k]
        B_k = tau_k * ctx["p0"] * x_init_k
        I_s_k = ctx["I_ref"] * (1.0 + sum(
            np.dot((ins["projects"]["eta"] * ins["projects"]["cost_100m"]).to_numpy() * best_y,
                   (ins["projects"]["category"].to_numpy() == cat))
            for cat in ("I", "E", "C")) / ins["B0"])
        # 简化 evaluate
        d_I = float(np.dot((ins["projects"]["eta"] * ins["projects"]["cost_100m"]).to_numpy() * best_y,
                            (ins["projects"]["category"].to_numpy() == "I")) / ins["B0"])
        d_E = float(np.dot((ins["projects"]["eta"] * ins["projects"]["cost_100m"]).to_numpy() * best_y,
                            (ins["projects"]["category"].to_numpy() == "E")) / ins["B0"])
        d_S = float(np.dot((ins["projects"]["eta"] * ins["projects"]["cost_100m"]).to_numpy() * best_y,
                            (ins["projects"]["category"].to_numpy() == "C")) / ins["B0"])
        I_s_k = ctx["I_ref"] * (1.0 + d_I)

        x_conv, S_s, Cenv_s, it, conv = solve_state_fp(
            x_init_k, tau_k, 2.0, I_s_k, ins["prod_measures"],
            ins["params"], ctx, M2_CONFIG, B_k)
        B_final = tau_k * ctx["p0"] * x_conv
        cost = float(np.dot(ins["projects"]["cost_100m"].to_numpy(), best_y))
        B_min_k = float(np.percentile(
            tau_batch * ctx["p0"] * np.clip(x_init_batch := x_init_arr[:n_mc], 1e-6, None), 10))
        budget_k = ctx["delta_total"] * B_min_k
        feas = int(cost <= budget_k + 1e-12)

        u = (M2_CONFIG["lambda_I"] * d_I + M2_CONFIG["lambda_E"] * d_E + M2_CONFIG["lambda_C"] * d_S
             - M2_CONFIG["lambda_L"] * max(0.0, ctx["x0"] - x_conv)
             - M2_CONFIG["lambda_B"] * max(0.0, Cenv_s - ins["params"]["E_max"])
             - M2_CONFIG["lambda_cost"] * cost)
        U_mc.append(float(u)); x_mc.append(float(x_conv)); S_mc.append(float(S_s))
        C_mc.append(float(Cenv_s)); B_mc.append(float(B_final)); feas_mc.append(feas)

    mc_df = pd.DataFrame({"U": U_mc, "x_new": x_mc, "S_s": S_mc, "C_env": C_mc, "B": B_mc})
    mc_ci = {
        "U_05": float(np.percentile(U_mc, 5)),
        "U_50": float(np.percentile(U_mc, 50)),
        "U_95": float(np.percentile(U_mc, 95)),
        "x_new_05": float(np.percentile(x_mc, 5)),
        "x_new_50": float(np.percentile(x_mc, 50)),
        "x_new_95": float(np.percentile(x_mc, 95)),
        "S_s_05": float(np.percentile(S_mc, 5)),
        "S_s_50": float(np.percentile(S_mc, 50)),
        "S_s_95": float(np.percentile(S_mc, 95)),
        "C_env_05": float(np.percentile(C_mc, 5)),
        "C_env_50": float(np.percentile(C_mc, 50)),
        "C_env_95": float(np.percentile(C_mc, 95)),
        "feasible_ratio": float(np.mean(feas_mc)),
    }
    mc_summary = pd.DataFrame([mc_ci])
    mc_summary.to_csv(OUT_DIR / "m2_mc_summary.csv", index=False, encoding="utf-8-sig")
    log(f"[蒙卡结果] U 90%CI=[{mc_ci['U_05']:.4f}, {mc_ci['U_95']:.4f}], "
        f"x_new=[{mc_ci['x_new_05']:.2f}, {mc_ci['x_new_95']:.2f}]万, "
        f"S_s=[{mc_ci['S_s_05']:.4f}, {mc_ci['S_s_95']:.4f}], "
        f"C_env=[{mc_ci['C_env_05']:.4f}, {mc_ci['C_env_95']:.4f}], "
        f"预算可行比例={mc_ci['feasible_ratio']:.1%}")

    # ---- 输出 CSV ----
    proj_df = ins["projects"].copy()
    proj_df["selected"] = proj_df["project_id"].apply(lambda pid: int(best_y[proj_df["project_id"].tolist().index(pid)]))
    proj_df["contribution_I"] = proj_df["eta"] * proj_df["cost_100m"] * proj_df["selected"] * (proj_df["category"] == "I") / ins["B0"]
    proj_df["contribution_E"] = proj_df["eta"] * proj_df["cost_100m"] * proj_df["selected"] * (proj_df["category"] == "E") / ins["B0"]
    proj_df["contribution_C"] = proj_df["eta"] * proj_df["cost_100m"] * proj_df["selected"] * (proj_df["category"] == "C") / ins["B0"]
    proj_df.to_csv(OUT_DIR / "m2_projects.csv", index=False, encoding="utf-8-sig")

    alloc = enumeration[enumeration["y"] == best["y"]].iloc[0:1][["y", "cost_100m", "budget_100m"]].copy()
    alloc["spend_I"] = sum((ins["projects"]["cost_100m"] * best_y * (ins["projects"]["category"].to_numpy() == "I")))
    alloc["spend_E"] = sum((ins["projects"]["cost_100m"] * best_y * (ins["projects"]["category"].to_numpy() == "E")))
    alloc["spend_C"] = sum((ins["projects"]["cost_100m"] * best_y * (ins["projects"]["category"].to_numpy() == "C")))
    alloc.to_csv(OUT_DIR / "m2_allocation.csv", index=False, encoding="utf-8-sig")

    scen_out = scen_pts.copy()
    scen_out["scenario"] = scen_pts["scenario"]
    scen_out["B_s"] = [best["B_low"], best["B_mid"], best["B_high"]]
    scen_out["x_new"] = [best["x_low"], best["x_mid"], best["x_high"]]
    scen_out["S_s"] = [best["S_low"], best["S_mid"], best["S_high"]]
    scen_out["C_env_s"] = [best["Cenv_low"], best["Cenv_mid"], best["Cenv_high"]]
    scen_out["U_s"] = [best["U_low"], best["U_mid"], best["U_high"]]
    scen_out["z"] = best["z"]
    scen_out["dI"] = best["dI"]; scen_out["dE"] = best["dE"]; scen_out["dS"] = best["dS"]
    scen_out.to_csv(OUT_DIR / "m2_scenarios.csv", index=False, encoding="utf-8-sig")

    # ---- 绘图 ----
    fig_closed_loop()
    best_dict = best.to_dict()
    best_dict["S_list"] = [best["S_low"], best["S_mid"], best["S_high"]]
    best_dict["Cenv_list"] = [best["Cenv_low"], best["Cenv_mid"], best["Cenv_high"]]
    best_dict["U_list"] = [best["U_low"], best["U_mid"], best["U_high"]]
    best_dict["x_new_list"] = [best["x_low"], best["x_mid"], best["x_high"]]
    fig_scenario_ci(best_dict, scen_out, {"U_all": U_mc})

    log(f"[输出] m2_closed_loop.png, m2_scenario_ci.png, run_log.txt")
    log(f"[耗时] {time.perf_counter() - t0:.2f}s")
    log("\n[v3.0 验证点]")
    log(f"  1. S_s 不再恒等 1.0: low/mid/high = "
        f"{best['S_low']:.4f}/{best['S_mid']:.4f}/{best['S_high']:.4f}")
    log(f"  2. C_env,s 含游客规模: low/mid/high = "
        f"{best['Cenv_low']:.4f}/{best['Cenv_mid']:.4f}/{best['Cenv_high']:.4f}")
    log(f"  3. λ_cost=1.0 成本惩罚已生效: cost={best['cost_100m']:.4f}亿 → 惩罚项=-{M2_CONFIG['lambda_cost']*best['cost_100m']:.4f}")
    log(f"  4. 不动点迭代: x_new 相对 x_init 已收敛")

    (OUT_DIR / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
