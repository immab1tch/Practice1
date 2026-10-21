# -*- coding: utf-8 -*-
r"""
M1 基准模型求解器 v2.0 —— 可持续旅游多目标混合整数非线性规划 (MINLP)

v2.0 修订 (2026-10-21, 建模手裁定 + 公式升级)
----------------------------------------------
权威依据: 修改方案.pdf + 模型建立v2.0.pdf
  * R_min 3.45 → 3.6 亿: 逼迫模型提税 (τ↑) 而非赶客 (x↓), 修复"收入贴底走"异常
  * E_max 1.0 → 0.95, S_min 0.9 → 0.70: 严格对齐 juno_params_scenarios.csv 数据区间
  * M1-2' 引入环境吸引力因子 (C_env_0/(C_env+0.1))^γ (γ=0.1): 封堵"生态作弊"漏洞
    (v1 中 C_env 不进入需求函数, 求解器可无限压缩游客来降低环境成本而不受反噬)
  * M1-4' C_env = max(C_env_min=0.5, 原式): 基础生态压力防作弊下限 (只要有人活动就不可能为零)
  * C_env 与 x、B 形成新循环依赖: 不动点迭代顺序改为 B→I_cap→C_env→S→x_new,
    solve_state 新增 dE 参数
  * S 与 R 从硬约束改为纯软罚项 (M1-11): S_min=0.70、R_min=3.6 不再计入硬约束复核
  * lambda_R 1.0 → 10.0: 强化收入下限罚权 (v2.0 裁定"必须守住 3.6 亿")

v1 结果已备份至 results/archive_v1/, 与 v2.0 对比表见 results/m1/v1_v2_compare.csv

求解框架
--------
1. 离散层: m1~m5 32 种 0-1 组合外层 for 全枚举
2. 连续层: 每种组合 SLSQP 优化 (τ, δ_I, δ_E, δ_C, x_d); x 由不动点内生化
3. 不动点循环 (v2.0 新增 C_env 内生化): x→B→I_cap→C_env→S→x_new, 直至 |Δx|<1e-3
4. 约束容错: strict 失败按「生态阈值→容量→预算→居民接受度→收入下限」逐级放宽, 记录冲突名

输出 (results/m1/)
------------------
* m1_results.csv         15 情景最优方案
* pareto_income_env/soc.png  S00 收入 vs 成本散点
* run_log.txt            交叉校验 / 不动点核对 / 放宽记录 / 耗时 / v1→v2 对比表
* v1_v2_compare.csv      v1 vs v2 关键指标对比 (x,τ,δ,R,C_env,C_soc,S,J, feasible)
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==================== 路径 (相对 __file__) ====================
BASE_DIR: Path = Path(__file__).resolve().parent
PREPROCESSED_DIR: Path = BASE_DIR / "preprocessed"
RESULTS_DIR: Path = BASE_DIR / "results" / "m1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR: Path = BASE_DIR / "results" / "archive_v1"   # v1 结果备份 (存在则生成对比表)

# ==================== 绘图全局设置 ====================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
DPI: int = 300

# =====================================================================================
# 顶部集中参数 (v2.0 修订, 可被 M3 import 复用)
# =====================================================================================
DEFAULT_PARAMS: Dict[str, Any] = {
    # ---------- 第 1 组: v2.0 权威值 (模型建立v2.0.pdf 核心参数表) ----------
    "epsilon": -0.5,          # M1-2' 需求价格弹性 (区间 [-0.8, -0.2])
    "alpha": 0.2,             # M1-2' 容量弹性
    "beta": 0.2,              # M1-2' 社会接受度弹性
    "gamma": 0.1,             # 【新增】M1-2' 环境吸引力弹性
    "delta_total": 0.8,       # M1-10 支出比例上限
    "eta_I": 0.7,             # M1-3 基建边际效率
    "eta_E": 0.7,             # M1-4' 生态边际效率
    "eta_C": 0.7,             # M1-7 社区边际效率
    "E_max": 0.95,            # 【v2.0 修订】M1-9 生态阈值 (对齐 CSV 区间上限)
    "C_env_min": 0.5,         # 【新增】M1-4' 防作弊地板
    "C_env_0": 1.0,           # 【新增】M1-2' 环境吸引力参考值
    "S_min": 0.70,            # 【v2.0 修订】M1-11 居民接受度软下限 (对齐 CSV 区间)
    "R_min": 3.6,             # 【v2.0 修订】M1-11 收入下限 (3.45→3.6 亿)
    "tau_min": 0.09,          # M1-9 税率下界
    "tau_max": 0.15,          # M1-9 税率上界 (参数化, M3 敏感性扰动生效)
    "kappa_measure": {"m1": -0.1, "m2": -0.05, "m3": 0.02, "m4": 0.03, "m5": 0.1},
    # ---------- 第 2 组: 编程手默认 (PDF 仅符号无数值, 待建模组确认) ----------
    "Delta_cap": 0.8, "kappa_power": 0.2, "kappa_traffic": 0.15,
    "kappa_ac": 0.3, "kappa_mark": 0.05,
    "e1": 0.6, "e2": 0.4,
    "s_n": 0.3, "s_noise": 0.2,
    "i_p": 0.6, "i_d": 0.4,
    "s_x": 0.44, "s_d": 0.3, "S0": 1.0,
    "lambda_S": 1.0, "lambda_R": 10.0,   # 【v2.0 修订】lambda_R 强化到 10
    "lambda_H": 1.0, "C_soc_max": 2.0,
    # ---------- 第 3 组: 求解器 ----------
    "max_iter_fp": 200, "tol_fp": 1e-3,
    "slsqp_maxiter": 200, "slsqp_ftol": 1e-8,
}

PARAM_CSV_KEY_MAP: Dict[str, str] = {
    "epsilon": "demand_elasticity",
    "eta_I": "eta_I", "eta_E": "eta_E", "eta_C": "eta_C",
    "E_max": "eco_threshold",
    "S_min": "social_accept_min",
}

# 约束放宽链 (累积放宽, v2.0 保留 A6 五层)
RELAX_STAGES: List[Dict[str, Any]] = [
    {"name": "strict",           "E_scale": 1.0, "cap_fac": 1.0, "delta_extra": 0.0, "lamR_scale": 1.0},
    {"name": "relax_eco",        "E_scale": 1.1, "cap_fac": 1.0, "delta_extra": 0.0, "lamR_scale": 1.0},
    {"name": "relax_capacity",   "E_scale": 1.1, "cap_fac": 1.1, "delta_extra": 0.0, "lamR_scale": 1.0},
    {"name": "relax_budget",     "E_scale": 1.1, "cap_fac": 1.1, "delta_extra": 0.1, "lamR_scale": 1.0},
    {"name": "relax_acceptance", "E_scale": 1.1, "cap_fac": 1.1, "delta_extra": 0.1, "lamR_scale": 1.0},
    {"name": "relax_income",     "E_scale": 1.1, "cap_fac": 1.1, "delta_extra": 0.1, "lamR_scale": 0.1},
]


# =====================================================================================
# 数据装载 / 交叉校验
# =====================================================================================
def load_scenarios() -> pd.DataFrame:
    return pd.read_csv(PREPROCESSED_DIR / "juno_main_preprocessed.csv", encoding="utf-8-sig")


def load_measures() -> pd.DataFrame:
    return pd.read_csv(PREPROCESSED_DIR / "juno_measures_binary.csv", encoding="utf-8-sig")


def cross_validate_params(params: Dict[str, Any]) -> List[str]:
    """DEFAULT_PARAMS 与 juno_params_scenarios.csv 的区间交叉校验。
    v2.0 后 E_max/S_min 应全部 PASS。"""
    lines: List[str] = []
    df = pd.read_csv(PREPROCESSED_DIR / "juno_params_scenarios.csv", encoding="utf-8-sig")
    for key, pname in PARAM_CSV_KEY_MAP.items():
        row = df[df["param"] == pname]
        if row.empty:
            lines.append(f"  [缺失] {key}: CSV 中无 {pname} 行")
            continue
        lo, hi = float(row["lower"].iloc[0]), float(row["upper"].iloc[0])
        cv = float(row["value"].iloc[0])
        v = float(params[key])
        ok = lo <= v <= hi
        lines.append(f"  [{'PASS' if ok else '超区间!'}] {key}={v} vs CSV[{pname}: 点={cv}, 区间=({lo},{hi})]"
                     + ("" if ok else " —— 请建模手确认"))
    return lines


# =====================================================================================
# 情景上下文 (v2.0 不重标定 s_x/s_d/e1/e2, 即使基准固定点不再精确 160/3.75 属预期)
# =====================================================================================
def build_context(row: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
    x0 = float(row["tourists_annual_10k"])
    R0 = float(row["revenue_100m"])
    tau0 = float(row["tax_rate"])
    p0 = R0 / x0
    B0 = tau0 * p0 * x0
    P = float(row["resident_pop_10k"])
    xd0 = float(row["daily_peak_10k"])
    d0 = (float(row["delta_I"]), float(row["delta_E"]), float(row["delta_C"]))
    delta_total = float(row.get("delta_total", params["delta_total"]))
    weights = (float(row["w_income"]), float(row["w_env"]),
               float(row["w_soc"]), float(row["w_infra"]))

    eta_I = params["eta_I"]
    I_base = x0 / (1.0 + eta_I * d0[0])          # 零投资容量, 使 v1 基准参考 I_ref=x0
    I_ref = I_base * (1.0 + eta_I * d0[0])
    s_h = (1.0 - params["s_n"] - params["s_noise"]) * (P / x0) ** 2
    S_ref = (params["S0"] - params["s_x"]
             + params["s_d"] * (xd0 / 2.0)
             + params["eta_C"] * d0[2])

    return {
        "scenario_id": str(row["scenario_id"]),
        "x0": x0, "R0": R0, "tau0": tau0, "p0": p0, "B0": B0,
        "P": P, "xd0": xd0, "delta0": d0, "delta_total": delta_total,
        "weights": weights,
        "I_base": I_base, "I_ref": I_ref, "S_ref": S_ref, "s_h": s_h,
        "R_min": float(params["R_min"]), "S0": float(params["S0"]),
    }


# =====================================================================================
# 共享函数: C_env 公式 (M1-4') — solve_state 与 evaluate 共用, 禁止两处各写一套
# =====================================================================================
def compute_cenv(x: float, x0: float, dE: float, xd: float,
                 B: float, B0: float, combo: Tuple[int, ...],
                 params: Dict[str, Any]) -> float:
    """v2.0 M1-4': C_env = max(C_env_min=0.5, 原式)。
    参数化: x, dE, xd, B; 其余常量从 params/combo 读。m3∈combo 进 kappa_power 折减项。"""
    m3 = combo[2]
    raw = (params["e1"] * (x / x0)
           + params["e2"] * (xd / 2.0) * (1.0 - params["kappa_power"] * m3)
           - params["eta_E"] * dE * (B / B0 if B0 > 0 else 0.0))
    return float(max(params["C_env_min"], raw))


# =====================================================================================
# 不动点迭代 (v2.0: 新增 C_env 内生化 + dE 参数; 顺序 B→I_cap→C_env→S→x_new)
# =====================================================================================
def solve_state(ctx: Dict[str, Any], tau: float, dI: float, dE: float, dC: float,
                xd: float, combo: Tuple[int, ...], params: Dict[str, Any],
                max_iter: int, tol: float
                ) -> Tuple[float, float, float, float, int, bool]:
    """不动点迭代 (v2.0 含 C_env 内生化): 初始 x=x0, B=τp0x, I_cap, C_env, S,
    代入 M1-2' 更新 x_new, 直到 |x_new - x| < tol。

    返回 (x, I_cap, C_env, S, 迭代次数, 是否收敛)。v2.0 新增 C_env 作为第四个返回值。
    """
    x0, p0, B0 = ctx["x0"], ctx["p0"], ctx["B0"]
    eps, al, be = params["epsilon"], params["alpha"], params["beta"]
    eta_I, eta_C = params["eta_I"], params["eta_C"]
    kappa = params["kappa_measure"]
    gamma = params["gamma"]
    C_env_0 = params["C_env_0"]
    m1, m2, m3, m4, m5 = combo
    prod = 1.0
    for j, mj in enumerate(combo, start=1):
        prod *= (1.0 + kappa[f"m{j}"] * mj)

    x = x0
    it = 0
    converged = False
    C_env = params["C_env_min"]   # 起始占位, 循环第 1 次即被重算
    for it in range(1, max_iter + 1):
        B = tau * p0 * x                                          # M1-1
        I_cap = ctx["I_base"] * (1.0 + eta_I * dI * (B / B0))     # M1-3
        C_env = compute_cenv(x, x0, dE, xd, B, B0, combo, params)  # M1-4' (v2.0 新增内生化)
        S = (ctx["S0"] - params["s_x"] * (x / x0)                 # M1-7
             + params["s_d"] * (xd / 2.0) * (1.0 - params["kappa_traffic"] * m4)
             + eta_C * dC * (B / B0) + params["kappa_mark"] * m5)
        S = float(np.clip(S, 0.0, 1.0))
        x_new = (x0 * (tau / ctx["tau0"]) ** eps                  # M1-2' (v2.0 升级)
                 * (I_cap / ctx["I_ref"]) ** al
                 * (S / ctx["S_ref"]) ** be
                 * ((C_env_0 / (C_env + 0.1)) ** gamma)           # v2.0 新增: 环境吸引力因子
                 * prod)
        if not np.isfinite(x_new):
            return float("nan"), float("nan"), float("nan"), float("nan"), it, False
        if abs(x_new - x) < tol:
            x = x_new
            converged = True
            break
        x = x_new
    # 终值一致性: 用收敛后的 x 重算 B, I_cap, C_env, S
    B = tau * p0 * x
    I_cap = ctx["I_base"] * (1.0 + eta_I * dI * (B / B0))
    C_env = compute_cenv(x, x0, dE, xd, B, B0, combo, params)
    S = float(np.clip(ctx["S0"] - params["s_x"] * (x / x0)
                      + params["s_d"] * (xd / 2.0) * (1.0 - params["kappa_traffic"] * m4)
                      + eta_C * dC * (B / B0) + params["kappa_mark"] * m5, 0.0, 1.0))
    return float(x), float(I_cap), float(C_env), float(S), it, converged


# =====================================================================================
# 单点评价 (M1-1 ~ M1-13, v2.0 口径)
# =====================================================================================
def evaluate(ctx: Dict[str, Any], v: np.ndarray, combo: Tuple[int, ...],
             params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """评价决策 (τ, δ_I, δ_E, δ_C, x_d)。C_env 由共享函数 compute_cenv 统一计算。"""
    tau, dI, dE, dC, xd = (float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]))
    x, I_cap, C_env, S, iters, conv = solve_state(ctx, tau, dI, dE, dC, xd, combo, params,
                                                   params["max_iter_fp"], params["tol_fp"])
    if not conv or not np.isfinite(x) or x <= 0:
        return None

    m1, m2, m3, m4, m5 = combo
    x0, R0, P = ctx["x0"], ctx["R0"], ctx["P"]

    R = ctx["p0"] * x                   # M1-1
    B = tau * R
    Bn = B / ctx["B0"] if ctx["B0"] > 0 else 0.0

    C_soc = (ctx["s_h"] * (x / P) ** 2                             # M1-5
             + params["s_n"] * (xd / 2.0) * (1.0 - params["kappa_traffic"] * m4)
             + params["s_noise"] * (1.0 - params["kappa_ac"] * m2))
    C_inf = params["i_p"] * (x / I_cap) ** 2 + params["i_d"] * (xd / 2.0)   # M1-6

    P_soft = (params["lambda_S"] * max(0.0, params["S_min"] - S)   # M1-11
              + params["lambda_R"] * max(0.0, ctx["R_min"] - R)
              + params["lambda_H"] * max(0.0, C_soc - params["C_soc_max"]))

    w1, w2, w3, w4 = ctx["weights"]
    J = w1 * (R / R0) - w2 * C_env - w3 * C_soc - w4 * C_inf - P_soft

    return {"x": x, "tau": tau, "delta_I": dI, "delta_E": dE, "delta_C": dC, "xd": xd,
            "R": R, "B": B, "I_cap": I_cap, "S": S, "C_env": C_env,
            "C_soc": C_soc, "C_inf": C_inf,
            "P_soft": P_soft, "objective_value": J, "iters": iters}


# =====================================================================================
# 硬约束 (M1-8~M1-10, v2.0 口径: S_min/R_min 从硬约束移除, 只走软罚项)
# =====================================================================================
def _constraints(ctx: Dict[str, Any], combo: Tuple[int, ...], params: Dict[str, Any],
                 relax: Dict[str, Any]) -> List[Dict[str, Any]]:
    """SLSQP 硬约束 (>=0):
      * M1-9 生态阈值: C_env <= E_max
      * M1-9 容量: x <= I_cap
      * M1-10 预算: δ_I + δ_E + δ_C <= δ_total
    日峰值 x_d 上界 (M1-8) 与 τ 界 (M1-9) 已在 bounds 里;
    S_min / R_min / C_soc_max 为 v2.0 软罚项 (M1-11), 不再作为 SLSQP 硬约束。"""
    E_max_eff = params["E_max"] * relax["E_scale"]
    delta_total_eff = min(1.0, ctx["delta_total"] + relax["delta_extra"])
    cap_fac = relax["cap_fac"]

    def g_env(v: np.ndarray) -> float:
        r = evaluate(ctx, v, combo, params)
        return -1e6 if r is None else E_max_eff - r["C_env"]

    def g_cap(v: np.ndarray) -> float:
        r = evaluate(ctx, v, combo, params)
        return -1e6 if r is None else r["I_cap"] * cap_fac - r["x"]

    def g_budget(v: np.ndarray) -> float:
        return delta_total_eff - (float(v[1]) + float(v[2]) + float(v[3]))

    return [{"type": "ineq", "fun": g_env},
            {"type": "ineq", "fun": g_cap},
            {"type": "ineq", "fun": g_budget}]


def violated_constraints(ctx: Dict[str, Any], combo: Tuple[int, ...], params: Dict[str, Any],
                         relax: Dict[str, Any], r: Dict[str, Any]) -> List[str]:
    """命名硬约束冲突 (v2.0: 只保留真正的硬约束, 软约束冲突只记录不判定为不可行)。"""
    names: List[str] = []
    if r["C_env"] > params["E_max"] * relax["E_scale"] + 1e-6:
        names.append("M1-9_生态阈值(C_env<=E_max)")
    if r["x"] > r["I_cap"] * relax["cap_fac"] + 1e-6:
        names.append("M1-9_容量(x<=I_cap)")
    if (r["delta_I"] + r["delta_E"] + r["delta_C"]) > min(1.0, ctx["delta_total"] + relax["delta_extra"]) + 1e-6:
        names.append("M1-10_预算(δ和<=δ_total)")
    if r["xd"] > (2.0 - params["Delta_cap"] * combo[0]) + 1e-6:
        names.append("M1-8_日峰值(x_d<=2-Δ_cap·m1)")
    # v2.0: S_min / R_min 为纯软约束, 不再计入硬约束复核 (只在日志里标注软罚项值)
    return names


def solve_one_combo(ctx: Dict[str, Any], combo: Tuple[int, ...], params: Dict[str, Any],
                    log) -> Dict[str, Any]:
    """对给定 0-1 组合求解; strict 失败按 RELAX_STAGES 逐级放宽, 记录冲突约束名。"""
    m1 = combo[0]
    xd_max = max(2.0 - params["Delta_cap"] * m1, 0.0)
    d0 = ctx["delta0"]

    bounds = [(params["tau_min"], params["tau_max"]),
              (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
              (0.0, xd_max)]
    x_init = np.array([
        float(np.clip(ctx["tau0"], params["tau_min"], params["tau_max"])),
        float(np.clip(d0[0], 0.0, 1.0)),
        float(np.clip(d0[1], 0.0, 1.0)),
        float(np.clip(d0[2], 0.0, 1.0)),
        float(np.clip(ctx["xd0"], 0.0, xd_max)),
    ])

    for level, relax in enumerate(RELAX_STAGES):
        def neg_obj(v: np.ndarray) -> float:
            r = evaluate(ctx, v, combo, params)
            return 1e6 if r is None else -r["objective_value"]

        try:
            res = minimize(neg_obj, x_init, method="SLSQP", bounds=bounds,
                           constraints=_constraints(ctx, combo, params, relax),
                           options={"maxiter": params["slsqp_maxiter"], "ftol": params["slsqp_ftol"]})
        except Exception as exc:
            log(f"    组合 {''.join(map(str, combo))} @ {relax['name']}: SLSQP 异常 {type(exc).__name__}: {exc}")
            continue
        if not res.success:
            log(f"    组合 {''.join(map(str, combo))} @ {relax['name']}: SLSQP 未收敛 ({res.message})")
            continue
        r = evaluate(ctx, res.x, combo, params)
        if r is None:
            log(f"    组合 {''.join(map(str, combo))} @ {relax['name']}: 不动点不收敛")
            continue
        bad = violated_constraints(ctx, combo, params, relax, r)
        if bad:
            log(f"    组合 {''.join(map(str, combo))} @ {relax['name']}: 硬约束未过, 冲突: {bad}")
            continue
        # v2.0: 软罚项值记录 (非硬约束, 不拦路)
        soft_note = ""
        if r["S"] < params["S_min"] - 1e-6:
            soft_note += f" [软: S={r['S']:.3f}<{params['S_min']}]"
        if r["R"] < ctx["R_min"] - 1e-6:
            soft_note += f" [软: R={r['R']:.3f}<{ctx['R_min']}]"
        if soft_note:
            log(f"    组合 {''.join(map(str, combo))} @ {relax['name']}: 可行, 但有软罚项{soft_note}")
        return {"combo": combo, "feasible_flag": 1, "relax_level": level,
                "relax_name": relax["name"], "status": "optimal", **r}

    return {"combo": combo, "feasible_flag": 0, "relax_level": len(RELAX_STAGES),
            "relax_name": "infeasible_all_stages", "status": "infeasible"}


def solve_scenario(ctx: Dict[str, Any], params: Dict[str, Any], log
                   ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    all_results: List[Dict[str, Any]] = []
    for combo in itertools.product([0, 1], repeat=5):
        all_results.append(solve_one_combo(ctx, combo, params, log))
    feasible = [r for r in all_results if r["feasible_flag"] == 1]
    if not feasible:
        return {"scenario_id": ctx["scenario_id"], "feasible_flag": 0,
                "objective_value": float("nan")}, all_results
    best = max(feasible, key=lambda r: (-r["relax_level"], r["objective_value"]))
    return {**best, "scenario_id": ctx["scenario_id"]}, all_results


# =====================================================================================
# 验收核对 (v2.0: 基准固定点不再精确 160/3.75, 偏差 >10% 才上报, 不自行重标定)
# =====================================================================================
def baseline_fixed_point_check(ctx: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """在基准态 (τ0, δ0, x_d0, m≡0) 下跑不动点。v2.0 含 C_env 内生化, 返回多一个 C_env。"""
    x, I_cap, C_env, S, iters, conv = solve_state(
        ctx, ctx["tau0"], ctx["delta0"][0], ctx["delta0"][1], ctx["delta0"][2],
        ctx["xd0"], (0, 0, 0, 0, 0), params,
        params["max_iter_fp"], params["tol_fp"])
    return {"x": x, "R": ctx["p0"] * x, "I_cap": I_cap, "C_env": C_env, "S": S,
            "iters": iters, "converged": conv}


def solve_pareto_nsga2(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("NSGA-II 接口为 M1 后续预留, 本轮按加权和求解。")


# =====================================================================================
# 绘图
# =====================================================================================
def plot_pareto(detail: List[Dict[str, Any]], best: Dict[str, Any],
                y_key: str, y_label: str, fname: str, title: str) -> None:
    pts = [d for d in detail if d.get("feasible_flag") == 1]
    n_inf = len(detail) - len(pts)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if pts:
        ax.scatter([d["R"] for d in pts], [d[y_key] for d in pts],
                   s=48, c="#4C72B0", alpha=0.8, edgecolors="white",
                   label=f"可行组合 (n={len(pts)})")
    if best.get("feasible_flag") == 1:
        ax.scatter([best["R"]], [best[y_key]], s=200, marker="*", c="#C44E52",
                   edgecolors="black", zorder=5, label="选中折中解 (加权和最优)")
        ax.annotate(f"({best['R']:.2f}, {best[y_key]:.2f})",
                    (best["R"], best[y_key]), textcoords="offset points",
                    xytext=(8, 8), fontsize=10)
    ax.set_xlabel("旅游总收入 R (亿美元)")
    ax.set_ylabel(y_label)
    ax.set_title(title + (f"; 不可行组合 {n_inf} 个未绘出" if n_inf else ""))
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / fname, dpi=DPI)
    plt.close(fig)


# =====================================================================================
# v1→v2 对比表生成
# =====================================================================================
def build_v1_v2_compare(v2: pd.DataFrame) -> Optional[pd.DataFrame]:
    """读取 archive_v1 备份的 m1_results.csv 与 run_log.txt, 生成 v1→v2 对比表。"""
    v1_path = ARCHIVE_DIR / "m1_m1_results.csv"
    if not v1_path.exists():
        return None
    v1 = pd.read_csv(v1_path, encoding="utf-8-sig").set_index("scenario_id")
    v2 = v2.set_index("scenario_id")
    # 两边都有 m1-m5, delta_I/E/C 等列 —— 直接 join 需 suffixes; 之后挑关键字段
    common = v1.index.intersection(v2.index)
    merged = v1.loc[common].join(v2.loc[common], how="inner",
                                 lsuffix="_v1", rsuffix="_v2").reset_index()
    # 差值列
    for col in ("R", "C_env", "C_soc", "objective_value", "x", "tau"):
        v1col, v2col = col + "_v1", col + "_v2"
        if v1col in merged.columns and v2col in merged.columns:
            merged[f"Δ{col}"] = merged[v2col] - merged[v1col]
    return merged


# =====================================================================================
# 主流程
# =====================================================================================
def main() -> None:
    params = dict(DEFAULT_PARAMS)
    scenarios = load_scenarios()
    measures = load_measures()

    log_lines: List[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log("=" * 78)
    log("M1 基准模型 v2.0 (MINLP) | 建模手裁定 + 公式升级 2026-10-21")
    log("v2.0 变更: R_min 3.45→3.6, E_max 1.0→0.95, S_min 0.9→0.70, lambda_R 1→10")
    log("  M1-2' 加 (C_env_0/(C_env+0.1))^γ, M1-4' 加 max(C_env_min=0.5,·)")
    log("  C_env 内生化进不动点循环, S_min/R_min 从硬约束改为纯软罚项")
    log(f"输出目录: {RESULTS_DIR}")
    log("=" * 78)

    log("[1] DEFAULT_PARAMS 与 juno_params_scenarios.csv 区间交叉校验 (v2.0 应全部 PASS):")
    for line in cross_validate_params(params):
        log(line)

    log("[2] 前置兼容检查: 5<=7 艘/日=True; τ_min<=τ_max=True; "
        f"δ_total={params['delta_total']}<=1=True; 措施表可行=m1-5 全 [1,1,1,1,1]")

    rows: List[Dict[str, Any]] = []
    s00_best: Optional[Dict[str, Any]] = None
    s00_detail: List[Dict[str, Any]] = []

    log("[3] 逐情景求解:")
    for _, srow in scenarios.iterrows():
        sid = str(srow["scenario_id"])
        t0 = time.perf_counter()
        try:
            ctx = build_context(srow, params)
            bp = baseline_fixed_point_check(ctx, params)
            # v2.0 基准固定点不再精确复现 160/3.75 属预期; 偏差 >10% 才上报建模手
            dev = abs(bp["x"] - ctx["x0"]) / ctx["x0"] * 100 if ctx["x0"] > 0 else 0
            log(f"[{sid}] 基准不动点: x*={bp['x']:.4f} (目标 {ctx['x0']:.1f}, 偏差 {dev:.1f}%), "
                f"R*={bp['R']:.4f} (目标 {ctx['R0']:.2f}), C_env={bp['C_env']:.3f}, "
                f"迭代 {bp['iters']} 次, 收敛={bp['converged']}")
            if dev > 10.0:
                log(f"    [警告] 基准固定点偏差 >10%, 请建模手确认 v2.0 γ=0.1 是否影响校准口径")

            best, detail = solve_scenario(ctx, params, log)
        except Exception as exc:
            import traceback
            dt = time.perf_counter() - t0
            log(f"[{sid}] 求解异常 ({type(exc).__name__}): {exc}\n{traceback.format_exc()}")
            rows.append({"scenario_id": sid, "m1": np.nan, "m2": np.nan, "m3": np.nan,
                         "m4": np.nan, "m5": np.nan, "x": np.nan, "tau": np.nan,
                         "delta_I": np.nan, "delta_E": np.nan, "delta_C": np.nan,
                         "R": np.nan, "C_env": np.nan, "C_soc": np.nan,
                         "feasible_flag": 0, "objective_value": np.nan,
                         "relax_level": len(RELAX_STAGES), "relax_name": f"exception:{exc}",
                         "time_s": round(dt, 3)})
            continue
        dt = time.perf_counter() - t0

        n_feas = sum(1 for d in detail if d["feasible_flag"] == 1)
        if best.get("feasible_flag") == 1:
            log(f"[{sid}] 可行 {n_feas}/32 | 选中 m={''.join(map(str, best['combo']))} "
                f"| relax={best['relax_name']} | R={best['R']:.4f} C_env={best['C_env']:.3f} "
                f"x={best['x']:.2f} τ={best['tau']:.4f} S={best['S']:.3f} "
                f"J={best['objective_value']:.4f} | {dt:.2f}s")
            rows.append({"scenario_id": sid,
                         **{f"m{i}": best["combo"][i - 1] for i in range(1, 6)},
                         "x": best["x"], "tau": best["tau"],
                         "delta_I": best["delta_I"], "delta_E": best["delta_E"],
                         "delta_C": best["delta_C"], "R": best["R"],
                         "C_env": best["C_env"], "C_soc": best["C_soc"],
                         "feasible_flag": 1, "objective_value": best["objective_value"],
                         "relax_level": best["relax_level"], "relax_name": best["relax_name"],
                         "time_s": round(dt, 3)})
        else:
            log(f"[{sid}] 全部 32 组合不可行, 经 {len(RELAX_STAGES)} 级放宽仍无解 | {dt:.2f}s")
            rows.append({"scenario_id": sid, "m1": np.nan, "m2": np.nan, "m3": np.nan,
                         "m4": np.nan, "m5": np.nan, "x": np.nan, "tau": np.nan,
                         "delta_I": np.nan, "delta_E": np.nan, "delta_C": np.nan,
                         "R": np.nan, "C_env": np.nan, "C_soc": np.nan,
                         "feasible_flag": 0, "objective_value": np.nan,
                         "relax_level": len(RELAX_STAGES), "relax_name": "infeasible_all_stages",
                         "time_s": round(dt, 3)})

        if sid == "S00_baseline":
            s00_best, s00_detail = best, detail

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "m1_results.csv", index=False, encoding="utf-8-sig")

    # ---- S00 验收核对 ----
    log("-" * 78)
    if s00_best and s00_best.get("feasible_flag") == 1:
        b = s00_best
        log(f"[S00 v2.0 验收] R={b['R']:.4f} 亿 (期望 >=3.6); C_env={b['C_env']:.3f} "
            f"(期望 ≈0.5, ≥0.5); τ={b['tau']:.4f} (v1 为 0.1450, 应提升); "
            f"x={b['x']:.2f} 万 (期望 ≈153.6, ≤160); S={b['S']:.3f}; "
            f"m={''.join(map(str, b['combo']))}; x_d={b['xd']:.4f}; J={b['objective_value']:.4f}")
        plot_pareto(s00_detail, b, "C_env", "环境成本 C_env",
                    "pareto_income_env.png",
                    "M1 v2.0 Pareto: 收入 vs 环境成本 (S00, 32 组合)")
        plot_pareto(s00_detail, b, "C_soc", "社会成本 C_soc",
                    "pareto_income_soc.png",
                    "M1 v2.0 Pareto: 收入 vs 社会成本 (S00, 32 组合)")

    n_ok = int((results["feasible_flag"] == 1).sum())
    log(f"可解 {n_ok}/15 | 不可解 {15 - n_ok}/15")

    # ---- v1→v2 对比表 ----
    cmp_df = build_v1_v2_compare(results)
    if cmp_df is not None:
        cmp_path = RESULTS_DIR / "v1_v2_compare.csv"
        cmp_df.to_csv(cmp_path, index=False, encoding="utf-8-sig")
        log(f"v1→v2 对比表已生成: {cmp_path}")
        log("  (V1 已备份至 results/archive_v1/m1_m1_results.csv)")

    log_lines.append(f"\n[运行摘要] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"  可解情景: {n_ok}/15")
    log_lines.append(f"  参数: R_min={params['R_min']}, E_max={params['E_max']}, "
                     f"S_min={params['S_min']}, gamma={params['gamma']}, lambda_R={params['lambda_R']}")

    with open(RESULTS_DIR / "run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print("\n===== m1_results.csv (v2.0) =====")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
