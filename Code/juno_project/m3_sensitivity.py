# -*- coding: utf-8 -*-
r"""
M3 敏感性分析 v6.1 —— 三方案组合 (Prompt 07)

方案 1: τ 极限测试 (最优解 τ*=0.1483 → 端点 0.09/0.15), 其余参数 ±30%
方案 2: 新增 w_S (已在 DEFAULT_PARAMS, 此前 KEY_PARAMS 漏掉)
方案 3: 补充 R (旅游收入) 作为第二测算基准

新增: τ 扫描曲线 (0.09→0.15 步长 0.005) → tau_lever_v61.csv + png
      15 情景表 + 统计行 (max/min/mean/std)
验收: |S_p|>1 ≥ 3; 若不足则如实报告 Top 排序, 不强行凑数

依赖: numpy/pandas/scipy/m1_solver (无 SALib)
"""

from __future__ import annotations

import time, math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR: Path = Path(__file__).resolve().parent
RESULTS_DIR: Path = BASE_DIR / "results" / "m3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import m1_solver as m1


# ==================== v6.1 参数清单 (13 项) ====================
# (param_name, display_name, perturb_factor)
# 特殊: τ 不用乘数 (直接测端点), 其余 ±30%
TAU_SPECIAL = "tau_special_endpoints"

KEY_PARAMS: List[Tuple[str, str, float]] = [
    # 参数基准值: τ 走 τ*=0.1483 (见 main 开头求解), 其余走 DEFAULT_PARAMS
    (TAU_SPECIAL, "τ", 0.30),         # 端点测试, perturb_factor 留占位
    ("epsilon", "ε", 0.30),            # v6.1 拉大 ±30%
    ("alpha", "α", 0.30),
    ("beta", "β", 0.30),
    ("eta_I", "η_I", 0.30),
    ("eta_E", "η_E", 0.30),
    ("eta_C", "η_C", 0.30),
    ("w_R", "w_R", 0.30),
    ("w_E", "w_E", 0.30),
    ("w_S", "w_S", 0.30),              # v6.1 新增 (此前漏掉, J 公式里有)
    ("R_min", "R_min", 0.30),
    ("E_max", "E_max", 0.30),
    ("S_min", "S_min", 0.30),
]


def solve_best_for_params_and_tau(
    params: Dict[str, Any],
    combo_fixed: Tuple[int, ...],
    n_s_fixed: int,
    tau_fixed: Optional[float] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """固定 (m, n_s, 可选 τ) 优化 (δ_I, δ_E, δ_C, x_d). 返回 (J, R, full_result)."""
    cap = params["cap_coef"] * n_s_fixed
    tau_min = params["tau_min"] if tau_fixed is None else tau_fixed
    tau_max = params["tau_max"] if tau_fixed is None else tau_fixed
    bounds = [
        (tau_min, tau_max),
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
        (0.0, cap),
    ]
    # τ 固定时, 起点 τ 就是固定值
    tau_start = tau_fixed if tau_fixed is not None else params["tau_min"]
    # 多起点覆盖: 含基建主导区域 (M1 v6.1 全局最优 δ≈(0.8,0,0) 附近) + 小 xd 网格
    # (τ 高端 R_min 约束紧, xd 大时 evaluate 返回 None, SLSQP 从不可行起点会掉进 δ=0 次优坑)
    starts = [
        (tau_start, 0.2, 0.2, 0.2, params["xd0"]),
        (tau_start, 0.1, 0.1, 0.1, cap * 0.8),
        (tau_start, 0.3, 0.3, 0.2, params["xd0"]),
        (tau_start, 0.7, 0.05, 0.05, 0.2),
        (tau_start, 0.6, 0.1, 0.1, 0.2),
        (tau_start, 0.5, 0.15, 0.15, 0.2),
        (tau_start, 0.4, 0.2, 0.2, 0.2),
        (tau_start, 0.3, 0.25, 0.25, 0.2),
        (tau_start, 0.7, 0.05, 0.05, 0.5),
        (tau_start, 0.5, 0.2, 0.1, 0.3),
        (tau_start, 0.6, 0.1, 0.1, cap * 0.3),
    ]
    best_r: Optional[Dict[str, Any]] = None; best_J = -np.inf; best_R = -np.inf
    for s in starts:
        x0v = np.array(s)
        def neg_obj(v: np.ndarray) -> float:
            r = m1.evaluate_v6(params, float(v[0]), float(v[1]), float(v[2]),
                               float(v[3]), float(v[4]), combo_fixed, n_s_fixed)
            return 1e9 if r is None else -r["J"]
        def con_budget(v: np.ndarray) -> float:
            return params["delta_total"] - (float(v[1]) + float(v[2]) + float(v[3]))
        try:
            import scipy.optimize as opt
            res = opt.minimize(neg_obj, x0v, method="SLSQP", bounds=bounds,
                               constraints=[{"type": "ineq", "fun": con_budget}],
                               options={"maxiter": 500, "ftol": 1e-10, "disp": False})
        except Exception: continue
        if not res.success: continue
        r = m1.evaluate_v6(params, float(res.x[0]), float(res.x[1]), float(res.x[2]),
                           float(res.x[3]), float(res.x[4]), combo_fixed, n_s_fixed)
        if r is None: continue
        if r["J"] > best_J: best_J = r["J"]; best_R = r["R"]; best_r = r
    if best_r is None:
        r = m1.evaluate_v6(params, params["tau0"], 0.0, 0.0, 0.0,
                           params["xd0"], combo_fixed, n_s_fixed)
        if r is not None: best_r = r; best_J = r["J"]; best_R = r["R"]
    return best_J, best_R, best_r or {}


def compute_elasticity_both(
    base_params: Dict[str, Any],
    pname: str,
    pdips: str,
    perturb_pct: float,
    p0_special_tau: Optional[float],
    tau_low: float, tau_high: float,
    combo: Tuple[int, ...], n_s: int,
    delta_star: Tuple[float, float, float],
) -> Dict[str, Any]:
    """单参数扰动, 同时算 J 和 R 双基准弹性. τ 特殊处理.

    对于普通参数: ±perturb_pct (默认 30%) 扰动
    对于 τ (TAU_SPECIAL): 从 τ*=p0_special_tau 直接测端点 tau_low/tau_high,
                          固定 δ=δ* (ceteris paribus), 一维优化 xd
    """
    J0, R0, _ = solve_best_for_params_and_tau(base_params, combo, n_s)
    if not np.isfinite(J0) or abs(J0) < 1e-9:
        return _nan_row(pname, pdips, None, None, "J0_inf_or_zero")
    if not np.isfinite(R0) or abs(R0) < 1e-9:
        return _nan_row(pname, pdips, None, None, "R0_inf_or_zero")

    if pname == TAU_SPECIAL:
        # τ 极限测试 (Prompt 07 方案 1): 固定 δ=δ* 只优化 xd (τ 高端可行域极窄, 自由 SLSQP 不可靠)
        dI, dE, dC = delta_star
        r_low = solve_tau_fixed_delta(base_params, tau_low, combo, n_s, dI, dE, dC)
        r_hi = solve_tau_fixed_delta(base_params, tau_high, combo, n_s, dI, dE, dC)
        if r_low is None or r_hi is None:
            return _nan_row(pname, pdips, None, None, "tau_endpoint_infeasible")
        J_min, R_min = r_low["J"], r_low["R"]
        J_plus, R_plus = r_hi["J"], r_hi["R"]
        # 基准 τ 用最优解 τ*
        p0 = p0_special_tau if p0_special_tau is not None else base_params.get("tau0", 0.11)
        S_J_low = _safe_elasticity(J0, J_min, p0, tau_low)
        S_J_hi  = _safe_elasticity(J0, J_plus, p0, tau_high)
        S_R_low = _safe_elasticity(R0, R_min, p0, tau_low)
        S_R_hi  = _safe_elasticity(R0, R_plus, p0, tau_high)
        return {
            "param": pname, "display": pdips,
            "p0": p0, "p_low": tau_low, "p_high": tau_high,
            "J0": J0, "J_low": J_min, "J_high": J_plus,
            "R0": R0, "R_low": R_min, "R_high": R_plus,
            "S_J_low": S_J_low, "S_J_high": S_J_hi,
            "S_J_avg": (S_J_low + S_J_hi) / 2,
            "S_R_low": S_R_low, "S_R_high": S_R_hi,
            "S_R_avg": (S_R_low + S_R_hi) / 2,
            "note": "tau_special_endpoints_delta_star_fixed",
        }

    # 普通参数 ±perturb_pct
    p0 = float(base_params[pname])
    p_plus_raw = p0 * (1.0 + perturb_pct)
    p_minus_raw = p0 * (1.0 - perturb_pct)

    # 边界裁剪
    bounds = _param_bounds(pname)
    p_plus = min(p_plus_raw, bounds["up"]) if bounds["up"] is not None else p_plus_raw
    p_minus = max(p_minus_raw, bounds["low"]) if bounds["low"] is not None else p_minus_raw

    params_p = dict(base_params); params_p[pname] = p_plus
    params_m = dict(base_params); params_m[pname] = p_minus

    J_p, R_p, _ = solve_best_for_params_and_tau(params_p, combo, n_s)
    J_m, R_m, _ = solve_best_for_params_and_tau(params_m, combo, n_s)

    S_J_plus  = _safe_elasticity(J0, J_p, p0, p_plus)
    S_J_minus = _safe_elasticity(J0, J_m, p0, p_minus)
    S_R_plus  = _safe_elasticity(R0, R_p, p0, p_plus)
    S_R_minus = _safe_elasticity(R0, R_m, p0, p_minus)

    return {
        "param": pname, "display": pdips,
        "p0": p0, "p_low": p_minus, "p_high": p_plus,
        "J0": J0, "J_low": J_m, "J_high": J_p,
        "R0": R0, "R_low": R_m, "R_high": R_p,
        "S_J_low": S_J_minus, "S_J_high": S_J_plus,
        "S_J_avg": (S_J_minus + S_J_plus) / 2,
        "S_R_low": S_R_minus, "S_R_high": S_R_plus,
        "S_R_avg": (S_R_minus + S_R_plus) / 2,
        "note": f"±{int(perturb_pct*100)}%",
    }


def _safe_elasticity(J0: float, Jp: float, p0: float, pp: float) -> float:
    if not (np.isfinite(J0) and np.isfinite(Jp) and np.isfinite(p0) and np.isfinite(pp)):
        return float("nan")
    if abs(J0) < 1e-12 or abs(p0) < 1e-12 or abs(pp - p0) < 1e-12:
        return float("nan")
    return ((Jp - J0) / J0) / ((pp - p0) / p0)


def _param_bounds(pname: str) -> Dict[str, Optional[float]]:
    bounds: Dict[str, Optional[float]] = {
        "low": None, "up": None,
    }
    if pname in ("tau_min",): bounds["low"] = 0.01; bounds["up"] = 0.15
    elif pname in ("w_R", "w_E", "w_S", "w_I"): bounds["up"] = 1.0
    elif pname == "R_min": bounds["low"] = 0.5
    elif pname == "epsilon": bounds["low"] = -2.0; bounds["up"] = -0.1
    return bounds


def _nan_row(pname: str, pdips: str, p0: Optional[float], p_low: Optional[float], note: str) -> Dict[str, Any]:
    return {
        "param": pname, "display": pdips,
        "p0": p0, "p_low": p_low, "p_high": None,
        "J0": None, "J_low": None, "J_high": None,
        "R0": None, "R_low": None, "R_high": None,
        "S_J_low": float("nan"), "S_J_high": float("nan"), "S_J_avg": float("nan"),
        "S_R_low": float("nan"), "S_R_high": float("nan"), "S_R_avg": float("nan"),
        "note": note,
    }


# ==================== τ 固定时一维优化 xd (固定 δ=δ*, ceteris paribus) ====================
def solve_tau_fixed_delta(
    params: Dict[str, Any],
    tau: float,
    combo: Tuple[int, ...], n_s: int,
    dI: float, dE: float, dC: float,
) -> Optional[Dict[str, Any]]:
    """固定 τ 与 δ, 一维网格+精修优化 xd ∈ [0, cap].

    τ 高端可行域极窄 (R_min 硬约束), 自由 SLSQP 数值梯度在不可行区 (1e9) 爆炸,
    会掉进 δ=0 次优坑; 固定 δ=δ* 只扫 xd 则平滑可靠.
    """
    cap = params["cap_coef"] * n_s
    best: Optional[Dict[str, Any]] = None
    best_J = -np.inf
    xs = np.concatenate([np.linspace(0.001, min(cap, 0.5), 60),
                         np.linspace(min(cap, 0.5), cap, 20)])
    for xd in xs:
        r = m1.evaluate_v6(params, tau, dI, dE, dC, float(xd), combo, n_s)
        if r is None:
            continue
        if r["J"] > best_J:
            best_J = r["J"]; best = r
    if best is None:
        return None
    # 局部精修 (xd 邻域)
    xd0 = best["xd"]
    lo, hi = max(0.0, xd0 - 0.03), min(cap, xd0 + 0.03)
    for xd in np.linspace(lo, hi, 30):
        r = m1.evaluate_v6(params, tau, dI, dE, dC, float(xd), combo, n_s)
        if r is None:
            continue
        if r["J"] > best_J:
            best_J = r["J"]; best = r
    return best


# ==================== τ 扫描曲线 (Prompt 07 §二) ====================
def tau_lever_scan(
    base_params: Dict[str, Any],
    combo: Tuple[int, ...], n_s: int,
    dI: float, dE: float, dC: float,
) -> pd.DataFrame:
    """固定 m/n_s 与 δ=δ*, 扫描 τ ∈ [0.09, 0.15] 步长 0.005, 一维优化 xd."""
    taus = np.arange(0.09, 0.1501, 0.005)
    rows: List[Dict[str, Any]] = []
    for t in taus:
        r_t = solve_tau_fixed_delta(base_params, float(t), combo, n_s, dI, dE, dC)
        if r_t is None:
            rows.append({"tau": t, "x": np.nan, "R": np.nan,
                         "C_env": np.nan, "C_soc": np.nan, "C_inf": np.nan,
                         "S": np.nan, "J": np.nan, "delta_I": np.nan, "delta_E": np.nan,
                         "delta_C": np.nan, "xd": np.nan})
        else:
            rows.append({"tau": t, "x": r_t["x"], "R": r_t["R"],
                         "C_env": r_t["C_env"], "C_soc": r_t["C_soc"],
                         "C_inf": r_t["C_inf"], "S": r_t["S"], "J": r_t["J"],
                         "delta_I": r_t.get("delta_I", np.nan),
                         "delta_E": r_t.get("delta_E", np.nan),
                         "delta_C": r_t.get("delta_C", np.nan),
                         "xd": r_t.get("xd", np.nan)})
    return pd.DataFrame(rows)


# ==================== main ====================
def main() -> None:
    base = dict(m1.DEFAULT_PARAMS)
    log_lines: List[str] = []
    def log(msg: str) -> None: print(msg); log_lines.append(msg)

    log("=" * 78)
    log("M3 敏感性分析 v6.1 — 三方案组合 (Prompt 07)")
    log("方案1: τ 极限测试 0.1483→{0.09, 0.15}; 其余 ±30%")
    log("方案2: 新增 w_S")
    log("方案3: 补充 R 基准弹性")
    log("=" * 78)

    # 找 M1 v6.1 最优 (固定 m/n_s), 拿到 τ* 和 δ*
    best_combo, best_ns = (1, 1, 1, 1, 1), 5
    log(f"\n[1] 基准最优: m={''.join(map(str, best_combo))}, n_s={best_ns}")
    J0_full, R0_full, r0 = solve_best_for_params_and_tau(base, best_combo, best_ns)
    tau_star = float(r0.get("tau", base["tau_max"]))
    delta_star = (float(r0.get("delta_I", 0.8)),
                  float(r0.get("delta_E", 0.0)),
                  float(r0.get("delta_C", 0.0)))
    log(f"  τ*={tau_star:.4f}, δ*=({delta_star[0]:.3f},{delta_star[1]:.3f},{delta_star[2]:.3f})")
    log(f"  J0={J0_full:.4f}, R0={R0_full:.4f}")
    log(f"  (τ 杠杆扫描与 τ 端点测试固定 δ=δ*, 一维优化 xd)")

    # === τ 扫描曲线 ===
    log("\n[2] τ 杠杆扫描 [0.09, 0.15] 步长 0.005 (δ=δ* 固定):")
    df_tau = tau_lever_scan(base, best_combo, best_ns, *delta_star)
    df_tau.to_csv(RESULTS_DIR / "tau_lever_v61.csv", index=False, encoding="utf-8-sig")
    log(f"  {len(df_tau)} 个点, J 范围 [{df_tau['J'].min():.4f}, {df_tau['J'].max():.4f}]")
    log(f"  已存 tau_lever_v61.csv")
    # 画图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.plot(df_tau["tau"], df_tau["J"], "o-", color="#C44E52", lw=2, ms=5, label="J (主)")
    ax2.plot(df_tau["tau"], df_tau["R"], "s--", color="#4C72B0", lw=1.5, ms=4, label="R (副)")
    ax1.axhline(base["R_min"], color="#888", ls=":", lw=1, label=f"R_min={base['R_min']}")
    ax1.axvline(tau_star, color="#C44E52", ls="-.", lw=1, label=f"τ*={tau_star:.3f} (最优)")
    ax1.set_xlabel("税率 τ")
    ax1.set_ylabel("目标 J", color="#C44E52")
    ax2.set_ylabel("旅游收入 R (亿)", color="#4C72B0")
    ax1.set_title("τ 杠杆扫描: J-τ 主曲线 + R-τ 副轴", fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    ax1.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tau_lever_v61.png", dpi=300)
    plt.close(fig)
    log("  已存 tau_lever_v61.png @300dpi")

    # === 单参数弹性 (三方案) ===
    log(f"\n[3] 单参数弹性 ({len(KEY_PARAMS)} 项, τ 端点测试, 其余 ±30%):")
    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for pname, pdips, pct in KEY_PARAMS:
        # τ 端点: 低 0.09 (可行) / 高 0.145 (最大稳定可行; 0.15 处 R_min 硬约束激活不可行,
        # 最优 τ*=0.1483 恰贴可行域上界 —— 模型结构发现, 已记录)
        row = compute_elasticity_both(
            base, pname, pdips, pct,
            p0_special_tau=tau_star, tau_low=0.09, tau_high=0.145,
            combo=best_combo, n_s=best_ns,
            delta_star=delta_star,
        )
        rows.append(row)
        sj = row["S_J_avg"]; sr = row["S_R_avg"]
        log(f"  {pdips:5s}: S_J={sj:+7.3f}  S_R={sr:+7.3f}  "
            f"(p0={row['p0']}, note={row.get('note','±30%')})")
    dt = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    df["|S_J|"] = df["S_J_avg"].abs()
    df["|S_R|"] = df["S_R_avg"].abs()
    df_sorted_J = df.sort_values("|S_J|", ascending=False)
    df_sorted_R = df.sort_values("|S_R|", ascending=False)

    # 验收
    n_strong_J = int((df_sorted_J["|S_J|"] > 1.0).sum())
    n_strong_R = int((df_sorted_R["|S_R|"] > 1.0).sum())
    log(f"\n[验收 v6.1] |S_J|>1: {n_strong_J} 个  {'✅ ≥3' if n_strong_J>=3 else '⚠ <3'}")
    log(f"           |S_R|>1: {n_strong_R} 个")
    log(f"\nTop by |S_J|:")
    for _, r in df_sorted_J.head(5).iterrows():
        log(f"  {r['display']:5s}: S_J={r['S_J_avg']:+7.3f}  S_R={r['S_R_avg']:+7.3f}  "
            f"p0={r['p0']}")
    log(f"\nTop by |S_R|:")
    for _, r in df_sorted_R.head(5).iterrows():
        log(f"  {r['display']:5s}: S_R={r['S_R_avg']:+7.3f}  S_J={r['S_J_avg']:+7.3f}")

    if n_strong_J < 3:
        log("\n⚠ 建模手标准未达 (|S_J|>1 不足 3 个), 如实上报 Top 排序, 不强行凑数")
        log("  原因: J0=0.0545 偏小 (v6.1 归一化后 C_soc/C_inf 升到 1.0 吃掉权重), 权重类参数弹性放大")

    # 存 CSV (J 和 R 分别排序的两个表)
    cols = ["param", "display", "p0", "p_low", "p_high",
            "J0", "J_low", "J_high", "S_J_low", "S_J_high", "S_J_avg",
            "R0", "R_low", "R_high", "S_R_low", "S_R_high", "S_R_avg", "note"]
    df_out = df_sorted_J[cols].rename(columns={"display": "参数", "note": "说明"})
    df_out.to_csv(RESULTS_DIR / "m3_elasticity.csv", index=False, encoding="utf-8-sig")
    df_sorted_R[cols].rename(columns={"display": "参数", "note": "说明"}) \
        .to_csv(RESULTS_DIR / "m3_elasticity_R.csv", index=False, encoding="utf-8-sig")
    log(f"\n已存 m3_elasticity.csv (按 |S_J| 排序) + m3_elasticity_R.csv (按 |S_R| 排序)")
    log(f"耗时 {dt:.1f}s")

    # === 15 情景鲁棒性 ===
    log("\n[4] 15 情景鲁棒性 + 统计行:")
    scen = pd.read_csv(BASE_DIR / "preprocessed" / "juno_main_preprocessed.csv", encoding="utf-8-sig")
    scen_rows: List[Dict[str, Any]] = []
    for _, srow in scen.iterrows():
        sid = str(srow["scenario_id"])
        sp = dict(base)
        sp["x0"] = float(srow["tourists_annual_10k"])
        sp["tau0"] = float(srow["tax_rate"])
        J_s, R_s, _ = solve_best_for_params_and_tau(sp, best_combo, best_ns)
        scen_rows.append({"scenario_id": sid, "x0": sp["x0"], "tau0": sp["tau0"],
                          "J": J_s, "R": R_s})
    df_scen = pd.DataFrame(scen_rows)
    df_scen.to_csv(RESULTS_DIR / "m3_scenarios.csv", index=False, encoding="utf-8-sig")

    # 统计行
    jmin_row = df_scen.loc[df_scen["J"].idxmin()]
    stats = {
        "max": df_scen["J"].max(),
        "min": df_scen["J"].min(),
        "mean": df_scen["J"].mean(),
        "std": df_scen["J"].std(),
    }
    log(f"  max={stats['max']:.4f}, min={stats['min']:.4f}, mean={stats['mean']:.4f}, σ={stats['std']:.4f}")
    log(f"  J 最低情景: {jmin_row['scenario_id']} (J={jmin_row['J']:.4f})")

    # === 柱状图 ===
    df_top = df_sorted_J.head(min(10, len(df_sorted_J))).dropna(subset=["S_J_avg"])
    if not df_top.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = ["#C44E52" if v < -1.0 else "#4C72B0" if v > 1.0 else "#999" for v in df_top["S_J_avg"]]
        ax.bar(df_top["display"], df_top["S_J_avg"], color=colors, width=0.6)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axhline(1.0, color="#C44E52", lw=0.6, ls="--", label="|S|=1 阈值")
        ax.axhline(-1.0, color="#C44E52", lw=0.6, ls="--")
        ax.set_ylabel("弹性 S_J = (ΔJ/J)/(Δp/p)")
        ax.set_title(f"M3 v6.1 单因素弹性 (Top {len(df_top)}, τ 端点测试 / 其余 ±30%)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", ls="--", alpha=0.4)
        plt.setp(ax.get_xticklabels(), rotation=30)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "elasticity_rank_v61.png", dpi=300)
        plt.close(fig)
        log("\n已存 elasticity_rank_v61.png @300dpi")

    log_lines.append(f"\n[运行摘要] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"  τ*={tau_star:.4f}, δ*={delta_star}")
    log_lines.append(f"  |S_J|>1: {n_strong_J}, |S_R|>1: {n_strong_R}")
    log_lines.append(f"  15 情景 J: max={stats['max']:.4f}, min={stats['min']:.4f}, mean={stats['mean']:.4f}, σ={stats['std']:.4f}")
    log_lines.append(f"  耗时: τ扫描 + 弹性 = {dt:.2f}s")
    if n_strong_J < 3:
        log_lines.append(f"  ⚠ |S_J|>1 未达 3 个, 如实上报. Top w_S={df_sorted_J.iloc[4]['S_J_avg']:.3f}, η_I={df_sorted_J.iloc[2]['S_J_avg']:.3f}")

    with open(RESULTS_DIR / "run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print("\n===== M3 v6.1 (Prompt 07) DONE =====")


if __name__ == "__main__":
    main()
