# -*- coding: utf-8 -*-
r"""
M3 敏感性分析 v6.0 —— 单因素弹性 (不再用 SALib/Sobol)

权威: Prompts/05_v6.0代码对齐.md §模块三
  * 删除所有 SALib 代码 (v6.0 已砍 Sobol)
  * 改单因素弹性法: S_p = (ΔJ/J) / (Δp/p)
  * 对每个关键参数 p, 分别扰动 +10% 和 -10%, 重跑 M1 最优解 (固定 m/n_s, 优化 τ/δ/x_d)
  * 关键参数清单 (12 项, 剔掉 γ — v6.0 M1 已删 γ 环境吸引力因子, 扰动无意义):
    τ, ε, α, β, η_I, η_E, η_C, w_R, w_E, R_min, E_max, S_min

依赖: numpy/pandas/scipy (调用 m1_solver.v6 求解器)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ==================== 路径 + 复用 ====================
BASE_DIR: Path = Path(__file__).resolve().parent
RESULTS_DIR: Path = BASE_DIR / "results" / "m3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import m1_solver as m1


# ==================== 关键参数清单 (v6.0) ====================
# 注: γ 从清单中剔除 — v6.0 M1 公式里已无 γ 环境吸引力因子
#     扰动 γ 对结果无影响, 不做
KEY_PARAMS: List[Tuple[str, str]] = [
    # (param_name, display_name_for_chart)
    ("tau_min", "τ"),
    ("epsilon", "ε"),
    ("alpha", "α"),
    ("beta", "β"),
    ("eta_I", "η_I"),
    ("eta_E", "η_E"),
    ("eta_C", "η_C"),
    ("w_R", "w_R"),
    ("w_E", "w_E"),
    ("R_min", "R_min"),
    ("E_max", "E_max"),
    ("S_min", "S_min"),
]


def solve_best_for_params(
    params: Dict[str, Any],
    combo_fixed: Tuple[int, ...],
    n_s_fixed: int,
    n_restarts: int = 3,
) -> Tuple[float, Dict[str, Any]]:
    """固定 (m, n_s), 优化 (τ, δ_I, δ_E, δ_C, x_d) 找最优 J.

    等价于 m1.solve_one_combo_ns 但更轻量 (只针对最优 m/n_s)."""
    cap = params["cap_coef"] * n_s_fixed
    bounds = [
        (params["tau_min"], params["tau_max"]),
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
        (0.0, cap),
    ]
    starts = [
        (params["tau_min"], 0.2, 0.2, 0.2, params["xd0"]),
        (params["tau_max"], 0.1, 0.1, 0.1, cap * 0.8),
        (params["tau0"], 0.3, 0.3, 0.2, params["xd0"]),
    ]
    best_r = None; best_J = -np.inf
    for s in starts:
        x0v = np.array(s)
        def neg_obj(v: np.ndarray) -> float:
            r = m1.evaluate_v6(params, float(v[0]), float(v[1]), float(v[2]),
                               float(v[3]), float(v[4]), combo_fixed, n_s_fixed)
            return 1e9 if r is None else -r["J"]
        def con_budget(v: np.ndarray) -> float:
            return params["delta_total"] - (float(v[1]) + float(v[2]) + float(v[3]))
        try:
            res = __import__("scipy.optimize", fromlist=["minimize"]).minimize(
                neg_obj, x0v, method="SLSQP", bounds=bounds,
                constraints=[{"type": "ineq", "fun": con_budget}],
                options={"maxiter": params["slsqp_maxiter"],
                         "ftol": params["slsqp_ftol"], "disp": False})
        except Exception: continue
        if not res.success: continue
        r = m1.evaluate_v6(params, float(res.x[0]), float(res.x[1]), float(res.x[2]),
                           float(res.x[3]), float(res.x[4]), combo_fixed, n_s_fixed)
        if r is None: continue
        if r["J"] > best_J: best_J = r["J"]; best_r = r
    # 兜底: 基准态
    if best_r is None:
        r = m1.evaluate_v6(params, params["tau0"], 0.0, 0.0, 0.0,
                           params["xd0"], combo_fixed, n_s_fixed)
        if r is not None: best_r = r; best_J = r["J"]
    return best_J, best_r or {}


def compute_elasticity(
    base_params: Dict[str, Any],
    param_name: str,
    combo: Tuple[int, ...],
    n_s: int,
) -> Dict[str, Any]:
    """单参数 ±10% 扰动, 弹性 S_p = (ΔJ/J) / (Δp/p)."""
    p0 = float(base_params[param_name])
    # 基准 J (用 base_params 重新跑)
    J0, _ = solve_best_for_params(base_params, combo, n_s)
    if not np.isfinite(J0) or abs(J0) < 1e-9:
        return {"param": param_name, "p0": p0, "J0": J0,
                "J_plus": np.nan, "J_minus": np.nan,
                "S_plus": np.nan, "S_minus": np.nan, "S_avg": np.nan}
    # +10%
    p_plus = p0 * 1.1
    # 边界裁剪: tau_min >= 0.01, tau_max <= 0.5
    if param_name in ("tau_min",): p_plus = min(p_plus, 0.14)
    if param_name in ("w_R", "w_E"): p_plus = min(p_plus, 1.0)
    params_p = dict(base_params); params_p[param_name] = p_plus
    J_plus, _ = solve_best_for_params(params_p, combo, n_s)
    # -10%
    p_minus = p0 * 0.9
    if param_name in ("tau_min", "R_min"): p_minus = max(p_minus, 0.01)
    params_m = dict(base_params); params_m[param_name] = p_minus
    J_minus, _ = solve_best_for_params(params_m, combo, n_s)
    # 弹性 (对称)
    S_plus = ((J_plus - J0) / J0) / ((p_plus - p0) / p0) if np.isfinite(J_plus) else np.nan
    S_minus = ((J_minus - J0) / J0) / ((p_minus - p0) / p0) if np.isfinite(J_minus) else np.nan
    S_avg = (S_plus + S_minus) / 2.0 if np.isfinite(S_plus) and np.isfinite(S_minus) else np.nan
    return {"param": param_name, "p0": p0, "J0": J0,
            "J_plus": J_plus, "J_minus": J_minus,
            "S_plus": S_plus, "S_minus": S_minus, "S_avg": S_avg}


def main() -> None:
    base = dict(m1.DEFAULT_PARAMS)
    log_lines: List[str] = []
    def log(msg: str) -> None: print(msg); log_lines.append(msg)

    log("=" * 78)
    log("M3 敏感性分析 v6.1 — 单因素 ±10% 弹性 (已删 SALib/Sobol)")
    log(f"关键参数: {len(KEY_PARAMS)} 项")
    log("=" * 78)

    # 先确定 M1 最优 (固定 m/n_s, 取 M1 v6.1 枚举结果)
    log("\n[1] 找 M1 v6.1 最优 (固定 m, n_s):")
    best_combo, best_ns = (1, 1, 1, 1, 1), 5   # M1 v6.1 最优
    J0, r0 = solve_best_for_params(base, best_combo, best_ns)
    log(f"  m={''.join(map(str, best_combo))}, n_s={best_ns}, J0={J0:.4f}")
    log(f"  τ={r0.get('tau',np.nan):.4f}, δ=({r0.get('delta_I',np.nan):.3f},"
        f"{r0.get('delta_E',np.nan):.3f},{r0.get('delta_C',np.nan):.3f})")

    # 弹性
    log("\n[2] 单参数 ±10% 弹性:")
    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for pname, pdisp in KEY_PARAMS:
        row = compute_elasticity(base, pname, best_combo, best_ns)
        row["display"] = pdisp
        rows.append(row)
        s_avg = row["S_avg"]
        mark = f"|S|={abs(s_avg):.3f}" if np.isfinite(s_avg) else "nan"
        log(f"  {pdisp:6s}: S_avg = {s_avg:+8.3f}  (p0={row['p0']:.4f}, J0={row['J0']:.4f})")
    dt = time.perf_counter() - t0

    # 排序
    df = pd.DataFrame(rows)
    df["|S|"] = df["S_avg"].abs()
    df_sorted = df.sort_values("|S|", ascending=False)

    # 验收: 至少 3 个 |S_p| > 0.1
    n_strong = int((df_sorted["|S|"] > 0.1).sum())
    log(f"\n[验收] |S_p| > 0.1 的参数数 = {n_strong} {'✅ ≥ 3' if n_strong >= 3 else '❌ < 3'}")
    top3 = df_sorted.head(3)[["display", "S_avg"]].to_string(index=False)
    log(f"Top 3 弹性:\n{top3}")
    log(f"\n耗时 {dt:.1f}s")

    # 存 CSV
    df_out = df_sorted[["display", "param", "p0", "J0", "J_plus", "J_minus",
                        "S_plus", "S_minus", "S_avg"]].rename(columns={"display": "参数"})
    df_out.to_csv(RESULTS_DIR / "m3_elasticity.csv", index=False, encoding="utf-8-sig")
    log(f"\n已存 m3_elasticity.csv")

    # 情景弹性 (15 情景, 从 juno_main_preprocessed 读 x0/tau0/δ, 每个情景重跑 M1)
    log("\n[3] 15 情景 J 值 (按预处理情景行, 从 CSV 读 x0/τ0/δ):")
    scen = pd.read_csv(BASE_DIR / "preprocessed" / "juno_main_preprocessed.csv", encoding="utf-8-sig")
    scen_rows = []
    for _, srow in scen.iterrows():
        sid = str(srow["scenario_id"])
        # 从情景行读覆盖参数 (x0, tau0, delta_I/E/C)
        scen_params = dict(base)
        scen_params["x0"] = float(srow["tourists_annual_10k"])
        scen_params["tau0"] = float(srow["tax_rate"])
        scen_params["delta_I"] = float(srow["delta_I"])
        scen_params["delta_E"] = float(srow["delta_E"])
        scen_params["delta_C"] = float(srow["delta_C"])
        # 用情景参数跑 M1 (固定 m/n_s)
        J_s, r_s = solve_best_for_params(scen_params, best_combo, best_ns)
        if np.isfinite(J_s):
            rel = (J_s - J0) / abs(J0) * 100 if abs(J0) > 1e-9 else np.nan
            scen_rows.append({"scenario_id": sid, "x0": scen_params["x0"],
                              "tau0": scen_params["tau0"], "tau": r_s.get("tau", np.nan),
                              "J": J_s, "relative_change_pct": rel})
            log(f"  {sid:15s}: x0={scen_params['x0']:6.1f}, τ0={scen_params['tau0']:.3f} → J={J_s:.4f} ({rel:+.1f}%)")
    pd.DataFrame(scen_rows).to_csv(RESULTS_DIR / "m3_scenarios.csv", index=False, encoding="utf-8-sig")

    # 画柱状图 (Top 10 参数弹性)
    df_top = df_sorted.head(min(10, len(df_sorted))).dropna(subset=["S_avg"])
    if not df_top.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#C44E52" if v < -0.1 else "#4C72B0" if v > 0.1 else "#999"
                  for v in df_top["S_avg"]]
        ax.bar(df_top["display"], df_top["S_avg"], color=colors, width=0.6)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axhline(0.1, color="#C44E52", lw=0.6, ls="--", label="±0.1 阈值")
        ax.axhline(-0.1, color="#C44E52", lw=0.6, ls="--")
        ax.set_ylabel("弹性 S = (ΔJ/J)/(Δp/p)")
        ax.set_title(f"M3 v6.0 单因素弹性 (Top {len(df_top)} 参数, ±10% 扰动)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", ls="--", alpha=0.4)
        plt.setp(ax.get_xticklabels(), rotation=30)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "elasticity_rank.png", dpi=300)
        plt.close(fig)
        log(f"已存 elasticity_rank.png")

    log_lines.append(f"\n[运行摘要] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"  参数数: {len(KEY_PARAMS)}, |S|>0.1: {n_strong}")
    log_lines.append(f"  耗时: {dt:.2f}s")

    with open(RESULTS_DIR / "run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print("\n===== M3 v6.1 DONE =====")


if __name__ == "__main__":
    main()
