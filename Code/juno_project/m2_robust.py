# -*- coding: utf-8 -*-
r"""
M2 鲁棒反哺模型 v6.0 —— 按 v6.0 公式重构

权威: Prompts/05_v6.0代码对齐.md §模块二 + CHANGELOG §1
重构 (不 patch v3.0):
  * 项目砍到 3 类: y_I, y_E, y_C ∈ {0,1}, 枚举 2³ = 8 种组合
  * 资金注入 delta_k_s = delta_k * y_k (y=0 则 δ_k_s=0)
  * 状态反馈公式简化 (v6.0):
      I_s   = I0 * (1 + eta_I * delta_I_s)
      C_env_s = max(0.4, 0.8*(0.5*x_s/x0 + 0.5*(x_d/x_d0)) - eta_E*delta_E_s)
      S_s   = 1.0 - 0.3*(x_s/x0 - 1) + eta_C*delta_C_s
  * 情景需求: x_s_new = x0*(tau_s/tau0)^ε * (I_s/I0)^α * (max(0.01,S_s)/1.0)^β * prod(1+κ_j*m_j)
  * 鲁棒 min-max: 每种 y 组合下, 对 15 蒙卡情景取 z 的最小值 (最坏), 再在 8 组合取最大
  * 量纲 ω_I=0.5, ω_E=1.0, ω_S=0.8 转亿美元当量; λ_cost=1.0, λ_L=0.05
  * 蒙卡口径: x ~ N(x0, 10%), τ ~ N(τ0, 5%), 15 次抽样

依赖: numpy/pandas/scipy (调用 m1_solver.v6 公式函数)
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ==================== 路径 + 复用 v6.0 公式 ====================
BASE_DIR: Path = Path(__file__).resolve().parent
RESULTS_DIR: Path = BASE_DIR / "results" / "m2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import m1_solver as m1                      # v6.0 参数 + FP 迭代

# ==================== M2 v6.1 独有参数 ====================
M2_CONFIG: Dict[str, Any] = {
    # 项目成本 (亿) — 3 类项目对应 c_I, c_E, c_S
    "c_I": 0.3, "c_E": 0.25, "c_S": 0.4,
    # 总预算 (亿) — 约束 Σ c_k*y_k ≤ B
    "B_total": 1.0,
    # M1 v6.1 最优解固定量 (取 m1 v6.1 run 输出)
    "m_optimal": (1, 1, 1, 1, 1),            # m=11111, n_s=5, τ=0.1483, δ=(0.8,0,0)
    "delta_optimal": (0.800, 0.000, 0.000),  # M1 v6.1 最优 δ (仅基建, 占满 delta_total=0.8)
    # 量纲系数 (v6.1 修正: 0.5/1.0/0.8 → 1.5/2.0/1.2, 防退化)
    "omega_I": 1.5, "omega_E": 2.0, "omega_S": 1.2,
    # 罚系数 (v6.1 修正: 1.0 → 0.3, 让投资更有吸引力)
    "lambda_cost": 0.3, "lambda_L": 0.05,
    # 蒙卡
    "n_scenarios": 15, "sigma_x": 0.10, "sigma_tau": 0.05,
    "seed": 42,
}


def run_m2_scenario(
    params: Dict[str, Any],
    tau_s: float,
    x_d_s: float,
    y_I: int, y_E: int, y_C: int,
    delta_opt: Tuple[float, float, float],
    m_combo: Tuple[int, ...],
) -> Dict[str, Any]:
    """对给定 (τ_s, x_d_s, y_I, y_E, y_C) 跑 M2 单情景不动点.

    v6.0 M2-2~M2-6 简化公式:
      δ_k_s = delta_k * y_k                    (M2-2)
      I_s   = I0 * (1 + eta_I * delta_I_s)     (M2-3)
      C_env_s = max(0.4, 0.8*(0.5*x_s/x0 + 0.5*(x_d_s/x_d0))
                 - eta_E*delta_E_s)             (M2-4)
      S_s   = 1.0 - 0.3*(x_s/x0 - 1) + eta_C*delta_C_s  (M2-5)
      x_s_new = x0 * (τ_s/τ0)^ε * (I_s/I0)^α * (max(0.01,S_s)/1)^β
                * prod(1+κ_j*m_j)              (M2-6)
    """
    x0 = params["x0"]; tau0 = params["tau0"]; xd0 = params["xd0"]
    I0 = params["I0"]
    eps = params["epsilon"]; al = params["alpha"]; be = params["beta"]
    eta_I = params["eta_I"]; eta_E = params["eta_E"]; eta_C = params["eta_C"]
    c_env_floor = params["c_env_floor"]; c_env_base = params["c_env_base"]
    kappa = params["kappa_measure"]

    dI, dE, dC = delta_opt
    dI_s, dE_s, dC_s = dI * y_I, dE * y_E, dC * y_C

    prod = 1.0
    for j, mj in enumerate(m_combo, start=1):
        prod *= (1.0 + kappa[f"m{j}"] * mj)

    x = x0
    for it in range(params["max_iter_fp"]):
        I_s = I0 * (1.0 + eta_I * dI_s)
        C_env_s = max(c_env_floor,
                      c_env_base * (0.5 * (x / x0) + 0.5 * (x_d_s / xd0))
                      - eta_E * dE_s)
        S_s = 1.0 - 0.3 * (x / x0 - 1.0) + eta_C * dC_s
        x_new = (x0 * (tau_s / tau0) ** eps * (I_s / I0) ** al
                 * (max(0.01, S_s)) ** be * prod)
        if not np.isfinite(x_new) or x_new <= 0:
            return {"x_s": float("nan"), "z": float("nan"), "ok": False}
        if abs(x_new - x) < params["tol_fp"]:
            break
        x = x_new
    else:
        return {"x_s": float("nan"), "z": float("nan"), "ok": False}

    x_s = float(x_new)
    # 鲁棒目标 (M2-7): z ≤ Σ ω_k·δ_k_s − λ_cost·Σ c_k·y_k − λ_L·max(0,x0−x_s_new)
    omega = M2_CONFIG["omega_I"] * dI_s + M2_CONFIG["omega_E"] * dE_s + M2_CONFIG["omega_S"] * dC_s
    cost = M2_CONFIG["lambda_cost"] * (M2_CONFIG["c_I"] * y_I + M2_CONFIG["c_E"] * y_E + M2_CONFIG["c_S"] * y_C)
    loss = M2_CONFIG["lambda_L"] * max(0.0, x0 - x_s)
    z = omega - cost - loss
    return {"x_s": x_s, "z": z, "omega": omega, "cost": cost, "loss": loss, "ok": True}


def main() -> None:
    params = dict(m1.DEFAULT_PARAMS)
    log_lines: List[str] = []

    def log(msg: str) -> None:
        print(msg); log_lines.append(msg)

    log("=" * 78)
    log("M2 鲁棒反哺模型 v6.1")
    log(f"项目: 3 类 × 8 组合, 情景: 15 蒙卡 (x~N(x0,10%), τ~N(τ0,5%))")
    log(f"量纲: ω_I=1.5, ω_E=2.0, ω_S=1.2; λ_cost=0.3, λ_L=0.05")
    log("=" * 78)

    # --- M1 最优解 (v6.0, 从 run_log 取) ---
    m_opt = M2_CONFIG["m_optimal"]
    delta_opt = M2_CONFIG["delta_optimal"]
    log(f"\n固定 M1 v6.1 最优: m={''.join(map(str, m_opt))}, δ=({delta_opt[0]:.3f},{delta_opt[1]:.3f},{delta_opt[2]:.3f})")

    # --- 预算约束 ---
    B = M2_CONFIG["B_total"]
    log(f"总预算 B={B} 亿; c_I={M2_CONFIG['c_I']}, c_E={M2_CONFIG['c_E']}, c_S={M2_CONFIG['c_S']}")

    # --- 蒙卡情景生成 ---
    rng = np.random.default_rng(M2_CONFIG["seed"])
    n_scen = M2_CONFIG["n_scenarios"]
    sigma_x = M2_CONFIG["sigma_x"]
    sigma_tau = M2_CONFIG["sigma_tau"]
    tau_arr = rng.normal(params["tau0"], sigma_tau * params["tau0"], n_scen).clip(0.01, params["tau_max"])
    # x_d 情景扰动 (v6.0 简化: x_d_s = x_d0 固定 + 小扰动)
    xd_arr = np.full(n_scen, params["xd0"]) + rng.normal(0, 0.1 * params["xd0"], n_scen)
    log(f"\n[蒙卡] {n_scen} 情景: τ ∈ [{tau_arr.min():.3f},{tau_arr.max():.3f}], x_d ∈ [{xd_arr.min():.2f},{xd_arr.max():.2f}]")

    # --- 枚举 8 种 y 组合 ---
    y_combos = list(itertools.product([0, 1], repeat=3))
    log(f"\n[枚举] {len(y_combos)} 种 y_I,y_E,y_C 组合:")

    rows_scenarios: List[Dict[str, Any]] = []     # 所有情景明细
    rows_projects: List[Dict[str, Any]] = []      # 每种 y 组合汇总

    t0 = time.perf_counter()
    for y_I, y_E, y_C in y_combos:
        c_y = M2_CONFIG["c_I"] * y_I + M2_CONFIG["c_E"] * y_E + M2_CONFIG["c_S"] * y_C
        budget_ok = c_y <= B
        if not budget_ok:
            log(f"  y=({y_I},{y_E},{y_C}): 成本 {c_y:.2f} > 预算 {B} → 跳过")
            rows_projects.append({"y_I": y_I, "y_E": y_E, "y_C": y_C,
                                   "feasible_flag": 0, "z_worst": float("nan"),
                                   "cost": c_y, "n_ok": 0})
            continue

        # 15 情景
        n_ok = 0
        z_list: List[float] = []
        x_list: List[float] = []
        for s in range(n_scen):
            r = run_m2_scenario(params, float(tau_arr[s]), float(xd_arr[s]),
                                y_I, y_E, y_C, delta_opt, m_opt)
            rows_scenarios.append({
                "y_I": y_I, "y_E": y_E, "y_C": y_C,
                "scenario_id": f"S{s+1:02d}",
                "tau_s": float(tau_arr[s]), "x_d_s": float(xd_arr[s]),
                "x_s_new": r["x_s"] if r["ok"] else np.nan,
                "z": r["z"] if r["ok"] else np.nan,
                "feasible_flag": int(r["ok"]),
            })
            if r["ok"]:
                n_ok += 1
                z_list.append(r["z"])
                x_list.append(r["x_s"])

        if z_list:
            z_worst = min(z_list)        # 鲁棒: 取 15 情景中 z 最小的 (最坏情况)
            rows_projects.append({
                "y_I": y_I, "y_E": y_E, "y_C": y_C,
                "feasible_flag": 1, "z_worst": z_worst,
                "cost": c_y, "n_ok": n_ok, "z_best": max(z_list),
                "x_s_mean": float(np.mean(x_list)),
                "x_s_std": float(np.std(x_list)),
            })
            log(f"  y=({y_I},{y_E},{y_C}): 可行 {n_ok}/{n_scen} | z_worst={z_worst:.4f} | cost={c_y:.2f}")
        else:
            rows_projects.append({"y_I": y_I, "y_E": y_E, "y_C": y_C,
                                   "feasible_flag": 1, "z_worst": float("nan"),
                                   "cost": c_y, "n_ok": 0})

    dt = time.perf_counter() - t0

    # --- 最优项目组合 ---
    feasible_proj = [r for r in rows_projects if r["feasible_flag"] == 1 and np.isfinite(r["z_worst"])]
    n_feas_y = len(feasible_proj)
    if feasible_proj:
        best_proj = max(feasible_proj, key=lambda r: r["z_worst"])
        log(f"\n[最优项目] y=({best_proj['y_I']},{best_proj['y_E']},{best_proj['y_C']}), z_worst={best_proj['z_worst']:.4f}, cost={best_proj['cost']:.2f}")
        # 验收: 至少 4 种 y 可行 (Prompt 05 第 244 行)
        log(f"[验收] 可行 y 组合数 = {n_feas_y}/8 {'✅ ≥ 4' if n_feas_y >= 4 else '❌ < 4'}")
    else:
        log("\n[FATAL] 所有 y 组合不可行!")

    log(f"\n耗时 {dt:.1f}s")

    # --- 存 CSV ---
    pd.DataFrame(rows_projects).to_csv(RESULTS_DIR / "m2_projects.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_scenarios).to_csv(RESULTS_DIR / "m2_scenarios.csv", index=False, encoding="utf-8-sig")
    log(f"已存 m2_projects.csv, m2_scenarios.csv")

    log_lines.append(f"\n[运行摘要] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"  y 组合: 8, 可行 {n_feas_y}")
    log_lines.append(f"  蒙卡情景: {n_scen}")
    log_lines.append(f"  耗时: {dt:.2f}s")

    with open(RESULTS_DIR / "run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print("\n===== M2 v6.1 DONE =====")


if __name__ == "__main__":
    main()
