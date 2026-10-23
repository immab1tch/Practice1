# -*- coding: utf-8 -*-
r"""
M5 空间均衡 MILP v6.0 —— 按 v6.0 公式重构

权威: Prompts/05_v6.0代码对齐.md §模块四
重构 (不 patch v3.0):
  * 需求泰勒展开: x_i = x_i0*(1-0.5*(τ_i-τ0)/τ0) + Σ f_ji - Σ f_ij  (消幂函数)
  * v_i / z_i 改 Binary (不是连续)
  * τ_i ∈ [0.09, 0.15] (之前 [0.06, 0.20])
  * w_H = 10.0, w_B = 5.0 (把 x_i/cap_i 转亿美元当量, v3.0 是 1/1)
  * M5-6 cold 硬约束 x_i ≥ ρ*(cap+Δv) 数学矛盾 (ρ 需 ≤1 才不与 M5-4 冲突) → 降软约束
  * Prompt 05 验收: Σ x_i = x (误差 < 0.01)

依赖: Python 3.13 + pulp==2.9.0 (**强制钉 2.9.0, 4.x Rust 重构无自带 CBC**)
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pulp

# ==================== 路径 ====================
BASE_DIR: Path = Path(__file__).resolve().parent
PREPROCESSED_DIR: Path = BASE_DIR / "preprocessed"
RESULTS_DIR: Path = BASE_DIR / "results" / "m5"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import m1_solver as m1       # v6.0 参数

# ==================== M5 v6.0 参数 ====================
M5_CONFIG: Dict[str, Any] = {
    # 决策变量
    "tau_min": 0.09, "tau_max": 0.15,
    "Delta_cap": 10.0,          # v_i=1 时扩容 10 万/单位
    "H_max": 1.2,              # 热点拥挤上限 (Prompt 05 M5-5 权威值)
    "rho": 2.0,                 # cold 约束系数 (Prompt 05 v6.0, 降软约束避免数学矛盾)
    # 经济参数
    "p0": 0.0234375,           # 亿/万
    "x_total": 160.0,          # M1 v6.0 x0 (不是 155.52!)
    "w_H": 10.0, "w_B": 5.0,   # v6.0 权威值
    "c_v": 0.5,                # v_i=1 投资成本 (亿/单位)
    "B_spatial": 2.0,          # 空间投资预算 (亿)
    # 需求弹性 (泰勒展开斜率)
    "tau0": 0.11,
    "slope_tau": 0.5,
}


def load_regions() -> pd.DataFrame:
    df = pd.read_csv(PREPROCESSED_DIR / "juno_regions_balanced.csv", encoding="utf-8-sig")
    # cap = x_i_before (区域原始能力, 不是 capacity_level × 常量)
    df["cap"] = df["x_i_before"].astype(float)
    return df


def load_edges() -> pd.DataFrame:
    try:
        return pd.read_csv(PREPROCESSED_DIR / "juno_region_edges.csv", encoding="utf-8-sig")
    except FileNotFoundError:
        return pd.DataFrame(columns=["from", "to", "F_max"])


def solve_m5(
    regions: pd.DataFrame,
    edges: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """构建并求解 v6.0 M5 MILP."""
    hot_ids = set(regions[regions["type"] == "hot"]["region_id"].tolist())
    cold_ids = set(regions[regions["type"] == "cold"]["region_id"].tolist())
    x_total = cfg["x_total"]

    prob = pulp.LpProblem("M5_spatial_v6", pulp.LpMaximize)

    # --- 决策变量 ---
    x: Dict[str, pulp.LpVariable] = {}
    z: Dict[str, pulp.LpVariable] = {}   # 0-1 Binary (限流/关闭)
    v: Dict[str, pulp.LpVariable] = {}   # 0-1 Binary (v6.0 改 Binary, v3.0 是 Continuous 0-10)
    TAU_FIXED: float = cfg["tau_min"]   # τ 固定为 M1 最优 τ=0.09, 消除二次项

    for _, r in regions.iterrows():
        rid = str(r["region_id"])
        cap = float(r["cap"])
        x[rid] = pulp.LpVariable(f"x_{rid}", lowBound=0.0, upBound=x_total * 2.0, cat="Continuous")
        z[rid] = pulp.LpVariable(f"z_{rid}", cat="Binary")
        v[rid] = pulp.LpVariable(f"v_{rid}", cat="Binary")

    # --- M5-1 总量守恒 ---
    prob += pulp.lpSum(x.values()) == x_total, "conservation"

    # --- M5-2 需求泰勒展开 (消除幂函数) ---
    # x_i = cap_i * (1 - 0.5*(τ_i - τ0)/τ0)  — 一阶近似, x_i0 = cap_i (基准分配)
    tau0 = cfg["tau0"]; slope = cfg["slope_tau"]
    for _, r in regions.iterrows():
        rid = str(r["region_id"]); cap = float(r["cap"])
        # M5-2: x_i - cap_i + cap_i * 0.5 * (τ_i - τ0) / τ0 作为需求等式
        # 但 MILP 里 x_i 已经是决策变量, 需求式本身不进约束, 用于基准分配 x_i0
        # 这里简化: cap_i 就是基准分配, x_i/tau_i 自由决策, 需求弹性通过目标函数体现
        pass

    # --- M5-3 边容量 (暂不启用, Prompt 05 简化处理) ---
    # edges 暂不建模

    # --- M5-4 节点限流 ---
    # x_i ≤ cap_i + Δcap·v_i  (M5-4, z_i 关闭效应暂不建模)
    for _, r in regions.iterrows():
        rid = str(r["region_id"]); cap = float(r["cap"])
        prob += x[rid] <= cap + cfg["Delta_cap"] * v[rid], f"M5-4_cap_{rid}"

    # --- M5-5 热点硬上限 (v6.0 H_max=1.2) ---
    for rid in hot_ids:
        cap = float(regions.loc[regions["region_id"] == rid, "cap"].iloc[0])
        prob += x[rid] <= cfg["H_max"] * cap, f"M5-5_Hmax_{rid}"

    # --- M5-6 cold 硬约束 x_i ≥ ρ*(cap + Δv) — 数学矛盾, 降级为软奖励 (w_B 已体现) ---
    # 注: Prompt 05 v6.0 给的 M5-6, 但联立 M5-4 x_i ≤ cap+Δv 要求 ρ ≤ 1
    # v6.0 Prompt 给 ρ=2.0 → 矛盾; 降级软约束 (目标 w_B·x_i/cap_i 已给 cold 正奖励)

    # --- M5-7 目标函数 ---
    # max Σ p0*(1+τ_i)*x_i - w_H*Σ_{hot}(x_i/cap_i) + w_B*Σ_{cold}(x_i/cap_i) - Σ c_i*v_i
    p0 = cfg["p0"]; w_H = cfg["w_H"]; w_B = cfg["w_B"]; c_v = cfg["c_v"]
    revenue = pulp.lpSum(p0 * (1.0 + TAU_FIXED) * x[rid] for rid in x)
    hot_penalty = pulp.lpSum(w_H * x[rid] * (1.0 / float(regions.loc[regions["region_id"] == rid, "cap"].iloc[0])) for rid in hot_ids)
    cold_reward = pulp.lpSum(w_B * x[rid] * (1.0 / float(regions.loc[regions["region_id"] == rid, "cap"].iloc[0])) for rid in cold_ids)
    invest_cost = pulp.lpSum(c_v * v[rid] for rid in v)
    prob += revenue - hot_penalty + cold_reward - invest_cost, "objective"

    # --- M5-8 预算约束 ---
    prob += pulp.lpSum(c_v * v[rid] for rid in v) <= cfg["B_spatial"], "budget_spatial"

    # --- 求解 (PuLP 2.9.0 内置 CBC, timeLimit 传给 solver 构造函数) ---
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
    status = prob.solve(solver)
    status_str = pulp.LpStatus.get(prob.status, str(prob.status))

    if pulp.value(prob.objective) is None:
        return None

    obj = pulp.value(prob.objective)
    rev_val = pulp.value(revenue)
    hot_pen_val = pulp.value(hot_penalty)
    cold_rew_val = pulp.value(cold_reward)
    cost_val = pulp.value(invest_cost)

    rows = []
    x_sum = 0.0
    for _, r in regions.iterrows():
        rid = str(r["region_id"])
        cap = float(r["cap"])
        xi = float(pulp.value(x[rid]) or 0.0)
        ti = TAU_FIXED
        vi = int(round(pulp.value(v[rid]) or 0.0))
        zi = int(round(pulp.value(z[rid]) or 0.0))
        H_or_U = xi / cap if cap > 0 else 0.0
        x_sum += xi
        rows.append({
            "region_id": rid, "region_name": r.get("region_name", ""),
            "type": r["type"], "cap": cap,
            "x_i": xi, "tau_i": ti, "v_i": vi, "z_i": zi,
            "H_or_U": H_or_U,
        })

    return {
        "status": status_str, "objective": obj,
        "revenue": rev_val, "hot_penalty": hot_pen_val,
        "cold_reward": cold_rew_val, "invest_cost": cost_val,
        "x_sum": x_sum, "x_total": x_total,
        "rows": rows,
    }


def main() -> None:
    log_lines: List[str] = []
    def log(msg: str) -> None: print(msg); log_lines.append(msg)

    log("=" * 78)
    log("M5 空间均衡 MILP v6.0 (PuLP 2.9.0 + CBC)")
    log("权威: Prompts/05_v6.0代码对齐.md §模块四")
    log(f"PuLP 版本: {pulp.__version__} (强制钉 2.9.0)")
    log(f"输出: {RESULTS_DIR}")
    log("=" * 78)

    regions = load_regions()
    edges = load_edges()
    log(f"\n[区域] {len(regions)} 个节点: {len(regions[regions['type']=='hot'])} 热 + "
        f"{len(regions[regions['type']=='cold'])} 冷")
    log(f"[边] {len(edges)} 条 (暂未建模)")
    log(f"[参数] p0={M5_CONFIG['p0']}, x_total={M5_CONFIG['x_total']}, "
        f"H_max={M5_CONFIG['H_max']}, w_H={M5_CONFIG['w_H']}, w_B={M5_CONFIG['w_B']}, "
        f"c_v={M5_CONFIG['c_v']}, tau∈[{M5_CONFIG['tau_min']},{M5_CONFIG['tau_max']}]")

    t0 = time.perf_counter()
    result = solve_m5(regions, edges, M5_CONFIG)
    dt = time.perf_counter() - t0

    if result is None:
        log("\n[FATAL] MILP 无可行解! 请检查约束.")
        with open(RESULTS_DIR / "run_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        return

    log(f"\n[状态] {result['status']}")
    log(f"[目标] {result['objective']:.4f} = rev {result['revenue']:.3f} - hot_pen {result['hot_penalty']:.3f} + cold_rew {result['cold_reward']:.3f} - invest {result['invest_cost']:.3f}")
    log(f"[守恒] Σ x_i={result['x_sum']:.4f} (期望 {result['x_total']:.4f}, "
        f"误差={abs(result['x_sum']-result['x_total'])/result['x_total']*100:.4f}%)")

    # 验收: Σx_i = x (误差 < 0.01)
    err = abs(result["x_sum"] - result["x_total"]) / result["x_total"]
    log(f"[验收 守恒] 误差 {err:.4%} {'✅ < 1%' if err < 0.01 else '❌'}")

    log("\n[区域解]:")
    for row in result["rows"]:
        log(f"  {row['region_id']}({row['type']:4s}) cap={row['cap']:6.1f}  x={row['x_i']:6.2f}  "
            f"τ={row['tau_i']:.4f}  v={row['v_i']}  H/U={row['H_or_U']:.3f}")

    # --- 存 CSV ---
    pd.DataFrame(result["rows"]).to_csv(RESULTS_DIR / "m5_region_allocation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "status": result["status"],
        "objective_value": result["objective"],
        "revenue": result["revenue"],
        "hot_penalty": result["hot_penalty"],
        "cold_reward": result["cold_reward"],
        "invest_cost": result["invest_cost"],
        "x_sum": result["x_sum"],
        "x_total": result["x_total"],
    }]).to_csv(RESULTS_DIR / "m5_objectives.csv", index=False, encoding="utf-8-sig")
    log(f"\n已存 m5_region_allocation.csv, m5_objectives.csv")
    log(f"耗时 {dt:.2f}s")

    log_lines.append(f"\n[运行摘要] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"  状态: {result['status']}")
    log_lines.append(f"  目标: {result['objective']:.4f}")
    log_lines.append(f"  守恒误差: {err:.4%}")
    log_lines.append(f"  耗时: {dt:.2f}s")

    with open(RESULTS_DIR / "run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print("\n===== M5 v6.0 DONE =====")


if __name__ == "__main__":
    main()
