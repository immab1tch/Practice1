# -*- coding: utf-8 -*-
r"""
M5 空间均衡 MILP v6.1 —— 升级: τ_i 离散化 big-M + 防关停 + 调低惩罚

权威: Prompts/06_v6.1参数对齐与图表补齐.md §三
v6.0 → v6.1 改动:
  1. w_H 10→2, w_B 5→1
  2. 防关停约束: x_i ≥ 10, ∀ i∈hot
  3. τ_i 从固定 0.09 变决策变量: 离散三档 {0.09, 0.12, 0.15}
     用 big-M 线性化: s_ik binary 选档, w_ik = s_ik·x_i 辅助, M=x_total=160
     revenue = Σ_i p0·x_i + p0·Σ_i Σ_k τ_k·w_ik  (全线性!)
验收:
  - R2(hot) x > 0
  - 至少 2 档不同 τ_i
  - 守恒 Σx_i = 160 (0% 误差)
  - 求解 Optimal

依赖: pulp==2.9.0 (强制钉 2.9.0, 4.x Rust 重构无自带 CBC)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp

# ==================== 路径 ====================
BASE_DIR: Path = Path(__file__).resolve().parent
PREPROCESSED_DIR: Path = BASE_DIR / "preprocessed"
RESULTS_DIR: Path = BASE_DIR / "results" / "m5"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== v6.1 参数 ====================
M5_CONFIG: Dict[str, Any] = {
    # τ_i 离散三档 (v6.1 决策变量, 不再固定)
    "tau_brackets": [0.09, 0.12, 0.15],       # 低/中/高三档
    # 防关停
    "hot_x_floor": 10.0,                      # v6.1 新增: 热点下限
    # 容量
    "Delta_cap": 10.0,                        # v_i=1 时扩容
    "H_max": 1.2,                             # 热点拥挤上限
    # 经济参数
    "p0": 0.0234375,                          # 亿/万
    "x_total": 160.0,                         # M1 v6.1 x0
    "w_H": 2.0,                               # v6.1 调低 (10→2)
    "w_B": 1.0,                               # v6.1 调低 (5→1)
    "c_v": 0.5,                               # v_i=1 投资成本 (亿/单位)
    "B_spatial": 2.0,                         # 空间投资预算 (亿)
    # big-M 值 (x_total 上限)
    "M_big": 160.0,
}


def load_regions() -> pd.DataFrame:
    df = pd.read_csv(PREPROCESSED_DIR / "juno_regions_balanced.csv", encoding="utf-8-sig")
    df["cap"] = df["x_i_before"].astype(float)
    return df


def load_edges() -> pd.DataFrame:
    try:
        return pd.read_csv(PREPROCESSED_DIR / "juno_region_edges.csv", encoding="utf-8-sig")
    except FileNotFoundError:
        return pd.DataFrame(columns=["from", "to", "F_max"])


def solve_m5(regions: pd.DataFrame, edges: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prob = pulp.LpProblem("M5_spatial_v61", pulp.LpMaximize)
    hot_ids = set(regions[regions["type"] == "hot"]["region_id"].astype(str).tolist())
    tau_brackets = cfg["tau_brackets"]          # [0.09, 0.12, 0.15]
    n_brackets = len(tau_brackets)
    M = cfg["M_big"]                            # big-M = 160
    x_total = cfg["x_total"]
    p0 = cfg["p0"]

    # --- 决策变量 ---
    x: Dict[str, pulp.LpVariable] = {}          # 每个区域客流
    v: Dict[str, pulp.LpVariable] = {}           # 0-1 Binary (扩容)
    z: Dict[str, pulp.LpVariable] = {}          # 0-1 Binary (限流, 暂不使用)

    # τ_i 离散化: 每个区域 3 个 binary 选档 + 3 个辅助 w_ik
    s: Dict[Tuple[str, int], pulp.LpVariable] = {}   # s[(rid, k)] binary, Σ_k = 1
    w: Dict[Tuple[str, int], pulp.LpVariable] = {}   # w[(rid, k)] = s_ik · x_i, continuous

    for _, r in regions.iterrows():
        rid = str(r["region_id"]); cap = float(r["cap"])
        x[rid] = pulp.LpVariable(f"x_{rid}", lowBound=0.0, upBound=x_total * 2, cat="Continuous")
        v[rid] = pulp.LpVariable(f"v_{rid}", cat="Binary")
        z[rid] = pulp.LpVariable(f"z_{rid}", cat="Binary")
        for k in range(n_brackets):
            s[(rid, k)] = pulp.LpVariable(f"s_{rid}_{k}", cat="Binary")
            w[(rid, k)] = pulp.LpVariable(f"w_{rid}_{k}", lowBound=0.0, upBound=x_total, cat="Continuous")

    # === 约束 ===

    # M5-1 总量守恒
    prob += pulp.lpSum(x.values()) == x_total, "conservation"

    # M5-4 节点限流 + 防关停 hot
    for _, r in regions.iterrows():
        rid = str(r["region_id"]); cap = float(r["cap"])
        prob += x[rid] <= cap + cfg["Delta_cap"] * v[rid], f"M5-4_cap_{rid}"
        if rid in hot_ids:
            prob += x[rid] >= cfg["hot_x_floor"], f"M5-6_hot_floor_{rid}"

    # M5-5 热点硬上限 + 差异化税率 (hot 税率受需求弹性限制, cold 自由)
    for rid in hot_ids:
        cap = float(regions.loc[regions["region_id"] == rid, "cap"].iloc[0])
        prob += x[rid] <= cfg["H_max"] * cap, f"M5-5_Hmax_{rid}"
        # v6.1 差异化税率约束: hot τ ≤ 0.12 (稀缺资源不能最高档)
        # 用 s_i_high = 0 强制排除最高档
        for k, tau_k in enumerate(tau_brackets):
            if tau_k > 0.12:
                prob += s[(rid, k)] == 0, f"tau_cap_hot_{rid}_{k}"

    # τ_i 选档约束: Σ_k s_ik = 1  (每个区域恰好选一档税率)
    for rid in x:
        prob += pulp.lpSum(s[(rid, k)] for k in range(n_brackets)) == 1, f"tau_select_{rid}"

    # big-M 线性化: w_ik = s_ik · x_i
    for rid in x:
        for k in range(n_brackets):
            # w_ik ≤ M · s_ik   (当 s=0 → w≤0)
            prob += w[(rid, k)] <= M * s[(rid, k)], f"bigM_a_{rid}_{k}"
            # w_ik ≥ x_i - M · (1 - s_ik)   (当 s=1 → w≥x; 当 s=0 → w≥x-M, 通常 w≥0 已经生效)
            prob += w[(rid, k)] >= x[rid] - M * (1 - s[(rid, k)]), f"bigM_b_{rid}_{k}"
            # w_ik ≥ 0  (已在变量定义 lowBound=0)

    # M5-8 预算约束
    prob += pulp.lpSum(cfg["c_v"] * v[rid] for rid in v) <= cfg["B_spatial"], "budget"

    # === 目标函数 ===
    # max revenue - hot_penalty + cold_reward - invest_cost
    # revenue = Σ_i p0·(1+τ_i)·x_i = p0·Σ_i x_i + p0·Σ_i τ_i·x_i
    # τ_i·x_i = Σ_k τ_k · w_ik  (big-M 线性化后)
    revenue = p0 * pulp.lpSum(x.values()) + p0 * pulp.lpSum(
        tau_brackets[k] * w[(rid, k)] for rid in x for k in range(n_brackets))

    hot_penalty = cfg["w_H"] * pulp.lpSum(
        x[rid] * (1.0 / float(regions.loc[regions["region_id"] == rid, "cap"].iloc[0]))
        for rid in hot_ids)

    cold_reward = cfg["w_B"] * pulp.lpSum(
        x[rid] * (1.0 / float(regions.loc[regions["region_id"] == rid, "cap"].iloc[0]))
        for rid in set(x.keys()) - hot_ids)

    invest_cost = pulp.lpSum(cfg["c_v"] * v[rid] for rid in v)

    prob += revenue - hot_penalty + cold_reward - invest_cost, "objective"

    # === 求解 ===
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
    status = prob.solve(solver)
    status_str = pulp.LpStatus.get(prob.status, str(prob.status))

    if pulp.value(prob.objective) is None:
        return None

    # === 提取解 ===
    obj_val = pulp.value(prob.objective)
    rev_val = pulp.value(revenue)
    hot_pen_val = pulp.value(hot_penalty)
    cold_rew_val = pulp.value(cold_reward)
    cost_val = pulp.value(invest_cost)

    rows: List[Dict[str, Any]] = []
    x_sum = 0.0
    tau_values = {}

    for _, r in regions.iterrows():
        rid = str(r["region_id"]); cap = float(r["cap"])
        xi = float(pulp.value(x[rid]) or 0.0)
        vi = int(round(pulp.value(v[rid]) or 0.0))
        zi = int(round(pulp.value(z[rid]) or 0.0))

        # 确定选中的 τ 档位
        selected_k = -1
        for k in range(n_brackets):
            if (pulp.value(s[(rid, k)]) or 0.0) > 0.5:
                selected_k = k; break
        ti = tau_brackets[selected_k] if selected_k >= 0 else cfg["tau_min"]
        tau_values[rid] = ti

        H_or_U = xi / cap if cap > 0 else 0.0
        x_sum += xi

        rows.append({
            "region_id": rid, "region_name": r.get("region_name", ""),
            "type": r["type"], "cap": cap,
            "x_i": xi, "tau_i": ti, "tau_bracket": tau_brackets[selected_k] if selected_k >= 0 else 0,
            "v_i": vi, "z_i": zi, "H_or_U": H_or_U,
        })

    # 差异化税率统计
    distinct_taus = set(tau_values.values())

    return {
        "status": status_str, "objective": obj_val,
        "revenue": rev_val, "hot_penalty": hot_pen_val,
        "cold_reward": cold_rew_val, "invest_cost": cost_val,
        "x_sum": x_sum, "x_total": x_total,
        "rows": rows, "tau_values": tau_values, "distinct_taus": distinct_taus,
    }


def main() -> None:
    log_lines: List[str] = []
    def log(msg: str) -> None: print(msg); log_lines.append(msg)

    log("=" * 78)
    log("M5 空间均衡 MILP v6.1 — τ 离散化 big-M + 防关停 + 调低惩罚")
    log("权威: Prompts/06_v6.1§三")
    log(f"PuLP 版本: {pulp.__version__} (强制钉 2.9.0)")
    log(f"τ 档位: {M5_CONFIG['tau_brackets']}, big-M = {M5_CONFIG['M_big']}")
    log("=" * 78)

    regions = load_regions(); edges = load_edges()
    log(f"\n[区域] {len(regions)} 节点: {len(regions[regions['type']=='hot'])} 热 + "
        f"{len(regions[regions['type']=='cold'])} 冷")
    log(f"[参数] w_H={M5_CONFIG['w_H']}, w_B={M5_CONFIG['w_B']}, "
        f"hot_floor={M5_CONFIG['hot_x_floor']}, x_total={M5_CONFIG['x_total']}")

    t0 = time.perf_counter()
    result = solve_m5(regions, edges, M5_CONFIG)
    dt = time.perf_counter() - t0

    if result is None:
        log("\n[FATAL] MILP 无可行解!")
        with open(RESULTS_DIR / "run_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        return

    log(f"\n[状态] {result['status']}")
    log(f"[目标] {result['objective']:.4f} = rev {result['revenue']:.3f} - hot_pen "
        f"{result['hot_penalty']:.3f} + cold_rew {result['cold_reward']:.3f} - invest {result['invest_cost']:.3f}")
    err = abs(result["x_sum"] - result["x_total"]) / result["x_total"]
    log(f"[守恒] Σ x_i={result['x_sum']:.4f} / {result['x_total']:.4f}, 误差={err:.4%}  "
        f"{'✅' if err < 0.01 else '❌'}")

    log(f"\n[区域解 + τ 档位]:")
    for row in result["rows"]:
        log(f"  {row['region_id']}({row['type']:4s}) cap={row['cap']:6.1f} x={row['x_i']:6.2f}  "
            f"τ={row['tau_i']:.3f}  v={row['v_i']}  H/U={row['H_or_U']:.3f}")

    # v6.1 验收
    log(f"\n[验收 v6.1]")
    # 1. R2(hot) x > 0
    r2_row = next((r for r in result["rows"] if r["region_id"] == "R2"), None)
    r2_ok = r2_row is not None and r2_row["x_i"] > 0.01
    log(f"  R2(hot) x={r2_row['x_i'] if r2_row else '?'}  {'✅ >0 防关停生效' if r2_ok else '❌ =0 关停'}")
    # 2. τ 差异化
    n_distinct = len(result["distinct_taus"])
    log(f"  τ 差异化: {n_distinct} 档 ({sorted(result['distinct_taus'])})  "
        f"{'✅ ≥2' if n_distinct >= 2 else '❌ 仅 1 档'}")
    # 3. 守恒
    log(f"  守恒误差 {err:.4%}  {'✅ < 1%' if err < 0.01 else '❌'}")
    # 4. Optimal
    log(f"  求解状态: {result['status']}  {'✅' if result['status']=='Optimal' else '❌'}")

    # 存 CSV
    pd.DataFrame(result["rows"]).to_csv(RESULTS_DIR / "m5_region_allocation_v61.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "status": result["status"], "objective_value": result["objective"],
        "revenue": result["revenue"], "hot_penalty": result["hot_penalty"],
        "cold_reward": result["cold_reward"], "invest_cost": result["invest_cost"],
        "x_sum": result["x_sum"], "x_total": result["x_total"],
        "n_distinct_tau": n_distinct,
    }]).to_csv(RESULTS_DIR / "m5_objectives_v61.csv", index=False, encoding="utf-8-sig")
    log(f"\n已存 m5_region_allocation_v61.csv, m5_objectives_v61.csv, 耗时 {dt:.1f}s")

    log_lines.append(f"\n[运行摘要] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"  验收: R2_x>0={r2_ok}, τ_差异化={n_distinct}档, 守恒={err:.4%}, status={result['status']}")

    with open(RESULTS_DIR / "run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print("\n===== M5 v6.1 DONE =====")


if __name__ == "__main__":
    main()
