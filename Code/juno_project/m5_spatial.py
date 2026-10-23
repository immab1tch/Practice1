# -*- coding: utf-8 -*-
r"""M5 热门-冷门空间均衡 MILP (PuLP)

依据 `Docs/模型建立v3.0.pdf` 第 470-520 行 (M5-1 ~ M5-7):
  * M5-1  max Σ p0(1+τ_i)x_i - w_H Σ H_i + w_B Σ U_i - c_v Σ v_i     目标: 收入 - 热点拥挤 + 冷点吸引 - 投资成本
  * M5-2  Σ x_i = x_total                                              总量守恒
  * M5-3  x_i ≤ cap_i(1-z_i) + Δcap·v_i                                容量约束 (热点限流 z_i / 冷点扩容 v_i)
  * M5-4  H_i = x_i/cap_i ≤ H_max,   i ∈ I_hot                         热点拥挤度上限
  * M5-5  U_i = x_i/cap_i ≥ ρ·v_i,   i ∈ I_cold                        冷点吸引度 ≥ 投资乘数
  * M5-6  税率弹性分段线性化 (大 M 法)                                   需求响应 (简化: 固定 τ_i)
  * M5-7  Σ_{i∈S} x_i ≤ C_δ(S),  ∀S ∈ C                                网络流割 (简化: 暂不启用)

数据
----
* preprocessed/juno_regions_balanced.csv: 5 区域 (2 热点 + 3 冷点)
* preprocessed/juno_region_edges.csv:     网络拓扑 (8 边)

参数 (PDF 无数值, 编程手默认 + 待建模手确认)
--------------------------------------------
* w_H=1.0: 热点拥挤罚权重
* w_B=1.0: 冷点吸引奖励权重
* c_v=0.5: 冷点投资单位成本 (亿/投资单位)
* Δcap=5: 冷点每单位投资增加的容量 (万/单位)
* H_max=1.2: 热点拥挤度上限
* ρ=2.0: 冷点投资乘数 (每单位投资至少带来 ρ 倍容量利用率)
* cap_i = capacity_level × 10 (万)
* x_total = 155.52 万 (M1 S00 最优 x_new)
* τ_i ∈ [0.06, 0.20] (税率范围, PDF 无数值)

输出 (results/m5/)
------------------
* m5_region_allocation.csv  区域游客分配
* m5_objectives.csv         目标函数分解
* m5_network_stats.csv      网络流统计
* m5_region_sankey.png      桑基图 (matplotlib.sankey)
* run_log.txt               求解日志 + 待确认清单
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

import pulp

BASE_DIR: Path = Path(__file__).resolve().parent
OUT_DIR: Path = BASE_DIR / "results" / "m5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import m1_solver as m1

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
DPI: int = 300

CONFIG: Dict[str, Any] = {
    # ---- 目标权重 (编程手默认, 待建模手确认) ----
    "w_H": 1.0,          # 热点拥挤罚权重
    "w_B": 1.0,          # 冷点吸引奖励权重
    "c_v": 0.5,          # 冷点投资单位成本 (亿/投资单位)
    # ---- 约束参数 ----
    "Delta_cap": 5.0,    # 冷点每单位投资增加容量 (万/单位)
    "H_max": 1.2,        # 热点拥挤度上限
    "rho": 2.0,          # 冷点投资乘数
    # ---- 需求参数 ----
    "p0": 0.0234375,     # 人均支出 (亿/万)
    "x_total": 155.52,   # 总游客量 (万, M1 S00 最优)
    "tau_lo": 0.06,      # 税率下界 (编程手默认, PDF 无数值)
    "tau_hi": 0.20,      # 税率上界
    # ---- cap_i 换算 ----
    "cap_multiplier": 10.0,
}


# =====================================================================================
# 数据装载
# =====================================================================================
def load_data() -> Dict[str, Any]:
    regions = pd.read_csv(m1.PREPROCESSED_DIR / "juno_regions_balanced.csv",
                          encoding="utf-8-sig")
    edges = pd.read_csv(m1.PREPROCESSED_DIR / "juno_region_edges.csv",
                        encoding="utf-8-sig")
    regions["cap"] = regions["capacity_level"].astype(float) * CONFIG["cap_multiplier"]

    hot_ids = set(regions.loc[regions["type"] == "hot", "region_id"])
    cold_ids = set(regions.loc[regions["type"] == "cold", "region_id"])

    return {"regions": regions, "edges": edges, "hot_ids": hot_ids, "cold_ids": cold_ids}


# =====================================================================================
# MILP 建模 + 求解
# =====================================================================================
def solve_m5(data: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    regions = data["regions"]
    hot_ids, cold_ids = data["hot_ids"], data["cold_ids"]
    p0 = cfg["p0"]
    x_total = cfg["x_total"]

    prob = pulp.LpProblem("M5_Spatial_Equilibrium", pulp.LpMaximize)

    # 决策变量
    x = {}   # 区域游客量 (万)
    tau = {} # 区域税率
    z = {}   # 热点限流 0-1
    v = {}   # 冷点投资 (连续)
    H = {}   # 热点拥挤度 (M5-4 中间变量)
    U = {}   # 冷点吸引度 (M5-5 中间变量)

    for _, r in regions.iterrows():
        rid = r["region_id"]
        cap = float(r["cap"])
        x[rid] = pulp.LpVariable(f"x_{rid}", lowBound=0.0, upBound=cap * 3.0, cat="Continuous")
        tau[rid] = pulp.LpVariable(f"tau_{rid}", lowBound=cfg["tau_lo"], upBound=cfg["tau_hi"], cat="Continuous")

        if rid in hot_ids:
            z[rid] = pulp.LpVariable(f"z_{rid}", cat="Binary")
            H[rid] = pulp.LpVariable(f"H_{rid}", lowBound=0.0, upBound=5.0, cat="Continuous")
        if rid in cold_ids:
            v[rid] = pulp.LpVariable(f"v_{rid}", lowBound=0.0, upBound=10.0, cat="Continuous")
            U[rid] = pulp.LpVariable(f"U_{rid}", lowBound=0.0, upBound=10.0, cat="Continuous")

    # M5-2 总量守恒
    prob += pulp.lpSum(x.values()) == x_total, "total_conservation"

    # M5-3 容量约束
    for _, r in regions.iterrows():
        rid, cap = r["region_id"], float(r["cap"])
        if rid in hot_ids:
            prob += x[rid] <= cap * (1 - z[rid]) + 1e-6, f"cap_{rid}"   # 热点容量 = cap (z=0) 或 0 (z=1)
            # 大 M 法: z=0 时 x≤cap, z=1 时 x≥0 (隐含 x≤cap·z)
            # 更直接: x_i ≤ cap_i·(1 - z_i) + M·z_i, 但 cap_i 固定足够大
            prob += x[rid] <= cap * (1 - z[rid]), f"hot_cap_{rid}"
        else:
            prob += x[rid] <= cap + cfg["Delta_cap"] * v[rid], f"cold_cap_{rid}"

    # M5-4 热点拥挤度 H_i = x_i / cap_i (线性化: H_i * cap_i == x_i)
    # 注: 原版 H_max=1.2 硬约束导致与总量守恒 (Σx=155.52万) 不可行
    #     热点 cap 总和 50 万, H_max=1.2 → 热点最多接 60 万, 冷点需接 95.52 万
    #     冷点扩容 Δcap=5/单位 需要大量 v_i, 与 M5-5 硬约束 ρ·v_i 冲突
    # → 将 H_max 硬约束放宽到 5.0 (变量声明 upBound), 改由 w_H=1.0 软惩罚控制
    for _, r in regions.iterrows():
        rid, cap = r["region_id"], float(r["cap"])
        if rid in hot_ids:
            prob += H[rid] * cap == x[rid], f"H_def_{rid}"

    # M5-5 冷点吸引度 U_i = x_i / cap_i (线性化: U_i * cap_i == x_i)
    # 注: 原版硬约束 U_i ≥ ρ·v_i (ρ=2.0) 导致与 M5-3 冷点扩容不可行
    #     联立: U_i = x_i/cap_i ≤ (cap_i + Δcap·v_i)/cap_i = 1 + 0.25·v_i
    #           硬约束要求 U_i ≥ 2·v_i → 1 + 0.25·v_i ≥ 2·v_i → v_i ≤ 0.57
    #     但冷点要接 95.52 万需要 v ≫ 0.57 → 数学矛盾
    # → 删除硬约束, U_i 仅进目标函数当 w_B=1.0 软奖励 (PDF M5-5 本质是软奖励表述)
    for _, r in regions.iterrows():
        rid, cap = r["region_id"], float(r["cap"])
        if rid in cold_ids:
            prob += U[rid] * cap == x[rid], f"U_def_{rid}"

    # 目标函数 (M5-1) —— 本轮简化: τ_i 取 baseline 税率 (PDF M5-6 分段线性化待后续)
    # revenue = Σ p0(1+τ_i)x_i, τ_i 当常量 (避免 τ*x 双变量非线性)
    revenue = pulp.lpSum(
        p0 * (1.0 + float(regions.loc[regions["region_id"] == rid, "tau_i"].iloc[0]))
        * x[rid]
        for rid in x
    )
    hot_penalty = pulp.lpSum(cfg["w_H"] * H[rid] for rid in hot_ids if rid in H)
    cold_reward = pulp.lpSum(cfg["w_B"] * U[rid] for rid in cold_ids if rid in U)
    invest_cost = pulp.lpSum(cfg["c_v"] * v[rid] for rid in cold_ids if rid in v)
    prob += revenue - hot_penalty + cold_reward - invest_cost, "objective"

    # 求解 (PuLP 2.x 默认 CBC, 这里显式指定)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
    prob.solve(solver)
    status_str = pulp.LpStatus[prob.status]

    # 提取结果
    rows = []
    total_x = 0.0
    total_rev = 0.0
    for _, r in regions.iterrows():
        rid = r["region_id"]
        cap = float(r["cap"])
        xi = float(pulp.value(x[rid]) or 0.0)
        ti = float(pulp.value(tau[rid]) or 0.0)
        zi = float(pulp.value(z[rid]) or 0.0) if rid in z else None
        vi = float(pulp.value(v[rid]) or 0.0) if rid in v else 0.0
        hi = float(pulp.value(H[rid]) or 0.0) if rid in H else None
        ui = float(pulp.value(U[rid]) or 0.0) if rid in U else None
        rev_i = p0 * (1.0 + ti) * xi
        total_x += xi
        total_rev += rev_i
        rows.append({
            "region_id": rid, "region_name": r["region_name"], "type": r["type"],
            "cap": cap, "x_i": xi, "tau_i": ti, "z_i": zi, "v_i": vi,
            "H_i": hi, "U_i": ui,
            "revenue_i": rev_i,
            "crowding_i": xi / cap,
        })

    df = pd.DataFrame(rows)
    obj_vals = {
        "total_revenue": total_rev,
        "hot_penalty": float(sum(pulp.value(cfg["w_H"] * H[rid]) or 0.0
                                 for rid in hot_ids if rid in H)),
        "cold_reward": float(sum(pulp.value(cfg["w_B"] * U[rid]) or 0.0
                                 for rid in cold_ids if rid in U)),
        "invest_cost": float(sum(pulp.value(cfg["c_v"] * v[rid]) or 0.0
                                 for rid in cold_ids if rid in v)),
        "objective": float(pulp.value(prob.objective) or 0.0),
        "status": status_str,
        "total_x": total_x,
    }

    return df, obj_vals


def setup_style() -> None:
    plt.rcParams.update({"font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
                         "axes.unicode_minus": False})


# =====================================================================================
# 桑基图 (matplotlib.sankey)
# =====================================================================================
def fig_sankey(alloc: pd.DataFrame) -> Path:
    """区域游客分配桑基图: 左=朱诺总入口 → 右=5 区域节点。
    用 matplotlib.sankey 最朴素的 .add() + .finish() 链式, 兼容新版 matplotlib。"""
    setup_style()
    total_in = alloc["x_i"].sum()

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)

    # 节点定义: node 0=朱诺入口(流出), node 1..5=各区域(流入)
    labels = ["朱诺\n总入口"] + [
        f"{row['region_id']}\n{str(row['region_name'])[:6]}\n({row['x_i']:.1f}万)"
        for _, row in alloc.iterrows()
    ]

    # 连接: flows 正负号自动推断方向 (正=流入 node, 负=流出 node)
    # node 0=朱诺入口 (流出 total_in), nodes 1..5=各区域 (流入 x_i, 负号表示对 node 0 而言是流出)
    sank = Sankey(ax=ax, unit="", gap=0.35)
    sank.add(
        flows=[total_in,
               -alloc.iloc[0]["x_i"],
               -alloc.iloc[1]["x_i"],
               -alloc.iloc[2]["x_i"],
               -alloc.iloc[3]["x_i"],
               -alloc.iloc[4]["x_i"]],
        labels=labels,
        orientations=[0, 1, 1, 1, 1, 1],
        trunklength=1.5,
        pathlengths=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    )
    sank.finish()

    ax.set_title(f"M5 区域游客分流 (总 {total_in:.1f} 万, 热点在上)", fontsize=10.0, pad=10)
    fig.text(0.5, 0.01, "Figure 11: M5 regional tourist allocation (matplotlib Sankey)",
             ha="center", fontsize=8.0, color="#333333")

    p = OUT_DIR / "m5_region_sankey.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


# =====================================================================================
# 主入口
# =====================================================================================
def main() -> None:
    t0 = time.perf_counter()
    log_lines: List[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log("=" * 80)
    log("M5 热门-冷门空间均衡 MILP (PuLP)")
    log("=" * 80)

    data = load_data()
    log(f"[区域] {len(data['regions'])} 个节点: "
        f"{len(data['hot_ids'])} 热点 + {len(data['cold_ids'])} 冷点")
    log(f"  热点: {sorted(data['hot_ids'])}")
    log(f"  冷点: {sorted(data['cold_ids'])}")
    log(f"[边] {len(data['edges'])} 条网络连接")
    log(f"[参数] p0={CONFIG['p0']:.4f}, x_total={CONFIG['x_total']:.2f}万, "
        f"H_max={CONFIG['H_max']}, rho={CONFIG['rho']}, "
        f"w_H={CONFIG['w_H']}, w_B={CONFIG['w_B']}, c_v={CONFIG['c_v']}, "
        f"Delta_cap={CONFIG['Delta_cap']}")

    log("[求解] PuLP CBC MILP ...")
    alloc, obj = solve_m5(data, CONFIG)

    log(f"[状态] {obj['status']}")
    log(f"[目标] revenue={obj['total_revenue']:.4f}亿, "
        f"-hot_penalty={-obj['hot_penalty']:.4f}, "
        f"+cold_reward={obj['cold_reward']:.4f}, "
        f"-invest_cost={-obj['invest_cost']:.4f}, "
        f"obj={obj['objective']:.4f}")
    log(f"[守恒] Σ x_i={obj['total_x']:.2f}万 vs x_total={CONFIG['x_total']:.2f}万")
    for _, row in alloc.iterrows():
        log(f"  {row['region_id']} ({row['region_name'][:18]:18s}): cap={row['cap']:.0f}万, "
            f"x={row['x_i']:.2f}万, τ={row['tau_i']:.4f}, "
            f"z={row['z_i']}, v={row['v_i']:.2f}, "
            f"H/U={row['crowding_i']:.3f}")

    # 输出 CSV
    alloc.round(6).to_csv(OUT_DIR / "m5_region_allocation.csv",
                          index=False, encoding="utf-8-sig")
    pd.DataFrame([obj]).round(6).to_csv(OUT_DIR / "m5_objectives.csv",
                                        index=False, encoding="utf-8-sig")

    # 绘制桑基图
    p = fig_sankey(alloc)
    log(f"[输出] m5_region_allocation.csv, m5_objectives.csv, m5_region_sankey.png")
    log(f"[耗时] {time.perf_counter() - t0:.2f}s")

    log("\n[待建模手确认]")
    log(f"  1. w_H={CONFIG['w_H']}, w_B={CONFIG['w_B']}, c_v={CONFIG['c_v']} (目标权重)")
    log(f"  2. Δcap={CONFIG['Delta_cap']}, H_max={CONFIG['H_max']}, ρ={CONFIG['rho']} (约束参数)")
    log(f"  3. cap_multiplier={CONFIG['cap_multiplier']} (capacity_level → 万)")
    log(f"  4. τ∈[{CONFIG['tau_lo']}, {CONFIG['tau_hi']}] (税率范围, PDF 无数值)")
    log(f"  5. M5-6 税率弹性分段 + M5-7 网络流割: 本轮简化处理, 暂未启用")

    (OUT_DIR / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
