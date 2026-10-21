# -*- coding: utf-8 -*-
"""O 奖风格演示图: 用现有 m1_solver v2.0 的数据, 画"权衡前沿 + 杠杆扫描"升级版图。
仅作演示, 非交付代码。"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent.parent   # juno_project 根
sys.path.insert(0, str(PROJECT))
import m1_solver as m1

OUT = PROJECT / "results" / "m1" / "fig_Ostyle_demo.png"

params = dict(m1.DEFAULT_PARAMS)
df = m1.load_scenarios()
row = df[df["scenario_id"] == "S00_baseline"].iloc[0]
ctx = m1.build_context(row, params)
log = lambda msg: None

# ---------- 1) 32 组合全解, 提取可行集与 Pareto 前沿 ----------
res = []
for i in range(32):
    combo = tuple((i >> j) & 1 for j in range(5))
    r = m1.solve_one_combo(ctx, combo, params, log)
    res.append(r)
feas = [r for r in res if r["feasible_flag"] == 1]
best = max(feas, key=lambda r: r["objective_value"])

def front(pts, key):
    p = sorted(pts, key=lambda r: r["R"], reverse=True)
    out = []
    for r in p:
        if not out or r[key] < out[-1][key]:
            out.append(r)
    return out

front_env, front_soc = front(feas, "C_env"), front(feas, "C_soc")

# ---------- 2) 税率杠杆扫描 (固定最优措施组合, τ ∈ [0.09, 0.15]) ----------
combo = best["combo"]
taus = np.linspace(0.09, 0.15, 61)
scan = []
for tau in taus:
    v = np.array([tau, best["delta_I"], best["delta_E"], best["delta_C"], best["xd"]])
    r = m1.evaluate(ctx, v, combo, params)
    if r is not None:
        scan.append((tau, r))
ts = np.array([s[0] for s in scan])
Rs = np.array([s[1]["R"] for s in scan])
Js = np.array([s[1]["objective_value"] for s in scan])
Cenvs = np.array([s[1]["C_env"] for s in scan])
Csocs = np.array([s[1]["C_soc"] for s in scan])

# ---------- 3) 绘图: 2 面板 O 奖风格 ----------
plt.rcParams.update({"font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "axes.edgecolor": "#888888",
                     "axes.linewidth": 0.8})
C1, C2, C3, C4 = "#4C72B0", "#DD8452", "#C44E52", "#55A868"

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=200,
                         gridspec_kw={"wspace": 0.28})

# --- 面板 1: 经济-环境权衡 + Pareto 前沿 + 填充 ---
ax = axes[0]
ax.scatter([r["R"] for r in feas], [r["C_env"] for r in feas], s=26,
           color=C1, alpha=0.9, edgecolor="white", lw=0.5, zorder=3,
           label="可行组合 (n=%d)" % len(feas))
fr = [r["R"] for r in front_env]; fc = [r["C_env"] for r in front_env]
ax.plot(fr, fc, color=C2, lw=2.0, marker="o", ms=4.5, zorder=4,
        label="Pareto 前沿 (收入↑ / 环境成本↓)")
ax.fill_between(fr, fc, fc[-1] - 0.06, color=C2, alpha=0.14, zorder=1)
ax.scatter([best["R"]], [best["C_env"]], marker="*", s=170, color=C3,
           edgecolor="white", lw=0.6, zorder=5, label="加权和最优折中解")
ax.annotate("($R$=%.3f, $C_{env}$=%.3f)" % (best["R"], best["C_env"]),
            (best["R"], best["C_env"]), textcoords="offset points",
            xytext=(8, 8), fontsize=8.5, color=C3)
ax.set_xlabel("旅游总收入 $R$ (亿美元)"); ax.set_ylabel("环境成本 $C_{env}$")
ax.set_title("(a) 经济-环境权衡 (Pareto 前沿)")
ax.legend(fontsize=8, loc="upper right", framealpha=0.92); ax.grid(alpha=0.25)

# --- 面板 2: 税率杠杆扫描 (量减价升机制) ---
ax = axes[1]
ax.plot(ts, Js, color=C4, lw=2.2, zorder=3, label="综合效益 $J$")
ax.fill_between(ts, Js, Js.min() - 0.01, color=C4, alpha=0.14, zorder=1)
ax.plot(ts, Rs, color=C1, lw=1.6, ls="--", zorder=3, label="旅游收入 $R$")
ax.plot(ts, Csocs, color=C2, lw=1.4, ls=":", zorder=3, label="社会成本 $C_{soc}$")
i_best = int(np.argmax(Js))
ax.scatter([ts[i_best]], [Js[i_best]], marker="*", s=170, color=C3, zorder=5,
           edgecolor="white", lw=0.6, label="最优税率 τ=%.3f" % ts[i_best])
ax.axvline(ts[i_best], color=C3, lw=0.9, ls="--", alpha=0.7)
ax.set_xlabel("综合税率 $\\tau$"); ax.set_ylabel("数值")
ax.set_title("(b) 税率杠杆: 量减价升 vs 成本上升 (S00, 固定措施组合)")
ax.legend(fontsize=8, loc="lower right", framealpha=0.92); ax.grid(alpha=0.25)

fig.suptitle("S00_baseline · M1 v2.0 权衡与政策杠杆 (O 奖风格演示)", fontsize=13, y=1.00)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("已生成:", OUT)
print("S00: R=%.4f, C_env=%.4f, tau=%.4f, J=%.4f, x=%.2f" %
      (best["R"], best["C_env"], best["tau"], best["objective_value"], best["x"]))
