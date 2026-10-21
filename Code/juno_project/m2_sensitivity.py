# -*- coding: utf-8 -*-
r"""M2 鲁棒解参数扰动敏感性分析（固定最优项目组合 y*）

口径 (用户确认)
----------------
固定 M2 最优组合 y* = 10100000, 对 c_k、eta_k、rho_E、lambda_* 做单因素扰动,
重新计算 M2-2 ~ M2-7, 观察鲁棒目标 z 的响应:
  * 中心差分弹性 S_z: ±10% 扰动
  * 龙卷风/响应曲线: ±20% 扰动
不重新优化 y* (按口径固定), 因此"最优性稳健性"不在本脚本范围内。

扰动因子 (8 项)
---------------
c_unit(c_k 成本换算)、eta(项目反哺效率)、rho_E(生态折减)、
lambda_I / lambda_E / lambda_C(反哺权重)、lambda_L(游客不足罚)、lambda_B(生态超阈罚)

输出 (results/m2/)
------------------
* m2_sensitivity.csv             每个扰动点的 z / U_s / 预算可行性
* m2_sensitivity_elasticity.csv  弹性 S_z 与排序
* m2_sensitivity.png             弹性排序 + z 响应曲线
* sensitivity_log.txt            口径、基准点、异常与待确认
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR: Path = Path(__file__).resolve().parent
OUT_DIR: Path = BASE_DIR / "results" / "m2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import m2_robust as m2

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
DPI: int = 300

C_MAIN, C_SEC, C_EMPH, C_ALT, C_GREY = m2.C_MAIN, m2.C_SEC, m2.C_EMPH, m2.C_ALT, m2.C_GREY

MS_CONFIG: Dict[str, Any] = {
    "perturb": 0.10,        # 中心差分弹性步长
    "range": 0.20,          # 曲线/龙卷风端点
    "factors": ["c_unit", "eta", "rho_E", "lambda_I", "lambda_E",
                "lambda_C", "lambda_L", "lambda_B"],
}

FACTOR_NAME: Dict[str, str] = {
    "c_unit": r"$c_k$ 成本", "eta": r"$\eta_k$ 效率", "rho_E": r"$\rho_E$ 折减",
    "lambda_I": r"$\lambda_I$ 基建", "lambda_E": r"$\lambda_E$ 生态",
    "lambda_C": r"$\lambda_C$ 社区", "lambda_L": r"$\lambda_L$ 缺客罚",
    "lambda_B": r"$\lambda_B$ 超阈罚",
}


def load_baseline() -> Dict[str, Any]:
    """复用 m2_robust 的输入装载与情景抽样, 并读回已求解的最优 y*。"""
    ins = m2.load_inputs()
    scen, _ = m2.make_scenarios(ins)
    proj_path = OUT_DIR / "m2_projects.csv"
    if not proj_path.exists():
        raise FileNotFoundError(f"未找到 {proj_path}, 请先运行 m2_robust.py")
    sel = pd.read_csv(proj_path, encoding="utf-8-sig").set_index("project_id")["selected"]
    y = np.asarray([int(sel.loc[pid]) for pid in ins["projects"]["project_id"]], dtype=int)
    return {"ins": ins, "scen": scen, "y": y}


def build_variant(ins: Dict[str, Any], name: str, factor: float
                 ) -> Tuple[Dict[str, Any], pd.DataFrame, float]:
    """构造单因素扰动后的 cfg / 项目表, 返回 (cfg, proj, 参数基准值)。"""
    cfg = dict(m2.M2_CONFIG)
    proj = ins["projects"].copy()
    if name == "c_unit":
        base = float(m2.M2_CONFIG["cost_unit_100m"])
        proj["cost_100m"] = (proj["cost_100m"] * factor).round(6)
    elif name == "eta":
        base = float(ins["projects"]["eta"].mean())
        proj["eta"] = proj["eta"] * factor
    else:
        base = float(m2.M2_CONFIG[name])
        cfg[name] = base * factor
    return cfg, proj, base


def run_point(y: np.ndarray, scen: pd.DataFrame, ins: Dict[str, Any],
              name: str, factor: float) -> Dict[str, Any]:
    cfg, proj, base = build_variant(ins, name, factor)
    ev = m2.evaluate_selection(y, scen, ins, cfg=cfg, proj=proj)
    return {"factor": name, "factor_name": FACTOR_NAME[name],
            "perturb": factor - 1.0, "param_base": base,
            "param_value": base * factor,
            "cost_100m": ev["cost_100m"], "budget_100m": ev["budget_100m"],
            "feasible": ev["feasible_budget"],
            "dI": ev["dI"], "dE": ev["dE"], "dS": ev["dS"],
            "I_s": ev["I_s"], "C_env_s": ev["C_env_s"], "S_s": ev["S_s"],
            "U_low": ev["U"][0], "U_mid": ev["U"][1], "U_high": ev["U"][2],
            "z": ev["z"],
            "x_low": ev["x_new"][0], "x_mid": ev["x_new"][1], "x_high": ev["x_new"][2]}


def elasticity(y_minus: float, y_base: float, y_plus: float, rel: float) -> float:
    vals = (y_minus, y_base, y_plus)
    if any(not np.isfinite(v) for v in vals) or abs(y_base) < 1e-12:
        return float("nan")
    return float((y_plus - y_minus) / (2.0 * y_base) / rel)


def make_figure(elast: pd.DataFrame, rows: pd.DataFrame, z0: float) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.85, bottom=0.28, wspace=0.55)

    # ---- (a) 弹性 S_z 排序 ----
    ax = axes[0]
    d = elast.sort_values("S_z")
    ypos = np.arange(len(d))
    colors = [C_EMPH if abs(v) > 0.05 else C_MAIN for v in d["S_z"]]
    ax.barh(ypos, d["S_z"], color=colors, alpha=0.85, height=0.62)
    ax.set_yticks(ypos)
    ax.set_yticklabels(d["factor_name"], fontsize=7.0)
    ax.axvline(0, color="#555555", lw=0.8)
    for yy, v in zip(ypos, d["S_z"]):
        ax.text(v + (0.03 if v >= 0 else -0.03), yy, "%.2f" % v,
                va="center", ha="left" if v >= 0 else "right", fontsize=5.6,
                color="#333333")
    lo = min(float(d["S_z"].min()) - 0.55, -0.05)
    hi = max(float(d["S_z"].max()) + 0.55, 0.1)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("中心差分弹性 $S_z$ (±10%)", fontsize=8.2)
    ax.set_title("(a) 鲁棒目标的单因素弹性", fontsize=9.2)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(labelsize=7.2)

    # ---- (b) z 对扰动的响应曲线 (仅画 |S|>0.05 的因子) ----
    ax = axes[1]
    top = elast[elast["S_z"].abs() > 0.05].sort_values("S_z", ascending=False)
    palette = [C_EMPH, C_SEC, C_MAIN, C_ALT, "#937860", "#8172B3"]
    levels = [-MS_CONFIG["range"], -MS_CONFIG["perturb"], 0.0,
              MS_CONFIG["perturb"], MS_CONFIG["range"]]
    for k, (_, r) in enumerate(top.iterrows()):
        sub = rows[rows["factor"] == r["factor"]].sort_values("perturb")
        col = palette[k % len(palette)]
        ax.plot(sub["perturb"] * 100, sub["z"], marker="o", ms=3.0, lw=1.3,
                color=col, label=r["factor_name"], zorder=3)
        bad = sub[sub["feasible"] == 0]
        if len(bad):
            ax.plot(bad["perturb"] * 100, bad["z"], marker="x", ms=5.5, lw=0,
                    color=C_EMPH, zorder=4)
    ax.axhline(z0, color=C_GREY, lw=0.9, ls=":", zorder=1)
    ax.text(-24.5, z0 - 0.004, "基准 $z$=%.4f" % z0, fontsize=6.0, color=C_GREY,
            ha="left", va="top")
    ax.set_xticks([v * 100 for v in levels])
    ax.set_xlim(-26, 26)
    ax.set_xlabel("参数扰动 (%)", fontsize=8.2)
    ax.set_ylabel("鲁棒目标 $z$", fontsize=8.2)
    ax.set_title("(b) $z$ 的扰动响应 (固定 $y^*$)", fontsize=9.2)
    ax.legend(fontsize=5.9, loc="upper left", ncol=2, framealpha=0.9,
              handlelength=1.4, columnspacing=0.9)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7.2)
    ax.text(0.97, 0.03, "× = 违反 M2-2 鲁棒预算\n其余 %d 项 $|S_z|<0.05$ 未画"
            % (len(elast) - len(top)),
            transform=ax.transAxes, fontsize=5.7, color=C_GREY, ha="right", va="bottom",
            linespacing=1.3)

    fig.text(0.5, 0.055, "Figure 8: M2 single-factor sensitivity of robust objective "
                         "under fixed project selection", ha="center", fontsize=8.4,
             color="#333333")
    path = OUT_DIR / "m2_sensitivity.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    t0 = time.perf_counter()
    log_lines: List[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    base = load_baseline()
    ins, scen, y = base["ins"], base["scen"], base["y"]
    rel = float(MS_CONFIG["perturb"])
    rng = float(MS_CONFIG["range"])

    z_base = m2.evaluate_selection(y, scen, ins)["z"]

    log("=" * 84)
    log("M2 参数扰动敏感性 (固定 y*, 单因素)")
    log("=" * 84)
    log(f"[基准] y*={''.join(map(str, y))}, z={z_base:.6f}, 步长 ±{rel:.0%}, 端点 ±{rng:.0%}")
    log(f"[口径] 固定最优组合不重优化; c_k/eta_k/rho_E/lambda_* 逐项扰动, 其余保持基准")

    rows = []
    for name in MS_CONFIG["factors"]:
        for p in (-rng, -rel, 0.0, rel, rng):
            rows.append(run_point(y, scen, ins, name, 1.0 + p))
    tab = pd.DataFrame(rows)
    tab["z_rel_pct"] = (tab["z"] / z_base - 1.0) * 100.0
    tab["z_rel_pct"] = tab["z_rel_pct"].round(4)
    tab = tab.round(6)
    tab.to_csv(OUT_DIR / "m2_sensitivity.csv", index=False, encoding="utf-8-sig")

    el = []
    for name in MS_CONFIG["factors"]:
        sub = tab[tab["factor"] == name].set_index("perturb")
        z_m, z_p = float(sub.loc[-rel, "z"]), float(sub.loc[rel, "z"])
        f_m, f_p = int(sub.loc[-rel, "feasible"]), int(sub.loc[rel, "feasible"])
        el.append({
            "factor": name, "factor_name": FACTOR_NAME[name],
            "param_base": float(sub.loc[0.0, "param_base"]),
            "z_base": z_base, "z_minus10": z_m, "z_plus10": z_p,
            "S_z": elasticity(z_m, z_base, z_p, rel) if (f_m and f_p) else np.nan,
            "S_U_low": elasticity(float(sub.loc[-rel, "U_low"]),
                                  float(sub.loc[0.0, "U_low"]), float(sub.loc[rel, "U_low"]), rel),
            "S_U_mid": elasticity(float(sub.loc[-rel, "U_mid"]),
                                  float(sub.loc[0.0, "U_mid"]), float(sub.loc[rel, "U_mid"]), rel),
            "S_U_high": elasticity(float(sub.loc[-rel, "U_high"]),
                                   float(sub.loc[0.0, "U_high"]), float(sub.loc[rel, "U_high"]), rel),
            "feasible_minus10": f_m, "feasible_plus10": f_p,
            "z_rel_minus20": float(sub.loc[-rng, "z_rel_pct"]),
            "z_rel_plus20": float(sub.loc[rng, "z_rel_pct"]),
            "feasible_plus20": int(sub.loc[rng, "feasible"]),
        })
    elast = pd.DataFrame(el)
    elast["abs_S_z"] = elast["S_z"].abs()
    elast = elast.sort_values("abs_S_z", ascending=False).reset_index(drop=True)
    elast["rank"] = np.arange(1, len(elast) + 1)
    elast = elast.round(6)
    elast.to_csv(OUT_DIR / "m2_sensitivity_elasticity.csv", index=False,
                 encoding="utf-8-sig")

    log("[弹性] 按 |S_z| 排序:")
    for _, r in elast.iterrows():
        log(f"  {r['rank']}. {r['factor']:<9s} {r['factor_name']:<14s} "
            f"S_z={r['S_z']:+.4f}  z(-10%)={r['z_minus10']:.4f} "
            f"z(+10%)={r['z_plus10']:.4f}  Δz(±20%)="
            f"[{r['z_rel_minus20']:+.2f}%, {r['z_rel_plus20']:+.2f}%]")

    # ---- 预算可行性边界 ----
    cu = tab[tab["factor"] == "c_unit"].set_index("perturb")
    feas = cu.index[cu["feasible"] == 1].max()
    log(f"[预算] c_unit 扰动下 y*=10100000 的鲁棒预算可行上限: 成本 "
        f"{cu.loc[0.0, 'cost_100m']:.4f} → 预算 {cu.loc[0.0, 'budget_100m']:.4f} 亿; "
        f"+20% 成本={cu.loc[rng, 'cost_100m']:.4f} 亿 -> "
        f"{'可行' if cu.loc[rng, 'feasible'] else '不可行 (违反 M2-2)'}; "
        f"可行扰动上限 {feas:+.0%}")
    log("[口径观察 3] lambda_C (ΔS≡0) 与 lambda_B (C_env,s=0.782 < E_max=0.95) "
        "弹性严格为 0 —— 两项在 M2 中均为非活跃项, 与 M1 v2.0 非紧约束结论同源")
    log("[口径观察 4] c_unit 与 eta 的 S_z 完全相同: 两者都只经 ΔI/ΔE 线性缩放进入 U, "
        "项目成本不进目标, 仅在 M2-2 中决定可行性 —— 成本差异对 z 无直接作用")
    log(f"[输出] m2_sensitivity.csv ({len(tab)} 行), m2_sensitivity_elasticity.csv "
        f"({len(elast)} 行), m2_sensitivity.png")

    p = make_figure(elast, tab, z_base)
    log(f"[图] {p.name}")
    log(f"[耗时] {time.perf_counter() - t0:.2f}s")
    (OUT_DIR / "sensitivity_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
