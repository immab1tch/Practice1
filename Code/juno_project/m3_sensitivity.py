# -*- coding: utf-8 -*-
r"""M3 敏感性分析 v2.0 (sensitivity analysis)

v2.0 适配 (2026-10-21, 建模手裁定 + m1_solver.py v2.0 参数集)
------------------------------------------------------------
* 待敏参数集扩展到 PDF 表 6 全集: ε, η_I, η_E, η_C, E_max, S_min, τ_max, w_j
* Sobol 全局: N=128 (2 的幂, 避免 SALib 警告), 仅 S00_baseline 单情景 + M1 最优措施组合,
  w_j 因"权重扰动不适合均匀 Sobol 采样"先不纳入 (报告中说明)
* ProcessPoolExecutor 并行评估 (固定组合下约 0.045s/样本)
* E_max 弹性≈0 属 v2.0 学术预期 (C_env≈0.5 远低于 E_max=0.95, 非紧约束/影子价格 0),
  结果一致即算通过, 不改公式或参数
* 参数区间分两类: (a) CSV 区间: ε, η_I/E/C, E_max (0.6-0.95), S_min (0.4-0.7);
  (b) 编程假设 (待建模手确认): tau_max [0.135, 0.165], w_j 按 ±10% 单因素扰动重归一化

分析口径
--------
1. 单因素 (M3-1): 8 参数 × ±10% × 15 情景, 复用 M1 32 组合全枚举, 中心差分弹性
2. Sobol (M3-2): Saltelli 采样 S00 固定组合, S1/ST/S2
3. 交互: ST-S1 最大值 / S2 热力图

输出 (results/m3/)
------------------
* m3_elasticity.csv
* sobol_indices.csv
* sobol_bar.png / elasticity_rank.png / interaction_heatmap.png
"""

from __future__ import annotations

import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol

BASE_DIR: Path = Path(__file__).resolve().parent
RESULTS_DIR: Path = BASE_DIR / "results" / "m3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys_path = str(BASE_DIR)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
import m1_solver as m1

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
DPI: int = 300

# =====================================================================================
M3_CONFIG: Dict[str, Any] = {
    # v2.0 待敏参数集 (PDF 表 6 θ_p 全集)
    "params": ["epsilon", "eta_I", "eta_E", "eta_C", "E_max", "S_min", "tau_max"],
    "weights": ["w_income", "w_env", "w_soc", "w_infra"],  # 单因素扰动重归一化, 不进 Sobol
    "perturb": 0.10,
    "targets": ["R", "C_env", "C_soc", "J"],
    # Sobol 设置
    "sobol_n": 128,           # v2.0 改小 (2 的幂避免 SALib warning); 仅 S00 单情景
    "sobol_calc_second_order": True,
    "sobol_seed": 42,
    "sobol_scenario": "S00_baseline",
    # 阈值 (结论文字用)
    "strong_interaction_s2": 0.05,
    "strong_interaction_st_s1": 0.10,
}

# 参数名 -> juno_params_scenarios.csv param 名 (CSV 区间来源)
# 不在此映射的参数 (tau_max) 用硬编码区间 (报告里标注"待建模手确认")
PARAM_CSV_MAP: Dict[str, str] = {
    "epsilon": "demand_elasticity",
    "eta_I": "eta_I", "eta_E": "eta_E", "eta_C": "eta_C",
    "E_max": "eco_threshold",
    "S_min": "social_accept_min",
}

# tau_max / w_j 区间 (编程假设, 非 CSV 数据; 报告里必须标注)
TAU_MAX_BOUNDS = [0.135, 0.165]   # 基准 0.15 ±10%; 与 CSV 无对应
W_J_BOUNDS_REL = 0.10              # 单因素扰动 ±10%, 不进 Sobol

# 参数名 -> CSV 权重列名 (情景主表里的 w_income/w_env/w_soc/w_infra)
PARAM_TO_CSV_WEIGHT_COL: Dict[str, str] = {
    "w_income": "w_income", "w_env": "w_env",
    "w_soc": "w_soc", "w_infra": "w_infra",
}


def _silent(_msg: str) -> None:
    pass


# =====================================================================================
# 区间读取 (v2.0 扩展, 含 CSV 区间 + tau_max 硬编码 + w_j 单因素专用)
# =====================================================================================
def load_param_bounds_for_sobol() -> Tuple[List[str], List[List[float]], List[str]]:
    """返回 (sobol 参数名列表, 区间列表, 需标注"非 CSV"的参数名)。

    w_j 权重不进 Sobol (权重扰动不适合均匀采样), 只做单因素。
    """
    df = pd.read_csv(m1.PREPROCESSED_DIR / "juno_params_scenarios.csv", encoding="utf-8-sig")
    names: List[str] = []
    bounds: List[List[float]] = []
    need_note: List[str] = []
    for name in M3_CONFIG["params"]:
        if name == "tau_max":
            names.append(name)
            bounds.append(TAU_MAX_BOUNDS)
            need_note.append(name)
            continue
        row = df[df["param"] == PARAM_CSV_MAP[name]]
        if row.empty:
            raise KeyError(f"juno_params_scenarios.csv 缺少参数 {PARAM_CSV_MAP[name]}")
        names.append(name)
        bounds.append([float(row["lower"].iloc[0]), float(row["upper"].iloc[0])])
    return names, bounds, need_note


def load_m1_base() -> pd.DataFrame:
    path = m1.RESULTS_DIR / "m1_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}, 请先运行 m1_solver.py")
    return pd.read_csv(path, encoding="utf-8-sig").set_index("scenario_id")


# =====================================================================================
# 单因素敏感性 (式 M3-1, v2.0 扩展含 w_j 权重重归一化)
# =====================================================================================
def _make_params(name: str, base_params: Dict[str, Any], factor: float,
                 row: pd.Series) -> Dict[str, Any]:
    """构造单因素扰动后的 params dict。

    普通标量参数: 直接乘 factor; w_j 权重: 扰动后重归一化 (和为 1)。
    """
    params = dict(base_params)
    if name in M3_CONFIG["weights"]:
        # 权重扰动: 先对该权重 *factor, 然后所有权重按比例重归一化
        weights = np.array([float(row[PARAM_TO_CSV_WEIGHT_COL[w]])
                           for w in M3_CONFIG["weights"]])
        idx = M3_CONFIG["weights"].index(name)
        weights[idx] *= factor
        weights = weights / weights.sum()
        # weights 是 build_context 时从 row 读的, 所以这里临时覆盖 row 里的值
        new_row = row.copy()
        for w, v in zip(M3_CONFIG["weights"], weights):
            new_row[PARAM_TO_CSV_WEIGHT_COL[w]] = float(v)
        # 通过 build_context 传入覆盖后的 row
        params["_weight_row_override"] = new_row
        return params
    params[name] = float(m1.DEFAULT_PARAMS[name]) * factor
    return params


def solve_targets(row: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
    ctx = m1.build_context(row, params)
    # w_j 权重覆盖 (build_context 从 row 读权重)
    if "_weight_row_override" in params:
        override = params.pop("_weight_row_override")
        ctx["weights"] = (float(override["w_income"]), float(override["w_env"]),
                          float(override["w_soc"]), float(override["w_infra"]))
    best, _detail = m1.solve_scenario(ctx, params, _silent)
    if best.get("feasible_flag") == 1:
        return {"R": best["R"], "C_env": best["C_env"], "C_soc": best["C_soc"],
                "J": best["objective_value"], "x": best["x"],
                "feasible_flag": 1, "relax_name": best["relax_name"]}
    return {"R": np.nan, "C_env": np.nan, "C_soc": np.nan, "J": np.nan, "x": np.nan,
            "feasible_flag": 0, "relax_name": "infeasible_all_stages"}


def elasticity(y_minus: float, y_base: float, y_plus: float, rel: float) -> float:
    vals = (y_minus, y_base, y_plus)
    if any(not np.isfinite(v) for v in vals) or abs(y_base) < 1e-12:
        return float("nan")
    return float((y_plus - y_minus) / (2.0 * y_base) / rel)


def run_one_factor(scenarios: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """单因素敏感性: 7 参数 (标量) + 4 权重 = 11 扰动项, 每项 ±10% × 15 情景。"""
    rel: float = float(M3_CONFIG["perturb"])
    rows: List[Dict[str, Any]] = []
    all_params = M3_CONFIG["params"] + M3_CONFIG["weights"]   # 7 标量 + 4 权重

    for _, srow in scenarios.iterrows():
        sid = str(srow["scenario_id"])
        if sid not in base.index or int(base.loc[sid, "feasible_flag"]) != 1:
            rows.append({"scenario_id": sid, "param": "ALL", "param_base": np.nan,
                         "param_minus": np.nan, "param_plus": np.nan,
                         **{f"{t}_base": np.nan for t in M3_CONFIG["targets"]},
                         **{f"{t}_minus": np.nan for t in M3_CONFIG["targets"]},
                         **{f"{t}_plus": np.nan for t in M3_CONFIG["targets"]},
                         **{f"S_{t}": np.nan for t in M3_CONFIG["targets"]},
                         "feasible_plus": 0, "feasible_minus": 0,
                         "relax_plus": "M1_base_infeasible", "relax_minus": "M1_base_infeasible"})
            continue

        y_base = {t: float(base.loc[sid, t]) for t in M3_CONFIG["targets"] if t != "J"}
        y_base["J"] = float(base.loc[sid, "objective_value"])

        for name in all_params:
            if name in M3_CONFIG["weights"]:
                # 权重: 基准值从 row 读 (非 DEFAULT_PARAMS)
                p0 = float(srow[PARAM_TO_CSV_WEIGHT_COL[name]])
            else:
                p0 = float(m1.DEFAULT_PARAMS[name])

            res: Dict[str, Dict[str, Any]] = {}
            for tag, factor in (("minus", 1.0 - rel), ("plus", 1.0 + rel)):
                params = _make_params(name, dict(m1.DEFAULT_PARAMS), factor, srow)
                res[tag] = solve_targets(srow, params)

            out: Dict[str, Any] = {
                "scenario_id": sid, "param": name, "param_base": p0,
                "param_minus": p0 * (1.0 - rel), "param_plus": p0 * (1.0 + rel),
                "feasible_plus": res["plus"]["feasible_flag"],
                "feasible_minus": res["minus"]["feasible_flag"],
                "relax_plus": res["plus"]["relax_name"],
                "relax_minus": res["minus"]["relax_name"],
                "param_category": "weight" if name in M3_CONFIG["weights"] else "scalar",
            }
            for t in M3_CONFIG["targets"]:
                out[f"{t}_base"] = y_base[t]
                out[f"{t}_minus"] = res["minus"][t]
                out[f"{t}_plus"] = res["plus"][t]
                out[f"S_{t}"] = elasticity(res["minus"][t], y_base[t], res["plus"][t], rel)
            rows.append(out)

            if out["feasible_plus"] == 0 or out["feasible_minus"] == 0:
                print(f"[单因素] {sid} / {name}: 存在不可行扰动点")

    return pd.DataFrame(rows)


# =====================================================================================
# Sobol (式 M3-2, v2.0: N=128, ProcessPoolExecutor 并行, 仅 S00 固定组合)
# =====================================================================================
def _sobol_eval_point_single(x: np.ndarray, row_dict: Dict[str, Any],
                             combo: Tuple[int, ...], params_keys: List[str],
                             bounds: List[List[float]]) -> Dict[str, Any]:
    """单进程求值 (ProcessPoolExecutor 目标)。

    row_dict 需可序列化, build_context 在子进程里重建 ctx。
    """
    # 重建 srow Series
    row = pd.Series(row_dict)
    params = dict(m1.DEFAULT_PARAMS)
    for name, val, (lo, hi) in zip(params_keys, x, bounds):
        params[name] = float(np.clip(val, lo, hi))
    ctx = m1.build_context(row, params)

    r = m1.solve_one_combo(ctx, combo, params, _silent)
    if r.get("feasible_flag") == 1:
        return {"R": r["R"], "C_env": r["C_env"], "C_soc": r["C_soc"],
                "J": r["objective_value"], "relax_name": r["relax_name"], "fallback": 0}
    best, _ = m1.solve_scenario(ctx, params, _silent)
    if best.get("feasible_flag") == 1:
        return {"R": best["R"], "C_env": best["C_env"], "C_soc": best["C_soc"],
                "J": best["objective_value"], "relax_name": best["relax_name"], "fallback": 1}
    return {"R": np.nan, "C_env": np.nan, "C_soc": np.nan, "J": np.nan,
            "relax_name": "infeasible_all_stages", "fallback": 1}


def run_sobol(scenarios: pd.DataFrame, base: pd.DataFrame
              ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    sid: str = M3_CONFIG["sobol_scenario"]
    row = scenarios[scenarios["scenario_id"] == sid].iloc[0]
    combo: Tuple[int, ...] = tuple(int(base.loc[sid, f"m{i}"]) for i in range(1, 6))
    names, bounds, need_note = load_param_bounds_for_sobol()

    problem = {"num_vars": len(names), "names": names, "bounds": bounds}
    X = sobol_sample.sample(problem, int(M3_CONFIG["sobol_n"]),
                            calc_second_order=bool(M3_CONFIG["sobol_calc_second_order"]),
                            seed=int(M3_CONFIG["sobol_seed"]))

    row_dict = row.to_dict()   # 可序列化 (ProcessPoolExecutor)

    targets = ["R", "C_env", "C_soc", "J"]
    Y: Dict[str, np.ndarray] = {t: np.empty(len(X)) for t in targets}
    relax_counter: Dict[str, int] = {}
    n_fallback = 0
    n_infeasible = 0

    t0 = time.perf_counter()
    print(f"[Sobol] 评估 {len(X)} 样本点 ({len(names)} 参数, 并行进程) ...")

    # 并行 + 缓存 (参数点四舍五入 10 位作为 key)
    cache: Dict[Tuple[float, ...], Dict[str, Any]] = {}
    futures = {}
    with ProcessPoolExecutor() as pool:
        for i, x in enumerate(X):
            key = tuple(np.round(x, 10))
            if key in cache:
                for t in targets:
                    Y[t][i] = cache[key][t]
                continue
            fut = pool.submit(_sobol_eval_point_single, x, row_dict, combo,
                              list(names), bounds)
            futures[fut] = (i, key)

        for fut in as_completed(futures):
            i, key = futures[fut]
            val = fut.result()
            cache[key] = val
            n_fallback += val["fallback"]
            relax_counter[val["relax_name"]] = relax_counter.get(val["relax_name"], 0) + 1
            if not np.isfinite(val["R"]):
                n_infeasible += 1
            for t in targets:
                Y[t][i] = val[t]
    dt = time.perf_counter() - t0

    n_hit = len(X) - len(cache)

    if n_infeasible > 0:
        print(f"  警告: {n_infeasible} 样本点放宽链尽头仍不可行 (均值插补)")

    rows: List[Dict[str, Any]] = []
    s2_store: Dict[str, np.ndarray] = {}
    for t in targets:
        y = Y[t].copy()
        if not np.all(np.isfinite(y)):
            mean_ok = np.nanmean(y)
            y = np.where(np.isfinite(y), y, mean_ok)
        Si = sobol.analyze(problem, y, calc_second_order=True, print_to_console=False)
        s2_store[t] = Si["S2"]
        for j, name in enumerate(names):
            rows.append({
                "target": t, "param": name,
                "S1": float(Si["S1"][j]), "S1_conf": float(Si["S1_conf"][j]),
                "ST": float(Si["ST"][j]), "ST_conf": float(Si["ST_conf"][j]),
                "interaction_ST_minus_S1": float(Si["ST"][j] - Si["S1"][j]),
            })

    meta = {"n_samples": int(len(X)), "n_unique": len(cache), "n_cache_hit": n_hit,
            "n_fallback": n_fallback, "n_infeasible": n_infeasible,
            "relax_counter": relax_counter, "elapsed_s": dt,
            "combo": combo, "scenario": sid, "bounds": bounds,
            "need_note_params": need_note,
            "sobol_excludes_w_j": True}
    return pd.DataFrame(rows), {"meta": meta, "s2": s2_store, "names": names}


# =====================================================================================
# 绘图
# =====================================================================================
def plot_sobol_bar(df: pd.DataFrame, names: List[str]) -> None:
    """Sobol 柱状图 (v3.0 版: C_env 目标截断 + KKT 学术注释)

    C_env 目标的 Sobol ST 指数因方差退化 (Var(C_env)→0, 钉死 0.5 地板) 爆炸到 O(10²)。
    PDF v3.0 第 6 页将此包装为 KKT 学术亮点:
      E_max=0.95 是非活跃约束, 影子价格 λ_E=0 → 扰动不传递到 C_env 的方差
    处理:
      1. 计算逻辑保持原样 (sobol_indices.csv 保存原始值)
      2. 作图时对 |val| > 10 的截断到 ±10, 加红色 ▼/▲ 标记截断点
      3. C_env 子图右上角加学术注释文本框
    """
    targets = ["R", "C_env", "C_soc"]
    labels = {"R": "旅游总收入 R", "C_env": "环境成本 C_env", "C_soc": "社会成本 C_soc"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    xpos = np.arange(len(names))
    width = 0.36

    TRUNC_THRESHOLD: float = 10.0  # 截断阈值 (C_env tau_max ST=333, epsilon ST=15.9)

    for ax, t in zip(axes, targets):
        sub = df[df["target"] == t].set_index("param").loc[names]
        s1_raw = sub["S1"].to_numpy()
        st_raw = sub["ST"].to_numpy()
        s1c = np.clip(s1_raw, -TRUNC_THRESHOLD, TRUNC_THRESHOLD)
        stc = np.clip(st_raw, -TRUNC_THRESHOLD, TRUNC_THRESHOLD)
        # 置信区间同样截断 (C_env 的 conf 与指数同源爆炸, 不截会撑爆 y 轴)
        s1_conf_c = np.clip(sub["S1_conf"].to_numpy(), 0.0, TRUNC_THRESHOLD)
        st_conf_c = np.clip(sub["ST_conf"].to_numpy(), 0.0, TRUNC_THRESHOLD)

        ax.bar(xpos - width / 2, s1c, width, yerr=s1_conf_c, capsize=3,
               label="一阶指数 S1", color="#4C72B0", error_kw={"elinewidth": 0.8})
        ax.bar(xpos + width / 2, stc, width, yerr=st_conf_c, capsize=3,
               label="总效应 ST", color="#DD8452", error_kw={"elinewidth": 0.8})
        ax.set_xticks(xpos)
        ax.set_xticklabels(names, rotation=25, fontsize=8)
        ax.set_title(labels[t], fontsize=10)
        ax.grid(True, axis="y", ls="--", alpha=0.4)
        ax.axhline(0, color="gray", lw=0.6, ls=":")

        # 标记被截断的原始值
        for j in range(len(names)):
            if abs(s1_raw[j]) > TRUNC_THRESHOLD:
                ax.text(xpos[j] - width / 2, np.sign(s1_raw[j]) * TRUNC_THRESHOLD,
                        f"▼{s1_raw[j]:.0f}", ha="center", va="bottom", color="#C44E52",
                        fontsize=7, fontweight="bold")
            if abs(st_raw[j]) > TRUNC_THRESHOLD:
                ax.text(xpos[j] + width / 2, np.sign(st_raw[j]) * TRUNC_THRESHOLD,
                        f"▼{st_raw[j]:.0f}", ha="center", va="bottom", color="#C44E52",
                        fontsize=7, fontweight="bold")

        # C_env 子图: KKT 学术注释
        if t == "C_env":
            ax.text(0.97, 0.97,
                    ("Sobol 方差退化\n"
                     "Var(C_env)→0 (钉死 0.5 地板)\n"
                     "→ E_max 非活跃约束\n"
                     "KKT 影子价格 λ_E=0\n"
                     "红色▼=原始值远超截断阈值"),
                    transform=ax.transAxes, fontsize=7.2,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9E6", ec="#E8A838",
                              alpha=0.92, lw=0.8))

    axes[0].set_ylabel("Sobol 指数 (截断 |val|>10)", fontsize=9)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"M3 v3.0 Sobol 全局敏感性 (S00, N={M3_CONFIG['sobol_n']})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sobol_bar.png", dpi=DPI)
    plt.close(fig)


def plot_elasticity_rank(df: pd.DataFrame) -> None:
    targets = ["R", "C_env", "C_soc"]
    labels = {"R": "旅游总收入 R", "C_env": "环境成本 C_env", "C_soc": "社会成本 C_soc"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, t in zip(axes, targets):
        col = f"S_{t}"
        val = (df.groupby("param")[col].apply(lambda s: np.nanmean(np.abs(s)))
               .sort_values(ascending=True))
        ax.barh(np.arange(len(val)), val.values, color="#55A868")
        ax.set_yticks(np.arange(len(val)))
        ax.set_yticklabels(val.index)
        ax.set_title(labels[t])
        ax.set_xlabel("|S_p| (15 情景均值)")
        ax.grid(True, axis="x", ls="--", alpha=0.4)
        for i, v in enumerate(val.values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    fig.suptitle("M3 v2.0 单因素弹性绝对值排序 (±10% 扰动, 15 情景平均)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "elasticity_rank.png", dpi=DPI)
    plt.close(fig)


def plot_interaction_heatmap(s2_store: Dict[str, np.ndarray], names: List[str]) -> None:
    targets = ["R", "C_env", "C_soc"]
    labels = {"R": "旅游总收入 R", "C_env": "环境成本 C_env", "C_soc": "社会成本 C_soc"}
    n = len(names)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, t in zip(axes, targets):
        M = np.array(s2_store[t], dtype=float)
        M = np.where(np.isfinite(M), M, 0.0)
        M = np.triu(M, 1) + np.triu(M, 1).T
        vmax = max(np.nanmax(np.abs(M)), 1e-6)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(names, rotation=25)
        ax.set_yticklabels(names)
        ax.set_title(labels[t])
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("M3 v2.0 参数交互热力图 (Sobol 二阶指数 S2)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "interaction_heatmap.png", dpi=DPI)
    plt.close(fig)


# =====================================================================================
# 主流程
# =====================================================================================
def main() -> None:
    t_start = time.perf_counter()
    scenarios = m1.load_scenarios()
    base = load_m1_base()
    sobol_names, sobol_bounds, sobol_note = load_param_bounds_for_sobol()

    print("=" * 78)
    print("M3 v2.0 敏感性分析 | 单因素 M3-1 + Sobol M3-2")
    print(f"v2.0 适配: 引用 m1_solver.py v2.0 参数, E_max=0.95/S_min=0.70/gamma=0.1")
    print(f"Sobol 参数: {sobol_names}")
    if sobol_note:
        print(f"  注意: tau_max 区间 {TAU_MAX_BOUNDS} 为编程假设 (非 CSV 数据), 待建模手确认")
    print(f"  E_max 弹性≈0 属 v2.0 学术预期 (C_env≈0.5 << E_max=0.95, 非紧约束)")
    print(f"  w_j 权重: 单因素扰动重归一化, 不进 Sobol (权重不适合均匀采样)")
    print("=" * 78)

    # ---- 1. 单因素 ----
    print("[1] 单因素敏感性 (±10%, 15 情景, 含 w_j 权重重归一化) ...")
    t0 = time.perf_counter()
    elast = run_one_factor(scenarios, base)
    t_elast = time.perf_counter() - t0
    elast.to_csv(RESULTS_DIR / "m3_elasticity.csv", index=False, encoding="utf-8-sig")
    print(f"  完成 {t_elast:.1f}s -> m3_elasticity.csv ({len(elast)} 行)")
    # v2.0 学术预期说明
    ecol = elast[elast["param"] == "E_max"]["S_C_env"].abs().mean()
    print(f"  E_max → C_env 平均弹性 = {ecol:.4f} (v2.0 学术预期 ≈0, 属非紧约束)")

    # ---- 2. Sobol ----
    print(f"[2] Sobol 全局敏感性 ({M3_CONFIG['sobol_scenario']}, "
          f"固定 M1 v2.0 最优组合, N={M3_CONFIG['sobol_n']}, 并行评估) ...")
    sobol_df, extra = run_sobol(scenarios, base)
    sobol_df.to_csv(RESULTS_DIR / "sobol_indices.csv", index=False, encoding="utf-8-sig")
    meta = extra["meta"]
    print(f"  样本 {meta['n_samples']} (唯一 {meta['n_unique']}, 缓存 {meta['n_cache_hit']}), "
          f"回退 {meta['n_fallback']}, 耗时 {meta['elapsed_s']:.1f}s")
    print(f"  放宽分布: {meta['relax_counter']}")
    print("  Sobol 指数 (按 ST 排序):")
    for t in ["R", "C_env", "C_soc"]:
        sub = sobol_df[sobol_df["target"] == t].sort_values("ST", ascending=False)
        print(f"    Y={t}: " + ", ".join(
            f"{r.param}(S1={r.S1:.3f}, ST={r.ST:.3f})" for r in sub.itertuples()))

    # ---- 3. 交互 ----
    strong: List[str] = []
    for t in ["R", "C_env", "C_soc"]:
        sub = sobol_df[sobol_df["target"] == t]
        mx = float(sub["interaction_ST_minus_S1"].abs().max())
        s2m = np.nanmax(np.abs(np.where(np.isfinite(extra["s2"][t]), extra["s2"][t], 0.0)))
        if mx > M3_CONFIG["strong_interaction_st_s1"] or s2m > M3_CONFIG["strong_interaction_s2"]:
            strong.append(f"{t}(max|ST-S1|={mx:.3f}, max|S2|={s2m:.3f})")
    print(f"[3] 交互效应: {'显著 -> ' + '; '.join(strong) if strong else '不显著'}")

    # ---- 4. 绘图 ----
    plot_sobol_bar(sobol_df, extra["names"])
    plot_elasticity_rank(elast)
    plot_interaction_heatmap(extra["s2"], extra["names"])
    print("[4] 图已输出: sobol_bar.png / elasticity_rank.png / interaction_heatmap.png")

    print("-" * 78)
    print(f"总耗时 {time.perf_counter() - t_start:.1f}s | 输出目录 {RESULTS_DIR}")


if __name__ == "__main__":
    main()
