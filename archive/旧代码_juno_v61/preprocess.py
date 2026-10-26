# -*- coding: utf-8 -*-
r"""
Data Preprocessing Module for Problem B: Sustainable Tourism (Juneau)

Project root: 脚本所在目录（相对路径，基于 __file__ 解析）
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import savemat

warnings.filterwarnings("ignore")

# ==================== Project paths ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "preprocessed"
FIG_DIR = OUTPUT_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "juno_raw.xlsx"
CREATE_DEMO_IF_MISSING = True
DPI = 300
# =======================================================

for d in [BASE_DIR, DATA_DIR, OUTPUT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


def make_demo_data():
    cols = [
        "scenario_id", "tourists_annual_10k", "daily_peak_10k", "revenue_100m",
        "resident_pop_10k", "port_physical_berths", "policy_berth_cap",
        "tax_rate", "delta_I", "delta_E", "delta_C", "delta_total",
        "season_start_month", "season_end_month", "glacier_retreat_fields",
        "pos_pct", "neg_pct", "two_pct", "no_pct", "mgmt_insufficient_pct",
        "crowding_pct", "vehicle_pct", "sidewalk_pct",
    ]
    rows = [
        ("S00_baseline",   160, 2.0, 3.75, 3, 7, 5, 0.11, 0.30, 0.30, 0.20, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
        ("S01_low_tour",   140, 1.7, 3.20, 3, 7, 5, 0.11, 0.28, 0.32, 0.20, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
        ("S02_high_tour",  180, 2.3, 4.10, 3, 7, 5, 0.11, 0.32, 0.28, 0.20, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
        ("S03_strict_eco", 160, 2.0, 3.75, 3, 7, 5, 0.12, 0.25, 0.40, 0.15, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
        ("S04_loose_eco",  160, 2.0, 3.75, 3, 7, 5, 0.10, 0.35, 0.22, 0.23, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
        ("S05_high_tax",   160, 2.0, 3.75, 3, 7, 5, 0.15, 0.30, 0.30, 0.20, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
        ("S06_low_tax",    160, 2.0, 3.75, 3, 7, 5, 0.09, 0.30, 0.30, 0.20, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
        ("S07_strict_cap", 150, 1.8, 3.50, 3, 7, 4, 0.12, 0.32, 0.30, 0.18, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
        ("S08_loose_cap",  170, 2.2, 3.95, 3, 7, 5, 0.11, 0.30, 0.28, 0.22, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
        ("S09_community",  160, 2.0, 3.75, 3, 7, 5, 0.11, 0.25, 0.25, 0.30, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
        ("S10_infra",      160, 2.0, 3.75, 3, 7, 5, 0.11, 0.40, 0.20, 0.20, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
        ("S11_ecology",    160, 2.0, 3.75, 3, 7, 5, 0.11, 0.20, 0.40, 0.20, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
        ("S12_balanced",   158, 1.9, 3.70, 3, 7, 5, 0.12, 0.30, 0.30, 0.20, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
        ("S13_perturbed",  175, 2.4, 4.05, 3, 7, 5, 0.13, 0.28, 0.30, 0.22, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
        ("S14_conserv",    145, 1.6, 3.30, 3, 7, 5, 0.10, 0.30, 0.35, 0.15, 0.80, 4, 10, 8, 31, 11, 46, 1, 56, 63, 61, 59),
    ]
    main = pd.DataFrame(rows, columns=cols)
    # 制造几个缺失/异常以测试
    main.loc[2, "tax_rate"] = np.nan
    main.loc[5, "delta_E"] = 0.95
    main.loc[7, "policy_berth_cap"] = 8

    measures = pd.DataFrame({
        "measure_id":   ["M1", "M2", "M3", "M4", "M5", "M6"],
        "measure_name": ["daily_visitor_cap", "alcohol_restriction", "shore_power",
                         "traffic_management", "marketing_diversion", "visitor_fee_hotel_tax"],
        "m_j":          [1, 0, 1, 1, 1, 1],
        "cost_level":   [3, 1, 4, 2, 2, 3],
        "feasible":     [1, 1, 1, 1, 1, 1],
    })
    projects = pd.DataFrame({
        "project_id":   ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
        "category":     ["I", "I", "E", "E", "C", "C", "E", "I"],
        "cost_level":   [3, 2, 4, 3, 2, 1, 2, 2],
        "eta":          [0.8, 0.7, 0.9, 0.8, 0.6, 0.5, 0.7, 0.6],
        "selected_yk":  [1, 0, 1, 0, 1, 0, 1, 1],
    })
    regions = pd.DataFrame({
        "region_id":      ["R1", "R2", "R3", "R4", "R5"],
        "region_name":    ["downtown_port_hotspot", "mendenhall_glacier_hotspot",
                           "whale_watching_cold", "rainforest_cold", "community_culture_cold"],
        "type":           ["hot", "hot", "cold", "cold", "cold"],
        "capacity_level": [3, 2, 2, 2, 1],
        "x_i_before":     [60, 40, 15, 10, 35],
        "x_i_after":      [55, 38, 20, 15, 32],
        "tau_i":          [0.12, 0.15, 0.08, 0.07, 0.08],
        "z_i":            [1, 1, 0, 0, 0],
        "y_i":            [0, 0, 1, 1, 1],
    })
    params = pd.DataFrame({
        "param":  ["demand_elasticity", "eta_I", "eta_E", "eta_C", "eco_threshold",
                   "social_accept_min", "fairness_min", "w_income", "w_env", "w_soc", "w_infra"],
        "value":  [-0.5, 0.7, 0.8, 0.6, 0.8, 0.5, 0.2, 0.35, 0.30, 0.20, 0.15],
        "lower":  [-0.8, 0.5, 0.6, 0.4, 0.6, 0.4, 0.1, 0.25, 0.20, 0.10, 0.05],
        "upper":  [-0.2, 0.9, 0.95, 0.8, 0.95, 0.7, 0.3, 0.45, 0.40, 0.30, 0.25],
        "source": ["assumed_interval"] * 11,
    })
    return {"main": main, "measures": measures, "projects": projects, "regions": regions, "params": params}


def load_raw_main(path: Path):
    """读取主表，如果文件损坏或格式不对，返回 None 并触发 Demo 数据。"""
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    try:
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        if suffix == ".csv":
            try:
                return pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding="gbk")
    except Exception as e:
        print("[WARNING] 读取文件失败：", e)
        print("[WARNING] 请检查文件是否真的是 Excel 格式，或者是否被其他软件占用。")
        return None
    raise ValueError("Unsupported format")


def load_or_make_data():
    main = load_raw_main(RAW_DATA_PATH)
    demo = make_demo_data()

    if main is None:
        if not CREATE_DEMO_IF_MISSING:
            raise FileNotFoundError("Raw data not found or unreadable.")
        print("[INFO] 使用示例数据 (Demo Data) 跑通流程。")
        print("[INFO] 若需使用真实数据，请将真实的 Excel 放到 %s" % DATA_DIR)
        main = demo["main"]

    data = {"main": main}
    extra_files = [
        ("measures", "juno_measures_binary.csv"),
        ("projects", "juno_projects_budget.csv"),
        ("regions",  "juno_regions.csv"),
        ("params",   "juno_params_scenarios.csv"),
    ]
    for key, fname in extra_files:
        p = DATA_DIR / fname
        if p.exists():
            try:
                data[key] = pd.read_csv(p, encoding="utf-8-sig")
                print("[LOAD] %s" % p)
            except Exception as e:
                print("[WARNING] 读取 %s 失败，使用 Demo 数据。" % fname)
                data[key] = demo[key]
        else:
            data[key] = demo[key]
            print("[DEMO] %s not found, demo structure is used." % fname)
    return data


def iqr_fix(series, name, report):
    s = pd.to_numeric(series, errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return s, pd.Series(np.zeros(len(s), dtype=int), index=s.index)
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flag = ((s < low) | (s > high)).astype(int)
    fixed = s.clip(lower=low, upper=high)
    report.append({"table": "main", "field": name, "method": "IQR_winsorize",
                   "low": float(low), "high": float(high), "outlier_count": int(flag.sum())})
    return fixed, flag


def validate_and_fix_main(main, report):
    df = main.copy()
    num_cols = [c for c in df.columns if c != "scenario_id" and df[c].dtype != object]
    for c in num_cols:
        if df[c].isna().any():
            med = df[c].median()
            if pd.isna(med): med = 0.0
            df[c] = df[c].fillna(med)
            df["%s_imputed" % c] = df[c].isna().astype(int)
            report.append({"table": "main", "field": c, "method": "median_fill", "fill_value": float(med)})

    rule_defaults = {"resident_pop_10k": 3, "port_physical_berths": 7, "policy_berth_cap": 5,
                     "season_start_month": 4, "season_end_month": 10}
    for c, v in rule_defaults.items():
        if c in df.columns: df[c] = df[c].fillna(v)

    outlier_cols = ["tourists_annual_10k", "daily_peak_10k", "revenue_100m", "tax_rate",
                    "delta_I", "delta_E", "delta_C", "pos_pct", "neg_pct", "two_pct", "no_pct"]
    for c in outlier_cols:
        if c in df.columns:
            df[c], flag = iqr_fix(df[c], c, report)
            df["%s_outlier_flag" % c] = flag

    df["tax_rate"] = df["tax_rate"].clip(0, 1)
    for c in ["delta_I", "delta_E", "delta_C"]: df[c] = df[c].clip(lower=0)

    ratio_sum = df[["delta_I", "delta_E", "delta_C"]].sum(axis=1)
    over = ratio_sum > df["delta_total"] + 1e-9
    scale = df.loc[over, "delta_total"] / ratio_sum[over]
    for c in ["delta_I", "delta_E", "delta_C"]: df.loc[over, c] = df.loc[over, c] * scale
    df["ratio_sum_flag"] = over.astype(int)

    conflict = df["policy_berth_cap"] > df["port_physical_berths"]
    df.loc[conflict, "policy_berth_cap"] = df.loc[conflict, "port_physical_berths"]
    df["berth_rule_flag"] = conflict.astype(int)

    pct_cols = ["pos_pct", "neg_pct", "two_pct", "no_pct"]
    pct_sum = df[pct_cols].sum(axis=1)
    bad_pct = (pct_sum - 100).abs() > 1
    for c in pct_cols: df.loc[bad_pct, c] = df.loc[bad_pct, c] / pct_sum[bad_pct] * 100
    df["pct_sum_flag"] = bad_pct.astype(int)

    bad_season = df["season_start_month"] > df["season_end_month"]
    df.loc[bad_season, "season_start_month"] = 4
    df.loc[bad_season, "season_end_month"] = 10
    df["season_rule_flag"] = bad_season.astype(int)
    df["peak_rule_flag"] = (df["daily_peak_10k"] > df["tourists_annual_10k"]).astype(int)

    std_cols = ["tourists_annual_10k", "daily_peak_10k", "revenue_100m", "tax_rate", "delta_I", "delta_E", "delta_C"]
    for c in std_cols:
        mu = df[c].mean()
        sd = df[c].std(ddof=0) + 1e-12
        df["%s_z" % c] = (df[c] - mu) / sd
        lo, hi = df[c].min(), df[c].max()
        df["%s_minmax" % c] = (df[c] - lo) / (hi - lo + 1e-12)

    df["w_income"] = 0.35
    df["w_env"] = 0.30
    df["w_soc"] = 0.20
    df["w_infra"] = 0.15
    df["weight_source"] = "initial_equal_AHP"
    df["data_flag"] = "main_scale_160w_375M_2w_7berths_5cap"
    return df, report


def validate_measures(measures, report):
    df = measures.copy()
    for c in ["m_j", "feasible"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    if "measure_name" in df.columns: df["measure_name_code"] = df["measure_name"].astype("category").cat.codes
    report.append({"table": "measures", "method": "0-1 rounding + range clip"})
    return df


def validate_projects(projects, report):
    df = projects.copy()
    if "selected_yk" in df.columns: df["selected_yk"] = pd.to_numeric(df["selected_yk"], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    if "eta" in df.columns: df["eta"] = pd.to_numeric(df["eta"], errors="coerce").fillna(0).clip(0, 1)
    if "category" in df.columns: df["category_code"] = df["category"].astype("category").cat.codes
    report.append({"table": "projects", "method": "0-1 rounding + eta clip"})
    return df


def validate_regions(regions, main, report):
    df = regions.copy()
    for c in ["z_i", "y_i"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round().clip(0, 1).astype(int)

    target = float(main["tourists_annual_10k"].iloc[0]) if "tourists_annual_10k" in main.columns else 160.0
    for col in ["x_i_before", "x_i_after"]:
        if col in df.columns:
            s = float(df[col].sum())
            if abs(s - target) > 1e-6 and s > 0:
                df[col] = df[col] / s * target
                report.append({"table": "regions", "field": col, "method": "conservation_scaling", "old_sum": s, "target": target})
    return df


def make_region_edges():
    edges = pd.DataFrame([
        ("R1", "R2", 20, 3), ("R1", "R3", 35, 2), ("R1", "R4", 40, 2), ("R1", "R5", 15, 1),
        ("R2", "R3", 30, 2), ("R2", "R4", 35, 2), ("R3", "R5", 25, 1), ("R4", "R5", 30, 1),
    ], columns=["from_region", "to_region", "travel_time", "capacity_level"])

    tt_min, tt_max = edges["travel_time"].min(), edges["travel_time"].max()
    edges["travel_time_norm"] = (edges["travel_time"] - tt_min) / (tt_max - tt_min + 1e-12)
    inv = 1.0 / edges["capacity_level"]
    inv_min, inv_max = inv.min(), inv.max()
    edges["capacity_inv_norm"] = (inv - inv_min) / (inv_max - inv_min + 1e-12)
    edges["edge_weight"] = 0.6 * edges["travel_time_norm"] + 0.4 * edges["capacity_inv_norm"]
    return edges


def visualize(raw_main, main, measures, regions):
    plt.figure(figsize=(12, 5))
    sns.heatmap(raw_main.isna(), cbar=False, cmap="viridis")
    plt.title("Missing Value Heatmap (Before Preprocessing)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_missing_heatmap_before.png", dpi=DPI)
    plt.close()

    box_cols = [c for c in ["tourists_annual_10k", "daily_peak_10k", "revenue_100m", "tax_rate", "delta_I", "delta_E", "delta_C"] if c in raw_main.columns]
    if box_cols:
        plt.figure(figsize=(12, 5))
        sns.boxplot(data=raw_main[box_cols], orient="h")
        plt.title("Key Numerical Fields Boxplot (Before Preprocessing)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "02_boxplot_before.png", dpi=DPI)
        plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(main["tourists_annual_10k"], bins=8, color="#4C72B0", alpha=0.85)
    axes[0, 0].set_title("Before Standardization: Annual Tourists (10k)")
    axes[0, 1].hist(main["tourists_annual_10k_z"], bins=8, color="#55A868", alpha=0.85)
    axes[0, 1].set_title("After Z-score: Annual Tourists")
    axes[1, 0].hist(main["tax_rate"], bins=8, color="#C44E52", alpha=0.85)
    axes[1, 0].set_title("Before Standardization: Tax Rate")
    axes[1, 1].hist(main["tax_rate_minmax"], bins=8, color="#8172B2", alpha=0.85)
    axes[1, 1].set_title("After Min-Max: Tax Rate")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_standardization_compare.png", dpi=DPI)
    plt.close()

    if "measure_name" in measures.columns and "m_j" in measures.columns:
        plt.figure(figsize=(10, 5))
        measures.sort_values("m_j", ascending=True).plot.barh(x="measure_name", y="m_j", color="#4C72B0", legend=False)
        plt.title("Discrete Measures Adoption (0-1)")
        plt.xlabel("m_j")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "04_measures_binary.png", dpi=DPI)
        plt.close()

    if "region_name" in regions.columns and "x_i_before" in regions.columns and "x_i_after" in regions.columns:
        regions.set_index("region_name")[["x_i_before", "x_i_after"]].plot(kind="bar", figsize=(12, 5), color=["#C44E52", "#55A868"])
        plt.title("Region Visitor Volume: Before vs After Balancing")
        plt.ylabel("Visitors (10k)")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "05_regions_before_after.png", dpi=DPI)
        plt.close()


def save_all(main, measures, projects, regions, params, edges):
    main.to_csv(OUTPUT_DIR / "juno_main_preprocessed.csv", index=False, encoding="utf-8-sig")
    measures.to_csv(OUTPUT_DIR / "juno_measures_binary.csv", index=False, encoding="utf-8-sig")
    projects.to_csv(OUTPUT_DIR / "juno_projects_budget.csv", index=False, encoding="utf-8-sig")
    regions.to_csv(OUTPUT_DIR / "juno_regions_balanced.csv", index=False, encoding="utf-8-sig")
    params.to_csv(OUTPUT_DIR / "juno_params_scenarios.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(OUTPUT_DIR / "juno_region_edges.csv", index=False, encoding="utf-8-sig")

    mat_dict = {}
    for name, df in [("main", main), ("measures", measures), ("projects", projects), ("regions", regions), ("params", params), ("edges", edges)]:
        for col in df.columns:
            mat_dict["%s_%s" % (name, col)] = df[col].values
    savemat(OUTPUT_DIR / "juno_preprocessed.mat", mat_dict)

    report_path = OUTPUT_DIR / "preprocess_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"project_root": str(BASE_DIR), "output_dir": str(OUTPUT_DIR), "files": sorted(os.listdir(OUTPUT_DIR))}, f, ensure_ascii=False, indent=2)
    return report_path


def validate_output(main, measures, regions):
    checks = []
    checks.append(("main_rows", len(main)))
    checks.append(("main_missing_ratio", float(main.isna().mean().mean())))
    checks.append(("measures_m_j_values", sorted(measures["m_j"].unique().tolist())))
    checks.append(("regions_z_i_values", sorted(regions["z_i"].unique().tolist())))
    checks.append(("regions_y_i_values", sorted(regions["y_i"].unique().tolist())))
    checks.append(("max_ratio_sum", float((main["delta_I"] + main["delta_E"] + main["delta_C"]).max())))
    checks.append(("max_delta_total", float(main["delta_total"].max())))
    checks.append(("ratio_sum_ok", bool(((main["delta_I"] + main["delta_E"] + main["delta_C"]) <= main["delta_total"] + 1e-9).all())))
    checks.append(("policy_le_physical", bool((main["policy_berth_cap"] <= main["port_physical_berths"]).all())))
    checks.append(("region_before_sum", float(regions["x_i_before"].sum())))
    checks.append(("region_after_sum", float(regions["x_i_after"].sum())))
    return checks


def main():
    print("=" * 70)
    print("Data Preprocessing Module - Problem B Sustainable Tourism")
    print("Project root: %s" % BASE_DIR)
    print("Raw data path: %s" % RAW_DATA_PATH)
    print("Output dir:   %s" % OUTPUT_DIR)
    print("=" * 70)

    data = load_or_make_data()
    raw_main = data["main"].copy()
    report = []

    main_df, report = validate_and_fix_main(data["main"], report)
    measures_df = validate_measures(data["measures"], report)
    projects_df = validate_projects(data["projects"], report)
    regions_df = validate_regions(data["regions"], main_df, report)
    params_df = data["params"].copy()
    edges_df = make_region_edges()

    visualize(raw_main, main_df, measures_df, regions_df)
    report_path = save_all(main_df, measures_df, projects_df, regions_df, params_df, edges_df)

    print("\n===== Preprocessed main table: head(10) =====")
    print(main_df.head(10).to_string(index=False))
    print("\n===== Preprocessed main table: tail(5) =====")
    print(main_df.tail(5).to_string(index=False))

    print("\n===== Integrity checks =====")
    for k, v in validate_output(main_df, measures_df, regions_df):
        print("%-25s: %s" % (k, v))

    print("\n[REPORT] %s" % report_path)
    print("[DONE] CSV / MAT / figures are saved under: %s" % OUTPUT_DIR)


if __name__ == "__main__":
    main()