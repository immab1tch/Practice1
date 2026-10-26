# -*- coding: utf-8 -*-
"""构建 juno_raw.xlsx —— 原始数据工作簿 (论文手反馈原文件 0 字节).

数据来源 (全部真实, 不编造):
  * passenger_volume: McKinley Research Group (2024) Figure 1 (2014-2020)
                      + problem statement (2019, 2021-2023)
  * spending_2023:    McKinley (2024) Executive Summary
  * resident_survey:  preprocessed/juno_main_preprocessed.csv (S00, Juneau survey 2023)
  * problem_baseline: problem statement 基准 (x0=160, R0=3.75, P=3 万...)
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
data_dir = BASE / "data"
out = data_dir / "juno_raw.xlsx"

# ---- Sheet 1: 2014-2023 邮轮乘客量 (万人) ----
passenger = pd.DataFrame({
    "year": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "passengers_10k": [96.1, 98.3, 101.5, 107.2, 115.1, 126.5, 11.7, 11.3, 47.4, 160.0],
    "source": ["McKinley2024_Fig1"] * 5 + ["ProblemStatement2019",
                                           "McKinley2024_Fig1",
                                           "ProblemStatement2021",
                                           "ProblemStatement2022",
                                           "ProblemStatement2023"],
    "note": [""] * 10,
})
passenger.loc[passenger["year"] == 2019, "note"] = "prev. peak 1.265M (problem) / 1.306M (report)"
passenger.loc[passenger["year"] == 2020, "note"] = "COVID-19 pandemic year"
passenger.loc[passenger["year"] == 2023, "note"] = "model baseline x0=160 (report: 1.67M unadjusted)"

# ---- Sheet 2: 2023 直接支出构成 (百万 USD) ----
spending = pd.DataFrame({
    "category": ["Cruise passenger spending", "Cruise line spending",
                 "Crew member spending", "Total direct spending"],
    "usd_million": [320, 39, 16, 375],
    "share_pct": [85.3, 10.4, 4.3, 100.0],
    "source": ["McKinley2024_ExecSummary"] * 4,
})

# ---- Sheet 3: 2023 居民调查 (preprocessed S00) ----
survey = pd.DataFrame({
    "item": ["Positive impact", "Negative impact", "Both positive and negative",
             "No impact", "Streets very crowded", "Vehicle traffic concern",
             "Sidewalk congestion", "Management insufficient"],
    "value_pct": [34.83, 12.36, 51.69, 1.12, 56, 63, 61, 56],
    "source": ["Juneau resident survey 2023 (problem / preprocessed S00)"] * 8,
})

# ---- Sheet 4: 模型基准参数 (problem statement) ----
baseline = pd.DataFrame({
    "param": ["x0 (2023 passengers, 10k)", "R0 (2023 revenue, 100M USD)",
              "P (resident population, 10k)", "xd0 (daily peak, 10k)",
              "port physical berths", "policy berth cap", "tax_rate tau0",
              "delta_total"],
    "value": [160.0, 3.75, 3.0, 2.0, 7, 5, 0.11, 0.8],
    "unit": ["10k persons", "USD 100M", "10k persons", "10k persons/day",
             "berths", "ships/day", "ratio", "ratio"],
    "note": ["", "", "", "", "", "", "baseline tax", "reinvestment cap"],
})

with pd.ExcelWriter(out, engine="openpyxl") as w:
    passenger.to_excel(w, sheet_name="passenger_volume", index=False)
    spending.to_excel(w, sheet_name="spending_2023", index=False)
    survey.to_excel(w, sheet_name="resident_survey", index=False)
    baseline.to_excel(w, sheet_name="problem_baseline", index=False)

print("OK:", out, out.stat().st_size, "bytes")
for s in ["passenger_volume", "spending_2023", "resident_survey", "problem_baseline"]:
    print("  sheet:", s)
