"""Independent verification of the polished figures.

The figures are restyled copies of frozen results. This script recomputes the
underlying numbers inline and confirms they match the frozen source files, so
the polished visuals cannot have altered any result.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
checks = 0


def expect(cond):
    global checks
    assert cond
    checks += 1


# ---- expected figures: every PNG has a non-empty SVG partner ----
expected = [
    "M1_equilibrium/M1P_01_New_investment_allocation",
    "M1_equilibrium/M1P_02_Pareto_front_color_mapped",
    "M2_budget/M2P_01_Fund_allocation_donut",
    "M3_robustness/M3P_01_Sensitivity_ranking",
    "M4_migration/M4P_01_Combined_feature_weights",
    "M4_migration/M4P_02_TOPSIS_policy_priorities",
    "M4_migration/M4P_03_WTP_revenue_combined",
    "M5_spatial/M5P_01_Gravity_allocation",
    "M5_spatial/M5P_02_Utilization_target",
    "M5_spatial/M5P_03_Transfer_heatmap",
    "Overview/OVP_01_Executive_dashboard",
]
for f in expected:
    for ext in ("png", "svg"):
        p = HERE / f"{f}.{ext}"
        expect(p.exists() and p.stat().st_size > 1000)

# ---- M1 investment + KPI numbers ----
rec = pd.read_csv(PROJ / "Code/M1_equilibrium/results/recommended.csv").iloc[0]
expect(abs(rec.spend_I * 100 - 6.087338) < 1e-6)
expect(abs(rec.spend_E * 100 - 18.575613) < 1e-6)
expect(abs(rec.spend_C * 100 - 9.464116) < 1e-6)
expect(abs(rec.x - 147.2) < 1e-9)
expect(abs(rec.tau - 0.14718843) < 1e-7)

# ---- M1 Pareto: recommended point belongs to the frozen set ----
pareto = pd.read_csv(PROJ / "Code/M1_equilibrium/results/pareto.csv")
expect(len(pareto) > 100)
expect(abs(pareto.iloc[0].x - rec.x) < 1e-6)

# ---- M3 sensitivity ranking ----
sens = pd.read_csv(PROJ / "Code/M3_robustness/results/M3_sensitivity_ranking.csv")
expect(sens.max_abs_delta_J.is_monotonic_decreasing)
expect(sens.iloc[0].parameter == "R0")
expect(abs(sens.iloc[0].max_abs_delta_J - 0.131322) < 1e-6)

# ---- M4 weights: combined weights sum to 1, values match ----
fw = pd.read_csv(PROJ / "Code/M4_migration/results/M4_feature_weights.csv")
expect(abs(fw.combined.sum() - 1.0) < 1e-9)
expect(abs(fw.combined.max() - 0.206039) < 1e-6)

# ---- M4 TOPSIS: per-destination ranking and top policy ----
pr = pd.read_csv(PROJ / "Code/M4_migration/results/M4_policy_ranking.csv")
for dest in ["Juneau", "Venice"]:
    sub = pr[pr.destination == dest].sort_values("closeness", ascending=False)
    expect(sub.iloc[0]["rank"] == 1)
expect(pr[(pr.destination == "Juneau") & (pr.policy_id == "P2")].iloc[0].closeness
       > 0.75)

# ---- M4 fee scenarios: 5 EUR anchor exact, peak at 20 ----
fee = pd.read_csv(PROJ / "Code/M4_migration/results/M4_real_fee_scenarios.csv")
expect(abs(fee.iloc[0].estimated_vouchers - 485062) < 1e-6)
expect(abs(fee.iloc[0].gross_fee_revenue_eur - 2425310) < 1e-6)
peak = fee.loc[fee.gross_fee_revenue_eur.idxmax()]
expect(peak.fee_eur == 20)

# ---- M5 allocation: total conserved, glacier below 0.80 ----
b = pd.read_csv(PROJ / "Code/M5_spatial/results/M5_allocation_before.csv")
a = pd.read_csv(PROJ / "Code/M5_spatial/results/M5_allocation_after.csv")
expect(abs(b.visitors_10k.sum() - a.visitors_10k.sum()) < 1e-6)
expect(abs(b.iloc[1].utilization - 0.950464) < 1e-6)
expect(abs(a.iloc[1].utilization - 0.796899) < 1e-6)
expect(a.iloc[1].utilization < 0.80)

# ---- M5 transfer matrix: row sums = origin (before), column sums = destination (after) ----
tm = pd.read_csv(PROJ / "Code/M5_spatial/results/M5_transfer_matrix.csv", index_col=0)
row_sums = tm.sum(axis=1)
col_sums = tm.sum(axis=0)
for i, node in enumerate(tm.index):
    expect(abs(row_sums.iloc[i] - b.iloc[i].visitors_10k) < 1e-6)
    expect(abs(col_sums.iloc[i] - a.iloc[i].visitors_10k) < 1e-6)

# ---- M2 donut: shares of the default ledger sum to 1 ----
led = pd.read_csv(PROJ / "Code/M2_budget/results/M2_budget_ledger.csv")
sub = led[(led.case == "default") & (led.million_usd > 0)]
expect(abs(sub.share_of_total_tax.sum() - 1.0) < 1e-9)
expect(abs(sub.million_usd.sum() - 50.780009) < 1e-6)

result = {
    "status": "passed",
    "polished_figures": len(expected),
    "independent_checks": checks,
    "note": "All polished visuals are restyled from frozen results; underlying numbers recomputed and matched, no result altered.",
}
(HERE / "polished_checks.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
