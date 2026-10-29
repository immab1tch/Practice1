"""M4 real-data independent checks: recompute every formula inline.

The math is written here (not imported from real_data) so this is a genuine
cross-check of the evidence-grounded fee scenarios.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
evidence = json.loads((ROOT / "inputs" / "real_venice_evidence.json").read_text())
checks = 0

# 29 observed days: sum and gross.
rows = evidence["daily_2024"]["rows"]
total = sum(r["paid_vouchers"] for r in rows)
assert total == 485062
assert total * 5 == 2425310
daily = pd.read_csv(OUT / "M4_real_voucher_daily.csv")
assert len(daily) == 29
assert daily.paid_vouchers.sum() == 485062
checks += 3

# Fee scenarios recomputed inline.
wtp = evidence["wtp_prior"]["open_ended_survival"]
q5 = next(w["share"] for w in wtp if w["fee_eur"] == 5)
anchor = 485062
scenarios = pd.read_csv(OUT / "M4_real_fee_scenarios.csv")
assert len(scenarios) == len(wtp)
for w in wtp:
    f = w["fee_eur"]
    mult = w["share"] / q5
    vouchers = anchor * mult
    gross = vouchers * f
    saved = scenarios.loc[scenarios.fee_eur == f].iloc[0]
    assert abs(saved.participation_multiplier - mult) < 1e-9
    assert abs(saved.estimated_vouchers - vouchers) < 1e-6
    assert abs(saved.gross_fee_revenue_eur - gross) < 1e-6
checks += len(wtp) * 3

# The 5-EUR scenario exactly reproduces the observed administrative anchor.
anchor_row = scenarios.loc[scenarios.fee_eur == 5].iloc[0]
assert anchor_row.estimated_vouchers == 485062
assert anchor_row.gross_fee_revenue_eur == 2425310
checks += 2

# Voucher statistics: mean and peak recomputed inline.
stats = pd.read_csv(OUT / "M4_real_voucher_stats.csv").set_index("metric")
v = np.array([r["paid_vouchers"] for r in rows], float)
assert abs(stats.loc["mean vouchers / day", "value"] - v.mean()) < 1e-9
assert abs(stats.loc["maximum (peak)", "value"] - v.max()) < 1e-9
assert abs(stats.loc["minimum (trough)", "value"] - v.min()) < 1e-9
checks += 3

# Net anchor: fixed cost and upper bound.
net = pd.read_csv(OUT / "M4_real_net_anchor.csv").set_index("item")
fixed = next(c["value_eur"] for c in evidence["cost_evidence"]
             if "fixed service remuneration" in c["metric"])
assert net.loc["identified fixed cost (Vela, ex VAT)", "eur"] == fixed
assert net.loc["net upper bound (other costs not identified)", "eur"] == 2425310 - fixed
checks += 2

# Feature evidence mapping: 7 features, scores in 1..5.
fmap = pd.read_csv(OUT / "M4_real_feature_evidence.csv")
assert len(fmap) == 7
assert fmap.expert_score.between(1, 5).all()
checks += 2

# Real-data figures.
fig_index = pd.read_csv(OUT / "figure_index_real_en.csv")
assert len(fig_index) == 4
for name in fig_index.file:
    for ext in ["png", "svg"]:
        assert (ROOT / "figures" / f"{name}.{ext}").stat().st_size > 1000
checks += len(fig_index) * 2

result = {
    "status": "passed",
    "M4_real_data_independent_checks": checks,
    "observed_anchor_vouchers": 485062,
    "observed_anchor_gross_eur": 2425310,
    "fee_levels": len(wtp),
    "real_figure_pairs": len(fig_index),
    "scope": "Cross-checks the evidence-grounded fee scenarios; WTP is stated preference, no causal elasticity or observed optimum claimed.",
}
(OUT / "real_data_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
