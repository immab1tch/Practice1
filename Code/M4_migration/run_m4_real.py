"""Run M4 real-data fee scenarios from the frozen primary-source evidence."""
import json
from pathlib import Path
import real_data as rd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

evidence = json.loads((ROOT / "inputs" / "real_venice_evidence.json").read_text())

daily = rd.voucher_daily(evidence)
daily.to_csv(OUT / "M4_real_voucher_daily.csv", index=False, encoding="utf-8-sig")

stats = rd.voucher_statistics(daily)
stats.to_csv(OUT / "M4_real_voucher_stats.csv", index=False, encoding="utf-8-sig")

scenarios = rd.fee_scenarios(evidence)
scenarios.to_csv(OUT / "M4_real_fee_scenarios.csv", index=False, encoding="utf-8-sig")

costs = rd.cost_components(evidence)
costs.to_csv(OUT / "M4_real_cost_components.csv", index=False, encoding="utf-8-sig")

net = rd.identified_net_anchor(evidence)
net.to_csv(OUT / "M4_real_net_anchor.csv", index=False, encoding="utf-8-sig")

feature_map = rd.feature_evidence_mapping(evidence)
feature_map.to_csv(OUT / "M4_real_feature_evidence.csv", index=False, encoding="utf-8-sig")

peak = scenarios.loc[scenarios.gross_fee_revenue_eur.idxmax()]
print("Real-data M4 scenarios written.")
print(f"Gross fee revenue peaks at {peak.fee_eur:.0f} EUR "
      f"({peak.gross_fee_revenue_eur:,.0f} EUR) under the stated-preference curve.")
print("This is a conditional scenario, not an observed optimum.")
