"""Run M5: gravity allocation, cold-point investment, max-flow check, transfer matrix."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from model import gravity_allocation, apply_investment, network_max_flow, build_transfer_matrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def allocation_frame(cfg, x, shares, cap, resistance, factor):
    rows = []
    for i, node in enumerate(cfg["nodes"]):
        rows.append(dict(node_id=node["id"], node_name=node["name_en"], type=node["type"],
                         visitors_10k=x[i], share=shares[i], capacity_10k=cap[i],
                         utilization=x[i] / cap[i], resistance=resistance[i],
                         resistance_factor=factor[i]))
    return pd.DataFrame(rows)


def main():
    cfg = json.loads((ROOT / "config.json").read_text())
    node_ids = [n["id"] for n in cfg["nodes"]]
    gamma = cfg["decay_gamma"]
    x_total = cfg["x_total"]
    daily_peak = cfg["daily_peak"]

    def scenario_with_coefficients(coeffs):
        """Recompute the post-investment outcome with a custom beta set."""
        cfg2 = json.loads(json.dumps(cfg))
        for key in ("beta_marketing", "beta_resistance", "beta_capacity", "beta_edge"):
            cfg2["investment"][key] = coeffs[key]
        A1, res1, cap1, edges1, _ = apply_investment(cfg2)
        x1, sh1, _ = gravity_allocation(A1, res1, gamma, x_total)
        daily1 = sh1 * daily_peak
        flow1, out1, ok1 = network_max_flow(edges1, daily1, node_ids)
        gi = node_ids.index("N2")
        return float(x1[gi] / cap1[gi]), float(flow1), float(out1), bool(ok1)

    # Coefficient sensitivity (makes the target-calibration transparent).
    sens_rows = []
    for sc in cfg.get("coefficient_sensitivity", []):
        gu, flow, demand, ok = scenario_with_coefficients(sc)
        sens_rows.append(dict(label=sc["label"], beta_marketing=sc["beta_marketing"],
                              beta_resistance=sc["beta_resistance"], beta_capacity=sc["beta_capacity"],
                              beta_edge=sc["beta_edge"], glacier_utilization=gu,
                              max_flow_10k=flow, demand_10k=demand, flow_feasible=ok,
                              glacier_below_080=bool(gu < 0.80)))
    pd.DataFrame(sens_rows).to_csv(OUT / "M5_coefficient_sensitivity.csv",
                                   index=False, encoding="utf-8-sig")

    # Baseline allocation.
    A0 = cfg["attractiveness_baseline"]
    res0 = cfg["resistance_baseline"]
    cap0 = cfg["capacity_baseline"]
    x0, sh0, fac0 = gravity_allocation(A0, res0, gamma, x_total)
    before = allocation_frame(cfg, x0, sh0, cap0, res0, fac0)
    before.to_csv(OUT / "M5_allocation_before.csv", index=False, encoding="utf-8-sig")

    # Baseline daily max flow.
    edges0 = {f"{e['from']}->{e['to']}": e["capacity_baseline"] for e in cfg["edges"]}
    daily0 = sh0 * daily_peak
    flow0, out0, ok0 = network_max_flow(edges0, daily0, node_ids)

    # Apply investment.
    A1, res1, cap1, edges1, inv_details = apply_investment(cfg)
    x1, sh1, fac1 = gravity_allocation(A1, res1, gamma, x_total)
    after = allocation_frame(cfg, x1, sh1, cap1, res1, fac1)
    after.to_csv(OUT / "M5_allocation_after.csv", index=False, encoding="utf-8-sig")

    # Post daily max flow.
    daily1 = sh1 * daily_peak
    flow1, out1, ok1 = network_max_flow(edges1, daily1, node_ids)

    pd.DataFrame(inv_details).to_csv(OUT / "M5_investment_details.csv", index=False, encoding="utf-8-sig")

    # Transfer matrix.
    T = build_transfer_matrix(x0, x1, node_ids)
    pd.DataFrame(T, index=node_ids, columns=node_ids).to_csv(
        OUT / "M5_transfer_matrix.csv", encoding="utf-8-sig")

    # Network flow summary.
    flow_df = pd.DataFrame([
        dict(scenario="before", max_flow_10k=flow0, outgoing_demand_10k=out0, feasible=ok0),
        dict(scenario="after", max_flow_10k=flow1, outgoing_demand_10k=out1, feasible=ok1),
    ])
    flow_df.to_csv(OUT / "M5_network_flow.csv", index=False, encoding="utf-8-sig")

    glacier_i = node_ids.index("N2")
    summary = dict(
        before=dict(glacier_utilization=float(x0[glacier_i] / cap0[glacier_i])),
        after=dict(glacier_utilization=float(x1[glacier_i] / cap1[glacier_i]),
                   visitors_10k={node_ids[i]: float(x1[i]) for i in range(len(node_ids))}),
        investment_budget_10k_usd=cfg["investment"]["budget_10k_usd"],
        max_flow_before=dict(flow=flow0, demand=out0, feasible=ok0),
        max_flow_after=dict(flow=flow1, demand=out1, feasible=ok1),
        cold_gain_10k=float(np.sum(np.where(x1 - x0 > 0, x1 - x0, 0))),
        glacier_target_below_080=bool(x1[glacier_i] / cap1[glacier_i] < 0.80),
        global_optimality_proven=False,
        note="Semi-quantitative gravity + max-flow; investment coefficients are assumptions.",
    )
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
