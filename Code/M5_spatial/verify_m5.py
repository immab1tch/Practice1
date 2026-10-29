"""M5 independent checks: recompute gravity, investment, max-flow and transfer.

The math is written inline (not imported from model) so this is a genuine
cross-check of the delivered evidence.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
cfg = json.loads((ROOT / "config.json").read_text())
checks = 0
node_ids = [n["id"] for n in cfg["nodes"]]
gamma = cfg["decay_gamma"]
x_total = cfg["x_total"]
daily_peak = cfg["daily_peak"]


def gravity(A, res):
    f = (1.0 + np.asarray(res)) ** (-gamma)
    raw = np.asarray(A) * f
    return raw / raw.sum() * x_total, f


# Baseline gravity.
A0 = np.asarray(cfg["attractiveness_baseline"])
res0 = np.asarray(cfg["resistance_baseline"])
x0, _ = gravity(A0, res0)
before = pd.read_csv(OUT / "M5_allocation_before.csv").sort_values("node_id")
assert np.allclose(x0, before.visitors_10k, atol=1e-6)
checks += len(node_ids)

# Investment effects (inline).
inv = cfg["investment"]
A_pot = np.asarray(cfg["attractiveness_potential"])
mkt_factor = np.where(A_pot > 0, A0 / A_pot, 1.0)
res1 = res0.copy()
cap1 = np.asarray(cfg["capacity_baseline"], float)
edge_cap = {e["to"]: e["capacity_baseline"] for e in cfg["edges"]}
A1 = A0.copy()
for node_id, spend in inv["allocation"].items():
    i = node_ids.index(node_id)
    mkt = spend * inv["marketing_fraction"]
    transit = spend * (1 - inv["marketing_fraction"])
    mkt_factor[i] = min(1.0, mkt_factor[i] + inv["beta_marketing"] * mkt)
    A1[i] = A_pot[i] * mkt_factor[i]
    res1[i] = max(0.0, res1[i] - inv["beta_resistance"] * transit)
    cap1[i] = cap1[i] + inv["beta_capacity"] * transit
    edge_cap[node_id] = edge_cap[node_id] + inv["beta_edge"] * transit
checks += len(inv["allocation"]) * 4

# Post gravity.
x1, _ = gravity(A1, res1)
after = pd.read_csv(OUT / "M5_allocation_after.csv").sort_values("node_id")
assert np.allclose(x1, after.visitors_10k, atol=1e-6)
assert np.allclose(cap1, after.capacity_10k, atol=1e-6)
checks += len(node_ids) * 2

# Glacier target.
g = node_ids.index("N2")
glacier_util = x1[g] / cap1[g]
assert glacier_util < 0.80
assert abs(x0[g] / np.asarray(cfg["capacity_baseline"])[g] - 0.95) < 0.01
checks += 2

# Max flow, independent cut formula for the star network: sum min(edge, demand).
sh0 = x0 / x_total
sh1 = x1 / x_total
flow_rows = pd.read_csv(OUT / "M5_network_flow.csv").set_index("scenario")
for tag, shares, edges in [("before", sh0, {e["to"]: e["capacity_baseline"] for e in cfg["edges"]}),
                           ("after", sh1, edge_cap)]:
    mf = 0.0
    for nid in node_ids[1:]:
        i = node_ids.index(nid)
        mf += min(edges[nid], shares[i] * daily_peak)
    saved = flow_rows.loc[tag, "max_flow_10k"]
    assert abs(mf - saved) < 0.01
checks += 2

# Transfer matrix: conservation and consistency.
T = pd.read_csv(OUT / "M5_transfer_matrix.csv", index_col=0).to_numpy()
assert np.allclose(T.sum(axis=0), x1, atol=1e-6)  # column totals = post allocation
assert np.allclose(T.sum(axis=1), x0, atol=1e-6)  # row totals = pre allocation
assert np.all(T >= 0)
checks += len(node_ids) * 2 + 1

# Delivered figures.
fig_index = pd.read_csv(OUT / "figure_index_en.csv")
assert len(fig_index) == 4
for name in fig_index.file:
    for ext in ["png", "svg"]:
        assert (ROOT / "figures" / f"{name}.{ext}").stat().st_size > 1000
checks += len(fig_index) * 2

result = {
    "status": "passed",
    "M5_independent_checks": checks,
    "glacier_utilization_before": float(x0[g] / np.asarray(cfg["capacity_baseline"])[g]),
    "glacier_utilization_after": float(glacier_util),
    "max_flow_feasible_after": bool(flow_rows.loc["after", "feasible"]),
    "figure_pairs": len(fig_index),
    "scope": "Cross-checks the gravity allocation, investment response, max-flow cut and transfer matrix; not empirical calibration.",
}
(OUT / "independent_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
