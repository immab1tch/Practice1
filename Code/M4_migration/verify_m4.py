"""M4 independent checks: recompute AHP/entropy/TOPSIS/recalibration from raw inputs.

The math is written inline (not imported from model) so this is a genuine
cross-check of the delivered evidence.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

cfg = json.loads((ROOT / "config.json").read_text())
checks = 0

# --- AHP: principal eigenvector (independent method) ---
A = np.asarray(cfg["ahp_matrix"], float)
n = A.shape[0]
assert np.allclose(A, 1.0 / A.T, atol=5e-4)
vals, vecs = np.linalg.eig(A)
idx = np.argmax(vals.real)
ahp_w = np.abs(vecs[:, idx].real)
ahp_w = ahp_w / ahp_w.sum()
lambda_max = vals[idx].real
ci = (lambda_max - n) / (n - 1)
cr = ci / RI[n]
assert cr < 0.1, cr
saved_w = pd.read_csv(OUT / "M4_feature_weights.csv")
assert np.allclose(ahp_w, saved_w.ahp, atol=1e-3)
checks += n + 2

# --- Entropy: inline ---
scale = cfg["score_scale"]
dests = list(cfg["raw_scores"].keys())
mat = np.vstack([np.asarray(cfg["raw_scores"][d], float) / scale for d in dests])
P = mat / mat.sum(axis=0)
e = np.zeros(mat.shape[1])
for j in range(mat.shape[1]):
    p = P[:, j]
    p = p[p > 0]
    e[j] = -np.sum(p * np.log(p)) / np.log(mat.shape[0])
d = 1.0 - e
entropy_w = d / d.sum()
assert np.allclose(entropy_w, saved_w.entropy, atol=1e-6)
checks += n

# --- Combined ---
theta = cfg["theta"]
combined = theta * ahp_w + (1 - theta) * entropy_w
combined = combined / combined.sum()
assert np.allclose(combined, saved_w.combined, atol=1e-3)
checks += n

# --- Threshold recalibration: inline ---
juneau_i = mat[dests.index("Juneau")]
venice_i = mat[dests.index("Venice")]
dev = venice_i - juneau_i
thr_saved = pd.read_csv(OUT / "M4_thresholds.csv").set_index("key")
for t in cfg["thresholds"]:
    factor = 1.0 + np.sum(np.asarray(t["gamma"], float) * dev)
    raw = t["base"] * factor
    value = min(max(raw, t["bounds"][0]), t["bounds"][1])
    assert abs(value - thr_saved.loc[t["key"], "venice_value"]) < 1e-9
    checks += 1

# --- Policy TOPSIS: inline ---
rel = np.asarray(cfg["policy_feature_relevance"], float)
power = cfg.get("intensity_power", 1.0)
rank_saved = pd.read_csv(OUT / "M4_policy_ranking.csv")
for d in dests:
    inten = np.asarray(cfg["raw_scores"][d], float) / scale
    eff = saved_w.combined.to_numpy() * inten ** power
    eff = eff / eff.sum()
    norm = np.sqrt((rel ** 2).sum(axis=0))
    V = (rel / norm) * eff
    ideal = V.max(axis=0)
    anti = V.min(axis=0)
    dplus = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    dminus = np.sqrt(((V - anti) ** 2).sum(axis=1))
    close = dminus / (dplus + dminus)
    sub = rank_saved[rank_saved.destination == d].sort_values("policy_id")
    assert np.allclose(close, sub.closeness, atol=1e-7)
    # Rank-1 policy must match the saved top.
    assert sub.iloc[np.argmax(close)].policy_id == sub.sort_values("rank").iloc[0].policy_id
    checks += len(rel) + 1

# --- Delivered figures ---
fig_index = pd.read_csv(OUT / "figure_index_en.csv")
assert len(fig_index) == 4
for name in fig_index.file:
    for ext in ["png", "svg"]:
        assert (ROOT / "figures" / f"{name}.{ext}").stat().st_size > 1000
checks += len(fig_index) * 2

result = {
    "status": "passed",
    "M4_independent_checks": checks,
    "ahp_consistency_ratio": float(cr),
    "destinations": len(dests),
    "policies": len(rel),
    "figure_pairs": len(fig_index),
    "scope": "Cross-checks the semi-quantitative MCDM math and delivered evidence; not empirical calibration to Venice microdata.",
}
(OUT / "independent_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
