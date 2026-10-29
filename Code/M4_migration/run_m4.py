"""Run M4: feature mapping, AHP-entropy weights, TOPSIS priorities, recalibration."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from model import ahp, entropy_weights, combine_weights, destination_weights, topsis, recalibrate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def main():
    cfg = json.loads((ROOT / "config.json").read_text())
    features = cfg["features"]
    fids = [f["id"] for f in features]
    fnames = [f["name_en"] for f in features]
    scale = cfg["score_scale"]
    destinations = list(cfg["raw_scores"].keys())

    # 1. Feature matrix (raw 1-5 -> normalized intensity).
    raw = {d: np.array(cfg["raw_scores"][d], float) for d in destinations}
    intensity = {d: raw[d] / scale for d in destinations}

    feature_rows = []
    for d in destinations:
        for i, fid in enumerate(fids):
            feature_rows.append(dict(destination=d, feature_id=fid, feature_name=fnames[i],
                                     raw=int(raw[d][i]), intensity=intensity[d][i]))
    pd.DataFrame(feature_rows).to_csv(OUT / "M4_features.csv", index=False, encoding="utf-8-sig")

    # 2. AHP weights.
    ahp_w, lambda_max, ci, cr = ahp(cfg["ahp_matrix"])

    # 3. Entropy weights (discrimination across the two destinations).
    decision = np.vstack([intensity[d] for d in destinations])
    entropy_w = entropy_weights(decision)

    # 4. Combined weights.
    combined_w = combine_weights(ahp_w, entropy_w, cfg["theta"])
    pd.DataFrame({"feature_id": fids, "feature_name": fnames,
                  "ahp": ahp_w, "entropy": entropy_w, "combined": combined_w}).to_csv(
        OUT / "M4_feature_weights.csv", index=False, encoding="utf-8-sig")

    # 5. Feature deviation Venice vs Juneau.
    base_d = "Juneau"
    dev = {d: intensity[d] - intensity[base_d] for d in destinations}

    # 6. Threshold recalibration.
    threshold_rows = []
    recalibrated = {}
    for t in cfg["thresholds"]:
        value, factor, raw_value = recalibrate(t["base"], t["gamma"], dev["Venice"], t["bounds"])
        recalibrated[t["key"]] = value
        threshold_rows.append(dict(key=t["key"], name=t["name_en"], juneau_base=t["base"],
                                   factor=factor, raw_recalibrated=raw_value, venice_value=value,
                                   lower=t["bounds"][0], upper=t["bounds"][1]))
    pd.DataFrame(threshold_rows).to_csv(OUT / "M4_thresholds.csv", index=False, encoding="utf-8-sig")

    # 7. Per-destination policy TOPSIS.
    relevance = np.array(cfg["policy_feature_relevance"], float)
    policies = cfg["policies"]
    power = cfg.get("intensity_power", 1.0)
    ranking_rows = []
    top_policies = {}
    for d in destinations:
        attention = intensity[d] ** power
        eff_w = destination_weights(combined_w, attention)
        closeness, d_plus, d_minus = topsis(relevance, eff_w, benefit=True)
        order = np.argsort(-closeness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        for r, p in enumerate(policies):
            ranking_rows.append(dict(destination=d, policy_id=p["id"], policy_name=p["name_en"],
                                     closeness=closeness[r], d_plus=d_plus[r], d_minus=d_minus[r],
                                     rank=int(ranks[r])))
        top_policies[d] = [policies[i]["name_en"] for i in order[:3]]
    ranking = pd.DataFrame(ranking_rows)
    ranking.to_csv(OUT / "M4_policy_ranking.csv", index=False, encoding="utf-8-sig")

    # 8. Summary.
    summary = dict(
        destinations=destinations,
        ahp=dict(lambda_max=lambda_max, CI=ci, CR=cr, acceptable=cr < 0.1),
        feature_weights=dict(combined={fids[i]: float(combined_w[i]) for i in range(len(fids))}),
        venice_thresholds={k: float(v) for k, v in recalibrated.items()},
        top_three_policies=top_policies,
        global_optimality_proven=False,
        note="Semi-quantitative expert coding; not calibrated to Venice microdata.",
    )
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
