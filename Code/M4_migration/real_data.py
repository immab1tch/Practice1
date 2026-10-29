"""M4 real-data engine: evidence-grounded Venice fee scenarios.

Unlike the AHP/TOPSIS framework (model.py), every number here is anchored to the
frozen primary-source evidence in inputs/real_venice_evidence.json:
  - the 29 observed 2024 chargeable days (administrative voucher counts),
  - the stated-preference willingness-to-pay (WTP) survival curve,
  - literature spending/waste ranges and partial management costs.
The WTP curve supplies the *relative* response to the fee; the absolute level is
anchored to the real 5-EUR count. No causal elasticity is observed or claimed.
"""
import numpy as np
import pandas as pd

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def voucher_daily(evidence):
    """The 29 observed chargeable days with weekend flag."""
    rows = []
    for r in evidence["daily_2024"]["rows"]:
        rows.append(dict(date=r["date"], weekday=WEEKDAY_NAMES[r["weekday"]],
                         is_weekend=bool(r["weekday"] >= 5),
                         paid_vouchers=r["paid_vouchers"], fee_eur=r["fee_eur"]))
    return pd.DataFrame(rows)


def voucher_statistics(daily):
    """Peak/trough and weekday/weekend structure of the fee-paying population."""
    v = daily.paid_vouchers.to_numpy(float)
    wd = daily.loc[~daily.is_weekend, "paid_vouchers"]
    we = daily.loc[daily.is_weekend, "paid_vouchers"]
    return pd.DataFrame([
        dict(metric="chargeable days", value=len(daily)),
        dict(metric="mean vouchers / day", value=float(v.mean())),
        dict(metric="standard deviation", value=float(v.std(ddof=1))),
        dict(metric="coefficient of variation", value=float(v.std(ddof=1) / v.mean())),
        dict(metric="minimum (trough)", value=float(v.min())),
        dict(metric="maximum (peak)", value=float(v.max())),
        dict(metric="weekday mean", value=float(wd.mean())),
        dict(metric="weekend mean", value=float(we.mean())),
        dict(metric="weekend / weekday ratio", value=float(we.mean() / wd.mean())),
    ])


def fee_scenarios(evidence):
    """Conditional fee scenarios anchored to the real 5-EUR administrative count.

    vouchers(f) = anchor_2024 * [q(f) / q(5)]; gross = vouchers * f, where q(f)
    is the stated-preference WTP survival share. The 5-EUR row reproduces the
    observed 485,062 vouchers and 2,425,310 EUR gross.
    """
    anchor = 485062
    wtp = evidence["wtp_prior"]["open_ended_survival"]
    q5 = next(w["share"] for w in wtp if w["fee_eur"] == 5)
    spend = evidence["spending_prior"]["day_tripper"]
    rows = []
    for w in wtp:
        f = w["fee_eur"]
        multiplier = w["share"] / q5
        vouchers = anchor * multiplier
        gross = vouchers * f
        onsite_lo, onsite_ce, onsite_hi = (vouchers * spend["lower"],
                                           vouchers * spend["central"],
                                           vouchers * spend["upper"])
        rows.append(dict(fee_eur=f, wtp_survival=w["share"],
                         participation_multiplier=multiplier,
                         estimated_vouchers=vouchers, gross_fee_revenue_eur=gross,
                         onsite_spend_low_eur=onsite_lo, onsite_spend_central_eur=onsite_ce,
                         onsite_spend_high_eur=onsite_hi,
                         total_contribution_central_eur=gross + onsite_ce))
    return pd.DataFrame(rows)


def cost_components(evidence):
    """Document the partial, differently-scoped cost evidence (no fake total)."""
    rows = []
    for c in evidence["cost_evidence"]:
        rows.append(dict(metric=c["metric"],
                         value_eur=c.get("value_eur", c.get("value_eur_approx")),
                         value_fraction=c.get("value_fraction"),
                         scope=c["use"], source_id=c["source_id"]))
    return pd.DataFrame(rows)


def identified_net_anchor(evidence):
    """Conservative net at the real 5-EUR anchor.

    Only the fixed Vela remuneration is identified for 2024; the channel split
    needed for the 5% commission and the full collection cost are not, so the
    result is an upper bound on net revenue, not a settled net figure.
    """
    gross = 2425310
    fixed = next(c["value_eur"] for c in evidence["cost_evidence"]
                 if "fixed service remuneration" in c["metric"])
    return pd.DataFrame([
        dict(item="observed gross (5 EUR, 2024)", eur=gross),
        dict(item="identified fixed cost (Vela, ex VAT)", eur=fixed),
        dict(item="net upper bound (other costs not identified)", eur=gross - fixed),
    ])


def feature_evidence_mapping(evidence):
    """Map each 1-5 feature score to the real evidence behind it (source IDs)."""
    mapping = {
        "F1": ("S1 urban historic centre, no wilderness pressure", 2),
        "F2": ("S1: 3.82m historic-centre arrivals / 9.09m nights", 5),
        "F3": ("S5: 49,304 residents vs 49,693 tourist beds", 5),
        "F4": ("S1: 5.66m municipal arrivals, concentrated in centre", 5),
        "F5": ("S4/S6: large-ship curbs, 25-person group cap", 5),
        "F6": ("S4: lagoon ecosystem under UNESCO protection", 4),
        "F7": ("S2/S5: access-fee rationale, beds exceed residents", 5),
    }
    return pd.DataFrame(
        [dict(feature=fid, expert_score=score, real_evidence=text)
         for fid, (text, score) in mapping.items()])
