"""Record checked primary-source values; never imported by the existing model.

Daily vouchers transcribed from the visible labels of CDA_2024_official.pdf p11.
All other values retain source scope and admissible uses in the JSON.
Run with Python 3.9+; standard library only.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "CDA24": {
        "title": "Comune di Venezia, Primi risultati, conferenza stampa 19 luglio 2024",
        "url": "https://dam.comune.venezia.it/api/resource/c1f4fd7d-88ce-4b39-9301-fcfd46a373be",
        "local_file": "originals/CDA_2024_official.pdf",
        "locator": "PDF pp10-11; p11 graph visually checked",
        "published": "2024-07-19", "evidence": "official preliminary administrative counts",
    },
    "CDA25": {
        "title": "Comune di Venezia, Contributo di accesso: i numeri del 2025",
        "url": "https://live.comune.venezia.it/it/2025/08/contributo-di-accesso-i-numeri-del-2025",
        "published": "2025-08-04", "evidence": "official preliminary administrative counts",
        "after_2025_competition": True,
    },
    "FLOW": {
        "title": "Ri-Pensare Venezia, Turisti e City Users: la fotografia aggiornata dei flussi",
        "url": "https://www.ripensarevenezia.it/turisti-e-city-users-la-fotografia-aggiornata-dei-flussi/",
        "published": "2025, before February workshops; exact date not shown",
        "evidence": "research-team summary of mobile-positioning estimates, not raw daily observations",
        "after_2025_competition": True,
    },
    "TCC": {
        "title": "Bertocchi et al. (2020), Venice and Overtourism",
        "url": "https://www.mdpi.com/2071-1050/12/2/512?type=check_update&version=1",
        "doi": "10.3390/su12020512", "published": "2020-01-09",
        "locator": "Tables 1-2, sections 3-4",
        "evidence": "peer-reviewed model inputs, mixed historical sources including 2018 survey",
    },
    "WTP": {
        "title": "Fabio Rubbi, Evaluating Tourists’ Willingness to Pay for an Entrance Fee to Cultural Heritage Cities: The Case Study of Venice",
        "url": "https://unitesi.unive.it/retrieve/b0fa03f6-59df-4605-a683-b9d0dc74bae9/Tesi_FabioRubbi_pdfA.pdf",
        "landing_url": "https://unitesi.unive.it/handle/20.500.14247/24139",
        "published": "academic year 2023/2024; public availability before competition not verified",
        "locator": "PDF pp23-27,34-36,40-43,51-52; equation 3.3 and table 3.2",
        "evidence": "master thesis; stated-preference survey, not causal observed-demand elasticity",
        "availability_before_competition_verified": False,
    },
    "CONTRACT": {
        "title": "City council question 903, quoted 2024 Vela service contract",
        "url": "https://consiglio2020.comune.venezia.it/atti-ispettivi/1/atto/913?m=3",
        "published": "2024-01-23",
        "evidence": "contract excerpt quoted in council question; not final full-program accounts",
    },
    "REPLY": {
        "title": "Finance assessor reply, protocol 73545",
        "url": "https://consiglio2020.comune.venezia.it/pdf/atti_ispettivi/risposte_1/843_risposta_interrogazione_902.pdf",
        "published": "2024-02-12", "locator": "p1",
        "evidence": "official clarification of channel-specific sales commission",
    },
    "VELA": {
        "title": "VE.LA. S.p.A. Bilancio di esercizio 2024",
        "url": "https://vela.avmspa.it/sites/default/files/bilancio%20vela%202024%20-%20no%20signed_low.pdf",
        "local_file": "originals/Vela_2024.pdf", "locator": "PDF p12, management report printed p6",
        "published": "2025", "after_2025_competition": True,
        "evidence": "provider annual accounts; provider revenue is not complete municipal program cost",
    },
}

dates = ["04-25", "04-26", "04-27", "04-28", "04-29", "04-30",
         "05-01", "05-02", "05-03", "05-04", "05-05", "05-11", "05-12",
         "05-18", "05-19", "05-25", "05-26", "06-08", "06-09", "06-15",
         "06-16", "06-22", "06-23", "06-29", "06-30", "07-06", "07-07",
         "07-13", "07-14"]
counts = [16201, 24686, 23015, 17264, 18406, 17104, 19835, 16122,
          19524, 21757, 13368, 19568, 14155, 20589, 19377, 17908,
          16395, 14477, 13139, 16809, 11607, 15285, 14895, 18296,
          13035, 13858, 12093, 14886, 11408]
daily = [{"date": "2024-" + d, "weekday": dt.date.fromisoformat("2024-" + d).weekday(),
          "paid_vouchers": n, "fee_eur": 5, "source_id": "CDA24"}
         for d, n in zip(dates, counts)]

data = {
    "retrieved_on": "2026-10-29",
    "scope": "Supplement for Venice optimization research; no existing model parameters replaced",
    "sources": SOURCES,
    "daily_2024": {
        "unit": "paid visit vouchers per chargeable day, not all visitors or distinct people",
        "spatial_scope": "Citta antica, fee-eligible visits in 08:30-16:00",
        "weekday_definition": "Monday=0, Sunday=6",
        "outside_list": "unobserved here; never impute zero visitors",
        "rows": daily,
    },
    "fee_totals": [
        {"year": 2024, "days": 29, "paid_vouchers": 485062, "gross_eur": 2425310,
         "source_id": "CDA24", "status": "preliminary, July 2024 release"},
        {"year": 2025, "days": 54, "paid_vouchers": 723497, "gross_eur": 5421425,
         "platform_vouchers": 704467, "off_platform_vouchers": 19030,
         "fee_options_eur": [5, 10], "source_id": "CDA25",
         "status": "preliminary, August 2025 release; use for later validation"},
    ],
    "day_visitors": {
        "year": 2024, "annual_visit_estimate": 6300000, "source_id": "FLOW",
        "scope": "historic centre; mobile-positioning classification of excursionists",
        "annual_estimate_divided_by_366": 6300000 / 366,
        "reported_approx_daily_estimate": 20000,
        "use": "alternative demand-base scenario; classification must be aligned before calibration",
        "warning": "Source daily approximation and annual sum do not reconcile exactly. No unique-person or chargeable-share interpretation; do not combine with lodging nights as distinct visitors.",
    },
    "spending_prior": {
        "source_id": "TCC", "unit": "EUR per visitor per day",
        "price_year": "not uniquely specified; historical inputs, not 2023/2024 price-adjusted",
        "hotel": {"lower": 190, "central": 210, "upper": 230},
        "extra_hotel": {"lower": 140, "central": 160, "upper": 180},
        "day_tripper": {"lower": 45, "central": 60, "upper": 80},
        "use": "literature ranges for scenarios, not a contemporary population estimate or confidence interval",
    },
    "waste_prior": {
        "source_id": "TCC", "unit": "kg per visitor per day",
        "hotel": [2.1, 2.3, 2.5], "extra_hotel": [1.8, 2.0, 2.2],
        "day_tripper": [0.9, 1.0, 1.1],
        "use": "historical literature scenario only; validate before contemporary calibration",
    },
    "wtp_prior": {
        "source_id": "WTP", "survey_period": "September-December 2024",
        "pilot_responses": 216, "final_raw_responses": 176, "analysis_n": 153,
        "mean_eur": 13.08, "median_eur": 7.00,
        "open_ended_survival": [
            {"fee_eur": f, "share": q / 100}
            for f, q in zip([5, 10, 15, 20, 25, 30, 40, 50],
                            [67.97, 46.41, 32.68, 27.45, 21.57, 14.38, 8.50, 6.54])
        ],
        "dc_logit_bid_coefficient": -0.067, "dc_logit_bid_standard_error": 0.017,
        "use": "alternative stated-preference demand scenario only",
        "warning": "Logit coefficient is not an elasticity. Mixed day/overnight sample, English-only self-selection, hypothetical fee additional to lodging tax. Do not treat the study's revenue-maximising fee as our sustainability optimum.",
    },
    "cost_evidence": [
        {"metric": "2024 quoted Vela fixed service remuneration excluding VAT", "value_eur": 229871,
         "source_id": "CONTRACT", "use": "partial service-contract component, not total annual collection cost"},
        {"metric": "commission on tracked Venezia Unica referral sales", "value_fraction": 0.05,
         "source_id": "REPLY", "use": "multiply by eligible channel sales only; direct cda.ve.it access has no such Vela fee"},
        {"metric": "Vela reported revenue increment associated with access-fee management", "value_eur_approx": 1420000,
         "source_id": "VELA", "use": "provider-side scale check, not municipal total cost; do not add to fixed remuneration without de-duplication"},
        {"metric": "2025 allocation to reducing waste charges", "lower_bound_eur": 1500000,
         "source_id": "CDA25", "use": "observed reported allocation, not unrestricted surplus or complete net income"},
    ],
    "unidentified": ["causal demand elasticity for fee-eligible day visitors",
                     "full daily visitors with chargeable eligibility and exemptions aligned",
                     "current site/time-specific sustainable capacity",
                     "full municipal collection costs, separated fixed/variable/start-up/recurring",
                     "local income floor and social/environmental preference constraints",
                     "causal investment-response coefficients"],
    "quarantined_claims": [
        {"source_id": "TCC", "issue": "Restaurant capacity 240000 in Table 1 versus central 24000 in equation 12; St Mark capacity 10000 versus 100000. Table 2 current-case day trippers 3250 differs from section 4's 14611 (alternative cases).",
         "action": "Do not auto-import whole capacity matrix or impose 52111 as current hard capacity."},
        {"source_id": "CDA25", "issue": "Reported daily mean 13046 corresponds approximately to platform-only 704467/54, not all 723497 vouchers. The retrospective 2024 mean 16676 differs from the 2024 official graph/total.",
         "action": "Compute means from explicit matching numerator and day count; never infer causal elasticity from year averages."},
        {"source_id": "FLOW", "issue": "Annual 6.3m versus approximate 20k/day; wording of >100k peak overlaps day-tripper/total-tourist categories.",
         "action": "Keep annual estimate as scenario; do not use ambiguous peak to calibrate day-tripper peak factor."},
        {"source_id": "CDA24", "issue": "Indexed 2025 official report mentions different retrospective 2024 receipt 2249505; full report link unavailable (404).",
         "action": "Keep release-specific July 2024 gross 2425310; alternative amount unresolved, not silently substituted or called final net revenue."},
    ],
}

checks = {
    "29_distinct_days": len(daily) == len(set(dates)) == 29,
    "chronological_dates": [r['date'] for r in daily] == sorted(r['date'] for r in daily),
    "positive_integer_counts": all(isinstance(n, int) and n > 0 for n in counts),
    "daily_sum_matches_official_total": sum(counts) == 485062,
    "gross_matches_five_euro_fee": sum(counts) * 5 == 2425310,
    "2025_channels_reconcile": 704467 + 19030 == 723497,
    "wtp_probabilities_nonincreasing": all(a['share'] >= b['share'] for a,b in zip(data['wtp_prior']['open_ended_survival'], data['wtp_prior']['open_ended_survival'][1:])),
    "all_spending_ranges_ordered": all(x['lower'] <= x['central'] <= x['upper'] for k,x in data['spending_prior'].items() if isinstance(x,dict)),
}
assert all(checks.values()), checks
data['derived_fee_means'] = [dict(year=r['year'], paid_vouchers_per_chargeable_day=r['paid_vouchers']/r['days']) for r in data['fee_totals']]
(ROOT/'优化参数与来源.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
(ROOT/'核对记录.json').write_text(json.dumps({
    'checks': checks, 'daily_chart_visual_review': 'CDA24 PDF p11 date and value labels inspected',
    'vela_visual_review': 'PDF p12 inspected',
    'wtp_visual_review': 'not completed; web text read; screenshot/download blocked',
    'capacity_visual_review': 'web HTML tables read; PDF download blocked',
    'interpretation': 'Arithmetic/transcription checks do not establish causal identification or empirical model validity',
    'sha256': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
               for f in (ROOT/'originals').glob('*.pdf')},
}, ensure_ascii=False, indent=2) + '\n')
print(f'Recorded {len(daily)} daily observations; {len(checks)} arithmetic checks passed.')
