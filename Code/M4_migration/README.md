# M4 — Semi-quantitative destination migration (Juneau → Venice)

This module demonstrates how the Juneau framework can be **transferred to a new
destination** without that destination's microdata. It follows the
"半定量推演" (semi-quantitative deduction) chapter of the v7.0 model: map the
new destination onto a shared pressure-feature space, re-weight the criteria,
re-rank policy types, and recalibrate the thresholds.

> **Provenance (read carefully).** The v7.0 PDF supplies the **method**
> (AHP + entropy + TOPSIS + threshold recalibration) and the **qualitative
> expected conclusions** (Juneau: capacity + shore power; Venice: daily cap,
> high fee, rent subsidy), but it does **not** list the seven feature scores,
> the AHP pairwise matrix, or the policy–feature relevance coefficients. Every
> concrete number in `config.json` is therefore a **declared expert coding
> constructed for this demonstration**, chosen to be consistent with the PDF's
> stated conclusions; it is not measured Venice data.

## 1. Pressure-feature space

Seven pressure features are coded on a 1–5 expert scale (1 = negligible,
5 = severe) from publicly-known destination characteristics:

| Feature | Juneau | Venice |
|---|---:|---:|
| F1 Natural-environment pressure | 5 | 2 |
| F2 Density and crowding | 2 | 5 |
| F3 Housing displacement | 2 | 5 |
| F4 Visitor concentration | 4 | 5 |
| F5 Transport/infra bottleneck | 3 | 5 |
| F6 Ecosystem fragility | 5 | 4 |
| F7 Resident social conflict | 3 | 5 |

The two profiles are structurally different: Juneau is **nature-dominated**
(F1/F6), Venice is **urban/social-dominated** (F2/F3/F5/F7).

## 2. Feature weighting (AHP + entropy)

- **AHP** (analytic hierarchy process, 层次分析法): a 7×7 reciprocal
  comparison matrix gives the structural weights. Result:
  λ_max = 7.106, CI = 0.0177, **CR = 0.0134 < 0.1** (consistent).
- **Entropy weights** (熵权): discrimination across the two destinations.
  Features shared by both (e.g. F4/F5/F6) receive low entropy weight.
- **Combined weights** (θ = 0.5):

| Feature | AHP | Entropy | Combined |
|---|---:|---:|---:|
| F1 | 0.110 | 0.263 | **0.187** |
| F2 | 0.065 | 0.263 | **0.164** |
| F3 | 0.149 | 0.263 | **0.206** |
| F4 | 0.191 | 0.017 | **0.104** |
| F5 | 0.120 | 0.088 | **0.104** |
| F6 | 0.231 | 0.017 | **0.124** |
| F7 | 0.134 | 0.088 | **0.111** |

## 3. Policy-priority ranking (TOPSIS)

For each destination the effective weights are the combined weights times the
destination's feature intensity (a power transform, exponent 2, so that
priorities are driven by the most severe pressure gaps); the seven policy types
are ranked by **TOPSIS** (technique for order of preference by similarity to
ideal solution, 逼近理想解排序) closeness to the ideal over the policy–feature
relevance matrix.

**Juneau** (closeness): Shore power 0.758 > Capacity expansion 0.677 >
Transit management 0.512 > Daily cap 0.458 > High fee 0.402 > Rent subsidy 0.147
> Alcohol control 0.144.

**Venice** (closeness): Daily cap 0.626 > Rent subsidy 0.611 > High fee 0.528 >
Transit management 0.420 > Capacity expansion 0.406 > Shore power 0.232 >
Alcohol control 0.225.

This matches the expected migration logic: a nature-cruise destination leads
with **shore power + capacity expansion** (relieve pressure without sacrificing
volume), while a dense urban destination leads with a **daily cap, rent subsidy
and high tourism fee** (control demand and protect residents).

**Ranking robustness.** Perturbing every feature score by ±1 (2,000 Monte Carlo
re-codings), the top-two set is retained for **Juneau in 95%** of cases and for
**Venice in 62%**. The Juneau ranking is robust; the Venice ordering within the
{daily cap, rent subsidy, high fee} cluster is sensitive to the exact coding,
although that cluster as a whole remains the priority.

## 4. Threshold recalibration

Thresholds are moved by the feature deviation between the destinations,
θ^V = clip(θ^J · [1 + Σ γ_k · dev_k], bounds):

| Threshold | Juneau | Venice (recalibrated) |
|---|---:|---:|
| Resident acceptance floor S_min | 0.70 | **0.89** |
| Social pressure ceiling H_max | 1.00 | **0.78** |
| Ecological pressure ceiling E_max | 1.00 | **0.83** |
| Daily peak ceiling (10k/day) | 2.00 | **1.80** |
| Revenue floor / baseline | 0.92 | **0.828** |

The tighter urban/social thresholds raise the acceptance floor and lower the
crowding ceilings. The absolute Venice revenue floor is the ratio times
**Venice's own baseline**, which is not supplied here.

## 5. Reproduce

```bash
python run_m4.py      # compute, write results/
python render_m4.py   # English figures (PNG + SVG, 300 dpi) -> figures/
python verify_m4.py   # 52 independent cross-checks
```

## 6. Boundary statements (honest scope)

- Every feature score, policy–feature relevance coefficient and threshold
  sensitivity is a **declared semi-quantitative expert coding**, not a measured
  or fitted value.
- The entropy weights are based on only two destinations and are therefore
  intentionally coarse; they are combined with AHP rather than used alone.
- Absolute Venice revenue/visitor numbers cannot be produced without Venice's
  baseline; the module delivers ratios, relative priorities and recalibrated
  thresholds only.
- `global_optimality_proven = false`: this is a multi-criteria framework
  demonstration, not a globally optimized solution.
