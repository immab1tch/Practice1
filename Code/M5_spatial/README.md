# M5 — Hot-cold spatial equilibrium (gravity model)

This module answers how to relieve the overloaded Mendenhall Glacier **without
capping the total number of visitors**. It follows the v7.0 model chapter 7:
allocate visitors to spatial nodes by a gravity model, invest in cold nodes, and
verify the diversion with a max-flow cut-set.

## 1. Four spatial nodes

| Node | Type | Baseline capacity (10k) |
|---|---|---:|
| N1 Port / downtown | hot | 80.0 |
| N2 Mendenhall Glacier | hot | 49.6 |
| N3 Whale watching | cold | 35.0 |
| N4 Rainforest | cold | 28.0 |

This is the **four-node** model specified in v7.0; it is not the five-node
mixed-integer program kept in `archive/`.

## 2. Gravity allocation (M5-1)

The recommended total x = 147.2 万 (from M1) is distributed by attractiveness
and traffic resistance:

```
x_i = A_i (1 + r_i)^(-gamma) / sum_j A_j (1 + r_j)^(-gamma) * x
```

- `A_i` is the effective attractiveness (吸引力); `r_i` is the traffic
  resistance (交通阻抗), derived from travel time from the port (0 / 20 / 35 /
  40 min, normalized).
- `gamma = 0.5` is the resistance-decay exponent. The PDF labels this exponent
  by the elasticity symbol ε; because the M1 elasticity ε = −0.5 is a *price*
  elasticity (negative), the spatial decay must use its positive magnitude, so
  the implemented exponent is gamma = |ε| = 0.5. This keeps the resistance
  factor decreasing with impedance, as a gravity model requires.

**Baseline allocation** (before investment):

| Node | Visitors (10k) | Share | Utilization |
|---|---:|---:|---:|
| N1 | 70.6 | 0.480 | 0.88 |
| N2 | 47.1 | 0.320 | **0.95** |
| N3 | 17.9 | 0.121 | 0.51 |
| N4 | 11.6 | 0.079 | 0.41 |

The glacier runs at 0.95 of its capacity.

## 3. Cold-point investment (5 million USD)

The 500万 budget is split between marketing (raises awareness/attractiveness)
and transit (lowers resistance, raises capacity and road throughput):

| Node | Total | Marketing | Transit | Awareness factor | Capacity (10k) |
|---|---:|---:|---:|---:|---:|
| N3 | 300万 | 150万 | 150万 | 0.44 → 0.89 | 35 → 44 |
| N4 | 200万 | 100万 | 100万 | 0.49 → 0.79 | 28 → 34 |

Cold nodes have a high *potential* attractiveness (whale watching 0.75,
rainforest 0.45) but a low initial awareness factor; marketing moves the
effective attractiveness toward that potential.

**Allocation after investment:**

| Node | Visitors (10k) | Share | Utilization |
|---|---:|---:|---:|
| N1 | 59.2 | 0.402 | 0.74 |
| N2 | 39.5 | 0.269 | **0.797** |
| N3 | 32.2 | 0.219 | 0.73 |
| N4 | 16.3 | 0.110 | 0.48 |

The glacier utilization falls from **0.95 to 0.797, below the 0.80 target**,
and total visitors are unchanged.

> **Calibration provenance (read carefully).** The v7.0 PDF states the *target*
> (500万 brings the glacier below 0.80) but supplies no response coefficients,
> so the primary β set was **target-calibrated** — chosen so the framework
> reproduces that stated outcome. The 0.797 figure is therefore a consistency
> demonstration, **not an independent proof that 500万 suffices**. This is
> recorded in `config.json` (`calibration_provenance`) and quantified below.

## 3a. Coefficient sensitivity

Recomputing the same investment with alternative (un-fitted) coefficient sets:

| Coefficient set | Glacier utilization | Below 0.80? | Max flow feasible? |
|---|---:|:---:|:---:|
| Calibrated (target) | 0.797 | Yes | Yes |
| Neutral round coefficients | 0.800 | No (boundary) | Yes |
| Conservative | 0.843 | No | No |
| Optimistic | 0.757 | Yes | Yes |

The honest conclusion: 500万 moves the glacier **to the 0.80 threshold** under
neutral assumptions; whether it lands just below or just above depends on the
assumed response coefficients. The diversion mechanism and the road-bottleneck
finding are robust; the exact margin is not. See `M5_coefficient_sensitivity.csv`
and figure `M5_05`.

## 4. Max-flow cut-set verification

The daily peak (1.84万) is routed from the port over road/boat edges to a
super-sink. The star network's max flow is the sum of min(edge, demand) per
node; `scipy.sparse.csgraph.maximum_flow` gives the same result.

| Scenario | Max flow (10k) | Outgoing demand (10k) | Feasible |
|---|---:|---:|:---:|
| Before | 0.87 | 0.96 | No |
| After | 1.10 | 1.10 | **Yes** |

Before investment the road cut is the bottleneck (which is why pressure
concentrates at the glacier); transit spending raises the max flow to meet
demand. This confirms the cold-point investment is genuinely limited by road
capacity, as the PDF states.

## 5. Hot-cold transfer matrix

The reallocation moves about **19.0万 visitors** to the cold nodes. The gravity
model identifies net changes; the delivered 4×4 transfer matrix distributes
each hot loss to the cold gains in proportion to their gain (rows = origin,
columns = destination):

| From \ To | N1 | N2 | N3 | N4 |
|---|---:|---:|---:|---:|
| N1 | 59.2 | 0 | 8.6 | 2.8 |
| N2 | 0 | 39.5 | 5.7 | 1.9 |
| N3 | 0 | 0 | 17.9 | 0 |
| N4 | 0 | 0 | 0 | 11.6 |

## 6. Tax discretization

A continuous per-node tax multiplied by a continuous flow would create a
bilinear term (two decision variables multiplied). The model avoids this:
- The spatial gravity uses the *traffic resistance* (fixed from travel time),
  not a node-specific tax.
- The tourism fee is the M1 global rate held fixed; if a node fee were varied,
  it would be snapped to the discrete brackets {0.09, 0.12, 0.15} rather than
  solved continuously. No continuous×continuous product is therefore present.

## 7. Reproduce

```bash
python run_m5.py      # compute, write results/
python render_m5.py   # English figures (PNG + SVG, 300 dpi) -> figures/
python verify_m5.py   # 45 independent cross-checks
```

## 8. Boundary statements (honest scope)

- Node attractiveness, resistance, capacity and the investment-response
  coefficients are **declared semi-quantitative inputs/assumptions**, calibrated
  to reproduce the known visitor concentration, not fitted to microdata.
- The primary β set is **target-calibrated** to the PDF's stated 0.80 outcome;
  under neutral coefficients the glacier sits at the 0.80 boundary. The model
  does not independently prove 500万 is sufficient.
- The transfer matrix is a proportional reconstruction of net changes, not an
  observed individual-level movement table.
- `global_optimality_proven = false`: this is a spatial simulation, not a
  globally optimized solution.
