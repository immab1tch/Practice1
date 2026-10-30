# Polished Figures

A curated, presentation-ready set of figures for the 2025 MCM/ICM Problem B
(*Managing Sustainable Tourism in Juneau*). The styling follows the visual
language of the six award-winning papers in `Reference/award_papers_B2025`:
low-saturation Morandi/macaron colors, generous whitespace, thin or removed
spines, no chart junk, and smooth gradients used only where they aid reading.

## Design system

The shared system is in `_style.py`.

- **Palette (hex):** ink `#2E3438`, teal `#3E7B8C`, coral `#DE8868`,
  gold `#D2A450`, sage `#8CA48B`, rose `#C78A9E`, slate `#6E8CA8`,
  lavender `#9B8FC0`, warm gray `#B6AEA4`, paper `#FBFAF7`.
- **Custom colormaps:** `cmap_teal`, `cmap_coral`, `cmap_gold`, `cmap_rose`,
  a diverging teal-neutral-coral map, and a sequential heat map
  (slate-teal-sage-gold-coral).
- **Gradient bars:** a vertical/horizontal gradient is clipped to a rounded
  rectangle (`FancyBboxPatch`); simple solid fills are kept for small multiples.
- **Typography/export:** DejaVu Sans, top/right spines removed, each figure
  saved as 300-dpi PNG and editable SVG with fonts converted to paths.

## Correctness guarantee

Every polished figure is a **restyled copy of a frozen result**, not a new
computation. The render scripts only read the frozen CSVs under each module's
`results/` directory; they never rerun or alter a model.

`verify_polished.py` independently recomputes the underlying numbers and
compares them against the frozen sources (file integrity, weight sums,
conservation of visitors, ranking order, fee anchors, and ledger shares).
Run it with:

```
python verify_polished.py
```

The latest run reports `status: passed` with 54 independent checks
(see `polished_checks.json`). No data or result was changed for appearance.

## Original figures

The original figures remain untouched in their module folders, e.g.
`Code/M1_equilibrium/results/figures`, `Code/M4_migration/figures`, and
`Code/M5_spatial/figures`. Nothing was deleted or overwritten.

## Reproduction

```
python render_m1.py
python render_m2.py
python render_m3.py
python render_m4.py
python render_m5.py
python render_overview.py
python render_premium.py
```

The full list of figures and their data sources is in `figure_index.csv`.

## Premium upgrade (second round)

To close the remaining visual gap with the strongest papers, `render_premium.py`
adds four figures that the references use but the first round did not include:

- **3D Pareto front** (`M1P_03`): the 458 real non-dominated points plotted in
  visitor-revenue-environment space and colored by satisfaction, with the
  recommended point marked as a star.
- **Gradient feasibility region** (`M1P_04`): a smooth monotonic revenue curve
  (PCHIP interpolation of the real points) with a gradient fill and an annotated
  optimum.
- **Policy radar** (`M4P_04`): Juneau vs. Venice TOPSIS profiles on seven axes.
- **Conceptual schematic** (`OVP_02`): natural vs. policy-shaped visitor
  distribution against a carrying-capacity line. Axes are normalized and it is
  explicitly labeled "schematic, not to scale"; it carries no measurements.

PCHIP smoothing and gradient fills are visual only; the real sample points are
kept on the chart, and no frozen value is changed. The verification now covers
15 figures with 70 independent checks.

## Honest limits of the remaining gap

A few elements in the reference screenshots cannot be reproduced exactly from
this project, for identifiable reasons rather than effort:

1. **Modeling difference.** The reference 3D surfaces are dense continuous
   Pareto surfaces. Our M1 front is a 458-point NSGA-II set, so an honest 3D
   figure is a point cloud (with a star), not a filled parametric surface.
   Fitting a continuous surface would present an approximation as a result.
2. **Data-structure difference.** Many reference heatmaps are smooth grids over
   continuous parameter sweeps. Our frozen heatmaps are small discrete matrices
   (e.g. 4x4, seven features); producing extra grid cells would require running
   the model over a new sweep, which is outside a restyling task and would
   invent values not present in the frozen results.
3. **Tooling / human edge.** Pixel-perfect editorial schematics, hand-drawn
   conceptual diagrams and bespoke callouts are faster to fine-tune by hand in
   vector tools (Adobe Illustrator, Figma, or PowerPoint) than through a
   plotting library. Matplotlib reaches roughly the same information design but
   the last mile of hand-tuned layout is where a human has the advantage.

