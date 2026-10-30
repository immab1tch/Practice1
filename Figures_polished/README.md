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
```

The full list of figures and their data sources is in `figure_index.csv`.
