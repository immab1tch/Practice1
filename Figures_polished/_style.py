"""Shared design system for the polished figures.

All polished figures read FROZEN result CSVs and only restyle them; no model
number is recomputed here. The palette is a low-saturation Morandi / macaron
scheme inspired by the 2025 B-problem award papers.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

# ---- Root of the whole project (this file lives in Figures_polished/) ----
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

# ---- Morandi / macaron palette ----
INK = "#2E3438"
TEAL = "#3E7B8C"
TEAL_MID = "#6FA6AE"
TEAL_SOFT = "#B6D3D2"
CORAL = "#DE8868"
CORAL_SOFT = "#EDB59C"
GOLD = "#D2A450"
GOLD_SOFT = "#E5CB92"
SAGE = "#8CA48B"
ROSE = "#C78A9E"
SLATE = "#6E8CA8"
LAVENDER = "#9B8FC0"
WARM_GRAY = "#B6AEA4"
PAPER = "#FBFAF7"

PALETTE = [TEAL, CORAL, GOLD, SAGE, SLATE, ROSE, LAVENDER, TEAL_MID, WARM_GRAY]

# ---- Custom sequential / diverging colormaps ----
cmap_teal = LinearSegmentedColormap.from_list("morandi_teal", [TEAL, TEAL_MID, TEAL_SOFT])
cmap_coral = LinearSegmentedColormap.from_list("morandi_coral", [CORAL, CORAL_SOFT])
cmap_gold = LinearSegmentedColormap.from_list("morandi_gold", [GOLD, GOLD_SOFT])
cmap_diverge = LinearSegmentedColormap.from_list(
    "morandi_div", [TEAL, "#A9C7C9", "#EFE7DC", "#E8B39A", CORAL])
cmap_heat = LinearSegmentedColormap.from_list(
    "morandi_heat", [SLATE, TEAL, SAGE, GOLD, CORAL])
cmap_rose = LinearSegmentedColormap.from_list("morandi_rose", [ROSE, "#E2C4CE"])


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.edgecolor": "#C9C2B8",
        "axes.linewidth": 0.9, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "svg.fonttype": "path", "savefig.facecolor": "white",
    })


def _grad(ax, extent, cmap, horizontal, rounding=0.0, zorder=2):
    """Draw a gradient rectangle clipped to an optional rounded box."""
    grad = np.linspace(0.12, 1.0, 256)
    grad = grad.reshape(1, -1) if horizontal else grad.reshape(-1, 1)
    im = ax.imshow(grad, extent=extent, aspect="auto", cmap=cmap,
                   zorder=zorder, origin="lower")
    if rounding > 0:
        x0, x1, y0, y1 = extent
        box = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                             boxstyle=f"round,pad=0,rounding_size={rounding}",
                             transform=ax.transData, zorder=zorder)
        im.set_clip_path(box)
    return im


def vbar(ax, x, height, width, cmap, bottom=0.0, rounding=0.10):
    """Vertical gradient bar (gradient runs bottom -> top)."""
    return _grad(ax, (x, x + width, bottom, bottom + height), cmap,
                 horizontal=False, rounding=rounding)


def hbar(ax, y, value, height, cmap, x0=0.0, rounding=0.12):
    """Horizontal gradient bar (gradient runs left -> right)."""
    return _grad(ax, (x0, value, y, y + height), cmap,
                 horizontal=True, rounding=rounding)


def save(fig, out_dir, name, dpi=300):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
