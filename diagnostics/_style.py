"""
Shared plotting style and colorblind-safe palette for the diagnostics tools.
============================================================================
One place to define how every diagnostic figure looks, so that a given model
is always the SAME colour across every tool (previously IFS was blue in one
script and red in another, and AIFS was orange / dark-red / blue depending on
the figure — actively misleading when comparing plots).

Palette: Okabe & Ito (2008), the canonical colourblind-safe qualitative set.
Model-pair separation was checked under deuteranopia / protanopia / tritanopia
(all pairwise ΔE well above the ΔE≥8 CVD floor and the 15 normal-vision floor).

Usage
-----
    import _style
    _style.apply_style()                 # set rcParams once, near the top of main()
    ax.plot(..., color=_style.C_FC1)     # model1 (reference)
    ax.plot(..., color=_style.C_FC2)     # model2
"""
from __future__ import annotations

import matplotlib as mpl

# ── Model / observation identity colours (Okabe-Ito) ─────────────────────────
C_FC1 = "#0072B2"   # model1 / reference (e.g. IFS)   — blue
C_FC2 = "#D55E00"   # model2 (e.g. AIFS)              — vermillion
C_OBS = "#009E73"   # observations                    — bluish green

MODEL_COLORS = (C_FC1, C_FC2)

# ── Contingency outcome colours (kept distinct from the two model colours) ────
C_HIT = "#009E73"   # hits         — green
C_MISS = "#E69F00"  # misses       — amber
C_FA = "#CC79A7"    # false alarms — reddish purple
OUTCOME_COLORS = {"hits": C_HIT, "misses": C_MISS, "false_alarms": C_FA}

# ── Neutral reference elements ───────────────────────────────────────────────
C_THRESHOLD = "#333333"   # threshold line
C_REF = "#777777"         # perfect / 1:1 / zero reference line
C_GOOD_BAND = "#009E73"   # tolerance band fill (low alpha)

# ── Colormaps (magnitude / polarity), consistent across tools ────────────────
CMAP_DIVERGING = "RdBu_r"   # bias, differences (two hues + neutral midpoint)
CMAP_SEQUENTIAL = "viridis"  # spread, magnitude (single-hue, perceptually uniform)
CMAP_COUNT = "YlOrRd"        # counts / frequencies / probabilities

# Extended CVD-safe cycle for the rare figure with >3 series.
_PROP_CYCLE = [C_FC1, C_FC2, C_OBS, C_MISS, C_FA, "#56B4E9", "#F0E442", "#000000"]


def model_color(i: int) -> str:
    """Colour for model index 0 (fc1) or 1 (fc2)."""
    return MODEL_COLORS[i]


def winner_color(model2_wins: bool) -> str:
    """Colour of the winning model (True → fc2, False → fc1)."""
    return C_FC2 if model2_wins else C_FC1


def significance_marker(is_significant) -> str:
    """Compact significance label from a bootstrap `*_is_significant` flag.

    Returns '' when the flag is missing/NaN so callers can append unconditionally.
    """
    if is_significant is None:
        return ""
    try:
        import math
        if isinstance(is_significant, float) and math.isnan(is_significant):
            return ""
    except Exception:
        pass
    if isinstance(is_significant, str):
        is_significant = is_significant.strip().lower() in ("true", "1", "yes")
    return "✓ sig." if bool(is_significant) else "n.s."


def apply_style(save_dpi: int = 150) -> None:
    """Set global matplotlib rcParams for a consistent look across all tools.

    Explicit per-artist colours still win; this mainly standardises fonts,
    gridlines, backgrounds, the default colour cycle and the save DPI so the
    figures from different scripts sit together as one product family.
    """
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": save_dpi,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#888888",
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.alpha": 0.4,
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "axes.prop_cycle": mpl.cycler(color=_PROP_CYCLE),
    })
