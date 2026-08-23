"""
Shared visual system for the infant-gut main figure.

Palette semantics (colour-blind safe, Okabe-Ito base; every colour is paired with
a non-colour redundant channel elsewhere -- arrow direction / open-vs-filled
marker / check-cross glyph -- so the figure survives greyscale printing):

  base_raw   -> muted grey        ("sparse / noisy" observed pipeline)
  imputed    -> deep slate        ("dense / denoised" MOT-(2D)^2 PCA pipeline)
  LinDA      -> orange  #E69F00   (the *liberal* engine)
  MaAsLin2   -> blue    #0072B2   (the *conservative* engine)
  enriched   -> vermillion #D55E00 (direction UP)
  depleted   -> blue    #0072B2   (direction DOWN)   [only used in d/e]
  true bio   -> green   #009E73  + check
  false/contam-> vermillion #D55E00 + cross
  null       -> grey            + circle

Method colours (orange/blue) live only in panels b,c; direction/verdict colours
live only in panels d,e, so there is no within-panel clash.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- colours -------------------------------------------------------------
C = dict(
    base="#9AA0A6",        # observed / sparse
    imputed="#33415C",     # imputed / dense
    linda="#E69F00",       # liberal engine
    maaslin="#0072B2",     # conservative engine
    up="#D55E00",          # enriched
    down="#2C7FB8",        # depleted
    true="#009E73",        # validated biology
    false="#D55E00",       # contamination / false positive
    null="#B8BDC4",        # null / grey
    ink="#1A1A1A",         # text
    faint="#E9ECEF",       # panel wash
    filter="#6A4C93",      # the "persistence filter" motif (purple)
)

VERDICT = {  # glyph + colour for the scorecard / lanes
    "recover": ("↑", C["true"]),      # up arrow, green
    "confirm": ("✓", C["true"]),      # check, green
    "suppress": ("✗", C["false"]),    # cross, vermillion
    "null": ("○", C["null"]),         # open circle, grey
}


def apply_rc():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "svg.fonttype": "none",          # keep text as text in the SVG (editable)
        "pdf.fonttype": 42,              # editable text in PDF
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#5A5F66",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
        "lines.linewidth": 1.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def panel_letter(ax, letter, dx=-0.02, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="right", color=C["ink"])


def message_title(ax, text, y=1.02):
    """Message-first panel title (states the conclusion, not the content)."""
    ax.set_title(text, loc="left", y=y, color=C["ink"])


def clean(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
