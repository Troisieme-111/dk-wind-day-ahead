"""Shared settings: paths, price areas, periods, plot style."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# DKW_OUT redirects all data/outputs (used by tests/make_fake_raw.py)
OUT = Path(os.environ.get("DKW_OUT", ROOT))
RAW = OUT / "data" / "raw"
PROCESSED = OUT / "data" / "processed"
FIGURES = OUT / "figures"
RESULTS = OUT / "results"
for _p in (RAW, PROCESSED, FIGURES, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

HOURLY_FILE = PROCESSED / "dk_hourly.csv.gz"

AREAS = ["DK1", "DK2"]
# Neighbouring price areas used as the exogenous "outside market" in the model.
# "SE" covers Sweden before its split into four areas (Nov 2011); "SYSTEM" is the
# Nordic system price, used only for hours where none of the others exists.
NEIGHBOURS = {"DK1": ["NO2", "SE3", "SE", "DE"], "DK2": ["SE4", "SE", "DE"]}

FIRST_YEAR = 2000          # Elspotprices starts 1 July 1999
LAST_YEAR = 2025
CALIBRATION_YEARS = range(2015, 2020)   # model fitted on these years
VALIDATION_YEARS = range(2020, 2026)    # and tested on these

EUR_PER_DKK = 1 / 7.45     # fallback when only DKK prices exist

# Plot style (validated categorical palette, light mode)
COLORS = {"DK1": "#2a78d6", "DK2": "#eb6834", "model": "#1baf7a",
          "grey": "#8a8984", "text": "#0b0b0b", "muted": "#52514e"}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": COLORS["grey"], "axes.labelcolor": COLORS["text"],
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
        "lines.linewidth": 2, "lines.markersize": 5, "font.size": 10,
        "legend.frameon": False,
    })
    return plt
