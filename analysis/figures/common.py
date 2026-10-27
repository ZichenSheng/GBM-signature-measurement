"""Shared paths, palette and helpers for the Figure 1 / Figure 5 implementation (2026-10-26).

All coordinates are PDF points from the top-left of the page or of a panel canvas.
"""
import os
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DESK = Path(os.environ["GBM_VALIDATION_ROOT"])
REV = DESK / "GBM_NPJSBA_REVISION"
IMPL = Path(os.environ.get("GBM_FIGURE_OUT", "outputs/final_figures")).resolve()
for _dir in ["panels", "validation", "work", "figures"]:
    (IMPL / _dir).mkdir(parents=True, exist_ok=True)
DESIGN = REV / "05_MAIN_FIGURE_REDESIGN"
SDC = Path(__file__).resolve().parents[2] / "results_reference/final"
SSD = Path(os.environ["GBM_SOURCE_ROOT"])
F5SRC = SSD / "Figure5_ACT5_SPATIAL_REBUILT/source_tables"
IVY = SSD / "IVYGAP_PATHOLOGY_ANATOMY_EXTENSION_V1"
R4 = SSD / "GBM_R4_STAGEB1_DS03_CALIBRATED_DETECTABILITY_TRANSPORT_2026-09-08"
R4CLOSE = SSD / "GBM_R4_POST_STAGEB1_STRATEGIC_CLOSURE_AUDIT_2026-09-09"

FIG1_SRC = DESK / "文稿/figure/Figure1.pdf"
FIG5_SRC = REV / "02_DELIVERABLES/Figures_revised/Figure5_NPJSBA_revised.pdf"

ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIAL_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
ARIAL_I = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"

# Figure 1 palette (figure1_production.py l. 62-101)
INK = "#152033"; SUBTLE = "#5E6878"; HAIR = "#D7DCE4"; GRID = "#E9ECF1"; NULL = "#C9CED6"
COHORT_1 = "#2A6FBB"; COHORT_2 = "#D85D57"
PURPLE = "#7651AE"; PURPLE_DARK = "#4D3278"; PURPLE_LIGHT = "#D9CEE9"
LETTER1 = "#111111"
# Figure 5 palette (Figure5_ACT5_FINAL_upgraded.py l. 15, 45)
REP = {"PF4C-B4-007": ("Liu EMT-5", "Liu", "#1458DB"),
       "PF4C-B4-029": ("Xiao EMT-4", "Xiao", "#D10A88"),
       "PF4C-FINAL-079": ("Arimappamagan-14", "Arim.", "#F26722")}
INK5 = "#231815"; TEXT5 = "#202027"; SEC5 = "#596575"; RULE5 = "#D4D5D8"; REF5 = "#9BA3AD"
STATES = ["LE", "IT", "CT", "PAN", "MVP"]


def style(base=6.6):
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": base, "axes.labelsize": base, "xtick.labelsize": 5.8, "ytick.labelsize": 5.8,
        "axes.linewidth": 0.55, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2,
        "ytick.major.size": 2.2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white", "axes.unicode_minus": True, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK})


class Canvas:
    """A matplotlib figure sized in points, addressed from the top-left in points."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / 72, h / 72))

    def ax(self, x0, y0, x1, y1, **kw):
        return self.fig.add_axes([x0 / self.w, 1 - y1 / self.h, (x1 - x0) / self.w, (y1 - y0) / self.h], **kw)

    def text(self, x, y, s, size=6.6, weight="normal", color=INK, ha="left", va="baseline", style="normal", **kw):
        return self.fig.text(x / self.w, 1 - y / self.h, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
                             fontstyle=style, **kw)

    def line(self, x0, y0, x1, y1, color=HAIR, lw=0.5, **kw):
        self.fig.add_artist(mpl.lines.Line2D([x0 / self.w, x1 / self.w], [1 - y0 / self.h, 1 - y1 / self.h],
                                             transform=self.fig.transFigure, color=color, lw=lw, **kw))

    def save(self, stem):
        out = IMPL / "panels"
        self.fig.savefig(out / f"{stem}.pdf", dpi=600)
        self.fig.savefig(out / f"{stem}.svg", dpi=600)
        plt.close(self.fig)
        return out / f"{stem}.pdf"


def clean(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left); ax.spines["bottom"].set_visible(bottom)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def log_json(name, obj):
    p = IMPL / "validation" / name
    p.write_text(json.dumps(obj, indent=1, default=str, ensure_ascii=False) + "\n")
    return p


def golden_jitter(n, half):
    """Deterministic, evenly spread offsets in [-half, half] (golden-ratio sequence; no random seed)."""
    import numpy as np
    return ((np.arange(n) * 0.6180339887498949) % 1.0 * 2 - 1) * half
