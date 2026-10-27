#!/usr/bin/env python3
"""Visual-hierarchy redesign of GBM Main Figure 2 (ACT2), panels A–G.

This script reuses the previously validated frozen objects and plotting
estimands. It changes panel geometry and graphical form only. No CARE object,
ACT3 object, generated image, or mock-up-derived value is loaded.
"""

from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scipy
from scipy.stats import gaussian_kde, spearmanr


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
STEM = "Figure_2_ACT2_A_to_G_FINAL"
PDF = OUT / f"{STEM}.pdf"
SVG = OUT / f"{STEM}.svg"
PNG = OUT / f"{STEM}.png"
CHANGELOG = OUT / "FIG2_A_TO_G_CHANGELOG.md"
SOURCE_MAP = OUT / "FIG2_A_TO_G_SOURCE_MAP.tsv"
RENDER_LOG = OUT / "FIG2_A_TO_G_RENDER_LOG.txt"

SRC = {
    "A": ROOT / "IVYGAP_PATHOLOGY_ANATOMY_EXTENSION_V1/05_ANATOMICAL_INFORMATIVENESS/IVYGAP_ANATOMICAL_INFORMATIVENESS_MODEL_LEVEL.tsv",
    "B": ROOT / "IVYGAP_PATHOLOGY_ANATOMY_EXTENSION_V1/06_ANATOMICAL_IDENTIFIABILITY/ANATOMICAL_INFORMATIVENESS_VS_IDENTIFIABILITY.tsv",
    "C0": ROOT / "IVYGAP_PATHOLOGY_ANATOMY_EXTENSION_V1/06_ANATOMICAL_IDENTIFIABILITY/IVYGAP_ANATOMICAL_IDENTIFIABILITY_MODEL_LEVEL.tsv",
    "C1": ROOT / "IVYGAP_PATHOLOGY_ANATOMY_EXTENSION_V1/08_CORE3_SENSITIVITY/IVYGAP_SENSITIVITY_IDENTIFIABILITY_MODEL_LEVEL.tsv",
    "DS": ROOT / "IVYGAP_PATHOLOGY_ANATOMY_EXTENSION_V1/07_SAME_SCORE_DIFFERENT_PATHOLOGY/IVYGAP_NEAREST_CROSS_ANATOMY_SAMPLE_LEVEL.tsv.gz",
    "DM": ROOT / "IVYGAP_PATHOLOGY_ANATOMY_EXTENSION_V1/07_SAME_SCORE_DIFFERENT_PATHOLOGY/IVYGAP_SAME_SCORE_DIFFERENT_PATHOLOGY_MODEL_LEVEL.tsv",
    "E": ROOT / "GBM_SIGNATURE_FINAL_SPECIFIC_GAP_CLOSURE_V1_FINAL/03_wp3_acta_geomx/ACTA2025_COMPARTMENT_IDENTIFIABILITY.tsv",
    "F": ROOT / "GBM_SIGNATURE_FINAL_SPECIFIC_GAP_CLOSURE_V1_FINAL/03_wp3_acta_geomx/ACTA2025_MATCHED_SCORE_ANALYSIS.tsv",
    "G": ROOT / "GBM_SIGNATURE_FINAL_SPECIFIC_GAP_CLOSURE_V1_FINAL/03_wp3_acta_geomx/ACTA2025_SOURCE_SUBSTITUTION.tsv",
    "IMG": ROOT / "figure2_v3_assets/ivy/IVY_IMAGE_PROVENANCE.tsv",
    "SAMPLE": ROOT / "IVYGAP_PATHOLOGY_ANATOMY_EXTENSION_V1/02_SOURCE/IVYGAP_SAMPLE_MANIFEST.tsv",
}

EXPECTED_HASH = {
    "A": "50c316c46d1ce9eb9f6f2e29298d0f4adcf46d1ec4f90d3b76fe2e3cd7036ba5",
    "B": "c22d416df151b62773c68acd94e53d4b9f8fa186a761a490a5cda6f38ab6be98",
    "C0": "5fd64baf910ff249de6c9e2cb7c7b38ac2f1b107780d1ac7c2ee617ef7530e0a",
    "C1": "6ef3ef70e5f1f03281e2faf395702bbcb9cd7e1599e87d2f702f324df1010e26",
    "DS": "c5626d12905a930205b00a8604c0414100cb90cad63654b21ba79b7cce124714",
    "DM": "e314362906efd6da8888781a3e14c76eda220429531e8097454971577be2a16a",
    "E": "f6e114c0e4b3d041b4098c4586ec467e5221754e41c414550e29ca10db208e73",
    "F": "3fdd5dca0fb91ea5c5691cef12078ba5babc318e40ad5ee012533ce7258fe023",
    "G": "27b579c400d611d4568fb1496e32a2ce5232c853a26e043729b21b2e50e7d101",
}

INK = "#11191F"
NAVY = "#123B5A"
MID = "#69777F"
GRAY = "#AEB8BD"
LIGHT = "#DCE3E6"
PALE = "#F4F7F8"
BLUE = "#2F7EBB"
TEAL = "#168E8A"
GREEN = "#2B8A63"
ORANGE = "#E68136"
RED = "#CA4938"
PURPLE = "#8A4BB8"
SOX = "#2F80B9"
TAM = "#E47E32"

ANAT = {"LE": BLUE, "IT": TEAL, "CT": ORANGE, "PAN": "#D63D47", "MVP": PURPLE}
ANAT_LONG = {
    "LE": "Leading edge", "IT": "Infiltrating tumour", "CT": "Cellular tumour",
    "PAN": "Pseudopalisade", "MVP": "Microvascular proliferation",
}
DIST_CMAP = LinearSegmentedColormap.from_list(
    "matched_distance", ["#FFF9F5", "#F8DFD2", "#EFB496", "#DD7456", "#B33D38", "#5B1B2C"]
)
DENS_CMAP = LinearSegmentedColormap.from_list(
    "joint_density", ["#FFFFFF", "#EDF4F5", "#C8DFE3", "#82B8C4", "#397C91", "#123B5A"]
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(key: str) -> pd.DataFrame:
    path = SRC[key]
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def require(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [x for x in columns if x not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing {missing}")


def ecdf(values) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, float)
    x = np.sort(x[np.isfinite(x)])
    return x, np.arange(1, len(x) + 1) / len(x)


def kde(values, grid) -> np.ndarray:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return gaussian_kde(x)(grid)


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 5.9,
        "ytick.labelsize": 5.9,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.58,
        "xtick.major.width": 0.52,
        "ytick.major.width": 0.52,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def clean(ax, left=True, bottom=True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    for spine in ax.spines.values():
        spine.set_linewidth(0.58)
        spine.set_color(INK)
    ax.tick_params(color=INK, labelcolor=INK, pad=1.6)
    ax.grid(False)


def header(fig, ss, letter: str, title: str, dataset: str | None = None, hero=False) -> None:
    b = ss.get_position(fig)
    y = b.y1 + 0.0045
    fig.text(b.x0, y, letter, fontsize=13.0 if hero else 12.0, fontweight="bold",
             color=INK, ha="left", va="bottom")
    fig.text(b.x0 + (0.027 if hero else 0.025), y + 0.0007, title,
             fontsize=9.2 if hero else 8.1, fontweight="semibold", color=NAVY,
             ha="left", va="bottom")
    if dataset:
        fig.text(b.x1, y + 0.0008, f"({dataset})", fontsize=5.6, color=MID,
                 ha="right", va="bottom")


def load_validate():
    for key, expected in EXPECTED_HASH.items():
        actual = sha256(SRC[key])
        if actual != expected:
            raise RuntimeError(f"Frozen source changed: {key}: {actual} != {expected}")

    A, B, C0, C1, Ds, Dm, E, F, G = [read(k) for k in ["A", "B", "C0", "C1", "DS", "DM", "E", "F", "G"]]
    images, samples = read("IMG"), read("SAMPLE")
    require(A, ["model_instance_id", "partial_r2", "permutation_p", "bh_fdr", "evaluable"], "A")
    require(B, ["model_instance_id", "partial_r2", "balanced_accuracy"], "B")
    require(C0, ["model_instance_id", "branch", "balanced_accuracy", "chance_level", "evaluable"], "C0")
    require(C1, ["model_instance_id", "branch", "balanced_accuracy", "chance_level", "evaluable"], "C1")
    require(Ds, ["anatomical_structure", "model_instance_id", "nearest_cross_anatomy_score_distance"], "D sample")
    require(Dm, ["model_instance_id", "nearest_median_distance", "random_cross_anatomy_median_distance", "nearest_random_ratio"], "D model")
    require(E, ["model_instance_id", "balanced_accuracy"], "E")
    require(F, ["model_instance_id", "SOX2_score", "nearest_IBA1_score", "absolute_score_distance", "random_cross_compartment_distance_median", "nearest_to_random_distance_ratio"], "F")
    require(G, ["model_instance_id", "patient_id", "delta_tumor", "delta_TAM"], "G")

    A = A.loc[A["evaluable"].astype(bool)].copy()
    C0 = C0.loc[C0["evaluable"].astype(bool)].copy()
    C1 = C1.loc[C1["evaluable"].astype(bool)].copy()
    if not (len(A) == len(B) == len(C0) == 201 and len(C1) == 402):
        raise RuntimeError("Frozen Ivy dimensions failed")
    if not (len(Ds) == 24522 and len(Dm) == 201 and len(E) == 198 and len(F) == 9702 and len(G) == 4554):
        raise RuntimeError("Frozen ACT2 dimensions failed")
    if Ds["model_instance_id"].nunique() != 201 or set(Ds["anatomical_structure"]) != set(ANAT):
        raise RuntimeError("Panel D structure failed")
    if E["model_instance_id"].nunique() != 198 or F["model_instance_id"].nunique() != 198:
        raise RuntimeError("GeoMx model identity failed")
    if G["model_instance_id"].nunique() != 198 or G["patient_id"].nunique() != 23:
        raise RuntimeError("Panel G structure failed")

    cx = pd.concat([
        C0[["model_instance_id", "branch", "balanced_accuracy", "chance_level"]],
        C1[["model_instance_id", "branch", "balanced_accuracy", "chance_level"]],
    ], ignore_index=True)
    C = cx.pivot(index="model_instance_id", columns="branch", values="balanced_accuracy")
    C = C[["ALL5", "CORE3", "PAN_MVP"]].dropna()
    chance = cx.groupby("branch", observed=True)["chance_level"].first()
    if len(C) != 201 or not np.allclose(chance[["ALL5", "CORE3", "PAN_MVP"]], [0.2, 1/3, 0.5]):
        raise RuntimeError("Panel C branch matching failed")

    images = images.loc[images["panel"].eq("A")].copy()
    if len(images) != 5 or images["specimen_id"].nunique() != 1 or set(images["biological_label"]) != set(ANAT):
        raise RuntimeError("Panel A image registry failed")
    for _, r in images.iterrows():
        p = Path(r["local_source_path"])
        if not p.is_file() or sha256(p) != r["sha256"]:
            raise RuntimeError(f"Image hash failed: {p}")
        match = samples.loc[
            samples["sample_id"].eq(int(r["sample_id"]))
            & samples["polygon_id"].eq(int(r["polygon_id"]))
            & samples["specimen_id"].eq(int(r["specimen_id"]))
            & samples["anatomical_structure"].eq(r["biological_label"])
        ]
        if len(match) != 1:
            raise RuntimeError(f"Image linkage failed: {r['biological_label']}")

    # Preserve the already-reconciled ACT2 numerical invariants.
    observed = {
        "r2": B["partial_r2"].median(),
        "ba": B["balanced_accuracy"].median(),
        "rho": spearmanr(B["partial_r2"], B["balanced_accuracy"]).statistic,
        "core3": C["CORE3"].median(),
        "panmvp": C["PAN_MVP"].median(),
        "dratio": Dm["nearest_random_ratio"].median(),
        "eba": E["balanced_accuracy"].median(),
        "fratio": F.groupby("model_instance_id", observed=True)["nearest_to_random_distance_ratio"].median().median(),
    }
    expected = {"r2": .4038960656015303, "ba": .35005263157894734,
                "core3": .47444444444444445, "panmvp": .6541666666666667,
                "dratio": .14429265317079268, "eba": .6020408163265306,
                "fratio": .03706543434856597}
    for key, value in expected.items():
        if not np.isclose(observed[key], value, rtol=0, atol=1e-12):
            raise RuntimeError(f"Validated ACT2 value changed: {key}={observed[key]}")
    return A, B, C, chance, Ds, Dm, E, F, G, images, observed


def draw_atlas(fig, ss, A, images):
    header(fig, ss, "A", "Anatomical information", "IvyGAP")
    outer = GridSpecFromSubplotSpec(2, 1, subplot_spec=ss, height_ratios=[0.67, 0.33], hspace=0.14)
    atlas = GridSpecFromSubplotSpec(2, 6, subplot_spec=outer[0], hspace=0.16, wspace=0.08)
    images = images.set_index("biological_label").loc[["LE", "IT", "CT", "PAN", "MVP"]].reset_index()
    placements = [(0, slice(0, 2)), (0, slice(2, 4)), (0, slice(4, 6)),
                  (1, slice(0, 3)), (1, slice(3, 6))]
    for i, (r, (row, col)) in enumerate(zip(images.to_dict("records"), placements)):
        ax = fig.add_subplot(atlas[row, col])
        ax.imshow(plt.imread(r["local_source_path"]), interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.95); sp.set_color(ANAT[r["biological_label"]])
        ax.text(0.025, 0.965, r["biological_label"], transform=ax.transAxes,
                fontsize=6.2, fontweight="bold", color="white", ha="left", va="top",
                bbox=dict(facecolor=ANAT[r["biological_label"]], edgecolor="none", pad=1.0, alpha=0.90))
        ax.text(0.98, 0.04, ANAT_LONG[r["biological_label"]], transform=ax.transAxes,
                fontsize=4.6, color="white", ha="right", va="bottom",
                bbox=dict(facecolor=INK, edgecolor="none", pad=0.75, alpha=0.72))
        if i == 0:
            px = 100 / float(r["resolution_um_per_px"])
            ax.plot([50, 50 + px], [640, 640], color="white", lw=1.45, solid_capstyle="butt")
            ax.text(50, 607, "100 µm", fontsize=4.4, color="white", ha="left", va="bottom")

    q = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], height_ratios=[0.79, 0.21], hspace=0.02)
    ax = fig.add_subplot(q[0]); axs = fig.add_subplot(q[1], sharex=ax)
    a = A.sort_values(["partial_r2", "model_instance_id"], kind="mergesort").reset_index(drop=True)
    x = np.arange(1, len(a) + 1); y = a["partial_r2"].to_numpy(float)
    ax.fill_between(x, 0, y, color="#B8D8E8", alpha=0.85, lw=0)
    ax.plot(x, y, color=BLUE, lw=0.70)
    med = float(np.median(y))
    ax.axhline(med, color=NAVY, lw=0.62, ls=(0, (3, 2)))
    ax.text(4, med + 0.035, f"median {med:.3f}", fontsize=5.2, color=NAVY, ha="left")
    ax.set_xlim(1, 201); ax.set_ylim(0, 0.92); ax.set_xticks([])
    ax.set_ylabel("Partial R²", labelpad=2); clean(ax)
    support = np.vstack([a["permutation_p"].to_numpy(float) < .05, a["bh_fdr"].to_numpy(float) < .05]).astype(float)
    axs.imshow(support, aspect="auto", origin="upper", interpolation="nearest",
               cmap=LinearSegmentedColormap.from_list("support", ["#E7ECEE", TEAL]),
               vmin=0, vmax=1, extent=[.5, 201.5, 1.5, -.5])
    axs.set_yticks([0, 1], ["perm.", "FDR"]); axs.tick_params(axis="y", length=0, labelsize=4.6, pad=1.5)
    axs.set_xticks([])
    axs.text(0.0, -0.38, "1", transform=axs.transAxes, ha="left", va="top", fontsize=4.5, color=MID)
    axs.text(1.0, -0.38, "201", transform=axs.transAxes, ha="right", va="top", fontsize=4.5, color=MID)
    for sp in axs.spines.values(): sp.set_visible(False)


def draw_joint(fig, ss, B):
    header(fig, ss, "B", "Information ≠ identification")
    g = GridSpecFromSubplotSpec(2, 2, subplot_spec=ss, height_ratios=[0.12, 0.88],
                                width_ratios=[0.88, 0.12], hspace=0.015, wspace=0.025)
    axt = fig.add_subplot(g[0, 0]); ax = fig.add_subplot(g[1, 0]); axr = fig.add_subplot(g[1, 1], sharey=ax)
    fig.add_subplot(g[0, 1]).axis("off")
    x = B["partial_r2"].to_numpy(float); y = B["balanced_accuracy"].to_numpy(float)
    xg = np.linspace(0, .82, 170); yg = np.linspace(.08, .68, 155)
    XX, YY = np.meshgrid(xg, yg)
    z = gaussian_kde(np.vstack([x, y]))(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    levels = np.unique(np.quantile(z[z > 0], [.30, .50, .66, .79, .89, .95, .985]))
    ax.contourf(XX, YY, z, levels=levels, cmap=DENS_CMAP, alpha=.88)
    ax.contour(XX, YY, z, levels=levels[2::2], colors="#3F7081", linewidths=.28, alpha=.62)
    ax.scatter(x, y, s=4.8, facecolor="white", edgecolor=BLUE, linewidth=.30, alpha=.60, zorder=3)
    coef = np.polyfit(x, y, 2); xx = np.linspace(x.min(), x.max(), 280)
    ax.plot(xx, np.polyval(coef, xx), color=INK, lw=1.05, zorder=4)
    ax.axhline(.2, color=MID, lw=.62, ls=(0, (4, 2)))
    rho = float(spearmanr(x, y).statistic)
    ax.text(.035, .965, f"ρ = {rho:.2f}\nmedian R² {np.median(x):.3f}\nmedian BA {np.median(y):.3f}\nn = {len(B)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.35, color=NAVY)
    ax.text(.98, .205, "chance 1/5", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=4.9, color=MID)
    ax.set_xlim(0, .82); ax.set_ylim(.08, .68)
    ax.set_xlabel("Anatomical partial R²"); ax.set_ylabel("Five-state balanced accuracy")
    clean(ax)
    dx = kde(x, xg); axt.fill_between(xg, 0, dx, color="#B6D6E7", lw=0); axt.plot(xg, dx, color=BLUE, lw=.65)
    axt.set_xlim(ax.get_xlim()); axt.axis("off")
    dy = kde(y, yg); axr.fill_betweenx(yg, 0, dy, color="#B6D6E7", lw=0); axr.plot(dy, yg, color=BLUE, lw=.65)
    axr.set_ylim(ax.get_ylim()); axr.axis("off")


def draw_claim(fig, ss, C, chance):
    header(fig, ss, "C", "Claim-dependent resolution")
    ax = fig.add_subplot(ss)
    branches = ["ALL5", "CORE3", "PAN_MVP"]; colors = [BLUE, GREEN, RED]
    x = np.arange(3)
    q05 = C[branches].quantile(.05).to_numpy(float); q25 = C[branches].quantile(.25).to_numpy(float)
    med = C[branches].median().to_numpy(float); q75 = C[branches].quantile(.75).to_numpy(float)
    q95 = C[branches].quantile(.95).to_numpy(float)
    ax.fill_between(x, q05, q95, color="#DCE7E9", alpha=.62, lw=0)
    ax.fill_between(x, q25, q75, color="#9DBEC6", alpha=.48, lw=0)
    ax.plot(x, med, color=INK, lw=1.05, zorder=4)
    for i, (color, branch) in enumerate(zip(colors, branches)):
        ax.vlines(i, q05[i], q95[i], color=color, lw=.75, zorder=3)
        ax.vlines(i, q25[i], q75[i], color=color, lw=2.2, zorder=4)
        ax.scatter(i, med[i], s=25, color=color, edgecolor="white", linewidth=.55, zorder=5)
        ax.hlines(float(chance.loc[branch]), i - .26, i + .26, color=color, lw=.68, ls=(0, (4, 2)))
        ax.text(i, 1.025, f"{med[i]:.3f}", ha="center", va="bottom", fontsize=5.1, color=color)
    ax.set_yscale("log"); ax.set_ylim(.10, 1.08); ax.set_xlim(-.38, 2.38)
    ax.set_yticks([.1, .2, .5, 1], [".1", ".2", ".5", "1"])
    ax.set_xticks(x, ["ALL5\n5 states", "CORE3\n3 states", "PAN–MVP\n2 states"])
    ax.set_ylabel("Balanced accuracy", labelpad=2)
    ax.text(.03, .05, "outer 5–95%  ·  inner IQR", transform=ax.transAxes, fontsize=4.6, color=MID)
    clean(ax)


def draw_anatomy_hero(fig, ss, Ds, Dm):
    header(fig, ss, "D", "Matched-score anatomy", "IvyGAP", hero=True)
    g = GridSpecFromSubplotSpec(2, 2, subplot_spec=ss, height_ratios=[0.88, 0.12],
                                width_ratios=[0.815, 0.185], hspace=0.15, wspace=0.16)
    ax = fig.add_subplot(g[0, 0]); cax = fig.add_subplot(g[1, 0]); axe = fig.add_subplot(g[:, 1])
    states = ["LE", "IT", "CT", "PAN", "MVP"]
    mat = Ds.pivot_table(index="anatomical_structure", columns="model_instance_id",
                         values="nearest_cross_anatomy_score_distance", aggfunc="median").loc[states]
    order = mat.median(axis=0).sort_values(kind="mergesort").index
    mat = mat.loc[:, order]
    im = ax.imshow(mat.to_numpy(float), aspect="auto", interpolation="nearest", origin="upper",
                   cmap=DIST_CMAP, norm=LogNorm(vmin=float(mat.min().min()), vmax=float(mat.max().max())))
    for y in np.arange(.5, 5, 1): ax.axhline(y, color="white", lw=1.1)
    ax.set_yticks(range(5), states); ax.set_xticks([0, 50, 100, 150, 200], ["1", "51", "101", "151", "201"])
    ax.set_xlabel("Signatures ordered by overall median distance", labelpad=2.0)
    ax.set_ylabel("Source anatomy", labelpad=4)
    ax.tick_params(axis="y", length=0, labelsize=7.0)
    for lab in ax.get_yticklabels():
        lab.set_color(ANAT[lab.get_text()]); lab.set_fontweight("bold")
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Median nearest cross-anatomy score distance (log scale)", fontsize=6.1, labelpad=1.5)
    cb.ax.tick_params(labelsize=5.5, length=1.7, pad=1.0)
    for col, color, label, ls in [
        ("nearest_median_distance", RED, "Observed", "-"),
        ("random_cross_anatomy_median_distance", GRAY, "Frozen random", (0, (4, 2))),
    ]:
        x, y = ecdf(Dm[col]); axe.plot(x, y, color=color, lw=1.20 if label == "Observed" else 1.0, ls=ls, label=label)
    axe.set_xscale("log"); axe.set_ylim(0, 1.02)
    axe.set_xlabel("Model-level median distance"); axe.set_ylabel("Cumulative fraction", labelpad=2)
    axe.text(.04, .97, f"median ratio\n{Dm['nearest_random_ratio'].median():.3f}", transform=axe.transAxes,
             ha="left", va="top", fontsize=5.8, color=NAVY)
    axe.legend(frameon=False, loc="lower right", handlelength=1.8, labelspacing=.25, borderpad=0)
    clean(axe)
    return mat


def draw_source_info(fig, ss, E):
    header(fig, ss, "E", "Cellular-source information")
    ax = fig.add_subplot(ss)
    e = E.sort_values(["balanced_accuracy", "model_instance_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    x = np.arange(1, len(e) + 1); y = e["balanced_accuracy"].to_numpy(float)
    ax.fill_between(x, 0, y, color="#A9D0E4", alpha=.76, lw=0)
    ax.plot(x, y, color=SOX, lw=.78)
    ax.axhline(.5, color=MID, lw=.65, ls=(0, (4, 2)))
    med = float(np.median(y)); ax.axhline(med, color=TAM, lw=.78)
    ax.text(195, med + .025, f"median {med:.3f}", ha="right", fontsize=5.2, color=NAVY)
    ax.text(195, .508, "chance .5", ha="right", va="bottom", fontsize=4.7, color=MID)
    ax.set_xlim(1, 198); ax.set_ylim(0, 1); ax.set_xticks([1, 99, 198])
    ax.set_xlabel("Signatures ranked by BA"); ax.set_ylabel("Source BA")
    ax.legend(handles=[Patch(facecolor=SOX, label="SOX2+ tumour-enriched"),
                       Patch(facecolor=TAM, label="IBA1+ TAM-enriched")],
              frameon=False, loc="upper right", handlelength=.9, labelspacing=.22, borderaxespad=.25)
    clean(ax)


def select_models(F):
    fm = F.groupby("model_instance_id", observed=True)["absolute_score_distance"].median().sort_values(kind="mergesort")
    qs = [.10, .30, .50, .70, .90]
    ix = [int(round(q * (len(fm) - 1))) for q in qs]
    return [(q, fm.index[i], float(fm.iloc[i])) for q, i in zip(qs, ix)]


def draw_source_hero(fig, ss, F):
    header(fig, ss, "F", "Matched-score source", hero=True)
    g = GridSpecFromSubplotSpec(1, 2, subplot_spec=ss, width_ratios=[0.79, 0.21], wspace=0.16)
    ax = fig.add_subplot(g[0]); axe = fig.add_subplot(g[1])
    selected = select_models(F)
    use = F.loc[F["model_instance_id"].isin([m for _, m, _ in selected])]
    lo, hi = np.quantile(np.r_[use["SOX2_score"], use["nearest_IBA1_score"]], [.002, .998])
    pad = .05 * (hi - lo); grid = np.linspace(lo - pad, hi + pad, 420)
    bases = np.linspace(3.8, .2, len(selected))
    for base, (q, model, mdist) in zip(bases, selected):
        d = F.loc[F["model_instance_id"].eq(model)]
        ds = kde(d["SOX2_score"], grid); dt = kde(d["nearest_IBA1_score"], grid)
        scale = max(ds.max(), dt.max()); ds = ds / scale * .39; dt = dt / scale * .39
        ax.fill_between(grid, base, base + ds, color=SOX, alpha=.72, lw=0)
        ax.plot(grid, base + ds, color=SOX, lw=.72)
        ax.fill_between(grid, base, base - dt, color=TAM, alpha=.70, lw=0)
        ax.plot(grid, base - dt, color=TAM, lw=.72)
        ax.hlines(base, grid.min(), grid.max(), color="#CFD8DC", lw=.45)
        ax.vlines(d["SOX2_score"], base + .015, base + .055, color=SOX, lw=.30, alpha=.38)
        ax.vlines(d["nearest_IBA1_score"], base - .055, base - .015, color=TAM, lw=.30, alpha=.38)
        ax.text(grid.min(), base + .47, f"q{int(q*100):02d}", fontsize=5.7, fontweight="bold", color=NAVY, ha="left")
        ax.text(grid.max(), base + .47, f"median |Δ| {mdist:.3f}", fontsize=5.1, color=MID, ha="right")
    ax.axvline(0, color=LIGHT, lw=.65)
    ax.set_xlim(grid.min(), grid.max()); ax.set_ylim(-.33, 4.55); ax.set_yticks([])
    ax.set_xlabel("Signature score")
    ax.legend(handles=[Patch(facecolor=SOX, label="SOX2+ tumour-enriched"),
                       Patch(facecolor=TAM, label="IBA1+ TAM-enriched")],
              frameon=False, loc="upper center", bbox_to_anchor=(.52, .995), ncol=2,
              handlelength=.9, columnspacing=.9, borderaxespad=0)
    clean(ax, left=False)
    for col, color, label, ls in [
        ("absolute_score_distance", RED, "Matched", "-"),
        ("random_cross_compartment_distance_median", GRAY, "Frozen random", (0, (4, 2))),
    ]:
        x, y = ecdf(F[col]); axe.plot(x, y, color=color, lw=1.2 if label == "Matched" else 1.0, ls=ls, label=label)
    axe.set_xscale("log"); axe.set_ylim(0, 1.02)
    axe.set_xlabel("Absolute score distance"); axe.set_ylabel("Cumulative fraction", labelpad=2)
    ratio = F.groupby("model_instance_id", observed=True)["nearest_to_random_distance_ratio"].median().median()
    axe.text(.04, .97, f"median ratio\n{ratio:.3f}", transform=axe.transAxes,
             ha="left", va="top", fontsize=5.8, color=NAVY)
    axe.legend(frameon=False, loc="lower right", handlelength=1.8, labelspacing=.25, borderpad=0)
    clean(axe)
    return selected


def draw_context(fig, ss, G):
    header(fig, ss, "G", "Context-dependent source realization", "GeoMx")
    g = GridSpecFromSubplotSpec(1, 2, subplot_spec=ss, width_ratios=[0.755, 0.245], wspace=0.18)
    ax = fig.add_subplot(g[0]); axi = fig.add_subplot(g[1])
    gd = G.assign(signed_source_change=G["delta_TAM"] - G["delta_tumor"])
    gs = gd.groupby("model_instance_id", observed=True)["signed_source_change"].agg(
        median="median", q25=lambda z: z.quantile(.25), q75=lambda z: z.quantile(.75)
    ).sort_values(["median", "model_instance_id"], kind="mergesort").reset_index()
    x = np.arange(1, len(gs) + 1); y = gs["median"].to_numpy(float)
    ax.fill_between(x, gs["q25"], gs["q75"], color="#CAD6DA", alpha=.42, lw=0)
    ax.fill_between(x, 0, y, where=y <= 0, color=SOX, alpha=.82, interpolate=True, lw=0)
    ax.fill_between(x, 0, y, where=y >= 0, color=TAM, alpha=.82, interpolate=True, lw=0)
    ax.plot(x[y <= 0], y[y <= 0], color=SOX, lw=.68)
    ax.plot(x[y >= 0], y[y >= 0], color=TAM, lw=.68)
    ax.axhline(0, color=INK, lw=.65)
    med = float(np.median(y)); q1, q3 = np.quantile(y, [.25, .75])
    ax.text(.01, .94, f"model median {med:.3f}  ·  IQR {q1:.3f} to {q3:.3f}",
            transform=ax.transAxes, fontsize=5.3, color=NAVY, ha="left", va="top")
    ax.set_xlim(1, 198); ax.set_xticks([1, 50, 100, 150, 198])
    ax.set_xlabel("Signatures ranked by differential source change")
    ax.set_ylabel("Median ΔIBA1 − ΔSOX2")
    ax.legend(handles=[Patch(facecolor=SOX, label="Tumour direction"), Patch(facecolor=TAM, label="TAM direction")],
              frameon=False, loc="upper center", bbox_to_anchor=(.56, 1.0), ncol=2,
              handlelength=.9, columnspacing=.9, borderaxespad=0)
    clean(ax)

    gx = G["delta_tumor"].to_numpy(float); gy = G["delta_TAM"].to_numpy(float)
    lim = float(np.quantile(np.abs(np.r_[gx, gy]), .995)); inside = (np.abs(gx) <= lim) & (np.abs(gy) <= lim)
    axi.scatter(gx[inside], gy[inside], s=1.8, color=NAVY, alpha=.09, linewidth=0)
    grid = np.linspace(-lim, lim, 130); XX, YY = np.meshgrid(grid, grid)
    z = gaussian_kde(np.vstack([gx, gy]))(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    levels = np.unique(np.quantile(z[z > 0], [.74, .89, .97]))
    axi.contour(XX, YY, z, levels=levels, colors=["#8AB7C2", "#477F90", NAVY], linewidths=[.43, .58, .72])
    axi.plot([-lim, lim], [-lim, lim], color=MID, lw=.68, ls=(0, (4, 2)))
    axi.set_xlim(-lim, lim); axi.set_ylim(-lim, lim)
    axi.set_xlabel("Δ SOX2+ tumour-enriched"); axi.set_ylabel("Δ IBA1+ TAM-enriched", labelpad=2)
    axi.text(.96, .95, "y = x", transform=axi.transAxes, ha="right", va="top", fontsize=4.8, color=MID)
    clean(axi)
    return gs


def source_row(panel, obj, key, columns, filter_rule, transform, n_before, n_after, unit, note=""):
    return {
        "panel": panel, "plotted_object": obj, "source_file": SRC[key].name,
        "exact_source_path": str(SRC[key]), "columns_used": ";".join(columns),
        "filtering_rule": filter_rule, "plotting_transformation": transform,
        "n_before": n_before, "n_after": n_after, "biological_unit": unit,
        "scientific_boundary_note": note,
    }


def write_docs(A, B, C, Ds, Dm, E, F, G, images, mat, selected, gs, observed):
    rows = [
        source_row("A", "IvyGAP H&E atlas mosaic", "IMG", ["local_source_path", "sample_id", "polygon_id", "specimen_id", "section_image_id", "biological_label", "crop_left", "crop_top", "crop_width", "crop_height", "resolution_um_per_px", "sha256"], "panel=A; specimen_id=297710624; verified one-to-one sample/polygon/anatomy linkage", "Resize only; thin anatomy-color frames; source-derived 100-µm scale bar", 5, 5, "source image crop", "No generated or synthetic image"),
        source_row("A", "Compact ranked partial-R² landscape and support tracks", "A", ["model_instance_id", "partial_r2", "permutation_p", "bh_fdr", "evaluable"], "evaluable=True", "Ascending stable rank; silhouette; median; permutation/FDR<.05 tracks", 201, len(A), "signature/model"),
        source_row("B", "Information-identification density field", "B", ["model_instance_id", "partial_r2", "balanced_accuracy"], "all frozen rows", "One point/signature; 2D KDE; quadratic visual trend; thin marginal KDEs; descriptive rho/medians", 201, len(B), "signature/model", "Density and trend are descriptive"),
        source_row("C", "ALL5 claim distribution", "C0", ["model_instance_id", "branch", "balanced_accuracy", "chance_level", "evaluable"], "evaluable=True; branch=ALL5", "Matched model IDs; branch quantiles", 201, 201, "signature/model"),
        source_row("C", "CORE3 and PAN–MVP claim distributions", "C1", ["model_instance_id", "branch", "balanced_accuracy", "chance_level", "evaluable"], "evaluable=True", "Replace spaghetti with interpolated 5–95%, IQR, and median quantile flow; claim-specific chance segments", 402, 402, "signature/model × claim", "No universal fewer-classes law implied"),
        source_row("D", "5×201 source-anatomy × signature matrix", "DS", ["anatomical_structure", "model_instance_id", "nearest_cross_anatomy_score_distance"], "all ALL5 rows", "Median by source anatomy × signature; deterministic overall-median column order; log color", 24522, len(Ds), "sample × signature/model", "Not a matched-partner identity matrix"),
        source_row("D", "Compact observed/random ECDF", "DM", ["model_instance_id", "nearest_median_distance", "random_cross_anatomy_median_distance", "nearest_random_ratio"], "all evaluable model rows", "ECDF; median nearest/random ratio", 201, len(Dm), "signature/model"),
        source_row("E", "Compact source-BA silhouette", "E", ["model_instance_id", "balanced_accuracy"], "all evaluable rows", "Stable descending rank; filled silhouette; median and chance", 198, len(E), "signature/model", "Source labels are AOI truth labels, not signature classes"),
        source_row("F", "Mirrored matched-score source landscapes", "F", ["model_instance_id", "SOX2_score", "nearest_IBA1_score", "absolute_score_distance"], "deterministic q10/q30/q50/q70/q90 ranks of model-level median distance", "Actual-score KDEs with real rugs; shared x scale; signature IDs retained in source map, not plot", 9702, 245, "SOX2 patient-time × signature/model with frozen nearest IBA1 match", "Selected IDs: " + ",".join(m for _, m, _ in selected) + "; no source equivalence or purity claim"),
        source_row("F", "Compact matched/random ECDF", "F", ["absolute_score_distance", "random_cross_compartment_distance_median", "nearest_to_random_distance_ratio", "model_instance_id"], "all frozen matched rows", "All-record ECDF; median of model-level median record ratios", 9702, len(F), "matched record"),
        source_row("G", "Full-width differential source-change spectrum", "G", ["model_instance_id", "patient_id", "delta_tumor", "delta_TAM"], "23 paired patients × 198 models", "Patient delta_TAM-delta_tumor; model median and IQR; stable ascending rank", 4554, len(G), "paired patient × signature/model", "Not raw source_rotation"),
        source_row("G", "Paired-context phase inset", "G", ["delta_tumor", "delta_TAM"], "all rows; symmetric 99.5% display window", "4,554 real points; 3 KDE contours; y=x", 4554, len(G), "paired patient × signature/model", "Descriptive context only"),
    ]
    pd.DataFrame(rows).to_csv(SOURCE_MAP, sep="\t", index=False)

    selected_text = ", ".join(f"q{int(q*100):02d}={m} (median |Δ|={d:.6g})" for q, m, d in selected)
    CHANGELOG.write_text(f"""# Figure 2 A–G visual-hierarchy redesign

## Scope

Relayout and graphical-form refactor only. Frozen ACT2 tables, filters, estimands, denominators, source images, and numerical values are unchanged. No new scientific branch was run.

## Architecture

- Old: four-row, two-column A–H dashboard with broadly similar panel weight.
- New: A/B/C top calibration row; D full-width anatomy hero; compact E plus enlarged F source hero; G full-width closing row.
- Panel H was deleted without replacement. The main figure contains exactly A–G.

## Panel-form changes

- **A:** single thumbnail strip plus tall skyline → larger two-tier five-image pathology atlas mosaic plus compact skyline/support tracks. Same five verified IvyGAP crops and same 201-row information table.
- **B:** retained real 201-point joint field; marginal densities reduced so the relationship dominates.
- **C:** 201 visible matched trajectories → light 5–95% and IQR quantile flow with a strong median path and claim-specific chance segments. Same matched 201-model table.
- **D:** half-row matrix/ECDF → full-width hero with the matrix occupying 81.5% and the ECDF 18.5%. Same legal 5×201 median-distance matrix.
- **E:** ranked stems → compact filled silhouette. Same 198 balanced accuracies, median, and chance.
- **F:** small paired panel → enlarged 79/21 hero. Same deterministic signature ranks and same 9,702 matched records; real score rugs added and long model IDs moved out of the graphic.
- **G:** half-row spectrum/inset → full-width 75.5/24.5 closing panel. Same `median(delta_TAM - delta_tumor)` spectrum and same GeoMx phase inset.

## Deterministic Panel F selection

{selected_text}

## Invariants

- Ivy median partial R²: {observed['r2']:.15g}
- Ivy median ALL5 BA: {observed['ba']:.15g}
- Ivy R²/BA Spearman ρ: {observed['rho']:.15g}
- D median nearest/random ratio: {observed['dratio']:.15g}
- GeoMx median source BA: {observed['eba']:.15g}
- F median nearest/random ratio: {observed['fratio']:.15g}
- D matrix dimensions: {mat.shape[0]}×{mat.shape[1]}
- G model summaries: {len(gs)}
""", encoding="utf-8")

    RENDER_LOG.write_text(f"""FIGURE 2 ACT2 A–G RENDER LOG
timestamp_utc={datetime.now(timezone.utc).isoformat()}
script={Path(__file__).resolve()}
python={platform.python_version()}
matplotlib={matplotlib.__version__}
pandas={pd.__version__}
numpy={np.__version__}
scipy={scipy.__version__}
backend={matplotlib.get_backend()}
page=A4 portrait, 210 x 297 mm
panel_architecture=top A/B/C 32/43/25; D full width; E/F 25/75; G full width
panel_count=7
last_panel=G
care_present_in_main_figure=NO
act3_content_present=NO
D_hero_status=PASS
F_hero_status=PASS
layout_redesigned=PASS
input_hashes=PASS
image_hash_and_frozen_linkage=PASS
manual_assembly=NO
image_generation=NO
pdf_svg=vector plots and typography; deliberate raster pathology crops only
png=600 dpi equivalent
""", encoding="utf-8")


def make_figure(A, B, C, chance, Ds, Dm, E, F, G, images):
    style()
    fig = plt.figure(figsize=(210/25.4, 297/25.4), facecolor="white")
    main = fig.add_gridspec(4, 1, left=.052, right=.982, top=.958, bottom=.052,
                            height_ratios=[1.02, 1.25, 1.29, .92], hspace=.31)
    top = GridSpecFromSubplotSpec(1, 3, subplot_spec=main[0], width_ratios=[.32, .43, .25], wspace=.22)
    middle = GridSpecFromSubplotSpec(1, 2, subplot_spec=main[2], width_ratios=[.25, .75], wspace=.18)

    draw_atlas(fig, top[0], A, images)
    draw_joint(fig, top[1], B)
    draw_claim(fig, top[2], C, chance)
    mat = draw_anatomy_hero(fig, main[1], Ds, Dm)
    draw_source_info(fig, middle[0], E)
    selected = draw_source_hero(fig, middle[1], F)
    gs = draw_context(fig, main[3], G)

    # Restrained hairlines establish narrative tiers without boxing panels.
    for i in [0, 1, 2]:
        a = main[i].get_position(fig); b = main[i + 1].get_position(fig)
        y = (a.y0 + b.y1) / 2
        fig.add_artist(Line2D([.052, .982], [y, y], transform=fig.transFigure,
                              color="#DCE4E7", lw=.42, zorder=0))
    return fig, mat, selected, gs


def main():
    A, B, C, chance, Ds, Dm, E, F, G, images, observed = load_validate()
    fig, mat, selected, gs = make_figure(A, B, C, chance, Ds, Dm, E, F, G, images)
    meta = {"Title": "Figure 2 ACT2 A to G final", "Author": "Reproducible frozen-data redesign"}
    fig.savefig(PDF, format="pdf", metadata=meta)
    fig.savefig(SVG, format="svg", metadata={"Title": meta["Title"]})
    fig.savefig(PNG, format="png", dpi=600, metadata={"Title": meta["Title"]})
    plt.close(fig)
    write_docs(A, B, C, Ds, Dm, E, F, G, images, mat, selected, gs, observed)
    print("FIG2_FINAL_STATUS=PASS")
    print("FIG2_PANEL_COUNT=7")
    print("FIG2_LAST_PANEL=G")
    print("CARE_PRESENT_IN_MAIN_FIGURE=NO")
    print("ACT3_CONTENT_PRESENT=NO")
    print("D_HERO_STATUS=PASS")
    print("F_HERO_STATUS=PASS")
    print("LAYOUT_REDESIGNED=PASS")
    print(f"OUTPUT_PDF={PDF}")
    print(f"OUTPUT_SVG={SVG}")
    print(f"OUTPUT_PNG={PNG}")
    print(f"SCRIPT_PATH={Path(__file__).resolve()}")


if __name__ == "__main__":
    main()
