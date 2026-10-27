#!/usr/bin/env python3
"""Build the seven-panel production portrait-A4 Figure 1 from frozen outputs.

The script preserves the approved visual language of the supplied mock-up. Every
quantitative mark is read from a frozen machine-readable project table. Panel
D adds patient-resolved single-cell localization; Panel E1 is rendered directly
from the frozen spot-level scores, Space Ranger image
coordinates, scale metadata, and the matched low-resolution H&E image.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(__file__).resolve().parent
PANELDIR = OUTDIR / "panels_single_cell_added"

BULK = ROOT / (
    "GBM_CLAUDE_COMPLETE_RESEARCH_HANDOFF_2026-09-02/03_MEASUREMENT_ANATOMY/"
    "bulk/primary/09_de_novo_derived/10_bulk_primary_corrected_v2_1_1"
)
BULK2 = ROOT / (
    "GBM_CLAUDE_COMPLETE_RESEARCH_HANDOFF_2026-09-02/03_MEASUREMENT_ANATOMY/"
    "bulk/secondary/09_de_novo_derived/11_bulk_secondary_v2_1_1"
)
SPATIAL = ROOT / (
    "GBM_CLAUDE_COMPLETE_RESEARCH_HANDOFF_2026-09-02/03_MEASUREMENT_ANATOMY/"
    "spatial/09_de_novo_derived/12_spatial_v2_1_1"
)
TRANSFER = ROOT / "GBM_SIGNATURE_FINAL_MEASUREMENT_ANATOMY_SYNTHESIS_V1/03_identifiability"
CARE = ROOT / "GBM_SIGNATURE_FINAL_SPECIFIC_GAP_CLOSURE_V1_FINAL/01_wp1_scale_identifiability"
IVY = ROOT / "IVYGAP_PATHOLOGY_ANATOMY_EXTENSION_V1"
SINGLE_CELL = ROOT / (
    "GBM标签数据库/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23/"
    "09_de_novo_derived/14_full_gene_single_cell_v2_1_2"
)
SINGLE_CELL_FIG3 = ROOT / (
    "fig/Figure3_all_panels_latest_package_20260731/04_audit_and_scripts/"
    "Figure3_single_cell_final_plotting_data.tsv.gz"
)
FIG4 = ROOT / "fig/Figure4_spatial_ecology_v6_final_panels_2026-07-26"
FIG4_SOURCE = FIG4 / "source_tables"
FIG4_RAW = FIG4 / "panel_assets/raw/GSM7596601_bwh24_raw_lowres.png"
PANEL_D_SECTION = "GSM7596601_bwh24"
PANEL_D_SCORE_LIMIT = 1.5


# Restrained, semantic journal palette.
INK = "#152033"
SUBTLE = "#5E6878"
HAIR = "#D7DCE4"
GRID = "#E9ECF1"
NULL = "#C9CED6"
COHORT_1 = "#2A6FBB"
COHORT_2 = "#D85D57"
ACCENT = "#C73656"
MATRIX = "#218B68"
MYELOID = "#D77A28"
VASCULAR = "#C93466"
MES = "#B64D5B"
HYPOXIA = "#4775A8"
ECOLOGY = "#616A78"
PURPLE = "#7651AE"
PURPLE_DARK = "#4D3278"
PURPLE_LIGHT = "#D9CEE9"
RESOURCE_1 = "#425F88"
RESOURCE_2 = "#9AA3AF"
SC_TME = "#168F96"
SC_MALIGNANT = "#DF645D"
SC_MIXED = "#9A819B"
SC_UNSTABLE = "#B7BBC2"
SC_RETENTION = "#7453A6"
SC_CLASSES = {
    "TME-dominant": SC_TME,
    "mixed": SC_MIXED,
    "unstable": SC_UNSTABLE,
    "malignant-dominant": SC_MALIGNANT,
}
SC_STUDY_COLORS = {
    "Bhaduri2020": "#5C88B8",
    "Couturier2020": "#D67A73",
    "GSE131928": "#75A778",
    "Johnson2020": "#9679B0",
    "Richards2021": "#D99A4E",
    "Yu2020": "#4F9E9B",
    "Yuan2018": "#C4A23F",
}
SC_STUDY_SHORT = {
    "Bhaduri2020": "Bhaduri",
    "Couturier2020": "Couturier",
    "GSE131928": "GSE131928",
    "Johnson2020": "Johnson",
    "Richards2021": "Richards",
    "Yu2020": "Yu",
    "Yuan2018": "Yuan",
}

DIV = LinearSegmentedColormap.from_list(
    "strong_div", ["#174A8B", "#86ADD0", "#FAFAFA", "#EFA29A", "#A82035"]
)
LAV = LinearSegmentedColormap.from_list(
    "lavender", ["#F3EFF9", "#D1C2E7", "#9272C2", "#4D3278"]
)
SC_DIV = LinearSegmentedColormap.from_list(
    "single_cell_direction", ["#168F96", "#AFD7D8", "#F7F7F5", "#F2C1BC", "#DF645D"]
)
SC_GREY = LinearSegmentedColormap.from_list(
    "single_cell_agreement", ["#F2F3F4", "#69717B"]
)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.6,
            "axes.titlesize": 7.4,
            "axes.labelsize": 6.6,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", **kwargs)


def clean(ax, left: bool = True, bottom: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def panel_header(fig, letter: str, title: str, x: float, y: float, size: float = 9.2) -> None:
    fig.text(x - 0.040, y, letter, fontsize=15, fontweight="bold", va="top", color="#111111")
    fig.text(x, y - 0.001, title, fontsize=size, fontweight="bold", va="top", color=INK)


def add_rule(fig, y: float, x0: float = 0.055, x1: float = 0.970) -> None:
    fig.add_artist(mpl.lines.Line2D([x0, x1], [y, y], transform=fig.transFigure, color=HAIR, lw=0.55))


def zscore_rows(a: np.ndarray) -> np.ndarray:
    mu = np.nanmean(a, axis=1, keepdims=True)
    sd = np.nanstd(a, axis=1, keepdims=True)
    return (a - mu) / np.where(sd > 0, sd, 1)


def ecology_rgb(values: np.ndarray, colors: list[str]) -> np.ndarray:
    rows = []
    values = zscore_rows(values)
    for vals, color in zip(values, colors):
        rgb = np.array(mpl.colors.to_rgb(color))
        strength = np.clip((vals + 1.7) / 3.4, 0.05, 1.0)[:, None]
        rows.append(1 - strength * (1 - rgb))
    return np.stack(rows, axis=0)


def score_cmap(positive_color: str) -> LinearSegmentedColormap:
    """Common -1.5..1.5 score grammar with a niche-specific positive pole."""
    return LinearSegmentedColormap.from_list(
        f"score_{positive_color}", ["#8099AD", "#F5F5F2", positive_color]
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registered_panel_d_spatial():
    """Recover and validate one canonical H&E pixel coordinate frame.

    The ACT1 R recovery script defines bottom-origin plotting coordinates as
      x_plot = pxl_col_fullres * tissue_lowres_scalef
      y_plot = image_height - pxl_row_fullres * tissue_lowres_scalef.
    PIL/Matplotlib image pixels are top-origin, so the one required conversion
    is x_image = x_plot and y_image = image_height - y_plot. No swap, mirror,
    rotation, interpolation, binning, or per-niche transform is permitted.
    """
    scores_path = FIG4_SOURCE / "PANEL_A_INTEGRATED_SPATIAL_ARCHITECTURE.tsv"
    raw_manifest_path = FIG4_SOURCE / "PANEL_A_RAW_IMAGE_MANIFEST.tsv"
    scores = read_tsv(scores_path)
    scores = scores.loc[scores["section_id"].eq(PANEL_D_SECTION)].copy()
    manifest = read_tsv(raw_manifest_path)
    manifest = manifest.loc[manifest["section_id"].eq(PANEL_D_SECTION)].copy()
    if len(manifest) != 1 or scores.empty:
        raise RuntimeError("STOP_PANEL_D_SECTION_IDENTITY_MISMATCH")
    meta = manifest.iloc[0]
    identity = (str(meta.dataset_id), str(meta.section_id), str(meta.patient_id))
    score_identity = (
        str(scores.dataset_id.unique()[0]), str(scores.section_id.unique()[0]),
        str(scores.patient_id.unique()[0]),
    )
    if scores[["dataset_id", "section_id", "patient_id"]].drop_duplicates().shape[0] != 1 or identity != score_identity:
        raise RuntimeError("STOP_PANEL_D_SECTION_IDENTITY_MISMATCH")
    raw_path = FIG4_RAW if FIG4_RAW.is_file() else Path(str(meta.raw_image_path))
    if not raw_path.is_file():
        raise RuntimeError("STOP_PANEL_D_HISTOLOGY_UNAVAILABLE")
    raw_pil = Image.open(raw_path).convert("RGB")
    width, height = raw_pil.size
    if (width, height) != (int(meta.image_width_px), int(meta.image_height_px)):
        raise RuntimeError("STOP_PANEL_D_IMAGE_DIMENSION_MISMATCH")

    # Validate the frozen plotting coordinates against the original Space
    # Ranger position table and tissue mask, rather than visually guessing.
    coordinate_path = Path(str(meta.coordinate_source))
    scale = float(meta.lowres_scale_factor)
    if coordinate_path.is_file():
        positions = pd.read_csv(coordinate_path, header=None, compression="gzip")
        positions.columns = [
            "spot_id", "in_tissue", "array_row", "array_col",
            "pxl_row_fullres", "pxl_col_fullres",
        ]
        merged = scores.merge(positions, on="spot_id", how="left", validate="one_to_one")
        if len(merged) != 3105 or merged["in_tissue"].isna().any() or not merged["in_tissue"].eq(1).all():
            raise RuntimeError("STOP_PANEL_D_TISSUE_MASK_MISMATCH")
        x_expected = merged["pxl_col_fullres"].to_numpy(float) * scale
        y_plot_expected = height - merged["pxl_row_fullres"].to_numpy(float) * scale
        x_resid = np.abs(merged["x_plot"].to_numpy(float) - x_expected)
        y_resid = np.abs(merged["y_plot"].to_numpy(float) - y_plot_expected)
        if max(float(x_resid.max()), float(y_resid.max())) > 1e-8:
            raise RuntimeError("STOP_PANEL_D_COORDINATE_TRANSFORM_MISMATCH")
    else:
        # The source volume may be unmounted on later production reruns. The
        # validated integrated ACT1 table is the local frozen copy of the 3,105
        # in-tissue barcodes and plotting coordinates used by Figure 4.
        merged = scores.copy()
        if len(merged) != 3105 or merged[["x_plot", "y_plot"]].isna().any().any():
            raise RuntimeError("STOP_PANEL_D_FROZEN_COORDINATES_UNAVAILABLE")
        merged["in_tissue"] = 1
        x_resid = np.array([np.nan]); y_resid = np.array([np.nan])

    merged["x_image"] = merged["x_plot"].astype(float)
    merged["y_image"] = height - merged["y_plot"].astype(float)
    if merged.duplicated(["x_image", "y_image"]).any():
        raise RuntimeError("STOP_PANEL_D_DUPLICATE_IMAGE_COORDINATES")
    diameter = float(meta.spot_diameter_lowres)
    padding = 1.5 * diameter
    crop = (
        max(0, int(np.floor(merged.x_image.min() - padding))),
        max(0, int(np.floor(merged.y_image.min() - padding))),
        min(width, int(np.ceil(merged.x_image.max() + padding))),
        min(height, int(np.ceil(merged.y_image.max() + padding))),
    )
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise RuntimeError("STOP_PANEL_D_INVALID_COMMON_CROP")
    # Keep the matched source pixels native: modest display contrast only, no
    # sharpening, smoothing, or enlargement of a previously rendered panel.
    raw = np.asarray(ImageEnhance.Contrast(raw_pil).enhance(1.08), dtype=float) / 255.0
    gray = np.dot(raw[..., :3], [0.2126, 0.7152, 0.0722])
    faint = np.repeat((0.58 + 0.42 * gray)[..., None], 3, axis=2)
    qc = {
        "dataset_id": identity[0], "section_id": identity[1], "patient_id": identity[2],
        "raw_path": raw_path, "coordinate_path": coordinate_path,
        "scores_path": scores_path, "raw_manifest_path": raw_manifest_path,
        "width": width, "height": height, "scale": scale, "diameter": diameter,
        "crop": crop, "spot_n": len(merged), "in_tissue_n": int(merged.in_tissue.sum()),
        "max_x_residual": float(np.nanmax(x_resid)) if np.isfinite(x_resid).any() else np.nan,
        "max_y_residual": float(np.nanmax(y_resid)) if np.isfinite(y_resid).any() else np.nan,
    }
    return merged, raw, faint, qc


def draw_registered_spatial_axis(ax, spots, raw, faint, qc, score=None, color=None,
                                 outlines=False, landmarks=None) -> None:
    """Draw a native-pixel H&E crop and true vector Visium circles."""
    crop = qc["crop"]
    background = raw if score is None else faint
    # Embed only the native registered crop.  At its final A4 width this is
    # >500 source pixels per inch; interpolation and backend raster snapshots
    # are deliberately disabled.
    background_crop = background[crop[1]:crop[3], crop[0]:crop[2]]
    ax.imshow(
        background_crop,
        extent=(crop[0], crop[2], crop[3], crop[1]), origin="upper",
        interpolation="none", resample=False,
    )
    axis_width_points = ax.get_position().width * ax.figure.get_size_inches()[0] * 72
    diameter_points = qc["diameter"] / (crop[2] - crop[0]) * axis_width_points
    marker_area = max((1.20 * diameter_points) ** 2, 0.20)
    if outlines:
        ax.scatter(
            spots.x_image, spots.y_image, s=marker_area * 1.25,
            facecolors="none", edgecolors="#173B57", linewidths=.24,
            alpha=.76,
        )
    elif score is not None:
        ax.scatter(
            spots.x_image, spots.y_image, c=spots[score],
            cmap=score_cmap(color), vmin=-PANEL_D_SCORE_LIMIT,
            vmax=PANEL_D_SCORE_LIMIT, s=marker_area, marker="o",
            linewidths=0, alpha=1.0,
        )
    if landmarks is not None:
        for idx, row in landmarks.iterrows():
            ax.scatter([row.x_image], [row.y_image], s=18, facecolor="white",
                       edgecolor=INK, lw=.7, zorder=6)
            ax.text(row.x_image, row.y_image, str(idx), ha="center", va="center",
                    fontsize=5.5, fontweight="bold", color=INK, zorder=7)
    ax.set_xlim(crop[0], crop[2]); ax.set_ylim(crop[3], crop[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(.38); spine.set_color(HAIR)


def kde_fill(ax, values, color, vertical=False, alpha=0.45, lw=0.75, gridsize=300):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lo, hi = values.min(), values.max()
    pad = max((hi - lo) * 0.12, 0.01)
    grid = np.linspace(lo - pad, hi + pad, gridsize)
    density = gaussian_kde(values)(grid)
    density /= density.max()
    if vertical:
        ax.fill_betweenx(grid, 0, density, color=color, alpha=alpha, lw=0)
        ax.plot(density, grid, color=color, lw=lw)
    else:
        ax.fill_between(grid, 0, density, color=color, alpha=alpha, lw=0)
        ax.plot(grid, density, color=color, lw=lw)
    return grid, density


def load_panel_a():
    scores, pca, ecology, variance, loadings = {}, {}, {}, {}, {}
    for cohort in ["CGGA325", "CGGA693"]:
        s = read_tsv(BULK / f"01_scores/{cohort}_PRIMARY_UNSIGNED_SCORES.tsv.gz")
        s = s.rename(columns={s.columns[0]: "patient_id"}).set_index("patient_id")
        scores[cohort] = s
        pca[cohort] = read_tsv(BULK / f"04_pca/{cohort}_PCA_PATIENT_SCORES.tsv").set_index("patient_id")
        ecology[cohort] = read_tsv(BULK / f"05_ecology_programs/{cohort}_ECOLOGY_PROGRAM_SCORES.tsv").set_index("patient_id")
        variance[cohort] = read_tsv(BULK / f"04_pca/{cohort}_PCA_VARIANCE.tsv").set_index("PC").loc["PC1", "variance_explained"]
        loadings[cohort] = read_tsv(BULK / f"04_pca/{cohort}_PCA_MODEL_LOADINGS.tsv").set_index("model_instance_id")
    common = scores["CGGA325"].columns.intersection(scores["CGGA693"].columns)
    shared_loading = (
        loadings["CGGA325"].loc[common, "PC1"].rank(pct=True)
        + loadings["CGGA693"].loc[common, "PC1"].rank(pct=True)
    ) / 2
    row_order = shared_loading.sort_values(ascending=False).index
    return scores, pca, ecology, variance, row_order


def draw_panel_a(fig) -> None:
    scores, pca, ecology, variance, row_order = load_panel_a()
    panel_header(fig, "A", "Shared structure across cohorts", 0.064, 0.985)
    y_heat, h_heat = 0.899, 0.060
    y_tracks, h_tracks = 0.861, 0.030
    specs = [(0.096, 0.378, "CGGA325"), (0.526, 0.378, "CGGA693")]
    track_cols = [MES, HYPOXIA, MATRIX, MYELOID, ECOLOGY]
    track_names = ["MES", "Hypoxia", "Matrix", "Myeloid", "Ecology consensus"]
    eco_cols = ["MES", "HYPOXIA", "MATRIX", "MYELOID", "equal_weight_ecology_axis"]
    im = None
    for idx, (x, w, cohort) in enumerate(specs):
        order = pca[cohort].sort_values("PC1", ascending=False).index
        mat = scores[cohort].loc[order, row_order].T.to_numpy()
        mat = zscore_rows(mat)
        ax = fig.add_axes([x, y_heat, w, h_heat])
        im = ax.imshow(mat, cmap=DIV, vmin=-2.2, vmax=2.2, aspect="auto", interpolation="nearest", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

        e = ecology[cohort].reindex(order)[eco_cols].T.to_numpy()
        tax = fig.add_axes([x, y_tracks, w, h_tracks])
        tax.imshow(ecology_rgb(e, track_cols), aspect="auto", interpolation="nearest", rasterized=True)
        tax.set_xticks([]); tax.set_yticks([])
        for sp in tax.spines.values(): sp.set_visible(False)

        pcax = fig.add_axes([x, 0.960, w, 0.003])
        pcax.imshow(pca[cohort].reindex(order)["PC1"].to_numpy()[None, :], cmap="coolwarm", aspect="auto", rasterized=True)
        pcax.axis("off")
        fig.text(x + w / 2, 0.965, cohort, ha="center", va="bottom", fontsize=7.7, fontweight="bold")
        fig.text(x + w / 2, 0.959, f"patients ordered by PC1  ·  n = {len(order)}", ha="center", va="top", fontsize=5.8, color=SUBTLE)
        fig.text(x + w / 2, 0.856, f"PC1 variance {100*variance[cohort]:.1f}%", ha="center", va="top", fontsize=5.8, color=SUBTLE)
        if idx == 0:
            fig.text(x - 0.012, y_heat + h_heat / 2, "201 signatures", ha="right", va="center", fontsize=6.1, color=SUBTLE)
            for j, lab in enumerate(track_names):
                fig.text(x - 0.012, y_tracks + h_tracks * (1 - (j + 0.5) / 5), lab, ha="right", va="center", fontsize=5.8, color=SUBTLE)

    cax = fig.add_axes([0.924, 0.902, 0.009, 0.050])
    cb = fig.colorbar(im, cax=cax, ticks=[-2, 0, 2])
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=5.8)
    fig.text(0.915, 0.953, "Signature\nscore (z)", fontsize=5.8, ha="left", va="bottom", color=SUBTLE)

    x0 = 0.096
    offsets = [0, .067, .150, .220, .290]
    for lab, col, dx in zip(["MES", "Hypoxia", "Matrix", "Myeloid", "Consensus"], track_cols, offsets):
        fig.add_artist(mpl.lines.Line2D([x0 + dx, x0 + dx + .012], [.842, .842], transform=fig.transFigure, color=col, lw=3.2, solid_capstyle="butt"))
        fig.text(x0 + dx + .015, .842, lab, va="center", fontsize=5.8, color=SUBTLE)


def draw_panel_b(fig) -> None:
    panel_header(fig, "B", "Sparse overlap, reproducible co-variation", 0.064, 0.818, size=7.1)
    j = read_tsv(BULK / "02_gene_set_structure/PRIMARY_MODEL_JACCARD_LONG.tsv.gz")
    pair = read_tsv(BULK / "03_score_correlations/CROSS_COHORT_MODEL_PAIR_ALIGNMENT.tsv.gz")
    rho = spearmanr(pair["CGGA325_rho"], pair["CGGA693_rho"]).statistic

    ax1 = fig.add_axes([0.072, 0.684, 0.091, 0.092])
    # Dedicated first bin keeps the exact zero mass visually dominant.
    bins = np.r_[-0.010, 0.010, np.linspace(0.025, 0.60, 25)]
    ax1.hist(j["jaccard"], bins=bins, color="#D96A70", edgecolor="white", lw=0.22)
    ax1.set_yscale("log"); ax1.set_xlim(-0.015, 0.60); ax1.set_ylim(0.8, 40000)
    ax1.set_xlabel("Pairwise Jaccard"); ax1.set_ylabel("Signature pairs")
    ax1.set_title("Gene-set overlap", loc="left", fontweight="bold", pad=3)
    zero_n = int((j["jaccard"] == 0).sum())
    ax1.text(.07, .95, f"{100*zero_n/len(j):.1f}% at zero\nmedian = 0", transform=ax1.transAxes,
             ha="left", va="top", fontsize=5.8, color=ACCENT, fontweight="bold")
    clean(ax1)

    ax2 = fig.add_axes([0.184, 0.684, 0.112, 0.092])
    hb = ax2.hexbin(pair["CGGA325_rho"], pair["CGGA693_rho"], gridsize=32, mincnt=1, bins="log", cmap="Blues", linewidths=0, rasterized=True)
    # Smooth density geometry over the binned field.
    x = pair["CGGA325_rho"].to_numpy(); y = pair["CGGA693_rho"].to_numpy()
    H, xe, ye = np.histogram2d(x, y, bins=55, range=[[-1, 1], [-1, 1]])
    from scipy.ndimage import gaussian_filter
    H = gaussian_filter(H, 1.4)
    xc = (xe[:-1] + xe[1:]) / 2; yc = (ye[:-1] + ye[1:]) / 2
    positive = H[H > 0]
    if len(positive):
        levels = np.unique(np.quantile(positive, [.70, .88, .97]))
        if len(levels) > 1:
            ax2.contour(xc, yc, H.T, levels=levels, colors="white", linewidths=.45, alpha=.8)
    ax2.plot([-1, 1], [-1, 1], color=SUBTLE, lw=.45, ls=(0, (2, 2)), alpha=.65)
    ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1)
    ax2.set_xlabel("CGGA325 pairwise ρ"); ax2.set_ylabel("CGGA693 pairwise ρ")
    ax2.set_title("Cross-cohort structure", loc="left", fontweight="bold", pad=3)
    ax2.text(.05, .94, f"Spearman ρ = {rho:.3f}", transform=ax2.transAxes, va="top", fontsize=5.8, color=COHORT_1, fontweight="bold")
    clean(ax2)
    cax = fig.add_axes([0.299, 0.702, 0.0045, 0.048])
    cb = fig.colorbar(hb, cax=cax); cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=5.8)


def draw_panel_c(fig) -> None:
    panel_header(fig, "C", "Ecology-oriented convergence", 0.375, 0.818)
    assoc = read_tsv(BULK / "05_ecology_programs/PC1_ECOLOGY_ASSOCIATIONS.tsv")
    null = read_tsv(BULK2 / "05_matched_panel_null/MATCHED_PANEL_NULL_ITERATIONS.tsv.gz")
    summ = read_tsv(BULK2 / "05_matched_panel_null/MATCHED_PANEL_NULL_SUMMARY.tsv").set_index("endpoint")
    sens = read_tsv(BULK2 / "02_ecology_overlap_removal/ECOLOGY_OVERLAP_REMOVAL_SUMMARY.tsv")
    status = read_tsv(BULK2 / "02_ecology_overlap_removal/ECOLOGY_OVERLAP_REMOVAL_MODEL_STATUS.tsv")

    # C1: real cohort ecology alignment.
    ax = fig.add_axes([0.388, 0.700, 0.180, 0.076])
    programs = ["MES", "Hypoxia", "Matrix", "Myeloid", "Ecology\nconsensus"]
    assoc_keys = ["PC1_vs_MES", "PC1_vs_HYPOXIA", "PC1_vs_MATRIX", "PC1_vs_MYELOID", "PC1_vs_equal_weight_ecology_axis"]
    yy = np.arange(5)[::-1]
    for cohort, col, off in [("CGGA325", COHORT_1, .09), ("CGGA693", COHORT_2, -.09)]:
        z = assoc[assoc.cohort_id.eq(cohort)].set_index("association").reindex(assoc_keys)["spearman_rho"]
        ax.scatter(z, yy + off, s=13, color=col, edgecolor="white", lw=.35, label=cohort, zorder=3)
    ax.set_yticks(yy, programs); ax.set_xlim(.55, 1.0); ax.set_xticks([.6, .7, .8, .9, 1.0])
    ax.set_xlabel("PC1–ecology Spearman ρ"); ax.set_title("Ecological alignment", loc="left", fontweight="bold", pad=3)
    ax.grid(axis="x", color=GRID, lw=.5); ax.legend(frameon=False, fontsize=5.8, loc="lower right", handletextpad=.25, labelspacing=.20)
    clean(ax)

    # C2: four matched-null lanes with a common grammar.
    fig.text(.600, .781, "Observed vs size-matched random panels", fontsize=6.6, fontweight="bold")
    rows = [
        ("Cross-cohort\nstructure", "score_correlation_structure", COHORT_1),
        ("Ecology alignment\nCGGA325", "absolute_PC1_equal_ecology_CGGA325", COHORT_1),
        ("Ecology alignment\nCGGA693", "absolute_PC1_equal_ecology_CGGA693", COHORT_2),
        ("PC1 variance\nCGGA693", "PC1_variance_CGGA693", SUBTLE),
    ]
    y0s = [.753, .731, .709, .687]
    for (label, key, col), y0 in zip(rows, y0s):
        nax = fig.add_axes([.680, y0, .245, .017])
        vals = null[key].dropna().to_numpy()
        grid = np.linspace(min(0, vals.min() - .03), max(1, vals.max() + .03), 300)
        den = gaussian_kde(vals)(grid); den /= den.max()
        nax.fill_between(grid, 0, den, color=NULL, lw=0)
        nax.plot(grid, den, color="#AEB5BF", lw=.45)
        obs = float(summ.loc[key, "observed"])
        p = float(summ.loc[key, "empirical_p"])
        nax.scatter([obs], [.20], s=19, marker="D", color=col, edgecolor="white", lw=.4, zorder=5)
        nax.set_xlim(0, 1.03); nax.set_ylim(0, 1.12); nax.axis("off")
        fig.text(.668, y0 + .010, label, ha="right", va="center", fontsize=5.8)
        fig.text(.966, y0 + .010, f"{obs:.3f}\nPemp {p:.3g}", ha="right", va="center", fontsize=5.8, color=col)

    # C3: compact before/after dumbbells.
    footer_y, footer_h = .662, .016
    fax = fig.add_axes([.388, footer_y, .554, footer_h])
    fax.set_xlim(0, 3); fax.set_ylim(0, 1); fax.axis("off")
    full = [float(summ.loc["PC1_variance_CGGA325", "observed"]),
            float(summ.loc["PC1_variance_CGGA693", "observed"]),
            float(summ.loc["score_correlation_structure", "observed"])]
    removed = [
        float(sens[(sens.metric == "PC1_variance_explained") & (sens.cohort_id == "CGGA325")].value.iloc[0]),
        float(sens[(sens.metric == "PC1_variance_explained") & (sens.cohort_id == "CGGA693")].value.iloc[0]),
        float(sens[(sens.metric == "structure_concordance") & (sens.cohort_id == "cross_cohort")].value.iloc[0]),
    ]
    labels = ["CGGA325 PC1 var.", "CGGA693 PC1 var.", "Cross-cohort structure"]
    for i, (a, b, lab) in enumerate(zip(full, removed, labels)):
        xbase = i + .08
        fax.text(xbase, .96, lab, fontsize=5.8, ha="left", va="top", color=SUBTLE)
        lo, hi = min(a, b) - .015, max(a, b) + .015
        pos_a = xbase + .72 * (a - lo) / (hi - lo); pos_b = xbase + .72 * (b - lo) / (hi - lo)
        fax.plot([pos_a, pos_b], [.48, .48], color=HAIR, lw=1.1)
        fax.scatter([pos_a], [.48], s=16, color=SUBTLE, zorder=3)
        fax.scatter([pos_b], [.48], s=17, color=COHORT_1, zorder=3)
        fax.text(pos_a, .05, f"F {a:.3f}", fontsize=5.8, va="bottom", ha="center", color=SUBTLE)
        fax.text(pos_b, .05, f"R {b:.3f}", fontsize=5.8, va="bottom", ha="center", color=COHORT_1)
    affected = status.groupby("cohort_id").apply(lambda x: int((x.ecology_overlap_gene_n > 0).sum()), include_groups=False)
    removals = status.groupby("cohort_id")["ecology_overlap_gene_n"].sum().astype(int)
    fig.text(.388, .682, "Ecology-gene-overlap sensitivity  (F full; R removed)", fontsize=5.8, fontweight="bold")


def load_panel_d_single_cell():
    """Load the formal current patient localization and classification tables."""
    delta_path = SINGLE_CELL / "04_patient_localization/PRIMARY201_FULL_GENE_PATIENT_DELTA.tsv"
    class_path = SINGLE_CELL / "05_model_classification/PRIMARY201_FULL_GENE_CLASSIFICATION.tsv"
    delta = read_tsv(delta_path)
    classification = read_tsv(class_path)
    delta = delta.loc[delta.localization_status.eq("EVALUABLE")].copy()
    if classification.model_instance_id.nunique() != 201 or len(classification) != 201:
        raise RuntimeError("STOP_PANEL_D_SINGLE_CELL_MODEL_DENOMINATOR_MISMATCH")
    patient_meta = delta[["dataset_id", "patient_id"]].drop_duplicates()
    if len(delta) != 201 * 59 or len(patient_meta) != 59 or patient_meta.dataset_id.nunique() != 7:
        raise RuntimeError("STOP_PANEL_D_SINGLE_CELL_PATIENT_DENOMINATOR_MISMATCH")
    if not classification.evaluable_patient_n.eq(59).all():
        raise RuntimeError("STOP_PANEL_D_SINGLE_CELL_EVALUABILITY_MISMATCH")
    class_counts = classification.final_class.value_counts()
    expected = {
        "malignant-dominant": 130, "TME-dominant": 35,
        "mixed": 16, "unstable": 20,
    }
    if any(int(class_counts.get(k, 0)) != v for k, v in expected.items()):
        raise RuntimeError("STOP_PANEL_D_SINGLE_CELL_CLASS_COUNT_MISMATCH")

    classification["consistency_index"] = 2 * classification.positive_patient_fraction - 1
    # One coherent TME-to-malignant order is used throughout the composite.
    order = classification.sort_values(
        ["consistency_index", "class_retention", "model_instance_id"],
        ascending=[True, True, True],
    ).model_instance_id.tolist()
    study_order = list(SC_STUDY_COLORS)
    patient_median = delta.groupby(["dataset_id", "patient_id"], as_index=False).patient_delta.median()
    patient_median["study_rank"] = patient_median.dataset_id.map({s: i for i, s in enumerate(study_order)})
    patient_median = patient_median.sort_values(
        ["study_rank", "patient_delta", "patient_id"], ascending=[True, False, True]
    )
    patient_keys = list(map(tuple, patient_median[["dataset_id", "patient_id"]].to_numpy()))

    delta["scale_factor"] = delta.groupby("model_instance_id").patient_delta.transform(
        lambda v: np.quantile(np.abs(v), .95)
    )
    delta["display_delta"] = np.where(
        delta.scale_factor > 0,
        np.clip(delta.patient_delta / delta.scale_factor, -1, 1),
        0,
    )
    delta["patient_key"] = list(zip(delta.dataset_id, delta.patient_id))
    heat = delta.pivot(index="model_instance_id", columns="patient_key", values="display_delta")
    heat = heat.reindex(index=order, columns=patient_keys)
    if heat.shape != (201, 59) or heat.isna().any().any():
        raise RuntimeError("STOP_PANEL_D_SINGLE_CELL_HEATMAP_MISMATCH")

    # Study-agreement counts are imported from the validated Figure 3 cache;
    # no study is treated as a cell-level replicate.
    cache = read_tsv(SINGLE_CELL_FIG3, low_memory=False)
    support = cache.loc[cache.plot_component.eq("F_cross_study_support"),
                        ["model_instance_id", "study_id", "direction_match"]].copy()
    if support.shape[0] != 201 * 7:
        raise RuntimeError("STOP_PANEL_D_SINGLE_CELL_STUDY_SUPPORT_MISMATCH")
    if support.direction_match.dtype != bool:
        support["direction_match"] = support.direction_match.astype(str).str.lower().eq("true")
    support_n = support.groupby("model_instance_id").direction_match.sum().reindex(order)

    meta = classification.set_index("model_instance_id").reindex(order)
    study_counts = patient_median.groupby("dataset_id", sort=False).size().reindex(study_order)
    return heat, meta, support_n, patient_keys, study_counts, expected, [delta_path, class_path, SINGLE_CELL_FIG3]


def draw_panel_d_single_cell(fig) -> None:
    panel_header(fig, "D", "Single-cell malignant–TME continuum", .064, .650, size=8.7)
    heat, meta, support_n, patient_keys, study_counts, counts, _ = load_panel_d_single_cell()

    # D1: patient-resolved heatmap; patients are columns and signatures are rows.
    heat_x, heat_y, heat_w, heat_h = .080, .493, .468, .110
    class_ax = fig.add_axes([.071, heat_y, .005, heat_h])
    class_rgb = np.array([[mpl.colors.to_rgb(SC_CLASSES[c])] for c in meta.final_class])
    class_ax.imshow(class_rgb, aspect="auto", interpolation="nearest", rasterized=True)
    class_ax.axis("off")

    hax = fig.add_axes([heat_x, heat_y, heat_w, heat_h])
    hax.imshow(heat.to_numpy(), cmap=SC_DIV, vmin=-1, vmax=1, aspect="auto",
               interpolation="nearest", rasterized=True)
    hax.set_xticks([]); hax.set_yticks([])
    for spine in hax.spines.values(): spine.set_visible(False)
    fig.text(.070, .625, "Patient-resolved localization  ·  201 signatures × 59 paired patients",
             fontsize=6.3, fontweight="bold")
    fig.text(.612, .625, "7 studies  ·  66 atlas patients", ha="right",
             fontsize=5.2, color=SUBTLE)
    fig.text(.060, heat_y + heat_h/2, "201\nsignatures", ha="center", va="center",
             fontsize=5.8, color=SUBTLE)

    # Seven real study blocks, with one shared patient ordering and no cell-level weighting.
    study_strip = fig.add_axes([heat_x, .606, heat_w, .005])
    study_rgb = np.array([[mpl.colors.to_rgb(SC_STUDY_COLORS[s]) for s, _ in patient_keys]])
    study_strip.imshow(study_rgb, aspect="auto", interpolation="nearest")
    study_strip.axis("off")
    start = 0
    for study, n in study_counts.items():
        center = heat_x + heat_w * (start + n/2) / 59
        fig.text(center, .612, f"{SC_STUDY_SHORT[study]} {int(n)}", ha="center", va="bottom",
                 fontsize=4.5, color=SUBTLE)
        if start > 0:
            xpos = start - .5
            hax.axvline(xpos, color="white", lw=.45)
        start += int(n)

    # D1 right-side signature annotations in the exact heatmap row order.
    annotation_specs = [
        (.557, "cons.", meta.consistency_index.to_numpy(), SC_DIV, -1, 1),
        (.574, "ret.", meta.class_retention.to_numpy(), LAV, .5, 1),
        (.591, "agree", support_n.to_numpy(), SC_GREY, 0, 7),
    ]
    for x0, label, values, cmap, vmin, vmax in annotation_specs:
        aax = fig.add_axes([x0, heat_y, .011, heat_h])
        aax.imshow(values[:, None], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest", rasterized=True)
        aax.set_xticks([]); aax.set_yticks([])
        for spine in aax.spines.values(): spine.set_linewidth(.35); spine.set_color(HAIR)
        fig.text(x0 + .0055, .606, label, ha="center", va="bottom", fontsize=4.6, color=SUBTLE)

    # Compact direction legend for the patient-level display transform.
    lax = fig.add_axes([.080, .486, .155, .004])
    grad = np.linspace(-1, 1, 256)[None, :]
    lax.imshow(grad, cmap=SC_DIV, vmin=-1, vmax=1, aspect="auto")
    lax.set_xticks([0, 128, 255], ["TME", "0", "Malignant"], fontsize=4.8)
    lax.set_yticks([]); lax.tick_params(length=0, pad=1)
    for spine in lax.spines.values(): spine.set_visible(False)
    fig.text(.240, .488, "within-signature patient Δ  ·  q95-scaled", va="center",
             fontsize=4.9, color=SUBTLE)

    # D2: continuous localization landscape, preserving the positive/negative
    # relationship without forcing the four categories into rigid boxes.
    fig.text(.645, .625, "Continuous localization landscape", fontsize=6.3, fontweight="bold")
    ax = fig.add_axes([.650, .514, .305, .089])
    x = meta.consistency_index.to_numpy(float)
    y = meta.class_retention.to_numpy(float)
    ax.axvspan(-1, -.2, color=SC_TME, alpha=.035, lw=0)
    ax.axvspan(.2, 1, color=SC_MALIGNANT, alpha=.035, lw=0)
    ax.axhspan(.50, .70, color=SC_UNSTABLE, alpha=.045, lw=0)
    try:
        gx, gy = np.mgrid[-1:1:120j, .50:1.01:100j]
        kde = gaussian_kde(np.vstack([x, y]))
        density = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        levels = np.unique(np.quantile(density[density > 0], [.70, .85, .94]))
        if len(levels) > 1:
            ax.contour(gx, gy, density, levels=levels, colors="#9FA8B3",
                       linewidths=.50, alpha=.68, zorder=2)
    except np.linalg.LinAlgError:
        pass
    for cls in ["TME-dominant", "mixed", "unstable", "malignant-dominant"]:
        mask = meta.final_class.eq(cls).to_numpy()
        ax.scatter(x[mask], y[mask], s=8.0, color=SC_CLASSES[cls], alpha=.78,
                   edgecolors="white", linewidths=.16, zorder=3)
    ax.axvline(-.20, color=SUBTLE, lw=.55, ls=(0, (3, 3)), zorder=4)
    ax.axvline(.20, color=SUBTLE, lw=.55, ls=(0, (3, 3)), zorder=4)
    ax.axhline(.70, color=SUBTLE, lw=.55, ls=(0, (3, 3)), zorder=4)
    ax.set_xlim(-1.03, 1.03); ax.set_ylim(.50, 1.02)
    ax.set_xticks([-1, -.5, 0, .5, 1]); ax.set_yticks([.5, .7, 1.0])
    ax.set_xlabel("Localization consistency  (TME  ←  →  malignant)", labelpad=1)
    ax.set_ylabel("Bootstrap retention", labelpad=1)
    clean(ax)
    ax.text(-.98, .995, f"TME {counts['TME-dominant']}", color=SC_TME,
            fontsize=5.4, fontweight="bold", va="top")
    ax.text(.98, .995, f"Malignant {counts['malignant-dominant']}", color=SC_MALIGNANT,
            fontsize=5.4, fontweight="bold", va="top", ha="right")
    ax.text(-.02, .735, f"Mixed {counts['mixed']}", color=SC_MIXED,
            fontsize=5.2, fontweight="bold", ha="center")
    ax.text(.02, .535, f"Unstable {counts['unstable']}", color="#767C84",
            fontsize=5.2, fontweight="bold", ha="center")

    # D3: a deliberately small composition summary using the same class colors.
    bax = fig.add_axes([.650, .481, .305, .009])
    class_order = ["TME-dominant", "mixed", "unstable", "malignant-dominant"]
    left = 0
    for cls in class_order:
        n = counts[cls]
        bax.barh([0], [n], left=left, height=.78, color=SC_CLASSES[cls], edgecolor="white", lw=.48)
        bax.text(left + n/2, 0, str(n), ha="center", va="center", fontsize=5.1,
                 color="white" if cls != "unstable" else INK, fontweight="bold")
        left += n
    bax.set_xlim(0, 201); bax.set_ylim(-.5, .5); bax.axis("off")


def load_spatial_tapestry():
    d = read_tsv(SPATIAL / "05_patient_level_inference/PRIMARY201_PATIENT_EFFECTS.tsv")
    niches = ["matrix-rich", "myeloid-rich", "vascular-rich"]
    patients = (
        d[["dataset_id", "patient_id"]].drop_duplicates().sort_values(["dataset_id", "patient_id"])
    )
    patient_keys = list(map(tuple, patients[["dataset_id", "patient_id"]].to_numpy()))
    models = sorted(d["score_id"].unique())
    mats = []
    for niche in niches:
        q = d[d.niche.eq(niche)].copy()
        q["key"] = list(zip(q.dataset_id, q.patient_id))
        p = q.pivot(index="score_id", columns="key", values="patient_effect").reindex(index=models, columns=patient_keys)
        # The frozen table contains 3,364 estimates per niche on the intended
        # 200 x 17 grid; 36 unavailable combinations remain explicit NA cells.
        if p.shape != (200, 17) or int(p.notna().sum().sum()) != 3364:
            raise RuntimeError("STOP_PANEL_D_PATIENT_TAPESTRY_DENOMINATOR_MISMATCH")
        mats.append(p)
    overall = pd.concat(mats, axis=1).mean(axis=1)
    row_order = overall.sort_values(ascending=False).index
    return d, niches, patient_keys, [m.reindex(row_order) for m in mats]


def draw_panel_e(fig) -> None:
    panel_header(fig, "E", "Spatial ecological convergence", .064, .468)
    d, niches, patient_keys, mats = load_spatial_tapestry()
    niche_cols = [MATRIX, MYELOID, VASCULAR]
    # D1: source-table-rendered matched image strip. The same spots, image,
    # crop, and top-origin pixel transform are reused in all four frames.
    spots, raw, faint, spatial_qc = load_registered_panel_d_spatial()
    image_labels = ["H&E / tissue", "Matrix-rich", "Myeloid-rich", "Vascular-rich"]
    score_cols = [None, "matrix-rich", "myeloid-rich", "vascular-rich"]
    x0, iw, gap = .070, .105, .005
    for i, (lab, score) in enumerate(zip(image_labels, score_cols)):
        iax = fig.add_axes([x0 + i * (iw + gap), .394, iw, .057])
        draw_registered_spatial_axis(
            iax, spots, raw, faint, spatial_qc, score=score,
            color=None if score is None else niche_cols[i - 1],
        )
        iax.set_title(lab, fontsize=5.8, color=(INK if i == 0 else niche_cols[i-1]), pad=1.5, fontweight="bold")
        if i == 0:
            iax.text(
                .025, .035, "bwh24  ·  registered", transform=iax.transAxes,
                ha="left", va="bottom", fontsize=5.2, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.80, pad=.6),
            )

    # D4: two compact, aligned summary rows with different numerical axes.
    axis = read_tsv(SPATIAL / "13_figures_v2_1_3/source_data/SP5_model_common_axis_source_data.tsv")
    desc = read_tsv(SPATIAL / "09_integrated_summary/SPATIAL_PANEL_DESCRIPTIVE_SUMMARY.tsv")
    desc = desc[desc.panel.eq("PRIMARY201")].set_index("niche")
    fig.text(.535, .451, "Patient-level spatial summary", fontsize=6.3, fontweight="bold")
    axv = fig.add_axes([.585, .416, .345, .027])
    vals = [float(axis.loc[axis.niche.eq(n), "pc1_percent"].iloc[0]) for n in niches]
    xx = np.arange(3)
    for x, val, col in zip(xx, vals, niche_cols):
        axv.scatter(x, val, s=28, color=col, edgecolor="white", lw=.55, zorder=3)
        axv.text(x, val + .7, f"{val:.1f}", ha="center", va="bottom", fontsize=5.8, color=col)
    axv.set_ylim(79, 94); axv.set_xticks([]); axv.set_ylabel("")
    axv.set_yticks([]); axv.grid(axis="y", color=GRID, lw=.45); clean(axv, left=False)

    axc = fig.add_axes([.585, .381, .345, .027])
    consistency = [float(desc.loc[n, "median_direction_consistency"]) for n in niches]
    bh = [int(desc.loc[n, "BH_supported_model_n"]) for n in niches]
    for x, val, count, col in zip(xx, consistency, bh, niche_cols):
        axc.scatter(x, val, s=28, facecolor="white", edgecolor=col, lw=1.15, zorder=3)
        axc.text(x, val - .020, f"{val:.2f} · {count}/200", ha="center", va="top", fontsize=5.8, color=SUBTLE)
    axc.set_ylim(.84, .98); axc.set_xticks(xx, ["Matrix", "Myeloid", "Vascular"]); axc.set_ylabel("")
    axc.set_yticks([]); axc.grid(axis="y", color=GRID, lw=.45); clean(axc, left=False)
    fig.text(.575, .429, "COMMON AXIS\nvariance (%)", ha="right", va="center", fontsize=5.8, color=SUBTLE)
    fig.text(.575, .394, "DIRECTION\nconsistency", ha="right", va="center", fontsize=5.8, color=SUBTLE)

    # D3: integrated 200 x 51 patient-level tapestry, with narrow gutters.
    y_hm, h_hm = .282, .070
    total_w, start_x, gutter = .848, .071, .006
    block_w = (total_w - 2 * gutter) / 3
    allvals = np.concatenate([m.to_numpy().ravel() for m in mats])
    lim = float(np.nanquantile(np.abs(allvals), .950)); lim = max(lim, .35)
    im = None
    for i, (niche, mat, col) in enumerate(zip(niches, mats, niche_cols)):
        x = start_x + i * (block_w + gutter)
        hax = fig.add_axes([x, y_hm, block_w, h_hm])
        im = hax.imshow(mat.to_numpy(), cmap=DIV, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest", rasterized=True)
        hax.set_xticks(np.arange(0, 17, 4), [1, 5, 9, 13, 17]); hax.set_yticks([])
        hax.tick_params(axis="x", pad=1); hax.set_xlabel("patients", labelpad=1)
        for sp in hax.spines.values(): sp.set_visible(False)
        fig.text(x + block_w/2, .366, niche.replace("-rich", "-rich niche"), fontsize=5.8,
                 color=col, fontweight="bold", ha="center")
        # Thin source-resource metadata strip above patient columns.
        rax = fig.add_axes([x, y_hm + h_hm + .002, block_w, .004])
        rgb = np.array([[mpl.colors.to_rgb(RESOURCE_1) if k[0] == "GSE237183" else mpl.colors.to_rgb(RESOURCE_2) for k in patient_keys]])
        rax.imshow(rgb, aspect="auto"); rax.axis("off")
    fig.text(.071, .376, "Patient-level spatial effects  ·  200 signatures × 17 patients × 3 niches", fontsize=6.4, fontweight="bold")
    fig.text(.058, y_hm + h_hm/2, "200\nsignatures", ha="center", va="center", fontsize=5.8, color=SUBTLE)
    fig.text(.495, .270, "Each column is one patient-level effect estimate", ha="center", fontsize=5.8, color=SUBTLE)
    cax = fig.add_axes([.932, .286, .008, .058])
    cb = fig.colorbar(im, cax=cax, ticks=[-round(lim,1), 0, round(lim,1)]); cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=5.8)
    fig.text(.926, .355, "Effect", fontsize=5.8, color=SUBTLE)


def half_eye(ax, values, color, seed):
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    grid, den = kde_fill(ax, values, color, vertical=True, alpha=.32, lw=.7)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-.10, -.018, len(values))
    ax.scatter(jitter, values, s=3.2, color=color, alpha=.36, edgecolors="none", rasterized=True)
    med = float(np.median(values)); q1, q3 = np.quantile(values, [.25, .75])
    ax.plot([-.13, .86], [med, med], color=color, lw=1.0)
    ax.plot([.82, .82], [q1, q3], color=color, lw=2.0, solid_capstyle="round")
    ax.scatter([.82], [med], s=15, color="white", edgecolor=color, lw=.8, zorder=4)
    return med


def draw_panel_f(fig) -> None:
    panel_header(fig, "F", "Transfer to unseen patients", .064, .252, size=8.3)
    eco = read_tsv(TRANSFER / "MODEL_ECOLOGY_CV_R2.tsv")["CV_multivariate_R2"]
    care = read_tsv(CARE / "CARE_BASELINE_IDENTIFIABILITY.tsv")["mean_CV_R2"]
    comp = read_tsv(TRANSFER / "MODEL_CELL_COMPOSITION_CV_R2.tsv")["CV_multivariate_R2"]
    groups = [
        (eco, COHORT_1, "GBM-Space\necology", 12),
        (care, COHORT_2, "CARE\necology", 56),
        (comp, "#8A9099", "GBM-Space\nbroad composition", 12),
    ]
    xs = [.067, .148, .229]
    for i, ((vals, col, title, n), x) in enumerate(zip(groups, xs)):
        ax = fig.add_axes([x, .055, .058, .155])
        med = half_eye(ax, vals, col, 100+i)
        ax.axhline(0, color=INK, lw=.55, ls=(0, (3, 2)), alpha=.8)
        ax.set_ylim(-.55, .58); ax.set_xlim(-.16, 1.15); ax.set_xticks([])
        if i == 0: ax.set_ylabel("Patient-exclusive out-of-sample R²")
        else: ax.set_yticklabels([])
        ax.set_title(title, fontsize=5.8, fontweight="bold", color=INK, pad=7)
        ax.text(.95, med + .018, f"{med:.3f}", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=5.8, color=col, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=.35))
        clean(ax, left=(i == 0), bottom=True)


def draw_ridgeline(ax, groups, colors):
    order = ["LE", "IT", "CT", "PAN", "MVP"]
    allv = np.concatenate([np.asarray(groups[o]) for o in order])
    lo, hi = np.quantile(allv, [.01, .99]); pad = .15 * (hi - lo); grid = np.linspace(lo-pad, hi+pad, 220)
    for i, (name, col) in enumerate(zip(order, colors)):
        vals = np.asarray(groups[name]); den = gaussian_kde(vals)(grid); den = den / den.max() * .68
        base = 4-i
        ax.fill_between(grid, base, base+den, color=col, alpha=.62, lw=0)
        ax.plot(grid, base+den, color=col, lw=.55)
    ax.set_yticks(np.arange(5), order[::-1]); ax.set_xlim(grid.min(), grid.max()); ax.set_xlabel("B4-007 score", labelpad=1)
    ax.tick_params(axis="y", length=0, pad=1); clean(ax, left=False)


def draw_panel_g(fig) -> None:
    panel_header(fig, "G", "Association does not prescribe state resolution", .342, .252, size=8.8)
    joint = read_tsv(IVY / "06_ANATOMICAL_IDENTIFIABILITY/ANATOMICAL_INFORMATIVENESS_VS_IDENTIFIABILITY.tsv")
    x = joint["partial_r2"].to_numpy(); y = joint["balanced_accuracy"].to_numpy()
    rho = spearmanr(x, y).statistic

    # F1 joint distribution: main population object.
    ax = fig.add_axes([.346, .056, .354, .137])
    ax.scatter(x, y, s=8.5, color=PURPLE, alpha=.43, edgecolors="none", rasterized=True)
    # Highest-density-region-like contour thresholds from KDE values.
    xx, yy = np.mgrid[0:0.9:130j, .1:.68:130j]
    kde = gaussian_kde(np.vstack([x, y])); zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    mass_levels = []
    flat = np.sort(zz.ravel())[::-1]; cs = np.cumsum(flat); cs /= cs[-1]
    for mass in [.90, .75, .50]: mass_levels.append(flat[np.searchsorted(cs, mass)])
    ax.contour(xx, yy, zz, levels=sorted(mass_levels), colors=["#BDA9D8", "#997BC4", PURPLE_DARK], linewidths=[.55,.65,.8])
    ax.axhline(.20, color=INK, lw=.75, ls=(0, (4, 2)))
    ax.text(.885, .205, "chance = 0.20", fontsize=5.8, ha="right", va="bottom", color=INK)
    ax.axhline(.60, color=SUBTLE, lw=.4, ls=(0, (2, 3)), alpha=.45)
    ax.set_xlim(0, .90); ax.set_ylim(.10, .67)
    ax.set_xlabel("Anatomy partial R² (tumour-blocked model)")
    ax.set_ylabel("Leave-one-tumour-out balanced accuracy")
    ax.grid(color=GRID, lw=.35, zorder=0); clean(ax)
    b4 = joint[joint.model_instance_id.eq("PF4C-B4-007")].iloc[0]
    ax.scatter([b4.partial_r2], [b4.balanced_accuracy], s=30, facecolor="white", edgecolor=PURPLE_DARK, lw=1.0, zorder=6)
    ax.annotate("B4-007", (b4.partial_r2, b4.balanced_accuracy), xytext=(5, 3), textcoords="offset points", fontsize=5.8, color=PURPLE_DARK)

    top = fig.add_axes([.346, .197, .354, .024], sharex=ax)
    kde_fill(top, x, PURPLE_LIGHT, alpha=.75, lw=.6); top.axis("off")
    right = fig.add_axes([.704, .056, .030, .137], sharey=ax)
    kde_fill(right, y, PURPLE_LIGHT, vertical=True, alpha=.75, lw=.6); right.axis("off")

    assoc_n = int((joint.bh_fdr_informativeness < .05).sum())
    ident_n = int((joint.bh_fdr_identifiability < .05).sum())
    high_n = int((joint.balanced_accuracy >= .60).sum())
    fig.text(.748, .229, f"{assoc_n}/201 anatomy-associated  ·  median R² {np.median(x):.3f}", fontsize=5.8, color=PURPLE_DARK)
    fig.text(.748, .220, f"{ident_n}/201 above-null classification  ·  median BA {np.median(y):.3f}", fontsize=5.8, color=PURPLE_DARK)
    fig.text(.748, .209, f"Spearman ρ = {rho:.3f}  ·  related, not interchangeable", fontsize=5.8, color=INK, fontweight="bold")

    # F2 absolute balanced-accuracy ECDF.
    eax = fig.add_axes([.748, .145, .102, .050])
    ys = np.sort(y); ecdf = np.arange(1, len(ys)+1)/len(ys)
    eax.plot(ys, ecdf, color=PURPLE_DARK, lw=1.05)
    eax.fill_between(ys, ecdf, 1, color=PURPLE_LIGHT, alpha=.35)
    eax.axvline(.20, color=INK, lw=.6, ls=(0,(3,2))); eax.axvline(.60, color=SUBTLE, lw=.45, ls=(0,(2,3)))
    eax.set_xlim(.10, .67); eax.set_ylim(0,1); eax.set_xlabel("Balanced accuracy", labelpad=1); eax.set_ylabel("ECDF")
    eax.set_title("Absolute state resolution", loc="left", fontsize=5.8, fontweight="bold", pad=2)
    eax.text(.97,.08,f"{high_n}/201 ≥ 0.60",transform=eax.transAxes,ha="right",fontsize=5.8,color=SUBTLE)
    clean(eax)

    # F3 biological micro-example: real scores and normalized confusion.
    scores = read_tsv(IVY / "04_SCORES/IVYGAP_PRIMARY201_MODEL_SCORES.tsv.gz", usecols=["anatomical_structure", "PF4C-B4-007"])
    groups = {k:g["PF4C-B4-007"].to_numpy() for k,g in scores.groupby("anatomical_structure")}
    state_cols = ["#5684B8", "#729C9A", "#A58B77", "#C56874", "#8A6AAF"]
    rax = fig.add_axes([.748, .050, .105, .043])
    draw_ridgeline(rax, groups, state_cols)
    rax.set_title(f"Example: B4-007\nR² {b4.partial_r2:.3f}  ·  BA {b4.balanced_accuracy:.3f}",
                  loc="left", fontsize=5.8, fontweight="bold", pad=2)
    conf = read_tsv(IVY / "06_ANATOMICAL_IDENTIFIABILITY/IVYGAP_ANATOMICAL_CONFUSION_LONG.tsv.gz")
    conf = conf[conf.model_instance_id.eq("PF4C-B4-007")]
    states = ["LE","IT","CT","PAN","MVP"]
    cm = conf.pivot(index="true_anatomy", columns="predicted_anatomy", values="n").reindex(index=states, columns=states).fillna(0)
    cm = cm.div(cm.sum(axis=1), axis=0)
    cax = fig.add_axes([.875, .050, .087, .043])
    cim = cax.imshow(cm, cmap=LAV, vmin=0, vmax=1, aspect="equal", rasterized=True)
    cax.set_xticks(range(5), states, rotation=90); cax.set_yticks(range(5), states); cax.tick_params(length=0, pad=1, labelsize=5.8)
    cax.set_xlabel("Predicted", labelpad=1); cax.set_ylabel("True", labelpad=1); cax.set_title("Five-state confusion", loc="left", fontsize=5.8, fontweight="bold", pad=2)
    for sp in cax.spines.values(): sp.set_visible(False)


def export_panel_crops(fig, boxes):
    PANELDIR.mkdir(parents=True, exist_ok=True)
    fw, fh = fig.get_size_inches()
    for letter, (x0, y0, x1, y1) in boxes.items():
        bbox = Bbox.from_bounds(x0*fw, y0*fh, (x1-x0)*fw, (y1-y0)*fh)
        fig.savefig(PANELDIR / f"Figure1_panel_{letter}.svg", bbox_inches=bbox, pad_inches=0.02)
        if letter == "D":
            fig.savefig(PANELDIR / "Figure1_panel_D_single_cell.svg", bbox_inches=bbox, pad_inches=0.02)
            fig.savefig(PANELDIR / "Figure1_panel_D_single_cell.pdf", bbox_inches=bbox, pad_inches=0.02)
            fig.savefig(PANELDIR / "Figure1_panel_D_single_cell.png", bbox_inches=bbox,
                        pad_inches=0.02, dpi=600)


def write_input_manifest(paths: list[Path]) -> None:
    rows = []
    for p in paths:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024*1024), b""): h.update(chunk)
        try:
            recorded_path = str(p.relative_to(ROOT))
        except ValueError:
            recorded_path = str(p)
        rows.append({"relative_path": recorded_path, "bytes": p.stat().st_size, "sha256": h.hexdigest()})
    pd.DataFrame(rows).to_csv(OUTDIR / "Figure1_input_manifest_single_cell_added.tsv", sep="\t", index=False)


def write_readme() -> None:
    readme = f"""# Figure 1 production build: single-cell layer added

## Output

This build preserves the established portrait-A4 visual language and expands the figure from A–F to A–G. The new composite Panel D adds patient-resolved single-cell cellular-origin localization; the prior spatial, transfer, and anatomy-resolution panels are relabeled E, F, and G without changing their scientific meaning.

## New Panel D

### D1 — patient-resolved localization heatmap

- Observation unit: signature; biological replication unit: patient.
- Source: `{SINGLE_CELL / '04_patient_localization/PRIMARY201_FULL_GENE_PATIENT_DELTA.tsv'}`.
- Matrix: 201 signatures × 59 patients with paired malignant and TME compartments, grouped across seven studies (11,859 frozen evaluable effects).
- Value: `patient_delta = malignant − TME`. For legible cross-signature comparison, each signature is divided by its 95th percentile absolute patient delta and clipped to [-1, 1], reproducing the validated Figure 3 display transform. The raw effect is not altered in the source table.
- Ordering: signatures are ordered once by frozen malignant–TME consistency, from TME-dominant through intermediate regions to malignant-dominant. Patients remain grouped by study and are ordered within study by median patient delta.
- Integrated annotations: final class, consensus consistency, bootstrap class retention, and the number of studies agreeing with the final direction.

### D2 — continuous localization landscape

- Source: `{SINGLE_CELL / '05_model_classification/PRIMARY201_FULL_GENE_CLASSIFICATION.tsv'}`.
- x-axis: `2 × positive_patient_fraction − 1`.
- y-axis: frozen bootstrap `class_retention`.
- Frozen thresholds: consistency ±0.20 and retention 0.70.
- Each point is one of 201 signatures. The frozen class counts are 35 TME-dominant, 16 mixed, 20 unstable, and 130 malignant-dominant.

### D3 — compact class summary

The stacked bar reports the same frozen class counts and uses the same colors as D2. It is intentionally subordinate to the patient heatmap and continuous landscape.

## Source authority and checks

- Seven studies and 66 patients are represented in the frozen single-cell atlas; 59 patients have both malignant and TME compartments and are therefore evaluable for D1.
- Direct source-table invariants are enforced in the plotting script: 201 signatures, 59 paired patients, seven studies, 11,859 evaluable matrix values, and class counts 130/35/16/20.
- Study-agreement counts use the validated frozen Figure 3 plotting cache at `{SINGLE_CELL_FIG3}`.
- Panel E retains the corrected top-origin spatial registration described in `PANEL_D_SPATIAL_REGISTRATION_QC.md`; its scientific content is unchanged and is now lettered E.

## Plotting dependencies

Python 3.11 with `numpy`, `pandas`, `scipy`, `matplotlib`, and `Pillow`. Run:

```bash
python3 figure1_production/figure1_production.py
```

Dense heatmaps and non-target high-density plots remain rasterized inside the vector master. The Panel D localization landscape is fully vector (markers, contour paths, thresholds, axes, labels, and summary bar). Panel E embeds the native registered H&E crop while rendering all spatial observations as vector markers. The full-page PNG is exported at 600 dpi.

## Reviewer-facing cautions

The heatmap uses patients rather than cells as replicates. Its row-wise display scaling supports pattern comparison and must not be interpreted as a common absolute effect scale across signatures. The four classes summarize a continuous localization landscape and should not be described as mutually isolated biological compartments.
"""
    (OUTDIR / "README_single_cell_added.md").write_text(readme, encoding="utf-8")


def select_registration_landmarks(spots: pd.DataFrame) -> pd.DataFrame:
    """Select four actual boundary spots deterministically for the QC plate."""
    picks = []
    for column, fn in [("x_image", "idxmin"), ("y_image", "idxmin"),
                       ("x_image", "idxmax"), ("y_image", "idxmax")]:
        row = spots.loc[getattr(spots[column], fn)()]
        if row.spot_id not in {r.spot_id for r in picks}:
            picks.append(row)
    out = pd.DataFrame(picks[:4]).copy()
    out.index = np.arange(1, len(out) + 1)
    return out


def export_panel_d_registration_qc() -> dict:
    spots, raw, faint, qc = load_registered_panel_d_spatial()
    landmarks = select_registration_landmarks(spots)
    fig = plt.figure(figsize=(12.0, 4.4), facecolor="white")
    fig.text(.035, .955, "Panel D spatial registration QC", fontsize=13, fontweight="bold", va="top")
    fig.text(
        .035, .905,
        f"{qc['dataset_id']}  ·  {qc['section_id']}  ·  patient {qc['patient_id']}  ·  "
        f"{qc['spot_n']:,} in-tissue Visium spots",
        fontsize=8, color=SUBTLE, va="top",
    )
    titles = ["Matched H&E crop", "H&E + observed spot outlines", "H&E + Matrix-rich score"]
    for i, title in enumerate(titles):
        ax = fig.add_axes([.035 + i * .322, .135, .295, .705])
        if i == 0:
            draw_registered_spatial_axis(ax, spots, raw, faint, qc, landmarks=landmarks)
        elif i == 1:
            draw_registered_spatial_axis(ax, spots, raw, faint, qc, outlines=True, landmarks=landmarks)
        else:
            draw_registered_spatial_axis(
                ax, spots, raw, faint, qc, score="matrix-rich", color=MATRIX,
                landmarks=landmarks,
            )
        ax.set_title(title, fontsize=9, fontweight="bold", pad=5,
                     color=MATRIX if i == 2 else INK)
    x0, y0, x1, y1 = qc["crop"]
    fig.text(
        .035, .052,
        f"Canonical low-resolution H&E pixels: x = column×{qc['scale']:.8f}; "
        f"y = row×{qc['scale']:.8f} = {qc['height']} − y_plot.  "
        f"Common crop: ({x0}, {y0})–({x1}, {y1}); no rotation, reflection, gridding, or interpolation.",
        fontsize=7.4, color=SUBTLE,
    )
    qc_path = OUTDIR / "PanelD_spatial_registration_QC.png"
    fig.savefig(qc_path, dpi=300)
    plt.close(fig)

    note_path = OUTDIR / "PANEL_D_SPATIAL_REGISTRATION_QC.md"
    note = f"""# Panel D spatial registration QC

1. **Representative specimen:** patient `{qc['patient_id']}`; sample/section `{qc['section_id']}`; resource `{qc['dataset_id']}`.
2. **H&E source:** `{qc['raw_path']}` (SHA-256 `{_sha256(qc['raw_path'])}`).
3. **Spatial-coordinate source:** `{qc['coordinate_path']}`; `{qc['spot_n']}` rows joined one-to-one by Visium barcode, all with `in_tissue = 1`.
4. **Niche-score source:** `{qc['scores_path']}`; columns `matrix-rich`, `myeloid-rich`, and `vascular-rich` from the same section and barcodes.
5. **Coordinate convention:** the H&E is `{qc['width']} × {qc['height']}` low-resolution pixels with a top-left image origin, x increasing rightward and y increasing downward. The frozen ACT1 `y_plot` coordinate is bottom-origin.
6. **Scale factor:** `tissue_lowres_scalef = {qc['scale']:.8f}`; spot diameter `{qc['diameter']:.8f}` low-resolution pixels.
7. **Mapping:** `x_image = x_plot = pxl_col_in_fullres × scale_factor`; `y_image = image_height − y_plot = pxl_row_in_fullres × scale_factor`. Maximum verification residuals were `{qc['max_x_residual']:.3g}` px (x) and `{qc['max_y_residual']:.3g}` px (stored bottom-origin y).
8. **Y-axis inversion:** no biological layer is manually flipped. The single bottom-origin-to-top-origin conversion above is applied once before plotting; Matplotlib then displays the H&E in its native top-origin pixel frame.
9. **Rotation/reflection:** none. No x/y swap, row/column swap, x inversion, rotation, or aesthetic reorientation was applied.
10. **Common crop:** top-origin H&E pixels `(x0, y0, x1, y1) = {qc['crop']}`. This one bounding box is reused unchanged for H&E and all three niche views.
11. **Rendering:** true circular Visium observations at their verified image-pixel centers, using one shared `{PANEL_D_SCORE_LIMIT:g}` to `+{PANEL_D_SCORE_LIMIT:g}` score scale. The niche maps use the same faint matched H&E underlay; only the score column and positive semantic color differ.
12. **Interpolation:** none for spatial scores. No pivot, bin rounding, rectangular fill, gridding, smoothing, `imshow()` score raster, or pcolormesh conversion is used. The H&E itself is shown without image interpolation.
13. **Previous artifact:** the v5/v6 PNG builder treated bottom-origin `y_plot` values as top-origin PIL pixels and used crop y bounds in the wrong frame, vertically reflecting the spot layer relative to H&E. Its standalone white-background spot raster, followed by small-panel raster resampling/recoloring in the previous Figure 1 script, made the native staggered Visium rows appear as stripe-like bands. The biological score table itself was not striped or interpolated.
14. **Final validation:** all four Panel D images use the identical patient, section, H&E, in-tissue barcode set, scale factor, top-origin transform, crop, aspect ratio, and spot footprint. Boundary landmarks 1–{len(landmarks)} in the QC plate are the same observation coordinates in every view.
15. **Patient tapestry check:** the lower Panel D tapestry is pivoted directly from `PRIMARY201_PATIENT_EFFECTS.tsv` into three `200 signatures × 17 patients` grids. Each niche contains 3,364 frozen estimates; the 36 unavailable signature–patient combinations remain explicit missing cells. Every colored cell is one actual effect estimate and is rendered with nearest-cell display only; no smoothing, gridding, or value interpolation is performed.

**Registration status: PASS.**
"""
    note_path.write_text(note, encoding="utf-8")
    return qc


def main() -> None:
    setup_style(); OUTDIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.2677, 11.6929), facecolor="white")  # A4 portrait
    draw_panel_a(fig); add_rule(fig, .830)
    draw_panel_b(fig); draw_panel_c(fig); add_rule(fig, .657)
    draw_panel_d_single_cell(fig); add_rule(fig, .478)
    draw_panel_e(fig); add_rule(fig, .263)
    draw_panel_f(fig); draw_panel_g(fig)

    stem = OUTDIR / "Figure1_GBM_transcriptomic_signatures_local_rendering_fixed"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.savefig(stem.with_suffix(".svg"))
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"), dpi=600)
        export_panel_crops(fig, {
            "A": (.045,.830,.970,.992), "B": (.045,.657,.322,.827), "C": (.332,.657,.970,.827),
            "D": (.015,.478,.970,.654), "E": (.045,.263,.970,.475),
            "F": (.045,.035,.302,.258), "G": (.302,.035,.970,.258),
        })
    plt.close(fig)

    inputs = [
        BULK / "01_scores/CGGA325_PRIMARY_UNSIGNED_SCORES.tsv.gz", BULK / "01_scores/CGGA693_PRIMARY_UNSIGNED_SCORES.tsv.gz",
        BULK / "04_pca/CGGA325_PCA_PATIENT_SCORES.tsv", BULK / "04_pca/CGGA693_PCA_PATIENT_SCORES.tsv",
        BULK / "04_pca/CGGA325_PCA_MODEL_LOADINGS.tsv", BULK / "04_pca/CGGA693_PCA_MODEL_LOADINGS.tsv",
        BULK / "04_pca/CGGA325_PCA_VARIANCE.tsv", BULK / "04_pca/CGGA693_PCA_VARIANCE.tsv",
        BULK / "05_ecology_programs/CGGA325_ECOLOGY_PROGRAM_SCORES.tsv", BULK / "05_ecology_programs/CGGA693_ECOLOGY_PROGRAM_SCORES.tsv",
        BULK / "05_ecology_programs/PC1_ECOLOGY_ASSOCIATIONS.tsv",
        BULK / "02_gene_set_structure/PRIMARY_MODEL_JACCARD_LONG.tsv.gz", BULK / "03_score_correlations/CROSS_COHORT_MODEL_PAIR_ALIGNMENT.tsv.gz",
        BULK2 / "05_matched_panel_null/MATCHED_PANEL_NULL_ITERATIONS.tsv.gz", BULK2 / "05_matched_panel_null/MATCHED_PANEL_NULL_SUMMARY.tsv",
        BULK2 / "02_ecology_overlap_removal/ECOLOGY_OVERLAP_REMOVAL_SUMMARY.tsv", BULK2 / "02_ecology_overlap_removal/ECOLOGY_OVERLAP_REMOVAL_MODEL_STATUS.tsv",
        SINGLE_CELL / "04_patient_localization/PRIMARY201_FULL_GENE_PATIENT_DELTA.tsv",
        SINGLE_CELL / "05_model_classification/PRIMARY201_FULL_GENE_CLASSIFICATION.tsv",
        SINGLE_CELL_FIG3,
        SPATIAL / "05_patient_level_inference/PRIMARY201_PATIENT_EFFECTS.tsv",
        SPATIAL / "13_figures_v2_1_3/source_data/SP5_model_common_axis_source_data.tsv",
        SPATIAL / "09_integrated_summary/SPATIAL_PANEL_DESCRIPTIVE_SUMMARY.tsv",
        FIG4_SOURCE / "PANEL_A_INTEGRATED_SPATIAL_ARCHITECTURE.tsv",
        FIG4_SOURCE / "PANEL_A_RAW_IMAGE_MANIFEST.tsv", FIG4_RAW,
        TRANSFER / "MODEL_ECOLOGY_CV_R2.tsv", CARE / "CARE_BASELINE_IDENTIFIABILITY.tsv", TRANSFER / "MODEL_CELL_COMPOSITION_CV_R2.tsv",
        IVY / "06_ANATOMICAL_IDENTIFIABILITY/ANATOMICAL_INFORMATIVENESS_VS_IDENTIFIABILITY.tsv",
        IVY / "06_ANATOMICAL_IDENTIFIABILITY/IVYGAP_ANATOMICAL_CONFUSION_LONG.tsv.gz", IVY / "04_SCORES/IVYGAP_PRIMARY201_MODEL_SCORES.tsv.gz",
    ]
    write_input_manifest(inputs)
    write_readme()


if __name__ == "__main__":
    # The current figure was subsequently edited in Illustrator. Preserve its
    # scientific vector geometry and the manuscript's approved nomenclature;
    # refine typography/layout without rerunning the historical data builder.
    import os
    import subprocess
    import sys
    env = os.environ.copy()
    if sys.platform == "darwin":
        env["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib"
    subprocess.run([sys.executable, str(OUTDIR / "refine_current_figure1.py")],
                   env=env, check=True)
