#!/usr/bin/env python3
"""Build the new Figure 1D (overlap-controlled ecological alignment) and the rebuilt Figure 1G
(anatomical information vs state resolution, Liu EMT-5 inset removed, layout expanded).
Inputs: verified tables only (05_MAIN_FIGURE_REDESIGN/source_data_candidates and the frozen Ivy table).
Every plotted number is asserted against the regression values in CODEX_FIGURE_IMPLEMENTATION_HANDOFF.md §4."""
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde, spearmanr

import os
from common import (COHORT_1, COHORT_2, GRID, HAIR, INK, IVY, LETTER1, NULL, PURPLE, PURPLE_DARK, PURPLE_LIGHT, SDC,
                    SUBTLE, Canvas, clean, golden_jitter, log_json, sha, style)

style()
log = {"inputs": {}, "asserts": {}}
T = lambda a, b, tol=1e-12: abs(float(a) - float(b)) <= tol  # noqa: E731

# ------------------------------------------------------------------ data + assertions (N1-N6)
ACT1 = __import__("pathlib").Path(os.environ["GBM_VALIDATION_ROOT"]) / "GBM_INDEPENDENT_SCIENTIFIC_AUDIT/logs/act1"
f_null_full, f_obs_full = ACT1 / "null_iterations_all.tsv.gz", ACT1 / "null_summary.tsv"   # full precision (authority)
f_null = SDC / "SD_Fig1D_independent_matched_null_500_panels.tsv.gz"                     # extracted candidate (10 s.f.)
f_obs = SDC / "SD_Fig1D_observed_and_null_summary.tsv"
f_ov = SDC / "SD_Fig1D_representation_programme_overlap.tsv"
for f in (f_null_full, f_obs_full, f_null, f_obs, f_ov):
    log["inputs"][str(f)] = sha(f)
it = pd.read_csv(f_null_full, sep="\t")
nul = it.melt(id_vars=["iteration", "transform", "family"], value_vars=["structure_rho", "absPC1_eco_CGGA325", "absPC1_eco_CGGA693"],
              var_name="endpoint", value_name="null_value")
obs = pd.read_csv(f_obs_full, sep="\t"); ov = pd.read_csv(f_ov, sep="\t")
cand = pd.read_csv(f_null, sep="\t").merge(nul, on=["iteration", "transform", "family", "endpoint"], suffixes=("_c", ""))
assert len(cand) == 500 * 4 * 2 * 3 and float((cand["null_value_c"] - cand["null_value"]).abs().max()) < 1e-9
ocand = pd.read_csv(f_obs, sep="\t").merge(obs, on=["transform", "family", "endpoint"], suffixes=("_c", ""))
assert float((ocand["observed_c"] - ocand["observed"]).abs().max()) < 1e-9
log["asserts"]["candidate_tables_vs_full_precision"] = {"null_rows": len(cand), "max_abs_diff": float((cand["null_value_c"] - cand["null_value"]).abs().max())}

def O(tr, fam, ep, col="observed"):
    r = obs[(obs["transform"] == tr) & (obs["family"] == fam) & (obs["endpoint"] == ep)]
    assert len(r) == 1, (tr, fam, ep); return float(r[col].iloc[0])

def NV(tr, fam, ep):
    v = nul[(nul["transform"] == tr) & (nul["family"] == fam) & (nul["endpoint"] == ep)]["null_value"].to_numpy()
    assert len(v) == 500; return v

EP = ["absPC1_eco_CGGA325", "absPC1_eco_CGGA693", "structure_rho"]
exp_obs = {"absPC1_eco_CGGA325": 0.9163581389608789, "absPC1_eco_CGGA693": 0.9333695351872493, "structure_rho": 0.6914194916493079}
exp_max = {"absPC1_eco_CGGA325": 0.7439837097371345, "absPC1_eco_CGGA693": 0.679985138186287, "structure_rho": 0.5874437399443826}
exp_full = {"absPC1_eco_CGGA325": 0.9390657781068741, "absPC1_eco_CGGA693": 0.9574818752560327, "structure_rho": 0.7309225783071415}
for ep in EP:
    v = NV("linear", "N2_ruleA_all201", ep); o = O("linear", "N2_ruleA_all201", ep)
    assert T(o, exp_obs[ep]) and T(v.max(), exp_max[ep]) and int((v >= o).sum()) == 0
    assert T(O("linear", "N1_full", ep), exp_full[ep])
    log["asserts"][f"N2-N4 {ep}"] = {"removed": o, "null_max": float(v.max()), "n_ge": 0, "full": O("linear", "N1_full", ep)}
SENS = [("All 201 · log2", "log2", "N2_ruleA_all201"),
        ("≤20% removed\n(n = 160)", "linear", "N2_ruleC_le20pct"),
        ("No programme gene\n(n = 114)", "linear", "N2_ruleD_nooverlap"),
        ("No programme gene\n· log2 (n = 114)", "log2", "N2_ruleD_nooverlap")]
exp_sens = {("log2", "N2_ruleA_all201"): (0.9473651733925706, 0.8557324447735409, 0.938418741128164, 0.7126906551582879),
            ("linear", "N2_ruleC_le20pct"): (0.9003455510304826, 0.5628162408984327, 0.9037983366200806, 0.7056027132336828),
            ("linear", "N2_ruleD_nooverlap"): (0.8905343699864249, 0.8160249290386279, 0.9101622414663655, 0.7151009364848),
            ("log2", "N2_ruleD_nooverlap"): (0.9224052819943231, 0.8490990990990992, 0.9128678537064029, 0.7366315127611536)}
for (tr, fam), e in exp_sens.items():
    got = (O(tr, fam, EP[0]), O(tr, fam, EP[0], "null_max"), O(tr, fam, EP[1]), O(tr, fam, EP[1], "null_max"))
    assert all(T(a, b) for a, b in zip(got, e)), (tr, fam, got)
    assert O(tr, fam, EP[0], "n_null_ge_obs") == 0 and O(tr, fam, EP[1], "n_null_ge_obs") == 0
log["asserts"]["N5 sensitivity"] = {f"{tr}/{fam}": e for (tr, fam), e in exp_sens.items()}
rep_n = {fam: int(obs[obs["family"] == fam]["rep_n"].iloc[0]) for fam in ["N2_ruleA_all201", "N2_ruleC_le20pct", "N2_ruleD_nooverlap"]}
assert rep_n == {"N2_ruleA_all201": 201, "N2_ruleC_le20pct": 160, "N2_ruleD_nooverlap": 114}, rep_n
log["asserts"]["N6 family n"] = rep_n
cls = ov["removal_class"].value_counts().to_dict()
assert len(ov) == 201 and cls == {"none": 114, "<=20%": 47, "20-50%": 30, ">=50%": 10}, cls
assert int((ov["eco_shared"] > 0).sum()) == 87 and int(ov["n"].sum()) == 1729 and int(ov["eco_shared"].sum()) == 176
log["asserts"]["N1 waffle"] = {"n": 201, "classes": cls, "with_gene": 87, "rows": 1729, "programme_rows": 176,
                               "note": "<=20% class has 47 representations; the rule-C family is 160 because one of the 161 with "
                                       "0-20% removal keeps <3 genes / <80% coverage after removal"}

# ------------------------------------------------------------------ Figure 1D (594.72 x 164 pt; page y 284-448)
W, H = 594.72, 164.0
c = Canvas(W, H)
c.text(10.99, 13.5, "D", 12, "bold", LETTER1)
c.text(28.83, 22.2, "Ecological alignment survives removal of programme genes", 10, "bold", INK)
c.text(588, 22.2, "post hoc sensitivity analysis · independent 500-panel matched null", 5.8, color=SUBTLE, ha="right", style="italic")

# D1 waffle
c.text(20, 36, "Programme-gene content", 6.6, "bold")
c.text(20, 43.5, "87/201 representations · 176/1,729 genes removed", 5.6, color=SUBTLE)
CL = [("none", "none", "white", 114), ("<=20%", "≤20%", "#C9CED6", 47), ("20-50%", "20–50%", "#8A9099", 30), (">=50%", "≥50%", INK, 10)]
order = ov.assign(k=ov["removal_class"].map({k: i for i, (k, *_r) in enumerate(CL)})).sort_values(["k", "frac", "rep"])
ax = c.ax(20, 49, 20 + 15 * 6.6, 49 + 14 * 6.6)
ax.set_xlim(0, 15 * 6.6); ax.set_ylim(14 * 6.6, 0); ax.axis("off")
fill = dict((k, col) for k, _l, col, _n in CL)
for i, k in enumerate(order["removal_class"]):
    r, q = divmod(i, 15)
    ax.add_patch(Rectangle((q * 6.6 + 0.35, r * 6.6 + 0.35), 5.9, 5.9, facecolor=fill[k], edgecolor="#AEB4BE" if k == "none" else "none", lw=0.35))
for j, (k, lab, col, n) in enumerate(CL):
    y = 66 + j * 11.5
    sw = c.ax(124, y - 5.2, 129.6, y + 0.4); sw.set_xlim(0, 1); sw.set_ylim(0, 1); sw.axis("off")
    sw.add_patch(Rectangle((0, 0), 1, 1, facecolor=col, edgecolor="#AEB4BE" if k == "none" else "none", lw=0.35))
    c.text(132, y, f"{lab}", 5.8, color=INK)
    c.text(166, y, f"{n}", 5.8, "bold", INK, ha="right")
c.text(124, 55, "share removed", 5.6, color=SUBTLE)
c.text(20, 153, "each square = one representation", 5.6, color=SUBTLE)

# D2 tracks
X0, X1 = 196, 392
c.text(186, 36, "Observed alignment vs 500 matched panels after removal", 6.6, "bold")
c.text(186, 43.5, "0/500 panels ≥ observed on every track · plus-one P = 0.002", 5.6, color=SUBTLE)
tracks = [("absPC1_eco_CGGA325", "Ecological alignment · CGGA325", COHORT_1, 62),
          ("absPC1_eco_CGGA693", "Ecological alignment · CGGA693", COHORT_2, 94),
          ("structure_rho", "Cross-cohort structure", INK, 126)]
ax = c.ax(X0, 44, X1, 138)
ax.set_xlim(0, 1); ax.set_ylim(138, 44)
for x in np.arange(0, 1.01, 0.2):
    ax.axvline(x, color=GRID, lw=0.35, zorder=0)
for ep, lab, col, yc in tracks:
    v = np.sort(NV("linear", "N2_ruleA_all201", ep)); o = O("linear", "N2_ruleA_all201", ep); f = O("linear", "N1_full", ep)
    ax.scatter(v, yc + golden_jitter(len(v), 6.2), s=1.5, color="#9AA1AC", alpha=0.75, lw=0, zorder=2)
    ax.plot([v.max()] * 2, [yc - 8.5, yc + 8.5], color=SUBTLE, lw=0.6, zorder=3)
    ax.text(v.max(), yc - 9.6, f"max {v.max():.3f}", fontsize=5.6, color=SUBTLE, ha="center", va="bottom")
    ax.plot([o, f], [yc, yc], color=col, lw=0.6, zorder=4)
    ax.plot([f], [yc], marker="D", ms=4.2, mfc="white", mec=col, mew=0.8, zorder=5)
    ax.plot([o], [yc], marker="D", ms=4.6, mfc=col, mec=col, mew=0.4, zorder=6)
    c.text(X0, yc - 10.5, lab, 6.0, "bold", col)
    c.text(397, yc + 1.2, f"{o:.3f}", 7.0, "bold", col)
    c.text(397, yc + 8.2, f"all genes {f:.3f}", 5.6, color=SUBTLE)
ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
ax.tick_params(axis="y", left=False, labelleft=False)
ax.spines[["top", "right", "left"]].set_visible(False)
c.text((X0 + X1) / 2, 153, "Spearman ρ (absolute value for alignment)", 6.6, ha="center")
lg = c.ax(186, 156, 440, 163); lg.axis("off"); lg.set_xlim(186, 440); lg.set_ylim(163, 156)
lg.plot([189], [159.5], marker="D", ms=3.6, mfc=INK, mec=INK, mew=0.4)
lg.plot([254], [159.5], marker="D", ms=3.4, mfc="white", mec=INK, mew=0.7)
lg.scatter([293], [159.5], s=1.8, color="#9AA1AC", lw=0)
c.text(193, 161.6, "genes removed", 5.6, color=SUBTLE)
c.text(258, 161.6, "all genes", 5.6, color=SUBTLE)
c.text(297, 161.6, "one matched panel (same removal)", 5.6, color=SUBTLE)

# D3 sensitivity dumbbells
c.text(452, 36, "Sensitivity settings", 6.6, "bold")
c.text(452, 43.5, "|ρ| with ecology: ● observed  | highest panel", 5.6, color=SUBTLE)
AX0, AX1 = 523, 588
ax = c.ax(AX0, 48, AX1, 132)
ax.set_xlim(0.5, 1.0); ax.set_ylim(132, 48)
for x in [0.5, 0.75, 1.0]:
    ax.axvline(x, color=GRID, lw=0.35, zorder=0)
for i, (lab, tr, fam) in enumerate(SENS):
    y0 = 57 + i * 20
    c.text(452, y0 + (0 if "\n" not in lab else -2.6), lab.split("\n")[0], 5.8, color=INK)
    if "\n" in lab:
        c.text(452, y0 + 4.4, lab.split("\n")[1], 5.6, color=SUBTLE)
    for dy, ep, col in [(-2.6, EP[0], COHORT_1), (3.4, EP[1], COHORT_2)]:
        o = O(tr, fam, ep); m = O(tr, fam, ep, "null_max")
        ax.plot([m, o], [y0 + dy] * 2, color=col, lw=0.6, alpha=0.6, zorder=3)
        ax.plot([m, m], [y0 + dy - 2.2, y0 + dy + 2.2], color=SUBTLE, lw=0.7, zorder=4)
        ax.plot([o], [y0 + dy], marker="o", ms=3.3, mfc=col, mec="white", mew=0.3, zorder=5)
ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xticklabels(["0.5", "0.75", "1.0"])
ax.tick_params(axis="y", left=False, labelleft=False)
ax.spines[["top", "right", "left"]].set_visible(False)
c.text(452, 153, "blue CGGA325 · red CGGA693", 5.6, color=SUBTLE)
c.text(452, 161.5, "0/500 panels ≥ observed in every setting", 5.6, color=SUBTLE)
pd1 = c.save("Fig1D_overlap_controlled_alignment")

# ------------------------------------------------------------------ Figure 1G rebuilt (page x 175.2-594.72, y 622-856.14)
f_g = IVY / "06_ANATOMICAL_IDENTIFIABILITY/ANATOMICAL_INFORMATIVENESS_VS_IDENTIFIABILITY.tsv"
log["inputs"][str(f_g)] = sha(f_g)
J = pd.read_csv(f_g, sep="\t"); assert len(J) == 201
x = J["partial_r2"].to_numpy(); y = J["balanced_accuracy"].to_numpy(); rho = spearmanr(x, y).statistic
assoc_n = int((J["bh_fdr_informativeness"] < .05).sum()); ident_n = int((J["bh_fdr_identifiability"] < .05).sum())
high_n = int((J["balanced_accuracy"] >= .60).sum())
assert (assoc_n, ident_n, high_n) == (195, 170, 2) and round(rho, 3) == 0.804
assert round(float(np.median(x)), 3) == 0.404 and round(float(np.median(y)), 3) == 0.350
liu = J[J["model_instance_id"] == "PF4C-B4-007"].iloc[0]
assert round(liu.partial_r2, 3) == 0.530 and round(liu.balanced_accuracy, 3) == 0.349
log["asserts"]["G"] = {"assoc": assoc_n, "ident": ident_n, "ba_ge_0.60": high_n, "rho": rho, "median_r2": float(np.median(x)),
                       "median_ba": float(np.median(y)), "liu": [float(liu.partial_r2), float(liu.balanced_accuracy)]}
GX0, GY0 = 175.2, 622.0
W, H = 594.72 - GX0, 856.14 - GY0
c = Canvas(W, H)
P = lambda px, py: (px - GX0, py - GY0)  # noqa: E731
c.text(*P(179.68, 637.45), "G", 12, "bold", LETTER1)
c.text(*P(204.88, 645.32), "Association does not prescribe state resolution", 10, "bold", INK)
mx0, mx1, my0, my1 = 212, 436, 676, 822
ax = c.ax(*P(mx0, my0), *P(mx1, my1))
ax.scatter(x, y, s=9, color=PURPLE, alpha=.43, edgecolors="none", rasterized=True, zorder=3)
xx, yy = np.mgrid[0:0.9:130j, .1:.68:130j]
kde = gaussian_kde(np.vstack([x, y])); zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
flat = np.sort(zz.ravel())[::-1]; cs = np.cumsum(flat); cs /= cs[-1]
lev = sorted(flat[np.searchsorted(cs, m)] for m in [.90, .75, .50])
ax.contour(xx, yy, zz, levels=lev, colors=["#BDA9D8", "#997BC4", PURPLE_DARK], linewidths=[.55, .65, .8], zorder=4)
ax.axhline(.20, color=INK, lw=.75, ls=(0, (4, 2)), zorder=2)
ax.text(.885, .205, "chance = 0.20", fontsize=5.8, ha="right", va="bottom", color=INK)
ax.axhline(.60, color=SUBTLE, lw=.45, ls=(0, (2, 3)), alpha=.6, zorder=2)
ax.set_xlim(0, .90); ax.set_ylim(.10, .67)
ax.set_xlabel("Anatomy partial R² (tumour-blocked model)", labelpad=2)
ax.set_ylabel("Leave-one-tumour-out balanced accuracy", labelpad=3)
ax.grid(color=GRID, lw=.35, zorder=0); clean(ax)
ax.scatter([liu.partial_r2], [liu.balanced_accuracy], s=30, facecolor="white", edgecolor=PURPLE_DARK, lw=1.0, zorder=6)
import matplotlib.patheffects as pe
ax.annotate("Liu EMT-5", (liu.partial_r2, liu.balanced_accuracy), xytext=(5, 3), textcoords="offset points", fontsize=5.8, color=PURPLE_DARK,
            path_effects=[pe.withStroke(linewidth=1.6, foreground="white")], zorder=7)

def kde_fill(a, v, vertical=False):
    lo, hi = v.min(), v.max(); pad = max((hi - lo) * .12, .01); g = np.linspace(lo - pad, hi + pad, 300)
    d = gaussian_kde(v)(g); d /= d.max()
    if vertical:
        a.fill_betweenx(g, 0, d, color=PURPLE_LIGHT, alpha=.75, lw=0); a.plot(d, g, color=PURPLE_LIGHT, lw=.6)
    else:
        a.fill_between(g, 0, d, color=PURPLE_LIGHT, alpha=.75, lw=0); a.plot(g, d, color=PURPLE_LIGHT, lw=.6)

top = c.ax(*P(mx0, 657), *P(mx1, my0 - 2), sharex=ax); kde_fill(top, x); top.axis("off")
rgt = c.ax(*P(mx1 + 2, my0), *P(mx1 + 20, my1), sharey=ax); kde_fill(rgt, y, vertical=True); rgt.axis("off")
RX = 468
c.text(*P(RX, 664), f"{assoc_n}/201 anatomy-associated", 5.8, color=PURPLE_DARK)
c.text(*P(RX, 671.5), f"median partial R² {np.median(x):.3f}", 5.8, color=PURPLE_DARK)
c.text(*P(RX, 682), f"{ident_n}/201 above-null classification", 5.8, color=PURPLE_DARK)
c.text(*P(RX, 689.5), f"median BA {np.median(y):.3f}", 5.8, color=PURPLE_DARK)
c.text(*P(RX, 700), f"Spearman ρ = {rho:.3f}", 5.8, "bold", INK)
c.text(*P(RX, 716), "Absolute state resolution", 5.8, "bold", INK)
eax = c.ax(*P(RX + 18, 722), *P(586, 808))
ys = np.sort(y); ecdf = np.arange(1, len(ys) + 1) / len(ys)
eax.plot(ys, ecdf, color=PURPLE_DARK, lw=1.05); eax.fill_between(ys, ecdf, 1, color=PURPLE_LIGHT, alpha=.35)
eax.axvline(.20, color=INK, lw=.6, ls=(0, (3, 2))); eax.axvline(.60, color=SUBTLE, lw=.45, ls=(0, (2, 3)))
eax.set_xlim(.10, .67); eax.set_ylim(0, 1); eax.set_xticks([.2, .4, .6]); eax.set_yticks([0, .5, 1])
eax.set_xlabel("Balanced accuracy", labelpad=2); eax.set_ylabel("ECDF", labelpad=2)
eax.text(.585, .06, f"{high_n}/201 ≥ 0.60", ha="right", va="bottom", fontsize=5.8, color=SUBTLE)
clean(eax)
pg = c.save("Fig1G_state_resolution_rebuilt")
log["outputs"] = {str(p): sha(p) for p in [pd1, pd1.with_suffix(".svg"), pg, pg.with_suffix(".svg")]}
log_json("build_fig1_panels.json", log)
print("Figure 1 panels built:", pd1, pg)
