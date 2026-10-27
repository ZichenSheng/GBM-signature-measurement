#!/usr/bin/env python3
"""Build the new Figure 5D (Ivy anatomical-state profiles of the three published membership representations) and the
rebuilt Figure 5G (reference-dependent detection and null calibration, three representations only).
Inputs: verified source-data candidates (05_MAIN_FIGURE_REDESIGN) and frozen ACT4 ledgers on the SSD.
Regression values N7-N15 (CODEX_FIGURE_IMPLEMENTATION_HANDOFF.md §4) are asserted before drawing."""
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

from common import (F5SRC, INK5, R4, R4CLOSE, REF5, REP, SDC, SEC5, STATES, TEXT5, Canvas, clean, log_json, sha, style)

style(6.6)
import matplotlib as mpl
mpl.rcParams.update({"text.color": TEXT5, "axes.labelcolor": TEXT5, "axes.edgecolor": "#33363D", "xtick.color": "#33363D",
                     "ytick.color": "#33363D"})
log = {"inputs": {}, "asserts": {}}
IDS = list(REP)

# ------------------------------------------------------------------ Fig 5D data + assertions (N7-N11)
f_tm, f_sum, f_st, f_gene, f_smp = (SDC / n for n in ["SD_Fig5D_ivy_tumour_state_means.tsv", "SD_Fig5D_ivy_state_profile_summary.tsv",
                                                      "SD_Fig5D_ivy_frozen_model_statistics.tsv", "SD_Fig5D_ivy_gene_state_profiles.tsv",
                                                      "SD_Fig5D_ivy_sample_scores.tsv"])
# Observation-level inputs remain external; aggregate tables above are public.
import os
from pathlib import Path
_external_sd = Path(os.environ["GBM_VALIDATION_ROOT"]) / "GBM_NPJSBA_REVISION/05_MAIN_FIGURE_REDESIGN/source_data_candidates"
f_tm = _external_sd / "SD_Fig5D_ivy_tumour_state_means.tsv"
f_smp = _external_sd / "SD_Fig5D_ivy_sample_scores.tsv"
f_coef = SDC / "A_coefficients_membership.tsv"
for f in (f_tm, f_sum, f_st, f_gene, f_smp, f_coef):
    log["inputs"][str(f)] = sha(f)
tm = pd.read_csv(f_tm, sep="\t", dtype={"tumor_id": str}); st = pd.read_csv(f_st, sep="\t"); gene = pd.read_csv(f_gene, sep="\t")
smp = pd.read_csv(f_smp, sep="\t", dtype={"tumor_id": str}); coef = pd.read_csv(f_coef, sep="\t")
# recompute tumour-state means from sample scores (independent of the candidate table)
chk = smp.groupby(["model_instance_id", "tumor_id", "anatomical_structure"])["score_minus_tumour_CT_mean"].mean().rename("re").reset_index()
mm = tm.merge(chk, on=["model_instance_id", "tumor_id", "anatomical_structure"])
assert len(mm) == len(tm) == 129 and float((mm["re"] - mm["state_mean_minus_CT"]).abs().max()) < 1e-9
EXP = {"PF4C-B4-007": (0.530064, 0.348719), "PF4C-B4-029": (0.59055, 0.370053), "PF4C-FINAL-079": (0.655243, 0.493035)}
for m, (r2, ba) in EXP.items():
    r = st[st["model_instance_id"] == m].iloc[0]
    assert abs(r.partial_r2_ALL5_tumour_blocked - r2) < 5e-7 and abs(r.balanced_accuracy_ALL5_LOTO - ba) < 5e-7 and r.ba_chance == 0.2
    assert abs(r.partial_r2_bh_fdr - 0.000121) < 5e-7
tps = tm.groupby("anatomical_structure")["tumor_id"].nunique().reindex(STATES).to_dict()
assert tps == {"LE": 8, "IT": 8, "CT": 10, "PAN": 8, "MVP": 9} and all(tm.groupby("model_instance_id").size() == 43)
P = {m: tm[tm["model_instance_id"] == m].pivot(index="tumor_id", columns="anatomical_structure", values="state_mean_minus_CT").reindex(columns=STATES) for m in IDS}
med = {m: P[m].median(axis=0, skipna=True) for m in IDS}
assert abs(med["PF4C-B4-007"]["MVP"] - 0.7746003) < 1e-6 and abs(med["PF4C-B4-029"]["MVP"] + 0.3756222) < 1e-6
assert abs(med["PF4C-B4-029"]["PAN"] - 0.6237169) < 1e-6 and abs(med["PF4C-FINAL-079"]["LE"] - 0.3692044) < 1e-6
cnt = lambda m, s, sign: (int((np.sign(P[m][s].dropna()) == sign).sum()), int(P[m][s].notna().sum()))  # noqa: E731
COUNTS = {"Liu MVP>CT": cnt("PF4C-B4-007", "MVP", 1), "Xiao MVP<CT": cnt("PF4C-B4-029", "MVP", -1),
          "Xiao PAN>CT": cnt("PF4C-B4-029", "PAN", 1), "Arim LE>CT": cnt("PF4C-FINAL-079", "LE", 1)}
assert COUNTS == {"Liu MVP>CT": (9, 9), "Xiao MVP<CT": (9, 9), "Xiao PAN>CT": (8, 8), "Arim LE>CT": (8, 8)}, COUNTS
ORDER = {m: coef[coef["signature_id"] == m]["gene"].str.upper().tolist() for m in IDS}
SIGN = {m: dict(zip(coef[coef["signature_id"] == m]["gene"].str.upper(), np.sign(coef[coef["signature_id"] == m]["coefficient"]))) for m in IDS}
for m in IDS:
    g = gene[gene["label"] == REP[m][0]]
    assert sorted(g["gene"]) == sorted(ORDER[m]) and len(ORDER[m]) == {"PF4C-B4-007": 5, "PF4C-B4-029": 4, "PF4C-FINAL-079": 14}[m]
    assert all(SIGN[m][r.gene] == r.coefficient_sign for r in g.itertuples())
assert [SIGN["PF4C-B4-007"][g] for g in ORDER["PF4C-B4-007"]] == [-1, 1, 1, 1, 1]
assert [SIGN["PF4C-B4-029"][g] for g in ORDER["PF4C-B4-029"]] == [1, -1, 1, -1]
log["asserts"].update({"N7": EXP, "N8": tps, "N9": {REP[m][0]: med[m].round(7).to_dict() for m in IDS}, "N10": COUNTS,
                       "N11": {REP[m][0]: [f"{g}{'+' if SIGN[m][g] > 0 else '−'}" for g in ORDER[m]] for m in IDS},
                       "tumour_means_recomputed_from_samples_max_diff": float((mm["re"] - mm["state_mean_minus_CT"]).abs().max())})

# ------------------------------------------------------------------ Fig 5D drawing (595.28 x 143 pt; page y 338-481)
W, H = 595.28, 143.0
c = Canvas(W, H)
c.text(12.88, 10.3, "D", 12, "bold", INK5)
c.text(26.5, 18.5, "Different anatomical states despite related biological associations", 10, "bold", INK5)
c.text(589, 18.5, "Ivy GAP · 122 wells · 10 tumours · unsigned membership scores", 5.8, color=SEC5, ha="right", style="italic")
DIV = LinearSegmentedColormap.from_list("ucmap_like", ["#2F5DA8", "#9DB5DB", "#F7F7F7", "#EBA39A", "#B2182B"])
norm = TwoSlopeNorm(0, -2, 2)
YL = (-0.8, 1.75)
COLX = [20, 211, 402]
ANN = {"PF4C-B4-007": [("MVP", "MVP > CT\n9/9 tumours", 1.08, "right")],
       "PF4C-B4-029": [("PAN", "PAN > CT 8/8", 1.30, "right"), ("MVP", "MVP < CT 9/9", -0.66, "right")],
       "PF4C-FINAL-079": [("LE", "LE > CT 8/8", 1.20, "left")]}
for k, m in enumerate(IDS):
    name, short, col = REP[m]; cx = COLX[k]; s = st[st["model_instance_id"] == m].iloc[0]
    c.text(cx, 29.5, name, 8, "bold", col)
    c.text(cx, 37.0, f"partial R² {s.partial_r2_ALL5_tumour_blocked:.3f} (tumour-blocked) · five-state BA {s.balanced_accuracy_ALL5_LOTO:.3f}",
           5.8, color=TEXT5)
    ax = c.ax(cx + 15, 46, cx + 80, 119)
    ax.axhline(0, color=REF5, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    for t, row in P[m].iterrows():
        ax.plot(range(5), row.to_numpy(float), color="#C3C8D0", lw=0.45, zorder=2)
        ax.scatter(range(5), row.to_numpy(float), s=3.2, color="#C3C8D0", lw=0, zorder=2)
    ax.plot(range(5), med[m].to_numpy(float), color=col, lw=1.25, zorder=4)
    ax.scatter(range(5), med[m].to_numpy(float), s=13, color=col, edgecolors="white", lw=0.4, zorder=5)
    for stt, txt, yy, ha in ANN[m]:
        xi = STATES.index(stt); ax.text(xi + (-0.18 if ha == "right" else 0.18), yy, txt, fontsize=5.8, color=col, ha=ha,
                                        va="center", fontweight="bold", linespacing=1.0, zorder=6)
    ax.set_xlim(-0.4, 4.4); ax.set_ylim(*YL); ax.set_xticks(range(5)); ax.set_xticklabels(STATES, fontsize=5.8)
    ax.set_yticks([-0.5, 0, 0.5, 1.0, 1.5]); ax.set_yticklabels(["−0.5", "0", "0.5", "1.0", "1.5"], fontsize=5.6)
    ax.tick_params(axis="x", length=1.8, pad=1.2); clean(ax)
    if k == 0:
        ax.set_ylabel("Score − tumour CT mean", fontsize=6.0, labelpad=1.5)
    # member-gene x state heat strip (genes as rows, panel A order)
    hx0, cw, rh, gy0 = cx + 120, 13.2, 6.0, 46.5
    hm = c.ax(hx0, gy0, hx0 + 5 * cw, gy0 + len(ORDER[m]) * rh)
    g = gene[gene["label"] == name].set_index("gene").reindex(ORDER[m])
    mat = g[[f"rel_CT_{s_}" for s_ in STATES]].to_numpy(float).clip(-2, 2)
    hm.pcolormesh(np.arange(6), np.arange(len(ORDER[m]) + 1), mat, cmap=DIV, norm=norm, edgecolors="white", linewidth=0.35)
    hm.set_ylim(len(ORDER[m]), 0); hm.set_xlim(0, 5); hm.axis("off")
    for j, s_ in enumerate(STATES):
        c.text(hx0 + (j + 0.5) * cw, gy0 - 1.8, s_, 5.4, color=TEXT5, ha="center")
    for i, gg in enumerate(ORDER[m]):
        sg = "+" if SIGN[m][gg] > 0 else "−"
        c.text(hx0 - 2, gy0 + (i + 0.5) * rh + 2.0, f"{gg} {sg}", 5.6, color=TEXT5 if sg == "+" else SEC5, ha="right",
               style="normal" if sg == "+" else "italic")
# colour bar + legend notes in the white space under the short Liu and Xiao strips
cbx, cby = COLX[0] + 120, 86
cb = c.ax(cbx, cby, cbx + 66, cby + 3.6)
cb.imshow(np.linspace(-2, 2, 200)[None, :], aspect="auto", cmap=DIV, norm=norm, extent=[-2, 2, 0, 1]); cb.set_yticks([])
cb.set_xticks([-2, 0, 2]); cb.set_xticklabels(["−2", "0", "2"], fontsize=5.4); cb.tick_params(length=1.5, pad=0.8)
for sp in cb.spines.values():
    sp.set_linewidth(0.3)
c.text(cbx, cby + 15.5, "gene z − tumour CT", 5.6, color=SEC5)
c.text(cbx, cby + 21.8, "(median; capped ±2)", 5.6, color=SEC5)
c.text(cbx, cby + 29.5, "+/−: coefficient sign", 5.6, color=SEC5)
nx = COLX[1] + 92
for i, line in enumerate(["Scores are unsigned membership", "means: SPOCK1 and CRLF1 enter", "positively although the source",
                          "predictor weights them negatively;", "the signed predictor's profile",
                          "differs (Supplementary Table 5)."]):
    c.text(nx, 82 + i * 6.6, line, 5.6, color=SEC5)
c.text(COLX[0] + 15, 140, "Grey: individual tumours (8–10 per state); colour: median over tumours. LE, leading edge; IT, infiltrating "
       "tumour; CT, cellular tumour; PAN, pseudopalisading; MVP, microvascular proliferation.", 5.6, color=SEC5)
pd5 = c.save("Fig5D_ivy_anatomical_profiles")

# ------------------------------------------------------------------ Fig 5G data + assertions (N12-N15)
f_w = F5SRC / "FG_worlds.tsv"; f_med = R4CLOSE / "03_null_geometry/DS03_BLOCKED_VS_NAIVE_REVERSAL_BY_SIGNATURE_Q.tsv"
f_null = R4 / "06_kmatched_null/DS03_FORMAL_NULL_LEDGER.tsv"; f_hr = SDC / "SD_Fig5G_act4_counts_999_vs_high_resolution.tsv"
for f in (f_w, f_med, f_null, f_hr):
    log["inputs"][str(f)] = sha(f)
w = pd.read_csv(f_w, sep="\t"); w = w[w["q"] == 1.0]
mdn = pd.read_csv(f_med, sep="\t"); mdn = mdn[mdn["q"].astype(float) == 1.0].set_index("signature_id")
nl = pd.read_csv(f_null, sep="\t"); nl = nl[nl["q"].astype(float) == 1.0]
hr = pd.read_csv(f_hr, sep="\t"); hr = hr[hr["q"] == 1.0].set_index("signature_id")
G = {}
for m in IDS:
    x = w[w["signature_id"] == m].sort_values("replicate_id"); assert len(x) == 20
    nd, bd = int((x["naive_p"] <= 0.05).sum()), int((x["blocked_p"] <= 0.05).sum())
    assert nd == int(x["NAIVE_DETECTED"].sum()) and bd == int(x["BLOCKED_DETECTED"].sum())
    z = nl[nl["signature_id"] == m]; assert len(z) == 100
    G[m] = dict(naive=x["naive_p"].to_numpy(float), blocked=x["blocked_p"].to_numpy(float), nd=nd, bd=bd,
                nmed=float(mdn.loc[m, "median_naive_p"]), bmed=float(mdn.loc[m, "median_blocked_p"]),
                tn=int(z["naive_rejected"].sum()), tb=int(z["blocked_rejected"].sum()), distinct_T=int(x["T_ALT"].nunique()))
    assert int(hr.loc[m, "naive_999"]) == int(hr.loc[m, "naive_200k"]) == nd and int(hr.loc[m, "blocked_999"]) == int(hr.loc[m, "blocked_exact"]) == bd
assert [(G[m]["nd"], G[m]["bd"]) for m in IDS] == [(0, 20), (0, 0), (20, 0)]
assert [(G[m]["nmed"], G[m]["bmed"]) for m in IDS] == [(0.1745, 0.003), (0.2915, 0.1065), (0.0295, 0.24)]
assert [(G[m]["tn"], G[m]["tb"]) for m in IDS] == [(0, 5), (0, 8), (55, 3)]
log["asserts"].update({"N12-N15": {REP[m][0]: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in G[m].items()} for m in IDS}})

# ------------------------------------------------------------------ Fig 5G drawing (595.28 x 143 pt; page y 699-842)
c = Canvas(W, H)
c.text(9.85, 12.0, "G", 12, "bold", INK5)
c.text(26.5, 20.3, "Reference-dependent detection and null calibration", 10, "bold", INK5)
c.text(589, 20.3, "GBM-Space · q100 planted alternatives · donor-blocked vs naive reference · 999-draw Monte Carlo P", 5.8,
       color=SEC5, ha="right", style="italic")
COLX = [20, 211, 402]
for k, m in enumerate(IDS):
    name, short, col = REP[m]; cx = COLX[k]; g = G[m]
    c.text(cx, 33.5, name, 8, "bold", col)
    import pymupdf as _pm
    nw = _pm.Font(fontfile="/System/Library/Fonts/Supplemental/Arial Bold.ttf").text_length(name, fontsize=8)
    c.text(cx + nw + 6, 33.5, f"{g['nd']}/20 → {g['bd']}/20", 8, "bold", TEXT5)
    c.text(cx, 41.5, f"median P {g['nmed']:g} → {g['bmed']:g}", 5.8, color=SEC5)
    ax = c.ax(cx + 20, 47, cx + 88, 121)
    ax.set_yscale("log"); ax.set_ylim(7e-4, 1.3); ax.set_xlim(-0.3, 1.3)
    ax.axhline(0.05, color="#8A9099", lw=0.55, ls=(0, (3, 2)), zorder=1)
    for a, b in zip(g["naive"], g["blocked"]):
        ax.plot([0, 1], [a, b], color=col, lw=0.6, alpha=0.38, zorder=2)
    jit = (np.arange(20) % 5 - 2) * 0.018
    ax.scatter(0 + jit, g["naive"], s=6, color=col, edgecolors="white", lw=0.25, zorder=3)
    ax.scatter(1 - jit, g["blocked"], s=6, color=col, edgecolors="white", lw=0.25, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Naive", "Donor-blocked"], fontsize=5.8)
    ax.set_yticks([1e-3, 1e-2, 0.05, 1e-1, 1]); ax.set_yticklabels(["0.001", "0.01", "0.05", "0.1", "1"], fontsize=5.6)
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator()); ax.tick_params(axis="x", length=0, pad=2)
    clean(ax)
    if k == 0:
        ax.set_ylabel("P value (log scale)", fontsize=6.0, labelpad=1.5)
    # true-null calibration: counts in 100 null worlds at q100
    bx0, bx1 = cx + 126, cx + 172
    c.text(cx + 98, 55, "True nulls rejected", 6.0, "bold", TEXT5)
    c.text(cx + 98, 62, "q100, 100 null worlds", 5.6, color=SEC5)
    bar = c.ax(bx0, 66, bx1, 92); bar.set_xlim(0, 0.6); bar.set_ylim(2, 0)
    for j, (lab, v, fc) in enumerate([("naive", g["tn"], col), ("blocked", g["tb"], "#6B7280")]):
        bar.add_patch(Rectangle((0, j + 0.22), v / 100, 0.56, facecolor=fc, edgecolor="none", alpha=0.85 if j == 0 else 0.7))
        c.text(cx + 98, 66 + (j + 0.5) * 13 + 2.0, lab, 5.8, color=TEXT5)
        bar.text(max(min(v / 100, 0.6), 0.05) + 0.02, j + 0.5, f"{v}/100", fontsize=5.8, va="center", ha="left", color=TEXT5,
                 fontweight="bold" if v > 10 else "normal")
    bar.axvline(0.05, color="#8A9099", lw=0.55, ls=(0, (3, 2)))
    bar.set_yticks([]); bar.set_xticks([0, 0.3, 0.6]); bar.set_xticklabels(["0", "30", "60%"], fontsize=5.4)
    bar.tick_params(axis="x", length=1.5, pad=0.8); clean(bar, left=False)
c.text(COLX[0] + 98, 108, "Lines: the 20 q100 realizations of", 5.6, color=SEC5)
c.text(COLX[0] + 98, 114.6, "one planted alternative (near-", 5.6, color=SEC5)
c.text(COLX[0] + 98, 121.2, "replicates; 14–15 distinct T)", 5.6, color=SEC5)
c.text(COLX[0] + 20, 140.6, "Detection: P ≤ 0.05. Decisions for these three representations are unchanged with exact donor-blocked P and "
       "200,000-draw naive P; dashed line, α = 0.05.", 5.6, color=SEC5)
assert sorted(G[m]["distinct_T"] for m in IDS) == [14, 14, 15]
pg5 = c.save("Fig5G_reference_dependent_detection")
log["outputs"] = {str(p): sha(p) for p in [pd5, pd5.with_suffix(".svg"), pg5, pg5.with_suffix(".svg")]}
log_json("build_fig5_panels.json", log)
print("Figure 5 panels built:", pd5, pg5)
