#!/usr/bin/env python3
"""K04 - design check + source-data extraction (read-only on inputs): the actual 500 independent matched-panel
values behind the overlap-removal result (prior independent audit, ACT1 N1/N2 families), verified against that
audit's null_summary.tsv, plus per-representation programme-gene overlap for the membership schematic.
Writes tidy candidate source tables to ../source_data_candidates/. No resampling, no new statistic."""
import hashlib, json, sys, os
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
A = (Path(os.environ["GBM_VALIDATION_ROOT"]) / 'GBM_INDEPENDENT_SCIENTIFIC_AUDIT/logs/act1')
IT, SU, OV = A / "null_iterations_all.tsv.gz", A / "null_summary.tsv", A / "representation_ecology_overlap.tsv"
OUT = Path(os.environ.get("GBM_PANEL_OUT", "outputs/panel_source")).resolve(); OUT.mkdir(parents=True, exist_ok=True)
FAM = {"N1_full": "Full gene sets (201)", "N2_ruleA_all201": "Programme genes removed (201)",
       "N2_ruleC_le20pct": "<=20% of genes removed (160)", "N2_ruleD_nooverlap": "Never contained a programme gene (114)"}
END = {"absPC1_eco_CGGA325": "Ecological alignment, CGGA325", "absPC1_eco_CGGA693": "Ecological alignment, CGGA693",
       "structure_rho": "Cross-cohort structure"}
it = pd.read_csv(IT, sep="\t"); su = pd.read_csv(SU, sep="\t")
long = it[it.family.isin(FAM)].melt(id_vars=["iteration", "transform", "family", "rep_n"], value_vars=list(END), var_name="endpoint", value_name="null_value")
long["family_label"] = long.family.map(FAM); long["endpoint_label"] = long.endpoint.map(END)
long = long.sort_values(["transform", "family", "endpoint", "iteration"])
obs = su[su.family.isin(FAM) & su.endpoint.isin(END)].copy()
chk = []
for r in obs.itertuples():
    v = long[(long["transform"] == r.transform) & (long["family"] == r.family) & (long["endpoint"] == r.endpoint)]["null_value"].to_numpy()
    ge = int((v >= r.observed).sum())
    chk.append({"transform": r.transform, "family": r.family, "endpoint": r.endpoint, "n": len(v),
                "max_matches": bool(np.isclose(v.max(), r.null_max, rtol=0, atol=1e-12)),
                "median_matches": bool(np.isclose(np.median(v), r.null_median, rtol=0, atol=1e-12)),
                "n_ge_matches": ge == r.n_null_ge_obs, "p_plus1": (1 + ge) / (1 + len(v))})
C = pd.DataFrame(chk); assert C.n.eq(500).all() and C.max_matches.all() and C.median_matches.all() and C.n_ge_matches.all(), C
long.to_csv(OUT / "SD_Fig1D_independent_matched_null_500_panels.tsv.gz", sep="\t", index=False, float_format="%.10g")
obs["family_label"] = obs.family.map(FAM); obs["endpoint_label"] = obs.endpoint.map(END)
obs.to_csv(OUT / "SD_Fig1D_observed_and_null_summary.tsv", sep="\t", index=False, float_format="%.10g")
ov = pd.read_csv(OV, sep="\t")
ov["removal_class"] = pd.cut(ov.frac, [-1e-9, 1e-9, 0.2 + 1e-9, 0.5 - 1e-9, 1.0], labels=["none", "<=20%", "20-50%", ">=50%"])
ov.to_csv(OUT / "SD_Fig1D_representation_programme_overlap.tsv", sep="\t", index=False)
meta = {"inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (IT, SU, OV)}, "checks": chk,
        "overlap": {"representations": int(len(ov)), "with_any_programme_gene": int((ov.eco_shared > 0).sum()),
                    "membership_rows": int(ov.n.sum()), "programme_rows": int(ov.eco_shared.sum()),
                    "class_counts": ov.removal_class.value_counts().reindex(["none", "<=20%", "20-50%", ">=50%"]).astype(int).to_dict(),
                    "by_programme_reps_with_gene": {k: int((ov[k] > 0).sum()) for k in ["MES", "HYPOXIA", "MATRIX", "MYELOID"]}}}
(OUT / "K04_fig1_overlap_panel_checks.json").write_text(json.dumps(meta, indent=1) + "\n")
print(C.to_string(index=False)); print(json.dumps(meta["overlap"], indent=1))
print(obs[["transform", "family", "endpoint", "observed", "null_median", "null_max", "n_null_ge_obs"]].round(4).to_string(index=False))
