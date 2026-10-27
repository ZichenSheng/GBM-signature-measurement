#!/usr/bin/env python3
"""K05 - source-data extraction for the proposed Fig. 5D (read-only on inputs). Frozen Ivy membership scores of the
three ACT5 representations -> per-sample table, per-tumour state means expressed relative to the same tumour's
cellular-tumour (CT) mean (CT is sampled in all 10 tumours), and descriptive within-tumour consistency counts.
Frozen model-level statistics (partial R2, BA, recall) are copied, not recomputed. Gene-level profiles come from K03."""
import hashlib, json, sys, os
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
IVY = Path(os.environ["GBM_SOURCE_ROOT"]) / "IVYGAP_PATHOLOGY_ANATOMY_EXTENSION_V1"
SC = IVY / "04_SCORES/IVYGAP_PRIMARY201_MODEL_SCORES.tsv.gz"
INF = IVY / "05_ANATOMICAL_INFORMATIVENESS/IVYGAP_ANATOMICAL_INFORMATIVENESS_MODEL_LEVEL.tsv"
BA = IVY / "06_ANATOMICAL_IDENTIFIABILITY/IVYGAP_ANATOMICAL_IDENTIFIABILITY_MODEL_LEVEL.tsv"
HERE = Path(os.environ.get("GBM_PANEL_OUT", "outputs/panel_source")).resolve(); HERE.mkdir(parents=True, exist_ok=True); OUT = HERE
REPS = {"PF4C-B4-007": "Liu EMT-5", "PF4C-B4-029": "Xiao EMT-4", "PF4C-FINAL-079": "Arimappamagan-14"}
ANAT = ["LE", "IT", "CT", "PAN", "MVP"]
S = pd.read_csv(SC, sep="\t", dtype={"sample_id": str, "tumor_id": str, "donor_id": str})
smp = S[["sample_id", "tumor_id", "donor_id", "anatomical_structure", *REPS]].melt(
    id_vars=["sample_id", "tumor_id", "donor_id", "anatomical_structure"], var_name="model_instance_id", value_name="frozen_membership_score")
smp["label"] = smp.model_instance_id.map(REPS)
ct = smp[smp.anatomical_structure == "CT"].groupby(["model_instance_id", "tumor_id"]).frozen_membership_score.mean().rename("tumour_CT_mean")
smp = smp.join(ct, on=["model_instance_id", "tumor_id"]); smp["score_minus_tumour_CT_mean"] = smp.frozen_membership_score - smp.tumour_CT_mean
smp.to_csv(OUT / "SD_Fig5D_ivy_sample_scores.tsv", sep="\t", index=False, float_format="%.10g")
tm = smp.groupby(["model_instance_id", "label", "tumor_id", "anatomical_structure"]).agg(
    n_samples=("sample_id", "size"), state_mean_minus_CT=("score_minus_tumour_CT_mean", "mean")).reset_index()
tm.to_csv(OUT / "SD_Fig5D_ivy_tumour_state_means.tsv", sep="\t", index=False, float_format="%.10g")
summ = tm.groupby(["model_instance_id", "label", "anatomical_structure"]).agg(
    tumours=("tumor_id", "nunique"), median_over_tumours=("state_mean_minus_CT", "median"),
    tumours_above_CT=("state_mean_minus_CT", lambda v: int((v > 1e-12).sum())),
    tumours_below_CT=("state_mean_minus_CT", lambda v: int((v < -1e-12).sum()))).reset_index()
summ["anatomical_structure"] = pd.Categorical(summ.anatomical_structure, ANAT); summ = summ.sort_values(["model_instance_id", "anatomical_structure"])
inf = pd.read_csv(INF, sep="\t"); ba = pd.read_csv(BA, sep="\t")
fz = []
for m, lab in REPS.items():
    i = inf[(inf.branch == "ALL5") & (inf.model_instance_id == m)].iloc[0]; b = ba[(ba.branch == "ALL5") & (ba.model_instance_id == m)].iloc[0]
    fz.append({"model_instance_id": m, "label": lab, "partial_r2_ALL5_tumour_blocked": i.partial_r2, "partial_r2_permutation_p": i.permutation_p,
               "partial_r2_bh_fdr": i.bh_fdr, "balanced_accuracy_ALL5_LOTO": b.balanced_accuracy, "ba_chance": b.chance_level,
               "ba_null_median": b.null_median, "ba_permutation_p": b.permutation_p, "ba_bh_fdr": b.bh_fdr,
               **{f"recall_{a}": b[f"recall_{a}"] for a in ANAT}, "sample_n": int(i.sample_n), "tumor_n": int(i.tumor_n)})
FZ = pd.DataFrame(fz)
summ.to_csv(OUT / "SD_Fig5D_ivy_state_profile_summary.tsv", sep="\t", index=False, float_format="%.6g")
FZ.to_csv(OUT / "SD_Fig5D_ivy_frozen_model_statistics.tsv", sep="\t", index=False, float_format="%.10g")
meta = {"inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (SC, INF, BA)},
        "definition": "per tumour: mean score of each state minus that tumour's CT mean; summary = median over tumours; counts are descriptive (no test)",
        "tumours_per_state": tm.groupby("anatomical_structure").tumor_id.nunique().reindex(ANAT).astype(int).to_dict()}
(HERE / "K05_fig5_ivy_panel_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
print(summ.to_string(index=False)); print(FZ.T.to_string()); print(json.dumps(meta["tumours_per_state"]))
