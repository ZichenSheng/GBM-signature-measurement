#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata, spearmanr, ttest_ind

from common import *


def fit_logistic(x, y):
    def objective(b):
        eta = b[0] + b[1] * x
        loss = np.sum(np.logaddexp(0, eta) - y * eta) + .5 * b[1] ** 2
        grad = np.array([np.sum(expit(eta)-y), np.sum((expit(eta)-y)*x)+b[1]])
        return loss, grad
    res = minimize(lambda b: objective(b)[0], np.zeros(2), jac=lambda b: objective(b)[1], method="BFGS", options={"gtol": 1e-10, "maxiter": 1000})
    return res.x


def auc_score(y, p):
    pos = y == 1; n1 = pos.sum(); n0 = (~pos).sum()
    if n1 == 0 or n0 == 0: return np.nan
    ranks = rankdata(p)
    return float((ranks[pos].sum() - n1*(n1+1)/2) / (n1*n0))


def grouped_logistic_cv(x, y, patient):
    pats = np.array(sorted(np.unique(patient)))
    folds = [set(v) for v in np.array_split(pats, 5)]
    prob = np.full(len(x), np.nan)
    for fold in folds:
        test = np.array([p in fold for p in patient]); train = ~test
        mu = x[train].mean(); sd = x[train].std(ddof=1)
        if not np.isfinite(sd) or sd <= EPSILON_SD: return np.nan, np.nan, np.nan
        b = fit_logistic((x[train]-mu)/sd, y[train])
        prob[test] = expit(b[0] + b[1]*(x[test]-mu)/sd)
    pred = prob >= .5
    sens = np.mean(pred[y == 1] == 1); spec = np.mean(pred[y == 0] == 0)
    bal = (sens+spec)/2
    loss = float(-np.mean(y*np.log(np.clip(prob,1e-15,1)) + (1-y)*np.log(np.clip(1-prob,1e-15,1))))
    return float(bal), auc_score(y, prob), loss


def welch(a, b):
    if len(a) < 2 or len(b) < 2: return np.nan, np.nan
    return float(np.mean(a)-np.mean(b)), float(ttest_ind(a,b,equal_var=False).pvalue)


def main():
    gate = json.loads((OUT / "00_governance/00_PREFLIGHT_STATUS.json").read_text())
    if gate["STOP_CORE_ANALYSIS"]: raise SystemExit("STOP_PREFLIGHT")
    manifest, genes, cw, gene_map = read_primary()
    meta = pd.read_csv(GEOMX / "metadata.csv")
    expr = pd.read_csv(GEOMX / "normalized.txt", sep="\t", index_col=0)
    if not expr.index.is_unique or not expr.columns.is_unique: raise SystemExit("STOP_GEOMX_EXPRESSION_IDENTITY")
    available_cols = set(expr.columns)
    meta["normalized_expression_available"] = meta["Unnamed: 0"].isin(available_cols)
    meta["patient_id"] = "ACTA-P" + meta.Patient_No.astype(int).astype(str).str.zfill(2)
    meta["timepoint"] = meta.tumor_setting.str.upper().map({"PRIMARY":"PRIMARY", "RECURRENCE":"RECURRENT"})
    meta["compartment"] = meta.SegmentLabel.str.upper().map({"SOX2":"SOX2", "IBA1":"IBA1"})
    meta["patient_treatment_group"] = np.where(meta.trial_setting.eq("Nivolumab"), "NIVOLUMAB", "CONTROL")
    meta["exposure_duration_class"] = np.where(
        meta.trial_setting.ne("Nivolumab"), "NONE",
        np.where(meta.timepoint.eq("PRIMARY"), "PRE_EXPOSURE",
                 np.where(meta.trial_setting_armB.astype(str).str.lower().eq("true"), "LONG_TERM", "SINGLE_DOSE_NEOADJUVANT")))
    meta["nivolumab_exposure"] = np.where(
        meta.patient_treatment_group.eq("CONTROL"), "NEVER_EXPOSED", meta.exposure_duration_class)
    meta["inclusion_status"] = np.where(meta.IDH_status.eq("IDH_WT") & meta.normalized_expression_available, "INCLUDED", "EXCLUDED")
    meta["exclusion_reason"] = np.where(meta.IDH_status.ne("IDH_WT"), "IDH_NOT_WILDTYPE", np.where(~meta.normalized_expression_available, "NORMALIZED_EXPRESSION_UNAVAILABLE", ""))
    sample_auth = pd.DataFrame({
        "aoi_id": meta["Unnamed: 0"], "patient_id": meta.patient_id, "primary_recurrent": meta.timepoint,
        "nivolumab_exposure": meta.nivolumab_exposure, "patient_treatment_group": meta.patient_treatment_group,
        "exposure_duration_class": meta.exposure_duration_class,
        "compartment": meta.compartment, "core_id": meta.TMA_coreID, "ROI": meta.ROILabel, "AOI": meta["Unnamed: 0"],
        "slide": meta.SlideName, "batch": meta.batch1, "normalized_expression_available": meta.normalized_expression_available,
        "IDH_status": meta.IDH_status, "QCFlags": meta.QCFlags, "inclusion_status": meta.inclusion_status, "exclusion_reason": meta.exclusion_reason,
    })
    write_tsv(sample_auth, OUT / "03_wp3_acta_geomx/ACTA2025_GEOMX_SAMPLE_AUTHORITY.tsv")
    included = meta[meta.inclusion_status.eq("INCLUDED")].copy().set_index("Unnamed: 0")
    if included.patient_id.nunique() != 26: raise SystemExit("STOP_GEOMX_IDHWT_PATIENT_IDENTITY")
    if set(included.index) - available_cols: raise SystemExit("STOP_GEOMX_AOI_CROSSWALK")

    external = pd.DataFrame([
        {"source": "Zenodo", "record": "16839828", "DOI": "10.5281/zenodo.16839828", "version": "v1", "retrieval_date": DATE, "license": "CC-BY-4.0", "file": p.name,
         "size_bytes": p.stat().st_size, "MD5": md5(p), "SHA256": sha256(p), "URL": f"https://zenodo.org/records/16839828/files/{p.name}?download=1", "role": "PROCESSED_INPUT"}
        for p in [GEOMX/"normalized.txt.tar.gz", GEOMX/"metadata.csv.tar.gz", GEOMX/"sampleAnno.xlsx.tar.gz", GEOMX/"Hs_R_NGS_WTA_v1.0.pkc.tar.gz"]
    ])
    write_tsv(external, OUT / "03_wp3_acta_geomx/ACTA2025_EXTERNAL_DATA_AUTHORITY.tsv")

    # Joint platform normalization across all included IDH-WT AOIs.
    mat = expr[included.index].T.astype(float)
    sd = mat.std(axis=0, ddof=1)
    usable = sd.index[np.isfinite(sd) & sd.gt(0)]
    z = (mat[usable] - mat[usable].mean(axis=0)) / sd[usable]
    scores, eval_table = reconstruct_unsigned_scores(z, manifest, gene_map)
    eval_table["platform_gene_n"] = len(usable)
    eval_table["joint_normalization"] = "ALL_IDHWT_AOIS_SOX2_AND_IBA1_TOGETHER"
    write_tsv(eval_table, OUT / "03_wp3_acta_geomx/ACTA2025_PRIMARY201_EVALUABILITY.tsv")
    evaluable = eval_table.loc[eval_table.evaluable, "model_instance_id"].tolist()
    if not evaluable: raise SystemExit("STOP_GEOMX_NO_EVALUABLE_MODELS")

    aoi = included[["patient_id", "timepoint", "compartment", "patient_treatment_group", "nivolumab_exposure", "exposure_duration_class", "TMA_coreID", "ROILabel", "SlideName", "batch1"]].join(scores[evaluable])
    keys = ["patient_id", "timepoint", "compartment"]
    aoi_reset = aoi.reset_index()
    agg = aoi_reset.groupby(keys, sort=True)[evaluable].mean().reset_index()
    meta_agg = aoi_reset.groupby(keys, sort=True).agg(
        patient_treatment_group=("patient_treatment_group", "first"),
        exposure_context=("nivolumab_exposure", lambda x: ";".join(sorted(set(x)))),
        AOI_n=("TMA_coreID", "size"),
    ).reset_index()
    agg = agg.merge(meta_agg, on=keys)
    score_long = agg.melt(id_vars=keys+["patient_treatment_group", "exposure_context", "AOI_n"], value_vars=evaluable, var_name="model_instance_id", value_name="model_score")
    score_long["aggregation_rule"] = "MEAN_WITHIN_PATIENT_TIME_COMPARTMENT"
    write_tsv(score_long, OUT / "03_wp3_acta_geomx/ACTA2025_COMPARTMENT_SCORE_MATRIX.tsv")

    # G1: patient-grouped compartment identity.
    ident_rows = []
    y = (agg.compartment == "SOX2").astype(int).to_numpy()
    pats = agg.patient_id.to_numpy(str)
    for model in evaluable:
        bal, auc, loss = grouped_logistic_cv(agg[model].to_numpy(float), y, pats)
        ident_rows.append({"model_instance_id": model, "patient_n": len(np.unique(pats)), "observation_n": len(agg), "balanced_accuracy": bal, "AUROC_descriptive": auc,
                           "cross_entropy": loss, "positive_class": "SOX2", "CV_scheme": "5-fold patient-grouped", "AOI_as_independent_patient": False})
    ident = pd.DataFrame(ident_rows)
    write_tsv(ident, OUT / "03_wp3_acta_geomx/ACTA2025_COMPARTMENT_IDENTIFIABILITY.tsv")

    matched_rows = []
    rng = np.random.default_rng(SEED + 300)
    sox = agg[agg.compartment.eq("SOX2")].reset_index(drop=True)
    iba = agg[agg.compartment.eq("IBA1")].reset_index(drop=True)
    for model in evaluable:
        for _, r in sox.iterrows():
            pool = iba[iba.patient_id.ne(r.patient_id)]
            dist = np.abs(pool[model].to_numpy(float)-float(r[model]))
            j = int(np.argmin(dist)); nearest = pool.iloc[j]
            random_dist = rng.choice(dist, size=100, replace=True)
            rand_med = float(np.median(random_dist))
            matched_rows.append({"model_instance_id": model, "SOX2_patient_id": r.patient_id, "SOX2_timepoint": r.timepoint,
                                 "nearest_IBA1_patient_id": nearest.patient_id, "nearest_IBA1_timepoint": nearest.timepoint,
                                 "SOX2_score": r[model], "nearest_IBA1_score": nearest[model], "absolute_score_distance": dist[j],
                                 "random_cross_compartment_distance_median": rand_med, "nearest_to_random_distance_ratio": dist[j]/rand_med if rand_med > EPSILON_SD else np.nan,
                                 "different_patient_enforced": True, "random_draw_n": 100})
    matched = pd.DataFrame(matched_rows)
    write_tsv(matched, OUT / "03_wp3_acta_geomx/ACTA2025_MATCHED_SCORE_ANALYSIS.tsv")
    median_bal = float(ident.balanced_accuracy.median())
    median_ratio = float(matched.groupby("model_instance_id").nearest_to_random_distance_ratio.median().median())
    g1a = median_bal <= .65; g1b = median_ratio <= .25
    G1 = "SUPPORTED" if g1a and g1b else ("PARTIALLY_SUPPORTED" if g1a or g1b else "NOT_SUPPORTED")

    # G2: patient-level matched recurrence source substitution.
    sub_rows = []
    for model in evaluable:
        q = score_long[score_long.model_instance_id.eq(model)].pivot_table(index=["patient_id", "patient_treatment_group"], columns=["timepoint", "compartment"], values="model_score")
        required = [("PRIMARY","SOX2"),("PRIMARY","IBA1"),("RECURRENT","SOX2"),("RECURRENT","IBA1")]
        for idx, r in q.dropna(subset=required).iterrows():
            dt = r[("RECURRENT","SOX2")]-r[("PRIMARY","SOX2")]; di = r[("RECURRENT","IBA1")]-r[("PRIMARY","IBA1")]
            dm = (dt+di)/2; rot = abs(dt-di)
            sub_rows.append({"model_instance_id": model, "patient_id": idx[0], "patient_treatment_group": idx[1], "delta_tumor": dt, "delta_TAM": di,
                             "delta_mean": dm, "absolute_delta_mean": abs(dm), "source_rotation": rot, "source_rotation_minus_absolute_delta_mean": rot-abs(dm)})
    sub = pd.DataFrame(sub_rows)
    model_sub = sub.groupby("model_instance_id").agg(patient_n=("patient_id","nunique"), median_absolute_delta_mean=("absolute_delta_mean","median"),
                                                       median_source_rotation=("source_rotation","median"), median_rotation_minus_overall=("source_rotation_minus_absolute_delta_mean","median")).reset_index()
    rho_rows = []
    for model, q in sub.groupby("model_instance_id"):
        rho, p = spearmanr(q.absolute_delta_mean, q.source_rotation)
        rho_rows.append({"model_instance_id": model, "abs_overall_vs_source_rotation_spearman": rho, "spearman_p_descriptive": p})
    model_sub = model_sub.merge(pd.DataFrame(rho_rows), on="model_instance_id")
    sub = sub.merge(model_sub, on="model_instance_id", suffixes=("", "_model_summary"))
    write_tsv(sub, OUT / "03_wp3_acta_geomx/ACTA2025_SOURCE_SUBSTITUTION.tsv")
    sub_lo, sub_hi, _ = publication_bootstrap(model_sub.rename(columns={"median_rotation_minus_overall":"value"}), "value", manifest, 301)
    sub_med = float(model_sub.median_rotation_minus_overall.median())
    G2 = "SUPPORTED" if sub_med > 0 and sub_lo > 0 else ("PARTIALLY_SUPPORTED" if sub_med > 0 else "NOT_SUPPORTED")

    # G3: longitudinal shift relative to within-time AOI variance.
    ratio_rows = []
    for model in evaluable:
        for (patient, compartment), q in aoi.reset_index().groupby(["patient_id","compartment"]):
            p = q[q.timepoint.eq("PRIMARY")][model].dropna().to_numpy(float); r = q[q.timepoint.eq("RECURRENT")][model].dropna().to_numpy(float)
            if len(p) < 2 or len(r) < 2: continue
            pooled = np.sqrt(((len(p)-1)*np.var(p,ddof=1)+(len(r)-1)*np.var(r,ddof=1))/(len(p)+len(r)-2))
            delta = abs(np.mean(r)-np.mean(p))
            ratio_rows.append({"model_instance_id": model, "patient_id": patient, "compartment": compartment, "primary_AOI_n": len(p), "recurrent_AOI_n": len(r),
                               "absolute_longitudinal_delta": delta, "pooled_within_time_SD": pooled, "LONGITUDINAL_TO_SPATIAL_RATIO": delta/pooled if pooled > EPSILON_SD else np.nan,
                               "epsilon_rule": "NA_IF_POOLED_SD_LE_1E-8", "ratio_gt_1_descriptive": delta/pooled > 1 if pooled > EPSILON_SD else np.nan})
    ratio = pd.DataFrame(ratio_rows)
    write_tsv(ratio, OUT / "03_wp3_acta_geomx/ACTA2025_LONGITUDINAL_SPATIAL_RATIO.tsv")
    model_ratio = ratio.groupby("model_instance_id").LONGITUDINAL_TO_SPATIAL_RATIO.median().rename("value").reset_index()
    ratio_lo, ratio_hi, _ = publication_bootstrap(model_ratio, "value", manifest, 302)
    ratio_med = float(ratio.LONGITUDINAL_TO_SPATIAL_RATIO.median())
    G3 = "SUPPORTED" if ratio_med > 1 and ratio_lo > 1 else ("PARTIALLY_SUPPORTED" if ratio_med > 1 else "NOT_SUPPORTED")

    # G4: patient-level negative-treatment stress test.
    treatment_rows = []
    for model in evaluable:
        q = sub[sub.model_instance_id.eq(model)]
        treated = q[q.patient_treatment_group.eq("NIVOLUMAB")].delta_mean.to_numpy(float); control = q[q.patient_treatment_group.eq("CONTROL")].delta_mean.to_numpy(float)
        effect, p = welch(treated, control)
        row = {"model_instance_id": model, "treated_patient_n": len(treated), "untreated_patient_n": len(control), "delta_mean_treated_minus_untreated": effect, "p_value": p}
        for comp, col in [("SOX2","delta_tumor"),("IBA1","delta_TAM")]:
            e, pv = welch(q[q.patient_treatment_group.eq("NIVOLUMAB")][col].to_numpy(float), q[q.patient_treatment_group.eq("CONTROL")][col].to_numpy(float))
            row[f"{comp}_effect"] = e; row[f"{comp}_p_descriptive"] = pv
        treatment_rows.append(row)
    treatment = pd.DataFrame(treatment_rows)
    treatment["BH_FDR"] = bh(treatment.p_value)
    treatment["BH_pass"] = treatment.BH_FDR < .05
    treatment["cross_compartment_same_sign_nominal"] = (np.sign(treatment.SOX2_effect)==np.sign(treatment.IBA1_effect)) & (treatment.SOX2_p_descriptive < .05) & (treatment.IBA1_p_descriptive < .05)
    write_tsv(treatment, OUT / "03_wp3_acta_geomx/ACTA2025_NIVOLUMAB_STRESS_TEST.tsv")
    bh_fraction = float(treatment.BH_pass.mean()); concordant_fraction = float(treatment.cross_compartment_same_sign_nominal.mean())
    G4 = "SUPPORTED" if bh_fraction <= .05 and concordant_fraction <= .05 else "NOT_SUPPORTED"
    g4_label = "NO_WIDESPREAD_FALSE_TREATMENT_SIGNAL" if G4 == "SUPPORTED" else "POTENTIAL_SIGNATURE_OVERINTERPRETATION_RISK"

    claims = pd.DataFrame([
        {"claim_id":"G1", "claim":"Compartment non-identifiability", "status":G1, "metric_1":"median_balanced_accuracy", "value_1":median_bal, "metric_2":"median_nearest_random_distance_ratio", "value_2":median_ratio},
        {"claim_id":"G2", "claim":"Source substitution beneath stable scalar", "status":G2, "metric_1":"median_rotation_minus_absolute_overall_delta", "value_1":sub_med, "metric_2":"publication_bootstrap_CI_low", "value_2":sub_lo},
        {"claim_id":"G3", "claim":"Longitudinal shift relative to spatial variance", "status":G3, "metric_1":"median_longitudinal_to_spatial_ratio", "value_1":ratio_med, "metric_2":"publication_bootstrap_CI_low_model_median", "value_2":ratio_lo},
        {"claim_id":"G4", "claim":g4_label, "status":G4, "metric_1":"BH_pass_fraction", "value_1":bh_fraction, "metric_2":"cross_compartment_same_sign_nominal_fraction", "value_2":concordant_fraction},
    ])
    write_tsv(claims, OUT / "03_wp3_acta_geomx/WP3_CLAIM_MATRIX.tsv")
    WP3 = "SUPPORTED" if (G1 == "SUPPORTED" or G2 == "SUPPORTED") else ("PARTIALLY_SUPPORTED" if ("PARTIALLY_SUPPORTED" in [G1,G2]) else "NOT_SUPPORTED")
    write_json({"WP3_GEOMX_EXTERNAL_STRESS_TEST": WP3, "IDHWT_patient_n": int(included.patient_id.nunique()), "IDHWT_AOI_n": len(included), "Primary201_evaluable_n": len(evaluable),
                "G1": G1, "median_compartment_balanced_accuracy": median_bal, "median_nearest_random_ratio": median_ratio,
                "G2": G2, "median_source_rotation_minus_absolute_overall_delta": sub_med, "source_substitution_publication_bootstrap_CI": [sub_lo,sub_hi],
                "G3": G3, "median_longitudinal_to_spatial_ratio": ratio_med, "ratio_publication_bootstrap_CI": [ratio_lo,ratio_hi],
                "G4": G4, "G4_label": g4_label, "nivolumab_BH_pass_n": int(treatment.BH_pass.sum()), "nivolumab_BH_family_n": len(treatment)}, OUT / "03_wp3_acta_geomx/WP3_STATUS.json")


if __name__ == "__main__": main()
