#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import *


def region_inputs(manifest, gene_map):
    z = pd.read_parquet(V1 / "04_model_scores/DS03/REQUIRED_GENE_Z_EXPRESSION.parquet")
    z = z[z.score_scope.eq("ALL_AUTHOR_ANNOTATED")].drop(columns="score_scope").set_index("score_row_id")
    scores, coverage = reconstruct_unsigned_scores(z, manifest, gene_map)
    meta = pd.read_parquet(V1 / "04_model_scores/DS03/PRIMARY201_MODEL_SCORES.parquet")
    meta = meta[meta.score_scope.eq("ALL_AUTHOR_ANNOTATED")][["score_row_id", "sample_id", "site_id", "patient_id"]].drop_duplicates("score_row_id").set_index("score_row_id").loc[z.index]
    axes = pd.read_parquet(V1 / "04_model_scores/DS03/FROZEN_ECOLOGY_AXIS_SCORES.parquet")
    axes = axes[axes.score_scope.eq("ALL_AUTHOR_ANNOTATED")].pivot(index="score_row_id", columns="axis_id", values="axis_score").loc[z.index].reindex(columns=AXES)
    return scores, coverage, meta, axes


def patient_bootstrap(contrib: pd.DataFrame, model_ids: list[str], patient_ids: list[str], seed_offset: int):
    sse = np.zeros((len(patient_ids), len(model_ids)))
    sst = np.zeros_like(sse)
    for i, p in enumerate(patient_ids):
        q = contrib[contrib.patient_id.eq(p)].set_index("model_instance_id")
        sse[i] = q.sse.reindex(model_ids).to_numpy(float)
        sst[i] = q.sst.reindex(model_ids).to_numpy(float)
    rng = np.random.default_rng(SEED + seed_offset)
    med = np.empty(BOOT_N)
    for b in range(BOOT_N):
        idx = rng.choice(len(patient_ids), size=len(patient_ids), replace=True)
        den = np.sum(sst[idx], axis=0)
        r2 = 1 - np.sum(sse[idx], axis=0) / np.where(den > 0, den, np.nan)
        med[b] = np.nanmedian(r2)
    return float(np.quantile(med, .025)), float(np.quantile(med, .975))


def repeated_fivefold_r2(s: np.ndarray, y: np.ndarray, folds: list[list[np.ndarray]]) -> np.ndarray:
    out = np.empty(len(folds))
    for r, split in enumerate(folds):
        sse = 0.0; sst = 0.0
        for test_idx in split:
            train = np.ones(len(s), dtype=bool); train[test_idx] = False
            X = np.column_stack([np.ones(train.sum()), s[train]])
            beta = np.linalg.lstsq(X, y[train], rcond=None)[0]
            pred = np.column_stack([np.ones(len(test_idx)), s[test_idx]]) @ beta
            center = y[train].mean(axis=0)
            sse += float(np.sum((y[test_idx] - pred) ** 2))
            sst += float(np.sum((y[test_idx] - center) ** 2))
        out[r] = 1 - sse / sst
    return out


def main():
    gate = json.loads((OUT / "00_governance/00_PREFLIGHT_STATUS.json").read_text())
    if gate["STOP_CORE_ANALYSIS"]: raise SystemExit("STOP_PREFLIGHT")
    manifest, genes, cw, gene_map = read_primary()
    scores, coverage, meta, axes = region_inputs(manifest, gene_map)
    evaluable = coverage.loc[coverage.evaluable, "model_instance_id"].tolist()

    # WP1A: exact reproduction of frozen M2 reference.
    region_rows = []
    groups = meta.patient_id.to_numpy(str)
    y = axes.to_numpy(float)
    for model in evaluable:
        r2, _, _ = grouped_linear_cv(scores[model].to_numpy(float), y, groups)
        region_rows.append({"model_instance_id": model, "R2_region_reproduced": r2, "residual_region_reproduced": 1-r2})
    region = pd.DataFrame(region_rows)
    frozen = pd.read_csv(MEAS / "03_identifiability/MODEL_ECOLOGY_CV_R2.tsv", sep="\t")[["model_instance_id", "CV_multivariate_R2", "RESIDUAL_BIOLOGY_FRACTION"]]
    region = region.merge(frozen, on="model_instance_id", how="outer")
    region["absolute_R2_difference"] = (region.R2_region_reproduced - region.CV_multivariate_R2).abs()
    region["reference_reproduction_status"] = np.where(region.absolute_R2_difference <= 1e-10, "PASS", "FAIL")
    write_tsv(region, OUT / "01_wp1_scale_identifiability/GBMSPACE_REGION_IDENTIFIABILITY_REFERENCE.tsv")
    if len(region) != 200 or (region.reference_reproduction_status != "PASS").any() or abs(region.R2_region_reproduced.median() - 0.17525034112478088) > 1e-10:
        raise SystemExit("STOP_WP1_REFERENCE_REPRODUCTION_FAILURE")

    # WP1B: mean aggregation on the inherited regional score scale.
    score_long = scores[evaluable].copy(); score_long["patient_id"] = groups
    score_cons = score_long.groupby("patient_id", sort=True)[evaluable].mean()
    axis_cons = axes.copy(); axis_cons["patient_id"] = groups
    axis_cons = axis_cons.groupby("patient_id", sort=True)[AXES].mean()
    consensus = axis_cons.join(score_cons).reset_index()
    consensus.insert(1, "region_n", meta.reset_index().groupby("patient_id").size().reindex(consensus.patient_id).to_numpy())
    write_tsv(consensus, OUT / "01_wp1_scale_identifiability/GBMSPACE_PATIENT_CONSENSUS_MATRIX.tsv")

    pgroups = consensus.patient_id.to_numpy(str)
    py = consensus[AXES].to_numpy(float)
    patient_rows, contrib_rows = [], []
    for model in evaluable:
        r2, sse, sst = grouped_linear_cv(consensus[model].to_numpy(float), py, pgroups)
        patient_rows.append({"model_instance_id": model, "patient_n": len(pgroups), "R2_region": float(region.set_index("model_instance_id").loc[model, "R2_region_reproduced"]),
                             "R2_patient_consensus": r2, "residual_region": 1-float(region.set_index("model_instance_id").loc[model, "R2_region_reproduced"]),
                             "residual_patient": 1-r2, "spatial_averaging_gain": r2-float(region.set_index("model_instance_id").loc[model, "R2_region_reproduced"]),
                             "model": "linear scalar to four frozen axes", "CV_scheme": "leave-one-patient-out", "negative_R2_retained": True})
        for i, p in enumerate(pgroups): contrib_rows.append({"patient_id": p, "model_instance_id": model, "sse": sse[i], "sst": sst[i]})
    patient = pd.DataFrame(patient_rows)
    write_tsv(patient, OUT / "01_wp1_scale_identifiability/GBMSPACE_PATIENT_IDENTIFIABILITY.tsv")
    pub_lo, pub_hi, _ = publication_bootstrap(patient, "residual_patient", manifest, 110)
    pat_lo, pat_hi = patient_bootstrap(pd.DataFrame(contrib_rows), evaluable, sorted(pgroups), 111)
    gbm_supported = patient.R2_patient_consensus.median() > 0 and patient.residual_patient.median() > .5 and pub_lo > .5

    # WP1C: frozen formal CARE baseline 56, 5-fold x100.
    membership = pd.read_csv(V2 / "00_governance/freeze/memberships/AS_CARE_LOPO56.tsv", sep="\t", dtype=str)
    cs = pd.read_parquet(V1 / "04_model_scores/DS01/PRIMARY201_MODEL_SCORES.parquet")
    cs = cs[(cs.score_scope == "ALL_AUTHOR_ANNOTATED") & (cs.timepoint == "T1") & cs.patient_id.isin(membership.patient_id)]
    cscore = cs.pivot(index="patient_id", columns="model_instance_id", values="model_score").sort_index()
    ca = pd.read_parquet(V1 / "04_model_scores/DS01/FROZEN_ECOLOGY_AXIS_SCORES.parquet")
    ca = ca[(ca.score_scope == "ALL_AUTHOR_ANNOTATED") & (ca.timepoint == "T1") & ca.patient_id.isin(membership.patient_id)]
    caxis = ca.pivot(index="patient_id", columns="axis_id", values="axis_score").reindex(cscore.index).reindex(columns=AXES)
    if len(cscore) != 56 or set(cscore.index) != set(membership.patient_id): raise SystemExit("STOP_CARE_BASELINE_IDENTITY")
    rng = np.random.default_rng(SEED + 120)
    folds = []
    for _ in range(100):
        order = rng.permutation(len(cscore)); folds.append([x.astype(int) for x in np.array_split(order, 5)])
    care_rows = []
    cy = caxis.to_numpy(float)
    care_eval = [m for m in manifest.model_instance_id if m in cscore and cscore[m].notna().all()]
    for model in care_eval:
        draws = repeated_fivefold_r2(cscore[model].to_numpy(float), cy, folds)
        care_rows.append({"model_instance_id": model, "patient_n": 56, "repeat_n": 100, "fold_n": 5,
                          "mean_CV_R2": float(np.mean(draws)), "median_repeat_CV_R2": float(np.median(draws)),
                          "repeat_CV_R2_q025": float(np.quantile(draws, .025)), "repeat_CV_R2_q975": float(np.quantile(draws, .975)),
                          "residual_biology": 1-float(np.mean(draws)), "negative_R2_retained": True})
    care = pd.DataFrame(care_rows)
    write_tsv(care, OUT / "01_wp1_scale_identifiability/CARE_BASELINE_IDENTIFIABILITY.tsv")
    care_lo, care_hi, _ = publication_bootstrap(care, "residual_biology", manifest, 121)
    care_supported = care.mean_CV_R2.median() > 0 and care.residual_biology.median() > .5 and care_lo > .5
    care_direction = care.mean_CV_R2.median() > 0 and care.residual_biology.median() > .5

    scale_summary = pd.DataFrame([
        {"dataset": "GBM-Space", "scale": "REGION", "patient_n": 12, "sample_n": 57, "model_n": len(region), "median_R2": region.R2_region_reproduced.median(), "median_residual": region.residual_region_reproduced.median(), "publication_bootstrap_CI_low_residual": np.nan, "publication_bootstrap_CI_high_residual": np.nan, "patient_bootstrap_CI_low_R2": np.nan, "patient_bootstrap_CI_high_R2": np.nan},
        {"dataset": "GBM-Space", "scale": "PATIENT_CONSENSUS", "patient_n": 12, "sample_n": 12, "model_n": len(patient), "median_R2": patient.R2_patient_consensus.median(), "median_residual": patient.residual_patient.median(), "publication_bootstrap_CI_low_residual": pub_lo, "publication_bootstrap_CI_high_residual": pub_hi, "patient_bootstrap_CI_low_R2": pat_lo, "patient_bootstrap_CI_high_R2": pat_hi},
        {"dataset": "CARE", "scale": "PATIENT_BASELINE", "patient_n": 56, "sample_n": 56, "model_n": len(care), "median_R2": care.mean_CV_R2.median(), "median_residual": care.residual_biology.median(), "publication_bootstrap_CI_low_residual": care_lo, "publication_bootstrap_CI_high_residual": care_hi, "patient_bootstrap_CI_low_R2": np.nan, "patient_bootstrap_CI_high_R2": np.nan},
    ])
    scale_summary["spatial_averaging_gain"] = np.where(scale_summary.scale.eq("PATIENT_CONSENSUS"), patient.R2_patient_consensus.median()-region.R2_region_reproduced.median(), np.nan)
    write_tsv(scale_summary, OUT / "01_wp1_scale_identifiability/GBMSPACE_SCALE_IDENTIFIABILITY_SUMMARY.tsv")
    wp1_status = "SUPPORTED" if gbm_supported and care_supported else ("PARTIALLY_SUPPORTED" if gbm_supported and care_direction else ("NOT_SUPPORTED" if not gbm_supported else "NOT_EVALUABLE"))
    cross = pd.DataFrame([
        {"dataset": "GBM-Space patient consensus", "patient_n": 12, "model_n": len(patient), "median_R2": patient.R2_patient_consensus.median(), "median_residual": patient.residual_patient.median(), "residual_CI_low": pub_lo, "residual_CI_high": pub_hi, "informative": patient.R2_patient_consensus.median() > 0, "strongly_nonidentifying": gbm_supported},
        {"dataset": "CARE baseline", "patient_n": 56, "model_n": len(care), "median_R2": care.mean_CV_R2.median(), "median_residual": care.residual_biology.median(), "residual_CI_low": care_lo, "residual_CI_high": care_hi, "informative": care.mean_CV_R2.median() > 0, "strongly_nonidentifying": care_supported},
    ])
    cross["WP1_FINAL_STATUS"] = wp1_status
    write_tsv(cross, OUT / "01_wp1_scale_identifiability/CROSS_DATASET_IDENTIFIABILITY.tsv")
    write_json({"WP1_SCALE_RESOLVED_IDENTIFIABILITY": wp1_status, "region_reference_R2": float(region.R2_region_reproduced.median()),
                "patient_consensus_R2": float(patient.R2_patient_consensus.median()), "spatial_averaging_gain": float(patient.R2_patient_consensus.median()-region.R2_region_reproduced.median()),
                "patient_consensus_residual": float(patient.residual_patient.median()), "patient_residual_publication_bootstrap_CI": [pub_lo, pub_hi],
                "CARE_R2": float(care.mean_CV_R2.median()), "CARE_residual": float(care.residual_biology.median()), "CARE_residual_publication_bootstrap_CI": [care_lo, care_hi]},
               OUT / "01_wp1_scale_identifiability/WP1_STATUS.json")


if __name__ == "__main__": main()
