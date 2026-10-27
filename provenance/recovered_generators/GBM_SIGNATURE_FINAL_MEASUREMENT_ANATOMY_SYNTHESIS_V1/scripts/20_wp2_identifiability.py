#!/usr/bin/env python3
"""WP2: patient-stratified identifiability and many-to-one projection analyses."""

from __future__ import annotations

import os as _gbm_os
from pathlib import Path as _GBMPath
def _gbm_external(env, relative):
    return _GBMPath(_gbm_os.environ.get(env, str(_GBMPath.cwd() / "external" / env.lower()))) / relative


import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.preprocessing import SplineTransformer, StandardScaler


ROOT = Path(str(_gbm_external('GBM_SOURCE_ROOT', '')))
OUT = ROOT / "GBM_SIGNATURE_FINAL_MEASUREMENT_ANATOMY_SYNTHESIS_V1"
V1 = ROOT / "GBM_SIGNATURE_CONTEXT_STRESS_TEST_EXTENSION_V1_FINAL"
V5 = ROOT / "GBM_SIGNATURE_MULTICELLULAR_COMMUNITY_CONTEXT_EXTENSION_V5_FINAL"
SEED = 20260829
BOOT_N = 1000
MIN_COVERAGE = 0.80


def bh(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    ok = values.notna()
    p = values.loc[ok].to_numpy(float)
    if not len(p):
        return out
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    restored = np.empty_like(adj)
    restored[order] = np.minimum(adj, 1.0)
    out.loc[ok] = restored
    return out


def model_map() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    manifest = pd.read_csv(OUT / "01_authority_snapshot/PRIMARY201_MODEL_MANIFEST_FROZEN.tsv", sep="\t", dtype=str)
    genes = pd.read_csv(OUT / "01_authority_snapshot/PRIMARY201_GENE_LONG_FROZEN.tsv", sep="\t", dtype=str)
    cw = pd.read_csv(OUT / "01_authority_snapshot/PRIMARY201_GENE_CROSSWALK_FROZEN.tsv", sep="\t", dtype=str)
    q = genes.assign(source_scoring_feature_id=genes.scoring_feature_id.astype(str)).merge(
        cw[["source_scoring_feature_id", "canonical_gene_symbol"]], on="source_scoring_feature_id", how="left"
    )
    q["gene"] = q.canonical_gene_symbol.fillna(q.source_scoring_feature_id).astype(str).str.upper()
    return manifest, q.groupby("model_instance_id").gene.agg(lambda x: sorted(set(x))).to_dict()


def reconstruct_scores(expr: pd.DataFrame, genes: dict[str, list[str]], manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, coverage = {}, []
    cols = set(expr.columns)
    original = manifest.set_index("model_instance_id").original_feature_n.astype(float).astype(int).to_dict()
    for model in manifest.model_instance_id:
        present = sorted(set(genes[model]) & cols)
        cov = len(present) / original[model] if original[model] else 0
        ok = original[model] >= 3 and len(present) >= 3 and cov >= MIN_COVERAGE
        rows[model] = expr[present].mean(axis=1).to_numpy(float) if ok else np.full(len(expr), np.nan)
        coverage.append({"model_instance_id": model, "original_gene_n": original[model], "mapped_gene_n": len(present), "mapping_coverage": cov, "evaluable": ok})
    return pd.DataFrame(rows, index=expr.index), pd.DataFrame(coverage)


def grouped_linear_cv(s: np.ndarray, y: np.ndarray, groups: np.ndarray, spline: bool = False) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.full_like(y, np.nan, dtype=float)
    sse = np.full(len(s), np.nan)
    sst = np.full(len(s), np.nan)
    for patient in np.unique(groups):
        test = groups == patient
        train = ~test
        xtr, xte = s[train, None], s[test, None]
        if spline:
            transform = SplineTransformer(n_knots=4, degree=3, knots="quantile", include_bias=False)
            xtr = transform.fit_transform(xtr)
            xte = transform.transform(xte)
        fit = LinearRegression().fit(xtr, y[train])
        predictions[test] = fit.predict(xte)
        center = np.nanmean(y[train], axis=0)
        sse[test] = np.sum((y[test] - predictions[test]) ** 2, axis=1)
        sst[test] = np.sum((y[test] - center) ** 2, axis=1)
    r2 = 1 - np.nansum(sse) / np.nansum(sst)
    return float(r2), predictions, sse, sst


def cluster_bootstrap_global(contrib: pd.DataFrame, model_ids: list[str], patient_order: list[str]) -> tuple[float, float, float, float, np.ndarray]:
    sse = np.zeros((len(patient_order), len(model_ids)))
    sst = np.zeros_like(sse)
    for i, p in enumerate(patient_order):
        q = contrib[contrib.patient_id.eq(p)].set_index("model_instance_id")
        sse[i] = q.sse.reindex(model_ids).to_numpy(float)
        sst[i] = q.sst.reindex(model_ids).to_numpy(float)
    rng = np.random.default_rng(SEED + 101)
    med = np.full(BOOT_N, np.nan)
    for b in range(BOOT_N):
        idx = rng.choice(len(patient_order), size=len(patient_order), replace=True)
        r2 = 1 - np.sum(sse[idx], axis=0) / np.where(np.sum(sst[idx], axis=0) > 0, np.sum(sst[idx], axis=0), np.nan)
        med[b] = np.nanmedian(r2)
    return float(np.nanmedian(med)), float(np.nanquantile(med, .025)), float(np.nanquantile(med, .975)), float(np.mean(np.isfinite(med))), med


def prepare_gbmspace(manifest: pd.DataFrame, genes: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    z = pd.read_parquet(V1 / "04_model_scores/DS03/REQUIRED_GENE_Z_EXPRESSION.parquet")
    z = z[z.score_scope.eq("ALL_AUTHOR_ANNOTATED")].drop(columns="score_scope").set_index("score_row_id")
    z.columns = z.columns.astype(str).str.upper()
    scores, coverage = reconstruct_scores(z, genes, manifest)
    meta = pd.read_parquet(V1 / "04_model_scores/DS03/PRIMARY201_MODEL_SCORES.parquet")
    meta = meta[meta.score_scope.eq("ALL_AUTHOR_ANNOTATED")][["score_row_id", "sample_id", "site_id", "patient_id"]].drop_duplicates("score_row_id").set_index("score_row_id").loc[z.index]
    axes = pd.read_parquet(V1 / "04_model_scores/DS03/FROZEN_ECOLOGY_AXIS_SCORES.parquet")
    axes = axes[axes.score_scope.eq("ALL_AUTHOR_ANNOTATED")].pivot(index="score_row_id", columns="axis_id", values="axis_score").loc[z.index]
    axes = axes.reindex(columns=["MES", "HYPOXIA", "MATRIX", "MYELOID"])
    return scores, coverage, meta, axes.to_numpy(float)


def ecology_cv(scores: pd.DataFrame, coverage: pd.DataFrame, meta: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    groups = meta.patient_id.to_numpy(str)
    rows, contrib_rows = [], []
    evaluable = coverage.loc[coverage.evaluable, "model_instance_id"].tolist()
    for model in evaluable:
        s = scores[model].to_numpy(float)
        r2, pred, sse, sst = grouped_linear_cv(s, y, groups, spline=False)
        r2_spline, _, _, _ = grouped_linear_cv(s, y, groups, spline=True)
        axis_r2 = []
        for j in range(y.shape[1]):
            _, p1, e1, t1 = grouped_linear_cv(s, y[:, [j]], groups, spline=False)
            axis_r2.append(1 - np.nansum(e1) / np.nansum(t1))
        rows.append({
            "model_instance_id": model, "dataset_id": "DS03_GBMSPACE", "patient_n": len(np.unique(groups)),
            "sample_n": len(groups), "CV_multivariate_R2": r2,
            "RESIDUAL_BIOLOGY_FRACTION": 1 - r2,
            "CV_R2_MES": axis_r2[0], "CV_R2_hypoxia": axis_r2[1],
            "CV_R2_matrix": axis_r2[2], "CV_R2_myeloid": axis_r2[3],
            "CV_multivariate_R2_spline_sensitivity": r2_spline,
            "primary_model": "linear scalar score to four frozen axes",
            "CV_scheme": "leave-one-patient-out; patient-exclusive folds",
            "nonlinearity_sensitivity": "fixed 4-knot cubic spline; no endpoint-based selection",
        })
        for p in np.unique(groups):
            idx = groups == p
            contrib_rows.append({"patient_id": p, "model_instance_id": model, "sse": float(np.nansum(sse[idx])), "sst": float(np.nansum(sst[idx]))})
    table = pd.DataFrame(rows)
    contrib = pd.DataFrame(contrib_rows)
    med, lo, hi, valid, draws = cluster_bootstrap_global(contrib, evaluable, sorted(np.unique(groups)))
    global_summary = {
        "domain": "FOUR_FROZEN_ECOLOGY_AXES", "model_n": len(evaluable), "patient_n": len(np.unique(groups)),
        "median_CV_R2": float(table.CV_multivariate_R2.median()),
        "patient_bootstrap_median_CV_R2": med, "bootstrap_CI_low": lo, "bootstrap_CI_high": hi,
        "median_residual_fraction": float(table.RESIDUAL_BIOLOGY_FRACTION.median()),
        "residual_bootstrap_CI_low": float(1 - hi), "residual_bootstrap_CI_high": float(1 - lo),
        "bootstrap_n": BOOT_N, "bootstrap_valid_fraction": valid,
    }
    return table, contrib, global_summary


def composition_cv(scores: pd.DataFrame, coverage: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    audit = pd.read_csv(V1 / "03_processed_data/DS03_GBMSpace/site_pseudobulk/DS03_SITE_PSEUDOBULK_AUDIT.tsv", sep="\t")
    counts = audit.pivot(index=["site_id", "patient_id"], columns="score_scope", values="cell_n").reset_index()
    counts = counts.dropna(subset=["ALL_AUTHOR_ANNOTATED", "MALIGNANT_ONLY"])
    counts["nonmalignant"] = counts.ALL_AUTHOR_ANNOTATED - counts.MALIGNANT_ONLY
    # Frozen two-compartment author labels; 0.5 cell pseudocount is fixed for zero handling.
    comp = counts[["MALIGNANT_ONLY", "nonmalignant"]].to_numpy(float) + 0.5
    prop = comp / comp.sum(axis=1, keepdims=True)
    logp = np.log(prop)
    y = logp - logp.mean(axis=1, keepdims=True)
    row_ids = counts.site_id.astype(str) + "|ALL_AUTHOR_ANNOTATED"
    keep = row_ids.isin(scores.index)
    counts, y, row_ids = counts.loc[keep].reset_index(drop=True), y[keep], row_ids[keep]
    groups = counts.patient_id.to_numpy(str)
    rows, contrib_rows = [], []
    evaluable = coverage.loc[coverage.evaluable, "model_instance_id"].tolist()
    for model in evaluable:
        s = scores.loc[row_ids, model].to_numpy(float)
        r2, pred, sse, sst = grouped_linear_cv(s, y, groups)
        rows.append({
            "model_instance_id": model, "dataset_id": "DS03_GBMSPACE",
            "composition_definition": "frozen broad malignant vs nonmalignant author compartments",
            "transform": "two-part CLR", "zero_handling": "add 0.5 cell to each compartment",
            "patient_n": len(np.unique(groups)), "sample_n": len(groups),
            "CV_multivariate_R2": r2, "RESIDUAL_COMPOSITION_FRACTION": 1-r2,
            "CV_scheme": "leave-one-patient-out; patient-exclusive folds",
        })
        for p in np.unique(groups):
            idx = groups == p
            contrib_rows.append({"patient_id": p, "model_instance_id": model, "sse": float(np.nansum(sse[idx])), "sst": float(np.nansum(sst[idx]))})
    table = pd.DataFrame(rows)
    contrib = pd.DataFrame(contrib_rows)
    med, lo, hi, valid, draws = cluster_bootstrap_global(contrib, evaluable, sorted(np.unique(groups)))
    summary = {
        "domain": "BROAD_CELL_COMPOSITION", "model_n": len(evaluable), "patient_n": len(np.unique(groups)),
        "median_CV_R2": float(table.CV_multivariate_R2.median()), "bootstrap_CI_low": lo, "bootstrap_CI_high": hi,
        "median_residual_fraction": float(table.RESIDUAL_COMPOSITION_FRACTION.median()),
        "residual_bootstrap_CI_low": float(1-hi), "residual_bootstrap_CI_high": float(1-lo),
        "bootstrap_n": BOOT_N, "bootstrap_valid_fraction": valid,
    }
    return table, counts, summary


def community_prediction() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_parquet(V5 / "03_primary201_community_mapping/PRIMARY201_LIN_PATIENT_SECTION_COMMUNITY_SCORES.parquet")
    rows, pred_rows = [], []
    for model, q in data.groupby("model_instance_id"):
        q = q.dropna(subset=["model_score", "patient_id", "community"]).reset_index(drop=True)
        if q.patient_id.nunique() < 3 or q.community.nunique() < 2:
            continue
        groups = q.patient_id.to_numpy(str)
        y = q.community.to_numpy(str)
        s = q.model_score.to_numpy(float)
        pred = np.empty(len(q), dtype=object)
        prob = np.full((len(q), len(np.unique(y))), np.nan)
        classes = np.sort(np.unique(y))
        ok = True
        for patient in np.unique(groups):
            test = groups == patient
            train = ~test
            if len(np.unique(y[train])) != len(classes) or np.nanstd(s[train]) <= 0:
                ok = False
                break
            scaler = StandardScaler().fit(s[train, None])
            fit = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED).fit(scaler.transform(s[train, None]), y[train])
            pred[test] = fit.predict(scaler.transform(s[test, None]))
            p = fit.predict_proba(scaler.transform(s[test, None]))
            for j, c in enumerate(fit.classes_):
                prob[test, np.where(classes == c)[0][0]] = p[:, j]
        if not ok:
            continue
        bal = balanced_accuracy_score(y, pred)
        macro = f1_score(y, pred, average="macro")
        loss = log_loss(y, prob, labels=classes)
        rows.append({
            "model_instance_id": model, "dataset_id": "LIN2026_STUDY_GENERATED_VISIUM",
            "patient_n": len(np.unique(groups)), "sample_n": len(q), "community_n": len(classes),
            "balanced_accuracy": bal, "macro_F1": macro, "cross_entropy_log_loss": loss,
            "model": "multinomial logistic regression, scalar score only",
            "CV_scheme": "leave-one-patient-out; all sections/communities patient-exclusive",
        })
        for i, (_, r) in enumerate(q.reset_index(drop=True).iterrows()):
            pred_rows.append({"model_instance_id": model, "patient_id": r.patient_id, "section_id": r.section_id, "true_community": y[i], "predicted_community": pred[i], "correct": y[i] == pred[i]})
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def nearest_indices(values: np.ndarray, groups: np.ndarray, i: int, k: int, rng: np.random.Generator, mode: str, biology: np.ndarray | None = None) -> np.ndarray:
    pool = np.flatnonzero(groups != groups[i])
    k = min(k, len(pool))
    if mode == "RANDOM":
        return rng.choice(pool, size=k, replace=False)
    if mode == "ECOLOGY":
        dist = np.linalg.norm(biology[pool] - biology[i], axis=1)
    else:
        dist = np.abs(values[pool] - values[i])
    return pool[np.argsort(dist, kind="mergesort")[:k]]


def matched_neighborhood(scores: pd.DataFrame, meta: pd.DataFrame, axes: np.ndarray, composition_counts: pd.DataFrame, community_data: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 303)
    rows = []
    groups = meta.patient_id.to_numpy(str)
    # Standardized four-axis distance.
    bio = StandardScaler().fit_transform(axes)
    comp_map = composition_counts.set_index("site_id")
    comp = comp_map.loc[meta.site_id, ["MALIGNANT_ONLY", "nonmalignant"]].to_numpy(float) + .5
    comp = np.log(comp / comp.sum(axis=1, keepdims=True))
    comp -= comp.mean(axis=1, keepdims=True)
    for model in scores.columns[scores.notna().all(axis=0)]:
        s = scores[model].to_numpy(float)
        for k in (1, 3, 5):
            for i in range(len(s)):
                for mode in ("SIGNATURE_SCORE", "RANDOM", "ECOLOGY"):
                    idx = nearest_indices(s, groups, i, k, rng, mode, bio)
                    rows.append({
                        "domain": "GBMSPACE_ECOLOGY_AND_COMPOSITION", "model_instance_id": model,
                        "sample_id": meta.site_id.iloc[i], "patient_id": groups[i], "k": k,
                        "benchmark": mode,
                        "mean_signature_score_distance": float(np.mean(np.abs(s[idx] - s[i]))),
                        "mean_ecology_distance": float(np.mean(np.linalg.norm(bio[idx] - bio[i], axis=1))),
                        "mean_cell_composition_CLR_distance": float(np.mean(np.linalg.norm(comp[idx] - comp[i], axis=1))),
                        "community_mismatch_fraction": np.nan,
                    })
    # Frozen Lin communities: mismatch among score-nearest different-patient samples.
    for model, q in community_data.groupby("model_instance_id"):
        groups_c = q.patient_id.to_numpy(str)
        s = q.model_score.to_numpy(float)
        label = q.community.to_numpy(str)
        for k in (1, 3, 5):
            for i in range(len(q)):
                for mode in ("SIGNATURE_SCORE", "RANDOM"):
                    idx = nearest_indices(s, groups_c, i, k, rng, mode)
                    rows.append({
                        "domain": "LIN2026_FROZEN_COMMUNITY", "model_instance_id": model,
                        "sample_id": f"{q.section_id.iloc[i]}|{label[i]}", "patient_id": groups_c[i], "k": k,
                        "benchmark": mode, "mean_signature_score_distance": float(np.mean(np.abs(s[idx]-s[i]))),
                        "mean_ecology_distance": np.nan, "mean_cell_composition_CLR_distance": np.nan,
                        "community_mismatch_fraction": float(np.mean(label[idx] != label[i])),
                    })
    return pd.DataFrame(rows)


def main() -> None:
    gate = json.loads((OUT / "00_preflight/PRE_FLIGHT_STATUS.json").read_text())
    if gate["STOP_CORE_ANALYSIS"]:
        raise RuntimeError("STOP_CORE_ANALYSIS")
    manifest, genes = model_map()
    scores, coverage, meta, axes = prepare_gbmspace(manifest, genes)
    ecology, eco_contrib, eco_summary = ecology_cv(scores, coverage, meta, axes)
    composition, composition_counts, comp_summary = composition_cv(scores, coverage, meta)
    community, community_predictions = community_prediction()
    community_data = pd.read_parquet(V5 / "03_primary201_community_mapping/PRIMARY201_LIN_PATIENT_SECTION_COMMUNITY_SCORES.parquet").dropna(subset=["model_score", "patient_id", "community"])
    neighborhood = matched_neighborhood(scores, meta.reset_index(), axes, composition_counts, community_data)

    ecology.to_csv(OUT / "03_identifiability/MODEL_ECOLOGY_CV_R2.tsv", sep="\t", index=False, na_rep="NA")
    composition.to_csv(OUT / "03_identifiability/MODEL_CELL_COMPOSITION_CV_R2.tsv", sep="\t", index=False, na_rep="NA")
    community.to_csv(OUT / "03_identifiability/MODEL_COMMUNITY_PREDICTION.tsv", sep="\t", index=False, na_rep="NA")
    neighborhood.to_csv(OUT / "03_identifiability/MATCHED_SCORE_NEIGHBORHOOD.tsv", sep="\t", index=False, na_rep="NA")
    community_predictions.to_parquet(OUT / "03_identifiability/COMMUNITY_CV_PREDICTIONS.parquet", index=False)
    coverage.to_csv(OUT / "03_identifiability/IDENTIFIABILITY_MODEL_COVERAGE.tsv", sep="\t", index=False)

    continuous = ecology[["model_instance_id", "CV_multivariate_R2", "RESIDUAL_BIOLOGY_FRACTION", "CV_multivariate_R2_spline_sensitivity"]].merge(
        composition[["model_instance_id", "CV_multivariate_R2", "RESIDUAL_COMPOSITION_FRACTION"]].rename(columns={"CV_multivariate_R2": "composition_CV_multivariate_R2"}),
        on="model_instance_id", how="outer",
    ).merge(
        community[["model_instance_id", "balanced_accuracy", "macro_F1", "cross_entropy_log_loss"]],
        on="model_instance_id", how="outer",
    )
    continuous.to_csv(OUT / "03_identifiability/MODEL_IDENTIFIABILITY_CONTINUOUS.tsv", sep="\t", index=False, na_rep="NA")

    global_rows = [eco_summary, comp_summary, {
        "domain": "FROZEN_MULTICELLULAR_COMMUNITY", "model_n": len(community),
        "patient_n": int(community.patient_n.max()) if len(community) else 0,
        "median_balanced_accuracy": float(community.balanced_accuracy.median()) if len(community) else np.nan,
        "median_macro_F1": float(community.macro_F1.median()) if len(community) else np.nan,
        "median_log_loss": float(community.cross_entropy_log_loss.median()) if len(community) else np.nan,
    }]
    global_table = pd.DataFrame(global_rows)
    global_table.to_csv(OUT / "03_identifiability/IDENTIFIABILITY_GLOBAL_SUMMARY.tsv", sep="\t", index=False, na_rep="NA")

    m2_supported = eco_summary["bootstrap_CI_low"] > 0 and eco_summary["median_residual_fraction"] > .50 and eco_summary["residual_bootstrap_CI_low"] > .50
    status = {
        "status": "PASS", "ecology_model_n": len(ecology), "composition_model_n": len(composition),
        "community_model_n": len(community), "ecology_patient_n": eco_summary["patient_n"],
        "median_ecology_CV_R2": eco_summary["median_CV_R2"],
        "ecology_CV_R2_bootstrap_CI": [eco_summary["bootstrap_CI_low"], eco_summary["bootstrap_CI_high"]],
        "median_residual_biology_fraction": eco_summary["median_residual_fraction"],
        "residual_fraction_bootstrap_CI": [eco_summary["residual_bootstrap_CI_low"], eco_summary["residual_bootstrap_CI_high"]],
        "M2_prespecified_gate_supported": bool(m2_supported), "patient_exclusive_folds": True,
        "region_level_random_split_used": False, "bootstrap_n": BOOT_N, "seed": SEED,
    }
    (OUT / "03_identifiability/WP2_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
