#!/usr/bin/env python3
"""WP2: cellular-source mechanism and model-specific sampling sufficiency."""

from __future__ import annotations

import os as _gbm_os
from pathlib import Path as _GBMPath
def _gbm_external(env, relative):
    return _GBMPath(_gbm_os.environ.get(env, str(_GBMPath.cwd() / "external" / env.lower()))) / relative


import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import rankdata, spearmanr


ROOT = Path(str(_gbm_external('GBM_SOURCE_ROOT', '')))
V1 = ROOT / "GBM_SIGNATURE_CONTEXT_STRESS_TEST_EXTENSION_V1_FINAL"
V2 = ROOT / "GBM_SIGNATURE_CONTEXT_EXTENSION_DEEPENING_V2"
OUT = V2 / "03_sampling_source"
BOOT_REPS = 1000
EMP_REPS = 1000
SEED_PUB = 20260826
SEED_EMP = 20260830
MAX_ANALYTIC_K = 15
CLASS_ORDER = ["malignant-dominant", "TME-dominant", "mixed", "unstable"]
METRICS = [
    "icc_all", "icc_malignant", "delta_icc", "single_region_error_all",
    "delta_single_region_error", "regional_range_all", "delta_regional_range",
    "regional_cv_all", "delta_regional_cv",
]


def bh(p: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    vals = p.loc[ok].to_numpy(float)
    if len(vals) == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    adjusted = np.minimum.accumulate((ranked * len(vals) / np.arange(1, len(vals) + 1))[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out.loc[ok] = restored
    return out


def bootstrap_p(samples: np.ndarray) -> float:
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return np.nan
    return float(min(1.0, 2 * min((1 + np.sum(samples <= 0)) / (len(samples) + 1),
                                  (1 + np.sum(samples >= 0)) / (len(samples) + 1))))


def cluster_bootstrap_group_contrast(df: pd.DataFrame, metric: str, group: str,
                                     rng: np.random.Generator) -> tuple[float, float, float, float, float]:
    pubs = df["publication_id"].drop_duplicates().to_numpy()
    observed = float(df.loc[df["cellular_class"].eq(group), metric].median() -
                     df.loc[df["cellular_class"].eq("malignant-dominant"), metric].median())
    draws = np.full(BOOT_REPS, np.nan)
    pub_groups = {p: d for p, d in df.groupby("publication_id", sort=False)}
    for b in range(BOOT_REPS):
        sampled = rng.choice(pubs, size=len(pubs), replace=True)
        parts = [pub_groups[p] for p in sampled]
        boot = pd.concat(parts, ignore_index=True)
        a = boot.loc[boot["cellular_class"].eq(group), metric].dropna()
        m = boot.loc[boot["cellular_class"].eq("malignant-dominant"), metric].dropna()
        if len(a) and len(m):
            draws[b] = a.median() - m.median()
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    return observed, float(lo), float(hi), bootstrap_p(draws), float(np.mean(np.isfinite(draws)))


def cluster_bootstrap_spearman(df: pd.DataFrame, x: str, y: str,
                               rng: np.random.Generator) -> tuple[float, float, float, float, float]:
    q = df[["publication_id", x, y]].dropna()
    observed = float(spearmanr(q[x], q[y]).statistic) if len(q) >= 5 else np.nan
    pubs = q["publication_id"].drop_duplicates().to_numpy()
    groups = {p: d for p, d in q.groupby("publication_id", sort=False)}
    draws = np.full(BOOT_REPS, np.nan)
    for b in range(BOOT_REPS):
        sampled = rng.choice(pubs, size=len(pubs), replace=True)
        boot = pd.concat([groups[p] for p in sampled], ignore_index=True)
        if len(boot) >= 5 and boot[x].nunique() > 1 and boot[y].nunique() > 1:
            draws[b] = spearmanr(boot[x], boot[y]).statistic
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    return observed, float(lo), float(hi), bootstrap_p(draws), float(np.mean(np.isfinite(draws)))


def publication_weighted_group_mean(df: pd.DataFrame, metric: str, group: str) -> float:
    q = df[df["cellular_class"].eq(group)].groupby("publication_id")[metric].mean()
    return float(q.mean()) if len(q) else np.nan


def exact_gene_cluster(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, d in g.groupby("model_instance_id", sort=False):
        members = sorted(set(d["scoring_feature_id"].astype(str).str.strip().str.upper()))
        canonical = "|".join(members)
        rows.append({
            "model_instance_id": model,
            "exact_gene_set_hash": hashlib.sha256(canonical.encode()).hexdigest(),
            "exact_gene_member_n": len(members),
        })
    return pd.DataFrame(rows)


def parse_ecology(text: object, axis: str) -> float:
    if not isinstance(text, str):
        return np.nan
    match = re.search(rf"(?:^|;){axis}:spearman_rho=([^,;]+)", text)
    try:
        return float(match.group(1)) if match else np.nan
    except ValueError:
        return np.nan


def vectorized_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = a - np.nanmean(a, axis=0)
    b0 = b - np.nanmean(b, axis=0)
    den = np.sqrt(np.nansum(a0 * a0, axis=0) * np.nansum(b0 * b0, axis=0))
    return np.nansum(a0 * b0, axis=0) / np.where(den > 0, den, np.nan)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(V1 / "00_governance/frozen_parent_inputs/01_PRIMARY201_MODEL_MANIFEST_V2_1_1.tsv", sep="\t")
    genes = pd.read_csv(V1 / "00_governance/frozen_parent_inputs/02_PRIMARY201_GENE_LONG_V2_1_1.tsv", sep="\t")
    classification = pd.read_csv(V1 / "00_governance/frozen_parent_results/PRIMARY201_FULL_GENE_CLASSIFICATION.tsv", sep="\t")
    parent = pd.read_csv(V1 / "00_governance/parent_current_readonly_import/GBM_CROSS_MODULE_MODEL_EVIDENCE_MASTER.tsv", sep="\t")
    primary_ids = set(manifest["model_instance_id"])
    parent = parent[parent["model_instance_id"].isin(primary_ids)].copy()
    if len(parent) != 201 or len(classification) != 201:
        raise RuntimeError("STOP_PRIMARY201_IDENTITY_DRIFT")

    icc = pd.read_csv(V1 / "05_modules/module_A_sampling/MODEL_REGIONAL_ICC.tsv", sep="\t")
    ip = icc.pivot(index="model_instance_id", columns="score_scope")
    sensitivity = pd.read_csv(V1 / "05_modules/module_A_sampling/MODEL_SPECIFIC_SAMPLING_SENSITIVITY.tsv", sep="\t")
    sp = sensitivity.pivot(index="model_instance_id", columns="score_scope")
    exact = exact_gene_cluster(genes)

    model = manifest[["model_instance_id", "publication_id", "original_feature_n", "mapped_feature_n", "duplicate_cluster_id"]].copy()
    model = model.merge(classification[["model_instance_id", "final_class", "median_patient_delta", "class_retention"]], on="model_instance_id", validate="one_to_one")
    model = model.merge(exact, on="model_instance_id", validate="one_to_one")
    model = model.rename(columns={"final_class": "cellular_class", "mapped_feature_n": "gene_count", "median_patient_delta": "cell_source_delta"})
    model["icc_all"] = model["model_instance_id"].map(ip["icc"]["ALL_AUTHOR_ANNOTATED"])
    model["icc_malignant"] = model["model_instance_id"].map(ip["icc"]["MALIGNANT_ONLY"])
    model["delta_icc"] = model["icc_malignant"] - model["icc_all"]
    mapping = {
        "single_region_error": "mean_single_region_consensus_absolute_error",
        "regional_range": "maximum_within_patient_range",
        "regional_cv": "median_patient_regional_cv_abs_mean",
    }
    for short, source in mapping.items():
        model[f"{short}_all"] = model["model_instance_id"].map(sp[source]["ALL_AUTHOR_ANNOTATED"])
        model[f"{short}_malignant"] = model["model_instance_id"].map(sp[source]["MALIGNANT_ONLY"])
        model[f"delta_{short}"] = model[f"{short}_malignant"] - model[f"{short}_all"]

    # Frozen ecology loadings/associations, kept cohort-specific.
    assoc = parent.set_index("model_instance_id")
    for cohort in ["CGGA325", "CGGA693"]:
        col = f"{cohort}_model_ecology_association"
        for axis in ["MES", "HYPOXIA", "MATRIX", "MYELOID"]:
            model[f"{cohort}_{axis.lower()}_loading"] = model["model_instance_id"].map(
                assoc[col].map(lambda x, a=axis: parse_ecology(x, a))
            )
    model.to_csv(OUT / "SAMPLING_MODEL_METRICS.tsv", sep="\t", index=False)

    rows: list[dict] = []
    rng = np.random.default_rng(SEED_PUB)
    # Descriptive class summaries and publication-cluster bootstrap contrasts.
    for metric in ["icc_all", "icc_malignant", "delta_icc", "single_region_error_all", "delta_single_region_error"]:
        for group in CLASS_ORDER:
            q = model.loc[model["cellular_class"].eq(group), metric].dropna()
            rows.append({
                "analysis_type": "CLASS_SUMMARY", "metric": metric, "cellular_class": group,
                "comparison": "NA", "cohort": "GBM_SPACE", "ecology_axis": "NA",
                "model_n": len(q), "publication_n": model.loc[model["cellular_class"].eq(group), "publication_id"].nunique(),
                "estimate": float(q.median()), "estimate_type": "model_median",
                "ci_low": np.nan, "ci_high": np.nan, "bootstrap_p": np.nan,
                "BH_FDR_family_B": np.nan, "bootstrap_valid_fraction": np.nan,
                "publication_weighted_estimate": publication_weighted_group_mean(model, metric, group),
                "gene_set_cluster_sensitivity": np.nan,
                "interpretation": "continuous class summary",
            })
        for group in ["TME-dominant", "mixed", "unstable"]:
            est, lo, hi, pval, valid = cluster_bootstrap_group_contrast(model, metric, group, rng)
            # Exact gene-set cluster sensitivity: one mean per exact set then class medians.
            gc = model.groupby(["exact_gene_set_hash", "cellular_class"], as_index=False)[metric].mean()
            g1 = gc.loc[gc["cellular_class"].eq(group), metric]
            gm = gc.loc[gc["cellular_class"].eq("malignant-dominant"), metric]
            gs = float(g1.median() - gm.median()) if len(g1) and len(gm) else np.nan
            rows.append({
                "analysis_type": "PUBLICATION_CLUSTER_BOOTSTRAP_CLASS_CONTRAST",
                "metric": metric, "cellular_class": group,
                "comparison": f"{group}_minus_malignant-dominant", "cohort": "GBM_SPACE", "ecology_axis": "NA",
                "model_n": int(model["cellular_class"].isin([group, "malignant-dominant"]).sum()),
                "publication_n": int(model.loc[model["cellular_class"].isin([group, "malignant-dominant"]), "publication_id"].nunique()),
                "estimate": est, "estimate_type": "difference_in_model_medians",
                "ci_low": lo, "ci_high": hi, "bootstrap_p": pval,
                "BH_FDR_family_B": np.nan, "bootstrap_valid_fraction": valid,
                "publication_weighted_estimate": publication_weighted_group_mean(model, metric, group) - publication_weighted_group_mean(model, metric, "malignant-dominant"),
                "gene_set_cluster_sensitivity": gs,
                "interpretation": "Primary inference resamples publications; no ordinary model-level t-test",
            })

    # Continuous cellular attribution.
    for metric in ["icc_all", "icc_malignant", "delta_icc", "single_region_error_all", "delta_single_region_error"]:
        est, lo, hi, pval, valid = cluster_bootstrap_spearman(model, "cell_source_delta", metric, rng)
        pubmean = model.groupby("publication_id")[["cell_source_delta", metric]].mean().dropna()
        pwrho = spearmanr(pubmean["cell_source_delta"], pubmean[metric]).statistic if len(pubmean) >= 5 else np.nan
        rows.append({
            "analysis_type": "PUBLICATION_CLUSTER_BOOTSTRAP_CONTINUOUS_SOURCE",
            "metric": metric, "cellular_class": "CONTINUOUS", "comparison": "cell_source_delta_spearman",
            "cohort": "PARENT_7_STUDY", "ecology_axis": "NA", "model_n": model[["cell_source_delta", metric]].dropna().shape[0],
            "publication_n": model.loc[model[["cell_source_delta", metric]].notna().all(axis=1), "publication_id"].nunique(),
            "estimate": est, "estimate_type": "spearman_rho", "ci_low": lo, "ci_high": hi,
            "bootstrap_p": pval, "BH_FDR_family_B": np.nan, "bootstrap_valid_fraction": valid,
            "publication_weighted_estimate": pwrho, "gene_set_cluster_sensitivity": np.nan,
            "interpretation": "positive cell_source_delta denotes stronger malignant-minus-TME attribution",
        })

    # Ecology-mechanism associations must be evaluated in both CGGA cohorts.
    for axis in ["mes", "hypoxia", "matrix", "myeloid"]:
        axis_rows = []
        for metric in ["icc_all", "single_region_error_all", "delta_icc"]:
            cohort_estimates = {}
            for cohort in ["CGGA325", "CGGA693"]:
                x = f"{cohort}_{axis}_loading"
                est, lo, hi, pval, valid = cluster_bootstrap_spearman(model, x, metric, rng)
                cohort_estimates[cohort] = est
                row = {
                    "analysis_type": "PUBLICATION_CLUSTER_BOOTSTRAP_ECOLOGY_ASSOCIATION",
                    "metric": metric, "cellular_class": "NA", "comparison": f"{axis}_loading_spearman",
                    "cohort": cohort, "ecology_axis": axis.upper(),
                    "model_n": model[[x, metric]].dropna().shape[0],
                    "publication_n": model.loc[model[[x, metric]].notna().all(axis=1), "publication_id"].nunique(),
                    "estimate": est, "estimate_type": "spearman_rho", "ci_low": lo, "ci_high": hi,
                    "bootstrap_p": pval, "BH_FDR_family_B": np.nan, "bootstrap_valid_fraction": valid,
                    "publication_weighted_estimate": np.nan, "gene_set_cluster_sensitivity": np.nan,
                    "interpretation": "cross-cohort direction assessed jointly; no cohort cherry-picking",
                }
                rows.append(row)
            concordant = np.sign(cohort_estimates["CGGA325"]) == np.sign(cohort_estimates["CGGA693"])
            rows.append({
                "analysis_type": "ECOLOGY_CROSS_COHORT_DIRECTION",
                "metric": metric, "cellular_class": "NA", "comparison": f"{axis}_loading",
                "cohort": "CGGA325_AND_CGGA693", "ecology_axis": axis.upper(),
                "model_n": 201, "publication_n": model["publication_id"].nunique(),
                "estimate": np.nan, "estimate_type": "direction_status", "ci_low": np.nan, "ci_high": np.nan,
                "bootstrap_p": np.nan, "BH_FDR_family_B": np.nan, "bootstrap_valid_fraction": np.nan,
                "publication_weighted_estimate": np.nan, "gene_set_cluster_sensitivity": np.nan,
                "interpretation": "DIRECTION_CONCORDANT" if concordant else "DIRECTION_DISCORDANT",
            })

    assoc_rows = pd.DataFrame(rows)
    infer_mask = assoc_rows["analysis_type"].isin([
        "PUBLICATION_CLUSTER_BOOTSTRAP_CLASS_CONTRAST",
        "PUBLICATION_CLUSTER_BOOTSTRAP_CONTINUOUS_SOURCE",
        "PUBLICATION_CLUSTER_BOOTSTRAP_ECOLOGY_ASSOCIATION",
    ])
    assoc_rows.loc[infer_mask, "BH_FDR_family_B"] = bh(assoc_rows.loc[infer_mask, "bootstrap_p"])

    # Gene-count sensitivity with cluster-robust SE by publication.
    reg = model[["icc_all", "gene_count", "cellular_class", "publication_id"]].dropna().copy()
    reg["log_gene_count"] = np.log(reg["gene_count"].clip(lower=1))
    reg["cellular_class"] = pd.Categorical(reg["cellular_class"], categories=CLASS_ORDER)
    fit = smf.ols(
        "icc_all ~ log_gene_count + C(cellular_class, Treatment(reference='malignant-dominant'))",
        data=reg,
    ).fit()
    robust = fit.get_robustcov_results(cov_type="cluster", groups=reg["publication_id"])
    for term, est, se, pval, lo, hi in zip(robust.model.exog_names, robust.params, robust.bse, robust.pvalues,
                                           robust.conf_int()[:, 0], robust.conf_int()[:, 1]):
        assoc_rows.loc[len(assoc_rows)] = {
            "analysis_type": "GENE_COUNT_CLUSTER_ROBUST_SENSITIVITY",
            "metric": "icc_all", "cellular_class": "NA", "comparison": term,
            "cohort": "GBM_SPACE", "ecology_axis": "NA", "model_n": len(reg),
            "publication_n": reg["publication_id"].nunique(), "estimate": est,
            "estimate_type": "OLS_coefficient", "ci_low": lo, "ci_high": hi,
            "bootstrap_p": pval, "BH_FDR_family_B": np.nan, "bootstrap_valid_fraction": np.nan,
            "publication_weighted_estimate": np.nan, "gene_set_cluster_sensitivity": np.nan,
            "interpretation": "Secondary sensitivity only; publication-cluster robust SE",
        }
    assoc_rows.to_csv(OUT / "SAMPLING_CELL_SOURCE_ASSOCIATION.tsv", sep="\t", index=False)

    # Analytic reliability and empirical patient-level consensus recovery.
    icc_all = icc[icc["score_scope"].eq("ALL_AUTHOR_ANNOTATED")].set_index("model_instance_id")
    score = pd.read_parquet(V1 / "04_model_scores/DS03/PRIMARY201_MODEL_SCORES.parquet")
    score = score[score["score_scope"].eq("ALL_AUTHOR_ANNOTATED")]
    wide = score.pivot(index=["patient_id", "sample_id"], columns="model_instance_id", values="model_score")
    region_counts = wide.reset_index().groupby("patient_id")["sample_id"].nunique()
    empirical_max_k = int(max(k for k in range(1, MAX_ANALYTIC_K + 1) if (region_counts >= k).sum() >= 6))
    consensus = wide.groupby(level="patient_id").mean().reindex(columns=model["model_instance_id"])
    by_patient = {pid: wide.xs(pid, level="patient_id").reindex(columns=model["model_instance_id"]).to_numpy(float)
                  for pid in consensus.index}
    rng_emp = np.random.default_rng(SEED_EMP)
    empirical_by_k = {}
    for k in range(1, empirical_max_k + 1):
        eligible = [pid for pid in consensus.index if by_patient[pid].shape[0] >= k]
        truth = consensus.loc[eligible].to_numpy(float)
        pearson_draws = np.full((EMP_REPS, truth.shape[1]), np.nan)
        spearman_draws = np.full((EMP_REPS, truth.shape[1]), np.nan)
        truth_rank = rankdata(truth, axis=0)
        for b in range(EMP_REPS):
            estimate = np.vstack([
                by_patient[pid][rng_emp.choice(by_patient[pid].shape[0], size=k, replace=False)].mean(axis=0)
                for pid in eligible
            ])
            pearson_draws[b] = vectorized_corr(estimate, truth)
            spearman_draws[b] = vectorized_corr(rankdata(estimate, axis=0), truth_rank)
        empirical_by_k[k] = {
            "patient_n": len(eligible),
            "pearson_median": np.nanmedian(pearson_draws, axis=0),
            "pearson_q1": np.nanquantile(pearson_draws, .25, axis=0),
            "pearson_q3": np.nanquantile(pearson_draws, .75, axis=0),
            "spearman_median": np.nanmedian(spearman_draws, axis=0),
        }

    suff_rows, curve_rows = [], []
    for j, row in model.reset_index(drop=True).iterrows():
        mid = row["model_instance_id"]
        between = float(icc_all.loc[mid, "between_patient_variance"])
        within = float(icc_all.loc[mid, "within_patient_regional_variance"])
        analytic = {}
        for k in range(1, MAX_ANALYTIC_K + 1):
            denom = between + within / k
            rel = between / denom if denom > 0 else np.nan
            analytic[k] = rel
            emp = empirical_by_k.get(k)
            curve_rows.append({
                "model_instance_id": mid, "sampled_region_n": k,
                "analytic_expected_icc": rel,
                "empirical_patient_n": emp["patient_n"] if emp else np.nan,
                "empirical_consensus_pearson_median": emp["pearson_median"][j] if emp else np.nan,
                "empirical_consensus_pearson_q1": emp["pearson_q1"][j] if emp else np.nan,
                "empirical_consensus_pearson_q3": emp["pearson_q3"][j] if emp else np.nan,
                "empirical_consensus_spearman_median": emp["spearman_median"][j] if emp else np.nan,
                "empirical_resampling_reps": EMP_REPS if emp else 0,
                "empirical_seed": SEED_EMP if emp else np.nan,
            })
        k80_num = next((k for k, v in analytic.items() if np.isfinite(v) and v >= .80), np.nan)
        k90_num = next((k for k, v in analytic.items() if np.isfinite(v) and v >= .90), np.nan)
        suff_rows.append({
            "model_instance_id": mid, "publication_id": row["publication_id"],
            "cellular_class": row["cellular_class"], "gene_count": row["gene_count"],
            "between_patient_variance": between, "within_patient_regional_variance": within,
            "icc_k1": analytic[1], "analytic_max_observed_k": MAX_ANALYTIC_K,
            "biopsy_k80": str(int(k80_num)) if np.isfinite(k80_num) else "NOT_REACHED",
            "biopsy_k90": str(int(k90_num)) if np.isfinite(k90_num) else "NOT_REACHED",
            "biopsy_k80_numeric": k80_num, "biopsy_k90_numeric": k90_num,
            "empirical_validation_max_k": empirical_max_k,
            "empirical_k1_pearson_median": empirical_by_k[1]["pearson_median"][j],
            "empirical_k1_spearman_median": empirical_by_k[1]["spearman_median"][j],
            "label_boundary": "SAMPLING_SUFFICIENCY_ESTIMATE_NOT_CLINICAL_BIOPSY_RECOMMENDATION",
        })
    suff = pd.DataFrame(suff_rows)
    curves = pd.DataFrame(curve_rows)
    suff.to_csv(OUT / "MODEL_SAMPLING_SUFFICIENCY.tsv", sep="\t", index=False)
    curves.to_csv(OUT / "MODEL_SAMPLING_SUFFICIENCY_CURVES.tsv", sep="\t", index=False)
    validation = curves[curves["sampled_region_n"].le(empirical_max_k)].dropna(subset=["empirical_consensus_pearson_median"])
    val_rho = spearmanr(validation["analytic_expected_icc"], validation["empirical_consensus_pearson_median"]).statistic
    pd.DataFrame([{
        "comparison": "analytic_expected_icc_vs_empirical_consensus_pearson",
        "model_k_row_n": len(validation), "model_n": validation["model_instance_id"].nunique(),
        "k_min": 1, "k_max": empirical_max_k, "spearman_rho": val_rho,
        "empirical_reps": EMP_REPS, "seed": SEED_EMP,
        "interpretation": "Validation of sampling-sufficiency estimates; not a clinical biopsy recommendation",
    }]).to_csv(OUT / "SAMPLING_SUFFICIENCY_EMPIRICAL_VALIDATION.tsv", sep="\t", index=False)

    (OUT / "WP2_STATUS.json").write_text(json.dumps({
        "module": "WP2_SAMPLING_CELL_SOURCE", "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_n": len(model), "publication_n": model["publication_id"].nunique(),
        "class_counts": model["cellular_class"].value_counts().to_dict(),
        "publication_bootstrap_reps": BOOT_REPS, "empirical_resampling_reps": EMP_REPS,
        "analytic_k_max": MAX_ANALYTIC_K, "empirical_k_max": empirical_max_k,
        "ordinary_model_level_t_test_used": False, "result_driven_change": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
