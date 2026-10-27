#!/usr/bin/env python3
"""Frozen Ivy GAP Q1-Q4 analysis. No model refitting or feature selection."""

from __future__ import annotations

import os as _gbm_os
from pathlib import Path as _GBMPath
def _gbm_external(env, relative):
    return _GBMPath(_gbm_os.environ.get(env, str(_GBMPath.cwd() / "external" / env.lower()))) / relative


import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(str(_gbm_external('GBM_SOURCE_ROOT', '')))
OUT = Path(_gbm_os.environ.get("GBM_IVY_OUT", "outputs/ivy"))
Z_PATH = Path(str(_gbm_external('GBM_DATA_ROOT', 'IvyGAP/PROCESSED/IVYGAP_ANATOMIC122_REQUIRED_GENE_Z.parquet')))
MODEL_PATH = ROOT / "GBM_SIGNATURE_FINAL_MEASUREMENT_ANATOMY_SYNTHESIS_V1/01_authority_snapshot/PRIMARY201_MODEL_MANIFEST_FROZEN.tsv"
SAMPLE_PATH = OUT / "02_SOURCE/IVYGAP_SAMPLE_MANIFEST.tsv"
EVAL_PATH = OUT / "03_EVALUABILITY/IVYGAP_MODEL_EVALUABILITY.tsv"
VALIDATION_PATH = OUT / "01_PROTOCOL/freeze/validation_report.json"

BRANCHES = {
    "ALL5": ["LE", "IT", "CT", "PAN", "MVP"],
    "CORE3": ["CT", "PAN", "MVP"],
    "PAN_MVP": ["PAN", "MVP"],
}
Q1_NPERM = 10_000
Q2_NPERM = 2_000
Q3_NBOOT = 10_000
PUB_NBOOT = 10_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    x = p[ok]
    if not len(x):
        return out
    order = np.argsort(x)
    ranked = x[order] * len(x) / np.arange(1, len(x) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty_like(x)
    tmp[order] = np.clip(ranked, 0, 1)
    out[ok] = tmp
    return out


def one_hot(values: np.ndarray, levels: list[str]) -> np.ndarray:
    return np.column_stack([(values == x).astype(float) for x in levels[1:]])


def orthonormal_basis(x: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if x.size == 0:
        return np.empty((x.shape[0], 0))
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    rank = int(np.sum(s > tol * max(x.shape) * (s[0] if len(s) else 1.0)))
    return u[:, :rank]


def permute_within_tumor(labels: np.ndarray, tumors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = labels.copy()
    for tumor in np.unique(tumors):
        idx = np.flatnonzero(tumors == tumor)
        out[idx] = rng.permutation(out[idx])
    return out


def q1_informativeness(y: np.ndarray, labels: np.ndarray, tumors: np.ndarray,
                       levels: list[str], rng: np.random.Generator) -> pd.DataFrame:
    tumor_levels = list(pd.unique(tumors))
    d = np.column_stack([np.ones(len(tumors)), one_hot(tumors, tumor_levels)])
    qd = orthonormal_basis(d)
    yres = y - qd @ (qd.T @ y)
    sse0 = np.sum(yres ** 2, axis=0)

    def f_for(lbl: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        a = one_hot(lbl, levels)
        ar = a - qd @ (qd.T @ a)
        qa = orthonormal_basis(ar)
        explained = np.sum((qa.T @ yres) ** 2, axis=0)
        sse1 = np.maximum(sse0 - explained, 0.0)
        df1 = qa.shape[1]
        df2 = len(lbl) - qd.shape[1] - df1
        with np.errstate(divide="ignore", invalid="ignore"):
            f = (explained / df1) / (sse1 / df2)
            r2 = explained / sse0
        return f, r2, df1, df2

    f_obs, r2, df1, df2 = f_for(labels)
    ge = np.zeros(y.shape[1], dtype=int)
    valid = np.zeros(y.shape[1], dtype=int)
    for _ in range(Q1_NPERM):
        f_perm, _, _, _ = f_for(permute_within_tumor(labels, tumors, rng))
        ok = np.isfinite(f_perm) & np.isfinite(f_obs)
        valid += ok
        ge += ok & (f_perm >= f_obs - 1e-12)
    p_emp = (ge + 1) / (valid + 1)
    return pd.DataFrame({
        "partial_r2": np.clip(r2, 0, 1),
        "omnibus_f": f_obs,
        "df_anatomy": df1,
        "df_residual": df2,
        "parametric_p": stats.f.sf(f_obs, df1, df2),
        "permutation_p": p_emp,
        "permutation_valid_n": valid,
        "bh_fdr": bh(p_emp),
    })


def grouped_predictions(y: np.ndarray, labels: np.ndarray, tumors: np.ndarray,
                        levels: list[str]) -> np.ndarray:
    n, m = y.shape
    pred = np.full((n, m), -1, dtype=np.int16)
    for tumor in pd.unique(tumors):
        test = tumors == tumor
        train = ~test
        centroids = np.full((len(levels), m), np.nan)
        for c, level in enumerate(levels):
            rows = train & (labels == level)
            if rows.any():
                centroids[c] = np.mean(y[rows], axis=0)
        distances = np.abs(y[test, None, :] - centroids[None, :, :])
        distances = np.where(np.isfinite(distances), distances, np.inf)
        pred[test] = np.argmin(distances, axis=1)
    return pred


def classification_metrics(pred: np.ndarray, labels: np.ndarray,
                           levels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true = np.array([levels.index(x) for x in labels], dtype=np.int16)
    m = pred.shape[1]
    recalls = np.full((len(levels), m), np.nan)
    f1s = np.full((len(levels), m), np.nan)
    confusion = np.zeros((len(levels), len(levels), m), dtype=int)
    for c in range(len(levels)):
        true_c = true == c
        for d in range(len(levels)):
            confusion[c, d] = np.sum(true_c[:, None] & (pred == d), axis=0)
        tp = confusion[c, c].astype(float)
        n_true = true_c.sum()
        n_pred = np.sum(pred == c, axis=0)
        recalls[c] = tp / n_true if n_true else np.nan
        denom = n_true + n_pred
        f1s[c] = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return np.nanmean(recalls, axis=0), np.nanmean(f1s, axis=0), confusion


def q2_identifiability(y: np.ndarray, labels: np.ndarray, tumors: np.ndarray,
                       levels: list[str], rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    pred = grouped_predictions(y, labels, tumors, levels)
    ba, macro_f1, confusion = classification_metrics(pred, labels, levels)
    # Observed BA and reference BA must share float64 precision so discrete ties count.
    null = np.empty((Q2_NPERM, y.shape[1]), dtype=np.float64)
    for b in range(Q2_NPERM):
        lbl = permute_within_tumor(labels, tumors, rng)
        pred_b = grouped_predictions(y, lbl, tumors, levels)
        null[b] = classification_metrics(pred_b, lbl, levels)[0]
    ge = np.sum(null >= ba[None, :] - 1e-12, axis=0)
    p_emp = (ge + 1) / (Q2_NPERM + 1)
    result = pd.DataFrame({
        "balanced_accuracy": ba,
        "macro_f1": macro_f1,
        "chance_level": 1 / len(levels),
        "null_median": np.median(null, axis=0),
        "null_ci_low": np.quantile(null, 0.025, axis=0),
        "null_ci_high": np.quantile(null, 0.975, axis=0),
        "permutation_p": p_emp,
        "permutation_valid_n": Q2_NPERM,
        "bh_fdr": bh(p_emp),
    })
    for c, level in enumerate(levels):
        denom = confusion[c].sum(axis=0)
        result[f"recall_{level}"] = np.divide(confusion[c, c], denom, out=np.full(y.shape[1], np.nan), where=denom > 0)
    return result, confusion


def weighted_median_boot(values: np.ndarray, groups: np.ndarray, counts: np.ndarray,
                         chunk: int = 500) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    groups = groups[order]
    out = np.empty(counts.shape[0])
    for start in range(0, counts.shape[0], chunk):
        cc = counts[start:start + chunk]
        weights = cc[:, groups]
        cum = np.cumsum(weights, axis=1)
        total = cum[:, -1]
        k1 = (total - 1) // 2
        k2 = total // 2
        i1 = np.argmax(cum > k1[:, None], axis=1)
        i2 = np.argmax(cum > k2[:, None], axis=1)
        out[start:start + len(cc)] = (values[i1] + values[i2]) / 2
    return out


def q3_same_score(y: np.ndarray, labels: np.ndarray, tumors: np.ndarray,
                  rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    tumor_levels = list(pd.unique(tumors))
    near_blocks, pair_blocks, sample_rows = [], [], []
    valid_tumors = []
    for ti, tumor in enumerate(tumor_levels):
        idx = np.flatnonzero(tumors == tumor)
        if len(np.unique(labels[idx])) < 2:
            continue
        local_y = y[idx]
        local_l = labels[idx]
        nearest = np.full((len(idx), y.shape[1]), np.nan)
        pairs = []
        for a in range(len(idx)):
            eligible = local_l != local_l[a]
            nearest[a] = np.min(np.abs(local_y[eligible] - local_y[a]), axis=0)
            for b in range(a + 1, len(idx)):
                if local_l[a] != local_l[b]:
                    pairs.append(np.abs(local_y[a] - local_y[b]))
        if not pairs:
            continue
        group_id = len(valid_tumors)
        valid_tumors.append(tumor)
        near_blocks.append((nearest, np.full(len(nearest), group_id, dtype=int)))
        pair_arr = np.vstack(pairs)
        pair_blocks.append((pair_arr, np.full(len(pair_arr), group_id, dtype=int)))
        for j, global_idx in enumerate(idx):
            sample_rows.append((global_idx, group_id, nearest[j]))

    near = np.vstack([x[0] for x in near_blocks])
    near_group = np.concatenate([x[1] for x in near_blocks])
    pair = np.vstack([x[0] for x in pair_blocks])
    pair_group = np.concatenate([x[1] for x in pair_blocks])
    numerator = np.median(near, axis=0)
    denominator = np.median(pair, axis=0)
    ratio = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)

    counts = rng.multinomial(len(valid_tumors), np.full(len(valid_tumors), 1 / len(valid_tumors)), size=Q3_NBOOT)
    boots = np.full((Q3_NBOOT, y.shape[1]), np.nan, dtype=np.float32)
    for model in range(y.shape[1]):
        boot_num = weighted_median_boot(near[:, model], near_group, counts)
        boot_den = weighted_median_boot(pair[:, model], pair_group, counts)
        boots[:, model] = np.divide(boot_num, boot_den, out=np.full_like(boot_num, np.nan), where=boot_den > 0)
    result = pd.DataFrame({
        "nearest_median_distance": numerator,
        "random_cross_anatomy_median_distance": denominator,
        "nearest_random_ratio": ratio,
        "ratio_ci_low": np.nanquantile(boots, 0.025, axis=0),
        "ratio_ci_high": np.nanquantile(boots, 0.975, axis=0),
        "bootstrap_valid_n": np.sum(np.isfinite(boots), axis=0),
        "contributing_tumor_n": len(valid_tumors),
        "eligible_sample_n": len(near),
        "valid_cross_anatomy_pair_n": len(pair),
    })
    sample_out = []
    for global_idx, _, values in sample_rows:
        for model, value in enumerate(values):
            sample_out.append((global_idx, model, value))
    return result, pd.DataFrame(sample_out, columns=["sample_row", "model_row", "nearest_cross_anatomy_score_distance"])


def publication_bootstrap(values: np.ndarray, publications: np.ndarray,
                          rng: np.random.Generator) -> tuple[float, float, float]:
    pubs = np.array(pd.unique(publications))
    members = {p: np.flatnonzero(publications == p) for p in pubs}
    estimates = np.empty(PUB_NBOOT)
    for b in range(PUB_NBOOT):
        selected = rng.choice(pubs, size=len(pubs), replace=True)
        idx = np.concatenate([members[p] for p in selected])
        estimates[b] = np.nanmedian(values[idx])
    return float(np.nanmedian(values)), float(np.nanquantile(estimates, 0.025)), float(np.nanquantile(estimates, 0.975))


def main() -> None:
    validation = json.loads(VALIDATION_PATH.read_text())
    assert validation["status"] == "PASS" and validation["blocker_count"] == 0
    samples = pd.read_csv(SAMPLE_PATH, sep="\t", dtype={"sample_id": str, "tumor_id": str, "donor_id": str})
    models = pd.read_csv(MODEL_PATH, sep="\t")
    evaluability = pd.read_csv(EVAL_PATH, sep="\t")
    assert len(samples) == 122 and samples.tumor_id.nunique() == 10 and samples.donor_id.nunique() == 10
    assert len(models) == 201 and models.model_instance_id.nunique() == 201
    assert len(evaluability) == 201 and evaluability.evaluable.all()
    assert models.model_instance_id.tolist() == evaluability.model_instance_id.tolist()
    assert not models.model_instance_id.astype(str).str.contains("MES95", case=False).any()

    z = pd.read_parquet(Z_PATH)
    z["sample_id"] = z["sample_id"].astype(str)
    z = samples[["sample_id"]].merge(z, on="sample_id", how="left", validate="one_to_one")
    scores = np.empty((len(samples), len(models)), dtype=float)
    for j, row in evaluability.iterrows():
        genes = str(row.present_genes).split(";")
        assert len(genes) == row.mapped_present_nonzero_variance_gene_n
        scores[:, j] = z[genes].mean(axis=1).to_numpy()
    assert np.isfinite(scores).all()

    score_df = samples[["sample_id", "tumor_id", "donor_id", "anatomical_structure"]].copy()
    score_df = pd.concat([score_df, pd.DataFrame(scores, columns=models.model_instance_id)], axis=1)
    score_path = OUT / "04_SCORES/IVYGAP_PRIMARY201_MODEL_SCORES.tsv.gz"
    score_df.to_csv(score_path, sep="\t", index=False, compression="gzip", float_format="%.10g")

    q1_rng = np.random.default_rng(2026083101)
    q2_rng = np.random.default_rng(2026083102)
    q3_rng = np.random.default_rng(2026083103)
    pub_rng = np.random.default_rng(2026083104)
    q1_all, q2_all, q3_all = [], [], []
    conf_all, nearest_all, summaries = [], [], []
    pubs = models.publication_id.to_numpy()

    for branch, levels in BRANCHES.items():
        keep = samples.anatomical_structure.isin(levels).to_numpy()
        y = scores[keep]
        labels = samples.loc[keep, "anatomical_structure"].to_numpy()
        tumors = samples.loc[keep, "tumor_id"].to_numpy()
        assert len(pd.unique(tumors)) >= 2 and set(pd.unique(labels)) == set(levels)

        r1 = q1_informativeness(y, labels, tumors, levels, q1_rng)
        r2, confusion = q2_identifiability(y, labels, tumors, levels, q2_rng)
        r3, nearest = q3_same_score(y, labels, tumors, q3_rng)
        common = models[["model_instance_id", "publication_id"]].copy()
        for r in (r1, r2, r3):
            r.insert(0, "branch", branch)
            r.insert(1, "model_instance_id", common.model_instance_id)
            r.insert(2, "publication_id", common.publication_id)
            r["sample_n"] = len(y)
            r["tumor_n"] = pd.Series(tumors).nunique()
            r["anatomy_n"] = len(levels)
            r["evaluable"] = True
        q1_all.append(r1)
        q2_all.append(r2)
        q3_all.append(r3)

        for c, true_level in enumerate(levels):
            for d, pred_level in enumerate(levels):
                tmp = common.copy()
                tmp["branch"] = branch
                tmp["true_anatomy"] = true_level
                tmp["predicted_anatomy"] = pred_level
                tmp["n"] = confusion[c, d]
                conf_all.append(tmp)

        branch_samples = samples.loc[keep].reset_index(drop=True)
        nearest = nearest.merge(branch_samples.reset_index(names="sample_row"), on="sample_row", how="left")
        nearest = nearest.merge(models[["model_instance_id"]].reset_index(names="model_row"), on="model_row", how="left")
        nearest["branch"] = branch
        nearest_all.append(nearest[["branch", "sample_id", "tumor_id", "anatomical_structure", "model_instance_id", "nearest_cross_anatomy_score_distance"]])

        med_r2, lo_r2, hi_r2 = publication_bootstrap(r1.partial_r2.to_numpy(), pubs, pub_rng)
        med_ba, lo_ba, hi_ba = publication_bootstrap(r2.balanced_accuracy.to_numpy(), pubs, pub_rng)
        med_ratio, lo_ratio, hi_ratio = publication_bootstrap(r3.nearest_random_ratio.to_numpy(), pubs, pub_rng)
        summaries.append({
            "branch": branch,
            "sample_n": len(y),
            "tumor_n": pd.Series(tumors).nunique(),
            "anatomy_n": len(levels),
            "evaluable_model_n": len(models),
            "publication_n": pd.Series(pubs).nunique(),
            "median_partial_r2": med_r2,
            "median_partial_r2_pubboot_ci_low": lo_r2,
            "median_partial_r2_pubboot_ci_high": hi_r2,
            "q1_fdr_lt_0_05_fraction": float(np.mean(r1.bh_fdr < 0.05)),
            "median_balanced_accuracy": med_ba,
            "median_balanced_accuracy_pubboot_ci_low": lo_ba,
            "median_balanced_accuracy_pubboot_ci_high": hi_ba,
            "q2_ba_ge_0_60_fraction": float(np.mean(r2.balanced_accuracy >= 0.60)),
            "q2_fdr_lt_0_05_fraction": float(np.mean(r2.bh_fdr < 0.05)),
            "median_nearest_random_ratio": med_ratio,
            "median_nearest_random_ratio_pubboot_ci_low": lo_ratio,
            "median_nearest_random_ratio_pubboot_ci_high": hi_ratio,
            "q3_ratio_lt_0_50_fraction": float(np.mean(r3.nearest_random_ratio < 0.50)),
        })

    q1 = pd.concat(q1_all, ignore_index=True)
    q2 = pd.concat(q2_all, ignore_index=True)
    q3 = pd.concat(q3_all, ignore_index=True)
    confusion = pd.concat(conf_all, ignore_index=True)
    nearest = pd.concat(nearest_all, ignore_index=True)
    summary = pd.DataFrame(summaries)

    all5 = "branch == 'ALL5'"
    q1.query(all5).to_csv(OUT / "05_ANATOMICAL_INFORMATIVENESS/IVYGAP_ANATOMICAL_INFORMATIVENESS_MODEL_LEVEL.tsv", sep="\t", index=False)
    summary.query(all5).to_csv(OUT / "05_ANATOMICAL_INFORMATIVENESS/IVYGAP_ANATOMICAL_INFORMATIVENESS_SUMMARY.tsv", sep="\t", index=False)
    q2.query(all5).to_csv(OUT / "06_ANATOMICAL_IDENTIFIABILITY/IVYGAP_ANATOMICAL_IDENTIFIABILITY_MODEL_LEVEL.tsv", sep="\t", index=False)
    confusion.query(all5).to_csv(OUT / "06_ANATOMICAL_IDENTIFIABILITY/IVYGAP_ANATOMICAL_CONFUSION_LONG.tsv.gz", sep="\t", index=False, compression="gzip")
    q1.query(all5)[["model_instance_id", "publication_id", "partial_r2", "bh_fdr"]].merge(
        q2.query(all5)[["model_instance_id", "balanced_accuracy", "macro_f1", "bh_fdr"]], on="model_instance_id", suffixes=("_informativeness", "_identifiability")
    ).to_csv(OUT / "06_ANATOMICAL_IDENTIFIABILITY/ANATOMICAL_INFORMATIVENESS_VS_IDENTIFIABILITY.tsv", sep="\t", index=False)
    q3.query(all5).to_csv(OUT / "07_SAME_SCORE_DIFFERENT_PATHOLOGY/IVYGAP_SAME_SCORE_DIFFERENT_PATHOLOGY_MODEL_LEVEL.tsv", sep="\t", index=False)
    nearest.query(all5).to_csv(OUT / "07_SAME_SCORE_DIFFERENT_PATHOLOGY/IVYGAP_NEAREST_CROSS_ANATOMY_SAMPLE_LEVEL.tsv.gz", sep="\t", index=False, compression="gzip")
    summary.query(all5).to_csv(OUT / "07_SAME_SCORE_DIFFERENT_PATHOLOGY/IVYGAP_SAME_SCORE_DIFFERENT_PATHOLOGY_SUMMARY.tsv", sep="\t", index=False)

    q1.query("branch != 'ALL5'").to_csv(OUT / "08_CORE3_SENSITIVITY/IVYGAP_SENSITIVITY_INFORMATIVENESS_MODEL_LEVEL.tsv", sep="\t", index=False)
    q2.query("branch != 'ALL5'").to_csv(OUT / "08_CORE3_SENSITIVITY/IVYGAP_SENSITIVITY_IDENTIFIABILITY_MODEL_LEVEL.tsv", sep="\t", index=False)
    q3.query("branch != 'ALL5'").to_csv(OUT / "08_CORE3_SENSITIVITY/IVYGAP_SENSITIVITY_SAME_SCORE_MODEL_LEVEL.tsv", sep="\t", index=False)
    confusion.query("branch != 'ALL5'").to_csv(OUT / "08_CORE3_SENSITIVITY/IVYGAP_SENSITIVITY_CONFUSION_LONG.tsv.gz", sep="\t", index=False, compression="gzip")
    nearest.query("branch != 'ALL5'").to_csv(OUT / "08_CORE3_SENSITIVITY/IVYGAP_SENSITIVITY_NEAREST_SAMPLE_LEVEL.tsv.gz", sep="\t", index=False, compression="gzip")
    summary.query("branch != 'ALL5'").to_csv(OUT / "08_CORE3_SENSITIVITY/IVYGAP_CORE3_AND_PAN_MVP_SUMMARY.tsv", sep="\t", index=False)

    run_manifest = {
        "status": "PASS",
        "freeze_validation_sha256": sha256(VALIDATION_PATH),
        "model_manifest_sha256": sha256(MODEL_PATH),
        "sample_manifest_sha256": sha256(SAMPLE_PATH),
        "evaluable_manifest_sha256": sha256(EVAL_PATH),
        "z_matrix_sha256": sha256(Z_PATH),
        "score_output_sha256": sha256(score_path),
        "sample_n": len(samples),
        "tumor_n": samples.tumor_id.nunique(),
        "model_n": len(models),
        "q1_permutations_per_branch": Q1_NPERM,
        "q2_permutations_per_branch": Q2_NPERM,
        "q3_tumor_bootstrap_iterations_per_branch": Q3_NBOOT,
        "publication_bootstrap_iterations_per_branch_metric": PUB_NBOOT,
        "cv_grouping": "LEAVE_ONE_TUMOR_OUT",
        "bootstrap_unit": "TUMOR",
    }
    (OUT / "10_QA/IVYGAP_FORMAL_ANALYSIS_RUN.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "PASS", "summaries": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
