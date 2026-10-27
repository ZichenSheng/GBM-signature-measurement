#!/usr/bin/env python3
from __future__ import annotations

import os as _gbm_os
from pathlib import Path as _GBMPath
def _gbm_external(env, relative):
    return _GBMPath(_gbm_os.environ.get(env, str(_GBMPath.cwd() / "external" / env.lower()))) / relative


import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(str(_gbm_external('GBM_SOURCE_ROOT', '')))
OUT = Path(os.environ.get(
    "GBM_GAP_OUT",
    "outputs/care",
)).resolve()
MEAS = ROOT / "GBM_SIGNATURE_FINAL_MEASUREMENT_ANATOMY_SYNTHESIS_V1"
V1 = ROOT / "GBM_SIGNATURE_CONTEXT_STRESS_TEST_EXTENSION_V1_FINAL"
V2 = ROOT / "GBM_SIGNATURE_CONTEXT_EXTENSION_DEEPENING_V2_FINAL"
V3 = ROOT / "GBM_SIGNATURE_CONTEXT_DEFORMATION_DEEPENING_V3"
V4 = ROOT / "GBM_SIGNATURE_PLASTICITY_REGULATORY_CLOSURE_V4_FINAL"
V5 = ROOT / "GBM_SIGNATURE_MULTICELLULAR_COMMUNITY_CONTEXT_EXTENSION_V5_FINAL"
V6 = ROOT / "GBM_SIGNATURE_3D_CROSS_LAYER_MEASUREMENT_VALIDATION_V6_FINAL"
AUTH = ROOT / "GBM标签数据库/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23"
CLIN = AUTH / "09_de_novo_derived/14_clinical_12model_v2_1_1"
GEOMX = Path(os.environ.get(
    "GBM_GAP_GEOMX",
    str(OUT / "00_governance/external_cache/ACTA2025_GEOMX"),
)).resolve()

VERSION = "GBM_SIGNATURE_FINAL_SPECIFIC_GAP_CLOSURE_V1"
DATE = "2026-08-30"
SEED = 20260830
BOOT_N = 2000
SIM_N = 10000
MIN_ORIGINAL = 3
MIN_MAPPED = 3
MIN_COVERAGE = 0.80
EPSILON_SD = 1e-8
AXES = ["MES", "HYPOXIA", "MATRIX", "MYELOID"]

DIRS = [
    "00_governance/freeze/memberships", "01_wp1_scale_identifiability",
    "02_wp2_clinical_error_ceiling", "03_wp3_acta_geomx", "04_integration",
    "05_tables", "06_qa", "07_handoff", "08_code", "logs",
]


def ensure_dirs() -> None:
    for d in DIRS:
        (OUT / d).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False, na_rep="NA", float_format="%.12g", lineterminator="\n")


def write_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_primary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    manifest = pd.read_csv(MEAS / "01_authority_snapshot/PRIMARY201_MODEL_MANIFEST_FROZEN.tsv", sep="\t", dtype=str)
    genes = pd.read_csv(MEAS / "01_authority_snapshot/PRIMARY201_GENE_LONG_FROZEN.tsv", sep="\t", dtype=str)
    cw = pd.read_csv(MEAS / "01_authority_snapshot/PRIMARY201_GENE_CROSSWALK_FROZEN.tsv", sep="\t", dtype=str)
    q = genes.assign(source_scoring_feature_id=genes.scoring_feature_id.astype(str)).merge(
        cw[["source_scoring_feature_id", "canonical_gene_symbol"]], on="source_scoring_feature_id", how="left"
    )
    q["canonical_gene"] = q.canonical_gene_symbol.fillna(q.source_scoring_feature_id).astype(str).str.upper()
    gene_map = q.groupby("model_instance_id").canonical_gene.agg(lambda x: sorted(set(x))).to_dict()
    return manifest, genes, cw, gene_map


def reconstruct_unsigned_scores(expr: pd.DataFrame, manifest: pd.DataFrame, gene_map: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr = expr.copy()
    expr.columns = expr.columns.astype(str).str.upper()
    cols = set(expr.columns)
    original = manifest.set_index("model_instance_id").original_feature_n.astype(float).astype(int).to_dict()
    out, audit = {}, []
    for model in manifest.model_instance_id:
        present = sorted(set(gene_map.get(model, [])) & cols)
        cov = len(present) / original[model] if original[model] else 0.0
        ok = original[model] >= MIN_ORIGINAL and len(present) >= MIN_MAPPED and cov >= MIN_COVERAGE
        out[model] = expr[present].mean(axis=1).to_numpy(float) if ok else np.full(len(expr), np.nan)
        audit.append({"model_instance_id": model, "original_gene_n": original[model], "mapped_gene_n": len(present), "mapping_coverage": cov, "evaluable": ok})
    return pd.DataFrame(out, index=expr.index), pd.DataFrame(audit)


def grouped_linear_cv(s: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    sse = np.full(len(s), np.nan)
    sst = np.full(len(s), np.nan)
    for patient in np.unique(groups):
        test = groups == patient
        train = ~test
        X = np.column_stack([np.ones(train.sum()), s[train]])
        beta = np.linalg.lstsq(X, y[train], rcond=None)[0]
        pred = np.column_stack([np.ones(test.sum()), s[test]]) @ beta
        center = np.nanmean(y[train], axis=0)
        sse[test] = np.sum((y[test] - pred) ** 2, axis=1)
        sst[test] = np.sum((y[test] - center) ** 2, axis=1)
    return float(1 - np.nansum(sse) / np.nansum(sst)), sse, sst


def publication_bootstrap(values: pd.DataFrame, value_col: str, manifest: pd.DataFrame, seed_offset: int = 0) -> tuple[float, float, np.ndarray]:
    x = values[["model_instance_id", value_col]].merge(manifest[["model_instance_id", "publication_id"]], on="model_instance_id", how="left").dropna(subset=[value_col])
    pubs = sorted(x.publication_id.fillna(x.model_instance_id).unique())
    by_pub = {p: x.loc[x.publication_id.fillna(x.model_instance_id).eq(p), value_col].to_numpy(float) for p in pubs}
    rng = np.random.default_rng(SEED + seed_offset)
    draws = np.empty(BOOT_N)
    for b in range(BOOT_N):
        sampled = rng.choice(pubs, size=len(pubs), replace=True)
        draws[b] = np.nanmedian(np.concatenate([by_pub[p] for p in sampled]))
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), draws


def bh(p: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    arr = p.loc[ok].to_numpy(float)
    if not len(arr):
        return out
    order = np.argsort(arr, kind="mergesort")
    ranked = arr[order]
    adj = np.minimum.accumulate((ranked * len(arr) / np.arange(1, len(arr) + 1))[::-1])[::-1]
    restored = np.empty_like(adj)
    restored[order] = np.minimum(adj, 1.0)
    out.loc[ok] = restored
    return out


def comparable_pairs(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, right = [], []
    for i in range(len(time)):
        for j in range(len(time)):
            if time[i] < time[j] and event[i] == 1:
                left.append(i); right.append(j)
    return np.asarray(left, int), np.asarray(right, int)


def cindex_from_pairs(score: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    d = score[left] - score[right]
    return float((np.sum(d > 0) + .5 * np.sum(d == 0)) / len(d))
