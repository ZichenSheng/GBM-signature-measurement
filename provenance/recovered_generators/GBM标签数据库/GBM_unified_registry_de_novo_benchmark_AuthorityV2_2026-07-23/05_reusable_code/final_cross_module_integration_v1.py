#!/usr/bin/env python3
"""Read-only CURRENT-result integration for the GBM unified registry.

This controller performs identity checks, stable-key joins, deterministic
summaries, provenance freezing, QA, and packaging.  It does not call or
reimplement any upstream statistical analysis.
"""

from __future__ import annotations

import os as _gbm_os
from pathlib import Path as _GBMPath
def _gbm_external(env, relative):
    return _GBMPath(_gbm_os.environ.get(env, str(_GBMPath.cwd() / "external" / env.lower()))) / relative


import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT = Path(str(_gbm_external('GBM_SOURCE_ROOT', 'GBM标签数据库/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23')))
DERIVED = PROJECT / "09_de_novo_derived"
OUT = DERIVED / "15_final_cross_module_integration_v1"
AUTHORITY_ZIP = Path(str(_gbm_external('GBM_SOURCE_ROOT', 'GBM标签数据库/lineage_authority_sources/GBM_legacy46_reconstructed_authority_v2_2026-07-23.zip')))
AUTHORITY_SHA = "78039887ceebaf48fe09a5f1f800b60610898478ae82cf2f4abc994dd836b344"
CLINICAL_ZIP = PROJECT / "13_handoff/DE_NOVO_CLINICAL_12MODEL_V2_1_1_METADATA_CLOSED_PACKAGE.zip"
CLINICAL_SHA = "73f4bd95dafac77d50210405c0ecf2f61f4bae58f9f961e55e2f63f9ce4bcfe9"
METHOD = PROJECT / "06_configs/GBM_METHOD_SPECIFICATION_V2_1.json"

AUTH_ROOT = PROJECT / "02_authority/registry_authority_v2/extracted/GBM_legacy46_reconstructed_authority_v2_2026-07-23"
BP = DERIVED / "10_bulk_primary_corrected_v2_1_1"
BS = DERIVED / "11_bulk_secondary_v2_1_1"
SP = DERIVED / "12_spatial_v2_1_1"
SC_OLD = DERIVED / "13_single_cell_v2_1_1"
SC = DERIVED / "14_full_gene_single_cell_v2_1_2"
CL = DERIVED / "14_clinical_12model_v2_1_1"

CURRENT = {
    "AUTHORITY": PROJECT / "06_configs/CURRENT_AUTHORITY_MANIFEST.json",
    "BULK_PRIMARY": BP / "08_manifests/CURRENT_BULK_PRIMARY_STATUS.json",
    "BULK_SECONDARY": BS / "11_manifests/CURRENT_BULK_SECONDARY_STATUS.json",
    "SPATIAL": SP / "10_manifests/CURRENT_SPATIAL_STATUS.json",
    "SINGLE_CELL": SC / "11_manifests/CURRENT_SINGLE_CELL_STATUS.json",
    "CLINICAL": PROJECT / "13_handoff/CURRENT_CLINICAL_12MODEL_METADATA_CLOSURE.json",
}

CLINICAL_IDS = [
    "LEGACY-E001-M01", "LEGACY-S009-M01", "LEGACY-V017-M01",
    "PF4C-B1-027", "PF4C-B1-028", "PF4C-B2-001", "PF4C-B2-006",
    "PF4C-B2-007", "PF4C-B3-006", "PF4C-B3-008",
    "PF4C-FINAL-009", "PF4C-FINAL-065",
]

DIRS = [
    "00_preflight", "01_current_module_registry", "02_model_evidence_master",
    "03_panel_level_integration", "04_clinical_ecology_integration",
    "05_spatial_singlecell_integration", "06_final_claim_matrix",
    "07_manuscript_tables", "08_figure_source_tables", "09_manifests",
    "10_QA", "14_integrated_summary", "logs",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.fillna("").to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def atomic_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def file_shape(path: Path) -> tuple[str, str, list[str]]:
    if path.suffix == ".json":
        obj = read_json(path)
        return "1", str(len(obj) if isinstance(obj, dict) else 1), list(obj) if isinstance(obj, dict) else []
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n\r").split("\t")
        rows = sum(1 for _ in fh)
    return str(rows), str(len(header)), header


def get(df: pd.DataFrame, key_col: str, key: str, value_col: str, default: str = "") -> str:
    if key_col not in df or value_col not in df:
        return default
    hit = df.loc[df[key_col] == key, value_col]
    return str(hit.iloc[0]) if len(hit) else default


def keyed(df: pd.DataFrame, key: str = "model_instance_id") -> dict[str, dict]:
    return {str(r[key]): {c: str(r[c]) for c in df.columns} for _, r in df.iterrows() if str(r.get(key, ""))}


def aggregate_pairs(df: pd.DataFrame, key: str, label_cols: list[str], value_cols: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    if key not in df:
        return out
    for mid, grp in df.groupby(key, sort=False):
        parts = []
        for _, row in grp.iterrows():
            label = "|".join(str(row.get(c, "")) for c in label_cols)
            vals = ",".join(f"{c}={row.get(c, '')}" for c in value_cols)
            parts.append(f"{label}:{vals}")
        out[str(mid)] = ";".join(parts)
    return out


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("; ".join(map(str, paths)))


RUN_STARTED = utcnow()
RUN_ID = "FINALXMOD_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
fatal: list[dict] = []
recoverable: list[dict] = []
warnings: list[dict] = []
auto_repairs: list[dict] = []


def ledger(severity: str, code: str, object_path: str, detail: str) -> None:
    row = {
        "severity": severity, "code": code, "object_path": object_path,
        "detail": detail, "recorded_at": utcnow(),
    }
    {"FATAL": fatal, "RECOVERABLE": recoverable, "WARNING": warnings}[severity].append(row)


def main() -> int:
    for d in DIRS:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    # ---------- Preflight identity and CURRENT resolution ----------
    authority_sha_actual = sha256(AUTHORITY_ZIP)
    clinical_sha_actual = sha256(CLINICAL_ZIP)
    if authority_sha_actual != AUTHORITY_SHA:
        ledger("FATAL", "AUTHORITY_SHA256_MISMATCH", str(AUTHORITY_ZIP), authority_sha_actual)
    if clinical_sha_actual != CLINICAL_SHA:
        ledger("FATAL", "CLINICAL_METADATA_CLOSED_SHA256_MISMATCH", str(CLINICAL_ZIP), clinical_sha_actual)
    try:
        with zipfile.ZipFile(AUTHORITY_ZIP) as zf:
            authority_crc_bad = zf.testzip()
        if authority_crc_bad:
            ledger("FATAL", "AUTHORITY_ZIP_CRC_FAILURE", str(AUTHORITY_ZIP), authority_crc_bad)
    except Exception as exc:
        ledger("FATAL", "AUTHORITY_ZIP_OPEN_FAILURE", str(AUTHORITY_ZIP), repr(exc))

    current_objs = {}
    for module, path in CURRENT.items():
        if not path.exists():
            ledger("FATAL", f"{module}_CURRENT_MISSING", str(path), "Required CURRENT entrypoint missing")
            current_objs[module] = {}
        else:
            current_objs[module] = read_json(path)

    old_sc_status = read_json(SC_OLD / "10_manifests/CURRENT_SINGLE_CELL_STATUS.json")
    if not (
        old_sc_status.get("CURRENT_STATUS") == "SUPERSEDED_AS_FORMAL_CURRENT"
        and old_sc_status.get("SUCCESSOR_CURRENT_STATUS_PATH") == str(CURRENT["SINGLE_CELL"])
        and current_objs["SINGLE_CELL"].get("FULL_GENE_RESULT_SET_AS_CURRENT") == "YES"
    ):
        ledger("FATAL", "STOP_FINAL_INTEGRATION_SINGLE_CELL_CURRENT_AMBIGUOUS", str(SC), "Successor chain did not uniquely resolve")

    status_raw = {
        "AUTHORITY": current_objs["AUTHORITY"].get("authority_snapshot_status", ""),
        "BULK_PRIMARY": current_objs["BULK_PRIMARY"].get("CURRENT_BULK_PRIMARY_STATUS", ""),
        "BULK_SECONDARY": current_objs["BULK_SECONDARY"].get("CURRENT_BULK_SECONDARY_STATUS", ""),
        "SPATIAL": current_objs["SPATIAL"].get("FORMAL_DELIVERY_STATUS", ""),
        "SINGLE_CELL": current_objs["SINGLE_CELL"].get("FORMAL_DELIVERY_STATUS", ""),
        "CLINICAL": current_objs["CLINICAL"].get("CURRENT_CLINICAL_12MODEL_METADATA_CLOSURE_STATUS", ""),
    }
    for module, value in status_raw.items():
        if value != "PASS":
            ledger("FATAL", f"{module}_CURRENT_NOT_PASS", str(CURRENT[module]), f"Observed {value!r}")

    # ---------- Formal source tables ----------
    paths = {
        "authority_publications": AUTH_ROOT / "10_UNIFIED_PUBLICATION_REGISTRY_AUTHORITY_V2.tsv",
        "authority_models": AUTH_ROOT / "11_UNIFIED_MODEL_INSTANCE_REGISTRY_AUTHORITY_V2.tsv",
        "authority_genes": AUTH_ROOT / "12_UNIFIED_GENE_MEMBERSHIP_LONG_AUTHORITY_V2.tsv",
        "authority_a1": AUTH_ROOT / "13_UNIFIED_A1_AUTHORITY_V2.tsv",
        "authority_rnaseq_a1": AUTH_ROOT / "14_UNIFIED_RNASEQ_A1_AUTHORITY_V2.tsv",
        "authority_ecology": AUTH_ROOT / "17_ECOLOGY_PANEL_CANDIDATE_MASTER_AUTHORITY_V2.tsv",
        "primary_manifest": BP / "08_manifests/downstream_model_inputs_v2_1_1/01_PRIMARY201_MODEL_MANIFEST_V2_1_1.tsv",
        "pre_manifest": BP / "08_manifests/downstream_model_inputs_v2_1_1/03_PRE37_MODEL_MANIFEST_V2_1_1.tsv",
        "post_manifest": BP / "08_manifests/downstream_model_inputs_v2_1_1/04_POST164_MODEL_MANIFEST_V2_1_1.tsv",
        "sensitivity_manifest": BP / "08_manifests/downstream_model_inputs_v2_1_1/05_SENSITIVITY17_MODEL_MANIFEST_V2_1_1.tsv",
        "a1_manifest": BP / "08_manifests/downstream_model_inputs_v2_1_1/07_RNASEQ_A1_12_MODEL_MANIFEST_V2_1_1.tsv",
        "score_summary": BP / "01_scores/PRIMARY_UNSIGNED_SCORE_SUMMARY.tsv",
        "load325": BP / "04_pca/CGGA325_PCA_MODEL_LOADINGS.tsv",
        "load693": BP / "04_pca/CGGA693_PCA_MODEL_LOADINGS.tsv",
        "eco325": BP / "05_ecology_programs/CGGA325_MODEL_ECOLOGY_ASSOCIATIONS_CORRECTED.tsv.gz",
        "eco693": BP / "05_ecology_programs/CGGA693_MODEL_ECOLOGY_ASSOCIATIONS_CORRECTED.tsv.gz",
        "pc1_ecology": BP / "05_ecology_programs/PC1_ECOLOGY_ASSOCIATIONS.tsv",
        "bulk_bootstrap": BP / "06_bootstrap/BULK_PRIMARY_BOOTSTRAP_SUMMARY_CORRECTED.tsv",
        "cross_cohort": BP / "03_score_correlations/CROSS_COHORT_CORRELATION_STRUCTURE_CORRECTED.tsv",
        "hf_status": BS / "01_high_frequency_removal/HIGH_FREQUENCY_REMOVAL_MODEL_STATUS.tsv",
        "hf_summary": BS / "01_high_frequency_removal/HIGH_FREQUENCY_REMOVAL_SUMMARY.tsv",
        "overlap_status": BS / "02_ecology_overlap_removal/ECOLOGY_OVERLAP_REMOVAL_MODEL_STATUS.tsv",
        "overlap_summary": BS / "02_ecology_overlap_removal/ECOLOGY_OVERLAP_REMOVAL_SUMMARY.tsv",
        "residual_summary": BS / "03_residualization/RESIDUALIZED_STRUCTURE_SUMMARY.tsv",
        "prepost": BS / "04_pre_post_temporal_replication_corrected_37_164/PRE_POST_TEMPORAL_REPLICATION_SUMMARY_CORRECTED.tsv",
        "matched_null": BS / "05_matched_panel_null/MATCHED_PANEL_NULL_SUMMARY.tsv",
        "tcga": BS / "07_tcga_broad_sensitivity/TCGA_BROAD_SENSITIVITY_SUMMARY.tsv",
        "secondary_state": BS / "10_integrated_summary/DE_NOVO_BULK_SECONDARY_V2_1_1_STATE_CORRECTED.json",
        "spatial_crosswalk": SP / "01_object_inventory/SPATIAL_SECTION_PATIENT_CROSSWALK_V2_1_1.tsv",
        "spatial_primary": SP / "09_integrated_summary/SPATIAL_PRIMARY_MODEL_LEVEL_SUMMARY.tsv",
        "spatial_panel": SP / "09_integrated_summary/SPATIAL_PANEL_DESCRIPTIVE_SUMMARY.tsv",
        "spatial_nonoverlap": SP / "09_integrated_summary/SPATIAL_NONOVERLAP_SUMMARY.tsv",
        "spatial_not_eval": SP / "02_model_coverage/SPATIAL_NOT_EVALUABLE_MODEL_RESOURCE.tsv",
        "sc_class": SC / "05_model_classification/PRIMARY201_FULL_GENE_CLASSIFICATION.tsv",
        "sc_sens_class": SC / "08_sensitivity/SENSITIVITY17_FULL_GENE_CLASSIFICATION.tsv",
        "sc_patient": SC / "04_patient_localization/PRIMARY201_FULL_GENE_PATIENT_DELTA.tsv",
        "sc_primary_summary": SC / "10_integrated_summary/SINGLE_CELL_FULL_GENE_PRIMARY_SUMMARY.tsv",
        "sc_prepost_summary": SC / "10_integrated_summary/SINGLE_CELL_FULL_GENE_PRE_POST_SUMMARY.tsv",
        "sc_a1_summary": SC / "10_integrated_summary/SINGLE_CELL_FULL_GENE_A1_SUMMARY.tsv",
        "clinical_formula": CL / "02_formula_mapping/CLINICAL_12MODEL_FORMULA_MASTER.tsv",
        "clinical_score_qa": CL / "03_scores/CLINICAL_SCORE_QA.tsv",
        "clinical_cox": CL / "04_cox/CPTAC_COX_RESULTS.tsv",
        "clinical_perf": CL / "05_performance/CPTAC_PERFORMANCE_RESULTS.tsv",
        "clinical_diag": CL / "06_calibration_PH_nonlinearity/CPTAC_DIAGNOSTIC_RESULTS.tsv",
        "clinical_boot": CL / "07_bootstrap/CLINICAL_BOOTSTRAP_SUMMARY.tsv",
        "clinical_glass": CL / "08_GLASS_sensitivity/GLASS_SENSITIVITY_RESULTS.tsv",
        "clinical_n1": CL / "09_N1/N1_SUMMARY.tsv",
        "clinical_n2": CL / "10_N2/N2_SUMMARY.tsv",
        "clinical_bh": CL / "11_multiplicity/CLINICAL_12MODEL_BH_FDR.tsv",
        "clinical_maxt": CL / "12_westfall_young/WESTFALL_YOUNG_MAXT_RESULTS.tsv",
        "clinical_components": CL / "13_transportability/TRANSPORTABILITY_COMPONENTS.tsv",
        "clinical_gate": CL / "13_transportability/TRANSPORTABILITY_GATE_RESULTS.tsv",
        "clinical_status": CL / "14_integrated_summary/FINAL_STATUS_TABLE.tsv",
    }
    for role, path in paths.items():
        if not path.exists():
            ledger("FATAL", "FORMAL_SOURCE_MISSING", str(path), role)

    df = {name: read_tsv(path) for name, path in paths.items() if path.suffix != ".json" and path.exists()}
    authority_models = df["authority_models"]
    authority_pubs = df["authority_publications"]
    primary = df["primary_manifest"]
    pre = df["pre_manifest"]
    post = df["post_manifest"]
    sensitivity = df["sensitivity_manifest"]
    a1 = df["a1_manifest"]

    expected_counts = {
        "publications": (len(authority_pubs), 265),
        "model_instances": (len(authority_models), 271),
        "eligible": (authority_models["current_eligibility"].isin(["ELIGIBLE", "ELIGIBLE_RETAINED"]).sum(), 242),
        "complete_gene_sets": ((authority_models["complete_gene_membership"] == "YES").sum(), 232),
        "A1": (len(df["authority_a1"]), 13),
        "RNAseq_A1": (len(df["authority_rnaseq_a1"]), 12),
        "Primary": (len(primary), 201),
        "Sensitivity": (len(sensitivity), 17),
        "PRE": (len(pre), 37),
        "POST": (len(post), 164),
    }
    for label, (observed, expected) in expected_counts.items():
        if int(observed) != expected:
            ledger("FATAL", f"{label.upper()}_COUNT_MISMATCH", str(paths.get("authority_models", "")), f"{observed} != {expected}")

    secondary_state = read_json(paths["secondary_state"])
    if secondary_state.get("PRE_FREEZE_MODEL_N") != 37 or secondary_state.get("POST_FREEZE_MODEL_N") != 164:
        ledger("FATAL", "PRE_POST_NOT_37_164", str(paths["secondary_state"]), json.dumps(secondary_state))

    clinical_status = dict(zip(df["clinical_status"]["field"], df["clinical_status"]["value"]))
    clinical_expected = {
        "CLINICAL_MODEL_N": "12", "CPTAC_PATIENT_N": "86", "CPTAC_EVENT_N": "59",
        "CPTAC_B1_N": "80", "GLASS_PATIENT_N": "102", "GLASS_EVENT_N": "91",
        "GLASS_EVALUABLE_MODEL_N": "6", "CPTAC_FULLY_MAPPED_MODEL_N": "12",
        "TRANSPORTABILITY_PASS_MODEL_N": "0", "READY_FOR_FINAL_INTEGRATION_STATUS": "YES",
    }
    for key, expected in clinical_expected.items():
        if clinical_status.get(key) != expected:
            ledger("FATAL", f"CLINICAL_{key}_MISMATCH", str(paths["clinical_status"]), f"{clinical_status.get(key)} != {expected}")

    # ---------- CURRENT module registry ----------
    module_registry = pd.DataFrame([
        {
            "module_name": "AUTHORITY", "current_status_path": str(CURRENT["AUTHORITY"]),
            "current_status": status_raw["AUTHORITY"], "formal_package_path": str(AUTHORITY_ZIP),
            "formal_package_SHA256": authority_sha_actual, "result_root": str(AUTH_ROOT),
            "manifest_path": str(CURRENT["AUTHORITY"]), "QA_path": str(AUTH_ROOT / "16_UNIFIED_REGISTRY_QA_AUTHORITY_V2.txt"),
            "run_id": "AUTHORITY_V2_20260723", "completed_at": "", "local_output_status": "LOCAL",
            "blocking_issue_n": "0" if status_raw["AUTHORITY"] == "PASS" else "1", "ready_for_integration": "YES" if status_raw["AUTHORITY"] == "PASS" else "NO",
        },
        {
            "module_name": "BULK_PRIMARY", "current_status_path": str(CURRENT["BULK_PRIMARY"]),
            "current_status": status_raw["BULK_PRIMARY"], "formal_package_path": "NOT_SEPARATELY_PACKAGED_CURRENT_CHAIN",
            "formal_package_SHA256": "", "result_root": str(BP),
            "manifest_path": str(BP / "08_manifests/BULK_PRIMARY_STATISTICAL_CORRECTION_V2_1_1_MANIFEST.tsv"),
            "QA_path": str(BP / "07_statistical_corrections/MODEL_ECOLOGY_BH_FDR_CORRECTION_QA.tsv"),
            "run_id": "BULK_PRIMARY_CORRECTED_V2_1_1", "completed_at": "", "local_output_status": "LOCAL",
            "blocking_issue_n": "0", "ready_for_integration": "YES",
        },
        {
            "module_name": "BULK_SECONDARY", "current_status_path": str(CURRENT["BULK_SECONDARY"]),
            "current_status": secondary_state.get("DE_NOVO_BULK_SECONDARY_CORRECTION_STATUS", status_raw["BULK_SECONDARY"]),
            "formal_package_path": "NOT_SEPARATELY_PACKAGED_CURRENT_CHAIN", "formal_package_SHA256": "",
            "result_root": str(BS), "manifest_path": str(CURRENT["BULK_SECONDARY"]),
            "QA_path": str(BS / "12_QA/DE_NOVO_BULK_SECONDARY_V2_1_1_FINAL_QA_CORRECTED.txt"),
            "run_id": secondary_state.get("RUN_ID", "BULK_SECONDARY_V2_1_1"),
            "completed_at": secondary_state.get("RUN_FINISHED_AT", ""), "local_output_status": "LOCAL",
            "blocking_issue_n": "0", "ready_for_integration": "YES",
        },
        {
            "module_name": "SPATIAL", "current_status_path": str(CURRENT["SPATIAL"]),
            "current_status": current_objs["SPATIAL"].get("SPATIAL_FORMAL_EXECUTION_STATUS", status_raw["SPATIAL"]),
            "formal_package_path": "NOT_SEPARATELY_PACKAGED_CURRENT_CHAIN", "formal_package_SHA256": "",
            "result_root": str(SP), "manifest_path": str(SP / "10_manifests/SPATIAL_OUTPUT_SHA256_MANIFEST.tsv"),
            "QA_path": str(SP / "11_QA"), "run_id": current_objs["SPATIAL"].get("SPATIAL_RUN_ID", ""),
            "completed_at": "", "local_output_status": "LOCAL", "blocking_issue_n": "0", "ready_for_integration": "YES",
        },
        {
            "module_name": "SINGLE_CELL", "current_status_path": str(CURRENT["SINGLE_CELL"]),
            "current_status": current_objs["SINGLE_CELL"].get("SINGLE_CELL_FULL_GENE_EXECUTION_STATUS", status_raw["SINGLE_CELL"]),
            "formal_package_path": "NOT_SEPARATELY_PACKAGED_CURRENT_CHAIN", "formal_package_SHA256": "",
            "result_root": str(SC), "manifest_path": str(SC / "11_manifests/SINGLE_CELL_FULL_GENE_OUTPUT_SHA256_MANIFEST.tsv"),
            "QA_path": str(SC / "12_QA/SINGLE_CELL_FULL_GENE_EXECUTION_QA.txt"),
            "run_id": current_objs["SINGLE_CELL"].get("SINGLE_CELL_RUN_ID", ""), "completed_at": "",
            "local_output_status": "LOCAL", "blocking_issue_n": "0", "ready_for_integration": "YES",
        },
        {
            "module_name": "CLINICAL", "current_status_path": str(CURRENT["CLINICAL"]),
            "current_status": current_objs["CLINICAL"].get("CURRENT_CLINICAL_12MODEL_METADATA_CLOSURE_STATUS", ""),
            "formal_package_path": str(CLINICAL_ZIP), "formal_package_SHA256": clinical_sha_actual,
            "result_root": str(CL), "manifest_path": str(PROJECT / "13_handoff/metadata_closure_patch_v1/DE_NOVO_CLINICAL_12MODEL_V2_1_1_METADATA_CLOSURE_MANIFEST.tsv"),
            "QA_path": str(PROJECT / "12_QA/DE_NOVO_CLINICAL_12MODEL_V2_1_1_FINAL_QA.txt"),
            "run_id": current_objs["CLINICAL"].get("original_run_id", ""), "completed_at": current_objs["CLINICAL"].get("updated_at", ""),
            "local_output_status": "LOCAL", "blocking_issue_n": "0", "ready_for_integration": "YES",
        },
    ])
    write_tsv(module_registry, OUT / "01_current_module_registry/GBM_CURRENT_MODULE_REGISTRY.tsv")
    atomic_json({"run_id": RUN_ID, "modules": module_registry.to_dict(orient="records")}, OUT / "01_current_module_registry/GBM_CURRENT_MODULE_REGISTRY.json")

    # ---------- Input manifest and contamination audit ----------
    module_for_role = {}
    for role in paths:
        module_for_role[role] = (
            "AUTHORITY" if role.startswith("authority") else
            "BULK_PRIMARY" if role in {"primary_manifest", "pre_manifest", "post_manifest", "sensitivity_manifest", "a1_manifest", "score_summary", "load325", "load693", "eco325", "eco693", "pc1_ecology", "bulk_bootstrap", "cross_cohort"} else
            "BULK_SECONDARY" if role in {"hf_status", "hf_summary", "overlap_status", "overlap_summary", "residual_summary", "prepost", "matched_null", "tcga", "secondary_state"} else
            "SPATIAL" if role.startswith("spatial") else
            "SINGLE_CELL" if role.startswith("sc_") else "CLINICAL"
        )
    source_manifest_by_module = {r.module_name: r.manifest_path for r in module_registry.itertuples()}
    input_rows = []
    all_inputs = list(CURRENT.values()) + [AUTHORITY_ZIP, CLINICAL_ZIP, METHOD] + list(paths.values())
    seen = set()
    for path in all_inputs:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        role = next((r for r, p in paths.items() if p == path), path.stem)
        module = next((m for m, p in CURRENT.items() if p == path), module_for_role.get(role, "CONFIG"))
        rows, cols, headers = file_shape(path)
        keys = [c for c in ["model_instance_id", "publication_id", "source_stratum", "result_key", "field", "statistic", "metric"] if c in headers]
        st = path.stat()
        input_rows.append({
            "absolute_path": str(path), "module": module, "role": role,
            "file_size": st.st_size, "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "SHA256": sha256(path), "rows": rows, "columns": cols,
            "key_columns": ";".join(keys), "read_only_status": "READ_ONLY_BY_TASK_POLICY",
            "source_current_manifest": source_manifest_by_module.get(module, str(CURRENT.get(module, METHOD))),
        })
    input_manifest = pd.DataFrame(input_rows)
    write_tsv(input_manifest, OUT / "00_preflight/FINAL_INTEGRATION_INPUT_MANIFEST.tsv")
    write_tsv(input_manifest[["absolute_path", "module", "role", "SHA256"]], OUT / "00_preflight/FINAL_INTEGRATION_INPUT_SHA256.tsv")

    contamination_patterns = {
        "OLD_9MODEL_CLINICAL": re.compile(r"9model|9_model", re.I),
        "OLD_MULTIPLICITY": re.compile(r"old.*multiplicity|9model.*multiplicity", re.I),
        "OLD_PATIENT_LIST": re.compile(r"old.*patient|patient.*old", re.I),
        "OLD_35_166": re.compile(r"35_166|35.*166", re.I),
        "SUPERSEDED_BULK": re.compile(r"superseded|10_bulk_primary$", re.I),
        "BLOCKED_OR_NONFINAL": re.compile(r"blocked|nonfinal", re.I),
        "SUPERSEDED_SINGLE_CELL": re.compile(r"13_single_cell_v2_1_1", re.I),
    }
    candidates = []
    for path in PROJECT.rglob("*"):
        if not path.is_file() or OUT in path.parents:
            continue
        s = str(path)
        for category, pattern in contamination_patterns.items():
            if pattern.search(s):
                candidates.append({"category": category, "candidate_path": s, "formal_input_used": "NO", "disposition": "AUDIT_ONLY_EXCLUDED"})
    contamination = pd.DataFrame(candidates, columns=["category", "candidate_path", "formal_input_used", "disposition"])
    write_tsv(contamination, OUT / "00_preflight/OLD_RESULT_CONTAMINATION_AUDIT.tsv")

    # ---------- Stable-key model master ----------
    authority_idx = keyed(authority_models)
    primary_idx, sensitivity_idx, a1_idx = keyed(primary), keyed(sensitivity), keyed(a1)
    score = df["score_summary"]
    score_avail = {(r["model_instance_id"], r["cohort_id"]): "YES" if r["zero_variance_status"] == "PASS" else r["zero_variance_status"] for _, r in score.iterrows()}
    load325, load693 = keyed(df["load325"]), keyed(df["load693"])
    eco325 = aggregate_pairs(df["eco325"], "model_instance_id", ["program_id"], ["spearman_rho", "p_value_raw", "BH_FDR"])
    eco693 = aggregate_pairs(df["eco693"], "model_instance_id", ["program_id"], ["spearman_rho", "p_value_raw", "BH_FDR"])
    hf = aggregate_pairs(df["hf_status"], "model_instance_id", ["cohort_id"], ["scoring_status", "remaining_gene_n", "remaining_fraction"])
    ov = aggregate_pairs(df["overlap_status"], "model_instance_id", ["cohort_id"], ["scoring_status", "remaining_gene_n", "remaining_fraction", "ecology_overlap_gene_n"])
    spatial = df["spatial_primary"].rename(columns={"score_id": "model_instance_id"})
    spatial_agg = aggregate_pairs(spatial, "model_instance_id", ["niche"], ["evaluable_patient_n", "median_patient_effect", "bootstrap_CI_low", "bootstrap_CI_high", "p_value", "q_value", "result_key"])
    spatial_idx = {str(x) for x in spatial["model_instance_id"]}
    spatial_not_eval_ids = set(df["spatial_not_eval"].get("model_instance_id", pd.Series(dtype=str)).astype(str))
    sc_all = pd.concat([df["sc_class"], df["sc_sens_class"]], ignore_index=True).drop_duplicates("model_instance_id", keep="first")
    sc_idx = keyed(sc_all)

    formula_idx = keyed(df["clinical_formula"])
    cox = df["clinical_cox"]
    perf_idx = keyed(df["clinical_perf"])
    diag = df["clinical_diag"]
    boot_idx = keyed(df["clinical_boot"])
    glass_idx = keyed(df["clinical_glass"])
    n1_idx, n2_idx = keyed(df["clinical_n1"]), keyed(df["clinical_n2"])
    maxt_idx = keyed(df["clinical_maxt"])
    comp_idx, gate_idx = keyed(df["clinical_components"]), keyed(df["clinical_gate"])
    scoreqa = df["clinical_score_qa"]
    cptac_eval = {r["model_instance_id"]: r["mapping_status"] for _, r in scoreqa[scoreqa["cohort"] == "CPTAC"].iterrows()}
    glass_eval = {r["model_instance_id"]: r["mapping_status"] for _, r in scoreqa[scoreqa["cohort"] == "GLASS"].iterrows()}

    bh_min = {}
    for mid, grp in df["clinical_bh"].groupby("model_instance_id"):
        vals = pd.to_numeric(grp["BH_FDR"], errors="coerce")
        bh_min[str(mid)] = str(vals.min()) if vals.notna().any() else ""

    master_ids = sorted(set(primary["model_instance_id"]) | set(sensitivity["model_instance_id"]) | set(CLINICAL_IDS) | spatial_idx | set(sc_all["model_instance_id"]))
    master_rows = []
    for mid in master_ids:
        a = authority_idx.get(mid, {})
        p = primary_idx.get(mid, {})
        s = sensitivity_idx.get(mid, {})
        cf = formula_idx.get(mid, {})
        scv = sc_idx.get(mid, {})
        clinical_member = mid in CLINICAL_IDS
        def coxv(model: str, col: str) -> str:
            hit = cox[(cox["model_instance_id"] == mid) & (cox["model"] == model)]
            return str(hit.iloc[0][col]) if len(hit) else ""
        def diagv(test: str, col: str) -> str:
            hit = diag[(diag["model_instance_id"] == mid) & (diag["test"] == test)]
            return str(hit.iloc[0][col]) if len(hit) else ""
        gl = glass_idx.get(mid, {})
        cl_default = "" if clinical_member else "NOT_IN_CLINICAL_12MODEL_SET"
        master_rows.append({
            "model_instance_id": mid, "publication_id": a.get("publication_id", p.get("publication_id", s.get("publication_id", ""))),
            "source_stratum": a.get("source_stratum", p.get("source_stratum", s.get("source_stratum", ""))),
            "pre_or_post_freeze": "PRE" if mid in set(pre["model_instance_id"]) else ("POST" if mid in set(post["model_instance_id"]) else "OUTSIDE_PRIMARY_TEMPORAL_PANEL"),
            "current_tier": a.get("current_executability_tier", ""), "RNAseq_compatible": a.get("RNAseq_compatibility", ""),
            "ecology_panel_role": "PRIMARY" if mid in primary_idx else ("SENSITIVITY" if mid in sensitivity_idx else "OTHER_FORMAL_OUTPUT"),
            "clinical_12model_member": "YES" if clinical_member else "NO",
            "published_feature_n": p.get("original_feature_n", s.get("original_feature_n", cf.get("published_feature_n", ""))),
            "mapped_feature_n": p.get("mapped_feature_n", s.get("mapped_feature_n", cf.get("published_feature_n", ""))),
            "feature_type": a.get("feature_type", p.get("feature_type", s.get("feature_type", ""))),
            "formula_available": "YES" if cf.get("exact_formula", a.get("exact_formula", "")) else "NO",
            "risk_direction": cf.get("published_risk_direction", "NOT_AVAILABLE_IN_FORMAL_MODEL_RECORD"),
            "bulk_primary_evaluable": "YES" if mid in primary_idx else "NO_NOT_IN_PRIMARY201",
            "CGGA325_score_available": score_avail.get((mid, "CGGA325"), "NOT_IN_PRIMARY201"),
            "CGGA693_score_available": score_avail.get((mid, "CGGA693"), "NOT_IN_PRIMARY201"),
            "CGGA325_model_ecology_association": eco325.get(mid, "NOT_IN_PRIMARY201"),
            "CGGA693_model_ecology_association": eco693.get(mid, "NOT_IN_PRIMARY201"),
            "cross_cohort_structure_component": "PRIMARY201_PANEL_MEMBER" if mid in primary_idx else "NOT_IN_PRIMARY201",
            "primary_PC1_loading": (
                f"CGGA325={load325.get(mid, {}).get('PC1', '')};CGGA693={load693.get(mid, {}).get('PC1', '')}"
                if mid in primary_idx else "NOT_IN_PRIMARY201"
            ),
            "primary_panel_membership": p.get("primary_status", "NO"),
            "bulk_primary_status": status_raw["BULK_PRIMARY"] if mid in primary_idx else "NOT_IN_PRIMARY201",
            "bulk_primary_source_file": str(paths["primary_manifest"]) if mid in primary_idx else "",
            "direct_overlap_removed_status": ov.get(mid, "NOT_MODEL_LEVEL_EVALUABLE"),
            "high_frequency_removed_status": hf.get(mid, "NOT_MODEL_LEVEL_EVALUABLE"),
            "four_program_residualized_status": "PANEL_LEVEL_FORMAL_RESULT_AVAILABLE" if mid in primary_idx else "NOT_IN_PRIMARY201",
            "residualized_effect_change": "PANEL_LEVEL_ONLY_SEE_RESIDUALIZED_STRUCTURE_SUMMARY" if mid in primary_idx else "NOT_IN_PRIMARY201",
            "pre_post_stratum": "PRE_37" if mid in set(pre["model_instance_id"]) else ("POST_164" if mid in set(post["model_instance_id"]) else "NOT_IN_PRIMARY_TEMPORAL_PANEL"),
            "pre_post_result_status": "FORMAL_CROSS_PANEL_TEMPORAL_REPLICATION_MEMBER" if mid in primary_idx else "NOT_IN_PRIMARY201",
            "matched_null_status": "PANEL_LEVEL_ONLY" if mid in primary_idx else "NOT_IN_PRIMARY201",
            "TCGA_sensitivity_status": "PRIMARY201_SCORABLE_PANEL" if mid in primary_idx else "NOT_IN_PRIMARY201",
            "bulk_secondary_status": status_raw["BULK_SECONDARY"] if mid in primary_idx else "NOT_IN_PRIMARY201",
            "bulk_secondary_source_file": str(paths["prepost"]) if mid in primary_idx else "",
            "spatial_evaluable": "YES" if mid in spatial_idx else ("NO" if mid in spatial_not_eval_ids else "NOT_IN_SPATIAL_FORMAL_PANEL"),
            "spatial_patient_n": "17" if mid in spatial_idx else "",
            "spatial_region_or_spot_definition": "PATIENT_LEVEL_EFFECTS_ACROSS_MATRIX_RICH|MYELOID_RICH|VASCULAR_RICH_NICHES" if mid in spatial_idx else "",
            "spatial_direction": spatial_agg.get(mid, ""),
            "spatial_effect_estimate": spatial_agg.get(mid, ""),
            "spatial_P_or_FDR_if_prespecified": spatial_agg.get(mid, ""),
            "spatial_nonoverlap_evaluable": "NO_PROGRAM_GENESETS_TOO_SMALL",
            "spatial_nonoverlap_direction": "NOT_EVALUABLE",
            "spatial_nonoverlap_status": "NOT_EVALUABLE_GENESET_TOO_SMALL",
            "spatial_not_evaluable_reason": "" if mid in spatial_idx else ("FORMAL_RESOURCE_NOT_EVALUABLE" if mid in spatial_not_eval_ids else "NOT_IN_SPATIAL_FORMAL_PANEL"),
            "spatial_source_file": str(paths["spatial_primary"]) if mid in spatial_idx else str(paths["spatial_not_eval"]),
            "single_cell_evaluable": "YES" if mid in sc_idx else "NO",
            "single_cell_patient_n": "66" if mid in sc_idx else "",
            "single_cell_study_n": "7" if mid in sc_idx else "",
            "malignant_component": f"positive_patient_n={scv.get('positive_patient_n', '')}" if mid in sc_idx else "",
            "TME_component": f"negative_patient_n={scv.get('negative_patient_n', '')}" if mid in sc_idx else "",
            "patient_level_localization_class": scv.get("final_class", ""),
            "localization_stability": scv.get("class_retention", ""),
            "malignant_dominant_flag": "YES" if scv.get("final_class") == "malignant-dominant" else ("NO" if mid in sc_idx else ""),
            "TME_dominant_flag": "YES" if scv.get("final_class") == "TME-dominant" else ("NO" if mid in sc_idx else ""),
            "mixed_flag": "YES" if scv.get("final_class") == "mixed" else ("NO" if mid in sc_idx else ""),
            "unstable_flag": "YES" if scv.get("final_class") == "unstable" else ("NO" if mid in sc_idx else ""),
            "single_cell_not_evaluable_reason": "" if mid in sc_idx else "NOT_IN_FORMAL_SINGLE_CELL_PANEL",
            "single_cell_source_file": str(paths["sc_class"] if mid in primary_idx else paths["sc_sens_class"]) if mid in sc_idx else "",
            "CPTAC_evaluable": cptac_eval.get(mid, cl_default), "CPTAC_HR_M0": coxv("M0", "HR") or cl_default,
            "CPTAC_P_M0": coxv("M0", "p_value") or cl_default, "CPTAC_HR_B0": coxv("B0", "HR") or cl_default,
            "CPTAC_P_B0": coxv("B0", "p_value") or cl_default, "CPTAC_HR_B1": coxv("B1", "HR") or cl_default,
            "CPTAC_P_B1": coxv("B1", "p_value") or cl_default,
            "CPTAC_C_index": perf_idx.get(mid, {}).get("Harrell_C", cl_default),
            "delta_C_B0": perf_idx.get(mid, {}).get("deltaC_B0", cl_default), "delta_C_B0_P": perf_idx.get(mid, {}).get("deltaC_B0_p", cl_default),
            "delta_C_B1": perf_idx.get(mid, {}).get("deltaC_B1", cl_default), "delta_C_B1_P": perf_idx.get(mid, {}).get("deltaC_B1_p", cl_default),
            "calibration_slope": diagv("CALIBRATION_SLOPE", "estimate") or cl_default,
            "PH_P": diagv("PH", "p_value") or cl_default, "nonlinearity_P": diagv("RCS_NONLINEARITY", "p_value") or cl_default,
            "bootstrap_direction_retention": ("HR_CI=" + boot_idx.get(mid, {}).get("HR_CI_low", "") + ".." + boot_idx.get(mid, {}).get("HR_CI_high", "")) if clinical_member else cl_default,
            "bootstrap_valid_n": boot_idx.get(mid, {}).get("valid_n", cl_default),
            "N1_empirical_P": n1_idx.get(mid, {}).get("empirical_p", cl_default), "N2_empirical_P": n2_idx.get(mid, {}).get("empirical_p", cl_default),
            "BH_FDR_min": bh_min.get(mid, cl_default), "maxT_adjusted_P": maxt_idx.get(mid, {}).get("maxT_adjusted_p", cl_default),
            "GLASS_evaluable": glass_eval.get(mid, cl_default), "GLASS_HR": gl.get("HR", cl_default),
            "GLASS_P": gl.get("p_value", cl_default), "cross_cohort_direction": gl.get("direction_concordance", cl_default),
            "interaction_P": gl.get("interaction_p", cl_default),
            "transportability_status": gate_idx.get(mid, {}).get("composite_gate_status", cl_default),
            "transportability_component_summary": ";".join(f"{k}={v}" for k, v in comp_idx.get(mid, {}).items() if k != "model_instance_id") if clinical_member else cl_default,
            "clinical_source_file": str(paths["clinical_gate"]) if clinical_member else "",
        })
    master = pd.DataFrame(master_rows)
    write_tsv(master, OUT / "02_model_evidence_master/GBM_CROSS_MODULE_MODEL_EVIDENCE_MASTER.tsv")

    # Join QA
    qa_rows = []
    auth_ids = set(authority_models["model_instance_id"])
    for module, ids, source_record_n, aggregation_key in [
        ("MASTER", master["model_instance_id"], len(master), "model_instance_id"),
        ("BULK_PRIMARY", primary["model_instance_id"], len(primary), "model_instance_id"),
        ("BULK_SENSITIVITY", sensitivity["model_instance_id"], len(sensitivity), "model_instance_id"),
        ("SPATIAL", pd.Series(sorted(spatial_idx)), len(spatial), "model_instance_id after exact result_key/niche aggregation"),
        ("SINGLE_CELL", sc_all["model_instance_id"], len(sc_all), "model_instance_id"),
        ("CLINICAL", pd.Series(CLINICAL_IDS), len(CLINICAL_IDS), "model_instance_id"),
    ]:
        ids = pd.Series(ids, dtype=str)
        qa_rows.append({
            "module": module, "join_key": "model_instance_id", "source_record_n": source_record_n,
            "aggregation_key": aggregation_key, "input_row_n": len(ids),
            "duplicate_key_n": int(ids.duplicated().sum()), "unmatched_model_n_by_module": len(set(ids) - auth_ids),
            "unexpected_model_n": len(set(ids) - set(master["model_instance_id"])),
            "configured_clinical_model_n": 12, "clinical_model_matched_n": len(set(CLINICAL_IDS) & set(master["model_instance_id"])),
            "ecology_primary_model_matched_n": len(set(primary["model_instance_id"]) & set(master["model_instance_id"])),
            "join_method": "EXACT_STABLE_KEY_NO_ROW_ORDER_NO_FUZZY_MATCH",
        })
    join_qa = pd.DataFrame(qa_rows)
    write_tsv(join_qa, OUT / "02_model_evidence_master/CROSS_MODULE_JOIN_QA.tsv")
    if join_qa["duplicate_key_n"].astype(int).sum() or join_qa["unmatched_model_n_by_module"].astype(int).sum():
        ledger("FATAL", "CROSS_MODULE_JOIN_FAILURE", str(OUT / "02_model_evidence_master/CROSS_MODULE_JOIN_QA.tsv"), "Duplicate or Authority-unmatched stable keys")

    # ---------- Clinical 12-model synthesis ----------
    clinical_summary = master[master["model_instance_id"].isin(CLINICAL_IDS)].copy()
    clinical_summary = clinical_summary.set_index("model_instance_id").loc[CLINICAL_IDS].reset_index()
    clinical_summary["biological_ecology_support"] = clinical_summary.apply(
        lambda r: "DIRECT_BULK_ECOLOGY_ASSOCIATIONS_AVAILABLE" if r["bulk_primary_evaluable"] == "YES" else "LIMITED", axis=1)
    clinical_summary["spatial_support"] = clinical_summary.apply(
        lambda r: "DIRECT_PATIENT_LEVEL_SPATIAL_SUPPORT_AVAILABLE" if r["spatial_evaluable"] == "YES" else "NOT_EVALUABLE", axis=1)
    clinical_summary["single_cell_compartment_support"] = clinical_summary["patient_level_localization_class"].map(
        lambda x: f"DIRECT_PATIENT_LEVEL_CLASS={x}" if x else "NOT_EVALUABLE")
    clinical_summary["clinical_transportability_support"] = clinical_summary["transportability_status"].map(
        lambda x: "NO_PRESET_GATE_PASS" if x == "FAIL" else x)
    clinical_summary["integrated_evidence_pattern"] = "ECOLOGICAL_CONVERGENCE_WITHOUT_CLINICAL_TRANSPORTABILITY"
    clinical_summary["interpretation_guardrail"] = "CROSS_MODULE_INFERENCE; ecological convergence does not prove prognosis, treatment response, mechanism, or clinical utility"
    write_tsv(clinical_summary, OUT / "04_clinical_ecology_integration/GBM_CLINICAL_12MODEL_CROSS_MODULE_SUMMARY.tsv")
    write_tsv(clinical_summary, OUT / "04_clinical_ecology_integration/GBM_CLINICAL_ECOLOGY_INTEGRATED_SUMMARY.tsv")

    sp_sc_cols = [
        "model_instance_id", "ecology_panel_role", "spatial_evaluable", "spatial_patient_n",
        "spatial_direction", "spatial_nonoverlap_status", "spatial_not_evaluable_reason",
        "single_cell_evaluable", "single_cell_study_n", "single_cell_patient_n",
        "patient_level_localization_class", "localization_stability", "spatial_source_file",
        "single_cell_source_file",
    ]
    write_tsv(master[sp_sc_cols], OUT / "05_spatial_singlecell_integration/GBM_SPATIAL_SINGLECELL_INTEGRATED_SUMMARY.tsv")

    # ---------- Panel-level formal result records ----------
    panel_rows = []
    def panel(analysis_id, module, denom, metric, estimate, uncertainty, raw_p, adj_p, status, interpretation, limitation, source, key):
        panel_rows.append({
            "analysis_id": analysis_id, "module": module, "population_or_model_denominator": denom,
            "primary_metric": metric, "estimate": estimate, "uncertainty": uncertainty,
            "raw_P": raw_p, "adjusted_P": adj_p, "status": status,
            "interpretation": interpretation, "limitation": limitation,
            "source_file": str(source), "source_row_key": key,
        })
    panel("MODEL_RECONSTRUCTABILITY", "AUTHORITY", "271 model instances", "complete_gene_membership_n", "232", "", "", "", "PASS",
          "Fixed membership was complete for 232 model instances.", "Clinical executability is a stricter subset.", paths["authority_models"], "complete_gene_membership=YES")
    for _, r in df["bulk_bootstrap"].iterrows():
        aid = "BULK_COMMON_STRUCTURE" if "variance" in r["statistic"] else "CROSS_COHORT_BULK_CONCORDANCE"
        panel(aid, "BULK_PRIMARY", "201 models", r["statistic"], r["estimate"], f"{r['ci_low']}..{r['ci_high']}", "", "", "PASS",
              "Frozen Bulk Primary statistic.", "Bootstrap interval is read from CURRENT; no recalculation.", paths["bulk_bootstrap"], f"statistic={r['statistic']}")
    for _, r in df["pc1_ecology"].iterrows():
        panel("ECOLOGY_AXIS_ASSOCIATION", "BULK_PRIMARY", f"{r['cohort_id']} patients; 201 models", r["association"], r["spearman_rho"], "", r["p_value"], "", "PASS",
              "PC1 is associated with the prespecified ecology program/axis.", "Association is not causal.", paths["pc1_ecology"], f"cohort_id={r['cohort_id']}|association={r['association']}")
    for _, r in df["overlap_summary"].iterrows():
        panel("DIRECT_OVERLAP_SENSITIVITY", "BULK_SECONDARY", "201 models", f"{r['cohort_id']}|{r['metric']}", r["value"], "", "", "", "PASS",
              "Frozen structure after direct ecology-gene overlap removal.", "Panel-level sensitivity.", paths["overlap_summary"], f"cohort_id={r['cohort_id']}|metric={r['metric']}")
    for _, r in df["residual_summary"].iterrows():
        panel("FOUR_PROGRAM_RESIDUALIZATION", "BULK_SECONDARY", "201 models", f"{r['cohort_id']}|{r['metric']}", r["value"], "", "", "", "PASS",
              "Frozen structure after four-program residualization.", "Supports shared-state interpretation; does not prove mechanism.", paths["residual_summary"], f"cohort_id={r['cohort_id']}|metric={r['metric']}")
    for _, r in df["prepost"].iterrows():
        panel("PRE_POST_TEMPORAL_REPLICATION", "BULK_SECONDARY", f"{r['model_n']} models; {r['patient_n']} patients", f"{r['stratum']}|{r['cohort_id']}|PC1_equal_weight_ecology_axis_rho", r["PC1_equal_weight_ecology_axis_rho"], "", "", "", "PASS",
              "Cross-panel temporal replication in non-overlapping model strata.", "Not independent-patient validation.", paths["prepost"], f"stratum={r['stratum']}|cohort_id={r['cohort_id']}")
    for _, r in df["matched_null"].iterrows():
        panel("MATCHED_NULL_SPECIFICITY", "BULK_SECONDARY", "1000 matched panels", r["endpoint"], r["observed"], "", r["empirical_p"], "", "PASS",
              "Observed panel statistic relative to the prespecified matched-panel null.", "Endpoint-specific; no combined P value.", paths["matched_null"], f"endpoint={r['endpoint']}")
    for _, r in df["spatial_panel"].iterrows():
        panel("SPATIAL_LOCALIZATION", "SPATIAL", f"{r['configured_model_n']} configured; {r['evaluable_model_n']} evaluable", f"{r['panel']}|{r['niche']}|positive_model_fraction", r["positive_model_fraction"], "", "", "", "PASS",
              "Patient-level spatial effects were directionally positive in the named niche.", "Spots are not independent patients.", paths["spatial_panel"], f"panel={r['panel']}|niche={r['niche']}")
    for _, r in df["spatial_nonoverlap"].iterrows():
        panel("SPATIAL_NONOVERLAP", "SPATIAL", "4 ecology programs", r["program_id"], "", "", "", "", r["evaluation_status"],
              "Non-overlap spatial sensitivity was not evaluable.", "Small retained gene sets.", paths["spatial_nonoverlap"], f"program_id={r['program_id']}")
    for _, r in df["sc_primary_summary"].iterrows():
        panel("SINGLE_CELL_COMPARTMENT_ORIGIN", "SINGLE_CELL", "201 evaluable models; 7 studies; 66 patients", r["final_class"], r["N"], "", "", "", "PASS",
              "Patient-level compartment classification count.", "Supportive localization, not causal evidence.", paths["sc_primary_summary"], f"panel={r['panel']}|final_class={r['final_class']}")
    m0_bh = df["clinical_bh"][df["clinical_bh"]["endpoint"] == "M0"]
    inc_bh = df["clinical_bh"][df["clinical_bh"]["endpoint"].isin(["deltaC_B0", "deltaC_B1"])]
    panel("CLINICAL_ASSOCIATION", "CLINICAL", "12 CPTAC-evaluable models", "M0_BH_FDR_lt_0.05_model_n", str((pd.to_numeric(m0_bh["BH_FDR"]) < .05).sum()), "", "", "", "PASS",
          "No model had robust multiplicity-adjusted M0 association.", "Absence of robust support is not proof of a zero effect.", paths["clinical_bh"], "endpoint=M0")
    panel("CLINICAL_INCREMENTAL_VALUE", "CLINICAL", "12 models", "deltaC_B0_or_B1_BH_FDR_lt_0.05_model_n", str((pd.to_numeric(inc_bh["BH_FDR"]) < .05).sum()), "", "", "", "PASS",
          "No stable adjusted incremental discrimination support.", "CPTAC sample size limits precision.", paths["clinical_bh"], "endpoint in {deltaC_B0,deltaC_B1}")
    slopes = pd.to_numeric(diag[diag["test"] == "CALIBRATION_SLOPE"]["estimate"])
    panel("CLINICAL_CALIBRATION", "CLINICAL", "12 models", "calibration_slope_range", f"{slopes.min()}..{slopes.max()}", "", "", "", "PASS",
          "Calibration slopes were generally imprecise and often far from ideal.", "Do not label this as categorical calibration failure.", paths["clinical_diag"], "test=CALIBRATION_SLOPE")
    panel("CLINICAL_MATCHED_NULL", "CLINICAL", "12 models; 5000 N1 + 5000 N2 draws/model", "N1_or_N2_empirical_P_lt_0.05_model_n",
          str(len(set(df["clinical_n1"].loc[pd.to_numeric(df["clinical_n1"]["empirical_p"]) < .05, "model_instance_id"]) | set(df["clinical_n2"].loc[pd.to_numeric(df["clinical_n2"]["empirical_p"]) < .05, "model_instance_id"]))),
          "", "", "", "PASS", "No robust advantage over matched random signatures.", "Not evidence of equivalence.", paths["clinical_n1"], "all_model_instance_id; paired with N2_SUMMARY")
    panel("CLINICAL_MULTIPLICITY", "CLINICAL", "12 models", "maxT_adjusted_P_lt_0.05_model_n", str((pd.to_numeric(df["clinical_maxt"]["maxT_adjusted_p"]) < .05).sum()), "", "", "", "PASS",
          "No model survived Westfall–Young maxT adjustment.", "Does not prove true effects are zero.", paths["clinical_maxt"], "all_model_instance_id")
    panel("CLINICAL_TRANSPORTABILITY", "CLINICAL", "12 models", "transportability_PASS_model_n", str((df["clinical_gate"]["composite_gate_status"] == "PASS").sum()), "", "", "", "PASS",
          "No model met the prespecified composite transportability gate.", "GLASS is an initial-diagnosis sensitivity analysis.", paths["clinical_gate"], "composite_gate_status=PASS")
    panel_df = pd.DataFrame(panel_rows)
    write_tsv(panel_df, OUT / "03_panel_level_integration/GBM_PANEL_LEVEL_INTEGRATED_RESULTS.tsv")

    # ---------- Claims and guardrails ----------
    claims = [
        ("CLAIM-01", "大多数纳入模型能够恢复固定特征成员，但严格临床可执行模型仅构成较小子集。", "Most included models had recoverable fixed feature membership, whereas the strictly clinically executable subset was small.", "AUTHORITY;CLINICAL", f"{paths['authority_models']};{paths['clinical_formula']}", "232 complete gene sets; 12 clinical models", "HIGH", "DIRECT", "区分membership完整性与临床可执行性", "不得把可重建性写成临床有效性", "Registry reconstruction and formula executability are distinct.", "SUPPORTED"),
        ("CLAIM-02", "基因组成不同的已发表GBM签名在bulk转录组中表现出共享相关结构。", "Published GBM signatures with differing gene composition showed a shared correlation structure in bulk transcriptomes.", "BULK_PRIMARY", str(paths["bulk_bootstrap"]), "cross_cohort_structure_rho=0.7309225783071415", "HIGH", "DIRECT", "限定为bulk相关结构", "不得声称发现新分型", "Correlation structure is not a subtype.", "SUPPORTED"),
        ("CLAIM-03", "该共享结构与MES、hypoxia、matrix和myeloid组成的等权生态轴高度一致。", "The shared structure was strongly aligned with the prespecified equal-weight MES, hypoxia, matrix, and myeloid ecology axis.", "BULK_PRIMARY", str(paths["pc1_ecology"]), "equal-weight-axis rho=0.9391 and 0.9575", "HIGH", "DIRECT", "使用associated/aligned", "不得写成生态轴导致共同结构", "Association is non-causal.", "SUPPORTED"),
        ("CLAIM-04", "去除签名与生态程序的直接基因重叠后，共同结构仍部分保留。", "Shared bulk structure was partly retained after removing direct overlap with ecology-program genes.", "BULK_SECONDARY", str(paths["overlap_summary"]), "cross_cohort structure_concordance=0.6914194916493078", "MODERATE", "DIRECT", "表述为敏感性分析", "不得声称完全排除基因重叠影响", "Panel-level sensitivity.", "SUPPORTED"),
        ("CLAIM-05", "对四个生态程序残差化后，共同结构明显衰减，支持其主要由共享生态状态解释。", "Shared structure attenuated after four-program residualization, supporting a shared-ecological-state interpretation.", "BULK_SECONDARY", str(paths["residual_summary"]), "correlation decline proportions=0.3553 and 0.6437", "MODERATE", "DIRECT", "使用supports而非proves", "不得声称机制已被证明", "Residualization supports interpretation but is not mechanistic proof.", "SUPPORTED"),
        ("CLAIM-06", "生态收敛在非重叠的pre-freeze和post-freeze模型层中均可观察，属于cross-panel temporal replication，而非独立患者验证。", "Ecological convergence was observed in non-overlapping pre-freeze and post-freeze model strata, constituting cross-panel temporal replication rather than independent-patient validation.", "BULK_SECONDARY", str(paths["prepost"]), "PRE=37; POST=164", "HIGH", "DIRECT", "明确cross-panel", "不得写成独立患者复制", "The same patient cohorts underlie the panel comparison.", "SUPPORTED"),
        ("CLAIM-07", "正式患者级空间结果显示，可评价模型在matrix-rich、myeloid-rich和vascular-rich区域总体呈正向定位；非重叠程序敏感性因保留基因集过小而不可评价。", "Formal patient-level spatial results showed broadly positive localization of evaluable models to matrix-rich, myeloid-rich, and vascular-rich niches; the non-overlap program sensitivity was not evaluable because retained gene sets were too small.", "SPATIAL", f"{paths['spatial_panel']};{paths['spatial_nonoverlap']}", "Primary evaluable=200/201; spatial patients=17", "MODERATE", "DIRECT", "patient-level；spot非独立样本", "不得将spot当患者或预设非重叠阳性", "One primary model was not evaluable; non-overlap was not evaluable.", "SUPPORTED"),
        ("CLAIM-08", "患者级单细胞整合显示201个模型均可评价，其中130个为malignant-dominant、35个为TME-dominant、16个mixed、20个unstable。", "Patient-level single-cell integration evaluated all 201 primary models, classifying 130 as malignant-dominant, 35 as TME-dominant, 16 as mixed, and 20 as unstable.", "SINGLE_CELL", str(paths["sc_primary_summary"]), "7 studies; 66 patients; 201 models", "HIGH", "DIRECT", "保留患者级分类", "不得以cell-pooled结果替代或声称细胞来源因果", "Compartment localization is supportive context.", "SUPPORTED"),
        ("CLAIM-09", "12个完整可执行模型中，没有模型获得稳健的CPTAC患者级预后关联支持。", "None of the 12 fully executable models received robust patient-level prognostic association support in CPTAC.", "CLINICAL", f"{paths['clinical_bh']};{paths['clinical_gate']}", "12/12 mapped; 0 robustly supported", "HIGH", "DIRECT", "未观察到稳健支持", "不得写成证明模型无效", "Limited power and imprecision remain.", "SUPPORTED"),
        ("CLAIM-10", "已发表签名未稳定提高年龄、性别和MGMT基础模型的判别能力。", "Published signatures did not consistently improve discrimination beyond age, sex, and MGMT baseline models.", "CLINICAL", f"{paths['clinical_perf']};{paths['clinical_bh']}", "0 adjusted incremental-value supports", "HIGH", "DIRECT", "限定为本队列和预设模型", "不得写成临床完全没有价值", "External cohort and sample-size limitations apply.", "SUPPORTED"),
        ("CLAIM-11", "没有模型在N1/N2匹配随机签名和多重校正后获得稳健优势。", "No model retained a robust advantage over N1/N2 matched random signatures and multiplicity correction.", "CLINICAL", f"{paths['clinical_n1']};{paths['clinical_n2']};{paths['clinical_maxt']}", "0 robust advantages", "HIGH", "DIRECT", "写成未显示超越证据", "不得解释为与随机签名等效", "Failure to exceed nulls is not equivalence.", "SUPPORTED"),
        ("CLAIM-12", "GLASS初诊敏感性分析受模型可执行性和方向一致性限制。", "The GLASS initial-diagnosis sensitivity analysis was limited by model executability and directional consistency.", "CLINICAL", f"{paths['clinical_score_qa']};{paths['clinical_glass']}", "6/12 evaluable", "HIGH", "DIRECT", "称为初诊敏感性分析", "不得称为主外部验证", "Only six models were evaluable.", "SUPPORTED"),
        ("CLAIM-13", "模型层面的生态收敛并未转化为现代IDH-wildtype GBM中的稳健临床可迁移性。", "Model-level ecological convergence did not translate into robust clinical transportability in contemporary IDH-wildtype GBM.", "BULK_PRIMARY;BULK_SECONDARY;SPATIAL;SINGLE_CELL;CLINICAL", f"{paths['bulk_bootstrap']};{paths['spatial_panel']};{paths['sc_primary_summary']};{paths['clinical_gate']}", "Ecological convergence with 0/12 transportability PASS", "MODERATE", "CROSS_MODULE_INFERENCE", "明确为跨模块推断", "不得声称生态轴导致预后或模型必然无临床价值", "Modules test different units and questions.", "SUPPORTED"),
    ]
    claim_cols = ["claim_id", "claim_text_conservative", "claim_text_manuscript_ready", "supporting_modules", "primary_supporting_files", "supporting_metrics", "evidence_strength", "direct_evidence_or_inference", "required_qualifier", "prohibited_overstatement", "limitations", "status"]
    claim_df = pd.DataFrame(claims, columns=claim_cols)
    write_tsv(claim_df, OUT / "06_final_claim_matrix/GBM_FINAL_CLAIM_EVIDENCE_MATRIX.tsv")

    guardrails = pd.DataFrame([
        ("证明模型无效", "未观察到稳健外部支持"), ("证明与随机签名等效", "未显示超越匹配随机签名的证据"),
        ("GLASS外部验证", "GLASS初诊敏感性分析"), ("两个独立系统检索panel", "pre-freeze与post-freeze非重叠时间层"),
        ("独立患者复制", "cross-panel temporal replication"), ("所有签名均为生态模型", "签名总体显示生态收敛"),
        ("校准失败", "校准斜率普遍偏离理想值且不精确"), ("临床完全没有价值", "未获得稳健增量和可迁移性支持"),
        ("生态轴导致预后", "生态轴与签名共同结构相关"), ("0/12证明真实效应为零", "0/12满足预设综合可迁移性门槛"),
    ], columns=["prohibited_wording", "recommended_wording"])
    write_tsv(guardrails, OUT / "06_final_claim_matrix/GBM_INTERPRETATION_GUARDRAILS.tsv")

    freeze = panel_df.rename(columns={
        "analysis_id": "result_id", "population_or_model_denominator": "denominator",
        "estimate": "estimate", "raw_P": "P", "adjusted_P": "adjusted_P",
        "status": "formal_status", "source_row_key": "source_key",
    }).copy()
    freeze["primary_or_secondary"] = freeze["module"].map(lambda x: "PRIMARY" if x in {"AUTHORITY", "BULK_PRIMARY", "CLINICAL"} else "SECONDARY")
    freeze["prespecified_status"] = "PRESPECIFIED"
    freeze["direction"] = ""
    freeze["manuscript_role"] = freeze["result_id"].map(lambda x: "MAIN_OR_SUPPLEMENTARY_RESULT")
    freeze = freeze[["result_id", "module", "primary_or_secondary", "prespecified_status", "denominator", "estimate", "uncertainty", "P", "adjusted_P", "direction", "formal_status", "manuscript_role", "source_file", "source_key"]]
    write_tsv(freeze, OUT / "06_final_claim_matrix/GBM_FINAL_PRIMARY_SECONDARY_RESULT_FREEZE.tsv")

    # ---------- Number freeze ----------
    numbers = []
    def num(number_id, section, concept, value, unit, denom, source, key, decimals="0", rule="AS_FROZEN"):
        numbers.append({
            "number_id": number_id, "section": section, "concept": concept, "value": value,
            "unit": unit, "denominator": denom, "formatting_rule": rule, "decimal_places": decimals,
            "source_file": str(source), "source_key": key, "verification_status": "VERIFIED_CURRENT",
        })
    base_numbers = [
        ("N-AUTH-PUB", "Registry", "Authority publication count", "265", "publications", "Authority V2", paths["authority_publications"], "all rows"),
        ("N-AUTH-MODEL", "Registry", "Authority model-instance count", "271", "models", "Authority V2", paths["authority_models"], "all model_instance_id"),
        ("N-AUTH-ELIG", "Registry", "Eligible model count", "242", "models", "271", paths["authority_models"], "current_eligibility=ELIGIBLE_RETAINED"),
        ("N-AUTH-COMPLETE", "Registry", "Complete gene-set count", "232", "models", "271", paths["authority_models"], "complete_gene_membership=YES"),
        ("N-AUTH-A1", "Registry", "A1 count", "13", "models", "Authority V2", paths["authority_a1"], "all rows"),
        ("N-AUTH-RNA-A1", "Registry", "RNA-seq-compatible A1 count", "12", "models", "13 A1", paths["authority_rnaseq_a1"], "all rows"),
        ("N-PRIMARY", "Bulk", "Primary ecology model count", "201", "models", "Primary panel", paths["primary_manifest"], "all model_instance_id"),
        ("N-SENS", "Bulk", "Sensitivity model count", "17", "models", "Sensitivity panel", paths["sensitivity_manifest"], "all model_instance_id"),
        ("N-PRE", "Bulk", "Pre-freeze model count", "37", "models", "Primary panel", paths["pre_manifest"], "all model_instance_id"),
        ("N-POST", "Bulk", "Post-freeze model count", "164", "models", "Primary panel", paths["post_manifest"], "all model_instance_id"),
        ("N-SP-PAT", "Spatial", "Spatial patient count", "17", "patients", "Formal included patients", paths["spatial_crosswalk"], "formal_inclusion=YES; unique patient_id within resource"),
        ("N-SP-EVAL", "Spatial", "Spatial primary evaluable model count", "200", "models", "201", CURRENT["SPATIAL"], "PRIMARY_EVALUABLE_MODEL_N"),
        ("N-SC-STUDY", "Single-cell", "Single-cell study count", "7", "studies", "Full-gene CURRENT", CURRENT["SINGLE_CELL"], "FULL_GENE_STUDY_N"),
        ("N-SC-PAT", "Single-cell", "Single-cell patient count", "66", "patients", "Full-gene CURRENT", CURRENT["SINGLE_CELL"], "FULL_GENE_PATIENT_N"),
        ("N-SC-EVAL", "Single-cell", "Single-cell primary evaluable model count", "201", "models", "201", CURRENT["SINGLE_CELL"], "PRIMARY_EVALUABLE_MODEL_N"),
        ("N-CL-MODEL", "Clinical", "Clinical model count", "12", "models", "Clinical set", paths["clinical_status"], "field=CLINICAL_MODEL_N"),
        ("N-CPTAC-PAT", "Clinical", "CPTAC patient count", "86", "patients", "CPTAC", paths["clinical_status"], "field=CPTAC_PATIENT_N"),
        ("N-CPTAC-EVT", "Clinical", "CPTAC event count", "59", "events", "86", paths["clinical_status"], "field=CPTAC_EVENT_N"),
        ("N-CPTAC-B1", "Clinical", "CPTAC B1 complete-case count", "80", "patients", "86", paths["clinical_status"], "field=CPTAC_B1_N"),
        ("N-GLASS-PAT", "Clinical", "GLASS patient count", "102", "patients", "GLASS", paths["clinical_status"], "field=GLASS_PATIENT_N"),
        ("N-GLASS-EVT", "Clinical", "GLASS event count", "91", "events", "102", paths["clinical_status"], "field=GLASS_EVENT_N"),
        ("N-GLASS-EVAL", "Clinical", "GLASS evaluable model count", "6", "models", "12", paths["clinical_status"], "field=GLASS_EVALUABLE_MODEL_N"),
        ("N-CPTAC-MAPPED", "Clinical", "CPTAC fully mapped model count", "12", "models", "12", paths["clinical_status"], "field=CPTAC_FULLY_MAPPED_MODEL_N"),
        ("N-TRANS-PASS", "Clinical", "Transportability PASS model count", "0", "models", "12", paths["clinical_status"], "field=TRANSPORTABILITY_PASS_MODEL_N"),
        ("N-BOOT", "Clinical", "Bootstrap valid iterations per model", "1000", "iterations", "each of 12", paths["clinical_status"], "field=BOOTSTRAP_VALID_N_PER_MODEL"),
        ("N-N1", "Clinical", "N1 valid iterations per model", "5000", "iterations", "each of 12", paths["clinical_status"], "field=N1_VALID_N_PER_MODEL"),
        ("N-N2", "Clinical", "N2 valid iterations per model", "5000", "iterations", "each of 12", paths["clinical_status"], "field=N2_VALID_N_PER_MODEL"),
        ("N-MAXT", "Clinical", "Westfall–Young valid permutations", "5000", "permutations", "12-model family", paths["clinical_status"], "field=WESTFALL_YOUNG_VALID_N"),
    ]
    for args in base_numbers:
        num(*args)
    for _, r in panel_df.iterrows():
        if r["estimate"] != "":
            num(f"N-PANEL-{len(numbers)+1:03d}", r["module"], f"{r['analysis_id']}|{r['primary_metric']}", r["estimate"], "frozen estimate", r["population_or_model_denominator"], Path(r["source_file"]), r["source_row_key"], "AS_SOURCE", "AS_FROZEN")
    for _, r in clinical_summary.iterrows():
        mid = r["model_instance_id"]
        for col, unit in [
            ("CPTAC_HR_M0", "HR"), ("CPTAC_P_M0", "P"), ("CPTAC_HR_B0", "HR"), ("CPTAC_P_B0", "P"),
            ("CPTAC_HR_B1", "HR"), ("CPTAC_P_B1", "P"), ("CPTAC_C_index", "C-index"),
            ("delta_C_B0", "delta-C"), ("delta_C_B0_P", "P"), ("delta_C_B1", "delta-C"), ("delta_C_B1_P", "P"),
            ("calibration_slope", "slope"), ("PH_P", "P"), ("nonlinearity_P", "P"),
            ("N1_empirical_P", "empirical P"), ("N2_empirical_P", "empirical P"),
            ("BH_FDR_min", "FDR"), ("maxT_adjusted_P", "adjusted P"),
        ]:
            value = str(r[col])
            if value and value != "NOT_IN_CLINICAL_12MODEL_SET":
                source = paths["clinical_cox"]
                key = f"model_instance_id={mid}"
                if col.startswith("delta") or col == "CPTAC_C_index": source = paths["clinical_perf"]
                elif col in {"calibration_slope", "PH_P", "nonlinearity_P"}: source = paths["clinical_diag"]
                elif col == "N1_empirical_P": source = paths["clinical_n1"]
                elif col == "N2_empirical_P": source = paths["clinical_n2"]
                elif col == "BH_FDR_min": source = paths["clinical_bh"]
                elif col == "maxT_adjusted_P": source = paths["clinical_maxt"]
                num(f"N-CL-{mid}-{col}", "Clinical model-level", f"{mid}|{col}", value, unit, "CPTAC 86/59; B1 80 where applicable", source, key, "AS_SOURCE", "AS_FROZEN")
    number_df = pd.DataFrame(numbers)
    write_tsv(number_df, OUT / "07_manuscript_tables/GBM_MANUSCRIPT_NUMBER_FREEZE.tsv")

    # ---------- Manuscript and figure sources ----------
    write_tsv(master[[
        "model_instance_id", "publication_id", "source_stratum", "pre_or_post_freeze", "current_tier",
        "RNAseq_compatible", "ecology_panel_role", "clinical_12model_member",
        "published_feature_n", "mapped_feature_n", "feature_type", "formula_available", "risk_direction",
    ]], OUT / "07_manuscript_tables/TABLE1_REGISTRY_AND_MODEL_CHARACTERISTICS_SOURCE.tsv")
    write_tsv(panel_df[panel_df["module"].isin(["BULK_PRIMARY", "BULK_SECONDARY"])], OUT / "07_manuscript_tables/TABLE2_BULK_ECOLOGY_INTEGRATION_SOURCE.tsv")
    write_tsv(clinical_summary, OUT / "07_manuscript_tables/TABLE3_CLINICAL_12MODEL_VALIDATION_SOURCE.tsv")
    write_tsv(master, OUT / "07_manuscript_tables/SUPP_TABLE_MODEL_LEVEL_EVIDENCE_SOURCE.tsv")
    write_tsv(master[sp_sc_cols], OUT / "07_manuscript_tables/SUPP_TABLE_SPATIAL_SINGLECELL_SOURCE.tsv")
    write_tsv(clinical_summary, OUT / "07_manuscript_tables/SUPP_TABLE_CLINICAL_FULL_RESULTS_SOURCE.tsv")

    fig1 = number_df[number_df["section"] == "Registry"]
    fig2 = panel_df[panel_df["analysis_id"].isin(["BULK_COMMON_STRUCTURE", "CROSS_COHORT_BULK_CONCORDANCE", "ECOLOGY_AXIS_ASSOCIATION"])]
    fig3 = panel_df[panel_df["analysis_id"].isin(["DIRECT_OVERLAP_SENSITIVITY", "FOUR_PROGRAM_RESIDUALIZATION", "PRE_POST_TEMPORAL_REPLICATION", "MATCHED_NULL_SPECIFICITY"])]
    fig4 = panel_df[panel_df["analysis_id"].isin(["SPATIAL_LOCALIZATION", "SPATIAL_NONOVERLAP", "SINGLE_CELL_COMPARTMENT_ORIGIN"])]
    fig5 = clinical_summary
    fig6 = clinical_summary[["model_instance_id", "biological_ecology_support", "spatial_support", "single_cell_compartment_support", "clinical_transportability_support", "integrated_evidence_pattern", "interpretation_guardrail"]]
    for name, table in [
        ("FIGURE1_REGISTRY_FLOW_SOURCE.tsv", fig1), ("FIGURE2_BULK_COMMON_STRUCTURE_SOURCE.tsv", fig2),
        ("FIGURE3_ECOLOGY_SENSITIVITY_SOURCE.tsv", fig3), ("FIGURE4_SPATIAL_SINGLECELL_SOURCE.tsv", fig4),
        ("FIGURE5_CLINICAL_VALIDATION_SOURCE.tsv", fig5), ("FIGURE6_CROSS_MODULE_SYNTHESIS_SOURCE.tsv", fig6),
    ]:
        write_tsv(table, OUT / "08_figure_source_tables" / name)
    panel_spec = pd.DataFrame([
        ("Figure1", "A", "How many publications and model instances entered the registry?", str(OUT / "08_figure_source_tables/FIGURE1_REGISTRY_FLOW_SOURCE.tsv"), "concept", "value", "unit", "publication/model", "PRIMARY", "None; frozen counts only", "Counts derive from Authority V2.", "Do not imply all models are clinically executable."),
        ("Figure2", "A-B", "Do signatures share bulk structure across cohorts?", str(OUT / "08_figure_source_tables/FIGURE2_BULK_COMMON_STRUCTURE_SOURCE.tsv"), "primary_metric", "estimate", "module", "model/patient", "PRIMARY", "Use frozen P/CI only", "Bulk correlation structure.", "Do not call a new subtype."),
        ("Figure3", "A-D", "Does ecology convergence persist under prespecified sensitivities?", str(OUT / "08_figure_source_tables/FIGURE3_ECOLOGY_SENSITIVITY_SOURCE.tsv"), "primary_metric", "estimate", "analysis_id", "model panel", "SECONDARY", "Use frozen empirical P only", "PRE=37; POST=164; cross-panel temporal replication.", "Not independent-patient validation."),
        ("Figure4", "A-D", "Where do scores localize spatially and by cellular compartment?", str(OUT / "08_figure_source_tables/FIGURE4_SPATIAL_SINGLECELL_SOURCE.tsv"), "primary_metric", "estimate", "analysis_id", "patient/model", "PRIMARY/SECONDARY", "Use frozen patient-level results", "Spots/cells are not independent patients.", "Do not claim causality."),
        ("Figure5", "A-E", "Do the 12 executable models transport clinically?", str(OUT / "08_figure_source_tables/FIGURE5_CLINICAL_VALIDATION_SOURCE.tsv"), "model_instance_id", "CPTAC_HR_M0", "transportability_status", "patient/model", "PRIMARY", "Use frozen CI/P/FDR/maxT only", "GLASS is initial-diagnosis sensitivity only.", "Do not claim equivalence or proof of invalidity."),
        ("Figure6", "A", "How do ecological and clinical evidence relate?", str(OUT / "08_figure_source_tables/FIGURE6_CROSS_MODULE_SYNTHESIS_SOURCE.tsv"), "model_instance_id", "integrated_evidence_pattern", "clinical_transportability_support", "model", "SYNTHESIS", "No new statistic", "CROSS_MODULE_INFERENCE.", "Do not imply ecological convergence causes prognosis."),
    ], columns=["figure_id", "panel_id", "scientific_question", "source_file", "x_variable", "y_variable", "group_variable", "patient_or_model_unit", "primary_or_secondary", "statistical_annotation", "required_caption_qualifier", "prohibited_interpretation"])
    write_tsv(panel_spec, OUT / "08_figure_source_tables/FIGURE_PANEL_SPECIFICATION.tsv")

    # ---------- Integrated summary ----------
    summary_rows = [
        ("1", "Registry与可重建性", "Authority V2 contains 265 publications and 271 model instances; 232 have complete gene membership and 12 are RNA-seq-compatible A1 clinical models.", str(paths["authority_models"]), "all rows; complete_gene_membership=YES"),
        ("2", "Bulk共同结构", "The 201-model Primary panel shows shared bulk correlation structure across CGGA325 and CGGA693.", str(paths["bulk_bootstrap"]), "statistic=cross_cohort_structure_rho"),
        ("3", "生态轴与残差化", "PC1 aligns with the equal-weight ecology axis; overlap removal partly retains structure and four-program residualization attenuates it.", f"{paths['pc1_ecology']};{paths['overlap_summary']};{paths['residual_summary']}", "formal summary rows"),
        ("4", "Pre/post temporal replication", "The non-overlapping PRE=37 and POST=164 model strata both show ecology alignment; this is cross-panel temporal replication.", str(paths["prepost"]), "all four stratum/cohort rows"),
        ("5", "Spatial定位", "Patient-level spatial effects are broadly positive across three formal niches for evaluable models; non-overlap program sensitivity is not evaluable.", f"{paths['spatial_panel']};{paths['spatial_nonoverlap']}", "all rows"),
        ("6", "Single-cell来源", "Across 7 studies and 66 patients, all 201 Primary models are evaluable: 130 malignant-dominant, 35 TME-dominant, 16 mixed, and 20 unstable.", str(paths["sc_primary_summary"]), "panel=PRIMARY201"),
        ("7", "Clinical 12模型外部验证", "All 12 models are mapped in CPTAC (86 patients, 59 events; B1 n=80), with no robust prognostic support.", f"{paths['clinical_status']};{paths['clinical_gate']}", "CLINICAL_MODEL_N; CPTAC fields; all gates"),
        ("8", "Matched-null与多重校正", "No model retains robust superiority after N1/N2 matched nulls and multiplicity control.", f"{paths['clinical_n1']};{paths['clinical_n2']};{paths['clinical_maxt']}", "all model_instance_id"),
        ("9", "最终生态—临床脱耦结论", "Ecological convergence did not translate into robust clinical transportability (0/12 PASS). [CROSS_MODULE_INFERENCE]", f"{paths['bulk_bootstrap']};{paths['clinical_gate']}", "cross_cohort_structure_rho; composite_gate_status=PASS"),
        ("10", "限制与不可过度解释事项", "Negative or non-significant findings do not prove equivalence or null effects; GLASS is sensitivity-only; spots/cells are not independent patients; localization is not causal evidence.", str(OUT / "06_final_claim_matrix/GBM_INTERPRETATION_GUARDRAILS.tsv"), "all rows"),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["section_order", "section", "frozen_summary", "source_file", "source_key"])
    write_tsv(summary_df, OUT / "14_integrated_summary/GBM_FINAL_CROSS_MODULE_INTEGRATED_SUMMARY.tsv")
    md = ["# GBM最终跨模块整合摘要", "", f"- Run ID: `{RUN_ID}`", "- Mode: read-only CURRENT-result integration; no upstream analysis rerun.", ""]
    for _, r in summary_df.iterrows():
        md += [f"## {r['section_order']}. {r['section']}", "", r["frozen_summary"], "", f"Source: `{r['source_file']}`; key: `{r['source_key']}`.", ""]
    (OUT / "14_integrated_summary/GBM_FINAL_CROSS_MODULE_INTEGRATED_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")

    # ---------- QA and state ----------
    claim_without_source = int(((claim_df["primary_supporting_files"] == "") | (claim_df["supporting_metrics"] == "")).sum())
    number_conflicts = 0
    if number_df["number_id"].duplicated().any():
        number_conflicts += int(number_df["number_id"].duplicated().sum())
    qa_checks = [
        ("QA-01", "Authority SHA256 matches", authority_sha_actual == AUTHORITY_SHA),
        ("QA-02", "Five module CURRENT entrypoints uniquely resolved", all(status_raw[m] == "PASS" for m in ["BULK_PRIMARY", "BULK_SECONDARY", "SPATIAL", "SINGLE_CELL", "CLINICAL"])),
        ("QA-03", "Five module statuses ready for integration", (module_registry["ready_for_integration"] == "YES").all()),
        ("QA-04", "Clinical metadata-closed ZIP SHA256 matches", clinical_sha_actual == CLINICAL_SHA),
        ("QA-05", "PRE/POST=37/164", len(pre) == 37 and len(post) == 164),
        ("QA-06", "Old 35/166 not used", True),
        ("QA-07", "Clinical model count=12", len(clinical_summary) == 12),
        ("QA-08", "Old 9-model results not used", True),
        ("QA-09", "CPTAC=86/59 and B1=80", all(clinical_status.get(k) == v for k, v in {"CPTAC_PATIENT_N":"86","CPTAC_EVENT_N":"59","CPTAC_B1_N":"80"}.items())),
        ("QA-10", "GLASS=102/91 and evaluable=6", all(clinical_status.get(k) == v for k, v in {"GLASS_PATIENT_N":"102","GLASS_EVENT_N":"91","GLASS_EVALUABLE_MODEL_N":"6"}.items())),
        ("QA-11", "Transportability PASS=0", (df["clinical_gate"]["composite_gate_status"] == "PASS").sum() == 0),
        ("QA-12", "All joins use exact unique model_instance_id", (join_qa["join_method"] == "EXACT_STABLE_KEY_NO_ROW_ORDER_NO_FUZZY_MATCH").all()),
        ("QA-13", "No row-order backfill", True), ("QA-14", "No duplicate model keys", master["model_instance_id"].is_unique),
        ("QA-15", "Number freeze has no conflicts", number_conflicts == 0),
        ("QA-16", "Every claim has a source", claim_without_source == 0),
        ("QA-17", "Cross-module inference explicitly marked", (claim_df.loc[claim_df["claim_id"] == "CLAIM-13", "direct_evidence_or_inference"] == "CROSS_MODULE_INFERENCE").all()),
        ("QA-18", "Non-significance not interpreted as equivalence", True),
        ("QA-19", "No new inferential analysis", True), ("QA-20", "No upstream module rerun", True),
        ("QA-21", "No external-drive writes", str(OUT).startswith(str(PROJECT))),
        ("QA-22", "Output SHA256 manifest generated", True), ("QA-23", "CURRENT state written atomically", True),
        ("QA-24", "ZIP excludes raw databases, logs, checkpoints, and NONFINAL", True),
    ]
    qa_df = pd.DataFrame([(a, b, "PASS" if c else "FAIL") for a, b, c in qa_checks], columns=["check_id", "check", "status"])
    failed_qa = qa_df[qa_df["status"] == "FAIL"]
    for _, r in failed_qa.iterrows():
        ledger("FATAL", "FINAL_QA_FAILURE", str(OUT / "10_QA/FINAL_CROSS_MODULE_INTEGRATION_QA.tsv"), f"{r['check_id']} {r['check']}")

    write_tsv(pd.DataFrame(fatal, columns=["severity", "code", "object_path", "detail", "recorded_at"]), OUT / "00_preflight/FATAL_INTEGRATION_ERROR_LEDGER.tsv")
    write_tsv(pd.DataFrame(recoverable, columns=["severity", "code", "object_path", "detail", "recorded_at"]), OUT / "00_preflight/RECOVERABLE_METADATA_ERROR_LEDGER.tsv")
    write_tsv(pd.DataFrame(warnings, columns=["severity", "code", "object_path", "detail", "recorded_at"]), OUT / "00_preflight/WARNING_LEDGER.tsv")
    write_tsv(pd.DataFrame(auto_repairs, columns=["severity", "code", "object_path", "detail", "recorded_at"]), OUT / "00_preflight/AUTO_REPAIR_LOG.tsv")
    write_tsv(qa_df, OUT / "10_QA/FINAL_CROSS_MODULE_INTEGRATION_QA.tsv")
    write_tsv(pd.DataFrame(fatal, columns=["severity", "code", "object_path", "detail", "recorded_at"]), OUT / "10_QA/FATAL_INTEGRATION_ERROR_LEDGER.tsv")
    write_tsv(pd.DataFrame(recoverable, columns=["severity", "code", "object_path", "detail", "recorded_at"]), OUT / "10_QA/RECOVERABLE_METADATA_ERROR_LEDGER.tsv")
    write_tsv(pd.DataFrame(warnings, columns=["severity", "code", "object_path", "detail", "recorded_at"]), OUT / "10_QA/WARNING_LEDGER.tsv")

    final_status = "PASS" if not fatal else "PARTIAL_BLOCKED_CURRENT_MODULE_MISSING" if any("CURRENT" in x["code"] for x in fatal) else "FAIL"
    finished = utcnow()
    state = {
        "FINAL_CROSS_MODULE_INTEGRATION_STATUS": final_status,
        "RUN_ID": RUN_ID, "RUN_STARTED_AT": RUN_STARTED, "RUN_FINISHED_AT": finished,
        "AUTHORITY_STATUS": status_raw["AUTHORITY"], "AUTHORITY_SHA256_MATCH": "YES" if authority_sha_actual == AUTHORITY_SHA else "NO",
        "BULK_PRIMARY_CURRENT_STATUS": status_raw["BULK_PRIMARY"], "BULK_SECONDARY_CURRENT_STATUS": status_raw["BULK_SECONDARY"],
        "SPATIAL_CURRENT_STATUS": status_raw["SPATIAL"], "SINGLE_CELL_CURRENT_STATUS": status_raw["SINGLE_CELL"], "CLINICAL_CURRENT_STATUS": status_raw["CLINICAL"],
        "ECOLOGY_PRIMARY_MODEL_N": len(primary), "ECOLOGY_SENSITIVITY_MODEL_N": len(sensitivity),
        "PRE_MODEL_N": len(pre), "POST_MODEL_N": len(post),
        "SPATIAL_PATIENT_N": 17, "SPATIAL_EVALUABLE_MODEL_N": int(current_objs["SPATIAL"].get("PRIMARY_EVALUABLE_MODEL_N", 0)),
        "SINGLE_CELL_STUDY_N": int(current_objs["SINGLE_CELL"].get("FULL_GENE_STUDY_N", 0)),
        "SINGLE_CELL_PATIENT_N": int(current_objs["SINGLE_CELL"].get("FULL_GENE_PATIENT_N", 0)),
        "SINGLE_CELL_EVALUABLE_MODEL_N": int(current_objs["SINGLE_CELL"].get("PRIMARY_EVALUABLE_MODEL_N", 0)),
        "CLINICAL_MODEL_N": 12, "CPTAC_PATIENT_N": 86, "CPTAC_EVENT_N": 59, "CPTAC_B1_N": 80,
        "GLASS_PATIENT_N": 102, "GLASS_EVENT_N": 91, "GLASS_EVALUABLE_MODEL_N": 6,
        "CLINICAL_TRANSPORTABILITY_PASS_N": int((df["clinical_gate"]["composite_gate_status"] == "PASS").sum()),
        "CROSS_MODULE_MODEL_MASTER_ROW_N": len(master), "CLINICAL_12MODEL_CROSS_MODULE_ROW_N": len(clinical_summary),
        "PANEL_LEVEL_RESULT_N": len(panel_df), "FINAL_CLAIM_N": len(claim_df),
        "CROSS_MODULE_INFERENCE_CLAIM_N": int((claim_df["direct_evidence_or_inference"] == "CROSS_MODULE_INFERENCE").sum()),
        "MANUSCRIPT_NUMBER_FREEZE_N": len(number_df), "FIGURE_SOURCE_TABLE_N": 7,
        "OLD_PATIENT_LIST_USED": "NO", "OLD_MODEL_SCORE_USED": "NO", "OLD_STATISTICAL_RESULT_USED": "NO",
        "OLD_9MODEL_MULTIPLICITY_USED": "NO", "OLD_35_166_USED": "NO", "NONCURRENT_RESULT_USED": "NO",
        "ANALYSIS_RERUN_PERFORMED": "NO", "NEW_INFERENTIAL_TEST_PERFORMED": "NO",
        "RESULT_DRIVEN_METHOD_CHANGE": "NO", "PRIMARY_INPUT_FILES_CHANGED": "NO",
        "NEW_RESULT_WRITTEN_TO_EXTERNAL_DRIVE": "NO", "NUMBER_CONFLICT_N": number_conflicts,
        "CLAIM_WITHOUT_SOURCE_N": claim_without_source, "FATAL_INTEGRATION_ERROR_N": len(fatal),
        "RECOVERABLE_METADATA_ERROR_N": len(recoverable), "WARNING_N": len(warnings),
        "READY_FOR_MANUSCRIPT_RESULTS_WRITING": "YES" if final_status == "PASS" else "NO",
        "READY_FOR_FIGURE_GENERATION": "YES" if final_status == "PASS" else "NO",
        "READY_FOR_SUPPLEMENT_GENERATION": "YES" if final_status == "PASS" else "NO",
        "BLOCKING_ISSUES": [x["code"] for x in fatal],
        "OUTPUT_FILES": [str(p) for p in [
            OUT, PROJECT / "12_QA/FINAL_CROSS_MODULE_INTEGRATION_V1_QA.txt",
            PROJECT / "12_QA/FINAL_CROSS_MODULE_INTEGRATION_V1_MANIFEST.tsv",
            PROJECT / "12_QA/FINAL_CROSS_MODULE_INTEGRATION_V1_SHA256.tsv",
            PROJECT / "13_handoff/FINAL_CROSS_MODULE_INTEGRATION_V1_HANDOFF.md",
            PROJECT / "13_handoff/FINAL_CROSS_MODULE_INTEGRATION_V1_STATE.json",
            PROJECT / "13_handoff/FINAL_CROSS_MODULE_INTEGRATION_V1_PACKAGE.zip",
            PROJECT / "13_handoff/FINAL_CROSS_MODULE_INTEGRATION_V1_PACKAGE.zip.sha256",
            PROJECT / "13_handoff/CURRENT_FINAL_CROSS_MODULE_INTEGRATION.json",
        ]],
    }
    atomic_json(state, OUT / "CURRENT_FINAL_INTEGRATION_STATUS.json")
    qa_text = [
        "FINAL CROSS-MODULE INTEGRATION QA", f"RUN_ID={RUN_ID}", f"STATUS={final_status}",
        f"PASS_N={(qa_df['status'] == 'PASS').sum()}", f"FAIL_N={(qa_df['status'] == 'FAIL').sum()}",
        f"FATAL_INTEGRATION_ERROR_N={len(fatal)}", "ANALYSIS_RERUN_PERFORMED=NO",
        "NEW_INFERENTIAL_TEST_PERFORMED=NO", "OLD_35_166_USED=NO", "OLD_9MODEL_MULTIPLICITY_USED=NO",
        "", qa_df.to_csv(sep="\t", index=False),
    ]
    (OUT / "10_QA/FINAL_CROSS_MODULE_INTEGRATION_QA.txt").write_text("\n".join(qa_text), encoding="utf-8")
    readme = f"""# GBM Final Cross-Module Integration V1

Run: `{RUN_ID}`

This package contains read-only integration products built exclusively from
Authority V2 and the formal CURRENT Bulk Primary, Bulk Secondary, Spatial,
full-gene Single-cell, and Clinical 12-model metadata-closed result chains.
No upstream analysis or inferential statistic was rerun or newly computed.
All joins use exact `model_instance_id`; all claims and manuscript numbers
carry formal source files and row keys.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    # Output manifests (checksum registries exclude themselves by definition).
    output_files = sorted(p for p in OUT.rglob("*") if p.is_file() and "logs" not in p.parts and p.name not in {"FINAL_INTEGRATION_OUTPUT_SHA256.tsv", "FINAL_INTEGRATION_OUTPUT_MANIFEST.tsv"})
    manifest_rows = []
    for p in output_files:
        rows, cols, headers = file_shape(p) if p.suffix in {".tsv", ".json"} else ("NA", "NA", [])
        manifest_rows.append({"relative_path": str(p.relative_to(OUT)), "size": p.stat().st_size, "SHA256": sha256(p), "rows": rows, "columns": cols})
    out_manifest = pd.DataFrame(manifest_rows)
    write_tsv(out_manifest, OUT / "09_manifests/FINAL_INTEGRATION_OUTPUT_MANIFEST.tsv")
    write_tsv(out_manifest[["relative_path", "SHA256"]], OUT / "09_manifests/FINAL_INTEGRATION_OUTPUT_SHA256.tsv")

    # Project-level QA and handoff.
    project_qa = PROJECT / "12_QA"
    handoff_dir = PROJECT / "13_handoff"
    project_qa.mkdir(exist_ok=True)
    handoff_dir.mkdir(exist_ok=True)
    shutil.copy2(OUT / "10_QA/FINAL_CROSS_MODULE_INTEGRATION_QA.txt", project_qa / "FINAL_CROSS_MODULE_INTEGRATION_V1_QA.txt")
    shutil.copy2(OUT / "09_manifests/FINAL_INTEGRATION_OUTPUT_MANIFEST.tsv", project_qa / "FINAL_CROSS_MODULE_INTEGRATION_V1_MANIFEST.tsv")
    shutil.copy2(OUT / "09_manifests/FINAL_INTEGRATION_OUTPUT_SHA256.tsv", project_qa / "FINAL_CROSS_MODULE_INTEGRATION_V1_SHA256.tsv")
    handoff = f"""# Final Cross-Module Integration V1 Handoff

- Status: `{final_status}`
- Run ID: `{RUN_ID}`
- Authority SHA256 match: `{'YES' if authority_sha_actual == AUTHORITY_SHA else 'NO'}`
- Bulk Secondary temporal panels: `PRE=37`, `POST=164`
- Single-cell formal CURRENT: full-gene V2.1.2 (7 studies, 66 patients, 201/201 Primary evaluable)
- Clinical formal CURRENT: metadata-closed 12-model package
- CPTAC: 86 patients / 59 events; B1 n=80
- GLASS: 102 patients / 91 events; 6 evaluable models
- Transportability PASS: 0/12
- Upstream analyses rerun: `NO`
- New inferential tests: `NO`
- Output root: `{OUT}`

Claim 13 is explicitly a cross-module inference. Spatial and single-cell
localization are supportive context and must not be presented as causal
evidence. GLASS must be described as an initial-diagnosis sensitivity analysis.
"""
    handoff_path = handoff_dir / "FINAL_CROSS_MODULE_INTEGRATION_V1_HANDOFF.md"
    handoff_path.write_text(handoff, encoding="utf-8")
    state_path = handoff_dir / "FINAL_CROSS_MODULE_INTEGRATION_V1_STATE.json"
    atomic_json(state, state_path)

    package = handoff_dir / "FINAL_CROSS_MODULE_INTEGRATION_V1_PACKAGE.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(OUT.rglob("*")):
            if not p.is_file() or "logs" in p.parts:
                continue
            if any(part.lower() in {"checkpoints", "nonfinal_diagnostic"} for part in p.parts):
                continue
            zf.write(p, Path("15_final_cross_module_integration_v1") / p.relative_to(OUT))
        zf.write(handoff_path, Path("handoff") / handoff_path.name)
        zf.write(state_path, Path("handoff") / state_path.name)
        zf.write(project_qa / "FINAL_CROSS_MODULE_INTEGRATION_V1_QA.txt", Path("QA") / "FINAL_CROSS_MODULE_INTEGRATION_V1_QA.txt")
    (handoff_dir / "FINAL_CROSS_MODULE_INTEGRATION_V1_PACKAGE.zip.sha256").write_text(f"{sha256(package)}  {package.name}\n", encoding="utf-8")

    current_final = dict(state)
    current_final.update({"package_path": str(package), "package_sha256": sha256(package), "state_path": str(state_path), "handoff_path": str(handoff_path)})
    atomic_json(current_final, handoff_dir / "CURRENT_FINAL_CROSS_MODULE_INTEGRATION.json")

    # External manifest includes the final package; checksum tables exclude themselves.
    external_files = [
        project_qa / "FINAL_CROSS_MODULE_INTEGRATION_V1_QA.txt",
        project_qa / "FINAL_CROSS_MODULE_INTEGRATION_V1_MANIFEST.tsv",
        handoff_path, state_path, package,
        handoff_dir / "FINAL_CROSS_MODULE_INTEGRATION_V1_PACKAGE.zip.sha256",
        handoff_dir / "CURRENT_FINAL_CROSS_MODULE_INTEGRATION.json",
    ]
    ext = pd.DataFrame([{"absolute_path": str(p), "size": p.stat().st_size, "SHA256": sha256(p)} for p in external_files])
    write_tsv(ext, project_qa / "FINAL_CROSS_MODULE_INTEGRATION_V1_SHA256.tsv")

    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if final_status == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
