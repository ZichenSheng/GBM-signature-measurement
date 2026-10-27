#!/usr/bin/env python3
"""Assemble the exact eight DS03 Phase-3.5 templates without endpoint inspection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    mapping = pd.read_csv(ROOT / "00_authority/PRIVATE_TEMPLATE_MAP.tsv", sep="\t", dtype=str)
    ds03 = mapping[mapping.context.eq("DS03")].copy()
    if len(ds03) != 8:
        raise RuntimeError("STOP_AND_REPORT_DS03_SCOPE_CONFLICT")
    if set(ds03.claim_geometry) != {"MES_MALIGNANT_STATE"}:
        raise RuntimeError("DS03 claim geometry conflict")
    refs = pd.read_csv(ROOT / "00_authority/CLAIM_REFERENCE_REGISTRY.tsv", sep="\t", dtype=str)
    ref = refs[refs.claim_family.eq("MES_MALIGNANT_STATE")]
    if len(ref) != 1:
        raise RuntimeError("SIGNATURE_STATUS=NOT_EVALUABLE_REFERENCE_UNRECOVERABLE")

    scores, biology, windows, template_rows = [], [], [], []
    common_groups = common_units = common_b = None
    for row in ds03.itertuples(index=False):
        path = ROOT / "02_source_reconstruction/templates" / f"{row.anonymous_template_id}.npz"
        z = np.load(path, allow_pickle=False)
        s = z["S"].astype(np.float64)
        b = z["B_template"].astype(np.float64)
        groups = z["patient_ids"].astype(str)
        units = z["unit_ids"].astype(str)
        if len(s) != 57 or b.shape != (57, 4) or len(set(groups)) != 12:
            raise RuntimeError("DS03 source object geometry conflict")
        if str(z["context"]) != "DS03" or str(z["claim_geometry"]) != "MES_MALIGNANT_STATE":
            raise RuntimeError("DS03 template scope conflict")
        if common_groups is None:
            common_groups, common_units, common_b = groups, units, b
        elif not (np.array_equal(groups, common_groups) and np.array_equal(units, common_units) and np.array_equal(b, common_b)):
            raise RuntimeError("DS03 shared measurement/B object conflict")
        scores.append(s); biology.append(b); windows.append(z["window"].astype(bool))
        template_rows.append({
            "anonymous_template_id": row.anonymous_template_id, "signature_id": row.signature_id,
            "context": row.context, "claim_geometry": row.claim_geometry,
            "B_claim_facets": ref.B_claim_facets.iat[0], "anchor": ref.A_claim.iat[0],
            "unit_n": len(s), "block_n": len(set(groups)), "template_sha256": sha(path),
        })
    np.savez_compressed(
        ROOT / "02_source_reconstruction/DS03_STAGEB1_FROZEN_INPUTS.npz",
        anonymous_template_ids=ds03.anonymous_template_id.to_numpy(dtype="U32"),
        signature_ids=ds03.signature_id.to_numpy(dtype="U64"),
        claim_geometry=np.asarray(["MES_MALIGNANT_STATE"] * 8, dtype="U64"),
        units=common_units.astype("U128"), groups=common_groups.astype("U32"),
        S=np.asarray(scores, dtype=np.float64), B=np.asarray(biology, dtype=np.float64),
        source_window=np.asarray(windows, dtype=bool),
    )
    pd.DataFrame(template_rows).to_csv(ROOT / "02_source_reconstruction/DS03_FROZEN_IDENTITY_REGISTRY.tsv", sep="\t", index=False, lineterminator="\n")
    unit = pd.DataFrame({"unit_position": np.arange(57), "unit_id": common_units, "block_id": common_groups})
    unit.to_csv(ROOT / "02_source_reconstruction/DS03_UNIT_REGISTRY.tsv", sep="\t", index=False, lineterminator="\n")
    blocks = unit.groupby("block_id", as_index=False).agg(unit_n=("unit_id", "size"))
    blocks.to_csv(ROOT / "02_source_reconstruction/DS03_BLOCK_SIZE_DISTRIBUTION.tsv", sep="\t", index=False, lineterminator="\n")
    receipt = {
        "status": "PASS", "scientific_role": "SOURCE_RECONSTRUCTION_NOT_BENCHMARK_RESULT",
        "context": "DS03", "score_scope": "MALIGNANT_ONLY", "measurement_unit": "ONE_OFFICIAL_GBMSPACE_REGIONAL_SITE_MALIGNANT_NUCLEUS_PSEUDOBULK",
        "unit_n": 57, "block_n": 12, "identity_n": 8,
        "identities": ds03.signature_id.tolist(), "anonymous_template_ids": ds03.anonymous_template_id.tolist(),
        "claim_geometry": "MES_MALIGNANT_STATE", "B_claim_facets": ref.B_claim_facets.iat[0],
        "anchor": ref.A_claim.iat[0], "endpoint_occupancy_inspected": False,
        "frozen_input_sha256": sha(ROOT / "02_source_reconstruction/DS03_STAGEB1_FROZEN_INPUTS.npz"),
        "unit_registry_sha256": sha(ROOT / "02_source_reconstruction/DS03_UNIT_REGISTRY.tsv"),
    }
    (ROOT / "02_source_reconstruction/DS03_SOURCE_RECONSTRUCTION_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
