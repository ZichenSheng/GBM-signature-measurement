#!/usr/bin/env python3
"""K03 - design check (read-only): which member genes drive the Ivy state profiles of the three ACT5 representations,
and does the coefficient-signed score (sensitivity only; the paper analyses unsigned membership) keep the same ordering?
Uses the frozen Ivy gene-z matrix (the scoring input); verifies that the unsigned mean reproduces the frozen scores."""
import hashlib, json, sys, os
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
Z = Path(os.environ["GBM_DATA_ROOT"]) / "IvyGAP/PROCESSED/IVYGAP_ANATOMIC122_REQUIRED_GENE_Z.parquet"
SC = (Path(os.environ["GBM_SOURCE_ROOT"]) / 'IVYGAP_PATHOLOGY_ANATOMY_EXTENSION_V1/04_SCORES/IVYGAP_PRIMARY201_MODEL_SCORES.tsv.gz')
CO = (Path(os.environ["GBM_SOURCE_ROOT"]) / 'Figure5_ACT5_SPATIAL_REBUILT/source_tables/A_coefficients_membership.tsv')
OUT = Path(os.environ.get("GBM_PANEL_OUT", "outputs/panel_source")).resolve(); OUT.mkdir(parents=True, exist_ok=True)
REPS = {"PF4C-B4-007": "Liu EMT-5", "PF4C-B4-029": "Xiao EMT-4", "PF4C-FINAL-079": "Arimappamagan-14"}
ANAT = ["LE", "IT", "CT", "PAN", "MVP"]
z = pd.read_parquet(Z); S = pd.read_csv(SC, sep="\t", dtype={"sample_id": str, "tumor_id": str})
co = pd.read_csv(CO, sep="\t")
print("coef table columns:", list(co.columns)); print(co.head(3).to_string())
if "sample_id" in z.columns: z = z.set_index("sample_id")
z.index = z.index.astype(str); z = z.loc[S.sample_id]
idc = [c for c in co.columns if "model" in c.lower() or "signature" in c.lower() or "identity" in c.lower()][0]
gc = [c for c in co.columns if "gene" in c.lower()][0]; cc = [c for c in co.columns if "coef" in c.lower()][0]
rows, chk = [], {}
def prof(v):
    x = pd.DataFrame({"t": S.tumor_id.values, "a": S.anatomical_structure.values, "v": np.asarray(v, float)})
    ct = x[x.a == "CT"].groupby("t").v.mean()
    x["rel"] = x.v - x.t.map(ct)  # within-tumour, relative to the tumour's cellular-tumour mean (CT present in all 10 tumours)
    tm = x.groupby(["t", "a"]).rel.mean().reset_index()
    return tm.groupby("a").rel.median().reindex(ANAT)
for m, lab in REPS.items():
    c = co[co[idc] == m] if (co[idc] == m).any() else co[co[idc].astype(str).str.contains(lab.split()[0])]
    genes = [g for g in c[gc].astype(str).str.upper() if g in z.columns]
    unsigned = z[genes].mean(axis=1).to_numpy()
    chk[lab] = {"genes": genes, "max_abs_diff_vs_frozen_score": float(np.nanmax(np.abs(unsigned - S[m].to_numpy())))}
    w = c.set_index(c[gc].astype(str).str.upper())[cc].reindex(genes).astype(float)
    signed = (z[genes] * np.sign(w.values)).mean(axis=1).to_numpy()
    pu, ps = prof(unsigned), prof(signed)
    chk[lab]["unsigned_profile_rel_CT"] = pu.round(3).to_dict(); chk[lab]["coef_signed_profile_rel_CT_SENSITIVITY"] = ps.round(3).to_dict()
    for g in genes:
        pg = prof(z[g].to_numpy())
        rows.append({"label": lab, "gene": g, "coefficient_sign": int(np.sign(w[g])), **{f"rel_CT_{a}": pg[a] for a in ANAT}})
G = pd.DataFrame(rows); G.to_csv(OUT / "K03_ivy_gene_level_profiles.tsv", sep="\t", index=False, float_format="%.4g")
meta = {"inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Z, SC, CO)}, "checks": chk,
        "note": "per-tumour state means minus that tumour's CT mean; median over tumours. Signed score is a sensitivity view only."}
(OUT / "K03_ivy_gene_level_drivers.json").write_text(json.dumps(meta, indent=1) + "\n")
print(json.dumps(chk, indent=1)); print(G.round(2).to_string(index=False))
