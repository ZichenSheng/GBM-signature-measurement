#!/usr/bin/env python3
"""K02 - design check (read-only recount, no new simulation): ACT4 detection counts per identity x q under
(a) the frozen 999-draw Monte Carlo P values and (b) the prior independent audit's exact blocked P / 200,000-draw naive P,
taken from GBM_INDEPENDENT_SCIENTIFIC_AUDIT/logs/act4/act4_world_level_independent.tsv. alpha = 0.05, P <= alpha."""
import hashlib, json, sys, os
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
SRC = (Path(os.environ["GBM_VALIDATION_ROOT"]) / 'GBM_INDEPENDENT_SCIENTIFIC_AUDIT/logs/act4/act4_world_level_independent.tsv')
OUT = Path(os.environ.get("GBM_PANEL_OUT", "outputs/panel_source")).resolve(); OUT.mkdir(parents=True, exist_ok=True)
LAB = {"PF4C-B1-004": "Peng GA-MSC-8", "PF4C-B4-007": "Liu EMT-5", "PF4C-B4-029": "Xiao EMT-4", "PF4C-FINAL-053": "Bao OS-61",
       "PF4C-FINAL-054": "Bao transcription-17", "PF4C-FINAL-055": "Bao adhesion-8", "PF4C-FINAL-056": "Bao stringent-5", "PF4C-FINAL-079": "Arimappamagan-14"}
A = 0.05
w = pd.read_csv(SRC, sep="\t")
w["nf"] = (w.p_naive_frozen <= A); w["bf"] = (w.p_blocked_frozen <= A)
w["ni"] = (w.p_NU_mc <= A); w["bi"] = (w.p_BK_exact.fillna(w.p_BK_mc) <= A)
w["naive_p_in_0.04_0.06"] = w.p_NU_mc.between(0.04, 0.06)
g = w.groupby(["signature_id", "q"]).agg(naive_999=("nf", "sum"), naive_200k=("ni", "sum"), blocked_999=("bf", "sum"), blocked_exact=("bi", "sum"),
                                       naive_p200k_median=("p_NU_mc", "median"), naive_p_near_alpha=("naive_p_in_0.04_0.06", "sum"), n=("nf", "size")).reset_index()
g.insert(1, "label", g.signature_id.map(LAB))
g["fingerprint_999"] = (g.blocked_999 - g.naive_999) / g.n; g["fingerprint_hr"] = (g.blocked_exact - g.naive_200k) / g.n
g.to_csv(OUT / "K02_act4_counts_999_vs_high_resolution.tsv", sep="\t", index=False, float_format="%.6g")
ev = w[w.signature_id != "PF4C-FINAL-053"]
def cats(d, n, b):
    return {"naive_only": int((d[n] & ~d[b]).sum()), "blocked_only": int((~d[n] & d[b]).sum()), "both": int((d[n] & d[b]).sum()), "neither": int((~d[n] & ~d[b]).sum())}
res = {"source": str(SRC), "sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
       "all_q_evaluable_560": {"frozen_999": cats(ev, "nf", "bf"), "high_resolution": cats(ev, "ni", "bi")},
       "by_q_evaluable": {str(q): {"frozen_999": cats(d, "nf", "bf"), "high_resolution": cats(d, "ni", "bi")} for q, d in ev.groupby("q")},
       "changed_cells": g[(g.naive_999 != g.naive_200k) | (g.blocked_999 != g.blocked_exact)][["label", "q", "naive_999", "naive_200k", "blocked_999", "blocked_exact"]].to_dict(orient="records")}
(OUT / "K02_act4_counts_999_vs_high_resolution.json").write_text(json.dumps(res, indent=1, default=int) + "\n")
print(g.to_string(index=False)); print(json.dumps(res["all_q_evaluable_560"], indent=1)); print(json.dumps(res["by_q_evaluable"], indent=1)); print(json.dumps(res["changed_cells"], indent=1))
