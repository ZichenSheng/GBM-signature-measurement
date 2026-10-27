#!/usr/bin/env python3
"""Figure 6: frozen GBM authority + evaluably schematic + CRC04 execution.

Usage: python3 build_figure6.py
All source paths are relative to this script. No historical analysis is rerun.
CRC04 null support is an exact enumeration for display from frozen derived d_i.
The release's statistical convention is reproduced with long-double summation.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import shutil
import xml.etree.ElementTree as ET
from PIL import Image

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
SRC = HERE / "sources"
REL = SRC / "ACT6_RELEASE_V2"
RES = REL / "gbm201data/inst/extdata"
ACT5 = SRC / "GBM_ACT6_CURRENT_CLAUDE_EVIDENCE_HANDOFF_2026-09-11/act5"
CRC = SRC / "CRC04_RESULT_REVIEW"
DATA = HERE / "source_data"
EXPORT = HERE / "exports"
for directory in (DATA, EXPORT):
    directory.mkdir(exist_ok=True)

provenance = []
def tsv(path, panel, purpose):
    provenance.append(dict(panel=panel, source=str(path.relative_to(HERE)),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(), purpose=purpose))
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def js(path, panel, purpose):
    provenance.append(dict(panel=panel, source=str(path.relative_to(HERE)),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(), purpose=purpose))
    return json.loads(path.read_text())

def save_tsv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

def one(rows, **criteria):
    found = [r for r in rows if all(r[k] == v for k, v in criteria.items())]
    assert len(found) == 1, (criteria, len(found))
    return found[0]

def short(identity):
    return identity.removeprefix("PF4C-")

manifest = tsv(RES / "PRIMARY201_MANIFEST.tsv", "A", "Primary201 identity, published provenance and mapped gene-set size")
members = tsv(RES / "GENE_MEMBERSHIP_LONG.tsv", "A", "Membership rows and canonical membership availability")
reps = tsv(RES / "REPRESENTATION_REGISTRY.tsv", "A", "Recorded representations; absent hashes retained as partial records")
evals = tsv(RES / "EVALUABILITY_MATRIX.tsv", "A", "StageB1 claim/evaluability records, regardless of outcome")
claims = tsv(RES / "CLAIM_REGISTRY.tsv", "A", "Definition of the scoped StageB1 claim")
ledger = tsv(ACT5 / "04_ACT5_V2_PROGRESSIVE_CLAIM_LEDGER.tsv", "B,D", "Frozen scientific claim records and formal qualifications")
stage = tsv(ACT5 / "06_ACT5_V2_COMPLETE_STAGEB1_DENOMINATOR.tsv", "D,E", "Prespecified denominator, exact-alpha gate and recovery")
ivy = tsv(ACT5 / "08_ACT5_V2_IVY_CONTEXT_SWITCH.tsv", "B,D", "Ivy coverage, legal endpoint and observed anatomical information")
vocab = tsv(ACT5 / "10_ACT5_V2_STATUS_VOCABULARY_MAP.tsv", "B,D", "Frozen descriptor/qualification distinction")
pairs = tsv(CRC / "11_REAL_PAIRED_DIFFERENCES.tsv", "F", "Primary-execution patient-level paired differences")
exact = js(CRC / "13_REAL_EXACT_TEST_RECEIPT.json", "F", "Frozen statistic, exact test convention and p value")
support = js(CRC / "12_REAL_SUPPORT_RECEIPT.json", "F", "Pair count, support cardinality and minimum attainable p")
crc_record = js(CRC / "15_REAL_EVIDENCE_RECORD.json", "F", "Actual prospective claim-scoped output record")
forensics = js(SRC / "CRC04_ADJUDICATION/22_REPRODUCTION_FAILURE_ADJUDICATION.json", "F", "Latest technical reproduction adjudication")
for name in ("preflight.R", "records.R", "qualification.R"):
    p = REL / "evaluably/R" / name
    provenance.append(dict(panel="C", source=str(p.relative_to(HERE)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                           purpose="Mechanism only; not evidence of historical GBM package execution"))
callback_path = SRC / "CRC04_PROTOCOL/10_MEASUREMENT_CALLBACKS.R"
provenance.append(dict(panel="F",source=str(callback_path.relative_to(HERE)),sha256=hashlib.sha256(callback_path.read_bytes()).hexdigest(),
                       purpose="Frozen statistic definition: sum(d); exhaustive paired sign flips and inclusive observed tail"))

assert len(manifest) == 201 and len(members) == 1729
membership = {}
for r in members:
    membership.setdefault(r["model_instance_id"], []).append(r)
rep_by_id = {r["model_instance_id"]: r for r in reps}
eval_by_id = {r["model_instance_id"]: r for r in evals}
claim_ids = {r["claim_id"] for r in claims}
ordered = sorted(manifest, key=lambda r: (int(r["n_genes_mapped"]), r["model_instance_id"]))
resource_rows = []
for rank, r in enumerate(ordered, 1):
    identity = r["model_instance_id"]
    assert int(r["n_genes_mapped"]) == len(membership[identity])
    rep = rep_by_id.get(identity)
    ev = eval_by_id.get(identity)
    resource_rows.append(dict(signature_id=identity, rank=rank, n_genes=int(r["n_genes_mapped"]),
        Identity=2 if all(r[k] not in ("", "NA") for k in ("model_instance_id", "model_id", "curation_lineage")) else 1,
        Membership=2 if membership[identity] and all(x["resolved"] == "TRUE" for x in membership[identity]) else 1,
        Provenance=2 if all(r[k] not in ("", "NA") for k in ("publication_id", "doi_or_pmid", "provenance_hash")) else 1,
        Representation=0 if rep is None else (2 if all(rep[k] not in ("", "NA") for k in ("representation_hash", "representation_rule_hash")) else 1),
        Claim_Evaluability=2 if ev and ev["claim_id"] in claim_ids else 0))
save_tsv(DATA / "panel_A_resource_layers.tsv", resource_rows)

b4 = "PF4C-B4-007"
bulk = one(ledger, signature_id=b4, dataset="CGGA325", claim_depth="L1_ASSOCIATION")
cap = one(ledger, signature_id=b4, claim_depth="L3_CLAIM_QUALIFIED_DETECTABILITY")
anat = one(ledger, signature_id=b4, context="Tissue/anatomical informativeness")
mal = one(ledger, signature_id=b4, context="Malignant-cell-specific MES state")
brows = [bulk, cap, anat, mal]
save_tsv(DATA / "panel_B_frozen_claim_records.tsv", brows)
drows = [cap,
         one(ledger, signature_id="PF4C-B4-029", claim_depth="L3_CLAIM_QUALIFIED_DETECTABILITY"),
         one(ledger, signature_id="PF4C-FINAL-053", claim_depth="L3_CLAIM_QUALIFIED_DETECTABILITY"), anat, mal]
save_tsv(DATA / "panel_D_frozen_claim_rows.tsv", drows)
f53 = one(stage, signature_id="PF4C-FINAL-053")
assert [f53[x] for x in ("endpoint_support", "constructibility", "q100_exact_alpha_reachable")] == ["YES", "YES", "NO"]
assert float(f53["q100_minimum_attainable_exact_p"]) == .0625
assert [r["qualification_state"] for r in drows] == ["SUPPORTED_WITHIN_DOMAIN", "UNRESOLVED", "NOT_EVALUABLE", "SUPPORTED_WITHIN_DOMAIN", "NOT_EVALUABLE"]
assert anat["alpha_reachability"] == "NOT_APPLICABLE_R4_NOT_RUN_FOR_THIS_IDENTITY"
assert mal["endpoint_compatibility"] == "NO"
r2 = float(one(ivy, signature_id=b4, dataset="Ivy", claim_scope="Tissue/anatomical informativeness")["partial_r2"])

stage_ids = ["B1-004", "B4-007", "B4-029", "FINAL-053", "FINAL-054", "FINAL-055", "FINAL-056", "FINAL-079"]
stage = [one(stage, signature_id="PF4C-" + sid) for sid in stage_ids]
stage_data = []
for r in stage:
    qualifies = r["q100_exact_alpha_reachable"] == "YES"
    recovered = int(r["q100_blocked_detections"]) == int(r["q100_world_n"])
    stage_data.append(dict(signature_id=r["signature_id"], prespecified=True,
        alpha_adjudicable=qualifies, recovered_among_adjudicable=recovered if qualifies else None,
        historical_recovery_n=int(r["q100_blocked_detections"]), historical_world_n=int(r["q100_world_n"]),
        q100_minimum_attainable_exact_p=r["q100_minimum_attainable_exact_p"]))
assert len(stage_data) == 8 and sum(r["alpha_adjudicable"] for r in stage_data) == 7
assert [short(r["signature_id"]) for r in stage_data if r["recovered_among_adjudicable"]] == ["B1-004", "B4-007", "FINAL-056"]
save_tsv(DATA / "panel_E_denominator.tsv", stage_data)

# Enumerate the statistic exactly as defined in the frozen callback: sum(s_i*d_i).
# np.longdouble accumulates like R's sum() on this host; no new primary run.
d = np.array([float(r["paired_difference"]) for r in pairs], dtype=np.longdouble)
null = np.array([np.sum(np.array([1 if mask & (1 << j) else -1 for j in range(len(d))], dtype=np.longdouble) * d)
                 for mask in range(2 ** len(d))], dtype=np.longdouble)
t_obs = float(exact["OBSERVED_TEST_STATISTIC"])
obs_derived = np.sum(d)
tail = null >= obs_derived
assert math.isclose(float(obs_derived), t_obs, rel_tol=0, abs_tol=3e-16)
assert len(d) == 9 and np.count_nonzero(d) == 9
assert len(null) == 512 and int(tail.sum()) == 16
assert float(tail.mean()) == exact["EXACT_P_VALUE"] == .03125
assert float(support["MIN_ATTAINABLE_P"]) == 1 / 512
assert crc_record["qualification"]["terminal_state"] == "SUPPORTED_WITHIN_DOMAIN"
pair_data = [dict(display_alias=f"P{i:02d}", patient_id=r["patient_id"], paired_difference=r["paired_difference"])
             for i, r in enumerate(pairs, 1)]
save_tsv(DATA / "panel_F_patient_differences.tsv", pair_data)
save_tsv(DATA / "panel_F_full_512_state_null.tsv", [dict(mask=i, statistic=format(float(v), ".17g"), in_observed_inclusive_tail=bool(tail[i]), probability=1/512) for i, v in enumerate(null)])
save_tsv(DATA / "panel_source_manifest.tsv", provenance)
figure_data = dict(resource_layers=resource_rows, B_records=brows, D_records=drows, E_collection=stage_data,
    F=dict(paired_differences=pair_data, exact_receipt=exact, support_receipt=support,
           primary_state=crc_record["qualification"]["terminal_state"], null_state_count=len(null), inclusive_tail_count=int(tail.sum())),
    scope=dict(A="frozen resource", B="frozen scientific claim ledger", C="software mechanism schematic",
               D="authority-derived qualification matrix; not runtime logs", E="prespecified frozen scientific collection", F="actual prospective primary execution"))
(DATA / "figure6_source_data.json").write_text(json.dumps(figure_data, indent=2) + "\n")

# Reuse the publication skill's native style, overriding dimensions/colours for this request.
style = Path("/external/PLOT_STYLE/publication.mplstyle")
if style.exists():
    plt.style.use(style)
plt.rcParams.update({"font.family": "Arial", "font.size": 7.2, "axes.labelsize": 7.2,
    "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "axes.linewidth": .5,
    "figure.constrained_layout.use": False, "savefig.bbox": None, "savefig.pad_inches": 0,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "text.color": "#111111",
    "axes.labelcolor": "#111111", "xtick.color": "#111111", "ytick.color": "#111111"})

W, H = 210, 297
fig = plt.figure(figsize=(W/25.4, H/25.4), dpi=150, facecolor="white")
canvas = fig.add_axes([0, 0, 1, 1], frameon=False)
canvas.set(xlim=(0,W), ylim=(H,0)); canvas.axis("off")
BLACK="#111111"; GRAY="#61676B"; LIGHT="#F1F3F4"; LINE="#D8DDDF"
BLUE="#577C93"; BLUE_PALE="#E9EFF3"; TEAL="#286D5D"; TEAL_PALE="#EAF3EF"
RED="#A84442"; RED_PALE="#F7ECEB"; PARTIAL="#A7BAC5"; PRESENT="#606F78"; ABSENT="#ECEFF1"
text_items=[]

def text(x,y,s,size=7.2,weight="normal",color=BLACK,ha="left",va="top",linespacing=1.18,**kw):
    artist=canvas.text(x,y,s,fontsize=size,fontweight=weight,color=color,ha=ha,va=va,linespacing=linespacing,**kw)
    text_items.append(artist)
    return artist

def rect(x,y,w,h,face=LIGHT,edge="none",lw=.5,**kw):
    p=Rectangle((x,y),w,h,facecolor=face,edgecolor=edge,linewidth=lw,**kw)
    canvas.add_patch(p);return p

def line(x1,y1,x2,y2,color=GRAY,lw=.6,**kw):
    canvas.add_line(Line2D([x1,x2],[y1,y2],color=color,lw=lw,**kw))

def arrow(x1,y1,x2,y2,color=GRAY,lw=.75,**kw):
    canvas.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=7,color=color,lw=lw,**kw))

def heading(letter,x,y,title,subtitle=None,size=10.4):
    text(x,y-.6,letter,18,"bold")
    text(x+7,y,title,size,"bold")
    if subtitle: text(x+7,y+5.3,subtitle,7.1,color=GRAY)

def axes_mm(x,y,w,h):
    return fig.add_axes([x/W,(H-y-h)/H,w/W,h/H])

def state_label(x,y,value,size=6.6,ha="left",split=False):
    color=TEAL if value=="SUPPORTED_WITHIN_DOMAIN" else RED if value in ("NOT_EVALUABLE","INVALID") else GRAY
    shown=value.replace("SUPPORTED_WITHIN_DOMAIN","SUPPORTED_\nWITHIN_DOMAIN") if split else value
    text(x,y,shown,size,"bold",color,ha=ha)

# A — ranked-size backdrop and primary availability matrix.
heading("A",8,8,"The GBM signature universe", "gbm201data · 201 curated signatures",10.4)
text(8,23,"Gene-set size",7.8,"bold")
text(48,23,"Resource layers for 201 signatures",7.8,"bold")
ax=axes_mm(17,36,26,36)
sizes=np.array([r["n_genes"] for r in resource_rows])
ax.scatter(np.arange(1,202),sizes,s=3,color="#A6AAAD",linewidths=0)
ax.set_yscale("log");ax.set_ylim(2.7,83);ax.set_xlim(-5,211)
ax.set_xticks([1,100,201]);ax.set_yticks([3,10,30,60]);ax.set_yticklabels(["3","10","30","60"])
ax.minorticks_off();ax.set_xlabel("Size rank",labelpad=2);ax.set_ylabel("Genes",labelpad=2)
for sid,offset,ha in [("B4-029",(7,8),"left"),("B4-007",(9,20),"left"),("FINAL-053",(-5,6),"right")]:
    r=next(r for r in resource_rows if short(r["signature_id"])==sid)
    ax.scatter(r["rank"],r["n_genes"],s=13,color=BLUE,zorder=4,linewidths=0)
    ax.annotate(sid,(r["rank"],r["n_genes"]),xytext=offset,textcoords="offset points",fontsize=6.3,ha=ha,
                arrowprops=dict(arrowstyle="-",lw=.4,color=GRAY),color=BLACK)
cols=["Identity","Membership","Provenance","Representation","Claim_Evaluability"]
headers=["Identity","Member-\nship","Prove-\nnance","Repr.\nrecorded","Claim /\nevaluab."]
mx,my,mw,mh=49,36,46,36
for j,h in enumerate(headers): text(mx+(j+.5)*mw/5,29,h,6.4,ha="center")
for i,r in enumerate(resource_rows):
    for j,col in enumerate(cols):
        rect(mx+j*mw/5,my+i*mh/201,mw/5-.3,mh/201+.005,[ABSENT,PARTIAL,PRESENT][r[col]])
text(mx,73.5,"Claim layer: StageB1; rows ranked by size",6.1,color=GRAY)
for x,c,label in [(49,PRESENT,"Present"),(65.2,PARTIAL,"Partial"),(80.2,ABSENT,"Absent")]:
    rect(x,79,2.7,2.7,c);text(x+3.6,79,label,6.4)
text(8,82,"1,729 membership rows",7.5,"bold")
text(8,87,"201 curated signatures ≠ 201 independently validated signatures.",7.0)

# B — four frozen claim identities, no runtime IDs or hashes.
heading("B",103,8,"One GBM subject, multiple claim-scoped\nscientific records (B4-007)",size=9.9)
text(110,19,"Frozen claim ledger · context-specific membership scores",6.9,color=GRAY)
bcontent=[
    ("Bulk programme association", "CGGA325 · bulk tumour sample", "Unsigned / unweighted; descriptive MES association"),
    ("Malignant-MES capability (q100)", "DS03 · 57 malignant regional pseudobulks / 12 donors", "Malignant-only; donor-blocked exact design · 20/20 recovered"),
    ("Anatomical informativeness", "IvyGAP · 122 LMD tissue wells / 10 tumours", "Tissue membership; tumour-aware anatomical design"),
    ("Malignant-cell MES state", "IvyGAP · LMD tissue; malignant endpoint not isolated", "Computable tissue score; scope stops at endpoint")]
for i,(r,(title,context,design)) in enumerate(zip(brows,bcontent)):
    y=26+i*14.0
    rect(104,y,98,12.5,TEAL_PALE if r["qualification_state"]=="SUPPORTED_WITHIN_DOMAIN" else RED_PALE)
    rect(104,y,.8,12.5,TEAL if r["qualification_state"]=="SUPPORTED_WITHIN_DOMAIN" else RED)
    text(107,y+1.2,title,7.9,"bold")
    state_label(200,y+1.6,r["qualification_state"],6.2,ha="right")
    text(107,y+5.1,context,6.9)
    text(107,y+8.7,design,6.6,color=GRAY)
text(105,84.5,"Subject × representation × claim × context × design",7.0,"bold")
text(105,88.5,"Evidence does not inherit across records.",7.0)

# C — an explicitly schematic software mechanism, no invented receipts.
heading("C",8,98,"The evaluably mechanism: from scientific question to an immutable, claim-scoped record",size=10.1)
text(15,104,"Software mechanism schematic; examples refer to frozen claims in B, not historical package runs.",7.1,color=GRAY)
text(8,114,"Scientific identity",9,"bold")
text(65,114,"Declare & freeze",9,"bold")
text(144,114,"Claim-scoped records",9,"bold")
identity=[("Subject","B4-007"),("Representation","Malignant-only membership"),("Claim","q100 measurement capability"),("Context","DS03 GBM-Space"),("Design","Donor-blocked exact inference")]
for i,(label,value) in enumerate(identity):
    y=122+i*8.1
    text(8,y,label,6.8,"bold",GRAY)
    text(8,y+3,value,7.4)
arrow(55.5,140,62,140)
text(65,120,"Ordered preflight",7.7,"bold")
gates=["Representation validity","Unit / endpoint legality","Truth isolation","Constructibility (valid-world yield)","Inferential support (α reachable)"]
for i,g in enumerate(gates):
    y=126+i*5.3
    rect(65,y-.4,68,4.6,BLUE_PALE)
    text(67,y,str(i+1),7.2,"bold",BLUE)
    text(71.5,y,g,7.3)
arrow(98,150.7,98,154.3)
text(65,155,"Execute once, only if legal → Qualify",7.7,"bold")
text(65,161,"A blocked gate stops downstream execution.",6.9,color=RED)
text(65,165.3,"Blocked checks still produce a durable record.",6.9)
rect(144,122,58,19,RED_PALE)
rect(144,145,58,19,TEAL_PALE)
text(147,124,"BLOCKED CLAIM-SCOPED RECORD",7.0,"bold")
text(147,129,"Schematic: B4-007 / Ivy malignant state",6.5)
state_label(147,134,"NOT_EVALUABLE",7.0)
text(147,147,"LEGAL CLAIM-SCOPED RECORD",7.2,"bold")
text(147,152,"Schematic: B4-007 / DS03 capability",6.6)
state_label(147,157,"SUPPORTED_WITHIN_DOMAIN",6.8)
arrow(134.5,157,142,154,color=TEAL,lw=.7)
arrow(134.5,138,142,132,color=RED,lw=.7)
text(144,168,"Immutable within the declared identity",6.9,"bold")
text(8,173,"Record A",7.6,"bold")
arrow(22,174.5,29,174.5,color=GRAY)
line(24.5,173.1,26.5,175.9,color=RED,lw=1)
text(31,173,"Record B   ·   Evidence does not inherit across distinct records.",7.6,"bold")

# D — claim qualification, deliberately omitting unadjudicated software gates.
heading("D",8,183,"Gate status across real GBM claim records",size=9.8)
text(15,188.5,"Authority-derived qualification; no historical package rerun",6.8,color=GRAY)
xcase=8
cx=[47,58,71,84]
heads=["Repre-\nsentation","Unit /\nendpoint","Construct-\nibility","α /\nsupport"]
for x,h in zip(cx,heads):text(x,195,h,6.5,ha="center")
text(100,195,"Evidence /\nanalysis",6.5,ha="center")
text(120,195,"Qualification\nstate",6.5,ha="center")
case_labels=["B4-007 · DS03\nCapability","B4-029 · DS03\nCapability","FINAL-053 · DS03\nCapability","B4-007 · Ivy\nAnatomical information","B4-007 · Ivy\nMalignant-state claim"]
gate_values=[["PASS"]*4,["PASS"]*4,["PASS","PASS","PASS","BLOCK"],["PASS","PASS","PASS","NA"],["PASS","BLOCK","AFTER_STOP","AFTER_STOP"]]
def tile(x,y,status,w=7.3,h=4.8):
    if status=="PASS":
        rect(x-w/2,y-h/2,w,h,BLUE_PALE);text(x,y,"+",8.5,"bold",BLUE,ha="center",va="center")
    elif status=="BLOCK":
        rect(x-w/2,y-h/2,w,h,RED_PALE,RED,.65);text(x,y,"×",8.8,"bold",RED,ha="center",va="center")
    elif status=="NA":
        rect(x-w/2,y-h/2,w,h,"white",LINE,.5);text(x,y,"N/A",6.0,color=GRAY,ha="center",va="center")
    else:
        rect(x-w/2,y-h/2,w,h,LIGHT);text(x,y,"···",8,color="#9FA6AB",ha="center",va="center")
display_evidence=["20/20", "0/20", "0/20†\nhistorical", f"R² = {r2:.3f}", "Not tested"]
gate_export=[]
for i,(r,label,g,ev) in enumerate(zip(drows,case_labels,gate_values,display_evidence)):
    y=207+i*6.6
    text(xcase,y-2.4,label,6.6,linespacing=1.08)
    for x,status in zip(cx,g):tile(x,y,status)
    text(100,y,ev,6.5,ha="center",va="center")
    state_label(120,y-2.5,r["qualification_state"],6.0,ha="center",split=True)
    gate_export.append(dict(signature_id=r["signature_id"],context=r["context"],representation=g[0],unit_endpoint=g[1],
                            constructibility=g[2],alpha_support=g[3],evidence_display=ev,qualification_state=r["qualification_state"]))
save_tsv(DATA / "panel_D_display_encodings.tsv",gate_export)
for x,status,label in [(10,"PASS","Pass"),(29,"BLOCK","Block"),(50,"NA","Not applicable"),(84,"AFTER_STOP","Not assessed after stop")]:
    tile(x,239,status,w=4.5,h=3);text(x+3.2,237.8,label,6.0)
text(8,243,"† FINAL-053: p_min = 0.0625; historical 0/20 is not an adjudicable negative.",6.4)

# E — denominator retained; no success-rate calculation.
heading("E",139,183,"Denominator preservation\nin GBM StageB1 collection",size=9.1)
text(146,192,"DS03 malignant-state capability",6.7,color=GRAY)
ex=np.arange(8)*5.2+158
for x,sid in zip(ex,stage_ids):text(x,205,sid,6.2,ha="left",va="bottom",rotation=60)
ey=[211,219,227]
for y,label,n in zip(ey,["Prespecified","Alpha-\nadjudicable","Recovered\n(adjudicable)"],[8,7,3]):
    text(139,y,label,6.4,va="center",linespacing=1.02);text(201,y,str(n),12,"bold",ha="center",va="center")
for i,r in enumerate(stage_data):
    for j in range(3):
        yes=True if j==0 else r["alpha_adjudicable"] if j==1 else r["recovered_among_adjudicable"]
        x,y=ex[i],ey[j]
        if yes is True:
            canvas.plot(x,y,"o",markersize=4.4,color=TEAL,mew=0)
        elif yes is None:
            text(x,y,"—",7.6,color=GRAY,ha="center",va="center")
        elif j==1:
            text(x,y,"×",9,"bold",RED,ha="center",va="center")
        else:
            canvas.plot(x,y,"o",markersize=4.2,mfc="white",mec=GRAY,mew=.6)
text(139,234,"FINAL-053 stays in the denominator.",6.7,"bold")
text(139,238,"× α-unreachable   — not adjudicable",6.1)
text(139,241.7,"○ No recovery among adjudicable designs",6.1)

# F — genuine prospective execution, using frozen primary patient differences.
heading("F",8,251,"Prospective external execution: CRC04 (colorectal cancer, held-out)",size=10.2)
text(15,256.5,"Prospective, paired · independent disease · strictly held-out",7.1,color=GRAY)
text(18,262,"Patient differences (n = 9)",7.5,"bold")
axp=axes_mm(18,269,48,17)
vals=np.asarray(d,dtype=float)
axp.axhline(0,color="#ABB2B5",lw=.6,ls=(0,(2,2)))
axp.vlines(np.arange(9),0,vals,color="#BAC4C9",lw=.8)
axp.scatter(np.arange(9),vals,s=12,color=BLUE,zorder=3,linewidths=0)
axp.set_xlim(-.6,8.6);axp.set_ylim(-.18,.6)
axp.set_xticks(np.arange(9));axp.set_xticklabels([r["display_alias"] for r in pair_data],fontsize=6)
axp.set_yticks([0,.25,.5]);axp.set_ylabel("Δ CRC-BUD11 score\n(FRONT − CORE)",fontsize=6.7,labelpad=2)
axp.set_xlabel("Patient (display alias)",fontsize=6.4,labelpad=2)
axp.tick_params(length=2,pad=1.5)
text(84,262,"Exact sign-flip null (512 states)",7.5,"bold")
axn=axes_mm(84,269,53,17)
bins=np.linspace(-float(np.abs(d).sum()),float(np.abs(d).sum()),27)
axn.hist([np.asarray(null[~tail],float),np.asarray(null[tail],float)],bins=bins,
         weights=[np.repeat(1/512,int((~tail).sum())),np.repeat(1/512,int(tail.sum()))],
         stacked=True,color=["#C6CDD1",TEAL],edgecolor="white",lw=.4)
axn.axvline(t_obs,color=BLACK,lw=.75,ls=(0,(2,1)))
axn.set_xlim(bins[0]-.03,bins[-1]+.03);axn.set_ylim(0,.10)
axn.set_yticks([0,.05,.10]);axn.set_xticks([-1.5,0,1.5])
axn.set_ylabel("Null probability",fontsize=6.7,labelpad=2)
axn.set_xlabel("T = Σ sᵢdᵢ",fontsize=6.8,labelpad=2)
axn.tick_params(length=2,pad=1.5)
axn.annotate("Tobs = 1.596",xy=(t_obs,.052),xytext=(.98,.095),ha="right",va="top",fontsize=6.2,
             arrowprops=dict(arrowstyle="-",lw=.5,color=GRAY))
text(148,263,"Frozen prospective contract",8.1,"bold")
text(148,269,"9 complete pairs · 9 nonzero differences",6.9)
text(148,273.4,"Patient-level exact one-sided sign-flip",6.9)
text(148,277.8,"p = 0.03125",7.5,"bold")
text(148,282.2,"p_min = 0.001953125",7.0)
rect(147,286.9,55,5.3,TEAL_PALE)
state_label(149,288,"SUPPORTED_WITHIN_DOMAIN",6.8)
text(8,294,"CRC04: technical reproduction discrepancy adjudicated; primary scientific result preserved (see source note).",6.1,color=GRAY)

# Save strictly A4. No bbox='tight', which would alter the page dimensions.
stem=EXPORT / "Figure6_ACT6_A4_portrait"
fig.savefig(stem.with_suffix(".svg"),format="svg")
fig.savefig(stem.with_suffix(".pdf"),format="pdf",metadata={"Title":"GBM claim-scoped evidence with evaluably", "Subject":"Frozen GBM authority, schematic mechanism, prospective CRC04 execution"})
fig.savefig(EXPORT / "Figure6_ACT6_A4_portrait_600dpi.png",dpi=600)
fig.savefig(EXPORT / "Figure6_ACT6_preview.png",dpi=180)
with Image.open(EXPORT / "Figure6_ACT6_preview.png") as preview:
    preview.convert("L").save(EXPORT / "Figure6_grayscale_QA.png")

# Geometry checks catch page-edge clipping while visual QA checks local layout.
fig.canvas.draw();renderer=fig.canvas.get_renderer()
out_of_page=[]
for artist in fig.findobj(Text):
    if not artist.get_text() or not artist.get_visible():continue
    b=artist.get_window_extent(renderer)
    if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.width+.5 or b.y1 > fig.bbox.height+.5:
        out_of_page.append(artist.get_text())
overlaps=[]
for i,a in enumerate(text_items):
    if a.get_rotation() != 0:continue  # Axis-aligned bounds overestimate rotated labels; visually checked.
    ba=a.get_window_extent(renderer)
    for b in text_items[i+1:]:
        if b.get_rotation() != 0:continue
        bb=b.get_window_extent(renderer)
        dx=min(ba.x1,bb.x1)-max(ba.x0,bb.x0)
        dy=min(ba.y1,bb.y1)-max(ba.y0,bb.y0)
        if dx>.7 and dy>.7:
            overlaps.append([a.get_text(),b.get_text(),round(dx*dy,2)])
qa=dict(page_mm=[210,297], png_dpi=600, expected_png_size=[int(W/25.4*600),int(H/25.4*600)],
        out_of_page_text=out_of_page,unrotated_canvas_text_overlaps=overlaps, n_signatures=len(manifest),membership_rows=len(members),
        FINAL053_constructibility=f53["constructibility"], FINAL053_exact_alpha_reachable=f53["q100_exact_alpha_reachable"],
        stageb1_counts=[len(stage_data),sum(r["alpha_adjudicable"] for r in stage_data),sum(r["recovered_among_adjudicable"] is True for r in stage_data)],
        crc04_statistic_frozen=t_obs,crc04_statistic_from_derived_pairs=float(obs_derived),crc04_tail_states=int(tail.sum()),crc04_support=len(null),
        crc04_p=float(tail.mean()), C_schematic=True,D_is_package_log=False,GBM_historical_analyses_rerun=False,
        states_used=sorted({r["qualification_state"] for r in drows}),
        resource_layer_definition="0 absent from named registry, 1 partial record (hash fields missing), 2 present; claim layer records existence, not favorable outcome")
(HERE / "FIGURE6_QA.json").write_text(json.dumps(qa,indent=2)+"\n")
(HERE / "PRODUCTION_STATUS.json").write_text(json.dumps(dict(status="COMPLETE_UNDER_REVISED_SCOPE", previous_preflight_pause="SUPERSEDED_BY_EXPLICIT_USER_SCOPE_REVISION", final_figure_rendered=True, GBM_analyses_rerun=False, C="SCHEMATIC", D="FROZEN_AUTHORITY_DERIVED_QUALIFICATION", F="FROZEN_PROSPECTIVE_PRIMARY_EXECUTION"),indent=2)+"\n")
print(json.dumps(qa,indent=2))
plt.close(fig)
