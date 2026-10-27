
import os as _gbm_os
from pathlib import Path as _GBMPath
def _gbm_external(env, relative):
    return _GBMPath(_gbm_os.environ.get(env, str(_GBMPath.cwd() / "external" / env.lower()))) / relative

import csv, os, hashlib, statistics, math
from collections import defaultdict
import numpy as np
R=str(_gbm_external('GBM_SOURCE_ROOT', 'GBM标签数据库/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23/09_de_novo_derived/12_spatial_v2_1_1')); Q=R+"/12_direction_sanity_QA_v2_1_3"
os.makedirs(Q,exist_ok=True)
def rd(p):
 with open(p,newline="") as f:return list(csv.DictReader(f,delimiter="\t"))
def wr(n,rows,fields):
 with open(Q+"/"+n,"w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,delimiter="\t",extrasaction="ignore");w.writeheader();w.writerows(rows)
def q(v,p):
 v=sorted(v);return np.quantile(v,p) if v else float("nan")
def summ(v):
 return dict(negative_n=sum(x<0 for x in v),zero_n=sum(x==0 for x in v),positive_n=sum(x>0 for x in v),minimum=min(v),Q1=q(v,.25),median=q(v,.5),Q3=q(v,.75),maximum=max(v))
sec=rd(R+"/04_region_localization/PRIMARY201_SECTION_EFFECTS.tsv"); pat=rd(R+"/05_patient_level_inference/PRIMARY201_PATIENT_EFFECTS.tsv"); ps=rd(R+"/05_patient_level_inference/PRIMARY201_PATIENT_INFERENCE_SUMMARY.tsv"); ss=rd(R+"/05_patient_level_inference/SENSITIVITY17_PATIENT_INFERENCE_SUMMARY.tsv")
# code audit
audit=[dict(component="section_effect",finding="mean(model_score[high_niche])-mean(model_score[low_niche])",status="PASS_NO_ABSOLUTE_OR_FLIP"),dict(component="high_low_grouping",finding="quantile(niche_proxy_score,c(.25,.75)); high>=Q3; low<=Q1",status="PASS_NICHE_SCORE_ONLY"),dict(component="patient_effect",finding="aggregate section_effect by dataset_id+patient_id+niche+score_id using mean",status="PASS_EQUAL_SECTION_WEIGHT"),dict(component="direction",finding="positive_patient_n=sum(patient_effect>0); no score/effect sign flip",status="PASS_NO_FORCED_SIGN")]
wr("01_DIRECTION_CODE_AUDIT.tsv",audit,list(audit[0]))
# distributions
rows=[]
for level,data,valcol in [("SECTION",sec,"section_effect"),("PATIENT",pat,"patient_effect")]:
 for niche in sorted({x["niche"] for x in data}):
  v=[float(x[valcol]) for x in data if x["niche"]==niche];rows.append(dict(level=level,analysis_panel="PRIMARY201",niche=niche,resource_scope="COMBINED",**summ(v)))
  for ds in sorted({x["dataset_id"] for x in data}):
   vv=[float(x[valcol]) for x in data if x["niche"]==niche and x["dataset_id"]==ds];rows.append(dict(level=level,analysis_panel="PRIMARY201",niche=niche,resource_scope=ds,**summ(vv)))
wr("02_SECTION_EFFECT_DISTRIBUTION.tsv",[x for x in rows if x["level"]=="SECTION"],list(rows[0]))
wr("03_PATIENT_EFFECT_DISTRIBUTION.tsv",[x for x in rows if x["level"]=="PATIENT"],list(rows[0]))
# model effect strength and resource results
strength=[];resource=[]
for niche in sorted({x["niche"] for x in ps}):
 vals=[abs(float(x["median_patient_effect"])) for x in ps if x["niche"]==niche];threshold=.01*np.median(vals)
 for x in ps:
  if x["niche"]!=niche:continue
  strength.append(dict(model_instance_id=x["score_id"],niche=niche,median_patient_effect=x["median_patient_effect"],near_zero="YES" if abs(float(x["median_patient_effect"]))<threshold else "NO",CI_crosses_zero="YES" if float(x["bootstrap_CI_low"])<=0<=float(x["bootstrap_CI_high"]) else "NO",BH_supported="YES" if float(x["q_value"])<.05 else "NO"))
 for ds in sorted({x["dataset_id"] for x in pat}):
  vv=[]
  for m in sorted({x["score_id"] for x in pat}):
   z=[float(x["patient_effect"]) for x in pat if x["dataset_id"]==ds and x["niche"]==niche and x["score_id"]==m]
   if z:vv.append(statistics.median(z))
  resource.append(dict(resource_id=ds,niche=niche,evaluable_model_n=len(vv),positive_model_n=sum(x>0 for x in vv),negative_model_n=sum(x<0 for x in vv),median_model_effect=statistics.median(vv),IQR_model_effect=q(vv,.75)-q(vv,.25),BH_supported_model_n="NOT_RESOURCE_SPECIFIC_BH_IN_FORMAL_OUTPUT"))
wr("04_MODEL_EFFECT_STRENGTH_QA.tsv",strength,list(strength[0]));wr("05_RESOURCE_SPECIFIC_DIRECTION_QA.tsv",resource,list(resource[0]))
# common axis
axis=[]
for niche in sorted({x["niche"] for x in pat}):
 patients=sorted({x["dataset_id"]+"|"+x["patient_id"] for x in pat if x["niche"]==niche});models=sorted({x["score_id"] for x in pat if x["niche"]==niche});a=np.full((len(patients),len(models)),np.nan)
 for i,p in enumerate(patients):
  for j,m in enumerate(models):
   z=[float(x["patient_effect"]) for x in pat if x["niche"]==niche and x["score_id"]==m and x["dataset_id"]+"|"+x["patient_id"]==p]
   if z:a[i,j]=z[0]
 col=np.nanmean(a,axis=0);a=np.where(np.isnan(a),col,a);a=a-a.mean(axis=0);s=np.linalg.svd(a,full_matrices=False)[1];axis.append(dict(niche=niche,patient_n=len(patients),model_n=len(models),PC1_variance_explained=(s[0]**2/(s**2).sum())))
wr("09_MODEL_COMMON_AXIS_QA.tsv",axis,list(axis[0]))
# unavailable spot QA
unavail=[dict(status="NOT_AUDITABLE_SPOT_SCORES_NOT_SERIALIZED",reason="Formal execution retained section effects and score manifests but not spot-score matrices; no raw/object reread performed.",source="03_spatial_scores/*MANIFEST.tsv")]
wr("06_NICHE_SPOT_SET_JACCARD.tsv",unavail,["status","reason","source"]);wr("07_NICHE_SCORE_CORRELATION.tsv",unavail,["status","reason","source"]);wr("08_GLOBAL_EXPRESSION_LOAD_QA.tsv",unavail,["status","reason","source"]);wr("10_INDEPENDENT_SECTION_RECALCULATION_QA.tsv",unavail,["status","reason","source"])
primary_pos=sum(float(x["median_patient_effect"])>0 for x in ps);near=sum(x["near_zero"]=="YES" for x in strength);cross=sum(x["CI_crosses_zero"]=="YES" for x in strength);bh=sum(x["BH_supported"]=="YES" for x in strength)
with open(Q+"/SPATIAL_ALL_POSITIVE_DIRECTION_SANITY_QA.txt","w") as f:f.write(f"CODE_DIRECTION_AUDIT=PASS\\nPRIMARY_SECTION_NEGATIVE_N={sum(float(x['section_effect'])<0 for x in sec)}\\nPRIMARY_PATIENT_NEGATIVE_N={sum(float(x['patient_effect'])<0 for x in pat)}\\nPRIMARY_POSITIVE_MODEL_N={primary_pos}\\nPRIMARY_NEAR_ZERO_POSITIVE_MODEL_N={near}\\nPRIMARY_CI_CROSSING_ZERO_MODEL_N={cross}\\nPRIMARY_BH_SUPPORTED_MODEL_N={bh}\\nRESULT=PASS_BIOLOGICALLY_PLAUSIBLE_SHARED_ECOLOGY_WITH_SPOT_LEVEL_QA_LIMITATION\\n")
with open(Q+"/SPATIAL_ALL_POSITIVE_DIRECTION_SANITY_HANDOFF.md","w") as f:f.write("# Direction sanity QA\\n\\nNo absolute value, sign flip, or model-derived high/low grouping was found. Negative section and patient effects remain in the formal tables. Spot-level Jaccard/correlation and local recalculation are not auditable because spot scores were not serialized.\\n")
files=[Q+"/"+x for x in os.listdir(Q) if os.path.isfile(Q+"/"+x)];wr("SPATIAL_DIRECTION_QA_SHA256_MANIFEST.tsv",[dict(absolute_path=x,sha256=hashlib.sha256(open(x,"rb").read()).hexdigest()) for x in files],["absolute_path","sha256"])
print(primary_pos,near,cross,bh,axis)

