"""Focused original-author CARE WP1C replay on external prepared inputs (not raw-data reconstruction)."""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

def run(source_root,output):
 sys.path.insert(0,str(Path(__file__).parent))
 spec=importlib.util.spec_from_file_location("care_author",Path(__file__).with_name("10_wp1.py"));A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
 root=Path(source_root);v1=root/"GBM_SIGNATURE_CONTEXT_STRESS_TEST_EXTENSION_V1_FINAL/04_model_scores/DS01"
 member=pd.read_csv(root/"GBM_SIGNATURE_CONTEXT_EXTENSION_DEEPENING_V2_FINAL/00_governance/freeze/memberships/AS_CARE_LOPO56.tsv",sep="\t",dtype=str)
 cs=pd.read_parquet(v1/"PRIMARY201_MODEL_SCORES.parquet");ca=pd.read_parquet(v1/"FROZEN_ECOLOGY_AXIS_SCORES.parquet")
 def filt(x):return x[(x.score_scope=="ALL_AUTHOR_ANNOTATED")&(x.timepoint=="T1")&x.patient_id.isin(member.patient_id)]
 cs=filt(cs);ca=filt(ca);s=cs.pivot(index="patient_id",columns="model_instance_id",values="model_score").sort_index();y=ca.pivot(index="patient_id",columns="axis_id",values="axis_score").reindex(s.index).reindex(columns=A.AXES).to_numpy(float)
 assert s.shape==(56,201) and set(s.index)==set(member.patient_id) and not set(s.index)&{"P53","P55","P64"}
 rng=np.random.default_rng(20260830+120);folds=[list(np.array_split(rng.permutation(56),5)) for _ in range(100)]
 assert all(np.array_equal(np.sort(np.concatenate(f)),np.arange(56)) for f in folds)
 rows=[]
 for model in s.columns:
  z=A.repeated_fivefold_r2(s[model].to_numpy(float),y,folds)
  rows.append({"model_instance_id":model,"mean_CV_R2":float(z.mean()),"median_repeat_CV_R2":float(np.median(z))})
 out=pd.DataFrame(rows);median=float(out.mean_CV_R2.median());assert abs(median-.0903096094913944)<1e-12
 frozen=pd.read_csv(root/"GBM_SIGNATURE_FINAL_SPECIFIC_GAP_CLOSURE_V1_FINAL/01_wp1_scale_identifiability/CARE_BASELINE_IDENTIFIABILITY.tsv",sep="\t");m=out.merge(frozen,on="model_instance_id");delta=float(abs(m.mean_CV_R2_x-m.mean_CV_R2_y).max());assert delta<1e-11
 # Fold-only leakage probe: author fit never accesses held-out outcomes.
 te=folds[0][0];tr=np.setdiff1d(np.arange(56),te);x=s.iloc[:,0].to_numpy();design=np.c_[np.ones(len(tr)),x[tr]]
 beta=np.linalg.lstsq(design,y[tr],rcond=None)[0];changed=y.copy();changed[te]=np.random.default_rng(1).normal(size=y[te].shape);beta2=np.linalg.lstsq(design,changed[tr],rcond=None)[0];assert np.array_equal(beta,beta2)
 output=Path(output);output.mkdir(parents=True,exist_ok=True);out.to_csv(output/"care_per_representation.tsv",sep="\t",index=False)
 res={"patient_n":56,"representation_n":201,"median_mean_CV_R2":median,"median_median_repeat_CV_R2":float(out.median_repeat_CV_R2.median()),"per_representation_max_difference":delta,"leakage_probe_max_difference":float(abs(beta-beta2).max())};(output/"care_result.json").write_text(json.dumps(res,indent=2)+"\n");print(json.dumps(res));return res
if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source-root",required=True);p.add_argument("--output",default="outputs/care");a=p.parse_args();run(a.source_root,a.output)
