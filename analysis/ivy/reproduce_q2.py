"""Run corrected production Q2 using the historical seed AND branch order.
Input: original externally obtained Ivy membership-score table. No raw data are distributed.
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

def load():
 p=Path(__file__).with_name("10_run_analysis.py")
 spec=importlib.util.spec_from_file_location("ivy_production",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def run(scores, output):
 A=load();s=pd.read_csv(scores,sep="\t",dtype={"sample_id":str,"tumor_id":str});y=s.iloc[:,4:].to_numpy(float)
 assert y.shape==(122,201)
 rng=np.random.default_rng(2026083102);counts={};out=[]
 for branch,levels in A.BRANCHES.items():
  k=s.anatomical_structure.isin(levels).to_numpy()
  r,_=A.q2_identifiability(y[k],s.anatomical_structure.to_numpy()[k],s.tumor_id.to_numpy()[k],levels,rng)
  r.insert(0,"model_instance_id",s.columns[4:]);r.insert(0,"branch",branch);out.append(r)
  counts[branch]=int((r.bh_fdr<.05).sum());print(branch,counts[branch],flush=True)
 assert counts=={"ALL5":170,"CORE3":134,"PAN_MVP":98},counts
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 pd.concat(out).to_csv(output/"corrected_q2_model_level.tsv",sep="\t",index=False)
 (output/"corrected_q2_counts.json").write_text(json.dumps(counts,indent=2)+"\n")
 return counts
if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scores",required=True);p.add_argument("--output",default="outputs/ivy_q2");a=p.parse_args();run(a.scores,a.output)
