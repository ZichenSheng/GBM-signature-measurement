"""Final-paper numerical checks from public reference distributions and simulation worlds.
This validates and reaggregates saved reference values; it does not regenerate source observations or random panels.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'results_reference/final'
def read(name):return pd.read_csv(REF/name,sep='\t')
def run(output):
 checks=[]
 def check(ok,name,actual,expected,level='FROZEN_REFERENCE_REAGGREGATION'):
  checks.append(dict(test=name,actual=str(actual),expected=str(expected),verification_level=level,status='PASS' if bool(ok) else 'FAIL'))
  if not ok:raise AssertionError((name,actual,expected))
 obs=read('SD_Fig1D_observed_and_null_summary.tsv');null=read('SD_Fig1D_independent_matched_null_500_panels.tsv.gz');ov=read('SD_Fig1D_representation_programme_overlap.tsv')
 for r in obs.itertuples():
  v=null[(null["transform"]==r.transform)&(null.family==r.family)&(null.endpoint==r.endpoint)].null_value
  ge=int((v>=r.observed).sum());actual=(len(v),ge,float(v.max()),(1+ge)/(1+len(v)))
  check(len(v)==500 and ge==r.n_null_ge_obs and abs(v.max()-r.null_max)<1e-8 and abs((1+ge)/501-r.p_plus1)<1e-10,'Fig1D '+r.transform+' '+r.family+' '+r.endpoint,actual,(500,r.n_null_ge_obs,r.null_max,r.p_plus1))
 check(len(ov)==201 and (ov.eco_shared>0).sum()==87 and ov.n.sum()==1729 and ov.eco_shared.sum()==176,'Fig1D membership/overlap',(len(ov),int((ov.eco_shared>0).sum()),int(ov.n.sum()),int(ov.eco_shared.sum())),(201,87,1729,176))
 check(set(obs.rep_n)=={201,160,114},'Fig1D sensitivity denominators',sorted(obs.rep_n.unique()),[114,160,201])
 s=read('SD_Fig5D_ivy_state_profile_summary.tsv');st=read('SD_Fig5D_ivy_frozen_model_statistics.tsv').set_index('model_instance_id');co=read('A_coefficients_membership.tsv');gene=read('SD_Fig5D_ivy_gene_state_profiles.tsv')
 for ident,state,direction,n in [('PF4C-B4-007','MVP','above',9),('PF4C-B4-029','MVP','below',9),('PF4C-B4-029','PAN','above',8),('PF4C-FINAL-079','LE','above',8)]:
  r=s[(s.model_instance_id==ident)&(s.anatomical_structure==state)].iloc[0];actual=(int(r['tumours_'+direction+'_CT']),int(r.tumours));check(actual==(n,n),'Fig5D '+ident+' '+state+' '+direction,actual,(n,n),'FROZEN_AGGREGATE_VALIDATION')
 for ident,r2,ba,n in [('PF4C-B4-007',.530064,.348719,5),('PF4C-B4-029',.590550,.370053,4),('PF4C-FINAL-079',.655243,.493035,14)]:
  r=st.loc[ident];check(abs(r.partial_r2_ALL5_tumour_blocked-r2)<5e-7 and abs(r.balanced_accuracy_ALL5_LOTO-ba)<5e-7 and r.sample_n==122 and r.tumor_n==10,'Fig5D frozen model statistics '+ident,(r.partial_r2_ALL5_tumour_blocked,r.balanced_accuracy_ALL5_LOTO,r.sample_n,r.tumor_n),(r2,ba,122,10),'FROZEN_AGGREGATE_VALIDATION')
  c=co[co.signature_id==ident];check(len(c)==n,'Fig5D member denominator '+ident,len(c),n)
  g=gene[gene.label==r.label].set_index('gene');sign=c.set_index(c.gene.str.upper()).coefficient.apply(np.sign);check(all(g.loc[k,'coefficient_sign']==v for k,v in sign.items()),'Fig5D source coefficient signs '+ident,sign.tolist(),g.coefficient_sign.tolist())
 w=read('act4_worlds_999_exact_200k.tsv');check(len(w)==640 and w.groupby(['signature_id','q']).size().eq(20).all(),'ACT4 world denominators',len(w),640)
 check(w.p_BK_exact.notna().sum()==620,'ACT4 exact enumeration coverage',int(w.p_BK_exact.notna().sum()),620)
 check((w.T_independent-w.T_frozen).abs().max()<1e-9,'ACT4 prior independent statistic agreement',(w.T_independent-w.T_frozen).abs().max(),'<1e-9','VALIDATION_OF_PRIOR_RECONSTRUCTION')
 b=w.p_BK_exact.fillna(w.p_BK_mc)<=.05;n=w.p_NU_mc<=.05;bf=w.p_blocked_frozen<=.05;nf=w.p_naive_frozen<=.05
 enum=w.p_BK_exact.notna();check(int(((b!=bf)&enum).sum())==3,'ACT4 exact vs 999 decision changes',int(((b!=bf)&enum).sum()),3)
 ev=w.signature_id!='PF4C-FINAL-053'
 def cats(mask,n,b):return [int((mask&n&~b).sum()),int((mask&~n&b).sum()),int((mask&n&b).sum()),int((mask&~n&~b).sum())]
 for key,mask,nn,bb,exp in [('historical formal all q',ev,nf,bf,[140,114,4,302]),('high-resolution formal all q',ev,n,b,[142,114,3,301]),('historical formal q100',ev&(w.q==1),nf,bf,[51,56,4,29]),('high-resolution formal q100',ev&(w.q==1),n,b,[46,57,3,34])]:
  actual=cats(mask,nn,bb);check(actual==exp,'ACT4 '+key,actual,exp)
 for ident,old,new in [('PF4C-FINAL-054',12,6),('PF4C-FINAL-055',19,20),('PF4C-FINAL-056',4,3)]:
  mask=(w.signature_id==ident)&(w.q==1);v=(int((mask&nf).sum()),int((mask&n).sum()));check(v==(old,new),'ACT4 q100 naive '+ident,v,(old,new))
 t=read('act4_true_null_999.tsv');p=read('act4_protagonist_worlds_999.tsv')
 for ident,nd,bd,tn,tb,unique in [('PF4C-B4-007',0,20,0,5,14),('PF4C-B4-029',0,0,0,8,14),('PF4C-FINAL-079',20,0,55,3,15)]:
  g=p[(p.signature_id==ident)&(p.q==1)];z=t[(t.signature_id==ident)&(t.q==1)];actual=(len(g),int((g.naive_p<=.05).sum()),int((g.blocked_p<=.05).sum()),len(z),int((z.naive_p<=.05).sum()),int((z.blocked_p<=.05).sum()),len(np.unique(np.round(g.T_ALT,9))))
  check(actual==(20,nd,bd,100,tn,tb,unique),'Fig5G detection, true-null calibration, near-replicates '+ident,actual,(20,nd,bd,100,tn,tb,unique))
  k=(w.signature_id==ident)&(w.q==1);check(((n==nf)&(b==bf))[k].all(),'Fig5G resolution invariance '+ident,'identical','identical')
 reg=read('cgga_registry_corrected.tsv');rg=reg[reg.in_registry_naive32=='yes'];check(len(reg)==34 and len(rg)==32,'CGGA historical registry audited subset',(len(reg),len(rg)),(34,32))
 # Classification is an evidence judgement; this is a recount, not another literature review.
 used=rg.status.eq('DOCUMENTED_CGGA_USE');check(int(used.sum())==19,'CGGA involvement in former naive32',int(used.sum()),19,'CLASSIFICATION_RECOUNT')
 x=pd.read_csv(ROOT/'docs/RESULT_CODE_CROSSWALK.tsv',sep='\t').fillna('');check(len(x)==85 and x.result_id.is_unique,'Crosswalk identity completeness',(len(x),x.result_id.is_unique),(85,True),'STRUCTURAL_VALIDATION')
 check((x.lifecycle=='WITHDRAWN').sum()==4 and (x.loc[x.lifecycle=='WITHDRAWN','figure_panel']=='NONE_WITHDRAWN').all(),'Withdrawn analyses outside active figures',int((x.lifecycle=='WITHDRAWN').sum()),4,'STRUCTURAL_VALIDATION')
 check(all((ROOT/p).exists() for p in x.public_reproduction_module),'Crosswalk public module paths','all exist','all exist','STRUCTURAL_VALIDATION')
 inv=pd.read_csv(ROOT/'provenance/SOURCE_RECOVERY_INVENTORY.tsv',sep='\t');inv=inv[inv.gap=='A'];counts=inv.classification.value_counts().to_dict();check(len(inv)==51 and counts=={'SOURCE_FOUND_AND_INDEPENDENTLY_REPRODUCED':36,'SOURCE_FOUND_AND_AUTHOR_CODE_REPRODUCED':15},'Recovered 51 provenance classifications',counts,{'independent':36,'author':15},'PRIOR_EXECUTION_INVENTORY_RECONCILIATION')
 output=Path(output);output.mkdir(parents=True,exist_ok=True);pd.DataFrame(checks).to_csv(output/'final_reference_checks.tsv',sep='\t',index=False)
 inputs=[p for p in REF.iterdir() if p.is_file()];pd.DataFrame([{'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]).to_csv(output/'reference_input_hashes.tsv',sep='\t',index=False)
 print('PASS:',len(checks),'scoped final-paper checks; no raw-data or full-simulation regeneration claim.');return checks
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='outputs/final_validation');a=p.parse_args();run(a.output)
