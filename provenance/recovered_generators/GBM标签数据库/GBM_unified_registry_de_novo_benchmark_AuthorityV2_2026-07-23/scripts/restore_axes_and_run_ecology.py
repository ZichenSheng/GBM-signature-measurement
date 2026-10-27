
import os as _gbm_os
from pathlib import Path as _GBMPath
def _gbm_external(env, relative):
    return _GBMPath(_gbm_os.environ.get(env, str(_GBMPath.cwd() / "external" / env.lower()))) / relative

from pathlib import Path
import pandas as pd, numpy as np, hashlib, json, shutil, gzip
from scipy.stats import spearmanr
P=Path(str(_gbm_external('GBM_SOURCE_ROOT', 'GBM标签数据库/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23')));D=Path('/external/GBM_LEGACY_DATA_ROOT/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23/09_de_novo_derived');O=D/'10_bulk_primary';F=Path(str(_gbm_external('GBM_SOURCE_ROOT', '04_single_cell/02_bulk_axis_freeze/frozen_reference_axes_gene_long.tsv'))); R=P/'02_authority/ecology_programs';R.mkdir(parents=True,exist_ok=True)
def h(p):
 q=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):q.update(b)
 return q.hexdigest()
def gz(d,p):
 with gzip.open(p,'wt') as f:d.to_csv(f,sep='\t',index=True)
def sv(d,p):d.to_csv(p,sep='\t',index=False)
assert F.exists() and h(F)=='2ec51f572444a5c5b5cadfb22f06353cfb4ff4d2df8f637fafe6728310a9c898'
orig=R/'frozen_reference_axes_gene_long_ORIGINAL.tsv';shutil.copy2(F,orig); a=pd.read_csv(F,sep='\t',dtype=str); target={'MES_Combined':'MES','Hallmark_Hypoxia':'HYPOXIA','Stromal_Matrix_Proxy':'MATRIX','Myeloid_Proxy':'MYELOID'}; z=a[a.axis_id.isin(target)].copy();z['program_id']=z.axis_id.map(target);z['program_name']=z.program_id;z['gene_symbol_normalized']=z.gene_symbol.str.strip().str.upper();z['source_file']=str(F);z['source_file_sha256']=h(F);z['definition_status']='PASS_EXACT_OLD_FROZEN_IMPLEMENTATION';z=z[['program_id','program_name','axis_id','gene_symbol','gene_symbol_normalized','frozen_order','source_file','source_file_sha256','definition_status']];z=z.drop_duplicates(['program_id','gene_symbol_normalized']);sv(z,R/'FROZEN_ECOLOGY_PROGRAM_MEMBERSHIP_STANDARDIZED.tsv')
defs=pd.DataFrame([{'program_id':'MES','program_name':'MES','subprogram_if_any':'MES_Combined (old frozen file also retains MES1_Neftel/MES2_Neftel)','gene_n':len(z[z.program_id=='MES']),'scoring_rule':'Current preprocessed expression: within-cohort gene z-standardization; mean across MES_Combined genes','aggregation_rule':'Use exact old MES_Combined membership; do not infer MES1/MES2 union','source_file':str(F)},{'program_id':'HYPOXIA','program_name':'hypoxia','subprogram_if_any':'Hallmark_Hypoxia','gene_n':len(z[z.program_id=='HYPOXIA']),'scoring_rule':'within-cohort gene z-standardization; mean','aggregation_rule':'mean','source_file':str(F)},{'program_id':'MATRIX','program_name':'matrix/ECM','subprogram_if_any':'Stromal_Matrix_Proxy','gene_n':len(z[z.program_id=='MATRIX']),'scoring_rule':'within-cohort gene z-standardization; mean','aggregation_rule':'mean','source_file':str(F)},{'program_id':'MYELOID','program_name':'myeloid','subprogram_if_any':'Myeloid_Proxy','gene_n':len(z[z.program_id=='MYELOID']),'scoring_rule':'within-cohort gene z-standardization; mean','aggregation_rule':'mean','source_file':str(F)}]);sv(defs,R/'FROZEN_ECOLOGY_PROGRAM_DEFINITIONS.tsv')
# Register recovered immutable method inputs.
ep=P/'06_configs/CURRENT_DERIVED_DATA_ENTRYPOINTS.tsv';E=pd.read_csv(ep,sep='\t');
for asset,path in [('FROZEN_ECOLOGY_PROGRAM_MEMBERSHIP',R/'FROZEN_ECOLOGY_PROGRAM_MEMBERSHIP_STANDARDIZED.tsv'),('FROZEN_ECOLOGY_PROGRAM_DEFINITIONS',R/'FROZEN_ECOLOGY_PROGRAM_DEFINITIONS.tsv')]:
 E=E[E.asset_id!=asset]; E=pd.concat([E,pd.DataFrame([{'asset_id':asset,'asset_type':'frozen_method_input','canonical_external_path':str(path),'local_manifest_path':str(path),'sha256':h(path),'status':'PASS','formal_role':'FROZEN_METHOD_INPUT','source_project':'OLD_FROZEN_METHOD_RECOVERY'}])],ignore_index=True)
E.to_csv(ep,sep='\t',index=False)
(P/'06_configs/CURRENT_BULK_PRIMARY_CONFIG.json').write_text(json.dumps({'ecology_program_membership_path':str(R/'FROZEN_ECOLOGY_PROGRAM_MEMBERSHIP_STANDARDIZED.tsv'),'ecology_program_definition_path':str(R/'FROZEN_ECOLOGY_PROGRAM_DEFINITIONS.tsv'),'ecology_program_membership_sha256':h(R/'FROZEN_ECOLOGY_PROGRAM_MEMBERSHIP_STANDARDIZED.tsv'),'ecology_program_definition_sha256':h(R/'FROZEN_ECOLOGY_PROGRAM_DEFINITIONS.tsv'),'ecology_program_source':'OLD_FROZEN_METHOD_RECOVERY','old_results_imported':False},indent=2))
membership={k:z.loc[z.program_id==k,'gene_symbol_normalized'].tolist() for k in target.values()}; outcome=[];modelout=[];boots=[];rng=np.random.default_rng(420462)
for cohort in ['CGGA325','CGGA693']:
 X=pd.read_csv(D/'03_preprocessed_expression'/f'{cohort}_STRICT_EXPRESSION_PREPROCESSED.tsv',sep='\t',index_col=0);X.index=X.index.astype(str).str.upper();Z=X.sub(X.mean(1),axis=0).div(X.std(1,ddof=1).replace(0,np.nan),axis=0)
 S=pd.DataFrame({k:Z.loc[[g for g in gs if g in Z.index]].mean(0) for k,gs in membership.items()},index=X.columns); assert S.notna().all().all() and np.isfinite(S.to_numpy()).all(); EZ=S.sub(S.mean()).div(S.std(ddof=1));S['equal_weight_ecology_axis']=EZ.mean(1);sv(S.reset_index(names='patient_id'),O/'05_ecology_programs'/f'{cohort}_ECOLOGY_PROGRAM_SCORES.tsv')
 PC=pd.read_csv(O/'04_pca'/f'{cohort}_PCA_PATIENT_SCORES.tsv',sep='\t').set_index('patient_id').loc[S.index,'PC1']; r0=spearmanr(PC,S.equal_weight_ecology_axis).statistic; flip=-1 if r0<0 else 1; PC=PC*flip
 for col in S.columns:
  rr,pp=spearmanr(PC,S[col]);outcome.append({'cohort_id':cohort,'association':'PC1_vs_'+col,'spearman_rho':rr,'p_value':pp,'pc1_sign_flip_applied':flip==-1})
 MS=pd.read_csv(O/'01_scores'/f'{cohort}_PRIMARY_UNSIGNED_SCORES.tsv.gz',sep='\t',index_col=0).loc[S.index]
 rows=[]
 for m in MS:
  for col in S.columns:
   rr,pp=spearmanr(MS[m],S[col]);rows.append({'model_instance_id':m,'program_id':col,'spearman_rho':rr,'p_value':pp})
 md=pd.DataFrame(rows);order=md.p_value.argsort();rank=np.empty(len(md),int);rank[order]=np.arange(1,len(md)+1);md['BH_FDR']=np.minimum.accumulate((md.p_value.to_numpy()[order]*len(md)/np.arange(1,len(md)+1))[::-1])[np.argsort(order)];gz(md.set_index(['model_instance_id','program_id']),O/'05_ecology_programs'/f'{cohort}_MODEL_ECOLOGY_ASSOCIATIONS.tsv.gz')
 for b in range(1000):
  ix=rng.integers(0,len(S),len(S));
  for col in S.columns: boots.append({'iteration':b+1,'cohort_id':cohort,'statistic':'PC1_vs_'+col,'rho':spearmanr(PC.to_numpy()[ix],S[col].to_numpy()[ix]).statistic})
sv(pd.DataFrame(outcome),O/'05_ecology_programs/PC1_ECOLOGY_ASSOCIATIONS.tsv');sv(pd.DataFrame(boots),O/'06_bootstrap/ECOLOGY_BOOTSTRAP_ITERATIONS.tsv'); B=pd.DataFrame(boots);sv(B.groupby(['cohort_id','statistic']).rho.agg(estimate='median',ci_low=lambda x:x.quantile(.025),ci_high=lambda x:x.quantile(.975)).reset_index(),O/'06_bootstrap/ECOLOGY_BOOTSTRAP_SUMMARY.tsv')
(O/'05_ecology_programs/ECOLOGY_PROGRAM_ANALYSIS_STATUS.txt').write_text('PASS\nOriginal frozen membership restored; scores and associations use new cohorts only.\n')
print(json.dumps({'membership_counts':{k:len(v) for k,v in membership.items()},'bootstrap_n':len(B)},indent=2))
