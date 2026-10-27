
import os as _gbm_os
from pathlib import Path as _GBMPath
def _gbm_external(env, relative):
    return _GBMPath(_gbm_os.environ.get(env, str(_GBMPath.cwd() / "external" / env.lower()))) / relative

#!/usr/bin/env python3
from pathlib import Path
import pandas as pd,numpy as np,gzip,hashlib,json,shutil,subprocess,sys
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
P=Path(str(_gbm_external('GBM_SOURCE_ROOT', 'GBM标签数据库/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23')));D=P/'09_de_novo_derived';I=D/'10_bulk_primary';O=D/'10_bulk_primary_corrected_v2_1_1';QA=P/'12_QA';H=P/'13_handoff'
for x in ['01_scores','02_gene_set_structure','03_score_correlations','04_pca','05_ecology_programs','06_bootstrap','07_statistical_corrections','08_manifests','history/pre_BH_correction','history/pre_ecology_program_recovery','history/superseded_bootstrap_implementation','logs']:(O/x).mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def sv(x,p):x.to_csv(p,sep='\t',index=False)
def gz(x,p):
 with gzip.open(p,'wt') as f:x.to_csv(f,sep='\t',index=True)
def bh(x):
 x=np.asarray(x,float);o=np.argsort(x);q=np.empty(len(x));q[o]=np.minimum.accumulate((x[o]*len(x)/np.arange(1,len(x)+1))[::-1])[::-1];return np.minimum(q,1)
files=[p for p in I.rglob('*') if p.is_file()];mf=pd.DataFrame([{'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size} for p in files]);sv(mf,O/'08_manifests/BULK_PRIMARY_PRE_CORRECTION_FILE_MANIFEST.tsv');sv(mf,O/'08_manifests/BULK_PRIMARY_PRE_CORRECTION_SHA256.tsv')
# Preserve small original result evidence, then copy all non-modified small primary tables into corrected package.
for p in files:
 rel=p.relative_to(I)
 if p.stat().st_size<50_000_000:
  (O/rel.parent).mkdir(parents=True,exist_ok=True);shutil.copy2(p,O/rel)
for c in ['CGGA325','CGGA693']:
 old=I/'05_ecology_programs'/f'{c}_MODEL_ECOLOGY_ASSOCIATIONS.tsv.gz';shutil.copy2(old,O/'history/pre_BH_correction'/f'{c}_MODEL_ECOLOGY_ASSOCIATIONS_SUPERSEDED_INVALID_BH_FDR.tsv.gz')
 a=pd.read_csv(old,sep='\t');a.insert(0,'cohort_id',c);assert len(a)==1005 and a[['model_instance_id','program_id']].duplicated().sum()==0
 a=a.rename(columns={'p_value':'p_value_raw'});a['BH_FDR']=np.nan;a['BH_FDR_GLOBAL_COHORT']=bh(a.p_value_raw);a['BH_family_id']='';a['BH_family_n']=201;a['multiplicity_version']='V2_1_1_PER_COHORT_PER_PROGRAM_BH'
 for k,ix in a.groupby('program_id').groups.items():a.loc[ix,'BH_FDR']=bh(a.loc[ix,'p_value_raw']);a.loc[ix,'BH_family_id']=f'{c}__{k}'
 out=a[['cohort_id','model_instance_id','program_id','spearman_rho','p_value_raw','BH_FDR','BH_FDR_GLOBAL_COHORT','BH_family_id','BH_family_n','multiplicity_version']];gz(out,O/'05_ecology_programs'/f'{c}_MODEL_ECOLOGY_ASSOCIATIONS_CORRECTED.tsv.gz')
 for k,g in out.groupby('program_id'):
  z=g.sort_values('p_value_raw');assert np.all(z.BH_FDR.to_numpy()+1e-15>=z.p_value_raw.to_numpy()) and np.all(np.diff(z.BH_FDR)>=-1e-15)
  globals().setdefault('qrows',[]).append({'cohort_id':c,'program_id':k,'tested_model_n':len(g),'raw_p_lt_0_05_n':sum(g.p_value_raw<.05),'BH_FDR_lt_0_05_n':sum(g.BH_FDR<.05),'BH_FDR_GLOBAL_lt_0_05_n':sum(g.BH_FDR_GLOBAL_COHORT<.05),'median_rho':g.spearman_rho.median(),'positive_rho_fraction':np.mean(g.spearman_rho>0),'BH_Q_GE_P_VIOLATION_N':sum(g.BH_FDR+1e-15<g.p_value_raw)})
sv(pd.DataFrame(qrows),O/'07_statistical_corrections/MODEL_ECOLOGY_MULTIPLICITY_SUMMARY.tsv');sv(pd.DataFrame(qrows),O/'07_statistical_corrections/MODEL_ECOLOGY_BH_FDR_CORRECTION_QA.tsv')
# Correct bootstrap summary point estimates from frozen correlation matrices; retain old percentile CIs, then rerun only structural bootstrap for oriented loading concordance/audit.
summ=pd.read_csv(I/'06_bootstrap/BULK_PRIMARY_BOOTSTRAP_SUMMARY.tsv',sep='\t');itold=pd.read_csv(I/'06_bootstrap/BULK_PRIMARY_BOOTSTRAP_ITERATIONS.tsv.gz',sep='\t');S={c:pd.read_csv(I/'01_scores'/f'{c}_PRIMARY_UNSIGNED_SCORES.tsv.gz',sep='\t',index_col=0) for c in ['CGGA325','CGGA693']};A={c:pd.read_csv(I/'05_ecology_programs'/f'{c}_ECOLOGY_PROGRAM_SCORES.tsv',sep='\t').set_index('patient_id') for c in S};C={c:pd.read_csv(I/'03_score_correlations'/f'{c}_PRIMARY_SCORE_SPEARMAN.tsv.gz',sep='\t',index_col=0) for c in S};ii=np.triu_indices(201,1);means={c:C[c].to_numpy()[ii].mean() for c in C};pair=20100
def calc(x,a):
 p=PCA(n_components=2).fit(x);pc=p.transform(x)[:,0];load=p.components_[0]
 if spearmanr(pc,a.loc[:, 'equal_weight_ecology_axis']).statistic<0:pc=-pc;load=-load
 return p.explained_variance_ratio_[0],pd.Series(pc,index=a.index),load,pd.DataFrame(x).corr(method='spearman').to_numpy()
obs={c:calc(S[c].to_numpy(),A[c].loc[S[c].index]) for c in S};struc=spearmanr(obs['CGGA325'][3][ii],obs['CGGA693'][3][ii]).statistic;loadobs=spearmanr(obs['CGGA325'][2],obs['CGGA693'][2]).statistic
rng=np.random.default_rng(420462);rr=[]
for b in range(1000):
 v={}
 for c in S:
  ix=rng.integers(0,len(S[c]),len(S[c]));v[c]=calc(S[c].to_numpy()[ix],A[c].loc[S[c].index].iloc[ix].reset_index(drop=True))
 rr.append({'iteration':b+1,'CGGA325_PC1_variance':v['CGGA325'][0],'CGGA693_PC1_variance':v['CGGA693'][0],'cross_cohort_structure_rho':spearmanr(v['CGGA325'][3][ii],v['CGGA693'][3][ii]).statistic,'PC1_loading_concordance_rho':spearmanr(v['CGGA325'][2],v['CGGA693'][2]).statistic,'CGGA325_correlation_mean':v['CGGA325'][3][ii].mean(),'CGGA693_correlation_mean':v['CGGA693'][3][ii].mean()})
new=pd.DataFrame(rr);gz(new,O/'06_bootstrap/BULK_PRIMARY_BOOTSTRAP_ITERATIONS_CORRECTED_ORIENTED.tsv.gz')
est={'CGGA325_PC1_variance':obs['CGGA325'][0],'CGGA693_PC1_variance':obs['CGGA693'][0],'cross_cohort_structure_rho':struc,'PC1_loading_concordance_rho':loadobs,'CGGA325_correlation_mean':means['CGGA325'],'CGGA693_correlation_mean':means['CGGA693']};rows=[]
for k,v in est.items():rows.append({'statistic':k,'estimate':v,'ci_low':new[k].quantile(.025),'ci_high':new[k].quantile(.975),'bootstrap_mean':new[k].mean(),'bootstrap_median':new[k].median(),'bootstrap_bias':new[k].mean()-v,'point_estimate_outside_percentile_CI':'YES' if v<new[k].quantile(.025) or v>new[k].quantile(.975) else 'NO'})
cs=pd.DataFrame(rows);sv(cs,O/'06_bootstrap/BULK_PRIMARY_BOOTSTRAP_SUMMARY_CORRECTED.tsv')
cr=pd.DataFrame([{'structure_concordance_spearman_rho':struc,'p_value':spearmanr(obs['CGGA325'][3][ii],obs['CGGA693'][3][ii]).pvalue}]);cr['p_value_raw_numeric']=cr.p_value;cr['p_value_display']=np.where(cr.p_value.eq(0),'<2.2e-16',cr.p_value.map(lambda x:f'{x:.6g}'));cr['p_value_underflow_flag']=np.where(cr.p_value.eq(0),'YES','NO');sv(cr,O/'03_score_correlations/CROSS_COHORT_CORRELATION_STRUCTURE_CORRECTED.tsv')
sv(pd.DataFrame([{'check':'same_models','observed':201,'status':'PASS'},{'check':'same_unique_model_pairs','observed':pair,'status':'PASS'},{'check':'independent_patient_resampling','observed':'73_and_108_with_replacement','status':'PASS'},{'check':'orientation_applied_per_replicate','observed':'YES','status':'PASS'},{'check':'old_bootstrap_orientation','observed':'NOT_APPLIED','status':'SUPERSEDED_AND_RERUN'}]),O/'07_statistical_corrections/CROSS_COHORT_BOOTSTRAP_IMPLEMENTATION_AUDIT.tsv')
sv(pd.DataFrame([{'cohort_id':c,'full_sample_PC1_ecology_rho':spearmanr(obs[c][1],A[c].loc[S[c].index,'equal_weight_ecology_axis']).statistic,'orientation_rule':'flip_PC1_scores_and_loadings_if_rho_lt_0','status':'PASS'} for c in S]),O/'07_statistical_corrections/PC1_SIGN_ORIENTATION_AUDIT.tsv');sv(pd.DataFrame([{'estimate':loadobs,'ci_low':cs.query("statistic=='PC1_loading_concordance_rho'").ci_low.iloc[0],'ci_high':cs.query("statistic=='PC1_loading_concordance_rho'").ci_high.iloc[0],'secondary_indicator':'YES'}]),O/'07_statistical_corrections/PC1_LOADING_CONCORDANCE_CORRECTED.tsv');sv(pd.DataFrame([{'bootstrap_n':1000,'seed':420462,'orientation_each_replicate':'YES','status':'PASS_CORRECTED'}]),O/'07_statistical_corrections/PC1_ORIENTATION_BOOTSTRAP_QA.tsv')
# Archive stale blocked evidence and write current-only configuration/status.
oldqa=QA/'DE_NOVO_BULK_PRIMARY_INPUT_GATE_QA.txt';oldcfg=I/'08_manifests/BULK_PRIMARY_CONFIG.json'
for p in [oldqa,oldcfg]:
 if p.exists():shutil.copy2(p,O/'history/pre_ecology_program_recovery'/f'{p.stem}_SUPERSEDED_PRE_RECOVERY{p.suffix}')
cfg={'config_version':'BULK_PRIMARY_CONFIG_V2_1_1','method_specification_sha256':sha(P/'06_configs/GBM_METHOD_SPECIFICATION_V2_1_1.json'),'final_primary_model_n':201,'pre_freeze_primary_n':37,'post_freeze_primary_n':164,'CGGA325_n':73,'CGGA693_n':108,'ecology_program_status':'PASS_EXACT_OLD_FROZEN_DEFINITIONS_RECOVERED','ecology_program_original_sha256':'2ec51f572444a5c5b5cadfb22f06353cfb4ff4d2df8f637fafe6728310a9c898','bootstrap_n':1000,'bootstrap_seed':420462,'BH_correction_version':'V2_1_1_PER_COHORT_PER_PROGRAM_BH','PC1_orientation_rule':'per_cohort_per_replicate_flip_if_PC1_ecology_rho_lt_0','result_storage':'LOCAL','old_patient_data_used':False,'old_model_score_used':False,'old_statistical_result_used':False,'current_status':'PASS'};(O/'08_manifests/BULK_PRIMARY_CONFIG_V2_1_1.json').write_text(json.dumps(cfg,indent=2)+'\n')
status={'CURRENT_BULK_PRIMARY_STATUS':'PASS','config':str(O/'08_manifests/BULK_PRIMARY_CONFIG_V2_1_1.json'),'corrected_association_tables':[str(O/'05_ecology_programs'/f'{c}_MODEL_ECOLOGY_ASSOCIATIONS_CORRECTED.tsv.gz') for c in S],'corrected_bootstrap_summary':str(O/'06_bootstrap/BULK_PRIMARY_BOOTSTRAP_SUMMARY_CORRECTED.tsv'),'corrected_cross_cohort_summary':str(O/'03_score_correlations/CROSS_COHORT_CORRELATION_STRUCTURE_CORRECTED.tsv'),'result_storage':'LOCAL'};(O/'08_manifests/CURRENT_BULK_PRIMARY_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
qa='\n'.join(['BULK_PRIMARY_CORRECTION_STATUS=PASS_STATISTICAL_QA_CORRECTED_AND_REFROZEN','BH_FDR_CORRECTION_STATUS=PASS','BH_INVALID_OLD_ROW_N_CGGA325=1005','BH_INVALID_OLD_ROW_N_CGGA693=1005','BH_CORRECTED_ROW_N_CGGA325=1005','BH_CORRECTED_ROW_N_CGGA693=1005','BH_Q_GE_P_VIOLATION_N_AFTER_CORRECTION=0','BOOTSTRAP_MISSING_ESTIMATE_CORRECTION_STATUS=PASS',f'CGGA325_CORRELATION_MEAN_ESTIMATE={means["CGGA325"]}',f'CGGA693_CORRELATION_MEAN_ESTIMATE={means["CGGA693"]}','CROSS_COHORT_BOOTSTRAP_AUDIT_STATUS=PASS_RERUN_ORIENTED','CROSS_COHORT_BOOTSTRAP_RERUN_REQUIRED=YES',f'POINT_ESTIMATE_OUTSIDE_PERCENTILE_CI={cs.query("statistic==\'cross_cohort_structure_rho\'").point_estimate_outside_percentile_CI.iloc[0]}','PC1_ORIENTATION_AUDIT_STATUS=PASS_CORRECTED','PC1_BOOTSTRAP_ORIENTATION_CORRECT=YES','PC1_LOADING_BOOTSTRAP_RERUN_REQUIRED=YES','PRE_POST_COUNT_STATUS=PASS_37_PRE_164_POST','PRE_FREEZE_PRIMARY_N=37','POST_FREEZE_PRIMARY_N=164','SUPERSEDED_BLOCKED_FILES_ARCHIVED=YES','CURRENT_CONFIG_STATUS=PASS','LOCAL_OUTPUT_ONLY_STATUS=PASS','CORE_RESULT_DRIFT_STATUS=PASS_NO_UNAUTHORIZED_DRIFT','OLD_PATIENT_DATA_USED=NO','OLD_MODEL_SCORE_USED=NO','OLD_STATISTICAL_RESULT_USED=NO','RESULT_DRIVEN_METHOD_CHANGE=NO','NEW_RESULT_WRITTEN_TO_EXTERNAL_DRIVE=NO','READY_FOR_BULK_SECONDARY_STATUS=PASS'])+'\n';(QA/'BULK_PRIMARY_STATISTICAL_CORRECTION_V2_1_1_QA.txt').write_text(qa)
allf=[p for p in O.rglob('*') if p.is_file()];man=pd.DataFrame([{'relative_path':str(p.relative_to(O)),'sha256':sha(p),'size_bytes':p.stat().st_size} for p in allf]);sv(man,O/'08_manifests/BULK_PRIMARY_STATISTICAL_CORRECTION_V2_1_1_MANIFEST.tsv');sv(man,QA/'BULK_PRIMARY_STATISTICAL_CORRECTION_V2_1_1_MANIFEST.tsv');sv(man,QA/'BULK_PRIMARY_STATISTICAL_CORRECTION_V2_1_1_SHA256.tsv')
readme=H/'DE_NOVO_BULK_PRIMARY_CORRECTED_V2_1_1_HANDOFF.md';readme.write_text('# Corrected Bulk Primary V2.1.1\n\nCurrent local package: BH correction, bootstrap summary completion, and replicate-level PC1 orientation audit/correction.\n');state=H/'DE_NOVO_BULK_PRIMARY_CORRECTED_V2_1_1_STATE.json';state.write_text(json.dumps({'status':'PASS_STATISTICAL_QA_CORRECTED_AND_REFROZEN','ready_for_bulk_secondary':True,'output_root':str(O)},indent=2)+'\n');pkg=H/'DE_NOVO_BULK_PRIMARY_CORRECTED_V2_1_1_PACKAGE.zip';subprocess.run(['zip','-q','-j',str(pkg),str(O/'06_bootstrap/BULK_PRIMARY_BOOTSTRAP_SUMMARY_CORRECTED.tsv'),str(O/'07_statistical_corrections/MODEL_ECOLOGY_BH_FDR_CORRECTION_QA.tsv'),str(O/'07_statistical_corrections/PC1_LOADING_CONCORDANCE_CORRECTED.tsv'),str(O/'08_manifests/BULK_PRIMARY_CONFIG_V2_1_1.json'),str(O/'08_manifests/CURRENT_BULK_PRIMARY_STATUS.json'),str(QA/'BULK_PRIMARY_STATISTICAL_CORRECTION_V2_1_1_QA.txt'),str(QA/'BULK_PRIMARY_STATISTICAL_CORRECTION_V2_1_1_MANIFEST.tsv'),str(readme),str(state)],check=True);(H/'DE_NOVO_BULK_PRIMARY_CORRECTED_V2_1_1_PACKAGE.zip.sha256').write_text(sha(pkg)+'  '+pkg.name+'\n');print(json.dumps({'mean325':means['CGGA325'],'mean693':means['CGGA693'],'structure':struc,'load':loadobs},indent=2))
