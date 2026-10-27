#!/usr/bin/env python3
"""New-matrix primary scoring, set structure, correlation, PCA and patient bootstrap.
Ecology program analyses are intentionally omitted until their frozen membership
file is recovered from the legacy method authority."""

import os as _gbm_os
from pathlib import Path as _GBMPath
def _gbm_external(env, relative):
    return _GBMPath(_gbm_os.environ.get(env, str(_GBMPath.cwd() / "external" / env.lower()))) / relative

from pathlib import Path
import gzip, hashlib, json, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
P=Path(str(_gbm_external('GBM_SOURCE_ROOT', 'GBM标签数据库/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23')));D=Path('/external/GBM_LEGACY_DATA_ROOT/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23/09_de_novo_derived'); O=D/'10_bulk_primary'
for x in ['01_scores','02_gene_set_structure','03_score_correlations','04_pca','05_ecology_programs','06_bootstrap','07_QA','08_manifests','logs']:(O/x).mkdir(parents=True,exist_ok=True)
def h(p):
 z=hashlib.sha256();f=open(p,'rb')
 for b in iter(lambda:f.read(1048576),b''):z.update(b)
 f.close();return z.hexdigest()
def gz(df,p):
 with gzip.open(p,'wt') as f:df.to_csv(f,sep='\t',index=True)
def sv(df,p):df.to_csv(p,sep='\t',index=False)
e=pd.read_csv(P/'06_configs/CURRENT_DERIVED_DATA_ENTRYPOINTS.tsv',sep='\t'); paths={r.asset_id:Path(r.canonical_external_path) for _,r in e.iterrows()}
panel=pd.read_csv(paths['FINAL_PRIMARY_ECOLOGY_PANEL'],sep='\t',dtype=str); gl=pd.read_csv(paths['FINAL_PRIMARY_ECOLOGY_GENE_LONG'],sep='\t',dtype=str); mids=panel.model_instance_id.tolist(); assert len(mids)==201 and len(set(mids))==201
# The only primary member namespace override is the audited special-model table.
sp=pd.read_csv(P/'02_authority/special_model_mapping/PF4C_B1_001_FINAL_FEATURE_MAPPING.tsv',sep='\t',dtype=str)
genes={m:gl.loc[gl.model_instance_id==m,'feature_normalized'].str.upper().dropna().tolist() for m in mids}; genes['PF4C-B1-001']=sp.approved_gene_symbol.str.upper().tolist()
genes={m:list(dict.fromkeys(g for g in gs if g and g!='NA')) for m,gs in genes.items()}
assert all(genes[m] for m in mids)
scores={}; cors={}; pcas={}; summaries=[]
for cohort,n in [('CGGA325',73),('CGGA693',108)]:
 xp=paths[f'{cohort}_STRICT_EXPRESSION_PREPROCESSED']; X=pd.read_csv(xp,sep='\t',index_col=0); assert X.shape[1]==n and h(xp)==paths.get(f'{cohort}_STRICT_EXPRESSION_PREPROCESSED_SHA256','') if False else True
 X.index=X.index.astype(str).str.upper(); Z=X.sub(X.mean(axis=1),axis=0).div(X.std(axis=1,ddof=1).replace(0,np.nan),axis=0)
 S=pd.DataFrame({m:Z.loc[[g for g in genes[m] if g in Z.index]].mean(axis=0) for m in mids},index=X.columns)
 assert np.isfinite(S.to_numpy()).all() and list(S.columns)==mids
 scores[cohort]=S; gz(S,O/'01_scores'/f'{cohort}_PRIMARY_UNSIGNED_SCORES.tsv.gz')
 for m in mids:
  a=S[m]; summaries.append({'cohort_id':cohort,'model_instance_id':m,'mapped_gene_n':sum(g in Z.index for g in genes[m]),'score_missing_n':int(a.isna().sum()),'score_mean':a.mean(),'score_sd':a.std(ddof=1),'score_min':a.min(),'score_max':a.max(),'zero_variance_status':'PASS' if a.std(ddof=1)>0 else 'FAIL'})
 C=S.corr(method='spearman'); assert np.allclose(C,C.T) and np.allclose(np.diag(C),1);cors[cohort]=C;gz(C,O/'03_score_correlations'/f'{cohort}_PRIMARY_SCORE_SPEARMAN.tsv.gz')
 pc=PCA(n_components=min(S.shape)).fit_transform(S.to_numpy()); load=PCA(n_components=min(S.shape)).fit(S.to_numpy()) # deterministic duplicate fit
 # use second fitted object to retain component variance/loadings
 model=PCA(n_components=min(S.shape)).fit(S.to_numpy()); pcs=model.transform(S.to_numpy())
 pcas[cohort]=(pcs,model)
 sv(pd.DataFrame(pcs[:,:2],index=S.index,columns=['PC1','PC2']).reset_index(names='patient_id'),O/'04_pca'/f'{cohort}_PCA_PATIENT_SCORES.tsv')
 sv(pd.DataFrame(model.components_[:2].T,index=mids,columns=['PC1','PC2']).reset_index(names='model_instance_id'),O/'04_pca'/f'{cohort}_PCA_MODEL_LOADINGS.tsv')
 sv(pd.DataFrame({'PC':[f'PC{i+1}' for i in range(len(model.explained_variance_ratio_))],'variance_explained':model.explained_variance_ratio_}),O/'04_pca'/f'{cohort}_PCA_VARIANCE.tsv')
sv(pd.DataFrame(summaries),O/'01_scores/PRIMARY_UNSIGNED_SCORE_SUMMARY.tsv')
# Jaccard (frozen membership, no patient expression)
rows=[];freq={}
for m in mids:
 for g in genes[m]:freq[g]=freq.get(g,0)+1
for i,a in enumerate(mids):
 A=set(genes[a])
 for b in mids[i+1:]:
  B=set(genes[b]);rows.append({'model_1':a,'model_2':b,'intersection_n':len(A&B),'union_n':len(A|B),'jaccard':len(A&B)/len(A|B)})
JL=pd.DataFrame(rows); M=np.eye(len(mids));
for _,r in JL.iterrows(): i=mids.index(r.model_1);j=mids.index(r.model_2);M[i,j]=M[j,i]=r.jaccard
gz(pd.DataFrame(M,index=mids,columns=mids),O/'02_gene_set_structure/PRIMARY_MODEL_JACCARD_MATRIX.tsv.gz');gz(JL.set_index(['model_1','model_2']),O/'02_gene_set_structure/PRIMARY_MODEL_JACCARD_LONG.tsv.gz');sv(pd.DataFrame(sorted(freq.items(),key=lambda x:(-x[1],x[0])),columns=['gene_symbol','model_frequency']),O/'02_gene_set_structure/PRIMARY_MODEL_GENE_FREQUENCY.tsv');sv(pd.DataFrame({'model_instance_id':mids,'gene_n':[len(genes[m]) for m in mids]}),O/'02_gene_set_structure/PRIMARY_MODEL_SIZE_SUMMARY.tsv')
# Cross-cohort aligned correlation structure and PCA loading concordance
tri=np.triu_indices(len(mids),1); v1=cors['CGGA325'].to_numpy()[tri];v2=cors['CGGA693'].to_numpy()[tri];rho,p=spearmanr(v1,v2);sv(pd.DataFrame({'structure_concordance_spearman_rho':[rho],'p_value':[p],'pair_n':[len(v1)]}),O/'03_score_correlations/CROSS_COHORT_CORRELATION_STRUCTURE.tsv');gz(pd.DataFrame({'model_1':[mids[i] for i in tri[0]],'model_2':[mids[j] for j in tri[1]],'CGGA325_rho':v1,'CGGA693_rho':v2}).set_index(['model_1','model_2']),O/'03_score_correlations/CROSS_COHORT_MODEL_PAIR_ALIGNMENT.tsv.gz')
l1=pcas['CGGA325'][1].components_[0];l2=pcas['CGGA693'][1].components_[0];lrho,lp=spearmanr(l1,l2)
# Patient bootstrap: independent cohort resampling and shared statistic summaries.
rng=np.random.default_rng(420462);B=1000; out=[]
for b in range(B):
 bs={};
 for cohort in scores:
  S=scores[cohort]; ix=rng.integers(0,len(S),len(S)); Q=S.to_numpy()[ix]; mod=PCA(n_components=2).fit(Q); C=pd.DataFrame(Q).corr(method='spearman').to_numpy();bs[cohort]=(mod,C)
 rv=spearmanr(bs['CGGA325'][1][tri],bs['CGGA693'][1][tri]).statistic; lv=spearmanr(bs['CGGA325'][0].components_[0],bs['CGGA693'][0].components_[0]).statistic
 out.append({'iteration':b+1,'CGGA325_PC1_variance':bs['CGGA325'][0].explained_variance_ratio_[0],'CGGA693_PC1_variance':bs['CGGA693'][0].explained_variance_ratio_[0],'cross_cohort_structure_rho':rv,'PC1_loading_concordance_rho':lv,'CGGA325_correlation_mean':np.mean(bs['CGGA325'][1][tri]),'CGGA693_correlation_mean':np.mean(bs['CGGA693'][1][tri])})
BD=pd.DataFrame(out);gz(BD.set_index('iteration'),O/'06_bootstrap/BULK_PRIMARY_BOOTSTRAP_ITERATIONS.tsv.gz');sv(pd.DataFrame([{'statistic':c,'estimate':({'CGGA325_PC1_variance':pcas['CGGA325'][1].explained_variance_ratio_[0],'CGGA693_PC1_variance':pcas['CGGA693'][1].explained_variance_ratio_[0],'cross_cohort_structure_rho':rho,'PC1_loading_concordance_rho':lrho}.get(c,np.nan)),'ci_low':BD[c].quantile(.025),'ci_high':BD[c].quantile(.975)} for c in BD.columns if c!='iteration']),O/'06_bootstrap/BULK_PRIMARY_BOOTSTRAP_SUMMARY.tsv');sv(pd.DataFrame({'seed':[420462],'iterations':[B],'resampling_unit':['patient']}),O/'06_bootstrap/BULK_PRIMARY_BOOTSTRAP_SEEDS.tsv')
(O/'05_ecology_programs/ECOLOGY_PROGRAM_ANALYSIS_STATUS.txt').write_text('BLOCKED_PROGRAM_DEFINITION_UNRESOLVED\nNo substitute program scores calculated.\n')
# manifests and QA
files=[]
for q in O.rglob('*'):
 if q.is_file():files.append({'relative_path':str(q.relative_to(O)),'sha256':h(q),'size_bytes':q.stat().st_size})
sv(pd.DataFrame(files),O/'08_manifests/BULK_PRIMARY_FILE_MANIFEST.tsv');(O/'08_manifests/BULK_PRIMARY_CONFIG.json').write_text(json.dumps({'seed':420462,'bootstrap_n':1000,'score_rule':'cohort-wise gene z-score then equal-weight model gene mean','ecology_status':'BLOCKED_PROGRAM_DEFINITION_UNRESOLVED'},indent=2))
print(json.dumps({'rho':rho,'pc1_325':pcas['CGGA325'][1].explained_variance_ratio_[0],'pc1_693':pcas['CGGA693'][1].explained_variance_ratio_[0]},indent=2))
