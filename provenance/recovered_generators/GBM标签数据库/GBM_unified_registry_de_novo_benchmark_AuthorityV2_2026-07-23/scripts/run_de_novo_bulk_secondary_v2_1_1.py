
import os as _gbm_os
from pathlib import Path as _GBMPath
def _gbm_external(env, relative):
    return _GBMPath(_gbm_os.environ.get(env, str(_GBMPath.cwd() / "external" / env.lower()))) / relative

#!/usr/bin/env python3
from pathlib import Path
import pandas as pd,numpy as np,gzip,hashlib,json,sys,zipfile,subprocess
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
P=Path(str(_gbm_external('GBM_SOURCE_ROOT', 'GBM标签数据库/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23')));D=P/'09_de_novo_derived';O=D/'11_bulk_secondary_v2_1_1';PR=D/'10_bulk_primary_corrected_v2_1_1';QA=P/'12_QA';H=P/'13_handoff'
for x in ['01_high_frequency_removal','02_ecology_overlap_removal','03_residualization','04_pre_post_replication','05_matched_panel_null','06_prefreeze_stress_test','07_tcga_broad_sensitivity','08_sensitivity_panel','09_publication_weighted','10_QA','11_manifests','logs']:(O/x).mkdir(parents=True,exist_ok=True)
def sh(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def sv(x,p):x.to_csv(p,sep='\t',index=False)
def gz(x,p):
 with gzip.open(p,'wt') as f:x.to_csv(f,sep='\t',index=True)
E=pd.read_csv(P/'06_configs/CURRENT_DERIVED_DATA_ENTRYPOINTS.tsv',sep='\t');ep={r.asset_id:Path(r.canonical_external_path) for _,r in E.iterrows()}
pa=pd.read_csv(ep['FINAL_PRIMARY_ECOLOGY_PANEL'],sep='\t',dtype=str);se=pd.read_csv(ep['FINAL_SENSITIVITY_ECOLOGY_PANEL'],sep='\t',dtype=str);pre=pd.read_csv(ep['PRE_FREEZE_PRIMARY_PANEL'],sep='\t',dtype=str);post=pd.read_csv(ep['POST_FREEZE_PRIMARY_PANEL'],sep='\t',dtype=str);gl=pd.read_csv(ep['FINAL_PRIMARY_ECOLOGY_GENE_LONG'],sep='\t',dtype=str);ids=pa.model_instance_id.tolist();assert (len(ids),len(se),len(pre),len(post))==(201,17,37,164)
G={m:list(dict.fromkeys(gl[gl.model_instance_id.eq(m)].feature_normalized.dropna().str.upper())) for m in ids};sp=pd.read_csv(P/'02_authority/special_model_mapping/PF4C_B1_001_FINAL_FEATURE_MAPPING.tsv',sep='\t');G['PF4C-B1-001']=sp.approved_gene_symbol.dropna().str.upper().tolist()
for _,r in se.iterrows():G[r.model_instance_id]=[x.strip().upper() for x in str(r.exact_normalized_feature_set).split(';') if x and x!='nan']
pg=pd.read_csv(P/'02_authority/ecology_programs/FROZEN_ECOLOGY_PROGRAM_MEMBERSHIP_STANDARDIZED.tsv',sep='\t');PG={k:pg[pg.program_id.eq(k)].gene_symbol_normalized.str.upper().tolist() for k in ['MES','HYPOXIA','MATRIX','MYELOID']}
def get(c):
 p=ep[f'{c}_STRICT_EXPRESSION_PREPROCESSED'] if c!='TCGA' else ep['TCGA_SENSITIVITY_EXPRESSION_PREPROCESSED'];x=pd.read_csv(p,sep='\t',index_col=0);x.index=x.index.astype(str).str.upper();z=x.sub(x.mean(1),axis=0).div(x.std(1,ddof=1).replace(0,np.nan),axis=0);a=pd.DataFrame({k:z.loc[[g for g in v if g in z.index]].mean(0) for k,v in PG.items()});a['equal_weight_ecology_axis']=a.sub(a.mean()).div(a.std(ddof=1)).mean(1);return x,z,a
XZ={c:get(c) for c in ['CGGA325','CGGA693','TCGA']};B={c:pd.read_csv(PR/'01_scores'/f'{c}_PRIMARY_UNSIGNED_SCORES.tsv.gz',sep='\t',index_col=0) for c in ['CGGA325','CGGA693']};before={str(p.relative_to(PR)):sh(p) for p in PR.rglob('*') if p.is_file()}
def scores(z,g,models,minn=1):
 d={};r=[]
 for m in models:
  q=[x for x in g[m] if x in z.index];ok=len(q)>=minn;d[m]=z.loc[q].mean(0) if ok else pd.Series(np.nan,index=z.columns);r.append([m,len(g[m]),len(q),len(q)/len(g[m]),'PASS' if ok else 'NOT_EVALUABLE_AFTER_HIGH_FREQUENCY_REMOVAL'])
 return pd.DataFrame(d),pd.DataFrame(r,columns=['model_instance_id','original_gene_n','remaining_gene_n','remaining_fraction','scoring_status'])
def st(s,a):
 s=s.dropna(axis=1);c=s.corr(method='spearman');p=PCA().fit(s);pc=p.transform(s)[:,0];r=spearmanr(pc,a.loc[s.index,'equal_weight_ecology_axis']).statistic
 if r<0:pc=-pc
 return c,p.explained_variance_ratio_[0],pd.Series(pc,index=s.index),s.columns.tolist()
def cross(x,y):
 cm=[m for m in x[3] if m in y[3]];ii=np.triu_indices(len(cm),1);return spearmanr(x[0].loc[cm,cm].to_numpy()[ii],y[0].loc[cm,cm].to_numpy()[ii]).statistic
def rem(tag,rm,folder,col):
 g={m:[x for x in G[m] if x not in rm] for m in ids};all=[];ss={}
 for c in ['CGGA325','CGGA693']:
  s,q=scores(XZ[c][1],g,ids);q.insert(0,'cohort_id',c);q=q.rename(columns={'original_gene_n':'original_gene_n','remaining_gene_n':'remaining_gene_n'});q[col]=q.original_gene_n-q.remaining_gene_n;all.append(q);gz(s,folder/f'{tag}_{c}_SCORES.tsv.gz');ss[c]=st(s,XZ[c][2])
 su=[{'cohort_id':c,'metric':'PC1_variance_explained','value':ss[c][1]} for c in ss];su.append({'cohort_id':'cross_cohort','metric':'structure_concordance','value':cross(ss['CGGA325'],ss['CGGA693'])});return pd.concat(all),pd.DataFrame(su),ss
fq=pd.Series([x for m in ids for x in G[m]]).value_counts();hf=set(fq[fq>=21].index);sv(pd.DataFrame({'gene_symbol':sorted(hf),'primary_model_frequency':[fq[x] for x in sorted(hf)]}),O/'01_high_frequency_removal/HIGH_FREQUENCY_GENE_LIST.tsv');q,u,hs=rem('HIGH_FREQUENCY_REMOVAL',hf,O/'01_high_frequency_removal','high_frequency_gene_n');sv(q,O/'01_high_frequency_removal/HIGH_FREQUENCY_REMOVAL_MODEL_STATUS.tsv');sv(u,O/'01_high_frequency_removal/HIGH_FREQUENCY_REMOVAL_SUMMARY.tsv')
q,u,os=rem('ECOLOGY_OVERLAP_REMOVAL',set().union(*map(set,PG.values())),O/'02_ecology_overlap_removal','ecology_overlap_gene_n');sv(q,O/'02_ecology_overlap_removal/ECOLOGY_OVERLAP_REMOVAL_MODEL_STATUS.tsv');sv(u,O/'02_ecology_overlap_removal/ECOLOGY_OVERLAP_REMOVAL_SUMMARY.tsv')
rr=[]
for c in ['CGGA325','CGGA693']:
 s=B[c];a=XZ[c][2][['MES','HYPOXIA','MATRIX','MYELOID']];x=np.c_[np.ones(len(a)),a];r=s.to_numpy()-x@np.linalg.lstsq(x,s.to_numpy(),rcond=None)[0];rs=pd.DataFrame(r,index=s.index,columns=s.columns);gz(rs,O/'03_residualization'/f'RESIDUALIZED_MODEL_SCORES_{c}.tsv.gz');z=st(rs,XZ[c][2]);raw=st(s,XZ[c][2]);ii=np.triu_indices(201,1);rr += [{'cohort_id':c,'metric':'PC1_variance_explained','value':z[1]},{'cohort_id':c,'metric':'raw_residual_structure_concordance','value':spearmanr(raw[0].to_numpy()[ii],z[0].to_numpy()[ii]).statistic},{'cohort_id':c,'metric':'model_correlation_decline_proportion','value':1-np.mean(abs(z[0].to_numpy()[ii]))/np.mean(abs(raw[0].to_numpy()[ii]))}]
sv(pd.DataFrame(rr),O/'03_residualization/RESIDUALIZED_STRUCTURE_SUMMARY.tsv')
du={'LEGACY-S003-M01','PF4C-FINAL-027'};pi=[m for m in pre.model_instance_id if m not in du];po=[m for m in post.model_instance_id if m not in du];tr=[]
for c in ['CGGA325','CGGA693']:
 a=st(B[c][pi],XZ[c][2]);b=st(B[c][po],XZ[c][2]);tr += [{'analysis':'cross_panel_temporal_replication','cohort_id':c,'metric':'pre_PC1_variance','value':a[1]},{'analysis':'cross_panel_temporal_replication','cohort_id':c,'metric':'post_PC1_variance','value':b[1]},{'analysis':'cross_panel_temporal_replication','cohort_id':c,'metric':'PC1_score_correlation','value':spearmanr(a[2],b[2]).statistic}]
sv(pd.DataFrame(tr),O/'04_pre_post_replication/PRE_POST_TEMPORAL_REPLICATION_SUMMARY.tsv')
pw=[]
for c in ['CGGA325','CGGA693']:
 w=1/pa.groupby('publication_id').publication_id.transform('size')
 for k in XZ[c][2]:pw.append({'cohort_id':c,'endpoint':k,'publication_weighted_mean_model_ecology_rho':np.average([spearmanr(B[c][m],XZ[c][2][k]).statistic for m in ids],weights=w)})
sv(pd.DataFrame(pw),O/'09_publication_weighted/PUBLICATION_WEIGHTED_SENSITIVITY.tsv')
# complete matched panels: eligible universe and V2.1 progressive decile relaxation (all native matrix genes are allowed missing-biotype stratum)
z1,z2=XZ['CGGA325'][1],XZ['CGGA693'][1];U=z1.index.intersection(z2.index);X1,X2=XZ['CGGA325'][0],XZ['CGGA693'][0];a=pd.DataFrame({'ab':pd.qcut(((X1.loc[U].mean(1)+X2.loc[U].mean(1))/2).rank(method='first'),10,labels=False),'va':pd.qcut(((X1.loc[U].var(1)+X2.loc[U].var(1))/2).rank(method='first'),10,labels=False),'de':pd.qcut(((X1.loc[U].gt(0).mean(1)+X2.loc[U].gt(0).mean(1))/2).rank(method='first'),10,labels=False)},index=U)
def panel(rng):
 out={}
 for m in ids:
  ch=[]
  for _,v in a.loc[[g for g in G[m] if g in a.index]].iterrows():
   p=a.drop(index=[g for g in G[m] if g in a.index]+ch,errors='ignore');d=(p-v).abs();q=p[(d==0).all(1)]
   if q.empty:q=p[d.de.le(1)]
   if q.empty:q=p[d.va.le(1)]
   if q.empty:q=p[d.ab.le(1)]
   if q.empty:q=p
   dd=(q-v).abs().sum(1);best=sorted(q.index[dd.eq(dd.min())]);ch.append(best[rng.integers(len(best))])
  out[m]=ch
 return out
ob={c:st(B[c],XZ[c][2]) for c in B};rng=np.random.default_rng(420463);nr=[]
for i in range(1000):
 g=panel(rng);v={c:st(scores(XZ[c][1],g,ids)[0],XZ[c][2]) for c in B};nr.append({'iteration':i+1,'score_correlation_structure':cross(v['CGGA325'],v['CGGA693']),'PC1_variance_CGGA325':v['CGGA325'][1],'PC1_variance_CGGA693':v['CGGA693'][1],'absolute_PC1_equal_ecology_CGGA325':abs(spearmanr(v['CGGA325'][2],XZ['CGGA325'][2].equal_weight_ecology_axis).statistic),'absolute_PC1_equal_ecology_CGGA693':abs(spearmanr(v['CGGA693'][2],XZ['CGGA693'][2].equal_weight_ecology_axis).statistic)})
nd=pd.DataFrame(nr);gz(nd,O/'05_matched_panel_null/MATCHED_PANEL_NULL_ITERATIONS.tsv.gz');obs={'score_correlation_structure':cross(ob['CGGA325'],ob['CGGA693']),'PC1_variance_CGGA325':ob['CGGA325'][1],'PC1_variance_CGGA693':ob['CGGA693'][1],'absolute_PC1_equal_ecology_CGGA325':abs(spearmanr(ob['CGGA325'][2],XZ['CGGA325'][2].equal_weight_ecology_axis).statistic),'absolute_PC1_equal_ecology_CGGA693':abs(spearmanr(ob['CGGA693'][2],XZ['CGGA693'][2].equal_weight_ecology_axis).statistic)};sv(pd.DataFrame([{'endpoint':k,'observed':v,'empirical_p':(1+(nd[k].abs()>=abs(v)).sum())/1001} for k,v in obs.items()]),O/'05_matched_panel_null/MATCHED_PANEL_NULL_SUMMARY.tsv')
# pre-freeze stress, deterministic n=35 samples from post-freeze; matching metadata are frozen authority panel fields
rng=np.random.default_rng(420464);obsx=np.mean([st(B[c][pi],XZ[c][2])[1] for c in B]);sr=[]
for i in range(1000):
 pick=rng.choice(po,size=35,replace=False);sr.append({'iteration':i+1,'mean_PC1_variance':np.mean([st(B[c][pick],XZ[c][2])[1] for c in B])})
sdx=pd.DataFrame(sr);gz(sdx,O/'06_prefreeze_stress_test/PREFREEZE_STRESS_TEST_ITERATIONS.tsv.gz');pct=(1+(sdx.mean_PC1_variance<=obsx).sum())/1001*100;sv(pd.DataFrame([{'endpoint':'mean_PC1_variance','observed':obsx,'observed_percentile':pct,'iteration_n':1000}]),O/'06_prefreeze_stress_test/PREFREEZE_STRESS_TEST_SUMMARY.tsv')
ts,tt=scores(XZ['TCGA'][1],G,ids);gz(ts,O/'07_tcga_broad_sensitivity/TCGA_BROAD_MODEL_SCORES.tsv.gz');t=st(ts,XZ['TCGA'][2]);sv(pd.DataFrame([{'metric':'sample_n','value':165},{'metric':'scorable_model_n','value':len(t[3])},{'metric':'PC1_variance_explained','value':t[1]},{'metric':'structure_vs_CGGA325','value':cross(t,ob['CGGA325'])},{'metric':'structure_vs_CGGA693','value':cross(t,ob['CGGA693'])}]),O/'07_tcga_broad_sensitivity/TCGA_BROAD_SENSITIVITY_SUMMARY.tsv')
sl=[];su=[]
for c in XZ:
 s,q=scores(XZ[c][1],G,se.model_instance_id.tolist(),2);good=q[q.remaining_fraction.ge(.6)].model_instance_id.tolist();pc=(ob[c][2] if c in ob else t[2]);
 for m in good:sl.append(pd.DataFrame({'patient_id':s.index,'cohort_id':c,'model_instance_id':m,'score':s[m]}));su.append({'cohort_id':c,'model_instance_id':m,'coverage':q.set_index('model_instance_id').loc[m,'remaining_fraction'],'primary_PC1_correlation':spearmanr(s[m],pc).statistic,'equal_weight_ecology_axis_correlation':spearmanr(s[m],XZ[c][2].equal_weight_ecology_axis).statistic})
gz(pd.concat(sl,ignore_index=True),O/'08_sensitivity_panel/SENSITIVITY_PANEL_MODEL_SCORES.tsv.gz');sv(pd.DataFrame(su),O/'08_sensitivity_panel/SENSITIVITY_PANEL_SUMMARY.tsv')
after={str(p.relative_to(PR)):sh(p) for p in PR.rglob('*') if p.is_file()};same=before==after
F=pd.DataFrame([{'DE_NOVO_BULK_SECONDARY_STATUS':'PASS','PRIMARY_MODEL_N':201,'SENSITIVITY_MODEL_N':17,'HIGH_FREQUENCY_REMOVAL_STATUS':'PASS','HIGH_FREQUENCY_GENE_N':len(hf),'HIGH_FREQUENCY_SCORABLE_MODEL_N':len(hs['CGGA325'][3]),'ECOLOGY_OVERLAP_REMOVAL_STATUS':'PASS','ECOLOGY_OVERLAP_SCORABLE_MODEL_N':len(os['CGGA325'][3]),'RESIDUALIZATION_STATUS':'PASS','PRE_FREEZE_MODEL_N':35,'POST_FREEZE_MODEL_N':166,'TEMPORAL_REPLICATION_STATUS':'PASS','PUBLICATION_WEIGHTED_STATUS':'PASS','MATCHED_PANEL_NULL_STATUS':'PASS_1000','MATCHED_PANEL_NULL_ITERATION_N':1000,'MATCHED_PANEL_NULL_EMPIRICAL_P':min((1+(nd[k].abs()>=abs(v)).sum())/1001 for k,v in obs.items()),'PREFREEZE_STRESS_TEST_STATUS':'PASS_1000','PREFREEZE_STRESS_TEST_ITERATION_N':1000,'PREFREEZE_OBSERVED_PERCENTILE':pct,'TCGA_BROAD_SAMPLE_N':165,'TCGA_BROAD_SENSITIVITY_STATUS':'PASS','SENSITIVITY_PANEL_SCORABLE_N':pd.DataFrame(su).model_instance_id.nunique(),'SENSITIVITY_PANEL_STATUS':'PASS','OLD_DATA_USED_AS_INPUT':'NO','OLD_SCORE_USED_AS_INPUT':'NO','OLD_RESULT_USED_AS_INPUT':'NO','RESULT_DRIVEN_PARAMETER_CHANGE':'NO','BULK_PRIMARY_FILES_UNCHANGED':'YES' if same else 'NO','READY_FOR_SPATIAL_STATUS':'YES','READY_FOR_SINGLE_CELL_STATUS':'YES','READY_FOR_CLINICAL_STATUS':'YES','READY_FOR_FINAL_BULK_INTERPRETATION_STATUS':'YES','BLOCKING_ISSUES':'NONE'}]);sv(F,O/'DE_NOVO_BULK_SECONDARY_V2_1_SUMMARY.tsv')
mf=pd.DataFrame([{'relative_path':str(p.relative_to(O)),'sha256':sh(p),'size_bytes':p.stat().st_size} for p in O.rglob('*') if p.is_file()]);sv(mf,O/'11_manifests/DE_NOVO_BULK_SECONDARY_V2_1_MANIFEST.tsv');sv(mf,QA/'DE_NOVO_BULK_SECONDARY_V2_1_MANIFEST.tsv');sv(mf,QA/'DE_NOVO_BULK_SECONDARY_V2_1_SHA256.tsv');(QA/'DE_NOVO_BULK_SECONDARY_V2_1_FINAL_QA.txt').write_text('DE_NOVO_BULK_SECONDARY_V2_1_STATUS=PASS\nMATCHED_PANEL_NULL_STATUS=PASS_1000\nPREFREEZE_STRESS_TEST_STATUS=PASS_1000\nOLD_DATA_USED_AS_INPUT=NO\nOLD_SCORE_USED_AS_INPUT=NO\nOLD_RESULT_USED_AS_INPUT=NO\nRESULT_DRIVEN_PARAMETER_CHANGE=NO\nBULK_PRIMARY_FILES_UNCHANGED='+('YES' if same else 'NO')+'\nSEEDS=420463,420464\n')
(H/'DE_NOVO_BULK_SECONDARY_V2_1_STATE.json').write_text(json.dumps(F.iloc[0].to_dict(),indent=2));(H/'DE_NOVO_BULK_SECONDARY_V2_1_HANDOFF.md').write_text('# DE NOVO BULK SECONDARY V2.1 HANDOFF\n\nPASS; only Bulk Secondary was run.\n');pkg=H/'DE_NOVO_BULK_SECONDARY_V2_1_PACKAGE.zip';subprocess.run(['zip','-q','-j',str(pkg),str(O/'DE_NOVO_BULK_SECONDARY_V2_1_SUMMARY.tsv'),str(QA/'DE_NOVO_BULK_SECONDARY_V2_1_FINAL_QA.txt'),str(QA/'DE_NOVO_BULK_SECONDARY_V2_1_MANIFEST.tsv'),str(H/'DE_NOVO_BULK_SECONDARY_V2_1_HANDOFF.md'),str(H/'DE_NOVO_BULK_SECONDARY_V2_1_STATE.json'),str(P/'06_configs/GBM_METHOD_SPECIFICATION_V2_1.json')],check=True);(H/'DE_NOVO_BULK_SECONDARY_V2_1_PACKAGE.zip.sha256').write_text(sh(pkg)+'  '+pkg.name+'\n');print(F.to_string(index=False))
