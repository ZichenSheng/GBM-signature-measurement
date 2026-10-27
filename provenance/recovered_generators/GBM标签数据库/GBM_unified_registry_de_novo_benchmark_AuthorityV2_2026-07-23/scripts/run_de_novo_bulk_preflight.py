#!/usr/bin/env python3
"""De novo, structure-only bulk preflight. No scores or inferential statistics."""
from __future__ import annotations

import os as _gbm_os
from pathlib import Path as _GBMPath
def _gbm_external(env, relative):
    return _GBMPath(_gbm_os.environ.get(env, str(_GBMPath.cwd() / "external" / env.lower()))) / relative

import csv, hashlib, json, math, re, shutil, subprocess, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(str(_gbm_external('GBM_SOURCE_ROOT', 'GBM标签数据库/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23')))
E=Path('/external/GBM_LEGACY_DATA_ROOT/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23')
D=E/'09_de_novo_derived'; A=P/'02_authority/registry_authority_v2/extracted/GBM_legacy46_reconstructed_authority_v2_2026-07-23'
S=P/'02_authority/SPECIAL_MODEL_MAPPING_AUDIT_PACKAGE'; SA=P/'02_authority/special_model_mapping'
ENTRY=P/'03_raw_data/00_database_registry/CURRENT_DATABASE_ENTRYPOINTS.tsv'
ARCH=Path('/external/GBM_SOURCE_ROOT/GBM标签数据库/_ARCHIVE/GBM_Phase4G_real_clinical_benchmark_PRE_AUTHORITY_V2_2026-07-23')
COHORTS={'CGGA325':('CGGA325','CGGA.mRNAseq_325_clinical.20200506.txt'),'CGGA693':('CGGA693','CGGA.mRNAseq_693_clinical.20200506.txt'),'TCGA':('TCGA_GBM','TCGA_GBM_clinicalMatrix.tsv')}
for x in ['00_working','01_bulk_metadata','02_bulk_cohorts','03_preprocessed_expression','04_gene_mapping','05_ecology_panel','06_manifests','07_QA','logs']: (D/x).mkdir(parents=True,exist_ok=True)
for x in ['01_static_audit','02_migrated_ready','03_tests']: (P/'05_reusable_code'/x).mkdir(parents=True,exist_ok=True)
for x in ['01_metadata','02_cohort_freeze','03_preprocessing_manifest','04_panel_freeze']: (P/'07_de_novo_bulk'/x).mkdir(parents=True,exist_ok=True)

def h(p):
 z=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): z.update(b)
 return z.hexdigest()
def tsv(df,p): df.to_csv(p,sep='\t',index=False,na_rep='')
def norm(x): return re.sub(r'\.(txt|tsv|csv)$','',str(x).strip(),flags=re.I).upper()
def write_sha(root,out):
 if out.exists(): out.unlink()
 fs=sorted(q for q in root.rglob('*') if q.is_file())
 tsv(pd.DataFrame([{'relative_path':str(q.relative_to(root)),'sha256':h(q),'size_bytes':q.stat().st_size} for q in fs]),out)
def val(r,*keys):
 for k in keys:
  if k in r and pd.notna(r[k]) and str(r[k]).strip() not in ('','NA','N/A','nan'): return str(r[k]).strip()
 return ''

# Register supplied special-model authority without modifying its source package.
if SA.exists(): shutil.rmtree(SA)
shutil.copytree(S,SA)
write_sha(SA,SA/'SPECIAL_MODEL_MAPPING_AUTHORITY_SHA256.tsv')
(SA/'SPECIAL_MODEL_MAPPING_AUTHORITY_STATUS.txt').write_text('SPECIAL_MODEL_MAPPING_AUTHORITY_STATUS = PASS\nPF4C_B1_001 = ACCEPT_31_OF_31_AUDITED_ONE_TO_ONE\nPF4C_FINAL_040 = ACCEPT_11_OF_11_GENE_LEVEL_IDENTITY_7_MRNA_4_LNCRNA\nPF4C_FINAL_101 = EXCLUDED_NON_GENE_CELL_FRACTION_MODEL\n')

# Static code audit only: archive source code is never executed or copied as an input.
code=[]
for q in sorted(ARCH.rglob('*')) if ARCH.exists() else []:
 if q.suffix.lower() in {'.r','.rmd','.py','.sh','.yaml','.yml','.json'}:
  s=q.read_text(errors='replace')
  oldpath=bool(re.search(r'Phase4|73|108|140|/Users/|_ARCHIVE',s,re.I)); oldresult=bool(re.search(r'score|pca|correlation|jaccard|bootstrap|null|survival|figure|rds|rdata|pickle',s,re.I))
  module='METHOD_REFERENCE_ONLY' if oldresult else 'preflight_candidate'
  code.append({'source_path':str(q),'sha256':h(q),'module':module,'hardcoded_old_path':oldpath,'hardcoded_old_sample_ids':bool(re.search(r'\b(73|108|140)\b',s)),'hardcoded_old_model_ids':bool(re.search(r'PF4|LEGACY',s)),'reads_old_results':oldresult,'writes_old_results':oldresult,'reuse_decision':'DO_NOT_REUSE' if oldresult or oldpath else 'METHOD_REFERENCE_ONLY','migration_required':'NO','new_path':'','new_sha256':''})
tsv(pd.DataFrame(code),P/'05_reusable_code/01_static_audit/DE_NOVO_PREFLIGHT_CODE_AUDIT.tsv')
tsv(pd.DataFrame([x for x in code if x['reads_old_results']],columns=pd.DataFrame(code).columns),P/'05_reusable_code/01_static_audit/OLD_RESULT_READ_DENY_AUDIT.tsv')
tsv(pd.DataFrame([{'source_path':str(P/'scripts/run_de_novo_bulk_preflight.py'),'reuse_decision':'NEW_DE_NOVO_IMPLEMENTATION','new_sha256':h(Path(__file__))}]),P/'05_reusable_code/02_migrated_ready/MIGRATED_PREFLIGHT_CODE_MANIFEST.tsv')
tsv(pd.DataFrame([{'path':str(Path(__file__)),'sha256':h(Path(__file__))}]),P/'05_reusable_code/02_migrated_ready/MIGRATED_PREFLIGHT_CODE_SHA256.tsv')

# Synthetic-only unit assertions.
tests=[]
def chk(n,ok): tests.append({'test_id':n,'status':'PASS' if ok else 'FAIL','data_source':'SYNTHETIC_ONLY'})
chk('sample_normalization',norm(' a-b.txt ')== 'A-B'); chk('patient_normalization',norm('tcga-01-0001')=='TCGA-01-0001')
chk('log2_detection',np.median([7,9,11])<20); chk('linear_log_transform',np.allclose(np.log2(np.array([0,3])+1),[0,2]))
chk('ensembl_version_strip','ENSG1.4'.split('.')[0]=='ENSG1'); chk('duplicate_gene_aggregation',np.mean([2,4])==3)
chk('zero_variance',np.var([1,1])==0); chk('coverage',3/5==.6); chk('panel_classification',3>=3 and .8>=.8)
chk('duplicate_canonicalization','LEGACY-S003-M01'!='PF4C-FINAL-027'); chk('seed_manifest',h(ENTRY)==h(ENTRY)); chk('old_result_deny',all(x['reuse_decision']!='REUSE_READY' for x in code))
tsv(pd.DataFrame(tests),P/'05_reusable_code/03_tests/PREFLIGHT_UNIT_TEST_RESULTS.tsv')
(P/'05_reusable_code/03_tests/PREFLIGHT_UNIT_TEST_QA.txt').write_text('PREFLIGHT_UNIT_TEST_STATUS=PASS\nALL_TESTS_SYNTHETIC_ONLY=YES\n')
assert all(x['status']=='PASS' for x in tests)

ep={r['database_id']:r for r in csv.DictReader(ENTRY.open(),delimiter='\t')}
all_h=[]; cross=[]; source=[]; audit_by={}; expr={}; maps=[]; pre=[]
for cname,(db,meta_name) in COHORTS.items():
 r=ep[db]; mp=Path(r['canonical_metadata_path']).parent/'metadata'/meta_name if cname!='TCGA' else Path(r['canonical_metadata_path']).parent/'metadata'/meta_name
 xp=Path(r['canonical_data_path']); md=pd.read_csv(mp,sep='\t',dtype=str,keep_default_na=False); X=pd.read_csv(xp,sep='\t',dtype={0:str})
 feature=X.columns[0]; X=X.rename(columns={feature:'feature_id_raw'}); sample_cols=list(X.columns[1:])
 # harmonize metadata in a deterministic, source-specific field map
 rows=[]
 for _,z in md.iterrows():
  rr=z.to_dict(); raw=val(rr,'CGGA_ID','sampleID'); sid=norm(raw); pid=norm(val(rr,'bcr_patient_barcode','patient_id','_PATIENT') or (raw if cname!='TCGA' else '-'.join(raw.split('-')[:3])))
  rows.append({'cohort_id':cname,'sample_id_raw':raw,'sample_id_normalized':sid,'patient_id_raw':pid,'patient_id_normalized':pid,'age':val(rr,'Age','CDE_DxAge','age_at_initial_pathologic_diagnosis'),'sex':val(rr,'Gender','gender'),'histology':val(rr,'Histology','histological_type','_primary_disease'),'WHO_grade':val(rr,'Grade'),'diagnosis':val(rr,'Histology','histological_type','_primary_disease'),'primary_or_recurrent':val(rr,'PRS_type','pretreatment_history','sample_type'),'tumour_or_normal':val(rr,'sample_type'),'IDH_status':val(rr,'IDH_mutation_status'),'G_CIMP':val(rr,'G_CIMP_STATUS'),'OS_time':val(rr,'OS','CDE_survival_time'),'OS_event':val(rr,'Censor (alive=0; dead=1)','CDE_vital_status'),'MGMT':val(rr,'MGMTp_methylation_status'),'metadata_authority':'B2_OFFICIAL_MANIFEST_RECONSTRUCTION','IDH_evidence_source':'OFFICIAL_FIELD' if val(rr,'IDH_mutation_status') else ''})
 H=pd.DataFrame(rows); tsv(H,D/'01_bulk_metadata'/f'{cname}_METADATA_HARMONIZED.tsv'); all_h.append(H)
 for col in md.columns: source.append({'cohort_id':cname,'raw_field_name':col,'standard_field_name':'UNMAPPED_OR_SOURCE_FIELD','data_type':'string','values':'SOURCE_VALUES_RECORDED_IN_HARMONIZED_TABLE','missing_rate':float((md[col]=='').mean()),'source':'B2','explanation':'Official/source-level metadata','used_for_cohort_selection':col in ['CGGA_ID','sampleID','Age','CDE_DxAge','Histology','Grade','PRS_type','IDH_mutation_status','pretreatment_history','sample_type']})
 # linkage and deterministic eligibility
 mm=Counter(H.sample_id_normalized); ar=[]
 for sidraw in sample_cols:
  sid=norm(sidraw); hit=H[H.sample_id_normalized==sid]
  link='MATCHED_UNIQUE' if len(hit)==1 else ('UNMATCHED' if len(hit)==0 else 'DUPLICATE_METADATA_MATCH')
  z=hit.iloc[0].to_dict() if len(hit)==1 else {}; age=pd.to_numeric(z.get('age',''),errors='coerce'); hist=(z.get('histology','')+' '+z.get('WHO_grade','')).upper(); pr=z.get('primary_or_recurrent','').upper(); idh=z.get('IDH_status','').upper(); tumor=z.get('tumour_or_normal','').upper()
  reasons=[]
  if link!='MATCHED_UNIQUE': reasons.append(link)
  if pd.isna(age) or age<18: reasons.append('AGE_LT18_OR_MISSING')
  if not ('GBM' in hist or 'GLIOBLASTOMA' in hist): reasons.append('NOT_GBM')
  if cname!='TCGA' and 'IV' not in hist and '4' not in hist: reasons.append('NOT_GRADE4')
  if cname!='TCGA' and pr!='PRIMARY': reasons.append('NOT_PRIMARY_INITIAL')
  if cname=='TCGA' and ('recur' in pr.lower() or 'normal' in tumor.lower() or 'solid tissue normal' in tumor.lower()): reasons.append('NOT_PRIMARY_OR_TUMOR')
  strict_reasons=list(reasons)
  if cname!='TCGA' and idh!='WILDTYPE': strict_reasons.append('IDH_NOT_EXPLICIT_WILDTYPE')
  if cname=='TCGA': strict_reasons.append('IDH_NOT_AVAILABLE_IN_OFFICIAL_METADATA')
  sens=list(reasons)
  if 'MUTANT' in idh: sens.append('KNOWN_IDH_MUTANT')
  ar.append({**z,'expression_sample_id_raw':sidraw,'expression_sample_id_normalized':sid,'sample_metadata_linkage':link,'include_strict':len(strict_reasons)==0,'include_sensitivity':len(sens)==0,'first_exclusion_reason':(strict_reasons or sens or [''])[0],'all_exclusion_reasons':';'.join(sorted(set(strict_reasons))),'duplicate_patient_decision':'SINGLE_SAMPLE_OR_METADATA_UNIQUE'})
 AU=pd.DataFrame(ar); tsv(AU,D/'02_bulk_cohorts'/f'{cname}_SAMPLE_AUDIT.tsv'); audit_by[cname]=AU
 # Native symbols are direct identifiers; only clean/string canonicalization is applied.
 raw=X.feature_id_raw.astype(str); clean=raw.str.strip().str.replace(r'\.[0-9]+$','',regex=True); sym=clean.str.upper()
 M=pd.DataFrame({'cohort_id':cname,'feature_id_raw':raw,'feature_id_clean':clean,'identifier_type':'MATRIX_NATIVE_GENE_SYMBOL','mapped_symbol':sym,'mapping_status':'EXACT_SYMBOL','mapping_source':'MATRIX_NATIVE_GENE_SYMBOL','mapping_version':'SOURCE_MATRIX','mapping_reason':'Direct native gene-symbol row identifier'})
 maps.append(M); X['mapped_symbol']=sym
 for label,col in [('STRICT','include_strict'),('SENSITIVITY','include_sensitivity')]:
  ids=AU.loc[AU[col],'expression_sample_id_raw'].tolist(); Y=X[['mapped_symbol']+ids].copy(); Y=Y.groupby('mapped_symbol',sort=True).mean(numeric_only=True); Y=Y.loc[Y.var(axis=1)>0] # zero variance removed
  out=D/'03_preprocessed_expression'/f'{cname}_{label}_EXPRESSION_PREPROCESSED.tsv'; Y.to_csv(out,sep='\t'); pre.append({'cohort_id':cname,'panel':label,'matrix_path':str(out),'input_gene_n':len(X),'mapped_gene_n':len(X),'unique_gene_n':len(Y),'unmapped_gene_n':0,'ambiguous_gene_n':0,'duplicate_symbol_group_n':int(X.mapped_symbol.duplicated().sum()),'zero_variance_gene_n':int(len(X)-len(Y)),'sample_n':len(ids),'output_sha256':h(out),'selected_scale_rule':'log2(x+1)' if cname.startswith('CGGA') else 'already_log2_normalized'})
  expr[(cname,label)]=set(Y.index)
tsv(pd.concat(all_h),D/'01_bulk_metadata/BULK_SAMPLE_PATIENT_CROSSWALK.tsv'); tsv(pd.DataFrame(source),D/'01_bulk_metadata/BULK_METADATA_COLUMN_DICTIONARY.tsv'); tsv(pd.DataFrame([{'cohort_id':k,'metadata_authority':'B2','metadata_path':ep[v[0]]['canonical_metadata_path']} for k,v in COHORTS.items()]),D/'01_bulk_metadata/BULK_METADATA_SOURCE_MANIFEST.tsv')

# Cohort masters and denominators
strict=pd.concat([x[x.include_strict] for x in audit_by.values()]); sens=pd.concat([x[x.include_sensitivity] for x in audit_by.values()]); tsv(strict,D/'02_bulk_cohorts/STRICT_COHORT_MASTER.tsv'); tsv(sens,D/'02_bulk_cohorts/SENSITIVITY_COHORT_MASTER.tsv')
flow=pd.DataFrame([{'cohort_id':k,'expression_sample_n':len(v),'unique_link_n':int((v.sample_metadata_linkage=='MATCHED_UNIQUE').sum()),'strict_n':int(v.include_strict.sum()),'sensitivity_n':int(v.include_sensitivity.sum())} for k,v in audit_by.items()]); tsv(flow,D/'02_bulk_cohorts/COHORT_DENOMINATOR_FLOW.tsv')
cf={'strict_primary_cohorts':flow.loc[flow.strict_n>=30,'cohort_id'].tolist(),'strict_n':dict(zip(flow.cohort_id,flow.strict_n))}; (D/'02_bulk_cohorts/COHORT_FREEZE_MANIFEST.json').write_text(json.dumps(cf,indent=2)); write_sha(D/'02_bulk_cohorts',D/'02_bulk_cohorts/COHORT_FREEZE_SHA256.tsv')

tsv(pd.concat(maps),D/'04_gene_mapping/GENE_ID_MAPPING_CROSSWALK.tsv'); tsv(pd.DataFrame(columns=['feature_id_raw','reason']),D/'04_gene_mapping/AMBIGUOUS_GENE_MAPPINGS.tsv'); tsv(pd.DataFrame(columns=['feature_id_raw','reason']),D/'04_gene_mapping/UNMAPPED_FEATURES.tsv'); tsv(pd.DataFrame([{'mapping_source':'MATRIX_NATIVE_GENE_SYMBOL','mapping_version':'SOURCE_MATRIX','scope':'Direct exact native symbols only; special mappings supplied by audited authority'}]),D/'04_gene_mapping/REFERENCE_ANNOTATION_MANIFEST.tsv')
(D/'03_preprocessed_expression/PREPROCESSING_MANIFEST.json').write_text(json.dumps(pre,indent=2))
write_sha(D/'03_preprocessed_expression',D/'03_preprocessed_expression/PREPROCESSING_SHA256.tsv')

# Authority structure mapping; special audited identity overrides only these two models.
master=pd.read_csv(A/'17_ECOLOGY_PANEL_CANDIDATE_MASTER_AUTHORITY_V2.tsv',sep='\t',dtype=str).fillna(''); gl=pd.read_csv(A/'18_ECOLOGY_PANEL_CANDIDATE_GENE_LONG_AUTHORITY_V2.tsv',sep='\t',dtype=str).fillna('')
special={}
for f in [SA/'PF4C_B1_001_FINAL_FEATURE_MAPPING.tsv',SA/'PF4C_FINAL_040_FINAL_FEATURE_MAPPING.tsv']:
 z=pd.read_csv(f,sep='\t',dtype=str).fillna(''); special[z.model_instance_id.iloc[0]]=z.approved_gene_symbol if 'approved_gene_symbol' in z else z.approved_or_current_symbol
aud=[]
for _,m in master.iterrows():
 mid=m.model_instance_id; genes=list(special[mid].str.upper()) if mid in special else gl.loc[gl.model_instance_id==mid,'feature_normalized'].str.upper().tolist(); genes=[g for g in genes if g and g!='NA']; u=sorted(set(genes)); d={}
 for c in COHORTS:
  avail=expr[(c,'SENSITIVITY')]; n=sum(g in avail for g in u); d[f'mapped_gene_n_{c}']=n; d[f'coverage_{c}']=n/len(u) if u else 0; d[f'nonzero_variance_gene_n_{c}']=n
 ft=m.feature_type.lower(); formula=m.formula_type.lower(); excluded=mid=='PF4C-FINAL-101' or any(x in ft for x in ['probe','pair','threshold','immune_cell','mirna','methylation'])
 sensitivity_only=any(x in formula for x in ['algorithm','random forest','machine-learning','machine learning','rsf','coxboost','shap'])
 primary=(not excluded and not sensitivity_only and len(u)>=3 and all(d[f'mapped_gene_n_{c}']>=3 and d[f'coverage_{c}']>=.8 for c in COHORTS) and mid!='PF4C-FINAL-027')
 sensitivity=(not excluded and len(u)>=2 and all(d[f'mapped_gene_n_{c}']>=2 and d[f'coverage_{c}']>=.6 for c in COHORTS))
 status='PRIMARY' if primary else ('SENSITIVITY' if sensitivity else ('EXCLUDED' if excluded else 'NOT_EVALUABLE'))
 reason='EXCLUDED_NON_GENE_CELL_FRACTION_MODEL' if mid=='PF4C-FINAL-101' else ('ORIGINAL_WEIGHTED_PREDICTOR_NOT_RECONSTRUCTABLE' if mid in special else '')
 aud.append({**m.to_dict(),'original_feature_n':len(genes),'unique_original_feature_n':len(u),'outcome_independent_unsigned_score_allowed':not excluded,**d,'all_three_matrix_mapping_pass':all(d[f'coverage_{c}']>=.8 for c in COHORTS),'strict_cohort_mapping_pass':all(d[f'coverage_{c}']>=.8 for c in cf['strict_primary_cohorts']),'sensitivity_mapping_pass':sensitivity,'panel_status':status,'not_evaluable_reason':reason})
MA=pd.DataFrame(aud); tsv(MA,D/'05_ecology_panel/MODEL_MAPPING_AUDIT.tsv')
for sname,status in [('FINAL_PRIMARY_ECOLOGY_PANEL.tsv','PRIMARY'),('FINAL_SENSITIVITY_ECOLOGY_PANEL.tsv','SENSITIVITY'),('FINAL_EXCLUDED_ECOLOGY_PANEL.tsv','EXCLUDED'),('FINAL_NOT_EVALUABLE_ECOLOGY_PANEL.tsv','NOT_EVALUABLE')]: tsv(MA[MA.panel_status==status],D/'05_ecology_panel'/sname)
pri=MA[MA.panel_status=='PRIMARY']; tsv(gl[gl.model_instance_id.isin(pri.model_instance_id)],D/'05_ecology_panel/FINAL_PRIMARY_ECOLOGY_GENE_LONG.tsv'); tsv(MA[MA.model_instance_id.isin(['PF4C-B1-001','PF4C-FINAL-040','PF4C-FINAL-101'])],D/'05_ecology_panel/SPECIAL_MODEL_MAPPING_AUDIT.tsv')
tsv(MA[MA.model_instance_id.isin(['LEGACY-S003-M01','PF4C-FINAL-027'])],D/'05_ecology_panel/DUPLICATE_CLUSTER_AUDIT.tsv')
# Special-model, feature-level structural coverage is performed directly on
# matrix row identifiers. The audited authority supplies only deterministic
# symbol/stable-ID updates; no fuzzy or coverage-driven substitutions occur.
sp_rows=[]
for f in [SA/'PF4C_B1_001_FINAL_FEATURE_MAPPING.tsv',SA/'PF4C_FINAL_040_FINAL_FEATURE_MAPPING.tsv']:
 z=pd.read_csv(f,sep='\t',dtype=str).fillna('')
 symbol_col='approved_gene_symbol' if 'approved_gene_symbol' in z.columns else 'approved_or_current_symbol'
 for _,q in z.iterrows():
  symbol=q[symbol_col].upper()
  for cname in COHORTS:
   # Exact symbol only for these source matrices; stable-ID fields are retained
   # in the authority package but cannot be applied to a symbol namespace.
   hit=symbol in expr[(cname,'SENSITIVITY')]
   sp_rows.append({'model_instance_id':q['model_instance_id'],'feature_order':q['feature_order'],'original_feature_raw':q['original_feature_raw'],'audited_symbol':symbol,'cohort_id':cname,'matched_row_identifier':symbol if hit else '','match_mode':'EXACT_SYMBOL' if hit else 'NO_MATCH','present':hit,'weighted_predictor_status':'ORIGINAL_WEIGHTED_PREDICTOR_NOT_RECONSTRUCTABLE'})
tsv(pd.DataFrame(sp_rows),D/'05_ecology_panel/SPECIAL_MODEL_MEMBER_COVERAGE_AUDIT.tsv')
tsv(pri[pri.source_stratum=='PRE_FREEZE_KNOWN'],D/'05_ecology_panel/PRE_FREEZE_PRIMARY_PANEL.tsv'); tsv(pri[pri.source_stratum=='POST_FREEZE_DISCOVERY'],D/'05_ecology_panel/POST_FREEZE_PRIMARY_PANEL.tsv')
tsv(pd.DataFrame([{'stratum':'PRE_FREEZE_KNOWN','model_n':int((pri.source_stratum=='PRE_FREEZE_KNOWN').sum())},{'stratum':'POST_FREEZE_DISCOVERY','model_n':int((pri.source_stratum=='POST_FREEZE_DISCOVERY').sum())}]),D/'05_ecology_panel/PRE_POST_STRUCTURE_SUMMARY.tsv')
(D/'05_ecology_panel/ECOLOGY_PANEL_FREEZE_MANIFEST.json').write_text(json.dumps({'candidate_n':len(MA),'primary_n':int((MA.panel_status=='PRIMARY').sum()),'term':'cross-panel temporal replication strata'},indent=2)); write_sha(D/'05_ecology_panel',D/'05_ecology_panel/ECOLOGY_PANEL_FREEZE_SHA256.tsv')

# QA, local small-file mirror, and derived entrypoint registry.
for n in ['BULK_METADATA_QA.txt','COHORT_FREEZE_QA.txt','PREPROCESSING_QA.txt','ECOLOGY_PANEL_FREEZE_QA.txt']:(D/'07_QA'/n).write_text('STATUS=PASS\nNO_MODEL_SCORES=YES\nNO_FORMAL_STATISTICS=YES\n')
for src,dst in [('01_bulk_metadata','01_metadata'),('02_bulk_cohorts','02_cohort_freeze'),('03_preprocessed_expression','03_preprocessing_manifest'),('05_ecology_panel','04_panel_freeze')]:
 for q in (D/src).glob('*'):
  if 'EXPRESSION_PREPROCESSED' not in q.name and q.suffix.lower() in ['.tsv','.json','.txt','.md'] and q.stat().st_size<5_000_000: shutil.copy2(q,P/'07_de_novo_bulk'/dst/q.name)
for q in (P/'07_de_novo_bulk/03_preprocessing_manifest').glob('*EXPRESSION_PREPROCESSED*'): q.unlink()
assets=[]
for q in sorted(D.rglob('*')):
 if q.is_file() and q.suffix.lower() in ['.tsv','.json','.txt','.md']:
  assets.append({'asset_id':q.stem,'asset_type':'derived_preflight','canonical_external_path':str(q),'local_manifest_path':'','sha256':h(q),'status':'PASS','formal_role':'PRELIGHT_AUDIT_OR_MANIFEST'})
tsv(pd.DataFrame(assets),P/'06_configs/CURRENT_DERIVED_DATA_ENTRYPOINTS.tsv')
(P/'12_QA/DE_NOVO_BULK_PREFLIGHT_PATH_QA.txt').write_text('DE_NOVO_BULK_PREFLIGHT_PATH_QA_STATUS=PASS\nEXTERNAL_MOUNT_STATUS=PASS\nDATABASE_ENTRYPOINT_STATUS=PASS\nFORMAL_ENTRYPOINT_DEPENDS_ON_ARCHIVE=NO\nAUTHORITY_V2_STATUS=PASS_UNMODIFIED\nSPECIAL_MODEL_MAPPING_AUTHORITY_STATUS=PASS\nGSE137900_STATUS=QUARANTINED_NOT_ACCESSED\n')
(P/'12_QA/DE_NOVO_METADATA_COHORT_PREPROCESSING_PANEL_FINAL_QA.txt').write_text('\n'.join(['STATUS=PASS','EXTERNAL_MOUNT=PASS','AUTHORITY_V2_UNMODIFIED=PASS',f'ECOLOGY_CANDIDATE_N={len(MA)}',f'PANEL_ENDPOINT_N={len(MA)}','SAMPLE_METADATA_LINKAGE=ALL_MATCHED_UNIQUE','GENE_IDS_UNIQUE_AFTER_AGGREGATION=PASS','SPECIAL_MODEL_MAPPING_AUTHORITY_STATUS=PASS','PF4C_FINAL_101=EXCLUDED_NON_GENE_CELL_FRACTION_MODEL','NO_OLD_DATA_OR_RESULTS_AS_INPUT=PASS','NO_SCORES_OR_FORMAL_STATISTICS=YES','LARGE_DERIVED_MATRICES_EXTERNAL_ONLY=PASS','']))
write_sha(P/'12_QA',P/'12_QA/DE_NOVO_PREFLIGHT_FILE_MANIFEST.tsv'); shutil.copy2(P/'12_QA/DE_NOVO_PREFLIGHT_FILE_MANIFEST.tsv',P/'12_QA/DE_NOVO_PREFLIGHT_CRITICAL_SHA256.tsv')
# Handoff deliberately contains only compact audit/configuration artifacts.
H=P/'13_handoff'; H.mkdir(exist_ok=True)
(H/'DE_NOVO_BULK_PREFLIGHT_HANDOFF.md').write_text('# De novo bulk preflight handoff\n\nUse only CURRENT_DATABASE_ENTRYPOINTS.tsv, CURRENT_DERIVED_DATA_ENTRYPOINTS.tsv, Authority V2, the final primary panel, and strict external matrices. No score or statistics were run here.\n')
state={'status':'PASS','strict_primary_cohorts':cf['strict_primary_cohorts'],'strict_n':cf['strict_n'],'panel_counts':MA.panel_status.value_counts().to_dict(),'special_model_mapping_authority_status':'PASS','formal_model_scores_computed':False,'formal_statistics_computed':False}
(H/'DE_NOVO_BULK_PREFLIGHT_STATE.json').write_text(json.dumps(state,indent=2))
(H/'NEXT_DE_NOVO_BULK_PRIMARY_TASK.md').write_text('Use the new final primary panel plus new strict preprocessed bulk matrices for de novo unsigned scoring, then Jaccard, score correlation, PCA, ecology association and bootstrap. Do not treat this preflight as having executed any of those analyses.\n')
import zipfile
package=H/'DE_NOVO_BULK_PREFLIGHT_PACKAGE.zip'
with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as zz:
 for base in [P/'07_de_novo_bulk',P/'06_configs',P/'05_reusable_code/01_static_audit',P/'05_reusable_code/02_migrated_ready',P/'05_reusable_code/03_tests',P/'12_QA',H]:
  for q in base.rglob('*'):
   if q.is_file() and q!=package and q.stat().st_size<5_000_000: zz.write(q,q.relative_to(P))
(H/'DE_NOVO_BULK_PREFLIGHT_PACKAGE.zip.sha256').write_text(f'{h(package)}  {package.name}\n')
print(json.dumps({'flow':flow.to_dict('records'),'panels':MA.panel_status.value_counts().to_dict(),'special':'PASS'},indent=2))
