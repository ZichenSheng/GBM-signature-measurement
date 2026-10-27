#!/usr/bin/env python3
"""Deterministic ACT3 figure production. Real frozen observations only.
Run with the project's analysis_env/bin/python. Final export requires resolved source authority for every critical endpoint.
Panel D denominators are authorized; Panel E remains explicitly gated.
"""
from pathlib import Path
import sys, importlib.util, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.transforms import Bbox
from matplotlib.ticker import NullFormatter, FixedLocator, FuncFormatter
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
OUT=Path(__file__).resolve().parents[1]; ROOT=OUT.parent
V3=ROOT/'GBM_SIGNATURE_CONTEXT_DEFORMATION_DEEPENING_V3'
V1=ROOT/'GBM_SIGNATURE_CONTEXT_STRESS_TEST_EXTENSION_V1_FINAL'
V2=ROOT/'GBM_SIGNATURE_CONTEXT_EXTENSION_DEEPENING_V2_FINAL'
R2=ROOT/'GBM_SIGNATURE_FINAL_MEASUREMENT_ANATOMY_SYNTHESIS_V1'
DATA=OUT/'03_PLOTTING_DATA'
for d in [DATA,OUT/'04_PANEL_OUTPUTS',OUT/'05_FINAL',OUT/'00_PROVENANCE']:d.mkdir(exist_ok=True)
NAVY='#003366'; BLUE='#0072B2'; RED='#E64B35'; ORANGE='#E69F00'; GREEN='#009E73'; PURPLE='#CC79A7'; GREY='#8A97A3'; CYAN='#CFE8F3'
COL=dict(CARE=BLUE,GLASS=ORANGE,GSE174554=GREEN); COHORTS=list(COL)
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'axes.titlesize':7,'text.color':NAVY,'axes.labelcolor':NAVY,'xtick.color':NAVY,'ytick.color':NAVY,'axes.edgecolor':NAVY,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','legend.fontsize':6.3,'xtick.major.size':2,'ytick.major.size':2})
SOURCES=[];AUDIT=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def src(p,panel,endpoint,columns='all',filter='none',derivation='frozen observations'):
 p=Path(p);SOURCES.append(dict(panel=panel,scientific_endpoint=endpoint,source_file=str(p),source_columns=columns,filter=filter,derivation=derivation,authority_status='FROZEN_SOURCE',sha256=sha(p)));return p

def read(rel,panel,endpoint,filter='none',derivation='frozen observations'):
 p=src(V3/rel,panel,endpoint,filter=filter,derivation=derivation);return pd.read_csv(p,sep='\t')
def save(d,n):d.to_csv(DATA/(n+'.tsv'),sep='\t',index=False)
def audit(p,m,v,expected=None,ok=True,note=''):
 AUDIT.append(dict(panel=p,metric=m,value=v,expected=expected,status='PASS' if ok else 'BLOCKED',note=note))
def check(p,m,v,expected):
 ok=bool(np.isclose(v,expected,rtol=1e-7,atol=1e-9));audit(p,m,v,expected,ok);assert ok,(p,m,v,expected)
def module(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
# Targeted source discrepancy tracing; no new bootstrap inference.
from trace_panel_e_contrast import run as trace_contrast
contrast_receipt=trace_contrast(ROOT,OUT)
for row in pd.read_csv(DATA/'Fig3E_contrast_source_trace.tsv',sep='\t').itertuples():
 src(row.source_file,'E','contrast reconciliation: '+row.source_type,derivation=row.definition)
# Governing summaries and source file integrity.
manifest=pd.read_csv(V3/'16_handoff/V3_OUTPUT_MANIFEST_SHA256.tsv',sep='\t').set_index('relative_path')
freeze=read('11_integration/V3_FINAL_NUMERIC_FREEZE.tsv','ALL','controlling numeric freeze')
read('11_integration/V3_CLAIM_MATRIX.tsv','ALL','claim boundaries')
src(V3/'16_handoff/V3_FINAL_HANDOFF_CN.md','ALL','handoff narrative; quantitative table wins over discrepant prose')
assert freeze.loc[freeze.metric.eq('central_claim_decision'),'value'].iloc[0]=='REFINE_V2_CLAIM'
# A: current authority and reconstructed exact nearest neighbour identities.
afile=ROOT/'GBM_ACT5_PROGRESSIVE_CLAIM_QUALIFICATION_V2_2026-09-10/ED_R2_MATCHED_SCORE_MAIN_CASES.tsv'
a=pd.read_csv(src(afile,'A','case-level mean score and ecology distance',filter='domain=GBMSPACE_ECOLOGY_AND_COMPOSITION; benchmark=SIGNATURE_SCORE'),sep='\t')
a=a[a.domain.eq('GBMSPACE_ECOLOGY_AND_COMPOSITION')&a.benchmark.eq('SIGNATURE_SCORE')].copy()
assert (a[['mean_signature_score_distance','mean_ecology_distance']]>0).all().all();save(a,'Fig3A_field')
r2=module('frozen_r2',src(R2/'scripts/20_wp2_identifiability.py','A','exact score and ecology lineage'))
for p in ['PRIMARY201_MODEL_MANIFEST_FROZEN.tsv','PRIMARY201_GENE_LONG_FROZEN.tsv','PRIMARY201_GENE_CROSSWALK_FROZEN.tsv']:src(R2/'01_authority_snapshot'/p,'A','frozen score reconstruction')
for p in ['REQUIRED_GENE_Z_EXPRESSION.parquet','PRIMARY201_MODEL_SCORES.parquet','FROZEN_ECOLOGY_AXIS_SCORES.parquet']:src(V1/'04_model_scores/DS03'/p,'A','real site ecological fingerprints')
m,g=r2.model_map();scores,cov,meta,eco=r2.prepare_gbmspace(m,g);ez=StandardScaler().fit_transform(eco)
meta=meta.reset_index();groups=meta.patient_id.to_numpy(str);candidates=[]
for row in a[a.k.eq(1)].itertuples():
 i=np.flatnonzero(meta.site_id.eq(row.sample_id))[0];s=scores[row.model_instance_id].to_numpy();pool=np.flatnonzero(groups!=groups[i]);j=pool[np.argsort(np.abs(s[pool]-s[i]),kind='mergesort')[0]]
 assert np.isclose(abs(s[i]-s[j]),row.mean_signature_score_distance)
 assert np.isclose(np.linalg.norm(ez[i]-ez[j]),row.mean_ecology_distance)
 candidates.append(dict(model_instance_id=row.model_instance_id,site1=meta.site_id[i],site2=meta.site_id[j],patient1=groups[i],patient2=groups[j],score_distance=abs(s[i]-s[j]),i=i,j=j))
rule='Current-authority k=1 signature-score cases; complete four-axis values; ascending score distance, model ID, site IDs; unique unordered site pairs and no reused sites; ecology not used for selection.'
selected=[];used=set()
for r in sorted(candidates,key=lambda r:(r['score_distance'],r['model_instance_id'],r['site1'],r['site2'])):
 if r['site1'] in used or r['site2'] in used:continue
 selected.append(r);used.update([r['site1'],r['site2']])
 if len(selected)==3:break
assert len(selected)==3
ex=[]
for n,r in enumerate(selected,1):
 for site,ix in [(1,r['i']),(2,r['j'])]:
  for axis,val in zip(['MES','HYPOXIA','MATRIX','MYELOID'],ez[ix]):ex.append(dict(pair=n,site_number=site,axis=axis,value=val,**r,selection_rule=rule))
ex=pd.DataFrame(ex);save(ex,'Fig3A_example_pair_registry')
# Near-score band is descriptive lower decile of current frozen score-distance cases.
near=float(a.mean_signature_score_distance.quantile(.10));audit('A','near_score_lower_decile',near,note='Deterministic score-only descriptive band; not inferential cutoff')
for col in ['mean_signature_score_distance','mean_ecology_distance']:
 audit('A',col+'_min',a[col].min());audit('A',col+'_max',a[col].max())
audit('A','real_case_rows',len(a));audit('A','exemplar_rule',rule)
for r in selected:audit('A','exemplar_'+r['site1'],r['site2'],note=f"{r['model_instance_id']}; score distance={r['score_distance']}")
# B.
bdim=read('02_intrinsic_dimension/INTRINSIC_DIMENSION_SUMMARY.tsv','B','supported dimensions')
b=read('02_intrinsic_dimension/DEFORMATION_VS_REFERENCE_DIMENSION.tsv','B','off-reference energy fraction',derivation='per-k patient median; full prespecified k range; no CI ribbon')
bcurve=b.groupby(['dataset_label','k'],as_index=False).orthogonal_energy_fraction.median();save(bcurve,'Fig3B_dimension_curves');save(bdim,'Fig3B_intrinsic_dimension')
for label in COHORTS:
 q=bdim[bdim.dataset_label.eq(label)].set_index('method');low=q.loc['supported_range_union_prespecified_methods','estimate_low'];high=q.loc['supported_range_union_prespecified_methods','estimate_high'];pa=q.loc['parallel_analysis_95','estimate']
 for metric,val in [('supported_low',low),('supported_high',high),('parallel_analysis',pa)]:audit('B',label+'_'+metric,val)
 for k in [2,int(high)]:audit('B',label+f'_k{k}_median',bcurve.loc[bcurve.dataset_label.eq(label)&bcurve.k.eq(k),'orthogonal_energy_fraction'].iloc[0])
 assert set(b[b.dataset_label.eq(label)].k)==set(range(1,min(20,dict(CARE=56,GLASS=111,GSE174554=20)[label]-2)+1))
# C.
c=read('03_sampling_null/REAL_VS_SAMPLING_DEFORMATION.tsv','C','patient observed vs null95',filter='reference_k=7');c=c[c.reference_k.eq(7)].copy();check('C','patient_n',c.patient_id.nunique(),56)
c['above_null95']=c.real_residual_magnitude>c.sampling_null_95_high_residual_magnitude;check('C','above_null95',c.above_null95.sum(),54);check('C','not_above_null95',(~c.above_null95).sum(),2);save(c,'Fig3C_patient_sampling')
cs=read('03_sampling_null/WP2_COHORT_SUMMARY.tsv','C','frozen patient bootstrap CI');cs=cs[cs.reference_k.eq(7)]
for col in ['real_residual_magnitude','sampling_null_median_residual_magnitude','biological_deformation_excess']:
 check('C',col+'_median',c[col].median(),cs.loc[cs.endpoint.eq(col),'median'].iloc[0])
 for col2 in ['ci_low','ci_high']:audit('C',col+'_'+col2,cs.loc[cs.endpoint.eq(col),col2].iloc[0])
# D and E; discrepancies are explicit, never filled.
dall=read('05_composition_intrinsic/CARE_COMPOSITION_INTRINSIC_SHAPLEY.tsv','D','nonadditive Shapley energies',filter='cell_threshold=50;reference_k=7')
d=dall[dall.cell_threshold.eq(50)&dall.reference_k.eq(7)].copy();d25=dall[dall.cell_threshold.eq(25)&dall.reference_k.eq(7)].copy();save(d,'Fig3D_shapley_patients');save(d25,'Fig3D_sensitivity25')
assert d.patient_id.is_unique and set(d.patient_id)<=set(c.patient_id)
assert np.allclose(d.composition_energy+d.intrinsic_energy+d.composition_intrinsic_cross_term,d.total_reconstructed_energy)
resolution=OUT/'00_PROVENANCE/authority_resolution.json'
denominator_resolved=resolution.exists() and json.loads(resolution.read_text()).get('use_frozen_D_denominators',False)
resolved=denominator_resolved and contrast_receipt['final_export_allowed']
assert denominator_resolved
check('D','patient_n',len(d),46)
check('D','sensitivity_patient_n',len(d25),53)
assert set(d.patient_id)<set(d25.patient_id)<set(c.patient_id)
audit('D','strict_nested_sets_50_in_25_in_CARE',True,True)
for thr,t in [(50,d),(25,d25)]:
 audit('D',f'excluded_patient_IDs_{thr}_cells',';'.join(sorted(set(c.patient_id)-set(t.patient_id))),note='Compared with full CARE frozen paired cohort; exact patient-level registry also saved')
audit('D','sensitivity_patient_n',len(d25));audit('D','composition_share_median',d.composition_energy_fraction_nonadditive.median());audit('D','sensitivity25_share_median',d25.composition_energy_fraction_nonadditive.median());audit('D','reconstruction_median',d.reconstruction_vs_original_delta_cosine.median())
em=read('12_figures/source_tables/FIG_V3C_C_MODEL_CLASS_SOURCE.tsv','E','model-level composition share',filter='cell_threshold=50',derivation='median across patients for each model_instance_id/final_class')
em=em[em.cell_threshold.eq(50)].groupby(['model_instance_id','final_class'],as_index=False).composition_absolute_share.median();save(em,'Fig3E_model_class')
ee=read('12_figures/source_tables/FIG_V3C_D_ECOLOGY_SOURCE.tsv','E','four ecology axes',filter='cell_threshold=50',derivation='abs(C)/(abs(C)+abs(I))');ee=ee[ee.cell_threshold.eq(50)].copy();ee['abs_C_share']=ee.composition_contribution.abs()/(ee.composition_contribution.abs()+ee.intrinsic_contribution.abs());save(ee,'Fig3E_ecology')
ec=read('05_composition_intrinsic/BOOTSTRAP_CLASS_CONTRASTS.tsv','E','publication-cluster bootstrap contrast');ec=ec[ec.class_a.eq('TME-dominant')&ec.class_b.eq('malignant-dominant')].iloc[0]
for col in ['mean_difference_a_minus_b','ci_low','ci_high']:audit('E','cluster_contrast_'+col,ec[col],note='Recovered frozen cluster-bootstrap table; exact row and definition traced, but competing 0.0535 authority remains unresolved; NOT currently annotated as final contrast')
for cl,q in em.groupby('final_class'):audit('E',cl+'_model_n',len(q));audit('E',cl+'_median',q.composition_absolute_share.median())
for axis,q in ee.groupby('ecology_axis'):audit('E',axis+'_n',len(q));audit('E',axis+'_median',q.abs_C_share.median())
audit('E','formal_contrast_authority_resolved',False,True,ok=False,note='0.091983 formal table/code versus 0.0535 hardcoded handoff prose; corresponding 0.0535 result row not recovered')
# F.
f=read('04_residual_modes/CARE_RESIDUAL_MODE_SIGNIFICANCE.tsv','F','three prespecified residual-mode gates',filter='dataset_label=CARE;reference_k=7;modes1:10');f=f[f.dataset_label.eq('CARE')&f.reference_k.eq(7)&f.residual_mode.between(1,10)].copy()
f['eigenvalue_ratio']=f.eigenvalue/f.permutation_eigenvalue_95
assert f.residual_mode.tolist()==list(range(1,11))
pass_all=(f.eigenvalue_ratio>1)&(f.bootstrap_loading_cosine_ci_low>=.6)&(f.loo_loading_cosine_min>=.6)
assert np.array_equal(pass_all,f.stable_axis);check('F','stable_axis_n',pass_all.sum(),0);save(f,'Fig3F_modes')
for r in f.itertuples():
 for col in ['eigenvalue_ratio','bootstrap_loading_cosine_ci_low','loo_loading_cosine_min']:audit('F',f'mode{r.residual_mode}_'+col,getattr(r,col))
# G: original frozen reader, primary-only scaler and basis.
vcommon=module('frozen_v3',src(V3/'15_code/v3_common.py','G','paired model matrix and deterministic loading orientation'))
coords=[];pairs=[];pm=[];rm=[];pca={};params=[]
for spec in vcommon.SPECS:
 src(V1/f'04_model_scores/{spec.dataset_id}/PRIMARY201_MODEL_SCORES.parquet','G','paired model scores',filter=f'score_scope={spec.scope}')
 membership=src(V2/'00_governance/freeze/memberships'/spec.membership,'G','frozen paired membership')
 xp,xr,models=vcommon.read_pair_matrix(spec);assert xp.shape[1]>=200 and np.isfinite(xp).all().all() and np.isfinite(xr).all().all()
 check('G',spec.label+'_pair_n',len(xp),spec.expected_n);audit('G',spec.label+'_complete_model_n',len(models))
 mu=xp.mean(axis=0);sd=xp.std(axis=0,ddof=1);keep=sd>vcommon.EPS
 P=((xp-mu)/sd).loc[:,keep].to_numpy();R=((xr-mu)/sd).loc[:,keep].to_numpy();basis,eig,exp=vcommon.fit_reference(P,2);pc=P@basis;rc=R@basis
 pca[spec.label]=(pc,rc,exp)
 for j in range(2):audit('G',spec.label+f'_PC{j+1}_explained_percent',100*exp[j])
 for mi in np.flatnonzero(keep):params.append(dict(dataset_label=spec.label,model_instance_id=models[mi],primary_mean=mu.iloc[mi],primary_sd=sd.iloc[mi],PC1_loading=basis[list(np.flatnonzero(keep)).index(mi),0],PC2_loading=basis[list(np.flatnonzero(keep)).index(mi),1]))
 mem=pd.read_csv(membership,sep='\t')
 if 'primary_sample_id' not in mem:
  pt=src(V1/'05_modules/module_C_longitudinal/PATIENT_LONGITUDINAL_SIGNATURE_SIMILARITY.tsv','G','sample pair identities')
  t=pd.read_csv(pt,sep='\t');t=t[t.dataset_id.eq(spec.dataset_id)&t.score_scope.eq(spec.scope)][['patient_id','primary_sample_id','recurrent_sample_id']].drop_duplicates();mem=mem.merge(t,on='patient_id',validate='one_to_one')
 for i,pid in enumerate(xp.index):
  mr=mem[mem.patient_id.astype(str).eq(pid)].iloc[0]
  pairs.append(dict(dataset_label=spec.label,patient_id=pid,primary_sample_id=mr.primary_sample_id,recurrent_sample_id=mr.recurrent_sample_id,feature_n=len(models)))
  for phase,vals in [('PRIMARY',pc[i]),('RECURRENT',rc[i])]:coords.append(dict(dataset_label=spec.label,patient_id=pid,phase=phase,PC1=vals[0],PC2=vals[1]))
 pm.append(xp.rename_axis('patient_id').reset_index().assign(dataset_label=spec.label));rm.append(xr.rename_axis('patient_id').reset_index().assign(dataset_label=spec.label))
for arr,name in [(pm,'primary'),(rm,'recurrent')]:pd.concat(arr).to_parquet(DATA/f'Fig3G_{name}_model_matrix.parquet',index=False)
save(pd.DataFrame(pairs),'Fig3G_pair_registry');save(pd.DataFrame(coords),'Fig3G_PCA_coordinates');save(pd.DataFrame(params),'Fig3G_primary_PCA_parameters')
# Validate exclusions against the actual frozen paired PCA cohort and compartment counts.
care_pairs=pd.DataFrame(pairs).query("dataset_label == 'CARE'")
care_set=set(care_pairs.patient_id)
assert set(d.patient_id)<set(d25.patient_id)<care_set and care_set==set(c.patient_id)
profiles_path=src(V3/'05_composition_intrinsic/CARE_BROAD_COMPARTMENT_REQUIRED_GENE_COUNTS.parquet','D','threshold eligibility from paired compartment cell counts')
profiles=pd.read_parquet(profiles_path,columns=['sample_id','broad_compartment','cell_n'])
eligibility=[]
for row in care_pairs.itertuples():
 pp=profiles[profiles.sample_id.eq(row.primary_sample_id)].set_index('broad_compartment').cell_n
 rr=profiles[profiles.sample_id.eq(row.recurrent_sample_id)].set_index('broad_compartment').cell_n
 common=pp.index.intersection(rr.index)
 for thr,table in [(50,d),(25,d25)]:
  eligible=[comp for comp in common if pp[comp]>=thr and rr[comp]>=thr]
  included=row.patient_id in set(table.patient_id)
  assert included==(len(eligible)>=2)
  eligibility.append(dict(patient_id=row.patient_id,primary_sample_id=row.primary_sample_id,recurrent_sample_id=row.recurrent_sample_id,cell_threshold=thr,included=included,eligible_explicit_compartment_n=len(eligible),eligible_compartments=';'.join(eligible),exclusion_reason='' if included else 'Fewer than two common explicit compartments meet threshold at both timepoints; original code skips pair'))
save(pd.DataFrame(eligibility),'Fig3D_threshold_membership_and_exclusions')
audit('D','eligibility_reproduced_from_cell_counts',True,True)
src(V2/'00_governance/freeze/memberships/AS_PRIMARY201_V2.tsv','G','complete frozen model registry')
ggeom=read('09_geometry_decomposition/GEOMETRY_DECOMPOSITION.tsv','G','high-dimensional geometry; separate from 2D projection');save(ggeom,'Fig3G_geometry')
for r in ggeom.itertuples():
 if r.geometry_component in ['centroid_translation','subspace_rotation_max_principal_angle_deg','variance_expansion_topk_trace_ratio','recurrent_direction_3_variance_ratio']:
  for col in ['estimate','ci_low','ci_high']:
   if pd.notna(getattr(r,col)):audit('G',r.dataset_label+'_'+r.geometry_component+'_'+col,getattr(r,col))
for s in SOURCES:
 p=Path(s['source_file'])
 if p.is_relative_to(V3) and str(p.relative_to(V3)) in manifest.index:
  assert sha(p)==manifest.loc[str(p.relative_to(V3)),'sha256'],f'Source hash mismatch: {p}'
pd.DataFrame(SOURCES).to_csv(OUT/'01_SOURCE_MAP.tsv',sep='\t',index=False)
pd.DataFrame(AUDIT).to_csv(OUT/'02_NUMERIC_AUDIT.tsv',sep='\t',index=False)
save(pd.DataFrame([dict(file=p.name,sha256=sha(p)) for p in DATA.iterdir() if p.is_file()]),'plotting_data_sha256')
# Layout in page fractions; reference A/B, C/D, E/F, full-width G.
fig=plt.figure(figsize=(210/25.4,297/25.4),facecolor='white')
RECT={'A':(.03,.700,.47,.245),'B':(.515,.700,.455,.245),'C':(.03,.450,.455,.245),'D':(.500,.450,.470,.245),'E':(.03,.245,.470,.197),'F':(.515,.245,.455,.197),'G':(.03,.030,.940,.207)}
def txt(x,y,s,size=7,**kw):return fig.text(x,y,s,fontsize=size,**kw)
def panel(letter,title):
 x,y,w,h=RECT[letter];txt(x+.001,y+h-.004,letter,16,weight='bold',va='top');txt(x+.039,y+h-.008,title,8.5,weight='bold',va='top')
def axin(letter,x,y,w,h):
 X,Y,W,H=RECT[letter];ax=fig.add_axes([X+x*W,Y+y*H,w*W,h*H]);ax.spines[['top','right']].set_visible(False);return ax
def pt(letter,x,y,s,size=7,**kw):
 X,Y,W,H=RECT[letter];return txt(X+x*W,Y+y*H,s,size,**kw)
def title(ax,s):ax.set_title(s,pad=6,fontsize=7)
def violin(ax,values,pos,color,width=.65):
 vals=np.asarray(values);assert np.isfinite(vals).all()
 vp=ax.violinplot(vals,positions=[pos],widths=width,showextrema=False)
 for body in vp['bodies']:body.set_facecolor(color);body.set_alpha(.25);body.set_edgecolor('none')
 # deterministic display-only horizontal offsets; every ordinate is a real observation.
 offset=((np.arange(len(vals))*.61803398875)%1-.5)*width*.58
 ax.scatter(pos+offset,vals,s=3,color=color,alpha=.7,linewidths=0,zorder=3)
 q1,med,q3=np.quantile(vals,[.25,.5,.75]);ax.plot([pos,pos],[q1,q3],color=NAVY,lw=2);ax.plot([pos-width*.25,pos+width*.25],[med,med],color=NAVY,lw=1.8);return med
# continuous separators, no boxes.
for y in [.696,.446,.241]:fig.add_artist(Line2D([.03,.97],[y,y],transform=fig.transFigure,color=CYAN,lw=.6))
for y0,y1,x in [(.700,.945,.506),(.450,.695,.492),(.245,.442,.506)]:fig.add_artist(Line2D([x,x],[y0,y1],transform=fig.transFigure,color=CYAN,lw=.6))
txt(.03,.975,'Figure 3',12.5,weight='bold',va='top')
fig.add_artist(Line2D([.151,.151],[.949,.977],transform=fig.transFigure,color=BLUE,lw=.6))
txt(.166,.975,'Longitudinal GBM state deformation persists after dimensional correction',12,weight='bold',va='top')
txt(.166,.956,'and is predominantly within-compartment, without a stable shared recurrence axis',10.4,va='top')
panel('A','Matched scores leave ecological freedom')
ax=axin('A',.115,.15,.49,.66);x=a.mean_signature_score_distance;y=a.mean_ecology_distance
hb=ax.hexbin(x,y,xscale='log',yscale='log',gridsize=32,mincnt=1,cmap='viridis',linewidths=0,rasterized=True)
ax.axvspan(x.min()*.9,near,color=BLUE,alpha=.055,zorder=-1);ax.axvline(near,color=GREY,ls='--',lw=.7)
ax.set(xlabel='Mean signature-score distance',ylabel='Mean four-axis ecological distance')
pt('A',.12,.86,f'Nearest-score decile ≤ {near:.3g}',6.3)
cax=axin('A',.618,.52,.016,.22);cb=fig.colorbar(hb,cax=cax);cb.set_label('Cases / bin',fontsize=6.3);cb.ax.tick_params(labelsize=6.3)
pt('A',.73,.86,'Matched-score pairs',6.3,ha='center')
for idx,r in enumerate(selected):
 ax=axin('A',.745,.65-idx*.24,.235,.125)
 for site,color in [(1,BLUE),(2,RED)]:
  q=ex[(ex.pair==idx+1)&ex.site_number.eq(site)];ax.plot(range(4),q.value,'o-',color=color,ms=2.7,lw=.7)
 ax.set_xticks(range(4),['MES','HYP','MAT','MYE']);ax.tick_params(axis='x',labelsize=6.3,pad=1);title(ax,f'Δscore = {r["score_distance"]:.3g}')
 ax.axhline(0,color=GREY,lw=.4,zorder=-1)
pt('A',.73,.03,'Site 1',6.3,color=BLUE);pt('A',.87,.03,'Site 2',6.3,color=RED)
panel('B','Dimensional correction reduces but does not\nremove deformation')
ax=axin('B',.15,.28,.31,.48);ax.axvline(2,color=GREY,ls='--',lw=.7)
for yi,label in enumerate(COHORTS[::-1]):
 q=bdim[bdim.dataset_label.eq(label)].set_index('method');lo=q.loc['supported_range_union_prespecified_methods','estimate_low'];hi=q.loc['supported_range_union_prespecified_methods','estimate_high'];pa=q.loc['parallel_analysis_95','estimate']
 ax.plot([lo,hi],[yi,yi],lw=8,color=COL[label],alpha=.25,solid_capstyle='butt');ax.scatter(pa,yi,s=24,color=COL[label]);ax.text(lo,yi+.19,f'{lo:g}',size=6.3);ax.text(hi,yi+.19,f'{hi:g}',size=6.3,ha='right')
ax.set(ylim=(-.5,2.5),xlim=(0,20),xlabel='Intrinsic dimension (k)');ax.set_yticks(range(3),COHORTS[::-1]);title(ax,'Primary-state dimension')
pt('B',.07,.135,'Range: prespecified methods',6.3);pt('B',.07,.075,'Point: parallel analysis',6.3)
ax=axin('B',.63,.18,.35,.51)
for label in COHORTS:
 q=bcurve[bcurve.dataset_label.eq(label)];ax.plot(q.k,q.orthogonal_energy_fraction,color=COL[label],lw=1,marker='o',ms=2,label=label)
 hi=int(bdim[(bdim.dataset_label.eq(label))&bdim.method.eq('supported_range_union_prespecified_methods')].estimate_high.iloc[0]);val=q[q.k.eq(hi)].orthogonal_energy_fraction.iloc[0];ax.scatter(hi,val,color=COL[label],s=15,zorder=5)
ax.set(xlabel='Reference dimension k',ylabel='Median off-reference fraction',yscale='log');ax.legend(frameon=False,fontsize=6.3,loc='lower left',handlelength=1);title(ax,'Longitudinal deformation')
# Exact endpoint annotations compactly above curve.
for i,label in enumerate(COHORTS):
 q=bcurve[bcurve.dataset_label.eq(label)];hi=int(bdim[(bdim.dataset_label.eq(label))&bdim.method.eq('supported_range_union_prespecified_methods')].estimate_high.iloc[0]);v2=q[q.k.eq(2)].orthogonal_energy_fraction.iloc[0];vh=q[q.k.eq(hi)].orthogonal_energy_fraction.iloc[0]
 pt('B',.60,.86-i*.043,f'k2 {v2:.3f} → k{hi} {vh:.3f}',6.3,color=COL[label])
panel('C','Deformation exceeds sampling expectation')
ax=axin('C',.13,.31,.53,.53)
x=c.sampling_null_95_high_residual_magnitude;y=c.real_residual_magnitude;lim=[min(x.min(),y.min())*.75,max(x.max(),y.max())*1.25]
ax.plot(lim,lim,'--',color=NAVY,lw=.7);ax.scatter(x,y,c=np.where(c.above_null95,RED,GREY),s=11,linewidths=0)
ax.set(xscale='log',yscale='log',xlim=lim,ylim=lim,xlabel='Patient-specific null 95th percentile',ylabel='Observed longitudinal\ndeformation magnitude')
ax.legend(handles=[Line2D([],[],marker='o',ls='',color=RED,label=f'> null95 (n={c.above_null95.sum()})'),Line2D([],[],marker='o',ls='',color=GREY,label=f'≤ null95 (n={(~c.above_null95).sum()})')],frameon=False,loc='upper left',handlelength=.7)
ax=axin('C',.84,.31,.13,.53);violin(ax,c.biological_deformation_excess,1,RED);ax.set_xticks([]);ax.set_ylabel('Biological excess\n(observed − null median)');
if (c.biological_deformation_excess>0).all():ax.set_yscale('log')
pt('C',.5,.13,f'{c.above_null95.sum()}/{len(c)} > null95',8,weight='bold',ha='center')
pt('C',.5,.065,f'Medians: observed {c.real_residual_magnitude.median():.2f}; null {c.sampling_null_median_residual_magnitude.median():.2f}; excess {c.biological_deformation_excess.median():.2f}',6.3,ha='center')
panel('D','Within-compartment change dominates')
ax=axin('D',.115,.23,.405,.60);x=d.composition_energy;y=d.intrinsic_energy;lim=[min(x.min(),y.min())*.6,max(x.max(),y.max())*1.5]
ax.plot(lim,lim,'--',lw=.7,color=NAVY);ax.scatter(x,y,color=BLUE,s=13,linewidths=0);ax.set(xscale='log',yscale='log',xlim=lim,ylim=lim,xlabel='Composition energy',ylabel='Intrinsic-state energy')
ax.text(.05,.96,'Intrinsic-dominant',transform=ax.transAxes,va='top',size=6.3);ax.text(.96,.04,'Composition-dominant',transform=ax.transAxes,ha='right',size=6.3)
ax=axin('D',.675,.57,.30,.25);vals=np.sort(d.composition_energy_fraction_nonadditive);ax.scatter(np.arange(1,len(vals)+1),vals,s=10,color=BLUE);ax.axhline(np.median(vals),color=RED,ls='--',lw=.7);ax.set_yscale('log');ax.set_xticks([]);ax.set_xlabel(f'Patients (n = {len(vals)}, ranked)');title(ax,'Composition-energy share')
ax.text(.04,.93,f'Median {100*np.median(vals):.2f}%',transform=ax.transAxes,size=6.3,color=RED,va='top')
pt('D',.64,.48,f'25-cell sensitivity: n={len(d25)}',6.3)
pt('D',.64,.435,f'Median {100*d25.composition_energy_fraction_nonadditive.median():.2f}%',6.3)
pt('D',.50,.025,f'CARE, k=7; ≥50 cells, n={len(d)} evaluable pairs',6.3,ha='center')
ax=axin('D',.675,.12,.30,.24);med=violin(ax,d.reconstruction_vs_original_delta_cosine,1,BLUE,width=.5);ax.set(xlim=(.45,1.9),ylim=(0,1.05),ylabel='Cosine similarity');ax.set_xticks([]);title(ax,'Reconstruction validity');ax.text(1.26,.62,f'Median\n{med:.3f}',size=6.3)
panel('E','Composition is model-aware but remains subdominant')
classes=['malignant-dominant','TME-dominant','mixed','unstable'];axes=['MES','HYPOXIA','MATRIX','MYELOID'];colors=[BLUE,RED,GREEN,PURPLE]
for side,(table,col,group,labels) in enumerate([(em,'composition_absolute_share','final_class',classes),(ee,'abs_C_share','ecology_axis',axes)]):
 ax=axin('E',.085+side*.51,.30,.39,.46)
 for i,(cl,color) in enumerate(zip(labels,colors)):
  vals=table.loc[table[group].eq(cl),col];med=violin(ax,vals,i,color);ax.text(i,-.33,f'{100*med:.1f}%',transform=ax.get_xaxis_transform(),ha='center',size=6.3)
 ax.set_yscale('log');ax.set_ylim(min(table[col].min()*.65,.01),1.15);ax.axhline(.5,color=GREY,ls='--',lw=.6);ax.set_ylabel('Composition share');ax.set_xticks(range(4),['Malignant\ndominant','TME\ndominant','Mixed','Unstable'] if side==0 else ['MES','HYP','MAT','MYE']);ax.tick_params(axis='x',labelsize=6.3);title(ax,'By model attribution class' if side==0 else 'By frozen ecology axes')
pt('E',.50,.025,'Formal contrast: source reconciliation pending',6.3,ha='center')
panel('F','No stable shared residual axis')
for i,(col,th,color,label) in enumerate([('eigenvalue_ratio',1,BLUE,'Eigenvalue /\npermutation 95th'),('bootstrap_loading_cosine_ci_low',.6,ORANGE,'Bootstrap loading\nCI lower bound'),('loo_loading_cosine_min',.6,GREEN,'LOO loading\nminimum cosine')]):
 ax=axin('F',.105+i*.303,.24,.245,.51);ax.axvline(th,color=GREY,ls='--',lw=.8);ax.scatter(f[col],f.residual_mode,color=color,s=13);ax.set_ylim(10.7,.3);ax.set_yticks(range(1,11));title(ax,label)
 if i:ax.set_yticklabels([])
 else:ax.set_ylabel('Residual mode')
 ax.set_xlim(0,max(1.02,1.08*f[col].max()) if i else 1.15*f[col].max());ax.set_xlabel(f'Gate ≥ {th:.2g}' if i else 'Gate > 1')
pt('F',.51,.04,f'{int(pass_all.sum())} modes pass all three gates',8,color=RED,weight='bold',ha='center',bbox=dict(facecolor='#FDECEE',edgecolor='none',pad=3))
panel('G','Longitudinal geometry reorganizes differently across cohorts')
# Covariance probability ellipses, calculated from each actual coordinate cloud.
def ellipse(ax,xy,color):
 cov=np.cov(xy,rowvar=False);ev,vec=np.linalg.eigh(cov);order=np.argsort(ev)[::-1];ev=ev[order];vec=vec[:,order];angle=np.degrees(np.arctan2(vec[1,0],vec[0,0]))
 for prob,ls in [(.8,':'),(.5,'--')]:
  wh=2*np.sqrt(ev*chi2.ppf(prob,2));ax.add_patch(Ellipse(xy.mean(axis=0),*wh,angle=angle,facecolor=color if prob==.5 else 'none',alpha=.08 if prob==.5 else .6,edgecolor=color,lw=.65,ls=ls))
for j,label in enumerate(COHORTS):
 ax=axin('G',.04+j*.201,.28,.158,.51);P,R,exp=pca[label]
 for p,r in zip(P,R):ax.plot([p[0],r[0]],[p[1],r[1]],color=GREY,alpha=.35,lw=.35,zorder=1)
 ellipse(ax,P,BLUE);ellipse(ax,R,RED);ax.scatter(P[:,0],P[:,1],s=7,color=BLUE,alpha=.75,linewidths=0);ax.scatter(R[:,0],R[:,1],s=9,color=RED,marker='^',alpha=.75,linewidths=0)
 for values,color in [(P,BLUE),(R,RED)]:center=values.mean(axis=0);ax.scatter(*center,s=37,color=color,edgecolors='white',linewidth=.5,zorder=5)
 ax.annotate('',xy=R.mean(axis=0),xytext=P.mean(axis=0),arrowprops=dict(arrowstyle='->',color=NAVY,lw=.9))
 ax.set(xlabel=f'PC1 ({100*exp[0]:.1f}%)',ylabel=f'PC2 ({100*exp[1]:.1f}%)');title(ax,f'{label} (n = {len(P)} pairs)');ax.locator_params(axis='both',nbins=4)
for j,(component,label,lim) in enumerate([('centroid_translation','Centroid translation',(0,9)),('subspace_rotation_max_principal_angle_deg','Max principal angle (°)',(0,90)),('variance_expansion_topk_trace_ratio','Top-k variance trace ratio',(0,1.7))]):
 ax=axin('G',.666,.65-j*.26,.145,.12)
 for i,co in enumerate(COHORTS):
  r=ggeom[ggeom.dataset_label.eq(co)&ggeom.geometry_component.eq(component)].iloc[0];ax.scatter(r.estimate,2-i,s=12,color=COL[co]);ax.annotate(f'{r.estimate:.1f}' if j==1 else f'{r.estimate:.3f}',(r.estimate,2-i),xytext=(3,0),textcoords='offset points',size=6.3,color=COL[co],ha='left',va='center')
 ax.set(ylim=(-.7,2.7),xlim=lim);ax.set_yticks([]);ax.tick_params(axis='x',labelsize=6.3);ax.set_title(label,pad=5,fontsize=6.3)
 if j==2:ax.axvline(1,color=GREY,ls='--',lw=.6)
ax=axin('G',.865,.54,.13,.10);r=ggeom[ggeom.dataset_label.eq('CARE')&ggeom.geometry_component.eq('recurrent_direction_3_variance_ratio')].iloc[0];ax.errorbar(r.estimate,0,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt='o',ms=3,color=NAVY,lw=.8,capsize=2);ax.axvline(1,color=GREY,ls='--',lw=.6);ax.set(xlim=(0,5),ylim=(-1,1));ax.set_yticks([]);ax.set_xticks([0,1,3,5]);pt('G',.93,.83,'CARE recurrent\ndirection 3',6.3,ha='center');pt('G',.93,.70,f'{r.estimate:.3f}\n[{r.ci_low:.3f}, {r.ci_high:.3f}]',6.3,ha='center')
pt('G',.845,.38,'Structured longitudinal\nreorganization',6.3,weight='bold');pt('G',.845,.235,'Translation + rotation\n+ cohort-specific\nvariance change',6.3);pt('G',.845,.12,'No single shared\nrecurrence axis',6.3)
handles=[Line2D([],[],color=BLUE,marker='o',ls='',label='Primary'),Line2D([],[],color=RED,marker='^',ls='',label='Recurrent'),Line2D([],[],color=GREY,lw=.6,label='Patient pair'),Line2D([],[],color=NAVY,marker='o',ls='',label='Centroids'),Line2D([],[],color=GREY,ls='--',label='50% covariance ellipse'),Line2D([],[],color=GREY,ls=':',label='80% covariance ellipse')]
fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.03,.035),ncol=6,frameon=False,fontsize=6.3,columnspacing=1,handlelength=1.5)
# Suppress crowded logarithmic minor labels without suppressing observations.
for ax in fig.axes:
 if ax.get_xscale()=='log':ax.xaxis.set_minor_formatter(NullFormatter())
 if ax.get_yscale()=='log':ax.yaxis.set_minor_formatter(NullFormatter())
# Numerically readable log ticks in narrow real-data ranges.
for ax in fig.axes:
 for dim in ['x','y']:
  if getattr(ax,'get_'+dim+'scale')()=='log':
   lo,hi=getattr(ax,'get_'+dim+'lim')();ticks=[v for v in [.1,.2,.5,1,2,5,10,20,50,100,200,500,1000] if lo<=v<=hi]
   if hi/lo<100:
    axis=getattr(ax,dim+'axis');axis.set_major_locator(FixedLocator(ticks));axis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
# Preliminary rendering permitted; final files only after all critical gates pass.
fig.savefig(OUT/'00_PROVENANCE/Figure3_realdata_preview.png',dpi=180)
fig.savefig(OUT/'00_PROVENANCE/Figure3_realdata_preview.pdf')
if resolved:
 assert all(r['status']=='PASS' for r in AUDIT)
 for ext in ['pdf','svg','png']:fig.savefig(OUT/'05_FINAL'/f'Figure3_ACT3_FINAL_REALDATA_A4_portrait.{ext}',dpi=600)
 for letter,(x,y,w,h) in RECT.items():
  bbox=Bbox.from_bounds(x*fig.get_figwidth(),y*fig.get_figheight(),w*fig.get_figwidth(),h*fig.get_figheight())
  for ext in ['pdf','svg','png']:fig.savefig(OUT/'04_PANEL_OUTPUTS'/f'Fig3{letter}.{ext}',dpi=600,bbox_inches=bbox)
(OUT/'00_PROVENANCE/production_status.json').write_text(json.dumps(dict(FIG3_ACT3_STATUS='BLOCKED_PANEL_E_CONTRAST',REAL_DATA_ONLY=True,SYNTHETIC_NUMERIC_DATA_USED='NO',PCA_PRIMARY_REFERENCE_ONLY=True,PCA_PAIR_COUNTS_VERIFIED=True,D_DENOMINATORS_RESOLVED=True,D_NESTED_SETS_VERIFIED=True,NUMERIC_AUDIT_PASS=False,FINAL_EXPORTED=False),indent=2))
print(json.dumps(dict(status='RENDERED_PENDING_VISUAL_QA' if resolved else 'BLOCKED_PANEL_E_CONTRAST',preview=str(OUT/'00_PROVENANCE/Figure3_realdata_preview.png'),synthetic_numeric_data_used=False),indent=2))
