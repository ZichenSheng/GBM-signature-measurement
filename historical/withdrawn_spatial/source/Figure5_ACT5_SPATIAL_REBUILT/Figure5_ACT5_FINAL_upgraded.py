#!/usr/bin/env python3
"""Frozen ACT5 protagonist figure. Run with Python, pandas, numpy, scipy, matplotlib.
No input authorities are modified. Null reconstruction: recover_null.py (frozen seeds).
"""
from pathlib import Path
import re,hashlib,json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.patches import Rectangle
R=Path(__file__).resolve().parent.parent;O=Path(__file__).resolve().parent;S=O/'source_tables';S.mkdir(exist_ok=True)
ids=['PF4C-B4-007','PF4C-B4-029','PF4C-FINAL-079'];names=['Liu EMT-5','Xiao EMT-4','Arimappamagan-14'];colors=['#1458DB','#D10A88','#F26722'];short=['Liu','Xiao','Arim.'];manifest=[]
def read(p,tag,filtercol=None):
 p=R/p;d=pd.read_csv(p,sep='\t');d=d[d[filtercol].isin(ids)].copy() if filtercol else d
 manifest.append(dict(panel_source=tag,path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows_exported=len(d)))
 d.to_csv(S/(tag+'.tsv'),sep='\t',index=False);return d
v1=Path('GBM_ACT5_REAL_USE_SIGNATURE_DECISION_DEMONSTRATION_V1_2026-09-09');v2=Path('GBM_ACT5_PROGRESSIVE_CLAIM_QUALIFICATION_V2_2026-09-10');base=Path('GBM标签数据库/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23/09_de_novo_derived');ds=Path('GBM_R4_STAGEB1_DS03_CALIBRATED_DETECTABILITY_TRANSPORT_2026-09-08')
prov=read(v1/'02_ACT5_SOURCE_PUBLICATION_PROVENANCE.tsv','A_publication','signature_id').set_index('signature_id')
bs=read(v2/'11_ACT5_V2_FIGURE5_SOURCE.tsv','B_ACT5_authority','signature_id')
c=read(base/'14_full_gene_single_cell_v2_1_2/04_patient_localization/PRIMARY201_FULL_GENE_PATIENT_DELTA.tsv','C_patient_delta','model_instance_id');c=c[c.localization_status=='EVALUABLE'];c.to_csv(S/'C_patient_delta.tsv',sep='\t',index=False)
cl=read(base/'14_full_gene_single_cell_v2_1_2/05_model_classification/PRIMARY201_FULL_GENE_CLASSIFICATION.tsv','C_classification','model_instance_id').set_index('model_instance_id')
d=read(base/'12_spatial_v2_1_1/05_patient_level_inference/PRIMARY201_PATIENT_EFFECTS.tsv','D_patient_effects','score_id')
dsm=read(base/'12_spatial_v2_1_1/05_patient_level_inference/PRIMARY201_PATIENT_INFERENCE_SUMMARY.tsv','D_frozen_intervals','score_id')
iv=read(v2/'08_ACT5_V2_IVY_CONTEXT_SWITCH.tsv','E_context','signature_id')
ic=read(Path('GBM_R4_IVY_GBMAP_ALTERNATIVE_CAPACITY_AUDIT_2026-09-08/03_ivy_structure_and_capacity/IVY_SIGNATURE_COVERAGE_AND_SCOPE.tsv'),'E_Ivy_coverage','signature_id').set_index('signature_id')
dc=read(Path('GBM_BRP_PHASE2_201_SIGNATURE_LANDSCAPE_2026-09-02/04_signature_scores/SCORE_COVERAGE_201.tsv'),'E_DS03_coverage','signature_id');dc=dc[(dc.dataset_id=='DS03')&(dc.score_scope=='MALIGNANT_ONLY')];dc.to_csv(S/'E_DS03_coverage.tsv',sep='\t',index=False)
w=read(Path('GBM_R4_POST_STAGEB1_STRATEGIC_CLOSURE_AUDIT_2026-09-09/03_null_geometry/DS03_BLOCKED_VS_NAIVE_WORLD_REVERSAL.tsv'),'FG_worlds');w=w[w.q==1];w.to_csv(S/'FG_worlds.tsv',sep='\t',index=False)
den=read(v2/'06_ACT5_V2_COMPLETE_STAGEB1_DENOMINATOR.tsv','G_denominator')
assert len(c)==177 and len(w)==160 and w.groupby('signature_id').size().eq(20).all()
assert d.groupby(['score_id','niche']).size().eq(17).all()
mat=c.pivot(index='model_instance_id',columns=['dataset_id','patient_id'],values='patient_delta').loc[ids].sort_index(axis=1)
coefs=[]
for sid in ids:
 formula=prov.loc[sid,'exact_coefficient_formula']
 if sid==ids[2]: pairs=[(g,float(v)) for g,v in re.findall(r'([A-Z0-9]+) (-?\d+\.\d+)',formula)]
 else:
  pairs=[(g,float(v.replace(' ',''))) for v,g in re.findall(r'([+-]?\s*\d+\.\d+)×([A-Z0-9]+)',formula)]
  order=['KDM5B','GJB2','TGM2','FAM3C','MARVELD1'] if sid==ids[0] else ['SPP1','SPOCK1','PLOD2','CRLF1'];pairs=[(g,dict(pairs)[g]) for g in order]
 assert len(pairs)==[5,4,14][ids.index(sid)]
 coefs.extend(dict(signature_id=sid,gene=g,coefficient=v,membership=1) for g,v in pairs)
coef=pd.DataFrame(coefs);coef.to_csv(S/'A_coefficients_membership.tsv',sep='\t',index=False)
mpl.rcParams.update({'font.family':'Arial','font.size':7,'axes.linewidth':.5,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2,'ytick.major.size':2,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
f=plt.figure(figsize=(210/25.4,297/25.4),facecolor='white')
def ax(x,y,width,height,clean=True):
 a=f.add_axes([x/210,1-(y+height)/297,width/210,height/297]);
 if clean:
  a.spines[['top','right']].set_visible(False)
 return a
def txt(x,y,s,size=7,weight='normal',color='#202027',ha='left',va='top',**kw):
 return f.text(x/210,1-y/297,s,fontsize=size,fontweight=weight,color=color,ha=ha,va=va,**kw)
def panel(letter,x,y,title):
 txt(x,y,letter,13,'bold');txt(x+6,y+.6,title,8.6,'bold')
def rule(x,y,w):
 f.add_artist(plt.Line2D([x/210,(x+w)/210],[1-y/297]*2,color='#D4D5D8',lw=.5))
from scipy.stats import gaussian_kde
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

def rain(a,values,y,color,width=.38):
 values=np.asarray(values);grid=np.linspace(values.min(),values.max(),180);density=gaussian_kde(values)(grid);density=density/density.max()*width
 a.fill_between(grid,y,y-density,color=color,alpha=.22,lw=.65,edgecolor=color)
 a.scatter(values,y+.13+.11*np.sin(np.arange(len(values))*2.399),s=6.5,color=color,alpha=.62,lw=.2,edgecolors='white',zorder=4)


rawax=ax;rawtxt=txt
def ax(x,y,width,height,clean=True): return rawax(x, y if y<20 else 20+(y-20)*.82,width,height*.82 if y>=20 else height,clean)
def txt(x,y,*args,**kw):return rawtxt(x,y if y<20 else 20+(y-20)*.82,*args,**kw)
# A compact support strip
text=txt(8,5,'Context qualifies what glioblastoma signatures measure',12,'bold')
panel('A',8,14,'Representation identity');txt(202,16,'Published signed predictor ≠ unsigned, unweighted membership',6.6,ha='right')
for k,sid in enumerate(ids):
 x=9+65*k;cd=coef[coef.signature_id==sid];n=len(cd);txt(x,22,names[k],8,'bold',colors[k]);txt(x,27,'Published coefficients',5.8);txt(x+49,27,'ACT5 membership',5.8,ha='center')
 a=ax(x+15,31,22,25 if n==14 else 21);a.axvline(0,color='#848992',lw=.6);a.barh(np.arange(n),cd.coefficient,height=.67,color=colors[k],alpha=.88);a.set_yticks(range(n),cd.gene,fontsize=4.6 if n==14 else 5.8);a.invert_yaxis();lim=max(abs(cd.coefficient))*1.15;a.set_xlim(-lim,lim);a.set_xticks([-round(lim/1.15,2),0,round(lim/1.15,2)]);a.tick_params(axis='y',length=0);a.spines['left'].set_visible(False);a.set_xlabel('Coefficient' if n<14 else '',fontsize=5.5,labelpad=1)
 a=ax(x+40,33,20,14);a.set_xlim(-.5,4.5);a.set_ylim(3,-1);a.axis('off')
 for j in range(n):a.scatter(j%5,j//5 if n>5 else .7,s=20,c=colors[k],edgecolors='white',lw=.3)
 txt(x+50,49,f'{n} equal members',6,ha='center');txt(x+50,53,'Unsigned · unweighted',5.6,ha='center')
 if k<2:f.add_artist(plt.Line2D([(x+62)/210]*2,[1-22/297,1-50/297],color='#D2D4D9',lw=.55))

ax=rawax;txt=rawtxt
# B: six genuine paired patient-score scatterplots
from scipy.stats import spearmanr
bulk=pd.read_csv(S/'B_patient_scatter.tsv',sep='\t')
panel('B',8,53,'Bulk patient-level MES association')
for k,sid in enumerate(ids):
 txt(32+k*23,60,names[k],5.6,'bold',colors[k],ha='center')
 for j,co in enumerate(['CGGA325','CGGA693']):
  q=bulk[(bulk.signature_id==sid)&(bulk.cohort==co)];a=ax(22+k*23,66+j*20,20,16);a.scatter(q.membership_score,q.MES_score,s=4.3,c=colors[k],alpha=.32,lw=0)
  slope,intercept=np.polyfit(q.membership_score,q.MES_score,1);xx=np.array([-2.5,2.5]);a.plot(xx,slope*xx+intercept,c=colors[k],lw=1.15);rho=spearmanr(q.membership_score,q.MES_score).statistic;a.text(.03,.94,f'ρ = {rho:.3f}',fontsize=5.4,transform=a.transAxes,va='top')
  a.set_xlim(-2.7,2.8);a.set_ylim(-2.5,2.5);a.set_xticks([-2,0,2]);a.set_yticks([-2,0,2]);a.tick_params(labelsize=4.8,pad=1);a.set_xlabel('');a.set_yticklabels(['−2','0','2'] if k==0 else []);a.set_xticklabels(['−2','0','2'] if j else [])
  if k==0:txt(8,70+j*20,co+f'\n(n={len(q)})',5.5)
txt(55,106,'Unsigned membership score',5.6,ha='center')
txt(5,84,'MES score',5.4,rotation=90,ha='center')
# C real existing UMAP plus unchanged frozen patient localization
panel('C',98,53,'Single-cell localization and patient consistency')
u=pd.read_csv(S/'C_single_cell_umap_scores.tsv.gz',sep='\t');typecols={'Malignant':'#477BCA','Myeloid':'#C97684','Lymphoid':'#73A8A1','Vascular':'#C79763','Glial / neuronal':'#9D95B1'}
txt(99,60,f'GBmap / Neftel2019 · {len(u):,} cells',5.5);txt(201,60,'59 patients · 7 studies',5.5,ha='right')
ucmap=LinearSegmentedColormap.from_list('cellscore',['#3065AD','#F4F4F2','#C74B43'])
for j in range(4):
 a=ax([99,128,153,178][j],67 if j==0 else 68,27 if j==0 else 23,18 if j==0 else 16);txt([112.5,139.5,164.5,189.5][j],64,(['Cell identity']+names)[j],5.2,'bold','#303741' if j==0 else colors[j-1],ha='center')
 if j==0:
  for typ,col in typecols.items():v=u[u.identity==typ];a.scatter(v.UMAP1,v.UMAP2,s=.22,c=col,lw=0)
 else:a.scatter(u.UMAP1,u.UMAP2,c=u[ids[j-1]],cmap=ucmap,vmin=-1.5,vmax=1.5,s=.22,lw=0)
 a.set_xlim(u.UMAP1.min()-1,u.UMAP1.max()+1);a.set_ylim(u.UMAP2.min()-1,u.UMAP2.max()+1);a.axis('off')
for j,(typ,col) in enumerate(typecols.items()):txt(99+(j%3)*34,85+(j//3)*2.3,typ,4.6,color=col)
txt(201,89,'All score UMAPs: mean gene z, shared ±1.5',4.7,ha='right')
a=ax(114,96,70,12);mesh=a.pcolormesh(np.arange(60),np.arange(4),mat.values,cmap=ucmap,norm=TwoSlopeNorm(0,-1.5,1.5),edgecolors='none');a.set_ylim(3,0);a.set_yticks(np.arange(3)+.5,short,fontsize=5.2);a.set_xticks([]);a.tick_params(length=0);a.spines[:].set_visible(False)
start=0;pal=['#798DB8','#C4AA86','#88A5A1','#AF94AD','#9DA6B1','#C49A9A','#81A8B8']
for j,(study,n) in enumerate(mat.columns.get_level_values(0).value_counts(sort=False).items()):
 x=114+70*start/59;wi=70*n/59;f.add_artist(Rectangle((x/210,1-95/297),wi/210,1/297,transform=f.transFigure,color=pal[j],lw=0));txt(x+wi/2,90.5,re.sub(r'20\d\d','',study)+f'\n{n}',4.2,ha='center')
 if start:a.axvline(start,c='white',lw=.6)
 start+=n
for yy in [1,2]:a.axhline(yy,c='white',lw=.8)
a=ax(190,96,12,12)
for k,sid in enumerate(ids):
 vv=mat.loc[sid].values;rain(a,vv,k+.4,colors[k],.25);a.scatter(np.median(vv),k+.4,s=10,facecolors='white',edgecolors=colors[k],lw=.6,zorder=5)
a.set_ylim(3,0);a.set_yticks([]);a.set_xticks([0,2]);a.tick_params(labelsize=4.5,pad=1);a.spines['left'].set_visible(False);a.axvline(0,c='#AAB1B9',lw=.4)
ca=ax(136,110,29,1);cb=f.colorbar(mesh,cax=ca,orientation='horizontal');cb.solids.set_rasterized(False);cb.set_ticks([-1.5,0,1.5]);cb.ax.tick_params(labelsize=4.3,pad=1);txt(121,114,'Malignant − TME (original score; colors capped ±1.5)',5)

rawax=ax;rawtxt=txt
def ax(x,y,w,h,clean=True):return rawax(x,y-3,w,h,clean)
def txt(x,y,*args,**kw):return rawtxt(x,y-3,*args,**kw)
# D1-D4: one real section, matched pixel orientation, crop and score scale
from PIL import Image
sp=pd.read_csv(S/'D_spot_scores_coordinates.tsv',sep='\t');meta=json.loads((O/'spatial_reconstruction_QA.json').read_text());tissue=np.asarray(Image.open(S/'D_tissue.png'));scale=meta['scale_factors']['lowres'];sx=sp.y.values*scale;sy=sp.x.values*scale
panel('D',8,123,'Spatial ecology');txt(202,125,'GSM7596589 / zh881t1 · 2,792 Visium spots',6,ha='right')
# Coordinates returned by GetTissueCoordinates are full-resolution image row/column, verified against Space Ranger positions.
xmin,xmax=max(0,sx.min()-12),min(tissue.shape[1],sx.max()+12);ymin,ymax=max(0,sy.min()-12),min(tissue.shape[0],sy.max()+12)
scorelim=2.;scmap=LinearSegmentedColormap.from_list('spatial_score',['#214E99','#6199C6','#F7F5F0','#DB8A64','#B53039'])
for j in range(4):
 x=8+j*25;txt(x+11.5,132,['H&E','Liu EMT-5','Xiao EMT-4','Arimappamagan-14'][j],5.6,'bold','#303741' if j==0 else colors[j-1],ha='center')
 a=ax(x,138,23,27);a.imshow(tissue,origin='upper',alpha=1 if j==0 else .14)
 if j:
  sc=a.scatter(sx,sy,c=sp[ids[j-1]],cmap=scmap,vmin=-scorelim,vmax=scorelim,s=2.3,linewidths=0,rasterized=False)
 a.set_xlim(xmin,xmax);a.set_ylim(ymax,ymin);a.set_aspect('equal');a.axis('off')
ca=ax(31,166,49,1.4);cb=f.colorbar(sc,cax=ca,orientation='horizontal',extend='neither');cb.solids.set_rasterized(False);cb.set_ticks([-2,0,2]);cb.ax.tick_params(labelsize=5,pad=1);txt(55,171,'Mean gene z; common color limits ±2',5.3,ha='center')
# D5-D7: frozen 17-patient ecological contrasts
for j,niche in enumerate(['matrix-rich','myeloid-rich','vascular-rich']):
 x=120+j*28;txt(x+11,132,niche.capitalize(),5.9,'bold',colors[j],ha='center');a=ax(x,139,23,26)
 for k,sid in enumerate(ids):
  vv=d[(d.score_id==sid)&(d.niche==niche)].patient_effect.values;rr=dsm[(dsm.score_id==sid)&(dsm.niche==niche)].iloc[0];rain(a,vv,k,colors[k],.23);a.plot([rr.bootstrap_CI_low,rr.bootstrap_CI_high],[k+.35]*2,color=colors[k],lw=1.3);a.scatter(rr.median_patient_effect,k+.35,s=17,color=colors[k],edgecolors='white',lw=.3,zorder=5)
 a.set_xlim(-.22,.86);a.set_ylim(2.8,-.55);a.set_yticks([0,1,2],short if j==0 else [],fontsize=5.5);a.tick_params(axis='y',length=0);a.axvline(0,color='#9BA3AD',ls=':',lw=.5);a.spines['left'].set_visible(False);a.set_xticks([0,.4,.8]);a.set_xlabel('High − low score',fontsize=5.1,labelpad=2)
txt(159,172,'17 patients · median and frozen bootstrap interval',5.2,ha='center')

ax=rawax;txt=rawtxt
# E: four aligned evidence blocks, without leftmost signature-name text
panel('E',8,174,'Computable ≠ informative ≠ evaluable')
for x,title in [(27,'Gene coverage'),(65,'Ivy anatomy'),(105,'Claim evaluability'),(166,'Independent Ivy tissue-scope control')]:txt(x,181,title,6.1,'bold',ha='center')
for x,t in [(18,'DS03'),(34,'Ivy'),(65,'Partial R²'),(99,'DS03'),(114,'Ivy')]:txt(x,185,t,5.3,ha='center')
for k,sid in enumerate(ids):
 y=190+k*5.2;n=int(ic.loc[sid,'genes_in_signature'])
 for xx in [14,30]:
  a=ax(xx,y,10,2.4);a.axis('off');a.set_xlim(-.5,6.5);a.set_ylim(1.5,-.5)
  for z in range(n):a.scatter(z%7,z//7,s=7,c=colors[k],lw=0)
  txt(xx+5,y+2.6,f'{n}/{n}',4.8,ha='center')
 a=ax(48,y,33,3.4);v=float(iv[(iv.signature_id==sid)&(iv.claim_scope=='Tissue/anatomical informativeness')].partial_r2.iloc[0]);a.plot([0,1],[0,0],c='#C5CBD3',lw=.5);a.plot([0,v],[0,0],c=colors[k],lw=.8);a.scatter(v,0,s=22,c=colors[k]);a.text(v,.25,f'{v:.3f}',ha='center',fontsize=5.2);a.set_xlim(0,1);a.set_ylim(-.3,.6);a.axis('off')
 for x,filled in [(99,True),(114,False)]:
  a=ax(x-2,y,4,3);a.scatter(0,0,s=25,facecolors='#33445A' if filled else 'white',edgecolors='#33445A',lw=.8);a.set_xlim(-1,1);a.set_ylim(-1,1);a.axis('off')
txt(48,205,'0                      0.5                     1',4.8);txt(87,205,'● Evaluable   ○ Not evaluable',5)
t= pd.read_csv(S/'E_tissue_control.tsv',sep='\t');cal=pd.read_csv(S/'E_tissue_calibration.tsv',sep='\t');a=ax(141,187,59,17);a.fill_between(t.q,t.q25_detectability_supported,t.q75_detectability_supported,color='#DCE4EF',alpha=.6);a.plot(t.q,t.median_detectability_supported,c='#244E96',marker='o',ms=3,lw=.85)
for rr in t.itertuples():a.text(rr.q,rr.median_detectability_supported+.07,f'{rr.median_detectability_supported:.2f}',ha='center',fontsize=5.2)
a.set_xlim(.2,1.05);a.set_ylim(0,1.22);a.set_xticks([.25,.5,.75,1]);a.set_yticks([0,.5,1]);a.tick_params(labelsize=4.8,pad=1);a.set_ylabel('Median recovery',fontsize=5.5);a.set_xlabel('q (top fraction)',fontsize=5.2,labelpad=1);txt(142,184,'61 identities · band: IQR',5.2)

rawax=ax;rawtxt=txt
def ax(x,y,w,h,clean=True):return rawax(x,y-4,w,h,clean)
def txt(x,y,*args,**kw):return rawtxt(x,y-4,*args,**kw)
# F: vivid identity-specific ambiguity + Jaccard + original legal-null density
panel('F',8,214,'Representation defines ambiguous units and legal null geometry')
m=pd.read_csv(S/'F_ambiguous_membership.tsv',sep='\t');units=sorted(m.unit_id.unique());amb=m.pivot(index='signature_id',columns='unit_id',values='middle_member').loc[ids,units]
txt(42,221,'DS03 ambiguous regions · 29/57 each',5.8,'bold',ha='center')
a=ax(23,227,57,15);a.set_xlim(0,57);a.set_ylim(3,0)
for k,sid in enumerate(ids):
 for z,v in enumerate(amb.loc[sid]):a.add_patch(Rectangle((z,k+.08),.85,.75,facecolor=colors[k] if v else '#E9E9E6',edgecolor='none'))
a.set_yticks([.45,1.45,2.45],short,fontsize=5.5);a.set_xticks([]);a.tick_params(length=0);a.spines[:].set_visible(False);txt(23,244,'Filled: middle band · pale: outside',5)
J=np.zeros((3,3))
for i in range(3):
 for j in range(3):J[i,j]=np.logical_and(amb.iloc[i],amb.iloc[j]).sum()/np.logical_or(amb.iloc[i],amb.iloc[j]).sum()
pd.DataFrame(J,index=ids,columns=ids).to_csv(S/'F_jaccard.tsv',sep='\t');txt(98,221,'Three protagonists\nJaccard',5.6,'bold',ha='center');a=ax(87,229,23,13);a.imshow(J,cmap=LinearSegmentedColormap.from_list('jaccard',['#F6F8FB','#AFC1DC']),vmin=0,vmax=1,aspect='auto');a.set_xticks([0,1,2],short,fontsize=4.7);a.xaxis.tick_top();a.set_yticks([]);a.tick_params(length=0);a.spines[:].set_visible(False)
for i in range(3):
 for j in range(3):a.text(j,i,'—' if i==j else f'{J[i,j]:.2f}',ha='center',va='center',fontsize=5.5,color=colors[i] if i!=j else '#596575')
for tick,col in zip(a.get_xticklabels(),colors):tick.set_color(col)
txt(98,244,f'3 pairs: median {np.median(J[np.triu_indices(3,1)]):.2f}',5.2,ha='center')
allj=pd.read_csv(S/'F_all8_jaccard_verified.tsv',sep='\t');assert len(allj)==28
txt(23,247,f'All 8 identities: median Jaccard = {allj.jaccard.median():.2f} (28 pairs)',5.2)
null=pd.read_csv(S/'F_recovered_blocked_null.tsv',sep='\t');txt(160,220,'Legal null geometry · 59,940 draws',5.8,'bold',ha='center')
for k,sid in enumerate(ids):
 x=119+28*k;ww=w[w.signature_id==sid];nv=null[null.signature_id==sid].T_null.values;txt(x+11.5,225,names[k],5.2,'bold',colors[k],ha='center');txt(x+11.5,229,f'{int(ww.BLOCKED_DETECTED.sum())}/20; Median p = {ww.blocked_p.median():.4g}',4.8,ha='center')
 a=ax(x,233,23,11);grid=np.linspace(nv.min(),nv.max(),250);density=gaussian_kde(nv)(grid);q95=np.quantile(nv,.95);a.fill_between(grid,0,density,color='#D4DBE4');a.plot(grid,density,c='#8E9BAB',lw=.6);a.fill_between(grid,0,density,where=grid>=q95,color='#9BAABB');a.axvline(q95,c=colors[k],ls=':',lw=.9)
 for v in ww.T_ALT:a.plot(v,-.06,'|',transform=a.get_xaxis_transform(),clip_on=False,c=colors[k],ms=5,mew=1.1)
 a.set_yticks([]);a.spines['left'].set_visible(False);a.xaxis.set_major_locator(mpl.ticker.MaxNLocator(2));a.tick_params(labelsize=4.4,pad=1);a.set_xlabel('Statistic T',fontsize=4.8,labelpad=1)

ax=rawax;txt=rawtxt
# G the quantitative climax, accurate endpoints with restrained connecting curves
panel('G',8,247,'Same-world bidirectional inferential reversal');txt(8,252,'Same positive world · same ALT statistic · different inferential null',6.5)
for k,sid in enumerate([ids[0],ids[2]]):
 x=19+k*66;i=ids.index(sid);ww=w[w.signature_id==sid].sort_values('replicate_id');txt(x+24,257,names[i],8,'bold',colors[i],ha='center');txt(x+24,261,f'{int(ww.NAIVE_DETECTED.sum())}/20 → {int(ww.BLOCKED_DETECTED.sum())}/20',10,'bold',ha='center');txt(x+24,266,f'Median p: {ww.naive_p.median():.4g} → {ww.blocked_p.median():.4g}',6,ha='center')
 a=ax(x,272,48,20);a.set_ylim(-3.1,0);a.set_xlim(-.1,1.1)
 for rr in ww.itertuples():
  y0,y1=np.log10([rr.naive_p,rr.blocked_p]);path=MPath([(0,y0),(.3,y0+.18*(y1-y0)),(.7,y1-.18*(y1-y0)),(1,y1)],[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4]);a.add_patch(PathPatch(path,facecolor='none',edgecolor=colors[i],lw=.65,alpha=.4));a.scatter([0,1],[y0,y1],s=11,c=colors[i],lw=.2,edgecolors='white',zorder=4)
 a.axhline(np.log10(.05),c='#717C89',ls='--',lw=.65);a.text(.48,np.log10(.05)+.10,'p = .05',fontsize=5.4,ha='center',color='#586473');a.set_xticks([0,1],['Naive','Blocked']);a.set_yticks([-3,-2,-1,0],['0.001','0.01','0.1','1']);a.set_ylabel('p-value (log scale)',fontsize=6.2)
# Denominator inset + proportional composition strip
x=153;txt(x,257,'Full q = 1 denominator (n = 160)',6.7,'bold');txt(x,262,'○ Naive   ♦ Blocked',5.7)
order=['B1-004','B4-007','B4-029','FINAL-053','FINAL-054','FINAL-055','FINAL-056','FINAL-079'];a=ax(171,267,24,16)
for j,s in enumerate(order):
 ww=w[w.signature_id=='PF4C-'+s];v0=ww.NAIVE_DETECTED.mean();v1=ww.BLOCKED_DETECTED.mean()
 if s=='FINAL-053':a.text(.5,j,'α unreachable',fontsize=5.1,ha='center',va='center');continue
 a.plot([v0,v1],[j,j],c='#ABB2BB',lw=.8);a.scatter(v0,j,s=20,facecolors='white',edgecolors='#858E9B',lw=.65,zorder=3);a.scatter(v1,j,s=16,marker='D',c='#323D4E',zorder=4)
a.set_ylim(7.5,-.5);a.set_xlim(-.12,1.12);a.set_xticks([0,.5,1]);a.set_yticks(range(8),order,fontsize=5.5);a.tick_params(axis='y',length=0);a.spines['left'].set_visible(False)
txt(151,287,'FINAL-053: 16 arrangements; p min .0625',5.0)
n=w.NAIVE_DETECTED.astype(bool);b=w.BLOCKED_DETECTED.astype(bool);cnt=[int((n&~b).sum()),int((~n&b).sum()),int((n&b).sum()),int((~n&~b).sum())];assert cnt==[51,56,4,49]
a=ax(151,290,51,2.8);start=0
for z,(count,col,hatch) in enumerate(zip(cnt,['#B36C59','#425A8A','#95759E','#DAD9D5'],['','','',''])):
 a.barh(0,count,left=start,height=.85,color=col,edgecolor='white' if z!=2 else '#6F7885',lw=.4,hatch=hatch)
 if count>4:a.text(start+count/2,0,str(count),ha='center',va='center',fontsize=5.5,color='white' if z<2 else '#333333')
 else:a.annotate('4',xy=(start+count/2,.45),xytext=(start+count/2,1.0),ha='center',fontsize=5.3,color='#785585',arrowprops=dict(arrowstyle='-',lw=.4,color='#785585'))
 start+=count
a.set_xlim(0,160);a.set_ylim(-.5,1.25);a.axis('off')
for xx,yy,label,col in [(151,293,'Naive-only','#B36C59'),(178,293,'Blocked-only','#425A8A'),(151,295,'Both','#95759E'),(178,295,'Neither','#777773')]:txt(xx,yy,label,5,color=col)
for ext in ['svg','pdf']:f.savefig(O/f'Figure5_ACT5_FINAL_upgraded.{ext}')
f.savefig(O/'Figure5_ACT5_FINAL_upgraded_600dpi.png',dpi=600);f.savefig(O/'preview.png',dpi=160)
pd.DataFrame(manifest).to_csv(O/'source_manifest.tsv',sep='\t',index=False)
print('Exported upgraded figure',flush=True)
