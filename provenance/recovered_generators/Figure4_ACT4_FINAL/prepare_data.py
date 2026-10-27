from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
O=Path(__file__).resolve().parent; P=O.parent
I=P/'GBM_R4_TRUTH_ISOLATED_DETECTABILITY_BENCHMARK_STAGEA_V1_1_2026-09-08'
D=P/'GBM_R4_STAGEB1_DS03_CALIBRATED_DETECTABILITY_TRANSPORT_2026-09-08'
C=P/'GBM_R4_POST_STAGEB1_STRATEGIC_CLOSURE_AUDIT_2026-09-09'
def read(p):return pd.read_csv(p,sep='\t')
def save(d,n):d.to_csv(O/'data'/n,sep='\t',index=False)
paths={'A_Ivy':I/'06_stageA_calibration/BLOCKED_NULL_SIZE_BY_Q.tsv','A_DS03':D/'07_calibration/DS03_BLOCKED_NULL_SIZE_BY_Q.tsv','B':I/'05_stageA_alt2/STAGEA_WORLD_LEDGER.tsv','C':D/'05_alt2_worlds/DS03_ALT2_WORLD_LEDGER.tsv','D':C/'02_ds03_bifurcation/DS03_STRUCTURAL_GEOMETRY_BY_SIGNATURE_Q.tsv','E':C/'02_ds03_bifurcation/DS03_Q100_EVIDENTIARY_BIFURCATION_TABLE.tsv','F':C/'03_null_geometry/DS03_BLOCKED_VS_NAIVE_WORLD_REVERSAL.tsv','G':D/'01_preexecution_freeze/STAGEB1_DS03_SEED_BANK.tsv'}
prov=pd.DataFrame([{'panel':k,'source':str(v.relative_to(P)),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in paths.items()]);save(prov,'provenance.tsv');print(prov.to_string(index=False))
a=[]
for ctx in ['Ivy','DS03']:
 for r in read(paths['A_'+ctx]).itertuples():
  for arm in ['blocked','naive']:
   k=getattr(r,arm+'_rejections');n=r.null_world_n;ci=binomtest(k,n).proportion_ci()
   a.append(dict(context=ctx,q=r.q,arm=arm,rejections=k,n=n,rate=k/n,low=ci.low,high=ci.high))
save(pd.DataFrame(a),'A.tsv')
i=read(paths['B']);assert len(i)==4880
b=i.groupby(['signature_id','q']).blocked_detected.agg(['sum','size']).reset_index();assert set(b['size'])=={20};b['fraction']=b['sum']/b['size'];save(b,'B.tsv')
w=read(paths['C']);v=read(paths['F']);s=read(paths['D']);e=read(paths['E'])
assert len(w)==len(v)==640
for col in ['per_block_K_matched','S_invariant','E_s_invariant','legal_invariants_pass']:assert set(w[col])=={'YES'}
keys=['signature_id','q','replicate_id'];m=w.merge(v,on=keys);assert np.allclose(m.blocked_p_x,m.blocked_p_y);assert np.allclose(m.naive_p_x,m.naive_p_y);assert np.allclose(m.reference_relative_residual_distance,m.T_ALT)
counts=v.REVERSAL_CELL.value_counts().to_dict();assert counts=={0:382,1:140,10:114,11:4}
c=w.groupby(['signature_id','q']).blocked_detected.agg(['sum','size']).reset_index().merge(s,on=['signature_id','q']);assert set(c['size'])=={20};c['reachable']=c.minimum_attainable_exact_p<=.05;c['recovery']=c['sum'].where(c.reachable)
for sid in ['PF4C-B4-007','PF4C-B4-029']:
 z=c[c.signature_id==sid].sort_values('q');assert list(z.K_target)==[2,4,5,7];assert list(z.P_s_q_n)==[2,3,4,5];assert set(z.E_s_n)=={10}
assert list(c[c.signature_id=='PF4C-FINAL-053'].minimum_attainable_exact_p)==[.25,.25,.0625,.0625]
save(c,'C_D.tsv');save(e,'E.tsv');save(v,'F.tsv')
# Deterministic illustrative selection: earliest q100 replicate in each named directional cell.
sel=[]
for sid,cell in [('PF4C-B4-007',10),('PF4C-FINAL-079',1)]:
 sel.append(v[(v.signature_id==sid)&(v.q==1)&(v.REVERSAL_CELL==cell)].sort_values('replicate_id').iloc[0])
sel=pd.DataFrame(sel);save(sel,'G_examples.tsv')
sp=importlib.util.spec_from_file_location('closure',C/'01_freeze/closure_audit.py');mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);mod.verify_authority()
engine,core,construct,frozen=mod.ds_modules()
ids,tids,families,units,groups,scores,biology,_=engine.load_inputs();support_map,_=engine.load_support(core,ids,units,groups,scores);labels,codes=np.unique(groups,return_inverse=True);seeds=read(paths['G']);rr=[]
for r in sel.itertuples():
 idx=list(ids).index(r.signature_id);support=support_map[r.signature_id];ss,bb=scores[idx],biology[idx]
 distance,weights,_=engine.geometry(core,r.signature_id,ss,bb,groups,support);pair=core.robust_b_distance(bb,groups);graph=core.score_graph(ss,groups);pools=core.padded_pools(groups,support['middle']);gi=np.flatnonzero(support['middle']).astype(np.int64);go=np.flatnonzero(~support['middle']).astype(np.int64)
 gate,prepared=engine.prepare(core,construct,frozen,ss,bb,groups,units,families[idx],support['middle']);_,target,assigned=construct.allocation(prepared['capacities'],1.);kv=np.array([assigned[str(g)] for g in labels],dtype=np.int64)
 rec=seeds[(seeds.signature_id==r.signature_id)&(seeds.q==1)&(seeds.world_type=='ALT2')&(seeds.replicate_id==r.replicate_id)].iloc[0]
 perm,meta=construct.search(bb,groups,prepared,1.,'ALT2',lambda attempt:core.attempt_seed(int(rec.world_seed),attempt),gate);obs=float(core._dispersion(np.asarray(perm,dtype=np.int64),distance,weights));assert np.isclose(obs,r.T_ALT,atol=5e-11)
 blocked,_=core.run_blocked(rec.blocked_null_seed,999,len(ss),pools,kv,kv,distance,weights,pair,graph,codes.astype(np.int64),True,True);naive=core.naive_ensemble(core.seed32(rec.naive_null_seed),999,len(ss),gi,go,int(target),distance,weights)
 for arm,vals in [('blocked',blocked),('naive',naive)]:
  pv,_=core.randomization_p(obs,vals);assert np.isclose(pv,getattr(r,arm+'_p'));assert np.isclose(np.quantile(vals,.95),getattr(r,arm+'_null_q95'))
  rr.extend(dict(signature_id=r.signature_id,replicate_id=r.replicate_id,arm=arm,draw=j+1,T=float(t),T_ALT=obs,p=pv) for j,t in enumerate(vals))
 print('G_REPRODUCED',r.signature_id,r.replicate_id,flush=True)
save(pd.DataFrame(rr),'G_null_draws.tsv')
report=['DS03_TOTAL_WORLDS=640','REVERSAL_COUNTS=382,140,114,4','FINAL053_Q100_MIN_EXACT_P=0.0625','FINAL053_Q25_Q50_MIN_EXACT_P=0.25','B4007_K=2,4,5,7','B4029_K=2,4,5,7','ALL_640_LEGALITY_INVARIANTS=PASS','G_SELECTION=earliest frozen q100 replicate in requested signature and reversal direction; illustrative, not a new confirmatory test',c.pivot(index='signature_id',columns='q',values='recovery').to_string(),b.groupby('q').fraction.quantile([.25,.5,.75]).to_string(),'IVY_Q100_COMPLETE='+str(sum((b.q==1)&(b.fraction==1))),'DISCREPANCIES: old image B1-004 2,8,17,20 -> frozen 1,9,9,20; FINAL-054 0,1,3,0 -> 0,0,0,0; FINAL-055 0,0,1,0 -> 5,0,0,0; FINAL-056 1,4,8,20 -> 3,7,5,20.']
(O/'validation_report.txt').write_text('\n'.join([f'PANEL_{k}_SOURCE={v}' for k,v in paths.items()]+report));print('\n'.join(report))
