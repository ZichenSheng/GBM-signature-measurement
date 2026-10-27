from pathlib import Path
import importlib.util,numpy as np,pandas as pd
R=Path(__file__).resolve().parent.parent
p=R/'GBM_R4_POST_STAGEB1_STRATEGIC_CLOSURE_AUDIT_2026-09-09/01_freeze/closure_audit.py'
s=importlib.util.spec_from_file_location('closure',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
engine,core,construct,frozen=m.ds_modules()
ids,tids,families,units,groups,scores,biology,_=engine.load_inputs();support_map,_=engine.load_support(core,ids,units,groups,scores)
seeds=pd.read_csv(m.DS/'01_preexecution_freeze/STAGEB1_DS03_SEED_BANK.tsv',sep='\t')
w=pd.read_csv(m.ROOT/'03_null_geometry/DS03_BLOCKED_VS_NAIVE_WORLD_REVERSAL.tsv',sep='\t')
rows=[]
for sid in ['PF4C-B4-007','PF4C-B4-029','PF4C-FINAL-079']:
 idx=list(ids).index(sid);support=support_map[sid];a,b=scores[idx],biology[idx];labels,codes=np.unique(groups,return_inverse=True)
 distance,weights,_=engine.geometry(core,sid,a,b,groups,support);pair=core.robust_b_distance(b,groups);graph=core.score_graph(a,groups);pools=core.padded_pools(groups,support['middle'])
 gate,prepared=engine.prepare(core,construct,frozen,a,b,groups,units,families[idx],support['middle']);_,target,assigned=construct.allocation(prepared['capacities'],1.);kvec=np.array([assigned[str(g)] for g in labels],dtype=np.int64)
 for rec in seeds[(seeds.signature_id==sid)&(seeds.q==1)&(seeds.world_type=='ALT2')].itertuples():
  null,_=core.run_blocked(rec.blocked_null_seed,999,len(a),pools,kvec,kvec,distance,weights,pair,graph,codes.astype(np.int64),True,True)
  old=w[(w.signature_id==sid)&(w.q==1)&(w.replicate_id==rec.replicate_id)].iloc[0]
  assert np.isclose(np.quantile(null,.95),old.blocked_null_q95,rtol=1e-9)
  assert np.isclose(core.randomization_p(old.T_ALT,null)[0],old.blocked_p)
  rows.extend((sid,rec.replicate_id,j+1,float(t)) for j,t in enumerate(null))
 print(sid,'verified',flush=True)
pd.DataFrame(rows,columns=['signature_id','replicate_id','null_draw','T_null']).to_csv(R/'Figure5_ACT5_FINAL/source_tables/F_recovered_blocked_null.tsv',sep='\t',index=False)
