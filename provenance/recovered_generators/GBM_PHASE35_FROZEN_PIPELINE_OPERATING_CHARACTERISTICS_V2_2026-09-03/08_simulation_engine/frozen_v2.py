"""Frozen Phase 3.5 adapter, with deterministically tested numerical kernels.

Source bytes are untouched. No fastmath; the same robust scale, legal neighbors,
tie policy, patient aggregation, normalization and bootstrap are retained.
"""
from __future__ import annotations
import ast
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.runtime_amendment002'))
import numpy as np
import pandas as pd
from scipy.stats import norm,rankdata,spearmanr
from numba import njit

SOURCES=ROOT/'01_phase35_authority/sources'
P2=SOURCES/'GBM_BRP_PHASE2_201_SIGNATURE_LANDSCAPE_2026-09-02/14_code/02_run_landscape.py'
P35=SOURCES/'GBM_BRP_PHASE35_CONSTRUCT_AND_PHASE4_EVALUABILITY_CLOSURE_2026-09-02/16_code/05_phase35_analytic_empirical_null.py'
P34=SOURCES/'GBM_BRP_PHASE35_CONSTRUCT_AND_PHASE4_EVALUABILITY_CLOSURE_2026-09-02/16_code/04_phase34_reliability.py'
NS=dict(np=np,pd=pd,norm=norm,rankdata=rankdata,spearmanr=spearmanr,Path=Path,EPS=1e-12)
def extract(path,names):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),NS)
extract(P2,['anchor_vector','robust_z','dist_matrix','local_geometry','patient_summary'])
extract(P35,['normal_scores','nearest_psd','anchor_weights','simulate_null_biology'])
extract(P34,['local_geometry_dual'])

def info(S,B,G,family):
    a=NS['anchor_vector'](family,[],B)
    by=pd.DataFrame(dict(score=S,anchor=a,group=G)).dropna().groupby('group')[['score','anchor']].median().dropna()
    if len(by)<10 or by.score.nunique()<3 or by.anchor.nunique()<3:
        return np.nan
    return abs(float(spearmanr(by.score,by.anchor).statistic))

def source_categories(delta,z,se,sd):
    master=pd.DataFrame(dict(delta_excess=delta,z_excess=z,MSBD_bootstrap_SE=se,MSBD_null_sd=sd))
    fn=next(n for n in ast.parse(P35.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    body=[n for n in fn.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) in ('conditions','choices',"master['stable_excess_flag']")]
    ns=dict(np=np,master=master)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(P35),'exec'),ns)
    return ns['master'].stable_excess_flag.to_numpy()

def fit(S,B,I,family):
    zs=NS['normal_scores'](S)
    zb=np.column_stack([NS['normal_scores'](B[:,j]) for j in range(B.shape[1])])
    R=NS['nearest_psd'](np.nan_to_num(np.corrcoef(zb,rowvar=False),nan=0.))
    scale=np.sqrt(np.diag(R));R=R/np.outer(scale,scale)
    w=NS['anchor_weights'](family,B.shape[1])
    denom=float(np.sqrt(max(w@R@w,1e-8)))
    latent_r=float(2*np.sin(np.pi*np.clip(abs(I),0,.999)/6))
    beta=latent_r*(R@w)/denom
    residual=NS['nearest_psd'](R-np.outer(beta,beta))
    return dict(zs=zs,R=R,beta=beta,residual=residual)

def generate(fitted,marginals,rng):
    eps=rng.multivariate_normal(np.zeros(marginals.shape[1]),fitted['residual'],size=len(marginals))
    u=norm.cdf(fitted['zs'][:,None]*fitted['beta'][None,:]+eps)
    B=np.column_stack([np.quantile(marginals[:,j],u[:,j],method='linear') for j in range(marginals.shape[1])])
    return B,eps

@njit(cache=True)
def score_graph(S,G):
    n=len(S);order=np.argsort(S,kind='mergesort');positions=np.empty(n,np.int64)
    for i in range(n):positions[order[i]]=i
    graph=np.full((n,3),-1,np.int64)
    for i in range(n):
        left=positions[i]-1;right=positions[i]+1;k=0
        while k<3 and (left>=0 or right<n):
            dl=abs(S[order[left]]-S[i]) if left>=0 else np.inf
            dr=abs(S[order[right]]-S[i]) if right<n else np.inf
            if dl<=dr:j=order[left];left-=1
            else:j=order[right];right+=1
            if G[j]!=G[i] and np.isfinite(S[j]):graph[i,k]=j;k+=1
    return graph

@njit(cache=True)
def metric(B,G,summary,graph):
    n,d=B.shape;z=np.empty_like(B)
    for j in range(d):
        med=np.median(B[:,j]);scale=1.4826*np.median(np.abs(B[:,j]-med))
        if not np.isfinite(scale) or scale<=1e-12:
            m=np.mean(B[:,j]);scale=np.sqrt(np.sum((B[:,j]-m)**2)/(n-1))
        if not np.isfinite(scale) or scale<=1e-12:scale=1.
        z[:,j]=(B[:,j]-med)/scale
    distances=np.zeros((n,n));pairs=np.empty(n*(n-1)//2);count=0
    for i in range(n):
        for j in range(i+1,n):
            val=0.
            for k in range(d):val+=(z[i,k]-z[j,k])**2
            dist=np.sqrt(max(0.,val));distances[i,j]=dist;distances[j,i]=dist
            if G[i]!=G[j]:pairs[count]=dist;count+=1
    den=np.median(pairs[:count]) if count else np.nan
    if not np.isfinite(den) or den<=1e-12:den=1.
    values=np.full(n,np.nan)
    for i in range(n):
        val=0.;k=0
        for j in graph[i]:
            if j>=0:val+=distances[i,j]/den;k+=1
        if k:values[i]=val/k
    gs=np.unique(summary);meds=np.empty(len(gs));cnt=0
    for g in gs:
        v=values[(summary==g)&np.isfinite(values)]
        if len(v):meds[cnt]=np.median(v);cnt+=1
    return np.median(meds[:cnt]) if cnt else np.nan

def original_metric(S,B,G,summary=None):
    d,_=NS['dist_matrix'](B,G,'euclidean')
    if summary is None:
        local=NS['local_geometry'](S,G,d,3)
        return float(NS['patient_summary'](local,'local_biological_distance').median())
    local=NS['local_geometry_dual'](S,G,summary,d,3)
    return float(local.groupby('group').local_biological_distance.median().median())

def pipeline(S,B,G,family,inner_seed,bootstrap_seed,base=None):
    labels,codes=np.unique(G,return_inverse=True)
    graph=score_graph(S,codes)
    I=info(S,B,G,family)
    # Frozen Phase2 writes point fields with %.10g before Phase3.5 consumes them.
    fitted=fit(S,B,float(format(I,'.10g')),family)
    inner=np.random.default_rng(inner_seed)
    null=np.empty(250)
    for j in range(250):
        b,_=generate(fitted,B,inner)
        null[j]=metric(b,codes,codes,graph)
    boot=np.random.default_rng(bootstrap_seed)
    draws=np.empty(250);base_draws=np.empty(250) if base is not None else None
    for j in range(250):
        sampled=boot.choice(labels,size=len(labels),replace=True)
        idx=np.concatenate([np.flatnonzero(G==g) for g in sampled])
        summary=np.concatenate([np.full(np.sum(G==g),i,np.int64) for i,g in enumerate(sampled)])
        legal=codes[idx];gr=score_graph(S[idx],legal)
        draws[j]=metric(B[idx],legal,summary,gr)
        if base is not None:base_draws[j]=metric(base[idx],legal,summary,gr)
    observed=float(format(metric(B,codes,codes,graph),'.10g'))
    mean=float(np.mean(null));sd=float(np.std(null,ddof=1));se=float(np.nanstd(draws,ddof=1))
    delta=observed-mean;z=delta/sd if sd>0 else np.nan
    flag=str(source_categories([delta],[z],[se],[sd])[0])
    result=dict(I=I,MSBD=observed,null_mean=mean,null_sd=sd,bootstrap_SE=se,delta_excess=delta,z_excess=z,
        stable_excess_flag=flag,positive_detected=bool(flag=='STABLE_EXCESS_YES' and delta>0),
        negative_stable=bool(flag=='STABLE_EXCESS_YES' and delta<0),any_stable=flag=='STABLE_EXCESS_YES',
        inner_draws=null.tolist(),bootstrap_draws=draws.tolist())
    if base is not None:
        # Paired base-system characterization on THIS world, not another outer draw.
        Ib=info(S,base,G,family);fb=fit(S,base,float(format(Ib,'.10g')),family)
        # Complete-vector within-patient permutation makes the fitted generator
        # inputs identical. With matched innovations, the base null draws equal
        # the alternative null draws; do not recompute an identical sequence.
        assert abs(I-Ib)<=.05 and np.allclose(fitted['R'],fb['R'],atol=1e-12,rtol=1e-10)
        assert np.array_equal(np.sort(B,axis=0),np.sort(base,axis=0))
        bn=null
        obsb=float(format(metric(base,codes,codes,graph),'.10g'))
        result['paired_base']=dict(I=Ib,MSBD=obsb,null_mean=float(np.mean(bn)),null_sd=float(np.std(bn,ddof=1)),
            bootstrap_SE=float(np.nanstd(base_draws,ddof=1)),delta_excess=obsb-float(np.mean(bn)),
            R_max_abs_change=float(np.max(np.abs(fitted['R']-fb['R']))),
            marginal_max_abs_change=float(np.max(np.abs(np.sort(B,axis=0)-np.sort(base,axis=0)))))
    return result
