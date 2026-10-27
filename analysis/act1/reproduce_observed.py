from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
PROGRAMS=["MES","HYPOXIA","MATRIX","MYELOID"]
def score_matrix(Z, sets, ids, minn=1):
    """Z: genes x patients (DataFrame). Returns patients x reps array (NaN column if < minn genes)."""
    gi = {g: i for i, g in enumerate(Z.index)}
    A = Z.to_numpy()
    out = np.full((A.shape[1], len(ids)), np.nan)
    nmap = []
    for j, m in enumerate(ids):
        idx = [gi[g] for g in sets[m] if g in gi]
        nmap.append(len(idx))
        if len(idx) >= minn:
            out[:, j] = np.nanmean(A[idx, :], axis=0)
    return out, np.array(nmap)


def ecology_scores(Z, eco):
    ps = {}
    for p in PROGRAMS:
        g = [x for x in eco[p] if x in Z.index]
        ps[p] = Z.loc[g].mean(axis=0)
    a = pd.DataFrame(ps)
    a['consensus'] = ((a - a.mean()) / a.std(ddof=1)).mean(axis=1)
    return a


def rank_cols(S):
    return pd.DataFrame(S).rank(axis=0).to_numpy()


def spearman_corr_matrix(S):
    R = rank_cols(S)
    return np.corrcoef(R, rowvar=False)


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def pca_pc1(S):
    X = S - S.mean(axis=0)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    frac = s[0] ** 2 / np.sum(s ** 2)
    return float(frac), U[:, 0] * s[0], Vt[0]


def endpoints(S1, S2, eco1, eco2, cols=None):
    """S1,S2 patients x reps; drop reps NaN in either cohort; returns dict of endpoints."""
    ok = ~(np.isnan(S1).any(0) | np.isnan(S2).any(0))
    if cols is not None:
        ok = ok & cols
    A, B = S1[:, ok], S2[:, ok]
    C1, C2 = spearman_corr_matrix(A), spearman_corr_matrix(B)
    iu = np.triu_indices(A.shape[1], 1)
    out = {'rep_n': int(ok.sum()), 'structure_rho': spearman(C1[iu], C2[iu])}
    for tag, S, e in [('CGGA325', A, eco1), ('CGGA693', B, eco2)]:
        f, pc, _ = pca_pc1(S)
        out[f'PC1_var_{tag}'] = f
        out[f'absPC1_eco_{tag}'] = abs(spearman(pc, e))
    return out

def run(expression_dir, output):
 m=pd.read_csv(ROOT/"resources/signatures/Primary201_canonical.tsv",sep="\t");G=m.groupby("model_instance_id",sort=False).gene_symbol.agg(list).to_dict();ids=list(G)
 e=pd.read_csv(ROOT/"resources/signatures/ecology_programs.tsv",sep="\t");eco=e.groupby("program_id").gene_symbol_normalized.agg(list).to_dict();ECO=set().union(*map(set,eco.values()));removed={m:[g for g in G[m] if g not in ECO] for m in ids}
 result={}
 for transform in ["linear","log2"]:
  S={};SR={};E={};nm={}
  for cohort,n in [("CGGA325",73),("CGGA693",108)]:
   x=pd.read_csv(Path(expression_dir)/(cohort+"_STRICT_EXPRESSION_PREPROCESSED.tsv"),sep="\t",index_col=0);x.index=x.index.astype(str).str.upper();assert x.shape[1]==n
   if transform=="log2":x=np.log2(x+1.)
   x=x[x.std(axis=1,ddof=1)>0];z=x.sub(x.mean(axis=1),axis=0).div(x.std(axis=1,ddof=1),axis=0)
   S[cohort],_=score_matrix(z,G,ids,minn=1);SR[cohort],nm[cohort]=score_matrix(z,removed,ids,minn=1);E[cohort]=ecology_scores(z,eco).consensus.to_numpy()
  rem=np.array([len(removed[m]) for m in ids]);original=np.array([len(G[m]) for m in ids]);ok=np.ones(201,dtype=bool)
  for cohort in S:ok&=(nm[cohort]>=3)&(nm[cohort]>=.8*np.maximum(rem,1))
  masks={"N1_full":np.ones(201,dtype=bool),"N2_ruleA_all201":np.ones(201,dtype=bool),"N2_ruleC_le20pct":ok&(rem>=.8*original),"N2_ruleD_nooverlap":np.array([not(set(G[m])&ECO) for m in ids])}
  for family,mask in masks.items():
   scores=SR if family.startswith("N2_ruleA") or family.startswith("N2_ruleC") else S
   result[transform+"|"+family]=endpoints(scores["CGGA325"],scores["CGGA693"],E["CGGA325"],E["CGGA693"],cols=mask)
 ref=pd.read_csv(ROOT/"results_reference/final/SD_Fig1D_observed_and_null_summary.tsv",sep="\t")
 for r in ref.itertuples():
  actual=result[r.transform+"|"+r.family];assert actual["rep_n"]==r.rep_n
  assert abs(actual[r.endpoint]-r.observed)<1e-8,(r.family,r.endpoint,actual[r.endpoint],r.observed)
 output=Path(output);output.mkdir(parents=True,exist_ok=True);(output/"act1_observed.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2));return result
if __name__=="__main__":
 p=argparse.ArgumentParser(description="ACT1 observations from external prepared linear-FPKM cohort matrices. No null resampling.");p.add_argument("--expression-dir",required=True);p.add_argument("--output",default="outputs/act1");a=p.parse_args();run(a.expression_dir,a.output)
