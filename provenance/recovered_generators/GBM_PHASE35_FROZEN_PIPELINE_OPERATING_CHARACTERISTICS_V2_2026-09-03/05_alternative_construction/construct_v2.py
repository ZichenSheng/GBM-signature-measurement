"""Amendment 002 exact complete-vector constructors. No diagnostic imports."""
from __future__ import annotations
import hashlib
import json
import numpy as np

def stream_seed(protocol_hash, row, stream, attempt=None):
    parts = [protocol_hash, row['context'], row['anonymous_template_id'],
             row['alternative'], f"{float(row['q']):.2f}", str(row['outer_replicate']), stream]
    if attempt is not None:
        parts.append(str(attempt))
    return int(hashlib.sha256('|'.join(parts).encode()).hexdigest(), 16)

def middle_window(S):
    order = np.argsort(S, kind='mergesort')
    n = len(S)
    # Integer expression of 0.25 <= (rank - 0.5)/n <= 0.75.
    included = (2 * (2*np.arange(n)+1) >= n) & (2 * (2*np.arange(n)+1) <= 3*n)
    out = np.zeros(n, bool)
    out[order[included]] = True
    return out

def leading_direction(cov):
    vals, vec = np.linalg.eigh(cov)
    lam = float(vals[-1])
    tied = np.abs(vals-lam) <= 1e-8*abs(lam)
    if np.sum(tied) == 1:
        v = vec[:, -1].copy()
    else:
        projector = vec[:, tied] @ vec[:, tied].T
        for j in range(len(vals)):
            v = projector[:, j].copy()
            if np.linalg.norm(v) > np.finfo(float).eps*len(vals):
                v /= np.linalg.norm(v)
                break
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return v, lam

def allocation(capacities, q):
    # Exact quarter-grid Hamilton allocation; patient IDs universally strings.
    numerator = int(round(float(q)*4))
    assert float(q) == numerator/4 and 0 <= numerator <= 4
    ids = sorted(capacities)
    total = sum(capacities.values())
    target = (numerator*total + 2)//4
    assigned = {p:(numerator*capacities[p])//4 for p in ids}
    remainder = target-sum(assigned.values())
    # Parentheses are intentional: descending integer remainders, then ID.
    priority = sorted(ids, key=lambda p: (-((numerator*capacities[p])%4), p))
    for p in priority[:remainder]:
        assigned[p] += 1
    assert sum(assigned.values()) == target
    assert all(0 <= assigned[p] <= capacities[p] for p in ids)
    return total, target, assigned

def prepare_pools(S, G, unit_ids, eps, covariance, alternative):
    M = middle_window(S)
    v, lam = leading_direction(covariance)
    u = eps @ v / np.sqrt(lam)
    # The recovered generator floors covariance eigenvalues at 1e-8. Full-rank
    # solve is the primary operation; pseudoinverse only for singular fixtures.
    if np.linalg.matrix_rank(covariance, tol=np.finfo(float).eps*len(v)*np.linalg.norm(covariance,2)) == len(v):
        solved = np.linalg.solve(covariance, eps.T).T
    else:
        solved = eps @ np.linalg.pinv(covariance, rcond=1e-12)
    norm2 = np.sum(eps*solved,axis=1)/eps.shape[1]
    r = np.sqrt(np.maximum(norm2, 0))
    pools, caps = {}, {}
    for p in sorted(set(map(str,G))):
        inside = np.flatnonzero((G==p)&M)
        outside = np.flatnonzero((G==p)&~M)
        if alternative == 'ALT1':
            plus = outside[u[outside]>0]
            minus = outside[u[outside]<0]
            pools[p] = (inside,plus,minus)
            caps[p] = min(len(inside)//2,len(plus),len(minus))
        elif alternative == 'ALT2':
            low = np.array(sorted(inside, key=lambda i: (r[i],str(unit_ids[i]))),int)[:len(inside)//2]
            high = np.array(sorted(outside, key=lambda i: (-r[i],str(unit_ids[i]))),int)[:len(outside)//2]
            pools[p] = (low,high)
            caps[p] = min(len(low),len(high))
        else:
            raise ValueError(alternative)
    return dict(M=M,u=u,r=r,pools=pools,capacities=caps)

def candidate(n, prepared, assigned, alternative, rng):
    permutation = np.arange(n)
    recipients, plus_ids, minus_ids, donor_ids = [], [], [], []
    for p in sorted(assigned):
        k = assigned[p]
        if not k:
            continue
        if alternative == 'ALT1':
            middle, plus, minus = prepared['pools'][p]
            rr = rng.permutation(rng.choice(middle,size=2*k,replace=False)).reshape(k,2)
            pp = rng.permutation(rng.choice(plus,size=k,replace=False))
            mm = rng.permutation(rng.choice(minus,size=k,replace=False))
            for j in range(k):
                for dst, src in ((rr[j,0],pp[j]),(rr[j,1],mm[j])):
                    permutation[dst],permutation[src] = src,dst
            recipients.extend(rr.ravel().tolist())
            plus_ids.extend(pp.tolist()); minus_ids.extend(mm.tolist())
            donor_ids.extend(pp.tolist()+mm.tolist())
        else:
            low, high = prepared['pools'][p]
            rr = rng.choice(low,size=k,replace=False)
            dd = rng.permutation(rng.choice(high,size=k,replace=False))
            permutation[rr] = dd; permutation[dd] = rr
            recipients.extend(rr.tolist()); donor_ids.extend(dd.tolist())
    return permutation, dict(recipients=recipients,plus_donors=plus_ids,minus_donors=minus_ids,donors=donor_ids)

def search(B, G, prepared, q, alternative, seed_for_attempt, legal_gate, rng_factory=np.random.default_rng):
    total,target,assigned = allocation(prepared['capacities'],q)
    identity = np.arange(len(B))
    base = dict(K_TOTAL=total,K_TARGET=target,allocated_units=assigned)
    if float(q)==0:
        return identity, dict(base,status='VALID_CONSTRUCTION',construction_attempts_used=0,
            construction_success_YN='NOT_APPLICABLE',accepted_candidate_attempt_id=None,
            construction_failure_reason='',recipients=[],plus_donors=[],minus_donors=[],donors=[])
    if total==0:
        return None,dict(base,status='CONSTRUCTION_FAILURE_ZERO_CAPACITY',construction_attempts_used=0,
            construction_success_YN='NO',accepted_candidate_attempt_id=None,
            construction_failure_reason='ZERO_PATIENT_CONSTRAINED_CAPACITY')
    last_reason=''
    for attempt in range(1,101):
        perm,meta = candidate(len(B),prepared,assigned,alternative,rng_factory(seed_for_attempt(attempt)))
        assert np.array_equal(np.sort(perm),identity) and np.array_equal(G[perm],G)
        assert len(meta['recipients']) == target*(2 if alternative=='ALT1' else 1)
        if alternative=='ALT1':
            assert len(meta['plus_donors'])==len(meta['minus_donors'])==target
        valid,last_reason = legal_gate(B[perm],perm)
        if valid:
            return perm,dict(base,**meta,status='VALID_CONSTRUCTION',construction_attempts_used=attempt,
                construction_success_YN='YES',accepted_candidate_attempt_id=attempt,construction_failure_reason='')
    return None,dict(base,status='CONSTRUCTION_FAILURE',construction_attempts_used=100,
        construction_success_YN='NO',accepted_candidate_attempt_id=None,construction_failure_reason=last_reason)
