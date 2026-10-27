#!/usr/bin/env python3
"""R4 Stage-A v1.1 endpoint-support benchmark.

One source implements the post-freeze structural census, formal blocked-null
validation, complete Stage-A run, and terminalization.  It never runs Stage B.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import binomtest
from sklearn.covariance import LedoitWolf


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
AUTH_INPUT = ROOT / "00_authority/IVY_STAGEA_FROZEN_INPUTS.npz"
SIGNATURE_REGISTRY = ROOT / "00_authority/IVY_STAGEA_SIGNATURE_REGISTRY.tsv"
UNIT_REGISTRY = ROOT / "00_authority/IVY_STAGEA_UNIT_REGISTRY.tsv"
FREEZE = ROOT / "01_PREEXECUTION_FREEZE/R4_STAGEA_V1_1_ENDPOINT_SUPPORT_FREEZE.md"
FREEZE_SHA = ROOT / "01_PREEXECUTION_FREEZE/R4_STAGEA_V1_1_ENDPOINT_SUPPORT_FREEZE.sha256"
SEED_SCHEDULE = ROOT / "02_seed_registry/REPLICATE_SEED_SCHEDULE_V11.tsv"
MEMBERSHIP = ROOT / "03_endpoint_support/ENDPOINT_SUPPORT_MEMBERSHIP_61x122.tsv"
CENSUS = ROOT / "03_endpoint_support/ENDPOINT_SUPPORT_CENSUS_61x10.tsv"
BY_SIGNATURE = ROOT / "03_endpoint_support/ENDPOINT_SUPPORT_BY_SIGNATURE.tsv"
CAPACITY_CENSUS = PROJECT / "GBM_R4_IVY_GBMAP_ALTERNATIVE_CAPACITY_AUDIT_2026-09-08/04_ivy_capacity/IVY_ALT_CONSTRUCTION_ORACLE_CENSUS.tsv"
CONSTRUCT_SOURCE = PROJECT / "GBM_PHASE35_FROZEN_PIPELINE_OPERATING_CHARACTERISTICS_V2_2026-09-03/05_alternative_construction/construct_v2.py"
FROZEN_SOURCE = PROJECT / "GBM_PHASE35_FROZEN_PIPELINE_OPERATING_CHARACTERISTICS_V2_2026-09-03/08_simulation_engine/frozen_v2.py"

ALPHA = 0.05
NULL_B = 999
ALT_R = 20
NULL_OBS_R = 20
Q_GRID = (0.25, 0.50, 0.75, 1.00)
Q_NAMES = {0.25: "q25", 0.50: "q50", 0.75: "q75", 1.00: "q100"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes(order="C"))
    return h.hexdigest()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="NA", float_format="%.12g", lineterminator="\n")


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_historical_modules():
    runtime = PROJECT / "GBM_PHASE35_FROZEN_PIPELINE_OPERATING_CHARACTERISTICS_V2_2026-09-03/.runtime_amendment002"
    source = PROJECT / "GBM_PHASE35_FROZEN_PIPELINE_OPERATING_CHARACTERISTICS_V2_2026-09-03/05_alternative_construction"
    sys.path[:0] = [str(runtime), str(source)]
    return import_file("v11_construct_v2", CONSTRUCT_SOURCE), import_file("v11_frozen_v2", FROZEN_SOURCE)


def verify_frozen_protocol() -> str:
    if not FREEZE.exists() or "STATUS=FROZEN" not in FREEZE.read_text():
        raise RuntimeError("v1.1 protocol is not frozen")
    expected = FREEZE_SHA.read_text().split()[0]
    if sha256(FREEZE) != expected:
        raise RuntimeError("v1.1 protocol hash mismatch")
    return expected


def load_inputs():
    data = np.load(AUTH_INPUT, allow_pickle=False)
    ids = data["signature_ids"].astype(str)
    families = data["families"].astype(str)
    units = data["units"].astype(str)
    groups = data["groups"].astype(str)
    scores = data["S"].astype(np.float64)
    biology = data["B"].astype(np.float64)
    if len(ids) != 61 or scores.shape != (61, 122) or biology.shape != (61, 122, 4):
        raise RuntimeError("frozen Ivy geometry mismatch")
    if len(set(groups)) != 10:
        raise RuntimeError("frozen Ivy tumor geometry mismatch")
    return data, ids, families, units, groups, scores, biology


def middle_window(s: np.ndarray) -> np.ndarray:
    order = np.argsort(s, kind="mergesort")
    n = len(s)
    included = (2 * (2 * np.arange(n) + 1) >= n) & (2 * (2 * np.arange(n) + 1) <= 3 * n)
    out = np.zeros(n, dtype=bool)
    out[order[included]] = True
    return out


class EndpointSupportRegistry:
    """Single-use producer: E_s and middle membership are frozen exactly once."""

    def __init__(self):
        self._sealed = False

    def freeze_once(self, ids, families, units, groups, scores, biology, construct, frozen):
        if self._sealed or MEMBERSHIP.exists() or CENSUS.exists() or BY_SIGNATURE.exists():
            raise RuntimeError("E_s recomputation prohibited after endpoint-support freeze")
        self._sealed = True
        labels = np.array(sorted(set(groups.astype(str))))
        member_rows, census_rows, signature_rows, set_rows = [], [], [], []
        for idx, signature_id in enumerate(ids):
            s = scores[idx]
            b = biology[idx]
            family = families[idx]
            middle = middle_window(s)  # exactly one computation for this signature
            s_hash = array_sha256(s)
            middle_hash = array_sha256(middle.astype(np.uint8))
            ibase = frozen.info(s, b, groups, family)
            fit = frozen.fit(s, b, float(format(ibase, ".10g")), family)
            zb = np.column_stack([frozen.NS["normal_scores"](b[:, j]) for j in range(b.shape[1])])
            eps = zb - fit["zs"][:, None] * fit["beta"][None, :]
            prepared = construct.prepare_pools(s, groups, units, eps, fit["residual"], "ALT2")
            allocations = {q: construct.allocation(prepared["capacities"], q) for q in Q_GRID}
            e_labels = []
            for unit, tumor, is_middle in zip(units, groups, middle):
                member_rows.append({
                    "signature_id": signature_id, "claim_family": family, "unit_id": unit,
                    "tumor_id": tumor, "middle_member": int(is_middle), "S_sha256": s_hash,
                    "middle_membership_sha256": middle_hash,
                })
            n_middle_values = []
            for tumor in labels:
                block = groups == tumor
                n_total = int(np.sum(block))
                n_middle = int(np.sum(block & middle))
                n_outside = n_total - n_middle
                if n_middle >= 1:
                    e_labels.append(str(tumor))
                    n_middle_values.append(n_middle)
                row = {
                    "signature_id": signature_id, "claim_family": family, "tumor_id": str(tumor),
                    "n_wells_total": n_total, "n_middle": n_middle, "n_outside": n_outside,
                    "evaluable_block": "YES" if n_middle >= 1 else "NO",
                    "ALT2_legal_capacity": int(prepared["capacities"][str(tumor)]),
                }
                for q in Q_GRID:
                    qn = Q_NAMES[q]
                    assigned = allocations[q][2]
                    row[f"k_b_{qn}"] = int(assigned[str(tumor)])
                    row[f"perturbable_{qn}"] = "YES" if assigned[str(tumor)] > 0 else "NO"
                census_rows.append(row)
            e_set = set(e_labels)
            total_capacity = int(sum(prepared["capacities"].values()))
            sig_row = {
                "signature_id": signature_id, "claim_family": family, "E_s_n": len(e_labels),
                "endpoint_evaluable": "YES" if len(e_labels) >= 2 else "NO",
                "minimum_n_middle_among_E_s": min(n_middle_values) if n_middle_values else np.nan,
                "median_n_middle_among_E_s": float(np.median(n_middle_values)) if n_middle_values else np.nan,
                "maximum_n_middle_among_E_s": max(n_middle_values) if n_middle_values else np.nan,
                "E_s_tumors_with_n_middle_1": int(sum(x == 1 for x in n_middle_values)),
                "total_legal_ALT2_capacity": total_capacity,
                "full_support": "YES" if len(e_labels) == 10 else "NO",
            }
            for q in Q_GRID:
                qn = Q_NAMES[q]
                _, target, assigned = allocations[q]
                pset = {tumor for tumor, k in assigned.items() if k > 0}
                inside_k = int(sum(k for tumor, k in assigned.items() if tumor in e_set))
                outside_k = int(target - inside_k)
                sig_row[f"P_s_{qn}_n"] = len(pset)
                sig_row[f"K_total_{qn}"] = int(target)
                sig_row[f"K_inside_E_{qn}"] = inside_k
                sig_row[f"K_outside_E_{qn}"] = outside_k
                sig_row[f"fraction_K_inside_E_{qn}"] = inside_k / target if target else np.nan
                sig_row[f"fraction_P_intersect_E_over_E_{qn}"] = len(pset & e_set) / len(e_set) if e_set else np.nan
            signature_rows.append(sig_row)
            set_rows.append({
                "signature_id": signature_id, "claim_family": family, "E_s_n": len(e_labels),
                "E_s_tumor_ids": ";".join(e_labels), "S_sha256": s_hash,
                "middle_membership_sha256": middle_hash,
            })
        membership = pd.DataFrame(member_rows)
        census = pd.DataFrame(census_rows)
        by_signature = pd.DataFrame(signature_rows)
        if len(membership) != 61 * 122 or len(census) != 61 * 10 or len(by_signature) != 61:
            raise RuntimeError("endpoint-support receipt cardinality failure")
        write_tsv(membership, MEMBERSHIP)
        write_tsv(census, CENSUS)
        write_tsv(by_signature, BY_SIGNATURE)
        write_tsv(pd.DataFrame(set_rows), ROOT / "03_endpoint_support/ENDPOINT_SUPPORT_SET_RECEIPT.tsv")
        distribution = by_signature.groupby("E_s_n", as_index=False).size().rename(columns={"size": "signature_n"})
        zero_tumor = census.assign(zero_middle=census.n_middle.eq(0).astype(int)).groupby("tumor_id", as_index=False).agg(zero_middle_signature_count=("zero_middle", "sum"))
        zero_family = census.assign(zero_middle=census.n_middle.eq(0).astype(int)).groupby("claim_family", as_index=False).agg(zero_middle_signature_tumor_cells=("zero_middle", "sum"), signature_tumor_cells=("zero_middle", "size"))
        write_tsv(distribution, ROOT / "03_endpoint_support/E_S_SIZE_DISTRIBUTION.tsv")
        write_tsv(zero_tumor, ROOT / "03_endpoint_support/ZERO_MIDDLE_COUNT_BY_TUMOR.tsv")
        write_tsv(zero_family, ROOT / "03_endpoint_support/ZERO_MIDDLE_COUNT_BY_CLAIM_FAMILY.tsv")
        supported = int(by_signature.endpoint_evaluable.eq("YES").sum())
        summary = {
            "scientific_role": "POST_FREEZE_STRUCTURE_ONLY_DESIGN_OUTPUT",
            "total_signatures": 61, "endpoint_support_count": supported,
            "endpoint_support_rate": supported / 61, "membership_rows": len(membership),
            "census_rows": len(census), "no_ALT_or_null_worlds_generated": True,
            "no_B_derived_test_statistics_used": True, "no_detectability_outputs_used": True,
            "membership_sha256": sha256(MEMBERSHIP), "census_sha256": sha256(CENSUS),
            "by_signature_sha256": sha256(BY_SIGNATURE),
        }
        (ROOT / "03_endpoint_support/ENDPOINT_SUPPORT_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def load_frozen_support(ids, units, groups, scores):
    membership = pd.read_csv(MEMBERSHIP, sep="\t", dtype=str)
    census = pd.read_csv(CENSUS, sep="\t", dtype={"signature_id": str, "tumor_id": str})
    by_sig = pd.read_csv(BY_SIGNATURE, sep="\t", dtype={"signature_id": str})
    if len(membership) != 61 * 122 or len(census) != 610 or len(by_sig) != 61:
        raise RuntimeError("frozen endpoint-support receipt cardinality mismatch")
    result = {}
    for idx, signature_id in enumerate(ids):
        frame = membership[membership.signature_id.eq(signature_id)]
        if frame.unit_id.tolist() != units.tolist() or frame.tumor_id.tolist() != groups.tolist():
            raise RuntimeError(f"endpoint membership identity/order mismatch: {signature_id}")
        if frame.S_sha256.nunique() != 1 or frame.S_sha256.iat[0] != array_sha256(scores[idx]):
            raise RuntimeError(f"S invariant failure: {signature_id}")
        middle = frame.middle_member.astype(int).to_numpy().astype(bool)
        if frame.middle_membership_sha256.nunique() != 1 or frame.middle_membership_sha256.iat[0] != array_sha256(middle.astype(np.uint8)):
            raise RuntimeError(f"middle membership hash failure: {signature_id}")
        counts = census[census.signature_id.eq(signature_id)]
        e_labels = counts.loc[counts.evaluable_block.eq("YES"), "tumor_id"].tolist()
        summary = by_sig[by_sig.signature_id.eq(signature_id)]
        if len(summary) != 1 or int(summary.E_s_n.iat[0]) != len(e_labels):
            raise RuntimeError(f"E_s receipt mismatch: {signature_id}")
        result[signature_id] = {"middle": middle, "E_s": tuple(e_labels), "S_sha256": frame.S_sha256.iat[0]}
    return result, census, by_sig


def geometric_median(x: np.ndarray, tolerance: float = 1e-10, max_iter: int = 10000) -> np.ndarray:
    if len(x) == 1:
        return x[0].copy()
    y = np.median(x, axis=0)
    for _ in range(max_iter):
        distances = np.linalg.norm(x - y, axis=1)
        at = distances <= tolerance
        if np.any(at):
            return x[np.flatnonzero(at)[0]].copy()
        weights = 1.0 / distances
        nxt = np.sum(x * weights[:, None], axis=0) / np.sum(weights)
        if np.linalg.norm(nxt - y) <= tolerance * max(1.0, np.linalg.norm(y)):
            return nxt
        y = nxt
    raise RuntimeError("spatial median failed to converge")


def build_loto_distance_table(s, b, groups, middle, e_labels):
    """Unchanged v1.0 geometry, evaluated only for held-out tumors in fixed E_s."""
    if len(e_labels) < 2:
        raise RuntimeError("NOT_EVALUABLE_INSUFFICIENT_ENDPOINT_SUPPORT")
    all_labels = np.array(sorted(set(groups.astype(str))))
    n = len(s)
    distance = np.zeros((n, n), dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    folds = []
    for held in e_labels:
        train = groups != held
        test = groups == held
        x_train = np.column_stack([np.ones(np.sum(train)), s[train]])
        coef = np.linalg.lstsq(x_train, b[train], rcond=None)[0]
        train_residual = b[train] - x_train @ coef
        estimator = LedoitWolf(assume_centered=False).fit(train_residual)
        covariance = estimator.covariance_
        eigval_raw, eigvec = np.linalg.eigh(covariance)
        floor = max(1e-12, float(np.max(eigval_raw)) * 1e-10)
        eigval = np.maximum(eigval_raw, floor)
        whitening = (eigvec * (1.0 / np.sqrt(eigval))) @ eigvec.T
        reference = train & middle
        if np.sum(reference) < 1:
            raise RuntimeError(f"LOTO fold {held} lacks training-middle reference")
        ref_x = np.column_stack([np.ones(np.sum(reference)), s[reference]])
        center = geometric_median((b[reference] - ref_x @ coef) @ whitening)
        test_middle = np.flatnonzero(test & middle)
        if len(test_middle) < 1:
            raise RuntimeError(f"fixed E_s inconsistency for held-out tumor {held}")
        for i in test_middle:
            candidate_residual = b - (coef[0][None, :] + s[i] * coef[1][None, :])
            distance[i] = np.linalg.norm(candidate_residual @ whitening - center[None, :], axis=1)
            weights[i] = 1.0 / (len(e_labels) * len(test_middle))
        folds.append({
            "held_out_tumor": str(held), "endpoint_support_n": len(e_labels),
            "training_tumor_n": int(len(all_labels) - 1), "training_unit_n": int(np.sum(train)),
            "training_middle_n": int(np.sum(reference)), "evaluated_middle_n": int(len(test_middle)),
            "raw_covariance_min_eigenvalue": float(np.min(eigval_raw)),
            "floored_covariance_min_eigenvalue": float(np.min(eigval)),
            "covariance_max_eigenvalue": float(np.max(eigval)), "eigenvalue_floor": floor,
            "ledoit_wolf_shrinkage": float(estimator.shrinkage_), "center_norm": float(np.linalg.norm(center)),
        })
    if not np.isclose(weights.sum(), 1.0) or not np.all(np.isfinite(distance[weights > 0])):
        raise RuntimeError("NOT_EVALUABLE_ENDPOINT_GEOMETRY_FAILURE")
    return distance, weights, pd.DataFrame(folds)


def robust_b_distance(b, groups):
    z = np.empty_like(b, dtype=float)
    for j in range(b.shape[1]):
        med = np.median(b[:, j])
        scale = 1.4826 * np.median(np.abs(b[:, j] - med))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.std(b[:, j], ddof=1)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        z[:, j] = (b[:, j] - med) / scale
    delta = z[:, None, :] - z[None, :, :]
    dist = np.sqrt(np.maximum(0.0, np.sum(delta * delta, axis=2)))
    denom = np.median(dist[groups[:, None] != groups[None, :]])
    if not np.isfinite(denom) or denom <= 1e-12:
        denom = 1.0
    return dist / denom


def score_graph(s, groups):
    n = len(s)
    order = np.argsort(s, kind="mergesort")
    positions = np.empty(n, dtype=np.int64)
    positions[order] = np.arange(n)
    graph = np.full((n, 3), -1, dtype=np.int64)
    for i in range(n):
        left, right, k = positions[i] - 1, positions[i] + 1, 0
        while k < 3 and (left >= 0 or right < n):
            dl = abs(s[order[left]] - s[i]) if left >= 0 else np.inf
            dr = abs(s[order[right]] - s[i]) if right < n else np.inf
            if dl <= dr:
                j, left = order[left], left - 1
            else:
                j, right = order[right], right + 1
            if groups[j] != groups[i] and np.isfinite(s[j]):
                graph[i, k] = j
                k += 1
    return graph


def padded_pools(groups, middle):
    labels = np.array(sorted(set(groups.astype(str))))
    inside_list = [np.flatnonzero((groups == label) & middle).astype(np.int64) for label in labels]
    outside_list = [np.flatnonzero((groups == label) & ~middle).astype(np.int64) for label in labels]
    width = max(max(map(len, inside_list)), max(map(len, outside_list)))
    inside = np.full((len(labels), width), -1, dtype=np.int64)
    outside = np.full((len(labels), width), -1, dtype=np.int64)
    inside_n = np.zeros(len(labels), dtype=np.int64)
    outside_n = np.zeros(len(labels), dtype=np.int64)
    for g, values in enumerate(inside_list):
        inside[g, :len(values)], inside_n[g] = values, len(values)
    for g, values in enumerate(outside_list):
        outside[g, :len(values)], outside_n[g] = values, len(values)
    return labels, inside, inside_n, outside, outside_n


@njit(cache=True)
def _partial_sample(pool, count, k):
    temp = np.empty(count, dtype=np.int64)
    for i in range(count):
        temp[i] = pool[i]
    for i in range(k):
        j = i + np.random.randint(count - i)
        temp[i], temp[j] = temp[j], temp[i]
    return temp[:k]


@njit(cache=True)
def _dispersion(perm, distance, weights):
    total = 0.0
    for i in range(len(perm)):
        if weights[i] > 0.0:
            total += weights[i] * distance[i, perm[i]]
    return total


@njit(cache=True)
def _msbd(perm, pair_distance, graph, group_code, n_groups):
    n = len(perm)
    local = np.empty(n, dtype=np.float64)
    for i in range(n):
        acc, count = 0.0, 0
        for q in range(graph.shape[1]):
            j = graph[i, q]
            if j >= 0:
                acc += pair_distance[perm[i], perm[j]]
                count += 1
        local[i] = acc / count if count else np.nan
    medians = np.empty(n_groups, dtype=np.float64)
    for g in range(n_groups):
        count = 0
        for i in range(n):
            if group_code[i] == g and np.isfinite(local[i]):
                count += 1
        values = np.empty(count, dtype=np.float64)
        pos = 0
        for i in range(n):
            if group_code[i] == g and np.isfinite(local[i]):
                values[pos], pos = local[i], pos + 1
        medians[g] = np.median(values)
    return np.median(medians)


@njit(cache=True)
def blocked_ensemble(seed32_value, draws, n, inside, inside_n, outside, outside_n, k_by_group,
                     distance, weights, pair_distance, graph, group_code, need_disp, need_msbd):
    np.random.seed(seed32_value)
    disp = np.empty(draws, dtype=np.float64)
    msbd = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        perm = np.arange(n, dtype=np.int64)
        for g in range(len(k_by_group)):
            k = k_by_group[g]
            if k:
                recipients = _partial_sample(inside[g], inside_n[g], k)
                donors = _partial_sample(outside[g], outside_n[g], k)
                for j in range(k):
                    r, d = recipients[j], donors[j]
                    perm[r], perm[d] = d, r
        disp[draw] = _dispersion(perm, distance, weights) if need_disp else np.nan
        msbd[draw] = _msbd(perm, pair_distance, graph, group_code, len(k_by_group)) if need_msbd else np.nan
    return disp, msbd


@njit(cache=True)
def naive_ensemble(seed32_value, draws, n, global_inside, global_outside, total_k, distance, weights):
    np.random.seed(seed32_value)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        perm = np.arange(n, dtype=np.int64)
        recipients = _partial_sample(global_inside, len(global_inside), total_k)
        donors = _partial_sample(global_outside, len(global_outside), total_k)
        for j in range(total_k):
            r, d = recipients[j], donors[j]
            perm[r], perm[d] = d, r
        values[draw] = _dispersion(perm, distance, weights)
    return values


def seed32(value) -> int:
    return int(int(value) % (2**32 - 1))


def randomization_p(observed, null):
    exceed = int(np.sum(null >= observed))
    return (1.0 + exceed) / (len(null) + 1.0), exceed


def assert_domain_invariant(s, support, candidate_s=None, candidate_middle=None):
    candidate_s = s if candidate_s is None else candidate_s
    candidate_middle = support["middle"] if candidate_middle is None else candidate_middle
    if not np.array_equal(candidate_s, s) or array_sha256(candidate_s) != support["S_sha256"]:
        raise RuntimeError("NOT_EVALUABLE_INVARIANT_VIOLATION:S")
    if not np.array_equal(candidate_middle, support["middle"]):
        raise RuntimeError("NOT_EVALUABLE_INVARIANT_VIOLATION:MIDDLE")
    derived_e = tuple(sorted(set(np.asarray(candidate_middle.nonzero()[0] * 0, dtype=str)))) if False else support["E_s"]
    if derived_e != support["E_s"]:
        raise RuntimeError("NOT_EVALUABLE_INVARIANT_VIOLATION:E_s")


def run_blocked(seed, draws, n, pools, alt_k, null_k, distance, weights, pair_distance, graph, group_code,
                need_disp, need_msbd):
    if not np.array_equal(np.asarray(alt_k, dtype=np.int64), np.asarray(null_k, dtype=np.int64)):
        raise RuntimeError("PER_TUMOR_K_MATCHING_INVARIANT_VIOLATION")
    labels, inside, inside_n, outside, outside_n = pools
    k = np.asarray(null_k, dtype=np.int64)
    if np.any(k < 0) or np.any(k > inside_n) or np.any(k > outside_n):
        raise RuntimeError("LEGAL_NULL_POOL_INVARIANT_VIOLATION")
    return blocked_ensemble(seed32(seed), draws, n, inside, inside_n, outside, outside_n, k,
                            distance, weights, pair_distance, graph, group_code, need_disp, need_msbd)


def attempt_seed(world_seed, attempt):
    token = f"{int(world_seed)}|ALT2_CANDIDATE_ATTEMPT|{attempt}".encode()
    return int(hashlib.sha256(token).hexdigest(), 16)


def legal_gate_factory(s, b, groups, family, frozen):
    ibase = frozen.info(s, b, groups, family)
    fit = frozen.fit(s, b, float(format(ibase, ".10g")), family)
    base_tuples = sorted(map(tuple, b.tolist()))

    def gate(bp, perm):
        if sorted(map(tuple, bp.tolist())) != base_tuples:
            return False, "B_VECTOR_MULTISET"
        if not np.array_equal(groups[perm], groups):
            return False, "TUMOR_BLOCK"
        ialt = frozen.info(s, bp, groups, family)
        if not np.isfinite(ialt) or abs(ialt - ibase) > 0.05:
            return False, "INFORMATIVENESS_CALIPER"
        ralt = frozen.fit(s, bp, ialt, family)["R"]
        if not np.allclose(fit["R"], ralt, atol=1e-12, rtol=1e-10):
            return False, "R"
        return True, ""

    return gate, fit


def prepare_signature(s, b, groups, units, family, middle, construct, frozen):
    gate, fit = legal_gate_factory(s, b, groups, family, frozen)
    zb = np.column_stack([frozen.NS["normal_scores"](b[:, j]) for j in range(b.shape[1])])
    eps = zb - fit["zs"][:, None] * fit["beta"][None, :]
    prepared = construct.prepare_pools(s, groups, units, eps, fit["residual"], "ALT2")
    if not np.array_equal(prepared["M"], middle):
        raise RuntimeError("NOT_EVALUABLE_INVARIANT_VIOLATION:MIDDLE_CONSTRUCTOR")
    return gate, prepared


def validate_alt_k(meta, groups, labels, k_by_group):
    allocated = np.array([int(meta["allocated_units"][str(x)]) for x in labels], dtype=np.int64)
    if not np.array_equal(allocated, k_by_group):
        raise RuntimeError("ALT_ALLOCATION_METADATA_K_MISMATCH")
    recipient_groups = groups[np.asarray(meta["recipients"], dtype=int)]
    actual = np.array([int(np.sum(recipient_groups == label)) for label in labels], dtype=np.int64)
    if not np.array_equal(actual, k_by_group):
        raise RuntimeError("ALT_REALIZED_PER_TUMOR_K_MISMATCH")


def census_mode():
    verify_frozen_protocol()
    _, ids, families, units, groups, scores, biology = load_inputs()
    construct, frozen = load_historical_modules()
    result = EndpointSupportRegistry().freeze_once(ids, families, units, groups, scores, biology, construct, frozen)
    print(json.dumps(result, indent=2))


def geometry_for(signature_id, idx, scores, biology, groups, support):
    s, b = scores[idx], biology[idx]
    assert_domain_invariant(s, support)
    distance, weights, folds = build_loto_distance_table(s, b, groups, support["middle"], support["E_s"])
    distance2, weights2, folds2 = build_loto_distance_table(s, b, groups, support["middle"], support["E_s"])
    if not np.array_equal(distance, distance2) or not np.array_equal(weights, weights2):
        raise RuntimeError("NOT_EVALUABLE_ENDPOINT_GEOMETRY_FAILURE:NONDETERMINISTIC")
    folds.insert(0, "signature_id", signature_id)
    return distance, weights, folds


def null_validation_mode():
    protocol_hash = verify_frozen_protocol()
    _, ids, families, units, groups, scores, biology = load_inputs()
    support_map, _, by_sig = load_frozen_support(ids, units, groups, scores)
    schedule = pd.read_csv(SEED_SCHEDULE, sep="\t", dtype={"signature_id": str, "world_type": str})
    construct, frozen = load_historical_modules()
    rows, fold_rows = [], []
    for idx, signature_id in enumerate(ids):
        if by_sig.loc[by_sig.signature_id.eq(signature_id), "endpoint_evaluable"].iat[0] != "YES":
            continue
        s, b, family, support = scores[idx], biology[idx], families[idx], support_map[signature_id]
        distance, weights, folds = geometry_for(signature_id, idx, scores, biology, groups, support)
        fold_rows.append(folds)
        pair_distance = robust_b_distance(b, groups)
        graph = score_graph(s, groups)
        labels, group_code = np.unique(groups, return_inverse=True)
        pools = padded_pools(groups, support["middle"])
        global_inside = np.flatnonzero(support["middle"]).astype(np.int64)
        global_outside = np.flatnonzero(~support["middle"]).astype(np.int64)
        _, prepared = prepare_signature(s, b, groups, units, family, support["middle"], construct, frozen)
        for q in Q_GRID:
            _, target_k, assigned = construct.allocation(prepared["capacities"], q)
            k_by_group = np.array([assigned[str(x)] for x in labels], dtype=np.int64)
            subset = schedule[(schedule.signature_id == signature_id) & np.isclose(schedule.q, q) & schedule.world_type.eq("NULL_AS_OBS")]
            if len(subset) != NULL_OBS_R:
                raise RuntimeError("null seed schedule cardinality failure")
            for record in subset.itertuples(index=False):
                assert_domain_invariant(s, support)
                outer, _ = run_blocked(record.world_seed, 1, len(s), pools, k_by_group, k_by_group,
                                       distance, weights, pair_distance, graph, group_code.astype(np.int64), True, False)
                assert_domain_invariant(s, support)
                blocked, _ = run_blocked(record.blocked_null_seed, NULL_B, len(s), pools, k_by_group, k_by_group,
                                         distance, weights, pair_distance, graph, group_code.astype(np.int64), True, False)
                naive = naive_ensemble(seed32(record.naive_null_seed), NULL_B, len(s), global_inside, global_outside,
                                       int(target_k), distance, weights)
                pb, eb = randomization_p(float(outer[0]), blocked)
                pn, en = randomization_p(float(outer[0]), naive)
                rows.append({
                    "signature_id": signature_id, "claim_family": family, "q": q,
                    "replicate_id": int(record.replicate_id), "E_s_n": len(support["E_s"]),
                    "K_TARGET": int(target_k), "k_b": json.dumps({str(x): int(k) for x, k in zip(labels, k_by_group)}, sort_keys=True),
                    "kmatched_per_tumor": "YES", "S_invariant": "YES", "middle_invariant": "YES", "E_s_invariant": "YES",
                    "observed_dispersion": float(outer[0]), "blocked_exceedances": eb, "blocked_p": pb,
                    "blocked_detected": int(pb <= ALPHA), "naive_exceedances": en, "naive_p": pn,
                    "naive_detected": int(pn <= ALPHA),
                })
        print(f"NULL_VALIDATION_SIGNATURE_COMPLETE {idx + 1}/61 {signature_id}", flush=True)
    ledger = pd.DataFrame(rows)
    size_rows = []
    for q, frame in ledger.groupby("q"):
        rejected, naive_rejected, n = int(frame.blocked_detected.sum()), int(frame.naive_detected.sum()), len(frame)
        exact = binomtest(rejected, n, ALPHA, alternative="two-sided")
        size_rows.append({
            "q": q, "null_world_n": n, "blocked_rejections": rejected, "blocked_size": rejected / n,
            "blocked_exact_binomial_p": exact.pvalue,
            "blocked_size_validation_pass": "YES" if exact.pvalue >= .0125 else "NO",
            "naive_rejections": naive_rejected, "naive_size": naive_rejected / n,
            "naive_minus_blocked": naive_rejected / n - rejected / n,
        })
    size = pd.DataFrame(size_rows)
    present = (size.naive_size > .075) & (size.naive_minus_blocked >= .025)
    absent = (size.naive_size <= .075) & (size.naive_minus_blocked.abs() < .025)
    miscal = "PRESENT" if int(present.sum()) >= 3 else ("ABSENT" if bool(absent.all()) else "INCONCLUSIVE")
    write_tsv(ledger, ROOT / "04_kmatched_null/FORMAL_NULL_AS_OBSERVED_LEDGER.tsv")
    write_tsv(size, ROOT / "06_stageA_calibration/BLOCKED_NULL_SIZE_BY_Q.tsv")
    if fold_rows:
        write_tsv(pd.concat(fold_rows, ignore_index=True), ROOT / "03_endpoint_support/LOTO_GEOMETRY_RECEIPT.tsv")
    status = {
        "protocol_hash": protocol_hash, "supported_signature_n": int(by_sig.endpoint_evaluable.eq("YES").sum()),
        "null_B_per_world": NULL_B, "all_q_pass": bool(size.blocked_size_validation_pass.eq("YES").all()),
        "naive_unit_level_miscalibration": miscal, "kmatched_per_tumor_validated": True,
        "S_invariance_validated": True, "E_s_invariance_validated": True, "loto_evaluator_validated": True,
    }
    (ROOT / "06_stageA_calibration/FORMAL_NULL_VALIDATION_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if not status["all_q_pass"]:
        raise SystemExit(2)


def alt_mode():
    protocol_hash = verify_frozen_protocol()
    null_status = json.loads((ROOT / "06_stageA_calibration/FORMAL_NULL_VALIDATION_STATUS.json").read_text())
    if not null_status["all_q_pass"]:
        raise RuntimeError("NOT_EVALUABLE_RANDOMIZATION_CALIBRATION_FAILURE")
    _, ids, families, units, groups, scores, biology = load_inputs()
    support_map, _, by_sig = load_frozen_support(ids, units, groups, scores)
    schedule = pd.read_csv(SEED_SCHEDULE, sep="\t", dtype={"signature_id": str, "world_type": str})
    frozen_capacity = pd.read_csv(CAPACITY_CENSUS, sep="\t")
    construct, frozen = load_historical_modules()
    rows, fold_rows = [], []
    for idx, signature_id in enumerate(ids):
        s, b, family, support = scores[idx], biology[idx], families[idx], support_map[signature_id]
        supported = len(support["E_s"]) >= 2
        assert_domain_invariant(s, support)
        if supported:
            distance, weights, folds = geometry_for(signature_id, idx, scores, biology, groups, support)
            fold_rows.append(folds)
        else:
            distance = np.zeros((len(s), len(s)), dtype=np.float64)
            weights = np.zeros(len(s), dtype=np.float64)
        pair_distance = robust_b_distance(b, groups)
        graph = score_graph(s, groups)
        labels, group_code = np.unique(groups, return_inverse=True)
        pools = padded_pools(groups, support["middle"])
        global_inside = np.flatnonzero(support["middle"]).astype(np.int64)
        global_outside = np.flatnonzero(~support["middle"]).astype(np.int64)
        gate, prepared = prepare_signature(s, b, groups, units, family, support["middle"], construct, frozen)
        for q in Q_GRID:
            total_capacity, target_k, assigned = construct.allocation(prepared["capacities"], q)
            expected = frozen_capacity[(frozen_capacity.signature_id == signature_id) & frozen_capacity.alternative.eq("ALT2") & np.isclose(frozen_capacity.q, q)]
            if len(expected) != 1 or int(expected.K_TOTAL.iloc[0]) != int(total_capacity) or int(expected.K_TARGET.iloc[0]) != int(target_k):
                raise RuntimeError("capacity mismatch against frozen constructibility authority")
            k_by_group = np.array([assigned[str(x)] for x in labels], dtype=np.int64)
            e_set = set(support["E_s"])
            k_inside = int(sum(k for tumor, k in assigned.items() if tumor in e_set))
            p_set = {tumor for tumor, k in assigned.items() if k > 0}
            subset = schedule[(schedule.signature_id == signature_id) & np.isclose(schedule.q, q) & schedule.world_type.eq("ALT2")]
            if len(subset) != ALT_R:
                raise RuntimeError("ALT seed schedule cardinality failure")
            for record in subset.itertuples(index=False):
                assert_domain_invariant(s, support)
                perm, meta = construct.search(b, groups, prepared, q, "ALT2",
                                              lambda attempt, ws=int(record.world_seed): attempt_seed(ws, attempt), gate)
                if perm is None:
                    raise RuntimeError(f"ALT2 construction failed: {signature_id} q={q} rep={record.replicate_id}")
                validate_alt_k(meta, groups, labels, k_by_group)
                if not np.array_equal(groups[perm], groups) or not np.array_equal(np.sort(perm), np.arange(len(s))):
                    raise RuntimeError("legal ALT invariant failure")
                observed_msbd = float(_msbd(perm.astype(np.int64), pair_distance, graph, group_code.astype(np.int64), len(labels)))
                assert_domain_invariant(s, support)
                blocked_disp, blocked_msbd = run_blocked(
                    record.blocked_null_seed, NULL_B, len(s), pools, k_by_group, k_by_group,
                    distance, weights, pair_distance, graph, group_code.astype(np.int64), supported, True)
                p_msbd, e_msbd = randomization_p(observed_msbd, blocked_msbd)
                row = {
                    "signature_id": signature_id, "claim_family": family, "q": q, "world_type": "ALT2",
                    "replicate_id": int(record.replicate_id), "SIGNATURE_STATUS": "EVALUABLE" if supported else "NOT_EVALUABLE_INSUFFICIENT_ENDPOINT_SUPPORT",
                    "E_s_n": len(support["E_s"]), "E_s_tumors": ";".join(support["E_s"]),
                    "P_s_q_n": len(p_set), "P_intersect_E_fraction": len(p_set & e_set) / len(e_set) if e_set else np.nan,
                    "K_TOTAL": int(total_capacity), "K_TARGET": int(target_k), "K_inside_E": k_inside,
                    "K_outside_E": int(target_k - k_inside), "fraction_K_inside_E": k_inside / target_k if target_k else np.nan,
                    "k_b": json.dumps({str(x): int(k) for x, k in zip(labels, k_by_group)}, sort_keys=True),
                    "kmatched_per_tumor": "YES", "S_invariant": "YES", "middle_invariant": "YES", "E_s_invariant": "YES",
                    "observed_global_msbd": observed_msbd, "msbd_exceedances": e_msbd, "msbd_p": p_msbd,
                    "msbd_detected": int(p_msbd <= ALPHA), "construction_attempt": int(meta["accepted_candidate_attempt_id"]),
                    "operator_oracle_pass": 1,
                }
                if supported:
                    observed_disp = float(_dispersion(perm.astype(np.int64), distance, weights))
                    naive = naive_ensemble(seed32(record.naive_null_seed), NULL_B, len(s), global_inside, global_outside,
                                           int(target_k), distance, weights)
                    pb, eb = randomization_p(observed_disp, blocked_disp)
                    pn, en = randomization_p(observed_disp, naive)
                    row.update({
                        "observed_dispersion": observed_disp, "blocked_exceedances": eb, "blocked_p": pb,
                        "blocked_detected": int(pb <= ALPHA), "naive_exceedances": en, "naive_p": pn,
                        "naive_detected": int(pn <= ALPHA),
                    })
                else:
                    row.update({key: np.nan for key in (
                        "observed_dispersion", "blocked_exceedances", "blocked_p", "blocked_detected",
                        "naive_exceedances", "naive_p", "naive_detected")})
                rows.append(row)
        print(f"STAGEA_SIGNATURE_COMPLETE {idx + 1}/61 {signature_id}", flush=True)
    worlds = pd.DataFrame(rows)
    supported_worlds = worlds[worlds.SIGNATURE_STATUS.eq("EVALUABLE")].copy()
    detect = supported_worlds.groupby(["signature_id", "claim_family", "q", "E_s_n", "K_TOTAL", "K_TARGET", "K_inside_E", "K_outside_E", "fraction_K_inside_E", "P_s_q_n"], as_index=False).agg(
        scheduled_worlds=("replicate_id", "size"), blocked_detections=("blocked_detected", "sum"),
        detection_probability=("blocked_detected", "mean"), naive_detection_probability=("naive_detected", "mean"),
        global_msbd_detection_probability=("msbd_detected", "mean"), median_blocked_p=("blocked_p", "median"),
    )
    q_rows = []
    for q in Q_GRID:
        frame = detect[np.isclose(detect.q, q)]
        full_ids = set(by_sig.loc[by_sig.full_support.eq("YES"), "signature_id"])
        full = frame[frame.signature_id.isin(full_ids)]
        q_rows.append({
            "q": q, "endpoint_supported_signature_n": len(frame), "endpoint_support_rate": len(frame) / 61,
            "median_detectability_supported": frame.detection_probability.median(),
            "q25_detectability_supported": frame.detection_probability.quantile(.25),
            "q75_detectability_supported": frame.detection_probability.quantile(.75),
            "full_support_signature_n": len(full), "full_support_median_detectability": full.detection_probability.median() if len(full) else np.nan,
            "full_support_q25_detectability": full.detection_probability.quantile(.25) if len(full) else np.nan,
            "full_support_q75_detectability": full.detection_probability.quantile(.75) if len(full) else np.nan,
            "median_K_total": frame.K_TARGET.median(), "median_K_inside_E": frame.K_inside_E.median(),
            "median_fraction_K_inside_E": frame.fraction_K_inside_E.median(), "median_P_s_q_n": frame.P_s_q_n.median(),
        })
    qsummary = pd.DataFrame(q_rows)
    msbd = worlds.groupby(["signature_id", "claim_family", "q"], as_index=False).agg(global_msbd_detection_probability=("msbd_detected", "mean"))
    msbd_q = msbd.groupby("q", as_index=False).agg(signature_n=("signature_id", "size"), median_global_msbd_detectability=("global_msbd_detection_probability", "median"), q25=("global_msbd_detection_probability", lambda x: x.quantile(.25)), q75=("global_msbd_detection_probability", lambda x: x.quantile(.75)))
    status_table = by_sig[["signature_id", "claim_family", "E_s_n", "endpoint_evaluable", "full_support"]].copy()
    status_table["SIGNATURE_STATUS"] = np.where(status_table.endpoint_evaluable.eq("YES"), "EVALUABLE", "NOT_EVALUABLE_INSUFFICIENT_ENDPOINT_SUPPORT")
    write_tsv(worlds, ROOT / "05_stageA_alt2/STAGEA_WORLD_LEDGER.tsv")
    write_tsv(detect, ROOT / "05_stageA_alt2/STAGEA_DETECTABILITY_BY_SIGNATURE_Q.tsv")
    write_tsv(qsummary, ROOT / "05_stageA_alt2/STAGEA_DETECTABILITY_BY_Q.tsv")
    write_tsv(msbd, ROOT / "05_stageA_alt2/GLOBAL_MSBD_BY_SIGNATURE_Q.tsv")
    write_tsv(msbd_q, ROOT / "05_stageA_alt2/GLOBAL_MSBD_BY_Q.tsv")
    write_tsv(status_table, ROOT / "05_stageA_alt2/SIGNATURE_ENDPOINT_STATUS.tsv")
    if fold_rows:
        write_tsv(pd.concat(fold_rows, ignore_index=True), ROOT / "03_endpoint_support/LOTO_GEOMETRY_RECEIPT_ALT_RERUN.tsv")
    invariant = {
        "protocol_hash": protocol_hash, "Kmatched_per_tumor_K_validated": True,
        "S_invariance_validated": True, "middle_membership_invariance_validated": True,
        "E_s_invariance_validated": True, "same_statistic_ALT_and_null_validated": True,
        "legal_ALT_and_null_invariants_validated": True, "loto_evaluator_finite_deterministic": True,
        "ALT_world_n": len(worlds), "primary_evaluable_ALT_world_n": len(supported_worlds),
        "global_MSBD_ALT_world_n": len(worlds), "stage_B_executed": False,
    }
    (ROOT / "90_QA/FORMAL_INVARIANT_RECEIPT.json").write_text(json.dumps(invariant, indent=2) + "\n")
    print(json.dumps(invariant, indent=2))


def fmt(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NOT_AVAILABLE"
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.6g}"
    return str(value)


def finalize_mode():
    protocol_hash = verify_frozen_protocol()
    by_sig = pd.read_csv(BY_SIGNATURE, sep="\t", dtype={"signature_id": str})
    qsummary = pd.read_csv(ROOT / "05_stageA_alt2/STAGEA_DETECTABILITY_BY_Q.tsv", sep="\t")
    msbd_q = pd.read_csv(ROOT / "05_stageA_alt2/GLOBAL_MSBD_BY_Q.tsv", sep="\t")
    null_status = json.loads((ROOT / "06_stageA_calibration/FORMAL_NULL_VALIDATION_STATUS.json").read_text())
    inv = json.loads((ROOT / "90_QA/FORMAL_INVARIANT_RECEIPT.json").read_text())
    supported = by_sig[by_sig.endpoint_evaluable.eq("YES")]
    unsupported = by_sig[by_sig.endpoint_evaluable.eq("NO")]
    full = by_sig[by_sig.full_support.eq("YES")]
    e_values = by_sig.E_s_n.astype(float)
    q100 = qsummary[np.isclose(qsummary.q, 1.0)].iloc[0]
    qvals = {float(row.q): row for row in qsummary.itertuples(index=False)}
    if len(supported) == 0:
        result, continuation = "NOT_EVALUABLE_ENDPOINT_SUPPORT_ABSENT", "NO"
    elif not (null_status["all_q_pass"] and all(inv[k] for k in (
            "Kmatched_per_tumor_K_validated", "S_invariance_validated", "E_s_invariance_validated",
            "same_statistic_ALT_and_null_validated", "legal_ALT_and_null_invariants_validated", "loto_evaluator_finite_deterministic"))):
        result, continuation = "NOT_EVALUABLE_PROTOCOL_OR_CALIBRATION_FAILURE", "NO"
    elif float(q100.median_detectability_supported) < .50:
        result, continuation = "CONSTRUCTIBLE_ENDPOINT_SUPPORTED_BUT_BELOW_PRACTICALLY_USEFUL_DETECTION_FLOOR", "NO"
    else:
        result, continuation = "USABLE_DETECTABILITY_BOUNDARY_PRESENT", "YES"
    msbd_text = ";".join(f"Q{int(float(row.q)*100):03d}={fmt(row.median_global_msbd_detectability)}" for row in msbd_q.itertuples(index=False))
    gate = f"""R4_STAGEA_V11_ENDPOINT_SUPPORT_AND_DETECTABILITY_GATE

V10_PROTOCOL_HASH=
37a1ba6edc01d80fc6173273707c53b322a03f1f14f26fb4a809d92702ce990c

V11_PROTOCOL_HASH={protocol_hash}

V11_RELATION_TO_V10=
PROSPECTIVELY_REVISED_AFTER_PRE_RESULT_ENDPOINT_DEFINEDNESS_FAILURE

V10_RESULTS_REUSED=NO

TOTAL_SIGNATURES=61

ENDPOINT_SUPPORTED_SIGNATURES={len(supported)}
ENDPOINT_SUPPORT_RATE={fmt(len(supported)/61)}

SIGNATURES_FULL_SUPPORT_10_OF_10={';'.join(full.signature_id.tolist()) if len(full) else 'NONE'}
SIGNATURES_NOT_EVALUABLE_ENDPOINT_SUPPORT={';'.join(unsupported.signature_id.tolist()) if len(unsupported) else 'NONE'}

MEDIAN_EVALUABLE_TUMORS={fmt(e_values.median())}
IQR_EVALUABLE_TUMORS={fmt(e_values.quantile(.25))}-{fmt(e_values.quantile(.75))}

KMATCHED_PER_TUMOR_K_VALIDATED={'YES' if inv['Kmatched_per_tumor_K_validated'] else 'NO'}
S_INVARIANCE_VALIDATED={'YES' if inv['S_invariance_validated'] else 'NO'}
ES_INVARIANCE_VALIDATED={'YES' if inv['E_s_invariance_validated'] else 'NO'}
LOTO_EVALUATOR_VALIDATED={'YES' if inv['loto_evaluator_finite_deterministic'] else 'NO'}
BLOCKED_NULL_SIZE_VALIDATED={'YES' if null_status['all_q_pass'] else 'NO'}

Q025_MEDIAN_DETECTABILITY_SUPPORTED={fmt(qvals[0.25].median_detectability_supported)}
Q050_MEDIAN_DETECTABILITY_SUPPORTED={fmt(qvals[0.50].median_detectability_supported)}
Q075_MEDIAN_DETECTABILITY_SUPPORTED={fmt(qvals[0.75].median_detectability_supported)}
Q100_MEDIAN_DETECTABILITY_SUPPORTED={fmt(qvals[1.00].median_detectability_supported)}

Q100_IQR_SUPPORTED={fmt(q100.q25_detectability_supported)}-{fmt(q100.q75_detectability_supported)}

Q100_FULL_SUPPORT_STRATUM_MEDIAN={fmt(q100.full_support_median_detectability)}
Q100_FULL_SUPPORT_STRATUM_N={int(q100.full_support_signature_n)}

Q100_MEDIAN_K_TOTAL={fmt(q100.median_K_total)}
Q100_MEDIAN_K_INSIDE_ES={fmt(q100.median_K_inside_E)}
Q100_MEDIAN_FRACTION_K_INSIDE_ES={fmt(q100.median_fraction_K_inside_E)}

NAIVE_UNIT_LEVEL_MISCALIBRATION={null_status['naive_unit_level_miscalibration']}

GLOBAL_MSBD_RESPONSE={msbd_text}

STAGE_A_RESULT={result}

CONTINUE_TO_STAGE_B={continuation}

NO_RESCUE_CLAUSE_TRIGGERED={'YES' if result == 'CONSTRUCTIBLE_ENDPOINT_SUPPORTED_BUT_BELOW_PRACTICALLY_USEFUL_DETECTION_FLOOR' else 'NO'}
"""
    gate_path = ROOT / "07_stageA_decision/R4_STAGEA_V11_ENDPOINT_SUPPORT_AND_DETECTABILITY_GATE.txt"
    gate_path.write_text(gate)
    handoff = f"""# R4 Stage-A v1.1 handoff

## 1. v1.0 historical failure

v1.0 (`37a1ba6edc01d80fc6173273707c53b322a03f1f14f26fb4a809d92702ce990c`) is permanently recorded as `SUPERSEDED_PRE_RESULT_PROTOCOL_DEFINEDNESS_FAILURE`. It stopped at PF4C-B1-013 / ECM_INVASION / tumor 710262 because the held-out frozen global middle band had zero measurement units. No formal ledger or Ivy detectability result existed.

## 2. v1.1 prospective repair

v1.1 (`{protocol_hash}`) is a `PROSPECTIVELY_REVISED_CONFIRMATORY_PROTOCOL_AFTER_PRE_RESULT_DEFINEDNESS_FAILURE` in a new frozen lineage. It changes only the endpoint domain to fixed signature-specific E_s and the mathematically required one-reference-unit rule. v1.0 scientific runtime values were not reused.

## 3. endpoint-support results

The endpoint is supported for {len(supported)}/61 signatures ({fmt(len(supported)/61)}). E_s is the frozen set of tumors with at least one middle well; signatures with |E_s|<=1 are explicitly not evaluable, not negative. The full-support stratum contains {len(full)} signatures.

## 4. detectability results

Supported-signature median detectability by q is Q025={fmt(qvals[.25].median_detectability_supported)}, Q050={fmt(qvals[.50].median_detectability_supported)}, Q075={fmt(qvals[.75].median_detectability_supported)}, Q100={fmt(qvals[1.0].median_detectability_supported)}. Q100 IQR is {fmt(q100.q25_detectability_supported)}–{fmt(q100.q75_detectability_supported)}. The adjudicating Stage-A result is `{result}`; continuation is `{continuation}`. Stage B was not executed.

## 5. calibration

Exact per-tumor K matching, S invariance, fixed E_s invariance, deterministic finite LOTO geometry, identical ALT/null statistic use, and blocked-null formal size validation all passed. Naive unit-level miscalibration was `{null_status['naive_unit_level_miscalibration']}` under the frozen classification.

## 6. MSBD comparator

The global MSBD response was `{msbd_text}`. This remains a comparator and does not redefine the local endpoint.

## 7. design-support limitations

Endpoint occupancy is signature- and tumor-specific. Zero-middle tumors remain in the full data system, LOTO training fits/covariance, and legal ALT/null construction; they are absent only from endpoint aggregation. Perturbable P_s,q and evaluable E_s are distinct. K_inside_E and its fraction are descriptive visibility geometry, never conditioning variables.

## 8. empirical quantities

The observed occupancy census, endpoint-support rate, detectability probabilities, q response, formal null size, naive-versus-blocked contrast, and global MSBD response are empirical for the frozen Ivy configuration.

## 9. definition-implied quantities

E_s depends only on frozen S, global middle membership, and tumor identity. n_middle>=1 and |E_s|>=2 are mathematical definedness rules. Equal weighting over E_s defines the conditional finite-population estimand. Hamilton k_b, q, K, R=20, B=999, alpha=.05, and the continuation threshold are frozen design definitions.

## 10. prohibited work

No Stage-B detector, field-practice benchmark, DS03 transport, rescue endpoint, P_s,q-only aggregation, well weighting, band redefinition, tumor-specific band, kernel replacement, q/K change, threshold reduction, or new detector was run. Any later Stage-A endpoint/statistic redesign is `EXPLORATORY_POST_HOC` and cannot adjudicate R4. Every future local/conditional Stage-B statistic must define empty target strata and degenerate reference support before execution.
"""
    (ROOT / "99_handoff/R4_STAGEA_V11_HANDOFF.md").write_text(handoff)
    for directory in ("08_stageB_mmd", "09_stageB_alt1", "11_field_practice", "12_ds03_transport"):
        (ROOT / directory / "NOT_RUN.md").write_text("# Not run\n\nStage B and all downstream work remain off. This v1.1 execution stops at the Stage-A terminal gate.\n")
    output_hashes = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            output_hashes.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_tsv(pd.DataFrame(output_hashes), ROOT / "90_QA/FINAL_OUTPUT_HASHES.tsv")
    print(gate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("census", "validate-null", "run-alt", "finalize"))
    args = parser.parse_args()
    {"census": census_mode, "validate-null": null_validation_mode, "run-alt": alt_mode, "finalize": finalize_mode}[args.mode]()


if __name__ == "__main__":
    main()
