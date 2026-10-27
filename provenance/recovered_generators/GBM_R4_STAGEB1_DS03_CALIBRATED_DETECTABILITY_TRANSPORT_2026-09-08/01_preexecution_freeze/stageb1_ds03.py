#!/usr/bin/env python3
"""Frozen Stage-B1 DS03 cross-context transport engine; no other Stage B methods."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
INPUT = ROOT / "02_source_reconstruction/DS03_STAGEB1_FROZEN_INPUTS.npz"
SEEDS = ROOT / "01_preexecution_freeze/STAGEB1_DS03_SEED_BANK.tsv"
FREEZE = ROOT / "01_preexecution_freeze/R4_STAGEB1_DS03_TRANSPORT_FREEZE.md"
FREEZE_SHA = ROOT / "01_preexecution_freeze/R4_STAGEB1_DS03_TRANSPORT_FREEZE.sha256"
MEMBERSHIP = ROOT / "03_endpoint_support/DS03_ENDPOINT_MEMBERSHIP_8x57.tsv"
CENSUS = ROOT / "03_endpoint_support/DS03_ENDPOINT_CAPACITY_CENSUS_8x12.tsv"
BY_SIGNATURE = ROOT / "03_endpoint_support/DS03_DESIGN_EVALUABILITY_BY_SIGNATURE.tsv"
BENCH_SET = ROOT / "03_endpoint_support/DS03_BENCHMARK_SET_RECEIPT.tsv"
IVY_ROOT = PROJECT / "GBM_R4_TRUTH_ISOLATED_DETECTABILITY_BENCHMARK_STAGEA_V1_1_2026-09-08"
CORE_SOURCE = IVY_ROOT / "01_PREEXECUTION_FREEZE/stagea_v11.py"
CONSTRUCT_SOURCE = PROJECT / "GBM_PHASE35_FROZEN_PIPELINE_OPERATING_CHARACTERISTICS_V2_2026-09-03/05_alternative_construction/construct_v2.py"
FROZEN_SOURCE = PROJECT / "GBM_PHASE35_FROZEN_PIPELINE_OPERATING_CHARACTERISTICS_V2_2026-09-03/08_simulation_engine/frozen_v2.py"
HISTORICAL_CAPACITY = PROJECT / "GBM_R4_IVY_GBMAP_ALTERNATIVE_CAPACITY_AUDIT_2026-09-08/02_existing_R4_reconstruction/02_ORIGINAL_CONTEXT_CAPACITY_RECONSTRUCTION.tsv"

Q_GRID = (.25, .50, .75, 1.00)
Q_NAMES = {.25: "q25", .50: "q50", .75: "q75", 1.00: "q100"}
ALPHA = .05
NULL_B = 999
ALT_R = 20


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tsv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="NA", float_format="%.12g", lineterminator="\n")


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def modules():
    core = import_file("stageb1_transport_core", CORE_SOURCE)
    construct, frozen = core.load_historical_modules()
    return core, construct, frozen


def verify_protocol():
    expected = FREEZE_SHA.read_text().split()[0]
    if "STATUS=FROZEN" not in FREEZE.read_text() or sha(FREEZE) != expected:
        raise RuntimeError("Stage-B1 protocol freeze mismatch")
    frozen_hashes = {
        INPUT: "9b599b95c64dd5ada1a5eba8386232f5b83350705e56bfe4d567e2e25da21db1",
        SEEDS: "2da5a39eb2bbc46f0a30088985d28490f6f67dd4fb341420441f8d78b76db769",
        CORE_SOURCE: "d7e8ad455c5d52a87d737a6544dd35ba4fee5ddf79f0f9f6525cf803b278296a",
        CONSTRUCT_SOURCE: "8f9359366baf61d776098fc48f1fcdffa1e27d3c03a51685a67202b9eb30c436",
        FROZEN_SOURCE: "6256bc04452a641fd94c06bdd926bc100e47e6006e7fb6d6ab921ea76b6c701c",
        HISTORICAL_CAPACITY: "c5b78f4432bfa7d4532f251089c5cb1ff5d94b231dba76ba34cd58a6ccf4f6e6",
    }
    for path, digest in frozen_hashes.items():
        if sha(path) != digest:
            raise RuntimeError(f"frozen source hash mismatch: {path}")
    return expected


def load_inputs():
    z = np.load(INPUT, allow_pickle=False)
    ids = z["signature_ids"].astype(str)
    tids = z["anonymous_template_ids"].astype(str)
    families = z["claim_geometry"].astype(str)
    units = z["units"].astype(str)
    groups = z["groups"].astype(str)
    scores = z["S"].astype(np.float64)
    biology = z["B"].astype(np.float64)
    source_window = z["source_window"].astype(bool)
    if len(ids) != 8 or scores.shape != (8, 57) or biology.shape != (8, 57, 4) or len(set(groups)) != 12:
        raise RuntimeError("DS03 frozen source geometry mismatch")
    return ids, tids, families, units, groups, scores, biology, source_window


def prepare(core, construct, frozen, s, b, groups, units, family, middle):
    gate, historical_fit = core.legal_gate_factory(s, b, groups, family, frozen)
    zb = np.column_stack([frozen.NS["normal_scores"](b[:, j]) for j in range(b.shape[1])])
    eps = zb - historical_fit["zs"][:, None] * historical_fit["beta"][None, :]
    prepared = construct.prepare_pools(s, groups, units, eps, historical_fit["residual"], "ALT2")
    if not np.array_equal(prepared["M"], middle):
        raise RuntimeError("middle-band implementation mismatch")
    return gate, prepared


def census_mode():
    protocol_hash = verify_protocol()
    core, construct, frozen = modules()
    ids, tids, families, units, groups, scores, biology, source_windows = load_inputs()
    labels = np.array(sorted(set(groups)))
    member_rows, census_rows, sig_rows = [], [], []
    for idx, signature_id in enumerate(ids):
        s, b, family = scores[idx], biology[idx], families[idx]
        middle = core.middle_window(s)
        if not np.array_equal(middle, source_windows[idx]):
            raise RuntimeError("source-frozen global middle membership conflict")
        s_hash = core.array_sha256(s)
        m_hash = core.array_sha256(middle.astype(np.uint8))
        gate, prepared = prepare(core, construct, frozen, s, b, groups, units, family, middle)
        allocations = {q: construct.allocation(prepared["capacities"], q) for q in Q_GRID}
        e_labels = [str(g) for g in labels if np.sum((groups == g) & middle) >= 1]
        e_set = set(e_labels)
        for unit, group, flag in zip(units, groups, middle):
            member_rows.append({"signature_id": signature_id, "unit_id": unit, "block_id": group,
                                "middle_member": int(flag), "S_sha256": s_hash, "middle_sha256": m_hash})
        for group in labels:
            block = groups == group
            nm = int(np.sum(block & middle)); no = int(np.sum(block & ~middle))
            row = {"signature_id": signature_id, "anonymous_template_id": tids[idx], "claim_geometry": family,
                   "block_id": str(group), "n_units_total": int(np.sum(block)), "n_middle": nm,
                   "n_outside": no, "endpoint_supported_block": "YES" if nm >= 1 else "NO",
                   "ALT2_capacity_b": int(prepared["capacities"][str(group)])}
            for q in Q_GRID:
                qn = Q_NAMES[q]; k = int(allocations[q][2][str(group)])
                row[f"k_b_{qn}"] = k; row[f"perturbable_{qn}"] = "YES" if k > 0 else "NO"
            census_rows.append(row)
        total_capacity = int(sum(prepared["capacities"].values()))
        endpoint = len(e_labels) >= 2
        constructible = total_capacity > 0
        benchmark = endpoint and constructible and family == "MES_MALIGNANT_STATE"
        sr = {"signature_id": signature_id, "anonymous_template_id": tids[idx], "claim_geometry": family,
              "block_n": len(labels), "E_s_n": len(e_labels), "E_s": ";".join(e_labels),
              "endpoint_supported": "YES" if endpoint else "NO", "ALT2_total_capacity": total_capacity,
              "ALT2_constructible": "YES" if constructible else "NO", "reference_recoverable": "YES",
              "benchmark_evaluable": "YES" if benchmark else "NO",
              "SIGNATURE_STATUS": "BENCHMARK_EVALUABLE" if benchmark else ("NOT_EVALUABLE_CONSTRUCTIBILITY" if not constructible else "NOT_EVALUABLE_INSUFFICIENT_ENDPOINT_SUPPORT")}
        for q in Q_GRID:
            qn = Q_NAMES[q]; _, target, assigned = allocations[q]
            pset = {g for g, k in assigned.items() if k > 0}
            inside = int(sum(k for g, k in assigned.items() if g in e_set))
            sr[f"K_target_{qn}"] = int(target); sr[f"P_s_{qn}_n"] = len(pset)
            sr[f"K_inside_E_{qn}"] = inside; sr[f"K_outside_E_{qn}"] = int(target - inside)
            sr[f"fraction_K_inside_E_{qn}"] = inside / target if target else np.nan
            sr[f"fraction_E_perturbed_{qn}"] = len(pset & e_set) / len(e_set) if e_set else np.nan
        sig_rows.append(sr)
    membership, census, by_sig = pd.DataFrame(member_rows), pd.DataFrame(census_rows), pd.DataFrame(sig_rows)
    if len(membership) != 8 * 57 or len(census) != 8 * 12 or len(by_sig) != 8:
        raise RuntimeError("DS03 census cardinality failure")
    write_tsv(membership, MEMBERSHIP); write_tsv(census, CENSUS); write_tsv(by_sig, BY_SIGNATURE)
    # Only after independent reconstruction, compare against the historical
    # authority as a source-consistency check. It never supplies runtime values.
    historical = pd.read_csv(HISTORICAL_CAPACITY, sep="\t")
    historical = historical[(historical.context.eq("DS03")) & (historical.alternative.eq("ALT2"))].copy()
    check = census.merge(
        historical[["signature_id", "block_id", "middle_units", "outside_units", "alt2_capacity", "reconstructed_K_TOTAL"]],
        on=["signature_id", "block_id"], how="outer", validate="one_to_one", indicator=True,
    )
    check["middle_match"] = check.n_middle.eq(check.middle_units)
    check["outside_match"] = check.n_outside.eq(check.outside_units)
    check["capacity_match"] = check.ALT2_capacity_b.eq(check.alt2_capacity)
    total_now = by_sig.set_index("signature_id").ALT2_total_capacity
    check["total_capacity_recomputed"] = check.signature_id.map(total_now)
    check["total_match"] = check.total_capacity_recomputed.eq(check.reconstructed_K_TOTAL)
    check["source_consistency_pass"] = (
        check._merge.eq("both") & check.middle_match & check.outside_match & check.capacity_match & check.total_match
    )
    write_tsv(check, ROOT / "03_endpoint_support/DS03_HISTORICAL_CAPACITY_CONSISTENCY_CHECK.tsv")
    if len(check) != 8 * 12 or not bool(check.source_consistency_pass.all()):
        raise RuntimeError("STOP_AND_REPORT_CAPACITY_RECONSTRUCTION_CONFLICT")
    bench = by_sig[by_sig.benchmark_evaluable.eq("YES")][["signature_id", "anonymous_template_id", "claim_geometry", "E_s_n", "ALT2_total_capacity"]].copy()
    write_tsv(bench, BENCH_SET)
    n = len(bench)
    if n:
        r0 = math.ceil(800 / n); target = r0 * n
    else:
        r0 = target = 0
    summary = {"protocol_hash": protocol_hash, "frozen_identity_n": 8,
               "ALT2_constructible_signature_n": int(by_sig.ALT2_constructible.eq("YES").sum()),
               "endpoint_supported_signature_n": int(by_sig.endpoint_supported.eq("YES").sum()),
               "benchmark_evaluable_signature_n": n, "null_worlds_per_signature_q": r0,
               "target_null_worlds_per_q": target, "no_ALT_or_null_worlds_generated": True,
               "no_detectability_inspected": True, "membership_sha256": sha(MEMBERSHIP),
               "census_sha256": sha(CENSUS), "benchmark_set_sha256": sha(BENCH_SET),
               "historical_capacity_consistency_validated": True}
    (ROOT / "03_endpoint_support/DS03_STRUCTURE_CENSUS_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def load_support(core, ids, units, groups, scores):
    member = pd.read_csv(MEMBERSHIP, sep="\t", dtype=str)
    by_sig = pd.read_csv(BY_SIGNATURE, sep="\t", dtype=str)
    result = {}
    for idx, signature_id in enumerate(ids):
        f = member[member.signature_id.eq(signature_id)]
        if f.unit_id.tolist() != units.tolist() or f.block_id.tolist() != groups.tolist():
            raise RuntimeError("sealed endpoint identity mismatch")
        middle = f.middle_member.astype(int).to_numpy().astype(bool)
        if f.S_sha256.iat[0] != core.array_sha256(scores[idx]) or f.middle_sha256.iat[0] != core.array_sha256(middle.astype(np.uint8)):
            raise RuntimeError("sealed S/middle hash mismatch")
        row = by_sig[by_sig.signature_id.eq(signature_id)].iloc[0]
        e_text = row.E_s if isinstance(row.E_s, str) else ""
        e_labels = tuple(e_text.split(";")) if e_text else tuple()
        result[signature_id] = {"middle": middle, "E_s": e_labels, "S_sha256": f.S_sha256.iat[0], "row": row}
    return result, by_sig


def geometry(core, signature_id, s, b, groups, support):
    core.assert_domain_invariant(s, support)
    d1, w1, f1 = core.build_loto_distance_table(s, b, groups, support["middle"], support["E_s"])
    d2, w2, _ = core.build_loto_distance_table(s, b, groups, support["middle"], support["E_s"])
    if not np.array_equal(d1, d2) or not np.array_equal(w1, w2):
        raise RuntimeError("NOT_EVALUABLE_ENDPOINT_GEOMETRY_FAILURE")
    f1.insert(0, "signature_id", signature_id)
    return d1, w1, f1


def seed_subset(schedule, signature_id, q, world_type, n):
    x = schedule[(schedule.signature_id == signature_id) & np.isclose(schedule.q, q) & schedule.world_type.eq(world_type)].sort_values("replicate_id").head(n)
    if len(x) != n:
        raise RuntimeError("seed bank insufficient")
    return x


def calibration_mode():
    protocol_hash = verify_protocol()
    core, construct, frozen = modules()
    ids, tids, families, units, groups, scores, biology, _ = load_inputs()
    support_map, by_sig = load_support(core, ids, units, groups, scores)
    bench = set(by_sig.loc[by_sig.benchmark_evaluable.eq("YES"), "signature_id"])
    nbench = len(bench); r0 = math.ceil(800 / nbench) if nbench else 0
    schedule = pd.read_csv(SEEDS, sep="\t", dtype={"signature_id": str, "world_type": str})
    rows, folds_all = [], []
    for idx, signature_id in enumerate(ids):
        if signature_id not in bench: continue
        s, b, family, support = scores[idx], biology[idx], families[idx], support_map[signature_id]
        distance, weights, folds = geometry(core, signature_id, s, b, groups, support); folds_all.append(folds)
        pair = core.robust_b_distance(b, groups); graph = core.score_graph(s, groups)
        labels, codes = np.unique(groups, return_inverse=True); pools = core.padded_pools(groups, support["middle"])
        gi = np.flatnonzero(support["middle"]).astype(np.int64); go = np.flatnonzero(~support["middle"]).astype(np.int64)
        _, prepared = prepare(core, construct, frozen, s, b, groups, units, family, support["middle"])
        for q in Q_GRID:
            _, target, assigned = construct.allocation(prepared["capacities"], q)
            k = np.array([assigned[str(g)] for g in labels], dtype=np.int64)
            for record in seed_subset(schedule, signature_id, q, "NULL_AS_OBS", r0).itertuples(index=False):
                core.assert_domain_invariant(s, support)
                outer, _ = core.run_blocked(record.world_seed, 1, len(s), pools, k, k, distance, weights, pair, graph, codes.astype(np.int64), True, False)
                blocked, _ = core.run_blocked(record.blocked_null_seed, NULL_B, len(s), pools, k, k, distance, weights, pair, graph, codes.astype(np.int64), True, False)
                naive = core.naive_ensemble(core.seed32(record.naive_null_seed), NULL_B, len(s), gi, go, int(target), distance, weights)
                pb, eb = core.randomization_p(float(outer[0]), blocked); pn, en = core.randomization_p(float(outer[0]), naive)
                rows.append({"signature_id": signature_id, "q": q, "replicate_id": int(record.replicate_id),
                             "K_target": int(target), "k_b": json.dumps({str(g): int(v) for g, v in zip(labels, k)}, sort_keys=True),
                             "per_block_K_matched": "YES", "observed_statistic": float(outer[0]),
                             "blocked_exceedances": eb, "blocked_p": pb, "blocked_rejected": int(pb <= ALPHA),
                             "naive_exceedances": en, "naive_p": pn, "naive_rejected": int(pn <= ALPHA)})
        print(f"DS03_CALIBRATION_SIGNATURE_COMPLETE {signature_id}", flush=True)
    ledger = pd.DataFrame(rows); size_rows = []
    for q, f in ledger.groupby("q"):
        n = len(f); reject = int(f.blocked_rejected.sum()); naive_reject = int(f.naive_rejected.sum())
        test = binomtest(reject, n, .05); ci = test.proportion_ci(confidence_level=.95, method="exact")
        passed = ci.low <= .05 <= ci.high and ci.high < .10
        size_rows.append({"q": q, "null_world_n": n, "blocked_rejections": reject, "blocked_size": reject/n,
                          "exact_95ci_low": ci.low, "exact_95ci_high": ci.high, "exact_binomial_p": test.pvalue,
                          "blocked_size_validation_pass": "YES" if passed else "NO", "naive_rejections": naive_reject,
                          "naive_size": naive_reject/n, "naive_minus_blocked": naive_reject/n-reject/n})
    size = pd.DataFrame(size_rows)
    present = (size.naive_size > .075) & (size.naive_minus_blocked >= .025)
    absent = (size.naive_size <= .075) & (size.naive_minus_blocked.abs() < .025)
    miscal = "PRESENT" if int(present.sum()) >= 3 else ("ABSENT" if bool(absent.all()) else "INCONCLUSIVE")
    write_tsv(ledger, ROOT / "06_kmatched_null/DS03_FORMAL_NULL_LEDGER.tsv")
    write_tsv(size, ROOT / "07_calibration/DS03_BLOCKED_NULL_SIZE_BY_Q.tsv")
    write_tsv(pd.concat(folds_all, ignore_index=True), ROOT / "04_loto_geometry/DS03_LOTO_GEOMETRY_RECEIPT.tsv")
    status = {"protocol_hash": protocol_hash, "benchmark_signature_n": nbench, "null_worlds_per_signature_q": r0,
              "all_q_pass": bool(size.blocked_size_validation_pass.eq("YES").all()),
              "naive_unit_level_miscalibration": miscal, "per_block_K_matched": True,
              "finite_deterministic_geometry": True, "identical_statistic_ALT_null": True}
    (ROOT / "07_calibration/DS03_CALIBRATION_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if not status["all_q_pass"]: raise SystemExit(2)


def alt_mode():
    protocol_hash = verify_protocol()
    calibration = json.loads((ROOT / "07_calibration/DS03_CALIBRATION_STATUS.json").read_text())
    if not calibration["all_q_pass"]: raise RuntimeError("NOT_EVALUABLE_RANDOMIZATION_CALIBRATION_FAILURE")
    core, construct, frozen = modules()
    ids, tids, families, units, groups, scores, biology, _ = load_inputs()
    support_map, by_sig = load_support(core, ids, units, groups, scores)
    bench = set(by_sig.loc[by_sig.benchmark_evaluable.eq("YES"), "signature_id"])
    schedule = pd.read_csv(SEEDS, sep="\t", dtype={"signature_id": str, "world_type": str})
    rows = []
    for idx, signature_id in enumerate(ids):
        if signature_id not in bench: continue
        s, b, family, support = scores[idx], biology[idx], families[idx], support_map[signature_id]
        distance, weights, _ = geometry(core, signature_id, s, b, groups, support)
        pair = core.robust_b_distance(b, groups); graph = core.score_graph(s, groups)
        labels, codes = np.unique(groups, return_inverse=True); pools = core.padded_pools(groups, support["middle"])
        gi = np.flatnonzero(support["middle"]).astype(np.int64); go = np.flatnonzero(~support["middle"]).astype(np.int64)
        gate, prepared = prepare(core, construct, frozen, s, b, groups, units, family, support["middle"])
        for q in Q_GRID:
            total, target, assigned = construct.allocation(prepared["capacities"], q)
            k = np.array([assigned[str(g)] for g in labels], dtype=np.int64)
            e_set = set(support["E_s"]); pset = {g for g, v in assigned.items() if v > 0}
            k_inside = int(sum(v for g, v in assigned.items() if g in e_set))
            for record in seed_subset(schedule, signature_id, q, "ALT2", ALT_R).itertuples(index=False):
                core.assert_domain_invariant(s, support)
                perm, meta = construct.search(b, groups, prepared, q, "ALT2", lambda attempt, ws=int(record.world_seed): core.attempt_seed(ws, attempt), gate)
                if perm is None: raise RuntimeError("ALT2 construction failed")
                core.validate_alt_k(meta, groups, labels, k)
                if not np.array_equal(groups[perm], groups) or not np.array_equal(np.sort(perm), np.arange(len(s))):
                    raise RuntimeError("ALT2 legal invariant failure")
                obs = float(core._dispersion(perm.astype(np.int64), distance, weights))
                obs_msbd = float(core._msbd(perm.astype(np.int64), pair, graph, codes.astype(np.int64), len(labels)))
                blocked, bmsbd = core.run_blocked(record.blocked_null_seed, NULL_B, len(s), pools, k, k, distance, weights, pair, graph, codes.astype(np.int64), True, True)
                naive = core.naive_ensemble(core.seed32(record.naive_null_seed), NULL_B, len(s), gi, go, int(target), distance, weights)
                pb, eb = core.randomization_p(obs, blocked); pn, en = core.randomization_p(obs, naive); pm, em = core.randomization_p(obs_msbd, bmsbd)
                rows.append({"signature_id": signature_id, "anonymous_template_id": tids[idx], "claim_geometry": family,
                             "q": q, "replicate_id": int(record.replicate_id), "E_s_n": len(support["E_s"]),
                             "P_s_q_n": len(pset), "fraction_E_perturbed": len(pset & e_set)/len(e_set),
                             "ALT2_total_capacity": int(total), "K_target": int(target), "K_inside_E": k_inside,
                             "K_outside_E": int(target-k_inside), "fraction_K_inside_E": k_inside/target if target else np.nan,
                             "k_b": json.dumps({str(g): int(v) for g, v in zip(labels, k)}, sort_keys=True),
                             "per_block_K_matched": "YES", "S_invariant": "YES", "E_s_invariant": "YES",
                             "reference_relative_residual_distance": obs, "blocked_exceedances": eb, "blocked_p": pb,
                             "blocked_detected": int(pb <= ALPHA), "naive_exceedances": en, "naive_p": pn,
                             "naive_detected": int(pn <= ALPHA), "global_msbd": obs_msbd, "msbd_exceedances": em,
                             "msbd_p": pm, "msbd_detected": int(pm <= ALPHA),
                             "construction_attempt": int(meta["accepted_candidate_attempt_id"]), "legal_invariants_pass": "YES"})
        print(f"DS03_ALT_SIGNATURE_COMPLETE {signature_id}", flush=True)
    worlds = pd.DataFrame(rows)
    detect = worlds.groupby(["signature_id", "anonymous_template_id", "claim_geometry", "q", "E_s_n", "ALT2_total_capacity", "K_target", "K_inside_E", "K_outside_E", "fraction_K_inside_E", "P_s_q_n"], as_index=False).agg(
        ALT_world_n=("replicate_id", "size"), blocked_detections=("blocked_detected", "sum"), detectability=("blocked_detected", "mean"),
        median_blocked_p=("blocked_p", "median"), naive_detectability=("naive_detected", "mean"), global_msbd_detectability=("msbd_detected", "mean"))
    qrows = []
    for q, f in detect.groupby("q"):
        qrows.append({"q": q, "signature_n": len(f), "median_detectability": f.detectability.median(),
                      "q25_detectability": f.detectability.quantile(.25), "q75_detectability": f.detectability.quantile(.75),
                      "min_detectability": f.detectability.min(), "max_detectability": f.detectability.max(),
                      "median_K": f.K_target.median(), "median_K_inside_E": f.K_inside_E.median(),
                      "median_fraction_K_inside_E": f.fraction_K_inside_E.median(),
                      "median_naive_detectability": f.naive_detectability.median(),
                      "median_global_msbd_detectability": f.global_msbd_detectability.median()})
    qsummary = pd.DataFrame(qrows)
    write_tsv(worlds, ROOT / "05_alt2_worlds/DS03_ALT2_WORLD_LEDGER.tsv")
    write_tsv(detect, ROOT / "08_detectability/DS03_DETECTABILITY_BY_SIGNATURE_Q.tsv")
    write_tsv(qsummary, ROOT / "08_detectability/DS03_DETECTABILITY_BY_Q.tsv")
    write_tsv(detect[["signature_id", "q", "global_msbd_detectability"]], ROOT / "09_msbd_comparator/DS03_GLOBAL_MSBD_BY_SIGNATURE_Q.tsv")
    write_tsv(qsummary[["q", "signature_n", "median_global_msbd_detectability"]], ROOT / "09_msbd_comparator/DS03_GLOBAL_MSBD_BY_Q.tsv")
    receipt = {"protocol_hash": protocol_hash, "ALT_world_n": len(worlds), "signature_q_n": len(detect),
               "all_20_worlds": bool(detect.ALT_world_n.eq(20).all()), "per_block_K_matched": True,
               "S_and_E_s_invariant": True, "legal_invariants_pass": True, "historical_detector_invoked": False,
               "MMD_invoked": False, "ALT1_invoked": False, "field_practice_invoked": False}
    (ROOT / "90_QA/DS03_FORMAL_EXECUTION_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def compare_mode():
    verify_protocol()
    ds = pd.read_csv(ROOT / "08_detectability/DS03_DETECTABILITY_BY_Q.tsv", sep="\t")
    dsize = pd.read_csv(ROOT / "07_calibration/DS03_BLOCKED_NULL_SIZE_BY_Q.tsv", sep="\t")
    dcal = json.loads((ROOT / "07_calibration/DS03_CALIBRATION_STATUS.json").read_text())
    support = json.loads((ROOT / "03_endpoint_support/DS03_STRUCTURE_CENSUS_SUMMARY.json").read_text())
    ivy = pd.read_csv(IVY_ROOT / "05_stageA_alt2/STAGEA_DETECTABILITY_BY_Q.tsv", sep="\t")
    isize = pd.read_csv(IVY_ROOT / "06_stageA_calibration/BLOCKED_NULL_SIZE_BY_Q.tsv", sep="\t")
    rows = []
    for q in Q_GRID:
        ir = ivy[np.isclose(ivy.q, q)].iloc[0]; dr = ds[np.isclose(ds.q, q)].iloc[0]
        ic = isize[np.isclose(isize.q, q)].iloc[0]; dc = dsize[np.isclose(dsize.q, q)].iloc[0]
        rows.extend([
            {"context": "IVY", "q": q, "measurement_units": 122, "blocks": 10, "frozen_identity_n": 61,
             "claim_geometry": "MULTI_FAMILY_TISSUE_SCOPE", "endpoint_supported_n": 61,
             "median_K": ir.median_K_total, "median_detectability": ir.median_detectability_supported,
             "median_global_msbd_detectability": {0.25:.05,0.5:.05,0.75:0,1.0:0}[q],
             "blocked_null_size": ic.blocked_size, "blocked_null_validated": ic.blocked_size_validation_pass,
             "naive_calibration": "ABSENT"},
            {"context": "DS03", "q": q, "measurement_units": 57, "blocks": 12, "frozen_identity_n": 8,
             "claim_geometry": "MES_MALIGNANT_STATE", "endpoint_supported_n": support["endpoint_supported_signature_n"],
             "median_K": dr.median_K, "median_detectability": dr.median_detectability,
             "median_global_msbd_detectability": dr.median_global_msbd_detectability,
             "blocked_null_size": dc.blocked_size, "blocked_null_validated": dc.blocked_size_validation_pass,
             "naive_calibration": dcal["naive_unit_level_miscalibration"]},
        ])
    write_tsv(pd.DataFrame(rows), ROOT / "11_ivy_ds03_comparison/IVY_VS_DS03_CALIBRATED_DETECTABILITY_COMPARISON.tsv")
    (ROOT / "11_ivy_ds03_comparison/COMPARISON_BOUNDARY.md").write_text(
        "# Ivy–DS03 comparison boundary\n\nDescriptive cross-context transport only. No pooling, equality test, t-test, replication claim, or shared q threshold is made. Ivy results are copied from their frozen summaries and are not reanalysed.\n")


def fmt(x):
    if pd.isna(x): return "NOT_AVAILABLE"
    return f"{float(x):.6g}" if isinstance(x, (float, np.floating)) else str(x)


def finalize_mode():
    protocol_hash = verify_protocol()
    inp = json.loads((ROOT / "02_source_reconstruction/DS03_SOURCE_RECONSTRUCTION_RECEIPT.json").read_text())
    support = json.loads((ROOT / "03_endpoint_support/DS03_STRUCTURE_CENSUS_SUMMARY.json").read_text())
    by_sig = pd.read_csv(BY_SIGNATURE, sep="\t", dtype=str)
    qsum = pd.read_csv(ROOT / "08_detectability/DS03_DETECTABILITY_BY_Q.tsv", sep="\t")
    cal = json.loads((ROOT / "07_calibration/DS03_CALIBRATION_STATUS.json").read_text())
    qmap = {float(r.q): r for r in qsum.itertuples(index=False)}
    identities = ";".join(inp["identities"])
    if not cal["all_q_pass"]: status, interpretation = "NOT_EVALUABLE_RANDOMIZATION_CALIBRATION_FAILURE", "TRANSPORT_BOUNDARY_NOT_EVALUABLE"
    else:
        status = "CALIBRATED_OPERATING_CURVE_AVAILABLE"
        curve = [qmap[q].median_detectability for q in Q_GRID]
        interpretation = "ARCHITECTURE_TRANSPORTS_WITH_SIMILAR_DIRECTION" if all(curve[i] <= curve[i+1] for i in range(3)) and curve[-1] > curve[0] else "ARCHITECTURE_TRANSPORTS_WITH_CONTEXT_DEPENDENT_DETECTABILITY"
    msbd = ";".join(f"Q{int(q*100):03d}={fmt(qmap[q].median_global_msbd_detectability)}" for q in Q_GRID)
    q100 = qmap[1.0]
    supported_claim = "A_CALIBRATED_REFERENCE_RELATIVE_ALT2_OPERATING_CURVE_IS_AVAILABLE_IN_THE_ORIGINAL_DS03_GBMSPACE_CONTEXT_WITH_CONTEXT_SPECIFIC_MAGNITUDE"
    prohibited_claim = "NO_NATURAL_ALT2_MECHANISM_NO_STATISTICAL_IDENTITY_WITH_IVY_NO_UNIVERSAL_Q_THRESHOLD_NO_UNIVERSAL_OPTIMALITY_NO_ABSENCE_FROM_NONDETECTION"
    gate = f"""R4_STAGEB1_DS03_TRANSPORT_GATE

PROTOCOL_HASH={protocol_hash}

SOURCE_DS03_UNIT_OBJECT_VALIDATED=YES
DS03_UNIT_COUNT=57
DS03_BLOCK_COUNT=12

FROZEN_DS03_IDENTITY_COUNT=8
FROZEN_DS03_IDENTITIES={identities}

CLAIM_GEOMETRY=MES_MALIGNANT_STATE

ALT2_CONSTRUCTIBLE_SIGNATURES={support['ALT2_constructible_signature_n']}
ENDPOINT_SUPPORTED_SIGNATURES={support['endpoint_supported_signature_n']}
BENCHMARK_EVALUABLE_SIGNATURES={support['benchmark_evaluable_signature_n']}

BLOCKED_NULL_SIZE_VALIDATED={'YES' if cal['all_q_pass'] else 'NO'}

Q025_MEDIAN_DETECTABILITY={fmt(qmap[.25].median_detectability)}
Q050_MEDIAN_DETECTABILITY={fmt(qmap[.50].median_detectability)}
Q075_MEDIAN_DETECTABILITY={fmt(qmap[.75].median_detectability)}
Q100_MEDIAN_DETECTABILITY={fmt(qmap[1.0].median_detectability)}

Q100_IQR={fmt(q100.q25_detectability)}-{fmt(q100.q75_detectability)}

Q025_MEDIAN_K={fmt(qmap[.25].median_K)}
Q050_MEDIAN_K={fmt(qmap[.50].median_K)}
Q075_MEDIAN_K={fmt(qmap[.75].median_K)}
Q100_MEDIAN_K={fmt(qmap[1.0].median_K)}

GLOBAL_MSBD_RESPONSE={msbd}

NAIVE_UNIT_LEVEL_MISCALIBRATION={cal['naive_unit_level_miscalibration']}

IVY_STAGEA_CURVE=
0.40,0.70,0.95,1.00

DS03_TRANSPORT_STATUS={status}

CROSS_CONTEXT_INTERPRETATION={interpretation}

STAGEB1_PRIMARY_SUPPORTED_CLAIM={supported_claim}

STAGEB1_PROHIBITED_CLAIM={prohibited_claim}

CONTINUE_TO_FURTHER_STAGEB=REQUIRES_USER_REVIEW
"""
    (ROOT / "99_handoff/R4_STAGEB1_DS03_TRANSPORT_GATE.txt").write_text(gate)
    handoff = f"""# R4 Stage-B1 DS03 transport handoff

## 1. Authority and scope

Stage-B1 is a narrow cross-context transport under protocol `{protocol_hash}`. It uses the frozen eight-identity DS03 universe and does not import Ivy identities or historical detector outcomes.

## 2. Original frozen DS03 object

The validated object is 57 official GBM-Space malignant-nucleus site pseudobulks nested in 12 donors, with block sizes 2, ten times 4, and 15. The eight identities are {identities}; all use `MES_MALIGNANT_STATE` and the source-recovered four-facet B reference.

## 3. Constructibility

ALT2 was independently recomputed from frozen S/middle/block geometry. {support['ALT2_constructible_signature_n']}/8 identities had capacity above zero. Historical worlds were not reused.

## 4. Endpoint support

{support['endpoint_supported_signature_n']}/8 identities met fixed `|E_s|>=2`; {support['benchmark_evaluable_signature_n']}/8 passed all pre-detection requirements. Unsupported identities, if any, are not negative.

## 5. Formal calibration

The blocked per-block K-matched null used {support['target_null_worlds_per_q']} null-as-observed worlds per q and B=999. All q levels passed the exact 95% CI criterion: `{cal['all_q_pass']}`. The same fixed reference-relative statistic was used for null and ALT.

## 6. ALT2 operating curve

DS03 median detectability was {fmt(qmap[.25].median_detectability)}, {fmt(qmap[.5].median_detectability)}, {fmt(qmap[.75].median_detectability)}, {fmt(qmap[1.0].median_detectability)} across q=.25,.50,.75,1.00. Q100 IQR was {fmt(q100.q25_detectability)}–{fmt(q100.q75_detectability)}. These are Monte Carlo operating characteristics, not biological prevalence.

## 7. Global MSBD comparator

Median global MSBD detectability was `{msbd}`. It remains a comparator to the claim-aligned local architecture; the historical stable-flag detector was not revived.

## 8. Naive arm

The frozen DS03 naive classification was `{cal['naive_unit_level_miscalibration']}` and is secondary, geometry-dependent calibration evidence only.

## 9. Descriptive comparison with Ivy

Ivy's frozen curve is 0.40,0.70,0.95,1.00. DS03 is compared side-by-side without pooling or equality testing. The prespecified interpretation is `{interpretation}`; this is not a formal replication claim.

## 10. Definition-implied versus empirical

Whole-vector/S/global-multiset preservation, residual-targeted ALT2, and Hamilton k allocation are definition-implied. Endpoint occupancy, absolute detectability, q response, MSBD, naive behavior, and the cross-context difference are empirical.

## 11. Supported claim

`{supported_claim}`. Manuscript-facing language should describe reference-relative residual magnitude, not pure dispersion.

## 12. Prohibited claim

`{prohibited_claim}`. The benchmark does not establish natural ALT2, mechanism, prevalence, context equality, or biological absence.

## 13. Potential next extension—not executed

Of the withheld options, a prospectively frozen MMD analysis would add the most genuinely new information because it probes broader distributional change beyond mean reference-relative magnitude. ALT1 Dip and field-practice benchmarking are lower priority here. No extension was executed; any continuation requires user review and a new explicit freeze defining empty/degenerate strata.
"""
    (ROOT / "99_handoff/R4_STAGEB1_DS03_TRANSPORT_HANDOFF.md").write_text(handoff)
    for name in ("MMD", "ALT1_DIP", "FIELD_PRACTICE"):
        (ROOT / "90_QA" / f"{name}_NOT_RUN.txt").write_text("NOT_RUN\n")
    files = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)})
    write_tsv(pd.DataFrame(files), ROOT / "90_QA/FINAL_OUTPUT_HASHES.tsv")
    print(gate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("census", "calibrate", "alt", "compare", "finalize"))
    args = parser.parse_args()
    {"census": census_mode, "calibrate": calibration_mode, "alt": alt_mode, "compare": compare_mode, "finalize": finalize_mode}[args.mode]()


if __name__ == "__main__": main()
