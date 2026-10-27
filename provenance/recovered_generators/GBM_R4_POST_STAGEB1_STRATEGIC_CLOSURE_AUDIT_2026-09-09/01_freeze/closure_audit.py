#!/usr/bin/env python3
"""Frozen bounded explanatory/feasibility engine; never runs a new detector."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.covariance import LedoitWolf


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
IVY = PROJECT / "GBM_R4_TRUTH_ISOLATED_DETECTABILITY_BENCHMARK_STAGEA_V1_1_2026-09-08"
DS = PROJECT / "GBM_R4_STAGEB1_DS03_CALIBRATED_DETECTABILITY_TRANSPORT_2026-09-08"
V6 = PROJECT / "GBM_SIGNATURE_3D_CROSS_LAYER_MEASUREMENT_VALIDATION_V6_FINAL"
P35 = PROJECT / "GBM_PHASE35_FROZEN_PIPELINE_OPERATING_CHARACTERISTICS_V2_2026-09-03"
P2 = PROJECT / "GBM_BRP_PHASE2_201_SIGNATURE_LANDSCAPE_2026-09-02"
HALLMARK_GMT = P35 / "01_phase35_authority/sources/03_bulk/redundancy/reference_sets/h.all.v2026.1.Hs.symbols.gmt"
INVENTORY = ROOT / "00_authority/AUTHORITY_FILE_INVENTORY.tsv"
FREEZE = ROOT / "01_freeze/DS03_BIFURCATION_EXPLANATORY_SCOPE_FREEZE.md"
FREEZE_SHA = ROOT / "01_freeze/DS03_BIFURCATION_EXPLANATORY_SCOPE_FREEZE.sha256"
DIR_SPEC = ROOT / "04_multimodal_feasibility/MULTIMODAL_DIRECTIONAL_ESTIMAND_SPEC.md"
DIR_SPEC_SHA = ROOT / "04_multimodal_feasibility/MULTIMODAL_DIRECTIONAL_ESTIMAND_SPEC.sha256"
SEED_BANK = ROOT / "01_freeze/COUPLING_ICC_NULL_SEED_BANK.tsv"
Q_GRID = (.25, .50, .75, 1.0)
EPS = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="NA", float_format="%.12g", lineterminator="\n")


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def verify_authority() -> str:
    expected = FREEZE_SHA.read_text().split()[0]
    if sha(FREEZE) != expected or "STATUS=FROZEN" not in FREEZE.read_text():
        raise RuntimeError("EXPLANATORY_SCOPE_FREEZE_MISMATCH")
    inventory = pd.read_csv(INVENTORY, sep="\t")
    for row in inventory.itertuples(index=False):
        path = PROJECT / row.relative_path
        if not path.is_file() or sha(path) != row.sha256:
            raise RuntimeError(f"AUTHORITY_HASH_MISMATCH:{row.relative_path}")
    if sha(SEED_BANK) != "7870effab4ea320c4f7377b09aeed066d7bebb7d54c881fa58553db537d4b824":
        raise RuntimeError("COUPLING_NULL_SEED_BANK_MISMATCH")
    return expected


def arrangement_count(middle_counts: dict[str, int], outside_counts: dict[str, int], k: dict[str, int]) -> int:
    total = 1
    for block in sorted(k):
        kb = int(k[block])
        total *= math.comb(int(middle_counts[block]), kb) * math.comb(int(outside_counts[block]), kb) * math.factorial(kb)
    return total


def norm_iqr(x: np.ndarray) -> float:
    return float(np.quantile(x, .75) - np.quantile(x, .25))


def null_stats(x: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_q25": float(np.quantile(x, .25)),
        f"{prefix}_q75": float(np.quantile(x, .75)),
        f"{prefix}_iqr": norm_iqr(x),
        f"{prefix}_q90": float(np.quantile(x, .90)),
        f"{prefix}_q95": float(np.quantile(x, .95)),
        f"{prefix}_q99": float(np.quantile(x, .99)),
        f"{prefix}_sd": float(np.std(x, ddof=1)),
        f"{prefix}_unique_n": int(len(np.unique(x))),
    }


def ds_modules():
    engine = import_file("frozen_stageb1_engine", DS / "01_preexecution_freeze/stageb1_ds03.py")
    core, construct, frozen = engine.modules()
    return engine, core, construct, frozen


def run_ds03() -> None:
    protocol_hash = verify_authority()
    engine, core, construct, frozen = ds_modules()
    ids, tids, families, units, groups, scores, biology, _ = engine.load_inputs()
    support_map, by_sig = engine.load_support(core, ids, units, groups, scores)
    seeds = pd.read_csv(DS / "01_preexecution_freeze/STAGEB1_DS03_SEED_BANK.tsv", sep="\t")
    original = pd.read_csv(DS / "05_alt2_worlds/DS03_ALT2_WORLD_LEDGER.tsv", sep="\t")
    original_detect = pd.read_csv(DS / "08_detectability/DS03_DETECTABILITY_BY_SIGNATURE_Q.tsv", sep="\t")
    block_census = pd.read_csv(DS / "03_endpoint_support/DS03_ENDPOINT_CAPACITY_CENSUS_8x12.tsv", sep="\t")
    labels, codes = np.unique(groups, return_inverse=True)
    world_rows, fold_rows, structural_rows = [], [], []

    for idx, signature_id in enumerate(ids):
        support = support_map[signature_id]
        s, b, family = scores[idx], biology[idx], families[idx]
        distance, weights, _ = engine.geometry(core, signature_id, s, b, groups, support)
        pair = core.robust_b_distance(b, groups)
        graph = core.score_graph(s, groups)
        pools = core.padded_pools(groups, support["middle"])
        gi = np.flatnonzero(support["middle"]).astype(np.int64)
        go = np.flatnonzero(~support["middle"]).astype(np.int64)
        gate, prepared = engine.prepare(core, construct, frozen, s, b, groups, units, family, support["middle"])
        identity = np.arange(len(s), dtype=np.int64)
        baseline_t = float(core._dispersion(identity, distance, weights))
        baseline_info = float(frozen.info(s, b, groups, family))
        per_block = block_census[block_census.signature_id.eq(signature_id)].copy()
        mcounts = dict(zip(per_block.block_id.astype(str), per_block.n_middle.astype(int)))
        ocounts = dict(zip(per_block.block_id.astype(str), per_block.n_outside.astype(int)))

        for q in Q_GRID:
            total_capacity, target, assigned = construct.allocation(prepared["capacities"], q)
            kvec = np.array([assigned[str(g)] for g in labels], dtype=np.int64)
            k_concentration = float(sum((v / target) ** 2 for v in assigned.values())) if target else np.nan
            arrangements = arrangement_count(mcounts, ocounts, assigned)
            seed_rows = seeds[(seeds.signature_id.eq(signature_id)) & np.isclose(seeds.q, q) & seeds.world_type.eq("ALT2")].sort_values("replicate_id")
            if len(seed_rows) != 20:
                raise RuntimeError("DS03_FROZEN_SEED_CARDINALITY_FAILURE")
            for rec in seed_rows.itertuples(index=False):
                perm, meta = construct.search(b, groups, prepared, q, "ALT2",
                                              lambda attempt, ws=int(rec.world_seed): core.attempt_seed(ws, attempt), gate)
                if perm is None:
                    raise RuntimeError("FROZEN_ALT_WORLD_RECONSTRUCTION_FAILURE")
                perm = np.asarray(perm, dtype=np.int64)
                core.validate_alt_k(meta, groups, labels, kvec)
                obs = float(core._dispersion(perm, distance, weights))
                obs_msbd = float(core._msbd(perm, pair, graph, codes.astype(np.int64), len(labels)))
                blocked, bmsbd = core.run_blocked(rec.blocked_null_seed, 999, len(s), pools, kvec, kvec,
                                                  distance, weights, pair, graph, codes.astype(np.int64), True, True)
                naive = core.naive_ensemble(core.seed32(rec.naive_null_seed), 999, len(s), gi, go, int(target), distance, weights)
                pb, eb = core.randomization_p(obs, blocked)
                pn, en = core.randomization_p(obs, naive)
                pm, em = core.randomization_p(obs_msbd, bmsbd)
                old = original[(original.signature_id.eq(signature_id)) & np.isclose(original.q, q) & original.replicate_id.eq(rec.replicate_id)]
                if len(old) != 1:
                    raise RuntimeError("DS03_WORLD_KEY_MISMATCH")
                old = old.iloc[0]
                checks = [np.isclose(obs, old.reference_relative_residual_distance, atol=5e-11), eb == old.blocked_exceedances,
                          np.isclose(pb, old.blocked_p), en == old.naive_exceedances, np.isclose(pn, old.naive_p),
                          np.isclose(obs_msbd, old.global_msbd, atol=5e-11), em == old.msbd_exceedances, np.isclose(pm, old.msbd_p)]
                if not all(checks):
                    raise RuntimeError(f"DS03_FROZEN_WORLD_REPRODUCTION_MISMATCH:{signature_id}:{q}:{rec.replicate_id}")
                recipients = np.asarray(meta["recipients"], dtype=int)
                donors = np.asarray(meta["donors"], dtype=int)
                donor_norm = prepared["r"][donors]
                recipient_norm = prepared["r"][recipients]
                row = {
                    "signature_id": signature_id, "q": q, "replicate_id": int(rec.replicate_id),
                    "K_target": int(target), "E_s_n": len(support["E_s"]),
                    "P_s_q_n": int(sum(v > 0 for v in assigned.values())), "K_concentration": k_concentration,
                    "legal_arrangement_count": str(arrangements), "minimum_attainable_exact_p": 1.0 / arrangements,
                    "monte_carlo_minimum_p": .001, "baseline_T": baseline_t, "T_ALT": obs,
                    "ALT_minus_baseline": obs - baseline_t,
                    "constructor_donor_norm_median": float(np.median(donor_norm)),
                    "constructor_recipient_norm_median": float(np.median(recipient_norm)),
                    "constructor_residual_contrast": float(np.median(donor_norm) - np.median(recipient_norm)),
                    "blocked_tail_mass_ge_ALT": eb / 999, "blocked_p": pb, "BLOCKED_DETECTED": int(pb <= .05),
                    "naive_tail_mass_ge_ALT": en / 999, "naive_p": pn, "NAIVE_DETECTED": int(pn <= .05),
                    "global_msbd": obs_msbd, "msbd_p": pm, "MSBD_DETECTED": int(pm <= .05),
                    **null_stats(blocked, "blocked_null"), **null_stats(naive, "naive_null"),
                }
                row["ALT_minus_blocked_null_median"] = obs - row["blocked_null_median"]
                row["ALT_blocked_null_empirical_percentile"] = float(np.mean(blocked <= obs))
                row["naive_minus_blocked_null_median"] = row["naive_null_median"] - row["blocked_null_median"]
                row["naive_minus_blocked_q95"] = row["naive_null_q95"] - row["blocked_null_q95"]
                row["naive_minus_blocked_iqr"] = row["naive_null_iqr"] - row["blocked_null_iqr"]
                row["REVERSAL_CELL"] = f"{row['BLOCKED_DETECTED']}{row['NAIVE_DETECTED']}"
                world_rows.append(row)
                for held in support["E_s"]:
                    hidx = np.flatnonzero((groups == held) & support["middle"])
                    w = weights[hidx]
                    fold_base = float(np.sum(w * distance[hidx, hidx]) / np.sum(w))
                    fold_alt = float(np.sum(w * distance[hidx, perm[hidx]]) / np.sum(w))
                    fold_rows.append({"signature_id": signature_id, "q": q, "replicate_id": int(rec.replicate_id),
                                      "held_out_block": held, "fold_T_baseline": fold_base,
                                      "fold_T_ALT": fold_alt, "fold_ALT_change": fold_alt - fold_base})

            middle_values = list(mcounts.values())
            structural_rows.append({
                "signature_id": signature_id, "q": q, "ALT2_total_capacity": int(total_capacity), "K_target": int(target),
                "E_s_n": len(support["E_s"]), "P_s_q_n": int(sum(v > 0 for v in assigned.values())),
                "K_concentration": k_concentration, "middle_min_all_blocks": min(middle_values),
                "middle_median_all_blocks": float(np.median(middle_values)), "middle_max_all_blocks": max(middle_values),
                "middle_zero_block_n": int(sum(v == 0 for v in middle_values)),
                "K_inside_E": int(sum(v for g, v in assigned.items() if g in set(support["E_s"]))),
                "fraction_K_inside_E": float(sum(v for g, v in assigned.items() if g in set(support["E_s"])) / target),
                "baseline_informativeness": baseline_info, "S_dynamic_range": float(np.ptp(s)),
                "baseline_reference_radial_scale": baseline_t, "legal_arrangement_count": str(arrangements),
                "minimum_attainable_exact_p": 1.0 / arrangements, "k_b": json.dumps(assigned, sort_keys=True),
            })
        print(f"DS03_EXPLANATORY_REPRODUCED {signature_id}", flush=True)

    worlds = pd.DataFrame(world_rows)
    folds = pd.DataFrame(fold_rows)
    structural = pd.DataFrame(structural_rows)
    write_tsv(worlds, ROOT / "03_null_geometry/DS03_BLOCKED_VS_NAIVE_WORLD_REVERSAL.tsv")
    write_tsv(folds, ROOT / "02_ds03_bifurcation/DS03_FOLDWISE_ALT_CHANGE.tsv")
    write_tsv(structural, ROOT / "02_ds03_bifurcation/DS03_STRUCTURAL_GEOMETRY_BY_SIGNATURE_Q.tsv")

    group_cols = ["signature_id", "q"]
    summaries = worlds.groupby(group_cols, as_index=False).agg(
        world_n=("replicate_id", "size"), cell_00=("REVERSAL_CELL", lambda x: int((x == "00").sum())),
        cell_01=("REVERSAL_CELL", lambda x: int((x == "01").sum())),
        cell_10=("REVERSAL_CELL", lambda x: int((x == "10").sum())),
        cell_11=("REVERSAL_CELL", lambda x: int((x == "11").sum())),
        blocked_detectability=("BLOCKED_DETECTED", "mean"), naive_detectability=("NAIVE_DETECTED", "mean"),
        median_ALT_T=("T_ALT", "median"), median_blocked_p=("blocked_p", "median"),
        median_naive_p=("naive_p", "median"), median_blocked_null_T=("blocked_null_median", "median"),
        median_blocked_null_q95=("blocked_null_q95", "median"), median_blocked_null_iqr=("blocked_null_iqr", "median"),
        median_naive_null_T=("naive_null_median", "median"), median_naive_null_q95=("naive_null_q95", "median"),
        median_naive_null_iqr=("naive_null_iqr", "median"), median_ALT_minus_null=("ALT_minus_blocked_null_median", "median"),
        median_constructor_residual_contrast=("constructor_residual_contrast", "median"),
        median_ALT_percentile=("ALT_blocked_null_empirical_percentile", "median"),
        global_MSBD_detectability=("MSBD_DETECTED", "mean"),
    )
    write_tsv(summaries, ROOT / "03_null_geometry/DS03_BLOCKED_VS_NAIVE_REVERSAL_BY_SIGNATURE_Q.tsv")

    q100 = summaries[np.isclose(summaries.q, 1.0)].merge(structural[np.isclose(structural.q, 1.0)], on=["signature_id", "q"])
    failure = []
    for row in q100.itertuples(index=False):
        f = worlds[(worlds.signature_id.eq(row.signature_id)) & np.isclose(worlds.q, 1.0)]
        if np.isclose(row.blocked_detectability, 1.0):
            label = "INDETERMINATE"
        else:
            shift_limited = float((f.T_ALT <= f.blocked_null_median).mean())
            null_limited = float(((f.T_ALT > f.blocked_null_median) & (f.T_ALT <= f.blocked_null_q95)).mean())
            label = "SHIFT_LIMITED" if shift_limited >= .75 else ("NULL_GEOMETRY_LIMITED" if null_limited >= .75 else "MIXED")
        failure.append(label)
    q100["failure_mode"] = failure
    q100["middle_occupancy_summary"] = q100.apply(
        lambda r: f"min={int(r.middle_min_all_blocks)};median={r.middle_median_all_blocks:g};max={int(r.middle_max_all_blocks)};zero={int(r.middle_zero_block_n)}", axis=1)
    contradiction = q100[[
        "signature_id", "K_target", "E_s_n", "P_s_q_n", "K_concentration", "middle_median_all_blocks", "middle_occupancy_summary",
        "baseline_informativeness", "S_dynamic_range", "median_constructor_residual_contrast", "median_ALT_T",
        "median_blocked_null_T", "median_blocked_null_q95", "median_ALT_minus_null", "blocked_detectability",
        "median_blocked_p", "median_naive_null_T", "median_naive_null_q95", "naive_detectability",
        "global_MSBD_detectability", "failure_mode", "legal_arrangement_count", "minimum_attainable_exact_p",
    ]].rename(columns={"K_target": "K_q100", "P_s_q_n": "P_s_q100_n", "median_constructor_residual_contrast": "constructor_residual_contrast",
                       "median_ALT_minus_null": "ALT_minus_null", "global_MSBD_detectability": "global_MSBD_detectability"})
    write_tsv(contradiction, ROOT / "02_ds03_bifurcation/DS03_Q100_EVIDENTIARY_BIFURCATION_TABLE.tsv")

    # Pretest-proxy audit uses only the frozen list of simple source/design scalars.
    detected = contradiction.blocked_detectability.eq(1)
    proxy_metrics = ["K_q100", "E_s_n", "P_s_q100_n", "K_concentration", "middle_median_all_blocks",
                     "baseline_informativeness", "S_dynamic_range", "constructor_residual_contrast"]
    proxy_frame = contradiction.copy()
    proxy_rows = []
    for metric in proxy_metrics:
        x = pd.to_numeric(proxy_frame[metric], errors="coerce")
        complete = x.notna().all()
        low_sep = complete and x[detected].max() < x[~detected].min()
        high_sep = complete and x[detected].min() > x[~detected].max()
        rho = float(spearmanr(x, detected.astype(int)).statistic) if complete and x.nunique() > 1 else np.nan
        proxy_rows.append({"metric": metric, "complete": complete, "detected_range": f"{x[detected].min():.6g}-{x[detected].max():.6g}",
                           "nondetected_range": f"{x[~detected].min():.6g}-{x[~detected].max():.6g}",
                           "perfect_nonoverlap": bool(low_sep or high_sep), "descriptive_spearman": rho})
    proxy = pd.DataFrame(proxy_rows)
    write_tsv(proxy, ROOT / "02_ds03_bifurcation/DS03_OBVIOUS_PRETEST_PROXY_AUDIT.tsv")
    proxy_status = "PRESENT" if bool(proxy.perfect_nonoverlap.any()) else ("ABSENT" if bool(proxy.complete.all()) else "INDETERMINATE")
    status = {
        "protocol_hash": protocol_hash, "frozen_worlds_reproduced": 640, "new_ALT_worlds_generated": 0,
        "exact_identities": ids.tolist(), "exact_q": list(Q_GRID), "worlds_per_signature_q": 20,
        "q100_detected": contradiction.loc[detected, "signature_id"].tolist(),
        "q100_nondetected": contradiction.loc[~detected, "signature_id"].tolist(),
        "legal_K_sufficient_proxy": "NO" if (contradiction.loc[~detected, "K_q100"].max() > contradiction.loc[detected, "K_q100"].min()) else "INDETERMINATE",
        "obvious_pretest_proxy": proxy_status,
        "world_reversal_present": bool(worlds.REVERSAL_CELL.isin(["01", "10"]).any()),
        "naive_distortion_direction": "BIDIRECTIONAL_SIGNATURE_DEPENDENT" if worlds.groupby("signature_id").REVERSAL_CELL.apply(lambda x: set(x) & {"01", "10"}).apply(bool).sum() >= 2 and worlds.REVERSAL_CELL.eq("01").any() and worlds.REVERSAL_CELL.eq("10").any() else "OTHER",
        "failure_modes": dict(zip(contradiction.signature_id, contradiction.failure_mode)),
    }
    (ROOT / "02_ds03_bifurcation/DS03_BIFURCATION_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


def msbd_components(pair: np.ndarray, graph: np.ndarray, groups: np.ndarray, perm: np.ndarray) -> dict[str, object]:
    edges = [(i, int(j)) for i in range(len(perm)) for j in graph[i] if j >= 0]
    base_edge = np.array([pair[i, j] for i, j in edges], dtype=float)
    alt_edge = np.array([pair[perm[i], perm[j]] for i, j in edges], dtype=float)
    changed_position = perm != np.arange(len(perm))
    affected = np.array([changed_position[i] or changed_position[j] for i, j in edges], dtype=bool)
    delta = alt_edge - base_edge
    base_local = np.array([np.mean([pair[i, int(j)] for j in graph[i] if j >= 0]) for i in range(len(perm))])
    alt_local = np.array([np.mean([pair[perm[i], perm[int(j)]] for j in graph[i] if j >= 0]) for i in range(len(perm))])
    labels = np.array(sorted(set(groups.astype(str))))
    base_block = np.array([np.median(base_local[groups == g]) for g in labels])
    alt_block = np.array([np.median(alt_local[groups == g]) for g in labels])
    fraction = float(affected.mean())
    affected_mean = float(delta[affected].mean()) if affected.any() else np.nan
    affected_abs_mean = float(np.abs(delta[affected]).mean()) if affected.any() else np.nan
    return {
        "eligible_directed_pair_n": len(edges), "affected_directed_pair_n": int(affected.sum()),
        "affected_pair_fraction": fraction, "affected_pair_mean_distance_change": affected_mean,
        "affected_pair_mean_absolute_change": affected_abs_mean,
        "maximum_mean_edge_contribution_abs": fraction * affected_abs_mean if affected.any() else np.nan,
        "dilution_predicted_mean_edge_shift": fraction * affected_mean if affected.any() else np.nan,
        "observed_mean_edge_shift": float(delta.mean()),
        "changed_local_unit_fraction": float(np.mean(~np.isclose(base_local, alt_local, atol=1e-12, rtol=1e-10))),
        "changed_block_median_n": int(np.sum(~np.isclose(base_block, alt_block, atol=1e-12, rtol=1e-10))),
        "changed_block_median_fraction": float(np.mean(~np.isclose(base_block, alt_block, atol=1e-12, rtol=1e-10))),
    }


def regenerate_msbd_context(context: str) -> pd.DataFrame:
    if context == "DS03":
        engine, core, construct, frozen = ds_modules()
        ids, _, families, units, groups, scores, biology, _ = engine.load_inputs()
        support_map, _ = engine.load_support(core, ids, units, groups, scores)
        seeds = pd.read_csv(DS / "01_preexecution_freeze/STAGEB1_DS03_SEED_BANK.tsv", sep="\t")
        ledger = pd.read_csv(DS / "05_alt2_worlds/DS03_ALT2_WORLD_LEDGER.tsv", sep="\t")
        prepare = lambda s, b, g, u, fam, mid: engine.prepare(core, construct, frozen, s, b, g, u, fam, mid)
    else:
        core = import_file("frozen_ivy_core_msbd", IVY / "01_PREEXECUTION_FREEZE/stagea_v11.py")
        _, ids, families, units, groups, scores, biology = core.load_inputs()
        support_map, _, _ = core.load_frozen_support(ids, units, groups, scores)
        construct, frozen = core.load_historical_modules()
        seeds = pd.read_csv(IVY / "02_seed_registry/REPLICATE_SEED_SCHEDULE_V11.tsv", sep="\t")
        ledger = pd.read_csv(IVY / "05_stageA_alt2/STAGEA_WORLD_LEDGER.tsv", sep="\t")
        prepare = lambda s, b, g, u, fam, mid: core.prepare_signature(s, b, g, u, fam, mid, construct, frozen)
    labels, codes = np.unique(groups, return_inverse=True)
    rows = []
    for idx, signature_id in enumerate(ids):
        support = support_map[signature_id]
        s, b, family = scores[idx], biology[idx], families[idx]
        pair = core.robust_b_distance(b, groups)
        graph = core.score_graph(s, groups)
        gate, prepared = prepare(s, b, groups, units, family, support["middle"])
        base_perm = np.arange(len(s), dtype=np.int64)
        base_msbd = float(core._msbd(base_perm, pair, graph, codes.astype(np.int64), len(labels)))
        for q in Q_GRID:
            _, _, assigned = construct.allocation(prepared["capacities"], q)
            kvec = np.array([assigned[str(g)] for g in labels], dtype=np.int64)
            subset = seeds[(seeds.signature_id.eq(signature_id)) & np.isclose(seeds.q, q) & seeds.world_type.eq("ALT2")].sort_values("replicate_id")
            if len(subset) != 20:
                raise RuntimeError(f"{context}_MSBD_SEED_CARDINALITY_FAILURE")
            for rec in subset.itertuples(index=False):
                attempt_seed = core.attempt_seed if context == "IVY" else core.attempt_seed
                perm, meta = construct.search(b, groups, prepared, q, "ALT2",
                                              lambda attempt, ws=int(rec.world_seed): attempt_seed(ws, attempt), gate)
                if perm is None:
                    raise RuntimeError(f"{context}_MSBD_ALT_RECONSTRUCTION_FAILURE")
                perm = np.asarray(perm, dtype=np.int64)
                core.validate_alt_k(meta, groups, labels, kvec)
                alt_msbd = float(core._msbd(perm, pair, graph, codes.astype(np.int64), len(labels)))
                old = ledger[(ledger.signature_id.eq(signature_id)) & np.isclose(ledger.q, q) & ledger.replicate_id.eq(rec.replicate_id)]
                old_value = old.iloc[0].global_msbd if context == "DS03" else old.iloc[0].observed_global_msbd
                if len(old) != 1 or not np.isclose(alt_msbd, old_value, atol=5e-11):
                    raise RuntimeError(f"{context}_MSBD_WORLD_REPRODUCTION_MISMATCH")
                comp = msbd_components(pair, graph, groups, perm)
                undiluted = abs(float(comp["affected_pair_mean_distance_change"]))
                attenuation_explained = 1.0 - min(1.0, abs(alt_msbd - base_msbd) / undiluted) if undiluted > EPS else np.nan
                rows.append({"context": context, "signature_id": signature_id, "q": q,
                             "replicate_id": int(rec.replicate_id), "K_target": int(sum(assigned.values())),
                             "baseline_MSBD": base_msbd, "ALT_MSBD": alt_msbd, "MSBD_shift": alt_msbd - base_msbd,
                             "attenuation_explained_fraction": attenuation_explained, **comp})
        print(f"MSBD_O1_O2_REPRODUCED {context} {signature_id}", flush=True)
    return pd.DataFrame(rows)


def run_msbd() -> None:
    verify_authority()
    ds = regenerate_msbd_context("DS03")
    ivy = regenerate_msbd_context("IVY")
    worlds = pd.concat([ds, ivy], ignore_index=True)
    write_tsv(worlds, ROOT / "05_msbd_construct_validity/MSBD_ALT2_FINITE_SAMPLE_DILUTION_WORLD.tsv")
    summary = worlds.groupby(["context", "q"], as_index=False).agg(
        world_n=("replicate_id", "size"), median_K=("K_target", "median"),
        median_affected_pair_fraction=("affected_pair_fraction", "median"),
        positive_affected_pair_change_fraction=("affected_pair_mean_distance_change", lambda x: float((x > 0).mean())),
        median_affected_pair_change=("affected_pair_mean_distance_change", "median"),
        median_predicted_mean_edge_shift=("dilution_predicted_mean_edge_shift", "median"),
        median_observed_mean_edge_shift=("observed_mean_edge_shift", "median"),
        median_MSBD_shift=("MSBD_shift", "median"), median_absolute_MSBD_shift=("MSBD_shift", lambda x: float(np.median(np.abs(x)))),
        median_changed_block_fraction=("changed_block_median_fraction", "median"),
        median_attenuation_explained=("attenuation_explained_fraction", "median"),
    )
    write_tsv(summary, ROOT / "05_msbd_construct_validity/MSBD_ALT2_DILUTION_BY_CONTEXT_Q.tsv")
    legal = bool((worlds.affected_directed_pair_n > 0).all())
    positive_by_context = worlds.groupby("context").affected_pair_mean_distance_change.apply(lambda x: float((x > 0).mean()))
    if legal and bool((positive_by_context >= .75).all()):
        target = "YES"
    elif legal and float((worlds.affected_pair_mean_distance_change > 0).mean()) >= .50:
        target = "PARTIAL"
    else:
        target = "NO"
    exp_by_context = worlds.groupby("context").attenuation_explained_fraction.median()
    if exp_by_context.notna().all() and bool((exp_by_context >= .80).all()):
        dilution = "NEARLY_ALL"
    elif exp_by_context.notna().any() and float(exp_by_context.median()) >= .20:
        dilution = "PARTLY"
    elif exp_by_context.notna().any():
        dilution = "LITTLE"
    else:
        dilution = "NOT_EVALUABLE"
    interpretation = "SUPPORTED_LIMITATION" if target == "YES" and dilution not in ("NEARLY_ALL", "NOT_EVALUABLE") else ("SCALE_MATCHING_ONLY" if target in ("YES", "PARTIAL") else "NOT_EVALUABLE")
    status = {"ALT2_IS_LEGAL_INSTANCE_OF_MSBD_TARGET": target,
              "MSBD_LOW_RESPONSE_EXPLAINED_BY_DILUTION": dilution,
              "MSBD_CONSTRUCT_VALIDITY_INTERPRETATION": interpretation,
              "positive_affected_pair_change_fraction_by_context": positive_by_context.to_dict(),
              "median_attenuation_explained_by_context": exp_by_context.to_dict(),
              "MSBD_k": 3, "patient_exclusion": "CROSS_BLOCK_NEIGHBORS_ONLY",
              "B_metric": "ROBUST_STANDARDIZED_EUCLIDEAN", "normalizer": "MEDIAN_CROSS_BLOCK_PAIR_DISTANCE",
              "aggregation": "MEAN_3_NEIGHBORS_THEN_BLOCK_MEDIAN_THEN_MEDIAN_BLOCKS"}
    (ROOT / "05_msbd_construct_validity/MSBD_CONSTRUCT_VALIDITY_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


def primary_crosswalk() -> pd.DataFrame:
    x = pd.read_csv(V6 / "03_authority/CELL3D_RNA_ATAC_CROSSWALK.tsv", sep="\t")
    raw = x[x.matched_RNA_ATAC.astype(str).str.lower().eq("true")].copy()
    primary = raw[raw.primary_cross_layer_patient_eligible.astype(str).str.lower().eq("true")].copy()
    if len(raw) != 73 or len(primary) != 69 or primary.patient_id.nunique() != 8:
        raise RuntimeError("V6_CROSS_LAYER_SUBSTRATE_REPRODUCTION_FAILURE")
    if not (primary.patient_matched_region_n.astype(int) >= 5).all():
        raise RuntimeError("V6_PRIMARY_INCLUSION_RULE_FAILURE")
    return primary.sort_values(["patient_id", "sample_id"]).reset_index(drop=True)


def claim_assets() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    taxonomy = pd.read_csv(P2 / "02_claim_taxonomy/CLAIM_TAXONOMY_201.tsv", sep="\t")
    registry = pd.read_csv(P2 / "03_claim_reference/CLAIM_REFERENCE_REGISTRY.tsv", sep="\t")
    members = pd.read_csv(P35 / "01_phase35_authority/sources/GBM_SIGNATURE_CONTEXT_STRESS_TEST_EXTENSION_V1/00_governance/frozen_parent_inputs/FROZEN_ECOLOGY_PROGRAM_MEMBERSHIP_V2.tsv", sep="\t")
    members.gene_symbol = members.gene_symbol.astype(str).str.upper()
    axis_genes = {a: sorted(set(d.gene_symbol)) for a, d in members.groupby("axis_id")}
    if sha(HALLMARK_GMT) != "eecaf6dad908334ae885406ec72bdc0646d8917588ed7c219fac92fc5363f596":
        raise RuntimeError("SUPPLEMENTARY_HALLMARK_AUTHORITY_HASH_MISMATCH")
    hallmark_names = {"HALLMARK_INTERFERON_ALPHA_RESPONSE": "Hallmark_IFNA",
                      "HALLMARK_INTERFERON_GAMMA_RESPONSE": "Hallmark_IFNG"}
    for line in HALLMARK_GMT.read_text().splitlines():
        fields = line.rstrip("\n").split("\t")
        if fields and fields[0] in hallmark_names:
            axis_genes[hallmark_names[fields[0]]] = sorted(set(g.upper() for g in fields[2:] if g))
    referenced = sorted(set(itertools.chain.from_iterable(registry.B_claim_facets.str.split(";"))))
    if not set(referenced).issubset(axis_genes):
        raise RuntimeError(f"CLAIM_FACET_MEMBERSHIP_MISSING:{sorted(set(referenced)-set(axis_genes))}")
    return taxonomy, registry, axis_genes


def score_claim_facets(primary: pd.DataFrame, registry: pd.DataFrame, axis_genes: dict[str, list[str]]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    sample_ids = primary.sample_id.astype(str).tolist()
    facets = sorted(set(itertools.chain.from_iterable(registry.B_claim_facets.str.split(";"))))
    all_genes = sorted(set(itertools.chain.from_iterable(axis_genes[f] for f in facets)))
    rna_path = V6 / "02_public_supplements/processed/CELL3D_REQUIRED_GENE_Z_EXPRESSION.parquet"
    import pyarrow.parquet as pq
    available_rna = set(pq.ParquetFile(rna_path).schema.names)
    rna_cols = [g for g in all_genes if g in available_rna]
    rna = pd.read_parquet(rna_path, columns=["sample_id", *rna_cols]).set_index("sample_id").reindex(sample_ids)
    atac_long = pd.read_csv(V6 / "07_atac/CELL3D_GENE_PROMOTER_ACCESSIBILITY.tsv", sep="\t",
                            usecols=["sample_id", "gene", "gene_promoter_accessibility", "promoter_peak_n"])
    atac_long.gene = atac_long.gene.astype(str).str.upper()
    atac_long = atac_long[atac_long.gene.isin(all_genes) & atac_long.sample_id.isin(sample_ids)]
    atac = atac_long.pivot(index="sample_id", columns="gene", values="gene_promoter_accessibility").reindex(sample_ids)
    available_atac = set(atac.columns[atac.notna().any(axis=0)])
    rna_scores, atac_scores, coverage_rows = {}, {}, []
    for facet in facets:
        genes = axis_genes[facet]
        rg = sorted(set(genes) & set(rna_cols))
        ag = sorted(set(genes) & available_atac)
        rcov, acov = len(rg) / len(genes), len(ag) / len(genes)
        r_ok, a_ok = len(rg) >= 3 and rcov >= .60, len(ag) >= 3 and acov >= .50
        rna_scores[facet] = rna[rg].mean(axis=1).to_frame(facet) if r_ok else pd.DataFrame(index=sample_ids, columns=[facet], dtype=float)
        atac_scores[facet] = atac[ag].mean(axis=1).to_frame(facet) if a_ok else pd.DataFrame(index=sample_ids, columns=[facet], dtype=float)
        coverage_rows.append({"facet": facet, "frozen_gene_n": len(genes), "RNA_present_gene_n": len(rg),
                              "RNA_coverage": rcov, "RNA_scoreable": r_ok, "ATAC_promoter_gene_n": len(ag),
                              "ATAC_promoter_coverage": acov, "ATAC_scoreable": a_ok,
                              "gene_membership_sha256": sha(P35 / "01_phase35_authority/sources/GBM_SIGNATURE_CONTEXT_STRESS_TEST_EXTENSION_V1/00_governance/frozen_parent_inputs/FROZEN_ECOLOGY_PROGRAM_MEMBERSHIP_V2.tsv")})
    return rna_scores, atac_scores, pd.DataFrame(coverage_rows)


def run_v6_authority() -> None:
    verify_authority()
    primary = primary_crosswalk()
    cross = pd.read_csv(V6 / "03_authority/CELL3D_RNA_ATAC_CROSSWALK.tsv", sep="\t")
    rna_eval = pd.read_csv(V6 / "04_rna/CELL3D_MODEL_EVALUABILITY.tsv", sep="\t")
    atac_eval = pd.read_csv(V6 / "07_atac/CELL3D_PRIMARY201_ATAC_EVALUABILITY.tsv", sep="\t")
    taxonomy, registry, axis_genes = claim_assets()
    _, _, coverage = score_claim_facets(primary, registry, axis_genes)
    write_tsv(coverage, ROOT / "04_multimodal_feasibility/V6_CLAIM_FACET_REFERENCE_CONSTRUCTIBILITY.tsv")
    rna_n = int(rna_eval.evaluable_status.eq("EVALUABLE").sum())
    atac_n = int(atac_eval.evaluable_primary_50pct.astype(str).str.lower().eq("true").sum())
    if len(rna_eval) != 201 or rna_n != 200 or len(atac_eval) != 201 or atac_n != 195:
        raise RuntimeError("V6_SIGNATURE_EVALUABILITY_REPRODUCTION_FAILURE")
    source_code = V6 / "12_code/run_v6_pipeline.py"
    ledger = pd.DataFrame([
        ("raw_matched_regions", 73, "matched_RNA_ATAC==True", "CELL3D_RNA_ATAC_CROSSWALK.tsv"),
        ("primary_matched_regions", 69, "matched_RNA_ATAC AND patient_matched_region_n>=5", "CELL3D_RNA_ATAC_CROSSWALK.tsv"),
        ("primary_matched_patients", 8, "primary_cross_layer_patient_eligible==True", "CELL3D_RNA_ATAC_CROSSWALK.tsv"),
        ("RNA_signature_evaluable", rna_n, "mapped_gene_n>=3 AND coverage>=0.60", "CELL3D_MODEL_EVALUABILITY.tsv"),
        ("ATAC_signature_evaluable", atac_n, "promoter mapped_gene_n>=3 AND coverage>=0.50", "CELL3D_PRIMARY201_ATAC_EVALUABILITY.tsv"),
        ("RNA_not_evaluable", ";".join(rna_eval.loc[~rna_eval.evaluable_status.eq("EVALUABLE"), "model_instance_id"]), "201 minus evaluable", "CELL3D_MODEL_EVALUABILITY.tsv"),
        ("ATAC_not_evaluable", ";".join(atac_eval.loc[~atac_eval.evaluable_primary_50pct.astype(str).str.lower().eq("true"), "model_instance_id"]), "201 minus evaluable", "CELL3D_PRIMARY201_ATAC_EVALUABILITY.tsv"),
        ("promoter_accessibility_definition", "peakwise z across matched samples; promoter peaks averaged by gene; present genes averaged by signature", "frozen code lines 650-680", "run_v6_pipeline.py"),
        ("cross_layer_coupling_estimator", "within-patient Spearman(RNA score,promoter-ATAC score), median across 8 patients", "physical matched regions", "run_v6_pipeline.py"),
        ("RNA_ICC_estimator", "random-intercept MixedLM REML; frozen ANOVA fallback", "patient grouping", "run_v6_pipeline.py"),
        ("ATAC_ICC_estimator", "random-intercept MixedLM REML; frozen ANOVA fallback", "patient grouping", "run_v6_pipeline.py"),
        ("publication_clustering", int(taxonomy.publication.nunique()), "publication identity retained per frozen taxonomy", "CLAIM_TAXONOMY_201.tsv"),
        ("claim_reference_families", len(registry), "six frozen claim-family rows", "CLAIM_REFERENCE_REGISTRY.tsv"),
        ("claim_facet_gene_mapping", len(axis_genes), "exact frozen axis_id-to-gene membership", "FROZEN_ECOLOGY_PROGRAM_MEMBERSHIP_V2.tsv"),
        ("historical_number_collision", "69=V6 matched physical regions; 61=Ivy frozen signatures", "different units and universes", "authority recovery"),
    ], columns=["authority_item", "reproduced_value", "definition_or_rule", "source"])
    ledger["source_sha256"] = ledger.source.map({
        "CELL3D_RNA_ATAC_CROSSWALK.tsv": sha(V6 / "03_authority/CELL3D_RNA_ATAC_CROSSWALK.tsv"),
        "CELL3D_MODEL_EVALUABILITY.tsv": sha(V6 / "04_rna/CELL3D_MODEL_EVALUABILITY.tsv"),
        "CELL3D_PRIMARY201_ATAC_EVALUABILITY.tsv": sha(V6 / "07_atac/CELL3D_PRIMARY201_ATAC_EVALUABILITY.tsv"),
        "run_v6_pipeline.py": sha(source_code), "CLAIM_TAXONOMY_201.tsv": sha(P2 / "02_claim_taxonomy/CLAIM_TAXONOMY_201.tsv"),
        "CLAIM_REFERENCE_REGISTRY.tsv": sha(P2 / "03_claim_reference/CLAIM_REFERENCE_REGISTRY.tsv"),
        "FROZEN_ECOLOGY_PROGRAM_MEMBERSHIP_V2.tsv": sha(P35 / "01_phase35_authority/sources/GBM_SIGNATURE_CONTEXT_STRESS_TEST_EXTENSION_V1/00_governance/frozen_parent_inputs/FROZEN_ECOLOGY_PROGRAM_MEMBERSHIP_V2.tsv")})
    write_tsv(ledger, ROOT / "04_multimodal_feasibility/V6_CROSS_LAYER_AUTHORITY_LEDGER.tsv")
    status = {
        "V6_SOURCE_SUBSTRATE_RECOVERED": "YES", "raw_matched_regions": 73, "primary_matched_regions": 69,
        "primary_matched_patients": 8, "RNA_signature_evaluable": rna_n, "ATAC_signature_evaluable": atac_n,
        "COMPUTATIONALLY_CONSTRUCTIBLE": "YES" if coverage.ATAC_scoreable.all() else "NO",
        "BIOLOGICALLY_INTERPRETABLE": "QUALIFIED_REGULATORY_ANALOGUE_NOT_TRANSCRIPTOMIC_EQUIVALENCE",
        "SOURCE_FROZEN": "COMPONENTS_YES_DERIVED_B_ATAC_NO",
        "ATAC_CLAIM_REFERENCE_STATUS": "PROSPECTIVELY_CONSTRUCTIBLE_FROM_FROZEN_COMPONENTS" if coverage.ATAC_scoreable.all() else "NOT_CONSTRUCTIBLE",
        "result_aware_feature_selection": "NO",
    }
    (ROOT / "04_multimodal_feasibility/V6_ATAC_REFERENCE_ADJUDICATION.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


def normal_score(x: np.ndarray) -> np.ndarray:
    keep = np.isfinite(x)
    out = np.full(len(x), np.nan)
    out[keep] = np.clip((rankdata(x[keep], method="average") - .5) / keep.sum(), 1e-6, 1 - 1e-6)
    from scipy.stats import norm
    out[keep] = norm.ppf(out[keep])
    return out


def residual_geometry(s: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sn = normal_score(s)
    bn = np.column_stack([normal_score(B[:, j]) for j in range(B.shape[1])])
    X = np.column_stack([np.ones(len(sn)), sn])
    coef = np.linalg.lstsq(X, bn, rcond=None)[0]
    eps = bn - X @ coef
    cov = LedoitWolf().fit(eps).covariance_
    vals, vecs = np.linalg.eigh(cov)
    floor = max(1e-12, float(vals.max()) * 1e-10)
    cov = (vecs * np.maximum(vals, floor)) @ vecs.T
    return eps, cov


def run_multimodal_census() -> None:
    verify_authority()
    if sha(DIR_SPEC) != DIR_SPEC_SHA.read_text().split()[0] or "STATUS=FROZEN" not in DIR_SPEC.read_text():
        raise RuntimeError("MULTIMODAL_DIRECTIONAL_SPEC_MISMATCH")
    adjud = json.loads((ROOT / "04_multimodal_feasibility/V6_ATAC_REFERENCE_ADJUDICATION.json").read_text())
    if adjud["ATAC_CLAIM_REFERENCE_STATUS"] not in ("LEGAL_FROZEN_ANALOGUE_AVAILABLE", "PROSPECTIVELY_CONSTRUCTIBLE_FROM_FROZEN_COMPONENTS"):
        raise RuntimeError("MULTIMODAL_REFERENCE_NOT_AUTHORISED")
    primary = primary_crosswalk()
    taxonomy, registry, axis_genes = claim_assets()
    rna_facets, atac_facets, _ = score_claim_facets(primary, registry, axis_genes)
    rna_score = pd.read_csv(V6 / "04_rna/CELL3D_PRIMARY201_RNA_SCORE_MATRIX.tsv", sep="\t").set_index("sample_id").reindex(primary.sample_id)
    atac_score = pd.read_csv(V6 / "07_atac/CELL3D_PRIMARY201_PROMOTER_ACCESSIBILITY_SCORE.tsv", sep="\t").set_index("sample_id").reindex(primary.sample_id)
    rna_eval = set(pd.read_csv(V6 / "04_rna/CELL3D_MODEL_EVALUABILITY.tsv", sep="\t").query("evaluable_status == 'EVALUABLE'").model_instance_id)
    ae = pd.read_csv(V6 / "07_atac/CELL3D_PRIMARY201_ATAC_EVALUABILITY.tsv", sep="\t")
    atac_eval = set(ae.loc[ae.evaluable_primary_50pct.astype(str).str.lower().eq("true"), "model_instance_id"])
    candidates = taxonomy[taxonomy.BRP_evaluable_prespecified.astype(str).str.upper().eq("YES") & taxonomy.signature_id.isin(rna_eval & atac_eval)].copy()
    construct = import_file("frozen_construct_multimodal", P35 / "05_alternative_construction/construct_v2.py")
    groups = primary.patient_id.astype(str).to_numpy()
    units = primary.sample_id.astype(str).to_numpy()
    rows = []
    for rec in candidates.itertuples(index=False):
        ref = registry[registry.claim_family.eq(rec.claim_family)].iloc[0]
        facets = ref.B_claim_facets.split(";")
        Brna = np.column_stack([rna_facets[f].loc[units, f].to_numpy(float) for f in facets])
        Batac = np.column_stack([atac_facets[f].loc[units, f].to_numpy(float) for f in facets])
        if not np.isfinite(Brna).all() or not np.isfinite(Batac).all():
            continue
        for direction, smat, Bdef in (("RNA_DEFINED", rna_score, Brna), ("ATAC_DEFINED", atac_score, Batac)):
            s = smat[rec.signature_id].to_numpy(float)
            if not np.isfinite(s).all():
                continue
            eps, cov = residual_geometry(s, Bdef)
            prepared = construct.prepare_pools(s, groups, units, eps, cov, "ALT2")
            total, target, assigned = construct.allocation(prepared["capacities"], 1.0)
            middle = prepared["M"]
            mcounts = {g: int(np.sum((groups == g) & middle)) for g in sorted(set(groups))}
            ocounts = {g: int(np.sum((groups == g) & ~middle)) for g in sorted(set(groups))}
            arrangements = arrangement_count(mcounts, ocounts, assigned)
            for g in sorted(set(groups)):
                pool_low, pool_high = prepared["pools"][g]
                rows.append({"signature_id": rec.signature_id, "claim_family": rec.claim_family, "direction": direction,
                             "patient_id": g, "region_n": int(np.sum(groups == g)), "middle_n": mcounts[g],
                             "outside_n": ocounts[g], "endpoint_supporting_patient": int(mcounts[g] >= 1),
                             "low_middle_recipient_n": len(pool_low), "high_residual_donor_n": len(pool_high),
                             "maximum_K_patient": int(prepared["capacities"][g]), "q100_k_b": int(assigned[g]),
                             "perturbable_patient": int(assigned[g] > 0), "total_K": int(total),
                             "perturbable_patient_n": int(sum(v > 0 for v in assigned.values())),
                             "endpoint_supported_patient_n": int(sum(v >= 1 for v in mcounts.values())),
                             "dual_layer_endpoint_signature_supported": int(sum(v >= 1 for v in mcounts.values()) >= 2),
                             "distinct_legal_K_matched_arrangements": str(arrangements),
                             "minimum_attainable_exact_p": 1.0 / arrangements if arrangements else np.nan,
                             "bundle_identity": "RNA_REGION_PLUS_PAIRED_ATAC_REGION_MOVES_TOGETHER",
                             "score_coordinate": "S_RNA_FIXED" if direction == "RNA_DEFINED" else "S_ATAC_FIXED"})
        print(f"MULTIMODAL_CENSUS {rec.signature_id}", flush=True)
    census = pd.DataFrame(rows)
    write_tsv(census, ROOT / "04_multimodal_feasibility/MULTIMODAL_CONSTRUCTIBILITY_CENSUS.tsv")
    sig = census.groupby(["signature_id", "claim_family", "direction"], as_index=False).agg(
        total_K=("total_K", "first"), median_legal_swaps_constructible_patients=("maximum_K_patient", lambda x: float(np.median(x[x > 0])) if (x > 0).any() else 0),
        patients_with_at_least_2_swaps=("maximum_K_patient", lambda x: int((x >= 2).sum())),
        perturbable_patient_n=("perturbable_patient_n", "first"), endpoint_supported_patient_n=("endpoint_supported_patient_n", "first"),
        endpoint_supported=("dual_layer_endpoint_signature_supported", "first"), arrangements=("distinct_legal_K_matched_arrangements", "first"),
        minimum_attainable_exact_p=("minimum_attainable_exact_p", "first"))
    write_tsv(sig, ROOT / "04_multimodal_feasibility/MULTIMODAL_CONSTRUCTIBILITY_BY_SIGNATURE_DIRECTION.tsv")
    pass_rows = []
    for direction, d in sig.groupby("direction"):
        s4 = float(d.median_legal_swaps_constructible_patients.median()) >= 2
        s5 = d.patients_with_at_least_2_swaps >= 4
        s7 = d.minimum_attainable_exact_p <= .05
        pass_id = set(d.loc[(d.endpoint_supported == 1) & s5 & s7, "signature_id"])
        pass_rows.append({"direction": direction, "candidate_signature_n": len(d), "S4_median_swaps": float(d.median_legal_swaps_constructible_patients.median()),
                          "S4_pass": s4, "signatures_passing_S5_S7_endpoint": len(pass_id), "passing_signatures": ";".join(sorted(pass_id))})
    direction_summary = pd.DataFrame(pass_rows)
    write_tsv(direction_summary, ROOT / "04_multimodal_feasibility/MULTIMODAL_STOP_RULES_BY_DIRECTION.tsv")
    sets = {r.direction: set(r.passing_signatures.split(";")) if r.passing_signatures else set() for r in direction_summary.itertuples()}
    endpoint_sets = {g: set(d.loc[d.endpoint_supported.eq(1), "signature_id"]) for g, d in sig.groupby("direction")}
    dual_endpoint = sorted(endpoint_sets.get("RNA_DEFINED", set()) & endpoint_sets.get("ATAC_DEFINED", set()))
    dual_capacity = sorted(sets.get("RNA_DEFINED", set()) & sets.get("ATAC_DEFINED", set()))
    s6 = len(dual_endpoint) >= 30
    alpha_reachable = bool((sig.minimum_attainable_exact_p <= .05).all())
    status = {"MULTIMODAL_DIRECTIONAL_ESTIMAND": "VALID", "candidate_signatures": int(candidates.shape[0]),
              "DUAL_LAYER_ENDPOINT_SUPPORTED_SIGNATURES": len(dual_endpoint), "dual_layer_signature_ids": dual_endpoint,
              "dual_direction_signatures_passing_endpoint_S5_S7": len(dual_capacity),
              "RNA_TO_ATAC_CONSTRUCTIBLE": "YES" if s6 and bool(direction_summary.set_index("direction").loc["RNA_DEFINED", "S4_pass"]) and len(sets.get("RNA_DEFINED", set())) >= 30 else "NO",
              "ATAC_TO_RNA_CONSTRUCTIBLE": "YES" if s6 and bool(direction_summary.set_index("direction").loc["ATAC_DEFINED", "S4_pass"]) and len(sets.get("ATAC_DEFINED", set())) >= 30 else "NO",
              "MULTIMODAL_RANDOMIZATION_ALPHA_REACHABLE": "YES" if alpha_reachable else "NO",
              "FULL_MULTIMODAL_OUTCOME_BENCHMARK_AUTHORISED": "YES" if s6 and direction_summary.S4_pass.all() and len(dual_capacity) >= 30 else "NO",
              "one_directional_progression_prespecified": "YES", "ALT_outcome_worlds_generated": 0, "detectability_calculated": "NO",
              "stop_rules": {"S1": "PASS", "S2": "PASS", "S3": "PASS", "S4": "PASS" if direction_summary.S4_pass.all() else "FAIL",
                             "S5": "PASS" if all(x >= 30 for x in direction_summary.signatures_passing_S5_S7_endpoint) else "FAIL",
                             "S7": "PASS" if alpha_reachable else "FAIL",
                             "S6": "PASS" if s6 else "FAIL", "S8": "PASS"}}
    (ROOT / "04_multimodal_feasibility/MULTIMODAL_FEASIBILITY_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


def rank_corr_rows(x: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    xr = rankdata(x, axis=1, method="average")
    fr = rankdata(fixed, method="average")
    xr -= xr.mean(axis=1, keepdims=True)
    fr -= fr.mean()
    return (xr @ fr) / np.sqrt(np.sum(xr * xr, axis=1) * np.sum(fr * fr))


def splitmix_uniform(seed: np.ndarray, n: int, salt: int) -> np.ndarray:
    """Vectorized deterministic U(0,1) keys, one frozen seed per row."""
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    base = seed.astype(np.uint64) ^ np.uint64(salt)
    out = np.empty((len(seed), n), dtype=float)
    for j in range(n):
        z = (base + np.uint64(0x9E3779B97F4A7C15) * np.uint64(j + 1)) & mask
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
        z = z ^ (z >> np.uint64(31))
        out[:, j] = (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
    return out


def run_coupling_null() -> None:
    verify_authority()
    primary = primary_crosswalk()
    rna = pd.read_csv(V6 / "04_rna/CELL3D_PRIMARY201_RNA_SCORE_MATRIX.tsv", sep="\t").set_index("sample_id").reindex(primary.sample_id)
    atac = pd.read_csv(V6 / "07_atac/CELL3D_PRIMARY201_PROMOTER_ACCESSIBILITY_SCORE.tsv", sep="\t").set_index("sample_id").reindex(primary.sample_id)
    stability = pd.read_csv(V6 / "07_atac/MODEL_RNA_ATAC_REGIONAL_STABILITY.tsv", sep="\t").set_index("model_id")
    source_patient = pd.read_csv(V6 / "07_atac/MODEL_PATIENT_RNA_ATAC_COUPLING.tsv", sep="\t")
    source_summary = pd.read_csv(V6 / "07_atac/MODEL_RNA_ATAC_COUPLING_SUMMARY.tsv", sep="\t").set_index("model_id")
    ids = sorted(set(rna.columns[1:]) & set(atac.columns[1:]))
    ids = [s for s in ids if rna[s].notna().all() and atac[s].notna().all() and s in stability.index]
    if len(ids) != 195:
        raise RuntimeError(f"COUPLING_UNIVERSE_REPRODUCTION_FAILURE:{len(ids)}")
    seeds = pd.read_csv(SEED_BANK, sep="\t").sort_values("null_world_id").rename(
        columns={"null_world_id": "replicate_id", "world_seed": "null_seed"})
    R = len(seeds)
    groups = primary.patient_id.astype(str).to_numpy()
    patients = sorted(set(groups))
    null_coupling = np.empty((R, len(ids)), dtype=float)
    observed = np.empty(len(ids), dtype=float)
    patient_rows = []
    for j, sid in enumerate(ids):
        pcorr = []
        null_patient = np.empty((R, len(patients)))
        for h, patient in enumerate(patients):
            ix = np.flatnonzero(groups == patient)
            xr = rankdata(rna[sid].to_numpy(float)[ix]); yr = rankdata(atac[sid].to_numpy(float)[ix])
            xc = xr - xr.mean(); yc = yr - yr.mean(); den = math.sqrt(float(xc @ xc) * float(yc @ yc))
            corr = float(xc @ yc / den)
            pcorr.append(corr)
            source = source_patient[(source_patient.model_id.eq(sid)) & (source_patient.patient_id.eq(patient))]
            if len(source) != 1 or not np.isclose(corr, source.iloc[0].rho_RNA_promoter_ATAC, atol=1e-12):
                raise RuntimeError(f"OBSERVED_PATIENT_COUPLING_REPRODUCTION_FAILURE:{sid}:{patient}")
            salt = int(hashlib.sha256(f"{sid}|{patient}".encode()).hexdigest()[:16], 16)
            # One independent permutation per frozen replicate; sorting deterministic U(0,1) keys is uniform.
            order = np.argsort(splitmix_uniform(seeds.null_seed.to_numpy(np.uint64), len(ix), salt), axis=1)
            null_patient[:, h] = (yc[order] @ xc) / den
            patient_rows.append({"model_id": sid, "patient_id": patient, "matched_region_n": len(ix), "reproduced_rho": corr})
        observed[j] = float(np.median(pcorr))
        if not np.isclose(observed[j], source_summary.loc[sid, "median_patient_RNA_ATAC_rho"], atol=1e-12):
            raise RuntimeError(f"OBSERVED_SIGNATURE_COUPLING_REPRODUCTION_FAILURE:{sid}")
        null_coupling[:, j] = np.median(null_patient, axis=1)
    write_tsv(pd.DataFrame(patient_rows), ROOT / "06_coupling_icc_retirement/OBSERVED_COUPLING_REPRODUCTION_BY_PATIENT.tsv")
    rna_icc = stability.loc[ids, "RNA_ICC"].to_numpy(float)
    atac_icc = stability.loc[ids, "ATAC_ICC"].to_numpy(float)
    obs_rna = float(spearmanr(observed, rna_icc).statistic)
    obs_atac = float(spearmanr(observed, atac_icc).statistic)
    null_rna = rank_corr_rows(null_coupling, rna_icc)
    null_atac = rank_corr_rows(null_coupling, atac_icc)
    world = pd.DataFrame({"replicate_id": seeds.replicate_id, "null_seed": seeds.null_seed,
                          "rho_null_coupling_RNA_ICC": null_rna, "rho_null_coupling_ATAC_ICC": null_atac})
    write_tsv(world, ROOT / "06_coupling_icc_retirement/COUPLING_ICC_DEFINITION_IMPLIED_NULL.tsv")
    signed_rna = max(0.0, math.copysign(1.0, obs_rna) * float(np.median(null_rna))) / abs(obs_rna)
    signed_atac = max(0.0, math.copysign(1.0, obs_atac) * float(np.median(null_atac))) / abs(obs_atac)
    reproduction = min(signed_rna, signed_atac)
    pattern = "RETIRED_DEFINITION_IMPLIED" if reproduction >= .70 else "RESIDUAL_NONTRIVIAL_COMPONENT_REQUIRES_USER_REVIEW"
    status = {"signature_n": len(ids), "patient_n": len(patients), "null_world_n": R,
              "observed_rho_coupling_RNA_ICC": obs_rna, "observed_rho_coupling_ATAC_ICC": obs_atac,
              "median_null_rho_RNA_ICC": float(np.median(null_rna)), "median_null_rho_ATAC_ICC": float(np.median(null_atac)),
              "signed_reproduction_fraction_RNA": signed_rna, "signed_reproduction_fraction_ATAC": signed_atac,
              "COUPLING_ICC_NULL_REPRODUCTION_FRACTION": reproduction, "retirement_threshold": .70,
              "COUPLING_STABILITY_PATTERN": pattern,
              "preserved": ["patient count", "regions per patient", "signature-specific marginal scores", "between/within-patient variance", "modality dynamic range", "publication and gene-set labels", "observed missingness"],
              "removed": "signature-specific within-patient regional RNA-ATAC pairing"}
    (ROOT / "06_coupling_icc_retirement/COUPLING_ICC_RETIREMENT_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ds03", "msbd", "v6-authority", "multimodal-census", "coupling-null"])
    args = parser.parse_args()
    {"ds03": run_ds03, "msbd": run_msbd, "v6-authority": run_v6_authority,
     "multimodal-census": run_multimodal_census, "coupling-null": run_coupling_null}[args.mode]()


if __name__ == "__main__":
    main()
