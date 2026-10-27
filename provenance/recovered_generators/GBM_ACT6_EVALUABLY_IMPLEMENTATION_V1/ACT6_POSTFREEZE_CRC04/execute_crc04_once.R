args <- commandArgs(trailingOnly = TRUE)
protocol_dir <- if (length(args)) normalizePath(args[[1L]]) else normalizePath(".")
workspace <- dirname(protocol_dir)
out_dir <- file.path(protocol_dir, "REAL_EXECUTION_2026-09-13")
lock_dir <- file.path(protocol_dir, "CRC04_REAL_EXECUTION_CLAIMED")
approval_path <- file.path(protocol_dir, "REAL_EXECUTION_APPROVAL.json")
review_path <- file.path(protocol_dir, "ACT6_CRC04_FINAL_PROTOCOL_FREEZE_REVIEW_2026-09-13.md")
archive_path <- file.path(protocol_dir, "sources", "Nanostring_GeoMx.zip")
object_path <- file.path(protocol_dir, "sources", "extracted", "Nanostring_GeoMx",
                         "exp2", "r_object", "geomx_exp2_target_Data_norm.RDS")

EXPECTED <- list(
  review_sha256 = "191d0ecdc10c0e74ab612112d77db5d5cff5168181f4b98586562aa2e90a7f2d",
  spec_hash = "4a5504e350edac6376468f6d580d9290bb9f77fcaeb1397fe2043d0e134bd833",
  audit_member_key = "5b09abf0c491f4ce6451e9f73a653a2e7f516c8be044d4c6016c790a0e3d51cd",
  archive_md5 = "dfe8b4548d394bce378919aeb069920c",
  archive_sha256 = "da92403e7fe4381d1cc8c7f884ea71b7f461a3e880183924ccd5cfe5496681c4",
  object_sha256 = "5c4eea69b73a0244f0b475265dae93db0ca0806fd06606419c5de4d0bdab968a",
  membership_sha256 = "f41e8048989584d3a8d135571afb9277ff0d788ce8c566435ecccf4f7470642e"
)

.libPaths(c(normalizePath(file.path(workspace, "evaluably.Rcheck")), .libPaths()))
suppressPackageStartupMessages(library(evaluably))
suppressPackageStartupMessages(library(GeomxTools))

sha256 <- function(path) digest::digest(file = path, algo = "sha256")
md5 <- function(path) unname(tools::md5sum(path))
now <- function() format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
write_json <- function(x, path) jsonlite::write_json(x, path, auto_unbox = TRUE,
                                                     pretty = TRUE, digits = 17,
                                                     na = "null", null = "null")
write_md <- function(lines, path) writeLines(lines, path, useBytes = TRUE)
write_tsv17 <- function(x, path, numeric_cols = character()) {
  y <- x
  for (nm in intersect(numeric_cols, names(y))) {
    y[[nm]] <- vapply(y[[nm]], function(z) {
      if (is.na(z)) "NA" else sprintf("%.17g", as.double(z))
    }, character(1L))
  }
  utils::write.table(y, path, sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
}
log_line <- function(...) {
  cat(paste0(now(), " ", paste0(..., collapse = ""), "\n"),
      file = file.path(out_dir, "REAL_EXECUTION_LOG.txt"), append = TRUE)
}

# Fail-closed pre-lock checks. None accesses an assay matrix element.
stopifnot(file.exists(approval_path), !dir.exists(lock_dir), !dir.exists(out_dir))
approval <- jsonlite::read_json(approval_path, simplifyVector = TRUE)
stopifnot(identical(approval$reviewer, "Claude_final_independent_CRC04_protocol_freeze_review"))
stopifnot(identical(approval$review_sha256, EXPECTED$review_sha256))
stopifnot(identical(approval$PROTOCOL_FROZEN, "YES"))
stopifnot(identical(approval$OUTCOME_EXECUTION_AUTHORIZED, "YES"))
stopifnot(identical(approval$CORRECTED_REVIEW_HASH_AUTHORIZATION, "YES"))
stopifnot(identical(approval$PREVIOUS_HASH_BLOCKER_RESOLVED, "YES"))
stopifnot(identical(approval$frozen_spec_hash, EXPECTED$spec_hash))
stopifnot(identical(approval$audit_member_key, EXPECTED$audit_member_key))
stopifnot(identical(approval$primary_executor_sha256,
                    sha256(file.path(protocol_dir, "execute_crc04_once.R"))))
stopifnot(identical(approval$fresh_session_reproducer_sha256,
                    sha256(file.path(protocol_dir, "reproduce_crc04_fresh_session.R"))))
stopifnot(identical(approval$result_packager_sha256,
                    sha256(file.path(protocol_dir, "package_crc04_result.py"))))
stopifnot(identical(sha256(review_path), EXPECTED$review_sha256))
stopifnot(identical(sha256(archive_path), EXPECTED$archive_sha256))
stopifnot(identical(md5(archive_path), EXPECTED$archive_md5))
stopifnot(identical(sha256(object_path), EXPECTED$object_sha256))
stopifnot(identical(sha256(file.path(protocol_dir, "05_WTA_ELIGIBLE_SEGMENT_MEMBERSHIP.tsv")),
                    EXPECTED$membership_sha256))

source(file.path(protocol_dir, "15_BUILD_CRC04_AUDIT_SPEC.R"))
b <- build_crc04(protocol_dir)
s <- b$spec
stopifnot(isTRUE(evaluably::verify_audit(s)))
stopifnot(identical(s$spec_hash, EXPECTED$spec_hash))
stopifnot(identical(s$audit_member_key, EXPECTED$audit_member_key))

# The atomic directory claim is the irreversible start of the one authorized run.
if (!dir.create(lock_dir, showWarnings = FALSE)) stop("Single-use execution lock already exists.")
dir.create(out_dir, showWarnings = FALSE)
lock <- list(
  CASE_ID = "ACT6-HO-CRC-04",
  lock_type = "ATOMIC_SINGLE_USE_DIRECTORY_CLAIM",
  lock_directory = lock_dir,
  started_at = now(),
  primary_execution_count_before_lock = 0L,
  PRIMARY_EXECUTION_COUNT = 1L,
  target_access_started_at_lock_creation = FALSE,
  NO_SECOND_PRIMARY_EXECUTION = TRUE,
  NO_RETRY_AFTER_RESULT_ACCESS = TRUE,
  NO_RESCUE = TRUE,
  approval_sha256 = sha256(approval_path),
  review_sha256 = EXPECTED$review_sha256,
  frozen_spec_hash = EXPECTED$spec_hash,
  audit_member_key = EXPECTED$audit_member_key)
write_json(lock, file.path(lock_dir, "claim.json"))
write_json(lock, file.path(protocol_dir, "REAL_EXECUTION_LOCK.json"))
file.copy(approval_path, file.path(out_dir, "REAL_EXECUTION_APPROVAL.json"), overwrite = FALSE)
file.copy(file.path(protocol_dir, "REAL_EXECUTION_LOCK.json"),
          file.path(out_dir, "REAL_EXECUTION_LOCK.json"), overwrite = FALSE)
writeLines(character(), file.path(out_dir, "REAL_EXECUTION_LOG.txt"))
log_line("PRIMARY_EXECUTION_STARTED; PRIMARY_EXECUTION_COUNT=1; ",
         "NO_SECOND_PRIMARY_EXECUTION=YES; NO_RESCUE=YES")

failed <- TRUE
on.exit({
  if (failed) {
    failure <- list(CASE_ID = "ACT6-HO-CRC-04", execution_status = "TECHNICAL_FAILURE_AFTER_LOCK",
                    PRIMARY_EXECUTION_COUNT = 1L, rerun_permitted = FALSE,
                    NO_SECOND_PRIMARY_EXECUTION = TRUE, NO_RESCUE = TRUE,
                    recorded_at = now())
    try(write_json(failure, file.path(out_dir, "REAL_TECHNICAL_FAILURE_RECEIPT.json")), silent = TRUE)
  }
}, add = TRUE)

input_receipt <- list(
  CASE_ID = "ACT6-HO-CRC-04", zenodo_record = "17671259",
  authoritative_archive = "Nanostring_GeoMx.zip",
  input_archive_md5 = md5(archive_path), input_archive_sha256 = sha256(archive_path),
  processed_object = "Nanostring_GeoMx/exp2/r_object/geomx_exp2_target_Data_norm.RDS",
  processed_object_sha256 = sha256(object_path), assay_element = "log_q_combat",
  review_sha256 = sha256(review_path), frozen_spec_hash = s$spec_hash,
  audit_member_key = s$audit_member_key, preexecution_hash_check = "PASS",
  input_identity_check = "PASS", renormalization_performed = FALSE)
write_json(input_receipt, file.path(out_dir, "REAL_INPUT_IDENTITY_RECEIPT.json"))
write_md(c("# Real input identity receipt", "",
           "- CASE_ID=ACT6-HO-CRC-04",
           "- ZENODO_RECORD=17671259",
           paste0("- INPUT_ARCHIVE_MD5=", input_receipt$input_archive_md5),
           paste0("- INPUT_ARCHIVE_SHA256=", input_receipt$input_archive_sha256),
           paste0("- PROCESSED_OBJECT_SHA256=", input_receipt$processed_object_sha256),
           "- ASSAY_ELEMENT=log_q_combat", "- PREEXECUTION_HASH_CHECK=PASS",
           "- INPUT_IDENTITY_CHECK=PASS", "- RENORMALIZATION_PERFORMED=NO"),
         file.path(out_dir, "REAL_INPUT_IDENTITY_RECEIPT.md"))

# First command that can access the eleven target-gene expression values.
log_line("IMMEDIATELY_BEFORE_TARGET_ACCESS; PRIMARY_EXECUTION_COUNT=1; ",
         "ASSAY_ELEMENT=log_q_combat")
obj <- readRDS(object_path)
full_assay <- Biobase::assayDataElement(obj, "log_q_combat")
log_line("REAL_TARGET_VALUES_ACCESSED=YES")

membership <- utils::read.delim(file.path(protocol_dir, "05_WTA_ELIGIBLE_SEGMENT_MEMBERSHIP.tsv"),
                                stringsAsFactors = FALSE, check.names = FALSE)
stopifnot(nrow(membership) == 85L, length(unique(membership$patient_id)) == 9L,
          all(membership$source_qc_pass), all(membership$source_included))
genes <- b$spec$representation_spec$feature_definition
stopifnot(length(genes) == 11L, identical(genes,
  c("TYK2", "IL2RG", "KRT17", "HLA-B", "NPPC", "WIF1", "IL32", "B2M", "CCND1", "CRIP1", "ITGB1")))
stopifnot(all(genes %in% rownames(full_assay)), all(membership$segment_id %in% colnames(full_assay)))
x <- full_assay[genes, membership$segment_id, drop = FALSE]
rm(full_assay)
md <- membership[, c("segment_id", "roi_id", "patient_id", "region", "compartment",
                     "source_qc_pass", "source_included"), drop = FALSE]
md$source_qc_pass <- as.logical(md$source_qc_pass)
md$source_included <- as.logical(md$source_included)
data <- list(
  expression = x, segment_metadata = md, features = genes,
  lymph_node_status = rep("PROTECTED_NOT_ACCESSED", 9L),
  distant_metastasis_status = rep("PROTECTED_NOT_ACCESSED", 9L),
  recurrence = rep(NA_character_, 9L), survival = rep(NA_real_, 9L),
  tumour_budding_grade = rep(NA_character_, 9L), other_clinical_outcome = list(),
  id_map = b$units)

measurement <- b$callbacks$crc04_measure(data)
data$paired_differences <- measurement$paired_differences
d <- measurement$paired_differences

seg_md <- measurement$eligible_segment_metadata
segment_out <- data.frame(
  segment_id = as.character(seg_md$segment_id), patient_id = as.character(seg_md$patient_id),
  region = ifelse(as.character(seg_md$region) == "INV", "INVASIVE_FRONT", "CORE"),
  segment_score = unname(measurement$segment_score[as.character(seg_md$segment_id)]),
  stringsAsFactors = FALSE)
patients <- names(d)
core_keys <- paste(patients, "CORE", sep = "\r")
inv_keys <- paste(patients, "INV", sep = "\r")
patient_out <- data.frame(
  patient_id = patients,
  core_patient_region_score = unname(measurement$patient_region_score[core_keys]),
  invasive_front_patient_region_score = unname(measurement$patient_region_score[inv_keys]),
  paired_difference = unname(d), stringsAsFactors = FALSE)
paired_out <- patient_out[, c("patient_id", "paired_difference"), drop = FALSE]
write_tsv17(segment_out, file.path(out_dir, "REAL_SEGMENT_SCORES.tsv"), "segment_score")
write_tsv17(patient_out, file.path(out_dir, "REAL_PATIENT_REGION_SCORES.tsv"),
            c("core_patient_region_score", "invasive_front_patient_region_score", "paired_difference"))
write_tsv17(paired_out, file.path(out_dir, "REAL_PAIRED_DIFFERENCES.tsv"), "paired_difference")

# Frozen support/reachability gate precedes the exact detection computation.
support <- b$callbacks$crc04_support(s, list(paired_differences = d))
support_receipt <- list(
  CASE_ID = "ACT6-HO-CRC-04", support_function_hash = s$inferential_rule$support_fn_hash,
  support_basis = "ABSOLUTE_PAIRED_DIFFERENCES_AND_EXACT_ZERO_PATTERN",
  sign_invariant = TRUE, n_pairs = length(d),
  N_NONZERO_DIFFERENCES = support$nonzero_pair_count,
  N_EXACT_ZERO_DIFFERENCES = support$exact_zero_count,
  MIN_ATTAINABLE_P = support$min_attainable_p,
  ALPHA = s$inferential_rule$alpha,
  ALPHA_REACHABLE = if (support$alpha_reachable) "YES" else "NO",
  SUPPORT_GATE_STATUS = if (support$alpha_reachable) "PASS" else "ALPHA_UNREACHABLE")
write_json(support_receipt, file.path(out_dir, "REAL_SUPPORT_RECEIPT.json"))
write_md(c("# Real support receipt", "",
           paste0("N_NONZERO_DIFFERENCES=", support$nonzero_pair_count),
           paste0("N_EXACT_ZERO_DIFFERENCES=", support$exact_zero_count),
           paste0("MIN_ATTAINABLE_P=", sprintf("%.17g", support$min_attainable_p)),
           paste0("ALPHA_REACHABLE=", support_receipt$ALPHA_REACHABLE),
           paste0("SUPPORT_GATE_STATUS=", support_receipt$SUPPORT_GATE_STATUS)),
         file.path(out_dir, "REAL_SUPPORT_RECEIPT.md"))

check <- evaluably::check_audit(s, data, attempts = 1L)
if (support$alpha_reachable) {
  stopifnot(identical(check$entry_gate, "PASS"))
  run <- evaluably::run_audit(check, data)
  result <- evaluably::apply_decision_map(run)
  exact_p <- b$callbacks$crc04_exact_p(d)
  detection <- as.integer(exact_p <= s$inferential_rule$alpha)
  stopifnot(identical(as.integer(run$recovery$detections), detection))
  record <- evaluably::evidence_record(run, render = FALSE)
  exact_status <- "EXECUTED"
} else {
  stopifnot(identical(check$entry_gate, "BLOCKED"),
            identical(check$support_gate, "ALPHA_UNREACHABLE"))
  result <- evaluably::apply_decision_map(check)
  exact_p <- NA_real_
  detection <- NA_integer_
  record <- evaluably::evidence_record(check, render = FALSE)
  exact_status <- "NOT_RUN_ALPHA_UNREACHABLE"
}
stopifnot(isTRUE(evaluably::verify_audit(record)))
terminal_state <- as.character(result$terminal_state)
if (support$alpha_reachable) {
  stopifnot(terminal_state %in% c("SUPPORTED_WITHIN_DOMAIN", "UNRESOLVED"))
  stopifnot(identical(terminal_state,
                      if (detection == 1L) "SUPPORTED_WITHIN_DOMAIN" else "UNRESOLVED"))
} else stopifnot(identical(terminal_state, "NOT_EVALUABLE"))

observed_stat <- sum(d)
observed_direction <- if (observed_stat > 0) "INVASIVE_FRONT_GREATER" else
  if (observed_stat < 0) "TUMOUR_CORE_GREATER" else "ZERO_TOTAL_DIFFERENCE"
exact_receipt <- list(
  CASE_ID = "ACT6-HO-CRC-04", exact_test_status = exact_status,
  TEST_FAMILY = "PERMUTATION_EXACT", NULL_SUPPORT = "PAIRED_SIGN_FLIP",
  ALTERNATIVE = "ONE_SIDED_GREATER", DIRECTION = "INVASIVE_FRONT_GREATER",
  P_VALUE_CONVENTION = "INCLUSIVE_OBSERVED", ALPHA = 0.05,
  n_pairs = length(d), full_support_cardinality = 2^length(d),
  OBSERVED_TEST_STATISTIC = observed_stat,
  EXACT_P_VALUE = if (is.na(exact_p)) NULL else exact_p,
  DETECTION = if (is.na(detection)) NULL else detection,
  exact_test_engine_hash = unname(s$stable_flag_rule_depends_on$functions[["crc04_exact_p"]]))
write_json(exact_receipt, file.path(out_dir, "REAL_EXACT_TEST_RECEIPT.json"))
write_md(c("# Real exact-test receipt", "",
           paste0("EXACT_TEST_STATUS=", exact_status),
           "TEST_FAMILY=PERMUTATION_EXACT", "NULL_SUPPORT=PAIRED_SIGN_FLIP",
           "ALTERNATIVE=ONE_SIDED_GREATER", "P_VALUE_CONVENTION=INCLUSIVE_OBSERVED",
           paste0("OBSERVED_TEST_STATISTIC=", sprintf("%.17g", observed_stat)),
           paste0("EXACT_P_VALUE=", if (is.na(exact_p)) "NOT_RUN" else sprintf("%.17g", exact_p)),
           paste0("DETECTION=", if (is.na(detection)) "NOT_ASSESSED" else detection)),
         file.path(out_dir, "REAL_EXACT_TEST_RECEIPT.md"))

decision_receipt <- list(
  CASE_ID = "ACT6-HO-CRC-04", decision_map_hash = s$decision_map_hash,
  P3_rule_count = 2L, P3_states = c("SUPPORTED_WITHIN_DOMAIN", "UNRESOLVED"),
  applied_rule = if (is.na(result$rule_id)) NULL else result$rule_id,
  precedence_rule_fired = result$precedence_rule_fired,
  DETECTION = if (is.na(detection)) NULL else detection,
  TERMINAL_STATE = terminal_state)
write_json(decision_receipt, file.path(out_dir, "REAL_DECISION_MAP_RECEIPT.json"))
write_md(c("# Real decision-map receipt", "", "P3_RULE_COUNT=2",
           "P3_STATES=SUPPORTED_WITHIN_DOMAIN|UNRESOLVED",
           paste0("APPLIED_RULE=", if (is.na(result$rule_id)) "ENTRY_GATE" else result$rule_id),
           paste0("TERMINAL_STATE=", terminal_state)),
         file.path(out_dir, "REAL_DECISION_MAP_RECEIPT.md"))

record <- evaluably::render_audit(record)
stopifnot(isTRUE(evaluably::verify_audit(record)))
tmp_record_dir <- file.path(out_dir, ".record_render")
dir.create(tmp_record_dir)
invisible(evaluably::render_audit(record, path = tmp_record_dir, formats = c("json", "md")))
stopifnot(file.rename(file.path(tmp_record_dir, "audit_record.json"),
                      file.path(out_dir, "REAL_EVIDENCE_RECORD.json")))
stopifnot(file.rename(file.path(tmp_record_dir, "audit_record.md"),
                      file.path(out_dir, "REAL_EVIDENCE_RECORD.md")))
unlink(tmp_record_dir, recursive = TRUE)
saveRDS(record, file.path(out_dir, "REAL_EVIDENCE_RECORD.rds"), version = 3)

interpretation <- if (terminal_state == "SUPPORTED_WITHIN_DOMAIN") {
  paste("The frozen derived CRC-BUD11 representation showed patient-level directional domain",
        "information distinguishing PanCK-positive epithelial invasive-front from tumour-core",
        "expression in the Nature 2026 GeoMx WTA cohort under the prespecified paired exact design.")
} else if (terminal_state == "UNRESOLVED") {
  paste("The prespecified directional domain-information claim was not demonstrated under",
        "the frozen CRC04 representation, cohort, and inferential design.")
} else {
  paste("The frozen support gate made alpha unreachable under the realized exact-zero pattern;",
        "this is not evidence of biological absence.")
}
evidence_summary <- list(
  CASE_ID = "ACT6-HO-CRC-04", SUBJECT_ID = "CRC-BUD11",
  CLAIM_CLASS = "DOMAIN_INFORMATIVENESS", PRIMARY_EXECUTION_COUNT = 1L,
  n_patients = length(patients), n_segments = nrow(segment_out), n_complete_pairs = length(d),
  target_gene_count = 11L, target_genes_used = 11L,
  paired_differences = as.list(stats::setNames(as.numeric(d), names(d))),
  N_NONZERO_DIFFERENCES = support$nonzero_pair_count,
  N_EXACT_ZERO_DIFFERENCES = support$exact_zero_count,
  MIN_ATTAINABLE_P = support$min_attainable_p,
  ALPHA_REACHABLE = support_receipt$ALPHA_REACHABLE,
  OBSERVED_DIRECTION = observed_direction, OBSERVED_TEST_STATISTIC = observed_stat,
  EXACT_P_VALUE = if (is.na(exact_p)) NULL else exact_p,
  DETECTION = if (is.na(detection)) NULL else detection,
  TERMINAL_STATE = terminal_state, interpretation = interpretation,
  record_scientific_hash = record$provenance$record_scientific_hash,
  record_file_hash = record$provenance$record_file_hash,
  NO_RESCUE_ANALYSIS_PERFORMED = "YES", SECOND_PRIMARY_EXECUTION_PERFORMED = "NO")
write_json(evidence_summary, file.path(out_dir, "REAL_CLAIM_SCOPED_RESULT.json"))
write_md(c("# CRC04 claim-scoped result", "", interpretation, "",
           paste0("TERMINAL_STATE=", terminal_state),
           "NO_RESCUE_ANALYSIS_PERFORMED=YES", "SECOND_PRIMARY_EXECUTION_PERFORMED=NO"),
         file.path(out_dir, "REAL_CLAIM_SCOPED_RESULT.md"))

capture.output(sessionInfo(), file = file.path(out_dir, "REAL_SESSION_INFO.txt"))
log_line("PRIMARY_EXECUTION_COMPLETED; TERMINAL_STATE=", terminal_state,
         "; NO_RESCUE_ANALYSIS_PERFORMED=YES; SECOND_PRIMARY_EXECUTION_PERFORMED=NO")

manifest_candidates <- sort(setdiff(list.files(out_dir, full.names = TRUE),
                                    c(file.path(out_dir, "REAL_EXECUTION_MANIFEST.json"),
                                      file.path(out_dir, "REAL_EXECUTION_HASHES.tsv"))))
manifest_files <- lapply(manifest_candidates, function(path) list(
  file = basename(path), bytes = unname(file.info(path)$size), sha256 = sha256(path)))
manifest <- list(
  CASE_ID = "ACT6-HO-CRC-04", manifest_version = "1.0.0",
  PRIMARY_EXECUTION_COUNT = 1L, execution_status = "PASS",
  terminal_state = terminal_state, frozen_spec_hash = s$spec_hash,
  audit_member_key = s$audit_member_key, review_sha256 = EXPECTED$review_sha256,
  files = manifest_files)
write_json(manifest, file.path(out_dir, "REAL_EXECUTION_MANIFEST.json"))
hash_paths <- sort(setdiff(list.files(out_dir, full.names = TRUE),
                          file.path(out_dir, "REAL_EXECUTION_HASHES.tsv")))
hash_table <- data.frame(file = basename(hash_paths),
                         sha256 = vapply(hash_paths, sha256, character(1L)),
                         stringsAsFactors = FALSE)
utils::write.table(hash_table, file.path(out_dir, "REAL_EXECUTION_HASHES.tsv"),
                   sep = "\t", row.names = FALSE, quote = FALSE)

failed <- FALSE
cat("PRIMARY_EXECUTION_STATUS=PASS\n")
cat("PRIMARY_EXECUTION_COUNT=1\n")
cat("TERMINAL_STATE=", terminal_state, "\n", sep = "")
