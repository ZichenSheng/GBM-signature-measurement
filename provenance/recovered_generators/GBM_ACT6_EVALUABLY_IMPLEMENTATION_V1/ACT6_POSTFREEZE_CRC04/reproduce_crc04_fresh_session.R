args <- commandArgs(trailingOnly = TRUE)
protocol_dir <- if (length(args)) normalizePath(args[[1L]]) else normalizePath(".")
workspace <- dirname(protocol_dir)
out_dir <- file.path(protocol_dir, "REAL_EXECUTION_2026-09-13")
.libPaths(c(normalizePath(file.path(workspace, "evaluably.Rcheck")), .libPaths()))
suppressPackageStartupMessages(library(evaluably))
suppressPackageStartupMessages(library(GeomxTools))
source(file.path(protocol_dir, "15_BUILD_CRC04_AUDIT_SPEC.R"))
b <- build_crc04(protocol_dir); s <- b$spec

sha256 <- function(path) digest::digest(file = path, algo = "sha256")
write_json <- function(x, path) jsonlite::write_json(x, path, auto_unbox = TRUE,
                                                     pretty = TRUE, digits = 17,
                                                     na = "null", null = "null")
cmp <- list()
add_cmp <- function(name, value) { cmp[[name]] <<- isTRUE(value); invisible(value) }

lock <- jsonlite::read_json(file.path(protocol_dir, "REAL_EXECUTION_LOCK.json"), simplifyVector = TRUE)
stopifnot(identical(lock$PRIMARY_EXECUTION_COUNT, 1L))
membership <- read.delim(file.path(protocol_dir, "05_WTA_ELIGIBLE_SEGMENT_MEMBERSHIP.tsv"),
                         stringsAsFactors = FALSE, check.names = FALSE)
obj <- readRDS(file.path(protocol_dir, "sources", "extracted", "Nanostring_GeoMx",
                         "exp2", "r_object", "geomx_exp2_target_Data_norm.RDS"))
assay <- Biobase::assayDataElement(obj, "log_q_combat")
genes <- s$representation_spec$feature_definition
x <- assay[genes, membership$segment_id, drop = FALSE]
md <- membership[, c("segment_id", "roi_id", "patient_id", "region", "compartment",
                     "source_qc_pass", "source_included"), drop = FALSE]
md$source_qc_pass <- as.logical(md$source_qc_pass)
md$source_included <- as.logical(md$source_included)
world <- list(expression = x, segment_metadata = md, features = genes)
measurement <- b$callbacks$crc04_measure(world)
d <- measurement$paired_differences
support <- b$callbacks$crc04_support(s, list(paired_differences = d))
exact_p <- if (support$alpha_reachable) b$callbacks$crc04_exact_p(d) else NA_real_
detection <- if (support$alpha_reachable) as.integer(exact_p <= s$inferential_rule$alpha) else NA_integer_
terminal_state <- if (!support$alpha_reachable) "NOT_EVALUABLE" else
  if (detection == 1L) "SUPPORTED_WITHIN_DOMAIN" else "UNRESOLVED"
observed_stat <- sum(d)

read_char_tsv <- function(path) read.delim(path, stringsAsFactors = FALSE,
                                           check.names = FALSE, colClasses = "character")
seg0 <- read_char_tsv(file.path(out_dir, "REAL_SEGMENT_SCORES.tsv"))
pr0 <- read_char_tsv(file.path(out_dir, "REAL_PATIENT_REGION_SCORES.tsv"))
pd0 <- read_char_tsv(file.path(out_dir, "REAL_PAIRED_DIFFERENCES.tsv"))
support0 <- jsonlite::read_json(file.path(out_dir, "REAL_SUPPORT_RECEIPT.json"), simplifyVector = TRUE)
exact0 <- jsonlite::read_json(file.path(out_dir, "REAL_EXACT_TEST_RECEIPT.json"), simplifyVector = TRUE)
decision0 <- jsonlite::read_json(file.path(out_dir, "REAL_DECISION_MAP_RECEIPT.json"), simplifyVector = TRUE)
result0 <- jsonlite::read_json(file.path(out_dir, "REAL_CLAIM_SCOPED_RESULT.json"), simplifyVector = TRUE)
record_rds <- readRDS(file.path(out_dir, "REAL_EVIDENCE_RECORD.rds"))
manifest0 <- jsonlite::read_json(file.path(out_dir, "REAL_EXECUTION_MANIFEST.json"), simplifyVector = TRUE)

seg_md <- measurement$eligible_segment_metadata
seg_ids <- as.character(seg_md$segment_id)
add_cmp("segment_ids", identical(seg0$segment_id, seg_ids))
add_cmp("segment_scores", identical(as.numeric(seg0$segment_score),
                                    unname(measurement$segment_score[seg_ids])))
patients <- names(d)
add_cmp("patient_ids", identical(pr0$patient_id, patients))
add_cmp("core_patient_region_scores", identical(as.numeric(pr0$core_patient_region_score),
  unname(measurement$patient_region_score[paste(patients, "CORE", sep = "\r")])))
add_cmp("invasive_front_patient_region_scores", identical(as.numeric(pr0$invasive_front_patient_region_score),
  unname(measurement$patient_region_score[paste(patients, "INV", sep = "\r")])))
add_cmp("paired_differences", identical(as.numeric(pd0$paired_difference), unname(d)))
add_cmp("zero_pattern", identical(as.numeric(pd0$paired_difference) == 0, unname(d) == 0))
add_cmp("support_nonzero", identical(as.integer(support0$N_NONZERO_DIFFERENCES),
                                     as.integer(support$nonzero_pair_count)))
add_cmp("support_zero", identical(as.integer(support0$N_EXACT_ZERO_DIFFERENCES),
                                  as.integer(support$exact_zero_count)))
add_cmp("minimum_attainable_p", identical(as.double(support0$MIN_ATTAINABLE_P),
                                          as.double(support$min_attainable_p)))
add_cmp("alpha_reachable", identical(support0$ALPHA_REACHABLE,
                                     if (support$alpha_reachable) "YES" else "NO"))
add_cmp("observed_test_statistic", identical(as.double(exact0$OBSERVED_TEST_STATISTIC),
                                             as.double(observed_stat)))
add_cmp("exact_p_value", if (support$alpha_reachable) identical(as.double(exact0$EXACT_P_VALUE),
                                                                as.double(exact_p)) else
  identical(exact0$exact_test_status, "NOT_RUN_ALPHA_UNREACHABLE"))
add_cmp("detection", if (support$alpha_reachable) identical(as.integer(exact0$DETECTION), detection) else TRUE)
add_cmp("terminal_state", identical(decision0$TERMINAL_STATE, terminal_state) &&
                                  identical(result0$TERMINAL_STATE, terminal_state))
add_cmp("evidence_record_rds_verifies", isTRUE(evaluably::verify_audit(record_rds)))
add_cmp("evidence_record_json_verifies", isTRUE(evaluably::verify_audit(
  file.path(out_dir, "REAL_EVIDENCE_RECORD.json"))))
manifest_files <- manifest0$files
record_ix <- which(manifest_files$file == "REAL_EVIDENCE_RECORD.json")
add_cmp("evidence_record_identity", length(record_ix) == 1L &&
  identical(manifest_files$sha256[[record_ix]], sha256(file.path(out_dir, "REAL_EVIDENCE_RECORD.json"))) &&
  identical(record_rds$provenance$record_scientific_hash, result0$record_scientific_hash) &&
  identical(record_rds$provenance$record_file_hash, result0$record_file_hash))
add_cmp("primary_execution_count_remains_one", identical(lock$PRIMARY_EXECUTION_COUNT, 1L))

status <- if (all(unlist(cmp, use.names = FALSE))) "PASS" else "FAIL"
receipt <- list(
  CASE_ID = "ACT6-HO-CRC-04", reproduction_process = "FRESH_R_SESSION",
  primary_execution_repeated = FALSE, immutable_primary_outputs_modified = FALSE,
  FRESH_SESSION_REPRODUCTION_STATUS = status,
  comparisons = cmp, n_segments = nrow(seg0), n_complete_pairs = length(d),
  N_NONZERO_DIFFERENCES = support$nonzero_pair_count,
  N_EXACT_ZERO_DIFFERENCES = support$exact_zero_count,
  MIN_ATTAINABLE_P = support$min_attainable_p,
  OBSERVED_TEST_STATISTIC = observed_stat,
  EXACT_P_VALUE = if (is.na(exact_p)) NULL else exact_p,
  DETECTION = if (is.na(detection)) NULL else detection,
  TERMINAL_STATE = terminal_state,
  evidence_record_json_sha256 = sha256(file.path(out_dir, "REAL_EVIDENCE_RECORD.json")),
  NO_RESCUE_ANALYSIS_PERFORMED = "YES", SECOND_PRIMARY_EXECUTION_PERFORMED = "NO")
write_json(receipt, file.path(protocol_dir, "CRC04_FRESH_SESSION_REPRODUCTION.json"))
lines <- c("# CRC04 fresh-session reproduction", "",
           paste0("FRESH_SESSION_REPRODUCTION_STATUS=", status),
           "PRIMARY_EXECUTION_REPEATED=NO", "IMMUTABLE_PRIMARY_OUTPUTS_MODIFIED=NO", "",
           "## Comparisons", "",
           vapply(names(cmp), function(nm) paste0("- ", nm, "=", if (cmp[[nm]]) "PASS" else "FAIL"), character(1L)),
           "", "NO_RESCUE_ANALYSIS_PERFORMED=YES", "SECOND_PRIMARY_EXECUTION_PERFORMED=NO")
writeLines(lines, file.path(protocol_dir, "CRC04_FRESH_SESSION_REPRODUCTION.md"), useBytes = TRUE)
cat("FRESH_SESSION_REPRODUCTION_STATUS=", status, "\n", sep = "")
if (status != "PASS") quit(status = 1L)
