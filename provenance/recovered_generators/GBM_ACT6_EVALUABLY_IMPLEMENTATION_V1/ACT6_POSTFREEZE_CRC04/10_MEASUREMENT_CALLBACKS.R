# CRC04 protocol-local measurement, exact test, support, and firewall callbacks.
# These functions do not open files and cannot execute the protected real outcome.

crc04_genes <- c(
  "TYK2", "IL2RG", "KRT17", "HLA-B", "NPPC", "WIF1",
  "IL32", "B2M", "CCND1", "CRIP1", "ITGB1"
)

crc04_measure <- function(world) {
  required <- c("expression", "segment_metadata", "features")
  if (!is.list(world) || !all(required %in% names(world))) {
    stop("CRC04 world must contain expression, segment_metadata, and features.")
  }
  x <- world$expression
  md <- world$segment_metadata
  features <- as.character(world$features)
  genes <- c("TYK2", "IL2RG", "KRT17", "HLA-B", "NPPC", "WIF1",
             "IL32", "B2M", "CCND1", "CRIP1", "ITGB1")
  if (!is.matrix(x) || !is.numeric(x) || is.null(rownames(x)) || is.null(colnames(x))) {
    stop("expression must be a named numeric matrix.")
  }
  if (anyDuplicated(rownames(x)) || anyDuplicated(colnames(x))) {
    stop("expression row and column names must be unique.")
  }
  if (length(features) != 11L || anyDuplicated(features) || !setequal(features, genes)) {
    stop("The frozen 11-gene feature set must be present exactly once.")
  }
  if (!all(genes %in% rownames(x))) stop("A frozen CRC-BUD11 gene is missing.")
  if (any(!is.finite(x[genes, colnames(x), drop = FALSE]))) {
    stop("Eligible source-processed expression values must be finite.")
  }
  required_md <- c("segment_id", "roi_id", "patient_id", "region", "compartment",
                   "source_qc_pass", "source_included")
  if (!is.data.frame(md) || !identical(sort(names(md)), sort(required_md))) {
    stop("segment_metadata schema does not match the frozen whitelist.")
  }
  if (anyNA(md) || anyDuplicated(md$segment_id)) stop("Segment metadata must be complete and unique.")
  if (!is.logical(md$source_qc_pass) || !is.logical(md$source_included)) {
    stop("source_qc_pass and source_included must be logical.")
  }
  if (!setequal(as.character(md$segment_id), colnames(x))) {
    stop("Expression columns and segment metadata must identify the same segments.")
  }
  if (!all(as.character(md$region) %in% c("CORE", "INV", "NOR", "ADE", "BONUS"))) {
    stop("Unknown histopathological region label.")
  }
  if (!all(as.character(md$compartment) %in% c("PanCK+", "PanCK-"))) {
    stop("Unknown compartment label.")
  }

  keep <- md$source_qc_pass & md$source_included &
    as.character(md$compartment) == "PanCK+" &
    as.character(md$region) %in% c("CORE", "INV")
  eligible <- md[keep, required_md, drop = FALSE]
  if (!nrow(eligible)) stop("No eligible PanCK-positive core/front segments.")
  included_patients <- sort(unique(as.character(md$patient_id[md$source_included])))
  if (length(included_patients) < 5L) stop("Fewer than five source-included patients.")
  pair_grid <- expand.grid(patient_id = included_patients,
                           region = c("CORE", "INV"),
                           stringsAsFactors = FALSE)
  observed_keys <- unique(paste(eligible$patient_id, eligible$region, sep = "\r"))
  required_keys <- paste(pair_grid$patient_id, pair_grid$region, sep = "\r")
  if (!all(required_keys %in% observed_keys)) {
    stop("Every source-included patient must have both eligible epithelial regions.")
  }

  eligible <- eligible[order(eligible$patient_id, eligible$region,
                             eligible$roi_id, eligible$segment_id), required_md, drop = FALSE]
  sx <- x[genes, as.character(eligible$segment_id), drop = FALSE]
  segment_score <- colMeans(sx)
  patient_region_score <- numeric(length(required_keys))
  for (j in seq_along(required_keys)) {
    key <- required_keys[[j]]
    bits <- strsplit(key, "\r", fixed = TRUE)[[1L]]
    ids <- as.character(eligible$segment_id[
      as.character(eligible$patient_id) == bits[[1L]] &
        as.character(eligible$region) == bits[[2L]]])
    patient_region_score[[j]] <- stats::median(segment_score[ids])
  }
  names(patient_region_score) <- required_keys
  core <- patient_region_score[paste(included_patients, "CORE", sep = "\r")]
  front <- patient_region_score[paste(included_patients, "INV", sep = "\r")]
  d <- as.numeric(front - core)
  names(d) <- included_patients
  list(
    segment_score = segment_score,
    patient_region_score = patient_region_score,
    paired_differences = d,
    patient_weights = stats::setNames(rep(1, length(d)), included_patients),
    eligible_segment_metadata = eligible
  )
}

crc04_metric <- function(world, spec = NULL) {
  crc04_measure(world)$paired_differences
}

crc04_exact_p <- function(d) {
  d <- as.numeric(d)
  if (!length(d) || any(!is.finite(d))) stop("d must be a nonempty finite vector.")
  n <- length(d)
  if (n > 20L) stop("Exact CRC04 enumeration is capped at 20 patient pairs.")
  total <- 2^n
  observed <- sum(d)
  ge <- 0L
  for (mask in 0:(total - 1)) {
    signs <- ifelse(bitwAnd(mask, bitwShiftL(1L, 0:(n - 1L))) != 0L, 1, -1)
    ge <- ge + as.integer(sum(signs * d) >= observed)
  }
  ge / total
}

crc04_support <- function(spec, data) {
  if (!is.list(data) || is.null(data$paired_differences)) {
    stop("support_fn requires data$paired_differences.")
  }
  a <- abs(as.numeric(data$paired_differences))
  if (!length(a) || any(!is.finite(a))) stop("Paired differences must be finite.")
  n <- length(a)
  if (n > 20L) stop("Exact CRC04 support enumeration is capped at 20 patient pairs.")
  total <- 2^n
  observed_max <- sum(a)
  at_max <- 0L
  for (mask in 0:(total - 1)) {
    signs <- ifelse(bitwAnd(mask, bitwShiftL(1L, 0:(n - 1L))) != 0L, 1, -1)
    at_max <- at_max + as.integer(sum(signs * a) >= observed_max)
  }
  p_min <- at_max / total
  list(
    support = seq_len(total),
    min_attainable_p = p_min,
    exact_zero_count = sum(a == 0),
    nonzero_pair_count = sum(a != 0),
    support_cardinality = total,
    alpha_reachable = p_min <= spec$inferential_rule$alpha
  )
}

crc04_detect <- function(values, spec) {
  crc04_exact_p(values) <= spec$inferential_rule$alpha
}

crc04_generator <- function(accessible, severity, seed, constraints) {
  if (!is.numeric(severity) || length(severity) != 1L || severity != 1) {
    stop("Only the deterministic observed realization at severity 1 is permitted.")
  }
  if (!is.list(constraints) || length(constraints)) stop("CRC04 declares no generator constraints.")
  if (!is.null(seed) && length(seed) != 1L) stop("seed must be scalar when supplied.")
  allowed <- c("expression", "segment_metadata", "features")
  if (!is.list(accessible) || !identical(sort(names(accessible)), sort(allowed))) {
    stop("Generator input must contain exactly the accessible whitelist.")
  }
  list(status = "CONSTRUCTED", world = accessible[allowed])
}
