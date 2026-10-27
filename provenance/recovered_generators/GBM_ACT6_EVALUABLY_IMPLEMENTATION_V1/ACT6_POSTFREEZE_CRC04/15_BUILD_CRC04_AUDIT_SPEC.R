# Construction only. This script never opens the real expression object or runs a real audit.
build_crc04 <- function(protocol_dir = ".") {
  protocol_dir <- normalizePath(protocol_dir)
  e <- new.env(parent = baseenv())
  sys.source(file.path(protocol_dir, "10_MEASUREMENT_CALLBACKS.R"), envir = e)
  null_deps <- list(packages = character(), constants = list(), functions = list())
  metric_deps <- list(packages = character(), constants = list(), functions = list(
    crc04_measure = list(fn = e$crc04_measure, depends_on = null_deps)))
  detect_deps <- list(packages = character(), constants = list(), functions = list(
    crc04_exact_p = list(fn = e$crc04_exact_p, depends_on = null_deps)))

  rep_json <- jsonlite::read_json(
    file.path(protocol_dir, "10_REPRESENTATION_DECLARATION.json"),
    simplifyVector = TRUE)
  rep_json$feature_mapping <- as.data.frame(rep_json$feature_mapping,
                                             stringsAsFactors = FALSE)
  representation <- do.call(evaluably::representation_spec, rep_json)

  membership <- utils::read.delim(
    file.path(protocol_dir, "05_WTA_ELIGIBLE_SEGMENT_MEMBERSHIP.tsv"),
    stringsAsFactors = FALSE, check.names = FALSE)
  id_map <- data.frame(
    panck_positive_epithelial_segment = membership$segment_id,
    patient_region = membership$patient_region_id,
    patient = membership$patient_id,
    stringsAsFactors = FALSE, check.names = FALSE)
  nesting <- data.frame(
    child = c("panck_positive_epithelial_segment", "patient_region"),
    parent = c("patient_region", "patient"), stringsAsFactors = FALSE)
  units <- evaluably::unit_spec(
    scientific = "patient",
    computational = "panck_positive_epithelial_segment",
    resampling = "patient",
    blocking = "patient",
    nesting = nesting,
    id_map = id_map,
    min_scientific_units = 5L,
    on_violation = "error")

  reference <- evaluably::reference_spec(
    type = "NO_VALID_REFERENCE", exactness = "EMPIRICAL", fn = NULL,
    source = paste(
      "No external signature is used. The independent endpoint is the source",
      "author/pathologist-defined histopathological CORE versus INV region label;",
      "EMPIRICAL is required released-API vocabulary and does not create a reference."),
    nuisance_preserved = character(), version = "CRC04-1.0.0",
    doi = "10.1038/s41586-026-10344-7")

  alternative <- evaluably::alternative_spec(
    name = "CRC04_OBSERVED_REALISATION_TRUTH_FIREWALL_ONLY",
    fn = e$crc04_generator,
    generator_accessible = c("expression", "segment_metadata", "features"),
    protected_evaluation_truth = c(
      "lymph_node_status", "distant_metastasis_status", "recurrence",
      "survival", "tumour_budding_grade", "other_clinical_outcome"),
    protected_assignment = c(
      "lymph_node_status", "distant_metastasis_status", "recurrence",
      "survival", "tumour_budding_grade", "other_clinical_outcome"),
    truth_dependence_declared = FALSE,
    truth_dependence_rationale = NULL,
    constraints = list(), capacity_fn = NULL,
    claims_capacity_check = FALSE,
    generator_fn_depends_on = null_deps)

  inference <- evaluably::inferential_rule(
    alpha = 0.05,
    test_family = "PERMUTATION_EXACT",
    alternative = "ONE_SIDED_GREATER",
    pvalue_convention = "INCLUSIVE_OBSERVED",
    exactness = "EXACT",
    reachability_basis = "EXACT_ENUMERATION",
    support_fn = e$crc04_support,
    justification = "Exact patient-level paired sign flips; reachability uses abs(d), exact-zero pattern, and pair count only.",
    support_fn_depends_on = null_deps)

  rules <- list(
    list(id = "R1", priority = 10L, when = "detections == 1L",
         then = "SUPPORTED_WITHIN_DOMAIN",
         because = "The inclusive exact paired sign-flip p-value is at or below 0.05 in the frozen invasive-front-greater direction."),
    list(id = "R2", priority = 20L, when = "detections == 0L",
         then = "UNRESOLVED",
         because = "Directional information was not demonstrated; reversal is not converted into an opposite-direction claim and biological absence is not established."))
  decisions <- evaluably::decision_map(rules, version = "CRC04-1.0.0")

  claim_text <- paste(
    "Within the PanCK-positive epithelial compartment of early-stage colorectal cancer,",
    "a frozen derived CRC-BUD11 representation carries patient-level information",
    "distinguishing invasive-front from matched tumour-core expression in the Nature 2026 GeoMx WTA cohort.")
  claim <- list(
    claim_id = "ACT6-HO-CRC-04",
    claim_family = "CRC04_ANATOMICAL_DOMAIN_INFORMATION",
    text = claim_text,
    estimand = "Mean across equally weighted patients of front-minus-core differences in median segment-level 11-gene mean scores",
    scope = "Nine source-included T1 patients; source-QC-passing PanCK-positive epithelial CORE and INV segments; Nature 2026 GeoMx WTA only")

  archive_sha256 <- "da92403e7fe4381d1cc8c7f884ea71b7f461a3e880183924ccd5cfe5496681c4"
  spec <- evaluably::audit_spec(
    subject = list(
      subject_id = "CRC-BUD11",
      subject_label = "Frozen eleven-gene colorectal tumour-budding signature membership",
      subject_provenance = "CRC-BUD11 source biology and invasive-front origin fixed independently of CRC04"),
    claim = claim,
    claim_class = "DOMAIN_INFORMATIVENESS",
    context = list(
      context_id = "NATURE2026_GEOMX_WTA_T1_PANCK_POS_INV_VS_CORE",
      description = claim$scope),
    endpoint_compatibility_declared = "SUPPORTED",
    metric_fn = e$crc04_metric,
    metric_direction = "greater",
    severity_grid = 1,
    representation_spec = representation,
    unit_spec = units,
    reference_spec = reference,
    alternative_spec = alternative,
    inferential_rule = inference,
    decision_map = decisions,
    n_worlds = 1L,
    metric_version = "CRC04-1.0.0",
    metric_fn_depends_on = metric_deps,
    min_constructibility = NULL,
    stable_flag_rule = e$crc04_detect,
    stable_flag_rule_depends_on = detect_deps,
    thresholds = list(
      target_gene_count = 11L,
      min_complete_patients = 5L,
      source_included_complete_patients = 9L,
      source_processed_assay = "log_q_combat",
      segment_gene_aggregation = "ARITHMETIC_MEAN",
      patient_region_aggregation = "MEDIAN",
      direction = "INVASIVE_FRONT_GREATER"),
    pairing = "paired",
    replacement_policy = "none",
    max_replacements = 0L,
    notes = paste(
      "Observed-realisation constructor only; no counterfactual biology.",
      "All nine source-included T1 patients require complete PanCK-positive CORE/INV geometry.",
      "No gene/segment/patient dropping, assay switch, renormalization, clinical adjustment, or alternative test."),
    source_locators = list(
      zenodo_record = "https://zenodo.org/records/17671259",
      archive = "Nanostring_GeoMx.zip",
      object = "Nanostring_GeoMx/exp2/r_object/geomx_exp2_target_Data_norm.RDS",
      assay_element = "log_q_combat"),
    source_sha256 = archive_sha256,
    scientific_replicates = 9L,
    audit_id = "ACT6-HO-CRC-04")
  spec <- evaluably::freeze_audit(spec)
  evaluably::verify_audit(spec)
  list(spec = spec, callbacks = e, units = id_map)
}
