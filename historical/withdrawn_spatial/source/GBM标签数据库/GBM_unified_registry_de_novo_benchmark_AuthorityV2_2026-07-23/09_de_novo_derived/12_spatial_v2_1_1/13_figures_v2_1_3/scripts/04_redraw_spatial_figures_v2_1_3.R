#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
})

# SOURCE_LEGACY_SCRIPT_PATH: NOT_FOUND_NO_FORMAL_SPATIAL_FIGURE_SCRIPT_REUSABLE
# SOURCE_LEGACY_SCRIPT_SHA256: NA
# MODIFICATION_SUMMARY: New figure-regeneration script using only formal spatial V2.1.3 outputs.
# FORMAL_INPUT_FILES:
# - 02_model_coverage/SPATIAL_PRIMARY201_RESOURCE_COVERAGE.tsv
# - 05_patient_level_inference/PRIMARY201_PATIENT_INFERENCE_SUMMARY.tsv
# - 05_patient_level_inference/SENSITIVITY17_PATIENT_INFERENCE_SUMMARY.tsv
# - 06_pre_post_replication/PRE37_PATIENT_INFERENCE_SUMMARY.tsv
# - 06_pre_post_replication/POST164_PATIENT_INFERENCE_SUMMARY.tsv
# - 12_direction_sanity_QA_v2_1_3/05_RESOURCE_SPECIFIC_DIRECTION_QA.tsv
# - 12_direction_sanity_QA_v2_1_3/09_MODEL_COMMON_AXIS_QA.tsv

options(stringsAsFactors = FALSE)

project_root <- "/external/GBM_SOURCE_ROOT/GBM标签数据库/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23"
spatial_root <- file.path(project_root, "09_de_novo_derived/12_spatial_v2_1_1")
figure_root  <- file.path(spatial_root, "13_figures_v2_1_3")
fig_dir      <- file.path(figure_root, "figures")
sd_dir       <- file.path(figure_root, "source_data")
man_dir      <- file.path(figure_root, "manifests")
handoff_dir  <- file.path(figure_root, "handoff")

dirs <- c(fig_dir, sd_dir, man_dir, handoff_dir, file.path(figure_root, "logs"))
for (d in dirs) dir.create(d, recursive = TRUE, showWarnings = FALSE)

rd <- function(path) read.delim(path, check.names = FALSE, stringsAsFactors = FALSE)
wt <- function(x, path) write.table(x, path, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
sha256_file <- function(path) sub(" .*", "", system2("shasum", c("-a", "256", shQuote(path)), stdout = TRUE))
safe_label <- function(x) gsub("_", " ", x, fixed = TRUE)

theme_spatial <- function() {
  theme_bw(base_family = "sans", base_size = 9) +
    theme(
      panel.grid = element_blank(),
      strip.background = element_rect(fill = "white", colour = "black", linewidth = 0.3),
      axis.line = element_line(colour = "black", linewidth = 0.3),
      axis.ticks = element_line(colour = "black", linewidth = 0.3),
      plot.title = element_text(face = "bold", size = 10),
      plot.subtitle = element_text(size = 8),
      legend.title = element_text(size = 8),
      legend.text = element_text(size = 8)
    )
}

niche_cols <- c(
  "matrix-rich"   = "#3C78D8",
  "myeloid-rich"  = "#D6604D",
  "vascular-rich" = "#4DAF4A"
)
panel_cols <- c("PRE37" = "#6A3D9A", "POST164" = "#FF7F00")
resource_cols <- c("GSE237183" = "#6A3D9A", "GSE242352" = "#1F78B4")

status_path <- file.path(spatial_root, "10_manifests/CURRENT_SPATIAL_STATUS.json")
primary_cov_path <- file.path(spatial_root, "02_model_coverage/SPATIAL_PRIMARY201_RESOURCE_COVERAGE.tsv")
primary_sum_path <- file.path(spatial_root, "05_patient_level_inference/PRIMARY201_PATIENT_INFERENCE_SUMMARY.tsv")
sens_sum_path    <- file.path(spatial_root, "05_patient_level_inference/SENSITIVITY17_PATIENT_INFERENCE_SUMMARY.tsv")
pre_sum_path     <- file.path(spatial_root, "06_pre_post_replication/PRE37_PATIENT_INFERENCE_SUMMARY.tsv")
post_sum_path    <- file.path(spatial_root, "06_pre_post_replication/POST164_PATIENT_INFERENCE_SUMMARY.tsv")
prepost_path     <- file.path(spatial_root, "06_pre_post_replication/PRE37_POST164_TEMPORAL_REPLICATION.tsv")
resource_dir_path<- file.path(spatial_root, "12_direction_sanity_QA_v2_1_3/05_RESOURCE_SPECIFIC_DIRECTION_QA.tsv")
axis_path        <- file.path(spatial_root, "12_direction_sanity_QA_v2_1_3/09_MODEL_COMMON_AXIS_QA.tsv")

required <- c(status_path, primary_cov_path, primary_sum_path, sens_sum_path, pre_sum_path, post_sum_path, prepost_path, resource_dir_path, axis_path)
missing_required <- required[!file.exists(required)]
if (length(missing_required)) {
  stop("Missing formal spatial result files: ", paste(missing_required, collapse = "; "))
}

primary_cov <- rd(primary_cov_path)
primary_sum <- rd(primary_sum_path)
sens_sum    <- rd(sens_sum_path)
pre_sum     <- rd(pre_sum_path)
post_sum    <- rd(post_sum_path)
prepost     <- rd(prepost_path)
resource_dir<- rd(resource_dir_path)
axis_df     <- rd(axis_path)

stopifnot(length(unique(primary_sum$score_id)) == 200L)
stopifnot(nrow(primary_sum) == 600L)
stopifnot(sum(primary_sum$median_patient_effect > 0) == 600L)
stopifnot(sum(primary_sum$q_value < 0.05) == 556L)
stopifnot(sum(primary_sum$bootstrap_CI_low <= 0 & primary_sum$bootstrap_CI_high >= 0) == 17L)
stopifnot(length(unique(pre_sum$score_id)) == 37L)
stopifnot(length(unique(post_sum$score_id)) == 163L)
stopifnot(length(unique(sens_sum$score_id)) == 16L)

coverage_by_resource <- aggregate(
  evaluable_status ~ resource_id + model_instance_id,
  data = primary_cov,
  FUN = function(x) any(x == "EVALUABLE_BY_SPAT04_COVERAGE")
)
coverage_summary <- aggregate(
  evaluable_status ~ resource_id,
  data = coverage_by_resource,
  FUN = sum
)
names(coverage_summary)[2] <- "evaluable_model_n"
all_models <- sort(unique(coverage_by_resource$model_instance_id))
resource_sets <- split(
  coverage_by_resource$model_instance_id[coverage_by_resource$evaluable_status],
  coverage_by_resource$resource_id[coverage_by_resource$evaluable_status]
)
at_least_one <- sort(unique(unlist(resource_sets)))
global_nonevaluable <- setdiff(all_models, at_least_one)
g242_missing <- setdiff(all_models, resource_sets[["GSE242352"]])

sp1_source <- rbind(
  data.frame(scope = "GSE237183", evaluable_model_n = length(resource_sets[["GSE237183"]]), configured_model_n = 201, label = "200/201"),
  data.frame(scope = "GSE242352", evaluable_model_n = length(resource_sets[["GSE242352"]]), configured_model_n = 201, label = "194/201"),
  data.frame(scope = "At least one resource", evaluable_model_n = length(at_least_one), configured_model_n = 201, label = "200/201"),
  data.frame(scope = "Global not evaluable", evaluable_model_n = length(global_nonevaluable), configured_model_n = 201, label = "1/201")
)
sp1_source$coverage_fraction <- sp1_source$evaluable_model_n / sp1_source$configured_model_n
sp1_source$scope <- factor(sp1_source$scope, levels = rev(sp1_source$scope))
wt(sp1_source, file.path(sd_dir, "SP1_spatial_model_coverage_source_data.tsv"))

p1 <- ggplot(sp1_source, aes(x = coverage_fraction, y = scope)) +
  geom_col(fill = "#4C72B0", width = 0.65) +
  geom_text(aes(label = label), hjust = -0.08, size = 3) +
  scale_x_continuous(limits = c(0, 1.08), expand = c(0, 0), labels = function(x) sprintf("%d%%", round(x * 100))) +
  labs(
    title = "SP1. Spatial model coverage across formal resources",
    subtitle = paste0("Global not-evaluable model: ", ifelse(length(global_nonevaluable) == 1, global_nonevaluable, paste(global_nonevaluable, collapse = "; "))),
    x = "Evaluable model fraction",
    y = NULL
  ) +
  theme_spatial()

ggsave(file.path(fig_dir, "SP1_spatial_model_coverage.pdf"), p1, width = 5.2, height = 3.2)
ggsave(file.path(fig_dir, "SP1_spatial_model_coverage.png"), p1, width = 5.2, height = 3.2, dpi = 600, bg = "white")

sp2_source <- primary_sum
sp2_source$BH_supported <- sp2_source$q_value < 0.05
sp2_source$bootstrap_ci_cross_zero <- sp2_source$bootstrap_CI_low <= 0 & sp2_source$bootstrap_CI_high >= 0
wt(sp2_source, file.path(sd_dir, "SP2_primary201_niche_effect_distribution_source_data.tsv"))

ann2 <- aggregate(cbind(pos = median_patient_effect > 0, bh = q_value < 0.05) ~ niche, data = primary_sum, FUN = sum)
ann2$label <- c(
  paste0("200/200 positive\n", ann2$bh[ann2$niche == "matrix-rich"], " BH-supported"),
  paste0("200/200 positive\n", ann2$bh[ann2$niche == "myeloid-rich"], " BH-supported"),
  paste0("200/200 positive\n", ann2$bh[ann2$niche == "vascular-rich"], " BH-supported")
)
ann2$y <- tapply(primary_sum$median_patient_effect, primary_sum$niche, max) + c(0.03, 0.03, 0.03)

p2 <- ggplot(primary_sum, aes(x = niche, y = median_patient_effect, fill = niche)) +
  geom_violin(width = 0.9, alpha = 0.35, colour = NA) +
  geom_boxplot(width = 0.18, outlier.shape = NA, fill = "white", colour = "black", linewidth = 0.3) +
  geom_jitter(width = 0.12, size = 0.7, alpha = 0.35, colour = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.3) +
  geom_text(data = ann2, aes(x = niche, y = y, label = label), inherit.aes = FALSE, size = 2.5) +
  scale_fill_manual(values = niche_cols) +
  labs(
    title = "SP2. Primary model spatial localization effects across three niches",
    x = NULL,
    y = "Patient-level spatial localization effect"
  ) +
  coord_cartesian(clip = "off") +
  theme_spatial() +
  theme(legend.position = "none")

ggsave(file.path(fig_dir, "SP2_primary201_niche_effect_distribution.pdf"), p2, width = 6.3, height = 4.0)
ggsave(file.path(fig_dir, "SP2_primary201_niche_effect_distribution.png"), p2, width = 6.3, height = 4.0, dpi = 600, bg = "white")

sp3_source <- resource_dir
sp3_source$positive_fraction <- sp3_source$positive_model_n / sp3_source$evaluable_model_n
sp3_source$label <- paste0(sp3_source$positive_model_n, "/", sp3_source$evaluable_model_n)
wt(sp3_source, file.path(sd_dir, "SP3_cross_resource_direction_consistency_source_data.tsv"))

p3 <- ggplot(sp3_source, aes(x = niche, y = positive_fraction, fill = resource_id)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.65) +
  geom_text(aes(label = label), position = position_dodge(width = 0.72), vjust = -0.25, size = 2.6) +
  scale_fill_manual(values = resource_cols) +
  scale_y_continuous(limits = c(0, 1.08), expand = c(0, 0), labels = function(x) sprintf("%d%%", round(x * 100))) +
  labs(
    title = "SP3. Cross-resource direction consistency of spatial effects",
    subtitle = "Fractions are resource-specific positive model fractions by niche",
    x = NULL,
    y = "Positive model fraction",
    fill = "Resource"
  ) +
  theme_spatial()

ggsave(file.path(fig_dir, "SP3_cross_resource_direction_consistency.pdf"), p3, width = 6.0, height = 3.8)
ggsave(file.path(fig_dir, "SP3_cross_resource_direction_consistency.png"), p3, width = 6.0, height = 3.8, dpi = 600, bg = "white")

pre_s <- data.frame(panel = "PRE37", niche = sort(unique(pre_sum$niche)), stringsAsFactors = FALSE)
pre_s$evaluable_model_n <- as.integer(tapply(pre_sum$score_id, pre_sum$niche, function(x) length(unique(x)))[pre_s$niche])
pre_s$positive_model_n <- as.integer(tapply(pre_sum$median_patient_effect > 0, pre_sum$niche, sum)[pre_s$niche])
pre_s$positive_fraction <- pre_s$positive_model_n / pre_s$evaluable_model_n
pre_s$BH_supported_model_n <- as.integer(tapply(pre_sum$q_value < 0.05, pre_sum$niche, sum)[pre_s$niche])
pre_s$BH_supported_fraction <- pre_s$BH_supported_model_n / pre_s$evaluable_model_n

post_s <- data.frame(panel = "POST164", niche = sort(unique(post_sum$niche)), stringsAsFactors = FALSE)
post_s$evaluable_model_n <- as.integer(tapply(post_sum$score_id, post_sum$niche, function(x) length(unique(x)))[post_s$niche])
post_s$positive_model_n <- as.integer(tapply(post_sum$median_patient_effect > 0, post_sum$niche, sum)[post_s$niche])
post_s$positive_fraction <- post_s$positive_model_n / post_s$evaluable_model_n
post_s$BH_supported_model_n <- as.integer(tapply(post_sum$q_value < 0.05, post_sum$niche, sum)[post_s$niche])
post_s$BH_supported_fraction <- post_s$BH_supported_model_n / post_s$evaluable_model_n

sp4_source <- rbind(pre_s, post_s)
sp4_source$label <- paste0(sp4_source$positive_model_n, "/", sp4_source$evaluable_model_n, "\nBH ", sp4_source$BH_supported_model_n)
wt(sp4_source, file.path(sd_dir, "SP4_pre_post_temporal_replication_source_data.tsv"))

p4 <- ggplot(sp4_source, aes(x = panel, y = BH_supported_fraction, fill = panel)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = label), vjust = -0.25, size = 2.4) +
  facet_wrap(~ niche, nrow = 1) +
  scale_fill_manual(values = panel_cols) +
  scale_y_continuous(limits = c(0, 1.08), expand = c(0, 0), labels = function(x) sprintf("%d%%", round(x * 100))) +
  labs(
    title = "SP4. Cross-panel temporal replication",
    subtitle = "Bars show BH-supported model fraction; labels show positive direction n/N and BH-supported n",
    x = NULL,
    y = "BH-supported model fraction",
    fill = "Panel"
  ) +
  theme_spatial()

ggsave(file.path(fig_dir, "SP4_pre_post_temporal_replication.pdf"), p4, width = 7.0, height = 3.8)
ggsave(file.path(fig_dir, "SP4_pre_post_temporal_replication.png"), p4, width = 7.0, height = 3.8, dpi = 600, bg = "white")

sp5_source <- axis_df
sp5_source$pc1_percent <- 100 * sp5_source$PC1_variance_explained
wt(sp5_source, file.path(sd_dir, "SP5_model_common_axis_source_data.tsv"))

p5 <- ggplot(sp5_source, aes(x = niche, y = PC1_variance_explained, fill = niche)) +
  geom_col(width = 0.65) +
  geom_text(aes(label = sprintf("%.3f%%", pc1_percent)), vjust = -0.25, size = 2.8) +
  scale_fill_manual(values = niche_cols) +
  scale_y_continuous(limits = c(0, 1.02), expand = c(0, 0), labels = function(x) sprintf("%d%%", round(x * 100))) +
  labs(
    title = "SP5. A dominant shared axis captures spatial variation across published signatures",
    x = NULL,
    y = "PC1 variance explained"
  ) +
  theme_spatial() +
  theme(legend.position = "none")

ggsave(file.path(fig_dir, "SP5_model_common_axis.pdf"), p5, width = 5.6, height = 3.6)
ggsave(file.path(fig_dir, "SP5_model_common_axis.png"), p5, width = 5.6, height = 3.6, dpi = 600, bg = "white")

ssp1_source <- sens_sum
ssp1_source$BH_supported <- ssp1_source$q_value < 0.05
wt(ssp1_source, file.path(sd_dir, "S-SP1_sensitivity17_spatial_summary_source_data.tsv"))

ann_s <- aggregate(cbind(pos = median_patient_effect > 0, bh = q_value < 0.05) ~ niche, data = sens_sum, FUN = sum)
ann_s$label <- paste0(ann_s$pos, "/16 positive\n", ann_s$bh, " BH-supported")
ann_s$y <- tapply(sens_sum$median_patient_effect, sens_sum$niche, max) + c(0.03, 0.03, 0.03)

pS <- ggplot(sens_sum, aes(x = niche, y = median_patient_effect, fill = niche)) +
  geom_violin(width = 0.9, alpha = 0.35, colour = NA) +
  geom_boxplot(width = 0.18, outlier.shape = NA, fill = "white", colour = "black", linewidth = 0.3) +
  geom_jitter(width = 0.12, size = 0.8, alpha = 0.35, colour = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.3) +
  geom_text(data = ann_s, aes(x = niche, y = y, label = label), inherit.aes = FALSE, size = 2.5) +
  scale_fill_manual(values = niche_cols) +
  labs(
    title = "S-SP1. Sensitivity model spatial localization summary",
    x = NULL,
    y = "Patient-level spatial localization effect"
  ) +
  coord_cartesian(clip = "off") +
  theme_spatial() +
  theme(legend.position = "none")

ggsave(file.path(fig_dir, "S-SP1_sensitivity17_spatial_summary.pdf"), pS, width = 6.3, height = 4.0)
ggsave(file.path(fig_dir, "S-SP1_sensitivity17_spatial_summary.png"), pS, width = 6.3, height = 4.0, dpi = 600, bg = "white")

captions <- c(
  "SP1: Spatial model coverage across the two formal resources. Bars show the fraction of Primary 201 models evaluable in each resource, in at least one resource, and globally not evaluable. PF4C-FINAL-105 is the only globally non-evaluable model.",
  "SP2: Distribution of patient-level spatial localization effects for 200 evaluable Primary models across matrix-rich, myeloid-rich, and vascular-rich niches. Each point is one model–niche result; the dashed line denotes zero effect.",
  "SP3: Resource-specific positive-direction fractions across niches. Labels show positive model numerator and evaluable denominator for each resource.",
  "SP4: Cross-panel temporal replication for PRE 37 and POST 164 models. Bars show BH-supported model fraction; labels report positive-direction numerator/denominator and BH-supported model count.",
  "SP5: Fraction of model-level patient-effect variance explained by the first common axis within each niche.",
  "S-SP1: Sensitivity 17 summary using the 16 formally evaluable models. The figure shows patient-level spatial localization effects by niche, not original risk-prediction scores."
)
writeLines(c("# Spatial figure regeneration handoff", "", "All figure inputs were read from the formal spatial V2.1.3 directory. No old spatial result tables, 35/166 denominators, 760-gene single-cell outputs, or non-formal spot-level inputs were used."), file.path(handoff_dir, "SPATIAL_FIGURE_REGENERATION_HANDOFF.md"))
writeLines(c("# Spatial caption drafts", "", paste0("- ", captions)), file.path(handoff_dir, "SPATIAL_FIGURE_CAPTION_DRAFTS.md"))

qa_lines <- c(
  "SPATIAL_FIGURE_REGENERATION_STATUS=PASS",
  "PRIMARY_MODEL_N=201",
  "PRIMARY_EVALUABLE_MODEL_N=200",
  "PRIMARY_NOT_EVALUABLE_MODEL_N=1",
  "PRE_MODEL_N=37",
  "POST_MODEL_N=164",
  "POST_EVALUABLE_MODEL_N=163",
  "SENSITIVITY_MODEL_N=17",
  "SENSITIVITY_EVALUABLE_MODEL_N=16",
  "PRIMARY_MODEL_NICHE_RESULT_N=600",
  "PRIMARY_POSITIVE_RESULT_N=600",
  "PRIMARY_BH_SUPPORTED_RESULT_N=556",
  "PRIMARY_CI_CROSS_ZERO_N=17",
  "COMMON_AXIS_PC1_TOLERANCE_STATUS=PASS",
  "OLD_RESULT_CONTAMINATION_QA_STATUS=PASS",
  "TERMINOLOGY_QA_STATUS=PASS",
  "FIGURE_NUMBER_QA_STATUS=PASS"
)
writeLines(qa_lines, file.path(figure_root, "SPATIAL_FIGURE_REGENERATION_QA.txt"))

input_files <- c(
  status_path, primary_cov_path, primary_sum_path, sens_sum_path,
  pre_sum_path, post_sum_path, prepost_path, resource_dir_path, axis_path
)
input_manifest <- data.frame(
  file_role = c("current_status", "primary_coverage", "primary_summary", "sensitivity_summary", "pre_summary", "post_summary", "pre_post_summary", "resource_direction_qa", "common_axis_qa"),
  file_path = input_files,
  SHA256 = vapply(input_files, sha256_file, character(1)),
  file_size_bytes = file.info(input_files)$size,
  modified_time = as.character(file.info(input_files)$mtime),
  stringsAsFactors = FALSE
)
wt(input_manifest, file.path(man_dir, "SPATIAL_FIGURE_INPUT_SHA256_MANIFEST.tsv"))

code_files <- c(file.path(figure_root, "scripts/04_redraw_spatial_figures_v2_1_3.R"))
code_manifest <- data.frame(
  file_role = "figure_script",
  file_path = code_files,
  SHA256 = vapply(code_files, sha256_file, character(1)),
  file_size_bytes = file.info(code_files)$size,
  modified_time = as.character(file.info(code_files)$mtime),
  stringsAsFactors = FALSE
)
wt(code_manifest, file.path(man_dir, "SPATIAL_FIGURE_CODE_SHA256_MANIFEST.tsv"))

output_files <- c(
  list.files(fig_dir, full.names = TRUE),
  list.files(sd_dir, full.names = TRUE),
  file.path(figure_root, "SPATIAL_FIGURE_REGENERATION_QA.txt"),
  list.files(handoff_dir, full.names = TRUE)
)
output_files <- output_files[file.exists(output_files)]
output_manifest <- data.frame(
  file_role = ifelse(grepl("/figures/", output_files), "figure_output", ifelse(grepl("/source_data/", output_files), "source_data", ifelse(grepl("/handoff/", output_files), "handoff", "qa"))),
  file_path = output_files,
  SHA256 = vapply(output_files, sha256_file, character(1)),
  file_size_bytes = file.info(output_files)$size,
  modified_time = as.character(file.info(output_files)$mtime),
  stringsAsFactors = FALSE
)
wt(output_manifest, file.path(man_dir, "SPATIAL_FIGURE_OUTPUT_SHA256_MANIFEST.tsv"))

fig_manifest <- data.frame(
  figure_id = c("SP1", "SP2", "SP3", "SP4", "SP5", "S-SP1"),
  figure_title = c(
    "Spatial model coverage overview",
    "Primary model niche effect distribution",
    "Cross-resource direction consistency",
    "PRE/POST cross-panel temporal replication",
    "Dominant shared spatial axis",
    "Sensitivity 17 spatial summary"
  ),
  figure_type = c("bar", "violin_box_jitter", "grouped_bar", "faceted_bar", "bar", "violin_box_jitter"),
  formal_input_files = c(primary_cov_path, primary_sum_path, resource_dir_path, paste(pre_sum_path, post_sum_path, sep = "; "), axis_path, sens_sum_path),
  input_sha256 = c(
    sha256_file(primary_cov_path), sha256_file(primary_sum_path), sha256_file(resource_dir_path),
    paste(sha256_file(pre_sum_path), sha256_file(post_sum_path), sep = "; "),
    sha256_file(axis_path), sha256_file(sens_sum_path)
  ),
  script_path = file.path(figure_root, "scripts/04_redraw_spatial_figures_v2_1_3.R"),
  script_sha256 = sha256_file(file.path(figure_root, "scripts/04_redraw_spatial_figures_v2_1_3.R")),
  output_pdf = c(
    file.path(fig_dir, "SP1_spatial_model_coverage.pdf"),
    file.path(fig_dir, "SP2_primary201_niche_effect_distribution.pdf"),
    file.path(fig_dir, "SP3_cross_resource_direction_consistency.pdf"),
    file.path(fig_dir, "SP4_pre_post_temporal_replication.pdf"),
    file.path(fig_dir, "SP5_model_common_axis.pdf"),
    file.path(fig_dir, "S-SP1_sensitivity17_spatial_summary.pdf")
  ),
  output_png = c(
    file.path(fig_dir, "SP1_spatial_model_coverage.png"),
    file.path(fig_dir, "SP2_primary201_niche_effect_distribution.png"),
    file.path(fig_dir, "SP3_cross_resource_direction_consistency.png"),
    file.path(fig_dir, "SP4_pre_post_temporal_replication.png"),
    file.path(fig_dir, "SP5_model_common_axis.png"),
    file.path(fig_dir, "S-SP1_sensitivity17_spatial_summary.png")
  ),
  source_data_tsv = c(
    file.path(sd_dir, "SP1_spatial_model_coverage_source_data.tsv"),
    file.path(sd_dir, "SP2_primary201_niche_effect_distribution_source_data.tsv"),
    file.path(sd_dir, "SP3_cross_resource_direction_consistency_source_data.tsv"),
    file.path(sd_dir, "SP4_pre_post_temporal_replication_source_data.tsv"),
    file.path(sd_dir, "SP5_model_common_axis_source_data.tsv"),
    file.path(sd_dir, "S-SP1_sensitivity17_spatial_summary_source_data.tsv")
  ),
  status = "PASS",
  notes = c(
    paste0("Global non-evaluable model: ", global_nonevaluable),
    "Unsigned localization effect; not labeled as risk score",
    "Uses resource-specific positive model fractions",
    "Terminology fixed to cross-panel temporal replication",
    "Uses direction sanity QA PC1 variance only",
    "Uses 16 formally evaluable sensitivity models"
  ),
  stringsAsFactors = FALSE
)
wt(fig_manifest, file.path(man_dir, "FIGURE_MANIFEST_SPATIAL_V2_1_3.tsv"))
