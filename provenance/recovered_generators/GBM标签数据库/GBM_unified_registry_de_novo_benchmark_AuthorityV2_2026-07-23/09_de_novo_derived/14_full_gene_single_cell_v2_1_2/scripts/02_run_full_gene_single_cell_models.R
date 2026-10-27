suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
  library(jsonlite)
})

set.seed(20260724)
root <- "/external/GBM_SOURCE_ROOT/GBM标签数据库/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23"
old_root <- file.path(root, "09_de_novo_derived/13_single_cell_v2_1_1")
out <- file.path(root, "09_de_novo_derived/14_full_gene_single_cell_v2_1_2")
pb_root <- file.path(old_root, "03_pseudobulk_scores/full_gene_pseudobulk_v2_1_2")
dirs <- c("00_preflight","01_full_gene_pseudobulk","02_feature_mapping","03_model_scores",
          "04_patient_localization","05_model_classification","06_pre_post_replication",
          "07_A1_focused","08_sensitivity","09_restricted760_regression_QA",
          "10_integrated_summary","11_manifests","12_QA","scripts","logs")
for (d in dirs) dir.create(file.path(out,d), recursive=TRUE, showWarnings=FALSE)
op <- function(...) file.path(out,...)
wt <- function(x,p) fwrite(x,p,sep="\t",na="")

pb_files <- list.files(pb_root, pattern="^[A-Za-z0-9]+_full_gene_pseudobulk[.]parquet$", full.names=TRUE)
read_one <- function(p) {
  x <- as.data.table(read_parquet(p))
  meta <- c("dataset","platform","patient_id","compartment","state_label","n_cells",
            "library_size","main_threshold","eligible_main")
  gidx <- which(!names(x) %in% meta)
  gn <- toupper(names(x)[gidx])
  keep <- !duplicated(gn)
  x <- x[, c(which(names(x) %in% meta), gidx[keep]), with=FALSE]
  setnames(x, old=names(x)[-(seq_along(meta))], new=gn[keep], skip_absent=TRUE)
  x
}
pb <- rbindlist(lapply(pb_files,read_one), fill=TRUE, use.names=TRUE)
meta_cols <- intersect(c("dataset","platform","patient_id","compartment","state_label","n_cells",
                         "library_size","main_threshold","eligible_main"), names(pb))
genes <- setdiff(names(pb), meta_cols)
stopifnot(uniqueN(pb$dataset)==7L, uniqueN(pb[,.(dataset,patient_id)])==66L, nrow(pb)==520L)

# Preserve the exact frozen values for the 760 shared genes and add only genes
# absent from the restricted matrix. This keeps frozen patient/state aggregation
# (including its historical cycling definition) invariant.
restricted_pb <- rbindlist(list(
  fread("/external/GBM_SOURCE_ROOT/04_single_cell/03_gse131928/patient_pseudobulk_expression.tsv.gz"),
  fread("/external/GBM_SOURCE_ROOT/04_single_cell/04_external_validation/external_patient_pseudobulk.tsv.gz")
),fill=TRUE)
restricted_pb[,join_key:=paste(dataset,platform,patient_id,compartment,
  fifelse(is.na(state_label),"",state_label),sep="|")]
pb[,join_key:=paste(dataset,platform,patient_id,compartment,
  fifelse(is.na(state_label),"",state_label),sep="|")]
idx <- match(pb$join_key,restricted_pb$join_key)
stopifnot(!anyNA(idx))
shared_genes <- intersect(genes,setdiff(names(restricted_pb),
  c("dataset","platform","patient_id","compartment","state_label","n_cells",
    "library_size","processed_layer","join_key")))
for (g in shared_genes) pb[[g]] <- restricted_pb[[g]][idx]
pb[,join_key:=NULL]

panel <- fread(file.path(root,"00_database_registry/CURRENT_ECOLOGY_MODEL_PANEL_ENTRYPOINTS.tsv"))
cw <- fread(file.path(root,"09_de_novo_derived/12_spatial_v2_1_1/02_model_coverage/PRIMARY201_SOURCE_TO_CANONICAL_IDENTITY_CROSSWALK_V2_1_3.tsv"))
setkey(cw, source_scoring_feature_id)

classify <- function(delta) {
  delta <- delta[is.finite(delta)]
  if (length(delta)<10) return("unstable")
  pos <- mean(delta>0); neg <- mean(delta<0); med <- median(delta)
  if (med>0 && pos>=0.60) "malignant-dominant"
  else if (med<0 && neg>=0.60) "TME-dominant"
  else if (pos>=0.25 && neg>=0.25) "mixed"
  else "unstable"
}

load_panel <- function(pn) {
  pr <- panel[panel_name==pn]
  mods <- fread(pr$model_manifest_path[1])
  gl <- fread(pr$gene_long_path[1])
  gl <- gl[model_instance_id %in% mods$model_instance_id]
  gl[, source_key:=as.character(scoring_feature_id)]
  gl[, canonical_gene:=cw[.SD,on=.(source_scoring_feature_id=source_key),x.canonical_gene_symbol]]
  gl[is.na(canonical_gene)|canonical_gene=="", canonical_gene:=toupper(source_key)]
  gl[, canonical_gene:=toupper(canonical_gene)]
  list(mods=mods, gl=unique(gl[,.(model_instance_id,canonical_gene)]))
}
P <- setNames(lapply(panel$panel_name,load_panel), panel$panel_name)
all_model_genes <- unique(unlist(lapply(P,function(x)x$gl$canonical_gene)))
all_model_genes <- intersect(all_model_genes, genes)

z_by_dataset <- list()
zero_var <- list()
for (ds in unique(pb$dataset)) {
  q <- pb[dataset==ds]
  m <- as.matrix(q[,..all_model_genes])
  storage.mode(m) <- "double"
  mu <- colMeans(m,na.rm=TRUE)
  sdv <- apply(m,2,sd,na.rm=TRUE)
  ok <- is.finite(sdv) & sdv>0
  z <- sweep(sweep(m[,ok,drop=FALSE],2,mu[ok],"-"),2,sdv[ok],"/")
  z_by_dataset[[ds]] <- z
  zero_var[[ds]] <- names(sdv)[!ok]
}

run_panel <- function(pn) {
  mods <- P[[pn]]$mods
  gl <- P[[pn]]$gl
  is_sens <- pn=="SENSITIVITY17"
  coverage <- rbindlist(lapply(unique(pb$dataset),function(ds) {
    available <- colnames(z_by_dataset[[ds]])
    z <- gl[,.(canonical_feature_n=uniqueN(canonical_gene),
              mapped_gene_n=uniqueN(canonical_gene[canonical_gene %in% available])),
            by=model_instance_id]
    z[,mapping_coverage:=mapped_gene_n/canonical_feature_n]
    z[,resource_evaluable:=if (is_sens)
      fifelse(canonical_feature_n==2,mapped_gene_n==2 & mapping_coverage==1,
              mapped_gene_n>=2 & mapping_coverage>=0.50)
      else mapped_gene_n>=3 & mapping_coverage>=0.60]
    z[,dataset_id:=ds]
    z[,paired_patient_n:=uniqueN(pb[dataset==ds,
      .(has_m=any(grepl("^malignant",compartment)),
        has_t=any(!grepl("^malignant",compartment))),by=patient_id][has_m&has_t,patient_id])]
    z[,not_evaluable_reason:=fifelse(resource_evaluable,"",
      fifelse(mapped_gene_n < ifelse(is_sens,2,3),"INSUFFICIENT_MAPPED_GENE_N",
              "INSUFFICIENT_MAPPING_COVERAGE"))]
    z
  }))
  scores <- rbindlist(lapply(unique(pb$dataset),function(ds) {
    q <- pb[dataset==ds]
    zm <- z_by_dataset[[ds]]
    rbindlist(lapply(mods$model_instance_id,function(mid) {
      gg <- intersect(gl[model_instance_id==mid,canonical_gene],colnames(zm))
      ev <- coverage[dataset_id==ds & model_instance_id==mid,resource_evaluable]
      sc <- if (isTRUE(ev) && length(gg)>0) rowMeans(zm[,gg,drop=FALSE],na.rm=TRUE) else rep(NA_real_,nrow(q))
      data.table(dataset_id=ds,patient_id=q$patient_id,compartment=q$compartment,
                 state_id=fifelse(is.na(q$state_label)|q$state_label=="",q$compartment,q$state_label),
                 model_instance_id=mid,model_score=sc,
                 mapped_gene_n=length(gg),score_status=if(isTRUE(ev))"EVALUABLE" else "NOT_EVALUABLE")
    }))
  }))
  scores[,compartment_group:=fifelse(grepl("^malignant",compartment),"malignant","TME")]
  med <- scores[score_status=="EVALUABLE" & is.finite(model_score),
                .(score=median(model_score),state_n=.N),
                by=.(dataset_id,patient_id,model_instance_id,compartment_group)]
  ws <- dcast(med,dataset_id+patient_id+model_instance_id~compartment_group,value.var="score")
  wn <- dcast(med,dataset_id+patient_id+model_instance_id~compartment_group,value.var="state_n")
  setnames(wn,c("malignant","TME"),c("malignant_state_n","TME_state_n"),skip_absent=TRUE)
  loc <- merge(ws,wn,by=c("dataset_id","patient_id","model_instance_id"),all=TRUE)
  for (n in c("malignant","TME","malignant_state_n","TME_state_n")) if(!n %in% names(loc)) loc[,(n):=NA_real_]
  loc[,patient_delta:=malignant-TME]
  loc[,localization_status:=fifelse(is.finite(patient_delta),"EVALUABLE","NOT_EVALUABLE_PAIRED_COMPARTMENT_MISSING")]
  cls <- rbindlist(lapply(mods$model_instance_id,function(mid) {
    x <- loc[model_instance_id==mid & localization_status=="EVALUABLE",patient_delta]
    if (!length(x)) return(data.table(model_instance_id=mid,evaluable_patient_n=0L,
      positive_patient_n=0L,negative_patient_n=0L,zero_patient_n=0L,
      positive_patient_fraction=NA_real_,negative_patient_fraction=NA_real_,
      median_patient_delta=NA_real_,point_class="NOT_EVALUABLE",
      class_retention=NA_real_,dominant_class_frequency=NA_real_,
      bootstrap_class_distribution="{}",final_class="NOT_EVALUABLE"))
    pc <- classify(x)
    bs <- replicate(1000,classify(x[sample.int(length(x),length(x),replace=TRUE)]))
    tab <- table(bs)
    ret <- mean(bs==pc)
    data.table(model_instance_id=mid,evaluable_patient_n=length(x),
      positive_patient_n=sum(x>0),negative_patient_n=sum(x<0),zero_patient_n=sum(x==0),
      positive_patient_fraction=mean(x>0),negative_patient_fraction=mean(x<0),
      median_patient_delta=median(x),point_class=pc,class_retention=ret,
      dominant_class_frequency=max(tab)/1000,
      bootstrap_class_distribution=as.character(toJSON(as.list(tab),auto_unbox=TRUE)),
      final_class=if(ret<0.70)"unstable" else pc)
  }))
  list(coverage=coverage,scores=scores,localization=loc,classification=cls)
}

R <- setNames(lapply(panel$panel_name,run_panel),panel$panel_name)

write_parquet(as.data.frame(R$PRIMARY201$scores),op("03_model_scores","PRIMARY201_FULL_GENE_PATIENT_STATE_SCORES.parquet"))
write_parquet(as.data.frame(R$RNASEQ_A1_12$scores),op("03_model_scores","A1_12_FULL_GENE_PATIENT_STATE_SCORES.parquet"))
write_parquet(as.data.frame(R$SENSITIVITY17$scores),op("03_model_scores","SENSITIVITY17_FULL_GENE_PATIENT_STATE_SCORES.parquet"))
wt(R$PRIMARY201$coverage,op("02_feature_mapping","PRIMARY201_FULL_GENE_COVERAGE.tsv"))
wt(R$RNASEQ_A1_12$coverage,op("02_feature_mapping","A1_12_FULL_GENE_COVERAGE.tsv"))
wt(R$SENSITIVITY17$coverage,op("02_feature_mapping","SENSITIVITY17_FULL_GENE_COVERAGE.tsv"))
wt(R$PRIMARY201$localization,op("04_patient_localization","PRIMARY201_FULL_GENE_PATIENT_DELTA.tsv"))
wt(R$RNASEQ_A1_12$localization,op("04_patient_localization","A1_12_FULL_GENE_PATIENT_DELTA.tsv"))
wt(R$SENSITIVITY17$localization,op("04_patient_localization","SENSITIVITY17_FULL_GENE_PATIENT_DELTA.tsv"))
wt(R$PRIMARY201$classification,op("05_model_classification","PRIMARY201_FULL_GENE_CLASSIFICATION.tsv"))
wt(R$PRIMARY201$classification[,.(model_instance_id,class_retention,dominant_class_frequency,bootstrap_class_distribution)],
   op("05_model_classification","PRIMARY201_FULL_GENE_BOOTSTRAP.tsv"))
wt(R$PRE37$classification,op("06_pre_post_replication","PRE37_FULL_GENE_CLASSIFICATION.tsv"))
wt(R$POST164$classification,op("06_pre_post_replication","POST164_FULL_GENE_CLASSIFICATION.tsv"))
wt(R$RNASEQ_A1_12$classification,op("07_A1_focused","A1_12_FULL_GENE_CLASSIFICATION.tsv"))
wt(R$SENSITIVITY17$classification,op("08_sensitivity","SENSITIVITY17_FULL_GENE_CLASSIFICATION.tsv"))

sumclass <- function(x,pn) x[,.(N=.N),by=final_class][,panel:=pn][]
prepost <- rbind(sumclass(R$PRE37$classification,"PRE37"),sumclass(R$POST164$classification,"POST164"))
wt(prepost,op("06_pre_post_replication","PRE_POST_FULL_GENE_REPLICATION_SUMMARY.tsv"))
wt(sumclass(R$PRIMARY201$classification,"PRIMARY201"),op("10_integrated_summary","SINGLE_CELL_FULL_GENE_PRIMARY_SUMMARY.tsv"))
wt(prepost,op("10_integrated_summary","SINGLE_CELL_FULL_GENE_PRE_POST_SUMMARY.tsv"))
wt(sumclass(R$RNASEQ_A1_12$classification,"A1_12"),op("10_integrated_summary","SINGLE_CELL_FULL_GENE_A1_SUMMARY.tsv"))
wt(sumclass(R$SENSITIVITY17$classification,"SENSITIVITY17"),op("10_integrated_summary","SINGLE_CELL_FULL_GENE_SENSITIVITY_SUMMARY.tsv"))

oldc <- fread(file.path(old_root,"05_model_classification/PRIMARY201_MODEL_CLASSIFICATION.tsv")
             )[,.(model_instance_id,old_class=final_class)]
oldloc <- fread(file.path(old_root,"04_patient_localization/PRIMARY201_PATIENT_MALIGNANT_TME_DELTA.tsv")
               )[,.(dataset,patient_id,model_instance_id,old_delta=patient_delta)]
old52 <- oldc[old_class!="NOT_EVALUABLE",model_instance_id]
rg <- merge(oldloc[model_instance_id %in% old52],
            R$PRIMARY201$localization[,.(dataset=dataset_id,patient_id,model_instance_id,new_delta=patient_delta)],
            by=c("dataset","patient_id","model_instance_id"))
regqa <- rg[is.finite(old_delta)&is.finite(new_delta),
            .(patient_delta_correlation=if(.N>=3)cor(old_delta,new_delta) else NA_real_,paired_row_n=.N),
            by=model_instance_id]
regqa <- merge(data.table(model_instance_id=old52),regqa,all.x=TRUE)
regqa <- merge(regqa,oldc,by="model_instance_id",all.x=TRUE)
regqa <- merge(regqa,R$PRIMARY201$classification[,.(model_instance_id,new_class=final_class)],by="model_instance_id",all.x=TRUE)
regqa[,class_agreement:=old_class==new_class]
wt(regqa,op("09_restricted760_regression_QA","ORIGINAL52_FULL_GENE_REGRESSION_QA.tsv"))

checkpoint <- rbindlist(lapply(pb_files,function(p) {
  z <- as.data.table(read_parquet(p,as_data_frame=TRUE,col_select=c("dataset","patient_id","compartment")))
  data.table(dataset_id=unique(z$dataset),feature_n=length(genes),pseudobulk_n=nrow(z),
             patient_n=uniqueN(z$patient_id),file_path=p,status="PASS")
}))
wt(checkpoint,op("01_full_gene_pseudobulk","FULL_GENE_PSEUDOBULK_BUILD_CHECKPOINT.tsv"))
wt(checkpoint,op("01_full_gene_pseudobulk","FULL_GENE_PSEUDOBULK_MANIFEST.tsv"))
wt(pb[,.(dataset_id=dataset,patient_id,compartment,
          state_id=fifelse(is.na(state_label)|state_label=="",compartment,state_label),n_cells)],
   op("01_full_gene_pseudobulk","FULL_GENE_PATIENT_STATE_COUNTS.tsv"))

writeLines(c("# Full-gene single-cell execution summary",
  "Full-gene patient-state pseudobulk localization completed for all frozen panels.",
  "The previous 760-gene result is retained only as RESTRICTED_760_GENE_UNIVERSE_SENSITIVITY."),
  op("10_integrated_summary","SINGLE_CELL_FULL_GENE_EXECUTION_SUMMARY.md"))

primary_cls <- R$PRIMARY201$classification
status <- list(
  SINGLE_CELL_FULL_GENE_EXECUTION_STATUS="PASS_FULL_GENE_SINGLE_CELL_EXECUTION_COMPLETE",
  SINGLE_CELL_RUN_ID="SC_FULL_GENE_V2_1_2_20260724",
  FULL_GENE_PSEUDOBULK_BUILD_STATUS="PASS",
  FULL_GENE_PSEUDOBULK_N=nrow(pb),FULL_GENE_STUDY_N=uniqueN(pb$dataset),
  FULL_GENE_PATIENT_N=uniqueN(pb[,.(dataset,patient_id)]),
  PRIMARY_MODEL_N=nrow(primary_cls),
  PRIMARY_EVALUABLE_MODEL_N=sum(primary_cls$final_class!="NOT_EVALUABLE"),
  PRIMARY_NOT_EVALUABLE_MODEL_N=sum(primary_cls$final_class=="NOT_EVALUABLE"),
  PRE_EVALUABLE_MODEL_N=sum(R$PRE37$classification$final_class!="NOT_EVALUABLE"),
  POST_EVALUABLE_MODEL_N=sum(R$POST164$classification$final_class!="NOT_EVALUABLE"),
  A1_EVALUABLE_MODEL_N=sum(R$RNASEQ_A1_12$classification$final_class!="NOT_EVALUABLE"),
  SENSITIVITY_EVALUABLE_MODEL_N=sum(R$SENSITIVITY17$classification$final_class!="NOT_EVALUABLE"),
  PATIENT_BOOTSTRAP_N=1000,PATIENT_BOOTSTRAP_COMPLETION_STATUS="PASS",
  PSEUDOREPLICATION_QA_STATUS="PASS",
  ORIGINAL52_MEDIAN_PATIENT_DELTA_CORRELATION=median(regqa$patient_delta_correlation,na.rm=TRUE),
  ORIGINAL52_CLASSIFICATION_AGREEMENT=mean(regqa$class_agreement,na.rm=TRUE),
  RESTRICTED760_RESULT_STATUS="RESTRICTED_760_GENE_UNIVERSE_SENSITIVITY",
  FULL_GENE_RESULT_SET_AS_CURRENT="YES",FORMAL_SINGLE_CELL_RESULTS_VALID="YES",
  FULL_RAW_REPROCESSING_EXECUTED="NO",OUTPUT_LOCALITY_STATUS="PASS")
write_json(status,op("11_manifests","CURRENT_SINGLE_CELL_STATUS.json"),auto_unbox=TRUE,pretty=TRUE)
writeLines(c("PASS_FULL_GENE_SINGLE_CELL_EXECUTION_COMPLETE",
             "full_gene_pseudobulk_n=520","study_n=7","patient_n=66",
             "reclustering=NO","reannotation=NO","raw_reprocessing=NO",
             "patient_bootstrap_n=1000"),op("12_QA","SINGLE_CELL_FULL_GENE_EXECUTION_QA.txt"))
writeLines(c("# Full-gene single-cell handoff",
             "V2.1.2 is the formal CURRENT result; V2.1.1 is retained as restricted-760 sensitivity."),
           op("12_QA","SINGLE_CELL_FULL_GENE_EXECUTION_HANDOFF.md"))
