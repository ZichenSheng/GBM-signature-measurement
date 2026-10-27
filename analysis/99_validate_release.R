# Scoped public release suite. External-input tests are documented separately.
a<-commandArgs(FALSE);f<-sub("^--file=","",a[grepl("^--file=",a)]);root<-dirname(dirname(normalizePath(f)));setwd(root)
run<-function(cmd,args){z<-system2(cmd,args);if(z!=0L)stop("FAIL: ",cmd," ",paste(args,collapse=" "))}
py<-Sys.getenv("MANUSCRIPT_PYTHON","python3")
run(py,c("analysis/08_aggregate_evidence.py"));run(py,c("analysis/09_validate_final.py"))
run(py,c("-m","unittest","discover","-s","tests","-p","'test_*.py'"))
for(s in c("analysis/01_inspect_signatures.R","analysis/05_design_capability.R","analysis/06_crc04_exact_inference.R","tests/test_spatial_normalization.R"))run(file.path(R.home("bin"),"Rscript"),s)
cat("PASS scoped public numerical validation. External observations and full simulations are not certified by this command.\n")
