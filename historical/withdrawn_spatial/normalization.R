# Corrected per-spot normalization kernel. WITHDRAWN as signature-specific evidence.
# Input is genes x spots. Do not use vector recycling for library-size division.
spot_log_cpm <- function(X) {
 X <- as.matrix(X)
 if (any(!is.finite(X)) || any(X < 0)) stop("nonnegative finite counts required")
 lib <- colSums(X)
 log2(sweep(X + 0.5, 2, lib + 1, "/") * 1e6)
}
spatial_binomial_p <- function(k,n) binom.test(k,n,.5,alternative="two.sided")$p.value
# Historical multiplicity family: 200 representations x 3 niches = 600 tests.
spatial_bh <- function(p) {stopifnot(length(p)==600);p.adjust(p,method="BH")}
