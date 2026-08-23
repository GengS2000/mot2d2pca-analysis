#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
methods <- "linda,maaslin2"
if (length(args)) {
  hit <- which(args == "--methods")
  if (length(hit) && hit[[1]] < length(args)) methods <- args[[hit[[1]] + 1L]]
}
methods <- unique(tolower(trimws(strsplit(methods, ",", fixed = TRUE)[[1]])))
allowed <- c("linda", "maaslin2", "glmmpql")
if (length(setdiff(methods, allowed))) stop("Unknown methods: ", paste(setdiff(methods, allowed), collapse = ", "))

required <- "ncdf4"
if ("linda" %in% methods) required <- c(required, "MicrobiomeStat")
if ("maaslin2" %in% methods) required <- c(required, "Maaslin2")
if ("glmmpql" %in% methods) required <- c(required, "MASS", "nlme")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing R packages: ", paste(missing, collapse = ", "))
if ("linda" %in% methods && !exists("linda", envir = asNamespace("MicrobiomeStat"), inherits = FALSE)) {
  stop("MicrobiomeStat::linda is unavailable")
}
cat("STAGE2_PREFLIGHT_OK methods=", paste(methods, collapse = ","), "\n", sep = "")
