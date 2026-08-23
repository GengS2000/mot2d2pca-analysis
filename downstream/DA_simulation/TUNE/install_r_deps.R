local_lib <- file.path(getwd(), ".Rlib")
user_lib <- file.path(
  Sys.getenv("LOCALAPPDATA"),
  "R",
  "win-library",
  paste(R.version$major, sub("\\..*$", "", R.version$minor), sep = ".")
)

dir.create(local_lib, showWarnings = FALSE, recursive = TRUE)
dir.create(user_lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(unique(c(normalizePath(local_lib, mustWork = FALSE), user_lib, .libPaths())))

options(repos = c(CRAN = "https://cloud.r-project.org"))

cran_pkgs <- c(
  "ncdf4", "MicrobiomeStat", "LDM",
  "dplyr", "tidyr", "purrr", "stringr", "readr", "ggplot2", "tibble",
  "BiocManager",
  "statmod", "matrixStats", "ggrepel", "lmerTest", "foreach", "iterators",
  "modeest", "Rcpp", "mlr3", "mlr3mbo", "bbotk", "paradox", "randomForest",
  "gtable", "scales", "vctrs", "rlang", "cli", "lifecycle", "glue",
  "magrittr", "generics", "pillar", "tidyselect", "withr", "isoband",
  "S7", "farver", "labeling", "R6", "viridisLite", "cpp11", "stringi",
  "hms", "prettyunits", "progress", "bit", "bit64", "clipr", "crayon",
  "tzdb", "utf8", "pkgconfig"
)

installed_local <- rownames(installed.packages(lib.loc = c(local_lib, .Library)))
missing_cran <- setdiff(cran_pkgs, installed_local)
print(missing_cran)
if (length(missing_cran) > 0) {
  install.packages(missing_cran, lib = local_lib, dependencies = TRUE, Ncpus = 4)
}

bioc_pkgs <- c("Maaslin2", "ANCOMBC", "TreeSummarizedExperiment", "S4Vectors")
missing_bioc <- bioc_pkgs[!vapply(bioc_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_bioc) > 0) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", lib = local_lib)
  }
  BiocManager::install(missing_bioc, lib = local_lib, ask = FALSE, update = FALSE, Ncpus = 4)
}

pkgs <- c(
  "ncdf4", "MicrobiomeStat", "LDM", "Maaslin2", "ANCOMBC",
  "TreeSummarizedExperiment", "S4Vectors",
  "dplyr", "tidyr", "readr", "ggplot2", "tibble", "stringr", "purrr"
)
print(setNames(pkgs %in% rownames(installed.packages()), pkgs))
for (p in pkgs) {
  suppressPackageStartupMessages(library(p, character.only = TRUE))
  cat(p, "loaded\n")
}
