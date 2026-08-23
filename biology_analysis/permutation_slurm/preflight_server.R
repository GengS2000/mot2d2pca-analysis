#!/usr/bin/env Rscript

script_dir <- local({
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = TRUE))
})
project_dir <- dirname(script_dir)
source(file.path(project_dir, "permutation_validation_small.R"))
cli <- parse_cli_args(commandArgs(trailingOnly = TRUE))
methods <- unique(tolower(split_csv_arg(cli$methods %||% "linda,maaslin2")))

cat("R:", R.version.string, "\n")
cat("Rscript:", file.path(R.home("bin"), "Rscript"), "\n")
cat(".libPaths():\n")
print(.libPaths())

packages <- c("ncdf4", "MicrobiomeStat", "LinDA", "linda", "Maaslin2", "MASS", "nlme")
available <- setNames(vapply(packages, requireNamespace, logical(1), quietly = TRUE), packages)
print(available)
if (available[["ncdf4"]]) cat("ncdf4", as.character(packageVersion("ncdf4")), "\n")
if (available[["Maaslin2"]]) cat("Maaslin2", as.character(packageVersion("Maaslin2")), "\n")
for (pkg in c("MicrobiomeStat", "LinDA", "linda")) {
  if (available[[pkg]]) {
    cat(pkg, as.character(packageVersion(pkg)),
        "linda_function=", exists("linda", envir = asNamespace(pkg), mode = "function"), "\n")
  }
}

preflight_packages(methods)
targets <- default_permutation_targets(project_dir)
required_paths <- unique(unlist(targets[c("cohort_dir", "subject_groups", "participant_order",
                                          "linda_script", "maaslin2_script")], use.names = FALSE))
if ("glmmpql" %in% methods) {
  required_paths <- unique(c(required_paths, file.path(project_dir, "glmmpql_longitudinal_helpers.R")))
}
missing_paths <- required_paths[!file.exists(required_paths)]
if (length(missing_paths)) stop("Missing project inputs:\n", paste(missing_paths, collapse = "\n"))
cat("PREFLIGHT_OK\n")
