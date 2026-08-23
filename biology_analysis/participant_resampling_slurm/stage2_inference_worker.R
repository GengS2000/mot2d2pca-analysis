#!/usr/bin/env Rscript

# Run one compact differential-inference task for one cohort/bootstrap/engine.

`%||%` <- function(x, y) if (is.null(x) || !length(x) || is.na(x[[1]])) y else x

parse_args <- function(args) {
  out <- list()
  i <- 1L
  while (i <= length(args)) {
    token <- args[[i]]
    if (!startsWith(token, "--")) stop("Unexpected argument: ", token)
    key <- gsub("-", "_", substring(token, 3), fixed = TRUE)
    if (i == length(args) || startsWith(args[[i + 1L]], "--")) {
      out[[key]] <- TRUE
      i <- i + 1L
    } else {
      out[[key]] <- args[[i + 1L]]
      i <- i + 2L
    }
  }
  out
}

write_rds_atomic <- function(object, path, compress = "gzip") {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp.", Sys.getpid())
  saveRDS(object, tmp, compress = compress)
  if (file.exists(path)) unlink(path)
  if (!file.rename(tmp, path)) stop("Cannot commit RDS: ", path)
  invisible(path)
}

tail_text <- function(path, n = 40L) {
  if (!file.exists(path)) return("")
  paste(tail(readLines(path, warn = FALSE), n), collapse = "\n")
}

task_spec <- function(task_id, n_bootstrap) {
  cohorts <- c("ECAM", "DIABIMMUNE")
  methods <- c("linda", "maaslin2")
  total <- length(cohorts) * length(methods) * n_bootstrap
  if (task_id < 0L || task_id >= total) stop("task-id outside valid range 0-", total - 1L)
  block <- task_id %/% n_bootstrap
  data.frame(
    task_id = task_id,
    cohort = cohorts[[block %/% length(methods) + 1L]],
    method = methods[[block %% length(methods) + 1L]],
    bootstrap_replicate = task_id %% n_bootstrap + 1L,
    stringsAsFactors = FALSE
  )
}

cohort_paths <- function(project_dir, cohort) {
  if (cohort == "ECAM") {
    cohort_dir <- file.path(project_dir, "ECAM_data")
    list(
      cohort_dir = cohort_dir,
      linda = file.path(cohort_dir, "differential_inference_netcdf.R"),
      glmmpql = file.path(cohort_dir, "differential_inference_netcdf.R"),
      maaslin2 = file.path(cohort_dir, "differential_inference_maaslin2_netcdf.R"),
      taxonomy = file.path(cohort_dir, "otu_taxonomy_cleaned_q2.csv")
    )
  } else {
    cohort_dir <- file.path(project_dir, "DIABIMMUNE_data")
    list(
      cohort_dir = cohort_dir,
      linda = file.path(cohort_dir, "differential_inference_netcdf.R"),
      glmmpql = file.path(cohort_dir, "differential_inference_netcdf.R"),
      maaslin2 = file.path(cohort_dir, "differential_inference_maaslin2_netcdf.R"),
      taxonomy = ""
    )
  }
}

main <- function() {
  cli <- parse_args(commandArgs(trailingOnly = TRUE))
  project_dir <- normalizePath(cli$project_dir %||% dirname(dirname(sys.frame(1)$ofile)), winslash = "/", mustWork = TRUE)
  output_root <- normalizePath(cli$output_root, winslash = "/", mustWork = FALSE)
  n_bootstrap <- as.integer(cli$n_bootstrap %||% 30L)
  task_id <- as.integer(cli$task_id)
  cores <- as.integer(cli$cores %||% 1L)
  rscript <- cli$rscript %||% file.path(R.home("bin"), "Rscript")
  lib_path <- cli$lib_path %||% paste(.libPaths(), collapse = ",")
  resume <- tolower(as.character(cli$resume %||% "true")) %in% c("true", "1", "yes")
  if (!is.null(cli$cohort) || !is.null(cli$method) || !is.null(cli$bootstrap_replicate)) {
    if (is.null(cli$cohort) || is.null(cli$method) || is.null(cli$bootstrap_replicate)) {
      stop("Provide --cohort, --method, and --bootstrap-replicate together.")
    }
    cohort_override <- toupper(as.character(cli$cohort))
    method_override <- tolower(as.character(cli$method))
    replicate_override <- as.integer(cli$bootstrap_replicate)
    if (!cohort_override %in% c("ECAM", "DIABIMMUNE")) stop("Unknown cohort: ", cohort_override)
    if (!method_override %in% c("linda", "maaslin2", "glmmpql")) stop("Unknown method: ", method_override)
    if (is.na(replicate_override) || replicate_override < 1L || replicate_override > n_bootstrap) {
      stop("bootstrap-replicate outside valid range 1-", n_bootstrap)
    }
    spec <- data.frame(
      task_id = task_id, cohort = cohort_override, method = method_override,
      bootstrap_replicate = replicate_override, stringsAsFactors = FALSE
    )
  } else {
    spec <- task_spec(task_id, n_bootstrap)
  }
  cohort <- spec$cohort[[1]]
  method <- spec$method[[1]]
  replicate <- spec$bootstrap_replicate[[1]]
  paths <- cohort_paths(project_dir, cohort)

  stage1_dir <- file.path(output_root, "stage1", cohort, sprintf("bootstrap_%03d", replicate))
  result_path <- file.path(output_root, "stage2", "results", cohort, method, sprintf("bootstrap_%03d.rds", replicate))
  status_path <- file.path(output_root, "stage2", "status", cohort, method, sprintf("bootstrap_%03d.rds", replicate))
  log_path <- file.path(output_root, "stage2", "logs", cohort, method, sprintf("bootstrap_%03d.log", replicate))
  required_stage1 <- file.path(stage1_dir, c(
    "SUCCESS.json", "tensor_base_raw.nc", "tensor_p2d2pca_imputed.nc",
    "participant_order.csv", "subject_groups_long.csv"
  ))

  if (resume && file.exists(result_path)) {
    existing <- try(readRDS(result_path), silent = TRUE)
    if (!inherits(existing, "try-error") && nrow(existing) &&
        all(existing$cohort == cohort) && all(existing$method == method) &&
        all(existing$bootstrap_replicate == replicate)) {
      status <- c(as.list(spec), list(status = "completed", reused_existing = TRUE,
        exit_code = 0L, elapsed_seconds = 0, result_rows = nrow(existing), error_message = ""))
      write_rds_atomic(status, status_path)
      message("Reusing ", result_path)
      return(invisible(TRUE))
    }
    stop("Existing compact result failed validation: ", result_path)
  }

  if (any(!file.exists(required_stage1))) {
    missing <- required_stage1[!file.exists(required_stage1)]
    status <- c(as.list(spec), list(status = "missing_stage1", reused_existing = FALSE,
      exit_code = 3L, elapsed_seconds = 0, result_rows = 0L,
      error_message = paste("Missing:", paste(missing, collapse = ", "))))
    write_rds_atomic(status, status_path)
    stop(status$error_message)
  }

  worker_tag <- paste(Sys.getenv("SLURM_JOB_ID", "local"), Sys.getenv("SLURM_ARRAY_TASK_ID", "0"), Sys.getpid(), sep = "_")
  work_dir <- file.path(output_root, "stage2", "work", worker_tag, cohort, method, sprintf("bootstrap_%03d", replicate))
  if (dir.exists(work_dir)) unlink(work_dir, recursive = TRUE, force = TRUE)
  dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(dirname(log_path), recursive = TRUE, showWarnings = FALSE)
  on.exit(if (dir.exists(work_dir)) unlink(work_dir, recursive = TRUE, force = TRUE), add = TRUE)

  script <- paths[[method]]
  engine_result <- switch(
    method,
    linda = "linda_longitudinal_results.csv",
    maaslin2 = "maaslin2_longitudinal_results.csv",
    glmmpql = "glmmpql_longitudinal_results.csv",
    stop("Unknown method: ", method)
  )
  args <- c(
    shQuote(script),
    "--base-nc", shQuote(file.path(stage1_dir, "tensor_base_raw.nc")),
    "--imputed-nc", shQuote(file.path(stage1_dir, "tensor_p2d2pca_imputed.nc")),
    "--subject-groups", shQuote(file.path(stage1_dir, "subject_groups_long.csv")),
    "--participant-order", shQuote(file.path(stage1_dir, "participant_order.csv")),
    "--out_dir", shQuote(work_dir),
    "--cores", as.character(cores),
    "--min-group-n", "2"
  )
  if (method == "linda") args <- c(args, "--engine", "linda", "--debug-linda", "false")
  if (method == "maaslin2") args <- c(args, "--debug-maaslin", "false")
  if (method == "glmmpql") {
    args <- c(args, "--engine", "glmmpql", "--debug-glmmpql", "false", "--glmmpql-niter", "10")
  }
  if (nzchar(lib_path)) args <- c(args, "--lib-path", shQuote(lib_path))
  if (cohort == "ECAM") {
    args <- c(args,
      "--taxon-index", shQuote(file.path(stage1_dir, "tensor_taxon_index.csv")),
      "--taxonomy", shQuote(paths$taxonomy))
  }

  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(paths$cohort_dir)
  started <- Sys.time()
  exit_code <- tryCatch(
    system2(rscript, args = args, stdout = log_path, stderr = log_path, wait = TRUE),
    error = function(e) structure(1L, error_message = conditionMessage(e))
  )
  ended <- Sys.time()
  error_message <- attr(exit_code, "error_message") %||% ""
  csv_path <- file.path(work_dir, engine_result)
  result_rows <- 0L
  status_name <- "failed"

  if (as.integer(exit_code) == 0L && file.exists(csv_path)) {
    result <- try(read.csv(csv_path, stringsAsFactors = FALSE, check.names = FALSE), silent = TRUE)
    required <- c("dataset", "group_mode", "diet_col", "feature_id", "effect_size", "p_value", "q_value")
    if (method == "glmmpql") required <- c(required, "std_error", "fit_status")
    if (!inherits(result, "try-error") && nrow(result) && !length(setdiff(required, names(result))) &&
        setequal(unique(result$dataset), c("base_raw", "p2d2pca_imputed"))) {
      result$cohort <- cohort
      result$method <- method
      result$bootstrap_replicate <- replicate
      result$task_id <- task_id
      result_rows <- nrow(result)
      write_rds_atomic(result, result_path, compress = "xz")
      status_name <- "completed"
    } else if (inherits(result, "try-error")) {
      error_message <- conditionMessage(attr(result, "condition"))
    } else {
      error_message <- "Output CSV was empty, incomplete, or lacked both base_raw and p2d2pca_imputed results."
    }
  } else if (!nzchar(error_message)) {
    error_message <- tail_text(log_path)
  }

  status <- c(as.list(spec), list(
    status = status_name, reused_existing = FALSE, exit_code = as.integer(exit_code),
    started_at = format(started, "%Y-%m-%d %H:%M:%S %z"),
    ended_at = format(ended, "%Y-%m-%d %H:%M:%S %z"),
    elapsed_seconds = as.numeric(difftime(ended, started, units = "secs")),
    result_rows = result_rows, result_path = result_path, log_path = log_path,
    error_message = error_message
  ))
  write_rds_atomic(status, status_path)
  unlink(work_dir, recursive = TRUE, force = TRUE)
  if (status_name != "completed") stop(error_message)
  message(sprintf("Completed %s/%s/bootstrap_%03d rows=%d", cohort, method, replicate, result_rows))
  invisible(TRUE)
}

tryCatch(main(), error = function(e) {
  message("STAGE2 WORKER ERROR: ", conditionMessage(e))
  quit(status = 2L)
})
