#!/usr/bin/env Rscript

# Slurm-friendly compact permutation worker.
# Each logical task is (cohort, method, permutation_id). Large engine outputs
# are created under out_dir/work and removed after a validated compact RDS is
# atomically committed. Labels, logs, compact taxon results, and status remain.

worker_script_dir <- local({
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) {
    dirname(normalizePath(ofile, winslash = "/", mustWork = TRUE))
  } else {
    file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    if (!length(file_arg)) stop("Cannot determine permutation_worker.R directory.")
    dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = TRUE))
  }
})
default_project_dir <- dirname(worker_script_dir)
source(file.path(default_project_dir, "permutation_validation_small.R"))

write_rds_atomic <- function(object, path, compress = "xz") {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp.", Sys.getpid())
  saveRDS(object, tmp, compress = compress)
  if (file.exists(path)) unlink(path)
  if (!file.rename(tmp, path)) stop("Could not atomically commit RDS: ", path)
  invisible(path)
}

write_csv_gz_atomic <- function(object, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp.", Sys.getpid())
  con <- gzfile(tmp, open = "wt", compression = 6)
  on.exit(try(close(con), silent = TRUE), add = TRUE)
  write.csv(object, con, row.names = FALSE, na = "NA")
  close(con)
  on.exit(NULL, add = FALSE)
  if (file.exists(path)) unlink(path)
  if (!file.rename(tmp, path)) stop("Could not atomically commit gzip CSV: ", path)
  invisible(path)
}

compact_result_path <- function(out_dir, cohort, method, permutation_id) {
  file.path(out_dir, "results", cohort, method, sprintf("perm_%04d.rds", permutation_id))
}

status_path_for <- function(out_dir, cohort, method, permutation_id) {
  file.path(out_dir, "status", cohort, method, sprintf("perm_%04d.rds", permutation_id))
}

label_path_for <- function(out_dir, cohort, permutation_id) {
  file.path(out_dir, "labels", cohort, sprintf("perm_%04d.csv.gz", permutation_id))
}

log_path_for <- function(out_dir, cohort, method, permutation_id) {
  file.path(out_dir, "logs", cohort, method, sprintf("perm_%04d.log", permutation_id))
}

parse_permutation_ids <- function(cli) {
  if (!is.null(cli$perm_ids_file)) {
    if (!file.exists(cli$perm_ids_file)) stop("perm_ids_file does not exist: ", cli$perm_ids_file)
    values <- scan(cli$perm_ids_file, what = integer(), quiet = TRUE)
  } else if (!is.null(cli$perm_ids)) {
    values <- as.integer(split_csv_arg(cli$perm_ids))
  } else if (!is.null(cli$perm_start) || !is.null(cli$perm_end)) {
    if (is.null(cli$perm_start) || is.null(cli$perm_end)) {
      stop("Provide both --perm-start and --perm-end.")
    }
    start <- as.integer(cli$perm_start)
    end <- as.integer(cli$perm_end)
    if (is.na(start) || is.na(end) || start > end) stop("Invalid permutation range.")
    values <- seq.int(start, end)
  } else {
    stop("Provide --perm-start/--perm-end, --perm-ids, or --perm-ids-file.")
  }
  if (anyNA(values) || any(values < 0L)) stop("Permutation IDs must be non-negative integers.")
  sort(unique(as.integer(values)))
}

validate_compact_result <- function(result, cohort, method, permutation_id, expected_taxa) {
  required <- c("cohort", "method", "permutation_id", "permutation_seed", "is_observed",
                "dataset", "group_mode", "diet_col", "feature_id", "effect_size", "p_value", "q_value")
  missing <- setdiff(required, names(result))
  if (length(missing)) stop("Compact result missing columns: ", paste(missing, collapse = ", "))
  if (!nrow(result)) stop("Compact result has zero rows.")
  if (!all(result$cohort == cohort) || !all(result$method == method) ||
      !all(result$permutation_id == permutation_id)) {
    stop("Compact result metadata does not match requested task.")
  }
  expected_datasets <- c("base_raw", "p2d2pca_imputed")
  if (!setequal(unique(result$dataset), expected_datasets)) {
    stop("Expected both base_raw and p2d2pca_imputed results.")
  }
  counts <- table(result$dataset)
  if (any(counts[expected_datasets] != expected_taxa)) {
    stop(
      "Unexpected taxon row counts: ",
      paste(names(counts), as.integer(counts), sep = "=", collapse = ", "),
      "; expected ", expected_taxa, " per dataset."
    )
  }
  invisible(TRUE)
}

tail_log <- function(path, n = 30L) {
  if (!file.exists(path)) return("")
  lines <- readLines(path, warn = FALSE)
  paste(tail(lines, n), collapse = "\n")
}

validate_existing_labels <- function(path, labels) {
  if (!file.exists(path)) return(FALSE)
  existing <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  comparable <- intersect(c("participant_id", "exposure_type", "exposure_value"), names(labels))
  if (!identical(names(existing[, comparable, drop = FALSE]), names(labels[, comparable, drop = FALSE])) ||
      nrow(existing) != nrow(labels)) return(FALSE)
  all(vapply(comparable, function(column) {
    identical(as.character(existing[[column]]), as.character(labels[[column]]))
  }, logical(1)))
}

run_compact_task <- function(target, method, permutation_id, labels_path, out_dir,
                             project_dir, rscript, lib_path, cores, resume, compact,
                             base_seed) {
  cohort <- target$cohort[[1]]
  expected_taxa <- if (cohort == "ECAM") 480L else 839L
  permutation_seed <- as.integer(base_seed + cohort_seed_offset(cohort) + permutation_id)
  result_path <- compact_result_path(out_dir, cohort, method, permutation_id)
  status_path <- status_path_for(out_dir, cohort, method, permutation_id)
  log_path <- log_path_for(out_dir, cohort, method, permutation_id)

  if (resume && file.exists(result_path)) {
    existing <- try(readRDS(result_path), silent = TRUE)
    valid <- !inherits(existing, "try-error") &&
      !inherits(try(validate_compact_result(existing, cohort, method, permutation_id, expected_taxa), silent = TRUE),
                "try-error") &&
      all(existing$permutation_seed == permutation_seed)
    if (valid) {
      status <- list(
        cohort = cohort, method = method, permutation_id = permutation_id,
        permutation_seed = permutation_seed, is_observed = permutation_id == 0L,
        status = "completed", exit_code = 0L, reused_existing = TRUE,
        elapsed_seconds = 0, result_path = result_path, log_path = log_path,
        result_rows = nrow(existing), compact = compact, cleanup_completed = TRUE,
        error_message = "", completed_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z")
      )
      write_rds_atomic(status, status_path, compress = "gzip")
      message(sprintf("[%s %s perm=%d] reusing compact result", cohort, method, permutation_id))
      return(status)
    }
    stop("Existing compact result failed validation: ", result_path)
  }

  worker_tag <- paste0(
    Sys.getenv("SLURM_JOB_ID", "local"), "_",
    Sys.getenv("SLURM_ARRAY_TASK_ID", "0"), "_", Sys.getpid()
  )
  work_dir <- file.path(out_dir, "work", worker_tag, cohort, method, sprintf("perm_%04d", permutation_id))
  if (dir.exists(work_dir)) unlink(work_dir, recursive = TRUE, force = TRUE)
  dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit({
    if (compact && dir.exists(work_dir)) unlink(work_dir, recursive = TRUE, force = TRUE)
  }, add = TRUE)

  script <- if (method %in% c("linda", "glmmpql")) target$linda_script[[1]] else target$maaslin2_script[[1]]
  args <- analysis_args(target, labels_path, method, work_dir, cores, lib_path)
  message(sprintf("[%s %s perm=%d] running", cohort, method, permutation_id))
  run_info <- run_child_script(rscript, script, args, target$cohort_dir[[1]], log_path)

  engine_result_path <- result_file_for_method(work_dir, method)
  error_message <- run_info$error_message
  result_rows <- 0L
  status_name <- "failed"
  cleanup_completed <- FALSE

  if (run_info$exit_code == 0L && file.exists(engine_result_path)) {
    result <- try(
      read_method_result(engine_result_path, cohort, method, permutation_id, permutation_seed),
      silent = TRUE
    )
    if (!inherits(result, "try-error")) {
      validation <- try(
        validate_compact_result(result, cohort, method, permutation_id, expected_taxa),
        silent = TRUE
      )
      if (!inherits(validation, "try-error")) {
        write_rds_atomic(result, result_path, compress = "xz")
        result_rows <- nrow(result)
        status_name <- "completed"
      } else {
        error_message <- conditionMessage(attr(validation, "condition"))
      }
    } else {
      error_message <- conditionMessage(attr(result, "condition"))
    }
  } else if (!nzchar(error_message)) {
    error_message <- tail_log(log_path)
  }

  if (compact) {
    unlink(work_dir, recursive = TRUE, force = TRUE)
    cleanup_completed <- !dir.exists(work_dir)
  }

  status <- list(
    cohort = cohort, method = method, permutation_id = permutation_id,
    permutation_seed = permutation_seed, is_observed = permutation_id == 0L,
    status = status_name, exit_code = run_info$exit_code, reused_existing = FALSE,
    started_at = run_info$started_at, ended_at = run_info$ended_at,
    elapsed_seconds = run_info$elapsed_seconds, result_path = result_path,
    log_path = log_path, result_rows = result_rows, compact = compact,
    cleanup_completed = cleanup_completed, error_message = error_message,
    completed_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z")
  )
  write_rds_atomic(status, status_path, compress = "gzip")
  status
}

run_worker <- function(cli) {
  project_dir <- normalizePath(cli$project_dir %||% default_project_dir, winslash = "/", mustWork = TRUE)
  out_dir <- normalize_output(cli$out_dir %||% file.path(project_dir, "artifacts", "permutation_validation_slurm"))
  permutation_ids <- parse_permutation_ids(cli)
  cohorts <- unique(toupper(split_csv_arg(cli$cohorts %||% "ECAM,DIABIMMUNE")))
  methods <- unique(tolower(split_csv_arg(cli$methods %||% "linda,maaslin2")))
  cores <- as.integer(cli$cores %||% 1L)
  base_seed <- as.integer(cli$seed %||% 20260715L)
  if (is.na(base_seed) || base_seed < 1L) stop("seed must be a positive integer.")
  resume <- as_flag(cli$resume %||% TRUE, TRUE)
  compact <- as_flag(cli$compact %||% TRUE, TRUE)
  rscript <- find_rscript(cli$rscript %||% NULL)
  lib_path <- cli$lib_path %||% paste(.libPaths(), collapse = ",")
  preflight_packages(methods, lib_path)

  targets <- default_permutation_targets(project_dir)
  targets <- targets[targets$cohort %in% cohorts, , drop = FALSE]
  if (!nrow(targets)) stop("No configured cohort targets remain.")
  if (length(setdiff(methods, c("linda", "maaslin2", "glmmpql")))) stop("Unknown method requested.")

  for (directory in c("results", "status", "labels", "logs", "work")) {
    dir.create(file.path(out_dir, directory), recursive = TRUE, showWarnings = FALSE)
  }
  config <- list(
    permutation_ids = permutation_ids, cohorts = cohorts, methods = methods,
    cores = cores, base_seed = base_seed, compact = compact, resume = resume, project_dir = project_dir,
    out_dir = out_dir, rscript = rscript, R = R.version.string,
    slurm_job_id = Sys.getenv("SLURM_JOB_ID", ""),
    slurm_array_task_id = Sys.getenv("SLURM_ARRAY_TASK_ID", ""),
    started_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z")
  )
  shard_id <- paste0(
    Sys.getenv("SLURM_JOB_ID", "local"), "_",
    Sys.getenv("SLURM_ARRAY_TASK_ID", Sys.getpid())
  )
  write_rds_atomic(config, file.path(out_dir, "worker_manifests", paste0("worker_", shard_id, ".rds")), "gzip")

  failures <- list()
  for (permutation_id in permutation_ids) {
    for (target_i in seq_len(nrow(targets))) {
      target <- targets[target_i, , drop = FALSE]
      cohort <- target$cohort[[1]]
      original_labels <- read_target_labels(target)
      permutation_seed <- as.integer(base_seed + cohort_seed_offset(cohort) + permutation_id)
      labels <- if (permutation_id == 0L) {
        original_labels
      } else {
        permute_participant_labels(original_labels, permutation_seed)
      }
      label_path <- label_path_for(out_dir, cohort, permutation_id)
      if (resume && file.exists(label_path)) {
        if (!validate_existing_labels(label_path, labels)) {
          stop("Existing label file conflicts with requested seed: ", label_path)
        }
      } else {
        write_csv_gz_atomic(labels, label_path)
      }

      for (method in methods) {
        status <- try(
          run_compact_task(
            target, method, permutation_id, label_path, out_dir, project_dir,
            rscript, lib_path, cores, resume, compact, base_seed
          ),
          silent = TRUE
        )
        if (inherits(status, "try-error")) {
          failures[[length(failures) + 1L]] <- data.frame(
            cohort = cohort, method = method, permutation_id = permutation_id,
            error_message = conditionMessage(attr(status, "condition")),
            stringsAsFactors = FALSE
          )
          warning("Task error: ", cohort, "/", method, "/", permutation_id, ": ",
                  conditionMessage(attr(status, "condition")))
        } else if (!identical(status$status, "completed")) {
          failures[[length(failures) + 1L]] <- data.frame(
            cohort = cohort, method = method, permutation_id = permutation_id,
            error_message = status$error_message, stringsAsFactors = FALSE
          )
        }
      }
    }
  }

  failure_df <- append_rows(failures)
  write_rds_atomic(failure_df, file.path(out_dir, "worker_manifests", paste0("failures_", shard_id, ".rds")), "gzip")
  message("Worker completed IDs: ", paste(permutation_ids, collapse = ","), "; failures=", nrow(failure_df))
  if (nrow(failure_df)) quit(status = 2L)
  invisible(TRUE)
}

if (sys.nframe() == 0L) {
  cli <- parse_cli_args(commandArgs(trailingOnly = TRUE))
  tryCatch(
    run_worker(cli),
    error = function(e) {
      message("WORKER ERROR: ", conditionMessage(e))
      quit(status = 1L)
    }
  )
}
