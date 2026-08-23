#!/usr/bin/env Rscript

# Stream compact permutation shards into calibration, paired-inflation,
# completeness, and taxon-specific empirical-p summaries.

aggregate_script_dir <- local({
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) {
    dirname(normalizePath(ofile, winslash = "/", mustWork = TRUE))
  } else {
    file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    if (!length(file_arg)) stop("Cannot determine aggregate_permutations.R directory.")
    dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = TRUE))
  }
})
default_project_dir <- dirname(aggregate_script_dir)
source(file.path(default_project_dir, "permutation_validation_small.R"))

write_lines_atomic <- function(lines, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp.", Sys.getpid())
  writeLines(as.character(lines), tmp, useBytes = TRUE)
  if (file.exists(path)) unlink(path)
  if (!file.rename(tmp, path)) stop("Could not commit text file: ", path)
}

result_path_for <- function(out_dir, cohort, method, permutation_id) {
  file.path(out_dir, "results", cohort, method, sprintf("perm_%04d.rds", permutation_id))
}

status_path_for <- function(out_dir, cohort, method, permutation_id) {
  file.path(out_dir, "status", cohort, method, sprintf("perm_%04d.rds", permutation_id))
}

label_path_for <- function(out_dir, cohort, permutation_id) {
  file.path(out_dir, "labels", cohort, sprintf("perm_%04d.csv.gz", permutation_id))
}

status_to_row <- function(status, expected_path) {
  if (is.null(status)) {
    return(data.frame(
      cohort = NA_character_, method = NA_character_, permutation_id = NA_integer_,
      status = "missing", exit_code = NA_integer_, reused_existing = FALSE,
      elapsed_seconds = NA_real_, result_rows = 0L, cleanup_completed = NA,
      error_message = "status file missing", result_path = expected_path,
      log_path = NA_character_, stringsAsFactors = FALSE
    ))
  }
  fields <- c("cohort", "method", "permutation_id", "permutation_seed", "is_observed",
              "status", "exit_code", "reused_existing", "started_at", "ended_at",
              "elapsed_seconds", "result_rows", "compact", "cleanup_completed",
              "error_message", "result_path", "log_path", "completed_at")
  row <- lapply(fields, function(field) status[[field]] %||% NA)
  names(row) <- fields
  as.data.frame(row, stringsAsFactors = FALSE)
}

taxon_key <- function(x) {
  paste(x$dataset, x$group_mode, normalize_diet_col(x$diet_col), x$feature_id, sep = "\r")
}

summarize_fit_status <- function(x) {
  if (!nrow(x) || !"fit_status" %in% names(x)) return(data.frame())
  keys <- c("cohort", "method", "permutation_id", "permutation_seed", "is_observed",
            "dataset", "group_mode", "diet_col")
  split_key <- interaction(x[keys], drop = TRUE, lex.order = TRUE)
  rows <- lapply(split(x, split_key), function(g) {
    status <- as.character(g$fit_status)
    ok <- !is.na(status) & status == "ok"
    status_category <- ifelse(is.na(status), "missing", sub(":.*$", "", status))
    finite_p <- is.finite(suppressWarnings(as.numeric(g$p_value)))
    data.frame(
      cohort = g$cohort[[1]], method = g$method[[1]],
      permutation_id = g$permutation_id[[1]], permutation_seed = g$permutation_seed[[1]],
      is_observed = g$is_observed[[1]], dataset = g$dataset[[1]],
      group_mode = g$group_mode[[1]], diet_col = normalize_diet_col(g$diet_col[[1]]),
      n_taxa = nrow(g), n_fit_ok = sum(ok), n_fit_failed = sum(!ok),
      n_finite_p = sum(finite_p), fit_success_fraction = mean(ok),
      status_counts = paste(names(table(status_category)), as.integer(table(status_category)), sep = "=", collapse = " | "),
      stringsAsFactors = FALSE
    )
  })
  append_rows(rows)
}

aggregate_empirical <- function(out_dir, complete_manifest, cohorts, methods, n_perm) {
  outputs <- list()
  task_is_complete <- function(cohort, method, permutation_id) {
    hit <- complete_manifest$cohort == cohort & complete_manifest$method == method &
      complete_manifest$permutation_id == permutation_id
    any(hit & complete_manifest$complete)
  }
  for (cohort in cohorts) {
    for (method in methods) {
      observed_path <- result_path_for(out_dir, cohort, method, 0L)
      if (!task_is_complete(cohort, method, 0L) || !file.exists(observed_path)) next
      observed <- readRDS(observed_path)
      observed_key <- taxon_key(observed)
      if (anyDuplicated(observed_key)) stop("Duplicate observed taxon keys: ", cohort, "/", method)
      exceedances <- integer(nrow(observed))
      effective <- integer(nrow(observed))

      for (permutation_id in seq_len(n_perm)) {
        path <- result_path_for(out_dir, cohort, method, permutation_id)
        if (!task_is_complete(cohort, method, permutation_id) || !file.exists(path)) next
        perm <- readRDS(path)
        idx <- match(observed_key, taxon_key(perm))
        valid <- !is.na(idx) & is.finite(perm$p_value[idx]) & is.finite(observed$p_value)
        effective[valid] <- effective[valid] + 1L
        exceedances[valid] <- exceedances[valid] +
          as.integer(perm$p_value[idx[valid]] <= observed$p_value[valid])
      }

      empirical <- data.frame(
        cohort = cohort, method = method,
        dataset = observed$dataset, group_mode = observed$group_mode,
        diet_col = normalize_diet_col(observed$diet_col),
        feature_id = observed$feature_id,
        taxon = if ("taxon" %in% names(observed)) observed$taxon else NA_character_,
        effect_observed = observed$effect_size,
        p_observed = observed$p_value,
        q_observed = observed$q_value,
        exceedances = exceedances,
        n_effective_permutations = effective,
        empirical_p = (1 + exceedances) / (1 + effective),
        stringsAsFactors = FALSE
      )
      empirical$empirical_q <- NA_real_
      task_split <- interaction(empirical[c("cohort", "method", "dataset", "group_mode", "diet_col")],
                                drop = TRUE, lex.order = TRUE)
      for (idx in split(seq_len(nrow(empirical)), task_split)) {
        empirical$empirical_q[idx] <- p.adjust(empirical$empirical_p[idx], method = "BH")
      }
      outputs[[length(outputs) + 1L]] <- empirical
    }
  }
  append_rows(outputs)
}

run_aggregation <- function(cli) {
  project_dir <- normalizePath(cli$project_dir %||% default_project_dir, winslash = "/", mustWork = TRUE)
  out_dir <- normalizePath(cli$out_dir %||% file.path(project_dir, "artifacts", "permutation_validation_slurm"),
                           winslash = "/", mustWork = TRUE)
  n_perm <- as.integer(cli$n_perm %||% 1000L)
  base_seed <- as.integer(cli$seed %||% 20260715L)
  cohorts <- unique(toupper(split_csv_arg(cli$cohorts %||% "ECAM,DIABIMMUNE")))
  methods <- unique(tolower(split_csv_arg(cli$methods %||% "linda,maaslin2")))
  aggregate_subdir <- cli$aggregate_subdir %||% "aggregate"
  if (!nzchar(aggregate_subdir) || basename(aggregate_subdir) != aggregate_subdir) {
    stop("aggregate-subdir must be a single directory name.")
  }
  aggregate_dir <- normalize_output(file.path(out_dir, aggregate_subdir))

  manifest_rows <- list()
  summaries <- list()
  shard_rows <- list()
  missing_rows <- list()
  label_rows <- list()
  fit_diagnostic_rows <- list()

  for (permutation_id in 0:n_perm) {
    for (cohort in cohorts) {
      label_path <- label_path_for(out_dir, cohort, permutation_id)
      label_exists <- file.exists(label_path)
      label_counts <- ""
      n_label_rows <- 0L
      if (label_exists) {
        labels <- try(read.csv(label_path, stringsAsFactors = FALSE, check.names = FALSE), silent = TRUE)
        if (inherits(labels, "try-error")) {
          label_exists <- FALSE
        } else {
          n_label_rows <- nrow(labels)
          label_counts <- label_count_text(labels)
        }
      }
      label_rows[[length(label_rows) + 1L]] <- data.frame(
        cohort = cohort, permutation_id = permutation_id,
        permutation_seed = base_seed + cohort_seed_offset(cohort) + permutation_id,
        is_observed = permutation_id == 0L, label_exists = label_exists,
        n_label_rows = n_label_rows, group_counts = label_counts,
        label_path = normalizePath(label_path, winslash = "/", mustWork = FALSE),
        stringsAsFactors = FALSE
      )
    }
  }

  for (permutation_id in 0:n_perm) {
    for (cohort in cohorts) {
      for (method in methods) {
        result_path <- result_path_for(out_dir, cohort, method, permutation_id)
        status_path <- status_path_for(out_dir, cohort, method, permutation_id)
        status <- if (file.exists(status_path)) try(readRDS(status_path), silent = TRUE) else NULL
        if (inherits(status, "try-error")) status <- NULL
        status_row <- status_to_row(status, result_path)
        status_row$cohort <- cohort
        status_row$method <- method
        status_row$permutation_id <- permutation_id
        result_ok <- file.exists(result_path)
        if (result_ok) {
          result <- try(readRDS(result_path), silent = TRUE)
          result_ok <- !inherits(result, "try-error") && nrow(result) > 0L
        }
        status_row$result_exists <- result_ok
        status_row$complete <- identical(status_row$status[[1]], "completed") && result_ok
        manifest_rows[[length(manifest_rows) + 1L]] <- status_row

        if (status_row$complete) {
          summaries[[length(summaries) + 1L]] <- summarize_one_result(result)
          fit_summary <- summarize_fit_status(result)
          if (nrow(fit_summary)) fit_diagnostic_rows[[length(fit_diagnostic_rows) + 1L]] <- fit_summary
          shard_rows[[length(shard_rows) + 1L]] <- data.frame(
            cohort = cohort, method = method, permutation_id = permutation_id,
            rows = nrow(result), bytes = file.info(result_path)$size,
            result_path = normalizePath(result_path, winslash = "/", mustWork = TRUE),
            stringsAsFactors = FALSE
          )
        } else {
          missing_rows[[length(missing_rows) + 1L]] <- data.frame(
            cohort = cohort, method = method, permutation_id = permutation_id,
            status = status_row$status[[1]], error_message = status_row$error_message[[1]],
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }

  manifest <- append_rows(manifest_rows)
  run_summary <- append_rows(summaries)
  shard_manifest <- append_rows(shard_rows)
  missing <- append_rows(missing_rows)
  label_manifest <- append_rows(label_rows)
  fit_diagnostics <- append_rows(fit_diagnostic_rows)
  paired <- make_paired_inflation(run_summary)
  calibration <- summarize_null_calibration(run_summary)
  empirical <- aggregate_empirical(out_dir, manifest, cohorts, methods, n_perm)

  write_csv_atomic(manifest, file.path(aggregate_dir, "task_manifest.csv"))
  write_csv_atomic(shard_manifest, file.path(aggregate_dir, "compact_result_manifest.csv"))
  write_csv_atomic(label_manifest, file.path(aggregate_dir, "label_manifest.csv"))
  write_csv_atomic(missing, file.path(aggregate_dir, "missing_or_failed_tasks.csv"))
  write_csv_atomic(run_summary, file.path(aggregate_dir, "permutation_run_summary.csv"))
  write_csv_atomic(paired, file.path(aggregate_dir, "paired_inflation_completed_vs_observed.csv"))
  write_csv_atomic(calibration, file.path(aggregate_dir, "null_calibration_summary.csv"))
  write_csv_atomic(empirical, file.path(aggregate_dir, "empirical_permutation_results.csv"))
  write_csv_atomic(fit_diagnostics, file.path(aggregate_dir, "glmmpql_fit_diagnostics.csv"))

  missing_ids <- unique(c(
    if (nrow(missing)) missing$permutation_id else integer(),
    label_manifest$permutation_id[!label_manifest$label_exists]
  ))
  missing_ids <- sort(as.integer(missing_ids))
  write_lines_atomic(missing_ids, file.path(aggregate_dir, "missing_permutation_ids.txt"))
  completeness <- data.frame(
    expected_tasks = (n_perm + 1L) * length(cohorts) * length(methods),
    completed_tasks = sum(manifest$complete), missing_or_failed_tasks = nrow(missing),
    expected_null_permutations = n_perm,
    missing_label_files = sum(!label_manifest$label_exists),
    complete = nrow(missing) == 0L && all(label_manifest$label_exists),
    generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
    stringsAsFactors = FALSE
  )
  write_csv_atomic(completeness, file.path(aggregate_dir, "completeness_report.csv"))
  capture.output(sessionInfo(), file = file.path(aggregate_dir, "sessionInfo.txt"))

  total_issues <- nrow(missing) + sum(!label_manifest$label_exists)
  if (total_issues > 0L) {
    message("Aggregation completed with ", total_issues,
            " missing/failed task or label issues. See missing_permutation_ids.txt")
    quit(status = 2L)
  }
  message("Aggregation complete: all ", completeness$expected_tasks, " tasks present.")
  invisible(TRUE)
}

if (sys.nframe() == 0L) {
  cli <- parse_cli_args(commandArgs(trailingOnly = TRUE))
  tryCatch(
    run_aggregation(cli),
    error = function(e) {
      message("AGGREGATION ERROR: ", conditionMessage(e))
      quit(status = 1L)
    }
  )
}
