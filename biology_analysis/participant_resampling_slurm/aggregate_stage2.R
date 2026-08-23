#!/usr/bin/env Rscript

# Memory-conscious aggregation of compact bootstrap LinDA/MaAsLin2 results.

`%||%` <- function(x, y) if (is.null(x) || !length(x) || is.na(x[[1]])) y else x

parse_args <- function(args) {
  out <- list(); i <- 1L
  while (i <= length(args)) {
    key <- gsub("-", "_", substring(args[[i]], 3), fixed = TRUE)
    out[[key]] <- args[[i + 1L]]; i <- i + 2L
  }
  out
}

write_csv_atomic <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp.", Sys.getpid())
  write.csv(x, tmp, row.names = FALSE, na = "NA")
  if (file.exists(path)) unlink(path)
  if (!file.rename(tmp, path)) stop("Cannot commit CSV: ", path)
}

write_csv_gz <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  con <- gzfile(path, "wt", compression = 6)
  on.exit(close(con), add = TRUE)
  write.csv(x, con, row.names = FALSE, na = "NA")
}

normalize_diet <- function(x) {
  x <- as.character(x)
  x[is.na(x) | x == "NA" | x == ""] <- ""
  x
}

task_spec <- function(task_id, n_bootstrap, methods = c("linda", "maaslin2")) {
  cohorts <- c("ECAM", "DIABIMMUNE")
  block <- task_id %/% n_bootstrap
  data.frame(
    task_id = task_id,
    cohort = cohorts[[block %/% length(methods) + 1L]],
    method = methods[[block %% length(methods) + 1L]],
    bootstrap_replicate = task_id %% n_bootstrap + 1L,
    stringsAsFactors = FALSE
  )
}

task_key <- function(dataset, group_mode, diet_col) {
  paste(dataset, group_mode, normalize_diet(diet_col), sep = "\034")
}

safe_quantile <- function(x, p) {
  x <- x[is.finite(x)]
  if (!length(x)) NA_real_ else unname(quantile(x, p, na.rm = TRUE, type = 7))
}

summarize_cohort_method <- function(output_root, cohort, method, n_bootstrap) {
  entries <- new.env(hash = TRUE, parent = emptyenv())
  run_summaries <- list()

  for (replicate in seq_len(n_bootstrap)) {
    path <- file.path(output_root, "stage2", "results", cohort, method, sprintf("bootstrap_%03d.rds", replicate))
    if (!file.exists(path)) next
    result <- try(readRDS(path), silent = TRUE)
    if (inherits(result, "try-error") || !nrow(result)) next
    result$diet_col <- normalize_diet(result$diet_col)
    result$effect_size <- suppressWarnings(as.numeric(result$effect_size))
    result$p_value <- suppressWarnings(as.numeric(result$p_value))
    result$q_value <- suppressWarnings(as.numeric(result$q_value))
    split_idx <- split(seq_len(nrow(result)), task_key(result$dataset, result$group_mode, result$diet_col))

    for (key in names(split_idx)) {
      g <- result[split_idx[[key]], , drop = FALSE]
      if (anyDuplicated(g$feature_id)) stop("Duplicate feature_id within task: ", key, " replicate=", replicate)
      ids <- as.character(g$feature_id)
      if (!exists(key, envir = entries, inherits = FALSE)) {
        label <- if ("taxon" %in% names(g)) as.character(g$taxon) else rep(NA_character_, nrow(g))
        entry <- list(
          dataset = as.character(g$dataset[[1]]), group_mode = as.character(g$group_mode[[1]]),
          diet_col = normalize_diet(g$diet_col[[1]]), feature_id = ids, taxon = label,
          effect = matrix(NA_real_, nrow(g), n_bootstrap),
          p = matrix(NA_real_, nrow(g), n_bootstrap),
          q = matrix(NA_real_, nrow(g), n_bootstrap)
        )
      } else {
        entry <- get(key, envir = entries, inherits = FALSE)
        new_ids <- setdiff(ids, entry$feature_id)
        if (length(new_ids)) {
          new_rows <- match(new_ids, ids)
          entry$feature_id <- c(entry$feature_id, new_ids)
          entry$taxon <- c(entry$taxon, if ("taxon" %in% names(g)) as.character(g$taxon[new_rows]) else rep(NA_character_, length(new_ids)))
          padding <- matrix(NA_real_, length(new_ids), n_bootstrap)
          entry$effect <- rbind(entry$effect, padding)
          entry$p <- rbind(entry$p, padding)
          entry$q <- rbind(entry$q, padding)
        }
      }
      pos <- match(ids, entry$feature_id)
      entry$effect[pos, replicate] <- g$effect_size
      entry$p[pos, replicate] <- g$p_value
      entry$q[pos, replicate] <- g$q_value
      assign(key, entry, envir = entries)

      run_summaries[[length(run_summaries) + 1L]] <- data.frame(
        cohort = cohort, method = method, bootstrap_replicate = replicate,
        dataset = entry$dataset, group_mode = entry$group_mode, diet_col = entry$diet_col,
        n_tested = sum(is.finite(g$p_value)),
        n_q_lt_0_05 = sum(g$q_value < 0.05, na.rm = TRUE),
        n_q_lt_0_10 = sum(g$q_value < 0.10, na.rm = TRUE),
        prop_p_lt_0_05 = mean(g$p_value < 0.05, na.rm = TRUE),
        n_fit_ok = if ("fit_status" %in% names(g)) sum(g$fit_status == "ok", na.rm = TRUE) else NA_integer_,
        n_fit_failed = if ("fit_status" %in% names(g)) sum(is.na(g$fit_status) | g$fit_status != "ok") else NA_integer_,
        fit_success_fraction = if ("fit_status" %in% names(g)) mean(g$fit_status == "ok", na.rm = TRUE) else NA_real_,
        status_counts = if ("fit_status" %in% names(g)) {
          status_category <- ifelse(is.na(g$fit_status), "missing", sub(":.*$", "", as.character(g$fit_status)))
          paste(names(table(status_category)), as.integer(table(status_category)), sep = "=", collapse = " | ")
        } else NA_character_,
        stringsAsFactors = FALSE
      )
    }
  }

  summaries <- list()
  for (key in ls(entries, all.names = TRUE)) {
    entry <- get(key, envir = entries, inherits = FALSE)
    n_valid <- rowSums(is.finite(entry$effect))
    q05 <- rowSums(entry$q < 0.05, na.rm = TRUE) / pmax(rowSums(is.finite(entry$q)), 1L)
    q10 <- rowSums(entry$q < 0.10, na.rm = TRUE) / pmax(rowSums(is.finite(entry$q)), 1L)
    pos <- rowSums(entry$effect > 0, na.rm = TRUE) / pmax(n_valid, 1L)
    neg <- rowSums(entry$effect < 0, na.rm = TRUE) / pmax(n_valid, 1L)
    summaries[[length(summaries) + 1L]] <- data.frame(
      cohort = cohort, method = method, dataset = entry$dataset,
      group_mode = entry$group_mode, diet_col = entry$diet_col,
      feature_id = entry$feature_id, taxon = entry$taxon,
      n_successful_bootstraps = n_valid,
      selection_frequency_q05 = q05, selection_frequency_q10 = q10,
      positive_effect_frequency = pos, negative_effect_frequency = neg,
      sign_stability = pmax(pos, neg),
      effect_median = apply(entry$effect, 1L, median, na.rm = TRUE),
      effect_mean = rowMeans(entry$effect, na.rm = TRUE),
      effect_ci_low = apply(entry$effect, 1L, safe_quantile, p = 0.025),
      effect_ci_high = apply(entry$effect, 1L, safe_quantile, p = 0.975),
      median_p_value = apply(entry$p, 1L, median, na.rm = TRUE),
      median_q_value = apply(entry$q, 1L, median, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  list(
    stability = if (length(summaries)) do.call(rbind, summaries) else data.frame(),
    run_summary = if (length(run_summaries)) do.call(rbind, run_summaries) else data.frame()
  )
}

main <- function() {
  cli <- parse_args(commandArgs(trailingOnly = TRUE))
  output_root <- normalizePath(cli$output_root, winslash = "/", mustWork = FALSE)
  n_bootstrap <- as.integer(cli$n_bootstrap %||% 30L)
  methods <- unique(tolower(trimws(strsplit(cli$methods %||% "linda,maaslin2", ",", fixed = TRUE)[[1]])))
  allowed_methods <- c("linda", "maaslin2", "glmmpql")
  if (!length(methods) || length(setdiff(methods, allowed_methods))) {
    stop("methods must contain linda, maaslin2, and/or glmmpql")
  }
  aggregate_subdir <- cli$aggregate_subdir %||% "aggregate"
  if (!nzchar(aggregate_subdir) || basename(aggregate_subdir) != aggregate_subdir) {
    stop("aggregate-subdir must be a single directory name.")
  }
  aggregate_dir <- file.path(output_root, aggregate_subdir)
  dir.create(aggregate_dir, recursive = TRUE, showWarnings = FALSE)
  total <- 2L * length(methods) * n_bootstrap

  manifest <- list(); missing <- integer()
  for (task_id in 0:(total - 1L)) {
    spec <- task_spec(task_id, n_bootstrap, methods)
    status_path <- file.path(output_root, "stage2", "status", spec$cohort, spec$method,
                             sprintf("bootstrap_%03d.rds", spec$bootstrap_replicate))
    result_path <- file.path(output_root, "stage2", "results", spec$cohort, spec$method,
                             sprintf("bootstrap_%03d.rds", spec$bootstrap_replicate))
    status <- if (file.exists(status_path)) try(readRDS(status_path), silent = TRUE) else NULL
    complete <- !is.null(status) && !inherits(status, "try-error") && identical(status$status, "completed") &&
      file.exists(result_path) && file.info(result_path)$size > 0
    if (!complete) missing <- c(missing, task_id)
    manifest[[length(manifest) + 1L]] <- data.frame(
      spec, complete = complete,
      status = if (is.list(status)) status$status %||% "unknown" else "missing",
      exit_code = if (is.list(status)) status$exit_code %||% NA_integer_ else NA_integer_,
      elapsed_seconds = if (is.list(status)) status$elapsed_seconds %||% NA_real_ else NA_real_,
      result_rows = if (is.list(status)) status$result_rows %||% 0L else 0L,
      error_message = if (is.list(status)) status$error_message %||% "" else "status file missing",
      result_path = result_path, stringsAsFactors = FALSE
    )
  }
  manifest_df <- do.call(rbind, manifest)
  write_csv_atomic(manifest_df, file.path(aggregate_dir, "stage2_completeness_report.csv"))
  writeLines(as.character(missing), file.path(aggregate_dir, "missing_stage2_task_ids.txt"))

  stability_parts <- list(); run_parts <- list()
  for (cohort in c("ECAM", "DIABIMMUNE")) {
    for (method in methods) {
      message("Aggregating ", cohort, "/", method)
      x <- summarize_cohort_method(output_root, cohort, method, n_bootstrap)
      if (nrow(x$stability)) stability_parts[[length(stability_parts) + 1L]] <- x$stability
      if (nrow(x$run_summary)) run_parts[[length(run_parts) + 1L]] <- x$run_summary
      gc()
    }
  }
  stability <- if (length(stability_parts)) do.call(rbind, stability_parts) else data.frame()
  run_summary <- if (length(run_parts)) do.call(rbind, run_parts) else data.frame()
  write_csv_atomic(run_summary, file.path(aggregate_dir, "bootstrap_inference_run_summary.csv"))
  write_csv_gz(stability, file.path(aggregate_dir, "bootstrap_inference_stability.csv.gz"))
  fit_diagnostics <- if (nrow(run_summary) && "fit_success_fraction" %in% names(run_summary)) {
    run_summary[run_summary$method == "glmmpql", , drop = FALSE]
  } else data.frame()
  write_csv_atomic(fit_diagnostics, file.path(aggregate_dir, "glmmpql_fit_diagnostics.csv"))

  if (nrow(stability)) {
    keys <- c("cohort", "method", "group_mode", "diet_col", "feature_id")
    raw <- stability[stability$dataset == "base_raw", c(keys, "selection_frequency_q05", "effect_median")]
    imp <- stability[stability$dataset == "p2d2pca_imputed", c(keys, "selection_frequency_q05", "effect_median")]
    paired <- merge(raw, imp, by = keys, suffixes = c("_base_raw", "_imputed"))
    paired$delta_selection_frequency_q05 <- paired$selection_frequency_q05_imputed - paired$selection_frequency_q05_base_raw
    paired$delta_effect_median <- paired$effect_median_imputed - paired$effect_median_base_raw
    write_csv_gz(paired, file.path(aggregate_dir, "imputed_vs_raw_bootstrap_stability.csv.gz"))

    concordance <- list()
    method_pairs <- if (length(methods) >= 2L) combn(methods, 2L, simplify = FALSE) else list()
    group_key <- interaction(stability[c("cohort", "dataset", "group_mode", "diet_col")], drop = TRUE, lex.order = TRUE)
    for (g in split(stability, group_key)) {
      for (method_pair in method_pairs) {
        a <- g[g$method == method_pair[[1]], c("feature_id", "effect_median", "selection_frequency_q05")]
        b <- g[g$method == method_pair[[2]], c("feature_id", "effect_median", "selection_frequency_q05")]
        names(a)[-1L] <- c("effect_a", "selection_a")
        names(b)[-1L] <- c("effect_b", "selection_b")
        pair <- merge(a, b, by = "feature_id")
        if (!nrow(pair)) next
        concordance[[length(concordance) + 1L]] <- data.frame(
          cohort = g$cohort[[1]], dataset = g$dataset[[1]], group_mode = g$group_mode[[1]], diet_col = g$diet_col[[1]],
          method_a = method_pair[[1]], method_b = method_pair[[2]], n_features = nrow(pair),
          spearman_effect = suppressWarnings(cor(pair$effect_a, pair$effect_b,
                                                 method = "spearman", use = "complete.obs")),
          effect_sign_agreement = mean(sign(pair$effect_a) == sign(pair$effect_b), na.rm = TRUE),
          spearman_selection_frequency = suppressWarnings(cor(pair$selection_a, pair$selection_b,
            method = "spearman", use = "complete.obs")),
          stringsAsFactors = FALSE
        )
      }
    }
    write_csv_atomic(if (length(concordance)) do.call(rbind, concordance) else data.frame(),
                     file.path(aggregate_dir, "engine_concordance_summary.csv"))
  }

  status <- data.frame(
    n_bootstrap = n_bootstrap, expected_stage2_tasks = total,
    completed_stage2_tasks = sum(manifest_df$complete), missing_stage2_tasks = length(missing),
    complete = !length(missing), generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z")
  )
  write_csv_atomic(status, file.path(aggregate_dir, "stage2_aggregate_status.csv"))
  capture.output(sessionInfo(), file = file.path(aggregate_dir, "stage2_sessionInfo.txt"))
  print(status)
  if (length(missing)) quit(status = 2L)
  invisible(TRUE)
}

tryCatch(main(), error = function(e) {
  message("STAGE2 AGGREGATE ERROR: ", conditionMessage(e)); quit(status = 1L)
})
