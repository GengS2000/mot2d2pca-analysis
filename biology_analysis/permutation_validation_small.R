#!/usr/bin/env Rscript

# Small-scale participant-label permutation validation for ECAM/DIABIMMUNE.
#
# This orchestrator deliberately reuses the existing cohort-specific LinDA and
# MaAsLin2 scripts. The completed tensors remain fixed. Only participant-level
# exposure labels are permuted; observed and completed data, and both DA engines,
# receive exactly the same label permutation.

.permutation_script_dir <- local({
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) {
    dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = FALSE))
  } else {
    ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
    if (!is.null(ofile)) dirname(normalizePath(ofile, winslash = "/", mustWork = FALSE)) else getwd()
  }
})

`%||%` <- function(x, y) {
  if (is.null(x) || !length(x) || (length(x) == 1L && is.na(x))) y else x
}

normalize_existing <- function(path, label = "file") {
  if (!file.exists(path)) stop(label, " does not exist: ", path)
  normalizePath(path, winslash = "/", mustWork = TRUE)
}

normalize_output <- function(path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  normalizePath(path, winslash = "/", mustWork = TRUE)
}

as_flag <- function(x, default = FALSE) {
  if (is.null(x) || !length(x)) return(default)
  tolower(trimws(as.character(x[[1]]))) %in% c("1", "true", "t", "yes", "y")
}

split_csv_arg <- function(x) {
  values <- trimws(strsplit(as.character(x), ",", fixed = TRUE)[[1]])
  values[nzchar(values)]
}

parse_cli_args <- function(args) {
  out <- list()
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) stop("Unexpected argument: ", key)
    key <- gsub("-", "_", substring(key, 3), fixed = TRUE)
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

find_rscript <- function(rscript = NULL) {
  candidates <- unique(c(
    rscript,
    Sys.which("Rscript"),
    file.path(R.home("bin"), if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript"),
    file.path(R.home("bin"), "x64", if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
  ))
  candidates <- candidates[!is.na(candidates) & nzchar(candidates)]
  found <- candidates[file.exists(candidates)]
  if (!length(found)) stop("Cannot locate Rscript. Pass rscript=... explicitly.")
  normalizePath(found[[1]], winslash = "/", mustWork = TRUE)
}

preflight_packages <- function(methods, lib_path = paste(.libPaths(), collapse = ",")) {
  requested_libs <- split_csv_arg(lib_path)
  requested_libs <- requested_libs[dir.exists(requested_libs)]
  old_libs <- .libPaths()
  on.exit(.libPaths(old_libs), add = TRUE)
  if (length(requested_libs)) .libPaths(unique(c(requested_libs, old_libs)))

  missing <- character()
  if (!requireNamespace("ncdf4", quietly = TRUE)) missing <- c(missing, "ncdf4")
  linda_available <- vapply(c("MicrobiomeStat", "LinDA", "linda"), function(pkg) {
    requireNamespace(pkg, quietly = TRUE) &&
      exists("linda", envir = asNamespace(pkg), mode = "function")
  }, logical(1))
  if ("linda" %in% methods && !any(linda_available)) {
    missing <- c(missing, "MicrobiomeStat (or LinDA/linda)")
  }
  if ("maaslin2" %in% methods && !requireNamespace("Maaslin2", quietly = TRUE)) {
    missing <- c(missing, "Maaslin2")
  }
  if ("glmmpql" %in% methods) {
    for (pkg in c("MASS", "nlme")) {
      if (!requireNamespace(pkg, quietly = TRUE)) missing <- c(missing, pkg)
    }
  }
  if (length(missing)) {
    stop(
      "Missing required R package/function(s): ", paste(missing, collapse = ", "),
      ". Current .libPaths(): ", paste(.libPaths(), collapse = "; "),
      ". Run from the same R environment used for the existing DA analyses, or pass lib_path=paste(.libPaths(), collapse=',')."
    )
  }
  invisible(TRUE)
}

default_permutation_targets <- function(project_dir = .permutation_script_dir) {
  project_dir <- normalizePath(project_dir, winslash = "/", mustWork = TRUE)
  data.frame(
    cohort = c("ECAM", "DIABIMMUNE"),
    group_mode = c("delivery_group", "abx_group"),
    diet_col = c("", ""),
    cohort_dir = file.path(project_dir, c("ECAM_data", "DIABIMMUNE_data")),
    subject_groups = file.path(
      project_dir,
      c(
        "ECAM_data/artifacts/ECAM_otu_linda_longitudinal/subject_groups_long.csv",
        "DIABIMMUNE_data/artifacts/subject_group_existing_methods_r/subject_groups_long.csv"
      )
    ),
    participant_order = file.path(
      project_dir,
      c(
        "ECAM_data/artifacts/ECAM_otu_linda_longitudinal/participant_order.csv",
        "DIABIMMUNE_data/participant_order.csv"
      )
    ),
    linda_script = file.path(
      project_dir,
      c("ECAM_data/differential_inference_netcdf.R", "DIABIMMUNE_data/differential_inference_netcdf.R")
    ),
    maaslin2_script = file.path(
      project_dir,
      c(
        "ECAM_data/differential_inference_maaslin2_netcdf.R",
        "DIABIMMUNE_data/differential_inference_maaslin2_netcdf.R"
      )
    ),
    stringsAsFactors = FALSE
  )
}

normalize_diet_col <- function(x) {
  x <- as.character(x)
  x[is.na(x) | x == "NA"] <- ""
  trimws(x)
}

read_target_labels <- function(target) {
  path <- normalize_existing(target$subject_groups, "subject_groups file")
  groups <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("participant_id", "group_mode", "diet_col", "exposure_type", "exposure_value", "rule")
  missing <- setdiff(required, names(groups))
  if (length(missing)) stop(path, " is missing columns: ", paste(missing, collapse = ", "))
  groups$diet_col <- normalize_diet_col(groups$diet_col)
  target_diet <- normalize_diet_col(target$diet_col)
  keep <- groups$group_mode == target$group_mode & groups$diet_col == target_diet
  task <- groups[keep, , drop = FALSE]
  if (!nrow(task)) {
    stop("No rows found for ", target$cohort, " / ", target$group_mode, " / ", target_diet)
  }
  if (anyDuplicated(task$participant_id)) {
    stop("Duplicate participant_id values in selected task: ", target$cohort, " / ", target$group_mode)
  }
  task
}

valid_label_index <- function(labels) {
  x <- as.character(labels$exposure_type)
  which(!is.na(x) & nzchar(trimws(x)) & tolower(trimws(x)) != "missing")
}

label_count_text <- function(labels) {
  idx <- valid_label_index(labels)
  counts <- table(as.character(labels$exposure_type[idx]))
  paste(names(counts), as.integer(counts), sep = "=", collapse = ", ")
}

permute_participant_labels <- function(labels, permutation_seed) {
  idx <- valid_label_index(labels)
  if (length(idx) < 4L) stop("Fewer than four participants have usable labels.")
  before <- sort(table(as.character(labels$exposure_type[idx])))
  set.seed(as.integer(permutation_seed))
  source_idx <- sample(idx, length(idx), replace = FALSE)
  out <- labels
  # Move the exposure label and its participant-level value together. The model
  # uses exposure_type, while moving exposure_value prevents an inconsistent
  # audit table for early/late diet tasks.
  shuffled_columns <- intersect(c("exposure_type", "exposure_value"), names(labels))
  out[idx, shuffled_columns] <- labels[source_idx, shuffled_columns, drop = FALSE]
  after <- sort(table(as.character(out$exposure_type[idx])))
  if (!identical(before, after)) stop("Permutation changed group sizes.")
  out
}

cohort_seed_offset <- function(cohort) {
  offsets <- c(ECAM = 1000000L, DIABIMMUNE = 2000000L)
  if (!cohort %in% names(offsets)) stop("No deterministic seed offset for cohort: ", cohort)
  offsets[[cohort]]
}

write_csv_atomic <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp")
  if (ncol(x) == 0L) {
    file.create(tmp)
  } else {
    write.csv(x, tmp, row.names = FALSE, na = "NA")
  }
  if (file.exists(path)) unlink(path)
  if (!file.rename(tmp, path)) stop("Could not move temporary CSV into place: ", path)
  invisible(path)
}

append_rows <- function(items) {
  items <- items[vapply(items, function(x) !is.null(x) && nrow(x) > 0L, logical(1))]
  if (!length(items)) return(data.frame())
  columns <- unique(unlist(lapply(items, names), use.names = FALSE))
  aligned <- lapply(items, function(x) {
    missing <- setdiff(columns, names(x))
    for (column in missing) x[[column]] <- NA
    x[, columns, drop = FALSE]
  })
  do.call(rbind, aligned)
}

run_child_script <- function(rscript, script, args, working_dir, log_path) {
  dir.create(dirname(log_path), recursive = TRUE, showWarnings = FALSE)
  script <- normalize_existing(script, "analysis script")
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(working_dir)
  quoted_args <- c(shQuote(script), unlist(lapply(args, shQuote), use.names = FALSE))
  started <- Sys.time()
  status <- tryCatch(
    system2(rscript, args = quoted_args, stdout = log_path, stderr = log_path, wait = TRUE),
    error = function(e) structure(1L, error_message = conditionMessage(e))
  )
  ended <- Sys.time()
  error_message <- attr(status, "error_message") %||% ""
  list(
    exit_code = as.integer(status),
    started_at = format(started, "%Y-%m-%d %H:%M:%S %z"),
    ended_at = format(ended, "%Y-%m-%d %H:%M:%S %z"),
    elapsed_seconds = as.numeric(difftime(ended, started, units = "secs")),
    error_message = error_message
  )
}

result_file_for_method <- function(run_dir, method) {
  filename <- switch(
    method,
    linda = "linda_longitudinal_results.csv",
    maaslin2 = "maaslin2_longitudinal_results.csv",
    glmmpql = "glmmpql_longitudinal_results.csv",
    stop("Unknown method: ", method)
  )
  file.path(run_dir, filename)
}

analysis_args <- function(target, labels_path, method, method_out_dir, cores, lib_path = "") {
  args <- c(
    "--subject-groups", normalizePath(labels_path, winslash = "/", mustWork = TRUE),
    "--participant-order", normalize_existing(target$participant_order, "participant order"),
    "--out_dir", normalize_output(method_out_dir),
    "--cores", as.character(as.integer(cores)),
    "--min-group-n", "2"
  )
  if (target$cohort == "ECAM") args <- c(args, "--group-modes", target$group_mode)
  if (method == "linda") {
    args <- c(args, "--engine", "linda", "--debug-linda", "false")
  } else if (method == "maaslin2") {
    args <- c(args, "--debug-maaslin", "false")
  } else if (method == "glmmpql") {
    args <- c(args, "--engine", "glmmpql", "--debug-glmmpql", "false", "--glmmpql-niter", "10")
  } else {
    stop("Unknown method: ", method)
  }
  if (nzchar(lib_path)) args <- c(args, "--lib-path", lib_path)
  args
}

read_method_result <- function(path, cohort, method, permutation_id, permutation_seed) {
  if (!file.exists(path)) return(data.frame())
  x <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  if (!nrow(x)) return(x)
  required <- c("dataset", "group_mode", "diet_col", "feature_id", "effect_size", "p_value", "q_value")
  missing <- setdiff(required, names(x))
  if (length(missing)) stop(path, " is missing result columns: ", paste(missing, collapse = ", "))
  x$diet_col <- normalize_diet_col(x$diet_col)
  x$p_value <- suppressWarnings(as.numeric(x$p_value))
  x$q_value <- suppressWarnings(as.numeric(x$q_value))
  x$effect_size <- suppressWarnings(as.numeric(x$effect_size))
  x$cohort <- cohort
  x$method <- method
  x$permutation_id <- as.integer(permutation_id)
  x$permutation_seed <- as.integer(permutation_seed)
  x$is_observed <- permutation_id == 0L
  # This is monotone in the two-sided p-value and is useful for diagnostics.
  # It is not labelled as a model t/z statistic because current consolidated
  # LinDA output does not preserve SE/statistic.
  clipped_p <- pmin(pmax(x$p_value, .Machine$double.xmin), 1)
  x$signed_p_score <- sign(x$effect_size) * stats::qnorm(clipped_p / 2, lower.tail = FALSE)
  front <- c("cohort", "method", "permutation_id", "permutation_seed", "is_observed")
  x[, c(front, setdiff(names(x), front)), drop = FALSE]
}

summarize_one_result <- function(x) {
  if (!nrow(x)) return(data.frame())
  keys <- c("cohort", "method", "permutation_id", "permutation_seed", "is_observed",
            "dataset", "group_mode", "diet_col")
  split_key <- interaction(x[keys], drop = TRUE, lex.order = TRUE)
  rows <- lapply(split(x, split_key), function(g) {
    p <- suppressWarnings(as.numeric(g$p_value))
    q <- suppressWarnings(as.numeric(g$q_value))
    p <- p[is.finite(p)]
    q <- q[is.finite(q)]
    data.frame(
      cohort = g$cohort[[1]], method = g$method[[1]],
      permutation_id = g$permutation_id[[1]], permutation_seed = g$permutation_seed[[1]],
      is_observed = g$is_observed[[1]], dataset = g$dataset[[1]],
      group_mode = g$group_mode[[1]], diet_col = normalize_diet_col(g$diet_col[[1]]),
      n_tested = length(p),
      prop_p_lt_0_01 = if (length(p)) mean(p < 0.01) else NA_real_,
      prop_p_lt_0_05 = if (length(p)) mean(p < 0.05) else NA_real_,
      median_p = if (length(p)) median(p) else NA_real_,
      min_p = if (length(p)) min(p) else NA_real_,
      n_q_lt_0_05 = if (length(q)) sum(q < 0.05) else NA_integer_,
      n_q_lt_0_10 = if (length(q)) sum(q < 0.10) else NA_integer_,
      any_q_lt_0_05 = if (length(q)) any(q < 0.05) else NA,
      any_q_lt_0_10 = if (length(q)) any(q < 0.10) else NA,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

make_paired_inflation <- function(run_summary) {
  if (!nrow(run_summary)) return(data.frame())
  keys <- c("cohort", "method", "permutation_id", "permutation_seed", "is_observed", "group_mode", "diet_col")
  observed <- run_summary[run_summary$dataset == "base_raw", , drop = FALSE]
  completed <- run_summary[run_summary$dataset == "p2d2pca_imputed", , drop = FALSE]
  if (!nrow(observed) || !nrow(completed)) return(data.frame())
  keep <- c(keys, "prop_p_lt_0_01", "prop_p_lt_0_05", "n_q_lt_0_05", "n_q_lt_0_10")
  paired <- merge(observed[, keep], completed[, keep], by = keys, suffixes = c("_observed", "_completed"))
  paired$delta_p_lt_0_01 <- paired$prop_p_lt_0_01_completed - paired$prop_p_lt_0_01_observed
  paired$delta_p_lt_0_05 <- paired$prop_p_lt_0_05_completed - paired$prop_p_lt_0_05_observed
  paired$rr_p_lt_0_01 <- ifelse(paired$prop_p_lt_0_01_observed > 0,
                                paired$prop_p_lt_0_01_completed / paired$prop_p_lt_0_01_observed, NA_real_)
  paired$rr_p_lt_0_05 <- ifelse(paired$prop_p_lt_0_05_observed > 0,
                                paired$prop_p_lt_0_05_completed / paired$prop_p_lt_0_05_observed, NA_real_)
  paired
}

summarize_null_calibration <- function(run_summary) {
  null <- run_summary[!run_summary$is_observed, , drop = FALSE]
  if (!nrow(null)) return(data.frame())
  keys <- c("cohort", "method", "dataset", "group_mode", "diet_col")
  split_key <- interaction(null[keys], drop = TRUE, lex.order = TRUE)
  rows <- lapply(split(null, split_key), function(g) {
    ci <- function(x) {
      x <- x[is.finite(x)]
      if (!length(x)) c(NA_real_, NA_real_) else unname(quantile(x, c(0.025, 0.975), na.rm = TRUE))
    }
    ci01 <- ci(g$prop_p_lt_0_01)
    ci05 <- ci(g$prop_p_lt_0_05)
    data.frame(
      cohort = g$cohort[[1]], method = g$method[[1]], dataset = g$dataset[[1]],
      group_mode = g$group_mode[[1]], diet_col = g$diet_col[[1]],
      n_permutations = length(unique(g$permutation_id)),
      mean_prop_p_lt_0_01 = mean(g$prop_p_lt_0_01, na.rm = TRUE),
      prop_p_lt_0_01_ci_low = ci01[[1]], prop_p_lt_0_01_ci_high = ci01[[2]],
      mean_prop_p_lt_0_05 = mean(g$prop_p_lt_0_05, na.rm = TRUE),
      prop_p_lt_0_05_ci_low = ci05[[1]], prop_p_lt_0_05_ci_high = ci05[[2]],
      probability_any_q_lt_0_05 = mean(g$any_q_lt_0_05, na.rm = TRUE),
      mean_n_q_lt_0_05 = mean(g$n_q_lt_0_05, na.rm = TRUE),
      probability_any_q_lt_0_10 = mean(g$any_q_lt_0_10, na.rm = TRUE),
      mean_n_q_lt_0_10 = mean(g$n_q_lt_0_10, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

compute_empirical_pilot <- function(taxon_results) {
  if (!nrow(taxon_results)) return(data.frame())
  required <- c("cohort", "method", "dataset", "group_mode", "diet_col", "feature_id",
                "permutation_id", "p_value", "q_value", "effect_size")
  if (length(setdiff(required, names(taxon_results)))) return(data.frame())
  keys <- c("cohort", "method", "dataset", "group_mode", "diet_col", "feature_id")
  split_key <- interaction(taxon_results[keys], drop = TRUE, lex.order = TRUE)
  rows <- lapply(split(taxon_results, split_key), function(g) {
    observed <- g[g$permutation_id == 0L & is.finite(g$p_value), , drop = FALSE]
    perm_p <- g$p_value[g$permutation_id > 0L & is.finite(g$p_value)]
    if (nrow(observed) != 1L || !length(perm_p)) return(NULL)
    data.frame(
      cohort = observed$cohort, method = observed$method, dataset = observed$dataset,
      group_mode = observed$group_mode, diet_col = observed$diet_col,
      feature_id = observed$feature_id,
      taxon = if ("taxon" %in% names(observed)) observed$taxon else NA_character_,
      effect_observed = observed$effect_size,
      p_observed = observed$p_value,
      q_observed = observed$q_value,
      n_effective_permutations = length(perm_p),
      empirical_p_pilot = (1 + sum(perm_p <= observed$p_value)) / (1 + length(perm_p)),
      stringsAsFactors = FALSE
    )
  })
  out <- append_rows(rows)
  if (!nrow(out)) return(out)
  out$empirical_q_pilot <- NA_real_
  task_keys <- c("cohort", "method", "dataset", "group_mode", "diet_col")
  task_split <- interaction(out[task_keys], drop = TRUE, lex.order = TRUE)
  for (idx in split(seq_len(nrow(out)), task_split)) {
    out$empirical_q_pilot[idx] <- p.adjust(out$empirical_p_pilot[idx], method = "BH")
  }
  out
}

run_small_permutation_validation <- function(
    n_perm = 5L,
    cohorts = c("ECAM", "DIABIMMUNE"),
    methods = c("linda", "maaslin2"),
    seed = 20260715L,
    cores = 1L,
    project_dir = .permutation_script_dir,
    out_dir = file.path(project_dir, "artifacts", "permutation_validation_small"),
    rscript = NULL,
    lib_path = paste(.libPaths(), collapse = ","),
    include_observed = TRUE,
    resume = TRUE,
    stop_on_error = FALSE) {

  n_perm <- as.integer(n_perm)
  cores <- as.integer(cores)
  seed <- as.integer(seed)
  if (is.na(n_perm) || n_perm < 1L) stop("n_perm must be at least 1.")
  if (is.na(cores) || cores < 1L) stop("cores must be at least 1.")
  methods <- unique(tolower(methods))
  if (length(setdiff(methods, c("linda", "maaslin2", "glmmpql")))) {
    stop("methods must be linda, maaslin2, and/or glmmpql.")
  }
  cohorts <- unique(toupper(cohorts))

  project_dir <- normalizePath(project_dir, winslash = "/", mustWork = TRUE)
  out_dir <- normalize_output(out_dir)
  rscript <- find_rscript(rscript)
  preflight_packages(methods, lib_path)
  targets <- default_permutation_targets(project_dir)
  targets <- targets[targets$cohort %in% cohorts, , drop = FALSE]
  if (!nrow(targets)) stop("No configured targets match cohorts: ", paste(cohorts, collapse = ", "))

  config <- data.frame(
    generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
    n_perm = n_perm, seed = seed, cores = cores,
    cohorts = paste(cohorts, collapse = ","), methods = paste(methods, collapse = ","),
    include_observed = include_observed, resume = resume, stop_on_error = stop_on_error,
    project_dir = project_dir, out_dir = out_dir, rscript = rscript,
    stringsAsFactors = FALSE
  )
  write_csv_atomic(config, file.path(out_dir, "run_config.csv"))

  taxon_results <- list()
  run_summaries <- list()
  run_manifest <- list()
  label_manifest <- list()
  permutation_ids <- if (include_observed) 0:n_perm else seq_len(n_perm)

  for (target_i in seq_len(nrow(targets))) {
    target <- targets[target_i, , drop = FALSE]
    target$cohort_dir <- normalize_existing(target$cohort_dir, "cohort directory")
    original_labels <- read_target_labels(target)
    original_counts <- label_count_text(original_labels)
    message("Target: ", target$cohort, " / ", target$group_mode, " (", original_counts, ")")

    for (permutation_id in permutation_ids) {
      permutation_seed <- seed + cohort_seed_offset(target$cohort) + as.integer(permutation_id)
      labels <- if (permutation_id == 0L) original_labels else permute_participant_labels(original_labels, permutation_seed)
      perm_tag <- sprintf("perm_%04d", permutation_id)
      perm_dir <- normalize_output(file.path(out_dir, target$cohort, perm_tag))
      labels_path <- file.path(perm_dir, "subject_groups_permuted.csv")
      if (resume && file.exists(labels_path)) {
        existing_labels <- read.csv(labels_path, stringsAsFactors = FALSE, check.names = FALSE)
        comparable <- intersect(c("participant_id", "exposure_type", "exposure_value"), names(labels))
        same_labels <- nrow(existing_labels) == nrow(labels) && all(vapply(comparable, function(column) {
          identical(as.character(existing_labels[[column]]), as.character(labels[[column]]))
        }, logical(1)))
        if (!same_labels) {
          stop(
            "Existing labels do not match the requested seed/configuration: ", labels_path,
            ". Use a new out_dir or set resume=FALSE to rerun and overwrite that pilot."
          )
        }
      }
      write_csv_atomic(labels, labels_path)
      label_manifest[[length(label_manifest) + 1L]] <- data.frame(
        cohort = target$cohort, group_mode = target$group_mode, diet_col = target$diet_col,
        permutation_id = permutation_id, permutation_seed = permutation_seed,
        is_observed = permutation_id == 0L, n_participants = nrow(labels),
        n_valid_labels = length(valid_label_index(labels)), group_counts = label_count_text(labels),
        labels_path = normalizePath(labels_path, winslash = "/", mustWork = TRUE),
        stringsAsFactors = FALSE
      )

      for (method in methods) {
        method_out_dir <- file.path(perm_dir, method)
        dir.create(method_out_dir, recursive = TRUE, showWarnings = FALSE)
        result_path <- result_file_for_method(method_out_dir, method)
        log_path <- file.path(perm_dir, "logs", paste0(method, ".log"))
        script <- if (method %in% c("linda", "glmmpql")) target$linda_script else target$maaslin2_script
        skipped_existing <- resume && file.exists(result_path) && file.info(result_path)$size > 0L

        message(sprintf("[%s %s %s] %s", target$cohort, perm_tag, method,
                        if (skipped_existing) "reusing existing output" else "running"))
        run_info <- if (skipped_existing) {
          list(exit_code = 0L, started_at = NA_character_, ended_at = NA_character_,
               elapsed_seconds = 0, error_message = "")
        } else {
          args <- analysis_args(target, labels_path, method, method_out_dir, cores, lib_path)
          run_child_script(rscript, script, args, target$cohort_dir, log_path)
        }

        status <- if (run_info$exit_code == 0L && file.exists(result_path)) "completed" else "failed"
        run_manifest[[length(run_manifest) + 1L]] <- data.frame(
          cohort = target$cohort, group_mode = target$group_mode, diet_col = target$diet_col,
          permutation_id = permutation_id, permutation_seed = permutation_seed,
          is_observed = permutation_id == 0L, method = method,
          status = status, exit_code = run_info$exit_code, reused_existing = skipped_existing,
          started_at = run_info$started_at, ended_at = run_info$ended_at,
          elapsed_seconds = run_info$elapsed_seconds, error_message = run_info$error_message,
          log_path = normalizePath(log_path, winslash = "/", mustWork = FALSE),
          result_path = normalizePath(result_path, winslash = "/", mustWork = FALSE),
          stringsAsFactors = FALSE
        )
        write_csv_atomic(append_rows(run_manifest), file.path(out_dir, "run_manifest.csv"))
        write_csv_atomic(append_rows(label_manifest), file.path(out_dir, "label_permutation_manifest.csv"))

        if (status == "failed") {
          warning("Failed: ", target$cohort, " / ", perm_tag, " / ", method,
                  ". See log: ", log_path)
          if (stop_on_error) stop("Stopping after failed child analysis.")
          next
        }

        result <- read_method_result(result_path, target$cohort, method, permutation_id, permutation_seed)
        if (nrow(result)) {
          taxon_results[[length(taxon_results) + 1L]] <- result
          run_summaries[[length(run_summaries) + 1L]] <- summarize_one_result(result)
        }
      }
    }
  }

  taxon_df <- append_rows(taxon_results)
  summary_df <- append_rows(run_summaries)
  manifest_df <- append_rows(run_manifest)
  label_df <- append_rows(label_manifest)
  paired_df <- make_paired_inflation(summary_df)
  calibration_df <- summarize_null_calibration(summary_df)
  empirical_df <- compute_empirical_pilot(taxon_df)

  write_csv_atomic(taxon_df, file.path(out_dir, "permutation_taxon_results.csv"))
  write_csv_atomic(summary_df, file.path(out_dir, "permutation_run_summary.csv"))
  write_csv_atomic(paired_df, file.path(out_dir, "paired_inflation_completed_vs_observed.csv"))
  write_csv_atomic(calibration_df, file.path(out_dir, "null_calibration_summary.csv"))
  write_csv_atomic(empirical_df, file.path(out_dir, "empirical_pilot.csv"))
  write_csv_atomic(manifest_df, file.path(out_dir, "run_manifest.csv"))
  write_csv_atomic(label_df, file.path(out_dir, "label_permutation_manifest.csv"))

  message("Permutation pilot finished. Output: ", out_dir)
  message("Important: empirical_pilot.csv is a workflow check only; small B has insufficient BH resolution.")
  invisible(list(
    taxon_results = taxon_df,
    run_summary = summary_df,
    paired_inflation = paired_df,
    null_calibration = calibration_df,
    empirical_pilot = empirical_df,
    run_manifest = manifest_df,
    label_manifest = label_df,
    out_dir = out_dir
  ))
}

print_small_permutation_strategy <- function() {
  cat(paste0(
    "Recommended R Console sequence:\n",
    "  1) source('permutation_validation_small.R')\n",
    "  2) default_permutation_targets()\n",
    "  3) Smoke test LinDA only:\n",
    "     pilot1 <- run_small_permutation_validation(n_perm=2, cohorts='ECAM', methods='linda')\n",
    "  4) Small paired pilot, both cohorts and engines:\n",
    "     pilot2 <- run_small_permutation_validation(n_perm=5, cohorts=c('ECAM','DIABIMMUNE'), methods=c('linda','maaslin2'))\n",
    "  5) Inspect:\n",
    "     pilot2$run_manifest\n",
    "     pilot2$null_calibration\n",
    "     pilot2$paired_inflation\n"
  ))
  invisible(NULL)
}

if (sys.nframe() == 0L && length(grep("^--file=", commandArgs(FALSE)))) {
  cli <- parse_cli_args(commandArgs(trailingOnly = TRUE))
  run_small_permutation_validation(
    n_perm = as.integer(cli$n_perm %||% 5L),
    cohorts = split_csv_arg(cli$cohorts %||% "ECAM,DIABIMMUNE"),
    methods = split_csv_arg(cli$methods %||% "linda,maaslin2"),
    seed = as.integer(cli$seed %||% 20260715L),
    cores = as.integer(cli$cores %||% 1L),
    project_dir = cli$project_dir %||% .permutation_script_dir,
    out_dir = cli$out_dir %||% file.path(.permutation_script_dir, "artifacts", "permutation_validation_small"),
    rscript = cli$rscript %||% NULL,
    lib_path = cli$lib_path %||% paste(.libPaths(), collapse = ","),
    include_observed = as_flag(cli$include_observed %||% TRUE, TRUE),
    resume = as_flag(cli$resume %||% TRUE, TRUE),
    stop_on_error = as_flag(cli$stop_on_error %||% FALSE, FALSE)
  )
}
