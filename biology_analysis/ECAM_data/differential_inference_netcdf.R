#!/usr/bin/env Rscript

# Subject-group differential inference with existing microbiome methods.
# The external subject-group path uses participant_idx as the merge key and
# treats exposure_type only as the modeling group.

script_args <- commandArgs(FALSE)
script_file_arg <- grep("^--file=", script_args, value = TRUE)
script_dir <- if (length(script_file_arg)) dirname(normalizePath(sub("^--file=", "", script_file_arg[[1]]))) else getwd()
source(file.path(script_dir, "netcdf_existing_method_helpers.R"))
source(file.path(dirname(script_dir), "glmmpql_longitudinal_helpers.R"))

get_arg_any <- function(args, names, default = NULL) {
  for (name in names) {
    if (!is.null(args[[name]])) return(args[[name]])
  }
  default
}

as_bool <- function(x, default = FALSE) {
  if (is.null(x) || length(x) == 0) return(default)
  x <- tolower(trimws(as.character(x[[1]])))
  if (x %in% c("1", "true", "t", "yes", "y")) return(TRUE)
  if (x %in% c("0", "false", "f", "no", "n")) return(FALSE)
  default
}

read_csv_checked <- function(path, required_cols = character()) {
  dat <- read_delimited_auto(path)
  missing <- setdiff(required_cols, names(dat))
  if (length(missing)) stop(path, " is missing columns: ", paste(missing, collapse = ", "))
  dat
}

is_reliable_tensor_participant_ids <- function(ids, n) {
  !is.null(ids) &&
    length(ids) == n &&
    !any(is.na(ids)) &&
    !anyDuplicated(ids) &&
    !identical(as.character(ids), as.character(seq_len(n)))
}

read_or_build_participant_order <- function(path, tensor, subject_groups, out_dir, participant_index_path = NULL) {
  n_participants <- length(tensor$participant)
  if (file.exists(path)) {
    message("Reading participant order: ", path)
    order <- read_delimited_auto(path)
  } else if (!is.null(participant_index_path) && nzchar(participant_index_path) && file.exists(participant_index_path)) {
    message("participant_order.csv not found; reading tensor participant index: ", participant_index_path)
    order <- read_participant_index_order(
      participant_index_path,
      n_participants,
      participant_ids = unique(subject_groups$participant_id)
    )
    write.csv(order, path, row.names = FALSE)
    message("Generated participant order: ", path)
  } else if (is_reliable_tensor_participant_ids(tensor$participant, n_participants)) {
    message("participant_order.csv not found; using tensor participant coordinate.")
    order <- data.frame(
      participant_idx = seq_len(n_participants),
      tensor_participant_index = seq_len(n_participants) - 1L,
      participant_id = tensor$participant,
      stringsAsFactors = FALSE
    )
    write.csv(order, path, row.names = FALSE)
    message("Generated participant order: ", path)
  } else {
    template_path <- file.path(out_dir, "participant_order_template.csv")
    template <- data.frame(
      participant_idx = seq_len(n_participants),
      participant_id = rep("", n_participants),
      stringsAsFactors = FALSE
    )
    write.csv(template, template_path, row.names = FALSE, na = "")
    stop("Cannot reliably map tensor participant_idx to metadata participant_id. Wrote template: ", template_path)
  }

  order <- normalize_participant_order(
    order,
    n_participants,
    participant_ids = unique(subject_groups$participant_id)
  )

  if (is_reliable_tensor_participant_ids(tensor$participant, n_participants) &&
      !identical(as.character(order$participant_id), as.character(tensor$participant))) {
    stop("participant_order participant_id order does not match tensor participant coordinate.")
  }

  order
}

classify_skip_type <- function(reason, default = "unknown_error") {
  reason <- as.character(reason %||% "")
  if (grepl("Only one valid group|Each group must have at least|Too few valid participants|Expected exactly two usable levels", reason)) {
    return("invalid_group")
  }
  if (grepl("align|aligned|metadata|feature|sample|dimension|participant_idx|coverage|direction|row names|column names", reason, ignore.case = TRUE)) {
    return("linda_input_error")
  }
  if (grepl("Cannot identify LinDA result table|undefined columns selected|result table|available object names", reason, ignore.case = TRUE)) {
    return("result_parse_error")
  }
  if (grepl("linda|MicrobiomeStat|model|fit|formula|singular|convergence", reason, ignore.case = TRUE)) {
    return("linda_fit_error")
  }
  default
}

append_skip <- function(skipped, group_mode, diet_col, dataset, reason,
                        skip_type = NULL, n_valid = NA_integer_,
                        counts = NA_character_, invalid_n = NA_integer_) {
  reason <- as.character(reason %||% "")
  if (is.null(skip_type) || length(skip_type) != 1 || is.na(skip_type)) {
    skip_type <- classify_skip_type(reason)
  }
  skipped[[length(skipped) + 1]] <- data.frame(
    subject_group_mode = group_mode %||% NA_character_,
    group_mode = group_mode %||% NA_character_,
    diet_col = diet_col %||% NA_character_,
    dataset = dataset %||% NA_character_,
    skip_type = skip_type,
    reason = reason,
    n_valid_participants = n_valid,
    group_counts = counts,
    invalid_group_n = invalid_n,
    stringsAsFactors = FALSE
  )
  skipped
}

empty_skipped_df <- function() {
  data.frame(
    subject_group_mode = character(),
    group_mode = character(),
    diet_col = character(),
    dataset = character(),
    skip_type = character(),
    reason = character(),
    n_valid_participants = integer(),
    group_counts = character(),
    invalid_group_n = integer(),
    stringsAsFactors = FALSE
  )
}

write_skipped_report <- function(skipped, path) {
  if (length(skipped)) {
    write.csv(do.call(rbind, skipped), path, row.names = FALSE)
  } else {
    write.csv(empty_skipped_df(), path, row.names = FALSE)
  }
  message("Wrote skipped report: ", path)
}

task_label <- function(group_mode, diet_col) {
  group_mode <- as.character(group_mode %||% "")
  diet_col <- as.character(diet_col %||% "")
  if (length(diet_col) != 1 || is.na(diet_col) || !nzchar(diet_col)) {
    make_safe_name(group_mode)
  } else {
    paste(make_safe_name(group_mode), make_safe_name(diet_col), sep = "_")
  }
}

prepare_subject_meta_for_task <- function(task_rows, participant_order, n_participants, min_group_n = 2) {
  task_rows$diet_col[is.na(task_rows$diet_col)] <- ""
  if (anyDuplicated(task_rows[, c("participant_id", "group_mode", "diet_col")])) {
    stop("Duplicate subject group rows for the same participant/group_mode/diet_col.")
  }

  task <- merge(participant_order, task_rows, by = "participant_id", all.x = TRUE, sort = FALSE)
  task <- task[order(task$participant_idx), , drop = FALSE]
  if (!identical(task$participant_idx, seq_len(n_participants))) {
    stop("Task rows do not cover tensor participant_idx 1:", n_participants, ".")
  }

  invalid <- is.na(task$exposure_type) | !nzchar(task$exposure_type) | task$exposure_type == "missing"
  valid <- task[!invalid, , drop = FALSE]
  if (nrow(valid) < min_group_n * 2) {
    stop("Too few valid participants after removing missing groups: ", nrow(valid), ".")
  }

  values <- unique(valid$exposure_type)
  if (length(values) < 2) {
    stop("Only one valid group: ", paste(values, collapse = ", "))
  }

  ref_levels <- unique(clean_missing(valid$reference_level %||% NA_character_))
  contrast_levels <- unique(clean_missing(valid$contrast_level %||% NA_character_))
  ref_levels <- ref_levels[!is.na(ref_levels)]
  contrast_levels <- contrast_levels[!is.na(contrast_levels)]
  group_modes <- unique(valid$group_mode)
  if (length(ref_levels) == 1L && length(contrast_levels) == 1L &&
      all(c(ref_levels[[1]], contrast_levels[[1]]) %in% values)) {
    levels <- c(ref_levels[[1]], contrast_levels[[1]])
  } else if (length(group_modes) == 1 && identical(group_modes[[1]], "delivery_group")) {
    levels <- c("Vaginal", "Cesarean")
    levels <- levels[levels %in% values]
  } else if (length(group_modes) == 1 && identical(group_modes[[1]], "abx_group")) {
    levels <- c("abx_unexposed", "abx_exposed")
    levels <- levels[levels %in% values]
  } else if (length(group_modes) == 1 && identical(group_modes[[1]], "diet_group")) {
    levels <- c("early", "late")
    levels <- levels[levels %in% values]
  } else {
    levels <- sort(values)
  }
  if (length(levels) != 2) {
    stop("Expected exactly two usable levels, got: ", paste(values, collapse = ", "))
  }

  valid$subject_group <- factor(valid$exposure_type, levels = levels)
  counts <- table(valid$subject_group)
  if (any(counts < min_group_n)) {
    stop("Each group must have at least ", min_group_n, " participants; counts: ",
         paste(names(counts), as.integer(counts), sep = "=", collapse = ", "))
  }
  valid$subject_group_binary <- ifelse(valid$subject_group == levels[[2]], 1, 0)
  valid$reference_level <- levels[[1]]
  valid$contrast_level <- levels[[2]]
  keep_cols <- c(
    "participant_idx", "tensor_participant_index", "participant_id",
    "exposure_type", "subject_group", "subject_group_binary",
    "reference_level", "contrast_level"
  )
  valid[, intersect(keep_cols, names(valid)), drop = FALSE]
}

# ---------------------------------------------------------------------------
# Two analysis modes replace the previous interaction/random-effect LinDA path,
# which failed under R >= 4.2 because MicrobiomeStat::linda evaluates an
# `if (<vector>)` (the "condition has length > 1" error) when handling
# `subject_group:bin_c` and `(1 | participant)` terms.
#
# Mode 1 (primary):    bin-stratified LinDA, abundance ~ subject_group per bin.
# Mode 2 (supplement): per-taxon lmer longitudinal trajectory LRT.
# ---------------------------------------------------------------------------

# Apply BH-FDR to a p-value column within groups defined by group_cols.
add_group_bh <- function(df, p_col, q_col, group_cols) {
  df[[q_col]] <- NA_real_
  if (!nrow(df)) return(df)
  key <- do.call(paste, c(lapply(group_cols, function(g) as.character(df[[g]])), sep = "\r"))
  for (g in unique(key)) {
    idx <- which(key == g)
    p <- suppressWarnings(as.numeric(df[[p_col]][idx]))
    ok <- is.finite(p)
    q <- rep(NA_real_, length(p))
    if (any(ok)) q[ok] <- stats::p.adjust(p[ok], method = "BH")
    df[[q_col]][idx] <- q
  }
  df
}

map_feature_to_taxon <- function(feature_ids, feature_map) {
  taxon <- as.character(feature_ids)
  if (!is.null(feature_map) && all(c("feature_id", "taxon") %in% names(feature_map))) {
    idx <- match(feature_ids, feature_map$feature_id)
    matched <- !is.na(idx)
    taxon[matched] <- feature_map$taxon[idx[matched]]
  }
  taxon
}

# Call linda() with only the arguments the installed implementation accepts.
# NOTE: linda() requires `formula` as a CHARACTER STRING (e.g. "~ subject_group").
# Passing a formula object makes linda's internal `grepl("\\(", formula)` return a
# length-2 vector, triggering "the condition has length > 1" under R >= 4.2.
call_linda_simple <- function(linda_fun, feature_mat, metadata_df, formula_text = "~ subject_group", cores = 1) {
  args <- list(
    feature.dat = feature_mat,
    meta.dat = metadata_df,
    formula = formula_text,
    feature.dat.type = "proportion",
    prev.filter = 0,
    mean.abund.filter = 0,
    max.abund.filter = 0,
    zero.handling = "pseudo-count",
    p.adj.method = "BH",
    is.winsor = FALSE,
    n.cores = cores,
    verbose = FALSE
  )
  args <- args[names(args) %in% names(formals(linda_fun))]
  do.call(linda_fun, args)
}

# Pull the subject_group contrast table out of a linda() fit.
extract_linda_subject_group_table <- function(fit) {
  out <- fit$output
  if (is.null(out) || !length(out)) stop("LinDA returned no output tables.")
  nms <- names(out)
  if (is.null(nms)) nms <- as.character(seq_along(out))
  idx <- grep("^subject_group", nms)
  if (!length(idx)) idx <- 1L
  df <- out[[idx[[1]]]]
  if (is.null(df) || !nrow(df)) stop("LinDA subject_group output table is empty.")
  cols <- colnames(df)
  effect_col <- if ("log2FoldChange" %in% cols) "log2FoldChange" else cols[[1]]
  p_col <- if ("pvalue" %in% cols) "pvalue" else NA_character_
  q_col <- if ("padj" %in% cols) "padj" else NA_character_
  df_col <- if ("df" %in% cols) "df" else NA_character_
  data.frame(
    feature_id = rownames(df),
    effect_size = suppressWarnings(as.numeric(df[[effect_col]])),
    p_value = if (!is.na(p_col)) suppressWarnings(as.numeric(df[[p_col]])) else NA_real_,
    q_value = if (!is.na(q_col)) suppressWarnings(as.numeric(df[[q_col]])) else NA_real_,
    df = if (!is.na(df_col)) suppressWarnings(as.numeric(df[[df_col]])) else NA_real_,
    stringsAsFactors = FALSE
  )
}

# Longitudinal LinDA: ONE inference per taxon, testing the subject_group effect
# while controlling for bin (time) and accounting for repeated measures per
# participant via a random intercept. This is the primary differential-abundance
# analysis. The feature table is the full participant x bin long table; missing
# timepoints are simply absent rows (the mixed model handles unbalanced data).
#
# Default formula: ~ subject_group + bin_c + (1 | participant)
#   - is.winsor MUST be FALSE: winsorization corrupts the centred log-ratio
#     matrix (all-NaN) on this longitudinal data, surfacing as a misleading
#     "contrasts" error.
#   - The reported q_value is LinDA's own BH adjustment across the tested taxa
#     within this single model (the correct per-task FDR).
run_longitudinal_linda <- function(tables, dataset_name, group_mode, diet_col, task_name,
                                    linda_fun,
                                    formula_text = "~ subject_group + bin_c + (1 | participant)",
                                    cores = 1, debug = TRUE) {
  meta <- tables$metadata
  feat <- tables$features
  feature_map <- tables$feature_map

  meta$subject_group <- droplevels(factor(meta$subject_group))
  meta$participant <- droplevels(factor(meta$participant))

  # Participant-level group counts (each participant contributes many samples).
  first_group <- tapply(as.character(meta$subject_group), meta$participant, function(x) x[[1]])
  gcounts <- table(first_group)
  counts_text <- paste(names(gcounts), as.integer(gcounts), sep = "=", collapse = ", ")
  n_participants <- nlevels(meta$participant)
  n_samples <- nrow(meta)

  if (debug) {
    message(sprintf("  [%s | %s] samples=%d, participants=%d, features=%d, groups: %s",
                    dataset_name, task_name, n_samples, n_participants, ncol(feat), counts_text))
  }
  if (nlevels(meta$subject_group) < 2) {
    stop("Only one valid group: ", counts_text)
  }

  # Drop features that are all-zero across the whole table (LinDA's proportion
  # zero-handling would otherwise compute 0.5 * min(numeric(0)) = Inf and poison
  # the log-ratio matrix). Such features carry no information for this task.
  feat_mat0 <- as.matrix(feat)
  nz <- colSums(feat_mat0, na.rm = TRUE) > 0
  n_dropped <- sum(!nz)
  if (sum(nz) < 2) {
    stop("Fewer than 2 non-empty features after dropping all-zero columns (", sum(nz), " kept).")
  }
  if (debug && n_dropped > 0) {
    message(sprintf("    dropped %d all-zero features; %d features tested", n_dropped, sum(nz)))
  }

  feat_in <- t(feat_mat0[, nz, drop = FALSE])
  storage.mode(feat_in) <- "double"
  colnames(feat_in) <- rownames(meta)
  if (!identical(colnames(feat_in), rownames(meta))) {
    stop("Feature matrix columns are not aligned to metadata rows.")
  }

  fit <- call_linda_simple(linda_fun, feat_in, meta, formula_text, cores = cores)
  tbl <- extract_linda_subject_group_table(fit)
  feature_meta <- feature_metadata_for_results(tbl$feature_id, feature_map)
  ref_level <- levels(meta$subject_group)[[1]]
  contrast_level <- levels(meta$subject_group)[[2]]

  task_df <- data.frame(
    dataset = dataset_name,
    group_mode = group_mode,
    diet_col = diet_col,
    feature_id = tbl$feature_id,
    reference_level = ref_level,
    contrast_level = contrast_level,
    effect_size = tbl$effect_size,
    p_value = tbl$p_value,
    q_value = tbl$q_value,
    df = tbl$df,
    n_samples = n_samples,
    n_participants = n_participants,
    group_counts = counts_text,
    formula = formula_text,
    stringsAsFactors = FALSE
  )
  cbind(
    task_df[, c("dataset", "group_mode", "diet_col", "feature_id", "reference_level", "contrast_level"), drop = FALSE],
    feature_meta[, setdiff(names(feature_meta), "feature_id"), drop = FALSE],
    task_df[, setdiff(names(task_df), c("dataset", "group_mode", "diet_col", "feature_id", "reference_level", "contrast_level")), drop = FALSE]
  )
}

run_external_subject_groups <- function(args, base_tensor, imputed_tensor, out_dir, engine, cores) {
  subject_groups_path <- get_arg_any(args, c("subject-groups", "subject_groups"), file.path(out_dir, "subject_groups_long.csv"))
  participant_order_path <- get_arg_any(args, c("participant-order", "participant_order"), file.path(out_dir, "participant_order.csv"))
  participant_index_path <- get_arg_any(args, c("participant-index", "participant_index"),
                                        file.path(dirname(base_tensor$path), "tensor_participant_index.csv"))
  min_group_n <- as.integer(get_arg_any(args, c("min-group-n", "min_group_n"), "2"))
  debug_linda <- as_bool(get_arg_any(args, c("debug-linda", "debug_linda"), "true"), default = TRUE)
  debug_glmmpql <- as_bool(get_arg_any(args, c("debug-glmmpql", "debug_glmmpql"), "true"), default = TRUE)
  glmmpql_niter <- as.integer(get_arg_any(args, c("glmmpql-niter", "glmmpql_niter"), "10"))
  # Longitudinal LinDA model. Default accounts for repeated measures within a
  # participant via a random intercept; override with --linda-formula if needed.
  linda_formula <- get_arg_any(args, c("linda-formula", "linda_formula"),
                               "~ subject_group + bin_c + (1 | participant)")

  message("Reading subject groups: ", subject_groups_path)
  subject_groups <- read_csv_checked(subject_groups_path, c(
    "participant_id", "group_mode", "diet_col", "exposure_type", "exposure_value", "rule"
  ))
  subject_groups$participant_id <- clean_missing(subject_groups$participant_id)
  subject_groups$group_mode <- clean_missing(subject_groups$group_mode)
  subject_groups$diet_col <- clean_missing(subject_groups$diet_col)
  subject_groups$exposure_type <- clean_missing(subject_groups$exposure_type)
  subject_groups$diet_col[is.na(subject_groups$diet_col)] <- ""
  if (any(is.na(subject_groups$participant_id)) || any(is.na(subject_groups$group_mode))) {
    stop("subject_groups_long.csv contains missing participant_id or group_mode.")
  }

  participant_order <- read_or_build_participant_order(
    participant_order_path,
    base_tensor,
    subject_groups,
    out_dir,
    participant_index_path = participant_index_path
  )
  message("participant_idx coverage: ", paste(range(participant_order$participant_idx), collapse = "-"),
          "; n=", nrow(participant_order))

  idx_groups <- merge(subject_groups, participant_order, by = "participant_id", all.x = TRUE, sort = FALSE)
  write.csv(
    idx_groups[order(idx_groups$participant_idx, idx_groups$group_mode, idx_groups$diet_col), ],
    file.path(out_dir, "participant_idx_subject_groups_long.csv"),
    row.names = FALSE
  )

  tasks <- unique(subject_groups[, c("group_mode", "diet_col"), drop = FALSE])
  requested_modes <- get_arg_any(args, c("group-modes", "group_modes"), "")
  if (nzchar(requested_modes)) {
    keep_modes <- split_arg(requested_modes)
    tasks <- tasks[tasks$group_mode %in% keep_modes, , drop = FALSE]
  }
  if (!nrow(tasks)) stop("No subject-group tasks remain after filtering.")
  datasets <- list(base_raw = base_tensor, p2d2pca_imputed = imputed_tensor)
  skipped <- list()                 # task-level skips -> skipped_subject_groups.csv
  method_results <- list()          # one result table per task/dataset
  linda_fun <- if (engine == "glmmpql") NULL else find_linda_fun()

  # Prepare participant-level metadata once per task (cheap, reused by both passes).
  prepared <- list()
  for (i in seq_len(nrow(tasks))) {
    group_mode <- tasks$group_mode[[i]]
    diet_col <- tasks$diet_col[[i]]
    task_rows <- subject_groups[subject_groups$group_mode == group_mode & subject_groups$diet_col == diet_col, , drop = FALSE]
    label <- task_label(group_mode, diet_col)
    message("Preparing group task: ", label)

    subject_meta <- try(
      prepare_subject_meta_for_task(task_rows, participant_order, length(base_tensor$participant), min_group_n = min_group_n),
      silent = TRUE
    )
    if (inherits(subject_meta, "try-error")) {
      reason <- conditionMessage(attr(subject_meta, "condition"))
      warning("Skipping ", label, ": ", reason)
      skipped <- append_skip(skipped, group_mode, diet_col, NA_character_, reason, skip_type = "invalid_group")
      next
    }
    counts <- table(subject_meta$subject_group)
    counts_text <- paste(names(counts), as.integer(counts), sep = "=", collapse = ", ")
    message("  group counts: ", counts_text)
    prepared[[length(prepared) + 1]] <- list(
      group_mode = group_mode, diet_col = diet_col, label = label,
      subject_meta = subject_meta, counts_text = counts_text
    )
  }

  # Build the long (participant x bin) method tables for a task/dataset.
  build_tables <- function(p, dataset_name) {
    tables <- tensor_to_method_tables_by_idx(
      tensor = datasets[[dataset_name]],
      subject_meta = p$subject_meta,
      extra_cols = c(
        "participant_id", "exposure_type", "subject_group", "subject_group_binary",
        "reference_level", "contrast_level"
      )
    )
    tables$metadata$subject_group <- factor(tables$metadata$subject_group,
                                            levels = levels(p$subject_meta$subject_group))
    tables$metadata$exposure_type <- as.character(tables$metadata$subject_group)
    tables$metadata$participant <- factor(tables$metadata$participant)
    if (engine == "glmmpql") tables <- glmmpql_use_tensor_log_values(tables, datasets[[dataset_name]])
    tables
  }

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  # ===== Longitudinal inference: one inference per taxon, controlling for bin. =====
  for (p in prepared) {
    for (dataset_name in names(datasets)) {
      tag <- paste(dataset_name, p$label, sep = "_")
      tables <- try(build_tables(p, dataset_name), silent = TRUE)
      if (inherits(tables, "try-error")) {
        reason <- conditionMessage(attr(tables, "condition"))
        warning("Skipping ", tag, ": ", reason)
        skipped <- append_skip(skipped, p$group_mode, p$diet_col, dataset_name, reason,
                               skip_type = paste0(engine, "_input_error"),
                               n_valid = nrow(p$subject_meta), counts = p$counts_text)
        next
      }
      write.csv(tables$feature_map, file.path(out_dir, paste0(tag, "_feature_map.csv")), row.names = FALSE)
      write.csv(tables$metadata, file.path(out_dir, paste0(tag, "_sample_metadata.csv")), row.names = FALSE)

      message("Running longitudinal ", if (engine == "glmmpql") "glmmPQL" else "LinDA", " for ", tag)
      res <- try({
        if (engine == "glmmpql") {
          run_longitudinal_glmmpql(
            tables = tables, dataset_name = dataset_name, group_mode = p$group_mode,
            diet_col = p$diet_col, task_name = p$label,
            niter = glmmpql_niter, debug = debug_glmmpql
          )
        } else {
          run_longitudinal_linda(
            tables = tables, dataset_name = dataset_name, group_mode = p$group_mode,
            diet_col = p$diet_col, task_name = p$label, linda_fun = linda_fun,
            formula_text = linda_formula, cores = cores, debug = debug_linda
          )
        }
      }, silent = TRUE)
      if (inherits(res, "try-error")) {
        reason <- conditionMessage(attr(res, "condition"))
        skip_type <- classify_skip_type(reason, default = paste0(engine, "_fit_error"))
        warning(if (engine == "glmmpql") "glmmPQL" else "LinDA", " failed for ", tag, ": ", reason)
        skipped <- append_skip(skipped, p$group_mode, p$diet_col, dataset_name, reason,
                               skip_type = skip_type,
                               n_valid = nrow(p$subject_meta), counts = p$counts_text)
      } else {
        method_results[[length(method_results) + 1]] <- res
      }
    }
  }

  # --- Output: one row per taxon per task; q_value is per-task BH FDR. ---
  result_path <- file.path(out_dir, if (engine == "glmmpql") {
    "glmmpql_longitudinal_results.csv"
  } else {
    "linda_longitudinal_results.csv"
  })
  out_cols <- c("dataset", "group_mode", "diet_col", "feature_id",
                "taxon", "taxon_label", "species_name",
                "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species", "lineage",
                "reference_level", "contrast_level",
                "effect_size", "std_error", "p_value", "q_value", "df", "fit_status", "da_method",
                "n_samples", "n_participants", "group_counts", "formula")
  if (length(method_results)) {
    result_df <- do.call(rbind, method_results)
    out_cols <- c(out_cols[out_cols %in% names(result_df)], setdiff(names(result_df), out_cols))
    write.csv(result_df[, out_cols], result_path, row.names = FALSE)
    message("Wrote longitudinal ", if (engine == "glmmpql") "glmmPQL" else "LinDA",
            " results (", nrow(result_df), " rows): ", result_path)
  } else {
    empty <- as.data.frame(setNames(rep(list(character()), length(out_cols)), out_cols))
    write.csv(empty, result_path, row.names = FALSE)
    message("No longitudinal results were produced; wrote empty table: ", result_path)
  }
  write_skipped_report(skipped, file.path(out_dir, "skipped_subject_groups.csv"))
}

run_legacy_internal_groups <- function(args, base_tensor, imputed_tensor, meta_path, out_dir, engine, cores) {
  message("No external --subject-groups file was supplied/found; using legacy internal grouping.")
  meta <- read_metadata(meta_path)
  diet_cols_arg <- get_arg_any(args, c("diet_cols", "diet-cols"), "ALL_DIETARY")
  subject_group_modes <- split_arg(get_arg_any(args, c("subject_group_modes", "subject-group-modes"), "antibiotic_use"))
  min_group_n <- as.integer(get_arg_any(args, c("min_group_n", "min-group-n"), "4"))
  linda_formula <- get_arg_any(args, c("linda-formula", "linda_formula"),
                               "~ subject_group + bin_c + (1 | participant)")
  diet_cols <- resolve_diet_cols(meta, diet_cols_arg)
  participants <- Reduce(intersect, list(base_tensor$participant, imputed_tensor$participant))
  datasets <- list(base_raw = base_tensor, p2d2pca_imputed = imputed_tensor)
  skipped <- list()
  linda_results <- list()
  linda_fun <- find_linda_fun()

  for (group_mode in subject_group_modes) {
    message("Preparing legacy subject group mode: ", group_mode)
    subject_meta <- try(
      build_subject_group_metadata(meta, participants, group_mode, diet_cols, min_group_n = min_group_n),
      silent = TRUE
    )
    if (inherits(subject_meta, "try-error")) {
      reason <- conditionMessage(attr(subject_meta, "condition"))
      warning("Skipping legacy group mode ", group_mode, ": ", reason)
      skipped <- append_skip(skipped, group_mode, NA_character_, NA_character_, reason, skip_type = "invalid_group")
      next
    }
    for (dataset_name in names(datasets)) {
      tables <- tensor_to_method_tables(
        tensor = datasets[[dataset_name]],
        subject_meta = subject_meta,
        extra_cols = c("subject_group", "subject_group_binary")
      )
      tables$metadata$subject_group <- factor(tables$metadata$subject_group,
                                              levels = levels(subject_meta$subject_group))
      tables$metadata$exposure_type <- as.character(tables$metadata$subject_group)
      tables$metadata$participant <- factor(tables$metadata$participant)

      res <- try(
        run_longitudinal_linda(
          tables = tables, dataset_name = dataset_name, group_mode = make_safe_name(group_mode),
          diet_col = "", task_name = make_safe_name(group_mode), linda_fun = linda_fun,
          formula_text = linda_formula, cores = cores, debug = FALSE
        ),
        silent = TRUE
      )
      if (inherits(res, "try-error")) {
        skipped <- append_skip(skipped, group_mode, NA_character_, dataset_name,
                               conditionMessage(attr(res, "condition")))
      } else {
        linda_results[[length(linda_results) + 1]] <- res
      }
    }
  }

  linda_path <- file.path(out_dir, "linda_longitudinal_results.csv")
  out_cols <- c("dataset", "group_mode", "diet_col", "feature_id",
                "taxon", "taxon_label", "species_name",
                "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species", "lineage",
                "reference_level", "contrast_level",
                "effect_size", "p_value", "q_value", "df",
                "n_samples", "n_participants", "group_counts", "formula")
  if (length(linda_results)) {
    linda_df <- do.call(rbind, linda_results)
    out_cols <- c(out_cols[out_cols %in% names(linda_df)], setdiff(names(linda_df), out_cols))
    write.csv(linda_df[, out_cols], linda_path, row.names = FALSE)
    message("Wrote longitudinal LinDA results (", nrow(linda_df), " rows): ", linda_path)
  }

  write_skipped_report(skipped, file.path(out_dir, "skipped_subject_groups.csv"))
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  lib_path <- get_arg_any(args, c("lib_path", "lib-path"), "")
  if (nzchar(lib_path)) .libPaths(c(split_arg(lib_path), .libPaths()))

  base_nc <- get_arg_any(args, c("base_nc", "base-nc"), "ECAM_otu_tensor_export/tensor_base_raw.nc")
  imputed_nc <- get_arg_any(args, c("imputed_nc", "imputed-nc"), "ECAM_otu_tensor_export/tensor_p2d2pca_imputed.nc")
  meta_path <- get_arg_any(args, c("meta", "metadata"), "otu_metadata_cleaned_q2.csv")
  out_dir <- get_arg_any(args, c("out_dir", "outdir"), "artifacts/ECAM_otu_linda_longitudinal")
  engine <- tolower(get_arg_any(args, c("engine"), "linda"))
  cores <- as.integer(get_arg_any(args, c("cores"), "1"))
  taxon_index_path <- get_arg_any(args, c("taxon-index", "taxon_index"),
                                  file.path(dirname(base_nc), "tensor_taxon_index.csv"))
  taxonomy_path <- get_arg_any(args, c("taxonomy", "taxonomy_path"), "otu_taxonomy_cleaned_q2.csv")

  if (!engine %in% c("linda", "maaslin2", "both", "glmmpql")) {
    stop("--engine must be one of: linda, maaslin2, both, glmmpql")
  }
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  message("Reading tensor: ", base_nc)
  base_tensor <- read_tensor_nc(base_nc)
  message("Reading tensor: ", imputed_nc)
  imputed_tensor <- read_tensor_nc(imputed_nc)
  base_tensor <- attach_taxon_annotations(base_tensor, taxon_index_path, taxonomy_path)
  imputed_tensor <- attach_taxon_annotations(imputed_tensor, taxon_index_path, taxonomy_path)
  if (length(base_tensor$participant) != length(imputed_tensor$participant)) {
    stop("Base and imputed tensors have different participant counts.")
  }

  subject_groups_arg <- get_arg_any(args, c("subject-groups", "subject_groups"), "")
  default_subject_groups <- file.path(out_dir, "subject_groups_long.csv")
  if (nzchar(subject_groups_arg) || file.exists(default_subject_groups)) {
    run_external_subject_groups(args, base_tensor, imputed_tensor, out_dir, engine, cores)
  } else if (engine == "glmmpql") {
    stop("--engine glmmpql requires an external --subject-groups file.")
  } else if (as_bool(get_arg_any(args, c("legacy-internal-groups", "legacy_internal_groups"), "false"), default = FALSE)) {
    run_legacy_internal_groups(args, base_tensor, imputed_tensor, meta_path, out_dir, engine, cores)
  } else {
    stop(
      "Subject group file not found: ", default_subject_groups,
      ". Run build_subject_groups.R first, or pass --subject-groups, or set --legacy-internal-groups true."
    )
  }

  message("Done. Outputs: ", out_dir)
}

tryCatch(
  main(),
  error = function(e) {
    message("ERROR: ", conditionMessage(e))
    quit(status = 1)
  }
)
