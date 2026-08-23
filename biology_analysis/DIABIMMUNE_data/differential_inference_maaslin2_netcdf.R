#!/usr/bin/env Rscript

# =============================================================================
# Subject-group differential inference — MaAsLin2 variant (DIABIMMUNE).
#
# WHAT THIS IS
#   A drop-in sibling of differential_inference_netcdf.R that keeps the EXACT
#   same data pre-processing and result-collation pipeline, and swaps ONLY the
#   differential-abundance engine from LinDA to MaAsLin2.
#
#   - Pre-processing is reused verbatim from netcdf_existing_method_helpers.R
#     (read_tensor_nc, tensor_to_method_tables_by_idx / tensor_to_method_tables,
#      participant-order handling, feature_map, sample_metadata, etc.).
#   - The MaAsLin2 call mirrors the modelling choices already sanctioned in this
#     repo (run_maaslin2_engine: normalization="NONE", transform="LOG",
#      analysis_method="LM", correction="BH") and the reference tuning script
#     DA_effect_sweep_maaslin2.R, adapted to the subject-group longitudinal model.
#
# MODEL CORRESPONDENCE (LinDA -> MaAsLin2)
#   LinDA : feature.dat.type="proportion";  ~ subject_group + bin_c + (1|participant)
#   MaAsLin2: input = per-sample relative abundance (same features table);
#             fixed_effects = c("subject_group", "bin_c")
#             random_effects = "participant"
#             normalization="NONE" (already proportions), transform="LOG",
#             analysis_method="LM", correction="BH".
#   The variable of interest is subject_group; bin_c is a continuous nuisance
#   covariate (NOT expanded into per-bin terms), matching the LinDA model so the
#   two engines answer the same question.
#
# MULTIPLE TESTING (important for a fair comparison)
#   LinDA reports padj = BH across taxa for the subject_group term within a task.
#   MaAsLin2's native qval in all_results.tsv is BH pooled over ALL metadata
#   terms (subject_group + bin_c) x features, a different (larger) hypothesis
#   pool. To keep the FDR scope identical to LinDA, q_value here is RECOMPUTED as
#   BH over the subject_group p-values across taxa, per task. MaAsLin2's own qval
#   is preserved separately in maaslin2_qval_native for auditing.
#
# OUTPUT
#   Writes maaslin2_longitudinal_results.csv with the SAME column schema as
#   linda_longitudinal_results.csv, so any downstream comparison script that
#   consumes the LinDA table works unchanged on the MaAsLin2 table.
#   Per-task raw MaAsLin2 runs are kept under <out_dir>/maaslin2_runs/<tag>/.
#   The LinDA results file is never overwritten (different filename); the skipped
#   report is written to skipped_subject_groups_maaslin2.csv.
#
# USAGE
#   Rscript differential_inference_maaslin2_netcdf.R
#   Rscript differential_inference_maaslin2_netcdf.R --transform LOG --cores 4
#   (Uses artifacts/subject_group_existing_methods_r/subject_groups_long.csv,
#    exactly like the LinDA run.)
# =============================================================================

script_args <- commandArgs(FALSE)
script_file_arg <- grep("^--file=", script_args, value = TRUE)
script_dir <- if (length(script_file_arg)) dirname(normalizePath(sub("^--file=", "", script_file_arg[[1]]))) else getwd()
source(file.path(script_dir, "netcdf_existing_method_helpers.R"))

# ---------------------------------------------------------------------------
# Small argument / IO utilities (copied verbatim from
# differential_inference_netcdf.R so this script is self-contained and does not
# source the LinDA script, which would auto-run its main()).
# ---------------------------------------------------------------------------
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
  if (!file.exists(path)) stop("File does not exist: ", path)
  dat <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
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

read_or_build_participant_order <- function(path, tensor, subject_groups, out_dir) {
  n_participants <- length(tensor$participant)
  if (file.exists(path)) {
    message("Reading participant order: ", path)
    order <- read_csv_checked(path, c("participant_idx", "participant_id"))
  } else if (is_reliable_tensor_participant_ids(tensor$participant, n_participants)) {
    message("participant_order.csv not found; using tensor participant coordinate.")
    order <- data.frame(
      participant_idx = seq_len(n_participants),
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

  order$participant_idx <- suppressWarnings(as.integer(order$participant_idx))
  order$participant_id <- clean_missing(order$participant_id)
  if (any(is.na(order$participant_idx)) || any(is.na(order$participant_id))) {
    stop("participant_order contains NA participant_idx or participant_id.")
  }
  if (anyDuplicated(order$participant_idx) || anyDuplicated(order$participant_id)) {
    stop("participant_order must have unique participant_idx and participant_id.")
  }
  if (!identical(sort(order$participant_idx), seq_len(n_participants))) {
    stop("participant_order participant_idx must cover tensor indices 1:", n_participants, ".")
  }
  order <- order[order(order$participant_idx), c("participant_idx", "participant_id")]

  if (is_reliable_tensor_participant_ids(tensor$participant, n_participants) &&
      !identical(as.character(order$participant_id), as.character(tensor$participant))) {
    stop("participant_order participant_id order does not match tensor participant coordinate.")
  }

  missing_ids <- setdiff(order$participant_id, unique(subject_groups$participant_id))
  if (length(missing_ids)) {
    stop("participant_order contains participant_id not present in subject groups: ", paste(missing_ids, collapse = ", "))
  }
  order
}

classify_skip_type <- function(reason, default = "unknown_error") {
  reason <- as.character(reason %||% "")
  if (grepl("Only one valid group|Each group must have at least|Too few valid participants|Expected exactly two usable levels", reason)) {
    return("invalid_group")
  }
  if (grepl("align|aligned|metadata|feature|sample|dimension|participant_idx|coverage|direction|row names|column names", reason, ignore.case = TRUE)) {
    return("input_error")
  }
  if (grepl("all_results|result table|MaAsLin2|Maaslin2|available object names", reason, ignore.case = TRUE)) {
    return("result_parse_error")
  }
  if (grepl("maaslin|model|fit|formula|singular|convergence|lmer", reason, ignore.case = TRUE)) {
    return("maaslin2_fit_error")
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

  group_modes <- unique(valid$group_mode)
  if (length(group_modes) == 1 && identical(group_modes[[1]], "abx_group")) {
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
  valid[, c("participant_idx", "participant_id", "exposure_type", "subject_group", "subject_group_binary"), drop = FALSE]
}

# BH within the subject_group term across taxa (matches LinDA's per-task FDR scope).
map_feature_to_taxon <- function(feature_ids, feature_map) {
  taxon <- as.character(feature_ids)
  if (!is.null(feature_map) && all(c("feature_id", "taxon") %in% names(feature_map))) {
    idx <- match(feature_ids, feature_map$feature_id)
    matched <- !is.na(idx)
    taxon[matched] <- feature_map$taxon[idx[matched]]
  }
  taxon
}

# ---------------------------------------------------------------------------
# MaAsLin2 engine (replaces call_linda_simple + extract_linda_subject_group_table).
# ---------------------------------------------------------------------------

# Pick the first present column name from candidates (case-insensitive fallback).
pick_col <- function(nm, candidates) {
  hit <- candidates[candidates %in% nm]
  if (length(hit)) return(hit[[1]])
  low <- tolower(nm)
  for (c in candidates) {
    j <- match(tolower(c), low)
    if (!is.na(j)) return(nm[[j]])
  }
  NA_character_
}

# Run MaAsLin2 for one (features, metadata) task and return the raw all_results
# table plus the reference/contrast levels. feature_df: samples (rows) x features
# (cols) relative abundances; meta: rownames = sample_id, with subject_group
# (2-level factor, reference = level 1), bin_c (numeric), participant (factor).
run_maaslin2_group_fit <- function(feature_df, meta, run_dir, cores = 1,
                                   normalization = "NONE", transform = "LOG") {
  require_pkg("Maaslin2")

  if (!"subject_group" %in% names(meta)) stop("metadata is missing subject_group.")
  if (!"bin_c" %in% names(meta)) stop("metadata is missing bin_c.")
  if (!"participant" %in% names(meta)) stop("metadata is missing participant.")

  ref_level <- levels(meta$subject_group)[[1]]
  contrast_level <- levels(meta$subject_group)[[2]]

  md <- data.frame(
    subject_group = factor(as.character(meta$subject_group), levels = levels(meta$subject_group)),
    bin_c = as.numeric(meta$bin_c),
    participant = factor(as.character(meta$participant)),
    stringsAsFactors = FALSE
  )
  rownames(md) <- rownames(meta)

  if (!identical(rownames(feature_df), rownames(md))) {
    stop("Feature rows are not aligned to metadata rows for MaAsLin2.")
  }

  dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
  fit <- Maaslin2::Maaslin2(
    input_data       = feature_df,        # samples x features (relative abundance)
    input_metadata   = md,                # samples x covariates
    output           = run_dir,
    fixed_effects    = c("subject_group", "bin_c"),
    random_effects   = "participant",
    reference        = paste0("subject_group,", ref_level),
    normalization    = normalization,     # NONE: features are already proportions
    transform        = transform,         # LOG: log relative abundance, LM inference
    analysis_method  = "LM",
    correction       = "BH",
    min_abundance    = 0,
    min_prevalence   = 0,
    min_variance     = 0,
    standardize      = FALSE,
    plot_heatmap     = FALSE,
    plot_scatter     = FALSE,
    cores            = cores
  )

  res_path <- file.path(run_dir, "all_results.tsv")
  if (!file.exists(res_path)) {
    stop("MaAsLin2 did not produce all_results.tsv in ", run_dir)
  }
  res <- read.delim(res_path, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
  list(res = res, ref_level = ref_level, contrast_level = contrast_level)
}

# Extract the per-taxon subject_group contrast from a MaAsLin2 all_results table.
# Returns a data.frame keyed by feature_id over the SUPPLIED tested_ids, with
# effect_size, p_value, q_value (per-task BH over the group term), maaslin2_qval_native.
extract_maaslin2_group_table <- function(res, tested_ids) {
  nm <- names(res)
  feat_col <- pick_col(nm, c("feature", "feature.names", "Feature"))
  meta_col <- pick_col(nm, c("metadata", "Metadata"))
  coef_col <- pick_col(nm, c("coef", "effect_size", "estimate", "Coefficient"))
  p_col    <- pick_col(nm, c("pval", "pvalue", "p.value", "P.value", "pval.value"))
  q_col    <- pick_col(nm, c("qval", "qvalue", "q.value", "Q.value"))
  if (is.na(feat_col) || is.na(meta_col) || is.na(coef_col) || is.na(p_col)) {
    stop("Unexpected MaAsLin2 result columns: ", paste(nm, collapse = ", "))
  }

  grp <- res[as.character(res[[meta_col]]) == "subject_group", , drop = FALSE]
  if (!nrow(grp)) stop("MaAsLin2 returned no subject_group rows in all_results.tsv.")

  grp_feature <- as.character(grp[[feat_col]])
  grp_effect  <- suppressWarnings(as.numeric(grp[[coef_col]]))
  grp_p       <- suppressWarnings(as.numeric(grp[[p_col]]))
  grp_qnative <- if (!is.na(q_col)) suppressWarnings(as.numeric(grp[[q_col]])) else NA_real_

  # For a 2-level factor there is exactly one subject_group row per feature; if a
  # future >2-level variant appears, keep the smallest-p row per feature.
  ord <- order(grp_feature, grp_p)
  grp_feature <- grp_feature[ord]; grp_effect <- grp_effect[ord]
  grp_p <- grp_p[ord]; grp_qnative <- grp_qnative[ord]
  keep <- !duplicated(grp_feature)
  grp_feature <- grp_feature[keep]; grp_effect <- grp_effect[keep]
  grp_p <- grp_p[keep]; grp_qnative <- grp_qnative[keep]

  # Map MaAsLin2 feature names back to our feature_ids (direct, then make.names).
  idx <- match(tested_ids, grp_feature)
  need <- is.na(idx)
  if (any(need)) {
    idx[need] <- match(make.names(tested_ids)[need], grp_feature)
  }

  out <- data.frame(
    feature_id = tested_ids,
    effect_size = grp_effect[idx],
    p_value = grp_p[idx],
    maaslin2_qval_native = grp_qnative[idx],
    stringsAsFactors = FALSE
  )
  # Recompute BH over the subject_group p-values across taxa -> identical FDR
  # scope to LinDA's per-task padj. NA p-values (features MaAsLin2 could not fit)
  # are left as NA, exactly like LinDA omitting a feature.
  out$q_value <- NA_real_
  ok <- is.finite(out$p_value)
  if (any(ok)) out$q_value[ok] <- stats::p.adjust(out$p_value[ok], method = "BH")
  out
}

# Longitudinal MaAsLin2: ONE inference per taxon for the subject_group effect,
# controlling for bin_c (time) with a per-participant random intercept. Mirrors
# run_longitudinal_linda's contract and output columns exactly.
run_longitudinal_maaslin2 <- function(tables, dataset_name, group_mode, diet_col, task_name,
                                      out_dir, cores = 1, debug = TRUE,
                                      normalization = "NONE", transform = "LOG") {
  meta <- tables$metadata
  feat <- tables$features
  feature_map <- tables$feature_map

  meta$subject_group <- droplevels(factor(meta$subject_group))
  meta$participant <- droplevels(factor(meta$participant))

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

  # Drop features that are all-zero across the whole table. MaAsLin2's LOG
  # transform replaces zeros with half the per-feature minimum non-zero value,
  # which is Inf for an all-zero column; dropping them also matches the exact set
  # of taxa LinDA tests, keeping the two engines' feature sets identical.
  feat_mat0 <- as.matrix(feat)
  nz <- colSums(feat_mat0, na.rm = TRUE) > 0
  n_dropped <- sum(!nz)
  if (sum(nz) < 2) {
    stop("Fewer than 2 non-empty features after dropping all-zero columns (", sum(nz), " kept).")
  }
  if (debug && n_dropped > 0) {
    message(sprintf("    dropped %d all-zero features; %d features tested", n_dropped, sum(nz)))
  }

  feature_df <- as.data.frame(feat_mat0[, nz, drop = FALSE], check.names = FALSE)
  if (!identical(rownames(feature_df), rownames(meta))) {
    stop("Feature matrix rows are not aligned to metadata rows.")
  }

  run_dir <- file.path(out_dir, "maaslin2_runs", paste(dataset_name, task_name, sep = "_"))
  fitres <- run_maaslin2_group_fit(
    feature_df, meta, run_dir, cores = cores,
    normalization = normalization, transform = transform
  )
  tbl <- extract_maaslin2_group_table(fitres$res, colnames(feature_df))
  tbl$taxon <- map_feature_to_taxon(tbl$feature_id, feature_map)

  formula_text <- sprintf(
    "maaslin2 LM | fixed: subject_group + bin_c | random: participant | norm=%s; transform=%s | q=BH(subject_group across taxa)",
    normalization, transform
  )

  data.frame(
    dataset = dataset_name,
    group_mode = group_mode,
    diet_col = diet_col,
    feature_id = tbl$feature_id,
    taxon = tbl$taxon,
    effect_size = tbl$effect_size,
    p_value = tbl$p_value,
    q_value = tbl$q_value,
    df = NA_real_,
    n_samples = n_samples,
    n_participants = n_participants,
    group_counts = counts_text,
    formula = formula_text,
    maaslin2_qval_native = tbl$maaslin2_qval_native,
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# Driver: external subject-group file path (same structure as the LinDA script,
# with the inference call swapped and MaAsLin2-specific output filenames).
# ---------------------------------------------------------------------------
run_external_subject_groups_maaslin2 <- function(args, base_tensor, imputed_tensor, out_dir, cores,
                                                 normalization, transform) {
  subject_groups_path <- get_arg_any(args, c("subject-groups", "subject_groups"), file.path(out_dir, "subject_groups_long.csv"))
  participant_order_path <- get_arg_any(args, c("participant-order", "participant_order"), "participant_order.csv")
  min_group_n <- as.integer(get_arg_any(args, c("min-group-n", "min_group_n"), "2"))
  debug_fit <- as_bool(get_arg_any(args, c("debug-maaslin", "debug_maaslin"), "true"), default = TRUE)

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

  participant_order <- read_or_build_participant_order(participant_order_path, base_tensor, subject_groups, out_dir)
  message("participant_idx coverage: ", paste(range(participant_order$participant_idx), collapse = "-"),
          "; n=", nrow(participant_order))

  idx_groups <- merge(subject_groups, participant_order, by = "participant_id", all.x = TRUE, sort = FALSE)
  write.csv(
    idx_groups[order(idx_groups$participant_idx, idx_groups$group_mode, idx_groups$diet_col), ],
    file.path(out_dir, "participant_idx_subject_groups_long.csv"),
    row.names = FALSE
  )

  tasks <- unique(subject_groups[, c("group_mode", "diet_col"), drop = FALSE])
  tasks <- tasks[tasks$group_mode %in% c("abx_group", "diet_group"), , drop = FALSE]
  datasets <- list(base_raw = base_tensor, p2d2pca_imputed = imputed_tensor)
  skipped <- list()
  maaslin_results <- list()

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

  build_tables <- function(p, dataset_name) {
    tables <- tensor_to_method_tables_by_idx(
      tensor = datasets[[dataset_name]],
      subject_meta = p$subject_meta,
      extra_cols = c("participant_id", "exposure_type", "subject_group", "subject_group_binary")
    )
    tables$metadata$subject_group <- factor(tables$metadata$subject_group,
                                            levels = levels(p$subject_meta$subject_group))
    tables$metadata$exposure_type <- as.character(tables$metadata$subject_group)
    tables$metadata$participant <- factor(tables$metadata$participant)
    tables
  }

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  for (p in prepared) {
    for (dataset_name in names(datasets)) {
      tag <- paste(dataset_name, p$label, sep = "_")
      tables <- try(build_tables(p, dataset_name), silent = TRUE)
      if (inherits(tables, "try-error")) {
        reason <- conditionMessage(attr(tables, "condition"))
        warning("Skipping ", tag, ": ", reason)
        skipped <- append_skip(skipped, p$group_mode, p$diet_col, dataset_name, reason,
                               skip_type = "input_error",
                               n_valid = nrow(p$subject_meta), counts = p$counts_text)
        next
      }
      write.csv(tables$feature_map, file.path(out_dir, paste0(tag, "_feature_map.csv")), row.names = FALSE)
      write.csv(tables$metadata, file.path(out_dir, paste0(tag, "_sample_metadata.csv")), row.names = FALSE)

      message("Running longitudinal MaAsLin2 for ", tag)
      res <- try(
        run_longitudinal_maaslin2(
          tables = tables, dataset_name = dataset_name, group_mode = p$group_mode,
          diet_col = p$diet_col, task_name = p$label,
          out_dir = out_dir, cores = cores, debug = debug_fit,
          normalization = normalization, transform = transform
        ),
        silent = TRUE
      )
      if (inherits(res, "try-error")) {
        reason <- conditionMessage(attr(res, "condition"))
        skip_type <- classify_skip_type(reason, default = "maaslin2_fit_error")
        warning("MaAsLin2 failed for ", tag, ": ", reason)
        skipped <- append_skip(skipped, p$group_mode, p$diet_col, dataset_name, reason,
                               skip_type = skip_type,
                               n_valid = nrow(p$subject_meta), counts = p$counts_text)
      } else {
        maaslin_results[[length(maaslin_results) + 1]] <- res
      }
    }
  }

  out_path <- file.path(out_dir, "maaslin2_longitudinal_results.csv")
  out_cols <- c("dataset", "group_mode", "diet_col", "feature_id", "taxon",
                "effect_size", "p_value", "q_value", "df",
                "n_samples", "n_participants", "group_counts", "formula",
                "maaslin2_qval_native")
  if (length(maaslin_results)) {
    res_df <- do.call(rbind, maaslin_results)
    out_cols <- c(out_cols[out_cols %in% names(res_df)], setdiff(names(res_df), out_cols))
    write.csv(res_df[, out_cols], out_path, row.names = FALSE)
    message("Wrote longitudinal MaAsLin2 results (", nrow(res_df), " rows): ", out_path)
  } else {
    empty <- as.data.frame(setNames(rep(list(character()), length(out_cols)), out_cols))
    write.csv(empty, out_path, row.names = FALSE)
    message("No longitudinal MaAsLin2 results were produced; wrote empty table: ", out_path)
  }
  write_skipped_report(skipped, file.path(out_dir, "skipped_subject_groups_maaslin2.csv"))
}

run_legacy_internal_groups_maaslin2 <- function(args, base_tensor, imputed_tensor, meta_path, out_dir, cores,
                                                normalization, transform) {
  message("No external --subject-groups file was supplied/found; using legacy internal grouping.")
  meta <- read_metadata(meta_path)
  diet_cols_arg <- get_arg_any(args, c("diet_cols", "diet-cols"), "ALL_DIETARY")
  subject_group_modes <- split_arg(get_arg_any(args, c("subject_group_modes", "subject-group-modes"), "antibiotic_use"))
  min_group_n <- as.integer(get_arg_any(args, c("min_group_n", "min-group-n"), "4"))
  diet_cols <- resolve_diet_cols(meta, diet_cols_arg)
  participants <- Reduce(intersect, list(base_tensor$participant, imputed_tensor$participant))
  datasets <- list(base_raw = base_tensor, p2d2pca_imputed = imputed_tensor)
  skipped <- list()
  maaslin_results <- list()

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
        run_longitudinal_maaslin2(
          tables = tables, dataset_name = dataset_name, group_mode = make_safe_name(group_mode),
          diet_col = "", task_name = make_safe_name(group_mode),
          out_dir = out_dir, cores = cores, debug = FALSE,
          normalization = normalization, transform = transform
        ),
        silent = TRUE
      )
      if (inherits(res, "try-error")) {
        skipped <- append_skip(skipped, group_mode, NA_character_, dataset_name,
                               conditionMessage(attr(res, "condition")))
      } else {
        maaslin_results[[length(maaslin_results) + 1]] <- res
      }
    }
  }

  out_path <- file.path(out_dir, "maaslin2_longitudinal_results.csv")
  out_cols <- c("dataset", "group_mode", "diet_col", "feature_id", "taxon",
                "effect_size", "p_value", "q_value", "df",
                "n_samples", "n_participants", "group_counts", "formula",
                "maaslin2_qval_native")
  if (length(maaslin_results)) {
    res_df <- do.call(rbind, maaslin_results)
    out_cols <- c(out_cols[out_cols %in% names(res_df)], setdiff(names(res_df), out_cols))
    write.csv(res_df[, out_cols], out_path, row.names = FALSE)
    message("Wrote longitudinal MaAsLin2 results (", nrow(res_df), " rows): ", out_path)
  }

  write_skipped_report(skipped, file.path(out_dir, "skipped_subject_groups_maaslin2.csv"))
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  lib_path <- get_arg_any(args, c("lib_path", "lib-path"), "")
  if (nzchar(lib_path)) .libPaths(c(split_arg(lib_path), .libPaths()))

  base_nc <- get_arg_any(args, c("base_nc", "base-nc"), "diabimmune_p2d2pca_netcdf_export/tensors/tensor_base_raw.nc")
  imputed_nc <- get_arg_any(args, c("imputed_nc", "imputed-nc"), "diabimmune_p2d2pca_netcdf_export/tensors/tensor_p2d2pca_imputed.nc")
  meta_path <- get_arg_any(args, c("meta", "metadata"), "11884_20190508-173103-added-month-abx.txt")
  out_dir <- get_arg_any(args, c("out_dir", "outdir"), "artifacts/subject_group_existing_methods_r")
  cores <- as.integer(get_arg_any(args, c("cores"), "1"))
  normalization <- toupper(get_arg_any(args, c("normalization", "norm"), "NONE"))
  transform <- toupper(get_arg_any(args, c("transform"), "LOG"))

  require_pkg("Maaslin2")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  message("Reading tensor: ", base_nc)
  base_tensor <- read_tensor_nc(base_nc)
  message("Reading tensor: ", imputed_nc)
  imputed_tensor <- read_tensor_nc(imputed_nc)
  if (length(base_tensor$participant) != length(imputed_tensor$participant)) {
    stop("Base and imputed tensors have different participant counts.")
  }

  subject_groups_arg <- get_arg_any(args, c("subject-groups", "subject_groups"), "")
  default_subject_groups <- file.path(out_dir, "subject_groups_long.csv")
  if (nzchar(subject_groups_arg) || file.exists(default_subject_groups)) {
    run_external_subject_groups_maaslin2(args, base_tensor, imputed_tensor, out_dir, cores, normalization, transform)
  } else {
    run_legacy_internal_groups_maaslin2(args, base_tensor, imputed_tensor, meta_path, out_dir, cores, normalization, transform)
  }

  message("Done. Outputs: ", out_dir)
}

# When launched via `Rscript foo.R`, exit with a non-zero status on error (for
# shells / CI). When sourced interactively (e.g. RStudio: source() or the Source
# button), just run main() and let any error surface normally so the session is
# NOT terminated by quit().
if (length(grep("^--file=", commandArgs(FALSE)))) {
  tryCatch(
    main(),
    error = function(e) {
      message("ERROR: ", conditionMessage(e))
      quit(status = 1)
    }
  )
} else {
  main()
}
