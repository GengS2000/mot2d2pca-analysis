# Shared glmmPQL implementation for the ECAM and DIABIMMUNE NetCDF runners.
# The caller is responsible for building the same sample-by-feature tables used
# by the existing LinDA/MaAsLin2 workflows.

glmmpql_use_tensor_log_values <- function(tables, tensor) {
  meta <- tables$metadata
  feature_map <- tables$feature_map
  required <- c("participant_idx", "bin")
  missing <- setdiff(required, names(meta))
  if (length(missing)) stop("glmmPQL table metadata missing: ", paste(missing, collapse = ", "))

  participant_index <- suppressWarnings(as.integer(meta$participant_idx))
  bin_value <- suppressWarnings(as.numeric(meta$bin))
  bin_index <- match(round(bin_value, 10), round(as.numeric(tensor$bin), 10))
  if (anyNA(participant_index) || anyNA(bin_index)) {
    stop("Could not map glmmPQL sample metadata back to tensor participant/bin indices.")
  }
  if (any(participant_index < 1L | participant_index > dim(tensor$values)[[1L]])) {
    stop("glmmPQL participant index is outside the tensor range.")
  }

  n_samples <- nrow(meta)
  n_features <- dim(tensor$values)[[2L]]
  log_values <- matrix(NA_real_, nrow = n_samples, ncol = n_features)
  for (i in seq_len(n_samples)) {
    log_values[i, ] <- tensor$values[participant_index[[i]], , bin_index[[i]]]
  }
  feature_ids <- if (!is.null(feature_map) && "feature_id" %in% names(feature_map)) {
    as.character(feature_map$feature_id)
  } else {
    colnames(tables$features)
  }
  if (length(feature_ids) != n_features) stop("glmmPQL feature map does not match tensor taxa.")
  colnames(log_values) <- feature_ids
  rownames(log_values) <- rownames(meta)
  tables$features <- as.data.frame(log_values, check.names = FALSE)
  tables
}

run_longitudinal_glmmpql <- function(
    tables,
    dataset_name,
    group_mode,
    diet_col,
    task_name,
    formula_text = "y ~ subject_group + bin_c; random = ~ 1 | participant; family = gaussian(identity)",
    niter = 10L,
    debug = TRUE) {
  if (!requireNamespace("MASS", quietly = TRUE)) stop("Package 'MASS' is required for glmmPQL.")
  if (!requireNamespace("nlme", quietly = TRUE)) stop("Package 'nlme' is required for glmmPQL.")

  meta <- tables$metadata
  features <- as.matrix(tables$features)
  feature_map <- tables$feature_map
  storage.mode(features) <- "double"

  required_meta <- c("subject_group", "bin_c", "participant")
  missing_meta <- setdiff(required_meta, names(meta))
  if (length(missing_meta)) {
    stop("glmmPQL metadata missing columns: ", paste(missing_meta, collapse = ", "))
  }
  if (nrow(features) != nrow(meta)) stop("glmmPQL feature and metadata row counts differ.")
  if (!is.null(rownames(features)) && !is.null(rownames(meta)) &&
      !identical(rownames(features), rownames(meta))) {
    stop("glmmPQL feature rows are not aligned to metadata rows.")
  }
  if (is.null(colnames(features)) || any(!nzchar(colnames(features)))) {
    stop("glmmPQL feature matrix must have non-empty feature IDs.")
  }

  meta$subject_group <- droplevels(factor(meta$subject_group))
  meta$participant <- droplevels(factor(meta$participant))
  meta$bin_c <- suppressWarnings(as.numeric(meta$bin_c))
  if (nlevels(meta$subject_group) != 2L) {
    stop("glmmPQL requires exactly two subject_group levels; found ",
         paste(levels(meta$subject_group), collapse = ", "))
  }

  reference_level <- levels(meta$subject_group)[[1L]]
  contrast_level <- levels(meta$subject_group)[[2L]]
  coefficient_name <- paste0("subject_group", contrast_level)
  first_group <- tapply(as.character(meta$subject_group), meta$participant, function(x) x[[1L]])
  group_counts <- table(first_group)
  group_counts_text <- paste(names(group_counts), as.integer(group_counts), sep = "=", collapse = ", ")

  if (debug) {
    message(sprintf(
      "  [glmmPQL %s | %s] samples=%d, participants=%d, features=%d, groups: %s",
      dataset_name, task_name, nrow(meta), nlevels(meta$participant), ncol(features), group_counts_text
    ))
  }

  estimate <- std_error <- p_value <- df_value <- rep(NA_real_, ncol(features))
  fit_status <- rep("ok", ncol(features))
  for (j in seq_len(ncol(features))) {
    dat <- data.frame(
      y = features[, j],
      subject_group = meta$subject_group,
      bin_c = meta$bin_c,
      participant = meta$participant
    )
    dat <- dat[is.finite(dat$y) & is.finite(dat$bin_c), , drop = FALSE]
    dat$subject_group <- droplevels(dat$subject_group)
    dat$participant <- droplevels(dat$participant)

    if (nrow(dat) < 10L) {
      fit_status[[j]] <- "skipped_too_few_rows"
      next
    }
    if (!is.finite(stats::sd(dat$y)) || stats::sd(dat$y) < 1e-12) {
      fit_status[[j]] <- "skipped_no_variation"
      next
    }
    if (nlevels(dat$subject_group) < 2L) {
      fit_status[[j]] <- "skipped_one_group"
      next
    }

    fit <- tryCatch(
      suppressWarnings(suppressMessages(
        MASS::glmmPQL(
          y ~ subject_group + bin_c,
          random = ~ 1 | participant,
          family = gaussian(link = "identity"),
          data = dat,
          niter = as.integer(niter),
          verbose = FALSE
        )
      )),
      error = function(e) e
    )
    if (inherits(fit, "error")) {
      fit_status[[j]] <- paste0("fit_error: ", conditionMessage(fit))
      next
    }

    coefficient_table <- tryCatch(summary(fit)$tTable, error = function(e) NULL)
    if (is.null(coefficient_table) || !coefficient_name %in% rownames(coefficient_table)) {
      candidates <- grep("^subject_group", rownames(coefficient_table), value = TRUE)
      if (length(candidates) == 1L) {
        coefficient_name_j <- candidates[[1L]]
      } else {
        fit_status[[j]] <- "no_subject_group_coefficient"
        next
      }
    } else {
      coefficient_name_j <- coefficient_name
    }
    estimate[[j]] <- coefficient_table[coefficient_name_j, "Value"]
    std_error[[j]] <- coefficient_table[coefficient_name_j, "Std.Error"]
    p_value[[j]] <- coefficient_table[coefficient_name_j, "p-value"]
    df_value[[j]] <- coefficient_table[coefficient_name_j, "DF"]
  }

  q_value <- rep(NA_real_, length(p_value))
  finite_p <- is.finite(p_value)
  if (any(finite_p)) q_value[finite_p] <- stats::p.adjust(p_value[finite_p], method = "BH")

  feature_ids <- colnames(features)
  annotation <- data.frame(feature_id = feature_ids, stringsAsFactors = FALSE)
  if (!is.null(feature_map) && nrow(feature_map) && "feature_id" %in% names(feature_map)) {
    map_index <- match(feature_ids, as.character(feature_map$feature_id))
    map_columns <- setdiff(names(feature_map), "feature_id")
    if (length(map_columns)) {
      annotation <- cbind(
        annotation,
        feature_map[map_index, map_columns, drop = FALSE],
        stringsAsFactors = FALSE
      )
      rownames(annotation) <- NULL
    }
  }
  if (!"taxon" %in% names(annotation)) annotation$taxon <- feature_ids

  identity_columns <- c("dataset", "group_mode", "diet_col", "feature_id",
                        "reference_level", "contrast_level")
  identity <- data.frame(
    dataset = dataset_name,
    group_mode = group_mode,
    diet_col = diet_col,
    feature_id = feature_ids,
    reference_level = reference_level,
    contrast_level = contrast_level,
    stringsAsFactors = FALSE
  )
  annotation_columns <- setdiff(names(annotation), "feature_id")
  statistics <- data.frame(
    effect_size = estimate,
    std_error = std_error,
    p_value = p_value,
    q_value = q_value,
    df = df_value,
    fit_status = fit_status,
    da_method = "glmmPQL",
    n_samples = nrow(meta),
    n_participants = nlevels(meta$participant),
    group_counts = group_counts_text,
    formula = formula_text,
    glmmPQL_niter = as.integer(niter),
    stringsAsFactors = FALSE
  )
  out <- cbind(identity, annotation[, annotation_columns, drop = FALSE], statistics)
  out[, c(identity_columns, setdiff(names(out), identity_columns)), drop = FALSE]
}
