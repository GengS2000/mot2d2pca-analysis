da_add_local_libpaths <- function() {
  local_lib <- file.path(getwd(), ".Rlib")
  dir.create(local_lib, showWarnings = FALSE, recursive = TRUE)
  user_lib <- file.path(
    Sys.getenv("LOCALAPPDATA"),
    "R",
    "win-library",
    paste(R.version$major, sub("\\..*$", "", R.version$minor), sep = ".")
  )
  .libPaths(unique(c(normalizePath(local_lib, mustWork = FALSE), user_lib, .libPaths())))
  invisible(.libPaths())
}

da_safe_div <- function(a, b, zero_value = NA_real_) {
  ifelse(b == 0, zero_value, a / b)
}

da_mean_or_na <- function(x) {
  if (length(x) == 0 || all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
}

da_parse_level_from_name <- function(x, prefix_regex = "^group_effect_") {
  val <- sub(prefix_regex, "", basename(x))
  val <- gsub("p", ".", val, fixed = TRUE)
  suppressWarnings(as.numeric(val))
}

da_format_level_label <- function(level_num) {
  out <- ifelse(
    is.na(level_num),
    "unknown",
    format(level_num, trim = TRUE, scientific = FALSE)
  )
  as.character(out)
}

da_as_logical <- function(x) {
  if (is.logical(x)) return(x)
  if (is.numeric(x) || is.integer(x)) return(!is.na(x) & x != 0)
  y <- tolower(trimws(as.character(x)))
  out <- rep(NA, length(y))
  out[y %in% c("true", "t", "1", "yes", "y")] <- TRUE
  out[y %in% c("false", "f", "0", "no", "n")] <- FALSE
  out
}

da_first_col <- function(df, candidates) {
  hit <- intersect(candidates, colnames(df))
  if (length(hit) == 0) NA_character_ else hit[1]
}

da_prepare_truth_df <- function(df_taxon_effect) {
  if (!"taxon" %in% colnames(df_taxon_effect)) {
    stop("Truth table must contain a taxon column.")
  }

  truth_col <- da_first_col(df_taxon_effect, c("is_true_diff", "is_differential", "truth"))
  if (is.na(truth_col)) {
    stop("Truth table must contain is_differential, is_true_diff, or truth.")
  }

  out <- df_taxon_effect
  out$taxon <- as.character(out$taxon)
  out$is_true_diff <- da_as_logical(out[[truth_col]])

  if (any(is.na(out$is_true_diff))) {
    bad <- unique(out$taxon[is.na(out$is_true_diff)])
    stop("Truth table has unrecognized truth values for taxa: ", paste(head(bad, 10), collapse = ", "))
  }

  out <- out[!duplicated(out$taxon), , drop = FALSE]
  out
}

da_infer_pvalue_col <- function(df) {
  da_first_col(df, c("pvalue", "p.value", "pval", "p", "raw.pvalue", "Pr(>|t|)"))
}

da_infer_qvalue_col <- function(df) {
  da_first_col(df, c("qvalue", "qval", "padj", "p.adj", "p.adjust", "adj.pvalue"))
}

da_infer_estimate_col <- function(df) {
  da_first_col(df, c("estimate", "Estimate", "lfc", "log2FoldChange", "beta", "coef"))
}

da_infer_statistic_col <- function(df) {
  da_first_col(df, c("statistic", "stat", "W", "z", "t", "t_value", "t-value"))
}

da_standardize_taxon_results <- function(res_df,
                                         df_taxon_effect,
                                         alpha = 0.05,
                                         method_name = NA_character_,
                                         data_type = NA_character_,
                                         masking_fraction = NA_real_,
                                         group_effect_mean = NA_real_,
                                         sim_seed = NA_integer_,
                                         mask_seed = NA_integer_,
                                         effect_name = NA_character_,
                                         frac_name = NA_character_,
                                         fit_status_default = "ok",
                                         fit_message_default = "",
                                         require_passed_ss = FALSE) {
  truth_df <- da_prepare_truth_df(df_taxon_effect)
  res_df <- as.data.frame(res_df, stringsAsFactors = FALSE, check.names = FALSE)

  if (nrow(res_df) == 0) {
    res_df <- data.frame(taxon = character(), stringsAsFactors = FALSE)
  }
  if (!"taxon" %in% colnames(res_df)) {
    stop("Result table must contain a taxon column.")
  }
  res_df$taxon <- as.character(res_df$taxon)

  p_col <- da_infer_pvalue_col(res_df)
  q_col <- da_infer_qvalue_col(res_df)
  estimate_col <- da_infer_estimate_col(res_df)
  statistic_col <- da_infer_statistic_col(res_df)

  if (!is.na(p_col) && is.na(q_col)) {
    valid <- !is.na(res_df[[p_col]])
    res_df$qvalue <- NA_real_
    if (any(valid)) {
      res_df$qvalue[valid] <- p.adjust(as.numeric(res_df[[p_col]][valid]), method = "BH")
    }
    q_col <- "qvalue"
  }

  merged <- merge(truth_df, res_df, by = "taxon", all.x = TRUE, sort = FALSE)
  matched <- merged$taxon %in% res_df$taxon

  if (!is.na(p_col) && p_col %in% colnames(merged)) {
    pvalue <- as.numeric(merged[[p_col]])
  } else {
    pvalue <- rep(NA_real_, nrow(merged))
  }

  if (!is.na(q_col) && q_col %in% colnames(merged)) {
    qvalue <- as.numeric(merged[[q_col]])
  } else {
    qvalue <- rep(NA_real_, nrow(merged))
  }

  if (!is.na(estimate_col) && estimate_col %in% colnames(merged)) {
    estimate <- as.numeric(merged[[estimate_col]])
  } else {
    estimate <- rep(NA_real_, nrow(merged))
  }

  if (!is.na(statistic_col) && statistic_col %in% colnames(merged)) {
    statistic <- as.numeric(merged[[statistic_col]])
  } else {
    statistic <- rep(NA_real_, nrow(merged))
  }

  if ("significant" %in% colnames(merged)) {
    significant <- da_as_logical(merged$significant)
    significant[is.na(significant)] <- FALSE
  } else if ("reject" %in% colnames(merged)) {
    significant <- da_as_logical(merged$reject)
    significant[is.na(significant)] <- FALSE
  } else {
    significant <- !is.na(qvalue) & qvalue <= alpha
  }

  ss_col <- da_first_col(merged, c("passed_ss", "passed_ss_x", "passed_ss_y", "passed_sensitivity"))
  if (require_passed_ss && !is.na(ss_col)) {
    passed_ss <- da_as_logical(merged[[ss_col]])
    passed_ss[is.na(passed_ss)] <- FALSE
    significant <- significant & passed_ss
  }

  fit_status <- if ("fit_status" %in% colnames(merged)) {
    as.character(merged$fit_status)
  } else {
    rep(fit_status_default, nrow(merged))
  }
  fit_status[is.na(fit_status) | fit_status == ""] <- ifelse(matched[is.na(fit_status) | fit_status == ""], fit_status_default, "no_result")
  fit_status[!matched] <- "no_result"

  fit_message <- if ("fit_message" %in% colnames(merged)) {
    as.character(merged$fit_message)
  } else {
    rep(fit_message_default, nrow(merged))
  }
  fit_message[is.na(fit_message)] <- ""
  fit_message[!matched & fit_message == ""] <- "taxon was not returned by method"

  std <- data.frame(
    taxon = merged$taxon,
    method_name = method_name,
    data_type = data_type,
    masking_fraction = masking_fraction,
    group_effect_mean = group_effect_mean,
    sim_seed = sim_seed,
    mask_seed = mask_seed,
    estimate = estimate,
    statistic = statistic,
    pvalue = pvalue,
    qvalue = qvalue,
    significant = significant,
    is_true_diff = as.logical(merged$is_true_diff),
    fit_status = fit_status,
    fit_message = fit_message,
    stringsAsFactors = FALSE
  )

  std$effect_name <- effect_name
  std$frac <- frac_name

  extra_cols <- setdiff(colnames(merged), colnames(std))
  out <- cbind(std, merged[, extra_cols, drop = FALSE])

  if (!"padj" %in% colnames(out)) out$padj <- out$qvalue
  if (!"reject" %in% colnames(out)) out$reject <- out$significant

  out
}

da_compute_metrics <- function(taxon_results) {
  if (!all(c("significant", "is_true_diff", "fit_status") %in% colnames(taxon_results))) {
    stop("Standardized taxon results must contain significant, is_true_diff, and fit_status.")
  }

  group_cols <- intersect(
    c("method_name", "data_type", "masking_fraction", "group_effect_mean", "sim_seed", "mask_seed"),
    colnames(taxon_results)
  )
  if (length(group_cols) == 0) group_cols <- NULL

  split_key <- if (is.null(group_cols)) {
    factor(rep("all", nrow(taxon_results)))
  } else {
    key_df <- taxon_results[, group_cols, drop = FALSE]
    for (nm in names(key_df)) {
      if (is.numeric(key_df[[nm]]) || is.integer(key_df[[nm]])) {
        key_df[[nm]][is.na(key_df[[nm]])] <- Inf
      } else {
        key_df[[nm]] <- as.character(key_df[[nm]])
        key_df[[nm]][is.na(key_df[[nm]])] <- "<NA>"
      }
    }
    interaction(key_df, drop = TRUE, lex.order = TRUE)
  }

  pieces <- split(taxon_results, split_key)
  out <- lapply(pieces, function(df) {
    called <- as.logical(df$significant)
    truth <- as.logical(df$is_true_diff)
    called[is.na(called)] <- FALSE
    truth[is.na(truth)] <- FALSE

    TP <- sum(called & truth, na.rm = TRUE)
    FP <- sum(called & !truth, na.rm = TRUE)
    FN <- sum(!called & truth, na.rm = TRUE)
    TN <- sum(!called & !truth, na.rm = TRUE)

    precision <- ifelse(TP + FP > 0, TP / (TP + FP), NA_real_)
    recall <- ifelse(TP + FN > 0, TP / (TP + FN), NA_real_)
    F1 <- ifelse(!is.na(precision) && !is.na(recall) && precision + recall > 0,
      2 * precision * recall / (precision + recall),
      0
    )
    FDR <- ifelse(TP + FP > 0, FP / (TP + FP), 0)
    specificity <- ifelse(TN + FP > 0, TN / (TN + FP), NA_real_)
    accuracy <- ifelse(TP + FP + TN + FN > 0, (TP + TN) / (TP + FP + TN + FN), NA_real_)

    ids <- if (is.null(group_cols)) {
      data.frame(stringsAsFactors = FALSE)
    } else {
      df[1, group_cols, drop = FALSE]
    }

    cbind(
      ids,
      data.frame(
        TP = TP,
        FP = FP,
        FN = FN,
        TN = TN,
        precision = precision,
        recall = recall,
        F1 = F1,
        specificity = specificity,
        accuracy = accuracy,
        FDR = FDR,
        n_discoveries = TP + FP,
        n_called_diff = TP + FP,
        n_true_diff = sum(truth, na.rm = TRUE),
        n_truth_diff = sum(truth, na.rm = TRUE),
        n_taxa_total = length(truth),
        n_successfully_fitted_taxa = sum(df$fit_status %in% c("ok", "warning"), na.rm = TRUE),
        n_failed_taxa = sum(!(df$fit_status %in% c("ok", "warning")), na.rm = TRUE),
        stringsAsFactors = FALSE
      )
    )
  })

  do.call(rbind, out)
}

da_evaluate_against_truth <- function(res_df,
                                      df_taxon_effect,
                                      alpha = 0.05,
                                      method_name = NA_character_,
                                      data_type = NA_character_,
                                      masking_fraction = NA_real_,
                                      group_effect_mean = NA_real_,
                                      sim_seed = NA_integer_,
                                      mask_seed = NA_integer_,
                                      effect_name = NA_character_,
                                      frac_name = NA_character_,
                                      require_passed_ss = FALSE) {
  res_eval <- da_standardize_taxon_results(
    res_df = res_df,
    df_taxon_effect = df_taxon_effect,
    alpha = alpha,
    method_name = method_name,
    data_type = data_type,
    masking_fraction = masking_fraction,
    group_effect_mean = group_effect_mean,
    sim_seed = sim_seed,
    mask_seed = mask_seed,
    effect_name = effect_name,
    frac_name = frac_name,
    require_passed_ss = require_passed_ss
  )
  metrics <- da_compute_metrics(res_eval)
  list(res_eval = res_eval, metrics = metrics)
}

da_approx_decreasing <- function(x, tolerance = 0.05, max_rebound = 0.15) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  if (length(x) <= 1) return(TRUE)
  diffs <- diff(x)
  all(diffs <= tolerance) && sum(pmax(diffs, 0), na.rm = TRUE) <= max_rebound
}

da_pass_fail_one_method <- function(metrics_df,
                                    method_label = NA_character_,
                                    tolerance = 0.05,
                                    max_rebound = 0.15) {
  needed <- c("data_type", "masking_fraction", "group_effect_mean", "TP", "recall", "F1", "FDR")
  missing <- setdiff(needed, colnames(metrics_df))
  if (length(missing) > 0) {
    stop("Metrics table is missing columns: ", paste(missing, collapse = ", "))
  }

  effects <- sort(unique(metrics_df$group_effect_mean))
  out <- lapply(effects, function(effect) {
    df <- metrics_df[metrics_df$group_effect_mean == effect, , drop = FALSE]
    z <- df[df$data_type == "zeroed", , drop = FALSE]
    im <- df[df$data_type == "imputed", , drop = FALSE]

    z <- z[order(z$masking_fraction), , drop = FALSE]
    im <- im[order(im$masking_fraction), , drop = FALSE]

    z06 <- z[abs(z$masking_fraction - 0.6) < 1e-8, , drop = FALSE]
    common_fracs <- intersect(z$masking_fraction, im$masking_fraction)
    zc <- z[match(common_fracs, z$masking_fraction), , drop = FALSE]
    ic <- im[match(common_fracs, im$masking_fraction), , drop = FALSE]

    cond1 <- nrow(z06) > 0 && any(z06$TP > 0, na.rm = TRUE)
    cond2 <- length(common_fracs) > 0 && sum((ic$TP > zc$TP) | (ic$recall > zc$recall), na.rm = TRUE) >= 7
    cond3 <- da_mean_or_na(im$F1) > da_mean_or_na(z$F1)
    cond4 <- da_mean_or_na(im$FDR) <= da_mean_or_na(z$FDR) + 0.10
    cond5 <- da_approx_decreasing(z$recall, tolerance, max_rebound) &&
      da_approx_decreasing(im$recall, tolerance, max_rebound)

    data.frame(
      method_name = method_label,
      group_effect_mean = effect,
      cond_zeroed_tp_at_0p6 = cond1,
      cond_imputed_better_7_of_9 = cond2,
      cond_mean_f1_better = cond3,
      cond_mean_fdr_not_inflated = cond4,
      cond_recall_approx_decreasing = cond5,
      passed = isTRUE(cond1 && cond2 && cond3 && cond4 && cond5),
      stringsAsFactors = FALSE
    )
  })

  do.call(rbind, out)
}
