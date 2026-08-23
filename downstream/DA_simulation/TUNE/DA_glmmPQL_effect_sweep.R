##############################
## 0. Load dependencies
##############################
required_pkgs <- c(
  "ncdf4", "MASS", "nlme", "dplyr", "tidyr",
  "purrr", "stringr", "readr", "ggplot2", "tibble"
)

to_install <- required_pkgs[!required_pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) {
  install.packages(to_install)
}

library(ncdf4)
library(MASS)
library(nlme)
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)
library(readr)
library(ggplot2)
library(tibble)

common_script <- file.path(getwd(), "DA_effect_sweep_common.R")
if (file.exists(common_script)) {
  source(common_script)
}

sweep_root_dir <- normalizePath(file.path(getwd(), "simulation3_effect_tune"), mustWork = FALSE)
results_dir_name <- "results_glmmPQL_compare"
alpha <- 0.05

##############################
## 1. Parameter settings
##############################
# Root directory of the effect sweep
# Name of the output directory inside each group_effect_* directory
results_dir_name <- "results_glmmPQL_compare"

# Significance threshold
alpha <- 0.05

##############################
## 2. Utility functions
##############################

safe_div <- function(a, b) {
  ifelse(b == 0, NA_real_, a / b)
}

parse_level_from_name <- function(x, prefix_regex) {
  val <- sub(prefix_regex, "", x)
  val <- gsub("p", ".", val, fixed = TRUE)
  suppressWarnings(as.numeric(val))
}

format_level_label <- function(level_num) {
  if (is.na(level_num)) return("unknown")
  format(level_num, trim = TRUE, scientific = FALSE)
}

find_truth_dir <- function(effect_dir, sweep_root_dir) {
  cand1 <- file.path(effect_dir, "truth")
  cand2 <- file.path(sweep_root_dir, "truth")

  if (dir.exists(cand1)) return(cand1)
  if (dir.exists(cand2)) return(cand2)

  stop(
    "truth directory not found. Checked:\n",
    "1) ", cand1, "\n",
    "2) ", cand2
  )
}

decode_nc_vals <- function(x) {
  if (is.null(x)) return(NULL)

  if (is.character(x)) return(as.vector(x))
  if (is.factor(x)) return(as.character(x))
  if (is.numeric(x) || is.integer(x)) return(as.vector(x))

  if (is.matrix(x)) {
    if (is.character(x)) {
      out1 <- apply(x, 2, paste0, collapse = "")
      out1 <- trimws(out1)
      if (length(out1) > 0) return(out1)

      out2 <- apply(x, 1, paste0, collapse = "")
      out2 <- trimws(out2)
      return(out2)
    } else {
      return(as.vector(x))
    }
  }

  as.vector(x)
}

get_nc_axis_values <- function(nc, axis_name) {
  if (axis_name %in% names(nc$dim)) {
    vals <- nc$dim[[axis_name]]$vals
    vals <- decode_nc_vals(vals)
    if (!is.null(vals)) return(vals)
  }

  if (axis_name %in% names(nc$var)) {
    vals <- ncvar_get(nc, axis_name)
    vals <- decode_nc_vals(vals)
    if (!is.null(vals)) return(vals)
  }

  stop("cannot read axis from the .nc: ", axis_name)
}

read_nc_tensor_standard <- function(nc_path) {
  nc <- nc_open(nc_path)
  on.exit(nc_close(nc), add = TRUE)

  coord_names <- c("participant", "taxon", "bin")
  var_names <- names(nc$var)

  preferred <- intersect(var_names, c(
    "y_zeroed", "y_imputed",
    "zero_ctf_imp_effect", "ctf_imp_effect", "da_2d2_imp_effect"
  ))

  if (length(preferred) == 1) {
    main_var <- preferred
  } else {
    non_coord_vars <- setdiff(var_names, coord_names)
    three_d_vars <- non_coord_vars[sapply(non_coord_vars, function(v) length(nc$var[[v]]$dim) == 3)]
    if (length(three_d_vars) != 1) {
      stop(
        "cannot uniquely identify the three-dimensional main variable: ", nc_path,
        "\nvisible variables: ", paste(var_names, collapse = ", ")
      )
    }
    main_var <- three_d_vars
  }

  arr <- ncvar_get(nc, main_var)
  var_dim_names <- sapply(nc$var[[main_var]]$dim, function(x) x$name)

  participants <- as.character(get_nc_axis_values(nc, "participant"))
  taxa <- as.character(get_nc_axis_values(nc, "taxon"))
  bins <- as.character(get_nc_axis_values(nc, "bin"))

  target_order <- c("participant", "taxon", "bin")
  perm <- match(target_order, var_dim_names)

  if (any(is.na(perm))) {
    stop(
      "participant/taxon/bin missing from the main variable dimensions: ", nc_path,
      "\ncurrent dimension order: ", paste(var_dim_names, collapse = " -> ")
    )
  }

  arr_std <- aperm(arr, perm = perm)

  if (dim(arr_std)[1] != length(participants)) {
    stop("participant dimension length mismatch: ", nc_path)
  }
  if (dim(arr_std)[2] != length(taxa)) {
    stop("taxon dimension length mismatch: ", nc_path)
  }
  if (dim(arr_std)[3] != length(bins)) {
    stop("bin dimension length mismatch: ", nc_path)
  }

  dimnames(arr_std) <- list(
    participant = participants,
    taxon = taxa,
    bin = bins
  )

  list(
    arr = arr_std,
    participants = participants,
    taxa = taxa,
    bins = bins,
    main_var = main_var,
    original_dim_order = var_dim_names
  )
}

infer_method_name <- function(file_path, main_var = NULL) {
  nm <- basename(file_path)

  if (str_detect(nm, regex("zero", ignore_case = TRUE))) return("zeroed")
  if (str_detect(nm, regex("imput", ignore_case = TRUE))) return("imputed")

  if (!is.null(main_var)) {
    if (str_detect(main_var, regex("zero", ignore_case = TRUE))) return("zeroed")
    if (str_detect(main_var, regex("imput", ignore_case = TRUE))) return("imputed")
  }

  return("unknown")
}

# Reshape one taxon's participant x bin slice into the long table glmmPQL needs
build_taxon_long_table <- function(arr_ptb, df_pseudo_group, taxon_name) {
  participants <- dimnames(arr_ptb)$participant
  bins <- dimnames(arr_ptb)$bin
  taxa <- dimnames(arr_ptb)$taxon

  if (!all(c("participant", "X") %in% colnames(df_pseudo_group))) {
    stop("effect_pseudo_group.csv must contain the participant and X columns")
  }

  df_pseudo_group <- df_pseudo_group %>%
    mutate(
      participant = as.character(participant),
      X = as.numeric(X)
    )

  missing_participants <- setdiff(participants, df_pseudo_group$participant)
  if (length(missing_participants) > 0) {
    stop(
      "the following participants are missing from effect_pseudo_group.csv: ",
      paste(missing_participants, collapse = ", ")
    )
  }

  idx_taxon <- match(taxon_name, taxa)
  if (is.na(idx_taxon)) {
    stop("Taxon not found in tensor: ", taxon_name)
  }

  y_vec <- as.numeric(arr_ptb[, idx_taxon, ])

  df_taxon <- expand.grid(
    participant = participants,
    bin = bins,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  ) %>%
    arrange(participant, bin) %>%
    mutate(
      y = y_vec,
      taxon = taxon_name,
      sample = paste(participant, bin, sep = "__")
    )

  idx <- match(df_taxon$participant, df_pseudo_group$participant)
  df_taxon$X <- df_pseudo_group$X[idx]

  df_taxon %>%
    mutate(
      participant = factor(participant),
      bin = factor(bin, levels = bins),
      X = as.numeric(X)
    ) %>%
    dplyr::select(y, participant, bin, X, taxon, sample)
}

extract_glmmpql_x_row <- function(fit) {
  sm <- summary(fit)
  ttab <- tryCatch(sm$tTable, error = function(e) NULL)
  if (is.null(ttab)) return(NULL)

  coef_df <- as.data.frame(ttab)
  coef_df$term <- rownames(coef_df)
  coef_df[coef_df$term == "X", , drop = FALSE]
}

fit_glmmpql_one_taxon <- function(df_taxon,
                                  alpha = 0.05,
                                  niter = 10,
                                  verbose = FALSE) {
  taxon_name <- unique(df_taxon$taxon)[1]
  out_na <- tibble(
    taxon = taxon_name,
    estimate = NA_real_,
    std_error = NA_real_,
    statistic = NA_real_,
    pvalue = NA_real_,
    padj = NA_real_,
    reject = NA,
    fit_ok = FALSE,
    fit_status = NA_character_
  )

  df_model <- df_taxon %>%
    filter(is.finite(y), !is.na(X), !is.na(participant), !is.na(bin)) %>%
    droplevels()

  if (nrow(df_model) < 4) {
    out_na$fit_status <- "insufficient_non_missing_rows"
    return(out_na)
  }
  if (length(unique(df_model$y)) < 2) {
    out_na$fit_status <- "constant_response"
    return(out_na)
  }
  if (length(unique(df_model$X)) < 2) {
    out_na$fit_status <- "single_X_level"
    return(out_na)
  }
  if (nlevels(df_model$participant) < 2) {
    out_na$fit_status <- "single_participant"
    return(out_na)
  }

  fit_try <- tryCatch(
    {
      fit <- suppressWarnings(
        glmmPQL(
          fixed = y ~ X + bin,
          random = ~ 1 | participant,
          family = gaussian(link = "identity"),
          data = df_model,
          correlation = NULL,
          weights = NULL,
          control = nlme::lmeControl(returnObject = TRUE),
          niter = niter,
          verbose = verbose
        )
      )
      list(ok = TRUE, fit = fit, err = NULL)
    },
    error = function(e) list(ok = FALSE, fit = NULL, err = e)
  )

  if (!isTRUE(fit_try$ok) || is.null(fit_try$fit)) {
    out_na$fit_status <- paste0("fit_error: ", conditionMessage(fit_try$err))
    return(out_na)
  }

  x_row <- extract_glmmpql_x_row(fit_try$fit)
  if (is.null(x_row) || nrow(x_row) != 1) {
    out_na$fit_status <- "X_term_not_found"
    return(out_na)
  }

  value_col <- intersect(c("Value", "Estimate", "estimate"), colnames(x_row))
  se_col <- intersect(c("Std.Error", "Std. Error", "SE"), colnames(x_row))
  stat_col <- intersect(c("t-value", "t value", "t_value", "statistic", "Statistic"), colnames(x_row))
  p_col <- intersect(c("p-value", "p value", "p_value", "Pr(>|t|)", "p"), colnames(x_row))

  if (length(value_col) < 1 || length(se_col) < 1 || length(stat_col) < 1 || length(p_col) < 1) {
    out_na$fit_status <- paste0("unexpected_tTable_columns: ", paste(colnames(x_row), collapse = ", "))
    return(out_na)
  }

  tibble(
    taxon = taxon_name,
    estimate = as.numeric(x_row[[value_col[1]]]),
    std_error = as.numeric(x_row[[se_col[1]]]),
    statistic = as.numeric(x_row[[stat_col[1]]]),
    pvalue = as.numeric(x_row[[p_col[1]]]),
    padj = NA_real_,
    reject = NA,
    fit_ok = TRUE,
    fit_status = "ok"
  )
}

fit_glmmpql_all_taxa <- function(arr_ptb,
                                 df_pseudo_group,
                                 alpha = 0.05,
                                 niter = 10,
                                 verbose = FALSE) {
  taxa <- dimnames(arr_ptb)$taxon

  res_df <- map_dfr(taxa, function(taxon_name) {
    df_taxon <- build_taxon_long_table(
      arr_ptb = arr_ptb,
      df_pseudo_group = df_pseudo_group,
      taxon_name = taxon_name
    )
    fit_glmmpql_one_taxon(
      df_taxon = df_taxon,
      alpha = alpha,
      niter = niter,
      verbose = verbose
    )
  })

  if (nrow(res_df) == 0) {
    return(
      tibble(
        taxon = character(),
        estimate = numeric(),
        std_error = numeric(),
        statistic = numeric(),
        pvalue = numeric(),
        padj = numeric(),
        reject = logical(),
        fit_ok = logical(),
        fit_status = character()
      )
    )
  }

  valid_idx <- !is.na(res_df$pvalue)
  res_df$padj <- NA_real_
  if (any(valid_idx)) {
    res_df$padj[valid_idx] <- p.adjust(res_df$pvalue[valid_idx], method = "BH")
  }
  res_df$reject <- !is.na(res_df$padj) & res_df$padj < alpha

  res_df
}

evaluate_against_truth <- function(res_df, df_taxon_effect, alpha = 0.05) {
  if (exists("da_evaluate_against_truth", mode = "function")) {
    return(da_evaluate_against_truth(
      res_df = res_df,
      df_taxon_effect = df_taxon_effect,
      alpha = alpha
    ))
  }

  if (!all(c("taxon", "is_differential") %in% colnames(df_taxon_effect))) {
    stop("effect_taxon_effect.csv must contain the taxon and is_differential columns")
  }

  df_taxon_effect <- df_taxon_effect %>%
    mutate(
      taxon = as.character(taxon),
      is_differential = as.integer(is_differential)
    )

  res_eval <- merge(
    df_taxon_effect,
    res_df,
    by = "taxon",
    all.x = TRUE
  )

  if (!("reject" %in% colnames(res_eval))) {
    if ("padj" %in% colnames(res_eval)) {
      res_eval$reject <- !is.na(res_eval$padj) & res_eval$padj < alpha
    } else if ("p.adj" %in% colnames(res_eval)) {
      res_eval$reject <- !is.na(res_eval$p.adj) & res_eval$p.adj < alpha
    } else if ("pvalue" %in% colnames(res_eval)) {
      res_eval$padj <- p.adjust(res_eval$pvalue, method = "BH")
      res_eval$reject <- !is.na(res_eval$padj) & res_eval$padj < alpha
    } else {
      stop("the result table lacks the reject / padj / p.adj / pvalue columns")
    }
  }

  truth <- as.logical(res_eval$is_differential)
  called <- as.logical(res_eval$reject)

  called[is.na(called)] <- FALSE
  truth[is.na(truth)] <- FALSE

  TP <- sum(called & truth, na.rm = TRUE)
  FP <- sum(called & !truth, na.rm = TRUE)
  FN <- sum(!called & truth, na.rm = TRUE)
  TN <- sum(!called & !truth, na.rm = TRUE)

  precision <- safe_div(TP, TP + FP)
  recall <- safe_div(TP, TP + FN)
  specificity <- safe_div(TN, TN + FP)
  accuracy <- safe_div(TP + TN, TP + FP + TN + FN)
  FDR <- safe_div(FP, TP + FP)

  F1 <- if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) {
    2 * precision * recall / (precision + recall)
  } else {
    NA_real_
  }

  metrics <- tibble(
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
    n_taxa_total = length(truth),
    n_truth_diff = sum(truth, na.rm = TRUE),
    n_called_diff = sum(called, na.rm = TRUE)
  )

  list(
    res_eval = res_eval,
    metrics = metrics
  )
}

run_one_nc_file <- function(nc_path,
                            df_pseudo_group,
                            df_taxon_effect,
                            frac_name,
                            effect_name,
                            effect_level,
                            alpha = 0.05) {
  nc_obj <- read_nc_tensor_standard(nc_path)
  arr_ptb <- nc_obj$arr

  res_df <- fit_glmmpql_all_taxa(
    arr_ptb = arr_ptb,
    df_pseudo_group = df_pseudo_group,
    alpha = alpha,
    niter = 10,
    verbose = FALSE
  )

  eval_obj <- evaluate_against_truth(
    res_df = res_df,
    df_taxon_effect = df_taxon_effect,
    alpha = alpha
  )

  method_name <- infer_method_name(nc_path, nc_obj$main_var)

  res_eval <- eval_obj$res_eval %>%
    mutate(
      effect_name = effect_name,
      effect_level = effect_level,
      frac = frac_name,
      method = method_name,
      file = basename(nc_path),
      main_var = nc_obj$main_var,
      original_dim_order = paste(nc_obj$original_dim_order, collapse = " -> "),
      formula_used = "y ~ X + bin; random = ~ 1 | participant"
    )

  metrics <- eval_obj$metrics %>%
    mutate(
      effect_name = effect_name,
      effect_level = effect_level,
      frac = frac_name,
      method = method_name,
      file = basename(nc_path),
      main_var = nc_obj$main_var,
      original_dim_order = paste(nc_obj$original_dim_order, collapse = " -> "),
      formula_used = "y ~ X + bin; random = ~ 1 | participant"
    ) %>%
    relocate(effect_name, effect_level, frac, method, file, main_var, original_dim_order, formula_used)

  list(
    nc_info = nc_obj,
    res_df = res_df,
    res_eval = res_eval,
    metrics = metrics
  )
}

make_three_plots <- function(metrics_df, output_dir, effect_level_label) {
  metric_plot_df <- metrics_df %>%
    dplyr::filter(method %in% c("zeroed", "imputed")) %>%
    dplyr::select(frac, method, precision, recall, F1, specificity, accuracy) %>%
    tidyr::pivot_longer(
      cols = c(precision, recall, F1, specificity, accuracy),
      names_to = "metric",
      values_to = "value"
    )

  conf_plot_df <- metrics_df %>%
    dplyr::filter(method %in% c("zeroed", "imputed")) %>%
    dplyr::select(frac, method, TP, FP, TN, FN) %>%
    tidyr::pivot_longer(
      cols = c(TP, FP, TN, FN),
      names_to = "conf_type",
      values_to = "count"
    )

  imp_good_df <- metrics_df %>%
    dplyr::filter(method %in% c("zeroed", "imputed")) %>%
    dplyr::select(frac, method, TP, FP, recall, F1) %>%
    tidyr::pivot_longer(
      cols = c(TP, FP, recall, F1),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(metric = factor(metric, levels = c("TP", "FP", "recall", "F1")))

  p_metric <- ggplot(metric_plot_df, aes(x = frac, y = value, fill = method)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68) +
    facet_wrap(~ metric, scales = "free_y", nrow = 2) +
    labs(
      title = paste0("glmmPQL performance: imputed vs zeroed (group effect = ", effect_level_label, ")"),
      x = "Fraction folder",
      y = "Metric value",
      fill = "Method"
    ) +
    theme_bw(base_size = 12) +
    theme(
      strip.text = element_text(face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )

  ggsave(
    filename = file.path(output_dir, "barplot_metrics_imputed_vs_zeroed.png"),
    plot = p_metric,
    width = 12,
    height = 7,
    dpi = 300
  )

  p_conf <- ggplot(conf_plot_df, aes(x = frac, y = count, fill = method)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68) +
    facet_wrap(~ conf_type, scales = "free_y", nrow = 2) +
    labs(
      title = paste0("glmmPQL confusion counts: imputed vs zeroed (group effect = ", effect_level_label, ")"),
      x = "Fraction folder",
      y = "Count",
      fill = "Method"
    ) +
    theme_bw(base_size = 12) +
    theme(
      strip.text = element_text(face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )

  ggsave(
    filename = file.path(output_dir, "barplot_confusion_counts.png"),
    plot = p_conf,
    width = 12,
    height = 7,
    dpi = 300
  )

  p_imp_good <- ggplot(imp_good_df, aes(x = frac, y = value, fill = method)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68) +
    facet_wrap(~ metric, scales = "free_y", nrow = 2) +
    labs(
      title = paste0("glmmPQL TP, FP, recall and F1: imputed vs zeroed (group effect = ", effect_level_label, ")"),
      x = "Fraction folder",
      y = "Value",
      fill = "Method"
    ) +
    theme_bw(base_size = 12) +
    theme(
      strip.text = element_text(face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )

  ggsave(
    filename = file.path(output_dir, "imp_good_barplot.png"),
    plot = p_imp_good,
    width = 12,
    height = 7,
    dpi = 300
  )
}

process_one_effect_dir <- function(effect_dir,
                                   sweep_root_dir,
                                   results_dir_name = "results_glmmPQL_compare",
                                   alpha = 0.05) {
  effect_name <- basename(effect_dir)
  effect_level <- parse_level_from_name(effect_name, "^group_effect_")
  effect_level_label <- format_level_label(effect_level)

  message("==============================")
  message("processing effect directory: ", effect_name)
  message("group effect level: ", effect_level_label)

  truth_dir <- find_truth_dir(effect_dir, sweep_root_dir)
  output_dir <- file.path(effect_dir, results_dir_name)
  taxon_output_dir <- file.path(output_dir, "taxon_results")

  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(taxon_output_dir, showWarnings = FALSE, recursive = TRUE)

  group_file <- file.path(truth_dir, "effect_pseudo_group.csv")
  taxon_file <- file.path(truth_dir, "effect_taxon_effect.csv")

  if (!file.exists(group_file)) stop("file not found: ", group_file)
  if (!file.exists(taxon_file)) stop("file not found: ", taxon_file)

  df_pseudo_group <- read_csv(group_file, show_col_types = FALSE) %>%
    mutate(participant = as.character(participant))

  df_taxon_effect <- read_csv(taxon_file, show_col_types = FALSE) %>%
    mutate(taxon = as.character(taxon))

  frac_dirs <- list.dirs(effect_dir, recursive = FALSE, full.names = TRUE)
  frac_dirs <- frac_dirs[grepl("^frac_", basename(frac_dirs))]

  if (length(frac_dirs) == 0) {
    stop("no frac_* subdirectory found under: ", effect_dir)
  }

  frac_tbl <- tibble(
    frac_dir = frac_dirs,
    frac_name = basename(frac_dirs),
    frac_value = parse_level_from_name(basename(frac_dirs), "^frac_")
  ) %>%
    arrange(frac_value, frac_name)

  frac_dirs <- frac_tbl$frac_dir
  frac_levels <- frac_tbl$frac_name

  all_metrics <- list()
  all_taxon_results <- list()

  for (frac_dir in frac_dirs) {
    frac_name <- basename(frac_dir)
    message("processing: ", effect_name, " / ", frac_name)

    nc_files <- list.files(frac_dir, pattern = "\\.nc$", full.names = TRUE)
    if (length(nc_files) == 0) {
      warning("no .nc file in this frac directory: ", frac_dir)
      next
    }

    folder_metrics <- list()
    folder_taxon_results <- list()

    for (nc_path in nc_files) {
      message("  reading file: ", basename(nc_path))

      one_res <- run_one_nc_file(
        nc_path = nc_path,
        df_pseudo_group = df_pseudo_group,
        df_taxon_effect = df_taxon_effect,
        frac_name = frac_name,
        effect_name = effect_name,
        effect_level = effect_level,
        alpha = alpha
      )

      method_name <- unique(one_res$metrics$method)
      if (length(method_name) != 1) method_name <- "unknown"

      folder_metrics[[basename(nc_path)]] <- one_res$metrics
      folder_taxon_results[[basename(nc_path)]] <- one_res$res_eval

      file_stem <- tools::file_path_sans_ext(basename(nc_path))
      write_csv(
        one_res$res_eval,
        file.path(
          taxon_output_dir,
          paste0("taxon_results_", frac_name, "_", method_name, "_", file_stem, ".csv")
        )
      )
    }

    all_metrics[[frac_name]] <- bind_rows(folder_metrics)
    all_taxon_results[[frac_name]] <- bind_rows(folder_taxon_results)
  }

  metrics_df <- bind_rows(all_metrics)
  taxon_df <- bind_rows(all_taxon_results)

  if (nrow(metrics_df) == 0) {
    warning("no valid results were produced for this effect directory: ", effect_dir)
    return(NULL)
  }

  metrics_df <- metrics_df %>%
    mutate(
      frac = factor(frac, levels = frac_levels),
      method = factor(method, levels = c("zeroed", "imputed", "unknown"))
    ) %>%
    arrange(frac, method)

  if (nrow(taxon_df) > 0) {
    taxon_df <- taxon_df %>%
      mutate(
        frac = factor(frac, levels = frac_levels),
        method = factor(method, levels = c("zeroed", "imputed", "unknown"))
      ) %>%
      arrange(frac, method, taxon)
  }

  write_csv(metrics_df, file.path(output_dir, "glmmPQL_metrics_summary.csv"))
  write_csv(taxon_df, file.path(output_dir, "glmmPQL_taxon_level_results_all.csv"))

  plot_metrics_df <- metrics_df %>%
    dplyr::filter(method %in% c("zeroed", "imputed"))

  if (nrow(plot_metrics_df) > 0) {
    make_three_plots(
      metrics_df = plot_metrics_df,
      output_dir = output_dir,
      effect_level_label = effect_level_label
    )
  } else {
    warning("no zeroed / imputed results available for plotting: ", effect_dir)
  }

  effect_info <- tibble(
    effect_name = effect_name,
    effect_level = effect_level,
    effect_level_label = effect_level_label,
    truth_dir = truth_dir,
    output_dir = output_dir,
    n_frac_dirs = length(frac_dirs),
    n_result_rows = nrow(metrics_df)
  )
  write_csv(effect_info, file.path(output_dir, "effect_run_info.csv"))

  message("done with effect directory: ", effect_name)
  message("results written to: ", output_dir)

  list(
    effect_name = effect_name,
    effect_level = effect_level,
    output_dir = output_dir,
    metrics_df = metrics_df,
    taxon_df = taxon_df
  )
}

##############################
## 3. Main loop over all group_effect_* directories
##############################
if (!dir.exists(sweep_root_dir)) {
  stop("sweep_root_dir does not exist: ", sweep_root_dir)
}

effect_dirs <- list.dirs(sweep_root_dir, recursive = FALSE, full.names = TRUE)
effect_dirs <- effect_dirs[grepl("^group_effect_", basename(effect_dirs))]

if (length(effect_dirs) == 0) {
  stop("no group_effect_* directory found under sweep_root_dir")
}

effect_tbl <- tibble(
  effect_dir = effect_dirs,
  effect_name = basename(effect_dirs),
  effect_value = parse_level_from_name(basename(effect_dirs), "^group_effect_")
) %>%
  arrange(effect_value, effect_name)

effect_dirs <- effect_tbl$effect_dir

all_effect_results <- list()
all_effect_metrics <- list()

for (effect_dir in effect_dirs) {
  one_effect <- process_one_effect_dir(
    effect_dir = effect_dir,
    sweep_root_dir = sweep_root_dir,
    results_dir_name = results_dir_name,
    alpha = alpha
  )

  if (!is.null(one_effect)) {
    all_effect_results[[basename(effect_dir)]] <- one_effect
    all_effect_metrics[[basename(effect_dir)]] <- one_effect$metrics_df
  }
}

##############################
## 4. Aggregate metrics across all effects
##############################
if (length(all_effect_metrics) > 0) {
  combined_output_dir <- file.path(sweep_root_dir, paste0(results_dir_name, "_all_effects"))
  dir.create(combined_output_dir, showWarnings = FALSE, recursive = TRUE)

  combined_metrics_df <- bind_rows(all_effect_metrics) %>%
    arrange(effect_level, frac, method)

  write_csv(
    combined_metrics_df,
    file.path(combined_output_dir, "glmmPQL_metrics_summary_all_effects.csv")
  )

  message("summaries for all effects written to: ", combined_output_dir)
} else {
  warning("no effect directory produced valid results")
}
