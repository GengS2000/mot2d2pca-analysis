##############################
## 0. Install and load packages
##############################
local_lib <- file.path(getwd(), ".Rlib")
dir.create(local_lib, showWarnings = FALSE, recursive = TRUE)
user_lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library", paste(R.version$major, sub("\\..*$", "", R.version$minor), sep = "."))
.libPaths(unique(c(normalizePath(local_lib, mustWork = FALSE), user_lib, .libPaths())))

required_pkgs <- c(
  "ncdf4", "MicrobiomeStat", "dplyr", "tidyr",
  "purrr", "stringr", "readr", "ggplot2", "tibble"
)

to_install <- required_pkgs[!required_pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) {
  install.packages(to_install)
}

library(ncdf4)
library(MicrobiomeStat)
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

##############################
## 1. Parameters: edit only this block
##############################
# Root directory of the effect sweep
# Override with the DA_SWEEP_ROOT environment variable.
sweep_root_dir <- normalizePath(
  Sys.getenv("DA_SWEEP_ROOT", unset = "simulation3_effect_tune"),
  mustWork = FALSE
)

# Name of the results directory created inside each group_effect_* directory
results_dir_name <- "results_linda_compare"

# Whether to back-transform feature.dat with exp()
# TRUE when the .nc holds log-scale values
# FALSE when the .nc already holds counts / abundances / proportions
apply_exp_transform <- TRUE

# LinDA input type
# Either "count" or "proportion"
feature_dat_type <- "count"

# Try the random-effect formula first; fall back to fixed effects if unsupported
use_random_effect_formula <- TRUE

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
    "truth directory not found. Tried:\n",
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

  stop("cannot read axis: ", axis_name)
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
        "\ncandidate variables: ", paste(var_names, collapse = ", ")
      )
    }
    main_var <- three_d_vars
  }

  arr <- ncvar_get(nc, main_var)
  var_dim_names <- sapply(nc$var[[main_var]]$dim, function(x) x$name)

  participants <- as.character(get_nc_axis_values(nc, "participant"))
  taxa         <- as.character(get_nc_axis_values(nc, "taxon"))
  bins         <- as.character(get_nc_axis_values(nc, "bin"))

  target_order <- c("participant", "taxon", "bin")
  perm <- match(target_order, var_dim_names)

  if (any(is.na(perm))) {
    stop(
      "participant/taxon/bin not found among the main variable dimensions: ", nc_path,
      "\ncurrent dimension order: ", paste(var_dim_names, collapse = " × ")
    )
  }

  arr_std <- aperm(arr, perm = perm)

  if (dim(arr_std)[1] != length(participants)) {
    stop("participant dimension length does not match its coordinate: ", nc_path)
  }
  if (dim(arr_std)[2] != length(taxa)) {
    stop("taxon dimension length does not match its coordinate: ", nc_path)
  }
  if (dim(arr_std)[3] != length(bins)) {
    stop("bin dimension length does not match its coordinate: ", nc_path)
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

is_target_nc_file <- function(nc_path) {
  method_name <- infer_method_name(nc_path)
  method_name %in% c("zeroed", "imputed")
}

build_linda_input <- function(arr_ptb, df_pseudo_group) {
  participants <- dimnames(arr_ptb)$participant
  taxa         <- dimnames(arr_ptb)$taxon
  bins         <- dimnames(arr_ptb)$bin

  if (!all(c("participant", "X") %in% colnames(df_pseudo_group))) {
    stop("effect_pseudo_group.csv must contain at least the participant and X columns")
  }

  df_pseudo_group <- df_pseudo_group %>%
    mutate(
      participant = as.character(participant),
      X = as.numeric(X)
    )

  missing_participants <- setdiff(participants, df_pseudo_group$participant)
  if (length(missing_participants) > 0) {
    stop(
      "the following participants are absent from the truth group table: ",
      paste(missing_participants, collapse = ", ")
    )
  }

  sample_ids <- character(0)
  feature_list <- list()
  meta_list <- list()

  for (p in seq_along(participants)) {
    for (b in seq_along(bins)) {
      sid <- paste(participants[p], bins[b], sep = "__")
      sample_ids <- c(sample_ids, sid)

      feature_list[[sid]] <- arr_ptb[p, , b]

      meta_list[[sid]] <- data.frame(
        sample = sid,
        participant = participants[p],
        bin = bins[b],
        stringsAsFactors = FALSE
      )
    }
  }

  feature.dat <- do.call(cbind, feature_list)
  feature.dat <- as.matrix(feature.dat)
  rownames(feature.dat) <- taxa
  colnames(feature.dat) <- sample_ids

  meta.dat <- do.call(rbind, meta_list)
  rownames(meta.dat) <- meta.dat$sample

  idx <- match(meta.dat$participant, df_pseudo_group$participant)
  meta.dat$X <- df_pseudo_group$X[idx]

  meta.dat <- meta.dat %>%
    mutate(
      participant = factor(participant),
      bin = factor(bin)
    )

  meta.dat <- meta.dat[match(colnames(feature.dat), meta.dat$sample), , drop = FALSE]
  stopifnot(all(meta.dat$sample == colnames(feature.dat)))

  list(
    feature.dat = feature.dat,
    meta.dat = meta.dat
  )
}

fit_linda_wrapper <- function(feature.dat, meta.dat,
                              feature_dat_type = "count",
                              use_random_effect_formula = TRUE) {
  formula1 <- if (use_random_effect_formula) "~ X * bin + (1 | participant)" else "~ X * bin"
  formula2 <- "~ X * bin"

  fit1 <- tryCatch(
    linda(
      feature.dat = feature.dat,
      meta.dat = meta.dat,
      formula = formula1,
      feature.dat.type = feature_dat_type,
      prev.filter = 0,
      mean.abund.filter = 0,
      max.abund.filter = 0,
      is.winsor = FALSE
    ),
    error = function(e) e
  )

  if (!inherits(fit1, "error")) {
    return(list(res = fit1, formula_used = formula1))
  }

  message("first LinDA fit failed; error: ", fit1$message)

  if (use_random_effect_formula) {
    message("falling back to the formula without random effects: ", formula2)
    fit2 <- linda(
      feature.dat = feature.dat,
      meta.dat = meta.dat,
      formula = formula2,
      feature.dat.type = feature_dat_type,
      prev.filter = 0,
      mean.abund.filter = 0,
      max.abund.filter = 0,
      is.winsor = FALSE
    )
    return(list(res = fit2, formula_used = formula2))
  } else {
    stop("LinDA fit failed: ", fit1$message)
  }
}

extract_main_effect_table <- function(linda_res, main_var = "X") {
  if (!("output" %in% names(linda_res))) {
    stop("the LinDA result has no output slot")
  }

  if (main_var %in% names(linda_res$output)) {
    out <- linda_res$output[[main_var]]
    out$taxon <- rownames(out)
    out <- out[, c("taxon", setdiff(colnames(out), "taxon")), drop = FALSE]
    return(out)
  }

  cand <- names(linda_res$output)[str_detect(names(linda_res$output), paste0("^", main_var, "$|", main_var))]
  if (length(cand) >= 1) {
    out <- linda_res$output[[cand[1]]]
    out$taxon <- rownames(out)
    out <- out[, c("taxon", setdiff(colnames(out), "taxon")), drop = FALSE]
    return(out)
  }

  stop(
    "main-effect result table not found in the LinDA output. Available names: ",
    paste(names(linda_res$output), collapse = ", ")
  )
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
    stop("effect_taxon_effect.csv must contain at least the taxon and is_differential columns")
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
      stop("the result table has neither reject nor a padj/p.adj/pvalue column from which to build it")
    }
  }

  truth  <- as.logical(res_eval$is_differential)
  called <- as.logical(res_eval$reject)

  called[is.na(called)] <- FALSE
  truth[is.na(truth)] <- FALSE

  TP <- sum(called & truth, na.rm = TRUE)
  FP <- sum(called & !truth, na.rm = TRUE)
  FN <- sum(!called & truth, na.rm = TRUE)
  TN <- sum(!called & !truth, na.rm = TRUE)

  precision   <- safe_div(TP, TP + FP)
  recall      <- safe_div(TP, TP + FN)
  specificity <- safe_div(TN, TN + FP)
  accuracy    <- safe_div(TP + TN, TP + FP + TN + FN)
  FDR         <- safe_div(FP, TP + FP)

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
                            apply_exp_transform = TRUE,
                            feature_dat_type = "count",
                            use_random_effect_formula = TRUE,
                            alpha = 0.05) {
  nc_obj <- read_nc_tensor_standard(nc_path)
  arr_ptb <- nc_obj$arr

  linda_input <- build_linda_input(arr_ptb, df_pseudo_group)
  feature.dat <- linda_input$feature.dat
  meta.dat <- linda_input$meta.dat

  if (apply_exp_transform) {
    feature.dat <- 10^feature.dat
  }

  fit_obj <- fit_linda_wrapper(
    feature.dat = feature.dat,
    meta.dat = meta.dat,
    feature_dat_type = feature_dat_type,
    use_random_effect_formula = use_random_effect_formula
  )

  res_df <- extract_main_effect_table(fit_obj$res, main_var = "X")

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
      original_dim_order = paste(nc_obj$original_dim_order, collapse = " × "),
      formula_used = fit_obj$formula_used
    )

  metrics <- eval_obj$metrics %>%
    mutate(
      effect_name = effect_name,
      effect_level = effect_level,
      frac = frac_name,
      method = method_name,
      file = basename(nc_path),
      main_var = nc_obj$main_var,
      original_dim_order = paste(nc_obj$original_dim_order, collapse = " × "),
      formula_used = fit_obj$formula_used
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
      title = paste0("Differential inference performance: imputed vs zeroed (group effect = ", effect_level_label, ")"),
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
      title = paste0("Confusion matrix counts: imputed vs zeroed (group effect = ", effect_level_label, ")"),
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
      title = paste0("TP, FP, recall and F1: imputed vs zeroed (group effect = ", effect_level_label, ")"),
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
                                   results_dir_name = "results_linda_compare",
                                   apply_exp_transform = TRUE,
                                   feature_dat_type = "count",
                                   use_random_effect_formula = TRUE,
                                   alpha = 0.05) {
  effect_name <- basename(effect_dir)
  effect_level <- parse_level_from_name(effect_name, "^group_effect_")
  effect_level_label <- format_level_label(effect_level)

  message("==============================")
  message("processing effect directory: ", effect_name)
  message("group-effect level: ", effect_level_label)

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
    stop(effect_dir, " contains no frac_... directory")
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
    nc_files <- nc_files[vapply(nc_files, is_target_nc_file, logical(1))]
    if (length(nc_files) == 0) {
      warning("no .nc file in the directory; skipping: ", frac_dir)
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
        apply_exp_transform = apply_exp_transform,
        feature_dat_type = feature_dat_type,
        use_random_effect_formula = use_random_effect_formula,
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
    warning("no results were produced for this effect directory: ", effect_dir)
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

  write_csv(metrics_df, file.path(output_dir, "linda_metrics_summary.csv"))
  write_csv(taxon_df, file.path(output_dir, "linda_taxon_level_results_all.csv"))

  plot_metrics_df <- metrics_df %>%
    dplyr::filter(method %in% c("zeroed", "imputed"))

  if (nrow(plot_metrics_df) > 0) {
    make_three_plots(
      metrics_df = plot_metrics_df,
      output_dir = output_dir,
      effect_level_label = effect_level_label
    )
  } else {
    warning("no zeroed / imputed results; skipping plots: ", effect_dir)
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

  message("done: ", effect_name)
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
## 3. Main loop over group_effect_*
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
    apply_exp_transform = apply_exp_transform,
    feature_dat_type = feature_dat_type,
    use_random_effect_formula = use_random_effect_formula,
    alpha = alpha
  )

  if (!is.null(one_effect)) {
    all_effect_results[[basename(effect_dir)]] <- one_effect
    all_effect_metrics[[basename(effect_dir)]] <- one_effect$metrics_df
  }
}

##############################
## 4. Overall summary (optional)
##############################
if (length(all_effect_metrics) > 0) {
  combined_output_dir <- file.path(sweep_root_dir, paste0(results_dir_name, "_all_effects"))
  dir.create(combined_output_dir, showWarnings = FALSE, recursive = TRUE)

  combined_metrics_df <- bind_rows(all_effect_metrics) %>%
    arrange(effect_level, frac, method)

  write_csv(
    combined_metrics_df,
    file.path(combined_output_dir, "linda_metrics_summary_all_effects.csv")
  )

  message("all done. Combined summary directory: ", combined_output_dir)
} else {
  warning("no effect directory produced any results.")
}
