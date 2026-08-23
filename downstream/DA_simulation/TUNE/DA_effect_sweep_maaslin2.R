# ========================================================
# Effect-sweep differential-abundance comparison, MaAsLin2 engine.
#
# Layout:
#   - traverses simulation3_effect_sweep/group_effect_*/frac_* (two levels);
#   - writes results_maaslin2_compare inside each effect directory, containing
#     three comparison plots, the summary table and the per-taxon results;
#   - writes the cross-effect summary results_maaslin2_compare_all_effects at
#     the repository root of the sweep.
# ========================================================

# =========================
# 0. Install / load packages
# =========================
local_lib <- file.path(getwd(), ".Rlib")
dir.create(local_lib, showWarnings = FALSE, recursive = TRUE)
user_lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library", paste(R.version$major, sub("\\..*$", "", R.version$minor), sep = "."))
.libPaths(unique(c(normalizePath(local_lib, mustWork = FALSE), user_lib, .libPaths())))

required_pkgs <- c(
  "ncdf4", "dplyr", "tidyr", "readr",
  "ggplot2", "tibble", "stringr"
)

to_install <- required_pkgs[!required_pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) {
  install.packages(to_install)
}

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}
if (!requireNamespace("Maaslin2", quietly = TRUE)) {
  BiocManager::install("Maaslin2")
}

suppressPackageStartupMessages({
  library(ncdf4)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(tibble)
  library(stringr)
  library(Maaslin2)
})

common_script <- file.path(getwd(), "DA_effect_sweep_common.R")
if (file.exists(common_script)) {
  source(common_script)
}

# =========================
# 1. Parameters: edit only this block
# =========================
# Override with the DA_SWEEP_ROOT environment variable.
root_dir <- normalizePath(
  Sys.getenv("DA_SWEEP_ROOT", unset = "simulation3_effect_tune"),
  mustWork = FALSE
)
alpha <- 0.05

# Effect ordering
expected_effect_dirs <- NULL

# Masking-fraction ordering
frac_levels <- paste0("frac_0p", 1:9)

# Output naming prefix
per_effect_output_folder <- "results_maaslin2_compare"
all_effects_output_folder <- "results_maaslin2_compare_all_effects"
output_prefix <- "maaslin2"

# =========================
# 2. Utility functions
# =========================
safe_div <- function(x, y) {
  ifelse(y == 0, NA_real_, x / y)
}

mean_or_na <- function(x) {
  if (length(x) == 0 || all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
}

bind_or_empty <- function(x) {
  if (length(x) == 0) return(data.frame())
  dplyr::bind_rows(x)
}

parse_effect_value <- function(x) {
  x <- basename(x)
  x <- str_remove(x, "^group_effect_")
  x <- str_replace_all(x, "p", ".")
  suppressWarnings(as.numeric(x))
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

infer_method_name <- function(file_path, main_var = NULL) {
  nm <- basename(file_path)

  if (str_detect(nm, regex("zero", ignore_case = TRUE))) return("zeroed")
  if (str_detect(nm, regex("imput", ignore_case = TRUE))) return("imputed")

  if (!is.null(main_var)) {
    if (str_detect(main_var, regex("zero", ignore_case = TRUE))) return("zeroed")
    if (str_detect(main_var, regex("imput", ignore_case = TRUE))) return("imputed")
  }

  "unknown"
}

is_target_nc_file <- function(nc_path) {
  method_name <- infer_method_name(nc_path)
  method_name %in% c("zeroed", "imputed")
}

resolve_truth_paths <- function(effect_dir, root_dir) {
  group_candidates <- c(
    file.path(effect_dir, "truth", "effect_pseudo_group.csv"),
    file.path(effect_dir, "effect_pseudo_group.csv"),
    file.path(root_dir, "truth", "effect_pseudo_group.csv"),
    file.path(root_dir, "effect_pseudo_group.csv")
  )

  truth_candidates <- c(
    file.path(effect_dir, "truth", "effect_taxon_effect.csv"),
    file.path(effect_dir, "effect_taxon_effect.csv"),
    file.path(root_dir, "truth", "effect_taxon_effect.csv"),
    file.path(root_dir, "effect_taxon_effect.csv")
  )

  group_path <- group_candidates[file.exists(group_candidates)][1]
  truth_path <- truth_candidates[file.exists(truth_candidates)][1]

  if (is.na(group_path) || is.na(truth_path)) {
    stop(
      paste0(
        "cannot locate the truth/group files in the effect directory:\n",
        "effect_dir = ", effect_dir, "\n",
        "group candidates = ", paste(group_candidates, collapse = " | "), "\n",
        "truth candidates = ", paste(truth_candidates, collapse = " | ")
      )
    )
  }

  list(group_path = group_path, truth_path = truth_path)
}

read_nc_tensor_maaslin2 <- function(nc_path) {
  nc <- nc_open(nc_path)
  on.exit(nc_close(nc), add = TRUE)

  coord_names <- c("participant", "taxon", "bin")
  var_names <- names(nc$var)

  preferred <- intersect(var_names, c("y_imputed", "y_zeroed"))
  if (length(preferred) >= 1) {
    main_var <- preferred[1]
  } else {
    non_coord_vars <- setdiff(var_names, coord_names)
    three_d_vars <- non_coord_vars[sapply(non_coord_vars, function(v) {
      length(nc$var[[v]]$dim) == 3
    })]

    if (length(three_d_vars) != 1) {
      stop(
        "cannot uniquely identify the three-dimensional main variable: ", nc_path,
        "\ncandidate variables: ", paste(var_names, collapse = ", ")
      )
    }
    main_var <- three_d_vars[1]
  }

  arr <- ncvar_get(nc, main_var)
  var_dim_names <- sapply(nc$var[[main_var]]$dim, function(x) x$name)

  participants <- as.character(get_nc_axis_values(nc, "participant"))
  taxa         <- as.character(get_nc_axis_values(nc, "taxon"))
  bins         <- as.character(get_nc_axis_values(nc, "bin"))

  target_order <- c("bin", "taxon", "participant")
  perm <- match(target_order, var_dim_names)

  if (any(is.na(perm))) {
    stop(
      "bin/taxon/participant not found among the main variable dimensions: ", nc_path,
      "\ncurrent dimension order: ", paste(var_dim_names, collapse = " × ")
    )
  }

  tensor <- aperm(arr, perm = perm)

  if (dim(tensor)[1] != length(bins)) {
    stop("bin dimension length does not match its coordinate: ", nc_path)
  }
  if (dim(tensor)[2] != length(taxa)) {
    stop("taxon dimension length does not match its coordinate: ", nc_path)
  }
  if (dim(tensor)[3] != length(participants)) {
    stop("participant dimension length does not match its coordinate: ", nc_path)
  }

  dimnames(tensor) <- list(
    bin = bins,
    taxon = taxa,
    participant = participants
  )

  list(
    tensor = tensor,
    participants = participants,
    taxa = taxa,
    bins = bins,
    main_var = main_var,
    original_dim_order = var_dim_names
  )
}

prepare_truth_df <- function(truth_path) {
  truth_df_raw <- read.csv(
    truth_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  possible_taxon_cols <- c(
    "taxon", "feature", "taxa", "Taxon",
    "Feature", "Taxa", "otu", "OTU"
  )
  taxon_col <- possible_taxon_cols[possible_taxon_cols %in% colnames(truth_df_raw)][1]

  if (is.na(taxon_col)) {
    stop(
      paste0(
        "taxon column not found in the truth file. Current column names: ",
        paste(colnames(truth_df_raw), collapse = ", ")
      )
    )
  }

  if (!"is_differential" %in% colnames(truth_df_raw)) {
    stop("the truth file lacks the is_differential column.")
  }

  truth_df <- truth_df_raw %>%
    mutate(
      taxon = trimws(as.character(.data[[taxon_col]])),
      truth_chr = tolower(trimws(as.character(is_differential))),
      truth = case_when(
        truth_chr %in% c("true", "t", "1", "yes") ~ TRUE,
        truth_chr %in% c("false", "f", "0", "no") ~ FALSE,
        TRUE ~ NA
      )
    ) %>%
    dplyr::select(taxon, truth) %>%
    distinct()

  if (any(is.na(truth_df$truth))) {
    stop("the truth file contains unrecognized is_differential values.")
  }

  if (anyDuplicated(truth_df$taxon) > 0) {
    warning("duplicate taxa in the truth file; kept unique taxa only.")
    truth_df <- truth_df %>% distinct(taxon, .keep_all = TRUE)
  }

  truth_df
}

prepare_group_df <- function(group_path) {
  group_df <- read.csv(
    group_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  if (!"group" %in% colnames(group_df) && "pseudo_group" %in% colnames(group_df)) {
    group_df$group <- group_df$pseudo_group
  }

  if (!all(c("participant", "group") %in% colnames(group_df))) {
    stop(
      paste0(
        "the group file must contain the participant and group columns. Current column names: ",
        paste(colnames(group_df), collapse = ", ")
      )
    )
  }

  group_df %>%
    mutate(
      participant = trimws(as.character(participant)),
      group = trimws(as.character(group))
    )
}

run_one_nc_file_maaslin2 <- function(nc_path,
                                     group_df,
                                     truth_df,
                                     effect_name,
                                     frac_name,
                                     output_dir,
                                     alpha = 0.05) {
  message("    reading file: ", basename(nc_path))

  nc_obj <- read_nc_tensor_maaslin2(nc_path)
  tensor <- nc_obj$tensor

  long_df <- as.data.frame.table(
    tensor,
    responseName = "abundance",
    stringsAsFactors = FALSE
  )
  colnames(long_df) <- c("bin", "taxon", "participant", "abundance")

  long_df <- long_df %>%
    mutate(
      bin = trimws(as.character(bin)),
      taxon = trimws(as.character(taxon)),
      participant = trimws(as.character(participant)),
      abundance = as.numeric(abundance)
    )

  long_df <- long_df %>%
    left_join(group_df, by = "participant")

  if (any(is.na(long_df$group))) {
    missing_ids <- unique(long_df$participant[is.na(long_df$group)])
    stop(
      paste0(
        "no group assignment in the group file for the following participants: ",
        paste(head(missing_ids, 20), collapse = ", ")
      )
    )
  }

  long_df <- long_df %>%
    mutate(sample_id = paste(participant, bin, sep = "__"))

  feature_tbl <- long_df %>%
    dplyr::select(sample_id, taxon, abundance) %>%
    tidyr::pivot_wider(
      names_from = taxon,
      values_from = abundance,
      values_fill = 0
    )

  feature_df <- as.data.frame(
    feature_tbl,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  rownames(feature_df) <- feature_df$sample_id
  feature_df$sample_id <- NULL

  taxon_map <- data.frame(
    taxon_original = colnames(feature_df),
    taxon_make = make.names(colnames(feature_df), unique = TRUE),
    stringsAsFactors = FALSE
  )

  meta_df <- long_df %>%
    distinct(sample_id, participant, bin, group) %>%
    as.data.frame(stringsAsFactors = FALSE)

  rownames(meta_df) <- meta_df$sample_id
  meta_df$sample_id <- NULL

  meta_df$participant <- factor(meta_df$participant)
  meta_df$bin <- factor(meta_df$bin)
  meta_df$group <- factor(meta_df$group)

  if (!"nonCD" %in% levels(meta_df$group)) {
    stop(
      paste0(
        "nonCD absent from group. Current levels: ",
        paste(levels(meta_df$group), collapse = ", ")
      )
    )
  }

  meta_df$group <- relevel(meta_df$group, ref = "nonCD")
  meta_df <- meta_df[rownames(feature_df), , drop = FALSE]

  if (!identical(rownames(meta_df), rownames(feature_df))) {
    stop("meta_df and feature_df sample orders are not aligned.")
  }

  method_name <- infer_method_name(nc_path, nc_obj$main_var)
  file_stem <- tools::file_path_sans_ext(basename(nc_path))
  one_output_dir <- file.path(output_dir, frac_name, paste0(method_name, "_", file_stem))
  dir.create(one_output_dir, recursive = TRUE, showWarnings = FALSE)

  write.csv(
    taxon_map,
    file.path(one_output_dir, "taxon_name_map.csv"),
    row.names = FALSE,
    quote = TRUE
  )

  fit <- Maaslin2(
    input_data       = feature_df,
    input_metadata   = meta_df,
    output           = one_output_dir,

    fixed_effects    = c("group", "bin"),
    random_effects   = c("participant"),
    reference        = c("group,nonCD"),

    normalization    = "NONE",
    transform        = "NONE",
    analysis_method  = "LM",

    min_abundance    = 0,
    min_prevalence   = 0,
    min_variance     = 0,
    standardize      = FALSE,
    plot_heatmap     = FALSE,
    plot_scatter     = FALSE
  )

  res_path <- file.path(one_output_dir, "all_results.tsv")
  if (!file.exists(res_path)) {
    stop("MaAsLin2 did not produce all_results.tsv: ", one_output_dir)
  }

  res <- read.delim(
    res_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  res <- res %>%
    mutate(
      feature = trimws(as.character(feature)),
      metadata = trimws(as.character(metadata))
    )

  res2 <- res %>%
    left_join(
      taxon_map %>% transmute(feature = taxon_original, taxon_exact = taxon_original),
      by = "feature"
    ) %>%
    left_join(
      taxon_map %>% transmute(feature_make = taxon_make, taxon_from_make = taxon_original),
      by = c("feature" = "feature_make")
    ) %>%
    mutate(
      taxon = dplyr::coalesce(taxon_exact, taxon_from_make, feature)
    ) %>%
    dplyr::select(-taxon_exact, -taxon_from_make)

  group_res <- res2 %>%
    filter(metadata == "group") %>%
    arrange(qval, pval)

  write.csv(
    group_res,
    file.path(one_output_dir, "group_differential_taxa_restored.csv"),
    row.names = FALSE,
    quote = TRUE
  )

  pred_df <- group_res %>%
    group_by(taxon) %>%
    summarise(
      min_pval = if (all(is.na(pval))) NA_real_ else min(pval, na.rm = TRUE),
      min_qval = if (all(is.na(qval))) NA_real_ else min(qval, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      predicted = !is.na(min_qval) & (min_qval < alpha)
    )

  write.csv(
    pred_df,
    file.path(one_output_dir, "predicted_taxa_from_maaslin2.csv"),
    row.names = FALSE,
    quote = TRUE
  )

  std_eval <- NULL
  if (exists("da_standardize_taxon_results", mode = "function")) {
    std_eval <- da_standardize_taxon_results(
      res_df = pred_df %>%
        transmute(
          taxon = taxon,
          pvalue = min_pval,
          qvalue = min_qval,
          significant = predicted
        ),
      df_taxon_effect = truth_df %>% transmute(taxon = taxon, is_true_diff = truth),
      alpha = alpha,
      method_name = "maaslin2",
      data_type = method_name,
      masking_fraction = da_parse_level_from_name(frac_name, "^frac_"),
      group_effect_mean = da_parse_level_from_name(effect_name, "^group_effect_"),
      effect_name = effect_name,
      frac_name = frac_name
    )
  }

  merge_diag <- full_join(
    truth_df %>% mutate(in_truth = TRUE),
    pred_df %>% mutate(in_pred = TRUE),
    by = "taxon"
  )

  matched_df <- merge_diag %>%
    filter(!is.na(in_truth) & !is.na(in_pred))

  truth_only_df <- merge_diag %>%
    filter(!is.na(in_truth) & is.na(in_pred))

  pred_only_df <- merge_diag %>%
    filter(is.na(in_truth) & !is.na(in_pred))

  write.csv(
    matched_df,
    file.path(one_output_dir, "merge_matched_taxa.csv"),
    row.names = FALSE,
    quote = TRUE
  )

  write.csv(
    truth_only_df,
    file.path(one_output_dir, "merge_truth_only_taxa.csv"),
    row.names = FALSE,
    quote = TRUE
  )

  write.csv(
    pred_only_df,
    file.path(one_output_dir, "merge_pred_only_taxa.csv"),
    row.names = FALSE,
    quote = TRUE
  )

  compare_df <- truth_df %>%
    left_join(pred_df, by = "taxon") %>%
    mutate(
      predicted = ifelse(is.na(predicted), FALSE, predicted),
      min_pval = ifelse(is.na(min_pval), 1, min_pval),
      min_qval = ifelse(is.na(min_qval), 1, min_qval),
      class = case_when(
        truth & predicted   ~ "TP",
        !truth & predicted  ~ "FP",
        !truth & !predicted ~ "TN",
        truth & !predicted  ~ "FN"
      ),
      effect = effect_name,
      frac = frac_name,
      method = method_name,
      file = basename(nc_path),
      main_var = nc_obj$main_var,
      original_dim_order = paste(nc_obj$original_dim_order, collapse = " × ")
    )

  if (!is.null(std_eval)) {
    std_keep <- std_eval %>%
      dplyr::select(
        taxon, method_name, data_type, masking_fraction, group_effect_mean,
        sim_seed, mask_seed, estimate, statistic, pvalue, qvalue,
        significant, is_true_diff, fit_status, fit_message
      )
    compare_df <- compare_df %>%
      dplyr::select(-dplyr::any_of(setdiff(colnames(std_keep), "taxon"))) %>%
      left_join(std_keep, by = "taxon")
  }

  write.csv(
    compare_df,
    file.path(one_output_dir, "taxon_truth_vs_prediction_comparison.csv"),
    row.names = FALSE,
    quote = TRUE
  )

  TP <- sum(compare_df$class == "TP")
  FP <- sum(compare_df$class == "FP")
  TN <- sum(compare_df$class == "TN")
  FN <- sum(compare_df$class == "FN")

  accuracy    <- safe_div(TP + TN, TP + TN + FP + FN)
  precision   <- safe_div(TP, TP + FP)
  recall      <- safe_div(TP, TP + FN)
  specificity <- safe_div(TN, TN + FP)
  f1_score    <- safe_div(2 * precision * recall, precision + recall)
  fdr         <- safe_div(FP, TP + FP)

  metrics_df <- data.frame(
    effect = effect_name,
    frac = frac_name,
    method = method_name,
    file = basename(nc_path),
    main_var = nc_obj$main_var,
    original_dim_order = paste(nc_obj$original_dim_order, collapse = " × "),
    alpha = alpha,
    TP = TP,
    FP = FP,
    TN = TN,
    FN = FN,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    specificity = specificity,
    F1 = f1_score,
    FDR = fdr,
    n_taxa_total = nrow(compare_df),
    n_truth_diff = sum(compare_df$truth),
    n_called_diff = sum(compare_df$predicted),
    stringsAsFactors = FALSE
  )

  if (!is.null(std_eval) && exists("da_compute_metrics", mode = "function")) {
    common_metrics <- da_compute_metrics(std_eval)
    metrics_df$TP <- common_metrics$TP[1]
    metrics_df$FP <- common_metrics$FP[1]
    metrics_df$TN <- common_metrics$TN[1]
    metrics_df$FN <- common_metrics$FN[1]
    metrics_df$accuracy <- common_metrics$accuracy[1]
    metrics_df$precision <- common_metrics$precision[1]
    metrics_df$recall <- common_metrics$recall[1]
    metrics_df$specificity <- common_metrics$specificity[1]
    metrics_df$F1 <- common_metrics$F1[1]
    metrics_df$FDR <- common_metrics$FDR[1]
    metrics_df$n_taxa_total <- common_metrics$n_taxa_total[1]
    metrics_df$n_truth_diff <- common_metrics$n_truth_diff[1]
    metrics_df$n_called_diff <- common_metrics$n_called_diff[1]
    metrics_df$n_discoveries <- common_metrics$n_discoveries[1]
    metrics_df$n_true_diff <- common_metrics$n_true_diff[1]
    metrics_df$n_successfully_fitted_taxa <- common_metrics$n_successfully_fitted_taxa[1]
    metrics_df$n_failed_taxa <- common_metrics$n_failed_taxa[1]
  }

  write.csv(
    metrics_df,
    file.path(one_output_dir, "taxon_classification_metrics.csv"),
    row.names = FALSE
  )

  confusion_mat <- table(
    Truth = factor(compare_df$truth, levels = c(FALSE, TRUE), labels = c("non_diff", "diff")),
    Pred  = factor(compare_df$predicted, levels = c(FALSE, TRUE), labels = c("non_diff", "diff"))
  )

  write.csv(
    as.data.frame.matrix(confusion_mat),
    file.path(one_output_dir, "taxon_confusion_matrix.csv"),
    row.names = TRUE
  )

  confusion_long <- as.data.frame(as.table(confusion_mat), stringsAsFactors = FALSE) %>%
    dplyr::rename(truth = Truth, pred = Pred, count = Freq) %>%
    mutate(
      effect = effect_name,
      frac = frac_name,
      method = method_name,
      file = basename(nc_path),
      main_var = nc_obj$main_var
    ) %>%
    relocate(effect, frac, method, file, main_var, truth, pred, count)

  group_res_out <- group_res %>%
    mutate(
      effect = effect_name,
      frac = frac_name,
      method = method_name,
      file = basename(nc_path),
      main_var = nc_obj$main_var,
      original_dim_order = paste(nc_obj$original_dim_order, collapse = " × ")
    ) %>%
    relocate(effect, frac, method, file, main_var, original_dim_order)

  list(
    metrics = metrics_df,
    compare = compare_df,
    group_res = group_res_out,
    confusion_long = confusion_long
  )
}

save_effect_level_outputs <- function(effect_name, metrics_df, compare_df_all,
                                      group_res_all, confusion_df_all,
                                      output_dir, frac_levels) {
  if (nrow(metrics_df) == 0) {
    warning("no valid results under this effect: ", effect_name)
    return(invisible(NULL))
  }

  metrics_df <- metrics_df %>%
    mutate(
      frac = factor(frac, levels = frac_levels),
      method = factor(method, levels = c("zeroed", "imputed", "unknown"))
    ) %>%
    arrange(frac, method, file)

  compare_df_all <- compare_df_all %>%
    mutate(
      frac = factor(frac, levels = frac_levels),
      method = factor(method, levels = c("zeroed", "imputed", "unknown"))
    ) %>%
    arrange(frac, method, file, taxon)

  group_res_all <- group_res_all %>%
    mutate(
      frac = factor(frac, levels = frac_levels),
      method = factor(method, levels = c("zeroed", "imputed", "unknown"))
    ) %>%
    arrange(frac, method, file)

  confusion_df_all <- confusion_df_all %>%
    mutate(
      frac = factor(frac, levels = frac_levels),
      method = factor(method, levels = c("zeroed", "imputed", "unknown"))
    ) %>%
    arrange(frac, method, file, truth, pred)

  write_csv(metrics_df, file.path(output_dir, paste0(output_prefix, "_metrics_summary.csv")))
  write_csv(compare_df_all, file.path(output_dir, paste0(output_prefix, "_taxon_truth_vs_prediction_all.csv")))
  write_csv(group_res_all, file.path(output_dir, paste0(output_prefix, "_group_differential_taxa_all.csv")))
  write_csv(confusion_df_all, file.path(output_dir, paste0(output_prefix, "_confusion_matrix_long_all.csv")))

  plot_metrics_df <- metrics_df %>%
    group_by(frac, method) %>%
    summarise(
      TP = mean_or_na(TP),
      FP = mean_or_na(FP),
      TN = mean_or_na(TN),
      FN = mean_or_na(FN),
      accuracy = mean_or_na(accuracy),
      precision = mean_or_na(precision),
      recall = mean_or_na(recall),
      specificity = mean_or_na(specificity),
      F1 = mean_or_na(F1),
      FDR = mean_or_na(FDR),
      n_files = dplyr::n(),
      .groups = "drop"
    ) %>%
    arrange(frac, method)

  write_csv(plot_metrics_df, file.path(output_dir, paste0(output_prefix, "_plot_metrics_summary.csv")))

  metric_plot_df <- plot_metrics_df %>%
    dplyr::filter(method %in% c("zeroed", "imputed")) %>%
    dplyr::select(frac, method, precision, recall, F1, specificity, accuracy) %>%
    tidyr::pivot_longer(
      cols = c(precision, recall, F1, specificity, accuracy),
      names_to = "metric",
      values_to = "value"
    )

  p_metric <- ggplot(metric_plot_df, aes(x = frac, y = value, fill = method)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68) +
    facet_wrap(~ metric, scales = "free_y", nrow = 2) +
    labs(
      title = paste0(effect_name, ": differential inference performance (imputed vs zeroed)"),
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

  conf_plot_df <- plot_metrics_df %>%
    dplyr::filter(method %in% c("zeroed", "imputed")) %>%
    dplyr::select(frac, method, TP, FP, TN, FN) %>%
    tidyr::pivot_longer(
      cols = c(TP, FP, TN, FN),
      names_to = "conf_type",
      values_to = "count"
    )

  p_conf <- ggplot(conf_plot_df, aes(x = frac, y = count, fill = method)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68) +
    facet_wrap(~ conf_type, scales = "free_y", nrow = 2) +
    labs(
      title = paste0(effect_name, ": confusion counts (imputed vs zeroed)"),
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

  imp_good_df <- plot_metrics_df %>%
    dplyr::filter(method %in% c("zeroed", "imputed")) %>%
    dplyr::select(frac, method, TP, FP, recall, F1) %>%
    tidyr::pivot_longer(
      cols = c(TP, FP, recall, F1),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(
      metric = factor(metric, levels = c("TP", "FP", "recall", "F1"))
    )

  p_imp_good <- ggplot(imp_good_df, aes(x = frac, y = value, fill = method)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68) +
    facet_wrap(~ metric, scales = "free_y", nrow = 2) +
    labs(
      title = paste0(effect_name, ": TP, FP, recall and F1 (imputed vs zeroed)"),
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

  invisible(list(
    metrics_df = metrics_df,
    plot_metrics_df = plot_metrics_df,
    compare_df_all = compare_df_all,
    group_res_all = group_res_all,
    confusion_df_all = confusion_df_all
  ))
}

# =========================
# 3. Scan the effect directories
# =========================
effect_dirs <- list.dirs(root_dir, recursive = FALSE, full.names = TRUE)
effect_dirs <- effect_dirs[grepl("^group_effect_", basename(effect_dirs))]
effect_dirs <- effect_dirs[order(parse_effect_value(basename(effect_dirs)), basename(effect_dirs))]
expected_effect_dirs <- basename(effect_dirs)

if (length(effect_dirs) == 0) {
  stop(
    "no group_effect_* directory found under root_dir: ",
    root_dir
  )
}

# =========================
# 4. Global containers for the cross-effect summary
# =========================
all_effects_metrics <- list()
all_effects_compare <- list()
all_effects_group_res <- list()
all_effects_confusion <- list()

# =========================
# 5. Iterate over each effect
# =========================
for (effect_dir in effect_dirs) {
  effect_name <- basename(effect_dir)
  message("============================")
  message("processing effect: ", effect_name)

  # truth/group paths for this effect
  paths <- resolve_truth_paths(effect_dir = effect_dir, root_dir = root_dir)
  group_df <- prepare_group_df(paths$group_path)
  truth_df <- prepare_truth_df(paths$truth_path)

  # Output directory for this effect
  output_dir <- file.path(effect_dir, per_effect_output_folder)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # Scan frac_*
  frac_dirs <- list.dirs(effect_dir, recursive = FALSE, full.names = TRUE)
  frac_dirs <- frac_dirs[basename(frac_dirs) %in% frac_levels]
  frac_dirs <- frac_dirs[order(match(basename(frac_dirs), frac_levels))]

  if (length(frac_dirs) == 0) {
    warning("no frac_0p1...frac_0p9 under this effect directory; skipping: ", effect_dir)
    next
  }

  effect_metrics <- list()
  effect_compare <- list()
  effect_group_res <- list()
  effect_confusion <- list()

  for (frac_dir in frac_dirs) {
    frac_name <- basename(frac_dir)
    message("  processing: ", frac_name)

    nc_files <- list.files(frac_dir, pattern = "\\.nc$", full.names = TRUE)
    nc_files <- nc_files[vapply(nc_files, is_target_nc_file, logical(1))]
    if (length(nc_files) == 0) {
      warning("no .nc file in the directory; skipping: ", frac_dir)
      next
    }

    for (nc_path in nc_files) {
      one_res <- run_one_nc_file_maaslin2(
        nc_path = nc_path,
        group_df = group_df,
        truth_df = truth_df,
        effect_name = effect_name,
        frac_name = frac_name,
        output_dir = output_dir,
        alpha = alpha
      )

      key <- paste(effect_name, frac_name, basename(nc_path), sep = "__")
      effect_metrics[[key]] <- one_res$metrics
      effect_compare[[key]] <- one_res$compare
      effect_group_res[[key]] <- one_res$group_res
      effect_confusion[[key]] <- one_res$confusion_long

      all_effects_metrics[[key]] <- one_res$metrics
      all_effects_compare[[key]] <- one_res$compare
      all_effects_group_res[[key]] <- one_res$group_res
      all_effects_confusion[[key]] <- one_res$confusion_long
    }
  }

  effect_metrics_df <- bind_or_empty(effect_metrics)
  effect_compare_df_all <- bind_or_empty(effect_compare)
  effect_group_res_all <- bind_or_empty(effect_group_res)
  effect_confusion_df_all <- bind_or_empty(effect_confusion)

  save_effect_level_outputs(
    effect_name = effect_name,
    metrics_df = effect_metrics_df,
    compare_df_all = effect_compare_df_all,
    group_res_all = effect_group_res_all,
    confusion_df_all = effect_confusion_df_all,
    output_dir = output_dir,
    frac_levels = frac_levels
  )
}

# =========================
# 6. Cross-effect summary
# =========================
all_effects_output_dir <- file.path(root_dir, all_effects_output_folder)
dir.create(all_effects_output_dir, recursive = TRUE, showWarnings = FALSE)

all_metrics_df <- bind_or_empty(all_effects_metrics)
all_compare_df <- bind_or_empty(all_effects_compare)
all_group_res_df <- bind_or_empty(all_effects_group_res)
all_confusion_df <- bind_or_empty(all_effects_confusion)

if (nrow(all_metrics_df) == 0) {
  stop("no metrics were produced after processing all effect directories.")
}

all_metrics_df <- all_metrics_df %>%
  mutate(
    effect = factor(effect, levels = expected_effect_dirs),
    frac = factor(frac, levels = frac_levels),
    method = factor(method, levels = c("zeroed", "imputed", "unknown"))
  ) %>%
  arrange(effect, frac, method, file)

all_compare_df <- all_compare_df %>%
  mutate(
    effect = factor(effect, levels = expected_effect_dirs),
    frac = factor(frac, levels = frac_levels),
    method = factor(method, levels = c("zeroed", "imputed", "unknown"))
  ) %>%
  arrange(effect, frac, method, file, taxon)

all_group_res_df <- all_group_res_df %>%
  mutate(
    effect = factor(effect, levels = expected_effect_dirs),
    frac = factor(frac, levels = frac_levels),
    method = factor(method, levels = c("zeroed", "imputed", "unknown"))
  ) %>%
  arrange(effect, frac, method, file)

all_confusion_df <- all_confusion_df %>%
  mutate(
    effect = factor(effect, levels = expected_effect_dirs),
    frac = factor(frac, levels = frac_levels),
    method = factor(method, levels = c("zeroed", "imputed", "unknown"))
  ) %>%
  arrange(effect, frac, method, file, truth, pred)

# Combined summary file
write_csv(
  all_metrics_df,
  file.path(all_effects_output_dir, "maaslin2_metrics_summary_all_effects.csv")
)

# Also store the detailed cross-effect table for later checking
write_csv(
  all_compare_df,
  file.path(all_effects_output_dir, "maaslin2_taxon_truth_vs_prediction_all_effects.csv")
)
write_csv(
  all_group_res_df,
  file.path(all_effects_output_dir, "maaslin2_group_differential_taxa_all_effects.csv")
)
write_csv(
  all_confusion_df,
  file.path(all_effects_output_dir, "maaslin2_confusion_matrix_long_all_effects.csv")
)

# Cross-effect mean version, for a quick overview
all_effects_plot_metrics <- all_metrics_df %>%
  group_by(effect, frac, method) %>%
  summarise(
    TP = mean_or_na(TP),
    FP = mean_or_na(FP),
    TN = mean_or_na(TN),
    FN = mean_or_na(FN),
    accuracy = mean_or_na(accuracy),
    precision = mean_or_na(precision),
    recall = mean_or_na(recall),
    specificity = mean_or_na(specificity),
    F1 = mean_or_na(F1),
    FDR = mean_or_na(FDR),
    n_files = dplyr::n(),
    .groups = "drop"
  ) %>%
  arrange(effect, frac, method)

write_csv(
  all_effects_plot_metrics,
  file.path(all_effects_output_dir, "maaslin2_plot_metrics_summary_all_effects.csv")
)

print(all_metrics_df)
message("all done.")
message("per-effect output directory: <effect_dir>/", per_effect_output_folder)
message("cross-effect summary directory: ", all_effects_output_dir)
