##############################
## 0. Install and load packages
##############################
required_pkgs <- c(
  "ncdf4", "NBZIMM", "nlme", "dplyr", "tidyr",
  "purrr", "stringr", "readr", "ggplot2", "tibble"
)

common_script <- file.path(getwd(), "DA_effect_sweep_common.R")
if (file.exists(common_script)) {
  source(common_script)
  if (exists("da_add_local_libpaths", mode = "function")) {
    da_add_local_libpaths()
  }
}

to_install <- required_pkgs[!required_pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) {
  install.packages(to_install)
}

library(ncdf4)
library(NBZIMM)
library(nlme)
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)
library(readr)
library(ggplot2)
library(tibble)

if (file.exists(common_script)) {
  source(common_script)
}

##############################
## 1. Parameters: edit only this block
##############################
# Root directory containing group_effect_0p1 / group_effect_0p2 / group_effect_0p4
root_dir <- normalizePath(file.path(getwd(), "simulation3_effect_tune"), mustWork = FALSE)

# Name of the output directory created inside each effect directory
output_dir_name <- "results_ZIGMM_compare"




# Back-transformation:
#   set 1 if the input was log10(count + 1)
#   set 0 if the input was log10(count)
# The upstream .nc stores log10(value + eps).
# For ZIGMM, default to inverting back to nonnegative abundance/proportion
# scale so encoded zeros become exact numeric zeros before mms(method = "zig").
input_is_log10 <- TRUE
zero_raw_tol <- 1e-10
eps <- 1e-2
use_abundance_scale_for_zig <- TRUE
log10_pseudocount <- 0           # Kept for compatibility; unused on the ZIG path
round_to_integer_count <- FALSE  # ZIG does not round; keep FALSE
nbzimm_method <- "zig"           

# NBZIMM parameters
alpha <- 0.05
nbzimm_niter <- 30
nbzimm_epsilon <- 1e-5

# min.p of mms: non-zero proportion filter threshold
# Set to 0 to match the LinDA script; no additional filtering
min_nonzero_prop <- 0

# Simplest zero-inflation component: intercept only
zi_fixed_formula <- ~1
zi_random_formula <- NULL

##############################
## 2. Utility functions
##############################

safe_div <- function(a, b) {
  ifelse(b == 0, NA_real_, a / b)
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

  if (dim(arr_std)[1] != length(participants)) stop("participant dimension length does not match its coordinate: ", nc_path)
  if (dim(arr_std)[2] != length(taxa))         stop("taxon dimension length does not match its coordinate: ", nc_path)
  if (dim(arr_std)[3] != length(bins))         stop("bin dimension length does not match its coordinate: ", nc_path)

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

# log10 -> count
# Matches the original script: apply 10^x only
# The upstream .nc stores log10(value + eps). For ZIGMM we first invert to
# nonnegative abundance/proportion scale so that encoded zeros become exact
# numeric zeros and nonzero entries follow 10^x - eps.
backtransform_log10_with_zero_recovery <- function(x,
                                                   eps = 1e-3,
                                                   zero_raw_tol = 1e-10) {
  out <- as.numeric(x)
  log10_eps <- log10(eps)

  out[!is.finite(out)] <- NA_real_

  zero_idx <- !is.na(out) & (
    abs(out - log10_eps) <= zero_raw_tol |
      out < log10_eps
  )
  pos_idx <- !is.na(out) & !zero_idx

  out[zero_idx] <- 0
  out[pos_idx] <- (10^out[pos_idx]) - eps
  out[is.na(out)] <- 0
  out[out < 0 & out >= -zero_raw_tol] <- 0
  out[out < 0] <- 0
  out[!is.finite(out)] <- 0

  out
}

build_nbzimm_input <- function(arr_ptb, df_pseudo_group,
                               input_is_log10 = TRUE,
                               log10_pseudocount = 1,
                               round_to_integer_count = TRUE,
                               eps = 1e-3,
                               zero_raw_tol = 1e-10,
                               use_abundance_scale_for_zig = TRUE,
                               diag_label = NULL) {
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

      vec <- arr_ptb[p, , b]
      
      # ZIG expects a continuous response on the log scale;
      # for log10(count) or log10(count+1) input, change base to log2 (matching the
      # NBZIMM tutorial's log2(count+1)) by multiplying by log2(10); no back-transform to counts.
      if (input_is_log10) {
        vec <- vec * 1   # keep log10(x) unchanged
        vec[!is.finite(vec)] <- 0
      }
      
      # The upstream .nc stores log10(value + eps). For ZIGMM we first
      # invert to nonnegative abundance/proportion scale so that encoded
      # zeros become exact numeric zeros.
      if (input_is_log10 && use_abundance_scale_for_zig) {
        vec <- backtransform_log10_with_zero_recovery(
          x = vec,
          eps = eps,
          zero_raw_tol = zero_raw_tol
        )
      }

      feature_list[[sid]] <- vec

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
  storage.mode(feature.dat) <- "numeric"
  rownames(feature.dat) <- taxa
  colnames(feature.dat) <- sample_ids

  # Defensive handling
  feature.dat[is.na(feature.dat)] <- 0
  #feature.dat[feature.dat < 0] <- 0

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

  # In the NBZIMM tutorial, N in offset(log(N)) is the library size
  meta.dat$N <- colSums(feature.dat, na.rm = TRUE)

  bad_idx <- which(!is.finite(meta.dat$N) | meta.dat$N <= 0)
  if (length(bad_idx) > 0) {
    warning(
      "library size <= 0 after reconstruction for the following samples; clamped to a small positive value for offset(log(N)): ",
      paste(meta.dat$sample[bad_idx], collapse = ", ")
    )
    meta.dat$N[bad_idx] <- 1e-12
  }

  # y for mms must be sample x taxa
  y <- t(feature.dat)
  stopifnot(nrow(y) == nrow(meta.dat))
  stopifnot(all(rownames(y) == rownames(meta.dat)))

  if (input_is_log10 && use_abundance_scale_for_zig) {
    log10_eps <- log10(eps)
    raw_zero_prop <- mean(
      (abs(arr_ptb - log10_eps) <= zero_raw_tol) | (arr_ptb < log10_eps),
      na.rm = TRUE
    )
    transformed_vals <- as.numeric(feature.dat)
    transformed_zero_prop <- mean(transformed_vals == 0, na.rm = TRUE)
    transformed_finite <- transformed_vals[is.finite(transformed_vals)]

    message(
      sprintf(
        "[ZIG input] %s raw_zero=%.4f transformed_zero=%.4f transformed[min/median/max]=[%.6g, %.6g, %.6g]",
        ifelse(is.null(diag_label), "input", diag_label),
        raw_zero_prop,
        transformed_zero_prop,
        min(transformed_finite),
        stats::median(transformed_finite),
        max(transformed_finite)
      )
    )
  }

  list(
    y = y,
    meta.dat = meta.dat,
    feature.dat = feature.dat
  )
}

fit_nbzimm_wrapper <- function(y, meta.dat,
                               niter = 30,
                               epsilon = 1e-5,
                               min_nonzero_prop = 0,
                               zi_fixed_formula = ~1,
                               zi_random_formula = NULL) {
  # Test the group effect of X after adjusting for bin
  fixed_formula <- ~ X + bin

  fit <- mms(
    y = y,
    fixed = fixed_formula,
    random = ~ 1 | participant,
    data = meta.dat,
    method = "zig",
    zi_fixed = zi_fixed_formula,
    zi_random = zi_random_formula,
    niter = niter,
    epsilon = epsilon,
    min.p = min_nonzero_prop,
    verbose = FALSE
  )

  list(
    res = fit,
    formula_used = deparse(fixed_formula)
  )
}

extract_main_effect_table_nbzimm <- function(nbzimm_res, main_var = "X", alpha = 0.05) {
  fx <- fixed(nbzimm_res)
  
  # ZIG / ZINBMM compatibility: fixed() may return list(dist=..., zero=...) or a data.frame
  if (is.list(fx) && "dist" %in% names(fx)) {
    out <- fx$dist
  } else if (is.data.frame(fx)) {
    out <- fx
  } else {
    stop("unrecognized structure returned by fixed(nbzimm_res)")
  }
  
  need_cols <- c("responses", "variables", "Estimate", "Std.Error", "pvalue", "padj")
  miss_cols <- setdiff(need_cols, colnames(out))
  if (length(miss_cols) > 0) {
    stop("the NBZIMM dist result lacks columns: ", paste(miss_cols, collapse = ", "))
  }

  out <- out %>%
    filter(variables == main_var) %>%
    transmute(
      taxon = as.character(responses),
      variable = as.character(variables),
      estimate = Estimate,
      std_error = Std.Error,
      pvalue = pvalue,
      padj = padj,
      reject = !is.na(padj) & padj < alpha
    )

  out
}

evaluate_against_truth <- function(res_df,
                                   df_taxon_effect,
                                   alpha = 0.05,
                                   method_name = NA_character_,
                                   effect_name = NA_character_,
                                   frac_name = NA_character_) {
  if (exists("da_evaluate_against_truth", mode = "function")) {
    return(da_evaluate_against_truth(
      res_df = res_df,
      df_taxon_effect = df_taxon_effect,
      alpha = alpha,
      method_name = method_name,
      effect_name = effect_name,
      frac_name = frac_name
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
    } else if ("pvalue" %in% colnames(res_eval)) {
      res_eval$padj <- p.adjust(res_eval$pvalue, method = "BH")
      res_eval$reject <- !is.na(res_eval$padj) & res_eval$padj < alpha
    } else {
      stop("the result table has neither reject nor a padj/pvalue column from which to build it")
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

add_run_metadata <- function(eval_obj,
                             nc_path,
                             method_name,
                             main_var,
                             original_dim_order,
                             formula_used,
                             effect_name,
                             effect_value,
                             frac_name) {
  original_dim_order_chr <- if (length(original_dim_order) > 0) {
    paste(original_dim_order, collapse = " x ")
  } else {
    NA_character_
  }

  res_eval <- eval_obj$res_eval %>%
    mutate(
      effect = .env$effect_name,
      effect_value = .env$effect_value,
      frac = .env$frac_name,
      method = .env$method_name,
      file = basename(.env$nc_path),
      main_var = .env$main_var,
      original_dim_order = .env$original_dim_order_chr,
      formula_used = .env$formula_used,
      model_type = "NBZIMM_ZIGMM_X_plus_bin"
    )

  metrics <- eval_obj$metrics %>%
    mutate(
      effect = .env$effect_name,
      effect_value = .env$effect_value,
      frac = .env$frac_name,
      method = .env$method_name,
      file = basename(.env$nc_path),
      main_var = .env$main_var,
      original_dim_order = .env$original_dim_order_chr,
      formula_used = .env$formula_used,
      model_type = "NBZIMM_ZIGMM_X_plus_bin"
    ) %>%
    relocate(effect, effect_value, frac, method, file, main_var, original_dim_order, formula_used, model_type)

  list(
    res_eval = res_eval,
    metrics = metrics
  )
}

make_failed_nc_result <- function(nc_path,
                                  df_taxon_effect,
                                  frac_name,
                                  effect_name,
                                  effect_value,
                                  alpha = 0.05,
                                  main_var = NA_character_,
                                  original_dim_order = character(0),
                                  fit_message = "model fitting failed") {
  method_name <- infer_method_name(nc_path, main_var)
  formula_used <- deparse(~ X + bin)

  res_df <- tibble(
    taxon = as.character(df_taxon_effect$taxon),
    variable = "X",
    estimate = NA_real_,
    std_error = NA_real_,
    pvalue = NA_real_,
    padj = NA_real_,
    reject = FALSE,
    fit_status = "error",
    fit_message = fit_message
  )

  eval_obj <- evaluate_against_truth(
    res_df = res_df,
    df_taxon_effect = df_taxon_effect,
    alpha = alpha,
    method_name = method_name,
    effect_name = effect_name,
    frac_name = frac_name
  )

  out <- add_run_metadata(
    eval_obj = eval_obj,
    nc_path = nc_path,
    method_name = method_name,
    main_var = main_var,
    original_dim_order = original_dim_order,
    formula_used = formula_used,
    effect_name = effect_name,
    effect_value = effect_value,
    frac_name = frac_name
  )

  list(
    nc_info = NULL,
    res_df = res_df,
    res_eval = out$res_eval,
    metrics = out$metrics
  )
}

run_one_nc_file <- function(nc_path,
                            df_pseudo_group,
                            df_taxon_effect,
                            frac_name,
                            effect_name,
                            effect_value,
                            input_is_log10 = TRUE,
                            log10_pseudocount = 1,
                            round_to_integer_count = TRUE,
                            eps = 1e-3,
                            zero_raw_tol = 1e-10,
                            use_abundance_scale_for_zig = TRUE,
                            niter = 30,
                            epsilon = 1e-5,
                            min_nonzero_prop = 0,
                            zi_fixed_formula = ~1,
                            zi_random_formula = NULL,
                            alpha = 0.05) {
  nc_obj <- read_nc_tensor_standard(nc_path)
  arr_ptb <- nc_obj$arr
  method_name <- infer_method_name(nc_path, nc_obj$main_var)

  nbzimm_input <- build_nbzimm_input(
    arr_ptb = arr_ptb,
    df_pseudo_group = df_pseudo_group,
    input_is_log10 = input_is_log10,
    log10_pseudocount = log10_pseudocount,
    round_to_integer_count = round_to_integer_count,
    eps = eps,
    zero_raw_tol = zero_raw_tol,
    use_abundance_scale_for_zig = use_abundance_scale_for_zig,
    diag_label = basename(nc_path)
  )

  fit_obj <- fit_nbzimm_wrapper(
    y = nbzimm_input$y,
    meta.dat = nbzimm_input$meta.dat,
    niter = niter,
    epsilon = epsilon,
    min_nonzero_prop = min_nonzero_prop,
    zi_fixed_formula = zi_fixed_formula,
    zi_random_formula = zi_random_formula
  )

  res_df <- extract_main_effect_table_nbzimm(
    nbzimm_res = fit_obj$res,
    main_var = "X",
    alpha = alpha
  )

  eval_obj <- evaluate_against_truth(
    res_df = res_df,
    df_taxon_effect = df_taxon_effect,
    alpha = alpha,
    method_name = method_name,
    effect_name = effect_name,
    frac_name = frac_name
  )

  method_name <- infer_method_name(nc_path, nc_obj$main_var)

  res_eval <- eval_obj$res_eval %>%
    mutate(
      effect = .env$effect_name,
      effect_value = .env$effect_value,
      frac = .env$frac_name,
      method = .env$method_name,
      file = basename(.env$nc_path),
      main_var = nc_obj$main_var,
      original_dim_order = paste(nc_obj$original_dim_order, collapse = " × "),
      formula_used = .env$fit_obj$formula_used,
      model_type = "NBZIMM_ZIGMM_X_plus_bin"
    )

  metrics <- eval_obj$metrics %>%
    mutate(
      effect = .env$effect_name,
      effect_value = .env$effect_value,
      frac = .env$frac_name,
      method = .env$method_name,
      file = basename(.env$nc_path),
      main_var = nc_obj$main_var,
      original_dim_order = paste(nc_obj$original_dim_order, collapse = " × "),
      formula_used = .env$fit_obj$formula_used,
      model_type = "NBZIMM_ZIGMM_X_plus_bin"
    ) %>%
    relocate(effect, effect_value, frac, method, file, main_var, original_dim_order, formula_used, model_type)

  list(
    nc_info = nc_obj,
    res_df = res_df,
    res_eval = res_eval,
    metrics = metrics
  )
}

parse_effect_value <- function(effect_name) {
  x <- str_remove(effect_name, "^group_effect_")
  x <- str_replace_all(x, "p", ".")
  suppressWarnings(as.numeric(x))
}

find_truth_files <- function(effect_dir, root_dir) {
  candidate_dirs <- c(
    file.path(effect_dir, "truth"),
    effect_dir,
    file.path(root_dir, "truth"),
    root_dir
  )
  candidate_dirs <- unique(candidate_dirs[dir.exists(candidate_dirs)])

  group_file <- NULL
  taxon_file <- NULL

  for (d in candidate_dirs) {
    g <- file.path(d, "effect_pseudo_group.csv")
    t <- file.path(d, "effect_taxon_effect.csv")

    if (is.null(group_file) && file.exists(g)) group_file <- g
    if (is.null(taxon_file) && file.exists(t)) taxon_file <- t

    if (!is.null(group_file) && !is.null(taxon_file)) break
  }

  if (is.null(group_file) || is.null(taxon_file)) {
    stop(
      "cannot find the truth files for directory: ", effect_dir,
      "\neffect_pseudo_group.csv and effect_taxon_effect.csv must both exist",
      "\nsearched locations: ", paste(candidate_dirs, collapse = " ; ")
    )
  }

  list(group_file = group_file, taxon_file = taxon_file)
}

plot_and_save_effect_results <- function(metrics_df, output_dir, effect_label) {
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

  if (nrow(metric_plot_df) == 0 || nrow(conf_plot_df) == 0 || nrow(imp_good_df) == 0) {
    warning(
      "No zeroed/imputed rows are available for plotting in ",
      effect_label,
      "; CSV outputs were still written.",
      call. = FALSE
    )
    return(invisible(FALSE))
  }

  p_metric <- ggplot(metric_plot_df, aes(x = frac, y = value, fill = method)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68) +
    facet_wrap(~ metric, scales = "free_y", nrow = 2) +
    labs(
      title = paste0("NBZIMM-ZIGMM performance: imputed vs zeroed (", effect_label, ")"),
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
      title = paste0("NBZIMM-ZIGMM confusion matrix counts: imputed vs zeroed (", effect_label, ")"),
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
      title = paste0("NBZIMM-ZIGMM TP, FP, recall and F1: imputed vs zeroed (", effect_label, ")"),
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
                                   root_dir,
                                   output_dir_name,
                                   input_is_log10,
                                   log10_pseudocount,
                                   round_to_integer_count,
                                   eps,
                                   zero_raw_tol,
                                   use_abundance_scale_for_zig,
                                   nbzimm_niter,
                                   nbzimm_epsilon,
                                   min_nonzero_prop,
                                   zi_fixed_formula,
                                   zi_random_formula,
                                   alpha) {
  effect_name <- basename(effect_dir)
  effect_value <- parse_effect_value(effect_name)
  output_dir <- file.path(effect_dir, output_dir_name)
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  truth_files <- find_truth_files(effect_dir, root_dir)

  message("========================================")
  message("processing effect directory: ", effect_name)
  message("truth(group): ", truth_files$group_file)
  message("truth(taxon): ", truth_files$taxon_file)
  message("output directory: ", output_dir)

  df_pseudo_group <- read_csv(truth_files$group_file, show_col_types = FALSE) %>%
    mutate(participant = as.character(participant))

  df_taxon_effect <- read_csv(truth_files$taxon_file, show_col_types = FALSE) %>%
    mutate(taxon = as.character(taxon))

  frac_dirs <- list.dirs(effect_dir, recursive = FALSE, full.names = TRUE)
  frac_dirs <- frac_dirs[grepl("^frac_0p[1-9]$", basename(frac_dirs))]

  if (length(frac_dirs) == 0) {
    stop("no frac_0p1 ... frac_0p9 directory found under: ", effect_dir)
  }

  frac_levels <- paste0("frac_0p", 1:9)
  frac_dirs <- frac_dirs[order(match(basename(frac_dirs), frac_levels))]

  all_metrics <- list()
  all_taxon_results <- list()

  for (frac_dir in frac_dirs) {
    frac_name <- basename(frac_dir)
    message("  processing: ", frac_name)

    nc_files <- list.files(frac_dir, pattern = "\\.nc$", full.names = TRUE)
    nc_files <- nc_files[!grepl("^mask", basename(nc_files), ignore.case = TRUE)]
    if (length(nc_files) == 0) {
      warning("no .nc file in the directory; skipping: ", frac_dir)
      next
    }

    folder_metrics <- list()
    folder_taxon_results <- list()

    for (nc_path in nc_files) {
      message("    reading file: ", basename(nc_path))

      one_res <- tryCatch(
        run_one_nc_file(
        nc_path = nc_path,
        df_pseudo_group = df_pseudo_group,
        df_taxon_effect = df_taxon_effect,
        frac_name = frac_name,
        effect_name = effect_name,
        effect_value = effect_value,
        input_is_log10 = input_is_log10,
        log10_pseudocount = log10_pseudocount,
        round_to_integer_count = round_to_integer_count,
        eps = eps,
        zero_raw_tol = zero_raw_tol,
        use_abundance_scale_for_zig = use_abundance_scale_for_zig,
        niter = nbzimm_niter,
        epsilon = nbzimm_epsilon,
        min_nonzero_prop = min_nonzero_prop,
        zi_fixed_formula = zi_fixed_formula,
        zi_random_formula = zi_random_formula,
          alpha = alpha
        ),
        error = function(e) {
          msg <- conditionMessage(e)
          warning("NBZIMM-ZIGMM failed for ", basename(nc_path), ": ", msg, call. = FALSE)
          make_failed_nc_result(
            nc_path = nc_path,
            df_taxon_effect = df_taxon_effect,
            frac_name = frac_name,
            effect_name = effect_name,
            effect_value = effect_value,
            alpha = alpha,
            fit_message = msg
          )
        }
      )

      method_name <- as.character(unique(one_res$metrics$method))
      method_name <- method_name[!is.na(method_name) & method_name != ""]
      if (length(method_name) != 1) method_name <- "unknown"

      folder_metrics[[basename(nc_path)]] <- one_res$metrics
      folder_taxon_results[[basename(nc_path)]] <- one_res$res_eval

      write_csv(
        one_res$res_eval,
        file.path(output_dir, paste0("taxon_results_", frac_name, "_", method_name, ".csv"))
      )
    }

    all_metrics[[frac_name]] <- bind_rows(folder_metrics)
    all_taxon_results[[frac_name]] <- bind_rows(folder_taxon_results)
  }

  metrics_df <- bind_rows(all_metrics) %>%
    mutate(
      frac = factor(frac, levels = frac_levels),
      method = factor(method, levels = c("zeroed", "imputed"))
    ) %>%
    arrange(frac, method)

  taxon_df <- bind_rows(all_taxon_results) %>%
    mutate(
      frac = factor(frac, levels = frac_levels),
      method = factor(method, levels = c("zeroed", "imputed"))
    ) %>%
    arrange(frac, method, taxon)

  write_csv(metrics_df, file.path(output_dir, "nbzimm_zinb_metrics_summary.csv"))
  write_csv(taxon_df, file.path(output_dir, "nbzimm_zinb_taxon_level_results_all.csv"))

  plot_and_save_effect_results(
    metrics_df = metrics_df,
    output_dir = output_dir,
    effect_label = effect_name
  )

  message("done: ", effect_name)
  print(metrics_df)

  invisible(list(
    effect_name = effect_name,
    effect_value = effect_value,
    output_dir = output_dir,
    metrics_df = metrics_df,
    taxon_df = taxon_df
  ))
}

##############################
## 3. Scan the effect directories and run them in batch
##############################
effect_dirs <- list.dirs(root_dir, recursive = FALSE, full.names = TRUE)
effect_dirs <- effect_dirs[grepl("^group_effect_", basename(effect_dirs))]

if (length(effect_dirs) == 0) {
  stop("no group_effect_0p1 / group_effect_0p2 / group_effect_0p4 directory found under root_dir")
}

effect_order <- c("group_effect_0p1", "group_effect_0p2", "group_effect_0p4")
other_effects <- setdiff(basename(effect_dirs), effect_order)
all_effect_levels <- c(effect_order, sort(other_effects))
effect_dirs <- effect_dirs[order(match(basename(effect_dirs), all_effect_levels))]

all_effect_outputs <- list()

for (effect_dir in effect_dirs) {
  effect_name <- basename(effect_dir)

  all_effect_outputs[[effect_name]] <- process_one_effect_dir(
    effect_dir = effect_dir,
    root_dir = root_dir,
    output_dir_name = output_dir_name,
    input_is_log10 = input_is_log10,
    log10_pseudocount = log10_pseudocount,
    round_to_integer_count = round_to_integer_count,
    eps = eps,
    zero_raw_tol = zero_raw_tol,
    use_abundance_scale_for_zig = use_abundance_scale_for_zig,
    nbzimm_niter = nbzimm_niter,
    nbzimm_epsilon = nbzimm_epsilon,
    min_nonzero_prop = min_nonzero_prop,
    zi_fixed_formula = zi_fixed_formula,
    zi_random_formula = zi_random_formula,
    alpha = alpha
  )
}

message("========================================")
message("all done. An output directory was created under every group_effect_*: ", output_dir_name)
