#!/usr/bin/env Rscript
# =============================================================================
# realdata_glmmPQL_ZIGMM_runner.R
#
# Adds glmmPQL and ZIGMM to the DIABIMMUNE / ECAM real-data differential
# analysis so all four DA methods (LinDA, MaAsLin2, glmmPQL, ZIGMM) are run on
# identical inputs.
#
# Inputs are taken from the tables the LinDA/MaAsLin2 pipeline already wrote:
#   <artifacts>/<dataset>_<task>_sample_metadata.csv   (rows = samples used)
#   <artifacts>/<dataset>_<task>_feature_map.csv       (columns = taxa tested)
# The abundance matrix is rebuilt from the exported tensors by indexing
# [participant_idx, taxon_index, bin_index], so every method sees exactly the
# same cells LinDA saw.
#
# Model specifications mirror DA_glmmPQL_effect_sweep.R / DA_ZIGMM_effect_sweep.R,
# with subject_group in place of the simulation's X and bin_c (continuous,
# already centred by the upstream pipeline) in place of the simulation's factor
# bin -- matching the real-data LinDA formula
# "~ subject_group + bin_c + (1 | participant)".
#
# Run:
#   "C:/Program Files/R/R-4.5.1/bin/Rscript.exe" realdata_glmmPQL_ZIGMM_runner.R --dataset ECAM
#   "C:/Program Files/R/R-4.5.1/bin/Rscript.exe" realdata_glmmPQL_ZIGMM_runner.R --dataset DIABIMMUNE
#   ... --engine glmmpql|zigmm|both   --tasks <regex>   --force
# =============================================================================

suppressPackageStartupMessages({
  library(ncdf4)
  library(MASS)
  library(nlme)
  library(NBZIMM)
})

# ---------------------------------------------------------------------------
# 0. Arguments
# ---------------------------------------------------------------------------
parse_args <- function() {
  raw <- commandArgs(trailingOnly = TRUE)
  out <- list()
  i <- 1
  while (i <= length(raw)) {
    a <- raw[[i]]
    if (grepl("^--", a)) {
      key <- sub("^--", "", a)
      if (grepl("=", key)) {
        kv <- strsplit(key, "=", fixed = TRUE)[[1]]
        out[[kv[[1]]]] <- paste(kv[-1], collapse = "=")
      } else if (i < length(raw) && !grepl("^--", raw[[i + 1]])) {
        out[[key]] <- raw[[i + 1]]; i <- i + 1
      } else {
        out[[key]] <- "true"
      }
    }
    i <- i + 1
  }
  out
}
ARGS <- parse_args()
arg <- function(name, default = NULL) if (!is.null(ARGS[[name]])) ARGS[[name]] else default

DATASET <- toupper(arg("dataset", "ECAM"))
ENGINE  <- tolower(arg("engine", "both"))
TASK_RE <- arg("tasks", NULL)
FORCE   <- identical(tolower(arg("force", "false")), "true")

ROOT <- tryCatch({
  sa <- commandArgs(FALSE)
  f <- grep("^--file=", sa, value = TRUE)
  if (length(f)) dirname(normalizePath(sub("^--file=", "", f[[1]]))) else getwd()
}, error = function(e) getwd())

# Per-dataset layout: where the tensors live, where the LinDA tables live,
# and the netCDF variable names for the two data conditions.
CFG <- list(
  ECAM = list(
    artifacts   = file.path(ROOT, "ECAM_data", "artifacts", "ECAM_otu_linda_longitudinal"),
    tensor_dir  = file.path(ROOT, "ECAM_data", "ECAM_otu_tensor_export"),
    tensors     = c(base_raw        = "tensor_base_raw.nc",
                    p2d2pca_imputed = "tensor_p2d2pca_imputed.nc"),
    varnames    = c(base_raw        = "abundance_binned_log",
                    p2d2pca_imputed = "abundance_binned_log_p2d2pca_imputed"),
    bin_index   = "tensor_bin_index.csv",
    n_taxa      = 480
  ),
  DIABIMMUNE = list(
    artifacts   = file.path(ROOT, "DIABIMMUNE_data", "artifacts", "subject_group_existing_methods_r"),
    tensor_dir  = file.path(ROOT, "DIABIMMUNE_data", "diabimmune_p2d2pca_netcdf_export", "tensors"),
    tensors     = c(base_raw        = "tensor_base_raw.nc",
                    p2d2pca_imputed = "tensor_p2d2pca_imputed.nc"),
    varnames    = c(base_raw        = "log10_abundance",
                    p2d2pca_imputed = "log10_abundance_imputed"),
    bin_index   = NA_character_,
    n_taxa      = 839
  )
)
if (!DATASET %in% names(CFG)) stop("--dataset must be ECAM or DIABIMMUNE")
cfg <- CFG[[DATASET]]
OUT <- file.path(cfg$artifacts, "glmmPQL_ZIGMM")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUT, "per_task"), showWarnings = FALSE, recursive = TRUE)

# The exported tensors store log10(value + EPS); the floor is log10(EPS).
# Both datasets floor at -3, i.e. EPS = 1e-3. ZIGMM needs true numeric zeros,
# so it inverts this transform (see backtransform_log10()).
EPS <- 1e-3
ALPHA <- 0.05

# Optional: treat back-transformed abundances below this value as zero before
# ZIGMM. Default 0 (off) -- thresholding manufactures the zeros the imputation
# removed, so it is a sensitivity knob only, never the headline analysis.
ZIGMM_ZERO_THRESH <- as.numeric(arg("zigmm-zero-threshold", "0"))

message(sprintf("[%s] artifacts: %s", DATASET, cfg$artifacts))
message(sprintf("[%s] tensors  : %s", DATASET, cfg$tensor_dir))

# ---------------------------------------------------------------------------
# 1. Tensor loading (cached; one read per data condition)
# ---------------------------------------------------------------------------
.tensor_cache <- new.env(parent = emptyenv())

load_tensor <- function(condition) {
  if (!is.null(.tensor_cache[[condition]])) return(.tensor_cache[[condition]])
  path <- file.path(cfg$tensor_dir, cfg$tensors[[condition]])
  if (!file.exists(path)) stop("Tensor not found: ", path)
  nc <- nc_open(path)
  on.exit(nc_close(nc), add = TRUE)
  vn <- cfg$varnames[[condition]]
  if (!vn %in% names(nc$var)) {
    stop("Variable '", vn, "' not in ", path, ". Available: ",
         paste(names(nc$var), collapse = ", "))
  }
  arr <- ncvar_get(nc, vn)
  message(sprintf("  loaded %s [%s] dims=%s", basename(path), vn,
                  paste(dim(arr), collapse = " x ")))
  .tensor_cache[[condition]] <- arr
  arr
}

# ncvar_get returns dimensions in the reverse of the xarray/Python order, so the
# axis roles cannot be hardcoded. Resolve them from the known extents instead:
# the taxon axis has n_taxa entries, and of the remaining two the participant
# axis is the one matching max(participant_idx).
resolve_axes <- function(dims, n_taxa, n_participants, n_bins, tag) {
  if (length(dims) != 3L) stop(tag, ": expected a 3-D tensor, got ", length(dims), "-D")
  tax_ax <- which(dims == n_taxa)
  if (!length(tax_ax)) {
    stop(tag, ": no tensor axis of length ", n_taxa, " (dims ",
         paste(dims, collapse = "x"), ")")
  }
  best <- NULL
  for (ta in tax_ax) {
    rest <- setdiff(seq_len(3L), ta)
    for (pa in rest) {
      ba <- setdiff(rest, pa)
      if (dims[pa] >= n_participants && dims[ba] >= n_bins) {
        cand <- list(taxon = ta, participant = pa, bin = ba,
                     exact = (dims[pa] == n_participants && dims[ba] == n_bins))
        if (is.null(best) || (cand$exact && !best$exact)) best <- cand
      }
    }
  }
  if (is.null(best)) {
    stop(tag, ": cannot map axes; dims=", paste(dims, collapse = "x"),
         " need taxa=", n_taxa, " participants>=", n_participants, " bins>=", n_bins)
  }
  best
}

# ---------------------------------------------------------------------------
# 2. Task discovery
# ---------------------------------------------------------------------------
discover_tasks <- function() {
  files <- list.files(cfg$artifacts, pattern = "_sample_metadata\\.csv$", full.names = FALSE)
  stems <- sub("_sample_metadata\\.csv$", "", files)
  conds <- names(cfg$tensors)
  rows <- list()
  for (s in stems) {
    hit <- conds[vapply(conds, function(c) startsWith(s, paste0(c, "_")), logical(1))]
    if (!length(hit)) next
    # longest matching prefix wins (base_raw vs p2d2pca_imputed never overlap,
    # but keep this robust to future condition names)
    cond <- hit[[which.max(nchar(hit))]]
    task <- sub(paste0("^", cond, "_"), "", s)
    rows[[length(rows) + 1]] <- data.frame(condition = cond, task = task,
                                          stem = s, stringsAsFactors = FALSE)
  }
  tasks <- do.call(rbind, rows)
  if (is.null(tasks) || !nrow(tasks)) stop("No *_sample_metadata.csv found in ", cfg$artifacts)
  if (!is.null(TASK_RE)) tasks <- tasks[grepl(TASK_RE, tasks$task), , drop = FALSE]
  # Only keep tasks present for BOTH conditions, so every comparison is paired.
  keep <- names(which(table(tasks$task) == length(conds)))
  dropped <- setdiff(unique(tasks$task), keep)
  if (length(dropped)) {
    message("  skipping unpaired tasks: ", paste(dropped, collapse = ", "))
  }
  tasks <- tasks[tasks$task %in% keep, , drop = FALSE]
  tasks[order(tasks$task, tasks$condition), , drop = FALSE]
}

# ---------------------------------------------------------------------------
# 3. Build the sample x taxon matrix for one (condition, task)
# ---------------------------------------------------------------------------
build_tables <- function(condition, task) {
  stem <- paste0(condition, "_", task)
  meta <- read.csv(file.path(cfg$artifacts, paste0(stem, "_sample_metadata.csv")),
                   stringsAsFactors = FALSE, check.names = FALSE)
  fmap <- read.csv(file.path(cfg$artifacts, paste0(stem, "_feature_map.csv")),
                   stringsAsFactors = FALSE, check.names = FALSE)
  for (nm in c("participant_idx", "bin", "subject_group", "bin_c", "participant")) {
    if (!nm %in% names(meta)) stop(stem, ": sample_metadata missing column '", nm, "'")
  }

  arr <- load_tensor(condition)
  dims <- dim(arr)

  # taxon_index is 0-based in the exports; fall back to feature_map row order.
  if ("taxon_index" %in% names(fmap)) {
    tax_i <- as.integer(fmap$taxon_index) + 1L
  } else {
    tax_i <- seq_len(nrow(fmap))
  }

  # bin -> bin_index. ECAM ships an explicit map; DIABIMMUNE bins are the
  # 0-based index already.
  bin_vals <- suppressWarnings(as.numeric(meta$bin))
  if (!is.na(cfg$bin_index) && file.exists(file.path(cfg$tensor_dir, cfg$bin_index))) {
    bidx <- read.csv(file.path(cfg$tensor_dir, cfg$bin_index), stringsAsFactors = FALSE)
    m <- match(round(bin_vals, 6), round(as.numeric(bidx$bin), 6))
    if (any(is.na(m))) stop(stem, ": ", sum(is.na(m)), " bin values not in ", cfg$bin_index)
    bin_i <- as.integer(bidx$bin_index[m]) + 1L
  } else {
    bin_i <- as.integer(round(bin_vals)) + 1L
  }
  par_i <- as.integer(meta$participant_idx)
  if (any(is.na(bin_i)) || any(is.na(par_i)) || any(is.na(tax_i))) {
    stop(stem, ": non-finite participant/bin/taxon index")
  }

  ax <- resolve_axes(dims, n_taxa = nrow(fmap),
                     n_participants = max(par_i), n_bins = max(bin_i), tag = stem)
  if (max(tax_i) > dims[ax$taxon]) stop(stem, ": taxon_index out of tensor range")
  if (max(bin_i) > dims[ax$bin]) stop(stem, ": bin index out of tensor range")
  if (max(par_i) > dims[ax$participant]) stop(stem, ": participant_idx out of tensor range")

  # Vectorised gather: one linear index per (sample, taxon) cell. Strides follow
  # the resolved axis roles so this is correct for either dimension order.
  stride <- cumprod(c(1L, dims[-3L]))          # stride per axis, 1-based
  ns <- nrow(meta); nt <- length(tax_i)
  pj <- rep(par_i, times = nt)
  bj <- rep(bin_i, times = nt)
  tj <- rep(tax_i, each = ns)
  lin <- 1L + (pj - 1L) * stride[ax$participant] +
              (tj - 1L) * stride[ax$taxon] +
              (bj - 1L) * stride[ax$bin]
  Y <- matrix(arr[lin], nrow = ns, ncol = nt)
  rownames(Y) <- if ("sample_id" %in% names(meta)) meta$sample_id else paste0("S", seq_len(ns))
  colnames(Y) <- fmap$feature_id

  meta$subject_group <- factor(meta$subject_group)
  if ("reference_level" %in% names(meta) && !all(is.na(meta$reference_level))) {
    ref <- meta$reference_level[[1]]
    if (ref %in% levels(meta$subject_group)) meta$subject_group <- relevel(meta$subject_group, ref)
  }
  meta$participant <- factor(meta$participant)
  meta$bin_c <- as.numeric(meta$bin_c)
  rownames(meta) <- rownames(Y)

  list(Y = Y, meta = meta, fmap = fmap)
}

# Invert log10(value + EPS) so encoded zeros become exact numeric zeros.
# Mirrors backtransform_log10_with_zero_recovery() in DA_ZIGMM_effect_sweep.R.
backtransform_log10 <- function(x, eps = EPS, tol = 1e-10) {
  out <- as.numeric(x)
  floor_val <- log10(eps)
  out[!is.finite(out)] <- NA_real_
  zero_idx <- !is.na(out) & (abs(out - floor_val) <= tol | out < floor_val)
  pos_idx  <- !is.na(out) & !zero_idx
  out[zero_idx] <- 0
  out[pos_idx]  <- (10^out[pos_idx]) - eps
  out[is.na(out)] <- 0
  out[out < 0] <- 0
  out[!is.finite(out)] <- 0
  out
}

# ---------------------------------------------------------------------------
# 4. glmmPQL: one LMM per taxon on the log10 response
#    y ~ subject_group + bin_c, random = ~ 1 | participant, gaussian(identity)
# ---------------------------------------------------------------------------
run_glmmpql <- function(tab, condition, task) {
  Y <- tab$Y; meta <- tab$meta
  grp_levels <- levels(meta$subject_group)
  coef_name <- paste0("subject_group", grp_levels[[2]])

  est <- se <- pval <- rep(NA_real_, ncol(Y))
  dfr <- rep(NA_real_, ncol(Y))
  status <- rep("ok", ncol(Y))

  for (j in seq_len(ncol(Y))) {
    y <- Y[, j]
    d <- data.frame(y = y, subject_group = meta$subject_group,
                    bin_c = meta$bin_c, participant = meta$participant)
    d <- d[is.finite(d$y), , drop = FALSE]
    # Need variation in the response and both groups still represented.
    if (nrow(d) < 10 || stats::sd(d$y) < 1e-12 || nlevels(droplevels(d$subject_group)) < 2) {
      status[j] <- "skipped_no_variation"; next
    }
    fit <- tryCatch(
      suppressWarnings(suppressMessages(
        MASS::glmmPQL(y ~ subject_group + bin_c,
                      random = ~ 1 | participant,
                      family = gaussian(link = "identity"),
                      data = d, niter = 10, verbose = FALSE)
      )),
      error = function(e) e
    )
    if (inherits(fit, "error")) { status[j] <- paste0("fit_error: ", conditionMessage(fit)); next }
    tt <- tryCatch(summary(fit)$tTable, error = function(e) NULL)
    if (is.null(tt) || !coef_name %in% rownames(tt)) { status[j] <- "no_coef"; next }
    est[j]  <- tt[coef_name, "Value"]
    se[j]   <- tt[coef_name, "Std.Error"]
    pval[j] <- tt[coef_name, "p-value"]
    dfr[j]  <- tt[coef_name, "DF"]
  }

  data.frame(
    feature_id = colnames(Y),
    effect_size = est, std_error = se, p_value = pval, df = dfr,
    q_value = p.adjust(pval, method = "BH"),
    fit_status = status,
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# 5. ZIGMM: NBZIMM::mms(method = "zig") over all taxa at once
#    fixed = ~ subject_group + bin_c, random = ~ 1 | participant
# ---------------------------------------------------------------------------
run_zigmm <- function(tab, condition, task) {
  Y <- tab$Y; meta <- tab$meta
  grp_levels <- levels(meta$subject_group)
  coef_name <- paste0("subject_group", grp_levels[[2]])

  # Back to non-negative abundance scale so encoded zeros are real zeros.
  Yb <- apply(Y, 2, backtransform_log10)
  Yb <- as.matrix(Yb)
  dimnames(Yb) <- dimnames(Y)
  if (ZIGMM_ZERO_THRESH > 0) Yb[Yb < ZIGMM_ZERO_THRESH] <- 0

  # ZIGMM models a zero-inflated response, so it is only defined when zeros
  # exist. The imputed tensor has none (every cell is filled with a positive
  # reconstruction), which makes the zero-inflation component degenerate. Detect
  # that up front and report it rather than letting mms() fail per taxon with an
  # opaque subscript error. In that limit the correct model IS the plain
  # Gaussian LMM, i.e. the glmmPQL result reported alongside.
  zero_frac <- mean(Yb == 0, na.rm = TRUE)
  n_zero_free <- sum(colSums(Yb == 0, na.rm = TRUE) == 0)
  message(sprintf("    ZIGMM input: zero fraction=%.4f, zero-free taxa=%d/%d",
                  zero_frac, n_zero_free, ncol(Yb)))
  if (n_zero_free == ncol(Yb)) {
    return(data.frame(
      feature_id = colnames(Y), effect_size = NA_real_, std_error = NA_real_,
      p_value = NA_real_, df = NA_real_, q_value = NA_real_,
      fit_status = "not_applicable_no_zeros",
      stringsAsFactors = FALSE))
  }

  dat <- data.frame(subject_group = meta$subject_group,
                    bin_c = meta$bin_c,
                    participant = meta$participant)
  dat$N <- rowSums(Yb, na.rm = TRUE)
  dat$N[!is.finite(dat$N) | dat$N <= 0] <- 1e-12

  ydf <- as.data.frame(Yb)

  fit <- tryCatch(
    suppressWarnings(suppressMessages(
      NBZIMM::mms(y = ydf,
                  fixed = ~ subject_group + bin_c,
                  random = ~ 1 | participant,
                  data = dat,
                  method = "zig",
                  zi_fixed = ~ 1,
                  zi_random = NULL,
                  niter = 30, epsilon = 1e-5,
                  min.p = 0, verbose = FALSE)
    )),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    message("    ZIGMM failed: ", conditionMessage(fit))
    return(data.frame(feature_id = colnames(Y), effect_size = NA_real_,
                      std_error = NA_real_, p_value = NA_real_, df = NA_real_,
                      q_value = NA_real_,
                      fit_status = paste0("mms_error: ", conditionMessage(fit)),
                      stringsAsFactors = FALSE))
  }

  fx <- tryCatch(NBZIMM::fixed(fit), error = function(e) NULL)
  tblz <- if (is.list(fx) && "dist" %in% names(fx)) fx$dist else if (is.data.frame(fx)) fx else NULL
  if (is.null(tblz)) {
    return(data.frame(feature_id = colnames(Y), effect_size = NA_real_,
                      std_error = NA_real_, p_value = NA_real_, df = NA_real_,
                      q_value = NA_real_, fit_status = "unparsed_fixed",
                      stringsAsFactors = FALSE))
  }
  sel <- tblz[as.character(tblz$variables) == coef_name, , drop = FALSE]
  idx <- match(colnames(Y), as.character(sel$responses))

  est <- as.numeric(sel$Estimate)[idx]
  sev <- as.numeric(sel$Std.Error)[idx]
  pv  <- as.numeric(sel$pvalue)[idx]

  data.frame(
    feature_id = colnames(Y),
    effect_size = est, std_error = sev, p_value = pv, df = NA_real_,
    q_value = p.adjust(pv, method = "BH"),
    fit_status = ifelse(is.na(idx), "not_fitted", "ok"),
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# 6. Driver
# ---------------------------------------------------------------------------
FORMULAS <- c(
  glmmPQL = "y ~ subject_group + bin_c; random = ~ 1 | participant; family = gaussian(identity)",
  ZIGMM   = "NBZIMM mms(method=zig) | fixed: ~ subject_group + bin_c | random: ~ 1 | participant | zi_fixed: ~ 1"
)

engines <- switch(ENGINE,
                  both = c("glmmPQL", "ZIGMM"),
                  glmmpql = "glmmPQL",
                  zigmm = "ZIGMM",
                  stop("--engine must be glmmpql, zigmm, or both"))

tasks <- discover_tasks()
message(sprintf("[%s] %d (condition, task) runs x %d engines",
                DATASET, nrow(tasks), length(engines)))

all_rows <- list()
for (eng in engines) {
  for (k in seq_len(nrow(tasks))) {
    cond <- tasks$condition[[k]]; task <- tasks$task[[k]]
    out_csv <- file.path(OUT, "per_task",
                         sprintf("%s_%s_%s.csv", eng, cond, task))
    if (file.exists(out_csv) && !FORCE) {
      message(sprintf("  [skip] %s | %s | %s (exists)", eng, cond, task))
      all_rows[[length(all_rows) + 1]] <- read.csv(out_csv, stringsAsFactors = FALSE)
      next
    }
    t0 <- Sys.time()
    message(sprintf("  [run ] %s | %s | %s", eng, cond, task))
    tab <- tryCatch(build_tables(cond, task), error = function(e) e)
    if (inherits(tab, "error")) {
      message("    build_tables failed: ", conditionMessage(tab)); next
    }
    res <- tryCatch(
      if (eng == "glmmPQL") run_glmmpql(tab, cond, task) else run_zigmm(tab, cond, task),
      error = function(e) e
    )
    if (inherits(res, "error")) {
      message("    engine failed: ", conditionMessage(res)); next
    }

    meta <- tab$meta
    first_group <- tapply(as.character(meta$subject_group), meta$participant,
                          function(x) x[[1]])
    gc_txt <- paste(names(table(first_group)), as.integer(table(first_group)),
                    sep = "=", collapse = ", ")
    keep_cols <- intersect(c("feature_id", "taxon", "taxon_index", "Kingdom", "Phylum",
                             "Class", "Order", "Family", "Genus", "Species",
                             "species_name", "taxon_label", "lineage"),
                           names(tab$fmap))
    res <- merge(tab$fmap[, keep_cols, drop = FALSE], res, by = "feature_id", all.y = TRUE)

    res$da_method       <- eng
    res$dataset         <- cond
    res$study           <- DATASET
    res$group_mode      <- task
    res$reference_level <- levels(meta$subject_group)[[1]]
    res$contrast_level  <- levels(meta$subject_group)[[2]]
    res$n_samples       <- nrow(meta)
    res$n_participants  <- nlevels(droplevels(meta$participant))
    res$group_counts    <- gc_txt
    res$formula         <- FORMULAS[[eng]]

    write.csv(res, out_csv, row.names = FALSE)
    all_rows[[length(all_rows) + 1]] <- res
    message(sprintf("    done in %.1fs | fitted %d/%d | q<0.10: %d",
                    as.numeric(difftime(Sys.time(), t0, units = "secs")),
                    sum(res$fit_status == "ok"), nrow(res),
                    sum(res$q_value < 0.10, na.rm = TRUE)))
  }
}

if (length(all_rows)) {
  combined <- do.call(rbind, lapply(all_rows, function(d) {
    d[, union(names(all_rows[[1]]), names(d)), drop = FALSE]
  }))
  out_path <- file.path(OUT, sprintf("%s_glmmPQL_ZIGMM_longitudinal_results.csv", DATASET))
  write.csv(combined, out_path, row.names = FALSE)
  message("\nWrote: ", out_path, "  (", nrow(combined), " rows)")
} else {
  message("\nNo results produced.")
}
