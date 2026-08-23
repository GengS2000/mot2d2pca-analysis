clean_missing <- function(x) {
  x <- as.character(x)
  x[trimws(tolower(x)) %in% c("", "na", "nan", "not provided", "not applicable", "restricted access")] <- NA
  x
}

local_lib <- file.path(getwd(), ".rlib")
if (dir.exists(local_lib)) .libPaths(c(local_lib, .libPaths()))

make_safe_name <- function(x) {
  x <- gsub("[^A-Za-z0-9_]+", "_", x)
  x <- gsub("_+", "_", x)
  gsub("^_|_$", "", x)
}

require_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(
      "Required R package '", pkg, "' is not installed or not on .libPaths(): ",
      paste(.libPaths(), collapse = "; "),
      call. = FALSE
    )
  }
}

`%||%` <- function(a, b) {
  if (is.null(a) || length(a) == 0 || (length(a) == 1 && is.na(a))) b else a
}

parse_args <- function(args) {
  out <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) stop("Unexpected argument: ", key)
    name <- substring(key, 3)
    if (i == length(args) || startsWith(args[[i + 1]], "--")) {
      out[[name]] <- TRUE
      i <- i + 1
    } else {
      out[[name]] <- args[[i + 1]]
      i <- i + 2
    }
  }
  out
}

get_arg <- function(args, name, default) {
  if (!is.null(args[[name]])) args[[name]] else default
}

split_arg <- function(x) {
  trimws(strsplit(x, ",", fixed = TRUE)[[1]])
}

read_delimited_auto <- function(path, required_cols = character()) {
  if (!file.exists(path)) stop("File does not exist: ", path)
  first <- readLines(path, n = 1, warn = FALSE)
  sep <- if (length(first) && lengths(regmatches(first, gregexpr("\t", first, fixed = TRUE))) >
             lengths(regmatches(first, gregexpr(",", first, fixed = TRUE)))) "\t" else ","
  dat <- if (identical(sep, "\t")) {
    read.delim(
      path,
      sep = "\t",
      header = TRUE,
      quote = "",
      comment.char = "",
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  } else {
    read.csv(
      path,
      header = TRUE,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  }
  missing <- setdiff(required_cols, names(dat))
  if (length(missing)) stop(path, " is missing columns: ", paste(missing, collapse = ", "))
  dat
}

first_existing_col <- function(dat, candidates) {
  hits <- candidates[candidates %in% names(dat)]
  if (length(hits)) hits[[1]] else NA_character_
}

standardize_metadata_subject_id <- function(meta, subject_col = NULL) {
  if (is.null(subject_col) || !nzchar(subject_col)) {
    subject_col <- first_existing_col(meta, c(
      "subjectid", "host_subject_id", "participant_id", "participant",
      "subject_id", "SampleID"
    ))
  }
  if (is.na(subject_col) || !subject_col %in% names(meta)) {
    stop("Metadata file must contain a subject ID column. Tried subjectid, host_subject_id, participant_id, participant, subject_id.")
  }
  meta$subjectid <- clean_missing(meta[[subject_col]])
  attr(meta, "subject_col") <- subject_col
  meta
}

normalize_participant_order <- function(order, n_participants, participant_ids = NULL) {
  if (!"participant_idx" %in% names(order)) {
    if ("participant_index" %in% names(order)) {
      order$participant_idx <- suppressWarnings(as.integer(order$participant_index)) + 1L
    } else {
      stop("Participant order must contain participant_idx or participant_index.")
    }
  } else {
    order$participant_idx <- suppressWarnings(as.integer(order$participant_idx))
    finite_idx <- order$participant_idx[is.finite(order$participant_idx)]
    if (length(finite_idx) && min(finite_idx) == 0L && max(finite_idx) == n_participants - 1L) {
      order$participant_idx <- order$participant_idx + 1L
    }
  }

  if (!"participant_id" %in% names(order)) {
    if ("participant" %in% names(order)) {
      order$participant_id <- order$participant
    } else {
      stop("Participant order must contain participant_id or participant.")
    }
  }
  if (!"tensor_participant_index" %in% names(order)) {
    order$tensor_participant_index <- order$participant_idx - 1L
  }

  order$participant_id <- clean_missing(order$participant_id)
  order$tensor_participant_index <- suppressWarnings(as.integer(order$tensor_participant_index))

  if (any(is.na(order$participant_idx)) || any(is.na(order$participant_id))) {
    stop("Participant order contains NA participant_idx or participant_id.")
  }
  if (anyDuplicated(order$participant_idx) || anyDuplicated(order$participant_id)) {
    stop("Participant order must have unique participant_idx and participant_id.")
  }
  if (!identical(sort(order$participant_idx), seq_len(n_participants))) {
    stop("Participant order participant_idx must cover tensor R indices 1:", n_participants, ".")
  }
  if (!is.null(participant_ids)) {
    missing_ids <- setdiff(order$participant_id, participant_ids)
    if (length(missing_ids)) {
      stop("Participant order contains participant_id not found in metadata/groups: ", paste(missing_ids, collapse = ", "))
    }
  }
  order[order(order$participant_idx), c("participant_idx", "tensor_participant_index", "participant_id")]
}

read_participant_index_order <- function(path, n_participants, participant_ids = NULL) {
  order <- read_delimited_auto(path)
  normalize_participant_order(order, n_participants, participant_ids = participant_ids)
}

taxonomy_rank_cols <- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")

normalize_taxonomy_table <- function(dat) {
  if (!"taxon" %in% names(dat)) {
    first_name <- names(dat)[[1]]
    if (!nzchar(first_name) || first_name %in% c("X", "X.1", "Feature.ID", "FeatureID", "OTU", "otu")) {
      names(dat)[[1]] <- "taxon"
    }
  }
  if (!"taxon" %in% names(dat)) stop("Taxonomy table must contain a taxon column or an unnamed first taxon-ID column.")
  dat$taxon <- clean_missing(dat$taxon)
  dat <- dat[!is.na(dat$taxon), , drop = FALSE]
  dat <- dat[!duplicated(dat$taxon), , drop = FALSE]
  for (col in taxonomy_rank_cols) {
    if (!col %in% names(dat)) dat[[col]] <- NA_character_
    dat[[col]] <- clean_missing(dat[[col]])
  }
  dat
}

make_species_name <- function(genus, species, fallback = NA_character_) {
  genus <- clean_missing(genus)
  species <- clean_missing(species)
  fallback <- clean_missing(fallback)
  if (length(fallback) == 1L && length(genus) > 1L) fallback <- rep(fallback, length(genus))
  out <- mapply(function(g, s, f) {
    if (!is.na(g) && !is.na(s)) {
      if (grepl(paste0("^", g, "\\b"), s)) s else paste(g, s)
    } else if (!is.na(g)) {
      g
    } else {
      f
    }
  }, genus, species, fallback, USE.NAMES = FALSE)
  clean_missing(out)
}

read_taxon_annotations <- function(taxon_index_path = "", taxonomy_path = "", taxa = NULL) {
  ann <- NULL
  if (!is.null(taxon_index_path) && nzchar(taxon_index_path) && file.exists(taxon_index_path)) {
    ann <- normalize_taxonomy_table(read_delimited_auto(taxon_index_path))
  }
  if (!is.null(taxonomy_path) && nzchar(taxonomy_path) && file.exists(taxonomy_path)) {
    tax <- normalize_taxonomy_table(read_delimited_auto(taxonomy_path))
    if (is.null(ann)) {
      ann <- tax
    } else {
      idx <- match(ann$taxon, tax$taxon)
      for (col in taxonomy_rank_cols) {
        fill <- is.na(ann[[col]]) & !is.na(idx)
        ann[[col]][fill] <- tax[[col]][idx[fill]]
      }
    }
  }
  if (is.null(ann)) {
    if (is.null(taxa)) return(NULL)
    ann <- data.frame(taxon = as.character(taxa), stringsAsFactors = FALSE)
    for (col in taxonomy_rank_cols) ann[[col]] <- NA_character_
  }
  ann$species_name <- make_species_name(ann$Genus, ann$Species, ann$taxon)
  ann$taxon_label <- ifelse(!is.na(ann$species_name), ann$species_name, ann$taxon)
  ann$lineage <- apply(ann[, taxonomy_rank_cols, drop = FALSE], 1, function(x) {
    vals <- clean_missing(x)
    vals <- vals[!is.na(vals)]
    paste(vals, collapse = "; ")
  })
  if (!is.null(taxa)) {
    taxa <- as.character(taxa)
    out <- data.frame(taxon = taxa, stringsAsFactors = FALSE)
    idx <- match(taxa, ann$taxon)
    for (col in setdiff(names(ann), "taxon")) out[[col]] <- ann[[col]][idx]
    out$taxon_label[is.na(out$taxon_label)] <- out$taxon[is.na(out$taxon_label)]
    out
  } else {
    ann
  }
}

attach_taxon_annotations <- function(tensor, taxon_index_path = "", taxonomy_path = "") {
  taxa <- as.character(tensor$taxon)
  if (!is.null(taxon_index_path) && nzchar(taxon_index_path) && file.exists(taxon_index_path)) {
    index_ann <- normalize_taxonomy_table(read_delimited_auto(taxon_index_path))
    if (length(taxa) == nrow(index_ann)) {
      sequential <- identical(taxa, as.character(seq_along(taxa)))
      no_overlap <- sum(taxa %in% index_ann$taxon) == 0L
      if (sequential || no_overlap) {
        taxa <- as.character(index_ann$taxon)
        tensor$taxon <- taxa
      }
    }
  }
  tensor$taxon_annotations <- read_taxon_annotations(taxon_index_path, taxonomy_path, taxa = taxa)
  tensor
}

decode_char_matrix <- function(x, expected_len = NULL) {
  if (is.character(x) && is.null(dim(x))) return(trimws(x, which = "right"))
  if (is.raw(x) && is.null(dim(x))) return(rawToChar(x[x != as.raw(0)]))

  d <- dim(x)
  if (is.null(d) || length(d) != 2) stop("Cannot decode NetCDF string coordinate.")

  collapse_one <- function(v) {
    if (is.raw(v)) rawToChar(v[v != as.raw(0)]) else paste0(v[!is.na(v)], collapse = "")
  }

  if (!is.null(expected_len) && d[[1]] == expected_len) {
    vals <- apply(x, 1, collapse_one)
  } else if (!is.null(expected_len) && d[[2]] == expected_len) {
    vals <- apply(x, 2, collapse_one)
  } else if (d[[1]] <= d[[2]]) {
    vals <- apply(x, 2, collapse_one)
  } else {
    vals <- apply(x, 1, collapse_one)
  }
  trimws(vals, which = "right")
}

read_string_coord <- function(nc, coord_name) {
  if (is.null(nc$dim[[coord_name]])) stop("NetCDF file is missing dimension '", coord_name, "'.")
  expected_len <- nc$dim[[coord_name]]$len
  if (!coord_name %in% names(nc$var)) return(as.character(seq_len(expected_len)))
  decode_char_matrix(ncdf4::ncvar_get(nc, coord_name), expected_len)
}

detect_data_var <- function(nc, preferred_var = NULL) {
  if (!is.null(preferred_var) && nzchar(preferred_var)) {
    if (!preferred_var %in% names(nc$var)) stop("Variable '", preferred_var, "' is not present.")
    return(preferred_var)
  }

  candidates <- names(nc$var)
  candidates <- candidates[!candidates %in% c("participant", "taxon", "bin")]
  for (v in candidates) {
    dims <- vapply(nc$var[[v]]$dim, function(d) d$name, character(1))
    if (identical(sort(dims), sort(c("participant", "taxon", "bin")))) return(v)
  }
  stop("Could not detect a participant x taxon x bin data variable.")
}

read_tensor_nc <- function(path, preferred_var = NULL) {
  require_pkg("ncdf4")
  nc <- ncdf4::nc_open(path)
  on.exit(ncdf4::nc_close(nc), add = TRUE)

  data_var <- detect_data_var(nc, preferred_var)
  dims <- vapply(nc$var[[data_var]]$dim, function(d) d$name, character(1))
  perm <- match(c("participant", "taxon", "bin"), dims)
  if (any(is.na(perm))) stop("Variable '", data_var, "' must have participant, taxon, bin dimensions.")

  arr <- ncdf4::ncvar_get(nc, data_var)
  if (!identical(perm, seq_along(perm))) arr <- aperm(arr, perm)

  eps_attr <- try(ncdf4::ncatt_get(nc, data_var, "eps")$value, silent = TRUE)
  eps <- if (inherits(eps_attr, "try-error") || is.null(eps_attr) || is.na(eps_attr)) 0.001 else as.numeric(eps_attr)

  list(
    path = path,
    variable = data_var,
    values = arr,
    participant = read_string_coord(nc, "participant"),
    taxon = read_string_coord(nc, "taxon"),
    bin = as.numeric(ncdf4::ncvar_get(nc, "bin")),
    eps = eps
  )
}

read_metadata <- function(path, subject_col = NULL) {
  meta <- read_delimited_auto(path)
  standardize_metadata_subject_id(meta, subject_col = subject_col)
}

resolve_diet_cols <- function(meta, diet_cols_arg) {
  if (identical(diet_cols_arg, "ALL_DIETARY")) {
    cols <- grep(
      "^(dietary_compound_.*_start_age_month|solid_start_approx_age_months|diet|diet_2|diet_2_month|diet_3)$",
      names(meta),
      value = TRUE
    )
  } else {
    cols <- split_arg(diet_cols_arg)
  }
  missing <- setdiff(cols, names(meta))
  if (length(missing)) stop("Metadata is missing requested diet columns: ", paste(missing, collapse = ", "))
  if (!length(cols)) stop("No diet columns found.")
  cols
}

subject_numeric_diet_values <- function(meta, participant_ids, diet_col, min_subjects = 8) {
  raw <- clean_missing(meta[[diet_col]])
  value <- suppressWarnings(as.numeric(raw))
  rows <- data.frame(participant = meta$subjectid, value = value, stringsAsFactors = FALSE)
  rows <- rows[!is.na(rows$participant) & rows$participant %in% participant_ids, , drop = FALSE]

  split_rows <- split(rows, rows$participant)
  out <- lapply(names(split_rows), function(id) {
    x <- split_rows[[id]]$value
    x <- x[is.finite(x)]
    data.frame(
      participant = id,
      diet_start_age_month = if (length(x)) stats::median(x) else NA_real_,
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, out)
  out <- out[is.finite(out$diet_start_age_month), , drop = FALSE]

  if (nrow(out) < min_subjects) stop("Only ", nrow(out), " subjects have usable numeric values.")
  s <- stats::sd(out$diet_start_age_month)
  if (!is.finite(s) || s <= 0) stop("Diet values have no variance across subjects.")

  out$diet_start_age_z <- as.numeric(scale(out$diet_start_age_month))
  out
}

subject_diet_median_split <- function(meta, participant_ids, diet_col, min_group_n = 4) {
  values <- subject_numeric_diet_values(meta, participant_ids, diet_col, min_subjects = 2 * min_group_n)
  threshold <- stats::median(values$diet_start_age_month, na.rm = TRUE)
  values$diet_group <- ifelse(values$diet_start_age_month <= threshold, "early_or_equal_median", "late")
  values$diet_group <- factor(values$diet_group, levels = c("early_or_equal_median", "late"))
  values$diet_group_binary <- ifelse(values$diet_group == "late", 1, 0)

  counts <- table(values$diet_group)
  if (any(counts < min_group_n)) {
    stop("Diet median split has too few subjects: ", paste(names(counts), as.integer(counts), sep = "=", collapse = ", "))
  }
  attr(values, "median_threshold") <- threshold
  values
}

clean_numeric <- function(x) {
  suppressWarnings(as.numeric(clean_missing(x)))
}

parse_event_flag <- function(x,
                             positive_values = c("true", "t", "yes", "y", "1", "antibiotics", "antibiotic", "abx"),
                             negative_values = c("false", "f", "no", "n", "0", "no antibiotics", "none")) {
  raw <- clean_missing(x)
  vals <- tolower(trimws(as.character(raw)))
  out <- rep(NA, length(vals))
  out[!is.na(vals) & vals %in% positive_values] <- TRUE
  out[!is.na(vals) & vals %in% negative_values] <- FALSE

  numeric_vals <- suppressWarnings(as.numeric(vals))
  numeric_idx <- is.finite(numeric_vals)
  out[numeric_idx] <- numeric_vals[numeric_idx] > 0
  out
}

check_subject_group_counts <- function(subject_meta, group_col = "subject_group", min_group_n = 4,
                                       group_name = "subject group") {
  counts <- table(subject_meta[[group_col]])
  if (length(counts) < 2 || sum(counts > 0) < 2) {
    stop(
      "Only one usable ", group_name, " level: ",
      paste(names(counts), as.integer(counts), sep = "=", collapse = ", ")
    )
  }
  if (any(counts < min_group_n)) {
    stop(
      group_name, " has too few subjects: ",
      paste(names(counts), as.integer(counts), sep = "=", collapse = ", ")
    )
  }
  invisible(counts)
}

subject_antibiotic_flags <- function(meta, participant_ids) {
  if (!"subjectid" %in% names(meta)) stop("Metadata file must contain a 'subjectid' column.")

  flag <- rep(FALSE, nrow(meta))
  observed <- rep(FALSE, nrow(meta))
  days <- rep(NA_real_, nrow(meta))

  if ("abx_ever" %in% names(meta)) {
    abx_ever <- parse_event_flag(meta$abx_ever)
    flag <- flag | (!is.na(abx_ever) & abx_ever)
    observed <- observed | !is.na(abx_ever)
  }

  if ("abx_month" %in% names(meta)) {
    abx_month <- parse_event_flag(meta$abx_month)
    flag <- flag | (!is.na(abx_month) & abx_month)
    observed <- observed | !is.na(abx_month)
  }

  if ("number_days_abx" %in% names(meta)) {
    days <- clean_numeric(meta$number_days_abx)
    flag <- flag | (is.finite(days) & days > 0)
    observed <- observed | is.finite(days)
  }

  if (!any(observed)) stop("Metadata does not contain usable antibiotic exposure columns.")

  rows <- data.frame(
    participant = clean_missing(meta$subjectid),
    antibiotic_flag = flag,
    antibiotic_observed = observed,
    number_days_abx = days,
    stringsAsFactors = FALSE
  )
  rows <- rows[!is.na(rows$participant) & rows$participant %in% participant_ids, , drop = FALSE]
  split_rows <- split(rows, rows$participant)

  out <- lapply(names(split_rows), function(id) {
    x <- split_rows[[id]]
    days_x <- x$number_days_abx[is.finite(x$number_days_abx)]
    data.frame(
      participant = id,
      antibiotic_use = any(x$antibiotic_flag, na.rm = TRUE),
      antibiotic_observed_rows = sum(x$antibiotic_observed, na.rm = TRUE),
      antibiotic_rows = sum(x$antibiotic_flag, na.rm = TRUE),
      number_days_abx_total = if (length(days_x)) sum(days_x) else NA_real_,
      number_days_abx_max = if (length(days_x)) max(days_x) else NA_real_,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

subject_antibiotic_use_group <- function(meta, participant_ids, min_group_n = 4) {
  out <- subject_antibiotic_flags(meta, participant_ids)
  out$subject_group <- ifelse(out$antibiotic_use, "antibiotic_use", "no_antibiotic_use")
  out$subject_group <- factor(out$subject_group, levels = c("no_antibiotic_use", "antibiotic_use"))
  out$subject_group_binary <- ifelse(out$subject_group == "antibiotic_use", 1, 0)
  check_subject_group_counts(out, min_group_n = min_group_n, group_name = "antibiotic_use group")
  attr(out, "group_mode") <- "antibiotic_use"
  attr(out, "group_reference") <- "no_antibiotic_use"
  out
}

subject_diet_change_flags <- function(meta, participant_ids, diet_cols,
                                      min_changed_diet_cols = 1,
                                      month_col = "month") {
  if (!"subjectid" %in% names(meta)) stop("Metadata file must contain a 'subjectid' column.")
  if (!month_col %in% names(meta)) stop("Metadata file must contain a '", month_col, "' column for diet-change grouping.")
  if (!length(diet_cols)) stop("No diet columns were supplied for diet-change grouping.")

  missing <- setdiff(diet_cols, names(meta))
  if (length(missing)) stop("Metadata is missing requested diet columns: ", paste(missing, collapse = ", "))

  meta$subjectid <- clean_missing(meta$subjectid)
  rows <- meta[!is.na(meta$subjectid) & meta$subjectid %in% participant_ids, , drop = FALSE]
  split_rows <- split(rows, rows$subjectid)

  out <- lapply(names(split_rows), function(id) {
    x <- split_rows[[id]]
    sample_months <- clean_numeric(x[[month_col]])
    sample_months <- sample_months[is.finite(sample_months)]

    diet_start_months <- vapply(diet_cols, function(col) {
      vals <- clean_numeric(x[[col]])
      vals <- vals[is.finite(vals)]
      if (length(vals)) stats::median(vals) else NA_real_
    }, numeric(1))

    has_window <- length(sample_months) > 0
    sample_min <- if (has_window) min(sample_months) else NA_real_
    sample_max <- if (has_window) max(sample_months) else NA_real_
    has_diet_value <- is.finite(diet_start_months)
    in_observed_window <- has_diet_value & has_window &
      diet_start_months >= sample_min & diet_start_months <= sample_max

    diet_change_n <- sum(in_observed_window, na.rm = TRUE)
    diet_value_n <- sum(has_diet_value, na.rm = TRUE)

    data.frame(
      participant = id,
      diet_change = if (diet_value_n > 0 && has_window) diet_change_n >= min_changed_diet_cols else NA,
      diet_change_n = diet_change_n,
      diet_value_n = diet_value_n,
      sample_min_month = sample_min,
      sample_max_month = sample_max,
      min_diet_start_month = if (diet_value_n > 0) min(diet_start_months[has_diet_value]) else NA_real_,
      max_diet_start_month = if (diet_value_n > 0) max(diet_start_months[has_diet_value]) else NA_real_,
      stringsAsFactors = FALSE
    )
  })

  out <- do.call(rbind, out)
  out[!is.na(out$diet_change), , drop = FALSE]
}

subject_diet_change_group <- function(meta, participant_ids, diet_cols,
                                      min_group_n = 4,
                                      min_changed_diet_cols = 1) {
  out <- subject_diet_change_flags(meta, participant_ids, diet_cols, min_changed_diet_cols)
  out$subject_group <- ifelse(out$diet_change, "diet_change", "no_observed_diet_change")
  out$subject_group <- factor(out$subject_group, levels = c("no_observed_diet_change", "diet_change"))
  out$subject_group_binary <- ifelse(out$subject_group == "diet_change", 1, 0)
  check_subject_group_counts(out, min_group_n = min_group_n, group_name = "diet_change group")
  attr(out, "group_mode") <- "diet_change"
  attr(out, "group_reference") <- "no_observed_diet_change"
  out
}

subject_exposure_type_group <- function(meta, participant_ids, diet_cols,
                                        min_group_n = 4,
                                        min_changed_diet_cols = 1) {
  abx <- subject_antibiotic_flags(meta, participant_ids)
  diet <- subject_diet_change_flags(meta, participant_ids, diet_cols, min_changed_diet_cols)
  out <- merge(abx, diet, by = "participant", all = FALSE)

  out$subject_group <- ifelse(
    out$antibiotic_use,
    "antibiotic_use",
    ifelse(out$diet_change, "diet_change", NA)
  )
  out <- out[!is.na(out$subject_group), , drop = FALSE]
  out$subject_group <- factor(out$subject_group, levels = c("diet_change", "antibiotic_use"))
  out$subject_group_binary <- ifelse(out$subject_group == "antibiotic_use", 1, 0)

  check_subject_group_counts(out, min_group_n = min_group_n, group_name = "exposure_type group")
  attr(out, "group_mode") <- "exposure_type"
  attr(out, "group_reference") <- "diet_change"
  out
}

build_subject_group_metadata <- function(meta, participant_ids, group_mode = "exposure_type",
                                         diet_cols = NULL, min_group_n = 4,
                                         min_changed_diet_cols = 1) {
  mode <- tolower(gsub("-", "_", group_mode))
  if (is.null(diet_cols) && mode %in% c("diet_change", "dietary_change", "exposure_type", "exposure")) {
    diet_cols <- resolve_diet_cols(meta, "ALL_DIETARY")
  }

  if (mode %in% c("abx", "abx_ever", "antibiotic", "antibiotics", "antibiotic_use")) {
    return(subject_antibiotic_use_group(meta, participant_ids, min_group_n = min_group_n))
  }
  if (mode %in% c("diet", "diet_change", "dietary_change")) {
    return(subject_diet_change_group(
      meta, participant_ids, diet_cols,
      min_group_n = min_group_n,
      min_changed_diet_cols = min_changed_diet_cols
    ))
  }
  if (mode %in% c("exposure", "exposure_type", "abx_vs_diet", "antibiotic_vs_diet")) {
    return(subject_exposure_type_group(
      meta, participant_ids, diet_cols,
      min_group_n = min_group_n,
      min_changed_diet_cols = min_changed_diet_cols
    ))
  }

  stop(
    "Unknown --subject_group_modes entry '", group_mode,
    "'. Use exposure_type, antibiotic_use, or diet_change."
  )
}

merge_subject_metadata <- function(left, right, by = "participant") {
  out <- merge(left, right, by = by, all = FALSE)
  if (!nrow(out)) stop("No subjects remain after merging subject metadata.")
  out
}

make_feature_ids <- function(taxa, taxon_annotations = NULL) {
  taxa <- as.character(taxa)
  feature_map <- data.frame(
    feature_id = sprintf("T%04d", seq_along(taxa)),
    taxon = taxa,
    stringsAsFactors = FALSE
  )
  if (!is.null(taxon_annotations)) {
    ann <- normalize_taxonomy_table(taxon_annotations)
    if (!"species_name" %in% names(ann)) ann$species_name <- make_species_name(ann$Genus, ann$Species, ann$taxon)
    if (!"taxon_label" %in% names(ann)) ann$taxon_label <- ifelse(!is.na(ann$species_name), ann$species_name, ann$taxon)
    if (!"lineage" %in% names(ann)) {
      ann$lineage <- apply(ann[, taxonomy_rank_cols, drop = FALSE], 1, function(x) {
        vals <- clean_missing(x)
        vals <- vals[!is.na(vals)]
        paste(vals, collapse = "; ")
      })
    }
    idx <- match(feature_map$taxon, ann$taxon)
    for (col in setdiff(names(ann), "taxon")) feature_map[[col]] <- ann[[col]][idx]
  }
  for (col in taxonomy_rank_cols) if (!col %in% names(feature_map)) feature_map[[col]] <- NA_character_
  if (!"species_name" %in% names(feature_map)) {
    feature_map$species_name <- make_species_name(feature_map$Genus, feature_map$Species, feature_map$taxon)
  }
  if (!"taxon_label" %in% names(feature_map)) {
    feature_map$taxon_label <- ifelse(!is.na(feature_map$species_name), feature_map$species_name, feature_map$taxon)
  }
  if (!"lineage" %in% names(feature_map)) {
    feature_map$lineage <- apply(feature_map[, taxonomy_rank_cols, drop = FALSE], 1, function(x) {
      vals <- clean_missing(x)
      vals <- vals[!is.na(vals)]
      paste(vals, collapse = "; ")
    })
  }
  feature_map
}

feature_metadata_for_results <- function(feature_ids, feature_map) {
  out <- data.frame(feature_id = as.character(feature_ids), stringsAsFactors = FALSE)
  if (!is.null(feature_map) && "feature_id" %in% names(feature_map)) {
    idx <- match(out$feature_id, feature_map$feature_id)
    for (col in setdiff(names(feature_map), "feature_id")) out[[col]] <- feature_map[[col]][idx]
  }
  if (!"taxon" %in% names(out)) out$taxon <- out$feature_id
  for (col in taxonomy_rank_cols) if (!col %in% names(out)) out[[col]] <- NA_character_
  if (!"species_name" %in% names(out)) out$species_name <- make_species_name(out$Genus, out$Species, out$taxon)
  if (!"taxon_label" %in% names(out)) out$taxon_label <- ifelse(!is.na(out$species_name), out$species_name, out$taxon)
  if (!"lineage" %in% names(out)) {
    out$lineage <- apply(out[, taxonomy_rank_cols, drop = FALSE], 1, function(x) {
      vals <- clean_missing(x)
      vals <- vals[!is.na(vals)]
      paste(vals, collapse = "; ")
    })
  }
  out
}

tensor_to_method_tables <- function(tensor, subject_meta, extra_cols) {
  arr <- tensor$values
  participants <- tensor$participant
  taxa <- tensor$taxon
  bins <- tensor$bin

  if (!identical(dim(arr), c(length(participants), length(taxa), length(bins)))) {
    stop("Tensor dimensions do not match participant/taxon/bin coordinates.")
  }

  keep_participant <- participants %in% subject_meta$participant
  arr <- arr[keep_participant, , , drop = FALSE]
  participants <- participants[keep_participant]

  feature_map <- make_feature_ids(taxa, tensor$taxon_annotations %||% NULL)
  n_samples <- length(participants) * length(bins)
  feature_mat <- matrix(NA_real_, nrow = n_samples, ncol = length(taxa))
  colnames(feature_mat) <- feature_map$feature_id

  sample_meta <- expand.grid(
    bin = bins,
    participant = participants,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  sample_meta <- sample_meta[, c("participant", "bin"), drop = FALSE]
  sample_meta$sample_id <- paste(sample_meta$participant, sample_meta$bin, sep = "__")
  sample_meta$bin_c <- sample_meta$bin - mean(bins, na.rm = TRUE)

  matched <- subject_meta[match(sample_meta$participant, subject_meta$participant), , drop = FALSE]
  for (col in extra_cols) sample_meta[[col]] <- matched[[col]]

  idx <- 1
  for (i in seq_along(participants)) {
    for (b in seq_along(bins)) {
      vals <- arr[i, , b]
      abund <- suppressWarnings(10^vals - tensor$eps)
      abund[!is.finite(abund) | abund < 0] <- 0
      total <- sum(abund)
      if (is.finite(total) && total > 0) feature_mat[idx, ] <- abund / total
      idx <- idx + 1
    }
  }

  keep_sample <- stats::complete.cases(sample_meta[, extra_cols, drop = FALSE]) & rowSums(feature_mat, na.rm = TRUE) > 0
  feature_mat <- feature_mat[keep_sample, , drop = FALSE]
  sample_meta <- sample_meta[keep_sample, , drop = FALSE]
  rownames(feature_mat) <- sample_meta$sample_id
  rownames(sample_meta) <- sample_meta$sample_id

  list(features = as.data.frame(feature_mat, check.names = FALSE), metadata = sample_meta, feature_map = feature_map)
}

tensor_to_method_tables_by_idx <- function(tensor, subject_meta, extra_cols) {
  arr <- tensor$values
  participants <- seq_along(tensor$participant)
  taxa <- tensor$taxon
  bins <- tensor$bin

  if (!"participant_idx" %in% names(subject_meta)) {
    stop("subject_meta must contain participant_idx.")
  }
  if (anyDuplicated(subject_meta$participant_idx)) {
    stop("subject_meta participant_idx must be unique.")
  }
  if (!identical(dim(arr), c(length(participants), length(taxa), length(bins)))) {
    stop("Tensor dimensions do not match participant/taxon/bin coordinates.")
  }

  keep_participant <- participants %in% subject_meta$participant_idx
  arr <- arr[keep_participant, , , drop = FALSE]
  participants <- participants[keep_participant]

  feature_map <- make_feature_ids(taxa, tensor$taxon_annotations %||% NULL)
  n_samples <- length(participants) * length(bins)
  feature_mat <- matrix(NA_real_, nrow = n_samples, ncol = length(taxa))
  colnames(feature_mat) <- feature_map$feature_id

  sample_meta <- expand.grid(
    bin = bins,
    participant_idx = participants,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  matched <- subject_meta[match(sample_meta$participant_idx, subject_meta$participant_idx), , drop = FALSE]
  sample_meta <- sample_meta[, c("participant_idx", "bin"), drop = FALSE]
  sample_meta$participant <- if ("participant_id" %in% names(matched)) {
    as.character(matched$participant_id)
  } else {
    as.character(sample_meta$participant_idx)
  }
  sample_meta$sample_id <- paste(sample_meta$participant, sample_meta$bin, sep = "__")
  sample_meta$bin_c <- sample_meta$bin - mean(bins, na.rm = TRUE)
  for (col in extra_cols) sample_meta[[col]] <- matched[[col]]

  idx <- 1
  for (i in seq_along(participants)) {
    for (b in seq_along(bins)) {
      vals <- arr[i, , b]
      abund <- suppressWarnings(10^vals - tensor$eps)
      abund[!is.finite(abund) | abund < 0] <- 0
      total <- sum(abund)
      if (is.finite(total) && total > 0) feature_mat[idx, ] <- abund / total
      idx <- idx + 1
    }
  }

  keep_sample <- stats::complete.cases(sample_meta[, extra_cols, drop = FALSE]) & rowSums(feature_mat, na.rm = TRUE) > 0
  feature_mat <- feature_mat[keep_sample, , drop = FALSE]
  sample_meta <- sample_meta[keep_sample, , drop = FALSE]
  rownames(feature_mat) <- sample_meta$sample_id
  rownames(sample_meta) <- sample_meta$sample_id

  list(features = as.data.frame(feature_mat, check.names = FALSE), metadata = sample_meta, feature_map = feature_map)
}

find_linda_fun <- function() {
  for (pkg in c("MicrobiomeStat", "LinDA", "linda")) {
    if (requireNamespace(pkg, quietly = TRUE)) {
      ns <- asNamespace(pkg)
      if (exists("linda", envir = ns, mode = "function")) return(get("linda", envir = ns))
    }
  }
  stop("Could not find a linda() function in MicrobiomeStat, LinDA, or linda packages.")
}

collect_data_frames <- function(x, prefix = "result") {
  out <- list()
  walk <- function(obj, path) {
    if (is.data.frame(obj)) {
      obj$source_path <- path
      out[[length(out) + 1]] <<- obj
    } else if (is.list(obj)) {
      nms <- names(obj)
      if (is.null(nms)) nms <- as.character(seq_along(obj))
      for (i in seq_along(obj)) walk(obj[[i]], paste(path, nms[[i]], sep = "."))
    }
  }
  walk(x, prefix)
  if (!length(out)) return(data.frame())
  all_cols <- unique(unlist(lapply(out, names)))
  out <- lapply(out, function(d) {
    missing <- setdiff(all_cols, names(d))
    for (m in missing) d[[m]] <- NA
    d[, all_cols, drop = FALSE]
  })
  do.call(rbind, out)
}

run_linda_engine <- function(feature_df, metadata_df, formula_text, out_dir, tag, cores = 1) {
  linda_fun <- find_linda_fun()
  args <- list(
    feature.dat = feature_df,
    meta.dat = metadata_df,
    formula = formula_text,
    feature.dat.type = "proportion",
    prev.filter = 0,
    mean.abund.filter = 0,
    max.abund.filter = 0,
    zero.handling = "pseudo-count",
    p.adj.method = "BH",
    n.cores = cores,
    verbose = TRUE
  )
  formal_names <- names(formals(linda_fun))
  args <- args[names(args) %in% formal_names]
  fit <- do.call(linda_fun, args)

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  saveRDS(fit, file.path(out_dir, paste0(tag, "_linda_fit.rds")))
  flat <- collect_data_frames(fit, "linda")
  write.csv(flat, file.path(out_dir, paste0(tag, "_linda_results.csv")), row.names = FALSE)
  invisible(flat)
}

run_maaslin2_engine <- function(feature_df, metadata_df, fixed_effects, random_effects, out_dir, tag, cores = 1) {
  require_pkg("Maaslin2")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  method_dir <- file.path(out_dir, paste0(tag, "_maaslin2"))
  dir.create(method_dir, recursive = TRUE, showWarnings = FALSE)

  metadata_df <- metadata_df[, unique(c(fixed_effects, random_effects, "sample_id")), drop = FALSE]
  rownames(metadata_df) <- rownames(feature_df)

  fit <- Maaslin2::Maaslin2(
    input_data = feature_df,
    input_metadata = metadata_df,
    output = method_dir,
    fixed_effects = fixed_effects,
    random_effects = random_effects,
    min_abundance = 0,
    min_prevalence = 0,
    min_variance = 0,
    normalization = "NONE",
    transform = "LOG",
    analysis_method = "LM",
    correction = "BH",
    standardize = FALSE,
    plot_heatmap = FALSE,
    plot_scatter = FALSE,
    cores = cores
  )

  results_path <- file.path(method_dir, "all_results.tsv")
  if (file.exists(results_path)) {
    results <- read.delim(results_path, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
    write.csv(results, file.path(out_dir, paste0(tag, "_maaslin2_results.csv")), row.names = FALSE)
  } else {
    results <- data.frame()
  }
  invisible(results)
}

run_requested_engines <- function(feature_df, metadata_df, feature_map, out_dir, tag, engine,
                                  linda_formula, maaslin_fixed_effects, maaslin_random_effects = "participant",
                                  cores = 1) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(feature_map, file.path(out_dir, paste0(tag, "_feature_map.csv")), row.names = FALSE)
  write.csv(metadata_df, file.path(out_dir, paste0(tag, "_sample_metadata.csv")), row.names = FALSE)

  engines <- if (identical(engine, "both")) c("linda", "maaslin2") else engine
  for (eng in engines) {
    if (identical(eng, "linda")) {
      message("Running LinDA for ", tag)
      run_linda_engine(feature_df, metadata_df, linda_formula, out_dir, tag, cores)
    } else if (identical(eng, "maaslin2")) {
      message("Running MaAsLin2 for ", tag)
      run_maaslin2_engine(feature_df, metadata_df, maaslin_fixed_effects, maaslin_random_effects, out_dir, tag, cores)
    } else {
      stop("--engine must be one of: linda, maaslin2, both")
    }
  }
}
