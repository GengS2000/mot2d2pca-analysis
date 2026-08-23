#!/usr/bin/env Rscript

script_args <- commandArgs(FALSE)
script_file_arg <- grep("^--file=", script_args, value = TRUE)
script_dir <- if (length(script_file_arg)) dirname(normalizePath(sub("^--file=", "", script_file_arg[[1]]))) else getwd()
source(file.path(script_dir, "netcdf_existing_method_helpers.R"))

get_arg_any <- function(args, names, default = NULL) {
  for (name in names) {
    if (!is.null(args[[name]])) return(args[[name]])
  }
  default
}

read_table_checked <- function(path) {
  read_delimited_auto(path)
}

first_values <- function(x, n = 8) {
  vals <- unique(clean_missing(x))
  vals <- vals[!is.na(vals)]
  paste(utils::head(vals, n), collapse = ";")
}

write_participant_order_template <- function(path, n_participants) {
  template <- data.frame(
    participant_idx = seq_len(n_participants),
    participant_id = rep("", n_participants),
    stringsAsFactors = FALSE
  )
  write.csv(template, path, row.names = FALSE, na = "")
  invisible(template)
}

is_numeric_like <- function(x) {
  vals <- clean_missing(x)
  vals <- vals[!is.na(vals)]
  if (!length(vals)) return(FALSE)
  nums <- suppressWarnings(as.numeric(vals))
  mean(is.finite(nums)) >= 0.8
}

parse_abx_signal <- function(x) {
  vals <- clean_missing(x)
  low <- tolower(trimws(as.character(vals)))
  out <- rep(NA, length(vals))
  positive <- c("true", "t", "yes", "y", "1", "antibiotics", "antibiotic", "abx", "exposed", "use", "used")
  negative <- c("false", "f", "no", "n", "0", "no antibiotics", "none", "unexposed", "not used")
  out[!is.na(low) & low %in% positive] <- TRUE
  out[!is.na(low) & low %in% negative] <- FALSE

  nums <- suppressWarnings(as.numeric(low))
  numeric_idx <- is.finite(nums)
  out[numeric_idx] <- nums[numeric_idx] > 0
  out
}

decode_tensor_participants <- function(tensor_path) {
  require_pkg("ncdf4")
  nc <- ncdf4::nc_open(tensor_path)
  on.exit(ncdf4::nc_close(nc), add = TRUE)

  if (is.null(nc$dim$participant)) stop("Tensor has no participant dimension: ", tensor_path)
  n_participants <- nc$dim$participant$len
  message("Tensor dimensions: ", paste(names(nc$dim), vapply(nc$dim, function(d) d$len, integer(1)), sep = "=", collapse = ", "))
  message("Tensor variables: ", paste(names(nc$var), collapse = ", "))

  ids <- NULL
  if ("participant" %in% names(nc$var)) {
    ids <- decode_char_matrix(ncdf4::ncvar_get(nc, "participant"), expected_len = n_participants)
    ids <- clean_missing(ids)
  } else if (!is.null(nc$dim$participant$vals)) {
    ids <- as.character(nc$dim$participant$vals)
    ids <- clean_missing(ids)
  }

  list(n = n_participants, ids = ids, dim_names = names(nc$dim), var_names = names(nc$var))
}

extract_reliable_tensor_ids <- function(tensor_info, metadata_ids) {
  ids <- tensor_info$ids
  if (is.null(ids) || length(ids) != tensor_info$n) return(NULL)
  if (any(is.na(ids)) || any(!nzchar(ids))) return(NULL)
  if (anyDuplicated(ids)) return(NULL)
  if (identical(ids, as.character(seq_len(tensor_info$n)))) return(NULL)
  overlap <- sum(ids %in% metadata_ids)
  if (overlap != tensor_info$n) return(NULL)
  ids
}

validate_participant_order <- function(order, n_participants, participant_ids) {
  normalize_participant_order(order, n_participants, participant_ids = participant_ids)
}

most_common_value <- function(x) {
  vals <- clean_missing(x)
  vals <- vals[!is.na(vals)]
  if (!length(vals)) return(NA_character_)
  counts <- sort(table(vals), decreasing = TRUE)
  names(counts)[[1]]
}

participant_split <- function(meta, participant_ids) {
  meta <- meta[meta$participant_id %in% participant_ids, , drop = FALSE]
  split(meta, factor(meta$participant_id, levels = participant_ids), drop = TRUE)
}

build_delivery_subject_groups <- function(split_meta, delivery_col = "delivery") {
  if (!delivery_col %in% names(split_meta[[1]])) return(NULL)
  rows <- lapply(names(split_meta), function(pid) {
    vals <- clean_missing(split_meta[[pid]][[delivery_col]])
    value <- most_common_value(vals)
    data.frame(
      participant_id = pid,
      group_mode = "delivery_group",
      diet_col = "",
      exposure_type = value %||% "missing",
      exposure_value = NA_real_,
      reference_level = "Vaginal",
      contrast_level = "Cesarean",
      rule = paste0("Subject-level most frequent ", delivery_col, "; contrast Cesarean vs Vaginal."),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

parse_abx_column_signal <- function(x, col) {
  signal <- parse_abx_signal(x)
  vals <- clean_missing(x)
  low <- tolower(trimws(as.character(vals)))
  text_positive <- !is.na(low) & !low %in% c("0", "false", "f", "no", "n", "none", "not used")
  if (grepl("name|class|pmp", col, ignore.case = TRUE)) {
    signal[is.na(signal) & text_positive] <- TRUE
  }
  signal
}

build_abx_subject_groups <- function(split_meta, abx_cols = NULL) {
  all_names <- names(split_meta[[1]])
  if (is.null(abx_cols) || !length(abx_cols)) {
    preferred <- c("antiexposedall", "abx_all_sources", "abx_name", "course_num", "abx_pmp_all")
    candidates <- unique(c(preferred, grep("abx|antibiotic|antimicrobial", all_names, ignore.case = TRUE, value = TRUE)))
    abx_cols <- candidates[candidates %in% all_names]
    abx_cols <- abx_cols[!grepl("mom|maternal|prenatal|pregnan|ld_abx", abx_cols, ignore.case = TRUE)]
  }
  abx_cols <- abx_cols[abx_cols %in% all_names]
  rows <- lapply(names(split_meta), function(pid) {
    d <- split_meta[[pid]]
    signals <- if (length(abx_cols)) {
      unlist(lapply(abx_cols, function(col) parse_abx_column_signal(d[[col]], col)), use.names = FALSE)
    } else {
      rep(NA, nrow(d))
    }
    observed <- signals[!is.na(signals)]
    exposed <- length(observed) > 0 && any(observed)
    exposure_type <- if (!length(observed)) "missing" else if (exposed) "abx_exposed" else "abx_unexposed"
    data.frame(
      participant_id = pid,
      group_mode = "abx_group",
      diet_col = "",
      exposure_type = exposure_type,
      exposure_value = if (!length(observed)) NA_real_ else as.numeric(exposed),
      reference_level = "abx_unexposed",
      contrast_level = "abx_exposed",
      rule = paste0(
        "Any positive non-maternal antibiotic signal across visits indicates exposure. Columns: ",
        paste(abx_cols, collapse = ";")
      ),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

build_categorical_diet_subject_groups <- function(split_meta, diet_cols = c("diet", "diet_2", "diet_3"),
                                                  min_group_n = 4) {
  all_names <- names(split_meta[[1]])
  diet_cols <- diet_cols[diet_cols %in% all_names]
  if (!length(diet_cols)) return(data.frame())
  out <- list()
  for (col in diet_cols) {
    vals_all <- clean_missing(unlist(lapply(split_meta, function(d) d[[col]]), use.names = FALSE))
    vals_all <- sort(unique(vals_all[!is.na(vals_all)]))
    for (value in vals_all) {
      flags <- vapply(split_meta, function(d) {
        vals <- clean_missing(d[[col]])
        any(vals == value, na.rm = TRUE)
      }, logical(1))
      counts <- c(no = sum(!flags), yes = sum(flags))
      if (any(counts < min_group_n)) next
      safe_value <- make_safe_name(value)
      rows <- data.frame(
        participant_id = names(split_meta),
        group_mode = paste("diet", make_safe_name(col), safe_value, sep = "_"),
        diet_col = col,
        exposure_type = ifelse(flags, paste0("has_", safe_value), paste0("no_", safe_value)),
        exposure_value = as.numeric(flags),
        reference_level = paste0("no_", safe_value),
        contrast_level = paste0("has_", safe_value),
        rule = paste0("Subject has at least one visit with ", col, " == '", value, "' vs no such visit."),
        stringsAsFactors = FALSE
      )
      out[[length(out) + 1]] <- rows
    }
  }
  if (!length(out)) return(data.frame())
  do.call(rbind, out)
}

group_counts_text <- function(x) {
  counts <- table(x, useNA = "ifany")
  paste(names(counts), as.integer(counts), sep = "=", collapse = ", ")
}

task_label <- function(group_mode, diet_col) {
  group_mode <- as.character(group_mode %||% "")
  diet_col <- as.character(diet_col %||% "")
  if (!nzchar(diet_col)) make_safe_name(group_mode) else paste(make_safe_name(group_mode), make_safe_name(diet_col), sep = "_")
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  metadata_path <- get_arg_any(args, c("metadata", "meta"), "otu_metadata_cleaned_q2.csv")
  tensor_path <- get_arg_any(args, c("tensor", "base_nc"), "ECAM_otu_tensor_export/tensor_base_raw.nc")
  out_dir <- get_arg_any(args, c("outdir", "out_dir"), "artifacts/ECAM_otu_linda_longitudinal")
  participant_order_path <- get_arg_any(args, c("participant-order", "participant_order"), file.path(out_dir, "participant_order.csv"))
  participant_index_path <- get_arg_any(args, c("participant-index", "participant_index"),
                                        file.path(dirname(tensor_path), "tensor_participant_index.csv"))
  subject_col <- get_arg_any(args, c("subject-col", "subject_col"), "host_subject_id")
  delivery_col <- get_arg_any(args, c("delivery-col", "delivery_col"), "delivery")
  diet_cols <- split_arg(get_arg_any(args, c("diet-cols", "diet_cols"), "diet,diet_2,diet_3"))
  abx_cols_arg <- get_arg_any(args, c("abx-cols", "abx_cols"), "")
  abx_cols <- if (nzchar(abx_cols_arg)) split_arg(abx_cols_arg) else NULL
  min_group_n <- as.integer(get_arg_any(args, c("min-group-n", "min_group_n"), "4"))
  if (!is.finite(min_group_n) || min_group_n < 1L) stop("--min-group-n must be a positive integer.")

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  message("Reading metadata: ", metadata_path)
  meta <- read_table_checked(metadata_path)
  if (anyDuplicated(names(meta))) {
    duplicated_names <- unique(names(meta)[duplicated(names(meta))])
    stop("Metadata has duplicated column names: ", paste(duplicated_names, collapse = ", "))
  }
  meta <- standardize_metadata_subject_id(meta, subject_col = subject_col)
  participant_col <- attr(meta, "subject_col")
  meta$participant_id <- meta$subjectid
  message("Metadata rows: ", nrow(meta), "; columns: ", ncol(meta), "; participant column: ", participant_col)

  id_keywords <- c("id", "sample", "subject", "participant", "host", "child", "case", "person", "individual", "patient")
  id_cols <- grep(paste(id_keywords, collapse = "|"), names(meta), ignore.case = TRUE, value = TRUE)
  id_diag <- do.call(rbind, lapply(id_cols, function(col) {
    vals <- clean_missing(meta[[col]])
    data.frame(
      col = col,
      n_total = length(vals),
      n_nonmissing = sum(!is.na(vals)),
      n_unique = length(unique(vals[!is.na(vals)])),
      first_values = first_values(vals),
      stringsAsFactors = FALSE
    )
  }))
  write.csv(id_diag, file.path(out_dir, "participant_id_diagnostics.csv"), row.names = FALSE)

  metadata_participant_ids <- sort(unique(meta$participant_id[!is.na(meta$participant_id)]))
  if (file.exists(participant_index_path)) {
    participant_index_raw <- read_delimited_auto(participant_index_path)
    n_participants <- nrow(participant_index_raw)
    participant_order <- normalize_participant_order(participant_index_raw, n_participants)
    if (file.exists(participant_order_path)) {
      participant_order <- validate_participant_order(
        read_delimited_auto(participant_order_path),
        n_participants,
        participant_order$participant_id
      )
    } else {
      write.csv(participant_order, participant_order_path, row.names = FALSE)
      message("Generated participant_order.csv from tensor participant index: ", participant_order_path)
    }
  } else {
    tensor_info <- decode_tensor_participants(tensor_path)
    n_participants <- tensor_info$n
    if (file.exists(participant_order_path)) {
      participant_order <- validate_participant_order(
        read_delimited_auto(participant_order_path),
        n_participants,
        metadata_participant_ids
      )
    } else {
      tensor_ids <- extract_reliable_tensor_ids(tensor_info, metadata_participant_ids)
      if (is.null(tensor_ids)) {
        template_path <- file.path(out_dir, "participant_order_template.csv")
        write_participant_order_template(template_path, n_participants)
        stop("Cannot reliably map tensor participant_idx to metadata participant_id. Wrote template: ", template_path)
      }
      participant_order <- data.frame(
        participant_idx = seq_len(n_participants),
        tensor_participant_index = seq_len(n_participants) - 1L,
        participant_id = tensor_ids,
        stringsAsFactors = FALSE
      )
      write.csv(participant_order, participant_order_path, row.names = FALSE)
    }
  }

  participant_ids <- participant_order$participant_id
  missing_meta <- setdiff(participant_ids, metadata_participant_ids)
  if (length(missing_meta)) {
    stop("Tensor participant IDs not found in metadata ", participant_col, ": ", paste(missing_meta, collapse = ", "))
  }
  message("Tensor participants mapped to metadata: ", length(participant_ids))

  meta_tensor <- meta[meta$participant_id %in% participant_ids, , drop = FALSE]
  split_meta <- participant_split(meta_tensor, participant_ids)
  participant_level <- data.frame(
    participant_id = names(split_meta),
    n_rows = vapply(split_meta, nrow, integer(1)),
    stringsAsFactors = FALSE
  )
  if (delivery_col %in% names(meta_tensor)) {
    participant_level$delivery <- vapply(split_meta, function(d) most_common_value(d[[delivery_col]]), character(1))
  }
  month_col <- first_existing_col(meta_tensor, c("month_of_life", "month", "day_of_life"))
  if (!is.na(month_col)) {
    participant_level$sample_min_time <- vapply(split_meta, function(d) {
      x <- clean_numeric(d[[month_col]])
      if (any(is.finite(x))) min(x[is.finite(x)]) else NA_real_
    }, numeric(1))
    participant_level$sample_max_time <- vapply(split_meta, function(d) {
      x <- clean_numeric(d[[month_col]])
      if (any(is.finite(x))) max(x[is.finite(x)]) else NA_real_
    }, numeric(1))
    participant_level$time_column <- month_col
  }

  abx_diag_cols <- grep("abx|antibiotic|antibiotics|antimicrobial", names(meta_tensor), ignore.case = TRUE, value = TRUE)
  abx_diag <- if (length(abx_diag_cols)) {
    do.call(rbind, lapply(abx_diag_cols, function(col) {
      vals <- clean_missing(meta_tensor[[col]])
      nums <- suppressWarnings(as.numeric(vals))
      finite_nums <- nums[is.finite(nums)]
      data.frame(
        col = col,
        class = paste(class(meta_tensor[[col]]), collapse = "/"),
        n_nonmissing = sum(!is.na(vals)),
        n_unique = length(unique(vals[!is.na(vals)])),
        example_values = first_values(vals),
        numeric_min = if (length(finite_nums)) min(finite_nums) else NA_real_,
        numeric_max = if (length(finite_nums)) max(finite_nums) else NA_real_,
        stringsAsFactors = FALSE
      )
    }))
  } else {
    data.frame(col = character(), class = character(), n_nonmissing = integer(),
               n_unique = integer(), example_values = character(),
               numeric_min = numeric(), numeric_max = numeric(), stringsAsFactors = FALSE)
  }
  write.csv(abx_diag, file.path(out_dir, "antibiotic_column_diagnostics.csv"), row.names = FALSE)

  delivery_groups <- build_delivery_subject_groups(split_meta, delivery_col = delivery_col)
  abx_groups <- build_abx_subject_groups(split_meta, abx_cols = abx_cols)
  diet_groups <- build_categorical_diet_subject_groups(split_meta, diet_cols = diet_cols, min_group_n = min_group_n)

  group_tables <- Filter(function(x) !is.null(x) && nrow(x) > 0, list(delivery_groups, abx_groups, diet_groups))
  if (!length(group_tables)) stop("No subject groups could be built from metadata.")
  all_cols <- unique(unlist(lapply(group_tables, names)))
  group_tables <- lapply(group_tables, function(d) {
    missing <- setdiff(all_cols, names(d))
    for (col in missing) d[[col]] <- NA
    d[, all_cols, drop = FALSE]
  })
  subject_groups <- do.call(rbind, group_tables)
  subject_groups$diet_col[is.na(subject_groups$diet_col)] <- ""
  write.csv(subject_groups, file.path(out_dir, "subject_groups_long.csv"), row.names = FALSE)

  participant_abx <- abx_groups[, c("participant_id", "exposure_type", "exposure_value", "rule"), drop = FALSE]
  names(participant_abx) <- c("participant_id", "abx_group", "abx_exposed", "abx_rule")
  write.csv(participant_abx, file.path(out_dir, "participant_abx_group.csv"), row.names = FALSE)
  write.csv(diet_groups, file.path(out_dir, "participant_diet_groups_long.csv"), row.names = FALSE)
  participant_level <- merge(participant_level, participant_abx[, c("participant_id", "abx_group")], by = "participant_id", all.x = TRUE, sort = FALSE)
  participant_level <- merge(participant_order, participant_level, by = "participant_id", all.x = TRUE, sort = FALSE)
  participant_level <- participant_level[order(participant_level$participant_idx), , drop = FALSE]
  write.csv(participant_level, file.path(out_dir, "participant_level_metadata.csv"), row.names = FALSE)

  idx_groups <- merge(subject_groups, participant_order, by = "participant_id", all.x = TRUE, sort = FALSE)
  idx_groups <- idx_groups[order(idx_groups$participant_idx, idx_groups$group_mode, idx_groups$diet_col), c(
    "participant_idx", "tensor_participant_index", "participant_id", "group_mode", "diet_col",
    "exposure_type", "exposure_value", "reference_level", "contrast_level", "rule"
  )]
  write.csv(idx_groups, file.path(out_dir, "participant_idx_subject_groups_long.csv"), row.names = FALSE)
  write.csv(idx_groups[idx_groups$group_mode == "abx_group", , drop = FALSE],
            file.path(out_dir, "participant_idx_group_map.csv"), row.names = FALSE)

  skipped <- data.frame(
    subject_group_mode = character(),
    diet_col = character(),
    reason = character(),
    stringsAsFactors = FALSE
  )
  write.csv(skipped, file.path(out_dir, "skipped_subject_groups.csv"), row.names = FALSE)

  message("Subject group counts:")
  tasks <- unique(subject_groups[, c("group_mode", "diet_col"), drop = FALSE])
  for (i in seq_len(nrow(tasks))) {
    rows <- subject_groups[subject_groups$group_mode == tasks$group_mode[[i]] &
                             subject_groups$diet_col == tasks$diet_col[[i]], , drop = FALSE]
    message("  ", task_label(tasks$group_mode[[i]], tasks$diet_col[[i]]), ": ", group_counts_text(rows$exposure_type))
  }
  message("Wrote participant-level metadata and subject group files to: ", out_dir)
}

tryCatch(
  main(),
  error = function(e) {
    message("ERROR: ", conditionMessage(e))
    quit(status = 1)
  }
)
