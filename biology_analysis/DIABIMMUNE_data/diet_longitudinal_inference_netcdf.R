#!/usr/bin/env Rscript

# Diet longitudinal inference with existing microbiome methods.
#
# This script keeps diet introduction age continuous.  It converts each NetCDF
# tensor into a sample-by-feature abundance table, joins subject groups from the
# target metadata file, and calls LinDA and/or MaAsLin2.
#
# Default:
#   Rscript diet_longitudinal_inference_netcdf.R
#
# Examples:
#   Rscript diet_longitudinal_inference_netcdf.R --engine linda
#   Rscript diet_longitudinal_inference_netcdf.R --engine maaslin2 --diet_cols solid_start_approx_age_months
#   Rscript diet_longitudinal_inference_netcdf.R --engine both --subject_group_modes none
#
# Main trajectory term:
#   LinDA formula term: diet_start_age_z:bin_c
#   MaAsLin2 metadata term: diet_start_age_z_by_bin_c

script_args <- commandArgs(FALSE)
script_file_arg <- grep("^--file=", script_args, value = TRUE)
script_dir <- if (length(script_file_arg)) dirname(normalizePath(sub("^--file=", "", script_file_arg[[1]]))) else getwd()
source(file.path(script_dir, "netcdf_existing_method_helpers.R"))

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  lib_path <- get_arg(args, "lib_path", "")
  if (nzchar(lib_path)) .libPaths(c(split_arg(lib_path), .libPaths()))

  base_nc <- get_arg(args, "base_nc", "diabimmune_p2d2pca_netcdf_export/tensors/tensor_base_raw.nc")
  imputed_nc <- get_arg(args, "imputed_nc", "diabimmune_p2d2pca_netcdf_export/tensors/tensor_p2d2pca_imputed.nc")
  meta_path <- get_arg(args, "meta", "11884_20190508-173103-added-month-abx.txt")
  out_dir <- get_arg(args, "out_dir", "artifacts/diet_longitudinal_existing_methods_r")
  diet_cols_arg <- get_arg(args, "diet_cols", "ALL_DIETARY")
  subject_group_modes <- split_arg(get_arg(args, "subject_group_modes", "exposure_type"))
  engine <- tolower(get_arg(args, "engine", "linda"))
  min_subjects <- as.integer(get_arg(args, "min_subjects", "8"))
  min_group_n <- as.integer(get_arg(args, "min_group_n", "4"))
  min_changed_diet_cols <- as.integer(get_arg(args, "min_changed_diet_cols", "1"))
  cores <- as.integer(get_arg(args, "cores", "1"))

  if (!engine %in% c("linda", "maaslin2", "both")) {
    stop("--engine must be one of: linda, maaslin2, both")
  }

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  message("Reading metadata: ", meta_path)
  meta <- read_metadata(meta_path)
  diet_cols <- resolve_diet_cols(meta, diet_cols_arg)

  message("Reading tensor: ", base_nc)
  base_tensor <- read_tensor_nc(base_nc)
  message("Reading tensor: ", imputed_nc)
  imputed_tensor <- read_tensor_nc(imputed_nc)
  participants <- Reduce(intersect, list(base_tensor$participant, imputed_tensor$participant))

  datasets <- list(base_raw = base_tensor, p2d2pca_imputed = imputed_tensor)
  skipped <- list()
  grouped_analysis <- !(length(subject_group_modes) == 1 && tolower(subject_group_modes[[1]]) == "none")
  subject_group_metas <- list()

  if (grouped_analysis) {
    for (group_mode in subject_group_modes) {
      message("Preparing subject group mode: ", group_mode)
      group_meta <- try(
        build_subject_group_metadata(
          meta = meta,
          participant_ids = participants,
          group_mode = group_mode,
          diet_cols = diet_cols,
          min_group_n = min_group_n,
          min_changed_diet_cols = min_changed_diet_cols
        ),
        silent = TRUE
      )
      if (inherits(group_meta, "try-error")) {
        reason <- conditionMessage(attr(group_meta, "condition"))
        warning("Skipping subject group mode ", group_mode, ": ", reason)
        skipped[[paste("subject_group", group_mode, sep = "__")]] <- data.frame(
          diet_col = NA_character_,
          subject_group_mode = group_mode,
          reason = reason,
          stringsAsFactors = FALSE
        )
        next
      }

      mode_safe <- make_safe_name(group_mode)
      write.csv(
        group_meta,
        file.path(out_dir, paste0("subject_group_", mode_safe, ".csv")),
        row.names = FALSE
      )
      subject_group_metas[[mode_safe]] <- list(mode = group_mode, data = group_meta)
    }
  } else {
    subject_group_metas[["none"]] <- list(mode = "none", data = NULL)
  }

  for (diet_col in diet_cols) {
    message("Preparing continuous diet variable: ", diet_col)
    diet_subject_meta <- try(subject_numeric_diet_values(meta, participants, diet_col, min_subjects), silent = TRUE)
    if (inherits(diet_subject_meta, "try-error")) {
      reason <- conditionMessage(attr(diet_subject_meta, "condition"))
      warning("Skipping ", diet_col, ": ", reason)
      skipped[[diet_col]] <- data.frame(
        diet_col = diet_col,
        subject_group_mode = NA_character_,
        reason = reason,
        stringsAsFactors = FALSE
      )
      next
    }

    for (group_key in names(subject_group_metas)) {
      group_info <- subject_group_metas[[group_key]]
      if (identical(group_info$mode, "none")) {
        subject_meta <- diet_subject_meta
        group_suffix <- ""
        extra_cols <- c("diet_start_age_month", "diet_start_age_z")
        linda_formula <- "~ diet_start_age_z + bin_c + diet_start_age_z:bin_c + (1 | participant)"
        maaslin_fixed_effects <- c("diet_start_age_z", "bin_c", "diet_start_age_z_by_bin_c")
      } else {
        subject_meta <- try(merge_subject_metadata(diet_subject_meta, group_info$data), silent = TRUE)
        if (inherits(subject_meta, "try-error")) {
          reason <- conditionMessage(attr(subject_meta, "condition"))
          warning("Skipping ", diet_col, " with subject group mode ", group_info$mode, ": ", reason)
          skipped[[paste(diet_col, group_info$mode, sep = "__")]] <- data.frame(
            diet_col = diet_col,
            subject_group_mode = group_info$mode,
            reason = reason,
            stringsAsFactors = FALSE
          )
          next
        }

        subject_meta$subject_group <- factor(
          subject_meta$subject_group,
          levels = levels(group_info$data$subject_group)
        )
        group_count_check <- try(
          check_subject_group_counts(
            subject_meta,
            min_group_n = min_group_n,
            group_name = paste("subject group for", diet_col)
          ),
          silent = TRUE
        )
        if (inherits(group_count_check, "try-error")) {
          reason <- conditionMessage(attr(group_count_check, "condition"))
          warning("Skipping ", diet_col, " with subject group mode ", group_info$mode, ": ", reason)
          skipped[[paste(diet_col, group_info$mode, "group_count", sep = "__")]] <- data.frame(
            diet_col = diet_col,
            subject_group_mode = group_info$mode,
            reason = reason,
            stringsAsFactors = FALSE
          )
          next
        }

        group_suffix <- paste0("_", group_key)
        extra_cols <- c("diet_start_age_month", "diet_start_age_z", "subject_group", "subject_group_binary")
        linda_formula <- paste(
          "~ subject_group + diet_start_age_z + bin_c +",
          "subject_group:bin_c + diet_start_age_z:bin_c + (1 | participant)"
        )
        maaslin_fixed_effects <- c(
          "subject_group_binary",
          "diet_start_age_z",
          "bin_c",
          "subject_group_binary_by_bin_c",
          "diet_start_age_z_by_bin_c"
        )
      }

      write.csv(
        subject_meta,
        file.path(out_dir, paste0("subject_diet_values_", make_safe_name(diet_col), group_suffix, ".csv")),
        row.names = FALSE
      )

      for (dataset_name in names(datasets)) {
        tag <- paste0(paste(dataset_name, make_safe_name(diet_col), sep = "_"), group_suffix)
        tables <- tensor_to_method_tables(
          tensor = datasets[[dataset_name]],
          subject_meta = subject_meta,
          extra_cols = extra_cols
        )

        tables$metadata$diet_start_age_z_by_bin_c <- tables$metadata$diet_start_age_z * tables$metadata$bin_c
        if (!identical(group_info$mode, "none")) {
          tables$metadata$subject_group <- factor(
            tables$metadata$subject_group,
            levels = levels(subject_meta$subject_group)
          )
          tables$metadata$subject_group_binary_by_bin_c <- tables$metadata$subject_group_binary * tables$metadata$bin_c
        }

        run_requested_engines(
          feature_df = tables$features,
          metadata_df = tables$metadata,
          feature_map = tables$feature_map,
          out_dir = out_dir,
          tag = tag,
          engine = engine,
          linda_formula = linda_formula,
          maaslin_fixed_effects = maaslin_fixed_effects,
          maaslin_random_effects = "participant",
          cores = cores
        )
      }
    }
  }

  if (length(skipped)) {
    skipped_path <- file.path(out_dir, "skipped_diet_columns.csv")
    write.csv(do.call(rbind, skipped), skipped_path, row.names = FALSE)
    message("Wrote skipped diet report: ", skipped_path)
  }

  message("Done. Outputs: ", out_dir)
}

tryCatch(
  main(),
  error = function(e) {
    message("ERROR: ", conditionMessage(e))
    quit(status = 1)
  }
)
