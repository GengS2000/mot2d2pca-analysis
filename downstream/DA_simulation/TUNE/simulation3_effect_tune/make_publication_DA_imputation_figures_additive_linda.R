# =============================================================================
# make_publication_DA_imputation_figures.R
#
# Purpose: Demonstrate that penalized_2d2_pca imputation improves differential
#          abundance (DA) inference relative to zeroed data, across four DA
#          methods (LinDA, MaAsLin2, ZIGMM, glmmPQL), multiple effect sizes,
#          and multiple masking fractions.
#
# IMPORTANT SCOPE RULES (enforced in code):
#   - ANCOMBC2 and LDM results are EXCLUDED entirely (never read, never plotted).
#   - Only method-level metrics summaries are used (NO taxon-level inference).
#   - glmmPQL `mask` rows are dropped; data_condition inferred from main_var.
#
# Run:
#   "C:/Program Files/R/R-4.5.1/bin/Rscript.exe" make_publication_DA_imputation_figures.R
#   (working directory should be .../simulation3_effect_tune)
# =============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(forcats)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(ragg)
})

# ----------------------------------------------------------------------------
# 0. Paths & global style
# ----------------------------------------------------------------------------
ROOT <- "."  # expected to be run from simulation3_effect_tune
if (!dir.exists(file.path(ROOT, "group_effect_0p5"))) {
  # allow running from TUNE/ as well
  if (dir.exists("simulation3_effect_tune")) ROOT <- "simulation3_effect_tune"
}
OUT <- file.path(ROOT, "publication_figures_imputation_DA_additive_linda")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# Colors
COL_IMPUTED <- "#1B9E89"   # muted teal / blue-green = imputed
COL_ZEROED  <- "#8C8C8C"   # neutral grey = zeroed
COND_COLORS <- c(imputed = COL_IMPUTED, zeroed = COL_ZEROED)
# Diverging palette for delta (0-centered): negative=red (imputed worse),
# positive=teal-green (imputed better)
DIV_LOW  <- "#B2182B"
DIV_MID  <- "#F7F7F7"
DIV_HIGH <- "#117A65"

DA_LEVELS <- c("LinDA", "MaAsLin2", "ZIGMM", "glmmPQL")

# Publication theme (no grey ggplot background)
theme_pub <- function(base_size = 9) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
      panel.border     = element_rect(color = "grey30", linewidth = 0.4),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      strip.background = element_rect(fill = "grey92", color = "grey30", linewidth = 0.4),
      strip.text       = element_text(size = base_size - 0.5, face = "bold"),
      axis.text        = element_text(size = base_size - 1, color = "black"),
      axis.title       = element_text(size = base_size),
      legend.key       = element_rect(fill = "white", color = NA),
      legend.title     = element_text(size = base_size - 0.5),
      legend.text      = element_text(size = base_size - 1),
      plot.title       = element_text(size = base_size + 1, face = "bold"),
      panel.spacing    = unit(0.55, "lines")
    )
}
theme_set(theme_pub())

# ----------------------------------------------------------------------------
# 1. Helpers for column standardization
# ----------------------------------------------------------------------------
# "frac_0p1" -> 0.1 ; "0p525" -> 0.525
parse_pnum <- function(x) {
  x <- as.character(x)
  x <- str_replace(x, "^frac_", "")
  x <- str_replace(x, "^group_effect_", "")
  as.numeric(str_replace(x, "p", "."))
}
# "group_effect_0p6" kept as effect_name; build canonical name from a numeric
effect_name_from_size <- function(sz) {
  paste0("group_effect_", str_replace(formatC(sz, format = "f", drop0trailing = TRUE), "\\.", "p"))
}

METRIC_COLS <- c("TP","FP","FN","TN","precision","recall","F1",
                 "specificity","accuracy","FDR","n_called_diff",
                 "n_truth_diff","n_taxa_total")

standardize <- function(df, da_method) {
  # ensure all metric cols exist
  for (m in METRIC_COLS) if (!m %in% names(df)) df[[m]] <- NA_real_
  df %>%
    transmute(
      da_method     = da_method,
      effect_name   = effect_name,
      effect_size   = effect_size,
      masking_fraction = masking_fraction,
      data_condition = data_condition,
      TP = as.numeric(TP), FP = as.numeric(FP),
      FN = as.numeric(FN), TN = as.numeric(TN),
      precision = as.numeric(precision), recall = as.numeric(recall),
      F1 = as.numeric(F1), specificity = as.numeric(specificity),
      accuracy = as.numeric(accuracy), FDR = as.numeric(FDR),
      n_called_diff = as.numeric(n_called_diff),
      n_truth_diff  = as.numeric(n_truth_diff),
      n_taxa_total  = as.numeric(n_taxa_total)
    )
}

# ----------------------------------------------------------------------------
# 2. Load each DA method (method-level summaries only)
# ----------------------------------------------------------------------------
inputs_used <- character(0)

## --- LinDA (all-effects summary) ----
linda_path <- file.path(ROOT, "results_linda_additive_compare_all_effects",
                        "linda_metrics_summary_all_effects.csv")
inputs_used <- c(inputs_used, linda_path)
linda <- read_csv(linda_path, show_col_types = FALSE) %>%
  filter(method %in% c("imputed","zeroed")) %>%
  mutate(
    effect_name = effect_name,
    effect_size = as.numeric(effect_level),
    masking_fraction = parse_pnum(frac),
    data_condition = method
  ) %>%
  standardize("LinDA")

## --- MaAsLin2 (per-file all-effects summary; plot summary used as cross-check) ----
maas_path <- file.path(ROOT, "results_maaslin2_compare_all_effects",
                       "maaslin2_metrics_summary_all_effects.csv")
inputs_used <- c(inputs_used, maas_path)
maaslin2 <- read_csv(maas_path, show_col_types = FALSE) %>%
  filter(method %in% c("imputed","zeroed")) %>%
  mutate(
    effect_name = effect,
    effect_size = parse_pnum(effect),
    masking_fraction = parse_pnum(frac),
    data_condition = method
  ) %>%
  standardize("MaAsLin2")

## --- ZIGMM / nbzimm_zinb (merge from group_effect_* subdirs; NO all-effects) ----
zigmm_files <- list.files(ROOT, pattern = "nbzimm_zinb_metrics_summary\\.csv$",
                          recursive = TRUE, full.names = TRUE)
zigmm_files <- zigmm_files[str_detect(zigmm_files, "results_ZIGMM_compare")]
inputs_used <- c(inputs_used, zigmm_files)
zigmm <- map_dfr(zigmm_files, ~read_csv(.x, show_col_types = FALSE)) %>%
  filter(method %in% c("imputed","zeroed")) %>%
  mutate(
    effect_name = effect,
    effect_size = as.numeric(effect_value),
    masking_fraction = parse_pnum(frac),
    data_condition = method
  ) %>%
  standardize("ZIGMM")

## --- glmmPQL (all-effects summary; method=NA -> infer from main_var; drop mask) ----
pql_path <- file.path(ROOT, "results_glmmPQL_compare_all_effects",
                      "glmmPQL_metrics_summary_all_effects.csv")
inputs_used <- c(inputs_used, pql_path)
glmmpql <- read_csv(pql_path, show_col_types = FALSE) %>%
  mutate(
    src = coalesce(as.character(main_var), as.character(file)),
    data_condition = case_when(
      str_detect(src, "mask")     ~ "MASK",
      str_detect(src, "imputed")  ~ "imputed",
      str_detect(src, "zeroed")   ~ "zeroed",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(data_condition %in% c("imputed","zeroed")) %>%   # drop mask + unknown
  mutate(
    effect_name = effect_name,
    effect_size = as.numeric(effect_level),
    masking_fraction = parse_pnum(frac)
  ) %>%
  standardize("glmmPQL")

# ----------------------------------------------------------------------------
# 3. Combine + safety filters
# ----------------------------------------------------------------------------
combined <- bind_rows(linda, maaslin2, zigmm, glmmpql) %>%
  mutate(da_method = factor(da_method, levels = DA_LEVELS),
         data_condition = factor(data_condition, levels = c("imputed","zeroed")))

# HARD SAFETY: drop any accidental ancombc2/LDM/taxon traces (by construction none)
stopifnot(!any(str_detect(tolower(as.character(combined$da_method)),
                          "ancombc2|ldm")))
# No method should be NA
stopifnot(all(!is.na(combined$da_method)))
stopifnot(all(!is.na(combined$data_condition)))

write_csv(combined, file.path(OUT, "combined_metrics_clean.csv"))

cat("Combined rows:", nrow(combined), "\n")
print(combined %>% count(da_method, data_condition))

# ----------------------------------------------------------------------------
# 4. Build paired (imputed - zeroed) deltas per (method, effect, frac)
# ----------------------------------------------------------------------------
metric_set <- c("F1","recall","precision","FDR","TP","FP","FN","TN",
                "specificity","accuracy","n_called_diff")

wide <- combined %>%
  select(da_method, effect_name, effect_size, masking_fraction,
         data_condition, all_of(metric_set)) %>%
  pivot_wider(names_from = data_condition,
              values_from = all_of(metric_set),
              names_glue = "{.value}_{data_condition}")

delta <- wide %>%
  mutate(
    delta_F1          = F1_imputed - F1_zeroed,
    delta_recall      = recall_imputed - recall_zeroed,
    delta_precision   = precision_imputed - precision_zeroed,
    delta_FDR         = FDR_imputed - FDR_zeroed,
    delta_TP          = TP_imputed - TP_zeroed,
    delta_FP          = FP_imputed - FP_zeroed,
    delta_FN          = FN_imputed - FN_zeroed,
    delta_specificity = specificity_imputed - specificity_zeroed,
    delta_accuracy    = accuracy_imputed - accuracy_zeroed
  )

write_csv(delta, file.path(OUT, "paired_delta_summary.csv"))

# ----------------------------------------------------------------------------
# 5. Statistical tests (paired Wilcoxon signed-rank + sign test), two-sided
# ----------------------------------------------------------------------------
safe_wilcox <- function(d) {
  d <- d[is.finite(d)]
  if (length(d[d != 0]) < 1) return(NA_real_)
  tryCatch(
    suppressWarnings(wilcox.test(d, mu = 0, exact = FALSE)$p.value),
    error = function(e) NA_real_)
}
safe_sign <- function(d) {
  d <- d[is.finite(d) & d != 0]
  n <- length(d); if (n < 1) return(NA_real_)
  pos <- sum(d > 0)
  tryCatch(binom.test(pos, n, 0.5)$p.value, error = function(e) NA_real_)
}
summarise_metric <- function(df, col) {
  d <- df[[col]]
  d_fin <- d[is.finite(d)]
  tibble(
    n_pairs        = length(d_fin),
    n_nonzero      = sum(d_fin != 0),
    median_delta   = median(d_fin),
    mean_delta     = mean(d_fin),
    prop_positive  = mean(d_fin > 0),
    p_wilcoxon     = safe_wilcox(d),
    p_sign         = safe_sign(d)
  )
}

make_tests <- function(df, group_label) {
  bind_rows(
    summarise_metric(df, "delta_F1")        %>% mutate(metric = "delta_F1"),
    summarise_metric(df, "delta_recall")    %>% mutate(metric = "delta_recall"),
    summarise_metric(df, "delta_precision") %>% mutate(metric = "delta_precision"),
    summarise_metric(df, "delta_FDR")       %>% mutate(metric = "delta_FDR")
  ) %>% mutate(da_method = group_label, .before = 1)
}

tests_by_method <- delta %>%
  group_split(da_method) %>%
  map_dfr(~make_tests(.x, as.character(.x$da_method[1])))
tests_overall <- make_tests(delta, "ALL (pooled)")

stat_tests <- bind_rows(tests_by_method, tests_overall) %>%
  select(da_method, metric, n_pairs, n_nonzero, median_delta, mean_delta,
         prop_positive, p_wilcoxon, p_sign)
write_csv(stat_tests, file.path(OUT, "statistical_tests_summary.csv"))

# Compact summary table requested in brief
summary_table <- delta %>%
  group_by(da_method) %>%
  summarise(
    n_pairs                   = sum(is.finite(delta_F1)),
    median_delta_F1           = median(delta_F1, na.rm = TRUE),
    mean_delta_F1             = mean(delta_F1, na.rm = TRUE),
    proportion_delta_F1_pos   = mean(delta_F1 > 0, na.rm = TRUE),
    median_delta_recall       = median(delta_recall, na.rm = TRUE),
    median_delta_FDR          = median(delta_FDR, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(
    stat_tests %>% filter(metric == "delta_F1") %>%
      select(da_method, p_value_delta_F1 = p_wilcoxon),
    by = "da_method") %>%
  left_join(
    stat_tests %>% filter(metric == "delta_recall") %>%
      select(da_method, p_value_delta_recall = p_wilcoxon),
    by = "da_method")
write_csv(summary_table, file.path(OUT, "compact_summary_table.csv"))
cat("\n--- Compact summary ---\n"); print(as.data.frame(summary_table))

# p-value formatting helper (exact text, used in tables / wide panels)
fmt_p <- function(p) ifelse(is.na(p), "NA",
                            ifelse(p < 0.001, "P<0.001",
                                   paste0("P=", formatC(p, format = "f", digits = 3))))
# compact significance stars for dense panels where exact P text cannot fit
fmt_star <- function(p) ifelse(is.na(p), "",
                        ifelse(p < 0.001, "***",
                        ifelse(p < 0.01,  "**",
                        ifelse(p < 0.05,  "*", "ns"))))

# ----------------------------------------------------------------------------
# 6. MAIN FIGURE (5 panels A-E)
# ----------------------------------------------------------------------------
REP_EFFECTS <- c(0.55, 0.60, 0.65, 0.70)

## Panel A: dF1 heatmap, facet per DA method, x=effect_size, y=masking_fraction
pa_dat <- delta %>%
  mutate(eff = factor(effect_size),
         frc = factor(masking_fraction))
lim_a <- max(abs(pa_dat$delta_F1), na.rm = TRUE)
pA <- ggplot(pa_dat, aes(eff, frc, fill = delta_F1)) +
  geom_tile(color = "white", linewidth = 0.3) +
  facet_wrap(~da_method, nrow = 1) +
  scale_fill_gradient2(low = DIV_LOW, mid = DIV_MID, high = DIV_HIGH,
                       midpoint = 0, limits = c(-lim_a, lim_a),
                       name = expression(Delta*F1)) +
  scale_x_discrete(name = "Effect size") +
  scale_y_discrete(name = "Masking fraction") +
  theme_pub() +
  theme(axis.text.x  = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 5.6),
        axis.text.y  = element_text(size = 6),
        panel.spacing.x = unit(0.35, "lines"),
        legend.position = "right") +
  ggtitle("A  ΔF1 = F1(imputed) − F1(zeroed)   (positive = imputed better)")

## Panel B: F1 vs masking_fraction, facet_grid(da_method ~ effect_size) for rep effects
pb_dat <- combined %>%
  filter(effect_size %in% REP_EFFECTS) %>%
  mutate(eff_lab = factor(paste0("effect ", effect_size),
                          levels = paste0("effect ", REP_EFFECTS)))
pB <- ggplot(pb_dat, aes(masking_fraction, F1, color = data_condition)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.1) +
  facet_grid(da_method ~ eff_lab) +
  scale_color_manual(values = COND_COLORS, name = NULL,
                     labels = c("imputed","zeroed")) +
  scale_x_continuous(breaks = seq(0.1, 0.9, 0.2), name = "Masking fraction") +
  scale_y_continuous(name = "F1", limits = c(0, 1)) +
  theme_pub() +
  theme(legend.position = "top") +
  ggtitle("B  F1 vs masking fraction (representative effect sizes)")

## Panel C: Recall vs FDR tradeoff, color by condition, facet by method
pc_dat <- combined %>% filter(is.finite(FDR), is.finite(recall))
pC <- ggplot(pc_dat, aes(FDR, recall, color = data_condition)) +
  geom_point(alpha = 0.55, size = 1.1) +
  facet_wrap(~da_method, nrow = 1) +
  scale_color_manual(values = COND_COLORS, name = NULL) +
  scale_x_continuous(name = "FDR", limits = c(0, 1),
                     breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1"),
                     expand = expansion(mult = 0.07)) +
  scale_y_continuous(name = "Recall", limits = c(0, 1),
                     breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1")) +
  theme_pub() +
  theme(legend.position = "top",
        panel.spacing.x = unit(0.7, "lines")) +
  ggtitle("C  Recall vs FDR trade-off (all conditions)")

## Panel D: median paired delta of TP / FP / FN per method (mechanism)
pd_dat <- delta %>%
  group_by(da_method) %>%
  summarise(`ΔTP` = median(delta_TP, na.rm = TRUE),
            `ΔFP` = median(delta_FP, na.rm = TRUE),
            `ΔFN` = median(delta_FN, na.rm = TRUE),
            .groups = "drop") %>%
  pivot_longer(-da_method, names_to = "count_type", values_to = "median_delta") %>%
  mutate(count_type = factor(count_type, levels = c("ΔTP","ΔFP","ΔFN")))
pD <- ggplot(pd_dat, aes(da_method, median_delta, fill = count_type)) +
  geom_col(position = position_dodge(0.7), width = 0.66, color = "grey20",
           linewidth = 0.25) +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  coord_flip() +
  scale_fill_manual(values = c("ΔTP" = "#117A65",
                               "ΔFP" = "#B2182B",
                               "ΔFN" = "#D49A0B"), name = NULL) +
  scale_y_continuous(name = "Median Δ count (imputed − zeroed)") +
  scale_x_discrete(name = NULL, limits = rev(DA_LEVELS)) +
  theme_pub() +
  theme(legend.position = "top",
        legend.key.size = unit(0.8, "lines")) +
  ggtitle("D  Median Δ counts")

## Panel E: paired improvement distribution for dF1, dRecall, dFDR + p-values
pe_long <- delta %>%
  select(da_method, delta_F1, delta_recall, delta_FDR) %>%
  pivot_longer(-da_method, names_to = "metric", values_to = "value") %>%
  filter(is.finite(value)) %>%
  mutate(metric = recode(metric,
                         delta_F1 = "ΔF1",
                         delta_recall = "ΔRecall",
                         delta_FDR = "ΔFDR"),
         metric = factor(metric, levels = c("ΔF1","ΔRecall","ΔFDR")))

# p-value annotation positions
pe_ann <- stat_tests %>%
  filter(da_method %in% DA_LEVELS,
         metric %in% c("delta_F1","delta_recall","delta_FDR")) %>%
  mutate(metric = recode(metric,
                         delta_F1 = "ΔF1",
                         delta_recall = "ΔRecall",
                         delta_FDR = "ΔFDR"),
         metric = factor(metric, levels = c("ΔF1","ΔRecall","ΔFDR")),
         da_method = factor(da_method, levels = DA_LEVELS),
         lab = fmt_star(p_wilcoxon))
# robust annotation height per facet: top of the data range + headroom
# (must not use value * constant, which flips sign for the negative ΔFDR facet)
yr_e <- pe_long %>% group_by(metric) %>%
  summarise(ymin = min(value, na.rm = TRUE),
            ymax = max(value, na.rm = TRUE), .groups = "drop") %>%
  mutate(y = ymax + 0.09 * (ymax - ymin))
pe_ann <- pe_ann %>% left_join(yr_e, by = "metric")

pE <- ggplot(pe_long, aes(da_method, value)) +
  geom_hline(yintercept = 0, color = "grey50", linetype = "dashed",
             linewidth = 0.4) +
  geom_violin(aes(fill = metric), color = NA, alpha = 0.25, scale = "width") +
  geom_boxplot(width = 0.16, outlier.size = 0.35, linewidth = 0.3,
               fill = "white") +
  facet_wrap(~metric, nrow = 1, scales = "free_y") +
  geom_text(data = pe_ann, aes(x = da_method, y = y, label = lab),
            size = 2.5, vjust = 0, fontface = "bold", inherit.aes = FALSE) +
  scale_fill_manual(values = c("ΔF1" = "#117A65",
                               "ΔRecall" = "#1B9E89",
                               "ΔFDR" = "#D49A0B"), guide = "none") +
  scale_x_discrete(name = NULL) +
  scale_y_continuous(name = "Paired difference (imputed − zeroed)",
                     expand = expansion(mult = c(0.05, 0.16))) +
  labs(caption = "Wilcoxon signed-rank: ***P<0.001  **P<0.01  *P<0.05  ns, not significant") +
  theme_pub() +
  theme(axis.text.x   = element_text(angle = 45, hjust = 1, size = 6.5),
        panel.spacing.x = unit(0.6, "lines"),
        plot.caption  = element_text(size = 6, hjust = 0, color = "grey30")) +
  ggtitle("E  Paired Δ per method (Wilcoxon P)")

## Assemble: A on top, B, then C, then bottom row (D narrow | E wide).
## E gets ~3x the width of D so its 3 facets, 4 method labels and Wilcoxon P
## annotations have room and no longer overlap.
bottom_row <- (pD | pE) + plot_layout(widths = c(1, 2.9))
main_fig <- (pA) / (pB) / (pC) / bottom_row +
  plot_layout(heights = c(1.0, 2.0, 1.0, 1.15))

ggsave(file.path(OUT, "publication_main_figure_DA_imputation.pdf"),
       main_fig, width = 7.4, height = 12.6, units = "in", device = cairo_pdf)
ggsave(file.path(OUT, "publication_main_figure_DA_imputation.png"),
       main_fig, width = 7.4, height = 12.6, units = "in", dpi = 600,
       device = ragg::agg_png)
cat("Main figure written.\n")

# ----------------------------------------------------------------------------
# 7. SUPPLEMENTARY FIGURE 1: full metric curves (F1, recall, precision, FDR)
#    across ALL effect sizes and masking fractions, per method.
# ----------------------------------------------------------------------------
sup_long <- combined %>%
  select(da_method, effect_size, masking_fraction, data_condition,
         F1, recall, precision, FDR) %>%
  pivot_longer(c(F1, recall, precision, FDR),
               names_to = "metric", values_to = "value") %>%
  mutate(metric = factor(metric, levels = c("F1","recall","precision","FDR")))

make_metric_curves <- function(meth) {
  d <- sup_long %>% filter(da_method == meth)
  ggplot(d, aes(masking_fraction, value, color = data_condition)) +
    geom_line(linewidth = 0.5) + geom_point(size = 0.7) +
    facet_grid(metric ~ effect_size, switch = "y") +
    scale_color_manual(values = COND_COLORS, name = NULL) +
    scale_x_continuous(breaks = c(0.1,0.5,0.9), name = "Masking fraction") +
    scale_y_continuous(name = NULL, limits = c(0, 1)) +
    theme_pub(base_size = 7.5) +
    theme(legend.position = "top",
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
    ggtitle(meth)
}
sup1 <- wrap_plots(lapply(DA_LEVELS, make_metric_curves), ncol = 1) +
  plot_annotation(title = "Supplementary Fig. S2: full metric curves (all effect sizes × masking fractions)")
ggsave(file.path(OUT, "supplementary_full_metric_curves.pdf"),
       sup1, width = 9.5, height = 13, units = "in", device = cairo_pdf)
ggsave(file.path(OUT, "supplementary_full_metric_curves.png"),
       sup1, width = 9.5, height = 13, units = "in", dpi = 600,
       device = ragg::agg_png)
cat("Supplementary S1 written.\n")

# ----------------------------------------------------------------------------
# 8. SUPPLEMENTARY FIGURE 2: full delta heatmaps (dF1, dRecall, dFDR)
# ----------------------------------------------------------------------------
heat_one <- function(col, title, palette_pos_good = TRUE) {
  d <- delta %>% mutate(eff = factor(effect_size), frc = factor(masking_fraction))
  lim <- max(abs(d[[col]]), na.rm = TRUE)
  # For dFDR, positive means imputed has HIGHER FDR (worse) -> flip colors
  if (palette_pos_good) {
    lo <- DIV_LOW; hi <- DIV_HIGH
  } else {
    lo <- DIV_HIGH; hi <- DIV_LOW
  }
  ggplot(d, aes(eff, frc, fill = .data[[col]])) +
    geom_tile(color = "white", linewidth = 0.25) +
    facet_wrap(~da_method, nrow = 1) +
    scale_fill_gradient2(low = lo, mid = DIV_MID, high = hi, midpoint = 0,
                         limits = c(-lim, lim), name = title) +
    scale_x_discrete(name = "Effect size") +
    scale_y_discrete(name = "Masking fraction") +
    theme_pub(base_size = 8) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
    ggtitle(title)
}
sup2 <- heat_one("delta_F1", "ΔF1 (positive = imputed better)", TRUE) /
        heat_one("delta_recall", "ΔRecall (positive = imputed better)", TRUE) /
        heat_one("delta_FDR", "ΔFDR (green = imputed lower FDR = better)", FALSE) +
  plot_annotation(title = "Supplementary Fig. S3: full Δ heatmaps across all conditions")
ggsave(file.path(OUT, "supplementary_delta_heatmaps.pdf"),
       sup2, width = 9.5, height = 9.5, units = "in", device = cairo_pdf)
ggsave(file.path(OUT, "supplementary_delta_heatmaps.png"),
       sup2, width = 9.5, height = 9.5, units = "in", dpi = 600,
       device = ragg::agg_png)
cat("Supplementary S2 written.\n")

# ----------------------------------------------------------------------------
# 9. SUPPLEMENTARY FIGURE 3: confusion-count summary (TP/FP/FN/TN)
# ----------------------------------------------------------------------------
conf_long <- combined %>%
  select(da_method, effect_size, masking_fraction, data_condition,
         TP, FP, FN, TN) %>%
  pivot_longer(c(TP, FP, FN, TN), names_to = "count_type", values_to = "count") %>%
  mutate(count_type = factor(count_type, levels = c("TP","FP","FN","TN")))

make_conf <- function(meth) {
  d <- conf_long %>% filter(da_method == meth)
  ggplot(d, aes(masking_fraction, count, color = data_condition)) +
    geom_line(linewidth = 0.5) + geom_point(size = 0.7) +
    facet_grid(count_type ~ effect_size, scales = "free_y", switch = "y") +
    scale_color_manual(values = COND_COLORS, name = NULL) +
    scale_x_continuous(breaks = c(0.1,0.5,0.9), name = "Masking fraction") +
    scale_y_continuous(name = NULL) +
    theme_pub(base_size = 7.5) +
    theme(legend.position = "top",
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
    ggtitle(meth)
}
sup3 <- wrap_plots(lapply(DA_LEVELS, make_conf), ncol = 1) +
  plot_annotation(title = "Supplementary Fig. S4: confusion-count summary (TP/FP/FN/TN) across conditions")
ggsave(file.path(OUT, "supplementary_confusion_counts.pdf"),
       sup3, width = 9.5, height = 13, units = "in", device = cairo_pdf)
ggsave(file.path(OUT, "supplementary_confusion_counts.png"),
       sup3, width = 9.5, height = 13, units = "in", dpi = 600,
       device = ragg::agg_png)
cat("Supplementary S3 written.\n")

cat("\nALL OUTPUTS WRITTEN TO:", normalizePath(OUT), "\n")
