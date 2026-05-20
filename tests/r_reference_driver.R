#!/usr/bin/env Rscript
# Drive Bioconductor lipidr reference on the bundled example dataset.
#
# Usage:
#   Rscript r_reference_driver.R <out_dir>
#
# Outputs (in out_dir):
#   annotations.tsv   annotate_lipids() class / category / chain features
#   pqn.tsv           normalize_pqn() normalized + log2 Area matrix
#   istd.tsv          normalize_istd() normalized + log2 Area matrix (if ISTDs)
#   de.tsv            de_analysis() moderated-t result
#   lsea.tsv          lsea() enrichment result
#   info.tsv          dataset metadata (molecules, groups)

suppressPackageStartupMessages({
  library(lipidr)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "R_out"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

extdata <- system.file("extdata", package = "lipidr")

# --- read the bundled Skyline example -------------------------------
d <- read_skyline(file.path(extdata, "A1_data.csv"))
clin <- read.csv(file.path(extdata, "clin.csv"))
d <- add_sample_annotation(d, clin)
d <- summarize_transitions(d, method = "max")

# --- annotations ----------------------------------------------------
rd <- as.data.frame(rowData(d))
ann <- rd[, c("Molecule", "Class", "total_cl", "total_cs", "istd")]
write.table(ann, file.path(out_dir, "annotations.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- normalize_pqn --------------------------------------------------
d_pqn <- normalize_pqn(d, measure = "Area", exclude = "blank", log = TRUE)
pqn_m <- assay(d_pqn, "Area")
rownames(pqn_m) <- rowData(d_pqn)$Molecule
write.table(pqn_m, file.path(out_dir, "pqn.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

# --- normalize_istd (only if internal standards present) ------------
if (sum(rowData(d)$istd) > 0) {
  d_istd <- tryCatch(
    normalize_istd(d, measure = "Area", exclude = "blank", log = TRUE),
    error = function(e) NULL)
  if (!is.null(d_istd)) {
    istd_m <- assay(d_istd, "Area")
    rownames(istd_m) <- rowData(d_istd)$Molecule
    write.table(istd_m, file.path(out_dir, "istd.tsv"),
                sep = "\t", quote = FALSE, col.names = NA)
  }
}

# --- de_analysis ----------------------------------------------------
d_norm <- normalize_pqn(d, measure = "Area", exclude = "blank", log = TRUE)
de <- de_analysis(d_norm, HighFat_water - NormalDiet_water, measure = "Area")
de_out <- de[, c("Molecule", "Class", "logFC", "AveExpr", "t",
                 "P.Value", "adj.P.Val")]
write.table(de_out, file.path(out_dir, "de.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- lsea -----------------------------------------------------------
set.seed(42)
enr <- lsea(de, rank.by = "logFC", nperm = 2000)
enr_df <- as.data.frame(enr)[, c("contrast", "set", "pval", "padj",
                                 "ES", "NES", "size")]
write.table(enr_df, file.path(out_dir, "lsea.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- info -----------------------------------------------------------
info <- data.frame(
  n_molecules = nrow(d),
  n_samples = ncol(d),
  n_classes = length(unique(rowData(d)$Class))
)
write.table(info, file.path(out_dir, "info.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("R lipidr reference done\n")
