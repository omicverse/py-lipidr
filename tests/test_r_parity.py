"""R-parity tests — pylipidr vs Bioconductor lipidr.

The R driver (:file:`r_reference_driver.R`) runs lipidr 2.20.0 on its own
bundled Skyline example dataset (``extdata/A1_data.csv`` + ``clin.csv``),
so both sides analyse the exact same input.  We compare:

* ``annotate_lipids`` — class assignments agree (> 95%).
* ``normalize_pqn``   — normalized values Pearson r > 0.99.
* ``normalize_istd``  — normalized values Pearson r > 0.99.
* ``de_analysis``     — logFC r > 0.99, P.Value r > 0.95.
* ``lsea``            — enrichment scores r > 0.9; top significant sets
  overlap.

Tests skip gracefully when the CMAP R env or lipidr is unavailable.
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import pylipidr as lp

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"
EXTDATA = Path("/scratch/users/steorra/env/CMAP/lib/R/library/lipidr/extdata")


def _r_available() -> bool:
    if not R_DRIVER.exists() or not (EXTDATA / "A1_data.csv").exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
             "&& Rscript -e 'library(lipidr); cat(\"OK\")'"],
            capture_output=True, text=True, timeout=120, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or lipidr not installed.",
)


@pytest.fixture(scope="module")
def r_reference(tmp_path_factory):
    """Run the lipidr R reference once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("lipidr_R")
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} {out_dir}"
    )
    res = subprocess.run(
        ["bash", "-lc", cmd], capture_output=True, text=True, timeout=900,
    )
    if res.returncode != 0:
        pytest.skip(f"R reference driver failed:\n{res.stderr[-2000:]}")
    return out_dir


@pytest.fixture(scope="module")
def py_experiment():
    """Python-side experiment matching the R driver's processing."""
    exp = lp.read_skyline(str(EXTDATA / "A1_data.csv"))
    exp = lp.add_sample_annotation(exp, str(EXTDATA / "clin.csv"))
    exp = lp.summarize_transitions(exp, method="max")
    return exp


# ----------------------------------------------------------------------
def test_annotate_class_agreement(r_reference, py_experiment):
    r_ann = pd.read_csv(r_reference / "annotations.tsv", sep="\t")
    r_ann = r_ann.set_index("Molecule")
    py_ann = py_experiment.adata.var.set_index("Molecule")
    common = r_ann.index.intersection(py_ann.index)
    assert len(common) > 50
    agree = (
        r_ann.loc[common, "Class"].astype(str).to_numpy()
        == py_ann.loc[common, "Class"].astype(str).to_numpy()
    ).mean()
    assert agree > 0.95, f"class agreement {agree:.3f} below 0.95"


def test_annotate_total_carbons_agreement(r_reference, py_experiment):
    r_ann = pd.read_csv(r_reference / "annotations.tsv", sep="\t")
    r_ann = r_ann.set_index("Molecule")
    py_ann = py_experiment.adata.var.set_index("Molecule")
    common = r_ann.index.intersection(py_ann.index)
    agree = (
        r_ann.loc[common, "total_cl"].fillna(-1).to_numpy()
        == py_ann.loc[common, "total_cl"].fillna(-1).to_numpy()
    ).mean()
    assert agree > 0.95, f"total_cl agreement {agree:.3f} below 0.95"


def test_normalize_pqn_vs_R(r_reference, py_experiment):
    norm = lp.normalize_pqn(py_experiment, measure="Area")
    py = norm.assay("Area")
    r = pd.read_csv(r_reference / "pqn.tsv", sep="\t", index_col=0)
    cm = r.index.intersection(py.index)
    cc = r.columns.intersection(py.columns)
    rv = r.loc[cm, cc].to_numpy().ravel()
    pv = py.loc[cm, cc].to_numpy().ravel()
    mask = np.isfinite(rv) & np.isfinite(pv)
    rho, _ = pearsonr(rv[mask], pv[mask])
    assert rho > 0.99, f"normalize_pqn Pearson r = {rho:.5f} (expected > 0.99)"


def test_normalize_istd_vs_R(r_reference, py_experiment):
    istd_file = r_reference / "istd.tsv"
    if not istd_file.exists():
        pytest.skip("R produced no normalize_istd output (no ISTDs).")
    norm = lp.normalize_istd(py_experiment, measure="Area")
    py = norm.assay("Area")
    r = pd.read_csv(istd_file, sep="\t", index_col=0)
    cm = r.index.intersection(py.index)
    cc = r.columns.intersection(py.columns)
    rv = r.loc[cm, cc].to_numpy().ravel()
    pv = py.loc[cm, cc].to_numpy().ravel()
    mask = np.isfinite(rv) & np.isfinite(pv)
    rho, _ = pearsonr(rv[mask], pv[mask])
    assert rho > 0.99, f"normalize_istd Pearson r = {rho:.5f} (expected > 0.99)"


def test_de_analysis_logfc_vs_R(r_reference, py_experiment):
    norm = lp.normalize_pqn(py_experiment, measure="Area")
    de = lp.de_analysis(norm, "HighFat_water - NormalDiet_water",
                        group_col="group")
    r = pd.read_csv(r_reference / "de.tsv", sep="\t").set_index("Molecule")
    py = de.set_index("Molecule")
    cm = r.index.intersection(py.index)
    rho_fc, _ = pearsonr(r.loc[cm, "logFC"], py.loc[cm, "logFC"])
    assert rho_fc > 0.99, f"de_analysis logFC r = {rho_fc:.5f} (expected > 0.99)"


def test_de_analysis_pvalue_vs_R(r_reference, py_experiment):
    norm = lp.normalize_pqn(py_experiment, measure="Area")
    de = lp.de_analysis(norm, "HighFat_water - NormalDiet_water",
                        group_col="group")
    r = pd.read_csv(r_reference / "de.tsv", sep="\t").set_index("Molecule")
    py = de.set_index("Molecule")
    cm = r.index.intersection(py.index)
    rho_p, _ = pearsonr(r.loc[cm, "P.Value"], py.loc[cm, "P.Value"])
    assert rho_p > 0.95, f"de_analysis P.Value r = {rho_p:.5f} (expected > 0.95)"


def test_lsea_enrichment_scores_vs_R(r_reference, py_experiment):
    norm = lp.normalize_pqn(py_experiment, measure="Area")
    de = lp.de_analysis(norm, "HighFat_water - NormalDiet_water",
                        group_col="group")
    enr = lp.lsea(de, rank_by="logFC", nperm=2000, seed=42)
    r = pd.read_csv(r_reference / "lsea.tsv", sep="\t").set_index("set")
    py = enr.set_index("set")
    cm = r.index.intersection(py.index)
    assert len(cm) > 5
    rho_es, _ = pearsonr(r.loc[cm, "ES"], py.loc[cm, "ES"])
    assert rho_es > 0.9, f"lsea ES Pearson r = {rho_es:.4f} (expected > 0.9)"


def test_lsea_pvalues_and_top_sets_vs_R(r_reference, py_experiment):
    norm = lp.normalize_pqn(py_experiment, measure="Area")
    de = lp.de_analysis(norm, "HighFat_water - NormalDiet_water",
                        group_col="group")
    enr = lp.lsea(de, rank_by="logFC", nperm=2000, seed=42)
    r = pd.read_csv(r_reference / "lsea.tsv", sep="\t").set_index("set")
    py = enr.set_index("set")
    cm = r.index.intersection(py.index)
    rho_p, _ = pearsonr(r.loc[cm, "pval"], py.loc[cm, "pval"])
    assert rho_p > 0.9, f"lsea pval Pearson r = {rho_p:.4f} (expected > 0.9)"
    # the significantly enriched sets must overlap
    r_sig = set(r.index[r["padj"] < 0.05])
    py_sig = set(py.index[py["padj"] < 0.05])
    if r_sig:
        overlap = len(r_sig & py_sig) / len(r_sig)
        assert overlap >= 0.5, (
            f"significant lipid sets overlap {overlap:.2f} "
            f"(R={r_sig}, PY={py_sig})"
        )
