"""Algorithmic smoke tests for pylipidr — no R required.

These check the internal consistency of each ported routine against
hand-derived expectations on synthetic and bundled lipidr data.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pylipidr as lp

warnings.filterwarnings("ignore")

# lipidr bundles its example data inside the installed R package; the tests
# use it when available so Python runs the exact same input as R.
_EXTDATA = Path(
    "/scratch/users/steorra/env/CMAP/lib/R/library/lipidr/extdata"
)
_HAS_EXTDATA = (_EXTDATA / "A1_data.csv").exists()


# ----------------------------------------------------------------------
# synthetic fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_experiment():
    """A small synthetic lipidomics experiment (lipids x samples)."""
    rng = np.random.default_rng(0)
    lipids = [
        "PC 32:0", "PC 34:1", "PC 36:2", "PE 34:1", "PE 36:2",
        "TG 52:2", "TG 54:3", "Cer d18:1/16:0", "SM d18:1/16:0",
        "LPC 18:0",
    ]
    samples = [f"S{i:02d}" for i in range(12)]
    base = rng.uniform(1e4, 1e6, size=len(lipids))
    mat = pd.DataFrame(
        base[:, None] * rng.lognormal(0, 0.25, size=(len(lipids), 12)),
        index=lipids, columns=samples,
    )
    # inject a group effect into the second half
    mat.iloc[:5, 6:] *= 2.0
    col = pd.DataFrame(
        {"group": (["ctrl"] * 6) + (["treat"] * 6)}, index=samples
    )
    exp = lp.as_lipidomics_experiment(mat.reset_index(), measure="Area")
    for c in col.columns:
        exp.adata.obs[c] = col[c].reindex(exp.adata.obs_names).to_numpy()
    lp.annotate_experiment(exp, no_match="ignore")
    return exp


# ----------------------------------------------------------------------
# annotation
# ----------------------------------------------------------------------
def test_annotate_lipids_basic():
    ann = lp.annotate_lipids(
        ["PC 32:0", "TG 54:3", "Cer d18:1/16:0"], no_match="ignore"
    )
    assert list(ann.loc["PC 32:0", ["Class", "Category"]]) == ["PC", "GP"]
    assert ann.loc["TG 54:3", "Category"] == "GL"
    assert ann.loc["TG 54:3", "total_cl"] == 54
    assert ann.loc["TG 54:3", "total_cs"] == 3
    assert ann.loc["Cer d18:1/16:0", "Category"] == "SP"
    assert ann.loc["Cer d18:1/16:0", "total_cl"] == 34
    assert not ann["not_parsed"].any()


def test_annotate_non_parsed_flagged():
    ann = lp.annotate_lipids(["PC 32:0", "totally not a lipid"], no_match="ignore")
    assert lp.non_parsed_molecules(ann) == ["totally not a lipid"]
    assert ann.loc["PC 32:0", "not_parsed"] == False  # noqa: E712


def test_annotate_istd_detection():
    ann = lp.annotate_lipids(
        ["PC 34:1", "PC 15:0-18:1(d7)", "So1P 17:1"], no_match="ignore"
    )
    assert not bool(ann.loc["PC 34:1", "istd"])
    assert bool(ann.loc["PC 15:0-18:1(d7)", "istd"])
    assert bool(ann.loc["So1P 17:1", "istd"])  # curated odd-chain standard


# ----------------------------------------------------------------------
# experiment + state flags
# ----------------------------------------------------------------------
def test_experiment_state_flags(synthetic_experiment):
    exp = synthetic_experiment
    assert exp.shape == (10, 12)
    assert not exp.is_logged()
    assert not exp.is_normalized()
    exp2 = exp.copy()
    exp2.set_logged(True).set_normalized(True)
    assert exp2.is_logged() and exp2.is_normalized()
    # flags must not leak back into the original
    assert not exp.is_logged()


# ----------------------------------------------------------------------
# QC
# ----------------------------------------------------------------------
def test_filter_by_cv_drops_variable_lipids():
    rng = np.random.default_rng(1)
    mat = pd.DataFrame(
        {
            "stable": [100, 101, 99, 100, 102, 98],   # CV ~1.5%
            "noisy": [10, 200, 5, 300, 1, 250],       # CV >> 20%
        }
    ).T
    mat.columns = [f"S{i}" for i in range(6)]
    exp = lp.as_lipidomics_experiment(mat.reset_index(), measure="Area")
    filt = lp.filter_by_cv(exp, cv_cutoff=20.0)
    assert "stable" in filt.molecules
    assert "noisy" not in filt.molecules


def test_impute_na_fills_missing():
    mat = pd.DataFrame(
        np.array([[1.0, 2.0, np.nan, 4.0], [10.0, np.nan, 30.0, 40.0]]),
        index=["L1", "L2"], columns=["S1", "S2", "S3", "S4"],
    )
    exp = lp.as_lipidomics_experiment(mat.reset_index(), measure="Area")
    for method in ("knn", "min", "zero"):
        imp = lp.impute_na(exp, method=method)
        assert np.isfinite(imp.assay("Area").to_numpy()).all()
    # zero method fills with exactly 0
    z = lp.impute_na(exp, method="zero").assay("Area")
    assert z.loc["L1", "S3"] == 0.0


def test_summarize_transitions_collapses(synthetic_experiment):
    # synthetic data has one transition per lipid -> n unchanged
    summ = lp.summarize_transitions(synthetic_experiment, method="max")
    assert summ.is_summarized()
    assert summ.shape[0] == synthetic_experiment.shape[0]


# ----------------------------------------------------------------------
# normalization
# ----------------------------------------------------------------------
def test_normalize_pqn_sets_flags(synthetic_experiment):
    norm = lp.normalize_pqn(synthetic_experiment, measure="Area", exclude=None)
    assert norm.is_normalized("Area")
    assert norm.is_logged("Area")  # log=True by default
    # normalized values are finite
    assert np.isfinite(norm.assay("Area").to_numpy()).all()


def test_normalize_pqn_factor_is_median_quotient():
    # build a matrix where one sample is uniformly 2x another
    mat = pd.DataFrame(
        {
            "S1": [10.0, 20.0, 30.0, 40.0],
            "S2": [20.0, 40.0, 60.0, 80.0],  # exactly 2x S1
        },
        index=["L1", "L2", "L3", "L4"],
    )
    exp = lp.as_lipidomics_experiment(mat.reset_index(), measure="Area")
    norm = lp.normalize_pqn(exp, measure="Area", exclude=None, log=False)
    out = norm.assay("Area")
    # after PQN the two samples should be brought onto a common scale
    np.testing.assert_allclose(
        out["S1"].to_numpy(), out["S2"].to_numpy(), rtol=1e-9
    )


def test_normalize_istd_requires_standards(synthetic_experiment):
    # synthetic data has no internal standards -> should raise
    with pytest.raises(ValueError, match="internal standard"):
        lp.normalize_istd(synthetic_experiment, measure="Area", exclude=None)


# ----------------------------------------------------------------------
# differential analysis
# ----------------------------------------------------------------------
def test_de_analysis_detects_effect(synthetic_experiment):
    # log2-transform without PQN (PQN's median-quotient would absorb the
    # global shift, since half of the synthetic lipids are spiked).
    exp = synthetic_experiment.copy()
    logged = np.log2(exp.assay("Area"))
    exp.set_assay(logged, "Area")
    exp.set_logged(True, "Area")
    de = lp.de_analysis(exp, "treat - ctrl", group_col="group")
    assert {"contrast", "Molecule", "logFC", "P.Value", "adj.P.Val"}.issubset(
        de.columns
    )
    de_idx = de.set_index("Molecule")
    # the 5 spiked lipids (spiked 2x -> logFC ~1) should be clearly positive
    spiked = ["PC 32:0", "PC 34:1", "PC 36:2", "PE 34:1", "PE 36:2"]
    assert (de_idx.loc[spiked, "logFC"] > 0.5).all()
    # and a smaller p-value than the un-spiked ones, on average
    unspiked = ["TG 52:2", "TG 54:3", "Cer d18:1/16:0", "SM d18:1/16:0"]
    assert de_idx.loc[spiked, "P.Value"].mean() < de_idx.loc[
        unspiked, "P.Value"
    ].mean()


def test_de_design_one_hot(synthetic_experiment):
    design = lp.de_design(synthetic_experiment, group_col="group")
    assert set(design.columns) == {"ctrl", "treat"}
    assert design.shape == (12, 2)
    # one-hot: each row sums to 1
    np.testing.assert_array_equal(design.sum(axis=1).to_numpy(), np.ones(12))


def _logged(exp):
    """Return a log2-transformed copy of an experiment (no normalization)."""
    out = exp.copy()
    out.set_assay(np.log2(out.assay("Area")), "Area")
    out.set_logged(True, "Area")
    return out


def test_significant_molecules_and_top_lipids(synthetic_experiment):
    de = lp.de_analysis(_logged(synthetic_experiment), "treat - ctrl",
                        group_col="group")
    sig = lp.significant_molecules(de, p_cutoff=0.05, logfc_cutoff=0.5)
    assert "treat - ctrl" in sig
    assert len(sig["treat - ctrl"]) >= 1
    top = lp.top_lipids(de, top_n=3)
    assert len(top) == 3


# ----------------------------------------------------------------------
# LSEA
# ----------------------------------------------------------------------
def test_gen_lipidsets_structure():
    mols = [
        "PC 32:0", "PC 34:1", "PC 36:2", "PE 34:1", "PE 36:2",
        "TG 52:2", "TG 54:3",
    ]
    sets = lp.gen_lipidsets(mols, min_size=2)
    # one set per class with >= 2 members
    assert "Class_PC" in sets
    assert "Class_PE" in sets
    assert len(sets["Class_PC"]) == 3
    # collections also include total_cl / total_cs
    assert any(k.startswith("total_cl_") for k in sets)
    assert any(k.startswith("total_cs_") for k in sets)


def test_lsea_runs_and_enriches(synthetic_experiment):
    de = lp.de_analysis(_logged(synthetic_experiment), "treat - ctrl",
                        group_col="group")
    enr = lp.lsea(de, rank_by="logFC", min_size=2, nperm=500, seed=0)
    assert {"contrast", "set", "pval", "padj", "ES", "NES", "size"}.issubset(
        enr.columns
    )
    assert len(enr) > 0
    # the PC set (all spiked up) should have a positive enrichment score
    pc = enr[enr["set"] == "Class_PC"]
    if not pc.empty:
        assert pc["ES"].iloc[0] > 0
    sig = lp.significant_lipidsets(enr, p_cutoff=0.5)
    assert isinstance(sig, dict)


# ----------------------------------------------------------------------
# I/O on bundled lipidr example data (if present)
# ----------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_EXTDATA, reason="lipidr extdata not installed")
def test_read_skyline_bundled():
    exp = lp.read_skyline(str(_EXTDATA / "A1_data.csv"))
    # lipidr reports 58 samples for A1_data.csv
    assert exp.shape[1] == 58
    assert "Molecule" in exp.adata.var.columns
    assert "Class" in exp.adata.var.columns


@pytest.mark.skipif(not _HAS_EXTDATA, reason="lipidr extdata not installed")
def test_full_pipeline_bundled():
    exp = lp.read_skyline(str(_EXTDATA / "A1_data.csv"))
    exp = lp.add_sample_annotation(exp, str(_EXTDATA / "clin.csv"))
    exp = lp.summarize_transitions(exp, method="max")
    norm = lp.normalize_pqn(exp, measure="Area")
    de = lp.de_analysis(norm, "HighFat_water - NormalDiet_water", group_col="group")
    assert len(de) == norm.shape[0]
    enr = lp.lsea(de, rank_by="logFC", nperm=500, seed=0)
    assert len(enr) > 0


def test_public_api_importable():
    """Every documented public function must be importable from pylipidr."""
    for name in lp.__all__:
        assert hasattr(lp, name), f"{name} missing from pylipidr namespace"
