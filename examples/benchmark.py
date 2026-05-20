"""Head-to-head speed + accuracy benchmark: R lipidr vs pylipidr.

Runs the full lipidomics pipeline -- read Skyline -> annotate ->
summarize transitions -> normalize (PQN) -> de_analysis -> lsea -- on
lipidr's own bundled example dataset (``extdata/A1_data.csv`` +
``clin.csv``), so both languages analyse identical input.

Reports, per stage:

* wall-clock time (Python via ``time.perf_counter``; R via ``Rscript``).
* accuracy of the Python output vs R: Pearson r / agreement for
  annotation classes, PQN, ISTD, DE logFC + p-values and LSEA scores.

Usage::

    python examples/benchmark.py --runs 2
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import pylipidr as lp

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
WORK = HERE / "compare_out"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"
EXTDATA = Path("/scratch/users/steorra/env/CMAP/lib/R/library/lipidr/extdata")
R_DRIVER = HERE.parent / "tests" / "r_reference_driver.R"
CONTRAST = "HighFat_water - NormalDiet_water"


def _pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def run_python(runs: int):
    """Run the full pylipidr pipeline ``runs`` times; return mean time + results."""
    elapsed = []
    de = enr = norm = exp = None
    for _ in range(runs):
        t0 = time.perf_counter()
        exp = lp.read_skyline(str(EXTDATA / "A1_data.csv"))
        exp = lp.add_sample_annotation(exp, str(EXTDATA / "clin.csv"))
        exp = lp.summarize_transitions(exp, method="max")
        norm = lp.normalize_pqn(exp, measure="Area")
        de = lp.de_analysis(norm, CONTRAST, group_col="group")
        enr = lp.lsea(de, rank_by="logFC", nperm=2000, seed=42)
        elapsed.append(time.perf_counter() - t0)
    return float(np.mean(elapsed)), exp, norm, de, enr


def run_R(out_dir: Path):
    """Run the lipidr R reference driver once; return wall time."""
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} {out_dir}"
    )
    t0 = time.perf_counter()
    res = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"R driver failed:\n{res.stderr[-2000:]}")
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    if not (EXTDATA / "A1_data.csv").exists():
        raise SystemExit(
            f"lipidr example data not found at {EXTDATA}. "
            "Install lipidr in the CMAP R env."
        )
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    r_out = WORK / "R_out"

    print(f"Dataset: lipidr bundled Skyline example (A1_data.csv)")

    print(f"\n--- Python pipeline (mean of {args.runs} runs) ---")
    py_time, exp, norm, de, enr = run_python(args.runs)
    print(f"  pylipidr total      {py_time*1000:9.1f} ms "
          f"({exp.shape[0]} lipids x {exp.shape[1]} samples)")

    print(f"\n--- R pipeline (single run) ---")
    r_time = run_R(r_out)
    print(f"  R lipidr total      {r_time*1000:9.1f} ms (incl. Rscript startup)")
    print(f"\nSpeedup (R / Python): {r_time / py_time:.2f}x")

    # ----- accuracy -----------------------------------------------------
    print(f"\n--- Accuracy (Python vs R) ---")
    summary = {
        "shape": list(exp.shape),
        "py_time_ms": py_time * 1000,
        "r_time_ms": r_time * 1000,
        "speedup": r_time / py_time,
    }

    # annotation
    r_ann = pd.read_csv(r_out / "annotations.tsv", sep="\t").set_index("Molecule")
    py_ann = exp.adata.var.set_index("Molecule")
    cm = r_ann.index.intersection(py_ann.index)
    cls_agree = (
        r_ann.loc[cm, "Class"].astype(str).to_numpy()
        == py_ann.loc[cm, "Class"].astype(str).to_numpy()
    ).mean()
    print(f"  annotate    class agreement   = {cls_agree:.3f}")
    summary["annotate_class_agreement"] = float(cls_agree)

    # PQN
    r_pqn = pd.read_csv(r_out / "pqn.tsv", sep="\t", index_col=0)
    py_pqn = norm.assay("Area")
    cm = r_pqn.index.intersection(py_pqn.index)
    cc = r_pqn.columns.intersection(py_pqn.columns)
    pqn_r = _pearson(
        r_pqn.loc[cm, cc].to_numpy().ravel(),
        py_pqn.loc[cm, cc].to_numpy().ravel(),
    )
    print(f"  normalize_pqn   Pearson r     = {pqn_r:.5f}")
    summary["pqn_pearson_r"] = pqn_r

    # ISTD
    istd_file = r_out / "istd.tsv"
    if istd_file.exists():
        ni = lp.normalize_istd(exp, measure="Area")
        r_istd = pd.read_csv(istd_file, sep="\t", index_col=0)
        py_istd = ni.assay("Area")
        cm = r_istd.index.intersection(py_istd.index)
        cc = r_istd.columns.intersection(py_istd.columns)
        istd_r = _pearson(
            r_istd.loc[cm, cc].to_numpy().ravel(),
            py_istd.loc[cm, cc].to_numpy().ravel(),
        )
        print(f"  normalize_istd  Pearson r     = {istd_r:.5f}")
        summary["istd_pearson_r"] = istd_r

    # DE
    r_de = pd.read_csv(r_out / "de.tsv", sep="\t").set_index("Molecule")
    py_de = de.set_index("Molecule")
    cm = r_de.index.intersection(py_de.index)
    fc_r = _pearson(r_de.loc[cm, "logFC"], py_de.loc[cm, "logFC"])
    p_r = _pearson(r_de.loc[cm, "P.Value"], py_de.loc[cm, "P.Value"])
    print(f"  de_analysis logFC Pearson r   = {fc_r:.5f}")
    print(f"  de_analysis P.Value Pearson r = {p_r:.5f}")
    summary["de_logfc_pearson_r"] = fc_r
    summary["de_pvalue_pearson_r"] = p_r

    # LSEA
    r_lsea = pd.read_csv(r_out / "lsea.tsv", sep="\t").set_index("set")
    py_lsea = enr.set_index("set")
    cm = r_lsea.index.intersection(py_lsea.index)
    es_r = _pearson(r_lsea.loc[cm, "ES"], py_lsea.loc[cm, "ES"])
    p_lsea_r = _pearson(r_lsea.loc[cm, "pval"], py_lsea.loc[cm, "pval"])
    print(f"  lsea ES Pearson r             = {es_r:.4f}")
    print(f"  lsea pval Pearson r           = {p_lsea_r:.4f}")
    summary["lsea_es_pearson_r"] = es_r
    summary["lsea_pval_pearson_r"] = p_lsea_r

    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFull report -> {WORK / 'summary.json'}")


if __name__ == "__main__":
    main()
