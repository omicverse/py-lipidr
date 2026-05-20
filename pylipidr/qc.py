"""Quality control and pre-processing.

Ports of lipidr's ``filter_by_cv``, ``impute_na`` and
``summarize_transitions``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .experiment import LipidomicsExperiment


def _cv(a: np.ndarray) -> float:
    """Coefficient of variation in percent (R ``.cv``)."""
    a = np.asarray(a, dtype=float)
    mean = np.nanmean(a)
    if mean == 0 or not np.isfinite(mean):
        return np.inf
    return (np.nanstd(a, ddof=1) / mean) * 100.0


def filter_by_cv(
    experiment: LipidomicsExperiment,
    cv_cutoff: float = 20.0,
    measure: Optional[str] = None,
    qc_samples: Optional[list] = None,
) -> LipidomicsExperiment:
    """Drop lipids whose CV exceeds ``cv_cutoff`` (R ``filter_by_cv``).

    Parameters
    ----------
    experiment
        The :class:`LipidomicsExperiment`.
    cv_cutoff
        CV threshold in percent (lipidr default 20).
    measure
        Which measure to compute CV on (defaults to the primary measure).
    qc_samples
        Optional list of QC-sample names; if given, CV is computed only
        across those (lipidr computes it across *all* samples by default).
    """
    measure = measure or experiment.default_measure
    mat = experiment.assay(measure)  # lipids x samples
    if qc_samples is not None:
        mat = mat[qc_samples]
    cv = mat.apply(lambda row: _cv(row.to_numpy()), axis=1)
    keep = cv.index[cv < cv_cutoff]
    return experiment.subset_molecules(keep)


def _impute_knn(mat: np.ndarray, k: int = 10) -> np.ndarray:
    """KNN imputation over features (rows = lipids, cols = samples).

    Mirrors ``imputeLCMD::impute.wrapper.KNN``: neighbours are chosen by
    Euclidean distance over the co-observed samples, missing cells filled
    with the neighbour mean.
    """
    out = mat.copy()
    n_rows = mat.shape[0]
    for i in range(n_rows):
        miss = ~np.isfinite(mat[i])
        if not miss.any():
            continue
        target = mat[i]
        dists = np.full(n_rows, np.inf)
        for j in range(n_rows):
            if j == i:
                continue
            both = np.isfinite(target) & np.isfinite(mat[j])
            if both.sum() < 1:
                continue
            d = target[both] - mat[j][both]
            dists[j] = np.sqrt(np.mean(d * d))
        order = np.argsort(dists)
        for col in np.where(miss)[0]:
            vals = []
            for j in order:
                if not np.isfinite(dists[j]):
                    break
                if np.isfinite(mat[j, col]):
                    vals.append(mat[j, col])
                if len(vals) >= k:
                    break
            if vals:
                out[i, col] = float(np.mean(vals))
            else:
                col_vals = mat[:, col][np.isfinite(mat[:, col])]
                out[i, col] = float(np.mean(col_vals)) if col_vals.size else 0.0
    return out


def _impute_min(mat: np.ndarray, kind: str) -> np.ndarray:
    """Minimum-value family imputation (minDet / minProb / zero / min)."""
    out = mat.copy()
    miss = ~np.isfinite(out)
    if kind == "zero":
        out[miss] = 0.0
        return out
    finite = out[np.isfinite(out)]
    gmin = float(np.min(finite)) if finite.size else 0.0
    if kind in ("min", "minDet"):
        # per-column minimum-detected value
        for col in range(out.shape[1]):
            cmiss = ~np.isfinite(out[:, col])
            if not cmiss.any():
                continue
            cfin = out[:, col][np.isfinite(out[:, col])]
            cmin = float(np.min(cfin)) if cfin.size else gmin
            out[cmiss, col] = cmin
        return out
    if kind == "minProb":
        rng = np.random.default_rng(0)
        sd = float(np.std(finite)) if finite.size else 1.0
        out[miss] = gmin + rng.normal(0.0, 0.01 * sd, size=int(miss.sum()))
        return out
    raise ValueError(f"unknown min-family method: {kind}")


def impute_na(
    experiment: LipidomicsExperiment,
    measure: Optional[str] = None,
    method: str = "knn",
    k: int = 10,
) -> LipidomicsExperiment:
    """Impute missing values (R ``impute_na``).

    Parameters
    ----------
    method
        One of ``"knn"``, ``"min"`` / ``"minDet"``, ``"minProb"`` or
        ``"zero"``.  lipidr also offers ``svd`` / ``mle`` / ``QRILC`` via
        imputeLCMD; the most-used ``knn`` and ``min`` variants are ported.
    k
        Neighbour count for the KNN method.
    """
    measure = measure or experiment.default_measure
    mat = experiment.assay(measure)
    arr = mat.to_numpy(dtype=float)
    if method == "knn":
        filled = _impute_knn(arr, k=k)
    elif method in ("min", "minDet", "minProb", "zero"):
        filled = _impute_min(arr, method)
    else:
        raise ValueError(
            f"Unsupported impute method '{method}'. "
            "Use one of: knn, min, minDet, minProb, zero."
        )
    out = experiment.copy()
    out.set_assay(
        pd.DataFrame(filled, index=mat.index, columns=mat.columns), measure
    )
    return out


def summarize_transitions(
    experiment: LipidomicsExperiment, method: str = "max"
) -> LipidomicsExperiment:
    """Collapse multiple transitions per lipid (R ``summarize_transitions``).

    Skyline data can contain several transitions (rows) for the same
    lipid molecule.  This collapses them to one value per molecule using
    ``"max"`` (default) or ``"average"``.
    """
    if method not in ("max", "average"):
        raise ValueError("method must be 'max' or 'average'")
    if experiment.is_summarized():
        raise ValueError("data is already summarized")
    var = experiment.adata.var
    if "Molecule" not in var.columns:
        raise ValueError("Experiment has no 'Molecule' column to summarize on.")

    agg = np.nanmax if method == "max" else np.nanmean
    measures = experiment.measure_names()
    groups = var["Molecule"].astype(str)
    uniq = list(pd.unique(groups))

    new_layers = {}
    for m in measures:
        mat = experiment.assay(m)  # lipids x samples
        collapsed = mat.groupby(groups.to_numpy()).agg(
            lambda block: agg(block.to_numpy()) if np.isfinite(block.to_numpy()).any() else np.nan
        )
        collapsed = collapsed.reindex(uniq)
        new_layers[m] = collapsed

    # per-molecule rowdata: first row of each group
    first_idx = var.groupby(groups.to_numpy(), sort=False).first()
    row_data = first_idx.reindex(uniq)
    row_data = row_data.drop(columns=[c for c in ("TransitionId",) if c in row_data.columns])
    row_data["Molecule"] = uniq

    from anndata import AnnData

    primary = new_layers[experiment.default_measure]
    ad = AnnData(
        X=primary.to_numpy(dtype=float).T,
        obs=experiment.adata.obs.copy(),
        var=row_data,
    )
    ad.var_names = pd.Index(uniq, name="Molecule")
    for m, mat in new_layers.items():
        if m == experiment.default_measure:
            continue
        ad.layers[m] = mat.to_numpy(dtype=float).T
    ad.uns["lipidr_default_measure"] = experiment.default_measure
    # carry processing-state flags
    ad.uns["lipidr_state"] = dict(experiment.adata.uns.get("lipidr_state", {}))
    out = LipidomicsExperiment(ad)
    out.set_summarized(True)
    return out
