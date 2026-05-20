"""Normalization — internal-standard and PQN.

Ports of lipidr's ``normalize_istd`` and ``normalize_pqn``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .experiment import LipidomicsExperiment


def _prenormalize(
    experiment: LipidomicsExperiment, measure: str,
    exclude: Union[None, str, Sequence[str]],
) -> LipidomicsExperiment:
    """R ``.prenormalize_check`` — validate, drop excluded samples, fill non-finite."""
    if measure not in experiment.measure_names():
        raise ValueError(f"{measure} is not in the dataset.")
    if experiment.is_normalized(measure):
        raise ValueError(f"{measure} is already normalized.")
    exp = experiment.copy()
    if exclude is not None:
        if exclude == "blank":
            # drop samples whose name contains 'blank'
            keep = [
                s for s in exp.adata.obs_names if "blank" not in str(s).lower()
            ]
            exp = exp.subset_samples(keep)
        else:
            ex = [exclude] if isinstance(exclude, str) else list(exclude)
            keep = [s for s in exp.adata.obs_names if s not in ex]
            if not keep:
                raise ValueError("You cannot exclude all samples.")
            exp = exp.subset_samples(keep)
    mat = exp.assay(measure)
    arr = mat.to_numpy(dtype=float, copy=True)
    if not np.isfinite(arr).all():
        finite = arr[np.isfinite(arr)]
        fill = float(np.min(finite)) if finite.size else 0.0
        arr[~np.isfinite(arr)] = fill
        exp.set_assay(pd.DataFrame(arr, index=mat.index, columns=mat.columns), measure)
    return exp


def _log_data(experiment: LipidomicsExperiment, measure: str, log: bool) -> LipidomicsExperiment:
    """R ``.log_data`` — log2-transform the measure if requested and not already logged."""
    if not log or experiment.is_logged(measure):
        return experiment
    mat = experiment.assay(measure)
    arr = mat.to_numpy(dtype=float, copy=True)
    arr[arr < 1] = 1.0
    logged = np.log2(arr)
    out = experiment.copy()
    out.set_assay(pd.DataFrame(logged, index=mat.index, columns=mat.columns), measure)
    out.set_logged(True, measure)
    return out


def normalize_pqn(
    experiment: LipidomicsExperiment,
    measure: str = "Area",
    exclude: Union[None, str, Sequence[str]] = "blank",
    log: bool = True,
) -> LipidomicsExperiment:
    """Probabilistic Quotient Normalization (R ``normalize_pqn``).

    For each sample, the normalization factor is the *median* of the
    quotient of each lipid's intensity over its row mean across samples;
    every sample is then divided by its factor.

    Algorithm (faithful to lipidr)::

        factor_n = median_lipid( m[i, n] / rowMean(m[i, :]) )
        m_norm[i, n] = m[i, n] / factor_n
    """
    exp = _prenormalize(experiment, measure, exclude)
    mat = exp.assay(measure)  # lipids x samples
    m = mat.to_numpy(dtype=float)
    row_mean = np.nanmean(m, axis=1, keepdims=True)
    quotient = m / row_mean
    # per-sample (column) median of the quotient
    factor_n = np.nanmedian(quotient, axis=0)
    normalized = m / factor_n[np.newaxis, :]
    exp.set_assay(
        pd.DataFrame(normalized, index=mat.index, columns=mat.columns), measure
    )
    exp.set_normalized(True, measure)
    return _log_data(exp, measure, log)


def normalize_istd(
    experiment: LipidomicsExperiment,
    measure: str = "Area",
    exclude: Union[None, str, Sequence[str]] = "blank",
    log: bool = True,
) -> LipidomicsExperiment:
    """Internal-standard normalization (R ``normalize_istd``).

    Each lipid is divided by the (mean-centred) signal of the internal
    standard(s) belonging to its own lipid *class*.  Lipids whose class
    has no internal standard are left unchanged (factor = 1).

    Requires the experiment's ``.var`` to carry ``istd`` and ``Class``
    columns (populated by :func:`pylipidr.annotate.annotate_experiment`).
    """
    exp = _prenormalize(experiment, measure, exclude)
    var = exp.adata.var
    if "istd" not in var.columns:
        raise ValueError(
            "No 'istd' column in annotations. Run annotate_experiment first."
        )
    istd_mask = var["istd"].to_numpy(dtype=bool)
    if istd_mask.sum() == 0:
        raise ValueError("No internal standards found in your lipid list.")

    mat = exp.assay(measure)  # lipids x samples
    m = mat.to_numpy(dtype=float)

    # mean-centre each ISTD row
    mistd = m[istd_mask, :].copy()
    istd_row_mean = np.nanmean(mistd, axis=1, keepdims=True)
    mistd = mistd / istd_row_mean
    istd_names = list(mat.index[istd_mask])
    mistd_df = pd.DataFrame(mistd, index=istd_names, columns=mat.columns)

    classes = var["Class"].astype(str).to_numpy()
    istd_class = classes[istd_mask]

    out = np.empty_like(m)
    for i in range(m.shape[0]):
        cls = classes[i]
        members = [
            istd_names[j] for j in range(len(istd_names)) if istd_class[j] == cls
        ]
        if len(members) == 0:
            factor = np.ones(m.shape[1])
        elif len(members) == 1:
            factor = mistd_df.loc[members[0]].to_numpy()
        else:
            factor = np.nanmean(mistd_df.loc[members].to_numpy(), axis=0)
        out[i, :] = m[i, :] / factor

    exp.set_assay(pd.DataFrame(out, index=mat.index, columns=mat.columns), measure)
    exp.set_normalized(True, measure)
    return _log_data(exp, measure, log)
