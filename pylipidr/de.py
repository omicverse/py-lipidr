"""Differential analysis — moderated-t DE via limma.

Ports of lipidr's ``de_design``, ``de_analysis``, ``significant_molecules``
and ``top_lipids``.  The moderated-t machinery is delegated to **pylimma**
(``python-limma``), the published pure-Python limma port that lipidr's own
``de_analysis`` calls under the hood in R.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

import pylimma

from .experiment import LipidomicsExperiment


def de_design(
    experiment: LipidomicsExperiment,
    group_col: Optional[str] = None,
    formula: Optional[str] = None,
) -> pd.DataFrame:
    """Build a design matrix from sample annotations (R ``de_design``).

    Parameters
    ----------
    group_col
        A single column of ``col_data`` to expand into a one-hot
        ``~ 0 + group`` design (the lipidr default).
    formula
        Alternatively, a list of columns (joined main-effects design).

    Returns
    -------
    pandas.DataFrame — samples x design columns.
    """
    obs = experiment.col_data
    if group_col is None and formula is None:
        if obs.shape[1] == 0:
            raise ValueError("Please add clinical data or specify a group column")
        group_col = obs.columns[0]
    if group_col is not None:
        groups = obs[group_col].astype(str)
        design = pd.get_dummies(groups, prefix="", prefix_sep="")
        design = design.astype(float)
        design.index = obs.index
        return design
    # formula: list of column names -> one-hot per term, dropped-first
    cols = formula if isinstance(formula, (list, tuple)) else [formula]
    parts = [pd.Series(1.0, index=obs.index, name="(Intercept)")]
    for c in cols:
        d = pd.get_dummies(obs[c].astype(str), prefix=c, drop_first=True).astype(float)
        d.index = obs.index
        parts.append(d)
    return pd.concat(parts, axis=1)


def _make_contrast_matrix(
    contrasts: Sequence[str], design_cols: Sequence[str]
) -> pd.DataFrame:
    """Parse simple ``"A - B"`` / ``"A"`` contrast strings into a matrix.

    Mirrors ``limma::makeContrasts``.  Supported tokens are design column
    names combined with ``+`` and ``-``.
    """
    cols = list(design_cols)
    mat = pd.DataFrame(0.0, index=cols, columns=list(contrasts))
    for con in contrasts:
        expr = con.replace(" ", "")
        # split into +/- terms
        terms = []
        sign = 1.0
        token = ""
        for ch in expr:
            if ch in "+-":
                if token:
                    terms.append((sign, token))
                sign = 1.0 if ch == "+" else -1.0
                token = ""
            else:
                token += ch
        if token:
            terms.append((sign, token))
        for s, name in terms:
            if name not in mat.index:
                raise ValueError(
                    f"Contrast term '{name}' is not a design column "
                    f"({list(mat.index)})"
                )
            mat.loc[name, con] += s
    return mat


def de_analysis(
    experiment: LipidomicsExperiment,
    contrasts: Union[str, Sequence[str], None] = None,
    measure: str = "Area",
    group_col: Optional[str] = None,
    design: Optional[pd.DataFrame] = None,
    coef: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """Moderated-t differential analysis (R ``de_analysis`` / ``de_design``).

    Parameters
    ----------
    experiment
        A :class:`LipidomicsExperiment` (normally normalized + logged).
    contrasts
        One or more contrast strings, e.g. ``"HighFat - Normal"``.  Each
        token must be a level of ``group_col`` (or a design column).
    measure
        Which measure to test.
    group_col
        Sample-annotation column defining the groups.  If omitted, the
        first ``col_data`` column is used.
    design
        Optional explicit design matrix (samples x terms).  When given,
        ``contrasts`` columns are interpreted against it.
    coef
        Alternatively, design column(s) to test directly as coefficients.

    Returns
    -------
    A tidy :class:`pandas.DataFrame` with one row per (contrast, molecule):
    ``contrast, Molecule, Class, total_cl, total_cs, istd,
    logFC, AveExpr, t, P.Value, adj.P.Val``.
    """
    exp = experiment
    if design is None:
        if group_col is None:
            if exp.col_data.shape[1] == 0:
                raise ValueError("Please add clinical data or specify a group column")
            group_col = exp.col_data.columns[0]
        groups = exp.col_data[group_col].astype(str)
        if contrasts is not None:
            con_list = [contrasts] if isinstance(contrasts, str) else list(contrasts)
            # restrict samples to the levels referenced by the contrasts
            referenced = set()
            for con in con_list:
                for tok in con.replace("+", " ").replace("-", " ").split():
                    referenced.add(tok)
            missing = referenced - set(groups.unique())
            if missing:
                raise ValueError(
                    f"These contrast variables are not present in {group_col}: "
                    + ", ".join(sorted(missing))
                )
            keep = groups.index[groups.isin(referenced)]
            exp = exp.subset_samples(keep)
            groups = exp.col_data[group_col].astype(str)
        design = pd.get_dummies(groups, prefix="", prefix_sep="").astype(float)
        design.index = exp.col_data.index

    # rank check
    if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        raise ValueError(
            "Tested variables are redundant (design matrix is not full rank)."
        )

    mat = exp.assay(measure)  # lipids x samples
    mat = mat.reindex(columns=design.index)
    fit = pylimma.lmFit(mat, design)

    coef_map: Dict[str, Union[int, str]] = {}
    if coef is not None:
        coef_list = [coef] if isinstance(coef, str) else list(coef)
        for c in coef_list:
            if c not in design.columns:
                raise ValueError(f"Coefficient '{c}' not in design matrix.")
            coef_map[c] = c
    elif contrasts is not None:
        con_list = [contrasts] if isinstance(contrasts, str) else list(contrasts)
        cmat = _make_contrast_matrix(con_list, design.columns)
        fit = pylimma.contrasts_fit(fit, cmat)
        coef_map = {name: i for i, name in enumerate(con_list)}
    else:
        # ANOVA-style: all non-first design columns
        coef_map = {c: c for c in design.columns[1:]}

    fit = pylimma.eBayes(fit)

    var = exp.adata.var
    ann_cols = [
        c for c in ("Molecule", "Class", "Category", "total_cl", "total_cs", "istd")
        if c in var.columns
    ]
    ann = var[ann_cols].copy()
    ann.index = mat.index

    frames: List[pd.DataFrame] = []
    for con_name, c in coef_map.items():
        tt = pylimma.topTable(fit, coef=c, number=np.inf, sort_by="P")
        tt = tt.rename(columns={"gene": "row_id"}).set_index("row_id")
        merged = ann.join(tt, how="right")
        merged.insert(0, "contrast", con_name)
        merged = merged.reset_index(drop=True)
        frames.append(merged)

    result = pd.concat(frames, ignore_index=True)
    if "Molecule" not in result.columns:
        result["Molecule"] = mat.index.to_numpy()[
            np.repeat(np.arange(len(mat.index)), len(coef_map))
        ][: len(result)]
    result.attrs["measure"] = measure
    return result


def significant_molecules(
    de_results: pd.DataFrame,
    p_cutoff: float = 0.05,
    logfc_cutoff: float = 1.0,
) -> Dict[str, List[str]]:
    """Extract significant lipids per contrast (R ``significant_molecules``).

    Returns a dict mapping each contrast to the list of significant
    molecule names (``adj.P.Val < p_cutoff`` and ``|logFC| > logfc_cutoff``).
    """
    df = de_results
    if "logFC" not in df.columns:
        hits = df[df["adj.P.Val"] < p_cutoff]
    else:
        hits = df[
            (df["adj.P.Val"] < p_cutoff) & (df["logFC"].abs() > logfc_cutoff)
        ]
    out: Dict[str, List[str]] = {}
    for con, sub in hits.groupby("contrast"):
        out[str(con)] = sub["Molecule"].astype(str).tolist()
    return out


def top_lipids(
    de_results: pd.DataFrame,
    top_n: int = 10,
    rank_by: str = "P.Value",
) -> pd.DataFrame:
    """Return the top-``n`` lipids per contrast (R ``top_lipids``).

    Note
    ----
    lipidr's ``top_lipids`` operates on PLS-DA loadings (``mvaResults``).
    Multivariate analysis is deferred to v0.2, so this Python helper
    instead ranks the :func:`de_analysis` result by ``rank_by``
    (``"P.Value"`` ascending, ``"logFC"`` by absolute value).
    """
    frames = []
    for con, sub in de_results.groupby("contrast"):
        if rank_by == "logFC" and "logFC" in sub.columns:
            ordered = sub.reindex(
                sub["logFC"].abs().sort_values(ascending=False).index
            )
        else:
            ordered = sub.sort_values("P.Value")
        frames.append(ordered.head(top_n))
    return pd.concat(frames, ignore_index=True)
