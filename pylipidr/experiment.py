"""LipidomicsExperiment — the central data container.

Port of lipidr's ``LipidomicsExperiment`` S4 class.  In R this is a
``SummarizedExperiment`` subclass carrying multiple assays (e.g. ``Area``,
``Retention.Time``), per-lipid annotations in ``rowData`` and per-sample
clinical data in ``colData``, together with processing-state flags
(``logged`` / ``normalized`` / ``summarized``).

Here we wrap an :class:`anndata.AnnData` with samples as observations
(``adata.obs``) and lipids as variables (``adata.var``).  The primary
measure lives in ``adata.X`` (samples x lipids); extra measures live in
``adata.layers``.  Processing-state flags are stored per-measure in
``adata.uns['lipidr_state']``.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

try:  # anndata is a hard dependency but keep the import error friendly.
    from anndata import AnnData
except Exception as exc:  # pragma: no cover
    raise ImportError("pylipidr requires the 'anndata' package") from exc


_STATE_KEY = "lipidr_state"


class LipidomicsExperiment:
    """Container for a lipidomics dataset.

    Parameters
    ----------
    adata
        An :class:`anndata.AnnData` with samples (rows) x lipids (columns).
        The default measure is ``adata.X``; named measures are
        ``adata.layers``.  The default-measure name is recorded in
        ``adata.uns['lipidr_default_measure']`` (defaults to ``"Area"``).
    """

    def __init__(self, adata: AnnData):
        if not isinstance(adata, AnnData):
            raise TypeError("LipidomicsExperiment expects an AnnData object")
        self.adata = adata
        if _STATE_KEY not in self.adata.uns:
            self.adata.uns[_STATE_KEY] = {}
        if "lipidr_default_measure" not in self.adata.uns:
            self.adata.uns["lipidr_default_measure"] = "Area"

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        measure: str = "Area",
        row_data: Optional[pd.DataFrame] = None,
        col_data: Optional[pd.DataFrame] = None,
    ) -> "LipidomicsExperiment":
        """Build from a lipids x samples DataFrame (``data`` rows = lipids)."""
        mat = data.to_numpy(dtype=float)
        # AnnData stores samples x lipids -> transpose.
        idx = data.index.astype(str)
        var = pd.DataFrame(index=idx)
        if row_data is not None:
            var = row_data.reindex(idx)
        # lipidr's rowData always carries a 'Molecule' column
        if "Molecule" not in var.columns:
            var["Molecule"] = idx.to_numpy()
        obs = pd.DataFrame(index=data.columns.astype(str))
        if col_data is not None:
            obs = col_data.reindex(data.columns.astype(str))
        ad = AnnData(X=mat.T, obs=obs, var=var)
        ad.uns["lipidr_default_measure"] = measure
        return cls(ad)

    # ------------------------------------------------------------------
    # measure access
    # ------------------------------------------------------------------
    @property
    def default_measure(self) -> str:
        return str(self.adata.uns.get("lipidr_default_measure", "Area"))

    def measure_names(self) -> list:
        return [self.default_measure] + list(self.adata.layers.keys())

    def assay(self, measure: Optional[str] = None) -> pd.DataFrame:
        """Return a lipids x samples DataFrame for ``measure`` (R ``assay()``)."""
        measure = measure or self.default_measure
        if measure == self.default_measure:
            mat = np.asarray(self.adata.X, dtype=float)
        elif measure in self.adata.layers:
            mat = np.asarray(self.adata.layers[measure], dtype=float)
        else:
            raise KeyError(f"{measure} is not in the dataset.")
        return pd.DataFrame(
            mat.T, index=self.adata.var_names, columns=self.adata.obs_names
        )

    def set_assay(self, mat: pd.DataFrame, measure: Optional[str] = None) -> None:
        """Write a lipids x samples DataFrame back into ``measure``."""
        measure = measure or self.default_measure
        arr = mat.reindex(
            index=self.adata.var_names, columns=self.adata.obs_names
        ).to_numpy(dtype=float)
        if measure == self.default_measure:
            self.adata.X = arr.T
        else:
            self.adata.layers[measure] = arr.T

    # ------------------------------------------------------------------
    # convenience views
    # ------------------------------------------------------------------
    @property
    def row_data(self) -> pd.DataFrame:
        """Per-lipid annotation table (R ``rowData``)."""
        return self.adata.var

    @property
    def col_data(self) -> pd.DataFrame:
        """Per-sample clinical table (R ``colData``)."""
        return self.adata.obs

    @property
    def molecules(self) -> pd.Index:
        return self.adata.var_names

    @property
    def samples(self) -> pd.Index:
        return self.adata.obs_names

    @property
    def shape(self) -> tuple:
        """(n_lipids, n_samples) — matching R ``dim()`` of the SE."""
        return (self.adata.n_vars, self.adata.n_obs)

    # ------------------------------------------------------------------
    # processing-state flags
    # ------------------------------------------------------------------
    def _state(self, measure: str) -> dict:
        st = self.adata.uns.setdefault(_STATE_KEY, {})
        return st.setdefault(measure, {})

    def is_logged(self, measure: Optional[str] = None) -> bool:
        return bool(self._state(measure or self.default_measure).get("logged", False))

    def is_normalized(self, measure: Optional[str] = None) -> bool:
        return bool(
            self._state(measure or self.default_measure).get("normalized", False)
        )

    def is_summarized(self, measure: Optional[str] = None) -> bool:
        return bool(self.adata.uns.get("lipidr_summarized", False))

    def set_logged(self, value: bool, measure: Optional[str] = None) -> "LipidomicsExperiment":
        self._state(measure or self.default_measure)["logged"] = bool(value)
        return self

    def set_normalized(self, value: bool, measure: Optional[str] = None) -> "LipidomicsExperiment":
        self._state(measure or self.default_measure)["normalized"] = bool(value)
        return self

    def set_summarized(self, value: bool) -> "LipidomicsExperiment":
        self.adata.uns["lipidr_summarized"] = bool(value)
        return self

    # ------------------------------------------------------------------
    # subsetting
    # ------------------------------------------------------------------
    def subset_samples(self, samples: Iterable) -> "LipidomicsExperiment":
        sub = self.adata[list(samples), :].copy()
        return LipidomicsExperiment(sub)

    def subset_molecules(self, molecules: Iterable) -> "LipidomicsExperiment":
        sub = self.adata[:, list(molecules)].copy()
        return LipidomicsExperiment(sub)

    def copy(self) -> "LipidomicsExperiment":
        return LipidomicsExperiment(self.adata.copy())

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n_lip, n_samp = self.shape
        flags = []
        m = self.default_measure
        if self.is_logged(m):
            flags.append("logged")
        if self.is_normalized(m):
            flags.append("normalized")
        if self.is_summarized():
            flags.append("summarized")
        flag_s = (", ".join(flags)) if flags else "raw"
        return (
            f"LipidomicsExperiment: {n_lip} lipids x {n_samp} samples "
            f"[measures={self.measure_names()}; {flag_s}]"
        )


def as_lipidomics_experiment(
    data, measure: str = "Area", **kwargs
) -> LipidomicsExperiment:
    """Coerce input to a :class:`LipidomicsExperiment` (R ``as_lipidomics_experiment``).

    Accepts an existing :class:`LipidomicsExperiment`, an :class:`AnnData`,
    or a lipids x samples :class:`pandas.DataFrame` whose first column may
    be a ``Molecule`` identifier column.
    """
    if isinstance(data, LipidomicsExperiment):
        return data
    if isinstance(data, AnnData):
        return LipidomicsExperiment(data)
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        # identify the molecule / lipid id column (first non-numeric column)
        id_col = None
        if "Molecule" in df.columns:
            id_col = "Molecule"
        else:
            for c in df.columns:
                low = str(c).lower()
                if low in ("lipids", "lipid", "molecule", "index", "name"):
                    id_col = c
                    break
            if id_col is None and not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                id_col = df.columns[0]
        if id_col is not None:
            df = df.set_index(id_col)
        # keep only numeric sample columns
        num = df.apply(pd.to_numeric, errors="coerce")
        return LipidomicsExperiment.from_dataframe(num, measure=measure, **kwargs)
    raise TypeError(f"Cannot coerce {type(data)!r} to LipidomicsExperiment")
