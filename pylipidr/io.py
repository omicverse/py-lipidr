"""Data importers — Skyline CSV and Metabolomics Workbench.

Ports of lipidr's ``read_skyline``, ``read_mwTab`` and
``read_mw_datamatrix``.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .annotate import annotate_experiment
from .experiment import LipidomicsExperiment, as_lipidomics_experiment

# Skyline column-name synonyms (lipidr's ``col_defs``).
_MOLECULE_COLS = [
    "Molecule", "Molecule Name", "Peptide", "PeptideModifiedSequence",
    "Lipid", "Compound",
]
_SAMPLE_COLS = ["Replicate", "Replicate Name", "Sample", "Sample Name", "FileName"]
_CLASS_COLS = ["Protein", "Protein Name", "Molecule List", "Class"]
_INTENSITY_COLS = ["Area", "Height", "Total Area", "Area Normalized"]


def _first_present(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lookup = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def _read_skyline_file(path_or_df) -> pd.DataFrame:
    """Read one Skyline export into a long (tidy) DataFrame."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)
    df = df.replace("#N/A", np.nan)
    if df.shape[0] == 0:
        raise ValueError("Skyline file does not contain any data.")

    mol_col = _first_present(df.columns, _MOLECULE_COLS)
    if mol_col is None:
        raise ValueError(
            "Could not find a molecule/lipid column in the Skyline file."
        )
    df = df.rename(columns={mol_col: "Molecule"})
    samp_col = _first_present(df.columns, _SAMPLE_COLS)
    if samp_col is None:
        raise ValueError("Could not find a sample/replicate column.")
    df = df.rename(columns={samp_col: "Sample"})

    cls_col = _first_present(df.columns, _CLASS_COLS)
    if cls_col is not None and cls_col != "Class":
        df = df.rename(columns={cls_col: "Class"})

    intensity_cols = [c for c in _INTENSITY_COLS if c in df.columns]
    if not intensity_cols:
        # fall back: any numeric column that is not metadata
        meta = {"Molecule", "Sample", "Class"}
        intensity_cols = [
            c for c in df.columns
            if c not in meta and pd.api.types.is_numeric_dtype(df[c])
        ]
    df["Molecule"] = df["Molecule"].astype(str)
    df["Sample"] = df["Sample"].astype(str)
    for c in intensity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_skyline(
    files: Union[str, pd.DataFrame, Sequence],
    measure: str = "Area",
) -> LipidomicsExperiment:
    """Read Skyline CSV export(s) into a :class:`LipidomicsExperiment`.

    Port of lipidr's ``read_skyline``.  Each input file is a *long* table
    with one row per transition (Molecule x Sample); the importer pivots
    to a lipids x samples matrix.  Multiple transitions of the same lipid
    are kept as separate rows tagged with a ``TransitionId`` and can be
    collapsed later with :func:`pylipidr.summarize_transitions`.

    Parameters
    ----------
    files
        A path, a DataFrame, or a list of either.
    measure
        Which Skyline intensity column to treat as the primary measure.
    """
    if isinstance(files, (str, pd.DataFrame)):
        files = [files]
    frames = []
    for f in files:
        sub = _read_skyline_file(f)
        src = f if isinstance(f, str) else "dataset"
        sub["filename"] = src
        frames.append(sub)
    long_df = pd.concat(frames, ignore_index=True)

    # Build a stable per-transition row id: a molecule may have several
    # transitions (rows) per sample -> rank within (filename, Sample).
    long_df["TransitionId"] = (
        long_df.groupby(["filename", "Sample"]).cumcount()
    )

    measures = [c for c in _INTENSITY_COLS if c in long_df.columns]
    if measure not in measures:
        measures = [measure] + measures if measure in long_df.columns else measures
    if not measures:
        raise ValueError("No intensity columns found in Skyline file(s).")

    # transition-level row key: Molecule + TransitionId keeps multi-transition
    transition_keys = (
        long_df[["Molecule", "TransitionId", "filename"]]
        .drop_duplicates(subset=["Molecule", "TransitionId"])
        .reset_index(drop=True)
    )
    transition_keys["row_id"] = (
        transition_keys["Molecule"].astype(str)
        + "_t"
        + transition_keys["TransitionId"].astype(str)
    )
    long_df = long_df.merge(
        transition_keys[["Molecule", "TransitionId", "row_id"]],
        on=["Molecule", "TransitionId"],
        how="left",
    )

    samples = list(pd.unique(long_df["Sample"]))
    row_ids = list(transition_keys["row_id"])

    layers = {}
    for m in measures:
        mat = long_df.pivot_table(
            index="row_id", columns="Sample", values=m, aggfunc="first"
        ).reindex(index=row_ids, columns=samples)
        layers[m] = mat

    row_data = transition_keys.set_index("row_id")[["Molecule", "TransitionId"]].copy()
    # carry per-transition class if present
    if "Class" in long_df.columns:
        cls_map = (
            long_df.drop_duplicates("row_id").set_index("row_id")["Class"]
        )
        row_data["Class_skyline"] = cls_map.reindex(row_ids).to_numpy()

    from anndata import AnnData

    primary = layers[measure]
    ad = AnnData(
        X=primary.to_numpy(dtype=float).T,
        obs=pd.DataFrame(index=pd.Index(samples, name="Sample")),
        var=row_data,
    )
    ad.var_names = pd.Index(row_ids)
    for m, mat in layers.items():
        if m == measure:
            continue
        ad.layers[m] = mat.to_numpy(dtype=float).T
    ad.uns["lipidr_default_measure"] = measure
    exp = LipidomicsExperiment(ad)
    annotate_experiment(exp, no_match="warn")
    return exp


def add_sample_annotation(
    experiment: LipidomicsExperiment, clinical: Union[str, pd.DataFrame]
) -> LipidomicsExperiment:
    """Attach clinical / sample metadata (R ``add_sample_annotation``).

    ``clinical`` must have a ``Sample`` column (or index) matching the
    experiment's sample names.
    """
    if isinstance(clinical, str):
        clin = pd.read_csv(clinical)
    else:
        clin = clinical.copy()
    if "Sample" in clin.columns:
        clin = clin.set_index("Sample")
    clin.index = clin.index.astype(str)
    exp = experiment.copy()
    aligned = clin.reindex(exp.adata.obs_names)
    for col in aligned.columns:
        exp.adata.obs[col] = aligned[col].to_numpy()
    return exp


def read_mwtab(path: str, measure: str = "Area") -> LipidomicsExperiment:
    """Read a Metabolomics Workbench ``mwTab`` file (R ``read_mwTab``).

    Parses the ``MS_METABOLITE_DATA`` block (data matrix) and the
    ``SUBJECT_SAMPLE_FACTORS`` block (sample annotations).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    # --- data matrix block --------------------------------------------
    data_rows: List[List[str]] = []
    in_data = False
    header: Optional[List[str]] = None
    for ln in lines:
        if "MS_METABOLITE_DATA_START" in ln:
            in_data = True
            header = None
            continue
        if "MS_METABOLITE_DATA_END" in ln:
            in_data = False
            continue
        if in_data:
            parts = ln.split("\t")
            if header is None:
                header = parts
                continue
            if parts and parts[0] not in ("Factors", "Samples"):
                data_rows.append(parts)
    if header is None or not data_rows:
        raise ValueError("No MS_METABOLITE_DATA block found in mwTab file.")

    ncol = len(header)
    fixed = []
    for r in data_rows:
        r = (r + [""] * ncol)[:ncol]
        fixed.append(r)
    mat = pd.DataFrame(fixed, columns=header)
    mat = mat.rename(columns={mat.columns[0]: "Molecule"})
    mat = mat.set_index("Molecule")
    mat = mat.apply(pd.to_numeric, errors="coerce")

    # --- subject / sample factors -------------------------------------
    factors: dict = {}
    for ln in lines:
        if ln.startswith("SUBJECT_SAMPLE_FACTORS"):
            parts = ln.split("\t")
            # SUBJECT_SAMPLE_FACTORS <subject> <sample> <factors>
            if len(parts) >= 4:
                sample = parts[2].strip()
                fac = parts[3].strip()
                kv = {}
                for tok in re.split(r"[|;]", fac):
                    if ":" in tok:
                        k, v = tok.split(":", 1)
                        kv[k.strip()] = v.strip()
                factors[sample] = kv

    exp = as_lipidomics_experiment(mat.reset_index(), measure=measure)
    if factors:
        clin = pd.DataFrame.from_dict(factors, orient="index")
        clin.index.name = "Sample"
        aligned = clin.reindex(exp.adata.obs_names)
        for col in aligned.columns:
            exp.adata.obs[col] = aligned[col].to_numpy()
    annotate_experiment(exp, no_match="ignore")
    return exp


def read_mw_datamatrix(path: str, measure: str = "Area") -> LipidomicsExperiment:
    """Read a Metabolomics Workbench *data matrix* TSV (R ``read_mw_datamatrix``).

    The first data row holds per-sample factors; remaining rows are
    metabolite intensities.  Adduct / stereochemistry suffixes are
    stripped from the molecule names so pygoslin can parse them.
    """
    raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if raw.shape[0] < 2:
        raise ValueError("mw datamatrix must have a factors row plus data rows.")
    raw = raw.rename(columns={raw.columns[0]: "Molecule"})

    factor_row = raw.iloc[0]
    data = raw.iloc[1:].copy()
    original_names = data["Molecule"].astype(str).tolist()

    # clean molecule names (mirrors lipidr's regex chain)
    cleaned = []
    for nm in original_names:
        s = re.sub(r" \[.*$", "", nm)                       # drop adduct
        s = re.sub(r"\((\d+[ZE]\.*)+\)", "", s)             # drop stereo
        s = re.sub(r"^(\w+)-", r"\1", s)                    # leading prefix
        s = re.sub(r"\d+:\d+ \(", "(", s)                   # collapse
        cleaned.append(s.strip())
    data["Molecule"] = cleaned

    sample_cols = [c for c in data.columns if c != "Molecule"]
    num = data.set_index("Molecule")[sample_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    num.index = pd.Index(original_names, name="Molecule")

    exp = LipidomicsExperiment.from_dataframe(num, measure=measure)
    # sample factors
    clin = {}
    for c in sample_cols:
        clin[c] = factor_row.get(c, "")
    clin_df = pd.DataFrame.from_dict(clin, orient="index", columns=["Factors"])
    # try to unnest key:value | key:value
    parsed = {}
    for samp, fac in clin_df["Factors"].items():
        kv = {}
        for tok in re.split(r"[|;]", str(fac)):
            if ":" in tok:
                k, v = tok.split(":", 1)
                kv[k.strip()] = v.strip()
        parsed[samp] = kv or {"Factors": str(fac)}
    parsed_df = pd.DataFrame.from_dict(parsed, orient="index")
    aligned = parsed_df.reindex(exp.adata.obs_names)
    for col in aligned.columns:
        exp.adata.obs[col] = aligned[col].to_numpy()
    annotate_experiment(exp, no_match="ignore")
    return exp
