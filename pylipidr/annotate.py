"""Lipid-name annotation, backed by pygoslin.

Port of lipidr's ``annotate_lipids`` / ``non_parsed_molecules`` /
``remove_non_parsed_molecules`` / ``update_molecule_names``.

lipidr parses lipid names (``CLS xx:x/yy:y``) into class / category /
total chain length / total unsaturation features.  Rather than re-implement
that regex machinery, the Python port delegates name parsing to
**pygoslin** (Kopczynski et al. 2020), the reference Goslin lipid-name
grammar.  Each parsed name yields:

* ``Class``           — the lipid class string (e.g. ``PC``, ``Cer``, ``TG``).
* ``Category``        — GP / GL / SP / ST / FA (Goslin lipid category).
* ``total_cl``        — summed acyl-chain carbon count.
* ``total_cs``        — summed double-bond count.
* ``chains``          — per-FA ``"C:DB"`` list (joined with ``;``).
* ``not_parsed``      — True if pygoslin could not parse the name.
* ``istd``            — True if the name looks like an internal standard.
"""
from __future__ import annotations

import re
import warnings
from functools import lru_cache
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

# lipidr's curated internal-standard molecule list (``lipidDefaults$clean_mols``
# rows flagged ``istd``).  Matched case-insensitively after whitespace
# collapse so naming variants (extra spaces) still resolve.
_CURATED_ISTD = {
    "gluccer 18:1/12:0", "lpc 18:1(d7)", "lpe 18:1(d7)", "laccer 18:1/12:0",
    "pc 15:0-18:1(d7)", "pe 15:0-18:1(d7)", "pg 15:0-18:1(d7)",
    "pi 15:0-18:1(d7)", "ps 33:1 (d7)", "sm 18:1 (d9)", "sa1p 17:0",
    "so1p 17:1", "sm 18:1(d9)", "cer 18:1/12:0", "cer1p 18:1/12:0",
    "sa 17:0", "so 17:1", "gluccer d18:1/12:0", "laccer d18:1/12:0",
    "15:0-18:1(d7) pg", "15:0-18:1(d7) pi", "15:0-18:1(d7) pe",
    "18:1-d9 sm", "cer d18:1/c12:0", "ps 33:1 d7", "15:0-18:1(d7) pc",
    "18:1(d7) lyso pc", "18:1(d7) lyso pe",
}

# Deuterium / labelled internal-standard name patterns used by lipidr's
# curated standard list (e.g. "LPC 18:1(d7)", "SM 18:1 (d9)", "SPLASH").
_ISTD_PATTERNS = [
    re.compile(r"\(d\d+\)", re.IGNORECASE),       # (d7), (d9)
    re.compile(r"\bd\d+(?!:)\b", re.IGNORECASE),  # standalone d7 (not d18:1)
    re.compile(r"-d\d+\b", re.IGNORECASE),        # 18:1-d9
    re.compile(r"\(\d+:\d+-d\d+\)", re.IGNORECASE),
    re.compile(r"SPLASH", re.IGNORECASE),
    re.compile(r"\bISTD\b", re.IGNORECASE),
    re.compile(r"\(IS\)", re.IGNORECASE),
]


def _looks_like_istd(name: str) -> bool:
    """Internal-standard detection from the molecule name.

    First checks lipidr's curated 30-name standard list, then falls back
    to deuterium-label / SPLASH / ISTD name patterns.
    """
    norm = re.sub(r"\s+", " ", str(name).strip().lower())
    if norm in _CURATED_ISTD:
        return True
    return any(p.search(name) for p in _ISTD_PATTERNS)


@lru_cache(maxsize=None)
def _get_parser():
    from pygoslin.parser.Parser import LipidParser

    return LipidParser()


# lipidr-specific lipid-class synonyms that pygoslin does not recognise.
# Sphingoid-base / sphingoid-1-phosphate shorthands used by Skyline.
_CLASS_SYNONYMS = {
    "So1P": "S1P",     # sphingosine-1-phosphate
    "Sa1P": "S1P",     # sphinganine-1-phosphate
    "So": "SPB",       # sphingosine
    "Sa": "SPB",       # sphinganine
    "Sph": "SPB",      # sphingosine base
}
_CLASS_SYN_RE = re.compile(
    r"^(" + "|".join(re.escape(k) for k in _CLASS_SYNONYMS) + r")\b"
)


def _clean_name(name: str) -> str:
    """Normalise a lipid name so pygoslin can parse it (R ``.clean_molecule_name``).

    Handles common Skyline / lipidr quirks: trailing adduct/polarity tags
    (`` NEG``, `` POS``), `chains CLASS` ordering, deuterium labels,
    plasmalogen ``p``/``e`` chain markers and sphingoid-base shorthands.
    """
    s = str(name).strip()
    # drop trailing polarity / adduct annotations
    s = re.sub(r"\s+(NEG|POS|\[[^\]]*\])\s*$", "", s, flags=re.IGNORECASE)
    # strip deuterium-label parentheses, e.g. "(d7)"
    s = re.sub(r"\s*\(d\d+\)", "", s)
    # "33:1p" plasmalogen marker -> standard ether notation handled below
    # "CLASS d 18:0" -> "CLASS 18:0" (the 'd' is a chain-type hint)
    s = re.sub(r"\b([dt])\s+(\d)", r"\2", s)
    # reorder "15:0-18:1 PE" -> "PE 15:0-18:1"
    m = re.match(r"^([\d:/\-]+(?:\([^)]*\))?)\s+([A-Za-z][A-Za-z0-9]*)$", s)
    if m and not re.match(r"^\d", m.group(2)):
        s = f"{m.group(2)} {m.group(1)}"
    # sphingoid-base shorthand classes
    syn = _CLASS_SYN_RE.match(s)
    if syn:
        s = _CLASS_SYNONYMS[syn.group(1)] + s[syn.end():]
    return s.strip()


def _fa_double_bonds(fa) -> int:
    """Robustly pull the double-bond count off a pygoslin FattyAcid."""
    if hasattr(fa, "get_double_bonds"):
        try:
            return int(fa.get_double_bonds())
        except Exception:
            pass
    db = getattr(fa, "double_bonds", 0)
    if isinstance(db, int):
        return db
    return int(getattr(db, "num_double_bonds", 0))


# Map pygoslin class strings back to lipidr's class conventions.
_CLASS_OUT = {
    "S1P": "SPH",   # sphingoid-1-phosphates reported as SPH by lipidr
    "SPB": "SPH",   # sphingoid bases
    "SPBP": "SPH",
    "SPH": "SPH",
}


def _normalise_class(cls: str) -> str:
    """Map a pygoslin class string to lipidr's class naming."""
    return _CLASS_OUT.get(str(cls), str(cls))


def _try_parse(text: str):
    """Attempt a pygoslin parse; return the lipid object or ``None``."""
    try:
        return _get_parser().parse(text).lipid
    except Exception:
        return None


@lru_cache(maxsize=8192)
def _parse_one(name: str) -> tuple:
    """Parse a single lipid name; cached.  Returns a tuple of features.

    Tries the raw name first, then a cleaned variant for Skyline / lipidr
    naming quirks pygoslin does not accept verbatim.
    """
    name = str(name).strip()
    try:
        lipid = _try_parse(name)
        if lipid is None:
            lipid = _try_parse(_clean_name(name))
        if lipid is None:
            raise ValueError("unparseable")
        hg = lipid.headgroup
        cls = _normalise_class(hg.headgroup)
        category = hg.lipid_category.name  # GP / GL / SP / ST / FA / UNDEFINED
        info = lipid.info
        total_cl = None
        total_cs = None
        if info is not None:
            total_cl = getattr(info, "num_carbon", None)
            total_cs = _fa_double_bonds(info)
        chains: List[str] = []
        cl_sum, cs_sum = 0, 0
        seen_fa = False
        for fa in (lipid.fa_list or []):
            c = int(getattr(fa, "num_carbon", 0))
            db = _fa_double_bonds(fa)
            if c > 0:
                chains.append(f"{c}:{db}")
                cl_sum += c
                cs_sum += db
                seen_fa = True
        if (total_cl is None or total_cl == 0) and seen_fa:
            total_cl = cl_sum
        if (total_cs is None) and seen_fa:
            total_cs = cs_sum
        return (
            str(cls),
            str(category),
            (int(total_cl) if total_cl is not None else np.nan),
            (int(total_cs) if total_cs is not None else np.nan),
            ";".join(chains),
            False,  # not_parsed
        )
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan, "", True)


def annotate_lipids(
    molecules: Iterable[str],
    no_match: str = "warn",
    istd: Optional[Iterable[bool]] = None,
) -> pd.DataFrame:
    """Annotate lipid molecule names (R ``annotate_lipids``).

    Parameters
    ----------
    molecules
        Iterable of lipid name strings.
    no_match
        ``"warn"`` (default), ``"remove"`` or ``"ignore"`` — controls what
        happens to names pygoslin cannot parse.
    istd
        Optional explicit boolean mask flagging internal standards.  If
        omitted, ISTDs are detected heuristically from the name.

    Returns
    -------
    pandas.DataFrame indexed by ``Molecule`` with columns
    ``Class, Category, total_cl, total_cs, chains, not_parsed, istd``.
    """
    if no_match not in ("warn", "remove", "ignore"):
        raise ValueError("no_match must be 'warn', 'remove' or 'ignore'")
    mols = [str(m) for m in molecules]
    rows = [_parse_one(m) for m in mols]
    df = pd.DataFrame(
        rows,
        columns=["Class", "Category", "total_cl", "total_cs", "chains", "not_parsed"],
        index=pd.Index(mols, name="Molecule"),
    )
    if istd is not None:
        df["istd"] = list(istd)
    else:
        df["istd"] = [_looks_like_istd(m) for m in mols]

    not_parsed = df["not_parsed"].to_numpy(dtype=bool)
    if not_parsed.any():
        bad = df.index[not_parsed].tolist()
        if no_match == "warn":
            warnings.warn(
                "Some lipid names couldn't be parsed by pygoslin: "
                + ", ".join(bad[:20])
                + (" ..." if len(bad) > 20 else "")
            )
        if no_match == "remove":
            df = df.loc[~not_parsed]
    return df


def non_parsed_molecules(annotations: pd.DataFrame) -> List[str]:
    """Return the molecule names that pygoslin failed to parse."""
    return annotations.index[annotations["not_parsed"].to_numpy(dtype=bool)].tolist()


def remove_non_parsed_molecules(experiment, no_match: str = "remove"):
    """Drop unparseable lipids from a :class:`LipidomicsExperiment`."""
    from .experiment import LipidomicsExperiment

    ann = annotate_lipids(experiment.molecules, no_match="ignore")
    keep = ann.index[~ann["not_parsed"].to_numpy(dtype=bool)]
    return experiment.subset_molecules(keep)


def update_molecule_names(experiment, old: Iterable[str], new: Iterable[str]):
    """Rename molecules in a :class:`LipidomicsExperiment` (R ``update_molecule_names``)."""
    mapping = dict(zip([str(o) for o in old], [str(n) for n in new]))
    exp = experiment.copy()
    new_names = [mapping.get(str(m), str(m)) for m in exp.adata.var_names]
    exp.adata.var_names = pd.Index(new_names)
    return exp


def annotate_experiment(experiment, no_match: str = "warn"):
    """Attach pygoslin annotations to a :class:`LipidomicsExperiment`'s ``.var``.

    This is the in-place equivalent of how lipidr stores annotation columns
    in ``rowData`` after :func:`read_skyline`.  Annotation is keyed on the
    ``Molecule`` column when present (Skyline data uses transition row
    ids), otherwise on the variable names.
    """
    var = experiment.adata.var
    if "Molecule" in var.columns:
        names = var["Molecule"].astype(str)
    else:
        names = pd.Series(experiment.molecules.astype(str), index=var.index)
    # parse the unique lipid names once
    uniq = list(pd.unique(names))
    ann_u = annotate_lipids(uniq, no_match="ignore")
    ann = ann_u.reindex(names.to_numpy())
    ann.index = var.index
    # preserve any pre-existing istd column from the importer
    pre_istd = var["istd"] if "istd" in var.columns else None
    for col in ["Class", "Category", "total_cl", "total_cs", "chains", "not_parsed"]:
        var[col] = ann[col].to_numpy()
    if pre_istd is not None:
        var["istd"] = pre_istd.to_numpy()
    else:
        var["istd"] = ann["istd"].to_numpy()
    if no_match == "warn":
        bad = non_parsed_molecules(ann)
        if bad:
            warnings.warn(
                "Some lipid names couldn't be parsed by pygoslin: "
                + ", ".join([str(b) for b in bad[:20]])
            )
    return experiment
