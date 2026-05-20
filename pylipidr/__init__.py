"""pylipidr: Pure-Python port of Bioconductor lipidr.

A standalone, dependency-light port of the lipidomics analysis toolkit
``lipidr`` (Mohamed, Molendijk & Hill, *J. Proteome Res.* 2020,
19(7):2890-2897).  It covers the computational core of lipidr: import,
annotation, QC, normalization, differential analysis and Lipid Set
Enrichment Analysis (LSEA).

Reused engines
--------------
* **pygoslin** — lipid-name parsing (replaces lipidr's regex parser).
* **python-limma** (``pylimma``) — moderated-t differential analysis.

Core data structure
-------------------
* :class:`LipidomicsExperiment` — an AnnData-backed samples x lipids
  container carrying lipid annotations in ``.var`` and processing-state
  flags (logged / normalized / summarized).
* :func:`as_lipidomics_experiment` — coerce DataFrame / AnnData input.

I/O
---
* :func:`read_skyline`        — Skyline CSV export(s).
* :func:`read_mwtab`          — Metabolomics Workbench ``mwTab``.
* :func:`read_mw_datamatrix`  — Metabolomics Workbench data matrix.
* :func:`add_sample_annotation` — attach clinical metadata.

Annotation
----------
* :func:`annotate_lipids` / :func:`annotate_experiment` — pygoslin-backed
  class / category / chain annotation.
* :func:`non_parsed_molecules`, :func:`remove_non_parsed_molecules`,
  :func:`update_molecule_names`.

QC / preprocessing
------------------
* :func:`filter_by_cv`, :func:`impute_na`, :func:`summarize_transitions`.

Normalization
-------------
* :func:`normalize_istd` — internal-standard normalization.
* :func:`normalize_pqn`  — probabilistic quotient normalization.

Differential analysis
---------------------
* :func:`de_design`, :func:`de_analysis`, :func:`significant_molecules`,
  :func:`top_lipids`.

Lipid-set enrichment
--------------------
* :func:`gen_lipidsets`, :func:`lsea`, :func:`significant_lipidsets`.

Deferred to v0.2
----------------
``mva`` (PCA / PLS-DA), all ``plot_*`` helpers,
``use_interactive_graphics`` and the network-heavy
``fetch_mw_study`` / ``list_mw_studies`` are *not* ported in v0.1.

Quick-start
-----------
>>> import pylipidr as lp
>>> exp = lp.read_skyline("A1_data.csv")
>>> exp = lp.add_sample_annotation(exp, "clin.csv")
>>> exp = lp.normalize_pqn(exp, measure="Area")
>>> de = lp.de_analysis(exp, "HighFat - Normal", group_col="group")
>>> hits = lp.significant_molecules(de)
>>> enr = lp.lsea(de, rank_by="logFC")
"""
from __future__ import annotations

from .annotate import (
    annotate_experiment,
    annotate_lipids,
    non_parsed_molecules,
    remove_non_parsed_molecules,
    update_molecule_names,
)
from .de import (
    de_analysis,
    de_design,
    significant_molecules,
    top_lipids,
)
from .experiment import LipidomicsExperiment, as_lipidomics_experiment
from .io import (
    add_sample_annotation,
    read_mw_datamatrix,
    read_mwtab,
    read_skyline,
)
from .lsea import gen_lipidsets, lsea, significant_lipidsets
from .normalize import normalize_istd, normalize_pqn
from .qc import filter_by_cv, impute_na, summarize_transitions

__version__ = "0.1.0"

__all__ = [
    # data structure
    "LipidomicsExperiment",
    "as_lipidomics_experiment",
    # I/O
    "read_skyline",
    "read_mwtab",
    "read_mw_datamatrix",
    "add_sample_annotation",
    # annotation
    "annotate_lipids",
    "annotate_experiment",
    "non_parsed_molecules",
    "remove_non_parsed_molecules",
    "update_molecule_names",
    # QC
    "filter_by_cv",
    "impute_na",
    "summarize_transitions",
    # normalization
    "normalize_istd",
    "normalize_pqn",
    # differential analysis
    "de_design",
    "de_analysis",
    "significant_molecules",
    "top_lipids",
    # LSEA
    "gen_lipidsets",
    "lsea",
    "significant_lipidsets",
]
