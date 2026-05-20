"""Lipid Set Enrichment Analysis (LSEA).

Ports of lipidr's ``gen_lipidsets``, ``lsea`` and
``significant_lipidsets``.

lipidr's ``lsea`` is a preranked GSEA (it calls ``fgsea::fgsea`` under the
hood) over lipid sets generated from the annotations.  The enrichment
score is the classic Subramanian et al. (2005) running-sum statistic; the
permutation p-value follows the fgsea / GSEA gene-permutation scheme.
Lipid sets are built by:

* ``Class``     — one set per lipid class.
* ``total_cl``  — one set per total acyl-chain length.
* ``total_cs``  — one set per total number of double bonds.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .annotate import annotate_lipids


def gen_lipidsets(
    molecules, min_size: int = 2
) -> Dict[str, List[str]]:
    """Generate lipid sets from annotations (R ``gen_lipidsets``).

    Parameters
    ----------
    molecules
        Either an iterable of lipid names, or a DataFrame already carrying
        ``Molecule``, ``Class``, ``total_cl``, ``total_cs`` columns (e.g. a
        :func:`pylipidr.de_analysis` result).
    min_size
        Minimum set size to keep (lipidr default 2).

    Returns
    -------
    dict mapping ``"<collection>_<value>"`` (e.g. ``"Class_PC"``,
    ``"total_cl_34"``, ``"total_cs_2"``) to lists of molecule names.
    """
    if isinstance(molecules, pd.DataFrame) and {
        "Molecule", "Class", "total_cl", "total_cs"
    }.issubset(molecules.columns):
        df = molecules[["Molecule", "Class", "total_cl", "total_cs"]].copy()
        if "istd" in molecules.columns:
            df["istd"] = molecules["istd"].to_numpy()
    else:
        ann = annotate_lipids(list(molecules), no_match="ignore")
        df = ann.reset_index()[["Molecule", "Class", "total_cl", "total_cs", "istd"]]

    if "istd" in df.columns:
        df = df[~df["istd"].fillna(False).astype(bool)]
    clean = df.drop_duplicates(subset=["Molecule"])

    sets: Dict[str, List[str]] = {}
    for collection in ("Class", "total_cl", "total_cs"):
        for value, sub in clean.groupby(collection):
            if pd.isna(value):
                continue
            if collection == "Class":
                key = f"Class_{value}"
            else:
                # integer-format numeric collections
                try:
                    key = f"{collection}_{int(value)}"
                except (TypeError, ValueError):
                    key = f"{collection}_{value}"
            members = sub["Molecule"].astype(str).tolist()
            if len(members) >= min_size:
                sets[key] = members

    # drop sets that contain every molecule
    all_mols = set(clean["Molecule"].astype(str))
    sets = {k: v for k, v in sets.items() if set(v) != all_mols or len(all_mols) <= min_size}
    return sets


# ---------------------------------------------------------------------
# preranked GSEA core (fgsea-style)
# ---------------------------------------------------------------------
def _enrichment_score(
    ranks: np.ndarray, hits: np.ndarray, weight: float = 1.0
) -> float:
    """Classic Subramanian running-sum enrichment score.

    ``ranks`` are the (sorted, descending) statistics; ``hits`` is a
    boolean mask of set membership in that order.  ``weight=1`` reproduces
    fgsea's default weighted statistic.
    """
    n = len(ranks)
    n_hit = int(hits.sum())
    if n_hit == 0 or n_hit == n:
        return 0.0
    if weight == 0:
        nr = float(n_hit)
        hit_inc = np.where(hits, 1.0 / nr, 0.0)
    else:
        w = np.abs(ranks) ** weight
        nr = float(np.sum(w[hits]))
        if nr == 0:
            return 0.0
        hit_inc = np.where(hits, w / nr, 0.0)
    miss_dec = np.where(hits, 0.0, 1.0 / (n - n_hit))
    running = np.cumsum(hit_inc - miss_dec)
    max_es = running.max()
    min_es = running.min()
    return float(max_es if abs(max_es) >= abs(min_es) else min_es)


def _fgsea(
    pathways: Dict[str, List[str]],
    stats: pd.Series,
    min_size: int = 2,
    max_size: Optional[int] = None,
    nperm: int = 10000,
    weight: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Preranked GSEA over ``pathways`` (mirrors ``fgsea::fgsea``).

    Returns a DataFrame with ``pathway, pval, padj, ES, NES, size,
    leadingEdge``.
    """
    stats = stats.dropna().astype(float)
    stats = stats.sort_values(ascending=False)
    ranks = stats.to_numpy()
    universe = list(stats.index.astype(str))
    pos = {m: i for i, m in enumerate(universe)}
    n = len(universe)
    rng = np.random.default_rng(seed)

    rows = []
    for name, members in pathways.items():
        idx = np.array([pos[m] for m in members if m in pos], dtype=int)
        size = int(idx.size)
        if size < min_size:
            continue
        if max_size is not None and size > max_size:
            continue
        hits = np.zeros(n, dtype=bool)
        hits[idx] = True
        es = _enrichment_score(ranks, hits, weight)

        # gene-permutation null
        null = np.empty(nperm, dtype=float)
        for p in range(nperm):
            perm = np.zeros(n, dtype=bool)
            perm[rng.choice(n, size=size, replace=False)] = True
            null[p] = _enrichment_score(ranks, perm, weight)

        if es >= 0:
            pos_null = null[null >= 0]
            pval = (np.sum(pos_null >= es) + 1) / (len(pos_null) + 1)
            mean_pos = pos_null.mean() if pos_null.size else np.nan
            nes = es / mean_pos if mean_pos and mean_pos != 0 else np.nan
        else:
            neg_null = null[null < 0]
            pval = (np.sum(neg_null <= es) + 1) / (len(neg_null) + 1)
            mean_neg = np.abs(neg_null).mean() if neg_null.size else np.nan
            nes = es / mean_neg if mean_neg and mean_neg != 0 else np.nan

        # leading edge: members up to the peak of the running sum
        if weight == 0:
            w = np.where(hits, 1.0, 0.0)
        else:
            w = np.where(hits, np.abs(ranks) ** weight, 0.0)
        nr = w.sum()
        hit_inc = w / nr if nr else w
        miss_dec = np.where(hits, 0.0, 1.0 / (n - size))
        running = np.cumsum(hit_inc - miss_dec)
        if es >= 0:
            peak = int(np.argmax(running))
            le_idx = idx[idx <= peak]
        else:
            peak = int(np.argmin(running))
            le_idx = idx[idx >= peak]
        leading = [universe[i] for i in sorted(le_idx)]

        rows.append(
            {
                "pathway": name,
                "pval": float(pval),
                "ES": float(es),
                "NES": float(nes),
                "size": size,
                "leadingEdge": leading,
            }
        )

    res = pd.DataFrame(rows)
    if res.empty:
        res["padj"] = []
        return res[["pathway", "pval", "padj", "ES", "NES", "size", "leadingEdge"]]
    # Benjamini-Hochberg
    res = res.sort_values("pval").reset_index(drop=True)
    m = len(res)
    ranks_bh = np.arange(1, m + 1)
    padj = (res["pval"].to_numpy() * m / ranks_bh)
    padj = np.minimum.accumulate(padj[::-1])[::-1]
    res["padj"] = np.clip(padj, 0, 1)
    return res[["pathway", "pval", "padj", "ES", "NES", "size", "leadingEdge"]]


def lsea(
    de_results: pd.DataFrame,
    rank_by: str = "logFC",
    min_size: int = 2,
    nperm: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Lipid Set Enrichment Analysis (R ``lsea``).

    Runs a preranked GSEA, per contrast, over the lipid sets generated by
    :func:`gen_lipidsets`, ranking molecules by the chosen DE statistic.

    Parameters
    ----------
    de_results
        Output of :func:`pylipidr.de_analysis`.
    rank_by
        ``"logFC"`` (default), ``"P.Value"`` or ``"adj.P.Val"`` — the
        statistic used to rank lipids.
    min_size
        Minimum lipid-set size.
    nperm
        Permutations for the null distribution.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    A :class:`pandas.DataFrame`: ``contrast, set, pval, padj, ES, NES,
    size, leadingEdge``, sorted by ``padj``.
    """
    if rank_by not in ("logFC", "P.Value", "adj.P.Val"):
        raise ValueError("rank_by must be 'logFC', 'P.Value' or 'adj.P.Val'")
    sets = gen_lipidsets(de_results, min_size=min_size)
    if not sets:
        raise ValueError(
            "Unable to generate lipid sets, possibly because of missing "
            "annotations."
        )

    frames: List[pd.DataFrame] = []
    for con, sub in de_results.groupby("contrast"):
        # one stat per molecule: keep the first after sorting descending
        ranked = (
            sub.sort_values(rank_by, ascending=False)
            .groupby("Molecule", sort=False)
            .first()
        )
        stats = ranked[rank_by]
        res = _fgsea(
            sets, stats, min_size=min_size, nperm=nperm, seed=seed
        )
        res.insert(0, "contrast", str(con))
        frames.append(res)

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"pathway": "set"})
    out = out.sort_values("padj").reset_index(drop=True)
    out.attrs["rank_by"] = rank_by
    out.attrs["sets"] = sets
    return out


def significant_lipidsets(
    enrich_results: pd.DataFrame,
    p_cutoff: float = 0.05,
    size_cutoff: int = 2,
) -> Dict[str, List[str]]:
    """Extract significantly enriched lipid sets (R ``significant_lipidsets``).

    Returns a dict mapping each contrast to its enriched set names
    (``padj < p_cutoff`` and ``size > size_cutoff``).
    """
    hits = enrich_results[
        (enrich_results["padj"] < p_cutoff)
        & (enrich_results["size"] > size_cutoff)
    ]
    out: Dict[str, List[str]] = {}
    for con, sub in hits.groupby("contrast"):
        out[str(con)] = sub["set"].astype(str).tolist()
    return out
