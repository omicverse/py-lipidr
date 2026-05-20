# py-lipidr

Pure-Python port of the Bioconductor **[lipidr](https://bioconductor.org/packages/release/bioc/html/lipidr.html)**
lipidomics analysis toolkit (Mohamed, Molendijk & Hill,
*J. Proteome Res.* 2020, 19(7):2890-2897).

`pylipidr` is a standalone, dependency-light implementation of lipidr's
**computational core**: data import, lipid-name annotation, QC,
normalization, differential analysis and Lipid Set Enrichment Analysis
(LSEA). It does not require R.

| | |
|---|---|
| PyPI / import name | `pylipidr` |
| License | MIT (same as upstream lipidr) |
| Upstream | Bioconductor lipidr 2.20.0 |

## Why this port reuses two existing engines

* **Lipid-name parsing** -> [`pygoslin`](https://github.com/lifs-tools/pygoslin),
  the reference Goslin lipid-name grammar. lipidr's regex-based name
  parser is replaced by `pygoslin`, which is more robust and standards-based.
* **Moderated-t differential analysis** -> [`python-limma`](https://pypi.org/project/python-limma/)
  (`pylimma`). lipidr's `de_analysis` calls limma under the hood in R;
  the Python port calls the published pure-Python limma port instead of
  reimplementing it.

## Install

```bash
pip install pylipidr            # once published
# or, from a checkout:
pip install -e .
```

Dependencies: `numpy`, `scipy`, `pandas`, `anndata`, `pygoslin`,
`python-limma`.

## Quick start

```python
import pylipidr as lp

# 1. read a Skyline CSV export -> a LipidomicsExperiment (AnnData-backed)
exp = lp.read_skyline("A1_data.csv")

# 2. attach clinical / sample metadata
exp = lp.add_sample_annotation(exp, "clin.csv")

# 3. collapse multiple transitions per lipid
exp = lp.summarize_transitions(exp, method="max")

# 4. QC + normalization
exp = lp.filter_by_cv(exp, cv_cutoff=20.0)
exp = lp.normalize_pqn(exp, measure="Area")        # log2 + PQN

# 5. moderated-t differential analysis (limma)
de = lp.de_analysis(exp, "HighFat - Normal", group_col="group")
hits = lp.significant_molecules(de, p_cutoff=0.05, logfc_cutoff=1.0)

# 6. Lipid Set Enrichment Analysis
enr = lp.lsea(de, rank_by="logFC")
sets = lp.significant_lipidsets(enr, p_cutoff=0.05)
```

## The `LipidomicsExperiment`

`LipidomicsExperiment` wraps an `anndata.AnnData` (samples x lipids):

* `.adata.var` -- per-lipid annotations (`Class`, `Category`,
  `total_cl`, `total_cs`, `chains`, `istd`, ...).
* `.adata.obs` -- per-sample clinical data.
* `.adata.X` / `.adata.layers` -- one or more intensity *measures*.
* processing-state flags `is_logged` / `is_normalized` / `is_summarized`
  are stored in `.adata.uns` and toggled with `set_logged` etc.

## What is ported

| lipidr (R) | pylipidr | notes |
|---|---|---|
| `LipidomicsExperiment`, `as_lipidomics_experiment` | `LipidomicsExperiment`, `as_lipidomics_experiment` | AnnData-backed |
| `read_skyline` | `read_skyline` | Skyline CSV export(s) |
| `read_mwTab` | `read_mwtab` | Metabolomics Workbench `mwTab` |
| `read_mw_datamatrix` | `read_mw_datamatrix` | MW data matrix TSV |
| `annotate_lipids` | `annotate_lipids`, `annotate_experiment` | pygoslin-backed |
| `non_parsed_molecules`, `remove_non_parsed_molecules`, `update_molecule_names` | same names | |
| `filter_by_cv` | `filter_by_cv` | CV filter |
| `impute_na` | `impute_na` | `knn` / `min` / `minDet` / `minProb` / `zero` |
| `summarize_transitions` | `summarize_transitions` | `max` / `average` |
| `normalize_pqn` | `normalize_pqn` | probabilistic quotient normalization |
| `normalize_istd` | `normalize_istd` | per-class internal-standard normalization |
| `de_design`, `de_analysis` | `de_design`, `de_analysis` | moderated-t via `pylimma` |
| `significant_molecules` | `significant_molecules` | |
| `top_lipids` | `top_lipids` | ranks DE result (see note below) |
| `gen_lipidsets` | `gen_lipidsets` | by class / chain length / unsaturation |
| `lsea` | `lsea` | preranked GSEA (fgsea-style) |
| `significant_lipidsets` | `significant_lipidsets` | |

## What is NOT ported (deferred to v0.2)

These are deliberately out of scope for v0.1 and are **documented here as
deferred**:

* **`mva`** -- PCA / PLS-DA / OPLS-DA multivariate analysis. omicverse
  already provides multivariate tooling; lipidr's `top_lipids` normally
  operates on `mva` loadings, so the v0.1 `top_lipids` instead ranks the
  `de_analysis` result.
* **All `plot_*` functions** -- `plot_samples`, `plot_molecules`,
  `plot_lipidclass`, `plot_chain_distribution`, `plot_results_volcano`,
  `plot_enrichment`, `plot_trend`, `plot_heatmap`, etc.
* **`use_interactive_graphics`** -- interactive plotly toggling.
* **`fetch_mw_study` / `list_mw_studies`** -- network helpers for the
  Metabolomics Workbench REST API.

## R-parity

`pylipidr` is validated against Bioconductor lipidr 2.20.0 on lipidr's
own bundled Skyline example dataset (`extdata/A1_data.csv` + `clin.csv`),
so both languages analyse identical input. Numbers from
`examples/benchmark.py`:

| step | metric | result |
|---|---|---|
| `annotate_lipids` | lipid-class agreement | **0.99** |
| `normalize_pqn` | Pearson r of normalized values | **1.000** |
| `normalize_istd` | Pearson r of normalized values | **0.997** |
| `de_analysis` | Pearson r of logFC | **1.000** |
| `de_analysis` | Pearson r of p-values | **1.000** |
| `lsea` | Pearson r of enrichment scores | **0.95** |
| `lsea` | Pearson r of p-values | **0.91** |

`lsea` agrees within target tolerance; small differences arise because R
lipidr's `fgsea` uses an adaptive *multilevel* permutation scheme while
`pylipidr` uses a fixed gene-permutation null. The *significantly*
enriched lipid sets agree.

Run the parity suite (skips gracefully if R is unavailable):

```bash
pytest tests/ -v
```

* `tests/test_smoke.py` -- 18 algorithmic tests, no R needed.
* `tests/test_r_parity.py` -- 8 tests vs Bioconductor lipidr.

## Benchmark

```bash
python examples/benchmark.py --runs 2
```

On the bundled example the full Python pipeline runs roughly **8x**
faster than the equivalent R pipeline (mostly by skipping Rscript /
Bioconductor startup). See `examples/compare_R_vs_Python.ipynb`.

## Citation

If you use `pylipidr`, please cite the original lipidr paper:

> Mohamed A, Molendijk J, Hill MM. **lipidr: A Software Tool for Data
> Mining and Analysis of Lipidomics Datasets.** *J. Proteome Res.* 2020,
> 19(7):2890-2897. doi:10.1021/acs.jproteome.0c00082

and, for the reused engines, the Goslin
(Kopczynski et al., *Anal. Chem.* 2020) and limma
(Ritchie et al., *Nucleic Acids Res.* 2015) papers.

## License

MIT -- the same license as upstream lipidr. See `LICENSE`.
