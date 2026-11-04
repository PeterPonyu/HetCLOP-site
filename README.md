# Calibration gains from added dispersion need not carry over to marker recovery in single-cell count decoders

Code, result records and figure sources for the article of the same title. This version is archived at
https://doi.org/10.5281/zenodo.23940846 together with the analysed corpus (`hetclop-corpus-3.0.3`, a separate download);
https://doi.org/10.5281/zenodo.21524011 always resolves to the latest version.

## Contents

- `src/`: decoders (deterministic, Gaussian, Poisson, negative-binomial and zero-inflated heads), evaluation
  metrics (expressed-gene variance calibration, energy distance, marker recall, rare-type F1), the tightness
  diagnostic, the shared gene panel and ortholog mapping.
- `experiments/`: the analyses of the original submission, one folder per analysis, with their scripts and result
  records (`results/`). Folder names are dated run identifiers.
- `revisions/major-revision/experiments/`: the analyses added in revision. Each folder has `design.json`, fixed and
  hashed (`design.sha256`) before the analysis was run, the scripts, a `verification.json` record and the result
  tables. All seeds and fits are kept.
- `revisions/major-revision/figures/` and `revisions/major-revision/manuscript/figs/`: figure and table sources,
  the scripts that compose the figures from saved results, the macro files holding every number reported in the
  article, and the rendered figures. `FIGURE-MAP.md` maps the submitted figures to the revised ones.
- `revisions/major-revision/evidence/`: generators of the number macros and supplementary tables, and verifiers.
- `ai-disclosure/`: the AI-use statement and, in `prompts/`, an English translation of the prompts used for the
  AI-assisted code work of the original submission.
- `tests/`: unit tests for the library.

## Requirements

Python 3.11 or later with the packages in `requirements.txt`, plus scikit-learn and statsmodels; scvi-tools 1.4.3
for the scVI analyses; a CUDA GPU for model fitting. The figure builder needs a TeX distribution with pgfplots,
`pdfcrop` and PyMuPDF; the UMAP figure needs R with ggplot2 and jsonlite.

## Reproducing the results

1. Download `hetclop-corpus-3.0.3` from the archive record and run
   `python scripts/restore_corpus.py /path/to/hetclop-corpus-3.0.3`. This places the analysed corpus where the
   scripts expect it and checks every file against its recorded SHA-256 digest. Analyses that use the scGPT or
   Geneformer encoders first need `python experiments/phase2a_decoder_2026-07-08/run_phase2a.py --stage encode`
   with the public model weights under `data/models/`.
2. Re-run any analysis from its folder, for example
   `python revisions/major-revision/experiments/e2-dose-response/run.py` followed by `analyze.py`. Existing outputs
   are reused unless removed.
3. Regenerate the numbers and check the recorded results:
   `python revisions/major-revision/evidence/followup_macros.py`,
   `python revisions/major-revision/evidence/supplement_tables.py` (Supplementary Tables S2 to S4),
   `python revisions/major-revision/evidence/verify_followup_experiments.py`, and the `verify.py` scripts of
   `armb-primary-report`, `e4-sampling-sensitivity`, `e8-matched-dispersion`, `estimated-dispersion` and
   `scientific-boundaries` (the E8 verifier reports its two checks that need the fitted arrays as not checkable). The
   `verify.py` scripts of `e1-armb-biology` and `e2-dose-response` also check the saved per-fit predictions of the
   original runs (about 14 GB) and per-dataset Geneformer embeddings, which are not distributed; their
   `verification.json` files record the checks made on those outputs. A few files pinned by digest differ here, mostly by reworded comments;
   `revisions/major-revision/experiments/released-source-digests.json` lists both digests and the change. Each
   verifier rewrites its `verification.json` when run.
4. Rebuild the figures: `python revisions/major-revision/figures/make_new_panels.py`, then
   `python revisions/major-revision/figures/build_figures_v2.py`, which also checks panel alignment and that every
   plotted coordinate of each source survives composition; `python revisions/major-revision/evidence/verify_figure_data.py`
   repeats the data check strictly.
5. Run the tests: `python -m pytest tests`. One test re-runs the simulation of
   `experiments/phase2m_lawgrid_2026-07-09` and rewrites its `results/lawgrid.json`; across library versions the
   rewritten values agree with the archived ones to about eight significant digits.

## Data

The corpus comprises 50 single-cell datasets from 45 public GEO accessions, listed with species and cell counts in
`revisions/major-revision/manuscript/figs/dataset_table.csv`. Each dataset was capped at 3000 cells by uniform
sampling (seed 0), filtered (at least 200 detected genes, mitochondrial fraction below 20%), reduced to its 8000
most variable genes, mapped to human gene symbols with the MGI ortholog table and projected onto a shared panel of
2500 genes. Cell types are cluster-level annotations. Cells were split 70/15/15 into training, validation and test
sets within each dataset (seed 42).

Rebuilding the corpus from the GEO matrices (`revisions/major-revision/experiments/e4-sampling-sensitivity/build_corpus.py`)
and the training-size experiment's extra cells require the per-dataset source files under `data/corpus_sources/`
(or `HETCLOP_CORPUS_SOURCE_ROOT`), which are not distributed; the per-group results of all eight corpora and of every
training size are included, so their summaries, intervals and figures can be recomputed. All other analyses need only
the deposited corpus.

## License

Code: MIT (`LICENSE`). Result records, figures and the deposited corpus: CC BY 4.0. The source datasets are public
GEO records and remain under their original terms; the vendored CFGen code keeps its own license and notice.
