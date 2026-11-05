# Calibration gains from added dispersion need not carry over to marker recovery in single-cell count decoders

This repository holds the code, saved results and figure sources for the article of the same title by Zeyu Fu.
Version 3.0.3 is archived at https://doi.org/10.5281/zenodo.23940846, where the analysed corpus is a separate file
(`hetclop-corpus-3.0.3.tar`). The DOI https://doi.org/10.5281/zenodo.21524011 always points to the newest version.

## What is here

- `src/`: the decoders (deterministic, Gaussian, Poisson, negative-binomial and zero-inflated output layers), the
  evaluation metrics (expressed-gene variance calibration, energy distance, marker recall, rare-type F1), the
  tightness diagnostic, the shared gene panel and the ortholog mapping.
- `experiments/`: the analyses of the first submission, one folder each, with their scripts and saved results
  (`results/`). Folder names are dated run labels.
- `revisions/major-revision/experiments/`: the analyses added during revision. Each folder has a design file
  (`design.json`) that was fixed and hashed before the analysis was run, the scripts, the result tables and a
  `verification.json` record. Every seed and fit is kept.
- `revisions/major-revision/figures/` and `revisions/major-revision/manuscript/figs/`: the scripts that draw the
  figures and tables from the saved results, the files that hold every number quoted in the article, and the
  rendered figures. `FIGURE-MAP.md` shows which submitted figure became which revised one.
- `revisions/major-revision/evidence/`: the scripts that regenerate the numbers and the supplementary tables, and
  the checks.
- `ai-disclosure/`: the AI-use statement and, in `prompts/`, an English translation of the prompts used for the
  AI-assisted code work of the first submission.
- `tests/`: unit tests for the library.

## Requirements

Python 3.11 or later with the packages in `requirements.txt`, plus scikit-learn and statsmodels. The scVI analyses
need scvi-tools 1.4.3, and fitting the models needs a CUDA GPU. Building the figures needs a TeX distribution with
pgfplots, `pdfcrop` and PyMuPDF; the UMAP figure needs R with ggplot2 and jsonlite.

## Reproducing the results

1. Download `hetclop-corpus-3.0.3.tar` from the Zenodo record, unpack it and run
   `python scripts/restore_corpus.py /path/to/hetclop-corpus-3.0.3`. The script puts the corpus where the
   analyses expect it and checks every file against its SHA-256 digest.
2. The analyses that use scGPT or Geneformer need their embeddings, which are not archived. Compute them with
   `python experiments/phase2a_decoder_2026-07-08/run_phase2a.py --stage encode`, with the public model weights in
   `data/models/`.
3. Run any analysis from its folder, for example `run.py` and then `analyze.py` in
   `revisions/major-revision/experiments/e2-dose-response/`. Saved outputs are reused unless you delete them.
4. Regenerate the numbers and tables, and check the saved results:
   - `python revisions/major-revision/evidence/followup_macros.py`
   - `python revisions/major-revision/evidence/supplement_tables.py` (Supplementary Tables S2 to S4)
   - `python revisions/major-revision/evidence/verify_followup_experiments.py`
   - `verify.py` in `armb-primary-report`, `e4-sampling-sensitivity`, `e8-matched-dispersion`,
     `estimated-dispersion` and `scientific-boundaries`

   Each `verify.py` rewrites its `verification.json` when it runs. The E8 check reports two tests that need the
   fitted arrays as not checkable. The checks in `e1-armb-biology` and `e2-dose-response` also read the saved
   predictions of the original fits (about 14 GB) and the per-dataset Geneformer embeddings. These files are not
   archived, so the committed `verification.json` files record the results obtained with them. A few files differ
   from the copies whose digests were recorded at run time, mostly in reworded comments;
   `revisions/major-revision/experiments/released-source-digests.json` lists both digests and the change.
5. Rebuild the figures with `python revisions/major-revision/figures/make_new_panels.py` and then
   `python revisions/major-revision/figures/build_figures_v2.py`. The second script also checks panel alignment and
   that every plotted value survives composition; `python revisions/major-revision/evidence/verify_figure_data.py`
   repeats the data check in strict mode.
6. Run the tests with `python -m pytest tests`. One test reruns the simulation in
   `experiments/phase2m_lawgrid_2026-07-09` and overwrites its `results/lawgrid.json`; with other library versions
   the new values agree with the archived ones to about eight significant digits.

## Data

The corpus has 50 single-cell datasets from 45 public GEO accessions, listed with species and cell counts in
`revisions/major-revision/manuscript/figs/dataset_table.csv`. Each dataset was capped at 3000 cells by uniform
sampling (seed 0) and filtered to cells with at least 200 detected genes and a mitochondrial fraction below 20%.
Its 8000 most variable genes were mapped to human gene symbols with the MGI ortholog table and projected onto a
shared panel of 2500 genes. Cell types are cluster-level annotations. Within each dataset, cells were split 70/15/15
into training, validation and test sets (seed 42).

Two steps need files that are not archived: rebuilding the corpus from the GEO matrices
(`revisions/major-revision/experiments/e4-sampling-sensitivity/build_corpus.py`) and adding the extra cells of the
training-size analysis. Both read per-dataset source files from `data/corpus_sources/` (or the folder named in
`HETCLOP_CORPUS_SOURCE_ROOT`). The per-group results of all eight corpora and every training size are included, so
their summaries, intervals and figures can still be recomputed. Every other analysis needs only the archived corpus.

## License

Code: MIT (`LICENSE`). Results, figures and the archived corpus: CC BY 4.0. The source datasets are public GEO
records and keep their original terms. The bundled copy of CFGen keeps its own license and notice.
