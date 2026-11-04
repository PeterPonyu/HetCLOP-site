Additional controlled experiments

Design: design.json was written before running the fixed 30-seed, four-regime experiment. The 120 run files contain all 77,760 group-level metric rows; no runs or regimes are omitted.

run.py reuses the original controlled generator and evaluation functions without modifying them. It tests original true-dispersion controls, then estimates dispersion from independent training counts at 25, 100 and 400 cells per group. The mean templates and group labels remain known. Estimated pooled dispersion is per-gene; the original global control is a scalar true median. These controls answer different questions.

analyze.py checks complete coverage and aggregates group medians within each dataset, then dataset medians within each seed. seed_summary.csv retains every seed. summary.csv gives medians and descriptive central 95% seed intervals. paired_vs_poisson.csv summarizes within-seed differences; these intervals are not confidence intervals. No additional null-hypothesis tests are introduced.

verify.py checks estimator boundaries, exact deterministic reproduction of one complete run into a temporary folder, direct independent variance-ratio arithmetic, and equality of the homogeneous oracle/global controls. verification.json records those checks and source hashes.

make_figures.py writes only the two additional figure sources and revision_numbers.tex. Original result files and the original number macros are untouched.

Run from the repository root:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python revisions/major-revision/experiments/estimated-dispersion/run.py
python revisions/major-revision/experiments/estimated-dispersion/analyze.py
python revisions/major-revision/experiments/estimated-dispersion/verify.py
python revisions/major-revision/experiments/estimated-dispersion/make_figures.py

Existing result files are reused by run.py. Archive the results directory before an intentional full rerun. Never replace the original five-seed experiment with these summaries: the additional experiment has its own specified aggregation and random seeds.
