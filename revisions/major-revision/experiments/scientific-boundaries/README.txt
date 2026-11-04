Scientific boundary tests and biological support sensitivity

This is new, isolated revision evidence. It does not replace original paper values.

run.py: 20 new seeds x 2 generating families x 3 mean specifications = 120 runs.
Every run contains 72 groups, 3 validation arms and 4 test arms (including an
independent real reference): 60,480 retained group-level records in results/.
Only theta is estimated. The mean perturbations are controlled, not newly trained
neural decoders. The correlated generator changes both higher marginal moments
and dependence; it is not a covariance-only intervention.

analyze.py: validation-only arm selection within each synthetic dataset; all test
endpoints, paired effects, selection counts, and optimistic test-best regret.
Group -> dataset median -> seed median. Central 95% seed intervals are not
confidence intervals. No post-hoc significance tests. Selection regret is only a
descriptive test diagnostic, never used to select the deployed arm.

The biological analysis uses archived per-type values, not new model fits. The
common support is 69 of 73 target types. Seed indices are paired only within a
stage. Stages have independent fitted models. Coverage-adjusted recall explicitly
assigns zero to an unscoreable target; it is a new composite, not original recall.
No type-level bootstrap is presented as dataset-clustered uncertainty.

verify.py checks complete finite outputs, direct primary-metric arithmetic,
validation-only selection, independent split covariates, matched count moments,
source hashes, common-support arithmetic, and one byte-identical complete rerun.
The implementation-correction/ archive records a centering bug: attenuation must shrink towards E[size], not 1. All 40 affected conditions
were rerun under the original mean-preserving design, with unchanged seeds. The
archive is excluded from all final analyses. Other 80 conditions are unchanged.

make_figures.py generates the figure sources of this analysis and its macro file,
boundary_numbers.tex.

Reproduce with the packages in requirements.txt:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run.py
  python analyze.py
  python verify.py
  python make_figures.py
Run these scripts by their absolute paths or from this directory. Existing run
files are reused; clean reruns should use a copied revision workspace.

Main findings: successful calibration selection does not imply recovery of gene
dependence; biased means can produce opposing calibration and energy conclusions;
matched scoring populations do not remove the fixed-mean/end-to-end marker reversal.
The tests do not establish external-corpus, donor-held-out, learned-mean, or
amortized-neural-dispersion generality. Those are stronger claims than this paper
now makes, not outcomes inferred from the present experiments.
