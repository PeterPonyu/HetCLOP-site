E8: shared-dispersion controls matched in the amount of extra variance (a post hoc control).

design.json (hashed in design.sha256 before any outcome was computed) states the question, the four comparators, the
endpoints, the inference and the interpretation rule. run.py original scores the saved E2 PCA-512 fits (arms A and B,
three seeds); run.py largest refits Arm B on the largest training set of E4b (three seeds) and scores it.
The refits reproduced the E4b fits exactly (identical validation loss and identical median-comparator scores), so
only the draws are new; the per-dataset direction counts therefore differ slightly from those of E4b. Every shared
dispersion is computed from validation cells only; every variant of a fit uses the same draw seed. analyze.py writes
summary.csv (paired dataset-level contrasts with dataset-bootstrap intervals and sign-flip p-values),
dataset-effects.csv, joint-direction.csv and variance.csv (achieved extra variance on test cells relative to the
per-cell dispersion). verify.py checks the design hash, completeness, the matching property on validation cells, the
validation-only median and that the analysis regenerates exactly. The fitted means and dispersions of the refits
(runs/x3_B_*/mu.npy, theta.npy) are not distributed.
