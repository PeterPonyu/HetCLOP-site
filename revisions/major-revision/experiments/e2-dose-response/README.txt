E2 is a new validation-to-test replication.

The original Arm A and Arm B outputs retain test effects but not the group-level
validation deficits or fitted predictions needed for this regression. This run
uses the existing cached corpus and training function. It does not overwrite or
replace the original tables or their ledger.

The design was fixed before execution. The unit is a dataset and cell-type group,
repeated over encoders, arms and fit seeds. Validation Poisson draws provide the
signed deficit predictor. Independent test draws provide NB-minus-Poisson
expressed-gene calibration effects. Both positive and negative deficits are kept.

The model includes a dataset random intercept and fixed arm, encoder and seed
effects. A sufficient-statistic implementation fits the REML objective and is
checked against statsmodels. Its structure follows the official description at
https://www.statsmodels.org/stable/mixed_linear.html . The confidence interval
resamples whole datasets with all their rows together. Duplicate sampled datasets
are independent bootstrap clusters. Two arm-specific models are sensitivities.

The test endpoint uses the archived expression threshold and variance convention.
Eligibility requires at least 20 cells in both splits. Excluded groups are saved.
A negative slope with an interval below zero supports an association between the
measured deficit and the calibration effect. It does not prove biological benefit,
a causal explanation, or a sufficient rule for selecting a count likelihood.

run.py creates fitted parameters and per-group observations. analyze.py creates the
summary, all bootstrap slopes and the separate ledger. verify.py checks the design,
inputs, every fit and summaries. verify_predictions.py replays validation draws
for every fit and test contrasts for one PCA fit in each arm. check_fit.py uses
only a generated implementation fixture, never manuscript evidence.
