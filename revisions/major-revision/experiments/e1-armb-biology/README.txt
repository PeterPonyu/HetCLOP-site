E1 compares fixed-mean biological effects in two matched experimental arms.

The three PCA fits in each arm come from the new E2 replication. PCA was selected
before these results because it passed the archived acceptance gate in both arms.
All four draws within a fit use exactly the same mean. Shared per-gene and
per-cell-type dispersions are validation medians of the same fitted per-cell
parameters. Type-specific estimates use labels and fall back to shared estimates
below 20 validation cells. These are allocation controls, not separately trained
scVI dispersion heads.

Every endpoint uses the same 1,250 target genes in both arms. Arm A's encoder saw
the target genes. Arm B's encoder did not. The decoders retain their original
training targets, so Arm A has a larger output head. The comparison extends the
biological design with a matched new MLP experiment. It does not reproduce or
replace the earlier scVI-based biological results.

Markers are defined within each dataset against its other cell types. Welch and
AUROC recall both use 50 genes. A classifier trained on synthetic half A predicts
real half B. The random split is shared by variants. The classifier and log-count
features follow the existing biological procedure. Convergence warnings are kept.
The rare class definition uses the full held-out test split, before halving it.

Inference averages seed effects within each dataset, then weights datasets equally.
This classification estimand differs from pooled whole-corpus F1, which is also
saved. Within-dataset macro-F1 uses fixed true-class support across variants.
Rare-class F1 uses the predefined rare identities present in that evaluation
dataset. The full-support endpoint counts abundant cells misclassified as rare.
The masked sensitivity evaluates only true rare cells. All three seeds and all
signed effects are retained. Intervals are nominal and endpoint-specific.

The arm interaction compares the per-cell-minus-Poisson effect across arms using
paired datasets. A secondary marker dose-response uses validation deficits on
this same target-gene support, not the larger Arm A panel used in primary E2.
These associations do not establish a prospective selection procedure.
