# Vendored CFGen decoder code (verbatim)

These files are copied **verbatim** (byte-identical; md5 verified at vendoring
time) from the CFGen repository:

- Source: `https://github.com/theislab/CFGen`
- Commit: see `CFGEN_COMMIT.txt` (`7a109b3216417a6020ffa1143147568cc8592fb6`, 2025-08-29)
- License: MIT (see `LICENSE`) — Copyright (c) 2022, Till Richter
- Paper: Palma, Richter, Theis et al., *CFGen*, ICLR 2025, arXiv:2407.11734.

Files:
- `cfgen/models/base/encoder_model.py` — CFGen's `EncoderModel`: the autoencoder /
  decoder stage that defines `p(x|z,l)`. Decoder: `mu = softmax(h_psi(z)) *
  size_factor`; likelihood: `NegativeBinomial(mu, theta=exp(self.theta))` with
  `self.theta` a **global gene-wise parameter** when `covariate_specific_theta=False`
  (CFGen's default) — the exact "per-cell-latent mean x size factor + separately
  fitted global gene-wise dispersion" arrangement F4 indicts.
- `cfgen/models/base/utils.py` — the `MLP` building block used by `EncoderModel`.

## What Phase 2j runs, and what it does NOT

`run_cfgen.py` instantiates CFGen's **actual `EncoderModel` class** (this vendored
code, unmodified) and trains it with a **matched custom loop** (AdamW lr=1e-3,
wd=1e-5, batch 256 — CFGen's `configure_optimizers` / `training_config` defaults;
encoder dims `[512,256,50]` — CFGen's `default.yaml`; encoder input `log1p(counts)`
= CFGen's `normalization_type: log_gexp`; NB target = raw counts; size factor =
per-cell total counts) on our corpus and split.

This is **CFGen's decoder code**, not a reimplementation of it. The deviations from
CFGen's full deployed pipeline are, explicitly:
- CFGen's Hydra/PyTorch-Lightning `Trainer` orchestration is replaced by a custom
  loop that calls CFGen's own `model.encode` / `model.decode` and `scvi`'s
  `NegativeBinomial` exactly as CFGen's `_step` does (`encoder_model.py` L121-140).
- The **flow-matching stage is not run** — it is irrelevant to F4, which is a
  within-decoder control on `p(x|z,l)` only (spec §20; the brief states the test
  "requires no retraining of the generative flow").
- Corpus/HVG panel/split are ours (2500-gene panel, 50 datasets), not CFGen's.

No CFGen source line was edited.
