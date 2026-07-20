# Multi-dataset bearing RUL model workflow

The canonical Colab notebook is `notebooks/train_models_colab.ipynb`; the historical root
notebook filename is generated with the same contents for continuity.

The implemented models are:

- FNN, CNN, and LSTM data-only baselines.
- AttnPINN with a corrected batch-first temporal attention encoder and a learned DeepHPM
  differential operator. This is a latent-physics model, not an explicit bearing law.
- Weak-prior PINN with RUL monotonicity, initial/terminal boundaries, vibration-derived
  health-indicator alignment, and a temperature trend prior when temperature exists.
- Strong PINN with the weak constraints plus separately weighted rolling-contact-fatigue
  Paris crack growth and Palmgren-Miner/bearing-life residuals. Temperature-dependent
  viscosity and contamination affect the residuals.

The strong model distinguishes physical equations from calibrated constitutive
approximations. The Paris coefficient/exponent and bearing-life structure come from the
reviewed literature. Contact pressure and missing bearing/lubricant parameters are recorded
as dataset-level assumptions. The differentiable `aSKF` expression approximates a catalog
curve and must be calibrated before it is described as quantitatively physical.

Dataset support:

- IMS raw snapshots.
- PRONOSTIA raw `acc_*.csv` snapshots.
- The hourly vibration/temperature run-to-failure CSV files, using streaming reads.
- A standardized CSV contract for the MATLAB v7.3 synthetic varying-degradation dataset.
- CWRU variants are acknowledged but excluded from direct RUL regression because they are
  classification samples rather than complete run-to-failure trajectories.

All data extraction is cached. Scaling is fit only on the training split. Run-level splits
are used when multiple bearings exist; the one-run vibration/temperature dataset uses an
explicit temporal split and must not be presented as cross-bearing validation.

The exact Run 2 configuration is preserved in
`configs/colab_experiments_run_02.json`. The active EXP-003/Run 3 configuration is stored in
both `configs/colab_experiments_run_03.json` and `configs/colab_experiments.json`.

Run 3 is an IMS-only, feature-based calibration experiment. LSTM and Weak-PINN/high are
predeclared references. Strong-PINN candidates cross Paris weights `[0.003, 0.01, 0.03]`
with Palmgren-Miner weights `[0.0003, 0.001, 0.003]` over the common seeds 42, 1042, and
2042. Candidates are trained, early-stopped, and ranked using validation data only. Only
the frozen validation winner is evaluated on test. Histories contain unweighted losses,
weighted contributions, periodic data/physics gradient norms, parameter counts, inference
times, normalized metrics, and original-time RUL metrics. The run also writes source,
configuration, split, feature-cache, environment, failure, and artifact manifests.

The Google Drive `MyDrive/Upload` directory is disposable between runs. Once a completed
output directory has been downloaded and preserved locally, the user deletes the old Drive
`Upload` and uploads the next complete local package. Every prepared `Upload/` must contain
all required source, configuration, notebook, dependency, and compact feature-cache files;
future runs must not depend on remnants of a previous Drive package.



## EXP-004 Run 4 preparation

Run 4 is prepared as a four-fold IMS held-out-bearing robustness experiment. The exact fold map, frozen model profiles, frozen Strong-PINN weights, cache hash, seeds, and success criterion are in `configs/colab_experiments_run_04.json`. The active `configs/colab_experiments.json` now points to EXP-004.

The fresh Upload package contains only the compact IMS cache and writes to `experiment_outputs_run_04`. It requires the user to commit/push the prepared repository and paste that exact SHA into the notebook before execution. This avoids treating an uncommitted Upload copy as thesis evidence.

Run 4 produces 36 jobs, resumable job artifacts, best-validation and final-epoch metrics, per-bearing summaries, a manifest, and `codex_results_bundle.zip`.