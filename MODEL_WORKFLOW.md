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

The exact Run 2 through Run 5 configurations are preserved in their numbered files.
The latest completed configuration is EXP-005/Run 5 and is stored in both
`configs/colab_experiments_run_05.json` and `configs/colab_experiments.json`.

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



## EXP-004 Run 4 result

Run 4 completed as a four-fold IMS held-out-bearing robustness experiment. Weak-PINN/high had the lowest equal-bearing macro RMSE at 0.314238, but every model had negative macro R2 and large between-bearing variation. The independently verified evidence and analysis are preserved under `saved results/run_04/`.

## EXP-005 Run 5 result

Run 5 repeats the exact Run 4 folds, models, weights, seeds, sequence construction, optimizer, and evaluation. Its single experimental change is robust baseline-relative normalization of vibration features using the first eight unlabeled snapshots of each physical bearing, followed by the unchanged training-only StandardScaler. Because eight equals the frozen sequence length, that fixed prefix exists before the first sequence target is predicted.

The run completed all 36 jobs on a Tesla T4 using exact clean commit `359fd3ceb0df7314e0714468414aba52c95b7783`. Independent verification reproduced all best-checkpoint and final-epoch metrics, configuration and split identities, source and artifact hashes, preprocessing parameters, and prediction identities. The prediction schema correction worked: `run_id` and `bearing_run_id` identify the physical trajectory, while `experiment_run_id` records `run_05`.

Weak-PINN/high ranked first by equal-bearing macro normalized RMSE at `0.288104`, ahead of Strong-PINN (`0.311587`) and LSTM (`0.371753`). It improved the Run 4 macro, worst-bearing, and between-bearing-variation results, but improved only two of four folds and worsened macro late-life MAE. All models retained negative macro R-squared values. Run 5 therefore failed the full predeclared success criterion and is a controlled partial/negative result.

The baseline-relative transform reduced mean label-free signal-feature Wasserstein shift from `1.4482` to `0.9342` in every fold, yet prediction error increased sharply on IMS-DS2/B1. This establishes that lower measured covariate discrepancy is not sufficient for better cross-bearing RUL transfer. Runs 4 and 5 are now the frozen diagnostic baseline pair. Before another Colab experiment, prepare a current-literature novelty matrix and locked publication protocol centered on when preprocessing assumptions and physics priors are trustworthy across bearings; do not automatically continue with another fixed-weight sweep.

## Publication research direction after Run 5

The literature and implementation audit is recorded under `research/`. It found that
attention-assisted PINNs, DeepHPM operators, generic adaptive loss weighting, adaptive
Paris-law parameter fitting, physics-consistency scores, Bayesian/ensemble uncertainty, and
conformal bearing-RUL intervals have already been published. Those mechanisms are baselines or
supporting components, not defensible standalone novelty.

The locked working question is whether the applicability of a candidate bearing-degradation
prior can be estimated without target-test RUL labels and used to prevent physics-induced
negative transfer. The proposed PriorCred-RUL study must first validate prior-validity
detection under controlled synthetic truth, then test a declared data-only fallback on
PRONOSTIA and the frozen IMS folds. Runs 4 and 5 provide motivation only.

Read these files before implementing EXP-006 or changing model mathematics:

- `research/PUBLICATION_NOVELTY_MATRIX.md`
- `research/PUBLICATION_PROTOCOL.md`
- `research/RESEARCH_ROADMAP.md`

EXP-006 is a data and physics-identifiability qualification milestone. It must inspect the
synthetic HDF5 schema, establish recoverable generator truth, define immutable publication
splits and physics applicability, and build derived-cache tooling before any new full neural
training. Do not prepare another fixed-weight PINN sweep.
