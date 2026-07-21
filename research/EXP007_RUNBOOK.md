# EXP-007 synthetic credibility feasibility runbook

Status: prepared for Google Colab execution

Experiment ID: `EXP-007`

Active configuration: `configs/experiment.yaml`

## Scientific purpose

EXP-007 is the stop/go experiment for the proposed physics-prior credibility direction. It
tests whether a frozen, label-safe diagnostic can distinguish supported from deliberately
misspecified simulator progression priors on complete held-out trajectories. It runs before
integration into a high-capacity PINN and before any real-bearing credibility claim.

This experiment does not claim that the official simulator's hidden P-code implements a known
published governing equation. The four declared progression families are adapted into
training-only empirical degradation templates. They are identified throughout the artifacts
as simulator-family templates, not Paris crack growth, ISO 281, or measured real-bearing laws.

## Fixed data and split

- Controlled cache: `data/processed_features/publication/exp006/controlled_synthetic_features.csv`
- SHA-256: `3199282d5abf674538797b41dc97240825cf6ec80853dffb5c9f8ca4f45bfdae`
- Training trajectories: 24
- Validation trajectories: 8
- Test trajectories: 8
- Progression families: linear, progressive, step-like, and gamma
- Causal feature sequence: 8 snapshots, stride 1
- Causal sequences before candidate expansion: 2,831 train, 504 validation, 382 test

The supplied-v2 cache is intentionally excluded from the known-truth validity endpoint because
its progression and fault labels are withheld. Its six-snapshot short trajectory is therefore
not silently removed from an EXP-007 dataset; that separate dataset is not used here.

## Cross-fitting and leakage boundary

Each of the 24 training trajectories is an out-of-fold trajectory exactly once. For its
cross-fitted evidence, that trajectory does not fit or tune:

- the StandardScaler;
- the causal LSTM RUL backbone;
- the vibration-to-degradation Extra Trees proxy; or
- the empirical progression templates.

The credibility classifier is trained on those out-of-fold training predictions. Validation
trajectories fit Platt calibration, the credibility threshold, early stopping for the final
backbone, and the validation-selected scalar comparator. Test RUL and validity labels are used
only after every component is frozen.

Forbidden credibility inputs include target RUL, hidden degradation, true family, validity
label, corruption type, total trajectory length, failure time, and future measurements.

## Candidate-prior stress test

At every causal checkpoint, the benchmark enumerates all four progression families crossed
with five declared time-scale settings, without consulting the trajectory's true family:

- exact correct family — valid;
- two mildly uncertain correct-family time scales — valid;
- two severely biased correct-family time scales — corrupt; and
- every wrong progression family/scale combination — corrupt.

This produces three valid and seventeen corrupt candidates. The natural imbalance is retained
for calibration, while the training classifier uses its declared balanced class weights. The
candidate pool cannot leak truth through a truth-dependent selection rule.

Training, validation, and test use different parameter-bias magnitudes while enumerating every
wrong law. Validation/test load, speed, and observation-noise shifts remain ordinary covariate
shift: they do not change a correct prior's validity label.

## Models and controls

- Small causal LSTM: identical data-only fallback.
- Extra Trees degradation proxy: maps available synthetic vibration/condition observations to
  simulator degradation without using target-test hidden degradation.
- Empirical family templates: fitted only from source training truth.
- Standardized logistic credibility estimator with validation-only Platt calibration.
- Controls: inverse residual, validation-selected scalar, random credibility, all-on, all-off,
  oracle, anti-oracle, and data-only.

PILE is explicitly not labeled as reproduced. Its GP/PDE evidence-selection formulation is not
faithfully applicable to these empirical simulator templates.

## Primary decision gate

The experiment proceeds only when all apply:

1. aggregate held-out trajectory-candidate AUROC is at least `0.80`;
2. the trajectory-first, seed-second bootstrap 95% AUROC lower bound is above `0.50`; and
3. neither all-on nor all-off behavior exceeds `90%` without a declared physical explanation.

If the gate fails, preserve and analyze the failure. Do not inspect the test result and then
change corruptions, thresholds, evidence, or network capacity.

RUL blending/regret in EXP-007 is a feasibility diagnostic. It is not yet the final
credibility-weighted physics-loss method or evidence of real-bearing improvement.

## Colab execution

1. Delete the previous `MyDrive/Upload` only after its results are preserved.
2. Upload the new local `Upload` folder.
3. Open `Upload/train_models_colab.ipynb`.
4. Select a T4 GPU.
5. Run all cells.

The notebook reads the exact pushed commit from `expected_commit.txt`; no SHA editing is
required. It trains under `/content/exp007_work` and synchronizes recovery state to
`MyDrive/Upload/experiment_outputs_exp007` after completed work units.

Budget approximately 30–60 minutes on a Colab T4, including dependency installation. Actual
time depends on early stopping and Colab I/O. Twenty small LSTM fits are scheduled: three
cross-fit models plus one final model for each of five seeds.

## Result handoff

Download `experiment_outputs_exp007` and place it under `results/incoming/`. Do not rename or
delete its manifest, seed directories, predictions, calibration parameters, failure records,
or bundle. Codex must independently reproduce the credibility metrics and gate decision before
EXP-008 is prepared.

## MATLAB and ANSYS

No additional simulation is required for EXP-007. MATLAB generated the locked benchmark in
EXP-006. ANSYS remains deferred until the credibility diagnostic passes and a separately
specified contact/crack/thermal validation would test a concrete physical claim.
