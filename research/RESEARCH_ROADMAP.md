# Research roadmap after EXP-005

Status: approved direction pending implementation

Date: 21 July 2026

## Destination

The target paper is a rigorous study of **physics-prior credibility and negative transfer in
bearing RUL**, not a larger collection of neural architectures. The central scientific result
should explain when bearing physics helps, when it harms, whether that can be detected without
test RUL labels, and whether a safe data-only fallback works across synthetic and real domains.

The novelty and protocol are defined in:

- `PUBLICATION_NOVELTY_MATRIX.md`
- `PUBLICATION_PROTOCOL.md`

Runs 4 and 5 remain the frozen motivating baseline pair. They are not to be re-tuned or
retroactively re-labeled as evidence for the proposed method.

## Stage 0 — Completed evidence recovery and design lock

Completed:

- independently verified EXP-004 and EXP-005;
- established that reduced feature Wasserstein shift did not reliably reduce RUL error;
- established that fixed strong physics can produce negative transfer;
- audited the local AttnPINN, general PINN reference, bearing simulation model, datasets, and
  relevant papers;
- identified 2026 novelty threats that rule out adaptive Paris weighting, generic gates, and
  UQ as standalone contributions; and
- locked the initial research questions, outcomes, leakage rules, baselines, and stop criteria.

## Stage 1 — EXP-006: data and physics identifiability qualification — completed

### Goal

Prove that the controlled benchmark and real datasets contain the information required by the
planned claims before writing a new neural model.

### Work

- Build a read-only inventory and fingerprint for synthetic v2, PRONOSTIA, IMS, and the
  vibration/temperature dataset.
- Implement a deterministic MATLAB v7.3/HDF5 reader or conversion utility that writes only to
  a derived cache.
- Recover synthetic trajectory IDs, official membership, time/sampling information, and every
  available generator parameter or degradation-family label.
- Create a machine-readable physics applicability table: required measurements, observed
  measurements, assumed parameters, units, and identifiability status per dataset/prior.
- Create immutable development/validation/test unit splits. Preserve official splits and the
  frozen IMS folds.
- Verify causal RUL labels and failure endpoints.
- Define corruption generators without altering raw data.
- Run only small deterministic checks locally; reserve heavy feature extraction for Colab.

### Decision gate — passed

Proceed only if valid/corrupt prior labels can be established without using test RUL labels.
If the supplied synthetic data do not expose enough truth, implement a transparent controlled
Python generator from published equations and record that as a new simulated benchmark. Do not
pretend unknown simulator internals are known.

The supplied v2 dataset intentionally withholds progression-family and fault labels, so it was
not relabeled. The official CC BY 4.0 MATLAB simulator was instead run from a predeclared
40-scenario design with seed 42006. It generated 3,997 labeled snapshots and exposes progression
family, hidden degradation, and fault location. The full qualification is under
`results/analyzed/EXP-006/`; exact reproduction details are in `EXP006_RUNBOOK.md`.

### Required artifacts

- `configs/publication_data_split.json`
- `configs/physics_priors.yaml`
- dataset and physics-applicability manifests
- conversion tests and cache schema
- an EXP-006 report documenting what is and is not identifiable

EXP-006 required no Colab Upload or neural training. EXP-007 synthetic credibility feasibility
is now the next implementation milestone.

## Stage 2 — EXP-007: synthetic credibility feasibility

### Goal

Test the diagnostic before integrating it into a high-capacity RUL model.

### Work

- Fit a simple, causal data-only backbone on synthetic training trajectories.
- Cross-fit label-safe credibility evidence.
- Train and calibrate the frozen credibility estimator on declared valid/corrupt regimes.
- Evaluate held-out law, parameter, noise, and operation-shift regimes.
- Compare inverse residual, validation-selected scalar weight, PILE-inspired evidence where
  feasible, random, all-on, all-off, and oracle controls.
- Measure AUROC, AUPRC, Brier score, calibration, fallback behavior, and parameter recovery.

### Decision gate

Stop method escalation if synthetic AUROC is below 0.80 or its 95% interval includes 0.50.
Diagnose identifiability rather than compensating with a larger network.

## Stage 3 — EXP-008: frozen real-data benchmark qualification

### Goal

Establish modern, leakage-safe PRONOSTIA and IMS baselines before testing PriorCred-RUL.

### Work

- Prepare PRONOSTIA using complete-bearing and official condition/test roles.
- Repeat the frozen IMS unit folds with the common artifact schema.
- Compare data-only, Weak-PINN, fixed Strong-PINN, AttnPINN/DeepHPM, adaptive-loss, adaptive
  Paris, and one representative domain-generalization baseline.
- Keep one feature representation, causal sequence definition, target, and test population
  common.
- Tune on development/validation bearings only. Seal the final test artifacts.

### Decision gate

The baselines must execute reliably, reproduce their own metrics from predictions, and expose
meaningful variation in physics regret. If all priors behave identically, revisit the study's
identifiability before proceeding.

## Stage 4 — EXP-009: PriorCred-RUL controlled comparison

### Goal

Test whether the frozen credibility mechanism reduces negative transfer.

### Work

- Use the same backbone, representation, splits, seeds, optimizer family, and evaluation as
  EXP-008.
- Add only the frozen credibility-controlled prior mechanism.
- Run synthetic oracle/anti-oracle controls and mandatory protocol ablations.
- Evaluate real-data association between label-free credibility and realized physics regret.
- Report accuracy, regret, calibration, conditional lifecycle results, failures, time, and
  parameter counts.

### Decision gate

Apply all five success criteria in `PUBLICATION_PROTOCOL.md`. Do not select a favorable
dataset, seed, or lifecycle segment after seeing the test results.

## Stage 5 — EXP-010: representation and multiphysics confirmation

### Goal

Determine whether the conclusion survives more informative sensor representations without
changing the central method.

### Work

- Freeze the EXP-009 method and weights.
- Compare feature-only, raw vibration, and hybrid raw/physical signal features.
- Use the temperature dataset only as a declared one-run mechanism case study.
- Use availability masks for temperature/lubrication priors.
- Report computation and ablate missing or corrupted sensor evidence.

This stage may strengthen external validity but cannot rescue a failed EXP-009 central test.

## Stage 6 — EXP-011: predeclared final multi-seed evaluation

### Goal

Produce the single evidence package used for the manuscript's main comparison.

### Work

- Freeze code, splits, hyperparameters, baselines, and claims.
- Refresh the literature matrix.
- Run at least five common seeds on the untouched final test populations.
- Perform hierarchical bootstrap, paired comparisons, Holm correction, lifecycle analysis,
  conditional reliability analysis, sensitivity analysis, and full artifact verification.
- Generate manuscript tables and figures directly from verified files.

No model selection follows EXP-011. A failed endpoint is reported as a failed endpoint.

## Manuscript contribution structure if the protocol succeeds

1. Empirical evidence that lower covariate discrepancy and lower assumed-law residual do not
   necessarily imply lower bearing-RUL error.
2. A benchmark protocol that separates valid physics, parameter uncertainty, law mismatch,
   ordinary domain shift, and sensor unavailability.
3. An anti-collapse, label-safe physics-prior credibility mechanism with an explicit
   data-only fallback.
4. Multi-dataset evidence that the mechanism reduces physics-induced negative-transfer risk,
   with per-bearing and conditional reliability reporting.

Accuracy improvements are welcome but secondary to the stronger reliability claim. If the
method merely improves average RMSE without detecting misspecification or reducing regret,
the intended paper claim has failed.

## Immediate next repository actions

The next coding turn should implement only Stage 1:

1. inspect the synthetic HDF5 schema and documentation without modifying `Data.mat`;
2. add the derived-cache converter and schema tests;
3. generate dataset fingerprints and the applicability manifest;
4. draft immutable publication splits and prior definitions;
5. validate locally without neural-network training; and
6. only then prepare and push a self-contained EXP-006 Colab package if heavy extraction is
   required.

This sequence prevents another expensive run from being launched before the physics labels,
units, and evaluation claims are actually identifiable.
