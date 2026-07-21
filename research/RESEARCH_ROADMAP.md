# Research roadmap after EXP-005

Status: EXP-007 completed; credibility escalation stopped at failed gate

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

EXP-006 required no Colab Upload or neural training. EXP-007 subsequently completed; its result
is recorded in Stage 2.

## Stage 2 — EXP-007: synthetic credibility feasibility — completed, gate failed

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

### Decision gate — failed

Stop method escalation if synthetic AUROC is below 0.80 or its 95% interval includes 0.50.
Diagnose identifiability rather than compensating with a larger network.

### Verified result

The exact clean five-seed T4 run completed and passed artifact-identity and metric-reproduction
checks. Correct within-seed AUROC is `0.660876 +/- 0.098617`; a mean-probability ensemble gives
`0.753676`. Four of five seeds exceed 90% all-off fallback. The gate and anti-collapse rule
therefore fail.

The source design has no speed/SNR variation, causing extreme out-of-support condition-shift
features on validation and test. The benchmark target is also misaligned: corrupt priors
generally reduce rather than increase RMSE relative to the weak data-only backbone. The result
does not support H1 or justify real-data PriorCred-RUL escalation. Full verification is under
`results/analyzed/EXP-007/`.

## Stage 2A - EXP-007A: counterfactual physics-harm correction - predeclared

EXP-007A replaces mathematical law-validity as the primary target with actual out-of-fold RUL
intervention safety. Each candidate model begins from the identical best data-only checkpoint
and is fine-tuned with differentiable simulator-progression value, rate, and monotonic losses.
Credibility evidence remains causal and excludes RUL, degradation truth, realized regret, and
future observations.

The official simulator design contains 64 multi-condition training, 16 interpolation
validation, and 16 fresh sealed-test trajectories. Development/validation use seed `420071`;
test uses seed `920071`. Before test evaluation, every seed must demonstrate at least 20% safe
and 20% harmful trajectory-candidate interventions. Full changes and gates are frozen in
`EXP007A_PROTOCOL_AMENDMENT.md`.

Open EXP-007 feature ablations are diagnostic only. They cannot select EXP-007A based on its
sealed outcomes. EXP-008 remains blocked until EXP-007A independently passes all credibility,
calibration, anti-collapse, harm-stress, and regret-reduction gates.

## Stage 3 — EXP-008: frozen real-data benchmark qualification — blocked

### Goal

Establish modern, leakage-safe PRONOSTIA and IMS baselines before testing PriorCred-RUL.

Do not implement this stage until a protocol amendment and a new controlled feasibility test
resolve the EXP-007 identifiability, anti-collapse, calibration, and target-alignment failures.

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

The corrective EXP-007A implementation follows these locked actions:

1. freeze EXP-007 as a negative feasibility result and amend the protocol before changing the
   method;
2. distinguish law correctness from counterfactual physics-intervention harm;
3. run oracle-evidence-ceiling and feature-ablation diagnostics only on the now-open EXP-007
   data, explicitly excluding them from confirmation evidence;
4. replace zero-variance condition standardization with predeclared physical scaling or a
   multi-condition source design;
5. require development corruptions to create measurable positive RUL regret; and
6. freeze the amended method and gates before using MATLAB to generate a fresh sealed test
   population with a new seed.

ANSYS is not indicated at this point: the blocker is statistical identifiability and benchmark
target construction, not missing contact or fracture simulation fidelity.
