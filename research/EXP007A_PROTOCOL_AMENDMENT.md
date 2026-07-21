# EXP-007A protocol amendment: counterfactual physics harm

Status: predeclared corrective protocol; no EXP-007A simulation outcomes or neural results
observed

Protocol version: 0.2

Date: 21 July 2026

## Reason for amendment

EXP-007 validly failed its credibility and anti-collapse gates. Its source population had one
speed/SNR condition, its standardized shift variables extrapolated far outside source support,
and the declared corrupt templates generally improved rather than harmed the weak data-only
backbone. Mathematical prior correctness was therefore not an adequate target for preventing
physics-induced RUL negative transfer.

EXP-007 and its test set remain frozen as a negative feasibility result. They may be used for
open-data diagnosis only and cannot validate this amendment.

## Revised primary target

For trajectory `i`, candidate prior `k`, and common seed `s`, define counterfactual physics
regret using an actual physics-regularized model and its identical parent data-only model:

```text
regret(i,k,s) = RMSE_physics(i,k,s) - RMSE_data_only(i,s)
```

A candidate is harmful when regret exceeds the predeclared normalized-RUL margin. The primary
credibility label is `safe_to_apply = 1 - harmful`. Governing-family correctness, parameter
corruption, and operating condition remain separate explanatory fields. Test RUL may create the
evaluation label after prediction, but it may never enter credibility features, calibration,
threshold selection, model selection, or physics weighting.

## Actual intervention

Each candidate physics model starts from the exact best data-only checkpoint for that fold and
seed, then is fine-tuned on the same training trajectories with a differentiable loss:

```text
L = L_data
    + lambda_prior * MSE(RUL_hat, RUL_candidate)
    + lambda_rate * SmoothL1(max(0, -dRUL_hat/dtau), candidate_damage_rate)
    + lambda_monotonic * mean(ReLU(dRUL_hat/dtau)^2)
```

Candidate RUL and rate come from a training-only empirical simulator-family template evaluated
at causal operating-condition-adjusted exposure. This is an adapted controlled-simulator
progression prior, not Paris crack growth, ISO 281 individual-life truth, or a reproduced hidden
simulator equation. Its purpose is a known-truth intervention stress test.

The data-only, fixed all-on, validation-selected scalar, inverse-residual, random, credibility-
controlled, oracle, and anti-oracle controls all use predictions from these trained
interventions. EXP-007A no longer treats post-hoc interpolation with an untrained template as
the physics intervention.

## Multi-condition benchmark

The official CC BY 4.0 MATLAB simulator will generate 96 complete trajectories:

- 64 training trajectories: 16 per degradation family;
- 16 validation trajectories: four per family; and
- 16 fresh sealed-test trajectories: four per family.

Training spans four loads, four speeds, four SNR levels, load variation, slip, and modulation
settings in a balanced deterministic design. Validation and test conditions are distinct but
inside the declared training ranges. Development/validation use simulator seed `420071`; the
sealed test uses seed `920071`. The scenario file is authoritative and must be committed before
simulation. Raw output is immutable derived evidence and is not committed.

Condition evidence uses declared physical ranges and reference values. A zero-variance fitted
standard deviation must never be used as an operating-shift denominator.

## Development qualification before test access

The final test population must not be evaluated unless cross-fitted training plus validation
interventions contain both:

- at least 20% harmful trajectory-candidate units; and
- at least 20% safe trajectory-candidate units.

If this fails, EXP-007A stops as a benchmark-design failure. Candidate corruptions, loss weights,
and thresholds may not be changed after examining sealed-test outcomes.

## Primary gate

All conditions must hold on the fresh sealed test:

1. Mean within-seed safe-intervention AUROC is at least `0.80`.
2. The trajectory-first, seed-second bootstrap 95% AUROC lower bound is above `0.50`.
3. No seed has unexplained all-on or all-off fallback above `90%`.
4. Mean Brier score is lower than the constant-prevalence Brier reference.
5. Fixed all-on physics produces measurable positive regret on harmful candidates.
6. Credibility control reduces mean positive physics regret relative to both fixed all-on and
   validation-selected scalar controls.

Failure blocks EXP-008. Passing permits preparation of EXP-008 but is not evidence that the
synthetic priors are valid real-bearing laws.

## Statistical and serialization corrections

- Compute classification metrics within each seed before seed aggregation.
- Bootstrap complete trajectories within selected seeds, then average seed-level statistics.
- Apply anti-collapse rules per seed, never only on pooled probabilities.
- Record parent experiment seed separately from derived optimization seed.
- Save prediction probabilities with 17 significant digits and recompute every reported metric
  from the serialized prediction files before finalization.
- Report prevalence AUPRC and constant-prevalence Brier baselines.

## Open EXP-007 diagnostics

Allowed-evidence nonlinear ceilings and feature ablations may be run on the now-open EXP-007
artifacts. They are explicitly diagnostic and cannot be cited as confirmation of version 0.2.

## Literature and claim boundary

The DSCN-AttnPINN bearing paper is adapted only as a learned-hidden-operator comparator; it does
not supply an explicit bearing governing law. The multisensory degradation-consistency RNN
supports monotonic weak constraints. Paris' law applies to stage-II crack growth when crack
length, stress-intensity range, cycles, and material parameters are identified; those quantities
are not invented here. SiMBA-style dynamic fusion is prior art and is not the novelty claim.

ANSYS remains deferred. The amendment tests statistical identifiability and negative-transfer
control, which additional finite-element fidelity would not by itself resolve.
