# EXP-007B protocol: causal selective physics-risk confirmation

Protocol version: **0.3.0**  
Status: **preregistered before fresh-test generation**

## Purpose

EXP-007B is a controlled confirmation of the only EXP-007A correction that met the nested
development criteria. It asks whether a prefix-local, abstaining controller can retain useful
physics-informed RUL improvement while limiting negative transfer on a newly generated sealed
simulator population.

EXP-007A's test trajectories are open diagnostic data. They are forbidden for EXP-007B model
selection, threshold selection, feature changes, or success decisions.

## Evidence and hypothesis

EXP-007A's complete-trajectory score was not available causally at early prediction times.
After correcting the decision level in a nested development-only audit, standardized logistic
safe-intervention ranking with a fixed 50% maximum physics blend improved macro-run RMSE by
`1.7659%`, with mean positive regret `0.002426`, harmful-run rate `0.09`, and maximum regret
`0.048386` across 400 trajectory-seed holdouts.

Hypothesis: on a fresh sealed simulator population, this frozen causal controller will improve
mean macro-run RUL RMSE by at least 1% relative to its exact data-only parents while keeping
mean positive control regret at or below `0.005`, harmful-run frequency at or below `0.10`,
and maximum run regret at or below `0.05`.

## Frozen data design

- Development trajectories: the exact 64 train and 16 validation trajectories generated with
  simulator seed `420071` for EXP-007A.
- Fresh sealed trajectories: the same 16 declared test conditions and progression-family
  balance, newly simulated with seed `920072`.
- Split unit: complete trajectory. No window, prefix, candidate, or seed variant of one
  trajectory may cross data roles.
- Feature construction, sequence length (`5`), stride (`1`), normalization, RUL definition,
  candidate physics models, candidate scales, backbone, optimizer, epochs, and five neural
  seeds remain unchanged from EXP-007A.
- EXP-007A seed `920071` test trajectories are excluded from the EXP-007B cache.

The scenario CSV, split JSON, derived-cache hashes, and exact Git commit are parameter
authorities. The fresh-test data may not be evaluated until every seed passes development
target and selector-threshold qualification.

## Frozen selector

The selector is an adaptation newly proposed for this bearing study; it is not reproduced from
the AttnPINN paper or claimed as a conformal algorithm.

1. Inputs are the same label-safe evidence fields used by EXP-007A and candidate-family one-hot
   indicators. Forbidden target, degradation truth, total-life, failure-time, and future fields
   remain excluded.
2. The selector is fitted separately for each neural seed on cross-fitted training evidence.
3. Training uses 12 evenly spaced observed prefixes per complete trajectory-candidate unit so
   long trajectories do not dominate. Each row contains only information available at that
   prefix.
4. The model is standardized logistic regression with balanced classes and `C=1.0`.
5. No Platt calibration is performed on the threshold-selection population.
6. At each prefix, candidates are ranked by safe-intervention probability. At most the single
   highest-scoring candidate can be used.
7. If its score is below the frozen threshold, prediction falls back exactly to data-only.
8. If selected, the output is `data_rul + 0.50 * (physics_rul - data_rul)`, clipped to `[0,1]`.

## Development threshold qualification

For each neural seed, thresholds are evaluated only on the 16 validation trajectories. The
chosen threshold minimizes validation macro-run RMSE among thresholds satisfying all of:

- mean positive run regret `<= 0.005`;
- harmful-run fraction `<= 0.10`, where harm is run RMSE regret `> 0.01`;
- prefix intervention coverage in `[0.05, 0.90]`.

If no threshold is feasible for any seed, that seed records a development-gate failure, the
fresh test remains untouched, and EXP-007B fails. All five seeds must qualify before the
fresh-test evidence is pooled.

## Confirmatory endpoints

The primary analysis unit is a complete test trajectory within neural seed (`16 x 5 = 80`
paired units). All metrics use normalized RUL unless explicitly labelled otherwise.

All primary gates must pass:

1. causal controller mean macro-run RMSE improves by at least `1.0%` versus data-only;
2. mean positive run regret is `<= 0.005`;
3. harmful-run fraction at the `0.01` regret margin is `<= 0.10`;
4. maximum observed run regret is `<= 0.05`;
5. pooled intervention coverage is within `[0.05, 0.90]`; and
6. every requested seed completed with a frozen feasible development threshold.

Secondary endpoints are mean signed regret, paired hierarchical-bootstrap uncertainty,
safe-intervention AUROC/AUPRC, calibration diagnostics, lifecycle-region risk and coverage,
family/condition/scale slices, training time, and stability across seeds. Secondary metrics do
not rescue a failed primary gate.

## Interpretation and stop rule

- Pass: the synthetic causal safety gate is supported and the next phase may evaluate the
  frozen controller on a higher-fidelity or real-bearing transfer design.
- Fail: preserve the negative result and stop physics-controller escalation. Do not tune on the
  newly opened population; any further correction requires another preregistration and fresh
  seed.

## Methodology references

- Angelopoulos et al., *Conformal Risk Control*, ICLR 2024,
  https://arxiv.org/abs/2208.02814. Used for risk-control framing; no conformal guarantee is
  claimed.
- Gangrade, Kag, and Saligrama, *Selective Classification via One-Sided Prediction*, AISTATS
  2021, https://proceedings.mlr.press/v130/gangrade21a.html. Used for the abstention and
  risk-coverage framing.

