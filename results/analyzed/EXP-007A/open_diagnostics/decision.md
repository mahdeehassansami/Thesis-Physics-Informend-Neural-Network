# EXP-007A opened-diagnostic decision

## Decision

Proceed to a separately preregistered **EXP-007B** confirmation run. The EXP-007A test
population is permanently diagnostic-only; EXP-007B must use a new simulator seed and may
not be rescored or redesigned after that population is opened.

## Direct observations

- EXP-007A's rolling evidence columns are causal, but its logistic score is fitted and applied
  to complete-trajectory means. Every trajectory-candidate therefore receives one constant
  score, including at early prediction times before the later evidence exists.
- Candidate negative transfer is concentrated in the `gamma` truth family and the `1.60`
  time-scale prior. On the opened test, their harmful fractions are `0.625` and `0.5625`,
  respectively.
- The original PriorCred controller intervened at every lifecycle sample. Its early-life regional
  mean regret was `0.013987`, harmful fraction `0.475`, and maximum regret `0.107786`; middle and
  late mean regrets were negative. This localizes the tail to exactly the period where a
  complete-trajectory score is least defensible as a causal decision.
- The old strict comparison against validation scalar control is ill-posed because that
  comparator selected zero physics for every seed and therefore had zero positive regret.

## Development-only method comparison

The corrective controller was selected using five nested, trajectory-grouped folds per neural
seed. In every outer fold, 48 trajectories fitted the selector, 16 calibrated its threshold,
and 16 remained held out. Candidate variants and causal prefixes from a trajectory never
crossed those roles. The opened EXP-007A test did not choose the method.

The selected controller is standardized logistic safe-intervention ranking evaluated at each
observed prefix. It selects at most one candidate, abstains to the exact data-only parent when
the threshold is not met, and limits the move toward the selected physics prediction to 50%.
Across 400 held-out trajectory-seed units it achieved:

- macro-run RMSE `0.173169` versus data-only `0.176282` (`1.7659%` improvement);
- mean positive control regret `0.002426`;
- harmful-run rate `0.0900` at the preregistered `0.01` regret margin;
- maximum observed control regret `0.048386`;
- mean intervention coverage `0.2978`; and
- feasible calibration in `96%` of outer folds.

The old opened test was used only as a stress audit after method selection. It did not reveal a
reason to cancel confirmation, but its numbers are not confirmatory evidence.

## Methodological scope

The next method is a causal selective risk controller, not a formally conformal controller.
[Conformal Risk Control](https://arxiv.org/abs/2208.02814) motivates explicit monotone risk
calibration, while [Selective Classification via One-Sided Prediction](https://proceedings.mlr.press/v130/gangrade21a.html)
motivates abstention and risk-coverage reporting. The available 16-trajectory validation set and
the present controller loss do not support an honest finite-sample conformal guarantee, so none
will be claimed.
