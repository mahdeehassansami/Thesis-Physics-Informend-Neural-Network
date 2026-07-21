# EXP-007A verified analysis

## Validity and execution

EXP-007A is a valid completed execution of commit `76d9f78227f8a8b9ff823c0883097d32d4edfc7e` on a Tesla T4.
All 739 inventoried full artifacts and all
376 lightweight-bundle entries passed size, SHA-256, and safe-path
checks. The configuration, 64/16/16 split, simulator scenario, feature cache, metadata, and five
seeds match the frozen experiment. All seeds completed, the development target gate passed for
every seed, and sealed-test access followed the declared order. No model, OOM, or numerical
failure was reported.

The manifest's `170.6` seconds describes a later resume/finalization
pass. The five saved seed jobs total `1264.0` seconds
(`21.1` minutes), while the first start-to-last-seed
wall interval in the log is `1293.6` seconds. This
runtime-scope discrepancy does not alter predictions but must be reported accurately.

## Independently verified primary endpoint

The test population contains 960 trajectory-candidate-seed units. Mean
within-seed safe-intervention AUROC is `0.666703 +/-
0.059375`. The trajectory-within-resampled-seed bootstrap 95% interval
is `[0.597772, 0.735509]`. The point
estimate is below the frozen `0.80` requirement, although the interval remains above chance.
Mean AUPRC is `0.725538` and mean Brier score is
`0.228371`, better than the `0.246436`
constant-prevalence reference. No seed crossed the 90% all-on/all-off collapse limit.

Performance is heterogeneous by true progression family: step-like AUROC is
`0.807459`, but gamma AUROC is only
`0.567852`. Seed AUROC ranges from
`0.589060` to `0.737500`.

## RUL control outcome

PriorCred's mean macro run RMSE is `0.164284` normalized
RUL, an improvement of `4.96%` over data-only and
`4.02%` over all-on physics. It beats data-only
on `61/80` seed-trajectory pairs and all-on on `65/80`. Oracle selection is
better (`0.150883`), while
anti-oracle is much worse (`0.211217`),
so intervention safety is consequential and useful headroom exists.

The gain is not risk-safe under the frozen definition. PriorCred's mean positive run regret is
`0.004073`, versus `0.003480` for all-on and exactly
`0.000000` for the validation-selected scalar. Its
worst positive regret is `0.078220`.
Every seed selected scalar zero, so the comparator reduced to data-only and had zero positive
regret. The strict requirement that PriorCred be below this nonnegative zero reference was
therefore impossible in this realized run. The criterion remains failed; it cannot be relaxed
after opening the test set.

## Physics interpretation

The harm stress worked: 431 harmful interventions have mean normalized-RUL regret
`0.042912`. However, simulator-family correctness is not a safety
proxy. Correct family/scale candidates are safe only `38.75%` of the time
and have mean regret `0.016419`, compared with `56.59%`
safe and `0.007709` mean regret for the other candidates. This supports the
research premise that a mathematically matched degradation prior can still cause negative
transfer when imposed through a learned RUL model.

## Convergence and stability

All 325 expected fits are represented with finite histories and separate nonnegative data,
prior-value, prior-rate, and monotonic losses. Median training length is
`9` epochs; the maximum is `52`.
The median final-to-best validation-MSE ratio is
`1.162`, consistent with checkpointed
early stopping rather than missing training. The final backbone has 22,625 parameters.

## Decision

EXP-007A fails the frozen publication gate because AUROC is below 0.80 and PriorCred does not
reduce positive regret below both declared comparators. EXP-008 remains blocked. The average RUL
improvement and law-correctness result are promising findings, but this opened synthetic test
set can now support diagnosis only, not confirmation of a revised method.
