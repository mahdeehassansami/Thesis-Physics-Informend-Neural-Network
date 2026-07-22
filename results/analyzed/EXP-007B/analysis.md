# EXP-007B verified analysis

## Validity and execution

EXP-007B is a valid completed execution of commit `76791b22a84fae52dac6737631a2d4add986ef45` on the required
Tesla T4. All 741 inventoried artifacts and all
378 lightweight-bundle entries passed size, SHA-256, and safe-path
checks. The frozen configuration, fresh simulator seed 920072, 64/16/16 split, scenario, feature
cache, metadata, and five neural seeds all match the committed inputs. Every development gate
was reproduced before test access, every seed completed, and no failed model, OOM retry, or
non-finite training value was found.

The complete run took `25.3` minutes: approximately
`20.5` minutes in the five seed jobs and
`4.3` minutes after the last seed
for aggregation, hashing, plots, and export. The short runtime is credible because this is an
8,268-row synthetic feature-sequence experiment with a 22,625-parameter backbone, not raw-signal
training.

## Confirmatory endpoint

The causal controller reduced mean macro-run RMSE from `0.161905` to `0.160892`
normalized RUL, a `0.626133%` improvement. This misses the preregistered
`1.0%` minimum, so the combined confirmation gate **fails**. The controller passed every safety
and exposure constraint: mean positive regret `0.001908` (limit `0.005`), harmful-run
fraction `0.0625` (limit `0.10` at regret margin `0.01`), maximum run regret
`0.013773` (limit `0.05`), and pooled coverage `0.2936` (allowed
`0.05-0.90`). The verified decision is `stop_and_preserve_exp007b_negative_result`.

The controller improved 28/80 complete trajectory-seed units and four of five
seed-level macro averages, but seed 3042 regressed by
`0.544%`.
The signed-regret bootstrap 95% interval is
`[-0.002895, 0.000924]`, which
crosses zero. Thus the modest average gain is neither large enough for the frozen gate nor
statistically stable in this hierarchy.

Always-on physics achieved `1.408%`
average improvement, more than the controller, but its harmful-run fraction was
`0.1125`, above the `0.10` safety limit. Oracle selection
shows substantial headroom (`14.79%`
improvement), confirming that intervention choice matters even though the learned selector did
not capture enough of it.

## Credibility, lifecycle, and selection behavior

Safe-intervention discrimination was weak: mean seed AUROC was
`0.560017 +/- 0.011059`, with bootstrap
95% interval `[0.500932,
0.618152]`. Mean Brier score `0.246173`
was worse than the constant-prevalence reference `0.243837`;
only one seed beat that reference. AUROC was secondary in EXP-007B, but this near-chance ranking
helps explain why development improvement did not generalize.

Intervention coverage varied sharply by neural seed, from
`0.2936` pooled but
`0.0364` to `0.6851` per seed.
`83.6%` of selected samples used the 0.60 time-scale prior, and only
`14.8%` selected the simulator's named true family. Family match
is not the optimization target, but the concentration shows that the controller mainly learned
a conservative short-time-scale correction rather than recovering the generating law.

The lifecycle result is asymmetric. Early-life intervention increased mean segment regret by
`0.003925`, while middle and late life improved it by
`0.002684` and `0.003240`.
Gamma and linear-increasing trajectories had positive mean full-run regret
(`0.002755` and
`0.001373`), whereas progressive and step-like
trajectories benefited. These are opened-test diagnostics and cannot be used to retune a new
confirmatory method.

## Training behavior and interpretation

All 325 expected fits have finite histories and separate nonnegative data, prior-value,
prior-rate, and monotonic losses. Median recorded training length was
`9` epochs and the maximum was
`52`; the five final backbones selected best epochs
[2, 6, 6, 7, 7]. This is consistent with functioning early
stopping and checkpoint reuse, not skipped training.

EXP-007B therefore supports a narrower finding than hoped: causal abstention reduced the tail
risk of physics intervention, but the frozen controller did not deliver the preregistered average
RUL improvement on a fresh population. Higher-fidelity or real-bearing confirmation is not yet
authorized by this gate.
