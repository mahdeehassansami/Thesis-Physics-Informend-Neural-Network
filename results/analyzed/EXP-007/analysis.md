# EXP-007 independent analysis

Status: **valid completed run; predeclared scientific gate failed**

## Facts verified from artifacts

- The run used exact clean commit `86e1e357f91f5c5e79d7f9d0fdcfd457ef09afe8` on a Tesla T4.
- Configuration, immutable 24/8/8 split, controlled-cache fingerprint, executed-notebook source,
  artifact inventory, and lightweight ZIP all match. All five seeds completed; no model failure
  or OOM was recorded.
- The complete run took 282.1 seconds.

## Independently recomputed credibility result

The saved report pooled differently calibrated probabilities across seeds and reported AUROC
`0.542776`. That is not the locked seed-level aggregation. Recomputing
AUROC within each seed gives mean `0.660876` (SD
`0.098617`) with a trajectory-first, seed-second bootstrap 95% interval
`[0.530015, 0.811965]`. The mean-probability
five-seed ensemble AUROC is `0.753676`. All remain below the required
`0.80` point estimate.

Mean AUPRC is `0.227513` versus class prevalence `0.15`. Mean
Brier score is `0.142381`, worse than the constant-prevalence reference
`0.127500`. Four of five seeds exceed 90% all-off fallback; three are exactly all-off
at the trajectory-candidate threshold level. The pooled 85.8% fallback rate therefore hides
seed-level collapse.

The diagnostic is strongly family- and corruption-dependent. Gamma mean AUROC is about 0.385,
and valid-versus-slow-time-scale AUROC is about 0.459, both worse than chance. Linear and
progressive families are easier, but their seed variation is large.

## Failure mechanism

Training uses only one speed/SNR condition. StandardScaler assigns scale 1 to those zero-
variance features, making median operation-shift score rise from about 0.31 in training to
69.4 in validation and 138.8 in test. Median covariate-shift score similarly rises from about
0.87 to 32.1 and 64.25. The credibility classifier therefore extrapolates far beyond its
training support. Validation thresholds yield roughly half fallback, but four test seeds
collapse almost completely off.

The target is also misaligned with the intended negative-transfer question. Corrupt priors did
not generally harm RUL prediction: mean all-on corrupt-prior macro regret is
`-0.024061`, and PriorCred corrupt-prior regret is `-0.003072` (negative
means improvement over data-only). Validation-selected scalar blending ranks first at mean
macro RMSE `0.234496`, while data-only is
`0.271236`. Because even wrong priors usually help this weak
backbone, law-correctness labels are not a usable proxy for intervention harm in this setup.

Parameter time-scale recovery is poor: mean absolute factor error is
`0.261560`. Final backbones also vary substantially: best
epochs are 103, 21, 160, 1, and 1, and final validation-to-training MSE ratios range from about
2.7 to 26.4.

## Conclusion

H1 is not supported, anti-collapse fails, and the benchmark does not express the required
negative-transfer stress. Do not proceed to EXP-008 or claim credibility-guided physics. This
is a useful negative feasibility result and a precise diagnosis of identifiability, scaling,
calibration, and target-alignment problems.
