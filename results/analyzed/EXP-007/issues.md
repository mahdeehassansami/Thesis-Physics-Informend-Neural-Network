# EXP-007 issues

1. The required AUROC >= 0.80 gate failed under both corrected seed aggregation and a mean-probability ensemble.
2. Four of five seeds exceed 90% all-off fallback; pooled averaging concealed this collapse.
3. Zero-variance source speed/SNR features make validation/test shift scores tens to hundreds of nominal standard deviations.
4. Slow time-scale corruption and gamma-family applicability are worse than chance.
5. Corrupt priors usually improve rather than harm the data-only backbone, so the benchmark does not instantiate the negative-transfer endpoint.
6. Mean Brier score is worse than a constant class-prevalence predictor, showing poor calibration.
7. The saved gate pools probabilities across seeds instead of aggregating seed-level AUROC as required by the protocol.
8. Seed 4042 job_result reports AUROC 0.597733, while its saved predictions reproduce 0.574142; serialized predictions are authoritative.
9. Cross-fit histories record derived optimization seeds (for example 143 and 244) without a separate parent experiment-seed field.
10. Time-scale parameter recovery and backbone convergence are unstable.
