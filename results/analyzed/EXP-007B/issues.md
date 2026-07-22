# EXP-007B issues

1. **The primary improvement gate failed.** Verified improvement is
   `0.626133%`, below the frozen `1.0%` requirement.
2. **The average gain is uncertain.** The signed-regret bootstrap interval
   `[-0.002895, 0.000924]`
   crosses zero, and seed 3042 regressed.
3. **Credibility does not generalize strongly.** Mean AUROC is
   `0.560017` and mean Brier score is worse than prevalence; test
   coverage ranges from `3.64%` to
   `68.51%` across seeds.
4. **Risk is lifecycle- and family-dependent.** Early-life, gamma, and linear-increasing slices
   show positive mean regret despite acceptable pooled full-run safety.
5. **The fresh test population is now open.** Threshold, feature, blend, family, or loss changes
   motivated by EXP-007B must be treated as exploratory and confirmed on another untouched
   simulator population.
6. **External validity remains limited.** This experiment uses normalized RUL from controlled
   synthetic feature trajectories; it does not establish performance on raw vibration,
   real-bearing failures, MATLAB simulation, or ANSYS physics.
