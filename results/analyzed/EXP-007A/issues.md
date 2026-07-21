# EXP-007A issues

1. **Primary discrimination gate failed.** Verified mean seed AUROC is
   `0.666703`, below `0.80`; gamma-family AUROC is especially weak at
   `0.567852`.
2. **Tail-risk gate failed.** PriorCred improves average RMSE but has mean positive regret
   `0.004073` and maximum `0.078220`.
3. **The scalar comparator became a zero-risk impossibility bound.** Validation selected scalar
   zero for all five seeds. Its positive regret is zero, so a strict lower-than comparison cannot
   be satisfied by a nonnegative positive-regret statistic. This is a protocol-design finding,
   not permission to rescore EXP-007A.
4. **Seed and family heterogeneity remain.** Seed AUROC spans `0.589060` to
   `0.737500`; true-family AUROC spans
   `0.567852` to `0.807459`.
5. **Manifest elapsed time is incomplete.** It records the final resume/finalization pass
   (`170.6` seconds), not the approximately
   `21.1` minutes of saved seed training.
6. **The test population is now open.** Further feature, threshold, candidate, or loss changes
   evaluated on these 16 trajectories are exploratory and cannot validate the revised method.
