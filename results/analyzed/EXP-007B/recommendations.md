# EXP-007B recommendations

1. Preserve EXP-007B as a valid negative confirmation. Do not relabel the completed run or
   relax its 1% gate after seeing the test result.
2. Do not proceed directly to the planned higher-fidelity/real-bearing EXP-008. The current
   controller first needs a development-only redesign that improves benefit without losing the
   demonstrated tail-risk control.
3. Use EXP-007B only to formulate hypotheses. Candidate directions are direct regret or
   benefit prediction, partial pooling of thresholds across neural seeds, and lifecycle-aware
   abstention. Select among them using development trajectories only, not these 16 test runs.
4. Strengthen the next protocol with multiple newly generated simulator population seeds.
   Five neural initializations quantify optimization variation but do not quantify variation in
   the test population itself.
5. Predefine both an average-benefit interval and safety bounds. Retain macro trajectory RMSE,
   mean positive regret, harm rate, maximum regret, and coverage so a method cannot pass by
   simply abstaining.
6. Once a revised controller passes a fresh synthetic confirmation, use MATLAB or ANSYS to
   create an independently governed degradation benchmark, then freeze the method before any
   real-bearing evaluation. ANSYS is valuable at that stage, not as a substitute for repairing
   the selector's generalization.
7. A publishable narrative is possible even if the method remains negative: matched physics
   laws can cause negative transfer, causal selective intervention controls tail harm, and
   apparent development gains can fail on a fresh degradation population. That claim requires
   the next study to quantify population-seed uncertainty explicitly.
