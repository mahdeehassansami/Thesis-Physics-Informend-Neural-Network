# EXP-005 issues

No structural, identity, hash, metric-reproduction, preprocessing-reconstruction, or job-completion defects were found.

Scientific limitations remain:

- All three models have negative equal-bearing macro R-squared.
- Weak-PINN improved on only two of four held-out bearings.
- Weak-PINN macro late-life MAE worsened despite a lower worst absolute late-life bias.
- IMS-DS2/B1 is a counterexample where the measured feature shift decreased but Weak-PINN and LSTM errors increased sharply.
- The experiment covers four IMS trajectories and cannot establish cross-dataset generalization.
