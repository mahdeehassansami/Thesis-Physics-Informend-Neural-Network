# EXP-007A recommendations

1. Do not prepare EXP-008 from these results. Record EXP-007A as a valid negative gate outcome.
2. Use the opened EXP-007A test artifacts only for diagnostics: localize the PriorCred regret
   tail by family, scale, condition, seed, and lifecycle; do not tune a confirmatory score on it.
3. Before a fresh run, replace the impossible strict comparison against zero positive regret
   with a preregistered clinically/engineering-meaningful risk objective, such as a non-inferiority
   margin, upper-tail regret bound, or constrained average-improvement plus harm-rate criterion.
4. Train the selector for the decision objective directly. AUROC ranking alone does not control
   rare large regret; test asymmetric or cost-sensitive learning using development data only.
5. Focus development diagnostics on gamma trajectories and high-regret 1.6-scale priors, while
   preserving progression-family and operating-condition coverage.
6. Any corrective EXP-007B must use a newly generated sealed simulator seed and a frozen protocol.
   The current 16 test trajectories cannot be reused as confirmation.
7. Defer ANSYS and real-bearing confirmation until the synthetic safety gate is coherent and
   passes on a fresh population. Higher-fidelity physics will not fix an ill-posed decision gate.
