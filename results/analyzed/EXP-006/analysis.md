# EXP-006 data and physics-identifiability qualification

Status: **completed**

## Outcome

EXP-006 passed its qualification gates without neural-network training. The supplied
Bearings with Varying Degradation Behaviors v2 file was preserved unchanged and exported
to a compact derived feature cache. It contains 40 trajectories and
1492 vibration snapshots.

The supplied dataset cannot be used as known-truth evidence for physics applicability.
Its documentation intentionally withholds the degradation progression and fault type, and
the exported cache therefore records `truth_available=false` rather than inferring labels.

The official CC BY 4.0 MATLAB simulator ran all 40 predeclared controlled
scenarios with seed 42006 on 9.14.0.2206163 (R2023a). The controlled
cache contains 3997 snapshots and retains the progression family,
hidden degradation value, fault location, bearing parameters, operating conditions, and
simulation details. This is the known-truth benchmark for EXP-007.

## Important data limitation

The supplied v2 data contain trajectories with as few as
6 snapshots. The frozen sequence length of eight from
Runs 4/5 cannot create samples for these runs. EXP-007 must declare a controlled sequence
policy or a causal variable-length model; it may not silently discard the short lives.

## Physics conclusion

Only the simulator progression family is known truth in the controlled benchmark. Paris
crack growth, ISO 281/L10-Miner, and temperature-lubrication terms remain conditional or
unidentified for the current real datasets. A low residual to those equations must not be
reported as proof that their physical assumptions are valid.

## ANSYS decision

ANSYS is not required for EXP-007, whose falsifiable target is progression-family
applicability and negative-transfer prevention. It may add a later, independent validation
layer if we design geometry/load-specific contact stress and crack-growth simulations with
measured units and a simulation-to-real gap analysis.
