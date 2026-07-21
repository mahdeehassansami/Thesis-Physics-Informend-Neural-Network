# EXP-006 issues and limitations

- Supplied v2 progression-family and fault labels are intentionally unavailable.
- 1 supplied trajectory has at
  most eight snapshots and cannot support the old fixed sequence construction.
- The simulator is distributed primarily as MATLAB P-code. Its inputs and outputs are
  auditable, but internal implementation lines cannot be independently inspected.
- Simulator truth establishes controlled synthetic validity, not real bearing-mechanics
  validity or a resolved simulation-to-real domain gap.
- The dataset documentation describes dynamic load rating as 32,000 N, while the stored
  supplied structures and official template use 32,500 N. Both values must remain visible;
  do not silently reconcile them.
