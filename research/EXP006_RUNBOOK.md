# EXP-006 data and physics qualification runbook

Status: completed

Experiment ID: `EXP-006`

Qualification source commit: `dc134aeefcfe7609d51d074974c6e39e7ef14ee2`

## Purpose

EXP-006 determines whether the supplied synthetic dataset and the official bearing simulator
can support the known-valid versus misspecified physics study defined in the publication
protocol. It performs simulation, signal feature extraction, schema validation, fingerprinting,
and split validation. It does not train a neural network.

## Result

- The supplied v2 dataset contains 28 official training and 12 official test lives, totaling
  1,492 snapshots.
- Its documentation intentionally withholds degradation-family and fault-location labels.
  It is suitable for RUL generalization but not known-truth physics-validity classification.
- One supplied trajectory has six snapshots, so an eight-step sequence produces no sample for
  it. EXP-007 must resolve this without silent exclusion.
- The official CC BY 4.0 MATLAB simulator completed 40 predeclared scenarios in 198.483 seconds
  with seed 42006.
- The controlled cache contains 3,997 snapshots across `linear_increasing`,
  `progressively_increasing`, `step_like`, and `gamma` progression families.
- Every controlled snapshot retains family, hidden degradation, fault location, bearing
  parameters, operating conditions, and simulation settings.
- All controlled trajectories have at least 20 snapshots.
- The immutable controlled split is 24 training, 8 validation, and 8 test trajectories. The
  validation and test sets introduce predeclared load, speed, and noise shift.

The verified report is under `results/analyzed/EXP-006/`.

## Environment

- MATLAB R2023a (`9.14.0.2206163`)
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- Communications Toolbox
- Python 3.11 project environment

The upstream simulator requires MATLAB R2022a or later and Windows. Its P-code is treated as an
external CC BY 4.0 reference implementation. Do not modify the reference archive.

## Authoritative files

- `configs/exp006_data_qualification.json`
- `configs/exp006_controlled_simulation_scenarios.csv`
- `configs/publication_data_split.json`
- `configs/physics_priors.yaml`
- `scripts/matlab/exp006_signal_features.m`
- `scripts/matlab/exp006_export_supplied_synthetic.m`
- `scripts/matlab/exp006_run_controlled_simulator.m`
- `scripts/matlab/exp006_export_controlled_results.m`
- `scripts/qualify_exp006.py`

## Local derived evidence

The raw datasets remain unchanged. Derived artifacts are stored under:

```text
data/processed_features/publication/exp006/
|-- supplied_synthetic_features.csv
|-- supplied_synthetic_metadata.json
|-- controlled_synthetic_features.csv
|-- controlled_synthetic_metadata.json
`-- controlled_simulator_results/
```

The full local preservation copy is under `saved results/run_06/`.

Important fingerprints:

- Supplied `Data.mat`: `94d0c09cc53bb85a59a54f141ffcc8aab8d7f098424fabba7d20ca19ce560728`
- Supplied feature cache: `55908929140aa66c0fac78288718ab62f66b477071334206baf29f3e20c6f096`
- Controlled feature cache: `3199282d5abf674538797b41dc97240825cf6ec80853dffb5c9f8ca4f45bfdae`
- Forty signal MAT files plus overview combined fingerprint:
  `e95127afa4a4cd8791b2351e878e7d8a1fe16e908cb71035e41cd8095276f30b`

The complete per-file inventory is in `results/analyzed/EXP-006/dataset_manifest.json`.

## Reproduction order

From the repository root, using the exact source commit:

1. Add `scripts/matlab/` to the MATLAB path.
2. Run `exp006_export_supplied_synthetic` using the three paths declared in the EXP-006 JSON.
3. Run `exp006_run_controlled_simulator` with the official simulator ZIP, scenario CSV, output
   root, seed 42006, and `smoke_only=false`.
4. Run `exp006_export_controlled_results` on the resulting directory.
5. Run:

   ```powershell
   .\.venv\Scripts\python.exe scripts\qualify_exp006.py
   ```

The simulator runner refuses to overwrite an existing result directory. Preserve or move an
existing run before intentionally reproducing it.

## MATLAB and ANSYS decision

MATLAB is sufficient for EXP-007 because the official simulator supplies labeled progression
truth and hidden degradation trajectories. ANSYS is not required for the next experiment.

ANSYS may become scientifically useful later for a separate geometry/load-specific validation
of contact stress, stress-intensity range, crack propagation, or thermomechanical effects. That
study must specify geometry, materials, boundary conditions, mesh convergence, units,
calibration, and simulation-to-real limitations. It should be added only after the basic
credibility mechanism passes the controlled EXP-007 gate.
