# EXP-007A counterfactual physics-harm runbook

Status: implementation frozen before simulator generation

Experiment ID: `EXP-007A`

Protocol: `research/EXP007A_PROTOCOL_AMENDMENT.md`, version 0.2.1

## Purpose

EXP-007A determines whether causal evidence can identify the safety of an actual
physics-regularized RUL intervention. It corrects EXP-007's zero-variance condition scaling,
pooled probability aggregation, per-seed collapse reporting, probability serialization, and
misalignment between law correctness and RUL harm.

## Frozen simulator design

- Scenario file: `configs/exp007a_multicondition_scenarios.csv`
- Split file: `configs/exp007a_data_split.json`
- Scenario SHA-256: `35c8968f392ecaefcb1bacd69aef0db364824b8dd17d88ca1e380e5b67754e48`
- Training: 64 trajectories, 16 per progression family
- Validation: 16 trajectories, four per family
- Fresh sealed test: 16 trajectories, four per family
- Development/validation simulator seed: `420071`
- Sealed-test simulator seed: `920071`
- Slip: fixed at the previously validated 1% after the preserved upstream array-index failure;
  see the protocol erratum

Generate raw derived results only after committing the method, protocol, scenarios, split, and
gates. Use:

```powershell
matlab -batch "addpath('scripts/matlab'); exp007a_run_multicondition_simulator( ...
  'Bearing_Simulation_Model-main/Bearing_Simulation_Model-main/Simulation_Model.zip', ...
  'configs/exp007a_multicondition_scenarios.csv', ...
  'data/processed_features/publication/exp007a/simulator_results', false)"
```

Then export the deterministic derived cache:

```powershell
matlab -batch "addpath('scripts/matlab'); exp007a_export_multicondition_results( ...
  'data/processed_features/publication/exp007a/simulator_results', ...
  'configs/exp007a_multicondition_scenarios.csv', ...
  'data/processed_features/publication/exp007a/multicondition_features.csv', ...
  'data/processed_features/publication/exp007a/multicondition_metadata.json')"
```

Do not modify raw result MAT files. Record the cache hash in the active configuration and copy
the complete simulator evidence to `saved results/run_07a/simulator/`.

### Preparation execution failures

1. Version 0.2 stopped in upstream signal code at development scenario 48. The partial 1.03 GB
   population is preserved under `saved results/run_07a/simulation_failure_01/`; protocol
   erratum 0.2.1 fixed slip at the previously validated 1%.
2. The first complete 0.2.1 invocation computed all 96 trajectories, but the wrapper allowed
   upstream P-code to overwrite its generic destination variable, so final raw directories were
   deleted with the temporary workspace. No cache or result artifact survived and no RUL/test
   evaluation occurred. The wrapper was corrected by invoking protected P-code inside a
   separate MATLAB function workspace. This changes no scenario, seed, model, target, or gate.
3. The first cache-export attempt stopped before writing either output because MATLAB R2023a
   requires a character vector, rather than a string scalar, when assigning one table variable
   name by brace indexing. The exporter compatibility repair changes no raw data, feature
   definition, identity mapping, split, model, target, or gate.

## Colab workflow

After cache qualification, final validation, commit, and push:

1. Build a fresh ignored `Upload/` using `scripts/build_upload_package.py`.
2. Upload it to `MyDrive`, replacing the preceding Upload folder only after preserving results.
3. Open `train_models_colab.ipynb` and select a T4 GPU.
4. Run all cells. The notebook reads the pushed SHA automatically.
5. If development target qualification fails, retain the failure and do not expose the sealed
   test through another manually edited path.
6. Download `codex_results_bundle.zip` and place it under `results/incoming/`.

Budget 60-120 minutes on a T4. The run is checkpoint-restartable at fold/candidate granularity.

## Decision

EXP-008 remains blocked until the complete EXP-007A bundle passes independent identity,
serialization, metric, calibration, anti-collapse, harm-stress, and regret-reduction checks.
