# EXP-007A counterfactual physics-harm runbook

Status: simulator and cache qualified; Colab neural run pending

Experiment ID: `EXP-007A`

Protocol: `research/EXP007A_PROTOCOL_AMENDMENT.md`, version 0.2.2

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
  'data/processed_features/publication/exp007a/simulator_results_final', false)"
```

Then export the deterministic derived cache:

```powershell
matlab -batch "addpath('scripts/matlab'); exp007a_export_multicondition_results( ...
  'data/processed_features/publication/exp007a/simulator_results_final', ...
  'configs/exp007a_multicondition_scenarios.csv', ...
  'data/processed_features/publication/exp007a/multicondition_features.csv', ...
  'data/processed_features/publication/exp007a/multicondition_metadata.json')"
```

Do not modify raw result MAT files. The complete simulator evidence is preserved at
`saved results/run_07a/simulator/`; the compact qualified cache remains at the active configured
path and is duplicated under `saved results/run_07a/qualified_cache/`.

Qualified cache identity:

- Feature rows: `7,772` across `96` trajectories.
- Train/validation/test trajectories: `64/16/16`.
- Feature SHA-256: `050db850cc5dd0177fc6c58cb0efb1227f305254aeae7fa0ac18a79974ac35af`.
- Metadata SHA-256: `81f327a8d73312240d43c571a987770322ab64cb45d480fba5269cd0b3a8ee2b`.
- Causal sequence length: `5` under protocol erratum 0.2.2, retaining the two five-snapshot
  training trajectories; validation/test minima are `9/20` snapshots.

### Preparation execution failures

1. Version 0.2 stopped in upstream signal code at development scenario 48. The partial 1.03 GB
   population is preserved under `saved results/run_07a/simulation_failure_01/`; protocol
   erratum 0.2.1 fixed slip at the previously validated 1%.
2. The first complete 0.2.1 invocation computed all 96 trajectories, but the wrapper allowed
   upstream P-code to overwrite its generic destination variable, so final raw directories were
   deleted with the temporary workspace. No cache or result artifact survived and no RUL/test
   evaluation occurred. The wrapper was corrected by invoking protected P-code inside a
   separate MATLAB function workspace. This changes no scenario, seed, model, target, or gate.
3. The first cache-export attempt wrote no output because MATLAB R2023a requires a character
   vector, rather than a string scalar, when assigning one table variable name by brace indexing.
   The second wrote the atomic feature CSV but stopped before metadata because R2023a rejects
   `numel` as the method in the used `groupsummary` call form. That incomplete CSV is preserved
   under `saved results/run_07a/export_failure_02/` and cannot be treated as a qualified cache.
   The exporter compatibility repairs select the equivalent default `GroupCount` and change no
   raw data, feature definition, identity mapping, split, model, target, or gate.
4. Structural qualification found two five-snapshot training trajectories. Protocol erratum
   0.2.2 reduced sequence length from eight to five before neural training so all 96 frozen
   trajectories remain eligible. No split, feature, target, architecture capacity,
   intervention, seed, or gate changed.
5. The first Colab handoff stopped during dependency installation before configuration loading
   or training. The requirements file contains the editable path `-e .`, but pip was launched
   from `/content`, so it resolved `.` outside the clone. A non-installing local pip dry-run
   reproduced the failure and succeeded when launched from the repository. The notebook now
   sets `cwd=CLONE` and does not suppress pip output. No data, model, target, seed, or gate
   changed, and the failed invocation produced no experiment result.

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
