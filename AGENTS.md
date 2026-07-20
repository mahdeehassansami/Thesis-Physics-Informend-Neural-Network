# AGENTS.md - Bearing RUL Thesis Workflow

## 1. Purpose

This file contains the persistent operating instructions for Codex in this
repository. Read it before inspecting, modifying, validating, or analyzing the
project.

The objective is to develop a rigorous and reproducible bearing remaining
useful life (RUL) thesis workflow in which:

1. Codex modifies and validates the local project.
2. The user commits and pushes an exact project version to GitHub.
3. The user runs the prepared notebook on a Google Colab T4 GPU.
4. Colab stores the complete experiment in Google Drive.
5. Colab creates a lightweight result bundle for Codex.
6. The user downloads that bundle into this repository.
7. Codex verifies and analyzes the evidence.
8. Codex prepares the next justified experiment.
9. The cycle repeats until the final controlled comparison is complete.

Do not depend on conversational memory. The repository, experiment
configuration, fixed split, Git commit, manifests, and downloaded artifacts are
the source of truth.

Do not begin implementing the entire workflow merely because this file exists.
Carry out the user's current request, and use these rules whenever that request
touches data, models, experiments, Colab, results, or the thesis.

## 2. Instruction priority and source of truth

Use the following project-level priority when interpreting the work:

1. The user's current explicit request.
2. This `AGENTS.md`.
3. The active experiment configuration and saved data split.
4. The manifest and artifacts belonging to the experiment being analyzed.
5. `Instructions.pdf`, especially its final iterative-workflow sections.
6. The root `README.md`, relevant papers, dataset documentation, and upstream
   model documentation.

If two project files disagree, do not silently choose the more convenient
version. Identify the disagreement and either resolve it from stronger evidence
or report it to the user.

The later sections of `Instructions.pdf` describe the intended final workflow.
Treat its earlier notebook-only snippets as examples rather than a competing
architecture.

## 3. Current repository context

The repository currently contains:

- A local thesis pipeline under `src/thesis_work/`.
- Tests under `tests/`.
- The historical notebook
  `Thesis_v3_with_extra_graphs_tables.ipynb`.
- Thesis material under `thesis/`.
- Generated local artifacts under `outputs/`.
- Locally extracted IMS data and feature caches under `data/`.
- Additional datasets under `Datasets/`.
- Research papers under `papers/`.
- An upstream AttnPINN reference repository under
  `AttnPINN-for-RUL-Estimation-English/`.
- Other reference or upstream repositories, including bearing simulation,
  general PINN, and RUL-dataset code.

The root `README.md` describes the existing local NASA IMS pipeline. Preserve
that functioning pipeline while adding the Colab workflow. Do not discard or
silently replace it.

Treat nested upstream repositories as reference material unless the user
explicitly asks to edit them. Prefer adapting required ideas into the primary
project package, with attribution and license awareness, instead of modifying
the reference copies in place.

## 4. Non-negotiable rules

### 4.1 Do not perform full neural-network training locally

Full PINN, AttnPINN, CNN, FNN, and LSTM training belongs in Google Colab.

Allowed local work includes:

- Reading source code, papers, dataset documentation, and metadata.
- Syntax and import checks.
- Unit tests.
- Configuration and schema validation.
- Notebook structural validation.
- Data discovery and small metadata inspections.
- Small preprocessing checks.
- Very small smoke tests when required to prove that code executes.
- Recomputing metrics from downloaded prediction files.
- Regenerating plots from existing results.
- Existing non-training commands such as figure regeneration.

Do not run the root command `uv run thesis-work run` when it would perform the
full training pipeline unless the user explicitly requests and authorizes local
training. Prefer `--skip-training` or a deliberately tiny smoke configuration
when local validation is necessary.

Never present smoke-test metrics as experiment results.

### 4.2 Preserve source data

Raw datasets are immutable inputs.

- Do not rename, move, overwrite, reformat, or delete raw dataset files.
- Do not modify files in `Datasets/` or `data/raw/`.
- Derived data must go to a clearly separate cache or processed-data directory.
- Never place generated outputs inside a raw dataset directory.
- Do not commit large datasets, papers, archives, or checkpoints unless the
  user explicitly asks.
- Before recursive file operations, verify the exact resolved target.

### 4.3 Keep experiments reproducible

Every real Colab experiment must have:

- A unique experiment ID.
- An exact Git commit SHA.
- A saved configuration.
- A saved, immutable data split.
- Recorded dataset identity and version.
- Recorded preprocessing details.
- Recorded random seeds.
- Recorded environment, GPU, and CUDA information.
- Complete metrics, histories, predictions, logs, and status information.

No unidentified run may be used in a thesis comparison.

### 4.4 Do not silently change experimental conditions

Never silently change:

- The train/validation/test split.
- Dataset selection or dataset version.
- RUL-label construction.
- Feature extraction.
- Normalization.
- Sequence length.
- Model architecture.
- Loss definitions or weights.
- Optimizer or learning rate.
- Batch size.
- Early stopping.
- Random seeds.
- Evaluation metrics.
- Failure threshold.

Any such change must be declared in the experiment configuration, rationale,
manifest, and final report.

### 4.5 Evidence before modification

After a result bundle has been downloaded, do not make arbitrary architecture
or hyperparameter changes. First validate and analyze the evidence. Every next
experiment must have:

- A specific observed problem or hypothesis.
- Evidence from the current or prior runs.
- A controlled requested change.
- A statement of what remains unchanged.
- A measurable success criterion.

Prefer one major experimental change at a time. Do not simultaneously alter the
split, preprocessing, architecture, loss, and optimizer because the source of
any improvement would become unknowable.

## 5. Division of responsibility

### 5.1 Codex is responsible for

- Reading the applicable instructions and project state.
- Understanding relevant papers and upstream implementations before changing
  model mathematics.
- Editing local Python source code.
- Editing configurations.
- Creating or updating the Colab notebook.
- Creating validation, artifact, and import utilities.
- Running safe local checks.
- Inspecting downloaded result bundles.
- Independently verifying reported metrics.
- Detecting failures, leakage, unfair comparisons, and reproducibility issues.
- Producing technical and thesis-ready analyses.
- Maintaining cross-experiment comparisons.
- Preparing the next evidence-supported experiment.
- Reporting every changed file and the reason for the change.

### 5.2 The user is responsible for

- Reviewing the prepared changes.
- Committing and pushing the intended project version to GitHub.
- Opening the notebook in Google Colab.
- Selecting and connecting to the GPU runtime.
- Approving Google Drive access.
- Starting the notebook execution.
- Downloading the lightweight result bundle.
- Placing it under `results/incoming/`.
- Telling Codex which completed experiment should be analyzed.

Codex must not claim that it directly controlled the user's Colab runtime.

### 5.3 Google Colab is responsible for

- T4 GPU training and evaluation.
- Prediction generation.
- Best-checkpoint generation.
- Training plots and metrics.
- Complete artifact packaging.

If a T4 is specifically required and another device is assigned, record the
actual device and stop or require explicit user approval to continue. Do not
silently treat a CPU run as a valid T4 experiment.

### 5.4 Google Drive is responsible for

- Persistent storage of complete experiment directories.
- Large checkpoints.
- Full logs and predictions.
- Executed notebooks.
- Archived run packages.

Train primarily under `/content/` for speed. Copy or synchronize finalized
artifacts and useful recovery checkpoints to Drive at controlled intervals.
Avoid excessive fine-grained I/O against mounted Drive.

### 5.5 The local repository is responsible for

- Authoritative source code.
- Configuration and fixed split.
- Colab notebook.
- Lightweight downloaded evidence.
- Analysis reports.
- Cross-experiment comparisons.
- Git history.

## 6. Target project organization

When the workflow is implemented, the intended organization is:

```text
.
|-- AGENTS.md
|-- README.md
|-- requirements-colab.txt
|-- notebooks/
|   `-- train_models_colab.ipynb
|-- configs/
|   |-- experiment.yaml
|   `-- data_split.json
|-- src/
|   `-- thesis_work/
|       |-- data/
|       |   |-- dataset.py
|       |   `-- preprocessing.py
|       |-- models/
|       |   |-- attnpinn.py
|       |   |-- pinn.py
|       |   |-- cnn.py
|       |   |-- fnn.py
|       |   `-- lstm.py
|       |-- training/
|       |   |-- trainer.py
|       |   |-- evaluation.py
|       |   `-- losses.py
|       `-- utils/
|           |-- artifacts.py
|           `-- reproducibility.py
|-- scripts/
|   |-- validate_project.py
|   `-- import_colab_results.py
`-- results/
    |-- incoming/
    |-- analyzed/
    `-- comparisons/
```

This is a target architecture, not permission to perform an unrelated mass
refactor. The project already has modules directly under `src/thesis_work/`.
Migrate them incrementally and preserve public behavior and tests. It is
acceptable to keep a coherent existing module if splitting it would add no
value.

The critical architectural rule is that model definitions, preprocessing,
training, evaluation, and artifact logic live in importable Python modules.
They must not be separately reimplemented inside the notebook.

## 7. Startup and recovery after a break

At the start of a continued workflow task:

1. Read this file.
2. Read the root `README.md`.
3. Check `git status` without modifying the worktree.
4. Inspect the active configuration, fixed split, and relevant manifests.
5. Inspect `results/incoming/`, `results/analyzed/`, and
   `results/comparisons/` if they exist.
6. Identify the highest valid experiment ID and its status.
7. Determine the current phase:
   - workflow preparation;
   - waiting for a Colab run;
   - result import;
   - result validation;
   - analysis;
   - next-experiment preparation; or
   - final comparison.
8. Read only the papers and dataset documentation relevant to the requested
   change.
9. State what evidence exists and what work remains.

Do not rely on a previous chat summary when repository evidence is available.
Do not rerun a completed experiment or redo a completed analysis without a
reason.

If a downloaded run and local code disagree, use the recorded Git commit to
identify the code that actually generated the run. Do not analyze the result as
if the current working tree produced it.

## 8. Literature-first rule

Before modifying AttnPINN, PINN physics, RUL-label formulation, attention,
degradation modeling, or evaluation methodology:

- Read the relevant paper or documentation in `papers/`.
- Read the corresponding upstream model README and implementation.
- Identify the original equation, assumption, or architectural idea.
- Record whether the local implementation is:
  - reproduced;
  - adapted;
  - simplified; or
  - newly proposed.
- Preserve citation and license information.
- Do not claim that a paper supports a change it does not discuss.

If an equation, symbol, or assumption is ambiguous, report the ambiguity
instead of inventing a physics constraint.

## 9. Dataset policy

### 9.1 Current dataset roles

The current dataset collection includes, at minimum:

- IMS bearing data: real run-to-failure vibration trajectories suitable for
  degradation and RUL work.
- Ball Bearing Vibration and Temperature Run-to-Failure data: real
  run-to-failure multimodal data suitable for RUL work.
- Bearings with Varying Degradation Behaviors v2: synthetic run-to-failure
  trajectories with a predefined train/test design, suitable for controlled
  RUL evaluation.
- PRONOSTIA/FEMTO data: real accelerated bearing run-to-failure trajectories
  suitable for RUL evaluation when prepared correctly.
- CWRU data: seeded-fault diagnostic data, primarily suitable for fault
  classification, pretraining, representation checks, or transfer studies.
- CWRU 48 kHz load-1 CNN data and CWRU NumPy derivatives: processed
  classification resources, not independent run-to-failure RUL ground truth.

Refresh the inventory before a new study because the user may add datasets.
Do not assume that every folder in `Datasets/` is suitable for direct RUL
training.

### 9.2 Dataset qualification

Before using a dataset, document:

- Dataset name and source.
- Dataset version or file fingerprint.
- Real or synthetic status.
- Run-to-failure or seeded-fault status.
- Sensors, channels, units, and sampling rate.
- Operating conditions.
- Bearing or unit identifiers.
- Failure endpoint definition.
- Available RUL labels or label-construction method.
- Official train/test split, if any.
- Missing, corrupt, or excluded files.
- License or citation requirements.

### 9.3 Split rules

For run-to-failure data:

- Split by bearing, engine, or complete run whenever possible.
- Never place windows from the same physical trajectory in both training and
  test sets.
- Do not perform a random row-level split after windowing.
- Preserve temporal order.
- Keep test data completely isolated from scaler fitting, feature selection,
  hyperparameter selection, and early stopping.

Save stable sample or unit identifiers in `configs/data_split.json`. The split
file must include enough information to reconstruct membership exactly.

If a dataset provides an official split, preserve it unless the experiment is
explicitly defined as a different split study.

### 9.4 Preprocessing and leakage controls

- Fit normalization, PCA, feature selection, and learned preprocessing on
  training data only.
- Apply the fitted transformation unchanged to validation and test data.
- Save fitted parameters or sufficient metadata to reproduce them.
- Record window length, stride, padding, truncation, resampling, and filtering.
- Record RUL clipping, piecewise-linear labels, normalization range, and
  inverse transformation.
- Do not use future measurements to construct a feature available at an earlier
  prediction time.
- Do not tune using the test set.
- Confirm that prediction and target arrays correspond to the same ordered
  samples.

## 10. Model policy

### 10.1 Model lineup

The active `configs/experiment.yaml` must explicitly list the models for each
experiment.

Treat AttnPINN as the principal self-attention-assisted physics-informed RUL
candidate introduced by the added upstream repository. PINN, CNN, FNN, and LSTM
are comparison candidates described in `Instructions.pdf` and the existing
project. Do not silently substitute AttnPINN for a conventional PINN or remove
a baseline.

When both PINN and AttnPINN are included, label them separately in every
configuration, log, metric, prediction file, plot, and thesis table.

When a model is intentionally excluded, record the reason. When a model fails,
record the failure rather than omitting its row.

### 10.2 Fair comparison

Models being compared must use:

- The same dataset version.
- The same fixed split.
- The same target definition.
- The same test population.
- Equivalent information at prediction time.
- The same fitted preprocessing where applicable.
- Declared and reproducible random seeds.
- The same metric implementation.

Model-specific sequence shapes or optimization settings are allowed only when
declared and justified. Do not describe models as fairly compared if one had
access to additional sensors, future data, a different split, or test-driven
tuning.

Record parameter counts, training time, best epoch, final epoch, and inference
time where feasible.

### 10.3 Physics-informed requirements

For PINN-family models, document:

- The physical or degradation quantity being modeled.
- The governing equation or constraint.
- Units and scaling.
- Boundary and initial conditions.
- Collocation or physics-evaluation points.
- Data loss.
- Physics loss.
- Any monotonicity, positivity, or failure-threshold constraints.
- Weighting between all loss terms.
- Whether weighting is fixed, scheduled, learned, or adaptive.

Save data-loss and physics-loss histories separately. A low combined loss is
not sufficient evidence that the physical constraint behaved correctly.

### 10.4 AttnPINN adaptation

The upstream AttnPINN repository was designed for C-MAPSS aircraft-engine RUL.
Do not assume its input representation or physics terms transfer unchanged to
bearing vibration.

Any bearing adaptation must explicitly document:

- Input feature mapping.
- Sequence construction.
- Attention dimensions.
- Target transformation.
- Bearing-specific degradation assumptions.
- Changed physics terms.
- Differences from the original paper and repository.

Validate tensor shapes and loss semantics before Colab training.

## 11. Experiment definition

Every experiment should have an ID such as `EXP-001`, `EXP-002`, and so on.
One Colab experiment should correspond to one exact Git commit.

Before implementing an experiment, define:

- Experiment ID and descriptive name.
- Goal.
- Evidence or baseline motivating it.
- Hypothesis.
- Dataset and split.
- Requested models.
- Changes from the previous experiment.
- Conditions held constant.
- Seeds.
- Primary and secondary metrics.
- Success criteria.
- Expected artifacts.
- Known risks.

Use `configs/experiment.yaml` as the single parameter authority. At minimum, it
should be capable of representing:

```yaml
experiment:
  id: EXP-001
  name: baseline
  description: Establish controlled model baselines

repository:
  expected_commit: null

data:
  dataset_id: null
  root: null
  split_file: configs/data_split.json
  target: normalized_rul
  preprocessing: {}

models:
  requested: [attnpinn, pinn, cnn, fnn, lstm]

training:
  seeds: [42]
  epochs: null
  batch_size: null
  learning_rate: null
  optimizer: Adam
  mixed_precision: true
  early_stopping: {}
  resume: true

evaluation:
  metrics: [mae, mse, rmse, r2]
  save_best_epoch_metrics: true
  save_final_epoch_metrics: true

artifacts:
  drive_project_directory: /content/drive/MyDrive/Bearing_RUL_Project
  create_lightweight_bundle: true
```

The exact schema may evolve, but there must be only one active value for each
parameter. Do not hardcode competing values in the notebook or model modules.

## 12. Reproducibility requirements

Set and record seeds for:

- Python.
- NumPy.
- PyTorch CPU.
- PyTorch CUDA.
- Data-loader workers.
- Split generation, if a split is being created.

Record deterministic settings and any operations that remain nondeterministic.
If multiple seeds are requested, preserve per-seed results and report aggregate
mean and standard deviation. Do not merge a single-seed and multi-seed result
without making that difference visible.

Record:

- Python version.
- Operating system.
- Git commit.
- Notebook hash or version.
- Package versions.
- PyTorch version.
- CUDA version.
- cuDNN version when available.
- GPU name and memory.
- Dataset fingerprint.
- Split-file fingerprint.
- Configuration fingerprint.
- Start time, end time, and elapsed time.

## 13. Colab notebook requirements

The notebook should be a thin, restartable controller. Its code cells should
call project modules instead of containing independent model implementations.

The notebook must contain clear sections for:

1. User-editable repository and Drive settings.
2. Google Drive mounting.
3. Repository cloning or updating.
4. Checkout and verification of the expected Git commit.
5. Dependency installation.
6. GPU, CUDA, and memory verification.
7. Random-seed and reproducibility setup.
8. Configuration loading and display.
9. Dataset discovery and validation.
10. Fixed-split validation.
11. Preprocessing or processed-cache loading.
12. Model construction.
13. Training and checkpoint resume.
14. Evaluation and comparison.
15. Plot and report generation.
16. Full artifact export to Drive.
17. Lightweight ZIP creation.
18. Bundle download.
19. A printed next-step message for local Codex analysis.

### 13.1 GPU behavior

At the beginning, print:

- `nvidia-smi`.
- CUDA availability.
- GPU name.
- CUDA and PyTorch versions.
- Total and available GPU memory.
- Selected training device.

Selecting a GPU runtime does not ensure that the model uses it. Explicitly move
models, input tensors, targets, and required loss tensors to the selected
device.

Use mixed precision on a T4 when it is numerically appropriate. Record whether
it was enabled. Keep numerically sensitive physics calculations in appropriate
precision when required.

### 13.2 Out-of-memory behavior

An OOM must not silently alter the experiment. The allowed policy must be
declared in the configuration:

- Fail and record the error; or
- Retry with a declared batch-size schedule.

Every retry and effective batch size must be written to the log and manifest.
If gradient accumulation is used, record the effective batch size.

### 13.3 Resume behavior

Support resuming from the latest compatible checkpoint. Before resuming,
validate the model, optimizer, configuration, dataset, split, and experiment
identity. Never resume from a checkpoint belonging to an incompatible run.

## 14. Full Google Drive artifact contract

Create a unique directory such as:

```text
/content/drive/MyDrive/Bearing_RUL_Project/experiments/
  EXP-004_2026-07-16_093015/
```

The complete run directory must contain:

```text
EXP-XXX_timestamp/
|-- run_manifest.json
|-- experiment_config.yaml
|-- git_commit.txt
|-- environment.txt
|-- dataset_summary.json
|-- data_split.json
|-- model_comparison.csv
|-- training_history.csv
|-- training.log
|-- executed_notebook.ipynb
|-- summary.md
|-- failure_report.json
|-- metrics/
|   |-- attnpinn_metrics.json
|   |-- pinn_metrics.json
|   |-- cnn_metrics.json
|   |-- fnn_metrics.json
|   `-- lstm_metrics.json
|-- predictions/
|   |-- attnpinn_predictions.csv
|   |-- pinn_predictions.csv
|   |-- cnn_predictions.csv
|   |-- fnn_predictions.csv
|   `-- lstm_predictions.csv
|-- plots/
|   |-- loss_curves.png
|   |-- actual_vs_predicted.png
|   |-- residual_distributions.png
|   `-- model_comparison.png
|-- checkpoints/
|   `-- *_best.pt
`-- codex_results_bundle.zip
```

Files for models not requested need not exist, but the manifest must state that
they were not requested. A failure report may be empty or explicitly state that
there were no failures.

### 14.1 Run manifest

`run_manifest.json` should include:

- Experiment ID and run ID.
- Status: completed, partial, or failed.
- Git commit SHA.
- Notebook identity.
- Configuration and split hashes.
- Dataset identity and fingerprint.
- Requested, completed, skipped, and failed models.
- Seeds.
- Device, GPU, CUDA, and package information.
- Start, end, and elapsed time.
- Effective runtime parameters and any retries.
- Artifact inventory with relative paths.
- Failure summaries.

### 14.2 Training history

`training_history.csv` should use a consistent long format, with fields such as:

- Experiment ID.
- Run ID.
- Seed.
- Model.
- Epoch.
- Train loss.
- Validation loss.
- Data loss.
- Physics loss.
- Learning rate.
- Elapsed seconds.

Use blank or documented null values for losses that do not apply to a model.

### 14.3 Model comparison

`model_comparison.csv` should contain one row per model and seed or a clearly
documented aggregate format. Include:

- Status.
- MAE.
- MSE.
- RMSE.
- R-squared.
- Any domain-standard RUL score used.
- Best validation epoch.
- Metrics at the best epoch.
- Metrics at the final epoch.
- Training time.
- Parameter count.
- Inference time where available.

Specify whether metrics are in normalized or original RUL units. Prefer
reporting original-unit metrics for interpretation and normalized metrics as
additional diagnostics.

### 14.4 Predictions

Each prediction file should contain enough identifiers to reconstruct errors:

- Dataset ID.
- Bearing or unit ID.
- Run ID.
- Sample or window ID.
- Time step or timestamp.
- Actual RUL.
- Predicted RUL.
- Absolute error.
- Normalized actual and predicted RUL when used.
- Health indicator and operating condition when available.
- Seed and model.

Before saving NumPy or scalar values, detach tensors and move them to CPU.

### 14.5 Failure reporting

Do not omit a failed model. Record:

- Model.
- Status.
- Error type.
- Error message.
- Last completed epoch.
- Checkpoint path if available.
- Whether a retry occurred.
- Effective parameters at failure.

## 15. Lightweight Codex result bundle

The downloadable `codex_results_bundle.zip` must contain:

- `run_manifest.json`.
- `experiment_config.yaml`.
- `git_commit.txt`.
- `environment.txt`.
- `dataset_summary.json`.
- `data_split.json`.
- `model_comparison.csv`.
- `training_history.csv`.
- Prediction CSV files.
- Per-model metric JSON files.
- Plots.
- Logs.
- `summary.md`.
- `failure_report.json`.

Exclude:

- Raw datasets.
- Processed datasets too large for analysis transfer.
- Large model checkpoints.
- Unrelated caches.

The full artifacts and checkpoints remain in Google Drive.

## 16. Local validation before handoff

Before telling the user that an experiment is ready for Colab:

1. Inspect all changed files.
2. Verify that model and training logic is imported from `src/`.
3. Verify that the notebook does not duplicate model definitions.
4. Validate configuration parsing.
5. Validate the fixed split and confirm its hash is unchanged unless the
   experiment explicitly changes it.
6. Run syntax and import checks.
7. Run relevant unit tests.
8. Run the repository validation script when it exists.
9. Validate notebook JSON and required section order.
10. Perform only a minimal smoke test if necessary.
11. Confirm that no full local training was run.
12. Check `git diff` and `git status`.

Current useful commands include:

```powershell
uv sync --extra dev
uv run pytest
uv run thesis-work validate-data
uv run thesis-work run --skip-training
uv run thesis-work regenerate-figures
```

Use only the commands relevant to the current change. Do not regenerate
unrelated outputs or overwrite thesis assets without need.

If a command would be slow, destructive, network-dependent, or train models,
state that before running it and remain within the user's authorization.

## 17. Git and experiment traceability

Before a Colab run, report:

- Experiment ID.
- Files changed.
- Reason for each change.
- Local validations completed.
- Whether the fixed split changed.
- Remaining manual Colab steps.
- Suggested commit message.
- Exact non-destructive Git commands the user can run.

Suggested pattern:

```powershell
git status
git add <explicit paths>
git commit -m "EXP-004 add adaptive AttnPINN loss weighting"
git push
git rev-parse HEAD
```

Do not automatically commit or push unless the user explicitly requests it.
Do not stage raw datasets, papers, large outputs, result bundles, or
checkpoints unintentionally.

The notebook must record the exact checked-out commit in:

- `git_commit.txt`.
- `run_manifest.json`.
- `summary.md`.

If the commit is dirty or differs from the configured expected commit, the
notebook should stop before training unless an explicit override is recorded.

## 18. Importing a completed Colab run

The user will place a downloaded ZIP or extracted directory under:

```text
results/incoming/
```

Do not trust a result bundle merely because it exists. First verify:

1. The archive extracts safely under the intended directory.
2. Required files exist.
3. The manifest and experiment IDs agree.
4. The recorded Git commit is present and plausible.
5. Configuration, split, and dataset identities are complete.
6. The split matches the intended or preceding experiment.
7. All requested models have a completed, failed, or skipped status.
8. Metrics are finite and use known units.
9. Predictions and targets have matching lengths and identifiers.
10. Normalization and inverse transformations are correct.
11. Metric calculations reproduce from prediction files.
12. Best-epoch and final-epoch results are distinguishable.
13. No model silently used a different test set.
14. Plots correspond to the numerical files.
15. Logs do not reveal hidden retries or configuration changes.

Reject or clearly flag a run if its experiment ID, Git commit, dataset version,
split, configuration, or seed cannot be identified.

Never execute arbitrary scripts contained in a downloaded result bundle.

## 19. Analysis requirements

For a valid completed experiment, analyze:

- Training and validation convergence.
- Overfitting and underfitting.
- Data-loss and physics-loss behavior.
- Learning-rate behavior.
- Early stopping and best epoch.
- Prediction residuals and error distributions.
- Error versus lifecycle position.
- Best and worst bearings, units, or runs.
- MAE, MSE, RMSE, R-squared, and any declared RUL score.
- Original-unit and normalized metrics.
- Training time, parameter count, and stability.
- Variation across seeds.
- Failure or OOM behavior.
- Suspiciously strong results.
- Possible train/test leakage.
- Fairness of the comparison.
- Whether the experiment hypothesis was supported.
- Whether the requested change improved the target metric without causing
  unacceptable regressions.
- Comparison with the latest valid previous experiment.

Create:

```text
results/analyzed/EXP-XXX/
|-- analysis.md
|-- verified_metrics.csv
|-- issues.md
`-- recommendations.md
```

Update an appropriate cross-experiment table or report under:

```text
results/comparisons/
```

The technical analysis must separate:

- Facts directly observed in artifacts.
- Metrics independently recomputed by Codex.
- Interpretations.
- Hypotheses for the next run.

Do not call a model "best" without naming the metric, unit, split, and
aggregation method.

## 20. Preparing the next experiment

Only prepare the next experiment after the current result has been validated
and analyzed, unless the user explicitly asks to establish the first baseline
or repair an execution failure.

The next experiment report must state:

1. Whether the previous run is valid.
2. Which model performed best and by which metric.
3. Whether the previous hypothesis succeeded.
4. Problems detected.
5. The single main objective of the next experiment.
6. Evidence supporting the proposed change.
7. Files to modify.
8. Variables held constant.
9. Success criteria.

Reasonable experiment progression is:

- Establish a clean baseline.
- Correct data, metric, or preprocessing problems.
- Validate the physics formulation.
- Improve loss balancing.
- Tune optimization or capacity in a controlled manner.
- Run a multi-seed final comparison.

Do not tune against the final test set. Use training and validation evidence for
model selection, then evaluate the frozen decision on the test set.

## 21. Reporting failures and blockers

Failures are evidence and must be preserved.

If training or analysis fails:

- Do not silently skip the failed component.
- Preserve the traceback or concise error.
- Record the last successful step.
- Identify whether partial artifacts are valid.
- Distinguish code failure, data failure, configuration failure, resource
  failure, and numerical instability.
- Propose the smallest justified repair.

If files are missing, first exhaust safe read-only inspection. Ask the user only
when the missing choice would materially change the experiment.

## 22. Communication requirements

For implementation work, Codex should:

- Lead with the intended outcome.
- Give concise progress updates during tool use.
- State assumptions that affect the experiment.
- Avoid asking for decisions that can be resolved safely from repository
  evidence.
- Never claim training or validation was completed when it was not.

At the end of a change-preparation task, report:

- Outcome.
- Files created or modified.
- Important implementation decisions.
- Validation commands and results.
- Confirmation that full local training was not run.
- Split-change status.
- Remaining user actions.

At the end of a result-analysis task, report:

- Run validity.
- Best model and metric.
- Hypothesis result.
- Important issues.
- Analysis files created.
- Next experiment objective.

## 23. Prohibited shortcuts

Do not:

- Duplicate source implementations inside the notebook.
- Train full models locally without explicit authorization.
- Modify raw datasets.
- Generate a fresh split for every run.
- Fit preprocessing on validation or test data.
- Select hyperparameters using test performance.
- Compare normalized metrics from one model with original-unit metrics from
  another.
- Drop failed models from comparison tables.
- Overwrite a prior experiment directory.
- Reuse an experiment ID for materially different settings.
- Compare runs without commit, dataset, split, configuration, and seed
  identity.
- Invent missing metrics or infer success from a plot alone.
- Change multiple major experimental factors without explicit justification.
- Commit or push to GitHub without explicit user authorization.
- Store large checkpoints in the normal Git history.
- Execute code received inside result bundles.
- Claim that upstream AttnPINN physics directly applies to bearing degradation
  without documenting the adaptation.

## 24. Completion criteria

The workflow setup is complete only when:

- The source code is modular and importable.
- The notebook is a thin Colab controller.
- A configuration and fixed split exist and validate.
- Local validation succeeds without full training.
- Colab can record its exact Git commit and environment.
- The full Drive artifact contract is implemented.
- The lightweight bundle contract is implemented.
- Results can be imported and independently verified locally.
- Cross-experiment analysis can be updated reproducibly.
- A new Codex session can identify the current experiment phase from repository
  files alone.

A thesis experiment is complete only when its run is identifiable, its metrics
are independently verifiable, its failures are documented, and its comparison
is fair.

## 25. Project-specific settings to establish

The workflow implementation must expose these settings in configuration rather
than scattering them through the notebook:

```text
GITHUB_REPOSITORY_URL: <SET_THIS_WHEN_IMPLEMENTING>
GOOGLE_DRIVE_PROJECT_DIRECTORY:
  /content/drive/MyDrive/Bearing_RUL_Project
COLAB_NOTEBOOK:
  notebooks/train_models_colab.ipynb
EXPERIMENT_CONFIG:
  configs/experiment.yaml
FIXED_DATA_SPLIT:
  configs/data_split.json
INCOMING_RESULTS:
  results/incoming/
ANALYZED_RESULTS:
  results/analyzed/
CROSS_EXPERIMENT_COMPARISONS:
  results/comparisons/
```

If the GitHub repository URL is not discoverable from the Git remote when the
workflow is implemented, ask the user for it at that point. Do not invent it.

## 26. Current experiment state (EXP-003 / Run 3)

As of the completed and independently verified Run 3 analysis:

- Run 1 is preserved under `saved results/run_01/` as a diagnostic pilot with direct
  elapsed-time target leakage. Do not use its headline scores as thesis evidence.
- Run 2 is preserved and checksummed under `saved results/run_02/`. Its leakage-free
  three-seed analysis is `saved results/run_02/RUN_02_ANALYSIS.md`.
- The exact Run 3 configuration is retained as `configs/colab_experiments_run_03.json`;
  the Run 2 configuration remains `configs/colab_experiments_run_02.json`.
- EXP-003 was an IMS-only feature-based calibration experiment. It held the Run 2 IMS
  split, features, scaling, sequence length, architecture sizes, optimizer, and batch size
  constant.
- Predeclared test references were LSTM/data-only and Weak-PINN/high, using common seeds
  42, 1042, and 2042.
- Strong-PINN calibration crossed Paris weights `[0.003, 0.01, 0.03]` with Miner weights
  `[0.0003, 0.001, 0.003]`, also over all three common seeds.
- Candidate selection used mean validation RMSE only. Exactly the selected candidate's
  three seeds produced test predictions; no unselected calibration candidate used test.
- Run 3 is preserved under `saved results/run_03/`; its complete output is in
  `saved results/run_03/experiment_outputs/` and its analysis is
  `saved results/run_03/RUN_03_ANALYSIS.md`.
- All 9 test metric rows and all 27 validation metric rows were independently reproduced.
  Configuration, split, feature-cache, artifact inventory, and Python/config source hashes
  verify. The recorded notebook hash differs from the current local notebook, and the run
  has no Git commit SHA; retain both as explicit traceability limitations.
- LSTM was best on test (RMSE `0.144667 +/- 0.011691`, R2
  `0.744132 +/- 0.040221`). Weak-PINN/high was second (RMSE
  `0.207321 +/- 0.023566`).
- The validation-selected Strong-PINN (`paris=0.003`, `miner=0.003`) failed the
  declared criterion: test RMSE was `0.528835 +/- 0.002243`, R2 was
  `-2.404354 +/- 0.028854`, and the validation-to-test RMSE gap was about `9.7x`.
  Its predictions collapsed toward high RUL, particularly late in life.
- Treat EXP-003 as valid negative/diagnostic evidence. It does not support claiming a
  Strong-PINN improvement. Large validation/test feature-distribution shifts and the
  single-bearing validation split make cross-bearing generalization the next uncertainty
  to resolve.
- Run 4 should be a fixed-configuration IMS leave-one-bearing-out robustness experiment,
  rotating the four trajectories as held-out test bearings and comparing LSTM,
  Weak-PINN/high, and the frozen Run 3 Strong-PINN over common seeds without test-driven
  tuning. Keep representation, architecture, sequence length, optimizer, and weights fixed.
- Postpone the features-only versus raw-only versus hybrid representation ablation to Run 5,
  after cross-bearing stability has been established.

## 27. Disposable Upload-folder policy

For every Colab run after Run 3, treat the Google Drive `MyDrive/Upload` folder as a
disposable, run-specific staging package:

1. The user first downloads and locally preserves the completed run's required output
   directory.
2. The user deletes the previous `MyDrive/Upload` folder from Google Drive.
3. Codex prepares a new, complete local `Upload/` folder for the next run.
4. The user uploads that new folder so its exact Drive path is again `MyDrive/Upload`.
5. The user opens and runs the notebook contained in that newly uploaded folder.

Never instruct the user to merge a new package into an old Drive `Upload`, reuse source or
configuration files left there by an earlier run, or assume that earlier feature caches
remain in Drive. Each newly prepared local `Upload/` must be self-contained and include the
notebook, active configuration, source package, dependencies, required compact feature
caches, instructions, and an empty output directory unique to that run.

Do not delete or replace a prior Drive `Upload` until its completed experiment output has
been downloaded and verified to exist locally. Full preserved results belong under
`saved results/run_XX/`, not in the reusable staging package.



## 28. EXP-004 Run 4 completed state

EXP-004 / Run 4 is complete, archived, and independently verified. Its purpose was fixed
IMS cross-bearing robustness, motivated by the severe Run 3 validation/test gap and
Strong-PINN high-RUL collapse.

- The complete output is preserved under `saved results/run_04/experiment_outputs/` and
  the analysis is `saved results/run_04/RUN_04_ANALYSIS.md`.
- The run used exact clean Git commit
  `a6e7ada1b41f1374d007304f6ec76f709faf617b`, a Tesla T4, four predeclared folds, and
  seeds 42, 1042, and 2042. All 36 jobs completed in 53.9 minutes with no failures.
- All best-checkpoint and final-epoch metrics were independently reproduced from saved
  predictions. Configuration, split, cache, committed source, executed notebook, and
  listed artifact hashes verify. The Run 3 split is reproduced exactly as Run 4 fold 3.
- Weak-PINN/high ranked first by the predeclared equal-bearing macro normalized RMSE
  (`0.314238`), Strong-PINN ranked second (`0.328065`), and LSTM ranked third
  (`0.346032`). All three macro R2 values were negative, and between-bearing variation
  was large, so no model is stable enough for a general cross-bearing claim.
- Per-bearing winners differed: Strong-PINN won IMS-DS1/B3 and IMS-DS3/B3, Weak-PINN won
  IMS-DS1/B4, and LSTM won IMS-DS2/B1. This supports the split-dependence hypothesis and
  rejects the Run 3 ranking as generally portable.
- Strong-PINN failed its portability criterion because absolute late-life bias was below
  `0.25` in only two of four folds. Weak-PINN and LSTM met it in three folds, but every
  model failed badly on IMS-DS3/B3.
- Artifact defect: best and final prediction CSVs store the experiment label `run_04` in
  `run_id` rather than the physical bearing ID. Bearing identity is reconstructable from
  `fold_id`, `data_split.json`, and artifact paths, so the run remains numerically valid.
  Run 5 must use separate `experiment_run_id` and `bearing_run_id` fields.
- The next controlled experiment should address domain shift with causal per-bearing
  baseline-relative feature normalization using only an initial healthy prefix, followed
  by the same training-only scaler. Hold folds, models, weights, architecture, optimizer,
  sequence length, seeds, and evaluation constant.
- Run 5 success requires Weak-PINN to improve macro RMSE `0.314238` and worst-bearing RMSE
  `0.497089`, improve at least three of four folds, and reduce between-bearing variation
  without worsening late-life bias. Postpone raw-only versus hybrid encoders until this
  normalization question is resolved.

## 29. EXP-005 Run 5 prepared state

EXP-005 / Run 5 is the controlled preprocessing experiment justified by the independently
verified Run 4 domain-shift evidence. Its authoritative configuration is
`configs/colab_experiments_run_05.json`; the same content is active in
`configs/colab_experiments.json`.

- The one substantive experimental change is per-bearing robust baseline-relative
  normalization of the eleven vibration signal features. For each physical run, the first
  eight chronological, unlabeled snapshots fit a feature-wise median and scale. The scale
  is `max(abs(median), 1.4826 * MAD, 1e-8)`, and transformed values are
  `(value - median) / scale`.
- Eight baseline samples equal the frozen sequence length. The first predicted sequence
  target therefore occurs only after all eight calibration observations are available.
  Baseline fitting must not use RUL, health indicator, total trajectory length, failure
  time, validation metrics, or test metrics.
- After the per-bearing transform, the existing `StandardScaler` is still fitted only on
  the two training runs in each fold and then applied unchanged to validation and test.
  Every fold must save its per-bearing baseline statistics and training-scaler parameters.
- The Run 4 artifact defect is corrected. Prediction files must keep the physical
  trajectory in both `run_id` and `bearing_run_id`; `experiment_run_id` separately records
  `run_05`. Do not overwrite physical identity with the experiment label.
- Held constant from Run 4: IMS feature-cache hash and RUL labels, all four folds, test
  population, feature set and feature-only representation, sequence length 8, model
  architecture, LSTM/data-only, Weak-PINN/high, frozen Strong-PINN
  `strong_paris_0p003_miner_0p003`, seeds 42/1042/2042, optimizer, learning rate, batch
  size, early stopping, physics equations and weights, checkpoint selection, and metric
  aggregation. There are 36 expected jobs.
- The predeclared Weak-PINN success criterion is: macro normalized RMSE below `0.314238`,
  worst-bearing RMSE below `0.497089`, improvement in at least three of four folds,
  between-bearing RMSE standard deviation below `0.186672`, and no worsening of absolute
  late-life bias.
- Run 5 code lives in `src/thesis_work/run5_baseline_normalization.py`; the preprocessing
  implementation remains in the shared `src/thesis_work/multi_dataset.py`. The notebook is
  a thin controller generated by `scripts/build_run5_colab_notebook.py`.
- Local verification includes syntax checks, all repository tests, a four-fold
  preprocessing/controller validation without optimization, a one-epoch one-seed
  synthetic runner smoke test whose metrics are discarded, Upload-package validation,
  source/import checks, and notebook structural validation. Full neural training remains
  reserved for Colab.
- A label-free preflight check on the fixed cache reduced mean train/test Wasserstein
  feature shift from `1.4482` to `0.9342` (35.5%) across the four folds, with reduction in
  every fold. This verifies that the transform acts on the observed covariate-shift
  problem; it is not model-performance evidence and does not replace the Colab run.
- The disposable local `Upload/` package writes only to an empty
  `experiment_outputs_run_05/` directory. The notebook intentionally retains
  `PASTE_40_CHARACTER_COMMIT_SHA` until the prepared changes are committed and pushed.
  Do not run the real experiment from an unidentified or dirty revision.
