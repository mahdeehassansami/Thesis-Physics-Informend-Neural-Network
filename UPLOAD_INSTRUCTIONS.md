# EXP-007A Google Drive and Colab instructions

## What this package runs

EXP-007A is the corrective counterfactual physics-harm feasibility experiment. It uses a new
96-trajectory multi-condition official-simulator cache. Candidate LSTM models are initialized
from identical data-only checkpoints and fine-tuned with differentiable simulator-progression
value, rate, and monotonic losses.

The run must first prove that development data contain both safe and harmful interventions. If
that qualification fails, the notebook stops before sealed-test evaluation. If it passes, the
frozen credibility estimator is evaluated on the separate test simulator seed.

## Upload and run

1. Delete the previous `MyDrive/Upload` only after its prior results are preserved locally.
2. Drag this entire new `Upload` folder into `MyDrive`.
3. Open `MyDrive/Upload/train_models_colab.ipynb` in Google Colab.
4. Select **Runtime > Change runtime type > T4 GPU**.
5. Run all cells in order and approve Google Drive access.

Do not edit a Git SHA. `expected_commit.txt` pins the exact pushed version automatically. The
notebook stops on a dirty or mismatched checkout, wrong GPU, cache mismatch, split mismatch, or
incompatible recovery directory.

## Runtime and recovery

Budget approximately 60-120 minutes on a Colab T4. The experiment schedules four cross-fit
data-only parents and 48 physics fine-tunes per seed, plus one final parent and 12 final physics
fine-tunes, across five seeds. Early stopping may reduce runtime.

Training occurs under `/content/exp007a_work`. Completed checkpoints and artifacts synchronize
to `MyDrive/Upload/experiment_outputs_exp007a`, so rerunning the same pinned notebook can resume.

## Return for analysis

Download `codex_results_bundle.zip` from
`MyDrive/Upload/experiment_outputs_exp007a/` and place it under `results/incoming/`. Keep the
full output directory in Drive. Do not begin EXP-008 until Codex independently validates the
manifest, serialized metrics, development qualification, per-seed collapse checks, calibration,
candidate regret, and final gate.
