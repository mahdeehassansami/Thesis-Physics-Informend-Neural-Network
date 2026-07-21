# EXP-007B Google Drive and Colab instructions

## What this package runs

EXP-007B is the preregistered causal selective physics-risk confirmation. It reuses the exact
80 EXP-007A development trajectories and replaces the opened test with 16 newly generated
seed-920072 trajectories. Candidate LSTMs, differentiable simulator-progression losses, and
five neural seeds remain unchanged.

The run must first prove that development data contain both safe and harmful interventions and
that every seed has a feasible validation risk/coverage threshold. If either qualification
fails, the notebook stops before the fresh test. The selector then works at each observed
prefix, chooses at most one candidate, abstains to data-only when unsafe, and limits physics to
a 50% blend.

## Upload and run

1. Delete the previous `MyDrive/Upload` only after its prior results are preserved locally.
2. Drag this entire new `Upload` folder into `MyDrive`.
3. Open `MyDrive/Upload/train_models_colab.ipynb` in Google Colab.
4. Select **Runtime > Change runtime type > T4 GPU**.
5. Run all cells in order and approve Google Drive access.

Do not edit a Git SHA. `expected_commit.txt` pins the exact pushed version automatically. The
notebook stops on a dirty or mismatched checkout, wrong GPU, cache mismatch, split mismatch, or
incompatible recovery directory.

The dependency cell deliberately runs pip from the cloned repository because the requirements
file contains `-e .`. Do not remove its `cwd=CLONE` setting. Pip output is intentionally visible
so a package-index or compatibility failure retains the actionable error message.

## Runtime and recovery

Budget approximately 60-120 minutes on a Colab T4. The experiment schedules four cross-fit
data-only parents and 48 physics fine-tunes per seed, plus one final parent and 12 final physics
fine-tunes, across five seeds. Early stopping may reduce runtime.

Training occurs under `/content/exp007b_work`. Completed checkpoints and artifacts synchronize
to `MyDrive/Upload/experiment_outputs_exp007b`, so rerunning the same pinned notebook can resume.

## Return for analysis

Download `codex_results_bundle.zip` from
`MyDrive/Upload/experiment_outputs_exp007b/` and place it under `results/incoming/`. Keep the
full output directory in Drive. Do not tune or rescore the newly opened population. Codex must
independently validate the manifest, serialized metrics, development thresholds, intervention
coverage, candidate regret, and every preregistered primary gate.
