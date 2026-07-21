# EXP-007 Google Drive and Colab instructions

Use a completely fresh Drive package. First preserve any previous experiment output, delete
the old `MyDrive/Upload`, and upload this complete local `Upload` directory to the root of
Google Drive.

EXP-007 is the controlled synthetic physics-prior credibility decision gate. It is not a new
fixed-weight PINN sweep and it does not use the supplied-v2 dataset whose progression truth is
withheld. It uses the 40 official-simulator trajectories qualified in EXP-006 and their frozen
24 training, 8 validation, and 8 test trajectory split.

The experiment performs five common-seed repetitions. Within each seed it:

1. cross-fits a small causal LSTM data-only RUL backbone across complete training trajectories;
2. cross-fits a vibration-to-degradation proxy and training-only empirical progression-family
   templates;
3. independently enumerates 20 candidate priors per causal checkpoint—three valid and seventeen
   deliberately corrupt—without using trajectory truth to select the candidate pool;
4. fits a logistic credibility estimator on cross-fitted training evidence;
5. calibrates it, chooses its threshold, and selects the scalar-weight comparator using only
   validation trajectories; and
6. evaluates the frozen diagnostic and controls once on the untouched test trajectories.

The test corruption magnitudes differ from training, and every wrong family is evaluated.
Validation and test also retain the load, speed, and noise shifts declared before the EXP-006 simulation.
Operation/noise shift does not turn a correct progression family into corrupt physics.

Before running:

1. Upload this fresh folder as `MyDrive/Upload`.
2. Open `MyDrive/Upload/train_models_colab.ipynb` in Google Colab.
3. Select `Runtime > Change runtime type > T4 GPU`.
4. Run all cells from the beginning.

No SHA editing is required. `expected_commit.txt` already contains the exact pushed commit, and
the notebook refuses a different or dirty checkout. It also verifies the controlled cache hash
before training.

Training runs primarily under `/content/exp007_work`. Completed fold/seed recovery artifacts
are synchronized to:

```text
/content/drive/MyDrive/Upload/experiment_outputs_exp007/
```

Do not add files to that output directory before the run. A compatible interrupted run may be
resumed by reopening the same fresh notebook and running from the beginning.

The predeclared gate is:

- aggregate held-out trajectory-candidate AUROC at least `0.80`;
- trajectory/seed bootstrap 95% AUROC lower bound strictly above `0.50`; and
- neither all-on nor all-off behavior above `90%` without a declared physical explanation.

If the gate fails, preserve the result. Do not tune against the test trajectories or enlarge
the network to force a pass.

When finished, download the complete `experiment_outputs_exp007` directory and place it under
`results/incoming/` in this repository. The included `codex_results_bundle.zip` is the compact
analysis package; the complete directory retains recovery checkpoints.
