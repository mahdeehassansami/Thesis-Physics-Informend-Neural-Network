# EXP-004 Run 4 Google Drive upload

Use a fresh Drive package. Download and preserve the previous run first, delete the old `MyDrive/Upload`, then upload this complete local `Upload` folder directly to the root of Google Drive.

Run 4 is EXP-004, an IMS-only fixed-model cross-bearing robustness experiment. It evaluates LSTM/data-only, Weak-PINN/high, and the frozen Run 3 Strong-PINN profile `strong_paris_0p003_miner_0p003`.

The experiment has four predeclared folds. Each IMS trajectory is held out once as test and once as validation; the other two trajectories train. Every model is trained over seeds 42, 1042, and 2042: 36 jobs total. Validation is used only for early stopping and scheduling. Test-fold metrics do not alter models or weights.

Before running Colab:

1. Commit and push the prepared repository.
2. Copy the resulting 40-character commit SHA.
3. Upload this fresh `Upload` folder to `MyDrive/Upload`.
4. Open `MyDrive/Upload/Thesis_v3_with_extra_graphs_tables.ipynb`.
5. Select a T4 GPU runtime.
6. In the first code cell, replace `PASTE_40_CHARACTER_COMMIT_SHA` with the pushed commit SHA.
7. Run all cells from the beginning.

The notebook clones and checks out that exact commit under `/content`. The Upload copy supplies the compact cache at `feature_cache/ims_features.csv` and stores output at `experiment_outputs_run_04`. The controller stops if the commit, working tree, GPU, or cache hash is wrong.

The complete output directory contains checkpoints, histories, predictions, final-epoch evidence, fold summaries, manifests, and logs. The lightweight `codex_results_bundle.zip` excludes checkpoints. Download the complete `experiment_outputs_run_04` directory for recovery and the ZIP for local analysis.

A successful run must report `status: completed`, `completed_jobs: 36`, and no failed jobs. Do not merge this package with an older Drive Upload.