# EXP-005 Run 5 Google Drive upload

Use a fresh Drive package. Download and preserve the previous run first, delete the old `MyDrive/Upload`, then upload this complete local `Upload` folder directly to the root of Google Drive.

Run 5 is EXP-005, an IMS-only controlled preprocessing experiment. It evaluates LSTM/data-only, Weak-PINN/high, and the frozen Run 3 Strong-PINN profile `strong_paris_0p003_miner_0p003` with the exact four Run 4 folds and seeds.

The only substantive experimental change is signal-feature preprocessing. For each physical bearing, the first eight unlabeled snapshots define a robust baseline. Each vibration feature becomes `(value - prefix median) / max(abs(prefix median), 1.4826 * prefix MAD, 1e-8)`, followed by the unchanged `StandardScaler` fitted only on the training runs. Eight samples equal the sequence length, so the baseline is available before the first predicted target. No RUL target, health indicator, total life, or failure time fits this transform.

The prediction metadata correction keeps the physical identifier in both `run_id` and `bearing_run_id`; the experiment execution label is stored separately as `experiment_run_id=run_05`.

Each IMS trajectory is held out once as test and once as validation; the other two trajectories train. Every model is trained over seeds 42, 1042, and 2042: 36 jobs total. Validation is used only for early stopping and scheduling. Test-fold metrics do not alter models, weights, or preprocessing.

Before running Colab:

1. Commit and push the prepared repository.
2. Copy the resulting 40-character commit SHA.
3. Upload this fresh `Upload` folder to `MyDrive/Upload`.
4. Open `MyDrive/Upload/Thesis_v3_with_extra_graphs_tables.ipynb`.
5. Select a T4 GPU runtime.
6. Verify that the first code cell contains the pushed EXP-005 commit SHA. If it still contains `PASTE_40_CHARACTER_COMMIT_SHA`, replace that placeholder before running.
7. Run all cells from the beginning.

The notebook clones and checks out that exact commit under `/content`. The Upload copy supplies the compact cache at `feature_cache/ims_features.csv` and stores output at `experiment_outputs_run_05`. The controller stops if the experiment identity, commit, working tree, T4 GPU, or cache hash is wrong.

The complete output directory contains baseline statistics, checkpoints, histories, predictions, final-epoch evidence, fold summaries, manifests, and logs. The lightweight `codex_results_bundle.zip` excludes checkpoints. Download the complete `experiment_outputs_run_05` directory for recovery and the ZIP for local analysis.

A successful run must report `status: completed`, `completed_jobs: 36`, and `failed_jobs: 0`. Run 4 took about 54 minutes on a T4; budget approximately 60–90 minutes for Run 5, allowing for Colab setup and variability. Do not merge this package with an older Drive Upload.
