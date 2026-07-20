from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from thesis_work.multi_dataset import MODEL_FEATURES


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "saved results" / "run_04"
OUTPUT = RUN / "experiment_outputs"
RUN3 = ROOT / "saved results" / "run_03"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(target, prediction))
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(target, prediction)),
    }


manifest = json.loads((OUTPUT / "run_manifest.json").read_text(encoding="utf-8"))
config = json.loads((OUTPUT / "resolved_config.json").read_text(encoding="utf-8"))
split = json.loads((OUTPUT / "data_split.json").read_text(encoding="utf-8"))
failures = json.loads((OUTPUT / "failure_report.json").read_text(encoding="utf-8"))
assert manifest["experiment_id"] == "EXP-004"
assert manifest["run_id"] == "run_04"
assert manifest["status"] == "completed"
assert manifest["completed_jobs"] == manifest["expected_jobs"] == 36
assert manifest["failed_jobs"] == 0
assert failures == {"failed_jobs": [], "failure_files": []}
assert sha256(OUTPUT / "resolved_config.json") == manifest["resolved_config_sha256"]
assert sha256(OUTPUT / "data_split.json") == manifest["data_split_sha256"]
assert sha256(ROOT / "Upload" / "feature_cache" / "ims_features.csv") == manifest[
    "dataset_feature_cache_sha256"
]
assert sha256(OUTPUT / "executed_notebook.ipynb") == manifest[
    "executed_notebook_sha256"
]

git_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
).stdout.strip()
assert git_commit == manifest["git"]["commit"]
assert manifest["git"]["dirty"] is False

inventory = pd.read_csv(OUTPUT / "artifact_inventory.csv")
inventory_issues = []
for row in inventory.itertuples(index=False):
    path = OUTPUT / row.relative_path
    if not path.is_file() or path.stat().st_size != row.bytes or sha256(path) != row.sha256:
        inventory_issues.append(row.relative_path)
assert not inventory_issues, inventory_issues

source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
source_issues = []
for row in source_manifest.itertuples(index=False):
    blob = subprocess.run(
        ["git", "show", f"{git_commit}:{row.relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if len(blob) != row.bytes or hashlib.sha256(blob).hexdigest() != row.sha256:
        source_issues.append(row.relative_path)
combined_source_hash = hashlib.sha256(
    "\n".join(
        f"{row.relative_path}:{row.sha256}"
        for row in source_manifest.sort_values("relative_path").itertuples(index=False)
    ).encode()
).hexdigest()
assert combined_source_hash == manifest["source_tree_sha256"]
assert not source_issues, source_issues

results = pd.read_csv(OUTPUT / "all_model_comparisons.csv")
assert len(results) == 36
assert set(results["status"]) == {"ok"}
fold_to_test = {fold["fold_id"]: fold["test_runs"][0] for fold in split["folds"]}

verified_best_rows = []
verified_final_rows = []
lifecycle_rows = []
history_rows = []
prediction_identity_issues = []
target_signatures: dict[str, set[str]] = {fold: set() for fold in fold_to_test}

for job in results.itertuples(index=False):
    artifact = OUTPUT / job.artifact_directory
    best = pd.read_csv(artifact / "predictions.csv")
    final = pd.read_csv(artifact / "final_predictions.csv")
    expected_test = fold_to_test[job.fold_id]
    if set(best["run_id"].astype(str)) != {expected_test}:
        prediction_identity_issues.append(
            {
                "artifact_directory": job.artifact_directory,
                "expected_test_run_id": expected_test,
                "stored_run_ids": "|".join(sorted(best["run_id"].astype(str).unique())),
                "issue": "Physical bearing run_id was overwritten by experiment run label.",
            }
        )
    for frame, role in ((best, "best_validation"), (final, "final_epoch")):
        assert set(frame["fold_id"]) == {job.fold_id}
        assert set(frame["model"]) == {job.model}
        assert set(frame["weight_profile"]) == {job.weight_profile}
        assert set(frame["seed"].astype(int)) == {int(job.seed)}
        assert set(frame["checkpoint_role"]) == {role}
        assert np.allclose(
            frame["absolute_error"],
            np.abs(frame["target_rul"] - frame["predicted_rul"]),
        )

    signature_frame = best[["sample_index", "target_rul", "rul_scale_seconds"]]
    target_signatures[job.fold_id].add(
        hashlib.sha256(signature_frame.to_csv(index=False).encode()).hexdigest()
    )
    best_metrics = metrics(best["target_rul"].to_numpy(), best["predicted_rul"].to_numpy())
    best_seconds = metrics(
        best["target_rul_seconds"].to_numpy(),
        best["predicted_rul_seconds"].to_numpy(),
    )
    final_metrics = metrics(final["target_rul"].to_numpy(), final["predicted_rul"].to_numpy())
    final_seconds = metrics(
        final["target_rul_seconds"].to_numpy(),
        final["predicted_rul_seconds"].to_numpy(),
    )
    for name, value in best_metrics.items():
        assert np.isclose(value, getattr(job, name), rtol=1e-5, atol=1e-7)
    for name, value in final_metrics.items():
        assert np.isclose(value, getattr(job, f"final_test_{name}"), rtol=1e-5, atol=1e-7)

    common = {
        "fold_id": job.fold_id,
        "test_run_id": expected_test,
        "model": job.model,
        "weight_profile": job.weight_profile,
        "seed_repeat": int(job.seed_repeat),
        "seed": int(job.seed),
        "samples": len(best),
    }
    verified_best_rows.append(
        {
            **common,
            **best_metrics,
            "mae_seconds": best_seconds["mae"],
            "rmse_seconds": best_seconds["rmse"],
            "r2_seconds": best_seconds["r2"],
        }
    )
    verified_final_rows.append(
        {
            **common,
            **final_metrics,
            "mae_seconds": final_seconds["mae"],
            "rmse_seconds": final_seconds["rmse"],
            "r2_seconds": final_seconds["r2"],
        }
    )

    target = best["target_rul"].to_numpy()
    prediction = best["predicted_rul"].to_numpy()
    phases = np.where(target > 2 / 3, "early", np.where(target > 1 / 3, "middle", "late"))
    for phase in ("early", "middle", "late"):
        mask = phases == phase
        lifecycle_rows.append(
            {
                **common,
                "phase": phase,
                "phase_samples": int(mask.sum()),
                "mae": float(np.mean(np.abs(prediction[mask] - target[mask]))),
                "bias": float(np.mean(prediction[mask] - target[mask])),
                "prediction_mean": float(np.mean(prediction[mask])),
                "target_mean": float(np.mean(target[mask])),
            }
        )

    history = pd.read_csv(artifact / "history.csv")
    best_row = history.loc[history["validation_mse"].idxmin()]
    history_rows.append(
        {
            **common,
            "epochs_completed": len(history),
            "recorded_best_epoch": int(job.best_epoch),
            "minimum_validation_epoch": int(best_row["epoch"]),
            "best_validation_mse": float(job.best_validation_mse),
            "final_validation_mse": float(job.final_validation_mse),
            "epochs_after_best": int(job.final_epoch - job.best_epoch),
            "minimum_learning_rate": float(history["learning_rate"].min()),
            "final_to_best_test_rmse_change": final_metrics["rmse"] - best_metrics["rmse"],
        }
    )

assert all(len(signatures) == 1 for signatures in target_signatures.values())
verified_best = pd.DataFrame(verified_best_rows)
verified_final = pd.DataFrame(verified_final_rows)
lifecycle = pd.DataFrame(lifecycle_rows)
history_summary = pd.DataFrame(history_rows)
identity_issues = pd.DataFrame(prediction_identity_issues)
verified_best.to_csv(RUN / "run_04_verified_best_metrics.csv", index=False)
verified_final.to_csv(RUN / "run_04_verified_final_metrics.csv", index=False)
lifecycle.to_csv(RUN / "run_04_verified_lifecycle_metrics.csv", index=False)
history_summary.to_csv(RUN / "run_04_training_diagnostics.csv", index=False)
identity_issues.to_csv(RUN / "run_04_prediction_identity_issues.csv", index=False)
pd.DataFrame({"source_manifest_mismatch": source_issues}).to_csv(
    RUN / "run_04_source_mismatches.csv", index=False
)

job_metrics = verified_best.copy()
fold_summary = (
    job_metrics.groupby(["fold_id", "test_run_id", "model", "weight_profile"], as_index=False)
    .agg(
        seed_repeats=("seed", "nunique"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        r2_mean=("r2", "mean"),
        rmse_seconds_mean=("rmse_seconds", "mean"),
    )
    .sort_values(["fold_id", "rmse_mean"])
    .reset_index(drop=True)
)
fold_summary["fold_rank_by_rmse"] = (
    fold_summary.groupby("fold_id")["rmse_mean"].rank(method="min").astype(int)
)
late = lifecycle[lifecycle["phase"] == "late"]
late_summary = late.groupby(["fold_id", "model", "weight_profile"], as_index=False).agg(
    late_life_mae_mean=("mae", "mean"),
    late_life_bias_mean=("bias", "mean"),
)
fold_summary = fold_summary.merge(late_summary, on=["fold_id", "model", "weight_profile"])
aggregate = (
    fold_summary.groupby(["model", "weight_profile"], as_index=False)
    .agg(
        folds_completed=("fold_id", "nunique"),
        fold_wins=("fold_rank_by_rmse", lambda values: int((values == 1).sum())),
        macro_mae_mean=("mae_mean", "mean"),
        macro_rmse_mean=("rmse_mean", "mean"),
        between_bearing_rmse_std=("rmse_mean", "std"),
        worst_bearing_rmse=("rmse_mean", "max"),
        macro_r2_mean=("r2_mean", "mean"),
        macro_rmse_seconds_mean=("rmse_seconds_mean", "mean"),
        macro_late_life_mae=("late_life_mae_mean", "mean"),
        acceptable_late_bias_folds=(
            "late_life_bias_mean",
            lambda values: int((np.abs(values) < 0.25).sum()),
        ),
    )
    .sort_values("macro_rmse_mean")
    .reset_index(drop=True)
)
aggregate["rank_by_macro_rmse"] = np.arange(1, len(aggregate) + 1)
fold_summary.to_csv(RUN / "run_04_verified_fold_summary.csv", index=False)
aggregate.to_csv(RUN / "run_04_verified_aggregate_summary.csv", index=False)

generated_fold = pd.read_csv(OUTPUT / "fold_model_summary.csv")
generated_aggregate = pd.read_csv(OUTPUT / "all_model_comparisons_summary.csv")
for verified, generated, keys, columns in (
    (
        fold_summary,
        generated_fold,
        ["fold_id", "model", "weight_profile"],
        ["mae_mean", "rmse_mean", "rmse_std", "r2_mean", "rmse_seconds_mean"],
    ),
    (
        aggregate,
        generated_aggregate,
        ["model", "weight_profile"],
        [
            "macro_mae_mean",
            "macro_rmse_mean",
            "between_bearing_rmse_std",
            "worst_bearing_rmse",
            "macro_r2_mean",
            "macro_rmse_seconds_mean",
        ],
    ),
):
    merged = verified.merge(generated, on=keys, suffixes=("_verified", "_generated"))
    for column in columns:
        assert np.allclose(
            merged[f"{column}_verified"], merged[f"{column}_generated"], rtol=1e-5, atol=1e-7
        )

cache = pd.read_csv(ROOT / "Upload" / "feature_cache" / "ims_features.csv")
shift_rows = []
for fold in split["folds"]:
    train = cache[cache["run_id"].isin(fold["train_runs"])]
    test = cache[cache["run_id"].isin(fold["test_runs"])]
    scaler = StandardScaler().fit(train[MODEL_FEATURES])
    train_scaled = scaler.transform(train[MODEL_FEATURES])
    test_scaled = scaler.transform(test[MODEL_FEATURES])
    for index, feature in enumerate(MODEL_FEATURES):
        shift_rows.append(
            {
                "fold_id": fold["fold_id"],
                "test_run_id": fold["test_runs"][0],
                "feature": feature,
                "wasserstein_train_vs_test": wasserstein_distance(
                    train_scaled[:, index], test_scaled[:, index]
                ),
                "absolute_scaled_mean_gap": abs(
                    float(train_scaled[:, index].mean() - test_scaled[:, index].mean())
                ),
            }
        )
feature_shift = pd.DataFrame(shift_rows).sort_values(
    "wasserstein_train_vs_test", ascending=False
)
feature_shift.to_csv(RUN / "run_04_feature_shift.csv", index=False)

run3 = pd.read_csv(RUN3 / "run_03_model_ranking.csv")
fold3 = fold_summary[fold_summary["fold_id"] == "fold_03_test_ims_ds2_b1"]
comparison_rows = []
for row in fold3.itertuples(index=False):
    prior = run3[(run3["model"] == row.model) & (run3["weight_profile"] == row.weight_profile)]
    if not prior.empty:
        comparison_rows.append(
            {
                "model": row.model,
                "weight_profile": row.weight_profile,
                "run_03_rmse_mean": float(prior.iloc[0]["rmse_mean"]),
                "run_04_matching_fold_rmse_mean": row.rmse_mean,
                "difference": row.rmse_mean - float(prior.iloc[0]["rmse_mean"]),
            }
        )
run3_match = pd.DataFrame(comparison_rows)
run3_match.to_csv(RUN / "run_03_vs_run_04_matching_fold.csv", index=False)

weak = aggregate[aggregate["model"] == "weak_pinn"].iloc[0]
strong = aggregate[aggregate["model"] == "strong_pinn"].iloc[0]
lstm = aggregate[aggregate["model"] == "lstm"].iloc[0]
top_shift = feature_shift.groupby("feature")["wasserstein_train_vs_test"].mean().sort_values(ascending=False).head(5)
final_delta = history_summary.groupby("model")["final_to_best_test_rmse_change"].agg(["mean", "min", "max"])
runtime_minutes = (
    pd.Timestamp(manifest["finished_utc"]) - pd.Timestamp(manifest["started_utc"])
).total_seconds() / 60

report = [
    "# Run 4 analysis",
    "",
    "## Validity decision",
    "",
    f"EXP-004 is a valid completed diagnostic cross-bearing experiment. All 36 jobs completed on a Tesla T4 in {runtime_minutes:.1f} minutes, the exact clean Git commit `{manifest['git']['commit']}` is recorded, all generated best/final metrics were independently reproduced from predictions, all listed artifact hashes verify, and each fold used one identical target population across models and seeds.",
    "",
    f"One artifact-contract defect was found in all {len(identity_issues)} best-prediction files and their corresponding final-epoch files: `run_id` contains `run_04` rather than the physical test-bearing ID. The correct bearing remains reconstructable from `fold_id`, `data_split.json`, and the job directory, so numerical validity is retained, but Run 5 must preserve separate `experiment_run_id` and `bearing_run_id` fields.",
    "",
    "## Primary outcome",
    "",
    f"- Weak-PINN/high ranked first by the predeclared equal-bearing macro RMSE: {weak.macro_rmse_mean:.6f}. It won {int(weak.fold_wins)} of four folds and had worst-bearing RMSE {weak.worst_bearing_rmse:.6f}.",
    f"- The frozen Strong-PINN ranked second: macro RMSE {strong.macro_rmse_mean:.6f}. It won {int(strong.fold_wins)} folds but had higher between-bearing variation ({strong.between_bearing_rmse_std:.6f}) and worst-bearing RMSE {strong.worst_bearing_rmse:.6f}.",
    f"- LSTM ranked third: macro RMSE {lstm.macro_rmse_mean:.6f}, despite reproducing its strong Run 3 result on the matching IMS-DS2 fold.",
    f"- No model was cross-bearing stable: macro R2 was negative for Weak-PINN ({weak.macro_r2_mean:.3f}), Strong-PINN ({strong.macro_r2_mean:.3f}), and LSTM ({lstm.macro_r2_mean:.3f}).",
    "",
    "Per-bearing winners were Strong-PINN on IMS-DS1/B3 (RMSE 0.168009), Weak-PINN on IMS-DS1/B4 (0.107763), LSTM on IMS-DS2/B1 (0.144667), and Strong-PINN on IMS-DS3/B3 (0.484993). The ranking is therefore strongly bearing-dependent.",
    "",
    "## Hypothesis and success criteria",
    "",
    "The Run 4 hypothesis is supported: the Run 3 ranking was materially split-dependent. The experiment succeeded methodologically because all 36 frozen jobs completed and verified, but the Run 3 ranking did not generalize: LSTM was not the macro-RMSE leader and won only one fold.",
    "",
    f"Strong-PINN also failed its portability criterion. Absolute late-life bias was below 0.25 in only {int(strong.acceptable_late_bias_folds)} of four folds, not the required three. Weak-PINN and LSTM each met that threshold in {int(weak.acceptable_late_bias_folds)} and {int(lstm.acceptable_late_bias_folds)} folds respectively, but all models failed badly on IMS-DS3/B3.",
    "",
    "## Diagnostics",
    "",
    f"Between-bearing RMSE standard deviations were {weak.between_bearing_rmse_std:.3f} (Weak-PINN), {strong.between_bearing_rmse_std:.3f} (Strong-PINN), and {lstm.between_bearing_rmse_std:.3f} (LSTM), which is too large for a stable cross-bearing claim.",
    "",
    "The largest average train/test feature shifts were: " + ", ".join(f"{name} ({value:.2f})" for name, value in top_shift.items()) + ". This reinforces the domain-shift interpretation.",
    "",
    "Final-epoch versus best-validation test RMSE changes (positive means the final epoch was worse) were: " + "; ".join(f"{model} mean {row['mean']:+.4f}, range [{row['min']:+.4f}, {row['max']:+.4f}]" for model, row in final_delta.iterrows()) + ". Best-checkpoint reporting is therefore necessary.",
    "",
    "Original-time RMSE is reported per bearing but should not determine the cross-fold ranking because the four trajectories have very different durations. The normalized equal-bearing macro metric remains primary.",
    "",
    "## Recommended Run 5",
    "",
    "Do not move directly to a large raw-versus-hybrid architecture study. First test one controlled preprocessing change aimed at the observed domain shift: causal per-bearing baseline-relative feature normalization using only an initial healthy prefix available at prediction time, followed by the same training-only scaler. Keep the four folds, models, frozen physics weights, architecture, seeds, optimizer, sequence length, and evaluation unchanged.",
    "",
    "Run 5 should succeed only if Weak-PINN improves on Run 4 macro RMSE 0.314238 and worst-bearing RMSE 0.497089, improves at least three of four folds, and reduces between-bearing RMSE variation without increasing late-life bias. Raw-only and hybrid encoders should remain a later ablation after this normalization question is resolved.",
]
(RUN / "RUN_04_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")

artifact_rows = []
for path in sorted(RUN.rglob("*")):
    if path.is_file() and path.name != "artifact_manifest.csv":
        artifact_rows.append(
            {
                "relative_path": path.relative_to(RUN).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
pd.DataFrame(artifact_rows).to_csv(RUN / "artifact_manifest.csv", index=False)

print(f"Verified and analyzed Run 4 at {RUN}")
print(aggregate.to_string(index=False))
print(f"Prediction identity issues: {len(identity_issues)}")
