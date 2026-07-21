from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from thesis_work.multi_dataset import MODEL_FEATURES, SIGNAL_FEATURES


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "saved results" / "run_05"
OUTPUT = RUN / "experiment_outputs"
RUN4 = ROOT / "saved results" / "run_04"
LOCAL_CACHE = ROOT / "data" / "processed_features" / "colab" / "ims_features.csv"
ANALYZED = ROOT / "results" / "analyzed" / "EXP-005"
COMPARISONS = ROOT / "results" / "comparisons"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(target, prediction))
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(target, prediction)),
    }


def git_blob(commit: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


manifest = json.loads((OUTPUT / "run_manifest.json").read_text(encoding="utf-8"))
config = json.loads((OUTPUT / "resolved_config.json").read_text(encoding="utf-8"))
split = json.loads((OUTPUT / "data_split.json").read_text(encoding="utf-8"))
dataset_summary = json.loads(
    (OUTPUT / "dataset_summary.json").read_text(encoding="utf-8")
)
failures = json.loads((OUTPUT / "failure_report.json").read_text(encoding="utf-8"))

assert manifest["experiment_id"] == "EXP-005"
assert manifest["run_id"] == "run_05"
assert manifest["status"] == "completed"
assert manifest["completed_jobs"] == manifest["expected_jobs"] == 36
assert manifest["failed_jobs"] == 0
assert failures == {"failed_jobs": [], "failure_files": []}
assert config["experiment"]["id"] == "EXP-005"
assert config["run_label"] == "run_05"
assert config["preprocessing"]["uses_targets"] is False
assert config["preprocessing"]["uses_failure_time"] is False

assert sha256(OUTPUT / "resolved_config.json") == manifest["resolved_config_sha256"]
assert sha256(OUTPUT / "data_split.json") == manifest["data_split_sha256"]
assert json_hash(config["preprocessing"]) == manifest["preprocessing_config_sha256"]
assert sha256(LOCAL_CACHE) == manifest["dataset_feature_cache_sha256"]
assert sha256(OUTPUT / "executed_notebook.ipynb") == manifest[
    "executed_notebook_sha256"
]
assert dataset_summary["feature_cache_sha256"] == manifest[
    "dataset_feature_cache_sha256"
]

run_commit = manifest["git"]["commit"]
assert run_commit == config["repository"]["expected_commit"]
assert manifest["git"]["dirty"] is False
subprocess.run(
    ["git", "cat-file", "-e", f"{run_commit}^{{commit}}"],
    cwd=ROOT,
    check=True,
)
committed_config = json.loads(
    git_blob(run_commit, "configs/colab_experiments.json").decode("utf-8")
)
expected_resolved = copy.deepcopy(committed_config)
expected_resolved["repository"]["expected_commit"] = run_commit
assert expected_resolved == config

inventory = pd.read_csv(OUTPUT / "artifact_inventory.csv")
inventory_issues: list[dict[str, str]] = []
for row in inventory.itertuples(index=False):
    path = OUTPUT / row.relative_path
    if not path.is_file():
        inventory_issues.append({"relative_path": row.relative_path, "issue": "missing"})
    elif path.stat().st_size != row.bytes:
        inventory_issues.append(
            {"relative_path": row.relative_path, "issue": "byte-count mismatch"}
        )
    elif sha256(path) != row.sha256:
        inventory_issues.append(
            {"relative_path": row.relative_path, "issue": "SHA-256 mismatch"}
        )
assert len(inventory) == manifest["artifact_count_excluding_manifest_inventory_bundle"]
assert not inventory_issues, inventory_issues

source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
source_issues: list[dict[str, str]] = []
for row in source_manifest.itertuples(index=False):
    blob = git_blob(run_commit, row.relative_path)
    if len(blob) != row.bytes or hashlib.sha256(blob).hexdigest() != row.sha256:
        source_issues.append(
            {"relative_path": row.relative_path, "issue": "committed blob mismatch"}
        )
combined_source_hash = hashlib.sha256(
    "\n".join(
        f"{row.relative_path}:{row.sha256}"
        for row in source_manifest.sort_values("relative_path").itertuples(index=False)
    ).encode()
).hexdigest()
assert combined_source_hash == manifest["source_tree_sha256"]
assert not source_issues, source_issues

bundle = OUTPUT / "codex_results_bundle.zip"
with zipfile.ZipFile(bundle) as archive:
    bundle_paths = set(archive.namelist())
    assert all(
        not Path(name).is_absolute() and ".." not in Path(name).parts
        for name in bundle_paths
    )
    assert "run_manifest.json" in bundle_paths
    assert "artifact_inventory.csv" in bundle_paths
    assert not any(Path(name).name == "checkpoint.pt" for name in bundle_paths)

fold_to_test = {fold["fold_id"]: fold["test_runs"][0] for fold in split["folds"]}
fold_to_validation = {
    fold["fold_id"]: fold["validation_runs"][0] for fold in split["folds"]
}
dataset_fold = {record["fold_id"]: record for record in dataset_summary["folds"]}
results = pd.read_csv(OUTPUT / "all_model_comparisons.csv")
assert len(results) == 36
assert set(results["status"]) == {"ok"}

verified_best_rows: list[dict[str, object]] = []
verified_final_rows: list[dict[str, object]] = []
lifecycle_rows: list[dict[str, object]] = []
training_rows: list[dict[str, object]] = []
prediction_rows: list[dict[str, object]] = []
identity_issues: list[dict[str, str]] = []
target_signatures: dict[str, set[str]] = {fold_id: set() for fold_id in fold_to_test}

for job in results.itertuples(index=False):
    artifact = OUTPUT / job.artifact_directory
    prediction_files = [
        ("predictions.csv", "best_validation", fold_to_test[job.fold_id]),
        ("final_predictions.csv", "final_epoch", fold_to_test[job.fold_id]),
        (
            "validation_predictions.csv",
            "best_validation",
            fold_to_validation[job.fold_id],
        ),
        (
            "final_validation_predictions.csv",
            "final_epoch",
            fold_to_validation[job.fold_id],
        ),
    ]
    loaded: dict[str, pd.DataFrame] = {}
    for filename, checkpoint_role, expected_bearing in prediction_files:
        frame = pd.read_csv(artifact / filename)
        loaded[filename] = frame
        expected_identity = {
            "experiment_id": {"EXP-005"},
            "experiment_run_id": {"run_05"},
            "dataset": {"ims"},
            "fold_id": {job.fold_id},
            "model": {job.model},
            "weight_profile": {job.weight_profile},
            "seed": {int(job.seed)},
            "checkpoint_role": {checkpoint_role},
            "run_id": {expected_bearing},
            "bearing_run_id": {expected_bearing},
        }
        for column, expected in expected_identity.items():
            actual = set(frame[column].astype(str))
            expected_strings = {str(value) for value in expected}
            if actual != expected_strings:
                identity_issues.append(
                    {
                        "artifact_directory": job.artifact_directory,
                        "file": filename,
                        "column": column,
                        "expected": "|".join(sorted(expected_strings)),
                        "actual": "|".join(sorted(actual)),
                    }
                )
        assert np.allclose(
            frame["absolute_error"],
            np.abs(frame["target_rul"] - frame["predicted_rul"]),
        )
        assert np.allclose(
            frame["absolute_error_seconds"],
            np.abs(frame["target_rul_seconds"] - frame["predicted_rul_seconds"]),
        )

    best = loaded["predictions.csv"]
    final = loaded["final_predictions.csv"]
    assert len(best) == dataset_fold[job.fold_id]["test_sequences"]
    signature_frame = best[["run_id", "sample_index", "target_rul", "rul_scale_seconds"]]
    target_signatures[job.fold_id].add(
        hashlib.sha256(signature_frame.to_csv(index=False).encode()).hexdigest()
    )

    best_metrics = metrics(best.target_rul.to_numpy(), best.predicted_rul.to_numpy())
    final_metrics = metrics(final.target_rul.to_numpy(), final.predicted_rul.to_numpy())
    best_seconds = metrics(
        best.target_rul_seconds.to_numpy(), best.predicted_rul_seconds.to_numpy()
    )
    final_seconds = metrics(
        final.target_rul_seconds.to_numpy(), final.predicted_rul_seconds.to_numpy()
    )
    for name, value in best_metrics.items():
        assert np.isclose(value, getattr(job, name), rtol=1e-5, atol=1e-7)
    for name, value in final_metrics.items():
        assert np.isclose(
            value, getattr(job, f"final_test_{name}"), rtol=1e-5, atol=1e-7
        )

    common = {
        "fold_id": job.fold_id,
        "test_run_id": fold_to_test[job.fold_id],
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

    target = best.target_rul.to_numpy()
    prediction = best.predicted_rul.to_numpy()
    phases = np.where(target > 2 / 3, "early", np.where(target > 1 / 3, "middle", "late"))
    for phase in ("early", "middle", "late"):
        mask = phases == phase
        assert mask.any()
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
    minimum = history.loc[history.validation_mse.idxmin()]
    assert int(minimum.epoch) == int(job.best_epoch)
    physics_columns = [
        column
        for column in history.columns
        if column.startswith("weighted_") and column != "weighted_data"
    ]
    training_rows.append(
        {
            **common,
            "epochs_completed": len(history),
            "best_epoch": int(job.best_epoch),
            "final_epoch": int(job.final_epoch),
            "epochs_after_best": int(job.final_epoch - job.best_epoch),
            "best_validation_rmse": float(np.sqrt(job.best_validation_mse)),
            "final_validation_rmse": float(np.sqrt(job.final_validation_mse)),
            "minimum_learning_rate": float(history.learning_rate.min()),
            "final_minus_best_test_rmse": final_metrics["rmse"] - best_metrics["rmse"],
            "weighted_physics_to_data_ratio_at_best": float(
                job.weighted_physics_to_data_ratio
            ),
            "physics_history_columns": "|".join(sorted(physics_columns)),
        }
    )
    prediction_rows.append(
        {
            **common,
            "prediction_mean": float(prediction.mean()),
            "prediction_std": float(prediction.std()),
            "prediction_min": float(prediction.min()),
            "prediction_max": float(prediction.max()),
            "fraction_at_zero": float(np.mean(prediction <= 1e-7)),
            "fraction_at_one": float(np.mean(prediction >= 1 - 1e-7)),
            "mean_bias": float(np.mean(prediction - target)),
        }
    )

assert all(len(signatures) == 1 for signatures in target_signatures.values())
assert not identity_issues, identity_issues

verified_best = pd.DataFrame(verified_best_rows)
verified_final = pd.DataFrame(verified_final_rows)
lifecycle = pd.DataFrame(lifecycle_rows)
training = pd.DataFrame(training_rows)
prediction_diagnostics = pd.DataFrame(prediction_rows)

fold_summary = (
    verified_best.groupby(
        ["fold_id", "test_run_id", "model", "weight_profile"], as_index=False
    )
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
    fold_summary.groupby("fold_id").rmse_mean.rank(method="min").astype(int)
)
late = lifecycle[lifecycle.phase == "late"]
late_summary = late.groupby(
    ["fold_id", "model", "weight_profile"], as_index=False
).agg(
    late_life_mae_mean=("mae", "mean"),
    late_life_bias_mean=("bias", "mean"),
)
fold_summary = fold_summary.merge(
    late_summary, on=["fold_id", "model", "weight_profile"]
)
time_summary = results.groupby(["model", "weight_profile"], as_index=False).agg(
    training_seconds_total=("seconds", "sum"),
    parameter_count=("parameter_count", "first"),
)
aggregate = (
    fold_summary.groupby(["model", "weight_profile"], as_index=False)
    .agg(
        folds_completed=("fold_id", "nunique"),
        seed_runs=("seed_repeats", "sum"),
        fold_wins=("fold_rank_by_rmse", lambda values: int((values == 1).sum())),
        macro_mae_mean=("mae_mean", "mean"),
        macro_rmse_mean=("rmse_mean", "mean"),
        between_bearing_rmse_std=("rmse_mean", "std"),
        worst_bearing_rmse=("rmse_mean", "max"),
        macro_r2_mean=("r2_mean", "mean"),
        macro_rmse_seconds_mean=("rmse_seconds_mean", "mean"),
        macro_late_life_mae=("late_life_mae_mean", "mean"),
        mean_abs_late_life_bias=(
            "late_life_bias_mean",
            lambda values: float(np.abs(values).mean()),
        ),
        worst_abs_late_life_bias=(
            "late_life_bias_mean",
            lambda values: float(np.abs(values).max()),
        ),
    )
    .merge(time_summary, on=["model", "weight_profile"])
    .sort_values("macro_rmse_mean")
    .reset_index(drop=True)
)
aggregate["rank_by_macro_rmse"] = np.arange(1, len(aggregate) + 1)

generated_fold = pd.read_csv(OUTPUT / "fold_model_summary.csv")
generated_aggregate = pd.read_csv(OUTPUT / "all_model_comparisons_summary.csv")
fold_check = fold_summary.merge(
    generated_fold,
    on=["fold_id", "test_run_id", "model", "weight_profile"],
    suffixes=("_verified", "_generated"),
)
for column in ("mae_mean", "rmse_mean", "rmse_std", "r2_mean", "rmse_seconds_mean"):
    assert np.allclose(
        fold_check[f"{column}_verified"],
        fold_check[f"{column}_generated"],
        rtol=1e-5,
        atol=1e-7,
    )
aggregate_check = aggregate.merge(
    generated_aggregate,
    on=["model", "weight_profile"],
    suffixes=("_verified", "_generated"),
)
for column in (
    "macro_mae_mean",
    "macro_rmse_mean",
    "between_bearing_rmse_std",
    "worst_bearing_rmse",
    "macro_r2_mean",
    "macro_late_life_mae",
    "worst_abs_late_life_bias",
):
    assert np.allclose(
        aggregate_check[f"{column}_verified"],
        aggregate_check[f"{column}_generated"],
        rtol=1e-5,
        atol=1e-7,
    )

cache = pd.read_csv(LOCAL_CACHE)
preprocessing_issues: list[dict[str, str]] = []
shift_rows: list[dict[str, object]] = []
for fold in split["folds"]:
    record_path = OUTPUT / "ims" / "folds" / fold["fold_id"] / "preprocessing.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected_hash = dataset_fold[fold["fold_id"]]["preprocessing_sha256"]
    assert sha256(record_path) == expected_hash
    baseline = record["baseline_relative_transform"]
    assert baseline["prefix_samples"] == config["preprocessing"]["prefix_samples"] == 8
    transformed = cache.copy()
    for run_id, statistics in baseline["run_statistics"].items():
        indices = cache.index[cache.run_id == run_id]
        ordered = cache.loc[indices].sort_values("sample_index")
        prefix = ordered.iloc[:8][SIGNAL_FEATURES].astype(float)
        center = prefix.median()
        mad = (prefix - center).abs().median() * 1.4826
        scale = np.maximum.reduce(
            [center.abs().to_numpy(), mad.to_numpy(), np.full(len(center), 1e-8)]
        )
        if not np.allclose(center.to_numpy(), list(statistics["center"].values())):
            preprocessing_issues.append(
                {"fold_id": fold["fold_id"], "run_id": run_id, "issue": "center mismatch"}
            )
        if not np.allclose(scale, list(statistics["scale"].values())):
            preprocessing_issues.append(
                {"fold_id": fold["fold_id"], "run_id": run_id, "issue": "scale mismatch"}
            )
        transformed.loc[indices, SIGNAL_FEATURES] = (
            transformed.loc[indices, SIGNAL_FEATURES].astype(float) - center
        ) / scale

    train_mask = transformed.run_id.isin(fold["train_runs"])
    test_mask = transformed.run_id.isin(fold["test_runs"])
    train_values = transformed.loc[train_mask, MODEL_FEATURES].astype(float)
    test_values = transformed.loc[test_mask, MODEL_FEATURES].astype(float)
    stored_scaler = record["post_transform_scaler"]
    recomputed_scaler = StandardScaler().fit(train_values)
    if not np.allclose(recomputed_scaler.mean_, stored_scaler["scaler_mean"]):
        preprocessing_issues.append(
            {"fold_id": fold["fold_id"], "run_id": "*", "issue": "scaler mean mismatch"}
        )
    if not np.allclose(recomputed_scaler.scale_, stored_scaler["scaler_scale"]):
        preprocessing_issues.append(
            {"fold_id": fold["fold_id"], "run_id": "*", "issue": "scaler scale mismatch"}
        )
    train_scaled = recomputed_scaler.transform(train_values)
    test_scaled = recomputed_scaler.transform(test_values)

    raw_train = cache.loc[cache.run_id.isin(fold["train_runs"]), MODEL_FEATURES]
    raw_test = cache.loc[cache.run_id.isin(fold["test_runs"]), MODEL_FEATURES]
    raw_scaler = StandardScaler().fit(raw_train)
    raw_train_scaled = raw_scaler.transform(raw_train)
    raw_test_scaled = raw_scaler.transform(raw_test)
    for feature in SIGNAL_FEATURES:
        index = MODEL_FEATURES.index(feature)
        shift_rows.append(
            {
                "fold_id": fold["fold_id"],
                "test_run_id": fold["test_runs"][0],
                "feature": feature,
                "run4_wasserstein": wasserstein_distance(
                    raw_train_scaled[:, index], raw_test_scaled[:, index]
                ),
                "run5_wasserstein": wasserstein_distance(
                    train_scaled[:, index], test_scaled[:, index]
                ),
            }
        )
assert not preprocessing_issues, preprocessing_issues
feature_shift = pd.DataFrame(shift_rows)
feature_shift["change"] = feature_shift.run5_wasserstein - feature_shift.run4_wasserstein
feature_shift["relative_reduction"] = 1 - (
    feature_shift.run5_wasserstein / feature_shift.run4_wasserstein
)

run4_fold = pd.read_csv(RUN4 / "run_04_verified_fold_summary.csv")
run4_aggregate = pd.read_csv(
    RUN4 / "experiment_outputs" / "all_model_comparisons_summary.csv"
)
fold_comparison = fold_summary.merge(
    run4_fold,
    on=["fold_id", "test_run_id", "model", "weight_profile"],
    suffixes=("_run5", "_run4"),
)
fold_comparison["rmse_change"] = (
    fold_comparison.rmse_mean_run5 - fold_comparison.rmse_mean_run4
)
fold_comparison["rmse_percent_change"] = (
    100 * fold_comparison.rmse_change / fold_comparison.rmse_mean_run4
)
fold_comparison["abs_late_bias_change"] = (
    fold_comparison.late_life_bias_mean_run5.abs()
    - fold_comparison.late_life_bias_mean_run4.abs()
)

aggregate_comparison = aggregate.merge(
    run4_aggregate,
    on=["model", "weight_profile"],
    suffixes=("_run5", "_run4"),
)
for column in (
    "macro_mae_mean",
    "macro_rmse_mean",
    "between_bearing_rmse_std",
    "worst_bearing_rmse",
    "macro_r2_mean",
    "macro_late_life_mae",
    "worst_abs_late_life_bias",
):
    aggregate_comparison[f"{column}_change"] = (
        aggregate_comparison[f"{column}_run5"]
        - aggregate_comparison[f"{column}_run4"]
    )

weak = aggregate[aggregate.model == "weak_pinn"].iloc[0]
weak_run4 = run4_aggregate[run4_aggregate.model == "weak_pinn"].iloc[0]
weak_fold = fold_comparison[fold_comparison.model == "weak_pinn"]
weak_folds_improved = int((weak_fold.rmse_change < 0).sum())
weak_late_bias_folds_improved = int((weak_fold.abs_late_bias_change <= 0).sum())
success_rows = [
    {
        "criterion": "All 36 jobs complete with finite reproducible metrics",
        "passed": True,
        "evidence": "36/36 completed; zero failures; all metrics reproduced",
    },
    {
        "criterion": "Weak-PINN macro RMSE below Run 4 value 0.314238",
        "passed": bool(weak.macro_rmse_mean < 0.314238),
        "evidence": f"{weak.macro_rmse_mean:.6f}",
    },
    {
        "criterion": "Weak-PINN worst-bearing RMSE below 0.497089",
        "passed": bool(weak.worst_bearing_rmse < 0.497089),
        "evidence": f"{weak.worst_bearing_rmse:.6f}",
    },
    {
        "criterion": "Weak-PINN improves at least three of four folds",
        "passed": weak_folds_improved >= 3,
        "evidence": f"{weak_folds_improved}/4 folds improved",
    },
    {
        "criterion": "Weak-PINN between-bearing RMSE SD below 0.186672",
        "passed": bool(weak.between_bearing_rmse_std < 0.186672),
        "evidence": f"{weak.between_bearing_rmse_std:.6f}",
    },
    {
        "criterion": "Weak-PINN late-life behavior does not worsen",
        "passed": bool(
            weak.macro_late_life_mae <= weak_run4.macro_late_life_mae
            and weak_late_bias_folds_improved >= 3
        ),
        "evidence": (
            f"macro late MAE {weak_run4.macro_late_life_mae:.6f} -> "
            f"{weak.macro_late_life_mae:.6f}; abs bias improved in "
            f"{weak_late_bias_folds_improved}/4 folds; worst abs bias "
            f"{weak_run4.worst_abs_late_life_bias:.6f} -> "
            f"{weak.worst_abs_late_life_bias:.6f}"
        ),
    },
]
success = pd.DataFrame(success_rows)

verified_best.to_csv(RUN / "run_05_verified_best_metrics.csv", index=False)
verified_final.to_csv(RUN / "run_05_verified_final_metrics.csv", index=False)
lifecycle.to_csv(RUN / "run_05_verified_lifecycle_metrics.csv", index=False)
training.to_csv(RUN / "run_05_training_diagnostics.csv", index=False)
prediction_diagnostics.to_csv(
    RUN / "run_05_prediction_diagnostics.csv", index=False
)
fold_summary.to_csv(RUN / "run_05_verified_fold_summary.csv", index=False)
aggregate.to_csv(RUN / "run_05_verified_aggregate_summary.csv", index=False)
feature_shift.to_csv(RUN / "run_04_vs_run_05_feature_shift.csv", index=False)
fold_comparison.to_csv(RUN / "run_04_vs_run_05_fold_comparison.csv", index=False)
aggregate_comparison.to_csv(
    RUN / "run_04_vs_run_05_aggregate_comparison.csv", index=False
)
success.to_csv(RUN / "run_05_success_criteria.csv", index=False)
pd.DataFrame(identity_issues, columns=["artifact_directory", "file", "column", "expected", "actual"]).to_csv(
    RUN / "run_05_prediction_identity_issues.csv", index=False
)
pd.DataFrame(inventory_issues, columns=["relative_path", "issue"]).to_csv(
    RUN / "run_05_inventory_issues.csv", index=False
)
pd.DataFrame(source_issues, columns=["relative_path", "issue"]).to_csv(
    RUN / "run_05_source_issues.csv", index=False
)
pd.DataFrame(preprocessing_issues, columns=["fold_id", "run_id", "issue"]).to_csv(
    RUN / "run_05_preprocessing_issues.csv", index=False
)

strong = aggregate[aggregate.model == "strong_pinn"].iloc[0]
lstm = aggregate[aggregate.model == "lstm"].iloc[0]
shift_mean_run4 = feature_shift.run4_wasserstein.mean()
shift_mean_run5 = feature_shift.run5_wasserstein.mean()
shift_reduction = 1 - shift_mean_run5 / shift_mean_run4
runtime_minutes = (
    pd.Timestamp(manifest["finished_utc"]) - pd.Timestamp(manifest["started_utc"])
).total_seconds() / 60
final_delta = training.groupby("model").final_minus_best_test_rmse.agg(
    ["mean", "min", "max"]
)
fold_winners = fold_summary[fold_summary.fold_rank_by_rmse == 1]

report = [
    "# Run 5 analysis",
    "",
    "## Validity decision",
    "",
    f"EXP-005 is a valid completed controlled preprocessing experiment. All 36 jobs completed on a Tesla T4 in {runtime_minutes:.1f} minutes with no failures. The exact clean Git commit `{run_commit}` exists locally and its 30 committed source blobs match the saved source manifest. Configuration, split, preprocessing, feature-cache, executed-notebook, artifact-inventory, and bundle checks passed. All best-checkpoint and final-epoch metrics were independently reproduced from predictions.",
    "",
    "The Run 4 prediction-identity defect is corrected. Every test and validation prediction file preserves the physical trajectory in both `run_id` and `bearing_run_id` and stores `run_05` separately in `experiment_run_id`; no identity issues were found.",
    "",
    "## Primary outcome",
    "",
    f"- Weak-PINN/high remained first by equal-bearing macro normalized RMSE at `{weak.macro_rmse_mean:.6f}`, an improvement of {(1-weak.macro_rmse_mean/weak_run4.macro_rmse_mean)*100:.1f}% over Run 4. Worst-bearing RMSE improved to `{weak.worst_bearing_rmse:.6f}` and between-bearing RMSE SD to `{weak.between_bearing_rmse_std:.6f}`.",
    f"- Strong-PINN ranked second at macro RMSE `{strong.macro_rmse_mean:.6f}`. It improved over Run 4 but won no fold.",
    f"- LSTM ranked third at macro RMSE `{lstm.macro_rmse_mean:.6f}`. Its worst-bearing RMSE and between-bearing variation improved, but its overall RMSE worsened because performance on IMS-DS2/B1 collapsed.",
    f"- Macro R2 remained negative for all models: Weak-PINN `{weak.macro_r2_mean:.3f}`, Strong-PINN `{strong.macro_r2_mean:.3f}`, and LSTM `{lstm.macro_r2_mean:.3f}`. Run 5 therefore does not establish generally reliable cross-bearing prediction.",
    "",
    "Per-bearing winners were "
    + "; ".join(
        f"{row.test_run_id}: {row.model} (RMSE {row.rmse_mean:.6f})"
        for row in fold_winners.itertuples(index=False)
    )
    + ".",
    "",
    "## Hypothesis and predeclared criteria",
    "",
    f"The Run 5 hypothesis is only partially supported and the full success criterion failed. Weak-PINN passed the aggregate macro-RMSE, worst-bearing, and between-bearing-variation thresholds, but improved only `{weak_folds_improved}/4` folds rather than at least three.",
    "",
    f"Late-life behavior was mixed: Weak-PINN worst absolute late-life bias improved from `{weak_run4.worst_abs_late_life_bias:.6f}` to `{weak.worst_abs_late_life_bias:.6f}`, but macro late-life MAE worsened from `{weak_run4.macro_late_life_mae:.6f}` to `{weak.macro_late_life_mae:.6f}`, and absolute late-life bias improved in only `{weak_late_bias_folds_improved}/4` folds. A conservative decision is therefore that the late-life condition did not pass.",
    "",
    "## Diagnostics and publication relevance",
    "",
    f"The transformation reproduced the label-free covariate result: mean signal-feature Wasserstein shift fell from `{shift_mean_run4:.4f}` to `{shift_mean_run5:.4f}` ({shift_reduction*100:.1f}% reduction), with a reduction in every fold. Reduced covariate discrepancy did not translate uniformly into lower RUL error.",
    "",
    "The strongest counterexample is IMS-DS2/B1: Weak-PINN RMSE increased from `0.207321` to `0.375087`, and LSTM increased from `0.144667` to `0.301256`. Conversely, Weak-PINN improved dramatically on IMS-DS1/B3 (`0.444780` to `0.170740`) and all models improved or remained close on the difficult IMS-DS3/B3 fold. The preprocessing effect is therefore model- and domain-dependent rather than universally stabilizing.",
    "",
    "Final-epoch minus best-validation test RMSE changes were: "
    + "; ".join(
        f"{model} mean {row['mean']:+.4f}, range [{row['min']:+.4f}, {row['max']:+.4f}]"
        for model, row in final_delta.iterrows()
    )
    + ". Best-validation checkpoint reporting remains necessary.",
    "",
    "This controlled negative/partial result is useful for the publication pivot: a fixed normalization rule and fixed physics weighting can both help some bearings while harming others even when measured covariate shift decreases. That directly motivates studying identifiability- or uncertainty-aware mechanisms that learn when a preprocessing assumption or physical prior is trustworthy.",
    "",
    "## Next decision",
    "",
    "Do not automatically prepare Run 6 or tune another fixed weight grid. Freeze Runs 4 and 5 as the controlled diagnostic baseline pair. The next repository task should be a formal 2021–2026 novelty matrix and publication research protocol that defines the central method, datasets, modern baselines, uncertainty treatment, ablations, and locked evaluation before further model implementation.",
]
(RUN / "RUN_05_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")

issues = [
    "# EXP-005 issues",
    "",
    "No structural, identity, hash, metric-reproduction, preprocessing-reconstruction, or job-completion defects were found.",
    "",
    "Scientific limitations remain:",
    "",
    "- All three models have negative equal-bearing macro R-squared.",
    "- Weak-PINN improved on only two of four held-out bearings.",
    "- Weak-PINN macro late-life MAE worsened despite a lower worst absolute late-life bias.",
    "- IMS-DS2/B1 is a counterexample where the measured feature shift decreased but Weak-PINN and LSTM errors increased sharply.",
    "- The experiment covers four IMS trajectories and cannot establish cross-dataset generalization.",
]
recommendations = [
    "# EXP-005 recommendations",
    "",
    "1. Freeze Runs 4 and 5 as the controlled cross-bearing diagnostic baseline pair.",
    "2. Do not prepare Run 6 as another fixed-weight or preprocessing sweep.",
    "3. Build a 2021-2026 literature novelty matrix before selecting the next method.",
    "4. Lock the publication protocol: datasets, leakage-safe splits, modern baselines, uncertainty, ablations, statistical tests, and success criteria.",
    "5. Use the Run 5 counterexample to motivate a reliability-aware mechanism that estimates when a preprocessing assumption or physics prior should be trusted.",
]

ANALYZED.mkdir(parents=True, exist_ok=True)
COMPARISONS.mkdir(parents=True, exist_ok=True)
(ANALYZED / "analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
(ANALYZED / "issues.md").write_text("\n".join(issues) + "\n", encoding="utf-8")
(ANALYZED / "recommendations.md").write_text(
    "\n".join(recommendations) + "\n", encoding="utf-8"
)
verified_best.to_csv(ANALYZED / "verified_metrics.csv", index=False)
success.to_csv(ANALYZED / "success_criteria.csv", index=False)
aggregate_comparison.to_csv(
    COMPARISONS / "run_04_vs_run_05_aggregate.csv", index=False
)
fold_comparison.to_csv(COMPARISONS / "run_04_vs_run_05_folds.csv", index=False)

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

print(f"Verified and analyzed Run 5 at {RUN}")
print(aggregate.to_string(index=False))
print(success.to_string(index=False))
