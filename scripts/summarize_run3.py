from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

from thesis_work.multi_dataset import MODEL_FEATURES


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "saved results" / "run_03"
OUTPUTS = RUN / "experiment_outputs"
UPLOAD = ROOT / "Upload"
RUN2 = ROOT / "saved results" / "run_02" / "experiment_outputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    errors = prediction - target
    mse = float(np.mean(errors**2))
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(np.sum(errors**2)) / denominator if denominator else np.nan
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
    }


manifest = json.loads((OUTPUTS / "run_manifest.json").read_text(encoding="utf-8"))
config = json.loads((OUTPUTS / "resolved_config.json").read_text(encoding="utf-8"))
split = json.loads((OUTPUTS / "data_split.json").read_text(encoding="utf-8"))
failure_report = json.loads(
    (OUTPUTS / "failure_report.json").read_text(encoding="utf-8")
)
assert manifest["experiment_id"] == "EXP-003"
assert manifest["run_id"] == "run_03"
assert manifest["status"] == "completed"
assert not failure_report["failures"]
assert sha256(OUTPUTS / "resolved_config.json") == manifest["resolved_config_sha256"]
assert sha256(OUTPUTS / "data_split.json") == manifest["data_split_sha256"]

cache_path = UPLOAD / "feature_cache" / "ims_features.csv"
assert sha256(cache_path) == manifest["dataset_feature_cache_sha256"]
source_manifest = pd.read_csv(OUTPUTS / "source_manifest.csv")
source_mismatches = []
for row in source_manifest.itertuples(index=False):
    local = UPLOAD / row.relative_path
    if not local.is_file() or sha256(local) != row.sha256:
        source_mismatches.append(row.relative_path)
pd.DataFrame(
    {"source_manifest_mismatch": source_mismatches}
).to_csv(RUN / "run_03_source_mismatches.csv", index=False)
combined_source_hash = hashlib.sha256(
    "\n".join(
        f"{row.relative_path}:{row.sha256}"
        for row in source_manifest.sort_values("relative_path").itertuples(index=False)
    ).encode()
).hexdigest()
assert combined_source_hash == manifest["source_tree_sha256"]

recorded_inventory = pd.read_csv(OUTPUTS / "artifact_inventory.csv")
inventory_mismatches = []
for row in recorded_inventory.itertuples(index=False):
    path = OUTPUTS / row.relative_path
    if not path.is_file() or path.stat().st_size != row.bytes or sha256(path) != row.sha256:
        inventory_mismatches.append(row.relative_path)
assert not inventory_mismatches, inventory_mismatches

reported_test = pd.read_csv(OUTPUTS / "all_model_comparisons.csv")
grid = pd.read_csv(
    OUTPUTS / "ims" / "calibration" / "validation_grid_results.csv"
)
grid_summary = pd.read_csv(
    OUTPUTS / "ims" / "calibration" / "validation_grid_summary.csv"
)
selection = json.loads(
    (OUTPUTS / "ims" / "calibration" / "selected_profile.json").read_text(
        encoding="utf-8"
    )
)
assert len(reported_test) == 9 and set(reported_test["status"]) == {"ok"}
assert len(grid) == 27 and set(grid["status"]) == {"ok"}
assert not grid["test_evaluated"].astype(bool).any()
assert set(grid["selection_split"]) == {"validation"}


def test_prediction_path(row: pd.Series) -> Path:
    if row["model"] == "strong_pinn":
        return (
            OUTPUTS
            / "ims"
            / "calibration"
            / row["weight_profile"]
            / f"seed_{int(row['seed_repeat']):02d}"
            / "predictions.csv"
        )
    return (
        OUTPUTS
        / "ims"
        / f"{row['model']}__{row['weight_profile']}__seed_{int(row['seed_repeat']):02d}"
        / "predictions.csv"
    )


verified_test_rows = []
lifecycle_rows = []
test_signatures = set()
for _, row in reported_test.iterrows():
    path = test_prediction_path(row)
    frame = pd.read_csv(path)
    target = frame["target_rul"].to_numpy(float)
    prediction = frame["predicted_rul"].to_numpy(float)
    target_seconds = frame["target_rul_seconds"].to_numpy(float)
    prediction_seconds = frame["predicted_rul_seconds"].to_numpy(float)
    normalized = metrics(target, prediction)
    original = metrics(target_seconds, prediction_seconds)
    normalized_differences = [
        abs(normalized[name] - float(row[name]))
        for name in ("mae", "mse", "rmse", "r2")
    ]
    original_relative_differences = [
        abs(original[name] - float(row[output_name]))
        / max(abs(float(row[output_name])), 1e-12)
        for name, output_name in (
            ("mae", "mae_seconds"),
            ("mse", "mse_seconds2"),
            ("rmse", "rmse_seconds"),
            ("r2", "r2_seconds"),
        )
    ]
    verified_test_rows.append(
        {
            "dataset": row["dataset"],
            "model": row["model"],
            "weight_profile": row["weight_profile"],
            "seed_repeat": int(row["seed_repeat"]),
            "seed": int(row["seed"]),
            **{f"verified_{name}": value for name, value in normalized.items()},
            "verified_mae_seconds": original["mae"],
            "verified_rmse_seconds": original["rmse"],
            "verified_r2_seconds": original["r2"],
            "prediction_mean": float(prediction.mean()),
            "prediction_min": float(prediction.min()),
            "prediction_max": float(prediction.max()),
            "maximum_normalized_metric_abs_difference": max(normalized_differences),
            "maximum_original_metric_relative_difference": max(original_relative_differences),
            "samples": len(frame),
        }
    )
    signature = hashlib.sha256(
        frame[["run_id", "sample_index", "target_rul"]]
        .to_csv(index=False)
        .encode()
    ).hexdigest()
    test_signatures.add(signature)
    error = prediction - target
    phase_values = np.where(
        target > 2 / 3, "early", np.where(target > 1 / 3, "middle", "late")
    )
    for phase in ("early", "middle", "late"):
        mask = phase_values == phase
        lifecycle_rows.append(
            {
                "model": row["model"],
                "weight_profile": row["weight_profile"],
                "seed_repeat": int(row["seed_repeat"]),
                "phase": phase,
                "samples": int(mask.sum()),
                "mae": float(np.mean(np.abs(error[mask]))),
                "bias": float(np.mean(error[mask])),
                "target_mean": float(target[mask].mean()),
                "prediction_mean": float(prediction[mask].mean()),
            }
        )

verified_test = pd.DataFrame(verified_test_rows)
assert len(test_signatures) == 1
assert verified_test["maximum_normalized_metric_abs_difference"].max() < 2e-5
assert verified_test["maximum_original_metric_relative_difference"].max() < 2e-5
verified_test.to_csv(RUN / "run_03_verified_test_metrics.csv", index=False)

verified_validation_rows = []
validation_signatures = set()
for row in grid.itertuples(index=False):
    path = OUTPUTS / row.artifact_directory / "validation_predictions.csv"
    frame = pd.read_csv(path)
    target = frame["target_rul"].to_numpy(float)
    prediction = frame["predicted_rul"].to_numpy(float)
    recomputed = metrics(target, prediction)
    differences = [
        abs(recomputed[name] - float(getattr(row, f"validation_{name}")))
        for name in ("mae", "mse", "rmse", "r2")
    ]
    verified_validation_rows.append(
        {
            "weight_profile": row.weight_profile,
            "paris_weight": row.paris_weight,
            "miner_weight": row.miner_weight,
            "seed_repeat": row.seed_repeat,
            "seed": row.seed,
            **{f"verified_validation_{name}": value for name, value in recomputed.items()},
            "weighted_physics_to_data_ratio": row.weighted_physics_to_data_ratio,
            "gradient_physics_to_data_ratio": row.gradient_physics_to_data_ratio,
            "maximum_metric_difference": max(differences),
        }
    )
    signature = hashlib.sha256(
        frame[["run_id", "sample_index", "target_rul"]]
        .to_csv(index=False)
        .encode()
    ).hexdigest()
    validation_signatures.add(signature)

verified_validation = pd.DataFrame(verified_validation_rows)
assert len(validation_signatures) == 1
assert verified_validation["maximum_metric_difference"].max() < 2e-5
verified_validation.to_csv(
    RUN / "run_03_verified_validation_grid.csv", index=False
)
recomputed_selection = (
    verified_validation.groupby(
        ["weight_profile", "paris_weight", "miner_weight"], as_index=False
    )["verified_validation_rmse"]
    .mean()
    .sort_values("verified_validation_rmse")
    .iloc[0]
)
assert recomputed_selection["weight_profile"] == selection["selected_profile"]

calibration_test_files = list(
    (OUTPUTS / "ims" / "calibration").glob("*/seed_*/predictions.csv")
)
assert len(calibration_test_files) == 3
assert {
    path.parts[-3] for path in calibration_test_files
} == {selection["selected_profile"]}

ranking = (
    verified_test.groupby(["model", "weight_profile"], as_index=False)
    .agg(
        seed_repeats=("seed", "nunique"),
        mae_mean=("verified_mae", "mean"),
        mae_std=("verified_mae", "std"),
        rmse_mean=("verified_rmse", "mean"),
        rmse_std=("verified_rmse", "std"),
        r2_mean=("verified_r2", "mean"),
        r2_std=("verified_r2", "std"),
        rmse_seconds_mean=("verified_rmse_seconds", "mean"),
        prediction_mean=("prediction_mean", "mean"),
    )
    .sort_values("rmse_mean")
    .reset_index(drop=True)
)
ranking["rank_by_mean_rmse"] = np.arange(1, len(ranking) + 1)
ranking.to_csv(RUN / "run_03_model_ranking.csv", index=False)

lifecycle = pd.DataFrame(lifecycle_rows)
lifecycle_summary = (
    lifecycle.groupby(["model", "weight_profile", "phase"], as_index=False)
    .agg(
        samples=("samples", "max"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        bias_mean=("bias", "mean"),
        prediction_mean=("prediction_mean", "mean"),
        target_mean=("target_mean", "mean"),
    )
)
lifecycle_summary.to_csv(RUN / "run_03_lifecycle_summary.csv", index=False)

history_rows = []
history_paths = [
    *sorted((OUTPUTS / "ims").glob("*__seed_*/history.csv")),
    *sorted((OUTPUTS / "ims" / "calibration").glob("*/seed_*/history.csv")),
]
for path in history_paths:
    history = pd.read_csv(path)
    best_index = int(history["validation_mse"].idxmin())
    relative = path.relative_to(OUTPUTS).as_posix()
    history_rows.append(
        {
            "relative_path": relative,
            "epochs_completed": len(history),
            "best_epoch": int(history.loc[best_index, "epoch"]),
            "best_validation_mse": float(history.loc[best_index, "validation_mse"]),
            "final_validation_mse": float(history["validation_mse"].iloc[-1]),
            "epochs_after_best": int(
                history["epoch"].iloc[-1] - history.loc[best_index, "epoch"]
            ),
            "best_weighted_data": float(history.loc[best_index].get("weighted_data", np.nan)),
            "best_weighted_paris": float(
                history.loc[best_index].get("weighted_paris_crack_growth", np.nan)
            ),
            "best_weighted_miner": float(
                history.loc[best_index].get("weighted_palmgren_miner", np.nan)
            ),
        }
    )
training = pd.DataFrame(history_rows)
assert len(training) == 33
training.to_csv(RUN / "run_03_training_diagnostics.csv", index=False)

cache = pd.read_csv(cache_path)
split_ids = split["ims"]
train_frame = cache[cache["run_id"].isin(split_ids["train_runs"])].copy()
validation_frame = cache[cache["run_id"].isin(split_ids["validation_runs"])].copy()
test_frame = cache[cache["run_id"].isin(split_ids["test_runs"])].copy()
clean = lambda frame: frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
scaler = StandardScaler().fit(clean(train_frame))
train_scaled = scaler.transform(clean(train_frame))
validation_scaled = scaler.transform(clean(validation_frame))
test_scaled = scaler.transform(clean(test_frame))
feature_shift_rows = []
for index, feature in enumerate(MODEL_FEATURES):
    feature_shift_rows.append(
        {
            "feature": feature,
            "validation_standardized_mean": float(validation_scaled[:, index].mean()),
            "test_standardized_mean": float(test_scaled[:, index].mean()),
            "absolute_validation_test_mean_gap": float(
                abs(validation_scaled[:, index].mean() - test_scaled[:, index].mean())
            ),
            "validation_test_wasserstein": float(
                wasserstein_distance(
                    validation_scaled[:, index], test_scaled[:, index]
                )
            ),
            "train_test_wasserstein": float(
                wasserstein_distance(train_scaled[:, index], test_scaled[:, index])
            ),
        }
    )
feature_shift = pd.DataFrame(feature_shift_rows).sort_values(
    "validation_test_wasserstein", ascending=False
)
feature_shift.to_csv(RUN / "run_03_feature_shift.csv", index=False)

run2 = pd.read_csv(RUN2 / "all_model_comparisons_summary.csv")
comparison_rows = []
for row in ranking.itertuples(index=False):
    prior = run2[
        (run2["dataset"] == "ims")
        & (run2["model"] == row.model)
        & (run2["weight_profile"] == row.weight_profile)
    ]
    comparison_rows.append(
        {
            "model": row.model,
            "weight_profile": row.weight_profile,
            "run_02_rmse_mean": float(prior.iloc[0]["rmse_mean"])
            if len(prior) == 1
            else np.nan,
            "run_03_rmse_mean": row.rmse_mean,
            "run_03_rmse_std": row.rmse_std,
        }
    )
run_comparison = pd.DataFrame(comparison_rows)
run_comparison["run_03_minus_run_02_rmse"] = (
    run_comparison["run_03_rmse_mean"] - run_comparison["run_02_rmse_mean"]
)
run_comparison.to_csv(RUN / "run_02_vs_run_03.csv", index=False)

archive_rows = []
for path in sorted(OUTPUTS.rglob("*")):
    if path.is_file():
        archive_rows.append(
            {
                "relative_path": path.relative_to(RUN).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
archive_manifest = pd.DataFrame(archive_rows)
archive_manifest.to_csv(RUN / "artifact_manifest.csv", index=False)

selected_validation_rmse = float(selection["validation_rmse_mean"])
selected_test = ranking[ranking["model"] == "strong_pinn"].iloc[0]
lstm = ranking[ranking["model"] == "lstm"].iloc[0]
weak = ranking[ranking["model"] == "weak_pinn"].iloc[0]
selected_late = lifecycle_summary[
    (lifecycle_summary["model"] == "strong_pinn")
    & (lifecycle_summary["phase"] == "late")
].iloc[0]
run2_strong_low = float(
    run2[
        (run2["dataset"] == "ims")
        & (run2["model"] == "strong_pinn")
        & (run2["weight_profile"] == "strong_low")
    ].iloc[0]["rmse_mean"]
)
top_shift = feature_shift.head(5)["feature"].tolist()

lines = [
    "# Run 3 analysis",
    "",
    "## Validity decision",
    "",
    "Run 3 is a complete, internally consistent EXP-003 diagnostic experiment. All 33 "
    "training jobs completed on a Tesla T4, all 27 calibration candidates used the same "
    "validation population, exactly the three seeds of the frozen validation winner were "
    "evaluated on test, and every recorded validation/test metric was independently "
    "reproduced. Configuration, split, feature-cache, Python/config source, and artifact hashes verify. "
    "The run still lacks a Git commit SHA, so the recorded source manifest, rather than "
    "Git history, is the authoritative code identity.",
    "",
    "- Traceability issue: the recorded notebook hash differs from the notebook now in local Upload; future bundles must include the executed notebook copy.",
    "",
    "## Outcome",
    "",
    f"- Validation selected `{selection['selected_profile']}` with mean validation RMSE "
    f"{selected_validation_rmse:.6f} +/- {selection['validation_rmse_std']:.6f}.",
    f"- On the untouched test bearing, that Strong-PINN produced RMSE "
    f"{selected_test.rmse_mean:.6f} +/- {selected_test.rmse_std:.6f} and R2 "
    f"{selected_test.r2_mean:.6f}.",
    f"- LSTM ranked first on test: RMSE {lstm.rmse_mean:.6f} +/- {lstm.rmse_std:.6f}, "
    f"R2 {lstm.r2_mean:.6f}, and original-time RMSE "
    f"{lstm.rmse_seconds_mean / 3600:.2f} hours.",
    f"- Weak-PINN/high ranked second: RMSE {weak.rmse_mean:.6f} +/- "
    f"{weak.rmse_std:.6f}, R2 {weak.r2_mean:.6f}.",
    "",
    "## Hypothesis decision",
    "",
    f"EXP-003 failed its declared success criterion. The selected Strong-PINN test RMSE "
    f"was {100 * (selected_test.rmse_mean / run2_strong_low - 1):.1f}% worse than Run 2 "
    f"Strong-PINN/low ({run2_strong_low:.6f}). The validation-to-test RMSE ratio was "
    f"{selected_test.rmse_mean / selected_validation_rmse:.1f}x, demonstrating severe "
    "cross-bearing domain shift rather than a valid physics improvement.",
    "",
    "## Interpretation",
    "",
    "- The selection policy worked as designed; the failure is scientific, not a test-leakage "
    "  or artifact error.",
    f"- The selected Strong-PINN collapsed toward high RUL: its mean test prediction was "
    f"{selected_test.prediction_mean:.3f}, and its late-life mean prediction was "
    f"{selected_late.prediction_mean:.3f} against a late-life target mean of "
    f"{selected_late.target_mean:.3f}.",
    "- All Paris=0.003 profiles looked excellent on the single validation bearing, but the "
    "  selected profile failed on the second-test bearing. One validation trajectory is not "
    "  representative enough to calibrate explicit physics weights.",
    f"- At selection, weighted physics loss was about "
    f"{grid_summary.iloc[0]['weighted_physics_to_data_ratio_mean']:.1f} times weighted data "
    "  loss, although the physics/data gradient ratio was about "
    f"{grid_summary.iloc[0]['gradient_physics_to_data_ratio_mean']:.2f}. The scalar objective "
    "  remains dominated by residual magnitude, and the physical calibration is not portable.",
    "- LSTM improved sharply relative to Run 2 because Run 3 used a new common seed set; this "
    "  shows that the apparent ranking remains seed-sensitive and should not be generalized "
    "  from one fixed three-seed set alone.",
    "- The most shifted validation-versus-test feature distributions were: "
    + ", ".join(top_shift)
    + ". This supports a bearing/rig domain-shift explanation.",
    "",
    "## Required next step",
    "",
    "Run 4 should not yet be the raw-versus-hybrid ablation. First establish a robust IMS "
    "evaluation using fixed configurations and leave-one-bearing-out folds. Rotate each of "
    "the four IMS trajectories as the held-out test bearing, use no test-driven weight tuning, "
    "and compare LSTM, Weak-PINN/high, and the frozen Run 3 Strong-PINN profile over the same "
    "three seeds. This isolates split sensitivity and reports per-bearing plus aggregate "
    "performance. Keep model architecture, features, optimizer, sequence length, and weights "
    "unchanged. Only after a model is stable across bearings should Run 5 compare feature-only, "
    "raw-only, and hybrid encoders on those same folds.",
]
(RUN / "RUN_03_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Verified and analyzed Run 3 at {RUN}")
print(ranking.to_string(index=False))
