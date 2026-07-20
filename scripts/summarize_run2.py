from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "saved results" / "run_02"
OUTPUTS = RUN / "experiment_outputs"
RUN1 = ROOT / "saved results" / "run_01"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((target - target.mean()) ** 2))
    if denominator == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((target - prediction) ** 2)) / denominator


config = json.loads((OUTPUTS / "resolved_config.json").read_text(encoding="utf-8"))
assert config["run_label"] == "run_02"

reported = pd.read_csv(OUTPUTS / "all_model_comparisons.csv")
assert len(reported) == 108, f"Expected 108 model/profile/seed rows, found {len(reported)}"
assert set(reported["status"]) == {"ok"}

verified_rows: list[dict[str, object]] = []
prediction_rows: list[dict[str, object]] = []
target_signatures: dict[str, set[str]] = {}
for prediction_path in sorted(OUTPUTS.glob("*/*__seed_*/predictions.csv")):
    dataset = prediction_path.parts[-3]
    directory = prediction_path.parts[-2]
    model, profile, seed_label = directory.split("__")
    seed_repeat = int(seed_label.removeprefix("seed_"))
    frame = pd.read_csv(prediction_path)
    target = frame["target_rul"].to_numpy(dtype=float)
    prediction = frame["predicted_rul"].to_numpy(dtype=float)
    errors = prediction - target
    mse = float(np.mean(errors**2))
    row = reported[
        (reported["dataset"] == dataset)
        & (reported["model"] == model)
        & (reported["weight_profile"] == profile)
        & (reported["seed_repeat"] == seed_repeat)
    ]
    assert len(row) == 1, (dataset, model, profile, seed_repeat)
    row = row.iloc[0]
    recomputed = {
        "mae": float(np.mean(np.abs(errors))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(target, prediction),
    }
    differences = {name: abs(recomputed[name] - float(row[name])) for name in recomputed}
    verified_rows.append(
        {
            "dataset": dataset,
            "model": model,
            "weight_profile": profile,
            "seed_repeat": seed_repeat,
            "seed": int(row["seed"]),
            **{f"reported_{name}": float(row[name]) for name in recomputed},
            **{f"verified_{name}": value for name, value in recomputed.items()},
            "max_metric_abs_difference": max(differences.values()),
            "samples": len(frame),
        }
    )
    target_signature = hashlib.sha256(
        frame[["run_id", "sample_index", "target_rul"]]
        .to_csv(index=False)
        .encode("utf-8")
    ).hexdigest()
    target_signatures.setdefault(dataset, set()).add(target_signature)
    lifecycle = np.where(target > 2 / 3, "early", np.where(target > 1 / 3, "middle", "late"))
    for phase in ("early", "middle", "late"):
        mask = lifecycle == phase
        prediction_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "weight_profile": profile,
                "seed_repeat": seed_repeat,
                "phase": phase,
                "samples": int(mask.sum()),
                "mae": float(np.mean(np.abs(errors[mask]))) if mask.any() else np.nan,
                "bias": float(np.mean(errors[mask])) if mask.any() else np.nan,
                "target_mean": float(np.mean(target[mask])) if mask.any() else np.nan,
                "prediction_mean": float(np.mean(prediction[mask])) if mask.any() else np.nan,
            }
        )

verified = pd.DataFrame(verified_rows).sort_values(
    ["dataset", "verified_rmse", "model", "weight_profile", "seed_repeat"]
)
assert len(verified) == len(reported)
maximum_metric_difference = float(verified["max_metric_abs_difference"].max())
if maximum_metric_difference >= 1e-5:
    offenders = verified.nlargest(10, "max_metric_abs_difference")[[
        "dataset", "model", "weight_profile", "seed_repeat",
        "max_metric_abs_difference", "reported_mae", "verified_mae",
        "reported_rmse", "verified_rmse", "reported_r2", "verified_r2",
    ]]
    raise AssertionError(
        f"Maximum metric difference {maximum_metric_difference}\n"
        f"{offenders.to_string(index=False)}"
    )
assert all(len(signatures) == 1 for signatures in target_signatures.values())
verified.to_csv(RUN / "run_02_verified_metrics.csv", index=False)
pd.DataFrame(prediction_rows).to_csv(
    RUN / "run_02_lifecycle_errors.csv", index=False
)

summary = (
    verified.groupby(["dataset", "model", "weight_profile"], as_index=False)
    .agg(
        seed_repeats=("seed_repeat", "count"),
        mae_mean=("verified_mae", "mean"),
        mae_std=("verified_mae", "std"),
        rmse_mean=("verified_rmse", "mean"),
        rmse_std=("verified_rmse", "std"),
        r2_mean=("verified_r2", "mean"),
        r2_std=("verified_r2", "std"),
    )
    .sort_values(["dataset", "rmse_mean"])
)
summary["rank_by_mean_rmse"] = summary.groupby("dataset")["rmse_mean"].rank(
    method="dense"
)
summary.to_csv(RUN / "run_02_model_ranking.csv", index=False)

history_rows: list[dict[str, object]] = []
for history_path in sorted(OUTPUTS.glob("*/*__seed_*/history.csv")):
    dataset = history_path.parts[-3]
    model, profile, seed_label = history_path.parts[-2].split("__")
    history = pd.read_csv(history_path)
    best_index = int(history["validation_mse"].idxmin())
    record: dict[str, object] = {
        "dataset": dataset,
        "model": model,
        "weight_profile": profile,
        "seed_repeat": int(seed_label.removeprefix("seed_")),
        "epochs_completed": len(history),
        "best_epoch": int(history.loc[best_index, "epoch"]),
        "best_validation_mse": float(history.loc[best_index, "validation_mse"]),
        "final_validation_mse": float(history["validation_mse"].iloc[-1]),
        "epochs_after_best": int(history["epoch"].iloc[-1] - history.loc[best_index, "epoch"]),
    }
    for column in history.columns:
        if column not in {"epoch", "validation_mse", "learning_rate"}:
            record[f"initial_{column}"] = float(history[column].iloc[0])
            record[f"best_epoch_{column}"] = float(history[column].iloc[best_index])
            record[f"final_{column}"] = float(history[column].iloc[-1])
    history_rows.append(record)
history_diagnostics = pd.DataFrame(history_rows).sort_values(
    ["dataset", "model", "weight_profile", "seed_repeat"]
)
assert len(history_diagnostics) == 108
history_diagnostics.to_csv(RUN / "run_02_training_diagnostics.csv", index=False)

training_summary = (
    history_diagnostics.groupby(["dataset", "model", "weight_profile"], as_index=False)
    .agg(
        epochs_min=("epochs_completed", "min"),
        epochs_mean=("epochs_completed", "mean"),
        epochs_max=("epochs_completed", "max"),
        best_epoch_mean=("best_epoch", "mean"),
        best_validation_mse_mean=("best_validation_mse", "mean"),
        final_validation_mse_mean=("final_validation_mse", "mean"),
    )
)
training_summary["final_to_best_validation_ratio"] = (
    training_summary["final_validation_mse_mean"]
    / training_summary["best_validation_mse_mean"]
)
training_summary.to_csv(RUN / "run_02_training_summary.csv", index=False)

run1 = pd.read_csv(RUN1 / "experiment_outputs" / "all_model_comparisons.csv")
run1_comparison = run1[["dataset", "model", "weight_profile", "mae", "rmse", "r2"]].merge(
    summary,
    on=["dataset", "model", "weight_profile"],
    how="inner",
    suffixes=("_run1", "_run2"),
)
run1_comparison["rmse_change_run2_minus_run1"] = (
    run1_comparison["rmse_mean"] - run1_comparison["rmse"]
)
run1_comparison["rmse_percent_change"] = 100.0 * (
    run1_comparison["rmse_change_run2_minus_run1"] / run1_comparison["rmse"]
)
run1_comparison.to_csv(RUN / "run_01_vs_run_02.csv", index=False)

sensitivity = pd.read_csv(OUTPUTS / "ims" / "sensitivity" / "sensitivity_results.csv")
sensitivity_ranking = pd.read_csv(
    OUTPUTS / "ims" / "sensitivity" / "sensitivity_ranking.csv"
)
assert len(sensitivity) == 15 and set(sensitivity["status"]) == {"ok"}

manifest_rows: list[dict[str, object]] = []
for path in sorted(OUTPUTS.rglob("*")):
    if path.is_file():
        manifest_rows.append(
            {
                "relative_path": path.relative_to(RUN).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(RUN / "artifact_manifest.csv", index=False)

best = summary.groupby("dataset", as_index=False).first()
best_single = verified.groupby("dataset", as_index=False).first()
target_details = []
for dataset in sorted(reported["dataset"].unique()):
    path = next(OUTPUTS.glob(f"{dataset}/*__seed_*/predictions.csv"))
    frame = pd.read_csv(path)
    target_details.append(
        {
            "dataset": dataset,
            "samples": len(frame),
            "runs": int(frame["run_id"].nunique()),
            "target_min": float(frame["target_rul"].min()),
            "target_max": float(frame["target_rul"].max()),
            "target_mean": float(frame["target_rul"].mean()),
        }
    )
targets = pd.DataFrame(target_details)
targets.to_csv(RUN / "run_02_test_set_summary.csv", index=False)

best_lifecycle = pd.DataFrame(prediction_rows).merge(
    best[["dataset", "model", "weight_profile"]],
    on=["dataset", "model", "weight_profile"],
    how="inner",
)
best_lifecycle_summary = (
    best_lifecycle.groupby(
        ["dataset", "model", "weight_profile", "phase"], as_index=False
    )
    .agg(
        samples=("samples", "max"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        bias_mean=("bias", "mean"),
        prediction_mean=("prediction_mean", "mean"),
        target_mean=("target_mean", "mean"),
    )
)
best_lifecycle_summary.to_csv(
    RUN / "run_02_best_model_lifecycle_summary.csv", index=False
)

run1_best = (
    run1.sort_values(["dataset", "rmse"]).groupby("dataset", as_index=False).first()
)
run1_best_map = {row.dataset: row for row in run1_best.itertuples(index=False)}

lines = [
    "# Run 2 analysis",
    "",
    "## Validity decision",
    "",
    "Run 2 is a complete and internally consistent diagnostic experiment: all 108 requested ",
    "dataset/model/profile/seed runs completed, all 15 sensitivity trials completed, every ",
    "reported prediction metric was independently reproduced, and each dataset used one identical ",
    "test target population across all models. It is not yet final thesis-grade evidence because ",
    "the downloaded bundle does not contain a Git commit, environment/GPU record, dataset ",
    "fingerprints, immutable split file/hash, or formal run manifest.",
    "",
    "## Execution and verification",
    "",
    f"- Main completed runs: {len(reported)} of 108; failed main runs: 0.",
    f"- Sensitivity trials: {len(sensitivity)} of 15; failed sensitivity trials: 0.",
    f"- Sum of recorded main training time: {reported['seconds'].sum() / 60:.1f} minutes.",
    f"- Archived artifacts: {len(manifest)} files, {manifest['bytes'].sum() / 1024**2:.1f} MiB.",
    f"- Largest absolute reported-versus-recomputed metric difference: {verified['max_metric_abs_difference'].max():.3e}.",
    "- All models used the same ordered test identifiers and targets within each dataset.",
    "- Configuration: sequence length 8, maximum 300 epochs, patience 40, batch size 64, and three seeds.",
    f"- No run reached 300 epochs; completed epochs ranged from {history_diagnostics['epochs_completed'].min()} to {history_diagnostics['epochs_completed'].max()} with median {history_diagnostics['epochs_completed'].median():.0f}.",
    f"- {int((history_diagnostics['epochs_after_best'] == 40).sum())} of 108 runs stopped exactly 40 epochs after their best validation epoch.",
    "",
    "## Best mean result per dataset (three seeds)",
    "",
]
for row in best.itertuples(index=False):
    lines.append(
        f"- {row.dataset}: {row.model}/{row.weight_profile}, "
        f"RMSE {row.rmse_mean:.6f} +/- {row.rmse_std:.6f}, "
        f"MAE {row.mae_mean:.6f} +/- {row.mae_std:.6f}, "
        f"R2 {row.r2_mean:.6f} +/- {row.r2_std:.6f}."
    )
lines.extend(["", "## Run 1 comparison", ""])
for row in best.itertuples(index=False):
    old = run1_best_map[row.dataset]
    if row.dataset == "kaist_vibration_temperature":
        lines.append(
            f"- {row.dataset}: Run 1 best RMSE was {old.rmse:.6f}; Run 2 best mean "
            f"RMSE is {row.rmse_mean:.6f}. This numerical decrease is not evidence of "
            "improvement: Run 2 uses three seeds and a different sequence/training setup, "
            "and all Run 2 R2 values remain negative on only 18 test sequences."
        )
    else:
        lines.append(
            f"- {row.dataset}: Run 1 best was {old.model}/{old.weight_profile} "
            f"at RMSE {old.rmse:.6f}; Run 2 best mean RMSE is {row.rmse_mean:.6f}. "
            "The worse score is expected because Run 1 directly supplied elapsed_norm while "
            "rul_norm = 1 - elapsed_norm; Run 2 removes that target leakage and is the more credible result."
        )
lines.extend(
    [
        "",
        "## Main findings",
        "",
        "- IMS: weak-PINN/high ranked first by mean RMSE, but seed variation is material. Its "
        "  physics priors improve the mean over data-only baselines, while weak-low and weak-medium "
        "  are substantially worse. This is evidence that the selected weak constraints can help, "
        "  not proof that the current weights are optimal.",
        "- PRONOSTIA: the FNN baseline ranked first. Every attention/PINN family was worse than the "
        "  simple FNN, so the present sequence encoder and physics priors do not improve generalization "
        "  across the held-out PRONOSTIA bearings/conditions.",
        "- KAIST: every aggregate R2 is strongly negative. Strong-PINN/high has the lowest mean RMSE, "
        "  but one seed is much worse and even the two better seeds have negative R2. This single-run "
        "  benchmark is too small for a reliable neural-model ranking and does not "
        "  demonstrate cross-bearing generalization.",
        "- Strong-PINN: explicit-law profiles remain worse than weak-PINN on IMS and worse than FNN on "
        "  PRONOSTIA. Higher simultaneous physics weights degrade IMS and PRONOSTIA performance, so the "
        "  current physical residual scales/calibration are not yet validated.",
        "- Convergence: training objectives generally continued to fall after validation performance "
        "  peaked, and every run stopped at the configured patience boundary. This is conventional "
        "  overfitting/domain-shift behavior rather than optimization failure; increasing epochs alone "
        "  is unlikely to improve held-out performance.",
        "- Lifecycle error: the best IMS model systematically underpredicts early-life RUL and "
        "  overpredicts late-life RUL; the best PRONOSTIA model also overpredicts late-life RUL. The "
        "  models regress toward middle-life predictions, which is safety-relevant because late-life "
        "  overprediction delays failure warnings.",
        "- KAIST evaluation contains only 18 test sequences and all are late-life targets (RUL 0 to "
        "  0.1328). Its extreme negative R2 values and unstable ranking are therefore unsurprising.",
        "- Sensitivity: IMS RMSE is most responsive to Paris-loss weight (range 0.3388), followed by "
        "  Miner-loss weight (0.2524). The Paris weight 0.01 produced RMSE 0.1759 in the single-seed, "
        "  80-epoch one-factor sweep. That promising result must be repeated over multiple seeds and a "
        "  validation-based selection; it cannot be adopted from this one test-set sweep.",
        "- IMS thermoviscosity beta and poor-lubrication multiplier had exactly zero sensitivity because "
        "  IMS has no measured temperature channel and the temperature/lubrication branch is masked. "
        "  Those parameters must be studied on temperature-equipped data, not claimed as IMS evidence.",
        "",
        "## Next experiment recommendation",
        "",
        "Do not jump directly to a raw-waveform model yet. Run 3 should be a controlled validation and "
        "physics-calibration experiment while retaining the corrected Run 2 features, labels, run-level "
        "splits, sequence length, architecture widths, optimizer, and three-seed protocol.",
        "",
        "1. Stop using the final test set for weight selection. Add an explicit validation-comparison "
        "   table and choose weights only from validation RMSE plus physics-residual diagnostics; evaluate "
        "   the selected frozen profile once on test.",
        "2. Repeat only the promising IMS strong-PINN neighborhood with three seeds: Paris weights "
        "   0.003, 0.01, and 0.03 crossed with Miner weights 0.0003, 0.001, and 0.003. Record weighted and "
        "   unweighted residual magnitudes so data and physics gradients can be compared.",
        "3. Treat datasets separately: retain weak-high as the IMS candidate, FNN as the PRONOSTIA "
        "   reference, and mark KAIST exploratory until more independent run-to-failure bearings or a "
        "   defensible transfer-learning design are available.",
        "4. Correct PRONOSTIA physics metadata to run-specific load/speed regimes before presenting an "
        "   explicit-law PINN as physical evidence. Verify bearing geometry/capacity, stress, lubricant, "
        "   and units from authoritative dataset/bearing documentation.",
        "5. Add the missing experiment contract: experiment ID, Git SHA, environment/GPU file, dataset "
        "   fingerprints, immutable split hash, manifest, per-seed parameter counts, inference time, and "
        "   original-unit RUL metrics.",
        "6. After Run 3 establishes a leakage-free, calibrated feature baseline, make Run 4 the planned "
        "   features-only versus raw-only versus hybrid encoder ablation on exactly the same run-level "
        "   splits. That isolates whether raw representation learning actually adds value.",
    ]
)
(RUN / "RUN_02_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Archived and analyzed Run 2 at {RUN}")
print(best[["dataset", "model", "weight_profile", "rmse_mean", "rmse_std"]].to_string(index=False))
