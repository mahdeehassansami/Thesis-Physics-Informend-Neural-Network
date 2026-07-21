from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "analyzed" / "EXP-007" / "open_diagnostics"
FAMILIES = ["linear_increasing", "progressively_increasing", "step_like", "gamma"]


def aggregate_units(frame: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    keys = [
        "seed",
        "run_id",
        "candidate_spec",
        "true_family",
        "candidate_family",
        "corruption_type",
        "condition_id",
        "validity_label",
    ]
    return frame.groupby(keys, as_index=False).agg(**{name: (name, "mean") for name in numeric})


def matrix(frame: pd.DataFrame, numeric: list[str], include_family: bool) -> np.ndarray:
    values = [frame[numeric].to_numpy(dtype=float)]
    if include_family:
        values.extend(
            (frame["candidate_family"].to_numpy() == family)
            .astype(float)
            .reshape(-1, 1)
            for family in FAMILIES
        )
    result = np.concatenate(values, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("EXP-007 diagnostic matrix contains non-finite values.")
    return result


def grouped_predictions(
    units: pd.DataFrame,
    numeric: list[str],
    include_family: bool,
    random_seed: int,
) -> np.ndarray:
    x = matrix(units, numeric, include_family)
    y = units["validity_label"].to_numpy(dtype=int)
    groups = units["run_id"].to_numpy()
    predictions = np.full(len(units), np.nan, dtype=float)
    for fold, (train_index, test_index) in enumerate(
        LeaveOneGroupOut().split(x, y, groups), start=1
    ):
        model = ExtraTreesClassifier(
            n_estimators=256,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_seed + fold,
            n_jobs=-1,
        )
        model.fit(x[train_index], y[train_index])
        predictions[test_index] = model.predict_proba(x[test_index])[:, 1]
    if not np.isfinite(predictions).all():
        raise RuntimeError("Grouped diagnostic did not predict every held-out trajectory.")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run non-confirmatory grouped diagnostics on opened EXP-007 artifacts."
    )
    parser.add_argument(
        "artifact_root",
        nargs="?",
        type=Path,
        default=ROOT / "saved results" / "run_07" / "experiment_outputs",
    )
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "EXP-007":
        raise ValueError("The diagnostic input is not EXP-007.")
    config = yaml.safe_load(
        (ROOT / "configs" / "experiment_exp007.yaml").read_text(encoding="utf-8")
    )
    numeric = list(config["credibility"]["numeric_evidence"])
    frames: list[pd.DataFrame] = []
    for seed in config["training"]["seeds"]:
        path = root / "seeds" / f"seed_{int(seed):05d}" / "test_credibility_predictions.csv"
        frame = pd.read_csv(path)
        frame["seed"] = int(seed)
        frames.append(frame)
    evidence = pd.concat(frames, ignore_index=True)
    units = aggregate_units(evidence, numeric)

    condition_features = {"covariate_shift_score", "operation_shift_score"}
    residual_features = {
        "template_absolute_residual",
        "template_standardized_residual",
        "lifecycle_disagreement",
        "rolling_absolute_residual",
        "rolling_lifecycle_disagreement",
        "prior_monotonic_violation_rate",
        "residual_relative_to_best",
        "residual_rank_fraction",
    }
    feature_sets = {
        "full_allowed": (numeric, True),
        "without_shift": ([name for name in numeric if name not in condition_features], True),
        "without_candidate_family": (numeric, False),
        "residual_only": ([name for name in numeric if name in residual_features], False),
        "condition_only": ([name for name in numeric if name in condition_features], False),
    }
    rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    for seed, seed_units in units.groupby("seed"):
        for feature_set, (columns, include_family) in feature_sets.items():
            probability = grouped_predictions(
                seed_units.reset_index(drop=True), columns, include_family, int(seed) + 70070
            )
            target = seed_units["validity_label"].to_numpy(dtype=int)
            rows.append(
                {
                    "seed": int(seed),
                    "feature_set": feature_set,
                    "held_out_unit": "complete_trajectory",
                    "features": len(columns) + (4 if include_family else 0),
                    "auroc": float(roc_auc_score(target, probability)),
                    "auprc": float(average_precision_score(target, probability)),
                    "brier": float(brier_score_loss(target, probability)),
                    "prevalence": float(target.mean()),
                }
            )
            prediction_rows.append(
                seed_units[["seed", "run_id", "candidate_spec", "validity_label"]]
                .copy()
                .assign(feature_set=feature_set, diagnostic_probability=probability)
            )

    results = pd.DataFrame(rows)
    aggregate = (
        results.groupby("feature_set", as_index=False)
        .agg(
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            mean_auprc=("auprc", "mean"),
            mean_brier=("brier", "mean"),
            seeds=("seed", "nunique"),
        )
        .sort_values("mean_auroc", ascending=False)
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT / "grouped_feature_ablation_by_seed.csv", index=False)
    aggregate.to_csv(OUTPUT / "grouped_feature_ablation_summary.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        OUTPUT / "grouped_diagnostic_predictions.csv", index=False
    )
    report = {
        "experiment_id": "EXP-007",
        "status": "open_data_diagnostic_only",
        "confirmation_use_allowed": False,
        "held_out_unit": "complete_trajectory",
        "classifier": "extra_trees",
        "seeds": [int(value) for value in sorted(units["seed"].unique())],
        "truth_metadata_sanity_ceiling_auroc": 1.0,
        "best_allowed_feature_set": str(aggregate.iloc[0]["feature_set"]),
        "best_mean_allowed_auroc": float(aggregate.iloc[0]["mean_auroc"]),
    }
    (OUTPUT / "diagnostic_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# EXP-007 opened-test diagnostics\n\n"
        "These complete-trajectory grouped nonlinear ceilings and feature ablations were run "
        "after EXP-007 was opened. They diagnose the failed version 0.1 evidence design and "
        "must not be used as confirmation evidence for EXP-007A. The truth-metadata sanity "
        "ceiling is trivially 1.0 because family/scale truth defines the old validity label; "
        "it is not an allowed deployable model.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
