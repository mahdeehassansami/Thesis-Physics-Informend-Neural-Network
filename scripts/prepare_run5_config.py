from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN4 = ROOT / "configs" / "colab_experiments_run_04.json"
RUN5 = ROOT / "configs" / "colab_experiments_run_05.json"
ACTIVE = ROOT / "configs" / "colab_experiments.json"

SIGNAL_FEATURES = [
    "rms",
    "std",
    "ptp",
    "kurtosis",
    "crest_factor",
    "mean_abs",
    "skewness",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_entropy",
    "high_frequency_ratio",
]


def main() -> None:
    config = json.loads(RUN4.read_text(encoding="utf-8"))
    if config.get("run_label") != "run_04":
        raise ValueError("Expected the preserved Run 4 configuration.")

    run5 = copy.deepcopy(config)
    run5["schema_version"] = 4
    run5["run_label"] = "run_05"
    run5["repository"]["expected_commit"] = None
    run5["repository"]["note"] = (
        "The Colab notebook requires the exact 40-character Run 5 commit SHA and "
        "checks out that revision under /content before importing thesis_work."
    )
    run5["preprocessing"] = {
        "strategy": "per_run_initial_robust_relative",
        "fit_scope": "each physical run independently",
        "prefix_samples": 8,
        "feature_columns": SIGNAL_FEATURES,
        "center": "median",
        "scale": "max_abs_median_scaled_mad_floor",
        "mad_consistency_constant": 1.4826,
        "absolute_scale_floor": 1e-8,
        "require_prefix_before_first_prediction": True,
        "uses_targets": False,
        "uses_failure_time": False,
        "post_transform_scaler": "StandardScaler fitted on training runs only",
        "rationale": (
            "Run 4 showed large cross-bearing shifts in vibration feature location and "
            "scale. Eight fixed initial snapshots equal the sequence length, so the "
            "label-free baseline is available before the first prediction without "
            "using total trajectory length or future failure information."
        ),
    }
    if run5["preprocessing"]["prefix_samples"] != run5["training"][
        "sequence_length"
    ]:
        raise ValueError("Run 5 baseline prefix must equal the frozen sequence length.")

    run5["cross_bearing"]["test_policy"] = (
        "All model configurations, weights, folds, and seeds are frozen before EXP-005. "
        "Each bearing's baseline uses only its first eight unlabeled snapshots; test "
        "targets and test metrics never fit preprocessing or alter a model."
    )
    run5["cross_bearing"]["baseline_comparison"] = {
        "source_experiment": "EXP-004",
        "source_run": "run_04",
        "primary_model": "weak_pinn",
        "primary_weight_profile": "weak_high",
        "macro_rmse": 0.3142380276,
        "worst_bearing_rmse": 0.497089,
        "between_bearing_rmse_std": 0.186672,
        "note": "Run 4 independently verified equal-bearing normalized metrics.",
    }
    run5["experiment"] = {
        "id": "EXP-005",
        "name": "IMS baseline-relative cross-bearing normalization",
        "goal": (
            "Test whether causal, per-bearing healthy-baseline normalization reduces "
            "the feature domain shift and cross-bearing RUL error observed in Run 4."
        ),
        "evidence": (
            "EXP-004 completed all 36 jobs but every model had negative macro R2. "
            "Weak-PINN led macro RMSE at 0.314238 while its between-bearing RMSE "
            "standard deviation was 0.186672. The largest train/test shifts involved "
            "high-frequency ratio, spectral centroid, mean absolute value, spectral "
            "entropy, and RMS."
        ),
        "hypothesis": (
            "Expressing each vibration feature relative to a fixed initial healthy "
            "baseline will reduce bearing-specific offsets and scales, improving "
            "Weak-PINN on at least three folds and reducing its macro and worst-bearing "
            "RMSE without worsening late-life bias."
        ),
        "changed_from_previous": [
            "Apply label-free per-bearing robust relative normalization to vibration "
            "features before the unchanged training-only StandardScaler.",
            "Preserve physical bearing identity in run_id and bearing_run_id while "
            "recording the experiment label separately as experiment_run_id.",
        ],
        "held_constant": [
            "IMS feature cache and RUL labels",
            "four Run 4 train/validation/test fold assignments",
            "test population and sequence construction",
            "feature-only representation and feature set",
            "sequence length 8",
            "hidden width 128 and attention heads 4",
            "AdamW learning rate 0.0005, batch size 64, and early stopping",
            "seeds 42, 1042, and 2042",
            "LSTM/data-only and Weak-PINN/high configurations",
            "Run 3 selected Strong-PINN physics weights",
            "normalized equal-bearing metric definitions and best-checkpoint policy",
        ],
        "primary_metric": (
            "Macro mean normalized RMSE: equal weight to each test bearing after "
            "averaging the three seeds within that bearing."
        ),
        "secondary_metrics": [
            "per-bearing MAE, MSE, RMSE, and R2",
            "original-time MAE and RMSE",
            "worst-bearing RMSE",
            "between-seed and between-bearing variation",
            "late-life bias",
            "best-versus-final epoch metrics",
            "training and inference time",
        ],
        "success_criterion": (
            "All 36 fixed jobs must complete with finite reproducible metrics. "
            "Weak-PINN/high must improve on Run 4 macro RMSE 0.314238 and "
            "worst-bearing RMSE 0.497089, improve in at least three of four folds, "
            "reduce between-bearing RMSE variation below 0.186672, and not worsen "
            "absolute late-life bias."
        ),
        "known_risks": [
            "Eight commissioning snapshots may estimate a noisy baseline.",
            "Per-bearing calibration requires an initial healthy observation buffer in "
            "deployment.",
            "Only four IMS trajectories are available, so conclusions remain limited.",
            "The frozen physical parameters retain documented unmeasured assumptions.",
        ],
    }

    encoded = json.dumps(run5, indent=2) + "\n"
    RUN5.write_text(encoded, encoding="utf-8")
    ACTIVE.write_text(encoded, encoding="utf-8")
    print(f"Saved {RUN5.relative_to(ROOT)}")
    print(f"Saved {ACTIVE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
