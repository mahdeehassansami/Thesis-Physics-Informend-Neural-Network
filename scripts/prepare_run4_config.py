from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN3 = ROOT / "configs" / "colab_experiments_run_03.json"
RUN4 = ROOT / "configs" / "colab_experiments_run_04.json"
ACTIVE = ROOT / "configs" / "colab_experiments.json"

RUNS = [
    "ims_ds1_b3",
    "ims_ds1_b4",
    "ims_ds2_b1",
    "ims_ds3_b3",
]
FOLDS = [
    {
        "fold_id": "fold_01_test_ims_ds1_b3",
        "train_runs": ["ims_ds2_b1", "ims_ds3_b3"],
        "validation_runs": ["ims_ds1_b4"],
        "test_runs": ["ims_ds1_b3"],
    },
    {
        "fold_id": "fold_02_test_ims_ds1_b4",
        "train_runs": ["ims_ds1_b3", "ims_ds2_b1"],
        "validation_runs": ["ims_ds3_b3"],
        "test_runs": ["ims_ds1_b4"],
    },
    {
        "fold_id": "fold_03_test_ims_ds2_b1",
        "train_runs": ["ims_ds1_b4", "ims_ds3_b3"],
        "validation_runs": ["ims_ds1_b3"],
        "test_runs": ["ims_ds2_b1"],
        "note": "Exact Run 3 train/validation/test assignment.",
    },
    {
        "fold_id": "fold_04_test_ims_ds3_b3",
        "train_runs": ["ims_ds1_b3", "ims_ds1_b4"],
        "validation_runs": ["ims_ds2_b1"],
        "test_runs": ["ims_ds3_b3"],
    },
]


def main() -> None:
    config = json.loads(RUN3.read_text(encoding="utf-8"))
    if config.get("run_label") != "run_03":
        raise ValueError("Expected the preserved Run 3 configuration.")

    run4 = copy.deepcopy(config)
    run4["schema_version"] = 3
    run4["run_label"] = "run_04"
    run4["repository"] = {
        "url": (
            "https://github.com/mahdeehassansami/"
            "Thesis-Physics-Informend-Neural-Network.git"
        ),
        "expected_commit": None,
        "require_clean_checkout": True,
        "note": (
            "The Colab notebook requires a 40-character commit SHA and checks out that "
            "exact revision under /content before importing thesis_work."
        ),
    }

    for dataset in run4["datasets"]:
        dataset["enabled"] = dataset["name"] == "ims"
        if dataset["name"] == "ims":
            dataset["split"] = {
                "strategy": "configured_cross_bearing_folds",
                "folds_reference": "cross_bearing.folds",
            }

    for model_name, model in run4["models"].items():
        model["enabled"] = model_name in {"lstm", "weak_pinn", "strong_pinn"}
    run4["models"]["lstm"]["profiles"] = ["data_only"]
    run4["models"]["weak_pinn"]["profiles"] = ["weak_high"]
    run4["models"]["strong_pinn"]["profiles"] = [
        "strong_paris_0p003_miner_0p003"
    ]
    run4["weight_profiles"]["strong_paris_0p003_miner_0p003"] = {
        "data": 1.0,
        "monotonic": 0.1,
        "health_indicator": 0.05,
        "boundary": 0.05,
        "temperature_prior": 0.02,
        "paris_crack_growth": 0.003,
        "palmgren_miner": 0.003,
        "crack_rate_positive": 0.01,
    }

    run4["training"].update(
        {
            "seeds": [42, 1042, 2042],
            "seed_repeats": 3,
            "resume": True,
            "oom_policy": "fail_and_record",
            "mixed_precision": False,
            "mixed_precision_rationale": (
                "Run 4 preserves Run 3 full-float training; physics derivatives remain "
                "in FP32 and numerical conditions are not changed."
            ),
            "save_final_evaluation": True,
        }
    )
    run4["sensitivity"]["enabled"] = False
    run4.pop("calibration", None)
    run4["runtime"] = {
        "require_cuda": True,
        "required_gpu_name_contains": "T4",
        "require_expected_commit": True,
        "require_clean_git": True,
    }
    run4["cross_bearing"] = {
        "enabled": True,
        "dataset": "ims",
        "all_runs": RUNS,
        "folds": FOLDS,
        "validation_policy": (
            "Fixed balanced assignment: every trajectory is test once and validation "
            "once; the remaining two trajectories train. Validation is used only for "
            "scheduler and early stopping."
        ),
        "test_policy": (
            "All model configurations and weights are frozen before any fold runs. "
            "Test-fold metrics are never used to change a model within EXP-004."
        ),
        "primary_aggregation": (
            "Macro mean of the four per-bearing, three-seed mean normalized RMSE values."
        ),
        "expected_feature_cache_sha256": (
            "07ede6448e8df30d1c6ad8284647ed09db22d6c59a758ab6d13043e55665f1cb"
        ),
        "frozen_strong_profile": {
            "name": "strong_paris_0p003_miner_0p003",
            "source_experiment": "EXP-003",
            "source_run": "run_03",
            "selection_split": "ims_ds1_b3",
            "paris_weight": 0.003,
            "miner_weight": 0.003,
        },
        "expected_jobs": 36,
    }
    run4["experiment"] = {
        "id": "EXP-004",
        "name": "IMS fixed-model cross-bearing robustness",
        "goal": (
            "Measure how the Run 3 model ranking changes when each of the four IMS "
            "trajectories is held out as test once."
        ),
        "evidence": (
            "Run 3 LSTM achieved mean test RMSE 0.144667, but the validation-selected "
            "Strong-PINN had a 9.7x validation/test gap and collapsed toward high RUL. "
            "Validation/test feature shifts indicate that one split is insufficient."
        ),
        "hypothesis": (
            "The Run 3 ranking and Strong-PINN failure are materially split-dependent; "
            "balanced held-out-bearing folds will reveal cross-bearing variance."
        ),
        "changed_from_previous": [
            "Replace the single IMS split with four predeclared held-out-bearing folds.",
            "Enable only the frozen Run 3 Strong-PINN winner instead of recalibrating.",
            "Record best-checkpoint and final-epoch test evidence for every fixed job.",
        ],
        "held_constant": [
            "IMS feature cache and RUL labels",
            "training-only StandardScaler fitting within each fold",
            "feature-only representation",
            "sequence length 8",
            "hidden width 128 and attention heads 4",
            "AdamW learning rate 0.0005, batch size 64, and early stopping",
            "seeds 42, 1042, and 2042",
            "LSTM/data-only and Weak-PINN/high configurations",
            "Run 3 selected Strong-PINN physics weights",
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
            "All 36 fixed jobs must complete with reproducible finite metrics. The Run 3 "
            "ranking is considered robust only if LSTM has the lowest macro RMSE and "
            "beats both PINNs in at least three of four held-out bearings. Strong-PINN "
            "portability additionally requires absolute late-life bias below 0.25 in at "
            "least three folds."
        ),
        "known_risks": [
            "Only four IMS trajectories are available, so each fold trains on two runs.",
            "Run durations differ, making normalized RMSE the primary cross-fold metric.",
            "The frozen physical parameters include documented unmeasured assumptions.",
            "Cross-validation uses all four trajectories as test across folds and is not "
            "an external-dataset validation.",
        ],
    }

    encoded = json.dumps(run4, indent=2) + "\n"
    RUN4.write_text(encoded, encoding="utf-8")
    ACTIVE.write_text(encoded, encoding="utf-8")
    print(f"Saved {RUN4.relative_to(ROOT)}")
    print(f"Saved {ACTIVE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
