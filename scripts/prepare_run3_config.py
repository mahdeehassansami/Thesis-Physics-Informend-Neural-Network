from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "configs" / "colab_experiments.json"
RUN2 = ROOT / "configs" / "colab_experiments_run_02.json"
RUN3 = ROOT / "configs" / "colab_experiments_run_03.json"


run2 = json.loads(ACTIVE.read_text(encoding="utf-8"))
if run2.get("run_label") != "run_02":
    raise RuntimeError("The active source configuration is not Run 2.")
run2.pop("_config_path", None)
RUN2.write_text(json.dumps(run2, indent=2) + "\n", encoding="utf-8")

run3 = copy.deepcopy(run2)
run3["schema_version"] = 2
run3["experiment"] = {
    "id": "EXP-003",
    "name": "IMS validation-only explicit-physics calibration",
    "goal": (
        "Calibrate Paris and Palmgren-Miner loss weights without using the "
        "test split for model selection."
    ),
    "evidence": (
        "Run 2 found weak-PINN/high best on IMS, while the one-factor "
        "Strong-PINN sweep showed large sensitivity to Paris and Miner weights."
    ),
    "hypothesis": (
        "A validation-selected Paris/Miner combination can improve Strong-PINN "
        "generalization without overwhelming the data loss."
    ),
    "held_constant": [
        "Run 2 IMS run-level split",
        "Run 2 feature extraction and training-only scaling",
        "sequence length 8",
        "hidden width 128 and attention heads 4",
        "AdamW learning rate 0.0005 and batch size 64",
        "three-seed evaluation",
    ],
    "primary_selection_metric": "mean validation RMSE across three seeds",
    "success_criterion": (
        "The frozen validation-selected Strong-PINN should improve mean test "
        "RMSE over Run 2 Strong-PINN/low and remain numerically stable across seeds."
    ),
}
run3["runtime"] = {
    "require_cuda": True,
    "required_gpu_name_contains": "T4",
}
for dataset in run3["datasets"]:
    dataset["enabled"] = dataset["name"] == "ims"

run3["training"].update(
    {
        "sequence_length": 8,
        "epochs": 300,
        "patience": 40,
        "batch_size": 64,
        "seed_repeats": 3,
        "seeds": [42, 1042, 2042],
        "seed_stride": 1000,
        "gradient_diagnostics_interval": 10,
    }
)
for name, model in run3["models"].items():
    model["enabled"] = name in {"lstm", "weak_pinn"}
run3["models"]["lstm"]["profiles"] = ["data_only"]
run3["models"]["weak_pinn"]["profiles"] = ["weak_high"]
run3["models"]["strong_pinn"]["enabled"] = False
run3["models"]["strong_pinn"]["profiles"] = []

run3["sensitivity"]["enabled"] = False
run3["calibration"] = {
    "enabled": True,
    "dataset": "ims",
    "model": "strong_pinn",
    "base_weight_profile": "strong_medium",
    "paris_weights": [0.003, 0.01, 0.03],
    "miner_weights": [0.0003, 0.001, 0.003],
    "selection_split": "validation",
    "selection_metric": "rmse",
    "test_policy": (
        "Train and rank all 27 candidate/seed runs on validation only; evaluate "
        "the test split only for the frozen winning profile."
    ),
}
run3["run_label"] = "run_03"
run3.pop("_config_path", None)

serialized = json.dumps(run3, indent=2) + "\n"
RUN3.write_text(serialized, encoding="utf-8")
ACTIVE.write_text(serialized, encoding="utf-8")
print(f"Saved {RUN2.relative_to(ROOT)}")
print(f"Saved {RUN3.relative_to(ROOT)}")
print(f"Activated {ACTIVE.relative_to(ROOT)}")
