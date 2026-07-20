from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pandas as pd

from thesis_work.experiment_runner import run_dataset_experiment
from thesis_work.multi_dataset import load_or_extract_dataset, prepare_sequence_dataset


ROOT = Path(__file__).resolve().parents[1]
UPLOAD = ROOT / "Upload"


def main() -> None:
    config = json.loads(
        (UPLOAD / "configs" / "colab_experiments.json").read_text(encoding="utf-8")
    )
    dataset_config = next(
        dataset
        for dataset in config["datasets"]
        if dataset["name"] == "kaist_vibration_temperature"
    )
    frame = load_or_extract_dataset(
        dataset_config,
        project_root=UPLOAD,
        cache_dir=UPLOAD / "feature_cache",
    )
    prepared = prepare_sequence_dataset(
        frame,
        dataset_config,
        sequence_length=config["training"]["sequence_length"],
    )

    smoke = copy.deepcopy(config)
    for name, model in smoke["models"].items():
        model["enabled"] = name in {"fnn", "strong_pinn"}
    smoke["models"]["strong_pinn"]["profiles"] = [
        "strong_low",
        "strong_medium",
    ]
    smoke["training"].update(
        {
            "epochs": 2,
            "patience": 1,
            "batch_size": 256,
            "seed_repeats": 2,
        }
    )

    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
        result, _ = run_dataset_experiment(
            prepared,
            dataset_config,
            smoke,
            output_root=directory,
        )
        if len(result) != 6:
            raise AssertionError(f"Expected six smoke runs, got {len(result)}")
        strong = result[result["model"] == "strong_pinn"]
        seed_sets = strong.groupby("weight_profile")["seed"].apply(set)
        if len({tuple(sorted(values)) for values in seed_sets}) != 1:
            raise AssertionError("Strong-PINN profiles did not use matching seeds.")
        summary = pd.read_csv(
            Path(directory)
            / dataset_config["name"]
            / "model_comparison_summary.csv"
        )
        if not (summary["seed_repeats"] == 2).all():
            raise AssertionError("Run 2 summary did not aggregate both seeds.")
    print("Run 2 repeated-seed runner validation passed.")


if __name__ == "__main__":
    main()
