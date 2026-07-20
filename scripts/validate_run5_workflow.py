from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

from thesis_work.multi_dataset import (
    MODEL_FEATURES,
    SIGNAL_FEATURES,
    load_or_extract_dataset,
    prepare_sequence_dataset,
)
from thesis_work.sequence_models import build_model, calculate_loss


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "configs" / "colab_experiments.json"
RUN4 = ROOT / "configs" / "colab_experiments_run_04.json"
NOTEBOOK = ROOT / "notebooks" / "train_models_colab.ipynb"
CACHE_DIR = ROOT / "data" / "processed_features" / "colab"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads(ACTIVE.read_text(encoding="utf-8"))
    run4 = json.loads(RUN4.read_text(encoding="utf-8"))
    assert config["experiment"]["id"] == "EXP-005"
    assert config["run_label"] == "run_05"
    assert config["training"] == run4["training"]
    assert config["models"] == run4["models"]
    assert config["weight_profiles"] == run4["weight_profiles"]
    assert config["physics"] == run4["physics"]
    assert config["cross_bearing"]["folds"] == run4["cross_bearing"]["folds"]
    assert config["cross_bearing"]["expected_jobs"] == 36

    preprocessing = config["preprocessing"]
    sequence_length = int(config["training"]["sequence_length"])
    assert preprocessing["prefix_samples"] == sequence_length == 8
    assert preprocessing["uses_targets"] is False
    assert preprocessing["uses_failure_time"] is False

    dataset = next(item for item in config["datasets"] if item.get("enabled"))
    assert dataset["name"] == "ims"
    cache_path = CACHE_DIR / "ims_features.csv"
    assert file_sha256(cache_path) == config["cross_bearing"][
        "expected_feature_cache_sha256"
    ]
    frame = load_or_extract_dataset(
        dataset, project_root=ROOT, cache_dir=CACHE_DIR, refresh=False
    )

    expected_runs = set(config["cross_bearing"]["all_runs"])
    run4_shift_values: list[float] = []
    run5_shift_values: list[float] = []
    for fold in config["cross_bearing"]["folds"]:
        fold_dataset = json.loads(json.dumps(dataset))
        fold_dataset["split"] = {
            "strategy": "run_ids",
            "train_runs": fold["train_runs"],
            "validation_runs": fold["validation_runs"],
            "test_runs": fold["test_runs"],
        }
        prepared = prepare_sequence_dataset(
            frame,
            fold_dataset,
            sequence_length,
            preprocessing_config=preprocessing,
        )
        metadata = prepared.preprocessing_metadata
        assert metadata["strategy"] == "per_run_initial_robust_relative"
        assert set(metadata["run_statistics"]) == expected_runs
        assert all(
            len(record["prefix_sample_indices"]) == sequence_length
            for record in metadata["run_statistics"].values()
        )
        assert len(prepared.train) > 0
        assert len(prepared.validation) > 0
        assert len(prepared.test) > 0
        unnormalized = prepare_sequence_dataset(
            frame,
            fold_dataset,
            sequence_length,
            preprocessing_config=None,
        )
        assert len(prepared.train) == len(unnormalized.train)
        assert len(prepared.validation) == len(unnormalized.validation)
        assert len(prepared.test) == len(unnormalized.test)
        raw_train = frame[frame.run_id.isin(fold["train_runs"])]
        raw_test = frame[frame.run_id.isin(fold["test_runs"])]
        raw_scaler = StandardScaler().fit(raw_train[MODEL_FEATURES])
        raw_train_scaled = raw_scaler.transform(raw_train[MODEL_FEATURES])
        raw_test_scaled = raw_scaler.transform(raw_test[MODEL_FEATURES])
        transformed_train = prepared.split_frames["train"]
        transformed_test = prepared.split_frames["test"]
        transformed_train_scaled = prepared.scaler.transform(
            transformed_train[MODEL_FEATURES]
        )
        transformed_test_scaled = prepared.scaler.transform(
            transformed_test[MODEL_FEATURES]
        )
        fold_run4_shifts = []
        fold_run5_shifts = []
        for feature in SIGNAL_FEATURES:
            feature_index = MODEL_FEATURES.index(feature)
            fold_run4_shifts.append(
                wasserstein_distance(
                    raw_train_scaled[:, feature_index],
                    raw_test_scaled[:, feature_index],
                )
            )
            fold_run5_shifts.append(
                wasserstein_distance(
                    transformed_train_scaled[:, feature_index],
                    transformed_test_scaled[:, feature_index],
                )
            )
        run4_fold_mean = float(np.mean(fold_run4_shifts))
        run5_fold_mean = float(np.mean(fold_run5_shifts))
        assert run5_fold_mean < run4_fold_mean
        run4_shift_values.extend(fold_run4_shifts)
        run5_shift_values.extend(fold_run5_shifts)
        for split in (prepared.train, prepared.validation, prepared.test):
            sample = split[0]
            assert torch.isfinite(sample["x"]).all()
            assert sample["x"].shape == (sequence_length, len(prepared.feature_columns))

    first_fold = config["cross_bearing"]["folds"][0]
    fold_dataset = json.loads(json.dumps(dataset))
    fold_dataset["split"] = {
        "strategy": "run_ids",
        "train_runs": first_fold["train_runs"],
        "validation_runs": first_fold["validation_runs"],
        "test_runs": first_fold["test_runs"],
    }
    prepared = prepare_sequence_dataset(
        frame,
        fold_dataset,
        sequence_length,
        preprocessing_config=preprocessing,
    )
    batch = torch.utils.data.default_collate([prepared.train[index] for index in range(2)])
    feature_indices = {
        name: index for index, name in enumerate(prepared.feature_columns)
    }
    for model_name in ("lstm", "weak_pinn", "strong_pinn"):
        model_config = config["models"][model_name]
        profile = model_config["profiles"][0]
        model = build_model(
            model_name,
            input_dim=len(prepared.feature_columns),
            model_config=model_config,
        )
        total, _, prediction = calculate_loss(
            model,
            batch,
            config["weight_profiles"][profile],
            feature_indices,
            config["physics"],
        )
        assert torch.isfinite(total)
        assert prediction.shape == batch["target"].shape

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert notebook["nbformat"] == 4
    assert "EXP-005" in source
    assert "run_run5_experiment" in source
    assert "experiment_outputs_run_05" in source
    assert "PASTE_40_CHARACTER_COMMIT_SHA" in source
    assert "nvidia-smi" in source
    assert "class " not in "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    finite_features = frame[preprocessing["feature_columns"]].to_numpy(dtype=float)
    assert np.isfinite(finite_features).all()
    run4_mean_shift = float(np.mean(run4_shift_values))
    run5_mean_shift = float(np.mean(run5_shift_values))
    reduction = 1.0 - run5_mean_shift / run4_mean_shift
    assert reduction > 0.35
    print(
        "Label-free mean feature shift: "
        f"{run4_mean_shift:.4f} -> {run5_mean_shift:.4f} "
        f"({reduction:.1%} reduction)."
    )
    print("Run 5 workflow validation passed without model training.")


if __name__ == "__main__":
    main()
