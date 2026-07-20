from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"Expected block not found in {relative}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def update_multi_dataset() -> None:
    replace_once("src/thesis_work/multi_dataset.py", '    "elapsed_norm",\n', "")
    replace_once("src/thesis_work/multi_dataset.py", '    "health_indicator",\n', "")
    replace_once(
        "src/thesis_work/multi_dataset.py",
        '    "total_cycles",\n',
        '    "cycles_per_time_unit",\n',
    )
    replace_once(
        "src/thesis_work/multi_dataset.py",
        '[float(target["elapsed_norm"])], dtype=torch.float32',
        '[float(target["_time_coordinate"])], dtype=torch.float32',
    )
    replace_once(
        "src/thesis_work/multi_dataset.py",
        "    split_frames: dict[str, pd.DataFrame]\n",
        "    split_frames: dict[str, pd.DataFrame]\n    time_scale_seconds: float\n",
    )
    replace_once(
        "src/thesis_work/multi_dataset.py",
        '''    train_frame, validation_frame, test_frame = split_feature_frame(
        frame, dataset_config["split"]
    )
    scaler = StandardScaler()
''',
        '''    train_frame, validation_frame, test_frame = split_feature_frame(
        frame, dataset_config["split"]
    )
    time_scale_seconds = max(float(train_frame["elapsed_seconds"].max()), 1.0)
    for split_frame in (train_frame, validation_frame, test_frame):
        split_frame["_time_coordinate"] = (
            split_frame["elapsed_seconds"] / time_scale_seconds
        )
        split_frame["cycles_per_time_unit"] = (
            time_scale_seconds * split_frame["speed_rpm"].fillna(0.0) / 60.0
        )
    scaler = StandardScaler()
''',
    )
    replace_once(
        "src/thesis_work/multi_dataset.py",
        '''            "test": test_frame,
        },
    )
''',
        '''            "test": test_frame,
        },
        time_scale_seconds=time_scale_seconds,
    )
''',
    )


def update_sequence_models() -> None:
    replace_once(
        "src/thesis_work/sequence_models.py",
        '''    initial_mask = (time <= 0.05).float()
    terminal_mask = (time >= 0.95).float()
''',
        '''    initial_mask = (target >= 0.95).float()
    terminal_mask = (target <= 0.05).float()
''',
    )
    path = ROOT / "src" / "thesis_work" / "sequence_models.py"
    text = path.read_text(encoding="utf-8")
    if text.count('meta["total_cycles"]') != 2:
        raise RuntimeError("Expected exactly two total_cycles physics references.")
    path.write_text(
        text.replace('meta["total_cycles"]', 'meta["cycles_per_time_unit"]'),
        encoding="utf-8",
    )


def update_experiment_runner() -> None:
    path = ROOT / "src" / "thesis_work" / "experiment_runner.py"
    text = path.read_text(encoding="utf-8")
    start = text.index("def run_dataset_experiment(")
    end = text.index("\ndef _set_nested(", start)
    replacement = '''def _comparison_summary(result: pd.DataFrame) -> pd.DataFrame:
    successful = result[result["status"] == "ok"].copy()
    if successful.empty:
        return pd.DataFrame()
    return (
        successful.groupby(["dataset", "model", "weight_profile"], as_index=False)
        .agg(
            seed_repeats=("seed", "nunique"),
            seconds_mean=("seconds", "mean"),
            seconds_total=("seconds", "sum"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .sort_values(["dataset", "rmse_mean"])
        .reset_index(drop=True)
    )


def run_dataset_experiment(
    prepared: PreparedDataset,
    dataset_config: dict[str, Any],
    config: dict[str, Any],
    output_root: str | Path,
    training_overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, torch.nn.Module]]:
    output_root = Path(output_root)
    dataset_output = output_root / dataset_config["name"]
    dataset_output.mkdir(parents=True, exist_ok=True)
    training = copy.deepcopy(config["training"])
    if training_overrides:
        training.update(training_overrides)

    rows: list[dict[str, Any]] = []
    trained_models: dict[str, torch.nn.Module] = {}
    seed_base = int(config.get("seed", 42))
    repeats = int(training.get("seed_repeats", 1))
    seed_stride = int(training.get("seed_stride", 1000))
    if repeats < 1:
        raise ValueError("seed_repeats must be at least one.")

    for model_index, (model_name, model_config) in enumerate(
        config["models"].items(), start=1
    ):
        if not model_config.get("enabled", True):
            continue
        for profile_name, weights in _weight_profiles_for_model(
            model_name, model_config, config
        ):
            for repeat_index in range(repeats):
                repeat_number = repeat_index + 1
                seed = seed_base + model_index + repeat_index * seed_stride
                label = (
                    f"{model_name}__{profile_name}__seed_{repeat_number:02d}"
                )
                artifact_dir = dataset_output / label
                started = time.time()
                try:
                    model, _, _, metrics = train_one_model(
                        prepared=prepared,
                        model_name=model_name,
                        model_config=model_config,
                        weights=weights,
                        physics=config["physics"],
                        training_config=training,
                        seed=seed,
                        artifact_dir=artifact_dir,
                    )
                    trained_models[label] = model
                    rows.append(
                        {
                            "dataset": dataset_config["name"],
                            "model": model_name,
                            "weight_profile": profile_name,
                            "seed_repeat": repeat_number,
                            "seed": seed,
                            "status": "ok",
                            "seconds": time.time() - started,
                            **metrics,
                        }
                    )
                except Exception as exc:
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    (artifact_dir / "failure.txt").write_text(
                        traceback.format_exc(), encoding="utf-8"
                    )
                    rows.append(
                        {
                            "dataset": dataset_config["name"],
                            "model": model_name,
                            "weight_profile": profile_name,
                            "seed_repeat": repeat_number,
                            "seed": seed,
                            "status": "failed",
                            "seconds": time.time() - started,
                            "error": str(exc),
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(dataset_output / "model_comparison.csv", index=False)
    _comparison_summary(result).to_csv(
        dataset_output / "model_comparison_summary.csv", index=False
    )
    return result, trained_models

'''
    text = text[:start] + replacement + text[end + 1 :]
    old_seed = 'seed=int(config.get("seed", 42)) + 10_000 + trial,'
    if old_seed not in text:
        raise RuntimeError("Sensitivity seed expression was not found.")
    text = text.replace(
        old_seed,
        'seed=int(config.get("seed", 42)) + 10_000,',
        1,
    )
    old_root = '''    all_results.to_csv(output_root / "all_model_comparisons.csv", index=False)
    all_sensitivity.to_csv(output_root / "all_sensitivity_results.csv", index=False)
'''
    new_root = '''    all_results.to_csv(output_root / "all_model_comparisons.csv", index=False)
    _comparison_summary(all_results).to_csv(
        output_root / "all_model_comparisons_summary.csv", index=False
    )
    all_sensitivity.to_csv(output_root / "all_sensitivity_results.csv", index=False)
'''
    if old_root not in text:
        raise RuntimeError("Root result write block was not found.")
    path.write_text(text.replace(old_root, new_root, 1), encoding="utf-8")


def update_config() -> None:
    path = ROOT / "configs" / "colab_experiments.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["run_label"] = "run_02"
    config["training"].update(
        {
            "sequence_length": 8,
            "epochs": 300,
            "patience": 40,
            "batch_size": 64,
            "seed_repeats": 3,
            "seed_stride": 1000,
        }
    )
    config["weight_profiles"]["strong_low"].update(
        {
            "paris_crack_growth": 0.0001,
            "palmgren_miner": 0.001,
            "crack_rate_positive": 0.001,
        }
    )
    config["weight_profiles"]["strong_medium"].update(
        {
            "paris_crack_growth": 0.001,
            "palmgren_miner": 0.01,
            "crack_rate_positive": 0.01,
        }
    )
    config["weight_profiles"]["strong_high"].update(
        {
            "paris_crack_growth": 0.01,
            "palmgren_miner": 0.1,
            "crack_rate_positive": 0.05,
        }
    )
    sensitivity = config["sensitivity"]
    sensitivity.update({"enabled": True, "epochs": 80, "patience": 15})
    for parameter in sensitivity["parameters"]:
        if parameter["path"].endswith("paris_crack_growth"):
            parameter["values"] = [0.0001, 0.001, 0.01]
        elif parameter["path"].endswith("palmgren_miner"):
            parameter["values"] = [0.001, 0.01, 0.1]
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def update_notebook_builder() -> None:
    replace_once(
        "scripts/build_colab_notebook.py",
        'OUTPUT_DIR = DRIVE_ROOT / "experiment_outputs"',
        'OUTPUT_DIR = DRIVE_ROOT / "experiment_outputs_run_02"',
    )
    replace_once(
        "scripts/build_colab_notebook.py",
        'config["training"].update({"epochs": 3, "patience": 2, "batch_size": 64})',
        'config["training"].update({"epochs": 3, "patience": 2, "batch_size": 64, "seed_repeats": 1})',
    )
    replace_once(
        "scripts/build_colab_notebook.py",
        'sns.barplot(data=ordered, x="model_profile", y="rmse", color="#4472C4")',
        'sns.barplot(data=ordered, x="model_profile", y="rmse", color="#4472C4", errorbar="sd")',
    )
    replace_once(
        "scripts/build_colab_notebook.py",
        '''    display(sensitivity_results.sort_values(["parameter", "value"]))
''',
        '''    sensitivity_results.to_csv(
        OUTPUT_DIR / "all_sensitivity_results.csv", index=False
    )
    display(sensitivity_results.sort_values(["parameter", "value"]))
''',
    )


def update_upload_instructions() -> None:
    path = ROOT / "UPLOAD_INSTRUCTIONS.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "`MyDrive/Upload/experiment_outputs`",
        "`MyDrive/Upload/experiment_outputs_run_02`",
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    update_multi_dataset()
    update_sequence_models()
    update_experiment_runner()
    update_config()
    update_notebook_builder()
    update_upload_instructions()
    print("Run 2 fixes applied.")
