from __future__ import annotations

import copy
import json
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from thesis_work.experiment_runner import train_one_model
from thesis_work.multi_dataset import (
    enabled_dataset_configs,
    load_or_extract_dataset,
    prepare_sequence_dataset,
)
from thesis_work.run3_calibration import (
    _best_epoch_physics_diagnostics,
    _environment,
    _git_state,
    _sha256,
    _training_seeds,
)
from thesis_work.run4_cross_bearing import (
    _aggregate,
    _annotate,
    _fold_summary,
    _folds,
    _hash,
    _lifecycle,
    _models,
    _source,
    _utc_now,
    finalize_run4_artifacts,
)


def finalize_run5_artifacts(root: str | Path) -> Path:
    """Refresh the Run 5 artifact inventory and lightweight result bundle."""

    return finalize_run4_artifacts(Path(root))


def validate_run5_runtime(
    config: dict[str, Any], project_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    environment = _environment()
    runtime = config.get("runtime", {})
    if runtime.get("require_cuda", True) and not environment["cuda_available"]:
        raise RuntimeError("Run 5 requires a CUDA GPU; select a Colab T4 runtime.")
    required_name = runtime.get("required_gpu_name_contains")
    if required_name and required_name.lower() not in str(
        environment.get("gpu_name", "")
    ).lower():
        raise RuntimeError(
            f"Run 5 requires {required_name}; detected {environment.get('gpu_name')}."
        )
    if environment["cuda_available"]:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        environment.update(
            {
                "selected_device": "cuda:0",
                "gpu_memory_free_bytes_at_validation": int(free_bytes),
                "gpu_memory_total_bytes_at_validation": int(total_bytes),
            }
        )
    else:
        environment["selected_device"] = "cpu"

    git = _git_state(Path(project_root))
    expected_commit = config.get("repository", {}).get("expected_commit")
    if runtime.get("require_expected_commit", True):
        if not isinstance(expected_commit, str) or len(expected_commit) != 40:
            raise RuntimeError("Set repository.expected_commit to the committed SHA.")
        if git["commit"] != expected_commit:
            raise RuntimeError(
                f"Expected commit {expected_commit}, detected {git['commit']}."
            )
    if runtime.get("require_clean_git", True) and git.get("dirty") is not False:
        raise RuntimeError("Run 5 requires a clean Git checkout.")
    return environment, git


def run_run5_experiment(
    config: dict[str, Any],
    project_root: str | Path,
    cache_dir: str | Path,
    output_root: str | Path,
    refresh_features: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the frozen Run 4 comparison with only Run 5 preprocessing changed."""

    project_root = Path(project_root).resolve()
    cache_dir = Path(cache_dir).resolve()
    output_root = Path(output_root).resolve()
    if config.get("run_label") != "run_05" or config["experiment"]["id"] != "EXP-005":
        raise ValueError("Run 5 requires EXP-005/run_05.")

    enabled = enabled_dataset_configs(config)
    if [dataset["name"] for dataset in enabled] != ["ims"]:
        raise ValueError("Run 5 must enable IMS only.")
    environment, git = validate_run5_runtime(config, project_root)
    dataset = enabled[0]
    models = _models(config)
    seeds = _training_seeds(config)
    preprocessing = copy.deepcopy(config["preprocessing"])
    sequence_length = int(config["training"]["sequence_length"])
    if preprocessing.get("prefix_samples") != sequence_length:
        raise ValueError(
            "Run 5 requires prefix_samples to equal the frozen sequence length."
        )

    frame = load_or_extract_dataset(
        dataset, project_root, cache_dir, refresh_features
    )
    cache = cache_dir / "ims_features.csv"
    cache_hash = _sha256(cache)
    if cache_hash != config["cross_bearing"]["expected_feature_cache_sha256"]:
        raise RuntimeError("IMS feature-cache hash mismatch.")
    folds = _folds(config, dataset, frame)
    split = {
        "dataset": "ims",
        "strategy": "fixed_cross_bearing_folds",
        "folds": folds,
    }
    clean_config = copy.deepcopy(config)
    clean_config.pop("_config_path", None)
    config_hash = _hash(clean_config)
    split_hash = _hash(split)
    preprocessing_hash = _hash(preprocessing)
    source, source_hash = _source(project_root)
    identity = {
        "experiment_id": "EXP-005",
        "experiment_run_id": "run_05",
        "git_commit": git["commit"],
        "source_tree_sha256": source_hash,
        "config_hash": config_hash,
        "split_hash": split_hash,
        "preprocessing_hash": preprocessing_hash,
        "cache_hash": cache_hash,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "run_state.json"
    if any(output_root.iterdir()):
        if not state_path.exists():
            raise FileExistsError("Non-empty Run 5 output lacks run_state.json.")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise RuntimeError("Run 5 resume identity mismatch.")
        if (output_root / "run_manifest.json").exists():
            return (
                pd.read_csv(output_root / "all_model_comparisons.csv"),
                pd.read_csv(output_root / "fold_model_summary.csv"),
                pd.read_csv(output_root / "all_model_comparisons_summary.csv"),
            )
    else:
        state = {
            "identity": identity,
            "started_utc": _utc_now(),
            "jobs_recorded": 0,
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    (output_root / "resolved_config.json").write_text(
        json.dumps(clean_config, indent=2), encoding="utf-8"
    )
    (output_root / "data_split.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8"
    )
    source.to_csv(output_root / "source_manifest.csv", index=False)
    (output_root / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    (output_root / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n",
        encoding="utf-8",
    )
    (output_root / "git_commit.txt").write_text(
        git["commit"] + "\n", encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    fold_info: list[dict[str, Any]] = []
    log_path = output_root / "training.log"
    for fold in folds:
        fold_dataset = copy.deepcopy(dataset)
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
        fold_root = output_root / "ims" / "folds" / fold["fold_id"]
        fold_root.mkdir(parents=True, exist_ok=True)
        preprocessing_record = {
            "baseline_relative_transform": prepared.preprocessing_metadata,
            "post_transform_scaler": {
                "fit_split": "train",
                "train_runs": fold["train_runs"],
                "validation_runs": fold["validation_runs"],
                "test_runs": fold["test_runs"],
                "feature_columns": prepared.feature_columns,
                "scaler_mean": prepared.scaler.mean_.tolist(),
                "scaler_scale": prepared.scaler.scale_.tolist(),
            },
            "sequence_length": sequence_length,
            "time_scale_seconds": prepared.time_scale_seconds,
        }
        (fold_root / "preprocessing.json").write_text(
            json.dumps(preprocessing_record, indent=2), encoding="utf-8"
        )
        fold_info.append(
            {
                "fold_id": fold["fold_id"],
                "train_runs": fold["train_runs"],
                "validation_runs": fold["validation_runs"],
                "test_runs": fold["test_runs"],
                "train_sequences": len(prepared.train),
                "validation_sequences": len(prepared.validation),
                "test_sequences": len(prepared.test),
                "time_scale_seconds": prepared.time_scale_seconds,
                "preprocessing_sha256": _sha256(fold_root / "preprocessing.json"),
            }
        )

        for model_name, profile, model_config, weights in models:
            for repeat, seed in enumerate(seeds, 1):
                artifact_dir = fold_root / (
                    f"{model_name}__{profile}__seed_{repeat:02d}"
                )
                artifact_dir.mkdir(parents=True, exist_ok=True)
                job_identity = {
                    "fold_id": fold["fold_id"],
                    "model": model_name,
                    "profile": profile,
                    "seed_repeat": repeat,
                    "seed": seed,
                    "identity": identity,
                }
                job_hash = _hash(job_identity)
                job_path = artifact_dir / "job_result.json"
                if job_path.exists():
                    saved = json.loads(job_path.read_text(encoding="utf-8"))
                    if saved.get("job_hash") != job_hash:
                        raise RuntimeError("Resume job identity mismatch.")
                    rows.append(saved["result"])
                    continue

                started = time.time()
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(
                        f"{_utc_now()} START {fold['fold_id']} "
                        f"{model_name} seed={seed}\n"
                    )
                try:
                    network, history, _, metrics = train_one_model(
                        prepared,
                        model_name,
                        model_config,
                        weights,
                        config["physics"],
                        config["training"],
                        seed,
                        artifact_dir,
                        evaluation_split="test",
                        save_final_evaluation=True,
                    )
                    best = _annotate(
                        artifact_dir / "predictions.csv",
                        config,
                        fold,
                        model_name,
                        profile,
                        repeat,
                        seed,
                        "best_validation",
                    )
                    for filename, role in [
                        ("validation_predictions.csv", "best_validation"),
                        ("final_predictions.csv", "final_epoch"),
                        ("final_validation_predictions.csv", "final_epoch"),
                    ]:
                        path = artifact_dir / filename
                        if path.exists():
                            _annotate(
                                path,
                                config,
                                fold,
                                model_name,
                                profile,
                                repeat,
                                seed,
                                role,
                            )
                    lifecycle = _lifecycle(best)
                    lifecycle.to_csv(
                        artifact_dir / "lifecycle_metrics.csv", index=False
                    )
                    late = lifecycle[lifecycle.phase == "late"].iloc[0]
                    row = {
                        "dataset": "ims",
                        "fold_id": fold["fold_id"],
                        "train_run_ids": "|".join(fold["train_runs"]),
                        "validation_run_id": fold["validation_runs"][0],
                        "test_run_id": fold["test_runs"][0],
                        "model": model_name,
                        "weight_profile": profile,
                        "seed_repeat": repeat,
                        "seed": seed,
                        "status": "ok",
                        "seconds": time.time() - started,
                        "late_life_mae": float(late.mae),
                        "late_life_bias": float(late.bias),
                        "artifact_directory": artifact_dir.relative_to(
                            output_root
                        ).as_posix(),
                        **metrics,
                        **_best_epoch_physics_diagnostics(history, weights),
                    }
                    del network
                except Exception as exc:
                    (artifact_dir / "failure.txt").write_text(
                        traceback.format_exc(), encoding="utf-8"
                    )
                    row = {
                        "dataset": "ims",
                        "fold_id": fold["fold_id"],
                        "train_run_ids": "|".join(fold["train_runs"]),
                        "validation_run_id": fold["validation_runs"][0],
                        "test_run_id": fold["test_runs"][0],
                        "model": model_name,
                        "weight_profile": profile,
                        "seed_repeat": repeat,
                        "seed": seed,
                        "status": "failed",
                        "seconds": time.time() - started,
                        "error": str(exc),
                        "artifact_directory": artifact_dir.relative_to(
                            output_root
                        ).as_posix(),
                    }
                job_path.write_text(
                    json.dumps({"job_hash": job_hash, "result": row}, indent=2),
                    encoding="utf-8",
                )
                rows.append(row)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(
                        f"{_utc_now()} {row['status'].upper()} {fold['fold_id']} "
                        f"{model_name} seed={seed}\n"
                    )
                pd.DataFrame(rows).to_csv(
                    output_root / "partial_model_comparisons.csv", index=False
                )
                state["jobs_recorded"] = len(rows)
                state["last_updated_utc"] = _utc_now()
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        fold_results = pd.DataFrame(
            [row for row in rows if row["fold_id"] == fold["fold_id"]]
        )
        fold_results.to_csv(fold_root / "model_comparison.csv", index=False)
        _fold_summary(fold_results).to_csv(
            fold_root / "model_comparison_summary.csv", index=False
        )

    results = pd.DataFrame(rows)
    fold_summary = _fold_summary(results)
    aggregate = _aggregate(fold_summary)
    results.to_csv(output_root / "all_model_comparisons.csv", index=False)
    fold_summary.to_csv(output_root / "fold_model_summary.csv", index=False)
    aggregate.to_csv(
        output_root / "all_model_comparisons_summary.csv", index=False
    )
    failures = results[results.status != "ok"].to_dict(orient="records")
    failure_files = [
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("failure.txt")
    ]
    (output_root / "failure_report.json").write_text(
        json.dumps(
            {"failed_jobs": failures, "failure_files": failure_files}, indent=2
        ),
        encoding="utf-8",
    )
    (output_root / "dataset_summary.json").write_text(
        json.dumps(
            {
                "dataset": "ims",
                "feature_cache_sha256": cache_hash,
                "feature_rows": len(frame),
                "run_ids": sorted(frame.run_id.unique()),
                "preprocessing": preprocessing,
                "folds": fold_info,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_root / "ims" / "assumptions.json").write_text(
        json.dumps(
            {
                "physics_assumptions": dataset.get("physics_assumptions", []),
                "operating_conditions": dataset.get("operating_conditions", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    expected = int(config["cross_bearing"]["expected_jobs"])
    completed = int((results.status == "ok").sum())
    status = "completed" if completed == expected and not failures else "partial"
    manifest = {
        "experiment_id": "EXP-005",
        "experiment_name": config["experiment"]["name"],
        "run_id": "run_05",
        "status": status,
        "started_utc": state["started_utc"],
        "finished_utc": _utc_now(),
        "git": git,
        "source_tree_sha256": source_hash,
        "source_file_count": len(source),
        "resolved_config_sha256": _sha256(output_root / "resolved_config.json"),
        "data_split_sha256": _sha256(output_root / "data_split.json"),
        "preprocessing_config_sha256": preprocessing_hash,
        "dataset_feature_cache_sha256": cache_hash,
        "folds": folds,
        "seeds": seeds,
        "requested_models": [
            {"model": model_name, "weight_profile": profile}
            for model_name, profile, _, _ in models
        ],
        "expected_jobs": expected,
        "completed_jobs": completed,
        "failed_jobs": len(failures),
        "environment": environment,
        "primary_aggregation": config["cross_bearing"]["primary_aggregation"],
        "test_access_policy": config["cross_bearing"]["test_policy"],
        "preprocessing_policy": preprocessing,
        "prediction_identity_schema": {
            "run_id": "physical bearing/run identifier",
            "bearing_run_id": "explicit copy of the physical bearing/run identifier",
            "experiment_run_id": "experiment execution label run_05",
        },
        "checkpoint_policy": (
            "Validation controls scheduler/early stopping; best-validation and "
            "final-epoch test metrics are both recorded without model changes."
        ),
        "failure_files": failure_files,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_root / "summary.md").write_text(
        "# EXP-005 Run 5 summary\n\n"
        f"Status: {status}\n"
        f"Completed jobs: {completed}/{expected}\n\n"
        "The only substantive experimental change from EXP-004 is fixed-prefix, "
        "per-bearing baseline-relative signal-feature normalization before the "
        "unchanged training-only StandardScaler.\n",
        encoding="utf-8",
    )
    finalize_run5_artifacts(output_root)
    return results, fold_summary, aggregate
