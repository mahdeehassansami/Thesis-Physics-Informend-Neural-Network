from __future__ import annotations

import os
import random

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np
import pandas as pd

from thesis_work.config import (
    CNN_SEED,
    DATA_BASELINE_SEED,
    EXPERIMENTS,
    GLOBAL_SEED,
    LSTM_SEED,
    MODEL_ORDER,
    PINN_SEED,
    SEED_REPEATS,
    SEED_REPEAT_STRIDE,
    TARGET_RUNS,
    ProjectPaths,
    default_paths,
)
from thesis_work.features import load_or_extract_run
from thesis_work.ims import list_snapshots, load_snapshot, resolve_dataset_source
from thesis_work.manifest import write_run_manifest
from thesis_work.preprocessing import compute_pca_hi, prepare_all_contexts, preprocess_features
from thesis_work.reports import (
    plot_final_predictions_from_table,
    plot_prediction_error_analysis,
    prediction_store_to_table,
    save_static_tables_and_figures,
)


def validate_data(paths: ProjectPaths | None = None) -> pd.DataFrame:
    paths = paths or default_paths()
    rows = []
    for dataset in ("1st_test", "2nd_test", "3rd_test"):
        source = resolve_dataset_source(paths.raw_data, dataset)
        files = list_snapshots(source)
        n_cols = 8 if dataset == "1st_test" else 4
        first_shape = load_snapshot(source, files[0], n_cols).shape if files else None
        rows.append(
            {
                "dataset": dataset,
                "source": str(source),
                "snapshots": len(files),
                "expected_columns": n_cols,
                "first_snapshot_shape": str(first_shape),
                "first_file": files[0] if files else "",
                "last_file": files[-1] if files else "",
            }
        )
    return pd.DataFrame(rows)


def load_or_build_all_features(
    paths: ProjectPaths | None = None,
    max_files: int | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    paths = paths or default_paths()
    all_runs = []
    for run_id, dataset, bearing in TARGET_RUNS:
        print(f"Loading {run_id}")
        all_runs.append(
            load_or_extract_run(
                paths,
                run_id=run_id,
                dataset=dataset,
                bearing=bearing,
                max_files=max_files,
                refresh=refresh,
            )
        )
    multi_df = pd.concat(all_runs, ignore_index=True)
    output = paths.processed_features / ("all_runs_features.csv" if max_files is None else f"all_runs_first_{max_files}_features.csv")
    multi_df.to_csv(output, index=False)
    print(f"Saved combined features: {output}")
    return multi_df


def run_pipeline(
    paths: ProjectPaths | None = None,
    max_files: int | None = None,
    refresh_features: bool = False,
    skip_training: bool = False,
    baseline_iterations: int = 15_000,
    pinn_iterations: int = 20_000,
    sequence_epochs: int = 60,
    sequence_patience: int = 8,
    sequence_batch_size: int = 128,
    sequence_length: int = 20,
    global_seed: int = GLOBAL_SEED,
    data_baseline_seed: int = DATA_BASELINE_SEED,
    pinn_seed: int = PINN_SEED,
    lstm_seed: int = LSTM_SEED,
    cnn_seed: int = CNN_SEED,
    seed_repeats: int = SEED_REPEATS,
    seed_repeat_stride: int = SEED_REPEAT_STRIDE,
) -> dict[str, object]:
    paths = paths or default_paths()
    random.seed(global_seed)
    np.random.seed(global_seed)
    multi_df = load_or_build_all_features(paths, max_files=max_files, refresh=refresh_features)
    proc_df = preprocess_features(multi_df)
    pca_hi_df, pca_hi_summary = compute_pca_hi(proc_df)
    pca_hi_summary.to_csv(paths.tables / "pca_hi_ablation_summary.csv", index=False)

    final_results = None
    prediction_store = None
    trained_models = None
    if not skip_training:
        from thesis_work.models import train_all_models

        contexts = prepare_all_contexts(proc_df)
        final_results, prediction_store, trained_models = train_all_models(
            contexts=contexts,
            experiments=EXPERIMENTS,
            baseline_iterations=baseline_iterations,
            pinn_iterations=pinn_iterations,
            sequence_epochs=sequence_epochs,
            sequence_patience=sequence_patience,
            sequence_batch_size=sequence_batch_size,
            sequence_length=sequence_length,
            data_baseline_seed=data_baseline_seed,
            pinn_seed=pinn_seed,
            lstm_seed=lstm_seed,
            cnn_seed=cnn_seed,
            seed_repeats=seed_repeats,
            seed_repeat_stride=seed_repeat_stride,
        )
        final_results["Model"] = pd.Categorical(final_results["Model"], categories=MODEL_ORDER, ordered=True)
        sort_cols = ["Experiment", "Seed repeat", "Model"] if "Seed repeat" in final_results.columns else ["Experiment", "Model"]
        final_results = final_results.sort_values(sort_cols).reset_index(drop=True)
        final_results["Model"] = final_results["Model"].astype(str)
        result_cols = ["Experiment", "Train runs", "Validation run", "Test run"]
        if "Seed repeat" in final_results.columns:
            result_cols.extend(["Seed repeat", "Seed"])
        result_cols.extend(["Model", "MAE", "RMSE", "R2"])
        final_results = final_results[result_cols]
        prediction_table = prediction_store_to_table(prediction_store)
        prediction_table.to_csv(paths.tables / "prediction_series.csv", index=False)

    save_static_tables_and_figures(
        paths=paths,
        multi_df=multi_df,
        proc_df=proc_df,
        pca_hi_df=pca_hi_df,
        pca_hi_summary=pca_hi_summary,
        final_results=final_results,
        prediction_store=prediction_store,
    )
    write_run_manifest(
        paths,
        command="run",
        options={
            "max_files": max_files,
            "refresh_features": refresh_features,
            "skip_training": skip_training,
            "baseline_iterations": baseline_iterations,
            "pinn_iterations": pinn_iterations,
            "sequence_epochs": sequence_epochs,
            "sequence_patience": sequence_patience,
            "sequence_batch_size": sequence_batch_size,
            "sequence_length": sequence_length,
            "seed_repeats": seed_repeats,
            "seed_repeat_stride": seed_repeat_stride,
            "source_of_truth": "python_project",
        },
        seeds={
            "global": global_seed,
            "data_baseline": data_baseline_seed,
            "pinn": pinn_seed,
            "lstm": lstm_seed,
            "cnn": cnn_seed,
        },
    )
    return {
        "multi_df": multi_df,
        "proc_df": proc_df,
        "pca_hi_df": pca_hi_df,
        "pca_hi_summary": pca_hi_summary,
        "final_results": final_results,
        "prediction_store": prediction_store,
        "trained_models": trained_models,
    }


def regenerate_figures_from_cache(paths: ProjectPaths | None = None) -> dict[str, object]:
    paths = paths or default_paths()
    feature_path = paths.processed_features / "all_runs_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing cached features at {feature_path}. Run `uv run thesis-work extract-features` first."
        )

    multi_df = pd.read_csv(feature_path)
    proc_df = preprocess_features(multi_df)
    pca_hi_df, pca_hi_summary = compute_pca_hi(proc_df)

    split_seed_results_path = paths.tables / "split_seed_results_table.csv"
    final_results_path = split_seed_results_path if split_seed_results_path.exists() else paths.tables / "final_results_table.csv"
    final_results = pd.read_csv(final_results_path) if final_results_path.exists() else None
    if final_results is not None and "R²" in final_results.columns and "R2" not in final_results.columns:
        final_results = final_results.rename(columns={"R²": "R2"})

    save_static_tables_and_figures(
        paths=paths,
        multi_df=multi_df,
        proc_df=proc_df,
        pca_hi_df=pca_hi_df,
        pca_hi_summary=pca_hi_summary,
        final_results=final_results,
        prediction_store=None,
    )
    prediction_path = paths.tables / "prediction_series.csv"
    if prediction_path.exists():
        prediction_table = pd.read_csv(prediction_path)
        plot_final_predictions_from_table(paths, prediction_table)
        plot_prediction_error_analysis(paths, prediction_table)
    write_run_manifest(
        paths,
        command="regenerate-figures",
        options={
            "feature_cache": str(feature_path),
            "final_results_table": str(final_results_path) if final_results_path.exists() else None,
            "prediction_series": str(prediction_path) if prediction_path.exists() else None,
            "source_of_truth": "python_project",
        },
        seeds={
            "global": GLOBAL_SEED,
            "data_baseline": DATA_BASELINE_SEED,
            "pinn": PINN_SEED,
            "lstm": LSTM_SEED,
            "cnn": CNN_SEED,
        },
        filename="figure_manifest.json",
    )
    return {
        "multi_df": multi_df,
        "proc_df": proc_df,
        "pca_hi_df": pca_hi_df,
        "pca_hi_summary": pca_hi_summary,
        "final_results": final_results,
    }


