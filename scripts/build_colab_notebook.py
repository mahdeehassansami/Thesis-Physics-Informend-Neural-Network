from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = [
    ROOT / "Thesis_v3_with_extra_graphs_tables.ipynb",
    ROOT / "notebooks" / "train_models_colab.ipynb",
]


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        r"""
# Bearing RUL: baselines, AttnPINN, weak-prior PINN, and explicit-law PINN

This is the canonical **Google Colab training controller**. The implementation lives in
`src/thesis_work/`; this notebook deliberately does not duplicate model definitions.

Model families:

1. data-only FNN, CNN, and LSTM baselines;
2. AttnPINN/DeepHPM, which learns an unknown latent differential operator;
3. weak-prior PINN, using monotonic degradation, boundary, health-indicator, and optional temperature priors;
4. strong PINN, adding separately weighted Paris rolling-contact-fatigue crack growth and Palmgren-Miner/bearing-life residuals with temperature/lubrication modifiers.

Important scientific limitation: missing load, contact pressure, bearing-capacity, and
lubricant values are explicit calibration assumptions in the configuration. The
differentiable `aSKF` term is a surrogate for catalog curves, not an unqualified exact ISO
lookup. Inspect every dataset's `assumptions.json` before interpreting the strong-PINN result.
"""
    ),
    markdown(
        """
## 1. Colab runtime and repository

Use a GPU runtime. Raw datasets remain read-only; extracted feature tables and all
checkpoints/results are written to Google Drive.
"""
    ),
    code(
        r"""
from google.colab import drive
drive.mount("/content/drive")

import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_URL = "https://github.com/mahdeehassansami/Thesis-Physics-Informend-Neural-Network.git"
REPO_DIR = Path("/content/thesis-work")
if not (REPO_DIR / ".git").exists():
    subprocess.run(["git", "clone", REPOSITORY_URL, str(REPO_DIR)], check=True)

os.chdir(REPO_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
print("Repository:", REPO_DIR)
"""
    ),
    markdown(
        """
## 2. Paths and experiment configuration

Edit dataset `enabled` flags and weight profiles in
`configs/colab_experiments.json`. The default suite runs three RUL datasets. CWRU is
intentionally excluded because it is fault-classification data, not run-to-failure RUL data.
"""
    ),
    code(
        r"""
import json
from pathlib import Path

from thesis_work.multi_dataset import (
    load_experiment_config,
    validate_dataset_config,
)

PROJECT_ROOT = REPO_DIR
DRIVE_ROOT = Path("/content/drive/MyDrive/thesis-work")
CACHE_DIR = DRIVE_ROOT / "feature_cache"
OUTPUT_DIR = DRIVE_ROOT / "experiment_outputs_run_02"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = PROJECT_ROOT / "configs" / "colab_experiments.json"
config = load_experiment_config(CONFIG_PATH)

status = [
    validate_dataset_config(dataset, PROJECT_ROOT)
    for dataset in config["datasets"]
]
status
"""
    ),
    code(
        r"""
# Optional quick-run overrides. Keep False for the full configured experiment.
QUICK_RUN = False
if QUICK_RUN:
    config["training"].update({"epochs": 3, "patience": 2, "batch_size": 64, "seed_repeats": 1})
    for dataset in config["datasets"]:
        dataset["enabled"] = dataset["name"] == "ims"
    for model_name, model_config in config["models"].items():
        if model_name in {"attnpinn", "weak_pinn", "strong_pinn"}:
            model_config["profiles"] = [model_config["profiles"][1]]

print(json.dumps({
    "enabled_datasets": [d["name"] for d in config["datasets"] if d.get("enabled")],
    "training": config["training"],
    "models": {name: value["profiles"] for name, value in config["models"].items() if value.get("enabled")},
}, indent=2))
"""
    ),
    markdown(
        """
## 3. Extract/cache features and train the complete model suite

Within each dataset, every model uses the same train/validation/test split and a scaler fit
only on training data. Physics-weight profiles are separate model runs, so low/medium/high
constraints can be compared directly.
"""
    ),
    code(
        r"""
from thesis_work.experiment_runner import run_all_experiments

results, _ = run_all_experiments(
    config=config,
    project_root=PROJECT_ROOT,
    cache_dir=CACHE_DIR,
    output_root=OUTPUT_DIR,
    refresh_features=False,
    run_sensitivity=False,
)
results.sort_values(["dataset", "rmse"], na_position="last")
"""
    ),
    code(
        r"""
import matplotlib.pyplot as plt
import seaborn as sns

successful = results[results["status"] == "ok"].copy()
successful["model_profile"] = successful["model"] + "\n" + successful["weight_profile"]
for dataset_name, dataset_results in successful.groupby("dataset"):
    ordered = dataset_results.sort_values("rmse")
    plt.figure(figsize=(12, 5))
    sns.barplot(data=ordered, x="model_profile", y="rmse", color="#4472C4", errorbar="sd")
    plt.title(f"{dataset_name}: test RMSE by model and physics weight")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    figure_path = OUTPUT_DIR / dataset_name / "model_rmse_comparison.png"
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.show()
"""
    ),
    markdown(
        """
## 4. One-at-a-time sensitivity analysis

The default sensitivity grid varies the two explicit-law loss weights, Paris exponent,
temperature-viscosity coefficient, and poor-lubrication multiplier. It uses shorter training
runs and writes both trial-level results and a range-based parameter ranking.
"""
    ),
    code(
        r"""
from thesis_work.experiment_runner import run_sensitivity_analysis
from thesis_work.multi_dataset import load_or_extract_dataset, prepare_sequence_dataset

RUN_SENSITIVITY = True
SENSITIVITY_DATASET = config["sensitivity"]["dataset"]
sensitivity_results = None

if RUN_SENSITIVITY:
    config["sensitivity"]["enabled"] = True
    dataset_config = next(
        dataset for dataset in config["datasets"]
        if dataset["name"] == SENSITIVITY_DATASET
    )
    feature_frame = load_or_extract_dataset(
        dataset_config,
        project_root=PROJECT_ROOT,
        cache_dir=CACHE_DIR,
        refresh=False,
    )
    prepared = prepare_sequence_dataset(
        feature_frame,
        dataset_config,
        sequence_length=config["training"]["sequence_length"],
    )
    sensitivity_results = run_sensitivity_analysis(
        prepared=prepared,
        dataset_config=dataset_config,
        config=config,
        output_root=OUTPUT_DIR,
    )
    sensitivity_results.to_csv(
        OUTPUT_DIR / "all_sensitivity_results.csv", index=False
    )
    display(sensitivity_results.sort_values(["parameter", "value"]))
"""
    ),
    code(
        r"""
if sensitivity_results is not None and not sensitivity_results.empty:
    sensitivity_ok = sensitivity_results[sensitivity_results["status"] == "ok"]
    grid = sns.FacetGrid(
        sensitivity_ok,
        col="parameter",
        col_wrap=2,
        sharex=False,
        sharey=True,
        height=3.5,
    )
    grid.map_dataframe(sns.lineplot, x="value", y="rmse", marker="o")
    grid.set_titles("{col_name}")
    grid.figure.tight_layout()
    sensitivity_figure = OUTPUT_DIR / SENSITIVITY_DATASET / "sensitivity" / "sensitivity_rmse.png"
    grid.figure.savefig(sensitivity_figure, dpi=200, bbox_inches="tight")
    plt.show()
"""
    ),
    markdown(
        """
## 5. Artifact audit

Each dataset directory contains model checkpoints, component-wise training histories,
predictions, comparison tables, failure traces (if any), and an assumptions file. Do not
report a failed model as a missing bar, and do not describe assumed operating values as
measured dataset variables.
"""
    ),
    code(
        r"""
from IPython.display import display

display(results)
print("Results:", OUTPUT_DIR)
print("Feature cache:", CACHE_DIR)
print("Failures:", list(OUTPUT_DIR.rglob("failure.txt")))
print("Assumption records:", list(OUTPUT_DIR.rglob("assumptions.json")))
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "Bearing_RUL_PINN_Experiments.ipynb", "provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


for output in OUTPUTS:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {output}")
