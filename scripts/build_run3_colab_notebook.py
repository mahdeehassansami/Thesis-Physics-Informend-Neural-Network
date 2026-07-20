from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "Thesis_v3_with_extra_graphs_tables.ipynb",
    ROOT / "notebooks" / "train_models_colab.ipynb",
]


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    source = textwrap.dedent(source).strip()
    ast.parse(source)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """
        # Run 3 — validation-only IMS Strong-PINN calibration

        This notebook is the thin Google Colab controller for **EXP-003**. Model,
        physics, training, selection, and artifact logic live under `src/thesis_work/`.

        Run 3 keeps the leakage-free Run 2 IMS features and split. It trains LSTM and
        Weak-PINN/high references, then trains a 3×3 Paris/Miner Strong-PINN grid over
        three common seeds. Grid candidates are ranked only on validation RMSE. The test
        split is opened only for the frozen winning Strong-PINN profile and the two
        predeclared reference models.
        """
    ),
    markdown(
        """
        ## 1. Mount the exact Upload workspace

        Select a **T4 GPU** runtime before running this notebook. Upload the complete
        `Upload` directory to the root of MyDrive. The controller refuses to fall back
        silently to an unidentified source tree.
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

        DRIVE_CANDIDATES = [
            Path("/content/drive/MyDrive/Upload"),
            Path("/content/drive/MyDrive/thesis-work"),
        ]
        DRIVE_WORKSPACE = next(
            (
                candidate
                for candidate in DRIVE_CANDIDATES
                if (candidate / "pyproject.toml").exists()
            ),
            None,
        )
        if DRIVE_WORKSPACE is None:
            raise FileNotFoundError(
                "Upload workspace not found. Expected MyDrive/Upload/pyproject.toml."
            )

        REPO_DIR = DRIVE_WORKSPACE
        os.chdir(REPO_DIR)
        SOURCE_DIR = REPO_DIR / "src"
        if not (SOURCE_DIR / "thesis_work" / "__init__.py").exists():
            raise FileNotFoundError(f"Missing thesis_work package under {SOURCE_DIR}")
        source_path = str(SOURCE_DIR.resolve())
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"],
            check=True,
        )

        import thesis_work
        print("Workspace:", REPO_DIR)
        print("thesis_work package:", Path(thesis_work.__file__).resolve())
        """
    ),
    markdown(
        """
        ## 2. Load the immutable Run 3 configuration

        The active configuration declares EXP-003, the fixed IMS split, three seeds,
        reference models, calibration grid, runtime requirement, and test-access policy.
        """
    ),
    code(
        r"""
        import copy
        import json

        from thesis_work.multi_dataset import load_experiment_config
        from thesis_work.run3_calibration import validate_run3_runtime

        PROJECT_ROOT = DRIVE_WORKSPACE
        CACHE_DIR = DRIVE_WORKSPACE / "feature_cache"
        OUTPUT_DIR = DRIVE_WORKSPACE / "experiment_outputs_run_03"
        CONFIG_PATH = PROJECT_ROOT / "configs" / "colab_experiments.json"
        config = load_experiment_config(CONFIG_PATH)

        if config.get("run_label") != "run_03":
            raise ValueError(f"Expected run_03 config, found {config.get('run_label')!r}")
        ims_cache = CACHE_DIR / "ims_features.csv"
        if not ims_cache.exists():
            raise FileNotFoundError(f"Missing cached IMS features: {ims_cache}")

        QUICK_RUN = False
        if QUICK_RUN:
            config = copy.deepcopy(config)
            config["training"].update(
                {
                    "epochs": 3,
                    "patience": 2,
                    "seed_repeats": 1,
                    "seeds": [42],
                    "gradient_diagnostics_interval": 1,
                }
            )
            config["calibration"]["paris_weights"] = [0.003, 0.01]
            config["calibration"]["miner_weights"] = [0.0003, 0.001]

        environment = validate_run3_runtime(config)
        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
            raise FileExistsError(
                f"{OUTPUT_DIR} is not empty. Rename/archive it before starting a new run."
            )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(json.dumps({
            "experiment": config["experiment"],
            "run_label": config["run_label"],
            "training": config["training"],
            "calibration": config["calibration"],
            "gpu": environment["gpu_name"],
            "feature_cache": str(ims_cache),
            "output": str(OUTPUT_DIR),
        }, indent=2))
        """
    ),
    markdown(
        """
        ## 3. Train references, select on validation, then evaluate the frozen winner

        Keep `QUICK_RUN = False` for the real experiment. This cell performs all training
        and writes traceable evidence to `experiment_outputs_run_03`.
        """
    ),
    code(
        r"""
        from thesis_work.run3_calibration import run_run3_experiment

        results, validation_grid = run_run3_experiment(
            config=config,
            project_root=PROJECT_ROOT,
            cache_dir=CACHE_DIR,
            output_root=OUTPUT_DIR,
            refresh_features=False,
        )
        display(results.sort_values(["model", "seed_repeat"]))
        """
    ),
    markdown(
        """
        ## 4. Inspect validation selection separately from test evidence

        `validation_grid_summary.csv` contains all nine candidate profiles. Only the
        profile recorded in `selected_profile.json` has test predictions.
        """
    ),
    code(
        r"""
        import pandas as pd

        calibration_dir = OUTPUT_DIR / "ims" / "calibration"
        validation_summary = pd.read_csv(
            calibration_dir / "validation_grid_summary.csv"
        )
        selected_profile = json.loads(
            (calibration_dir / "selected_profile.json").read_text(encoding="utf-8")
        )
        print(json.dumps(selected_profile, indent=2))
        display(validation_summary)
        """
    ),
    code(
        r"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        heatmap = validation_summary.pivot(
            index="paris_weight",
            columns="miner_weight",
            values="validation_rmse_mean",
        )
        plt.figure(figsize=(7, 5))
        sns.heatmap(heatmap, annot=True, fmt=".4f", cmap="viridis_r")
        plt.title("IMS Strong-PINN validation RMSE (three-seed mean)")
        plt.xlabel("Palmgren-Miner loss weight")
        plt.ylabel("Paris loss weight")
        plt.tight_layout()
        figure_path = calibration_dir / "validation_grid_rmse.png"
        plt.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.show()
        """
    ),
    markdown(
        """
        ## 5. Final artifact check

        Download the complete `experiment_outputs_run_03` folder after confirming the
        run manifest says `completed` and the failure list is empty.
        """
    ),
    code(
        r"""
        manifest = json.loads(
            (OUTPUT_DIR / "run_manifest.json").read_text(encoding="utf-8")
        )
        failures = json.loads(
            (OUTPUT_DIR / "failure_report.json").read_text(encoding="utf-8")
        )
        display(pd.read_csv(OUTPUT_DIR / "all_model_comparisons_summary.csv"))
        print(json.dumps(manifest, indent=2))
        print("Failures:", failures["failures"])
        print("Download this folder:", OUTPUT_DIR)
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": "Bearing_RUL_EXP_003_Run_3.ipynb",
            "provenance": [],
        },
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

serialized = json.dumps(notebook, indent=1) + "\n"
for target in TARGETS:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialized, encoding="utf-8")
    print(f"Saved {target.relative_to(ROOT)}")
