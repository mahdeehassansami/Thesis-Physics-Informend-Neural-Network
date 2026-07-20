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
    normalized = textwrap.dedent(source).strip()
    ast.parse(normalized)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": normalized.splitlines(keepends=True),
    }


cells = [
    markdown(
        """
        # EXP-005 Run 5 — causal baseline-relative normalization

        This notebook is a thin Colab controller. It repeats the frozen EXP-004 IMS
        four-fold, three-model, three-seed comparison while changing only signal-feature
        preprocessing. Each physical bearing is expressed relative to its first eight
        unlabeled snapshots before the unchanged training-only StandardScaler is fitted.
        """
    ),
    markdown(
        """
        ## 1. Mount Drive and check out the exact committed source

        Select a **T4 GPU** runtime. Commit and push the prepared EXP-005 repository,
        then replace `PASTE_40_CHARACTER_COMMIT_SHA` below with that exact commit. The
        uploaded folder supplies the compact feature cache and receives the run output;
        model and experiment code are imported from the verified Git checkout.
        """
    ),
    code(
        r"""
        from google.colab import drive
        drive.mount("/content/drive")

        import json, os, shutil, subprocess, sys
        from pathlib import Path

        UPLOAD = Path("/content/drive/MyDrive/Upload")
        REPOSITORY_URL = "https://github.com/mahdeehassansami/Thesis-Physics-Informend-Neural-Network.git"
        EXPECTED_COMMIT = "PASTE_40_CHARACTER_COMMIT_SHA"
        if not UPLOAD.joinpath("feature_cache", "ims_features.csv").exists():
            raise FileNotFoundError("Missing MyDrive/Upload/feature_cache/ims_features.csv")
        if len(EXPECTED_COMMIT) != 40 or any(ch not in "0123456789abcdef" for ch in EXPECTED_COMMIT.lower()):
            raise ValueError("Paste the exact 40-character Run 5 Git commit SHA.")

        CLONE = Path("/content/thesis_work_exp005")
        if CLONE.exists():
            shutil.rmtree(CLONE)
        subprocess.run(["git", "clone", "--quiet", REPOSITORY_URL, str(CLONE)], check=True)
        subprocess.run(["git", "checkout", "--quiet", EXPECTED_COMMIT], cwd=CLONE, check=True)
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=CLONE, check=True,
            capture_output=True, text=True
        ).stdout.strip()
        if actual != EXPECTED_COMMIT:
            raise RuntimeError(f"Git checkout mismatch: expected {EXPECTED_COMMIT}, got {actual}")
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=CLONE, check=True,
            capture_output=True, text=True
        ).stdout.strip()
        if dirty:
            raise RuntimeError("Checked-out source is dirty.")

        os.chdir(CLONE)
        sys.path.insert(0, str(CLONE / "src"))
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(CLONE / "requirements-colab.txt")],
            check=True,
        )
        import thesis_work
        print("Committed source:", actual)
        print("thesis_work package:", Path(thesis_work.__file__).resolve())
        """
    ),
    markdown(
        """
        ## 2. Verify the T4 runtime and load the frozen EXP-005 configuration

        The controller stops before training if the GPU, Git checkout, feature-cache
        identity, experiment identity, or resumable output identity is wrong.
        """
    ),
    code(
        r"""
        subprocess.run(["nvidia-smi"], check=True)

        import torch
        from thesis_work.multi_dataset import load_experiment_config
        from thesis_work.run5_baseline_normalization import validate_run5_runtime

        PROJECT_ROOT = CLONE
        CACHE_DIR = UPLOAD / "feature_cache"
        OUTPUT_DIR = UPLOAD / "experiment_outputs_run_05"
        CONFIG_PATH = PROJECT_ROOT / "configs" / "colab_experiments.json"
        config = load_experiment_config(CONFIG_PATH)
        config["repository"]["expected_commit"] = EXPECTED_COMMIT
        if config.get("run_label") != "run_05" or config["experiment"]["id"] != "EXP-005":
            raise ValueError("The checked-out source does not contain EXP-005/run_05.")
        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()) and not (OUTPUT_DIR / "run_state.json").exists():
            raise FileExistsError("Run 5 output exists without a compatible resumable run_state.json.")

        environment, git = validate_run5_runtime(config, PROJECT_ROOT)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        environment["gpu_memory_free_bytes_at_start"] = int(free_bytes)
        environment["gpu_memory_total_bytes_at_start"] = int(total_bytes)
        environment["selected_device"] = "cuda:0"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(json.dumps({
            "experiment": config["experiment"],
            "preprocessing": config["preprocessing"],
            "folds": config["cross_bearing"]["folds"],
            "environment": environment,
            "commit": git["commit"],
            "output": str(OUTPUT_DIR),
        }, indent=2))
        """
    ),
    markdown(
        """
        ## 3. Execute the fixed four-fold experiment

        This schedules 36 jobs: four held-out test bearings × three frozen models ×
        three common seeds. The first eight samples of each bearing fit only that
        bearing's label-free baseline. Validation controls scheduling and early stopping;
        test metrics never alter preprocessing, architecture, weights, or checkpoints.
        """
    ),
    code(
        r"""
        from thesis_work.run5_baseline_normalization import run_run5_experiment

        results, fold_summary, aggregate = run_run5_experiment(
            config=config,
            project_root=PROJECT_ROOT,
            cache_dir=CACHE_DIR,
            output_root=OUTPUT_DIR,
            refresh_features=False,
        )
        display(aggregate)
        display(fold_summary)
        """
    ),
    markdown(
        """
        ## 4. Inspect the predeclared comparison

        Weak-PINN/high is the primary Run 5 model. Success requires improvement over
        EXP-004 macro RMSE 0.314238 and worst-bearing RMSE 0.497089, improvement in at
        least three folds, reduced between-bearing variation, and no worse late-life bias.
        """
    ),
    code(
        r"""
        import pandas as pd

        aggregate = pd.read_csv(OUTPUT_DIR / "all_model_comparisons_summary.csv")
        folds = pd.read_csv(OUTPUT_DIR / "fold_model_summary.csv")
        display(aggregate)
        display(folds)
        display(pd.DataFrame([config["cross_bearing"]["baseline_comparison"]]))
        """
    ),
    markdown(
        """
        ## 5. Finalize the complete output and lightweight bundle

        Save this notebook before running the cell so the Drive copy reflects the run.
        The complete directory retains checkpoints; `codex_results_bundle.zip` excludes
        checkpoints and is the lightweight evidence package.
        """
    ),
    code(
        r"""
        import shutil
        from thesis_work.run5_baseline_normalization import finalize_run5_artifacts

        notebook_source = UPLOAD / "Thesis_v3_with_extra_graphs_tables.ipynb"
        if notebook_source.exists():
            shutil.copy2(notebook_source, OUTPUT_DIR / "executed_notebook.ipynb")
        bundle = finalize_run5_artifacts(OUTPUT_DIR)
        manifest = json.loads((OUTPUT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
        failures = json.loads((OUTPUT_DIR / "failure_report.json").read_text(encoding="utf-8"))
        print(json.dumps({
            "status": manifest["status"],
            "completed_jobs": manifest["completed_jobs"],
            "expected_jobs": manifest["expected_jobs"],
            "failed_jobs": manifest["failed_jobs"],
            "failures": failures,
            "bundle": str(bundle),
            "complete_output": str(OUTPUT_DIR),
            "next_step": "Download experiment_outputs_run_05 and place it in thesis-work for Codex analysis.",
        }, indent=2))
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": "Bearing_RUL_EXP_005_Run_5.ipynb",
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
encoded = json.dumps(notebook, indent=1) + "\n"
for target in TARGETS:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(encoded, encoding="utf-8")
    print(f"Saved {target.relative_to(ROOT)}")
