from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "train_models_colab.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def main() -> None:
    cells = [
        markdown(
            """
# EXP-007A - counterfactual physics-harm credibility

This thin, restartable controller runs the frozen protocol-v0.2 corrective experiment. Candidate physics models are trained with differentiable progression value/rate/monotonic losses from identical data-only checkpoints. Development counterfactual regret labels must qualify before the fresh sealed test is evaluated.
"""
        ),
        markdown(
            """
## 1. Mount Drive and verify the exact pushed source

Upload the newly prepared `Upload` folder to `MyDrive`, open this notebook from that folder, select a **T4 GPU**, and run all cells. The pushed SHA is read automatically from `expected_commit.txt`.
"""
        ),
        code(
            """
from google.colab import drive
drive.mount("/content/drive")

import json, os, shutil, subprocess, sys
from pathlib import Path

UPLOAD = Path("/content/drive/MyDrive/Upload")
FEATURE_PATH = UPLOAD / "feature_cache" / "multicondition_features.csv"
METADATA_PATH = UPLOAD / "feature_cache" / "multicondition_metadata.json"
COMMIT_FILE = UPLOAD / "expected_commit.txt"
REPOSITORY_URL = "https://github.com/mahdeehassansami/Thesis-Physics-Informend-Neural-Network.git"
for required in (FEATURE_PATH, METADATA_PATH, COMMIT_FILE):
    if not required.is_file():
        raise FileNotFoundError(f"Missing EXP-007A Upload file: {required}")
EXPECTED_COMMIT = COMMIT_FILE.read_text(encoding="utf-8").strip().lower()
if len(EXPECTED_COMMIT) != 40 or any(ch not in "0123456789abcdef" for ch in EXPECTED_COMMIT):
    raise ValueError("expected_commit.txt must contain one full lowercase Git SHA.")

CLONE = Path("/content/thesis_work_exp007a")
if CLONE.exists():
    shutil.rmtree(CLONE)
subprocess.run(["git", "clone", "--quiet", REPOSITORY_URL, str(CLONE)], check=True)
subprocess.run(["git", "checkout", "--quiet", EXPECTED_COMMIT], cwd=CLONE, check=True)
actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CLONE, check=True, capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain"], cwd=CLONE, check=True, capture_output=True, text=True).stdout.strip()
if actual != EXPECTED_COMMIT or dirty:
    raise RuntimeError(f"Repository identity failure: actual={actual}, dirty={bool(dirty)}")
print({"expected_commit": EXPECTED_COMMIT, "actual_commit": actual, "clean": not bool(dirty)})
"""
        ),
        markdown("## 2. Install the committed package and dependencies"),
        code(
            """
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(CLONE / "requirements-colab.txt")], check=True)
"""
        ),
        markdown("## 3. Verify the T4, CUDA, PyTorch, and memory"),
        code(
            """
subprocess.run(["nvidia-smi"], check=True)
import torch
print({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
    raise RuntimeError("EXP-007A requires a Colab Tesla T4 runtime.")
free, total = torch.cuda.mem_get_info()
print({"gpu_memory_free_bytes": free, "gpu_memory_total_bytes": total, "selected_device": "cuda:0"})
"""
        ),
        markdown("## 4. Load, pin, and validate the single experiment configuration"),
        code(
            """
sys.path.insert(0, str(CLONE / "src"))
from thesis_work.exp7a_harm_credibility import load_exp7a_config, validate_exp7a_runtime

CONFIG_PATH = CLONE / "configs" / "experiment.yaml"
config = load_exp7a_config(CONFIG_PATH)
config["repository"]["expected_commit"] = EXPECTED_COMMIT
environment, git_state, qualification = validate_exp7a_runtime(config, CLONE, FEATURE_PATH)
print(json.dumps({"experiment": config["experiment"], "qualification": qualification}, indent=2))

LOCAL_ROOT = Path(config["runtime"]["train_work_directory"])
OUTPUT_ROOT = LOCAL_ROOT / "experiment_outputs_exp007a"
RECOVERY_ROOT = UPLOAD / "experiment_outputs_exp007a"
LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
if RECOVERY_ROOT.exists() and any(RECOVERY_ROOT.iterdir()):
    state = RECOVERY_ROOT / "run_state.json"
    if not state.is_file():
        raise FileExistsError("Nonempty EXP-007A Drive output lacks compatible run_state.json.")
else:
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
print({"local_output": str(OUTPUT_ROOT), "drive_recovery": str(RECOVERY_ROOT)})
"""
        ),
        markdown(
            """
## 5. Train, qualify the development harm target, and conditionally evaluate the sealed test

The runner resumes completed fold/candidate checkpoints. If development contains insufficient safe or harmful interventions, it records a benchmark-design failure and does not evaluate the sealed test.
"""
        ),
        code(
            """
from thesis_work.exp7a_harm_credibility import run_exp7a_experiment

gate = run_exp7a_experiment(
    config=config,
    project_root=CLONE,
    feature_path=FEATURE_PATH,
    output_root=OUTPUT_ROOT,
    recovery_root=RECOVERY_ROOT,
)
print(json.dumps(gate, indent=2))
"""
        ),
        markdown("## 6. Finalize the complete Drive artifacts and lightweight Codex bundle"),
        code(
            """
from thesis_work.exp7a_harm_credibility import finalize_exp7a_artifacts

notebook_source = UPLOAD / "train_models_colab.ipynb"
if notebook_source.is_file():
    shutil.copy2(notebook_source, OUTPUT_ROOT / "executed_notebook.ipynb")
finalize_exp7a_artifacts(OUTPUT_ROOT)
shutil.copytree(OUTPUT_ROOT, RECOVERY_ROOT, dirs_exist_ok=True)
finalize_exp7a_artifacts(RECOVERY_ROOT)
bundle = RECOVERY_ROOT / "codex_results_bundle.zip"
if not bundle.is_file():
    raise FileNotFoundError(bundle)
print({"full_drive_artifacts": str(RECOVERY_ROOT), "lightweight_bundle": str(bundle)})
"""
        ),
        markdown("## 7. Download the lightweight bundle and return it for independent analysis"),
        code(
            """
from google.colab import files
files.download(str(RECOVERY_ROOT / "codex_results_bundle.zip"))
print("Place the downloaded EXP-007A bundle under results/incoming and ask Codex to verify it. Do not start EXP-008 from the notebook output alone.")
"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
