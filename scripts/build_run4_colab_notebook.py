from __future__ import annotations
import ast, json, textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TARGETS=[ROOT/"Thesis_v3_with_extra_graphs_tables.ipynb",ROOT/"notebooks"/"train_models_colab.ipynb"]

def md(s):
    return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(s).strip().splitlines(keepends=True)}
def code(s):
    s=textwrap.dedent(s).strip(); ast.parse(s)
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.splitlines(keepends=True)}

cells=[
md("""# EXP-004 Run 4 — IMS held-out-bearing robustness
This notebook is a thin Colab controller. It evaluates LSTM, Weak-PINN/high, and the frozen Run 3 Strong-PINN profile over four fixed IMS folds and three common seeds. Validation is used only for scheduler and early stopping; all model choices are frozen before test-fold evaluation."""),
md("""## 1. Mount Drive and check out the exact committed source
Select a **T4 GPU** runtime. Before running this cell, commit and push the prepared repository, then replace `EXPECTED_COMMIT` with the resulting 40-character SHA. The notebook clones that exact revision; the Upload copy is retained as a self-contained reference and provides the compact cache."""),
code(r"""
from google.colab import drive
drive.mount("/content/drive")

import json, os, shutil, subprocess, sys
from pathlib import Path

UPLOAD = Path("/content/drive/MyDrive/Upload")
REPOSITORY_URL = "https://github.com/mahdeehassansami/Thesis-Physics-Informend-Neural-Network.git"
EXPECTED_COMMIT = "PASTE_40_CHARACTER_COMMIT_SHA"
if not UPLOAD.joinpath("feature_cache", "ims_features.csv").exists():
    raise FileNotFoundError("Missing MyDrive/Upload/feature_cache/ims_features.csv")
if len(EXPECTED_COMMIT) != 40:
    raise ValueError("Set EXPECTED_COMMIT to the commit SHA after pushing Run 4.")
CLONE = Path("/content/thesis_work_exp004")
if CLONE.exists():
    shutil.rmtree(CLONE)
subprocess.run(["git", "clone", "--quiet", REPOSITORY_URL, str(CLONE)], check=True)
subprocess.run(["git", "checkout", "--quiet", EXPECTED_COMMIT], cwd=CLONE, check=True)
actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CLONE, check=True, capture_output=True, text=True).stdout.strip()
if actual != EXPECTED_COMMIT:
    raise RuntimeError(f"Git checkout mismatch: {actual}")
if subprocess.run(["git", "status", "--porcelain"], cwd=CLONE, check=True, capture_output=True, text=True).stdout.strip():
    raise RuntimeError("Checked-out source is dirty.")
os.chdir(CLONE)
sys.path.insert(0, str(CLONE / "src"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(CLONE / "requirements-colab.txt")], check=True)
import thesis_work
print("Committed source:", actual)
print("thesis_work package:", Path(thesis_work.__file__).resolve())
"""),
md("""## 2. Load EXP-004 and validate the T4 runtime
The configuration defines the four folds, frozen models, frozen physics weights, cache hash, and artifact policy. Do not change these values for the real run."""),
code(r"""
from thesis_work.multi_dataset import load_experiment_config
from thesis_work.run4_cross_bearing import validate_run4_runtime

PROJECT_ROOT = CLONE
CACHE_DIR = UPLOAD / "feature_cache"
OUTPUT_DIR = UPLOAD / "experiment_outputs_run_04"
CONFIG_PATH = PROJECT_ROOT / "configs" / "colab_experiments.json"
config = load_experiment_config(CONFIG_PATH)
config["repository"]["expected_commit"] = EXPECTED_COMMIT
if config.get("run_label") != "run_04" or config["experiment"]["id"] != "EXP-004":
    raise ValueError("The checked-out source does not contain EXP-004.")
if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()) and not (OUTPUT_DIR / "run_state.json").exists():
    raise FileExistsError("Run 4 output exists without a resumable run_state.json.")
environment, git = validate_run4_runtime(config, PROJECT_ROOT)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(json.dumps({"experiment": config["experiment"], "folds": config["cross_bearing"]["folds"], "gpu": environment["gpu_name"], "commit": git["commit"], "output": str(OUTPUT_DIR)}, indent=2))
"""),
md("""## 3. Execute the fixed four-fold experiment
This schedules 36 jobs: four held-out test bearings × three models × three seeds. It resumes completed jobs only when the source, config, split, and cache identities match."""),
code(r"""
from thesis_work.run4_cross_bearing import run_run4_experiment
results, fold_summary, aggregate = run_run4_experiment(
    config=config,
    project_root=PROJECT_ROOT,
    cache_dir=CACHE_DIR,
    output_root=OUTPUT_DIR,
    refresh_features=False,
)
display(aggregate)
display(fold_summary)
"""),
md("""## 4. Inspect fold robustness
The primary comparison is the macro mean normalized RMSE, giving each held-out bearing equal weight. The final Strong-PINN profile is not recalibrated in this run."""),
code(r"""
import pandas as pd
display(pd.read_csv(OUTPUT_DIR / "all_model_comparisons_summary.csv"))
display(pd.read_csv(OUTPUT_DIR / "fold_model_summary.csv"))
"""),
md("""## 5. Finalize the downloadable evidence bundle
Save the notebook before running this cell so the executed copy can be preserved with the results. The lightweight ZIP excludes checkpoints; keep the complete output directory for Drive recovery."""),
code(r"""
import shutil
from thesis_work.run4_cross_bearing import finalize_run4_artifacts
notebook_source = UPLOAD / "Thesis_v3_with_extra_graphs_tables.ipynb"
if notebook_source.exists():
    shutil.copy2(notebook_source, OUTPUT_DIR / "executed_notebook.ipynb")
bundle = finalize_run4_artifacts(OUTPUT_DIR)
manifest = json.loads((OUTPUT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
failures = json.loads((OUTPUT_DIR / "failure_report.json").read_text(encoding="utf-8"))
print(json.dumps({"status": manifest["status"], "completed_jobs": manifest["completed_jobs"], "expected_jobs": manifest["expected_jobs"], "failures": failures, "bundle": str(bundle), "complete_output": str(OUTPUT_DIR)}, indent=2))
"""),
]
nb={"cells":cells,"metadata":{"accelerator":"GPU","colab":{"name":"Bearing_RUL_EXP_004_Run_4.ipynb","provenance":[]},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
data=json.dumps(nb,indent=1)+"\n"
for target in TARGETS:
    target.parent.mkdir(parents=True,exist_ok=True); target.write_text(data,encoding="utf-8"); print("Saved",target.relative_to(ROOT))
