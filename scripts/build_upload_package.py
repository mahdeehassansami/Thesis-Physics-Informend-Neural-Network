from __future__ import annotations
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
UPLOAD=ROOT/"Upload"
CACHE_SOURCE=ROOT/"data"/"processed_features"/"colab"/"ims_features.csv"
FILES=["AGENTS.md","MODEL_WORKFLOW.md","README.md","UPLOAD_INSTRUCTIONS.md","Thesis_v3_with_extra_graphs_tables.ipynb","pyproject.toml","pytest.ini","requirements-colab.txt","uv.lock"]
DIRECTORIES=["configs","notebooks","scripts","src","tests"]

def ignore_generated(_directory,names):
    return {n for n in names if n in {"__pycache__",".pytest_cache"} or n.endswith((".pyc",".pyo"))}

def main():
    resolved=UPLOAD.resolve()
    if resolved.parent != ROOT.resolve() or resolved.name != "Upload":
        raise RuntimeError(f"Refusing unsafe Upload target: {resolved}")
    if not CACHE_SOURCE.is_file(): raise FileNotFoundError(CACHE_SOURCE)
    if UPLOAD.exists(): shutil.rmtree(UPLOAD)
    UPLOAD.mkdir(parents=True)
    for rel in FILES:
        src=ROOT/rel
        if src.is_file(): shutil.copy2(src,UPLOAD/rel)
    for rel in DIRECTORIES:
        shutil.copytree(ROOT/rel,UPLOAD/rel,ignore=ignore_generated)
    cache=UPLOAD/"feature_cache"; cache.mkdir()
    shutil.copy2(CACHE_SOURCE,cache/"ims_features.csv")
    (UPLOAD/"experiment_outputs_run_04").mkdir()
    print(f"Fresh EXP-004 Upload package prepared at {UPLOAD}")

if __name__=="__main__": main()
