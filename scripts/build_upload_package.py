from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
UPLOAD=ROOT/"Upload"
CACHE_SOURCE=ROOT/"data"/"processed_features"/"colab"/"ims_features.csv"
FILES=["AGENTS.md","MODEL_WORKFLOW.md","README.md","UPLOAD_INSTRUCTIONS.md","Thesis_v3_with_extra_graphs_tables.ipynb","pyproject.toml","pytest.ini","requirements-colab.txt","uv.lock"]
DIRECTORIES=["configs","notebooks","scripts","src","tests"]

def ignore_generated(_directory,names):
    return {n for n in names if n in {"__pycache__",".pytest_cache"} or n.endswith((".pyc",".pyo"))}

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

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
    (UPLOAD/"experiment_outputs_run_05").mkdir()
    config=json.loads((UPLOAD/"configs"/"colab_experiments.json").read_text(encoding="utf-8"))
    inventory=[]
    for path in sorted(UPLOAD.rglob("*")):
        if path.is_file() and path.name!="UPLOAD_PACKAGE_MANIFEST.json":
            inventory.append({"relative_path":path.relative_to(UPLOAD).as_posix(),"bytes":path.stat().st_size,"sha256":sha256(path)})
    manifest={
        "experiment_id":config["experiment"]["id"],
        "run_id":config["run_label"],
        "active_config":"configs/colab_experiments.json",
        "notebook":"Thesis_v3_with_extra_graphs_tables.ipynb",
        "feature_cache":"feature_cache/ims_features.csv",
        "feature_cache_sha256":sha256(cache/"ims_features.csv"),
        "empty_output_directory":"experiment_outputs_run_05",
        "expected_commit_state":"Notebook placeholder must be replaced after commit and push.",
        "file_count_excluding_manifest":len(inventory),
        "files":inventory,
    }
    (UPLOAD/"UPLOAD_PACKAGE_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Fresh EXP-005 Upload package prepared at {UPLOAD}")

if __name__=="__main__": main()
