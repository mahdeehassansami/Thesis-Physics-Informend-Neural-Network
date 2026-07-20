from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from thesis_work.multi_dataset import load_or_extract_dataset, prepare_sequence_dataset
from thesis_work.sequence_models import build_model, calculate_loss
ROOT=Path(__file__).resolve().parents[1]
UPLOAD=ROOT/"Upload"

def main():
    config=json.loads((UPLOAD/"configs"/"colab_experiments.json").read_text(encoding="utf-8"))
    assert config["experiment"]["id"]=="EXP-004" and config["run_label"]=="run_04"
    dataset=next(d for d in config["datasets"] if d.get("enabled"))
    frame=load_or_extract_dataset(dataset,project_root=UPLOAD,cache_dir=UPLOAD/"feature_cache")
    fold=config["cross_bearing"]["folds"][0]; dc=json.loads(json.dumps(dataset)); dc["split"]={"strategy":"run_ids","train_runs":fold["train_runs"],"validation_runs":fold["validation_runs"],"test_runs":fold["test_runs"]}
    prepared=prepare_sequence_dataset(frame,dc,sequence_length=config["training"]["sequence_length"])
    batch=next(iter(DataLoader(prepared.train,batch_size=4,shuffle=False)))
    indices={name:i for i,name in enumerate(prepared.feature_columns)}
    for name,mc in config["models"].items():
        if not mc.get("enabled"): continue
        profile=mc["profiles"][0]; model=build_model(name,input_dim=len(prepared.feature_columns),model_config=mc)
        total,components,prediction=calculate_loss(model,batch,config["weight_profiles"][profile],indices,config["physics"])
        assert torch.isfinite(total) and prediction.shape==batch["target"].shape
        total.backward(); print(name,profile,float(total.detach()),sorted(components))
    print("Run 4 Upload package validation passed.")

if __name__=="__main__": main()
