from __future__ import annotations
import json
from pathlib import Path
import torch
from thesis_work.sequence_models import build_model
ROOT=Path(__file__).resolve().parents[1]

def test_experiment_config_has_distinct_physics_variants():
    config=json.loads((ROOT/"configs"/"colab_experiments.json").read_text(encoding="utf-8"))
    assert {"attnpinn","weak_pinn","strong_pinn"} <= set(config["models"])
    assert config["weight_profiles"]["weak_low"] != config["weight_profiles"]["weak_high"]
    assert config["weight_profiles"]["strong_low"] != config["weight_profiles"]["strong_high"]
    assert config["physics"]["crack_growth"]["paris_exponent"]==7.5

def test_all_model_families_produce_bounded_rul():
    config=json.loads((ROOT/"configs"/"colab_experiments.json").read_text(encoding="utf-8")); x=torch.randn(2,3,20); t=torch.tensor([[.25],[.75]])
    for name in ("fnn","cnn","lstm","attnpinn","weak_pinn","strong_pinn"):
        model=build_model(name,20,config["models"][name]); prediction=model(x,t)
        assert prediction.shape==(2,1) and torch.all((prediction>=0)&(prediction<=1))

def test_run4_has_balanced_frozen_cross_bearing_design():
    config=json.loads((ROOT/"configs"/"colab_experiments.json").read_text(encoding="utf-8"))
    assert config["experiment"]["id"]=="EXP-004" and config["run_label"]=="run_04"
    assert config["training"]["seeds"]==[42,1042,2042] and config["cross_bearing"]["expected_jobs"]==36
    folds=config["cross_bearing"]["folds"]; runs=set(config["cross_bearing"]["all_runs"])
    assert len(folds)==4
    assert sorted(next(iter(f["test_runs"])) for f in folds)==sorted(runs)
    assert sorted(next(iter(f["validation_runs"])) for f in folds)==sorted(runs)
    enabled=[n for n,m in config["models"].items() if m.get("enabled")]
    assert enabled==["lstm","weak_pinn","strong_pinn"]
    assert config["models"]["strong_pinn"]["profiles"]==["strong_paris_0p003_miner_0p003"]
