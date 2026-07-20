from __future__ import annotations
import copy, hashlib, json, time, traceback, zipfile
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd, torch
from thesis_work.experiment_runner import train_one_model
from thesis_work.multi_dataset import enabled_dataset_configs, load_or_extract_dataset, prepare_sequence_dataset
from thesis_work.run3_calibration import _best_epoch_physics_diagnostics, _environment, _git_state, _sha256, _training_seeds

EXCLUDED={"run_manifest.json","artifact_inventory.csv","codex_results_bundle.zip"}

def _utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def _source(root):
    paths=[root/"AGENTS.md",root/"MODEL_WORKFLOW.md",root/"README.md",root/"UPLOAD_INSTRUCTIONS.md",root/"pyproject.toml",root/"requirements-colab.txt",root/"Thesis_v3_with_extra_graphs_tables.ipynb",*(root/"configs").glob("*.json"),*(root/"src"/"thesis_work").glob("*.py")]
    rows=[{"relative_path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":_sha256(p)} for p in paths if p.is_file()]
    frame=pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)
    combined=hashlib.sha256("\n".join(f"{r.relative_path}:{r.sha256}" for r in frame.itertuples(index=False)).encode()).hexdigest()
    return frame,combined

def _inventory(root):
    rows=[]
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in EXCLUDED:
            rows.append({"relative_path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":_sha256(p)})
    return pd.DataFrame(rows)

def finalize_run4_artifacts(root):
    root=Path(root); inv=_inventory(root); inv.to_csv(root/"artifact_inventory.csv",index=False)
    mp=root/"run_manifest.json"; m=json.loads(mp.read_text(encoding="utf-8"))
    m["artifact_count_excluding_manifest_inventory_bundle"]=len(inv)
    nb=root/"executed_notebook.ipynb"; m["executed_notebook_sha256"]=_sha256(nb) if nb.exists() else None
    mp.write_text(json.dumps(m,indent=2),encoding="utf-8")
    bundle=root/"codex_results_bundle.zip"
    if bundle.exists(): bundle.unlink()
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p != bundle and p.name != "checkpoint.pt":
                z.write(p,p.relative_to(root).as_posix())
    return bundle

def validate_run4_runtime(config, root):
    env=_environment(); runtime=config.get("runtime",{})
    if runtime.get("require_cuda",True) and not env["cuda_available"]:
        raise RuntimeError("Run 4 requires a CUDA GPU; select a Colab T4 runtime.")
    needed=runtime.get("required_gpu_name_contains")
    if needed and needed.lower() not in str(env.get("gpu_name","")).lower():
        raise RuntimeError(f"Run 4 requires {needed}; detected {env.get('gpu_name')}.")
    git=_git_state(Path(root))
    expected=config.get("repository",{}).get("expected_commit")
    if runtime.get("require_expected_commit",True):
        if not isinstance(expected,str) or len(expected)!=40: raise RuntimeError("Set repository.expected_commit to the committed SHA.")
        if git["commit"] != expected: raise RuntimeError(f"Expected {expected}, detected {git['commit']}.")
    if runtime.get("require_clean_git",True) and git.get("dirty") is not False:
        raise RuntimeError("Run 4 requires a clean Git checkout.")
    return env,git

def _folds(config, dataset, frame):
    cross=config["cross_bearing"]; expected=set(cross["all_runs"])
    if set(frame.run_id.unique()) != expected: raise ValueError("IMS cache run IDs do not match config.")
    folds=copy.deepcopy(cross["folds"]); tests=[]; vals=[]
    for f in folds:
        tr=set(f["train_runs"]); va=set(f["validation_runs"]); te=set(f["test_runs"])
        if len(tr)!=2 or len(va)!=1 or len(te)!=1 or tr&va or tr&te or va&te or tr|va|te != expected:
            raise ValueError(f"Invalid fold {f['fold_id']}")
        tests.append(next(iter(te))); vals.append(next(iter(va)))
    if sorted(tests)!=sorted(expected) or sorted(vals)!=sorted(expected): raise ValueError("Each run must be test and validation exactly once.")
    return folds

def _models(config):
    expected=["lstm","weak_pinn","strong_pinn"]; out=[]
    for name in expected:
        mc=config["models"][name]
        if not mc.get("enabled") or len(mc.get("profiles",[]))!=1: raise ValueError(f"Run 4 requires one frozen profile for {name}.")
        profile=mc["profiles"][0]
        out.append((name,profile,mc,config["weight_profiles"][profile]))
    return out

def _annotate(path, config, fold, model, profile, repeat, seed, role):
    f=pd.read_csv(path)
    meta={"experiment_id":config["experiment"]["id"],"run_id":config["run_label"],"dataset":"ims","fold_id":fold["fold_id"],"model":model,"weight_profile":profile,"seed_repeat":repeat,"seed":seed,"checkpoint_role":role}
    for k,v in reversed(list(meta.items())):
        if k in f.columns: f[k]=v
        else: f.insert(0,k,v)
    f["absolute_error"]=(f["target_rul"]-f["predicted_rul"]).abs()
    f["absolute_error_seconds"]=(f["target_rul_seconds"]-f["predicted_rul_seconds"]).abs()
    f.to_csv(path,index=False); return f

def _lifecycle(f):
    t=f.target_rul.to_numpy(); p=f.predicted_rul.to_numpy()
    phase=np.where(t>2/3,"early",np.where(t>1/3,"middle","late")); rows=[]
    for name in ("early","middle","late"):
        m=phase==name
        rows.append({"phase":name,"samples":int(m.sum()),"mae":float(np.mean(np.abs(p[m]-t[m]))),"bias":float(np.mean(p[m]-t[m])),"prediction_mean":float(np.mean(p[m])),"target_mean":float(np.mean(t[m]))})
    return pd.DataFrame(rows)

def _fold_summary(r):
    ok=r[r.status=="ok"]
    if ok.empty:return pd.DataFrame()
    s=ok.groupby(["dataset","fold_id","test_run_id","model","weight_profile"],as_index=False).agg(seed_repeats=("seed","nunique"),training_seconds_total=("seconds","sum"),mae_mean=("mae","mean"),rmse_mean=("rmse","mean"),rmse_std=("rmse","std"),r2_mean=("r2","mean"),rmse_seconds_mean=("rmse_seconds","mean"),late_life_mae_mean=("late_life_mae","mean"),late_life_bias_mean=("late_life_bias","mean"),best_epoch_mean=("best_epoch","mean"),final_epoch_mean=("final_epoch","mean"),parameter_count=("parameter_count","first"))
    s["fold_rank_by_rmse"]=s.groupby("fold_id").rmse_mean.rank(method="min").astype(int)
    return s.sort_values(["fold_id","rmse_mean"])

def _aggregate(s):
    if s.empty:return pd.DataFrame()
    a=s.groupby(["dataset","model","weight_profile"],as_index=False).agg(folds_completed=("fold_id","nunique"),seed_runs=("seed_repeats","sum"),fold_wins=("fold_rank_by_rmse",lambda x:int((x==1).sum())),macro_mae_mean=("mae_mean","mean"),macro_rmse_mean=("rmse_mean","mean"),between_bearing_rmse_std=("rmse_mean","std"),worst_bearing_rmse=("rmse_mean","max"),macro_r2_mean=("r2_mean","mean"),macro_rmse_seconds_mean=("rmse_seconds_mean","mean"),macro_late_life_mae=("late_life_mae_mean","mean"),worst_abs_late_life_bias=("late_life_bias_mean",lambda x:float(np.abs(x).max())),training_seconds_total=("training_seconds_total","sum"),parameter_count=("parameter_count","first")).sort_values("macro_rmse_mean").reset_index(drop=True)
    a["rank_by_macro_rmse"]=np.arange(1,len(a)+1); return a

def run_run4_experiment(config, project_root, cache_dir, output_root, refresh_features=False):
    project_root=Path(project_root).resolve(); cache_dir=Path(cache_dir).resolve(); output_root=Path(output_root).resolve()
    if config.get("run_label")!="run_04" or config["experiment"]["id"]!="EXP-004": raise ValueError("Run 4 requires EXP-004/run_04.")
    enabled=enabled_dataset_configs(config)
    if [d["name"] for d in enabled]!=["ims"]: raise ValueError("Run 4 must enable IMS only.")
    env,git=validate_run4_runtime(config,project_root); dataset=enabled[0]; models=_models(config); seeds=_training_seeds(config)
    frame=load_or_extract_dataset(dataset,project_root,cache_dir,refresh_features)
    cache=cache_dir/"ims_features.csv"; cache_hash=_sha256(cache)
    if cache_hash!=config["cross_bearing"]["expected_feature_cache_sha256"]: raise RuntimeError("IMS feature-cache hash mismatch.")
    folds=_folds(config,dataset,frame); split={"dataset":"ims","strategy":"fixed_cross_bearing_folds","folds":folds}
    clean=copy.deepcopy(config); clean.pop("_config_path",None); config_hash=_hash(clean); split_hash=_hash(split); source,source_hash=_source(project_root)
    identity={"experiment_id":"EXP-004","run_id":"run_04","git_commit":git["commit"],"source_tree_sha256":source_hash,"config_hash":config_hash,"split_hash":split_hash,"cache_hash":cache_hash}
    output_root.mkdir(parents=True,exist_ok=True); state_path=output_root/"run_state.json"
    if any(output_root.iterdir()):
        if not state_path.exists(): raise FileExistsError("Non-empty Run 4 output lacks run_state.json.")
        state=json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity")!=identity: raise RuntimeError("Run 4 resume identity mismatch.")
        if (output_root/"run_manifest.json").exists():
            return pd.read_csv(output_root/"all_model_comparisons.csv"),pd.read_csv(output_root/"fold_model_summary.csv"),pd.read_csv(output_root/"all_model_comparisons_summary.csv")
    else:
        state={"identity":identity,"started_utc":_utc_now(),"jobs_recorded":0}
        state_path.write_text(json.dumps(state,indent=2),encoding="utf-8")
    (output_root/"resolved_config.json").write_text(json.dumps(clean,indent=2),encoding="utf-8")
    (output_root/"data_split.json").write_text(json.dumps(split,indent=2),encoding="utf-8")
    source.to_csv(output_root/"source_manifest.csv",index=False)
    (output_root/"environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
    (output_root/"environment.txt").write_text("\n".join(f"{k}: {v}" for k,v in env.items())+"\n",encoding="utf-8")
    (output_root/"git_commit.txt").write_text(git["commit"]+"\n",encoding="utf-8")
    rows=[]; fold_info=[]; log=output_root/"training.log"
    for fold in folds:
        fc=copy.deepcopy(dataset); fc["split"]={"strategy":"run_ids","train_runs":fold["train_runs"],"validation_runs":fold["validation_runs"],"test_runs":fold["test_runs"]}
        prepared=prepare_sequence_dataset(frame,fc,int(config["training"]["sequence_length"]))
        root=output_root/"ims"/"folds"/fold["fold_id"]; root.mkdir(parents=True,exist_ok=True)
        (root/"preprocessing.json").write_text(json.dumps({"fit_split":"train","train_runs":fold["train_runs"],"validation_runs":fold["validation_runs"],"test_runs":fold["test_runs"],"feature_columns":prepared.feature_columns,"scaler_mean":prepared.scaler.mean_.tolist(),"scaler_scale":prepared.scaler.scale_.tolist(),"sequence_length":config["training"]["sequence_length"],"time_scale_seconds":prepared.time_scale_seconds},indent=2),encoding="utf-8")
        fold_info.append({"fold_id":fold["fold_id"],"train_runs":fold["train_runs"],"validation_runs":fold["validation_runs"],"test_runs":fold["test_runs"],"train_sequences":len(prepared.train),"validation_sequences":len(prepared.validation),"test_sequences":len(prepared.test),"time_scale_seconds":prepared.time_scale_seconds})
        for model,profile,mc,weights in models:
            for repeat,seed in enumerate(seeds,1):
                ad=root/f"{model}__{profile}__seed_{repeat:02d}"; ad.mkdir(parents=True,exist_ok=True)
                jid={"fold_id":fold["fold_id"],"model":model,"profile":profile,"seed_repeat":repeat,"seed":seed,"identity":identity}; jhash=_hash(jid); jp=ad/"job_result.json"
                if jp.exists():
                    saved=json.loads(jp.read_text(encoding="utf-8"))
                    if saved.get("job_hash")!=jhash: raise RuntimeError("Resume job identity mismatch.")
                    rows.append(saved["result"]); continue
                started=time.time(); log.open("a",encoding="utf-8").write(f"{_utc_now()} START {fold['fold_id']} {model} seed={seed}\n")
                try:
                    net,history,pred,metrics=train_one_model(prepared,model,mc,weights,config["physics"],config["training"],seed,ad,evaluation_split="test",save_final_evaluation=True)
                    best=_annotate(ad/"predictions.csv",config,fold,model,profile,repeat,seed,"best_validation")
                    for fn,role in [("validation_predictions.csv","best_validation"),("final_predictions.csv","final_epoch"),("final_validation_predictions.csv","final_epoch")]:
                        if (ad/fn).exists(): _annotate(ad/fn,config,fold,model,profile,repeat,seed,role)
                    life=_lifecycle(best); life.to_csv(ad/"lifecycle_metrics.csv",index=False); late=life[life.phase=="late"].iloc[0]
                    row={"dataset":"ims","fold_id":fold["fold_id"],"train_run_ids":"|".join(fold["train_runs"]),"validation_run_id":fold["validation_runs"][0],"test_run_id":fold["test_runs"][0],"model":model,"weight_profile":profile,"seed_repeat":repeat,"seed":seed,"status":"ok","seconds":time.time()-started,"late_life_mae":float(late.mae),"late_life_bias":float(late.bias),"artifact_directory":ad.relative_to(output_root).as_posix(),**metrics,**_best_epoch_physics_diagnostics(history,weights)}
                    del net
                except Exception as exc:
                    (ad/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
                    row={"dataset":"ims","fold_id":fold["fold_id"],"train_run_ids":"|".join(fold["train_runs"]),"validation_run_id":fold["validation_runs"][0],"test_run_id":fold["test_runs"][0],"model":model,"weight_profile":profile,"seed_repeat":repeat,"seed":seed,"status":"failed","seconds":time.time()-started,"error":str(exc),"artifact_directory":ad.relative_to(output_root).as_posix()}
                jp.write_text(json.dumps({"job_hash":jhash,"result":row},indent=2),encoding="utf-8"); rows.append(row)
                log.open("a",encoding="utf-8").write(f"{_utc_now()} {row['status'].upper()} {fold['fold_id']} {model} seed={seed}\n")
                pd.DataFrame(rows).to_csv(output_root/"partial_model_comparisons.csv",index=False)
                state["jobs_recorded"]=len(rows); state["last_updated_utc"]=_utc_now(); state_path.write_text(json.dumps(state,indent=2),encoding="utf-8")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        fr=pd.DataFrame([r for r in rows if r["fold_id"]==fold["fold_id"]]); fr.to_csv(root/"model_comparison.csv",index=False); _fold_summary(fr).to_csv(root/"model_comparison_summary.csv",index=False)
    results=pd.DataFrame(rows); fs=_fold_summary(results); agg=_aggregate(fs)
    results.to_csv(output_root/"all_model_comparisons.csv",index=False); fs.to_csv(output_root/"fold_model_summary.csv",index=False); agg.to_csv(output_root/"all_model_comparisons_summary.csv",index=False)
    failures=results[results.status!="ok"].to_dict(orient="records"); (output_root/"failure_report.json").write_text(json.dumps({"failed_jobs":failures,"failure_files":[p.relative_to(output_root).as_posix() for p in output_root.rglob("failure.txt")]},indent=2),encoding="utf-8")
    (output_root/"dataset_summary.json").write_text(json.dumps({"dataset":"ims","feature_cache_sha256":cache_hash,"feature_rows":len(frame),"run_ids":sorted(frame.run_id.unique()),"folds":fold_info},indent=2),encoding="utf-8")
    (output_root/"ims"/"assumptions.json").write_text(json.dumps({"physics_assumptions":dataset.get("physics_assumptions",[]),"operating_conditions":dataset.get("operating_conditions",{})},indent=2),encoding="utf-8")
    expected=int(config["cross_bearing"]["expected_jobs"]); done=int((results.status=="ok").sum()); status="completed" if done==expected and not failures else "partial"
    manifest={"experiment_id":"EXP-004","experiment_name":config["experiment"]["name"],"run_id":"run_04","status":status,"started_utc":state["started_utc"],"finished_utc":_utc_now(),"git":git,"source_tree_sha256":source_hash,"source_file_count":len(source),"resolved_config_sha256":_sha256(output_root/"resolved_config.json"),"data_split_sha256":_sha256(output_root/"data_split.json"),"dataset_feature_cache_sha256":cache_hash,"folds":folds,"seeds":seeds,"requested_models":[{"model":m,"weight_profile":p} for m,p,_,_ in models],"expected_jobs":expected,"completed_jobs":done,"failed_jobs":len(failures),"environment":env,"primary_aggregation":config["cross_bearing"]["primary_aggregation"],"test_access_policy":config["cross_bearing"]["test_policy"],"checkpoint_policy":"Validation controls scheduler/early stopping; best-validation and final-epoch test metrics are both recorded without model changes.","failure_files":[p.relative_to(output_root).as_posix() for p in output_root.rglob("failure.txt")]}
    (output_root/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (output_root/"summary.md").write_text(f"# EXP-004 Run 4 summary\n\nStatus: {status}\nCompleted jobs: {done}/{expected}\n\nFixed four-fold held-out-bearing evaluation; test metrics were not used for tuning.\n",encoding="utf-8")
    finalize_run4_artifacts(output_root); return results,fs,agg
