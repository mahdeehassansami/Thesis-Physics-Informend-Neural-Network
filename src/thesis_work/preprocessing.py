from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from thesis_work.config import (
    EXPERIMENTS,
    FEATURE_COLS_MULTI,
    MODEL_COLS,
    TARGET_COL,
)


def preprocess_features(multi_df: pd.DataFrame) -> pd.DataFrame:
    proc_df = multi_df.copy()
    proc_df["sigma_H_norm"] = 1.0

    for run_id in proc_df["run_id"].unique():
        run = proc_df[proc_df["run_id"] == run_id].sort_values("elapsed_hours").copy()
        n_healthy = max(20, int(0.05 * len(run)))
        healthy = run.iloc[:n_healthy]
        for col in MODEL_COLS:
            base = healthy[col].median() + 1e-12
            rel = np.maximum(run[col].values / base - 1.0, 0.0)
            rel = np.log1p(rel)
            rel = pd.Series(rel, index=run.index).rolling(window=15, min_periods=1).median()
            proc_df.loc[run.index, col] = rel.values

    proc_df["elapsed_scaled"] = proc_df["elapsed_hours"] / 1200.0
    return proc_df


def make_balanced_train_df(df: pd.DataFrame, run_ids: list[str]) -> pd.DataFrame:
    run_lengths = [len(df[df["run_id"] == run_id]) for run_id in run_ids]
    n_per_run = min(run_lengths)
    pieces = []
    for run_id in run_ids:
        run_df = df[df["run_id"] == run_id].sort_values("elapsed_scaled").copy()
        sample_idx = np.linspace(0, len(run_df) - 1, n_per_run).round().astype(int)
        pieces.append(run_df.iloc[sample_idx])
    return pd.concat(pieces, ignore_index=True)


def select_runs(df: pd.DataFrame, run_ids: list[str]) -> pd.DataFrame:
    return (
        df[df["run_id"].isin(run_ids)]
        .sort_values(["run_id", "elapsed_scaled", "snapshot_index"])
        .copy()
    )


def prepare_experiment_context(proc_df: pd.DataFrame, spec: dict[str, object]) -> dict[str, object]:
    if spec.get("balanced_train", False):
        train_df = make_balanced_train_df(proc_df, spec["train_runs"])
    else:
        train_df = select_runs(proc_df, spec["train_runs"])

    val_df = select_runs(proc_df, spec["validation_runs"])
    test_df = select_runs(proc_df, spec["test_runs"])

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_df[FEATURE_COLS_MULTI]).astype("float32")
    x_val = scaler.transform(val_df[FEATURE_COLS_MULTI]).astype("float32")
    x_test = scaler.transform(test_df[FEATURE_COLS_MULTI]).astype("float32")
    x_train[:, 1:] = np.clip(x_train[:, 1:], 0, 1)
    x_val[:, 1:] = np.clip(x_val[:, 1:], 0, 1)
    x_test[:, 1:] = np.clip(x_test[:, 1:], 0, 1)

    return {
        "spec": spec,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "scaler": scaler,
        "X_train": x_train,
        "X_val": x_val,
        "X_test": x_test,
        "y_train": train_df[[TARGET_COL]].values.astype("float32"),
        "y_val": val_df[[TARGET_COL]].values.astype("float32"),
        "y_test": test_df[[TARGET_COL]].values.astype("float32"),
    }


def prepare_all_contexts(proc_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {spec["name"]: prepare_experiment_context(proc_df, spec) for spec in EXPERIMENTS}


def monotonic_increase_score(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan")
    return float(np.mean(np.diff(values) >= -1e-8))


def compute_pca_hi(proc_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pca_test_runs = sorted({run for spec in EXPERIMENTS for run in spec["test_runs"]})
    pca_hi_df = proc_df.sort_values(["run_id", "elapsed_scaled"]).reset_index(drop=True).copy()
    feature_matrix = (
        pca_hi_df[MODEL_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    )
    hi_scaled = StandardScaler().fit_transform(feature_matrix)
    hi_raw = PCA(n_components=1).fit_transform(hi_scaled).ravel()
    pca_hi = (hi_raw - hi_raw.min()) / (hi_raw.max() - hi_raw.min() + 1e-12)

    pca_hi_df["damage_norm"] = 1.0 - pca_hi_df["rul_norm"]
    if pd.Series(pca_hi).corr(pca_hi_df["damage_norm"], method="spearman") < 0:
        pca_hi = 1.0 - pca_hi

    pca_hi_df["pca_hi"] = pca_hi
    pca_hi_df["pca_hi_smooth"] = pca_hi_df.groupby("run_id")["pca_hi"].transform(
        lambda s: s.rolling(window=21, min_periods=1, center=True).median()
    )
    pca_hi_df["life_norm_for_plot"] = pca_hi_df.groupby("run_id")["elapsed_scaled"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)
    )

    rows = []
    for run_id in pca_test_runs:
        run = pca_hi_df[pca_hi_df["run_id"] == run_id].sort_values("life_norm_for_plot")
        rows.append(
            {
                "Run": run_id,
                "Monotonic increase score": monotonic_increase_score(run["pca_hi_smooth"]),
                "Spearman corr(PCA-HI, damage)": float(
                    run["pca_hi"].corr(run["damage_norm"], method="spearman")
                ),
                "Spearman corr(PCA-HI, RUL)": float(
                    run["pca_hi"].corr(run["rul_norm"], method="spearman")
                ),
            }
        )
    return pca_hi_df, pd.DataFrame(rows)
