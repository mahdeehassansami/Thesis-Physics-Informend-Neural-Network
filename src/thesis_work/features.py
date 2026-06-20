from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import get_window, hilbert
from tqdm.auto import tqdm

from thesis_work.config import FS, ProjectPaths, expected_cols
from thesis_work.ims import (
    FAULT_FREQS,
    channel_to_bearing,
    list_snapshots,
    load_snapshot,
    parse_ims_timestamp,
    resolve_dataset_source,
)


def basic_time_features(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    centered = x - np.mean(x)
    rms = np.sqrt(np.mean(x**2))
    m2 = np.mean(centered**2)
    m4 = np.mean(centered**4)
    return {
        "rms": float(rms),
        "std": float(np.std(centered)),
        "ptp": float(np.ptp(x)),
        "kurtosis": float(m4 / (m2**2 + 1e-12)),
        "crest_factor": float(np.max(np.abs(x)) / (rms + 1e-12)),
        "mean_abs": float(np.mean(np.abs(x))),
    }


def spectrum_features(x: np.ndarray, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    x = x - np.mean(x)
    envelope = np.abs(hilbert(x))
    envelope = envelope - np.mean(envelope)
    window = get_window("hann", len(envelope))
    spec = np.abs(np.fft.rfft(envelope * window))
    freqs = np.fft.rfftfreq(len(envelope), d=1 / fs)
    return freqs, spec


def band_energy(
    freqs: np.ndarray,
    spec: np.ndarray,
    center: float,
    bandwidth: float = 5.0,
    harmonics: int = 4,
) -> float:
    total = 0.0
    for harmonic in range(1, harmonics + 1):
        f0 = harmonic * center
        mask = (freqs >= f0 - bandwidth) & (freqs <= f0 + bandwidth)
        total += float(np.sum(spec[mask] ** 2))
    return total


def extract_channel_features(signal: np.ndarray) -> dict[str, float]:
    feats = basic_time_features(signal)
    freqs, spec = spectrum_features(signal)
    for name, f0 in FAULT_FREQS.items():
        feats[f"E_{name}"] = band_energy(freqs, spec, f0, bandwidth=5.0, harmonics=4)
    feats["E_kin"] = feats["E_FTF"] + feats["E_BPFO"] + feats["E_BPFI"] + feats["E_BSF"]
    return feats


def build_feature_table_for_target(
    paths: ProjectPaths,
    dataset: str,
    target_bearing: int,
    max_files: int | None = None,
) -> pd.DataFrame:
    source = resolve_dataset_source(paths.raw_data, dataset)
    files = list_snapshots(source)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise ValueError(f"No IMS snapshot files found in {source}.")

    n_cols = expected_cols(dataset)
    t0 = parse_ims_timestamp(files[0])
    tf = parse_ims_timestamp(files[-1])
    rows: list[dict[str, object]] = []

    for idx, inner in enumerate(tqdm(files, desc=f"{dataset} bearing {target_bearing}")):
        ts = parse_ims_timestamp(inner)
        arr = load_snapshot(source, inner, n_cols)
        for ch in range(n_cols):
            bearing, axis = channel_to_bearing(dataset, ch)
            if bearing != target_bearing:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "file": Path(inner).name,
                    "timestamp": ts,
                    "snapshot_index": idx,
                    "elapsed_min": (ts - t0).total_seconds() / 60,
                    "rul_min": (tf - ts).total_seconds() / 60,
                    "bearing": bearing,
                    "axis": axis,
                    **extract_channel_features(arr[:, ch]),
                }
            )

    df = pd.DataFrame(rows)
    group_cols = [
        "dataset",
        "file",
        "timestamp",
        "snapshot_index",
        "elapsed_min",
        "rul_min",
        "bearing",
    ]
    numeric_cols = [c for c in df.columns if c not in group_cols + ["axis"]]
    df = df.groupby(group_cols, as_index=False)[numeric_cols].mean()
    df["time_norm"] = df["elapsed_min"] / (df["elapsed_min"].max() + 1e-12)
    df["rul_norm"] = df["rul_min"] / (df["rul_min"].max() + 1e-12)
    return df


def load_or_extract_run(
    paths: ProjectPaths,
    run_id: str,
    dataset: str,
    bearing: int,
    max_files: int | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    suffix = f"_first_{max_files}" if max_files is not None else ""
    cache_path = paths.processed_features / f"{run_id}{suffix}_features.csv"
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path)
    else:
        df = build_feature_table_for_target(paths, dataset, bearing, max_files=max_files)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    df["run_id"] = run_id
    df["elapsed_hours"] = df["elapsed_min"] / 60.0
    return df
