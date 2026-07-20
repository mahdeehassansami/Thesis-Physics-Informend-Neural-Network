from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import get_window
from sklearn.preprocessing import StandardScaler

from thesis_work.ims import (
    channel_to_bearing,
    list_snapshots,
    load_snapshot,
    parse_ims_timestamp,
)


SIGNAL_FEATURES = [
    "rms",
    "std",
    "ptp",
    "kurtosis",
    "crest_factor",
    "mean_abs",
    "skewness",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_entropy",
    "high_frequency_ratio",
]

MODEL_FEATURES = [
    *SIGNAL_FEATURES,
    "temperature_c",
    "ambient_temperature_c",
    "temperature_delta_c",
    "load_n",
    "speed_rpm",
    "temperature_available",
    "load_available",
    "contact_pressure_available",
]

PHYSICS_COLUMNS = [
    "temperature_c",
    "ambient_temperature_c",
    "temperature_delta_c",
    "load_n",
    "speed_rpm",
    "contact_pressure_mpa",
    "dynamic_capacity_n",
    "fatigue_limit_n",
    "viscosity_ref_cst",
    "viscosity_required_cst",
    "contamination_factor",
    "cycles_per_time_unit",
    "temperature_available",
    "load_available",
    "contact_pressure_available",
]


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(path.resolve())
    return config


def _natural_key(path: str | Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", Path(path).name)
    ]


def _safe_numeric_frame(path: Path, usecols: list[int] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, usecols=usecols, low_memory=False)
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if frame.empty:
        raise ValueError(f"No numeric signal rows were found in {path}.")
    return frame


def extract_signal_features(signal: np.ndarray, sampling_hz: float) -> dict[str, float]:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        raise ValueError("At least eight finite signal samples are required.")

    mean = float(np.mean(x))
    centered = x - mean
    rms = float(np.sqrt(np.mean(x**2)))
    std = float(np.std(centered))
    m2 = float(np.mean(centered**2))
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))

    windowed = centered * get_window("hann", len(centered))
    power = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sampling_hz)
    power_sum = float(power.sum()) + 1e-12
    probability = power / power_sum
    centroid = float(np.sum(freqs * probability))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * probability)))
    entropy = float(-np.sum(probability * np.log(probability + 1e-12)))
    entropy /= math.log(max(2, len(probability)))
    high_frequency_ratio = float(
        power[freqs >= 0.25 * (sampling_hz / 2.0)].sum() / power_sum
    )

    return {
        "rms": rms,
        "std": std,
        "ptp": float(np.ptp(x)),
        "kurtosis": m4 / (m2**2 + 1e-12),
        "crest_factor": float(np.max(np.abs(x)) / (rms + 1e-12)),
        "mean_abs": float(np.mean(np.abs(x))),
        "skewness": m3 / (m2**1.5 + 1e-12),
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_entropy": entropy,
        "high_frequency_ratio": high_frequency_ratio,
    }


def _mean_axis_features(signals: list[np.ndarray], sampling_hz: float) -> dict[str, float]:
    features = [extract_signal_features(signal, sampling_hz) for signal in signals]
    return {
        name: float(np.mean([axis_features[name] for axis_features in features]))
        for name in SIGNAL_FEATURES
    }


def _metadata_values(dataset_config: dict[str, Any]) -> dict[str, float]:
    operating = dataset_config.get("operating_conditions", {})
    values = {
        "temperature_c": float("nan"),
        "ambient_temperature_c": float("nan"),
        "temperature_delta_c": float("nan"),
        "load_n": operating.get("load_n", float("nan")),
        "speed_rpm": operating.get("speed_rpm", float("nan")),
        "contact_pressure_mpa": operating.get("contact_pressure_mpa", float("nan")),
        "dynamic_capacity_n": operating.get("dynamic_capacity_n", float("nan")),
        "fatigue_limit_n": operating.get("fatigue_limit_n", float("nan")),
        "viscosity_ref_cst": operating.get("viscosity_ref_cst", float("nan")),
        "viscosity_required_cst": operating.get(
            "viscosity_required_cst", float("nan")
        ),
        "contamination_factor": operating.get(
            "contamination_factor", float("nan")
        ),
    }
    return {name: float(value) for name, value in values.items()}


def _finalize_feature_frame(
    frame: pd.DataFrame, dataset_config: dict[str, Any]
) -> pd.DataFrame:
    frame = frame.sort_values(["run_id", "sample_index"]).reset_index(drop=True).copy()
    metadata = _metadata_values(dataset_config)
    for column, default in metadata.items():
        if column not in frame:
            frame[column] = default
        else:
            frame[column] = frame[column].fillna(default)

    frame["elapsed_norm"] = frame.groupby("run_id")["elapsed_seconds"].transform(
        lambda series: (series - series.min())
        / (series.max() - series.min() + 1e-12)
    )
    frame["rul_norm"] = 1.0 - frame["elapsed_norm"]

    frame["temperature_delta_c"] = (
        frame["temperature_c"] - frame["ambient_temperature_c"]
    )
    frame["temperature_available"] = frame["temperature_c"].notna().astype(float)
    frame["load_available"] = frame["load_n"].notna().astype(float)
    frame["contact_pressure_available"] = (
        frame["contact_pressure_mpa"].notna().astype(float)
    )

    speed = frame["speed_rpm"].fillna(0.0)
    run_duration = frame.groupby("run_id")["elapsed_seconds"].transform("max")
    frame["total_cycles"] = run_duration * speed / 60.0

    indicator_columns = ["rms", "kurtosis", "crest_factor", "high_frequency_ratio"]
    frame["health_indicator"] = 0.0
    for _, indices in frame.groupby("run_id", sort=False).groups.items():
        run = frame.loc[indices].sort_values("sample_index")
        components = []
        healthy_count = max(3, int(round(0.1 * len(run))))
        for column in indicator_columns:
            values = run[column].replace([np.inf, -np.inf], np.nan)
            values = values.interpolate(limit_direction="both").fillna(0.0)
            baseline = float(values.iloc[:healthy_count].median())
            scale = float(values.quantile(0.95) - baseline)
            component = np.clip((values.to_numpy() - baseline) / (abs(scale) + 1e-12), 0, 1)
            components.append(component)
        raw_indicator = np.mean(np.vstack(components), axis=0)
        smoothed = (
            pd.Series(raw_indicator)
            .rolling(window=7, min_periods=1, center=False)
            .median()
            .cummax()
            .clip(0, 1)
            .to_numpy()
        )
        frame.loc[run.index, "health_indicator"] = smoothed

    for column in MODEL_FEATURES + PHYSICS_COLUMNS:
        if column not in frame:
            frame[column] = 0.0

    model_fill = {
        "temperature_c": 0.0,
        "ambient_temperature_c": 0.0,
        "temperature_delta_c": 0.0,
        "load_n": 0.0,
        "speed_rpm": 0.0,
    }
    for column, value in model_fill.items():
        frame[column] = frame[column].fillna(value)
    return frame


def extract_ims_dataset(
    dataset_config: dict[str, Any], project_root: Path
) -> pd.DataFrame:
    root = (project_root / dataset_config["path"]).resolve()
    sampling_hz = float(dataset_config.get("sampling_hz", 20_000))
    rows: list[dict[str, Any]] = []

    for run in dataset_config["runs"]:
        source = root / run["folder"]
        files = list_snapshots(source)
        if not files:
            raise ValueError(f"No IMS snapshots found in {source}.")
        first_time = parse_ims_timestamp(files[0])
        expected_columns = 8 if run["folder"] == "1st_test" else 4
        for sample_index, filename in enumerate(files):
            timestamp = parse_ims_timestamp(filename)
            snapshot = load_snapshot(source, filename, expected_columns)
            axes = []
            for channel_index in range(expected_columns):
                bearing, _ = channel_to_bearing(run["folder"], channel_index)
                if bearing == int(run["bearing"]):
                    axes.append(snapshot[:, channel_index])
            if not axes:
                raise ValueError(
                    f"No channels for bearing {run['bearing']} in {source / filename}."
                )
            rows.append(
                {
                    "dataset": dataset_config["name"],
                    "run_id": run["run_id"],
                    "sample_index": sample_index,
                    "elapsed_seconds": (timestamp - first_time).total_seconds(),
                    **_mean_axis_features(axes, sampling_hz),
                }
            )
    return _finalize_feature_frame(pd.DataFrame(rows), dataset_config)


def extract_pronostia_dataset(
    dataset_config: dict[str, Any], project_root: Path
) -> pd.DataFrame:
    root = (project_root / dataset_config["path"]).resolve()
    sampling_hz = float(dataset_config.get("sampling_hz", 25_600))
    interval_seconds = float(dataset_config.get("snapshot_interval_seconds", 10))
    rows: list[dict[str, Any]] = []

    for run in dataset_config["runs"]:
        run_path = root / run["relative_path"]
        files = sorted(run_path.glob("acc_*.csv"), key=_natural_key)
        if not files:
            raise FileNotFoundError(f"No PRONOSTIA acc_*.csv files found in {run_path}.")
        for sample_index, path in enumerate(files):
            raw = _safe_numeric_frame(path)
            signals = [raw.iloc[:, -2].to_numpy(), raw.iloc[:, -1].to_numpy()]
            rows.append(
                {
                    "dataset": dataset_config["name"],
                    "run_id": run["run_id"],
                    "sample_index": sample_index,
                    "elapsed_seconds": sample_index * interval_seconds,
                    **_mean_axis_features(signals, sampling_hz),
                }
            )
    return _finalize_feature_frame(pd.DataFrame(rows), dataset_config)


def _stream_kaist_file(
    path: Path,
    sampling_hz: float,
    chunksize: int,
    samples_per_chunk: int,
) -> tuple[dict[str, float], float, float]:
    sampled_x: list[np.ndarray] = []
    sampled_y: list[np.ndarray] = []
    bearing_temperature_sum = 0.0
    ambient_temperature_sum = 0.0
    temperature_count = 0

    for chunk in pd.read_csv(path, header=None, chunksize=chunksize, low_memory=False):
        chunk = chunk.iloc[:, :4].apply(pd.to_numeric, errors="coerce").dropna()
        if chunk.empty:
            continue
        stride = max(1, len(chunk) // samples_per_chunk)
        sampled = chunk.iloc[::stride].iloc[:samples_per_chunk]
        sampled_x.append(sampled.iloc[:, 0].to_numpy(dtype=np.float64))
        sampled_y.append(sampled.iloc[:, 1].to_numpy(dtype=np.float64))
        bearing_temperature_sum += float(chunk.iloc[:, 2].sum())
        ambient_temperature_sum += float(chunk.iloc[:, 3].sum())
        temperature_count += len(chunk)

    if not sampled_x or temperature_count == 0:
        raise ValueError(f"No usable four-column data found in {path}.")
    signals = [np.concatenate(sampled_x), np.concatenate(sampled_y)]
    return (
        _mean_axis_features(signals, sampling_hz),
        bearing_temperature_sum / temperature_count,
        ambient_temperature_sum / temperature_count,
    )


def extract_kaist_temperature_dataset(
    dataset_config: dict[str, Any], project_root: Path
) -> pd.DataFrame:
    root = (project_root / dataset_config["path"]).resolve()
    files = sorted(root.glob("*.csv"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No vibration/temperature CSV files found in {root}.")

    sampling_hz = float(dataset_config.get("sampling_hz", 25_600))
    interval_seconds = float(dataset_config.get("snapshot_interval_seconds", 3600))
    chunksize = int(dataset_config.get("csv_chunksize", 500_000))
    samples_per_chunk = int(dataset_config.get("fft_samples_per_chunk", 4096))
    run_id = dataset_config.get("run_id", "kaist_run_1")
    rows = []
    for sample_index, path in enumerate(files):
        features, bearing_temperature, ambient_temperature = _stream_kaist_file(
            path, sampling_hz, chunksize, samples_per_chunk
        )
        rows.append(
            {
                "dataset": dataset_config["name"],
                "run_id": run_id,
                "sample_index": sample_index,
                "elapsed_seconds": sample_index * interval_seconds,
                "temperature_c": bearing_temperature,
                "ambient_temperature_c": ambient_temperature,
                **features,
            }
        )
    return _finalize_feature_frame(pd.DataFrame(rows), dataset_config)


def load_standardized_csv(
    dataset_config: dict[str, Any], project_root: Path
) -> pd.DataFrame:
    path = (project_root / dataset_config["path"]).resolve()
    frame = pd.read_csv(path)
    required = {"run_id", "sample_index", "elapsed_seconds", *SIGNAL_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing standardized columns: {missing}")
    frame["dataset"] = dataset_config["name"]
    return _finalize_feature_frame(frame, dataset_config)


def load_or_extract_dataset(
    dataset_config: dict[str, Any],
    project_root: str | Path,
    cache_dir: str | Path,
    refresh: bool = False,
) -> pd.DataFrame:
    project_root = Path(project_root).resolve()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset_config['name']}_features.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    kind = dataset_config["kind"]
    if kind == "ims":
        frame = extract_ims_dataset(dataset_config, project_root)
    elif kind == "pronostia":
        frame = extract_pronostia_dataset(dataset_config, project_root)
    elif kind == "kaist_temperature":
        frame = extract_kaist_temperature_dataset(dataset_config, project_root)
    elif kind == "standardized_csv":
        frame = load_standardized_csv(dataset_config, project_root)
    elif kind == "classification_only":
        raise ValueError(
            f"{dataset_config['name']} is classification-only and is intentionally "
            "excluded from direct RUL training."
        )
    else:
        raise ValueError(f"Unsupported dataset adapter kind: {kind}")

    frame.to_csv(cache_path, index=False)
    return frame


def split_feature_frame(
    frame: pd.DataFrame, split_config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategy = split_config["strategy"]
    if strategy == "run_ids":
        subsets = []
        for key in ("train_runs", "validation_runs", "test_runs"):
            run_ids = split_config[key]
            subset = frame[frame["run_id"].isin(run_ids)].copy()
            if subset.empty:
                raise ValueError(f"The {key} split is empty for run IDs {run_ids}.")
            subsets.append(subset)
        return tuple(subsets)  # type: ignore[return-value]

    if strategy == "single_run_temporal":
        ratios = split_config.get("ratios", [0.65, 0.15, 0.20])
        if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
            raise ValueError("Temporal split ratios must contain three values summing to 1.")
        if frame["run_id"].nunique() != 1:
            raise ValueError("single_run_temporal requires exactly one run.")
        ordered = frame.sort_values("sample_index")
        first_cut = max(1, int(len(ordered) * ratios[0]))
        second_cut = max(first_cut + 1, int(len(ordered) * sum(ratios[:2])))
        return (
            ordered.iloc[:first_cut].copy(),
            ordered.iloc[first_cut:second_cut].copy(),
            ordered.iloc[second_cut:].copy(),
        )
    raise ValueError(f"Unsupported split strategy: {strategy}")


class SequenceFrameDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        scaled_features: np.ndarray,
        sequence_length: int,
    ) -> None:
        import torch

        self.samples: list[dict[str, Any]] = []
        indexed = frame.reset_index(drop=True).copy()
        indexed["_row_position"] = np.arange(len(indexed))
        for _, run in indexed.groupby("run_id", sort=False):
            run = run.sort_values("sample_index")
            positions = run["_row_position"].to_numpy(dtype=int)
            if len(run) <= sequence_length:
                continue
            for end in range(sequence_length, len(run)):
                target = run.iloc[end]
                start = end - sequence_length
                meta = {
                    column: torch.tensor(
                        [float(target.get(column, 0.0))], dtype=torch.float32
                    )
                    for column in PHYSICS_COLUMNS
                }
                self.samples.append(
                    {
                        "x": torch.tensor(
                            scaled_features[positions[start:end]], dtype=torch.float32
                        ),
                        "time": torch.tensor(
                            [float(target["_time_coordinate"])], dtype=torch.float32
                        ),
                        "target": torch.tensor(
                            [float(target["rul_norm"])], dtype=torch.float32
                        ),
                        "rul_scale_seconds": torch.tensor(
                            [float(target["run_duration_seconds"])],
                            dtype=torch.float32,
                        ),
                        "health_indicator": torch.tensor(
                            [float(target["health_indicator"])], dtype=torch.float32
                        ),
                        "run_id": str(target["run_id"]),
                        "sample_index": int(target["sample_index"]),
                        "meta": meta,
                    }
                )
        if not self.samples:
            raise ValueError(
                "No sequences were created. Reduce sequence_length or inspect the split."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


@dataclass
class PreparedDataset:
    name: str
    feature_columns: list[str]
    scaler: StandardScaler
    preprocessing_metadata: dict[str, Any]
    train: SequenceFrameDataset
    validation: SequenceFrameDataset
    test: SequenceFrameDataset
    split_frames: dict[str, pd.DataFrame]
    time_scale_seconds: float


def apply_initial_baseline_normalization(
    frame: pd.DataFrame,
    preprocessing_config: dict[str, Any] | None,
    sequence_length: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize each run against a fixed, label-free commissioning prefix.

    The transform is intentionally fitted independently for every physical run so a
    test bearing uses only its own initial observations.  With ``prefix_samples`` no
    larger than ``sequence_length``, the complete baseline is available before the
    first sequence target is predicted.  RUL, elapsed-time, health-indicator, and
    physics metadata columns are never used to fit the transform.
    """

    config = dict(preprocessing_config or {})
    strategy = config.get("strategy", "none")
    if strategy in {None, "none"}:
        return frame.copy(), {"strategy": "none"}
    if strategy != "per_run_initial_robust_relative":
        raise ValueError(f"Unsupported preprocessing strategy: {strategy}")
    if config.get("center", "median") != "median":
        raise ValueError("Baseline normalization currently supports median centering only.")
    if config.get("scale", "max_abs_median_scaled_mad_floor") != (
        "max_abs_median_scaled_mad_floor"
    ):
        raise ValueError("Unsupported baseline scale definition.")
    if config.get("uses_targets", False) is not False or config.get(
        "uses_failure_time", False
    ) is not False:
        raise ValueError("Baseline normalization must remain target- and failure-time-free.")

    prefix_samples = int(config.get("prefix_samples", sequence_length))
    if prefix_samples < 2:
        raise ValueError("Baseline normalization requires at least two prefix samples.")
    if config.get("require_prefix_before_first_prediction", True) and (
        prefix_samples > sequence_length
    ):
        raise ValueError(
            "prefix_samples cannot exceed sequence_length when the baseline must be "
            "available before the first prediction."
        )

    feature_columns = list(config.get("feature_columns", SIGNAL_FEATURES))
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Baseline-normalization feature columns must be unique.")
    unknown = sorted(set(feature_columns) - set(SIGNAL_FEATURES))
    if unknown:
        raise ValueError(
            "Baseline normalization is restricted to vibration signal features; "
            f"unsupported columns: {unknown}"
        )
    missing = [column for column in feature_columns if column not in frame]
    if missing:
        raise ValueError(f"Missing baseline-normalization features: {missing}")

    mad_constant = float(config.get("mad_consistency_constant", 1.4826))
    absolute_floor = float(config.get("absolute_scale_floor", 1e-8))
    if mad_constant <= 0.0 or absolute_floor <= 0.0:
        raise ValueError("Baseline scale constants must be positive.")

    transformed = frame.copy()
    run_metadata: dict[str, Any] = {}
    for run_id, run in transformed.groupby("run_id", sort=True):
        ordered = run.sort_values("sample_index")
        if ordered["sample_index"].duplicated().any():
            raise ValueError(f"Run {run_id} contains duplicate sample_index values.")
        if len(ordered) < prefix_samples:
            raise ValueError(
                f"Run {run_id} has {len(ordered)} rows; {prefix_samples} baseline "
                "samples are required."
            )
        prefix = (
            ordered.iloc[:prefix_samples][feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        center = prefix.median(axis=0)
        scaled_mad = (prefix - center).abs().median(axis=0) * mad_constant
        scale = pd.Series(
            np.maximum.reduce(
                [
                    center.abs().to_numpy(dtype=float),
                    scaled_mad.to_numpy(dtype=float),
                    np.full(len(feature_columns), absolute_floor, dtype=float),
                ]
            ),
            index=feature_columns,
        )
        values = (
            transformed.loc[ordered.index, feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        transformed.loc[ordered.index, feature_columns] = (values - center) / scale
        run_metadata[str(run_id)] = {
            "prefix_sample_indices": [
                int(value) for value in ordered.iloc[:prefix_samples]["sample_index"]
            ],
            "center": {key: float(value) for key, value in center.items()},
            "scale": {key: float(value) for key, value in scale.items()},
        }

    metadata = {
        "strategy": strategy,
        "fit_scope": "each physical run independently",
        "uses_targets": False,
        "uses_failure_time": False,
        "prefix_samples": prefix_samples,
        "feature_columns": feature_columns,
        "center": "median of the fixed initial prefix",
        "scale": (
            f"max(abs(prefix median), {mad_constant:g} * prefix MAD, absolute floor)"
        ),
        "mad_consistency_constant": mad_constant,
        "absolute_scale_floor": absolute_floor,
        "require_prefix_before_first_prediction": bool(
            config.get("require_prefix_before_first_prediction", True)
        ),
        "run_statistics": run_metadata,
    }
    return transformed, metadata


def prepare_sequence_dataset(
    frame: pd.DataFrame,
    dataset_config: dict[str, Any],
    sequence_length: int,
    preprocessing_config: dict[str, Any] | None = None,
) -> PreparedDataset:
    frame, preprocessing_metadata = apply_initial_baseline_normalization(
        frame,
        preprocessing_config=preprocessing_config,
        sequence_length=sequence_length,
    )
    frame["run_duration_seconds"] = frame.groupby("run_id")[
        "elapsed_seconds"
    ].transform(lambda values: float(values.max() - values.min()))
    train_frame, validation_frame, test_frame = split_feature_frame(
        frame, dataset_config["split"]
    )
    time_scale_seconds = max(float(train_frame["elapsed_seconds"].max()), 1.0)
    for split_frame in (train_frame, validation_frame, test_frame):
        split_frame["_time_coordinate"] = (
            split_frame["elapsed_seconds"] / time_scale_seconds
        )
        split_frame["cycles_per_time_unit"] = (
            time_scale_seconds * split_frame["speed_rpm"].fillna(0.0) / 60.0
        )
    scaler = StandardScaler()
    train_features = scaler.fit_transform(
        train_frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    validation_features = scaler.transform(
        validation_frame[MODEL_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    test_features = scaler.transform(
        test_frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    return PreparedDataset(
        name=dataset_config["name"],
        feature_columns=list(MODEL_FEATURES),
        scaler=scaler,
        preprocessing_metadata=preprocessing_metadata,
        train=SequenceFrameDataset(train_frame, train_features, sequence_length),
        validation=SequenceFrameDataset(
            validation_frame, validation_features, sequence_length
        ),
        test=SequenceFrameDataset(test_frame, test_features, sequence_length),
        split_frames={
            "train": train_frame,
            "validation": validation_frame,
            "test": test_frame,
        },
        time_scale_seconds=time_scale_seconds,
    )


def enabled_dataset_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dataset
        for dataset in config["datasets"]
        if dataset.get("enabled", False)
        and dataset.get("kind") != "classification_only"
    ]


def validate_dataset_config(
    dataset_config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    project_root = Path(project_root)
    path = project_root / dataset_config["path"]
    status = {
        "dataset": dataset_config["name"],
        "kind": dataset_config["kind"],
        "enabled": dataset_config.get("enabled", False),
        "path": str(path),
        "path_exists": path.exists(),
        "physics_assumptions": dataset_config.get("physics_assumptions", []),
    }
    if dataset_config["kind"] == "classification_only":
        status["note"] = "Excluded: fault classification data do not supply run-to-failure RUL."
    elif dataset_config["kind"] == "standardized_csv" and not path.exists():
        status["note"] = (
            "Adapter contract is ready, but the standardized feature CSV must be generated."
        )
    else:
        status["note"] = "Ready" if path.exists() else "Missing path"
    return status
