from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import numpy as np

from thesis_work.config import BEARING_PARAMS, FR


def parse_ims_timestamp(filename: str | Path) -> dt.datetime:
    return dt.datetime.strptime(Path(filename).name, "%Y.%m.%d.%H.%M.%S")


def resolve_dataset_source(raw_data_dir: Path, dataset: str) -> Path:
    directory = raw_data_dir / dataset
    zip_path = raw_data_dir / f"{dataset}.zip"
    if directory.is_dir():
        return directory
    if zip_path.is_file():
        return zip_path
    raise FileNotFoundError(
        f"Missing {dataset}. Expected either {directory} or {zip_path}."
    )


def list_snapshots(source: Path) -> list[str]:
    if source.is_dir():
        files = [p.name for p in source.iterdir() if p.is_file()]
    else:
        with zipfile.ZipFile(source) as zf:
            files = [i.filename for i in zf.infolist() if not i.is_dir()]
    return sorted(files, key=parse_ims_timestamp)


def load_snapshot(source: Path, inner_file: str, expected_cols: int) -> np.ndarray:
    if source.is_dir():
        raw = (source / inner_file).read_text(encoding="ascii", errors="ignore")
    else:
        with zipfile.ZipFile(source) as zf:
            raw = zf.read(inner_file).decode("ascii", errors="ignore")

    arr = np.fromstring(raw, sep="\t", dtype=np.float64)
    if arr.size % expected_cols != 0:
        arr = np.fromstring(raw.replace("\t", " "), sep=" ", dtype=np.float64)
    if arr.size % expected_cols != 0:
        raise ValueError(
            f"{inner_file} has {arr.size} numeric values, not divisible by {expected_cols}."
        )
    return arr.reshape(-1, expected_cols)


def channel_to_bearing(dataset: str, channel_index_zero_based: int) -> tuple[int, str]:
    if dataset == "1st_test":
        bearing = channel_index_zero_based // 2 + 1
        axis = "x" if channel_index_zero_based % 2 == 0 else "y"
        return bearing, axis
    return channel_index_zero_based + 1, "single"


def bearing_fault_frequencies(
    fr: float = FR,
    n: int = 16,
    d: float = 0.331,
    D: float = 2.815,
    theta_deg: float = 15.17,
) -> dict[str, float]:
    theta = np.deg2rad(theta_deg)
    ratio = (d / D) * np.cos(theta)
    ftf = fr / 2 * (1 - ratio)
    bpfo = n * fr / 2 * (1 - ratio)
    bpfi = n * fr / 2 * (1 + ratio)
    bsf = D * fr / (2 * d) * (1 - ratio**2)
    return {"FTF": ftf, "BPFO": bpfo, "BPFI": bpfi, "BSF": bsf}


FAULT_FREQS = bearing_fault_frequencies(**BEARING_PARAMS)
