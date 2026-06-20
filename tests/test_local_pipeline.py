from __future__ import annotations

import numpy as np

from thesis_work.config import ProjectPaths
from thesis_work.features import build_feature_table_for_target
from thesis_work.ims import list_snapshots, load_snapshot, resolve_dataset_source
from thesis_work.pipeline import validate_data


def _write_snapshot(path, rows: int, cols: int, offset: float = 0.0) -> None:
    data = np.arange(rows * cols, dtype=float).reshape(rows, cols) / 100.0 + offset
    path.write_text(
        "\n".join("\t".join(f"{value:.5f}" for value in row) for row in data),
        encoding="ascii",
    )


def _make_raw_dataset(root, dataset: str, cols: int, names: list[str]) -> None:
    folder = root / "data" / "raw" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        _write_snapshot(folder / name, rows=128, cols=cols, offset=i)


def _paths(tmp_path) -> ProjectPaths:
    return ProjectPaths(
        root=tmp_path,
        raw_data=tmp_path / "data" / "raw",
        processed_features=tmp_path / "data" / "processed_features",
        outputs=tmp_path / "outputs",
        figures=tmp_path / "outputs" / "figures",
        tables=tmp_path / "outputs" / "tables",
    )


def test_directory_dataset_source_and_snapshot_loading(tmp_path):
    _make_raw_dataset(
        tmp_path,
        "2nd_test",
        cols=4,
        names=["2004.02.12.10.52.39", "2004.02.12.10.42.39"],
    )
    paths = _paths(tmp_path)

    source = resolve_dataset_source(paths.raw_data, "2nd_test")
    files = list_snapshots(source)
    arr = load_snapshot(source, files[0], expected_cols=4)

    assert source.is_dir()
    assert files == ["2004.02.12.10.42.39", "2004.02.12.10.52.39"]
    assert arr.shape == (128, 4)


def test_feature_table_for_target_from_local_directory(tmp_path):
    _make_raw_dataset(
        tmp_path,
        "1st_test",
        cols=8,
        names=["2003.10.22.12.06.24", "2003.10.22.12.09.13", "2003.10.22.12.14.13"],
    )
    paths = _paths(tmp_path)

    df = build_feature_table_for_target(paths, dataset="1st_test", target_bearing=3)

    assert len(df) == 3
    assert set(["rms", "E_FTF", "E_BPFO", "E_BPFI", "E_BSF", "E_kin", "rul_norm"]).issubset(df.columns)
    assert df["bearing"].eq(3).all()


def test_validate_data_reports_all_local_datasets(tmp_path):
    _make_raw_dataset(tmp_path, "1st_test", cols=8, names=["2003.10.22.12.06.24"])
    _make_raw_dataset(tmp_path, "2nd_test", cols=4, names=["2004.02.12.10.42.39"])
    _make_raw_dataset(tmp_path, "3rd_test", cols=4, names=["2004.03.04.09.27.46"])
    paths = _paths(tmp_path)

    report = validate_data(paths)

    assert report["dataset"].tolist() == ["1st_test", "2nd_test", "3rd_test"]
    assert report["snapshots"].tolist() == [1, 1, 1]
    assert report["first_snapshot_shape"].tolist() == ["(128, 8)", "(128, 4)", "(128, 4)"]
