from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

FS = 20_000
RPM = 2_000
FR = RPM / 60

BEARING_PARAMS = {
    "n": 16,
    "d": 0.331,
    "D": 2.815,
    "theta_deg": 15.17,
}

TARGET_RUNS = [
    ("ds2_b1", "2nd_test", 1),
    ("ds1_b3", "1st_test", 3),
    ("ds1_b4", "1st_test", 4),
    ("ds3_b3", "3rd_test", 3),
]

MODEL_COLS = [
    "rms",
    "std",
    "ptp",
    "kurtosis",
    "crest_factor",
    "mean_abs",
    "E_FTF",
    "E_BPFO",
    "E_BPFI",
    "E_BSF",
    "E_kin",
]

FEATURE_COLS_MULTI = ["elapsed_scaled", *MODEL_COLS, "sigma_H_norm"]
TARGET_COL = "rul_norm"
TIME_COL = FEATURE_COLS_MULTI.index("elapsed_scaled")
EKIN_COL = FEATURE_COLS_MULTI.index("E_kin")

PROPOSED_MODEL_NAME = "Proposed DeepXDE Physics-Informed RUL Model"
DATA_BASELINE_NAME = "Data-only neural baseline"
LSTM_BASELINE_NAME = "LSTM baseline"
CNN_BASELINE_NAME = "CNN baseline"
MODEL_ORDER = [
    DATA_BASELINE_NAME,
    PROPOSED_MODEL_NAME,
    LSTM_BASELINE_NAME,
    CNN_BASELINE_NAME,
]

GLOBAL_SEED = 42
DATA_BASELINE_SEED = 42
PINN_SEED = 42
LSTM_SEED = 42
CNN_SEED = 123
SEED_REPEATS = 3
SEED_REPEAT_STRIDE = 100

EXPERIMENTS = [
    {
        "name": "S01",
        "train_runs": ["ds1_b4", "ds3_b3"],
        "validation_runs": ["ds1_b3"],
        "test_runs": ["ds2_b1"],
        "balanced_train": True,
    },
    {
        "name": "S02",
        "train_runs": ["ds1_b3", "ds3_b3"],
        "validation_runs": ["ds1_b4"],
        "test_runs": ["ds2_b1"],
        "balanced_train": True,
    },
    {
        "name": "S03",
        "train_runs": ["ds1_b3", "ds1_b4"],
        "validation_runs": ["ds3_b3"],
        "test_runs": ["ds2_b1"],
        "balanced_train": True,
    },
    {
        "name": "S04",
        "train_runs": ["ds1_b4", "ds3_b3"],
        "validation_runs": ["ds2_b1"],
        "test_runs": ["ds1_b3"],
        "balanced_train": True,
    },
    {
        "name": "S05",
        "train_runs": ["ds2_b1", "ds3_b3"],
        "validation_runs": ["ds1_b4"],
        "test_runs": ["ds1_b3"],
        "balanced_train": True,
    },
    {
        "name": "S06",
        "train_runs": ["ds2_b1", "ds1_b4"],
        "validation_runs": ["ds3_b3"],
        "test_runs": ["ds1_b3"],
        "balanced_train": True,
    },
    {
        "name": "S07",
        "train_runs": ["ds1_b3", "ds3_b3"],
        "validation_runs": ["ds2_b1"],
        "test_runs": ["ds1_b4"],
        "balanced_train": True,
    },
    {
        "name": "S08",
        "train_runs": ["ds2_b1", "ds3_b3"],
        "validation_runs": ["ds1_b3"],
        "test_runs": ["ds1_b4"],
        "balanced_train": True,
    },
    {
        "name": "S09",
        "train_runs": ["ds2_b1", "ds1_b3"],
        "validation_runs": ["ds3_b3"],
        "test_runs": ["ds1_b4"],
        "balanced_train": True,
    },
    {
        "name": "S10",
        "train_runs": ["ds1_b3", "ds1_b4"],
        "validation_runs": ["ds2_b1"],
        "test_runs": ["ds3_b3"],
        "balanced_train": True,
    },
    {
        "name": "S11",
        "train_runs": ["ds2_b1", "ds1_b4"],
        "validation_runs": ["ds1_b3"],
        "test_runs": ["ds3_b3"],
        "balanced_train": True,
    },
    {
        "name": "S12",
        "train_runs": ["ds2_b1", "ds1_b3"],
        "validation_runs": ["ds1_b4"],
        "test_runs": ["ds3_b3"],
        "balanced_train": True,
    },
]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = ROOT_DIR
    raw_data: Path = ROOT_DIR / "data" / "raw"
    processed_features: Path = ROOT_DIR / "data" / "processed_features"
    outputs: Path = ROOT_DIR / "outputs"
    figures: Path = ROOT_DIR / "outputs" / "figures"
    tables: Path = ROOT_DIR / "outputs" / "tables"
    thesis_images: Path | None = None
    raw_datasets: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.thesis_images is None:
            object.__setattr__(
                self,
                "thesis_images",
                self.root / "thesis" / "latex" / "assets" / "images",
            )
        if not self.raw_datasets:
            object.__setattr__(
                self,
                "raw_datasets",
                {name: self.raw_data / name for name in ("1st_test", "2nd_test", "3rd_test")},
            )

    def ensure_output_dirs(self) -> None:
        for path in [self.processed_features, self.outputs, self.figures, self.tables, self.thesis_images]:
            path.mkdir(parents=True, exist_ok=True)


def default_paths() -> ProjectPaths:
    paths = ProjectPaths()
    paths.ensure_output_dirs()
    return paths


def expected_cols(dataset: str) -> int:
    return 8 if dataset == "1st_test" else 4
