from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from thesis_work.multi_dataset import load_or_extract_dataset


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "colab_experiments.json"
UPLOAD_CACHE = ROOT / "Upload" / "feature_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["ims", "pronostia", "kaist_vibration_temperature"],
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    by_name = {dataset["name"]: dataset for dataset in config["datasets"]}
    UPLOAD_CACHE.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        if name not in by_name:
            raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(by_name)}")
        started = time.time()
        print(f"START {name}", flush=True)
        frame = load_or_extract_dataset(
            by_name[name],
            project_root=ROOT,
            cache_dir=UPLOAD_CACHE,
            refresh=args.refresh,
        )
        output = UPLOAD_CACHE / f"{name}_features.csv"
        print(
            f"DONE {name}: rows={len(frame):,}, "
            f"size_mb={output.stat().st_size / 1024**2:.2f}, "
            f"seconds={time.time() - started:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
