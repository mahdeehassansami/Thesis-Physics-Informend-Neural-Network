from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_work.exp6_data_qualification import qualification_paths, run_qualification


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate EXP-006 supplied and controlled synthetic evidence."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--config",
        default="configs/exp006_data_qualification.json",
        help="Repository-relative EXP-006 configuration path.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analyzed/EXP-006",
        help="Repository-relative tracked analysis directory.",
    )
    args = parser.parse_args()
    paths = qualification_paths(args.project_root, args.config, args.output_dir)
    summary = run_qualification(paths)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
