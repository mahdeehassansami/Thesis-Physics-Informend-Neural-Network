from __future__ import annotations

import argparse

from thesis_work.config import default_paths
from thesis_work.pipeline import (
    load_or_build_all_features,
    regenerate_figures_from_cache,
    run_pipeline,
    validate_data,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local NASA IMS bearing RUL thesis pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate-data", help="Check local IMS datasets under data/raw")

    extract = subparsers.add_parser("extract-features", help="Extract and cache feature tables")
    extract.add_argument("--max-files", type=int, default=None, help="Only process the first N snapshots per run")
    extract.add_argument("--refresh", action="store_true", help="Rebuild cached feature CSV files")

    run = subparsers.add_parser("run", help="Run feature extraction, analysis, training, and report outputs")
    run.add_argument("--max-files", type=int, default=None, help="Only process the first N snapshots per run")
    run.add_argument("--refresh-features", action="store_true", help="Rebuild cached feature CSV files")
    run.add_argument("--skip-training", action="store_true", help="Only generate features, PCA-HI, and static thesis tables/figures")
    run.add_argument("--baseline-iterations", type=int, default=15_000)
    run.add_argument("--pinn-iterations", type=int, default=20_000)
    run.add_argument("--sequence-epochs", type=int, default=60)
    run.add_argument("--sequence-patience", type=int, default=8)
    run.add_argument("--sequence-batch-size", type=int, default=128)
    run.add_argument("--sequence-length", type=int, default=20)

    subparsers.add_parser(
        "regenerate-figures",
        help="Regenerate polished figures from cached full-run features and result tables",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = default_paths()

    if args.command == "validate-data":
        print(validate_data(paths).to_string(index=False))
        return

    if args.command == "extract-features":
        df = load_or_build_all_features(paths, max_files=args.max_files, refresh=args.refresh)
        print(df.groupby("run_id").size().to_string())
        return

    if args.command == "run":
        results = run_pipeline(
            paths=paths,
            max_files=args.max_files,
            refresh_features=args.refresh_features,
            skip_training=args.skip_training,
            baseline_iterations=args.baseline_iterations,
            pinn_iterations=args.pinn_iterations,
            sequence_epochs=args.sequence_epochs,
            sequence_patience=args.sequence_patience,
            sequence_batch_size=args.sequence_batch_size,
            sequence_length=args.sequence_length,
        )
        final_results = results["final_results"]
        if final_results is not None:
            print(final_results.to_string(index=False))
        print(f"Outputs saved under {paths.outputs}")
        return

    if args.command == "regenerate-figures":
        regenerate_figures_from_cache(paths)
        print(f"Figures regenerated under {paths.figures}")
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
