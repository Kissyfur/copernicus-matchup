from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from copernicus_matchup.config import load_data_config
from copernicus_matchup.layout import RunLayout
from copernicus_matchup.preprocessing import TargetTransform, preprocess_matchups
from copernicus_matchup.stages import create_matchups, download_products

STAGES = ("download", "matchup", "preprocess")


def _target_transform(args: argparse.Namespace) -> TargetTransform:
    floor = None
    if args.target_floor_quantile is not None:
        floor = {"method": "positive_quantile", "q": args.target_floor_quantile}
    return TargetTransform(
        transform=args.target_transform,
        offset=args.target_offset,
        floor=floor,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="copernicus-matchup",
        description=(
            "Build aligned feature cubes from in-situ observations and Copernicus Marine "
            "products. Runs the download, matchup and preprocess stages in order unless "
            "--stage is given."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to a JSON/YAML data config.")
    parser.add_argument(
        "--run-root",
        required=True,
        help="Directory to write raw/, processed/ and datasets/ into.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        help="Run only this stage; repeatable. Default: all three, in order.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redo stages whose outputs already exist.",
    )
    parser.add_argument(
        "--target-transform",
        default="none",
        choices=("none", "log", "log1p"),
        help="Transformation applied to the target column. Default: none.",
    )
    parser.add_argument(
        "--target-offset",
        type=float,
        default=0.0,
        help="Added to the target before the transform, to accommodate zeros.",
    )
    parser.add_argument(
        "--target-floor-quantile",
        type=float,
        help=(
            "Raise targets at or below this quantile of the strictly positive values up "
            "to it, before the transform. Required if the target contains exact zeros "
            "and --target-transform is log."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Only report warnings and errors.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = load_data_config(args.config)
    run_root = Path(args.run_root)
    stages = args.stage or list(STAGES)

    artifacts: dict[str, Path] = {}
    if "download" in stages:
        download_products(config, run_root, overwrite=args.overwrite)
    if "matchup" in stages:
        create_matchups(config, run_root, overwrite=args.overwrite)
    if "preprocess" in stages:
        artifacts = preprocess_matchups(
            config, run_root, target_transform=_target_transform(args)
        )

    layout = RunLayout(run_root)
    summary = {
        "run_root": str(layout.root),
        "stages": stages,
        "datasets": {name: str(path) for name, path in sorted(artifacts.items())},
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
