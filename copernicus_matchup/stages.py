from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from copernicus_matchup.config import DataConfig, ProductSpec
from copernicus_matchup.copernicus import (
    open_copernicus_dataset,
    rename_common_dimensions,
    save_dataset,
)
from copernicus_matchup.layout import RunLayout
from copernicus_matchup.matchups import create_product_matchups, save_matchups
from copernicus_matchup.targets import load_target_table, save_standard_target_table

logger = logging.getLogger("copernicus_matchup.stages")


def _write_remote_product_marker(product: ProductSpec, path: Path, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = {
        "name": product.name,
        "source": product.source,
        "dataset_ids": product.dataset_ids,
        "variables": product.variables,
        "open_dataset_kwargs": product.open_dataset_kwargs,
        "matchup": product.matchup,
        "preprocess": product.preprocess,
        "note": (
            "Remote Copernicus products are opened lazily during matchup creation. "
            "Only target-centered time/lat/lon windows are materialized as NetCDF matchups."
        ),
    }
    path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return path


def download_products(
    config: DataConfig,
    run_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Prepare every configured product under `run_root`.

    Remote products are deliberately not materialized: a whole Copernicus dataset is far
    larger than the windows a run needs, so only a marker recording how to open it is
    written here and the actual slicing happens in `create_matchups`. Local products are
    opened, their dimensions renamed to the internal lat/lon convention, and saved whole.

    Returns the artifact path per product name.
    """
    layout = RunLayout(run_root)
    layout.mkdirs("raw")
    artifacts: dict[str, Path] = {}

    for product in config.products:
        if product.source != "local":
            logger.info(
                "Preparing remote product %s without materializing the full dataset; "
                "matchup stage will save only sliced windows.",
                product.name,
            )
            artifacts[product.name] = _write_remote_product_marker(
                product, layout.remote_marker(product.name), overwrite
            )
            continue

        output_path = layout.raw_product(product.name)
        if output_path.exists() and not overwrite:
            logger.info("Skipping existing raw product %s", output_path)
            artifacts[product.name] = output_path
            continue

        logger.info("Opening local product %s", product.name)
        dataset = rename_common_dimensions(open_copernicus_dataset(product), product)
        artifacts[product.name] = save_dataset(dataset, output_path)
        logger.info("Saved %s", output_path)

    return artifacts


def create_matchups(
    config: DataConfig,
    run_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Cut a target-centred window out of every product, for every observation.

    Writes the standardized target table alongside, since downstream stages read the
    observations back from it rather than re-reading the original file.

    Returns the matchup path per product name; a product that matched nothing is absent
    from the mapping but still leaves its unmatched-rows CSV behind.
    """
    layout = RunLayout(run_root)
    layout.mkdirs("processed", "matchups")
    artifacts: dict[str, Path] = {}

    targets = load_target_table(config.target)
    save_standard_target_table(targets, layout.targets)

    for product in config.products:
        matchup_path = layout.product_matchup(product.name)
        if matchup_path.exists() and not overwrite:
            logger.info("Skipping existing matchups %s", matchup_path)
            artifacts[product.name] = matchup_path
            continue

        raw_path = layout.raw_product(product.name)
        local_raw_path = raw_path if product.source == "local" and raw_path.exists() else None
        logger.info("Creating matchups for %s", product.name)
        matchups, unmatched = create_product_matchups(
            product,
            targets,
            config.matchup,
            raw_path=local_raw_path,
        )
        save_matchups(matchups, unmatched, matchup_path, layout.product_unmatched(product.name))
        if matchups is not None:
            artifacts[product.name] = matchup_path
            logger.info("Saved %s", matchup_path)
        logger.info("Unmatched observations for %s: %s", product.name, len(unmatched))

    return artifacts


def build_dataset(
    config: DataConfig,
    run_root: str | Path,
    overwrite: bool = False,
    target_transform=None,
) -> dict[str, Path]:
    """Run the three acquisition stages in order: download, matchup, preprocess.

    This is the whole package in one call, for a caller that has a config file and wants
    cubes. `target_transform` is passed through to the preprocessing stage.
    """
    from copernicus_matchup.preprocessing import preprocess_matchups

    download_products(config, run_root, overwrite=overwrite)
    create_matchups(config, run_root, overwrite=overwrite)
    return preprocess_matchups(config, run_root, target_transform=target_transform)
