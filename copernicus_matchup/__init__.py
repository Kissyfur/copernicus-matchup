"""Match in-situ observations against Copernicus Marine products and build feature cubes.

Given a table of observations (latitude, longitude, time) and a list of gridded products,
this package downloads the products, cuts a space-time window around each observation,
resamples the windows onto a common grid, applies per-channel transformations, and writes
one aligned cube per feature group.

It knows nothing about what will be predicted from those cubes. The hand-off is on disk:
one NetCDF `DataArray` per feature group, carrying an `Id` dimension that indexes the
observations and a `variable` coordinate that names the channels.
"""

from copernicus_matchup.config import (
    DataConfig,
    MatchupConfig,
    PreprocessConfig,
    ProductSpec,
    RegridConfig,
    TargetConfig,
    load_data_config,
)
from copernicus_matchup.layout import RunLayout
from copernicus_matchup.preprocessing import TargetTransform, preprocess_matchups
from copernicus_matchup.stages import build_dataset, create_matchups, download_products

__all__ = [
    "DataConfig",
    "MatchupConfig",
    "PreprocessConfig",
    "ProductSpec",
    "RegridConfig",
    "RunLayout",
    "TargetConfig",
    "TargetTransform",
    "build_dataset",
    "create_matchups",
    "download_products",
    "load_data_config",
    "preprocess_matchups",
    "__version__",
]

__version__ = "0.2.0"
