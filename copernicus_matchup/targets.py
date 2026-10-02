from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from copernicus_matchup.config import DEFAULT_TARGET_NAME, TargetConfig
from copernicus_matchup.features.coordinates import dms_to_decimal
from copernicus_matchup.features.temporal import day_to_circle_x, day_to_circle_y

ID = "Id"
LAT = "lat"
LON = "lon"
TIME = "time"
TARGET = DEFAULT_TARGET_NAME


def load_target_table(config: TargetConfig) -> pd.DataFrame:
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"Target table does not exist: {path}")

    if path.suffix.lower() in {".xls", ".xlsx"}:
        data = pd.read_excel(path, sheet_name=config.sheet_name)
    else:
        data = pd.read_csv(path)

    target_names = config.target_names
    missing_targets = [
        column for column in config.target_columns if column not in data.columns
    ]
    if missing_targets:
        raise ValueError(f"Target table is missing configured target column(s): {missing_targets}")

    rename_map = {
        config.id_column: ID,
        config.lat_column: LAT,
        config.lon_column: LON,
        config.time_column: TIME,
    }
    # One entry per output. For a vector each name maps to itself, so this is a no-op
    # rename that still keeps the two paths on one code path.
    rename_map.update(dict(zip(config.target_columns, target_names)))
    data = data.rename(columns=rename_map)
    if ID not in data.columns:
        data[ID] = range(len(data))

    if config.filter_column:
        if config.filter_column not in data.columns:
            raise ValueError(f"Target table is missing configured filter column: {config.filter_column}")
        if config.filter_values:
            data = data[data[config.filter_column].isin(config.filter_values)].copy()

    required = {ID, LAT, LON, TIME, *target_names}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Target table is missing required columns: {sorted(missing)}")

    data[LAT] = data[LAT].map(dms_to_decimal).astype(float)
    data[LON] = data[LON].map(dms_to_decimal).astype(float)
    data[TIME] = pd.to_datetime(data[TIME])
    # A row missing any one output is dropped: the outputs share a feature cube, so a row
    # cannot be kept for some heads and dropped for others without splitting the dataset.
    data = data.dropna(subset=[ID, LAT, LON, TIME, *target_names]).copy()
    return data


def save_standard_target_table(data: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def target_to_dataarray(data: pd.DataFrame, target_names: list[str] | None = None) -> xr.DataArray:
    """Build the (Id, variable) target array, one variable per output.

    The returned variable axis follows `target_names` exactly -- every downstream artifact
    (model head order, per-output metrics, predictions.csv columns) is positional, so the
    order has to come from the config rather than from however pandas/xarray happen to sort
    the columns.
    """
    target_names = [TARGET] if target_names is None else list(target_names)
    missing = [name for name in target_names if name not in data.columns]
    if missing:
        raise ValueError(f"Target table is missing target column(s): {missing}")
    target = data[[ID, *target_names]].set_index(ID)
    array = target.to_xarray().to_dataarray(dim="variable").transpose(ID, "variable")
    return array.sel(variable=target_names)


def metadata_to_dataarray(
    data: pd.DataFrame,
    metadata_columns: list[str] | None = None,
    include_spatial_metadata: bool = True,
    include_day_metadata: bool = True,
    include_cyclic_day_metadata: bool = True,
) -> xr.DataArray:
    metadata_columns = [] if metadata_columns is None else metadata_columns
    missing_metadata = [col for col in metadata_columns if col not in data.columns]
    if missing_metadata:
        raise ValueError(f"Target table is missing configured metadata columns: {missing_metadata}")
    columns = [TIME] + list(metadata_columns)
    if include_spatial_metadata:
        columns = [LAT, LON] + columns
    meta = data[[ID] + columns].copy()
    meta[TIME] = pd.to_datetime(meta[TIME])
    meta["day"] = meta[TIME].dt.dayofyear
    meta["x_day"] = meta["day"].map(day_to_circle_x)
    meta["y_day"] = meta["day"].map(day_to_circle_y)
    base_columns = []
    if include_spatial_metadata:
        base_columns.extend([LAT, LON])
    if include_cyclic_day_metadata:
        base_columns.extend(["x_day", "y_day"])
    if include_day_metadata:
        base_columns.append("day")
    numeric_columns = [
        col
        for col in base_columns + metadata_columns
        if col in meta.columns and col != TIME
    ]
    if not numeric_columns:
        return xr.DataArray(
            np.empty((len(meta), 0), dtype=np.float32),
            dims=(ID, "variable"),
            coords={ID: meta[ID].values, "variable": []},
        )
    return meta[[ID] + numeric_columns].set_index(ID).to_xarray().to_dataarray(
        dim="variable"
    ).transpose(ID, "variable")
