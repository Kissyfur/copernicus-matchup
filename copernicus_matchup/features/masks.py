from __future__ import annotations

import numpy as np
import xarray as xr


def get_cloud_and_land_masks(
    data: xr.DataArray,
    variable_dim: str = "variable",
    time_dim: str = "time",
    land_threshold: float = 0.95,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Create cloud and static land masks from missing values."""
    is_missing = np.isnan(data).all(variable_dim)
    if time_dim in is_missing.dims:
        missing_frequency = is_missing.mean(dim=time_dim)
    else:
        missing_frequency = is_missing.astype(float)

    is_land_static = missing_frequency > land_threshold
    mask_land = is_missing.copy()
    mask_land[:] = is_land_static
    mask_cloud = is_missing & (~mask_land.astype(bool))

    mask_cloud = mask_cloud.astype(np.float32).expand_dims({variable_dim: ["cloud_mask"]})
    mask_land = mask_land.astype(np.float32).expand_dims({variable_dim: ["land_mask"]})
    return mask_cloud, mask_land


def valid_water_coverage(
    cloud_mask: xr.DataArray,
    land_mask: xr.DataArray,
    dims: tuple[str, ...] = ("lat", "lon", "time"),
) -> xr.DataArray:
    dims = tuple(dim for dim in dims if dim in cloud_mask.dims)
    water_zone = land_mask == 0
    total_water_pixels = water_zone.sum(dim=dims)
    useful_pixels = water_zone & (cloud_mask == 0)
    return useful_pixels.sum(dim=dims) / total_water_pixels
