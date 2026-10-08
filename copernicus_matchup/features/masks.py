from __future__ import annotations

import numpy as np
import xarray as xr


def missing_mask(
    data: xr.DataArray,
    variable_dim: str = "variable",
    name: str = "cloud_mask",
) -> xr.DataArray:
    """Cells with no valid value at that timestep.

    The right definition for a gappy L3 observation product, where a missing value means
    the sensor returned nothing -- cloud, glint, or land. It deliberately does not try to
    separate land out: that is a different product's job (see `static_missing_mask`), and
    guessing at it from persistence is what made the old land mask track the weather.
    """
    missing = np.isnan(data).all(variable_dim)
    return missing.astype(np.float32).expand_dims({variable_dim: [name]})


def static_missing_mask(
    data: xr.DataArray,
    variable_dim: str = "variable",
    time_dim: str = "time",
    name: str = "land_mask",
) -> xr.DataArray:
    """Cells missing at every timestep, broadcast back across time.

    The right definition for a gap-free L4 model product: its fields are computed on an
    ocean grid, so a cell is missing if and only if it is land, at every timestep. That
    makes the result a genuine static geography rather than a record of which days
    happened to be cloudy -- measured on one station's 517 matchups, an L4 source yields
    3 distinct land patterns where the L3 source yields 302.
    """
    missing = np.isnan(data).all(variable_dim)
    if time_dim in missing.dims:
        static = missing.all(dim=time_dim)
        missing = missing.copy()
        missing[:] = static
    return missing.astype(np.float32).expand_dims({variable_dim: [name]})


def get_cloud_and_land_masks(
    data: xr.DataArray,
    variable_dim: str = "variable",
    time_dim: str = "time",
    land_threshold: float = 0.95,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Create cloud and static land masks from one product's missing values.

    Retained for a product asked to supply BOTH masks, which forces land to be inferred
    from persistence and so conflates it with a window that was clouded throughout. Prefer
    taking the cloud mask from an L3 product via `missing_mask` and the land mask from an
    L4 product via `static_missing_mask`.
    """
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
