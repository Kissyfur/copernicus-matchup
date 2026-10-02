from __future__ import annotations

import numpy as np
import xarray as xr


def positive_quantile(
    data: xr.DataArray,
    quantile: float = 0.01,
    dims: tuple[str, ...] = ("Id", "time", "lat", "lon"),
    fallback: float = 1e-12,
) -> xr.DataArray:
    dims = tuple(dim for dim in dims if dim in data.dims)
    quant = data.where(data > 0).quantile(quantile, dim=dims, skipna=True)
    quant = quant.where(np.isfinite(quant) & (quant > 0), other=fallback)
    return np.maximum(data, quant)


def interpolate_gaps(data, dims: tuple[str, ...] = ("lat", "lon")):
    """Linear gap-fill along each dim, for a Dataset or a DataArray.

    Accepts a DataArray because gap-filling now runs *after* the variables have been
    stacked onto a `variable` axis -- it has to happen after cloud/land detection and the
    positive-quantile floor, both of which must see the real gaps (see
    _prepare_product_array in src/data/preprocessing.py).
    """
    if isinstance(data, xr.Dataset):
        interpolated = data.copy()
        for var_name in interpolated.data_vars:
            for dim in dims:
                if dim in interpolated[var_name].dims:
                    interpolated[var_name] = interpolated[var_name].interpolate_na(
                        dim=dim, method="linear", use_coordinate=False
                    )
        return interpolated

    for dim in dims:
        if dim in data.dims:
            data = data.interpolate_na(dim=dim, method="linear", use_coordinate=False)
    return data


def monthly_anomaly(data: xr.DataArray) -> xr.DataArray:
    """Subtract a global monthly mean while preserving 2D time coordinates."""
    if "time" not in data.coords:
        return data
    months_map = data.time.dt.month
    climatology = [
        data.where(months_map == month).mean(skipna=True) for month in range(1, 13)
    ]
    climatology_da = xr.DataArray(climatology, coords={"month": range(1, 13)}, dims="month")
    return data - climatology_da.sel(month=months_map)

